"""Tests for agents_core.phala_tee (agents-core-phala-tee-contractor-tier-v0).

Covers: JCS canonicalization + AAD/HKDF fixed vectors (no network), the
X25519/AES-GCM seal-open round trip, `verify_report_binding`'s five checks
(accept valid, reject each tampered check independently), and a
PHALA_API_KEY-gated live integration test (skipped, not failed, when the
credential is absent — same convention as tests/test_retrieval.py's
`@pytest.mark.integration` real-backend test).
"""
import copy
import os
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from agents_core import phala_tee


# ---------------------------------------------------------------------------
# JCS / AAD / HKDF fixed vectors — no network.
# ---------------------------------------------------------------------------


def test_canonicalize_sorts_keys_and_encodes_json_types():
    s = phala_tee.canonicalize({"b": 1, "a": [1, 2, 3], "c": None, "d": True})
    assert s == '{"a":[1,2,3],"b":1,"c":null,"d":true}'


def test_canonicalize_rejects_non_integer_numbers():
    with pytest.raises(phala_tee.AciFormatError):
        phala_tee.canonicalize({"x": 1.5})


def test_request_aad_matches_fixed_vector():
    aad = phala_tee.request_aad(
        algo="x25519-aes-256-gcm-hkdf-sha256",
        model="m",
        field="messages.0.content",
        nonce="deadbeef",
        ts=1700000000,
    )
    assert aad == (
        b'{"algo":"x25519-aes-256-gcm-hkdf-sha256","field":"messages.0.content",'
        b'"model":"m","nonce":"deadbeef","purpose":"aci.e2ee.request.v2","ts":1700000000}'
    )


def test_response_aad_matches_fixed_vector():
    aad = phala_tee.response_aad(
        algo="x25519-aes-256-gcm-hkdf-sha256",
        model="m",
        id="resp-1",
        field="choices.0.message.content",
        nonce="deadbeef",
        ts=1700000000,
    )
    assert aad == (
        b'{"algo":"x25519-aes-256-gcm-hkdf-sha256","field":"choices.0.message.content",'
        b'"id":"resp-1","model":"m","nonce":"deadbeef",'
        b'"purpose":"aci.e2ee.response.v2","ts":1700000000}'
    )


def test_derive_e2ee_key_matches_fixed_vectors():
    # HKDF-SHA256(salt=b"", info=b"aci.e2ee.v2.x25519", len=32) over a fixed
    # shared secret — deterministic given fixed input, no randomness involved.
    key_zero = phala_tee.derive_e2ee_key(bytes(32))
    assert key_zero.hex() == "8eae6dd8e57a7ecdcb76c0b9e900ad377b5a056f56ca507f498dbad379a8e4f2"
    key_ones = phala_tee.derive_e2ee_key(bytes([0x11] * 32))
    assert key_ones.hex() == "ea36878c5fba2152e5e13ee4ca6a451a4962ed49668fe0b500140eae844af60f"
    assert key_zero != key_ones


# ---------------------------------------------------------------------------
# seal_field / open_field round trip.
# ---------------------------------------------------------------------------


def test_seal_open_field_round_trip():
    recipient_priv = X25519PrivateKey.generate()
    recipient_pub_raw = recipient_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    aad = b"some-aad"
    blob_hex = phala_tee.seal_field(recipient_pub_raw, b"hello confidential world", aad)
    plaintext = phala_tee.open_field(recipient_priv, blob_hex, aad)
    assert plaintext == b"hello confidential world"


def test_open_field_rejects_wrong_aad():
    from cryptography.exceptions import InvalidTag

    recipient_priv = X25519PrivateKey.generate()
    recipient_pub_raw = recipient_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    blob_hex = phala_tee.seal_field(recipient_pub_raw, b"secret", b"correct-aad")
    with pytest.raises(InvalidTag):
        phala_tee.open_field(recipient_priv, blob_hex, b"wrong-aad")


# ---------------------------------------------------------------------------
# Synthetic AttestationReport helpers.
# ---------------------------------------------------------------------------


def _build_valid_report(*, not_after=None, nonce="test-nonce-123"):
    identity_priv = Ed25519PrivateKey.generate()
    identity_pub_hex = identity_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()

    service_priv = X25519PrivateKey.generate()
    service_pub_hex = service_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()

    if not_after is None:
        not_after = int(time.time()) + 3600

    keyset = {
        "workload_identity": {"public_key": {"algo": "ed25519", "public_key": identity_pub_hex}},
        "keyset_epoch": {"version": 1, "not_after": not_after},
        "receipt_signing_keys": [],
        "e2ee_public_keys": [{"algo": phala_tee.E2EE_ALGO, "public_key": service_pub_hex}],
    }

    workload_id = phala_tee.compute_workload_id(keyset["workload_identity"]["public_key"])
    workload_keyset_digest = phala_tee.compute_keyset_digest(keyset)
    report_data = phala_tee.compute_report_data(workload_id, workload_keyset_digest, nonce)
    endorsement_sig = identity_priv.sign(
        phala_tee.keyset_endorsement_payload(workload_keyset_digest)
    ).hex()

    report = {
        "api_version": "1",
        "workload_id": workload_id,
        "workload_keyset_digest": workload_keyset_digest,
        "attestation": {
            "workload_keyset": keyset,
            "report_data": report_data,
            "keyset_endorsement": {"algo": "ed25519", "value": endorsement_sig},
        },
    }
    return report, nonce, identity_priv, service_priv


# ---------------------------------------------------------------------------
# verify_report_binding — accept valid, reject each tampered check.
# ---------------------------------------------------------------------------


def test_verify_report_binding_accepts_valid_report():
    report, nonce, _, _ = _build_valid_report()
    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is True
    assert all(c.ok for c in verification.checks)
    names = {c.name for c in verification.checks}
    assert names == {
        "workload_id",
        "workload_keyset_digest",
        "report_data",
        "keyset_endorsement",
        "keyset_epoch.not_after",
    }


def test_verify_report_binding_rejects_tampered_workload_id():
    report, nonce, _, _ = _build_valid_report()
    report["workload_id"] = "sha256:" + "0" * 64
    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "workload_id" in failed


def test_verify_report_binding_rejects_tampered_workload_keyset_digest():
    report, nonce, _, _ = _build_valid_report()
    report["workload_keyset_digest"] = "sha256:" + "1" * 64
    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "workload_keyset_digest" in failed


def test_verify_report_binding_rejects_tampered_report_data():
    report, nonce, _, _ = _build_valid_report()
    report["attestation"]["report_data"] = "f" * 64
    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "report_data" in failed


def test_verify_report_binding_rejects_tampered_endorsement():
    report, nonce, _, _ = _build_valid_report()
    tampered = bytearray(bytes.fromhex(report["attestation"]["keyset_endorsement"]["value"]))
    tampered[0] ^= 0xFF
    report["attestation"]["keyset_endorsement"]["value"] = bytes(tampered).hex()
    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "keyset_endorsement" in failed


def test_verify_report_binding_rejects_expired_epoch():
    report, nonce, _, _ = _build_valid_report(not_after=int(time.time()) - 10)
    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "keyset_epoch.not_after" in failed


def test_require_verified_report_binding_raises_on_failure():
    report, nonce, _, _ = _build_valid_report()
    report["attestation"]["report_data"] = "f" * 64
    with pytest.raises(phala_tee.ReportVerificationError) as excinfo:
        phala_tee.require_verified_report_binding(report, nonce)
    assert "report_data" in str(excinfo.value)


def test_require_verified_report_binding_passes_valid_report():
    report, nonce, _, _ = _build_valid_report()
    verification = phala_tee.require_verified_report_binding(report, nonce)
    assert verification.ok is True


# ---------------------------------------------------------------------------
# open_e2ee_channel / E2eeChannel full seal-open flow against a verified
# synthetic report (still no network — the "gateway" side is simulated
# locally with the service's own X25519 private key).
# ---------------------------------------------------------------------------


def test_open_e2ee_channel_refuses_unverified_report():
    report, nonce, _, _ = _build_valid_report()
    report["attestation"]["report_data"] = "f" * 64
    verification = phala_tee.verify_report_binding(report, nonce)
    with pytest.raises(phala_tee.ReportVerificationError):
        phala_tee.open_e2ee_channel(report, verification)


def test_e2ee_channel_seal_and_simulated_gateway_round_trip():
    report, nonce, _, service_priv = _build_valid_report()
    verification = phala_tee.require_verified_report_binding(report, nonce)
    channel = phala_tee.open_e2ee_channel(report, verification)

    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is the secret ingredient?"},
    ]
    sealed, headers = channel.seal_messages(messages, model="some-model")

    assert headers["X-E2EE-Version"] == "2"
    assert sealed[0]["content"] != messages[0]["content"]
    assert sealed[1]["content"] != messages[1]["content"]

    # Simulate the gateway: decrypt each sealed message content with the
    # service's private key and the same request AAD the client used.
    nonce_hdr = headers["X-E2EE-Nonce"]
    ts_hdr = int(headers["X-E2EE-Timestamp"])
    for i, (orig, sealed_m) in enumerate(zip(messages, sealed)):
        aad = phala_tee.request_aad(
            algo=phala_tee.E2EE_ALGO, model="some-model", field=f"messages.{i}.content",
            nonce=nonce_hdr, ts=ts_hdr,
        )
        recovered = phala_tee.open_field(service_priv, sealed_m["content"], aad).decode("utf-8")
        assert recovered == orig["content"]

    # Simulate the gateway's response: encrypt a reply to the client's
    # X-Client-Pub-Key using the response AAD tag, and confirm open_response
    # decrypts it correctly.
    client_pub_raw = bytes.fromhex(headers["X-Client-Pub-Key"])
    reply_text = "the secret ingredient is paprika"
    response_id = "resp-abc"
    reply_aad = phala_tee.response_aad(
        algo=phala_tee.E2EE_ALGO, model="some-model", id=response_id,
        field="choices.0.message.content", nonce=nonce_hdr, ts=ts_hdr,
    )
    reply_blob = phala_tee.seal_field(client_pub_raw, reply_text.encode("utf-8"), reply_aad)

    response = {
        "id": response_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_blob}}],
    }
    opened = channel.open_response(response)
    assert opened["choices"][0]["message"]["content"] == reply_text


# ---------------------------------------------------------------------------
# Live integration test — real network call to Phala; skipped (not failed)
# when PHALA_API_KEY is unset, per this repo's credentialed-integration-test
# convention (see tests/test_retrieval.py's @pytest.mark.integration test).
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PHALA_API_KEY"),
    reason="PHALA_API_KEY not set — live Phala ACI integration test skipped",
)
def test_integration_phala_live_round_trip_and_forged_report_rejected():
    """Run locally:  PHALA_API_KEY=... pytest -m integration tests/test_phala_tee.py"""
    model = os.environ.get("PHALA_MODEL", "deepseek-ai/DeepSeek-V3")
    client = phala_tee.PhalaTeeClient()

    response = client.chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly the word: acknowledged"}],
        model=model,
    )
    content = response["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip()

    # A deliberately forged attestation report must be rejected by
    # verify_report_binding before any key material is used.
    nonce = "integration-test-nonce"
    real_report = client.fetch_attestation(nonce=nonce)
    forged = copy.deepcopy(real_report)
    forged["attestation"]["report_data"] = "0" * 64
    with pytest.raises(phala_tee.ReportVerificationError):
        phala_tee.require_verified_report_binding(forged, nonce)
