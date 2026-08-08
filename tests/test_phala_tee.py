"""Tests for agents_core.phala_tee (agents-core-phala-aci1-restoration-v0).

Covers: JCS canonicalization + AAD/HKDF fixed vectors (no network), the
X25519/AES-GCM seal-open round trip, the `aci` CLI delegation
`verify_report_binding` now uses (fail-closed on missing binary / timeout /
non-zero exit / unparseable transcript / digest-nonce mismatch / a required
check reported as skip; distinguishable exception types per fault; a
malicious/hostile-verifier transcript refused even when it claims
VERIFIED/exit-0/all-pass; nonce-bound results never cached), the
capability-gated e2ee path in `chat_completion` (seals when advertised,
sends plaintext only to loopback when not), and a PHALA_API_KEY-gated live
integration test (skipped, not failed, when the credential is absent — same
convention as tests/test_retrieval.py's `@pytest.mark.integration` real-
backend test).
"""
import os
import time
import unittest.mock as mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
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


def test_response_aad_matches_fixed_vector_for_reasoning_content():
    aad = phala_tee.response_aad(
        algo="x25519-aes-256-gcm-hkdf-sha256",
        model="m",
        id="resp-1",
        field="choices.0.message.reasoning_content",
        nonce="deadbeef",
        ts=1700000000,
    )
    assert aad == (
        b'{"algo":"x25519-aes-256-gcm-hkdf-sha256","field":"choices.0.message.reasoning_content",'
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
# verify_ecdsa_secp256k1 — unit-level, one test per failure mode. Kept as a
# generic crypto primitive even though verify_report_binding no longer calls
# it directly (aci/1 dropped the software keyset-endorsement signature it
# was verifying — see module docstring).
# ---------------------------------------------------------------------------


def _ecdsa_sign_raw(private_key, message: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes

    der_sig = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_sig)
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _ecdsa_pub_raw(private_key) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )


def test_verify_ecdsa_secp256k1_accepts_valid_signature():
    priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = _ecdsa_sign_raw(priv, message)
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), sig_raw, message) is True


def test_verify_ecdsa_secp256k1_rejects_wrong_pubkey():
    priv = ec.generate_private_key(ec.SECP256K1())
    other_priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = _ecdsa_sign_raw(priv, message)
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(other_priv), sig_raw, message) is False


def test_verify_ecdsa_secp256k1_rejects_wrong_signature_bytes():
    priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = bytearray(_ecdsa_sign_raw(priv, message))
    sig_raw[0] ^= 0xFF
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), bytes(sig_raw), message) is False


def test_verify_ecdsa_secp256k1_rejects_non_64_byte_signature():
    priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = _ecdsa_sign_raw(priv, message)
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), sig_raw[:-1], message) is False
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), sig_raw + b"\x00", message) is False


def test_verify_ecdsa_secp256k1_rejects_r_zero():
    priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = bytearray(_ecdsa_sign_raw(priv, message))
    sig_raw[0:32] = b"\x00" * 32
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), bytes(sig_raw), message) is False


def test_verify_ecdsa_secp256k1_rejects_s_zero():
    priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = bytearray(_ecdsa_sign_raw(priv, message))
    sig_raw[32:64] = b"\x00" * 32
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), bytes(sig_raw), message) is False


def test_verify_ecdsa_secp256k1_rejects_r_at_or_above_curve_order():
    priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = bytearray(_ecdsa_sign_raw(priv, message))
    sig_raw[0:32] = phala_tee.SECP256K1_ORDER.to_bytes(32, "big")
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), bytes(sig_raw), message) is False


def test_verify_ecdsa_secp256k1_rejects_s_at_or_above_curve_order():
    priv = ec.generate_private_key(ec.SECP256K1())
    message = b"some message"
    sig_raw = bytearray(_ecdsa_sign_raw(priv, message))
    sig_raw[32:64] = phala_tee.SECP256K1_ORDER.to_bytes(32, "big")
    assert phala_tee.verify_ecdsa_secp256k1(_ecdsa_pub_raw(priv), bytes(sig_raw), message) is False


# ---------------------------------------------------------------------------
# aci/1 report + aci CLI transcript fixtures.
# ---------------------------------------------------------------------------


def _build_aci1_report(*, e2ee_versions=None):
    """A synthetic aci/1-shaped attestation report — no workload_id,
    workload_identity, keyset_endorsement, or keyset_epoch, matching the
    live wire format captured 2026-08-07 (finding/brix-phala-aci1-break-
    confirmed-and-restoration-in-flight-2026-08-07)."""
    service_priv = X25519PrivateKey.generate()
    service_pub_hex = service_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    keyset = {
        "e2ee_public_keys": [{"algo": phala_tee.E2EE_ALGO, "public_key": service_pub_hex}],
        "not_after": int(time.time()) + 3600,
        "receipt_signing_keys": [],
        "subject": None,
        "tls_public_keys": [],
    }
    keyset_digest = phala_tee.compute_keyset_digest(keyset)
    report = {
        "api_version": "aci/1",
        "attestation": {
            "evidence": {},
            "report_data": "deadbeef" * 8,
            "source_provenance": {
                "repo_commit": "b7ca97aef029e52a16d7dca147f9fbcbe5fa3120",
                "repo_url": "https://github.com/Dstack-TEE/private-ai-gateway.git",
            },
            "tee_type": "tdx",
            "workload_keyset": keyset,
        },
        "service_capabilities": {
            "supported_e2ee_versions": e2ee_versions or [],
            "serving": "aggregator",
        },
        "workload_keyset_digest": keyset_digest,
    }
    return report, keyset_digest, service_priv


def _passing_transcript(*, checks, keyset_digest=None, nonce=None, exit_code=0, verdict="VERIFIED"):
    transcript = {
        "verdict": verdict,
        "checks": [{"id": cid, "status": status} for cid, status in checks],
        "_exit_code": exit_code,
    }
    if keyset_digest is not None:
        transcript["keyset_digest"] = keyset_digest
    if nonce is not None:
        transcript["nonce"] = nonce
    return transcript


def _install_aci_cli_stub(monkeypatch, *, audit_transcript, verify_transcript):
    """Stub `_run_aci_json` (the one real subprocess boundary) so
    `verify_report_binding`'s two-leg delegation and gating logic can be
    exercised without a real `aci` binary. Returns a call counter dict so
    tests can assert on cache behavior (DoD 12)."""
    calls = {"audit": 0, "verify": 0}

    def fake_run_aci_json(args, *, input_bytes=None, timeout=30.0):
        if args[0] == "audit":
            calls["audit"] += 1
            return dict(audit_transcript)
        if args[0] == "verify":
            calls["verify"] += 1
            return dict(verify_transcript)
        raise AssertionError(f"unexpected aci args {args}")

    monkeypatch.setattr(phala_tee, "_run_aci_json", fake_run_aci_json)
    monkeypatch.setattr(phala_tee, "_online_verify_cache", {})
    return calls


def _default_transcripts(*, keyset_digest, nonce):
    audit = _passing_transcript(
        checks=[("id-2", "pass"), ("id-3", "pass"), ("id-4", "pass")],
        keyset_digest=keyset_digest,
        nonce=nonce,
    )
    verify = _passing_transcript(
        checks=[("id-1", "pass"), ("id-5", "skip"), ("id-6", "pass")],
    )
    return audit, verify


# ---------------------------------------------------------------------------
# verify_report_binding / require_verified_report_binding — aci CLI
# delegation, fail-closed, per-check gating (DoD 1, 3, 4, 5, 6, 11, 12).
# ---------------------------------------------------------------------------


def test_verify_report_binding_accepts_valid_aci1_report(monkeypatch):
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is True
    assert verification.workload_id == keyset_digest
    assert verification.workload_keyset_digest == keyset_digest
    # id-5 reported "skip", not "pass" — must be surfaced, never absorbed.
    assert verification.custody_skipped is True


def test_verify_report_binding_reports_custody_verified_when_id5_passes(monkeypatch):
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, _ = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    verify = _passing_transcript(checks=[("id-1", "pass"), ("id-5", "pass"), ("id-6", "pass")])
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is True
    assert verification.custody_skipped is False


@pytest.mark.parametrize("failing_id", ["id-2", "id-3", "id-4"])
def test_verify_report_binding_rejects_failing_required_audit_check(monkeypatch, failing_id):
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    checks = [("id-2", "pass"), ("id-3", "pass"), ("id-4", "pass")]
    checks = [(cid, "fail" if cid == failing_id else status) for cid, status in checks]
    audit = _passing_transcript(checks=checks, keyset_digest=keyset_digest, nonce=nonce)
    _, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert f"aci.{failing_id}" in failed


@pytest.mark.parametrize("failing_id", ["id-1", "id-6"])
def test_verify_report_binding_rejects_failing_required_verify_check(monkeypatch, failing_id):
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, _ = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    checks = [("id-1", "pass"), ("id-5", "skip"), ("id-6", "pass")]
    checks = [(cid, "fail" if cid == failing_id else status) for cid, status in checks]
    verify = _passing_transcript(checks=checks)
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert f"aci.{failing_id}" in failed


@pytest.mark.parametrize("required_id,leg", [("id-2", "audit"), ("id-1", "verify")])
def test_verify_report_binding_rejects_required_check_reported_as_skip(monkeypatch, required_id, leg):
    """DoD 5: a transcript that is VERIFIED, exit 0, but reports a REQUIRED
    check as skip must be refused — gating is per-check status, never the
    top-line verdict or exit code alone."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    if leg == "audit":
        audit = _passing_transcript(
            checks=[(cid, "skip" if cid == required_id else "pass") for cid, _ in
                    [("id-2", None), ("id-3", None), ("id-4", None)]],
            keyset_digest=keyset_digest, nonce=nonce,
        )
    else:
        verify = _passing_transcript(
            checks=[(cid, "skip" if cid == required_id else "pass") for cid, _ in
                    [("id-1", None), ("id-5", None), ("id-6", None)]],
        )
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    with pytest.raises(phala_tee.ReportVerificationError):
        phala_tee.require_verified_report_binding(report, nonce)


def test_verify_report_binding_rejects_hostile_verifier_digest_mismatch(monkeypatch):
    """DoD 11: a transcript reporting VERIFIED, exit 0, and every required
    check as pass, but carrying a mismatched keyset digest, must be
    refused — proves the parse is a real check, not decoration."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest="sha256:" + "0" * 64, nonce=nonce)
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "aci_audit.keyset_digest_binding" in failed


def test_verify_report_binding_rejects_hostile_verifier_nonce_mismatch(monkeypatch):
    """DoD 4 + 11: a wrong nonce in the transcript is refused even though
    every declared check reports pass."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce="b" * 64)
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "aci_audit.nonce_binding" in failed


def test_verify_report_binding_rejects_non_zero_exit_even_with_passing_checks(monkeypatch):
    """DoD 3: a non-zero exit must raise, not pass, even if every declared
    check in the transcript says pass — exit code is necessary, never
    sufficient."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    audit["_exit_code"] = 1
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert "aci_audit.exit_code" in failed


def test_require_verified_report_binding_raises_on_failure(monkeypatch):
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest="sha256:" + "0" * 64, nonce=nonce)
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    with pytest.raises(phala_tee.ReportVerificationError) as excinfo:
        phala_tee.require_verified_report_binding(report, nonce)
    assert "aci_audit.keyset_digest_binding" in str(excinfo.value)


def test_require_verified_report_binding_passes_valid_report(monkeypatch):
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.require_verified_report_binding(report, nonce)
    assert verification.ok is True


def test_verify_report_binding_never_caches_the_nonce_bound_audit_leg(monkeypatch):
    """DoD 12: a second call with a DIFFERENT nonce must re-run the offline
    binding check rather than reusing a cached result — only the online
    (nonce-independent) leg may be cached."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce1, nonce2 = "a" * 64, "b" * 64
    audit1, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce1)
    calls = _install_aci_cli_stub(monkeypatch, audit_transcript=audit1, verify_transcript=verify)

    v1 = phala_tee.verify_report_binding(report, nonce1)
    assert v1.ok is True
    assert calls == {"audit": 1, "verify": 1}

    # Second call, different nonce: the stub still returns the SAME audit
    # transcript (bound to nonce1) — since the audit leg must never be
    # cached, verify_report_binding re-runs it fresh and this now correctly
    # fails the nonce-binding check against nonce2.
    v2 = phala_tee.verify_report_binding(report, nonce2)
    assert calls["audit"] == 2  # re-ran, not served from cache
    assert calls["verify"] == 1  # served from the one-hour keyset-digest cache
    assert v2.ok is False
    failed = {c.name for c in v2.checks if not c.ok}
    assert "aci_audit.nonce_binding" in failed


# ---------------------------------------------------------------------------
# aci CLI toolchain faults — distinguishable exception types, all three
# refuse the call, none degrade to an unverified pass (DoD 13).
# ---------------------------------------------------------------------------


def test_run_aci_json_raises_not_found_on_missing_binary(monkeypatch):
    monkeypatch.setattr(
        phala_tee.subprocess, "run",
        mock.Mock(side_effect=FileNotFoundError("no such file")),
    )
    with pytest.raises(phala_tee.AciVerifierNotFoundError):
        phala_tee._run_aci_json(["audit", "-"], input_bytes=b"{}")


def test_run_aci_json_raises_timeout(monkeypatch):
    import subprocess as subprocess_mod

    monkeypatch.setattr(
        phala_tee.subprocess, "run",
        mock.Mock(side_effect=subprocess_mod.TimeoutExpired(cmd="aci", timeout=30.0)),
    )
    with pytest.raises(phala_tee.AciVerifierTimeoutError):
        phala_tee._run_aci_json(["verify", "https://inference.phala.com"])


def test_run_aci_json_raises_protocol_error_on_unparseable_output(monkeypatch):
    proc = mock.Mock(stdout=b"not json", returncode=0)
    monkeypatch.setattr(phala_tee.subprocess, "run", mock.Mock(return_value=proc))
    with pytest.raises(phala_tee.AciVerifierProtocolError):
        phala_tee._run_aci_json(["audit", "-"], input_bytes=b"{}")


def test_aci_verifier_exception_types_are_distinguishable_and_all_refuse():
    """DoD 13: missing binary, timeout, and failed verification raise
    distinguishable exception types. None of them is a subclass of another
    (besides the common AciError base) — an operator (or a caller's except
    clause) can tell them apart."""
    assert issubclass(phala_tee.AciVerifierNotFoundError, phala_tee.AciError)
    assert issubclass(phala_tee.AciVerifierTimeoutError, phala_tee.AciError)
    assert issubclass(phala_tee.AciVerifierProtocolError, phala_tee.AciError)
    assert issubclass(phala_tee.ReportVerificationError, phala_tee.AciError)
    types = {
        phala_tee.AciVerifierNotFoundError,
        phala_tee.AciVerifierTimeoutError,
        phala_tee.AciVerifierProtocolError,
        phala_tee.ReportVerificationError,
    }
    assert len(types) == 4  # all four are genuinely distinct types
    for t in types:
        assert not any(t is not other and issubclass(t, other) for other in types)


# ---------------------------------------------------------------------------
# open_e2ee_channel / E2eeChannel full seal-open flow against a verified
# aci/1 report (still no network — the "gateway" side is simulated locally
# with the service's own X25519 private key).
# ---------------------------------------------------------------------------


def test_open_e2ee_channel_refuses_unverified_report():
    report, keyset_digest, _ = _build_aci1_report()
    bad_verification = phala_tee.ReportVerification(
        ok=False,
        checks=[phala_tee.Check("aci.id-2", False, "forged")],
        workload_id=keyset_digest,
        workload_keyset_digest=keyset_digest,
    )
    with pytest.raises(phala_tee.ReportVerificationError):
        phala_tee.open_e2ee_channel(report, bad_verification)


def test_e2ee_channel_seal_and_simulated_gateway_round_trip():
    report, keyset_digest, service_priv = _build_aci1_report(e2ee_versions=["2"])
    verification = phala_tee.ReportVerification(
        ok=True, checks=[], workload_id=keyset_digest, workload_keyset_digest=keyset_digest,
    )
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


def test_open_response_decrypts_reasoning_content():
    report, keyset_digest, service_priv = _build_aci1_report(e2ee_versions=["2"])
    verification = phala_tee.ReportVerification(
        ok=True, checks=[], workload_id=keyset_digest, workload_keyset_digest=keyset_digest,
    )
    channel = phala_tee.open_e2ee_channel(report, verification)

    sealed, headers = channel.seal_messages(
        [{"role": "user", "content": "how many rs in strawberry?"}], model="some-model"
    )
    nonce_hdr = headers["X-E2EE-Nonce"]
    ts_hdr = int(headers["X-E2EE-Timestamp"])
    client_pub_raw = bytes.fromhex(headers["X-Client-Pub-Key"])
    response_id = "resp-reasoning"

    content_aad = phala_tee.response_aad(
        algo=phala_tee.E2EE_ALGO, model="some-model", id=response_id,
        field="choices.0.message.content", nonce=nonce_hdr, ts=ts_hdr,
    )
    reasoning_aad = phala_tee.response_aad(
        algo=phala_tee.E2EE_ALGO, model="some-model", id=response_id,
        field="choices.0.message.reasoning_content", nonce=nonce_hdr, ts=ts_hdr,
    )
    content_blob = phala_tee.seal_field(client_pub_raw, b"three", content_aad)
    reasoning_blob = phala_tee.seal_field(
        client_pub_raw, b"count the letters: s-t-r-a-w-b-e-r-r-y", reasoning_aad
    )

    response = {
        "id": response_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content_blob, "reasoning_content": reasoning_blob},
        }],
    }
    opened = channel.open_response(response)
    assert opened["choices"][0]["message"]["content"] == "three"
    assert opened["choices"][0]["message"]["reasoning_content"] == "count the letters: s-t-r-a-w-b-e-r-r-y"


def test_open_response_raises_on_corrupted_reasoning_content():
    report, keyset_digest, service_priv = _build_aci1_report(e2ee_versions=["2"])
    verification = phala_tee.ReportVerification(
        ok=True, checks=[], workload_id=keyset_digest, workload_keyset_digest=keyset_digest,
    )
    channel = phala_tee.open_e2ee_channel(report, verification)

    sealed, headers = channel.seal_messages(
        [{"role": "user", "content": "how many rs in strawberry?"}], model="some-model"
    )
    nonce_hdr = headers["X-E2EE-Nonce"]
    ts_hdr = int(headers["X-E2EE-Timestamp"])
    client_pub_raw = bytes.fromhex(headers["X-Client-Pub-Key"])
    response_id = "resp-reasoning-corrupt"

    content_aad = phala_tee.response_aad(
        algo=phala_tee.E2EE_ALGO, model="some-model", id=response_id,
        field="choices.0.message.content", nonce=nonce_hdr, ts=ts_hdr,
    )
    content_blob = phala_tee.seal_field(client_pub_raw, b"three", content_aad)

    reasoning_aad = phala_tee.response_aad(
        algo=phala_tee.E2EE_ALGO, model="some-model", id=response_id,
        field="choices.0.message.reasoning_content", nonce=nonce_hdr, ts=ts_hdr,
    )
    reasoning_blob = bytearray(
        bytes.fromhex(phala_tee.seal_field(client_pub_raw, b"secret reasoning", reasoning_aad))
    )
    reasoning_blob[-1] ^= 0xFF  # corrupt the GCM tag

    response = {
        "id": response_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content_blob,
                "reasoning_content": bytes(reasoning_blob).hex(),
            },
        }],
    }
    with pytest.raises(phala_tee.ReasoningContentDecryptionError) as excinfo:
        channel.open_response(response)
    assert excinfo.value.index == 0


# ---------------------------------------------------------------------------
# Live integration test — real network call to Phala; skipped (not failed)
# when PHALA_API_KEY is unset, per this repo's credentialed-integration-test
# convention (see tests/test_retrieval.py's @pytest.mark.integration test).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PhalaTeeClient timeout default + latency-warning signal
# (agents-core-phala-gate-voicing-v0)
# ---------------------------------------------------------------------------


def test_phala_tee_client_default_timeout_matches_other_deliberation_adapters():
    """Raised from 30.0 to 300.0 to match LlamaAdapter/GravityWellAdapter (both
    300s) — a 30s ceiling shredded Council turns under live 2026-08-04 testing."""
    client = phala_tee.PhalaTeeClient()
    assert client._timeout == 300.0


def test_phala_tee_client_timeout_remains_caller_overridable():
    client = phala_tee.PhalaTeeClient(timeout=45.0)
    assert client._timeout == 45.0


def test_phala_tee_client_default_nonce_is_64_hex_chars(monkeypatch):
    """B1 fix: aci/1 requires exactly 64 lowercase hex (32 bytes) — the old
    os.urandom(16).hex() default was half-length and 400'd on every call."""
    seen_nonces = []
    orig_urandom = phala_tee.os.urandom

    def spy_urandom(n):
        b = orig_urandom(n)
        return b

    monkeypatch.setattr(phala_tee.os, "urandom", spy_urandom)
    session, _resp = _stub_chat_completion_deps(monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"})
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    captured = {}
    orig_fetch = client.fetch_attestation

    def spy_fetch(*, nonce=None):
        captured["nonce"] = nonce
        return orig_fetch(nonce=nonce)

    monkeypatch.setattr(client, "fetch_attestation", spy_fetch)
    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")
    assert len(captured["nonce"]) == 64
    bytes.fromhex(captured["nonce"])  # must be valid hex


def _stub_chat_completion_deps(monkeypatch, *, capabilities, applied=True):
    """Stub the attestation/e2ee machinery so chat_completion's HTTP+locality
    path can be exercised without real crypto or network. Returns the mock
    session whose .post/.get the test controls."""
    fake_verification = mock.MagicMock(ok=True, workload_keyset_digest="deadbeef", custody_skipped=True)
    fake_channel = mock.MagicMock()
    fake_channel.seal_messages.return_value = ([{"role": "user", "content": "sealed"}], {"X-E2EE-Version": "2"})
    fake_channel.open_response.side_effect = lambda resp: resp

    monkeypatch.setattr(phala_tee, "require_verified_report_binding",
                         lambda report, nonce, base_url=None: fake_verification)
    monkeypatch.setattr(phala_tee, "open_e2ee_channel",
                         lambda report, verification: fake_channel)

    session = mock.MagicMock()
    resp = mock.MagicMock()
    resp.raise_for_status = mock.MagicMock()
    resp.headers = {"x-e2ee-applied": "true" if applied else "false"}
    resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    session.post.return_value = resp
    session.get.return_value.raise_for_status = mock.MagicMock()
    session.get.return_value.json.return_value = {
        "attestation": {},
        "service_capabilities": capabilities,
    }
    return session, resp


def test_phala_tee_client_latency_warning_flagged_over_old_ceiling(monkeypatch):
    """A call that would have failed under the retired 30s ceiling (>30000ms)
    must carry extra.latency_warning=True — a distinct signal so a slow seat
    stays discoverable rather than silently absorbed by the timeout raise."""
    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": ["2"], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(session=session)

    times = iter([0.0, 45.0])  # start, end -> duration_ms = 45000
    monkeypatch.setattr(phala_tee.time, "monotonic", lambda: next(times))

    recorded = {}
    monkeypatch.setattr(
        phala_tee.locality, "record",
        lambda **kw: recorded.update(kw),
    )

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    assert recorded["extra"]["latency_warning"] is True


def test_phala_tee_client_no_latency_warning_under_old_ceiling(monkeypatch):
    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": ["2"], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(session=session)

    times = iter([0.0, 5.0])  # duration_ms = 5000, well under 30000
    monkeypatch.setattr(phala_tee.time, "monotonic", lambda: next(times))

    recorded = {}
    monkeypatch.setattr(
        phala_tee.locality, "record",
        lambda **kw: recorded.update(kw),
    )

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    assert recorded["extra"]["latency_warning"] is False


# ---------------------------------------------------------------------------
# Piece 3 — capability-gated e2ee, honest locality rows (DoD 6, 7, 8).
# ---------------------------------------------------------------------------


def test_chat_completion_seals_when_gateway_advertises_supported_version(monkeypatch):
    session, resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": ["2"], "serving": "aggregator"}, applied=True
    )
    client = phala_tee.PhalaTeeClient(session=session)

    recorded = {}
    monkeypatch.setattr(phala_tee.locality, "record", lambda **kw: recorded.update(kw))

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    posted_body = session.post.call_args.kwargs["json"]
    assert posted_body["messages"] == [{"role": "user", "content": "sealed"}]
    assert recorded["extra"]["e2ee_applied"] is True
    assert recorded["extra"]["channel"] == "e2ee"


def test_chat_completion_skips_sealing_and_does_not_raise_when_capability_empty(monkeypatch):
    session, resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}, applied=False
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    recorded = {}
    monkeypatch.setattr(phala_tee.locality, "record", lambda **kw: recorded.update(kw))

    messages = [{"role": "user", "content": "hi"}]
    # Must NOT raise E2eeNotAppliedError even though x-e2ee-applied is absent/false.
    client.chat_completion(messages=messages, model="m")

    posted_body = session.post.call_args.kwargs["json"]
    assert posted_body["messages"] == messages  # sent plaintext, unmodified
    assert recorded["extra"]["e2ee_applied"] is False
    assert recorded["extra"]["channel"] == "plaintext-loopback"
    assert recorded["extra"]["custody_unverified"] is True
    assert recorded["ok"] is True  # ok is the call-completed claim, never collapsed with e2ee_applied


def test_chat_completion_raises_when_capability_empty_and_base_url_not_loopback(monkeypatch):
    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url=phala_tee.DEFAULT_BASE_URL)

    with pytest.raises(phala_tee.PlaintextChannelNotLoopbackError):
        client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    session.post.assert_not_called()  # plaintext must never leave the process off-box


def test_chat_completion_raises_e2ee_not_applied_when_capability_advertised_but_not_applied(monkeypatch):
    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": ["2"], "serving": "aggregator"}, applied=False
    )
    client = phala_tee.PhalaTeeClient(session=session)

    with pytest.raises(phala_tee.E2eeNotAppliedError):
        client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PHALA_API_KEY"),
    reason="PHALA_API_KEY not set — live Phala ACI integration test skipped",
)
def test_integration_phala_live_round_trip_and_forged_report_rejected():
    """Run locally:  PHALA_API_KEY=... pytest -m integration tests/test_phala_tee.py

    NOTE: as of this restoration, live end-to-end verification depends on
    Piece 1 (the vendored `aci` binary, host-provided, outside this repo) —
    see agents-core-phala-aci1-restoration-v0 spec. Not a worker DoD item."""
    model = os.environ.get("PHALA_MODEL", "deepseek-ai/DeepSeek-V3")
    client = phala_tee.PhalaTeeClient()

    response = client.chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly the word: acknowledged"}],
        model=model,
    )
    content = response["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip()

    # A deliberately forged attestation report must be rejected before any
    # key material is used.
    nonce = "c" * 64
    real_report = client.fetch_attestation(nonce=nonce)
    import copy

    forged = copy.deepcopy(real_report)
    forged["attestation"]["report_data"] = "0" * 64
    with pytest.raises(phala_tee.ReportVerificationError):
        phala_tee.require_verified_report_binding(forged, nonce)
