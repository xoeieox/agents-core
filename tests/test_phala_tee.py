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
import json
import os
import time
import unittest.mock as mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from agents_core import phala_tee

# conftest.py's autouse `_locality_ledger_isolated` fixture already points
# LOCALITY_LEDGER_ROOT at a per-test tmp_path — chat_completion's new spend-
# cap check (agents-core-phala-spend-cap-v0) reads that same isolated root,
# so no additional isolation is needed here.


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


def _build_aci1_report(*, e2ee_versions=None, not_after_offset=3600):
    """A synthetic aci/1-shaped attestation report — no workload_id,
    workload_identity, keyset_endorsement, or keyset_epoch, matching the
    live wire format captured 2026-08-07 (finding/brix-phala-aci1-break-
    confirmed-and-restoration-in-flight-2026-08-07). `not_after_offset`
    (default 3600, unchanged behavior) makes the keyset's wall-clock
    validity bound testable for the verified-attestation cache's
    not_after clamp (DoD-3, DoD-9)."""
    service_priv = X25519PrivateKey.generate()
    service_pub_hex = service_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    keyset = {
        "e2ee_public_keys": [{"algo": phala_tee.E2EE_ALGO, "public_key": service_pub_hex}],
        "not_after": int(time.time()) + not_after_offset,
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


def _transcript(*, checks, keyset_digest=None, nonce=None, exit_code=0, verified=True, failed=0):
    """Build a transcript matching the REAL `aci --json` shape (Defect B):
    exactly two top-level keys, `checks` and `verdict`. `verdict` carries
    `workload_keyset_digest` (not top-level `keyset_digest`/
    `workload_keyset_digest`), and there is no structured `nonce` field
    anywhere — a `nonce` embeds into id-2's `detail` string instead, the
    only place the real binary puts it
    ('statement digest for nonce "<nonce>" matches report_data')."""
    checks_list = []
    for cid, status in checks:
        detail = None
        if cid == "id-2" and nonce is not None:
            detail = f'statement digest for nonce "{nonce}" matches report_data'
        checks_list.append({"id": cid, "status": status, "detail": detail})
    verdict = {
        "verified": verified,
        "passed": sum(1 for _, s in checks if s == "pass"),
        "failed": failed,
        "skipped": sum(1 for _, s in checks if s == "skip"),
    }
    if keyset_digest is not None:
        verdict["workload_keyset_digest"] = keyset_digest
    return {"checks": checks_list, "verdict": verdict, "_exit_code": exit_code}


# Old name kept as an alias so any external callers aren't broken by the
# rename to `_transcript` (Defect B: the old name/shape encoded the bug).
_passing_transcript = _transcript


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
    # phala-attestation-cache-v0: reset the verified-bundle cache too —
    # chat_completion tests must start from a clean module-level store.
    monkeypatch.setattr(phala_tee, "_attestation_cache", {})
    return calls


def _default_transcripts(*, keyset_digest, nonce):
    # exit_code=1 on the offline leg matches the REAL binary (Defect A):
    # `aci audit` exits 1 whenever the verdict isn't VERIFIED, and an
    # offline audit can never be VERIFIED (id-1/id-6 need online/live-TLS
    # data an offline audit doesn't have — they're structurally always
    # skipped here). Zero failed, all three required checks pass: this is
    # the measured real-binary shape, and must still verify.
    audit = _passing_transcript(
        checks=[
            ("id-1", "skip"), ("id-2", "pass"), ("id-3", "pass"),
            ("id-4", "pass"), ("id-5", "skip"), ("id-6", "skip"),
        ],
        keyset_digest=keyset_digest,
        nonce=nonce,
        exit_code=1,
        verified=False,
        failed=0,
    )
    verify = _passing_transcript(
        checks=[("id-1", "pass"), ("id-5", "skip"), ("id-6", "pass")],
        exit_code=0,
        verified=False,  # id-5 skipped, not "pass" -> verdict.verified is False too
        failed=0,
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


def test_verify_report_binding_accepts_non_zero_exit_when_required_checks_pass_and_none_failed(monkeypatch):
    """Defect A: `aci audit` exits 1 whenever the verdict isn't VERIFIED,
    and an offline audit can NEVER be VERIFIED (id-1/id-6 need online/
    live-TLS data it structurally doesn't have). Gating must not use the
    exit code — `_default_transcripts` already encodes exactly this shape
    (audit exit=1, verdict.failed=0, id-2/3/4 pass), so this is really an
    assertion that the happy path from `_default_transcripts` verifies,
    named explicitly so the exit-code trap can't silently regress."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    assert audit["_exit_code"] != 0  # sanity: this is the structurally-can't-pass leg
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is True


@pytest.mark.parametrize("leg", ["audit", "verify"])
def test_verify_report_binding_rejects_verdict_failed_nonzero_even_with_zero_exit(monkeypatch, leg):
    """Defect A, the flip side: `verdict.failed > 0` must gate the result
    even when the leg's own exit code is 0 — exit code is informational
    only, never load-bearing in either direction."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    if leg == "audit":
        audit["_exit_code"] = 0
        audit["verdict"]["failed"] = 1
    else:
        verify["_exit_code"] = 0
        verify["verdict"]["failed"] = 1
    _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    verification = phala_tee.verify_report_binding(report, nonce)
    assert verification.ok is False
    failed = {c.name for c in verification.checks if not c.ok}
    assert f"aci_{leg}.no_failed_checks" in failed


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


@pytest.mark.parametrize("loopback_url", ["http://127.0.0.1:4180", "http://localhost:4180", "http://[::1]:4180"])
def test_verify_report_binding_refuses_loopback_verification_target(monkeypatch, loopback_url):
    """Defect C: a verification target resolving to loopback must be
    refused fail-closed, naming both settings — verifying the local
    traffic-routing hop instead of the real upstream is the exact
    misconfiguration that made id-6 fail to bind (plain HTTP, no TLS
    handshake to bind to)."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    audit, verify = _default_transcripts(keyset_digest=keyset_digest, nonce=nonce)
    calls = _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    with pytest.raises(phala_tee.LoopbackVerificationTargetError) as excinfo:
        phala_tee.verify_report_binding(report, nonce, base_url=loopback_url)
    assert "PHALA_BASE_URL" in str(excinfo.value)
    assert phala_tee.PHALA_VERIFY_TARGET_ENV in str(excinfo.value)
    # Must fail before ever shelling out to either leg.
    assert calls == {"audit": 0, "verify": 0}


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
# Contract test against the REAL `aci` binary (agents-core-aci-audit-cli-
# interface-fix-v0, leg 3). Every other test above mocks at the
# `_run_aci_json` boundary, which is correct for logic tests but structurally
# cannot catch an interface mismatch between the argv this module builds and
# the argv the real binary accepts — that gap is exactly what shipped the
# stdin-form `aci audit` defect this unit fixes. This test asserts argv
# *parses*, never a verification verdict: reaching report-schema validation
# on a stub report is a PASS (it proves the args were accepted), and a clap
# usage error (exit 2, "unexpected argument" / "For more information, try
# '--help'") is a FAIL.
# ---------------------------------------------------------------------------

import subprocess as _subprocess_mod  # noqa: E402 (test-local, avoid clash with mocked module)


def _resolve_real_aci_binary():
    binary = os.environ.get(phala_tee.ACI_VERIFIER_BIN_ENV) or phala_tee.DEFAULT_ACI_VERIFIER_BIN
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        pytest.skip(
            f"real `aci` binary not found at {binary!r} — set "
            f"{phala_tee.ACI_VERIFIER_BIN_ENV} to a built `aci` binary to run this contract test"
        )
    return binary


def _is_clap_usage_error(returncode, stderr_text):
    if returncode != 2:
        return False
    return "unexpected argument" in stderr_text or (
        "error:" in stderr_text and "For more information, try '--help'" in stderr_text
    )


def test_real_binary_accepts_audit_report_flag_argv(tmp_path):
    """The argv this module now builds — `audit --report <FILE> --nonce
    <NONCE> --json` — must be accepted (not a clap usage error) by the real
    binary. A stub report failing schema validation is the expected,
    PASSING outcome here; it proves the args parsed."""
    binary = _resolve_real_aci_binary()
    report_path = tmp_path / "report.json"
    report_path.write_bytes(b"{}")

    proc = _subprocess_mod.run(
        [binary, "audit", "--report", str(report_path), "--nonce", "a" * 64, "--json"],
        capture_output=True, timeout=30.0,
    )
    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    assert not _is_clap_usage_error(proc.returncode, stderr_text), (
        f"real `aci audit --report ...` argv was rejected as a usage error "
        f"(exit {proc.returncode}): {stderr_text!r}"
    )


def test_real_binary_accepts_verify_argv():
    """The `verify <BASE_URL>` leg is already correct; this is what keeps it
    that way. `--upstream` unreachable is fine (network/connect failure, not
    a usage error); a clap usage error is not."""
    binary = _resolve_real_aci_binary()

    proc = _subprocess_mod.run(
        [binary, "verify", "http://127.0.0.1:1", "--json"],
        capture_output=True, timeout=30.0,
    )
    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    assert not _is_clap_usage_error(proc.returncode, stderr_text), (
        f"real `aci verify <BASE_URL>` argv was rejected as a usage error "
        f"(exit {proc.returncode}): {stderr_text!r}"
    )


def test_real_binary_rejects_the_old_broken_stdin_audit_argv(tmp_path):
    """Negative case (DoD 8): the OLD, broken form — a positional `-` and no
    `--report` — must actually fail argument parsing against the real
    binary, so this contract test is proven to discriminate rather than
    passing vacuously."""
    binary = _resolve_real_aci_binary()

    proc = _subprocess_mod.run(
        [binary, "audit", "-", "--nonce", "a" * 64, "--json"],
        input=b"{}", capture_output=True, timeout=30.0,
    )
    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    assert _is_clap_usage_error(proc.returncode, stderr_text), (
        f"expected the old stdin-form argv to fail clap argument parsing, "
        f"got exit {proc.returncode}, stderr: {stderr_text!r}"
    )


def test_real_binary_audit_transcript_shape_matches_what_module_reads(tmp_path):
    """Defect D: the argv contract tests above prove the CLI *accepts* our
    arguments; they say nothing about the *shape* of what it hands back.
    That gap is exactly what shipped Defect B (this module read
    `keyset_digest`/`workload_keyset_digest`/`nonce` at the top level and a
    structured nonce field that never existed). Capture a REAL transcript
    from the real binary — never a hand-written fixture asserting our own
    assumptions, which would just reproduce the bug this test exists to
    catch — and assert every field `verify_report_binding` actually reads
    is present where it reads it."""
    binary = _resolve_real_aci_binary()
    report, _keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    proc = _subprocess_mod.run(
        [binary, "audit", "--report", str(report_path), "--nonce", nonce, "--json"],
        capture_output=True, timeout=30.0,
    )
    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    assert not _is_clap_usage_error(proc.returncode, stderr_text), (
        f"real `aci audit` argv rejected as a usage error (exit "
        f"{proc.returncode}): {stderr_text!r}"
    )
    transcript = json.loads(proc.stdout.decode("utf-8"))

    # Top-level shape: exactly what the code destructures.
    assert set(transcript.keys()) == {"checks", "verdict"}, (
        f"real transcript top-level keys {sorted(transcript.keys())} != "
        f"{{'checks', 'verdict'}} — this module's field reads must be updated"
    )

    verdict = transcript["verdict"]
    assert isinstance(verdict, dict)
    assert "failed" in verdict, "verdict.failed is what gating reads (Defect A)"
    assert isinstance(verdict["failed"], int)
    assert "workload_keyset_digest" in verdict, (
        "verdict.workload_keyset_digest is what the digest-binding check reads (Defect B)"
    )

    checks = transcript["checks"]
    assert isinstance(checks, list) and checks
    for entry in checks:
        assert "id" in entry and "status" in entry and "detail" in entry, (
            f"check entry missing id/status/detail: {entry!r}"
        )

    id2 = next((c for c in checks if c.get("id") == "id-2"), None)
    assert id2 is not None, "id-2 (nonce binding statement) must be present in the audit transcript"
    assert isinstance(id2.get("detail"), str) and nonce in id2["detail"], (
        f"expected nonce {nonce!r} inside id-2's detail, got {id2.get('detail')!r} — "
        f"this is the only place the audited nonce is exposed (Defect B)"
    )


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


# ---------------------------------------------------------------------------
# PhalaTeeClient verify_target / PHALA_BASE_URL split (Defect C).
# ---------------------------------------------------------------------------


def test_phala_tee_client_verify_target_defaults_to_default_base_url():
    """The verification target must default to the real upstream, distinct
    from base_url, even when base_url is pinned to the loopback hop."""
    client = phala_tee.PhalaTeeClient(base_url="http://127.0.0.1:4180")
    assert client._base_url == "http://127.0.0.1:4180"
    assert client._verify_target == phala_tee.DEFAULT_BASE_URL


def test_phala_tee_client_verify_target_env_fallback(monkeypatch):
    monkeypatch.setenv(phala_tee.PHALA_VERIFY_TARGET_ENV, "https://verify.example.com")
    client = phala_tee.PhalaTeeClient()
    assert client._verify_target == "https://verify.example.com"


def test_phala_tee_client_verify_target_explicit_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv(phala_tee.PHALA_VERIFY_TARGET_ENV, "https://from-env.example.com")
    client = phala_tee.PhalaTeeClient(verify_target="https://from-arg.example.com")
    assert client._verify_target == "https://from-arg.example.com"


def test_chat_completion_passes_verify_target_not_base_url_to_report_binding(monkeypatch):
    """The online verify leg must audit the verification target, never the
    loopback traffic route — this is the exact split Defect C fixes."""
    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(
        session=session, base_url="http://127.0.0.1:4180", verify_target="https://real.example.com",
    )

    captured = {}

    def spy_require_verified(report, nonce, base_url=None):
        captured["base_url"] = base_url
        return mock.MagicMock(ok=True, workload_keyset_digest="deadbeef", custody_skipped=True)

    monkeypatch.setattr(phala_tee, "require_verified_report_binding", spy_require_verified)

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")
    assert captured["base_url"] == "https://real.example.com"


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


# ---------------------------------------------------------------------------
# Spend cap (Leg 2) + reconciliation (Leg 3) — agents-core-phala-spend-cap-v0.
# ---------------------------------------------------------------------------


def _seed_ledger_cost(cost_usd, *, cost_class="paid-phala-tee"):
    """Write one locality ledger row directly (bypassing chat_completion) so
    cap/reconciliation tests can pre-load today's recorded spend."""
    phala_tee.locality.record(
        requested_operator="phala-tee",
        served_model="m",
        host="phala",
        cost_class=cost_class,
        seam="phala_tee",
        cost_usd=cost_usd,
        ok=True,
    )


def test_chat_completion_threads_usage_cost_and_token_counts_into_ledger(monkeypatch):
    """Leg 1: usage.cost -> cost_usd, plus prompt/completion/total/reasoning
    token counts into extra — the response's own meter, not an estimate."""
    session, resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}, applied=False
    )
    resp.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 10,
            "total_tokens": 17,
            "reasoning_tokens": 0,
            "cost": 5.4e-06,
        },
    }
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    recorded = {}
    monkeypatch.setattr(phala_tee.locality, "record", lambda **kw: recorded.update(kw))

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    assert recorded["cost_usd"] == 5.4e-06
    assert recorded["extra"]["prompt_tokens"] == 7
    assert recorded["extra"]["completion_tokens"] == 10
    assert recorded["extra"]["total_tokens"] == 17
    assert recorded["extra"]["reasoning_tokens"] == 0
    assert recorded["extra"]["cost_unpriced"] is False


def test_chat_completion_records_none_cost_and_marker_when_usage_absent(monkeypatch):
    """DoD 2: no usage at all -> cost_usd=None, distinct marker, never a
    silent zero."""
    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}, applied=False
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    recorded = {}
    monkeypatch.setattr(phala_tee.locality, "record", lambda **kw: recorded.update(kw))

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    assert recorded["cost_usd"] is None
    assert recorded["extra"]["cost_unpriced"] is True
    assert recorded["ok"] is True  # other fields unchanged


def test_chat_completion_records_none_cost_when_cost_field_missing_but_tokens_present(monkeypatch):
    """usage present without `cost` (e.g. a non-metered path some day) must
    still record cost_usd=None + marker, never fabricate a cost."""
    session, resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}, applied=False
    )
    resp.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    recorded = {}
    monkeypatch.setattr(phala_tee.locality, "record", lambda **kw: recorded.update(kw))

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    assert recorded["cost_usd"] is None
    assert recorded["extra"]["cost_unpriced"] is True
    assert recorded["extra"]["prompt_tokens"] == 3


def test_spend_cap_boundary_exactly_at_cap_refuses(monkeypatch):
    """DoD 11 boundary case: recorded total == cap must refuse (any further
    call would exceed it)."""
    monkeypatch.setenv("PHALA_DAILY_SPEND_CAP_USD", "2.00")
    _seed_ledger_cost(2.00)

    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    with pytest.raises(phala_tee.PhalaSpendCapExceededError):
        client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")
    session.get.assert_not_called()  # refused before any network I/O, including attestation
    session.post.assert_not_called()


def test_spend_cap_under_cap_allows_call(monkeypatch):
    monkeypatch.setenv("PHALA_DAILY_SPEND_CAP_USD", "2.00")
    _seed_ledger_cost(1.00)

    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}, applied=False
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")
    session.post.assert_called_once()


def test_spend_cap_above_cap_refuses_with_distinct_named_exception(monkeypatch):
    """DoD 6: a distinct exception, not a generic/timeout error, and no
    silent fallback to another operator (the caller never gets a response)."""
    monkeypatch.setenv("PHALA_DAILY_SPEND_CAP_USD", "2.00")
    _seed_ledger_cost(3.00)

    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    with pytest.raises(phala_tee.PhalaSpendCapExceededError) as excinfo:
        client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")
    assert excinfo.value.spent_usd == 3.00
    assert excinfo.value.cap_usd == 2.00
    assert not issubclass(phala_tee.PhalaSpendCapExceededError, TimeoutError)
    session.post.assert_not_called()


def test_spend_cap_refusal_notifies_pushover_once_per_day(monkeypatch):
    """DoD 7: at most one Pushover per day, not one per blocked call."""
    monkeypatch.setenv("PHALA_DAILY_SPEND_CAP_USD", "2.00")
    _seed_ledger_cost(3.00)

    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    calls = []
    monkeypatch.setattr(
        "agents_core.notify.send_notification",
        lambda *a, **kw: calls.append((a, kw)) or True,
    )

    for _ in range(3):
        with pytest.raises(phala_tee.PhalaSpendCapExceededError):
            client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    assert len(calls) == 1


def test_spend_cap_refusal_surfaces_unknown_total_when_unpriced_rows_present(monkeypatch):
    """DoD 8: unpriced rows in the window are surfaced as unknown, in both
    the raised exception and the notification — never folded into a
    falsely confident total."""
    monkeypatch.setenv("PHALA_DAILY_SPEND_CAP_USD", "2.00")
    _seed_ledger_cost(3.00)
    _seed_ledger_cost(None)  # unpriced row

    session, _resp = _stub_chat_completion_deps(
        monkeypatch, capabilities={"supported_e2ee_versions": [], "serving": "aggregator"}
    )
    client = phala_tee.PhalaTeeClient(session=session, base_url="http://127.0.0.1:4180")

    calls = []
    monkeypatch.setattr(
        "agents_core.notify.send_notification",
        lambda *a, **kw: calls.append((a, kw)) or True,
    )

    with pytest.raises(phala_tee.PhalaSpendCapExceededError) as excinfo:
        client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m")

    assert excinfo.value.unpriced_count == 1
    assert "unpriced" in str(excinfo.value).lower()
    assert calls and "unpriced" in calls[0][0][0].lower()


def test_spend_cap_default_is_two_dollars_overridable_by_env(monkeypatch):
    monkeypatch.delenv("PHALA_DAILY_SPEND_CAP_USD", raising=False)
    assert phala_tee._phala_daily_spend_cap_usd() == 2.00
    monkeypatch.setenv("PHALA_DAILY_SPEND_CAP_USD", "0.05")
    assert phala_tee._phala_daily_spend_cap_usd() == 0.05


def test_reconcile_daily_spend_reports_material_divergence_without_gating(monkeypatch):
    """Leg 3 must report, never gate — no exception, no refusal."""
    _seed_ledger_cost(1.00)
    result = phala_tee.reconcile_daily_spend(2.50)
    assert result["ledger_total_usd"] == 1.00
    assert result["divergence_usd"] == pytest.approx(-1.50)
    assert result["material_divergence"] is True


def test_reconcile_daily_spend_no_divergence_when_matching(monkeypatch):
    _seed_ledger_cost(1.00)
    result = phala_tee.reconcile_daily_spend(1.00)
    assert result["material_divergence"] is False


def test_reconcile_daily_spend_unknown_provider_value_still_reports_ledger_total(monkeypatch):
    _seed_ledger_cost(1.00)
    result = phala_tee.reconcile_daily_spend(None)
    assert result["ledger_total_usd"] == 1.00
    assert result["divergence_usd"] is None
    assert result["material_divergence"] is False


# ---------------------------------------------------------------------------
# Verified-attestation bundle cache (phala-attestation-cache-v0, 2026-09-14).
#
# These tests use the REAL `verify_report_binding` + `_install_aci_cli_stub`
# (the call-counting fake at the single subprocess boundary, which resets
# both caches) + a MagicMock session counting `.get`/`.post` — NOT
# `_stub_chat_completion_deps`, which stubs `require_verified_report_binding`
# out (audit count 0, defeating the DoD-1 discriminator). The discriminators
# are the attestation GET count and the audit-leg count: the online-leg cache
# alone would also give verify==1, so GET + audit are what prove the
# verified bundle was served from the cache.
# ---------------------------------------------------------------------------


def _http_error(status_code):
    import requests as _requests

    resp = mock.Mock()
    resp.status_code = status_code
    return _requests.HTTPError(response=resp)


def _make_cache_client(monkeypatch, *, report, keyset_digest, nonce, session, base_url="http://127.0.0.1:4180"):
    """Wire the real two-leg verify against a MagicMock session and build a
    client pinned at loopback (plaintext path, no e2ee sealing). The audit
    transcript is nonce-bound, so a hit (which never re-runs the audit leg)
    is the only thing that keeps the audit count at 1."""
    audit = _passing_transcript(
        checks=[
            ("id-1", "skip"), ("id-2", "pass"), ("id-3", "pass"),
            ("id-4", "pass"), ("id-5", "skip"), ("id-6", "skip"),
        ],
        keyset_digest=keyset_digest,
        nonce=nonce,
        exit_code=1,
        verified=False,
        failed=0,
    )
    verify = _passing_transcript(
        checks=[("id-1", "pass"), ("id-5", "skip"), ("id-6", "pass")],
        exit_code=0,
        verified=False,
        failed=0,
    )
    calls = _install_aci_cli_stub(monkeypatch, audit_transcript=audit, verify_transcript=verify)

    resp = mock.MagicMock()
    resp.raise_for_status = mock.MagicMock()
    resp.headers = {}
    resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    session.post.return_value = resp
    get_resp = mock.MagicMock()
    get_resp.raise_for_status = mock.MagicMock()
    get_resp.json.return_value = report
    session.get.return_value = get_resp

    client = phala_tee.PhalaTeeClient(session=session, base_url=base_url, verify_target="https://real.example.com")
    return client, calls


def test_attestation_cache_hit_skips_fetch_and_both_legs(monkeypatch):
    """DoD-1: two chat_completion calls within ttl produce exactly ONE
    attestation GET and ONE audit leg (the amortization proof). The
    discriminators are the GET count and the audit count — the online-leg
    cache alone would also give verify==1, so GET + audit are what prove
    the verified bundle was served from the cache."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    assert session.get.call_count == 1
    assert session.post.call_count == 1
    assert calls == {"audit": 1, "verify": 1}

    client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)
    assert session.get.call_count == 1  # hit: no second attestation GET
    assert session.post.call_count == 2
    assert calls["audit"] == 1  # hit: no second audit leg
    assert calls["verify"] == 1


def test_attestation_cache_miss_after_ttl_refetches_and_reinserts(monkeypatch):
    """DoD-2: advancing the fake monotonic clock past ttl makes the second
    call a miss — it re-fetches, re-audits, and re-inserts."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    clock = [1000.0]
    monkeypatch.setattr(phala_tee.time, "monotonic", lambda: clock[0])

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    assert session.get.call_count == 1
    assert calls["audit"] == 1

    # Past the default 900s ttl (well under the 3600s ceiling).
    clock[0] = 1000.0 + 901.0
    client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)
    assert session.get.call_count == 2  # miss: re-fetches
    assert calls["audit"] == 2  # miss: re-runs the audit leg
    assert session.post.call_count == 2

    # Re-inserted: a call back inside ttl is a hit again.
    clock[0] = 1000.0 + 901.0 + 10.0
    client.chat_completion(messages=[{"role": "user", "content": "third"}], model="m", nonce=nonce)
    assert session.get.call_count == 2
    assert calls["audit"] == 2


def test_attestation_cache_not_after_clamp_wins_over_ttl(monkeypatch):
    """DoD-3: an entry stored with not_after 60s out while ttl is 900 must
    be a MISS at t+120s — the wall-clock clamp wins over ttl. The report
    is built AFTER the clock patch so not_after is in the fake domain."""
    clock = [1000.0]
    monkeypatch.setattr(phala_tee.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(phala_tee.time, "time", lambda: clock[0])

    report, keyset_digest, _ = _build_aci1_report(not_after_offset=60)
    nonce = "a" * 64
    session = mock.MagicMock()
    client, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    assert session.get.call_count == 1
    assert calls["audit"] == 1

    # +120s: still inside the 900s ttl, but past not_after (+60s) -> MISS.
    clock[0] = 1120.0
    client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)
    assert session.get.call_count == 2
    assert calls["audit"] == 2


def test_attestation_cache_fail_closed_invalidation_on_aci_error(monkeypatch):
    """DoD-4: the first call succeeds and caches; the second call is a
    forced miss that fails with AciError -> the cache entry is gone,
    chat_completion raises, and no HTTP POST was issued on the failing
    call."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    assert session.post.call_count == 1
    assert (client._base_url, client._verify_target) in phala_tee._attestation_cache

    # Force a miss, then make the audit leg fail with a toolchain fault.
    phala_tee.invalidate_attestation_cache()
    monkeypatch.setattr(
        phala_tee, "_run_aci_json",
        mock.Mock(side_effect=phala_tee.AciVerifierTimeoutError("aci", 30.0)),
    )

    with pytest.raises(phala_tee.AciVerifierTimeoutError):
        client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)

    assert (client._base_url, client._verify_target) not in phala_tee._attestation_cache
    assert session.post.call_count == 1  # no POST issued on the failing call


def test_attestation_cache_post_reject_pops_entry_and_reraises(monkeypatch):
    """M2 pop-list: a chat-completion POST rejected with 401 (keyset
    rotation rejected at the hop) pops the cached entry and re-raises
    fail-closed — the next call re-verifies fresh instead of reusing a
    stale bundle for the full ttl."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client, _calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    key = (client._base_url, client._verify_target)
    assert key in phala_tee._attestation_cache

    session.post.side_effect = _http_error(401)
    with pytest.raises(Exception):
        client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)
    assert key not in phala_tee._attestation_cache  # popped on the 401

    # Transient statuses do NOT pop: a fresh store survives a 502.
    session.post.side_effect = None
    client.chat_completion(messages=[{"role": "user", "content": "fresh"}], model="m", nonce=nonce)
    assert key in phala_tee._attestation_cache
    session.post.side_effect = _http_error(502)
    with pytest.raises(Exception):
        client.chat_completion(messages=[{"role": "user", "content": "x"}], model="m", nonce=nonce)
    assert key in phala_tee._attestation_cache  # still there after a 5xx


def test_attestation_cache_ledger_honesty_hit_and_miss_rows(monkeypatch):
    """DoD-5: the locality row's existing fields are unchanged; cache status
    is additive in extra only. Hit rows carry attestation_cache_hit=True +
    a non-negative age; miss rows carry hit=False + age None."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client, _calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    rows = []
    monkeypatch.setattr(phala_tee.locality, "record", lambda **kw: rows.append(kw))

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)

    assert len(rows) == 2
    miss_extra, hit_extra = rows[0]["extra"], rows[1]["extra"]

    # Existing row fields unchanged (spend-cap doctrine).
    for extra in (miss_extra, hit_extra):
        assert extra["report_verified"] is True
        assert extra["custody_unverified"] is True
        assert extra["e2ee_applied"] is False
        assert extra["channel"] == "plaintext-loopback"

    assert miss_extra["attestation_cache_hit"] is False
    assert miss_extra["attestation_cache_age_s"] is None
    assert hit_extra["attestation_cache_hit"] is True
    assert isinstance(hit_extra["attestation_cache_age_s"], float)
    assert hit_extra["attestation_cache_age_s"] >= 0.0


def test_attestation_cache_ttl_zero_disables_cache_but_never_verification(monkeypatch):
    """DoD-6: ttl<=0 disables the cache (every call fetches + audits) but
    never verification — the loopback-target refusal still fires."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)
    monkeypatch.setenv(phala_tee.PHALA_ATTESTATION_CACHE_TTL_ENV, "0")

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)

    assert session.get.call_count == 2  # cache disabled: every call fetches
    assert calls["audit"] == 2  # ... and re-audits
    assert len(phala_tee._attestation_cache) == 0  # never stored

    # The loopback verify-target refusal is untouched (existing behavior).
    with pytest.raises(phala_tee.LoopbackVerificationTargetError):
        phala_tee.verify_report_binding(report, nonce, base_url="http://127.0.0.1:4180")


def test_invalidate_attestation_cache_returns_drop_count_and_forces_reverify(monkeypatch):
    """DoD-7: invalidate_attestation_cache() returns the drop count and
    forces a fresh verify on the next call (canary hook)."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    assert len(phala_tee._attestation_cache) == 1

    assert phala_tee.invalidate_attestation_cache() == 1
    assert len(phala_tee._attestation_cache) == 0

    client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)
    assert session.get.call_count == 2  # forced fresh verify, not a hit
    assert calls["audit"] == 2

    # Dimension filters: None = wildcard, a value = exact match.
    assert phala_tee.invalidate_attestation_cache(base_url="http://nope.example.com") == 0
    assert len(phala_tee._attestation_cache) == 1
    assert phala_tee.invalidate_attestation_cache(base_url=client._base_url) == 1
    assert len(phala_tee._attestation_cache) == 0


def test_attestation_cache_shared_across_client_instances(monkeypatch):
    """DoD-8: two chat_completion calls via TWO DISTINCT PhalaTeeClient
    instances within ttl still yield session .get count == 1 — the store is
    MODULE-level (process-wide). An instance-level cache would give
    .get == 2 and never amortize on the node2 per-pass client."""
    report, keyset_digest, _ = _build_aci1_report()
    nonce = "a" * 64
    session = mock.MagicMock()
    client_a, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    client_a.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    assert session.get.call_count == 1
    assert calls["audit"] == 1

    client_b = phala_tee.PhalaTeeClient(session=session, base_url=client_a._base_url, verify_target=client_a._verify_target)
    client_b.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)
    assert session.get.call_count == 1  # hit across instances
    assert calls["audit"] == 1
    assert session.post.call_count == 2


def test_attestation_cache_ttl_ceiling_in_expression(monkeypatch):
    """DoD-9: ttl_env=86400 is clamped to the 3600s ceiling IN the
    expression — a second call at t+3601s is a MISS (effective ttl ==
    3600, not 86400). not_after is far out so the wall-clock term does not
    interfere (the ceiling is the binding clamp). The report is built AFTER
    the clock patch so not_after is in the fake domain."""
    clock = [1000.0]
    monkeypatch.setattr(phala_tee.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(phala_tee.time, "time", lambda: clock[0])
    monkeypatch.setenv(phala_tee.PHALA_ATTESTATION_CACHE_TTL_ENV, "86400")

    report, keyset_digest, _ = _build_aci1_report(not_after_offset=86400)
    nonce = "a" * 64
    session = mock.MagicMock()
    client, calls = _make_cache_client(monkeypatch, report=report, keyset_digest=keyset_digest, nonce=nonce, session=session)

    client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="m", nonce=nonce)
    assert session.get.call_count == 1
    assert calls["audit"] == 1

    # t+3601s: past the 3600s ceiling -> MISS (would be a hit if the
    # ceiling were not in the clamp expression).
    clock[0] = 1000.0 + 3601.0
    client.chat_completion(messages=[{"role": "user", "content": "again"}], model="m", nonce=nonce)
    assert session.get.call_count == 2
    assert calls["audit"] == 2


def test_attestation_cache_ttl_env_read_and_clamped():
    assert phala_tee._attestation_cache_ttl_secs() == phala_tee.DEFAULT_ATTESTATION_CACHE_TTL_SECS


def test_attestation_cache_ttl_env_overrides_and_clamps_to_ceiling(monkeypatch):
    monkeypatch.setenv(phala_tee.PHALA_ATTESTATION_CACHE_TTL_ENV, "120")
    assert phala_tee._attestation_cache_ttl_secs() == 120.0
    monkeypatch.setenv(phala_tee.PHALA_ATTESTATION_CACHE_TTL_ENV, "86400")
    assert phala_tee._attestation_cache_ttl_secs() == phala_tee._ATTESTATION_CACHE_TTL_CEILING_SECS
    monkeypatch.setenv(phala_tee.PHALA_ATTESTATION_CACHE_TTL_ENV, "0")
    assert phala_tee._attestation_cache_ttl_secs() == 0.0
    monkeypatch.setenv(phala_tee.PHALA_ATTESTATION_CACHE_TTL_ENV, "garbage")
    assert phala_tee._attestation_cache_ttl_secs() == phala_tee.DEFAULT_ATTESTATION_CACHE_TTL_SECS


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
