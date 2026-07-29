"""agents_core.phala_tee — verified client for Phala's Attested Confidential
Inference (ACI) API (https://inference.phala.com).

Ported from the Dstack-TEE/private-ai-gateway reference client
(``clients/verifier-ts``: ``report.ts``, ``digest.ts``, ``crypto.ts``,
``jcs.ts``, ``e2ee.ts``, ``e2ee-channel.ts``). Implements that client's
"Level 2, checks 2-6" report-binding verification — workload_id, keyset
digest, report_data/nonce binding, keyset endorsement signature, and epoch
freshness — and its ``x25519-aes-256-gcm-hkdf-sha256`` e2ee seal/open flow
for ``messages[].content`` on chat completions.

SCOPE BOUNDARY (matches the reference client, not broader): this module does
NOT verify "Level 1" — the hardware TEE quote's chain to the Intel/NVIDIA
vendor root — and does NOT verify the GPU-to-CPU-TEE channel binding. A
report that passes ``verify_report_binding`` is cryptographically
self-consistent and endorsed by the identity key it claims; it is not
independently proven to have come from real TDX/Hopper hardware. Do not
read "verified" in this module as "fully verified" — read it as "Level 2
verified, Level 1 assumed."

Streaming, ``prompt``/``input`` (completions/embeddings) field paths, and
multi-field response decryption (audio, reasoning_content) are out of scope
for v0 — only ``messages[].content`` request sealing and
``choices[].message.content`` response opening are implemented. Add more
when a caller needs them.

CONFIDENTIALITY IS NOT CONTENT-TRUST (Mirror Council gate, 2026-07-29). A
successful ``verify_report_binding`` and the resulting ``report_verified``
flag / ``"paid-phala-tee"`` locality cost class prove the *channel* was
sealed — Phala's gateway and node cannot observe the request/response. They
prove NOTHING about whether the model behind that channel is trustworthy,
unmanipulated, or giving safe advice. Council's words: "a sealed room with a
poisoned actor is merely a slower death" — a lie told over an attested
channel is still a lie, and it cannot be caught by anything watching the
wire, because nothing can watch the wire. Treat every claim this module
makes ("verified", "e2ee_applied", "report_verified") as a PRIVACY claim,
never a TRUST claim. A future classifier/router that reads
``report_verified: True`` as license to trust a Phala response's content
*more* than an equivalent local or contractor_safe call is misusing it —
closing that misuse is scope for the eventual tou-contractor-pipeline
classifier/router, not this module.

Residual, accepted gap (not resolved here): Phala's own attestation already
flags ``serving_software_known_good: unknown`` and
``model_weights_provenance: unknown`` — the hardware and OS image are
vouched for by this module's checks, the model serving stack and weights
are not.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from agents_core import locality

DEFAULT_BASE_URL = "https://inference.phala.com"
E2EE_ALGO = "x25519-aes-256-gcm-hkdf-sha256"
_HKDF_INFO = b"aci.e2ee.v2.x25519"


# ---------------------------------------------------------------------------
# Errors — a failed report-binding *check* is reported as ok=False in a Check
# (never thrown); these exceptions are for "the caller tried to use an
# unverified or malformed result", mirroring errors.ts's split.
# ---------------------------------------------------------------------------


class AciError(Exception):
    """Base class for every error this module raises."""


class AciFormatError(AciError):
    """A JCS/hex input was malformed."""


class UnsupportedAlgorithmError(AciError):
    """A signature/identity algorithm this module cannot verify (only ed25519 is supported)."""

    def __init__(self, algorithm: str, context: str):
        super().__init__(
            f'unsupported algorithm "{algorithm}" for {context}: this client '
            f"verifies only ed25519. Do not treat an unverifiable report as verified."
        )
        self.algorithm = algorithm


class ReportVerificationError(AciError):
    """Raised when a report fails one or more report-binding checks. The
    report is refused outright — never degraded to an unverified fallback."""

    def __init__(self, verification: "ReportVerification"):
        failed = [c.name for c in verification.checks if not c.ok]
        super().__init__(
            f"attestation report failed binding verification: {', '.join(failed)}"
        )
        self.verification = verification


class E2eeNotAppliedError(AciError):
    """Raised when a request asked for e2ee but the gateway's response did
    not carry ``x-e2ee-applied: true``. A caller must never believe a
    request was confidential when the gateway didn't apply it."""


# ---------------------------------------------------------------------------
# JCS (RFC 8785 subset) — ACI restricts numbers to integers (spec/aci.md §3).
# ---------------------------------------------------------------------------


def canonicalize(value: Any) -> str:
    """Canonicalize `value` to its RFC 8785 (JCS) string form, restricted to
    the ACI subset (integers only, no floats)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        raise AciFormatError(f"JCS: ACI restricts numbers to integers, got {value!r}")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(canonicalize(v) for v in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value.keys())
        parts = [
            json.dumps(k, ensure_ascii=False) + ":" + canonicalize(value[k])
            for k in keys
        ]
        return "{" + ",".join(parts) + "}"
    raise AciFormatError(f"JCS: unsupported type {type(value)!r}")


def jcs_bytes(value: Any) -> bytes:
    """Canonicalize and UTF-8 encode — the bytes fed to SHA-256 and signatures."""
    return canonicalize(value).encode("utf-8")


# ---------------------------------------------------------------------------
# Digests (digest.ts port) — spec/aci.md §4.1, §4.2, §4.3, §4.4.
# ---------------------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_prefixed(data: bytes) -> str:
    return "sha256:" + sha256_hex(data)


def compute_workload_id(public_key: dict) -> str:
    """`workload_id` (§4.1): sha256: || hex(sha256(JCS({algo, public_key})))."""
    return sha256_prefixed(
        jcs_bytes({"algo": public_key["algo"], "public_key": public_key["public_key"]})
    )


def compute_keyset_digest(keyset: dict) -> str:
    """`workload_keyset_digest` (§4.2): sha256: || hex(sha256(JCS(keyset)))."""
    return sha256_prefixed(jcs_bytes(keyset))


def attestation_statement(workload_id: str, workload_keyset_digest: str, nonce) -> dict:
    return {
        "purpose": "aci.report_data.v1",
        "workload_id": workload_id,
        "workload_keyset_digest": workload_keyset_digest,
        "nonce": nonce,
    }


def compute_report_data(workload_id: str, workload_keyset_digest: str, nonce) -> str:
    """`report_data` (§4.4): bare hex sha256 of the JCS statement, no prefix."""
    return sha256_hex(jcs_bytes(attestation_statement(workload_id, workload_keyset_digest, nonce)))


def keyset_endorsement_payload(workload_keyset_digest: str) -> bytes:
    """JCS bytes of the keyset endorsement payload (§4.3), signed by the identity key."""
    return jcs_bytes(
        {"purpose": "aci.keyset.endorsement.v1", "workload_keyset_digest": workload_keyset_digest}
    )


# ---------------------------------------------------------------------------
# Signature verification (crypto.ts port) — ed25519 only.
# ---------------------------------------------------------------------------


def verify_ed25519(public_key_raw: bytes, signature: bytes, message: bytes) -> bool:
    """Verify an Ed25519 signature. Returns False on a bad signature or
    malformed key — never raises for those."""
    try:
        key = Ed25519PublicKey.from_public_bytes(public_key_raw)
        key.verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


# ---------------------------------------------------------------------------
# Report-binding verification (report.ts port) — spec/aci.md §10.1 checks 2-6.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str | None = None


@dataclass(frozen=True)
class ReportVerification:
    ok: bool
    checks: list = field(default_factory=list)
    workload_id: str = ""
    workload_keyset_digest: str = ""


def _check_equal(name: str, actual, expected) -> Check:
    ok = actual == expected
    return Check(name, ok, None if ok else f"report {actual!r} != recomputed {expected!r}")


def verify_report_binding(report: dict, nonce, *, now: int | None = None) -> ReportVerification:
    """Verify the report's cryptographic bindings for `nonce` (§10.1 checks
    2-6): workload_id, workload_keyset_digest, report_data, keyset
    endorsement signature, and epoch freshness.

    Does NOT verify §10.1 check 1 (the TEE hardware quote's chain to the
    vendor root) or the GPU-to-CPU-TEE channel binding — see module
    docstring. A failed individual check is reported as `ok=False` in the
    returned `Check`, never raised; callers that must refuse on any failure
    should use `require_verified_report_binding`.
    """
    if now is None:
        now = int(time.time())
    checks: list[Check] = []

    attestation = report["attestation"]
    keyset = attestation["workload_keyset"]
    identity_key = keyset["workload_identity"]["public_key"]

    workload_id = compute_workload_id(identity_key)
    checks.append(_check_equal("workload_id", report.get("workload_id"), workload_id))

    workload_keyset_digest = compute_keyset_digest(keyset)
    checks.append(
        _check_equal(
            "workload_keyset_digest", report.get("workload_keyset_digest"), workload_keyset_digest
        )
    )

    expected_report_data = compute_report_data(workload_id, workload_keyset_digest, nonce)
    checks.append(_check_equal("report_data", attestation.get("report_data"), expected_report_data))

    endorsement = attestation["keyset_endorsement"]
    endorsement_algo = endorsement.get("algo")
    identity_algo = identity_key.get("algo")
    if endorsement_algo != identity_algo:
        checks.append(
            Check(
                "keyset_endorsement",
                False,
                f'endorsement.algo {endorsement_algo!r} != identity key algo {identity_algo!r}',
            )
        )
    elif identity_algo != "ed25519":
        raise UnsupportedAlgorithmError(identity_algo, "keyset endorsement")
    else:
        try:
            ok = verify_ed25519(
                bytes.fromhex(identity_key["public_key"]),
                bytes.fromhex(endorsement["value"]),
                keyset_endorsement_payload(workload_keyset_digest),
            )
        except ValueError:
            ok = False
        checks.append(
            Check("keyset_endorsement", ok, None if ok else "endorsement signature failed under identity key")
        )

    not_after = keyset["keyset_epoch"]["not_after"]
    epoch_ok = now < not_after
    checks.append(
        Check(
            "keyset_epoch.not_after",
            epoch_ok,
            None if epoch_ok else f"now {now} >= not_after {not_after}",
        )
    )

    return ReportVerification(
        ok=all(c.ok for c in checks),
        checks=checks,
        workload_id=workload_id,
        workload_keyset_digest=workload_keyset_digest,
    )


def require_verified_report_binding(report: dict, nonce, *, now: int | None = None) -> ReportVerification:
    """Verify the report and raise `ReportVerificationError` if any check
    failed. Use this before ever encrypting to the keyset's e2ee key — a
    report that fails a check must be refused, not degraded."""
    verification = verify_report_binding(report, nonce, now=now)
    if not verification.ok:
        raise ReportVerificationError(verification)
    return verification


# ---------------------------------------------------------------------------
# E2EE AAD builders (e2ee.ts port) — spec/aci.md §7.3.
# ---------------------------------------------------------------------------


def request_aad(*, algo: str, model: str, field: str, nonce: str, ts: int) -> bytes:
    """Request AAD (tag `aci.e2ee.request.v2`) as UTF-8 JCS bytes."""
    return jcs_bytes(
        {
            "purpose": "aci.e2ee.request.v2",
            "algo": algo,
            "model": model,
            "field": field,
            "nonce": nonce,
            "ts": ts,
        }
    )


def response_aad(*, algo: str, model: str, id: str, field: str, nonce: str, ts: int) -> bytes:
    """Response AAD (tag `aci.e2ee.response.v2`) as UTF-8 JCS bytes."""
    return jcs_bytes(
        {
            "purpose": "aci.e2ee.response.v2",
            "algo": algo,
            "model": model,
            "id": id,
            "field": field,
            "nonce": nonce,
            "ts": ts,
        }
    )


# ---------------------------------------------------------------------------
# E2EE seal/open primitives (e2ee-channel.ts port) — X25519 + HKDF-SHA256 +
# AES-256-GCM, spec/aci.md §7.1.
# ---------------------------------------------------------------------------


def derive_e2ee_key(shared_secret: bytes) -> bytes:
    """Derive the AES-256-GCM key from a raw X25519 shared secret: HKDF-SHA256,
    empty salt, info=`aci.e2ee.v2.x25519` (§7.1)."""
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=b"", info=_HKDF_INFO).derive(shared_secret)


def seal_field(recipient_pub_raw: bytes, plaintext: bytes, aad: bytes) -> str:
    """Encrypt one field to `recipient_pub_raw` with a fresh ephemeral X25519
    key -> wire hex: ephemeral_pub(32) || iv(12) || ciphertext+tag."""
    eph_priv = X25519PrivateKey.generate()
    eph_pub_raw = eph_priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    recipient_pub = X25519PublicKey.from_public_bytes(recipient_pub_raw)
    shared = eph_priv.exchange(recipient_pub)
    key = derive_e2ee_key(shared)
    iv = os.urandom(12)
    ct = AESGCM(key).encrypt(iv, plaintext, aad)
    return (eph_pub_raw + iv + ct).hex()


def open_field(recipient_priv: X25519PrivateKey, blob_hex: str, aad: bytes) -> bytes:
    """Decrypt one field addressed to `recipient_priv`'s static key."""
    try:
        blob = bytes.fromhex(blob_hex)
    except ValueError as e:
        raise AciFormatError(f"open_field: invalid hex blob: {e}") from e
    if len(blob) < 32 + 12:
        raise AciFormatError("open_field: blob too short for ephemeral key + iv")
    eph_pub_raw, iv, ct = blob[:32], blob[32:44], blob[44:]
    eph_pub = X25519PublicKey.from_public_bytes(eph_pub_raw)
    shared = recipient_priv.exchange(eph_pub)
    key = derive_e2ee_key(shared)
    return AESGCM(key).decrypt(iv, ct, aad)


# ---------------------------------------------------------------------------
# E2EE channel (e2ee-channel.ts port, v0 scope) — chat-completions
# `messages[].content` only. No streaming/embeddings/completions in v0.
# ---------------------------------------------------------------------------


@dataclass
class _SentContext:
    model: str
    nonce: str
    ts: int


class E2eeChannel:
    """An encrypted channel bound to one verified workload's e2ee key."""

    def __init__(self, *, service_pub_raw: bytes, service_pub_hex: str):
        self._service_pub_raw = service_pub_raw
        self._service_pub_hex = service_pub_hex
        self._client_priv = X25519PrivateKey.generate()
        self._client_pub_hex = self._client_priv.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()
        self._sent: _SentContext | None = None

    def seal_messages(self, messages: list, model: str) -> tuple[list, dict]:
        """Encrypt `messages[].content`; returns (sealed_messages, X-E2EE-* headers)."""
        nonce = os.urandom(32).hex()
        ts = int(time.time())
        self._sent = _SentContext(model=model, nonce=nonce, ts=ts)

        def enc_field(text: str, field_path: str) -> str:
            aad = request_aad(algo=E2EE_ALGO, model=model, field=field_path, nonce=nonce, ts=ts)
            return seal_field(self._service_pub_raw, text.encode("utf-8"), aad)

        sealed = []
        for i, m in enumerate(messages):
            content = m.get("content") if isinstance(m, dict) else None
            if content is None:
                sealed.append(m)
                continue
            text = content if isinstance(content, str) else json.dumps(content)
            new_m = dict(m)
            new_m["content"] = enc_field(text, f"messages.{i}.content")
            sealed.append(new_m)

        headers = {
            "X-E2EE-Version": "2",
            "X-Client-Pub-Key": self._client_pub_hex,
            "X-Model-Pub-Key": self._service_pub_hex,
            "X-E2EE-Nonce": nonce,
            "X-E2EE-Timestamp": str(ts),
        }
        return sealed, headers

    def open_response(self, response: dict) -> dict:
        """Decrypt `choices[].message.content` in a buffered chat-completion response."""
        if self._sent is None:
            raise AciError("open_response: call seal_messages first")
        sent = self._sent
        response_id = response.get("id") or ""

        def dec_field(blob_hex: str, field_path: str) -> str:
            aad = response_aad(
                algo=E2EE_ALGO, model=sent.model, id=response_id, field=field_path,
                nonce=sent.nonce, ts=sent.ts,
            )
            return open_field(self._client_priv, blob_hex, aad).decode("utf-8")

        out = dict(response)
        choices = out.get("choices")
        if isinstance(choices, list):
            new_choices = []
            for pos, c in enumerate(choices):
                idx = c.get("index", pos) if isinstance(c, dict) else pos
                if not isinstance(idx, int):
                    idx = pos
                c = dict(c) if isinstance(c, dict) else c
                message = c.get("message") if isinstance(c, dict) else None
                if isinstance(message, dict):
                    message = dict(message)
                    if isinstance(message.get("content"), str):
                        message["content"] = dec_field(message["content"], f"choices.{idx}.message.content")
                    c["message"] = message
                new_choices.append(c)
            out["choices"] = new_choices
        return out


def open_e2ee_channel(report: dict, verification: ReportVerification) -> E2eeChannel:
    """Open an e2ee channel to the workload `report` describes, once
    `verification` (from `verify_report_binding`/`require_verified_report_binding`
    for that report) has passed. Refuses (raises) unless `verification.ok` and
    the verified digest matches the report's claimed digest — you cannot
    encrypt to a key that is not in a verified, endorsed keyset."""
    if not verification.ok or verification.workload_keyset_digest != report.get("workload_keyset_digest"):
        raise ReportVerificationError(verification)

    keys = report["attestation"]["workload_keyset"].get("e2ee_public_keys") or []
    service = next((k for k in keys if k.get("algo") == E2EE_ALGO), None)
    if service is None:
        raise AciError(f"open_e2ee_channel: no attested {E2EE_ALGO} key in the keyset")

    return E2eeChannel(
        service_pub_raw=bytes.fromhex(service["public_key"]),
        service_pub_hex=service["public_key"],
    )


# ---------------------------------------------------------------------------
# High-level client
# ---------------------------------------------------------------------------


class PhalaTeeClient:
    """Client for Phala's Attested Confidential Inference API.

    Every call through `chat_completion` fetches a fresh attestation report,
    verifies its binding (`require_verified_report_binding`), refuses to
    proceed on any failed check, and records one locality-ledger entry
    (`cost_class="paid-phala-tee"`) carrying `report_verified` /
    `e2ee_applied` so the two claims — "sent to Phala" and "sent to Phala
    and confidentiality was verified" — are never collapsed into one row.
    See the module docstring: this is a confidentiality claim, not a
    content-trust claim.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ):
        self._api_key = api_key or os.environ.get("PHALA_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout

    def _auth_headers(self) -> dict:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def fetch_attestation(self, *, nonce: str | None = None) -> dict:
        """GET /v1/aci/attestation, optionally binding `nonce` into the report."""
        params = {"nonce": nonce} if nonce is not None else {}
        resp = self._session.get(
            f"{self._base_url}/v1/aci/attestation",
            params=params,
            headers=self._auth_headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def chat_completion(
        self,
        *,
        messages: list,
        model: str,
        nonce: str | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        """Fetch + verify attestation, seal `messages[].content`, POST
        /v1/chat/completions, and decrypt the response. Raises
        `ReportVerificationError` if the report fails binding verification
        and `E2eeNotAppliedError` if the gateway didn't apply e2ee — never
        silently falls back to an unverified/unsealed call."""
        if nonce is None:
            nonce = os.urandom(16).hex()

        report = self.fetch_attestation(nonce=nonce)
        verification = require_verified_report_binding(report, nonce)
        channel = open_e2ee_channel(report, verification)

        sealed_messages, e2ee_headers = channel.seal_messages(messages, model)
        body = dict(extra_body or {})
        body["model"] = model
        body["messages"] = sealed_messages

        headers = dict(self._auth_headers())
        headers.update(e2ee_headers)

        start = time.monotonic()
        resp = self._session.post(
            f"{self._base_url}/v1/chat/completions",
            json=body,
            headers=headers,
            timeout=self._timeout,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        resp.raise_for_status()

        applied = resp.headers.get("x-e2ee-applied", "").strip().lower() == "true"
        response_json = resp.json()
        if applied:
            response_json = channel.open_response(response_json)

        locality.record(
            requested_operator="phala-tee",
            served_model=model,
            host="phala",
            cost_class="paid-phala-tee",
            seam="phala_tee",
            duration_ms=duration_ms,
            ok=applied,
            extra={
                "receipt_id": resp.headers.get("x-receipt-id"),
                "workload_keyset_digest": verification.workload_keyset_digest,
                "e2ee_applied": applied,
                "report_verified": verification.ok,
            },
        )

        if not applied:
            raise E2eeNotAppliedError(
                "Phala gateway did not report x-e2ee-applied: true for a request "
                "that required e2ee — refusing to treat this response as confidential."
            )

        return response_json
