"""agents_core.phala_tee — verified client for Phala's Attested Confidential
Inference (ACI) API (https://inference.phala.com), protocol ``aci/1``.

Ported from the Dstack-TEE/private-ai-gateway reference client
(``clients/verifier-ts``: ``report.ts``, ``digest.ts``, ``crypto.ts``,
``jcs.ts``, ``e2ee.ts``, ``e2ee-channel.ts``) plus this module's own
``x25519-aes-256-gcm-hkdf-sha256`` e2ee seal/open flow for
``messages[].content`` on chat completions, when the gateway advertises it.

RESTORED FOR aci/1 (agents-core-phala-aci1-restoration-v0, 2026-08-08). Phala
moved ``inference.phala.com`` to protocol ``aci/1`` and dropped the fields the
previous "Level 2" checks read directly (``workload_id``, ``workload_identity``,
``keyset_endorsement``, ``keyset_epoch``) — under ``aci/1`` the TDX hardware
quote binds the keyset digest directly, so the keyset digest **is** the
workload identity and there is no separate software endorsement signature to
check. Report-binding verification (``verify_report_binding``) now delegates
to the vendored reference verifier (the ``aci`` CLI, built from
``Dstack-TEE/private-ai-gateway`` at the commit the live report itself
attests in ``source_provenance``) instead of hand-rolling TDX quote parsing —
see that function's docstring for the two-leg (offline audit / online
verify), fail-closed, per-check-gated design.

SCOPE: this module still does not verify the GPU-to-CPU-TEE channel binding
beyond what the vendored verifier's ``id-6`` check does. A report that passes
``verify_report_binding`` is refused unless BOTH the offline binding-chain
checks (fresh per call, nonce-bound) AND the online hardware-root/channel-
binding checks (cached per keyset digest) pass — see that docstring for the
declared required-check constants and the id-5 (private-key custody)
allowance, which is a permitted skip against the current vendored verifier
build, always surfaced, never silently absorbed.

TWO CACHES EXIST (phala-attestation-cache-v0, 2026-09-14), both in-process
and both fail-closed:

1. The online ``aci verify`` leg's transcript is cached per keyset digest
   for one hour (``_online_verify_cache``) — that leg is nonce-independent,
   so the transcript is safe to reuse.
2. The whole VERIFIED attestation bundle (report bytes, the nonce used, the
   ``ReportVerification`` result, the keyset digest) is cached
   process-wide under the ``(base_url, verify_target)`` tuple for
   ``min(ttl_env, 3600, not_after)`` seconds (``_attestation_cache``, TTL
   env ``PHALA_ATTESTATION_CACHE_TTL_SECS``, default 900s, hard ceiling
   3600s IN the clamp expression). A cache HIT skips BOTH the attestation
   GET and both CLI legs; a MISS runs the full fresh-nonce flow verbatim.
   The store is MODULE-level (process-wide, shared across client
   instances) because the node2 consumer drops the client per pass — an
   instance-level cache would never hit there.

Nonce posture under a hit (deliberate, bounded): a hit reuses the cached
(report, nonce) pair instead of minting fresh ones — the pair was fully
verified at store time, a hit re-presents a still-valid proof rather than a
fabricated one, mints no fresh nonce, performs no attestation traffic, and
never re-sends the pair to the gateway. The replay window is bounded by
``min(ttl_env, 3600, not_after)``; freshness-of-hardware (id-1) is the
online leg's collateral job and inherits the same bounded staleness the
existing 1-hour online cache already accepted. The nonce-bound offline
audit leg is NEVER cached by itself — its verdict is bound to the nonce —
the only reuse on a hit is re-presenting the same verified pair. A keyset
rotation at the same target is not detected at hit time; a POST rejected
with 400/401/403/422 pops the entry and re-raises fail-closed, otherwise
detection is delayed up to ``min(ttl, 3600, not_after)``. The locality row
stays honest: ``report_verified`` / ``custody_unverified`` / ``e2ee_applied``
/ ``channel`` are unchanged; cache status is additive in ``extra``
(``attestation_cache_hit`` / ``attestation_cache_age_s``) — "verified 14
minutes ago" never reads as "verified this instant".

Streaming, ``prompt``/``input`` (completions/embeddings) field paths, and
audio response decryption are out of scope for v0 — request sealing covers
``messages[].content``; response opening covers ``choices[].message.content``
and ``choices[].message.reasoning_content``. Add more when a caller needs
them.

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

CONFIDENTIALITY UNDER aci/1 IS CHANNEL BINDING, NOT BODY-LEVEL E2EE BY
DEFAULT. The gateway's ``service_capabilities.supported_e2ee_versions`` may
be empty (it is, live, as of this restoration) — ``serving: "aggregator"``
means the gateway no longer terminates body-level sealing. When it is empty,
this module sends plaintext, and refuses to do so anywhere but a loopback
base URL (``PhalaTeeClient`` must be pointed at a local verifying hop — see
``systemd/aci-serve.service``). Confidentiality then comes from TLS pinned to
a key in the hardware-attested keyset (spec 9.1 id-6) plus signed per-request
receipts, not from this module's own sealing. If the gateway ever readvertises
the e2ee version this module implements, sealing resumes automatically — the
sealing code is capability-gated, never deleted.

Residual, accepted gap (not resolved here): Phala's own attestation already
flags ``serving_software_known_good: unknown`` and
``model_weights_provenance: unknown`` — the hardware and OS image are
vouched for by the vendored verifier's checks, the model serving stack and
weights are not. Likewise id-5 (private-key custody) is not evaluated by the
current vendored verifier build (dstack KMS chain validation is not wired
into it) — this module surfaces that as an explicit, visible allowance
(``ReportVerification.custody_skipped``), never as a silent pass.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from agents_core import locality

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://inference.phala.com"
# The traffic route (PHALA_BASE_URL) and the verification target
# (PHALA_VERIFY_TARGET) are different concepts and must never be confused —
# see PlaintextChannelNotLoopbackError vs LoopbackVerificationTargetError.
# PHALA_BASE_URL points client traffic at the loopback aci-serve verifying
# hop; PHALA_VERIFY_TARGET is what the online `aci verify` leg audits, and
# defaults to the real upstream so the online leg verifies Phala's actual
# TLS-terminating service, not the hop.
PHALA_VERIFY_TARGET_ENV = "PHALA_VERIFY_TARGET"
E2EE_ALGO = "x25519-aes-256-gcm-hkdf-sha256"
_HKDF_INFO = b"aci.e2ee.v2.x25519"
# The e2ee version string this module's sealing code implements, matched
# against service_capabilities.supported_e2ee_versions before sealing.
_SUPPORTED_E2EE_VERSION = "2"
# secp256k1 group order — canonicality bound for raw r||s signatures.
SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# ---------------------------------------------------------------------------
# Spend cap (agents-core-phala-spend-cap-v0, 2026-08-08 ratification). The
# gateway returns an authoritative per-call `usage.cost` from the metered
# provider synchronously, in the response body — this is not an estimate
# and not Phala's usage panel (Erah's caveat: "either the panel is slow or
# something's + or - here" — the panel must never be what enforces this).
# Leg 1 threads that cost into the locality ledger below; Leg 2 (below) sums
# today's already-recorded ledger rows and refuses BEFORE sending a call
# that would push the Pacific day over cap. Never a price table, never a
# poll.
# ---------------------------------------------------------------------------
PHALA_COST_CLASS = "paid-phala-tee"
PHALA_DAILY_SPEND_CAP_ENV = "PHALA_DAILY_SPEND_CAP_USD"
# Q2a-1 ratification: $2/day, ~5x Erah's observed 2026-08-08 usage ($0.36
# for 3.6M tokens at $0.20/M input, $0.40/M output).
DEFAULT_PHALA_DAILY_SPEND_CAP_USD = 2.00
_PACIFIC = ZoneInfo("America/Los_Angeles")
# Reconciliation (Leg 3) divergence threshold — informational only, never
# gates a call. Env-overridable like the cap for the same testing reasons.
PHALA_RECONCILE_DIVERGENCE_ENV = "PHALA_RECONCILE_DIVERGENCE_USD"
DEFAULT_PHALA_RECONCILE_DIVERGENCE_USD = 0.50

# ---------------------------------------------------------------------------
# Vendored reference verifier (`aci` CLI) delegation — Piece 1 (the binary
# itself) is host-provided, built outside this repo from the exact commit the
# live report attests in source_provenance. This module never builds it and
# never assumes a path beyond the documented default.
# ---------------------------------------------------------------------------

ACI_VERIFIER_BIN_ENV = "ACI_VERIFIER_BIN"
# Documented default — where the BRIX host lane vendors the build (see
# agents-core-phala-aci1-restoration-v0 spec, Piece 1). Override with
# ACI_VERIFIER_BIN if the binary lives elsewhere.
DEFAULT_ACI_VERIFIER_BIN = "/srv/fast/aci-build/private-ai-gateway/target/release/aci"

# Required-check policy (Facets/Council, 2026-08-08 run 2026-08-08-075006-da851e):
# a declared, reviewable constant, not an implication of the exit code.
# Offline `aci audit` leg (report bytes + fresh nonce, run every call, NEVER
# cached — its result is nonce-bound):
_AUDIT_REQUIRED_CHECKS = ("id-2", "id-3", "id-4")
# Online `aci verify` leg (hardware root + channel binding, nonce-independent,
# cached per keyset digest for one hour):
_VERIFY_REQUIRED_CHECKS = ("id-1", "id-6")
# id-5 (private-key custody) is the sole permitted skip — the vendored
# verifier does not wire in dstack KMS chain validation. Never required, but
# never silently absorbed either — see ReportVerification.custody_skipped.
_PERMITTED_SKIP_CHECKS = ("id-5",)

_ONLINE_VERIFY_CACHE_TTL_SECS = 3600.0
# keyset_digest -> (cached_at_monotonic, transcript). Module-level: the cache
# key is nonce-independent by construction, so sharing it across clients is
# safe and matches "cached per keyset digest for one hour" in the spec.
_online_verify_cache: dict[str, tuple[float, dict]] = {}

# ---------------------------------------------------------------------------
# Verified-attestation bundle cache (phala-attestation-cache-v0, 2026-09-14).
# The verifying client used to pay the attestation GET + the two `aci` CLI
# verify legs on EVERY call, which at production call rates gets Phala
# rate-limiting (correction/phala-guarded-client-per-call-attestation-rate-
# limited-2026-09-14). The cache stores the whole VERIFIED bundle — report
# bytes, the nonce used, the ReportVerification result, the keyset digest —
# under one TTL, so a hit skips the attestation GET AND both CLI legs.
#
# Nonce posture (deliberate, bounded): a hit reuses the cached report and its
# nonce instead of minting fresh ones. The (report, nonce) pair was fully
# verified at store time; a hit re-presents a still-valid proof, it never
# mints a fresh nonce, performs no attestation traffic, and the pair is NOT
# re-sent to the gateway anywhere (the nonce rode only on the store-time GET
# query and the local `aci audit --nonce` arg). The replay window this opens
# is bounded by min(ttl_env, _ATTESTATION_CACHE_TTL_CEILING_SECS, not_after).
#
# Key is the TUPLE (base_url, verify_target) — NOT a flat f-string: a pipe in
# a URL would collide, and the tuple makes the invalidation semantics
# precise (a None dimension in invalidate_attestation_cache = no filter).
# The key deliberately does NOT include the keyset digest — it is only
# knowable post-verify, and the whole point of a hit is to skip the verify.
# A keyset rotation at the same target is therefore not detected at hit
# time; it is caught on a POST auth/keyset rejection (see
# _ATTESTATION_POST_REJECT_STATUSES) with detection otherwise delayed up to
# min(ttl, ceiling, not_after).
#
# MODULE-LEVEL store, process-wide, shared across client instances: the
# node2 consumer (lapis-pm corroboration_adapter) builds a fresh client per
# corroboration pass, so a client-instance-level cache would never hit there
# and every pass + the grammar-degrade retry would re-pay fetch + two legs —
# the exact rate-limit failure mode this cache exists to remove. In-process
# only, no cross-process coordination (daemon restart re-verifies once;
# acceptable, matching _online_verify_cache house style).
# ---------------------------------------------------------------------------
PHALA_ATTESTATION_CACHE_TTL_ENV = "PHALA_ATTESTATION_CACHE_TTL_SECS"
DEFAULT_ATTESTATION_CACHE_TTL_SECS = 900.0
# Hard ceiling on the effective TTL, IN the clamp expression (evaluated at
# store time): ttl_effective = min(ttl_env, ceiling, max(0.0, not_after -
# time.time())). The ceiling must be in the min itself, not a prose clause —
# an implementer copying a ceiling-less formula ships 24h staleness with
# ttl_env=86400.
_ATTESTATION_CACHE_TTL_CEILING_SECS = 3600.0
# A chat-completion POST rejected with one of these statuses is a keyset
# rotation (or auth failure) rejected at the hop: the cached bundle is stale
# for the gateway, so the entry is popped and the error re-raised fail-
# closed. Transient statuses (5xx, ...) do NOT pop — the bundle is still
# valid, the failure is the gateway's.
_ATTESTATION_POST_REJECT_STATUSES = frozenset({400, 401, 403, 422})


@dataclass(frozen=True)
class _VerifiedAttestation:
    """A fully verified attestation bundle, cached process-wide under
    (base_url, verify_target). Stored ONLY after
    `require_verified_report_binding` has passed — no code path can reach
    the POST with a bundle that did not pass the full two-leg gate at store
    time."""

    report: dict
    nonce: str
    verification: ReportVerification
    keyset_digest: str
    cached_at_monotonic: float
    not_after: int  # wall clock (unix secs) — the report's keyset validity bound


# (base_url, verify_target) -> (cached_at_monotonic, _VerifiedAttestation).
# Module-level, time.monotonic() stamped, in-process only — see the section
# comment above for why the store is process-wide and not client-instance
# level.
_attestation_cache: dict[tuple[str, str], tuple[float, _VerifiedAttestation]] = {}


def _attestation_cache_ttl_secs() -> float:
    """Read the env TTL at call time (not module load) so tests/ops can
    override without a process restart — same pattern as `_aci_verifier_bin`.
    Returns ONLY the env-ceiling clamp min(ttl_env,
    _ATTESTATION_CACHE_TTL_CEILING_SECS); the per-report `not_after` term is
    applied at store time against the report. A ttl_env <= 0 disables the
    cache (never store) — it is NOT a verify-off switch: with the cache
    disabled every call still runs both legs."""
    raw = os.environ.get(PHALA_ATTESTATION_CACHE_TTL_ENV)
    if raw is None:
        ttl_env = DEFAULT_ATTESTATION_CACHE_TTL_SECS
    else:
        try:
            ttl_env = float(raw)
        except ValueError:
            ttl_env = DEFAULT_ATTESTATION_CACHE_TTL_SECS
    return min(ttl_env, _ATTESTATION_CACHE_TTL_CEILING_SECS)


def invalidate_attestation_cache(*, base_url: str | None = None, verify_target: str | None = None) -> int:
    """Drop verified-attestation cache entries, returning the count dropped.

    A `None` argument is NO filter on that dimension (wildcard); all-`None`
    drops everything. For tests, the contractor canary, and the retry path.
    """
    dropped = 0
    for key in list(_attestation_cache.keys()):
        k_base, k_target = key
        if base_url is not None and k_base != base_url:
            continue
        if verify_target is not None and k_target != verify_target:
            continue
        del _attestation_cache[key]
        dropped += 1
    return dropped


# ---------------------------------------------------------------------------
# Errors — a failed report-binding *check* is reported as ok=False in a Check
# (never thrown); these exceptions are for "the caller tried to use an
# unverified or malformed result", mirroring errors.ts's split. The three
# AciVerifier* errors are toolchain/protocol faults, distinguishable from a
# report that ran through the verifier and failed — "your verifier is not
# installed" must never look like "this report did not verify".
# ---------------------------------------------------------------------------


class AciError(Exception):
    """Base class for every error this module raises."""


class AciFormatError(AciError):
    """A JCS/hex input was malformed."""


class UnsupportedAlgorithmError(AciError):
    """A signature/identity algorithm this module cannot verify (only ed25519
    and ecdsa-secp256k1 are supported)."""

    def __init__(self, algorithm: str, context: str):
        super().__init__(
            f'unsupported algorithm "{algorithm}" for {context}: this client '
            f"verifies only ed25519 and ecdsa-secp256k1. Do not treat an "
            f"unverifiable report as verified."
        )
        self.algorithm = algorithm


class AciVerifierNotFoundError(AciError):
    """Raised when the `aci` verifier binary cannot be found or executed —
    missing, not on PATH, or not executable. A toolchain fault, never a
    report-verification failure: it means "the verifier didn't run", and
    must never degrade to an unverified pass."""

    def __init__(self, binary: str):
        super().__init__(
            f"aci verifier binary not found or not executable: {binary!r} "
            f"(set {ACI_VERIFIER_BIN_ENV} to the built `aci` binary path; "
            f"default {DEFAULT_ACI_VERIFIER_BIN!r})"
        )
        self.binary = binary


class AciVerifierTimeoutError(AciError):
    """Raised when the `aci` verifier binary does not return within the
    configured timeout. A toolchain fault, distinguishable from a report
    that ran and failed to verify."""

    def __init__(self, binary: str, timeout: float):
        super().__init__(f"aci verifier {binary!r} timed out after {timeout}s")
        self.binary = binary
        self.timeout = timeout


class AciVerifierProtocolError(AciError):
    """Raised when the `aci` verifier's stdout cannot be parsed as the
    expected `--json` transcript, or a leg's transcript is missing the
    `checks` list required to gate on per-check status. A toolchain/protocol
    fault — refuses the call exactly like the other two, never degrades to
    an unverified pass."""


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
    """Raised when a request sealed to the gateway's advertised e2ee
    capability but the gateway's response did not carry
    ``x-e2ee-applied: true``. A caller must never believe a request was
    confidential when the gateway didn't apply it. Capability-gated: this is
    never raised for a request the gateway never advertised e2ee support
    for in the first place — see ``PlaintextChannelNotLoopbackError`` for
    that path's guard."""


class PlaintextChannelNotLoopbackError(AciError):
    """Raised when the gateway does not advertise a supported e2ee version
    (``service_capabilities.supported_e2ee_versions`` doesn't include the
    version this module implements) and the configured base URL is not a
    loopback address. Plaintext must never leave the process to anywhere but
    a local, attestation-verifying hop (see ``systemd/aci-serve.service``) —
    this is the invariant that keeps the seat honest under aci/1's
    channel-binding confidentiality model."""


class LoopbackVerificationTargetError(AciError):
    """Raised when the online `aci verify` leg's target resolves to
    loopback. ``PHALA_BASE_URL`` (the traffic route) and the verification
    target are different concepts that must never be confused: traffic is
    meant to flow through the loopback verifying hop, but the thing that
    gets *verified* must be the real upstream, or the online leg ends up
    checking the hop's own plain-HTTP loopback listener instead of Phala's
    TLS-terminating service (id-6 can never bind over plain HTTP). Fail
    closed here rather than let that surface as a confusing id-6 failure."""

    def __init__(self, verify_target: str):
        super().__init__(
            f"verification target {verify_target!r} resolves to loopback — "
            f"refusing to verify the local traffic hop instead of the real "
            f"upstream. PHALA_BASE_URL routes traffic through the loopback "
            f"verifying hop (correct, leave it); {PHALA_VERIFY_TARGET_ENV} "
            f"(default {DEFAULT_BASE_URL!r}) is what gets verified and must "
            f"point at the real service."
        )
        self.verify_target = verify_target


class PhalaSpendCapExceededError(AciError):
    """Raised when today's recorded Phala spend (Pacific day, cost_class
    ``paid-phala-tee``, summed from OUR OWN locality ledger — never Phala's
    usage panel) is at or over the daily cap. Refused BEFORE the call is
    sent — fail closed (Q2a-2, 2026-08-08 ratification: "a spend guard that
    keeps spending is not a guard"). Distinct from every other AciError in
    this module on purpose: those mean a broken/unverifiable seat, this
    means "we chose not to spend" — a caller must be able to tell the two
    apart. Never degrades to another operator; the caller must handle this
    explicitly (e.g. skip the call, fall back to a non-metered seat by its
    own choice, or surface it upstream)."""

    def __init__(self, spent_usd: float, cap_usd: float, unpriced_count: int):
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        self.unpriced_count = unpriced_count
        msg = (
            f"Phala daily spend cap reached: ${spent_usd:.4f} recorded today "
            f"(America/Los_Angeles day) against a ${cap_usd:.2f}/day cap"
        )
        if unpriced_count:
            msg += (
                f" — {unpriced_count} unpriced call(s) also recorded today; "
                f"true spend may be higher than the total above (unknown, not zero)"
            )
        msg += ". Refusing this call rather than risk exceeding the cap."
        super().__init__(msg)


class ReasoningContentDecryptionError(AciError):
    """Raised by `E2eeChannel.open_response` when a choice's
    `reasoning_content` field is present but cannot be decrypted (tamper,
    corruption, wrong nonce). A caller catching this must never substitute
    an empty string or placeholder text and continue as if decryption
    succeeded — a plain string in a text field is indistinguishable from
    real content to any downstream code that doesn't specifically check for
    this exception, which would recreate the exact truth-leakage this error
    exists to prevent. The correct response is to refuse the choice (or the
    whole response), not to paper over it."""

    def __init__(self, index: int, reason: str):
        super().__init__(
            f"choices.{index}.message.reasoning_content: failed to decrypt "
            f"({reason}) — refusing to substitute placeholder text for "
            f"unopened ciphertext"
        )
        self.index = index
        self.reason = reason


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
# Digests (digest.ts port) — spec/aci.md §4.2. `workload_keyset_digest` is
# the only digest this module still computes locally under aci/1: the TDX
# hardware quote binds it directly, so it doubles as the workload identity
# (see ReportVerification.workload_id) and it is what the vendored verifier's
# online-check cache is keyed on.
# ---------------------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_prefixed(data: bytes) -> str:
    return "sha256:" + sha256_hex(data)


def compute_keyset_digest(keyset: dict) -> str:
    """`workload_keyset_digest` (§4.2): sha256: || hex(sha256(JCS(keyset)))."""
    return sha256_prefixed(jcs_bytes(keyset))


# ---------------------------------------------------------------------------
# Signature verification (crypto.ts port) — ed25519 and ecdsa-secp256k1.
# Generic primitives, not schema-specific; kept as capability-gated legacy
# building blocks even though verify_report_binding no longer calls them
# directly (aci/1 dropped the software keyset-endorsement signature these
# were verifying — see module docstring).
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


def verify_ecdsa_secp256k1(public_key_raw: bytes, signature_raw: bytes, message: bytes) -> bool:
    """Verify an ECDSA secp256k1 signature over SHA-256(message).
    `public_key_raw` is an uncompressed SEC1 point (65 bytes, `04 || X || Y`).
    `signature_raw` is a raw `r || s` pair (64 bytes) — not DER; it is
    re-encoded to DER internally before verification. Returns False on a bad
    signature or malformed key/signature bytes — never raises for those
    (same contract as `verify_ed25519`). Rejects any signature that is not
    exactly 64 raw bytes, and rejects non-canonical/out-of-range `r`/`s`
    (zero or >= the curve order) before ever calling into the underlying
    verify — those must never reach it."""
    if len(signature_raw) != 64:
        return False
    r = int.from_bytes(signature_raw[:32], "big")
    s = int.from_bytes(signature_raw[32:], "big")
    if not (1 <= r < SECP256K1_ORDER) or not (1 <= s < SECP256K1_ORDER):
        return False
    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), public_key_raw)
        der_signature = encode_dss_signature(r, s)
        key.verify(der_signature, message, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False


# ---------------------------------------------------------------------------
# Report-binding verification — spec/aci.md §9.1/§10.1, delegated to the
# vendored `aci` reference verifier under aci/1 (report.ts's hand-rolled
# Level-2 checks are dead: aci/1 dropped workload_id/workload_identity/
# keyset_endorsement/keyset_epoch from the wire format entirely).
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
    # Under aci/1 the keyset digest IS the workload identity (the TDX quote
    # binds it directly) — workload_id and workload_keyset_digest are always
    # equal here. The field is kept, not renamed, so locality rows keep their
    # shape (agents-core-phala-aci1-restoration-v0 spec).
    workload_id: str = ""
    workload_keyset_digest: str = ""
    # True unless the vendored verifier's id-5 (private-key custody) check
    # reported "pass". id-5 is a permitted skip (see _PERMITTED_SKIP_CHECKS)
    # so this never gates `ok` — but it must never be silently absorbed
    # either. Callers must surface it (locality rows do, as a warning).
    custody_skipped: bool = True


def _check_equal(name: str, actual, expected) -> Check:
    ok = actual == expected
    return Check(name, ok, None if ok else f"report {actual!r} != recomputed {expected!r}")


def _aci_verifier_bin() -> str:
    return os.environ.get(ACI_VERIFIER_BIN_ENV, DEFAULT_ACI_VERIFIER_BIN)


def _run_aci_json(args: list[str], *, input_bytes: bytes | None = None, timeout: float = 30.0) -> dict:
    """Run the `aci` verifier CLI with `--json` and return the parsed
    transcript, tagged with `_exit_code`. Fails closed and distinguishably:
    `AciVerifierNotFoundError` for a missing/non-executable binary,
    `AciVerifierTimeoutError` on timeout, `AciVerifierProtocolError` for
    unparseable output. Never returns a degraded/partial result — any of
    these three raise instead."""
    binary = _aci_verifier_bin()
    cmd = [binary, *args, "--json"]
    try:
        proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout)
    except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
        raise AciVerifierNotFoundError(binary) from e
    except subprocess.TimeoutExpired as e:
        raise AciVerifierTimeoutError(binary, timeout) from e

    try:
        transcript = json.loads(proc.stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise AciVerifierProtocolError(
            f"{' '.join(cmd)}: unparseable --json output (exit {proc.returncode}): {e}"
        ) from e
    if not isinstance(transcript, dict):
        raise AciVerifierProtocolError(
            f"{' '.join(cmd)}: expected a JSON object transcript, got {type(transcript).__name__}"
        )
    transcript["_exit_code"] = proc.returncode
    return transcript


def _check_statuses(transcript: dict, *, leg: str) -> dict[str, str]:
    """Extract {check_id: status} from a transcript's `checks` list. Gating
    must read THIS, never the top-line verdict/exit code alone — the
    vendored verifier's own conformance doc
    (docs/reviews/aci-spec-conformance-gaps.md:9-21) is explicit that exit 0
    does not distinguish a skip from a pass."""
    checks = transcript.get("checks")
    if not isinstance(checks, list):
        raise AciVerifierProtocolError(f"aci {leg}: transcript missing a 'checks' list")
    statuses: dict[str, str] = {}
    for entry in checks:
        if not isinstance(entry, dict) or "id" not in entry or "status" not in entry:
            raise AciVerifierProtocolError(f"aci {leg}: malformed check entry {entry!r}")
        statuses[entry["id"]] = entry["status"]
    return statuses


def _run_aci_audit(report: dict, nonce: str, *, timeout: float = 30.0) -> dict:
    """Offline `aci audit` leg over the exact report bytes + fresh `nonce` —
    id-2/id-3/id-4. Nonce-bound: run on EVERY call, NEVER cached.

    `aci audit` takes the report as a `--report <FILE>` path, not stdin —
    there is no positional/stdin form. Serialize the report exactly once
    (same `json.dumps` shape as ever — no `sort_keys`, no canonicalization;
    the nonce binding is computed over these exact bytes) and write that
    single object to a 0600 temp file, removed on every exit path."""
    report_bytes = json.dumps(report).encode("utf-8")
    fd, path = tempfile.mkstemp(prefix="aci-audit-report-", suffix=".json")
    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(report_bytes)
        return _run_aci_json(["audit", "--report", path, "--nonce", nonce], timeout=timeout)
    finally:
        os.remove(path)


def _run_aci_verify_online(
    base_url: str, *, workload_keyset_digest: str, timeout: float = 30.0
) -> dict:
    """Online `aci verify` leg against `base_url` — id-1 (hardware root via
    DCAP collateral), id-6 (channel binding), id-5 (custody, permitted
    skip). Nonce-independent: cached per `workload_keyset_digest` for one
    hour. A nonce-bound result must NEVER be cached — see `_run_aci_audit`,
    which never is."""
    now = time.monotonic()
    cached = _online_verify_cache.get(workload_keyset_digest)
    if cached is not None:
        cached_at, transcript = cached
        if now - cached_at < _ONLINE_VERIFY_CACHE_TTL_SECS:
            return transcript

    transcript = _run_aci_json(["verify", base_url], timeout=timeout)
    _online_verify_cache[workload_keyset_digest] = (now, transcript)
    return transcript


def _is_loopback_url(url: str) -> bool:
    hostname = urlparse(url).hostname or ""
    return hostname in ("127.0.0.1", "::1", "localhost")


def _normalize_keyset_digest(value: str | None) -> str | None:
    """Strip an optional ``sha256:`` prefix so the transcript's claimed
    digest and our independently recomputed one (``compute_keyset_digest``,
    always ``sha256:``-prefixed) compare on the same representation
    regardless of whether the transcript's value carries the prefix."""
    if value is None:
        return None
    prefix = "sha256:"
    return value[len(prefix):] if value.startswith(prefix) else value


def _verdict_failed_count(transcript: dict) -> int | None:
    """`verdict.failed` from a transcript's top-level `verdict` object, or
    None if the transcript doesn't have the shape we expect — treated as
    "not zero" (fails closed) by the caller, never as a pass."""
    verdict = transcript.get("verdict")
    if not isinstance(verdict, dict):
        return None
    return verdict.get("failed")


def _nonce_in_id2_detail(audit_transcript: dict, nonce: str) -> bool:
    """The transcript exposes no structured nonce field — the only place the
    audited nonce appears is inside id-2's `detail` string ('statement
    digest for nonce "<nonce>" matches report_data'). String-level because
    the transcript exposes no structured field; a future verifier build
    that adds one is an obvious upgrade over this, not a rediscovery of the
    check. Absence (missing id-2, missing detail, or nonce not present) is
    False — a failure, never a pass-by-default."""
    checks_list = audit_transcript.get("checks")
    if not isinstance(checks_list, list):
        return False
    for entry in checks_list:
        if isinstance(entry, dict) and entry.get("id") == "id-2":
            detail = entry.get("detail")
            return isinstance(detail, str) and nonce in detail
    return False


def verify_report_binding(
    report: dict, nonce: str, *, now: int | None = None, base_url: str = DEFAULT_BASE_URL
) -> ReportVerification:
    """Verify the report's binding for `nonce` under aci/1 by delegating to
    the vendored reference verifier (the `aci` CLI) — see module docstring
    for why this module no longer hand-rolls TDX quote parsing or reads the
    fields aci/1 dropped.

    `base_url` means "the thing to verify" — pass the verification target
    (e.g. `PhalaTeeClient`'s `PHALA_VERIFY_TARGET`, default
    `https://inference.phala.com`), never the traffic route
    (`PHALA_BASE_URL`, which stays pinned to the loopback verifying hop). A
    `base_url` that resolves to loopback is refused outright
    (`LoopbackVerificationTargetError`) — verifying the local hop instead of
    the real upstream is exactly the misconfiguration that made id-6 fail
    to bind.

    Two legs, both fail-closed, both gated on PER-CHECK transcript status
    and `verdict.failed == 0` (never on the exit code alone — an offline
    `aci audit` exits 1 whenever the verdict isn't VERIFIED, which is
    *every* offline audit, since id-1/id-6 are structurally always skipped
    without a live/online channel; exit code is kept as informational-only
    detail, never used to gate):

    - offline `aci audit` over the exact report bytes + fresh `nonce`, run
      on EVERY call, never cached (nonce-bound). Required: id-2 (binding
      chain), id-3 (keyset not expired), id-4 (provenance/compose-hash).
    - online `aci verify` against `base_url`, chaining the TDX quote to the
      Intel vendor root via DCAP collateral. Nonce-independent, cached per
      `workload_keyset_digest` for one hour. Required: id-1 (hardware
      root), id-6 (channel binding to the attested keyset).

    Beyond the per-check status, this also independently recomputes
    `workload_keyset_digest` and compares it (normalizing the `sha256:`
    prefix explicitly) against `verdict.workload_keyset_digest` — plus
    asserts `nonce` appears in id-2's `detail` string, the only place the
    audited nonce is exposed — against what the verifier transcript itself
    claims to have audited. A verifier that reports VERIFIED/all-required-
    pass/zero-failed but audited a different digest or nonce than what was
    actually asked for is refused: the malicious/hostile-verifier case this
    defends against (Facets `trickster`, 2026-08-08).

    id-5 (private-key custody) is the sole permitted skip against the
    current vendored verifier build — never required, never silently
    absorbed; see `ReportVerification.custody_skipped`.

    Raises `LoopbackVerificationTargetError` if `base_url` resolves to
    loopback. Raises `AciVerifierNotFoundError` / `AciVerifierTimeoutError`
    / `AciVerifierProtocolError` for toolchain faults ("the verifier didn't
    run") — distinguishable from a report that ran through it and failed
    (reported as `ok=False` in the returned `Check`s here; use
    `require_verified_report_binding` to raise `ReportVerificationError` on
    that). Never degrades to an unverified pass for any of these.
    """
    del now  # kept for call-site compatibility; freshness is the verifier's job (id-3).

    if _is_loopback_url(base_url):
        raise LoopbackVerificationTargetError(base_url)

    attestation = report["attestation"]
    keyset = attestation["workload_keyset"]
    workload_keyset_digest = compute_keyset_digest(keyset)

    checks: list[Check] = []

    # --- offline leg: aci audit — nonce-bound, run fresh every call, never cached.
    audit_transcript = _run_aci_audit(report, nonce)
    audit_statuses = _check_statuses(audit_transcript, leg="audit")

    # Structural gate: no check failed outright, on this leg. An offline
    # audit's exit code cannot be used for this — `aci audit` exits 1
    # whenever the verdict is not VERIFIED, and an offline audit can never
    # be VERIFIED (id-1/id-6 need online/live-TLS data it doesn't have, so
    # they're always skipped here). The exit code is kept visible in the
    # detail string as informational only; it never gates.
    audit_failed = _verdict_failed_count(audit_transcript)
    checks.append(Check(
        "aci_audit.no_failed_checks",
        audit_failed == 0,
        f"verdict.failed={audit_failed!r} (aci audit exit={audit_transcript.get('_exit_code')}, informational)",
    ))

    # Hostile-verifier defence (Facets `trickster`, 2026-08-08): a verifier
    # reporting all-required-pass while having audited a *different* digest
    # or nonce than the one asked for must be refused. Read from
    # verdict.workload_keyset_digest (top-level keyset_digest/
    # workload_keyset_digest do not exist in the real transcript shape).
    # The transcript's digest and our recomputed one are both
    # sha256:-prefixed; normalize explicitly rather than substring-matching.
    audit_verdict = audit_transcript.get("verdict")
    raw_digest = audit_verdict.get("workload_keyset_digest") if isinstance(audit_verdict, dict) else None
    checks.append(_check_equal(
        "aci_audit.keyset_digest_binding",
        _normalize_keyset_digest(raw_digest),
        _normalize_keyset_digest(workload_keyset_digest),
    ))

    # There is no structured nonce field in the transcript — the audited
    # nonce only appears inside id-2's `detail` string. String-level
    # because the transcript exposes no structured field (see
    # _nonce_in_id2_detail); absence is a failure, never a pass-by-default.
    nonce_present = _nonce_in_id2_detail(audit_transcript, nonce)
    checks.append(Check(
        "aci_audit.nonce_binding", nonce_present,
        None if nonce_present else f"nonce {nonce!r} not found in id-2 detail",
    ))

    for check_id in _AUDIT_REQUIRED_CHECKS:
        status = audit_statuses.get(check_id)
        checks.append(Check(f"aci.{check_id}", status == "pass", f"aci audit {check_id} status={status!r}"))

    # --- online leg: aci verify — nonce-independent, cached per keyset digest.
    verify_transcript = _run_aci_verify_online(base_url, workload_keyset_digest=workload_keyset_digest)
    verify_statuses = _check_statuses(verify_transcript, leg="verify")

    # Same structural gate as the offline leg: no check failed outright.
    # Exit code stays visible in the detail string, informational only.
    verify_failed = _verdict_failed_count(verify_transcript)
    checks.append(Check(
        "aci_verify.no_failed_checks",
        verify_failed == 0,
        f"verdict.failed={verify_failed!r} (aci verify exit={verify_transcript.get('_exit_code')}, informational)",
    ))

    for check_id in _VERIFY_REQUIRED_CHECKS:
        status = verify_statuses.get(check_id)
        checks.append(Check(f"aci.{check_id}", status == "pass", f"aci verify {check_id} status={status!r}"))

    custody_status = verify_statuses.get("id-5")
    custody_skipped = custody_status != "pass"

    return ReportVerification(
        ok=all(c.ok for c in checks),
        checks=checks,
        workload_id=workload_keyset_digest,
        workload_keyset_digest=workload_keyset_digest,
        custody_skipped=custody_skipped,
    )


def require_verified_report_binding(
    report: dict, nonce: str, *, now: int | None = None, base_url: str = DEFAULT_BASE_URL
) -> ReportVerification:
    """Verify the report and raise `ReportVerificationError` if any required
    check failed. Use this before ever encrypting to the keyset's e2ee key —
    a report that fails a check must be refused, not degraded."""
    verification = verify_report_binding(report, nonce, now=now, base_url=base_url)
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
# Capability-gated: only opened when the gateway advertises
# `_SUPPORTED_E2EE_VERSION` — see PhalaTeeClient.chat_completion.
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
        """Decrypt `choices[].message.content` and, when present,
        `choices[].message.reasoning_content` in a buffered chat-completion
        response. An absent `reasoning_content` field is a no-op, not an
        error.

        Raises `ReasoningContentDecryptionError` if a present
        `reasoning_content` blob fails to decrypt (tamper, corruption, wrong
        nonce). A caller catching that exception must never substitute an
        empty string or placeholder text and continue as if decryption
        succeeded — that would recreate the exact truth-leakage this check
        exists to prevent; the correct response is to refuse the choice (or
        the whole response)."""
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
                    if isinstance(message.get("reasoning_content"), str):
                        try:
                            message["reasoning_content"] = dec_field(
                                message["reasoning_content"],
                                f"choices.{idx}.message.reasoning_content",
                            )
                        except (AciFormatError, InvalidTag, ValueError) as e:
                            raise ReasoningContentDecryptionError(idx, str(e)) from e
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
# Spend cap (Leg 2) + reconciliation (Leg 3) — agents-core-phala-spend-cap-v0.
# ---------------------------------------------------------------------------


def _phala_daily_spend_cap_usd() -> float:
    """Read at call time (not module load) so tests/ops can override via env
    without a process restart — same pattern as locality._max_bytes()."""
    raw = os.environ.get(PHALA_DAILY_SPEND_CAP_ENV)
    if raw is None:
        return DEFAULT_PHALA_DAILY_SPEND_CAP_USD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_PHALA_DAILY_SPEND_CAP_USD


def _phala_reconcile_divergence_usd() -> float:
    raw = os.environ.get(PHALA_RECONCILE_DIVERGENCE_ENV)
    if raw is None:
        return DEFAULT_PHALA_RECONCILE_DIVERGENCE_USD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_PHALA_RECONCILE_DIVERGENCE_USD


def _pacific_day_bounds(day: date | None = None, *, now_utc: datetime | None = None):
    """Return (start_utc, end_utc) for one America/Los_Angeles calendar day,
    matching how TOU windows and the night DAG reckon their day boundary.
    `day=None` means "today" (Pacific), and end_utc is then `now_utc` rather
    than midnight, so a same-day window never looks into the future."""
    now_utc = now_utc or datetime.now(timezone.utc)
    now_pacific_date = now_utc.astimezone(_PACIFIC).date()
    if day is None:
        day = now_pacific_date
    start_pacific = datetime(day.year, day.month, day.day, tzinfo=_PACIFIC)
    start_utc = start_pacific.astimezone(timezone.utc)
    if day == now_pacific_date:
        end_utc = now_utc
    else:
        end_utc = (start_pacific + timedelta(days=1)).astimezone(timezone.utc)
    return start_utc, end_utc


def _notify_spend_cap_once(spent_usd: float, cap_usd: float, unpriced_count: int) -> None:
    """Pushover the spend-cap refusal, at most once per Pacific day (Q2a-2:
    a cap that pages on every blocked call is alert fatigue by construction).
    Best-effort: never raises, never blocks the caller's refusal."""
    try:
        day = datetime.now(_PACIFIC).strftime("%Y-%m-%d")
        marker = locality.root() / f".phala_spend_cap_notified.{day}"
        marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return  # already notified today

        message = (
            f"Phala daily spend cap (${cap_usd:.2f}) reached: "
            f"${spent_usd:.4f} recorded today."
        )
        if unpriced_count:
            message += (
                f" {unpriced_count} unpriced call(s) also recorded today — "
                f"true spend may be higher (unknown, not zero)."
            )

        from agents_core.notify import Priority, send_notification

        send_notification(
            message,
            title="Phala spend cap reached",
            priority=Priority.HIGH,
            source="phala_tee",
        )
    except Exception as e:
        logger.warning("[phala_tee] spend-cap notification failed: %s", e)


def _check_phala_spend_cap() -> None:
    """Sum today's OWN-LEDGER cost for cost_class paid-phala-tee (never
    Phala's usage panel) and refuse before a call is sent if the Pacific
    day is already at or over the cap — fail closed. Called at the top of
    `PhalaTeeClient.chat_completion`, before any network I/O."""
    cap = _phala_daily_spend_cap_usd()
    since, until = _pacific_day_bounds()
    summary = locality.cost_class_window_summary(PHALA_COST_CLASS, since=since, until=until)
    spent = summary["total_cost_usd"]
    if spent >= cap:
        _notify_spend_cap_once(spent, cap, summary["unpriced_count"])
        raise PhalaSpendCapExceededError(spent, cap, summary["unpriced_count"])


def reconcile_daily_spend(provider_reported_usd: float | None, *, day: date | None = None) -> dict:
    """Leg 3 — compare our ledger's daily total against the provider's own
    reported usage for the same Pacific day. Reports a divergence; NEVER
    gates a call (that's Leg 2, and Leg 2 never reads the provider panel —
    Erah's caveat "either the panel is slow or something's + or - here" is
    exactly why this leg is a slow, informational cross-check and not the
    enforcement path).

    `provider_reported_usd` is supplied by the caller (e.g. read off Phala's
    usage panel or API out of band) — this function does not fetch it, so
    it never becomes a second polling path into the panel.
    """
    since, until = _pacific_day_bounds(day)
    summary = locality.cost_class_window_summary(PHALA_COST_CLASS, since=since, until=until)
    ledger_total = summary["total_cost_usd"]

    divergence_usd = None
    material_divergence = False
    if provider_reported_usd is not None:
        divergence_usd = ledger_total - provider_reported_usd
        material_divergence = abs(divergence_usd) >= _phala_reconcile_divergence_usd()

    return {
        "ledger_total_usd": ledger_total,
        "provider_reported_usd": provider_reported_usd,
        "divergence_usd": divergence_usd,
        "material_divergence": material_divergence,
        "unpriced_count": summary["unpriced_count"],
        "window": {"since": since.isoformat(), "until": until.isoformat()},
    }


# ---------------------------------------------------------------------------
# High-level client
# ---------------------------------------------------------------------------


class PhalaTeeClient:
    """Client for Phala's Attested Confidential Inference API.

    Every call through `chat_completion` verifies the attestation binding
    (`require_verified_report_binding`) before proceeding, refuses on any
    failed required check, and records one locality-ledger entry
    (`cost_class="paid-phala-tee"`) carrying `report_verified` /
    `e2ee_applied` / `channel` / `custody_unverified` so the claims — "sent
    to Phala", "sent to Phala with the channel verified", "sent to Phala
    with body-level e2ee applied" — are never collapsed into one row. See
    the module docstring: this is a confidentiality claim, not a
    content-trust claim.

    Attestation fetch/verify is amortized by the module-level
    verified-attestation bundle cache (phala-attestation-cache-v0): a MISS
    fetches a fresh attestation report, mints a fresh nonce, runs the full
    two-leg verify, and stores the verified bundle; a HIT within
    `min(ttl_env, 3600, not_after)` reuses the stored verified bundle and
    skips the attestation GET and both CLI legs. A hit re-presents the same
    verified (report, nonce) pair — it mints no fresh nonce and performs no
    attestation traffic — so the locality row marks it honestly via
    `extra.attestation_cache_hit` / `extra.attestation_cache_age_s`
    ("verified N seconds ago", never "verified this instant"). The spend
    cap still runs pre-I/O on hits and misses alike, and no code path
    reaches the POST with a bundle that did not pass the full two-leg gate
    at store time.
    """

    # Old ceiling this default replaces (agents-core-phala-gate-voicing-v0). Live
    # 2026-08-04 testing found deepseek/deepseek-v4-flash-0731 succeeding 1-of-3
    # attempts, both failures our own 30s read timeout, not upstream error — a
    # 30s ceiling shreds Council deliberation turns and presents as model
    # unreliability. `_LEGACY_TIMEOUT_SECS` is kept as the threshold for the
    # latency-warning signal below so a call that would have failed under the
    # old ceiling stays discoverable rather than silently absorbed by the raise.
    _LEGACY_TIMEOUT_SECS = 30.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        verify_target: str | None = None,
        session: requests.Session | None = None,
        # Raised from 30.0 to match the other deliberation adapters (LlamaAdapter
        # 300s, GravityWellAdapter 300s) — caller-overridable, unchanged contract.
        timeout: float = 300.0,
    ):
        self._api_key = api_key or os.environ.get("PHALA_API_KEY")
        # PHALA_BASE_URL (Piece 3b, agents-core-phala-aci1-restoration-v0):
        # points this client's TRAFFIC at the loopback aci-serve verifying
        # hop instead of directly at inference.phala.com — same os.environ
        # fallback pattern as PHALA_API_KEY above, explicit arg always wins.
        self._base_url = (base_url or os.environ.get("PHALA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        # PHALA_VERIFY_TARGET is a distinct concept from PHALA_BASE_URL: what
        # the online `aci verify` leg audits, NOT where traffic is routed.
        # Defaults to the real upstream (Erah 2026-08-08: "split the config
        # — verify the upstream, route through the hop"), so the online leg
        # verifies Phala's actual TLS-terminating service rather than the
        # loopback hop it can never TLS-handshake with. Same env-fallback
        # pattern, explicit arg always wins.
        self._verify_target = (
            verify_target or os.environ.get(PHALA_VERIFY_TARGET_ENV) or DEFAULT_BASE_URL
        ).rstrip("/")
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
        """Fetch + verify attestation, seal `messages[].content]` IF the
        gateway advertises a supported e2ee version, POST
        /v1/chat/completions, and decrypt the response when sealed. Raises
        `ReportVerificationError` if the report fails binding verification;
        `E2eeNotAppliedError` if the gateway advertised e2ee support, this
        call sealed to it, and the response didn't carry
        `x-e2ee-applied: true`; `PlaintextChannelNotLoopbackError` if the
        gateway does NOT advertise e2ee and `base_url` is not loopback —
        never silently falls back to an unverified/unsealed call over a
        non-local hop; `PhalaSpendCapExceededError` if today's own-ledger
        spend for this cost class is already at or over the daily cap —
        refused before any network I/O, fail closed
        (agents-core-phala-spend-cap-v0)."""
        _check_phala_spend_cap()

        cache_key = (self._base_url, self._verify_target)

        # --- verified-attestation bundle cache (phala-attestation-cache-v0).
        # A HIT skips the attestation GET and BOTH CLI legs: the stored
        # bundle passed the full two-leg gate at store time, and the
        # (report, nonce) pair is re-presented as-is — no fresh nonce, no
        # attestation traffic. The loopback verify-target refusal fires
        # inside require_verified_report_binding on the miss path BEFORE
        # any subprocess (it fires before the fetch today; the hit path
        # never re-reads a target that could not have been verified).
        cache_hit = False
        cache_age_s: float | None = None
        cached = _attestation_cache.get(cache_key)
        if cached is not None:
            cached_at, bundle = cached
            # Hit check in the monotonic domain (house style, matching
            # _online_verify_cache). The store-time ttl_effective was
            # clamped to min(ttl_env, ceiling, max(0.0, not_after -
            # store_now_wall)), so a wall-clock-expired report (not_after
            # in the past) was never stored. Re-checking the store-time
            # ttl_env alone would drift the not_after clamp by the wall
            # clock's movement since store time (the two clocks are not
            # locked), so the hit additionally requires the report's
            # keyset to still be within its own validity window (L4) —
            # this is the wall-clock clamp winning over the monotonic ttl.
            if time.monotonic() - cached_at < _attestation_cache_ttl_secs() and bundle.not_after > time.time():
                cache_hit = True
                cache_age_s = time.monotonic() - cached_at

        if cache_hit:
            report = bundle.report
            nonce = bundle.nonce
            verification = bundle.verification
        else:
            if nonce is None:
                nonce = os.urandom(32).hex()  # aci/1 requires 64 hex chars (32 bytes).

            # verify_report_binding's base_url means "the thing to verify" —
            # pass the verification target, not the traffic route:
            # self._base_url is pinned to the loopback aci-serve hop and can
            # never TLS-handshake (id-6 needs a live TLS channel to the real
            # service).
            try:
                report = self.fetch_attestation(nonce=nonce)
                verification = require_verified_report_binding(report, nonce, base_url=self._verify_target)
            except (AciError, requests.RequestException):
                # Fail-closed invalidation: a toolchain fault (any AciError
                # subclass — the verify legs, the loopback-target refusal,
                # the plaintext-not-loopback refusal, E2eeNotAppliedError)
                # or a fetch HTTP failure means the bundle is not
                # trustworthy; drop any stale entry for this key and
                # re-raise. Never degrade to an unverified send.
                _attestation_cache.pop(cache_key, None)
                raise

            # Insert the bundle IMMEDIATELY after verification, BEFORE the
            # POST: a transient POST failure does not invalidate a still-
            # valid bundle (the intended amortization path). ttl_env <= 0
            # disables the cache (never store) — not a verify-off switch.
            ttl_env = _attestation_cache_ttl_secs()
            if ttl_env > 0:
                not_after = (
                    report.get("attestation", {}).get("workload_keyset", {}).get("not_after")
                    or 0
                )
                # Clamp evaluated at STORE time, ceiling IN the expression:
                # min(ttl_env, ceiling, max(0.0, not_after - now_wall)).
                # A wall-clock-expired report (not_after in the past) is a
                # miss — never stored, never served.
                ttl_effective = min(
                    ttl_env,
                    _ATTESTATION_CACHE_TTL_CEILING_SECS,
                    max(0.0, float(not_after) - time.time()),
                )
                if ttl_effective > 0:
                    _attestation_cache[cache_key] = (
                        time.monotonic(),
                        _VerifiedAttestation(
                            report=report,
                            nonce=nonce,
                            verification=verification,
                            keyset_digest=verification.workload_keyset_digest,
                            cached_at_monotonic=time.monotonic(),
                            not_after=int(not_after),
                        ),
                    )

        capabilities = report.get("service_capabilities") or {}
        supported_versions = capabilities.get("supported_e2ee_versions") or []
        seal = _SUPPORTED_E2EE_VERSION in supported_versions

        channel = None
        e2ee_headers: dict = {}
        if seal:
            channel = open_e2ee_channel(report, verification)
            sealed_messages, e2ee_headers = channel.seal_messages(messages, model)
        else:
            if not _is_loopback_url(self._base_url):
                raise PlaintextChannelNotLoopbackError(
                    f"gateway does not advertise a supported e2ee version "
                    f"(supported_e2ee_versions={supported_versions!r}) and base_url "
                    f"{self._base_url!r} is not loopback — refusing to send plaintext "
                    f"off-box. Point base_url/PHALA_BASE_URL at a loopback-pinned "
                    f"verifying hop (see systemd/aci-serve.service)."
                )
            sealed_messages = messages

        body = dict(extra_body or {})
        body["model"] = model
        body["messages"] = sealed_messages

        headers = dict(self._auth_headers())
        headers.update(e2ee_headers)

        start = time.monotonic()
        try:
            resp = self._session.post(
                f"{self._base_url}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=self._timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Keyset rotation rejected at the hop: a POST 400/401/403/422
            # means the gateway no longer accepts the cached keyset — the
            # bundle is stale, so pop the entry (the node2 grammar-degrade
            # retry then re-verifies fresh on its next call) and re-raise
            # fail-closed. Transient statuses (5xx, ...) do NOT pop: the
            # bundle is still valid, the failure is the gateway's.
            status = getattr(e.response, "status_code", None)
            if status in _ATTESTATION_POST_REJECT_STATUSES:
                _attestation_cache.pop(cache_key, None)
            raise

        applied = resp.headers.get("x-e2ee-applied", "").strip().lower() == "true"
        response_json = resp.json()
        if seal and applied:
            response_json = channel.open_response(response_json)

        e2ee_applied = bool(seal and applied)

        # Leg 1 (agents-core-phala-spend-cap-v0): the gateway returns an
        # authoritative per-call `usage.cost` from the metered provider,
        # synchronously, in the response body — not an estimate, not the
        # (untrusted-fresh) usage panel. Thread it into the ledger row so
        # Leg 2's cap can sum OUR OWN recorded spend instead of polling
        # anything. If usage or cost is absent, record cost_usd=None with a
        # distinct marker — never substitute an estimate or a silent zero,
        # which would make the cap under-count.
        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        cost_usd = None
        usage_extra: dict[str, Any] = {}
        if isinstance(usage, dict):
            for field_name in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
                value = usage.get(field_name)
                if isinstance(value, (int, float)):
                    usage_extra[field_name] = value
            raw_cost = usage.get("cost")
            if isinstance(raw_cost, (int, float)):
                cost_usd = raw_cost
        usage_extra["cost_unpriced"] = cost_usd is None

        locality.record(
            requested_operator="phala-tee",
            served_model=model,
            host="phala",
            cost_class="paid-phala-tee",
            seam="phala_tee",
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            # `ok` is the call-completed claim only (HTTP succeeded above) —
            # it must never be collapsed with the e2ee_applied claim (module
            # docstring, Piece 3 of agents-core-phala-aci1-restoration-v0).
            ok=True,
            extra={
                "receipt_id": resp.headers.get("x-receipt-id"),
                "workload_keyset_digest": verification.workload_keyset_digest,
                "e2ee_applied": e2ee_applied,
                # Honest locality row: which hop actually carried the request.
                "channel": "e2ee" if seal else "plaintext-loopback",
                "report_verified": verification.ok,
                # id-5 custody allowance, surfaced — never absorbed silently.
                "custody_unverified": verification.custody_skipped,
                # Verified-attestation cache status, ADDITIVE in extra only
                # (spend-cap ledger-row doctrine: report_verified /
                # custody_unverified / e2ee_applied / channel above are
                # unchanged). "Verified 14 minutes ago" must never read as
                # "verified this instant".
                "attestation_cache_hit": cache_hit,
                "attestation_cache_age_s": cache_age_s,
                # Distinct signal (Facets `transmuter`, 2026-08-04) paired with the
                # 30s→300s timeout raise above: a call that would have failed under
                # the old ceiling must not simply succeed silently at 200s — flag it
                # so a slow seat stays discoverable instead of absorbed.
                "latency_warning": duration_ms > self._LEGACY_TIMEOUT_SECS * 1000,
                **usage_extra,
            },
        )

        if seal and not applied:
            raise E2eeNotAppliedError(
                "Phala gateway advertised e2ee support and this request sealed to "
                "it, but the response did not carry x-e2ee-applied: true — "
                "refusing to treat this response as confidential."
            )

        return response_json
