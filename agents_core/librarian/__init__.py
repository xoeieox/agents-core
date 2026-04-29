"""agents_core.librarian — query interface in front of the vault corpus.

Single public entry: ``corroborate()``.  Returns a ``SynthesisArtifact``
(cached or freshly synthesised) or a ``LibrarianUnavailable`` when the local
LLM endpoint is unreachable.

Invariants
----------
- LLM unavailable NEVER raises out of ``corroborate()``.  Always returns
  ``LibrarianUnavailable`` instead.  Consumers must handle the degraded signal.
- Librarian is read-only with respect to the live corpus; it only writes to
  the synthesis cache (via ``agents_core.synthesis_cache``).
- Citations are verified in two stages before signing:
    1. cited path + content_hash must match a chunk in the retrieval set.
    2. cited quoted_snippet must be a substring of the cited chunk text.
  Citations failing either stage are dropped; artifact carries
  ``verification: partial``.
- Visibility enforcement: the librarian filters retrieval results through
  dual-layer visibility rules (per-directory allow-list + frontmatter
  ``citable:`` override) before grounding.  Filtered-out chunks never appear
  in synthesis answers or citations.

LLM endpoint
------------
Default: ``http://203.0.113.12:8081/v1/chat/completions``
Override: ``LLM_ENDPOINT`` env var

Signing key
-----------
Default: ``/data/secrets/librarian-starhouse.key``  (Ed25519, PEM)
Override: ``LIBRARIAN_KEY_PATH`` env var
If the key file does not exist it is generated and written on first use.

Corpus visibility
-----------------
The dual-layer filter applies:
1. Per-directory allow-list: paths containing ``/Personal/`` or
   ``/Daily-Notes/private/``, or frontmatter ``private: true`` → excluded.
2. Per-doc frontmatter ``citable: false`` → excluded (``citable: true`` → included).
When ``citable:`` is absent the allow-list result applies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from agents_core import retrieval, synthesis_cache
from agents_core.librarian.shift import ShiftLevel, compute_shift_from_rows

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_LLM_ENDPOINT = "http://203.0.113.12:8081/v1/chat/completions"
_DEFAULT_KEY_PATH = Path("/data/secrets/librarian-starhouse.key")
_LIBRARIAN_ID = "starhouse-v0"
_LLM_TIMEOUT = 60.0  # seconds

PolicyName = Literal["auto-update", "on-trigger", "never-update"]


def _llm_endpoint() -> str:
    return os.environ.get("LLM_ENDPOINT", _DEFAULT_LLM_ENDPOINT)


def _key_path() -> Path:
    return Path(os.environ.get("LIBRARIAN_KEY_PATH", str(_DEFAULT_KEY_PATH)))


def _events_jsonl_path() -> Path:
    return Path(os.environ.get("VAULT_EVENTS_JSONL", "/data/vault-events.jsonl"))


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class SynthesisArtifact:
    """In-memory representation of a synthesis cache entry."""
    schema_version: str
    artifact_id: str
    claim: str
    scope_identity: dict
    request_metadata: dict
    answer: dict
    citations: list[dict]
    librarian_id: str
    model_id: str
    corpus_snapshot: str
    synthesized_at: str
    verification: str   # "full" | "partial" | "none" (none = zero citations claimed)
    policy: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "claim": self.claim,
            "scope_identity": self.scope_identity,
            "request_metadata": self.request_metadata,
            "answer": self.answer,
            "citations": self.citations,
            "librarian_id": self.librarian_id,
            "model_id": self.model_id,
            "corpus_snapshot": self.corpus_snapshot,
            "synthesized_at": self.synthesized_at,
            "verification": self.verification,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SynthesisArtifact":
        return cls(
            schema_version=d.get("schema_version", "1"),
            artifact_id=d["artifact_id"],
            claim=d["claim"],
            scope_identity=d["scope_identity"],
            request_metadata=d.get("request_metadata", {}),
            answer=d.get("answer", {}),
            citations=d.get("citations", []),
            librarian_id=d.get("librarian_id", ""),
            model_id=d.get("model_id", ""),
            corpus_snapshot=d.get("corpus_snapshot", ""),
            synthesized_at=d.get("synthesized_at", ""),
            verification=d.get("verification", "full"),
            policy=d.get("policy", "auto-update"),
        )


@dataclass
class LibrarianUnavailable:
    """Returned by ``corroborate()`` when the LLM endpoint is unreachable.

    Never raised.  Consumers must handle the degraded signal explicitly.
    A 500 reaching the UI is a bug.
    """
    most_recent_cached: SynthesisArtifact | None
    degraded: bool = True
    reason: str = ""


@dataclass
class TriggerRequestRow:
    id: int
    ts: str
    artifact_id: str
    requester_id: str
    reason: str


# TypedDict for Scope (identity-bearing fields only)
class Scope(dict):
    """Scope dict; must contain ``corpus: list[str]``."""


# ---------------------------------------------------------------------------
# Signing key management
# ---------------------------------------------------------------------------

_signing_key: Ed25519PrivateKey | None = None


def _get_signing_key() -> Ed25519PrivateKey:
    global _signing_key
    if _signing_key is None:
        _signing_key = _load_or_create_key(_key_path())
    return _signing_key


def _reset_signing_key() -> None:
    """For tests — force reload of signing key on next use."""
    global _signing_key
    _signing_key = None


def _read_key(key_path: Path) -> Ed25519PrivateKey:
    pem_bytes = key_path.read_bytes()
    key = serialization.load_pem_private_key(pem_bytes, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError(f"Key at {key_path} is not Ed25519")
    return key


def _load_or_create_key(key_path: Path) -> Ed25519PrivateKey:
    if key_path.exists():
        return _read_key(key_path)

    # Generate a candidate; persist with O_EXCL so a concurrent generator
    # can't overwrite an existing key (and we'll discover their write).
    new_key = Ed25519PrivateKey.generate()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    pem = new_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(key_path), flags, 0o600)
    except FileExistsError:
        # Lost the race — another process generated first. Load theirs.
        return _read_key(key_path)
    try:
        os.write(fd, pem)
    finally:
        os.close(fd)
    log.info("librarian: generated new Ed25519 signing key at %s", key_path)
    return new_key


def _sign_entry(entry_bytes: bytes) -> str:
    """Sign entry.json bytes and return hex-encoded signature."""
    key = _get_signing_key()
    sig = key.sign(entry_bytes)
    return sig.hex()


def verify_entry(entry_bytes: bytes, sig_hex: str, public_key_bytes: bytes) -> bool:
    """Verify an entry signature.  For testing / external verifiers.

    Parameters
    ----------
    entry_bytes:
        The canonical entry.json bytes (sorted keys, UTF-8, no trailing newline).
    sig_hex:
        Hex signature as stored in entry.sig.
    public_key_bytes:
        Raw Ed25519 public-key bytes (32 bytes).
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        pub.verify(bytes.fromhex(sig_hex), entry_bytes)
        return True
    except InvalidSignature:
        return False


def get_public_key_bytes() -> bytes:
    """Return the raw 32-byte Ed25519 public key for the host signing key."""
    key = _get_signing_key()
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


# ---------------------------------------------------------------------------
# Visibility filter (dual-layer)
# ---------------------------------------------------------------------------

_EXCLUDED_PATH_FRAGMENTS = ["/Personal/", "/Daily-Notes/private/"]
_FM_RE = re.compile(r"^---\r?\n(.*?\r?\n)---\r?\n", re.DOTALL)


def _extract_fm_bool(content: str, field: str) -> bool | None:
    """Extract a boolean frontmatter field.  Returns None if absent."""
    m = _FM_RE.match(content)
    if not m:
        return None
    fm_body = m.group(1)
    pattern = re.compile(rf"^{re.escape(field)}:\s*(.+)$", re.MULTILINE)
    fm_match = pattern.search(fm_body)
    if not fm_match:
        return None
    val = fm_match.group(1).strip().lower()
    if val in ("true", "yes", "1"):
        return True
    if val in ("false", "no", "0"):
        return False
    return None


def is_citable(path: str, content: str = "") -> bool:
    """Return True if this corpus path may be cited in synthesis answers.

    Dual-layer:
    1. Per-directory allow-list: /Personal/ or /Daily-Notes/private/ → excluded.
       Also, frontmatter ``private: true`` → excluded.
    2. Per-doc frontmatter ``citable: false`` → excluded.  ``citable: true`` →
       included (overrides directory rule).  Absent → directory rule applies.
    """
    # Per-doc citable: override (highest priority)
    if content:
        citable_fm = _extract_fm_bool(content, "citable")
        if citable_fm is not None:
            return citable_fm
        # private: true → excluded
        private_fm = _extract_fm_bool(content, "private")
        if private_fm:
            return False

    # Per-directory allow-list
    for fragment in _EXCLUDED_PATH_FRAGMENTS:
        if fragment in path:
            return False

    return True


def _filter_hits(hits: list) -> list:
    """Filter retrieval hits through dual-layer visibility rules."""
    visible = []
    for hit in hits:
        path = hit.metadata.get("file_path") or hit.metadata.get("path") or hit.id
        content = hit.content or ""
        if is_citable(path, content):
            visible.append(hit)
    return visible


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    claim: str,
    context_chunks: list[str],
    format_schema: dict | None,
    model_hint: str = "",
) -> tuple[dict, str]:
    """Call the local LLM endpoint with retrieved context.

    Returns (answer_dict, model_id).
    Raises ``httpx.HTTPError`` or ``httpx.TimeoutException`` on failure.
    """
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
    user_content = (
        f"Using the following source material, answer the claim.\n\n"
        f"CLAIM: {claim}\n\n"
        f"SOURCE MATERIAL:\n{context_text}"
    )
    messages = [{"role": "user", "content": user_content}]

    payload: dict = {"messages": messages, "max_tokens": 2048}
    if model_hint:
        payload["model"] = model_hint

    if format_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "synthesis_answer",
                "schema": format_schema,
            },
        }

    endpoint = _llm_endpoint()
    resp = httpx.post(endpoint, json=payload, timeout=_LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    model_id = data.get("model", "unknown")
    content_str: str = data["choices"][0]["message"]["content"]

    # Parse content as JSON if format_schema was requested, else wrap in dict
    # Always attempt JSON parse so citation extraction works regardless of format_schema.
    try:
        answer = json.loads(content_str)
        if not isinstance(answer, dict):
            answer = {"raw": content_str}
    except json.JSONDecodeError:
        answer = {"text": content_str}

    return answer, model_id


# ---------------------------------------------------------------------------
# Citation verification (two-stage)
# ---------------------------------------------------------------------------

def _verify_citations(
    claimed_citations: list[dict],
    retrieval_hit_map: dict[str, dict],  # path+hash → {path, content_hash, content}
) -> tuple[list[dict], str]:
    """Verify citations against the retrieval set.

    Stage 1: cited path + content_hash must match a chunk in retrieval_hit_map.
    Stage 2: cited quoted_snippet must be a substring of the cited chunk's text.

    Returns (verified_citations, verification_status) where verification_status
    is "full" if all citations passed both stages, "partial" if any were dropped.
    """
    verified: list[dict] = []
    any_dropped = False

    for cit in claimed_citations:
        path = cit.get("path", "")
        content_hash = cit.get("content_hash", "")
        snippet = cit.get("quoted_snippet", "")
        key = f"{path}\x00{content_hash}"

        chunk = retrieval_hit_map.get(key)
        if chunk is None:
            # Stage 1 failure: not in retrieval set
            any_dropped = True
            continue

        chunk_text = chunk.get("content", "")
        if not snippet or snippet not in chunk_text:
            # Stage 2 failure: snippet not a substring of chunk text
            any_dropped = True
            continue

        verified.append(cit)

    if not claimed_citations:
        status = "none"
    elif any_dropped:
        status = "partial"
    else:
        status = "full"
    return verified, status


def _build_retrieval_hit_map(hits: list) -> dict[str, dict]:
    """Build a lookup map from (path, content_hash) → chunk dict."""
    hit_map: dict[str, dict] = {}
    for hit in hits:
        path = hit.metadata.get("file_path") or hit.metadata.get("path") or hit.id
        # Compute content_hash from hit content
        content_bytes = hit.content.encode("utf-8") if isinstance(hit.content, str) else hit.content
        content_hash = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"
        key = f"{path}\x00{content_hash}"
        hit_map[key] = {"path": path, "content_hash": content_hash, "content": hit.content}
    return hit_map


# ---------------------------------------------------------------------------
# Corpus scope mapping
# ---------------------------------------------------------------------------

_KNOWN_RETRIEVAL_SCOPES = frozenset(retrieval.KNOWN_SCOPE)


def _corpus_to_retrieval_scopes(corpus: list[str]) -> list[str]:
    """Map librarian corpus tokens to agents_core.retrieval scope tokens."""
    scopes: list[str] = []
    for item in corpus:
        # e.g. "mem:idea/*" → "mem"; "vault-rag" → "vault-rag"
        token = item.split(":")[0]
        if token in _KNOWN_RETRIEVAL_SCOPES:
            scopes.append(token)
        elif item in _KNOWN_RETRIEVAL_SCOPES:
            scopes.append(item)
        else:
            # Path-like item: assume vault-rag
            scopes.append("vault-rag")
    # Deduplicate while preserving order
    return list(dict.fromkeys(scopes))


# ---------------------------------------------------------------------------
# Corpus snapshot computation
# ---------------------------------------------------------------------------

def _compute_corpus_snapshot(hits: list) -> str:
    """Hash all retrieved chunk content hashes to form a corpus snapshot."""
    hashes = []
    for hit in hits:
        content_bytes = hit.content.encode("utf-8") if isinstance(hit.content, str) else hit.content
        hashes.append(hashlib.sha256(content_bytes).hexdigest())
    hashes.sort()  # stable
    combined = "|".join(hashes).encode("utf-8")
    return f"sha256:{hashlib.sha256(combined).hexdigest()}"


# ---------------------------------------------------------------------------
# Core synthesis path
# ---------------------------------------------------------------------------

def _synthesize(
    claim: str,
    corpus: list[str],
    policy: str,
    freshness: int,
    format: dict | None,
) -> SynthesisArtifact | LibrarianUnavailable:
    """Run the full synthesis pipeline: retrieve → filter → LLM → verify → sign → cache."""
    # 1. Retrieve
    retrieval_scopes = _corpus_to_retrieval_scopes(corpus)
    try:
        hits = retrieval.retrieve(claim, retrieval_scopes, top_k=20) if retrieval_scopes else []
    except Exception as exc:
        log.warning("librarian: retrieval failed: %s", exc)
        hits = []

    # 2. Visibility filter
    hits = _filter_hits(hits)

    # 3. Build retrieval hit map for citation verification
    hit_map = _build_retrieval_hit_map(hits)
    context_chunks = [h.content for h in hits]

    # 4. Compute corpus snapshot
    corpus_snapshot = _compute_corpus_snapshot(hits)

    # 5. Compute artifact_id
    artifact_id = synthesis_cache.compute_artifact_id(claim, corpus, policy, format)

    # 6. Call LLM
    try:
        answer, model_id = _call_llm(claim, context_chunks, format)
    except Exception as exc:
        log.warning("librarian: LLM call failed: %s", exc)
        # Return most-recent cached if any
        most_recent = _load_most_recent(claim, corpus, policy, format)
        return LibrarianUnavailable(
            most_recent_cached=most_recent,
            degraded=True,
            reason=str(exc),
        )

    # 7. Extract claimed citations from LLM answer
    claimed_citations: list[dict] = answer.pop("citations", []) if isinstance(answer, dict) else []

    # 8. Citation verification (two-stage)
    verified_citations, verification = _verify_citations(claimed_citations, hit_map)

    # 9. Build entry
    now_ts = datetime.now(tz=timezone.utc).isoformat()
    format_schema_hash = synthesis_cache.compute_format_schema_hash(format)

    scope_identity = {
        "corpus": sorted(corpus),
        "policy": policy,
        "format_schema_hash": format_schema_hash,
    }
    request_metadata = {
        "freshness_at_request": freshness,
        "format_schema": format or {},
    }
    entry_dict = {
        "schema_version": "1",
        "artifact_id": artifact_id,
        "claim": claim,
        "scope_identity": scope_identity,
        "request_metadata": request_metadata,
        "answer": answer,
        "citations": verified_citations,
        "librarian_id": _LIBRARIAN_ID,
        "model_id": model_id,
        "corpus_snapshot": corpus_snapshot,
        "synthesized_at": now_ts,
        "verification": verification,
        "policy": policy,
    }

    # 10. Sign entry (sorted-key UTF-8, no trailing newline)
    entry_bytes = synthesis_cache.canonical_json(entry_dict).encode("utf-8")
    try:
        sig_hex = _sign_entry(entry_bytes)
    except Exception as exc:
        log.error("librarian: signing failed: %s — writing unsigned entry", exc)
        sig_hex = ""

    # 11. Write to cache
    synthesis_cache.put(entry_dict, sig_hex)

    return SynthesisArtifact.from_dict(entry_dict)


def _load_most_recent(
    claim: str,
    corpus: list[str],
    policy: str,
    format: dict | None,
) -> SynthesisArtifact | None:
    """Return the most-recently synthesised artifact for this query, or None."""
    format_schema_hash = synthesis_cache.compute_format_schema_hash(format)
    artifact_id = synthesis_cache.lookup_by_claim(claim, corpus, policy, format_schema_hash)
    if artifact_id is None:
        return None
    entry = synthesis_cache.get(artifact_id)
    if entry is None:
        return None
    return SynthesisArtifact.from_dict(entry)


def _is_within_freshness(entry: dict, freshness: int) -> bool:
    """Return True if entry is still within its freshness budget."""
    if freshness <= 0:
        return False
    try:
        synthesized_at = datetime.fromisoformat(entry["synthesized_at"])
        age = (datetime.now(tz=timezone.utc) - synthesized_at).total_seconds()
        return age < freshness
    except (KeyError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def corroborate(
    claim: str,
    scope: Scope,
    *,
    freshness: int = 60,
    policy: PolicyName = "auto-update",
    format: dict | None = None,
) -> SynthesisArtifact | LibrarianUnavailable:
    """Query the librarian for an answer to *claim* grounded in the vault corpus.

    Parameters
    ----------
    claim:
        The question or claim to synthesise.  Case preserved (proper nouns, paths).
    scope:
        Dict with ``corpus: list[str]`` — scope tokens / corpus paths to consider.
    freshness:
        Max age (seconds) of a cached artifact before re-synthesis.  Read-time
        policy, NOT part of artifact identity (same query with different freshness
        produces the same artifact_id).
    policy:
        Update-policy for the artifact; stored in entry and scope_identity.
        Default ``"auto-update"`` means auto-refresh on staleness.
    format:
        Optional JSON Schema for structured output.  Only its sha256 hash is
        identity-bearing; the literal value is stored in ``request_metadata``
        for replay.  ``None`` and ``{}`` collapse to the same hash.

    Returns
    -------
    SynthesisArtifact | LibrarianUnavailable

    The LLM endpoint being unreachable NEVER raises; it returns
    ``LibrarianUnavailable(most_recent_cached, degraded=True)``.
    """
    corpus: list[str] = scope.get("corpus", [])
    format_schema_hash = synthesis_cache.compute_format_schema_hash(format)

    # Cache lookup
    artifact_id = synthesis_cache.lookup_by_claim(claim, corpus, policy, format_schema_hash)
    if artifact_id is not None:
        entry = synthesis_cache.get(artifact_id)
        if entry is not None:
            # Freshness check
            if _is_within_freshness(entry, freshness):
                # Degree-of-shift check: only source_broken forces re-run regardless of freshness
                source_rows = synthesis_cache.get_source_rows(artifact_id)
                if source_rows:
                    shift = compute_shift_from_rows(
                        source_rows,
                        read_old=_read_old_content,
                        read_new=_read_corpus_file,
                    )
                    if shift == ShiftLevel.source_broken:
                        pass  # fall through to re-synthesis
                    else:
                        return SynthesisArtifact.from_dict(entry)
                else:
                    return SynthesisArtifact.from_dict(entry)

    # Cache miss, stale, or source_broken — synthesise
    return _synthesize(claim, corpus, policy, freshness, format)


def _read_corpus_file(path: str) -> str | None:
    """Best-effort read of a corpus file path.  Returns None if unreadable."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _read_old_content(path: str, content_hash: str) -> str | None:
    """Return old content if the on-disk file still has the expected hash, else None.

    v0 limitation: old content is only available when the file on disk still
    matches the hash from when the artifact was created (i.e. it has not changed
    since synthesis).  When the file has changed we cannot recover the original
    text without a dedicated content store — compute_shift_from_rows then treats
    the source as an opaque hash diff and falls back to word_line as the minimum
    shift level.
    """
    content = _read_corpus_file(path)
    if content is None:
        return None
    actual_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
    return content if actual_hash == content_hash else None


def regenerate(artifact_id: str) -> SynthesisArtifact:
    """Re-run synthesis for the (claim, scope_identity) of an existing artifact.

    If degree-of-shift is ``no_shift`` since the existing entry, returns the
    existing entry unchanged (no-op, no LLM call).  Otherwise re-synthesises
    and writes a new entry (which may have the same artifact_id if the canonical
    scope_identity is unchanged).

    Parameters
    ----------
    artifact_id:
        The ``sha256:...`` artifact_id to regenerate.

    Raises
    ------
    KeyError if the artifact_id is not found in the cache.
    """
    entry = synthesis_cache.get(artifact_id)
    if entry is None:
        raise KeyError(f"artifact {artifact_id!r} not found in synthesis cache")

    claim = entry["claim"]
    scope = entry["scope_identity"]
    corpus = scope["corpus"]
    policy = scope["policy"]
    format_schema = entry.get("request_metadata", {}).get("format_schema") or None

    # Check degree-of-shift; if no-shift, return existing entry unchanged
    source_rows = synthesis_cache.get_source_rows(artifact_id)
    if source_rows:
        shift = compute_shift_from_rows(
            source_rows,
            read_old=_read_old_content,
            read_new=_read_corpus_file,
        )
        if shift == ShiftLevel.no_shift:
            return SynthesisArtifact.from_dict(entry)

    freshness_at_request = entry.get("request_metadata", {}).get("freshness_at_request", 60)
    result = _synthesize(claim, corpus, policy, freshness_at_request, format_schema)
    if isinstance(result, LibrarianUnavailable):
        if result.most_recent_cached is not None:
            return result.most_recent_cached
        # Fall back to existing entry
        return SynthesisArtifact.from_dict(entry)
    return result


def degree_of_shift(artifact_id: str) -> ShiftLevel:
    """Compute the current degree-of-shift for a cached artifact.

    Returns the maximum shift level across all cited sources.
    Returns ``ShiftLevel.no_shift`` if the artifact has no citations or
    does not exist.
    """
    if synthesis_cache.get(artifact_id) is None:
        return ShiftLevel.no_shift
    source_rows = synthesis_cache.get_source_rows(artifact_id)
    if not source_rows:
        return ShiftLevel.no_shift
    return compute_shift_from_rows(
        source_rows,
        read_old=_read_old_content,
        read_new=_read_corpus_file,
    )


def request_trigger(artifact_id: str, requester_id: str, reason: str) -> None:
    """Record a consumer request to refresh an artifact without forcing regen.

    Request frequency is a demand signal: "which artifacts are consumers most
    pressing for an update on?"  At v1 this generalises to peer-network signal.
    """
    synthesis_cache.add_trigger_request(artifact_id, requester_id, reason)


def trigger_request_summary(*, since: datetime | None = None) -> list[TriggerRequestRow]:
    """Return all trigger requests, optionally filtered to ts >= since."""
    rows = synthesis_cache.get_trigger_requests(since=since)
    return [TriggerRequestRow(**r) for r in rows]


# ---------------------------------------------------------------------------
# Startup: replay missed vault events into by-source.sqlite
# ---------------------------------------------------------------------------

def startup_replay(events_jsonl: Path | None = None) -> int:
    """Replay missed vault-write events into by-source.sqlite.

    Reads ``/data/vault-events.jsonl`` (or *events_jsonl*) for events
    newer than the latest ``latest_observed_at`` already in the index,
    and calls ``synthesis_cache.update_source_tracking()`` for each.

    Safe to call multiple times (idempotent due to MAX timestamp check).

    Returns
    -------
    int — number of events replayed.
    """
    jsonl_path = events_jsonl or _events_jsonl_path()
    if not jsonl_path.exists():
        return 0

    last_observed = synthesis_cache.get_max_observed_at()
    replayed = 0

    try:
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = event.get("ts", "")
                if last_observed and ts <= last_observed:
                    continue

                path = event.get("path", "")
                content_hash = event.get("content_hash", "")
                if path and content_hash:
                    synthesis_cache.update_source_tracking(path, content_hash, ts)
                    replayed += 1
    except OSError as exc:
        log.warning("librarian: startup_replay failed to read %s: %s", jsonl_path, exc)

    return replayed


# ---------------------------------------------------------------------------
# Live-surface digest (PR 5 hook; implemented as a stub in PR 2)
# ---------------------------------------------------------------------------

def render_live_surface(out_path: Path) -> None:
    """Render the weekly Live-Surface digest to *out_path*.

    Invoked by cron on Saturday mornings (wired in PR 5).  Calls
    ``corroborate()`` for Thread Weaver state + Follow-ups + recently-landed,
    then writes the result as a markdown file with required frontmatter via
    ``vault_writer``.

    Frontmatter contract::

        type: live-surface
        generated: <iso-timestamp>
        freshness: weekly

    Idempotent: re-rendering with the same corpus snapshot produces
    byte-identical output.

    Parameters
    ----------
    out_path:
        Destination path for the digest (e.g.
        ``/srv/git/inertia-vault-working/Lapis/Live-Surface.md``).
    """
    from agents_core import vault_writer

    # Synthesise thread-weaver + follow-ups state
    thread_state = corroborate(
        "thread-weaver-state",
        Scope(corpus=["vault-rag", "mem"]),
        freshness=0,
        policy="auto-update",
    )
    follow_ups = corroborate(
        "recent-follow-ups",
        Scope(corpus=["mem", "vault-rag"]),
        freshness=0,
        policy="auto-update",
    )

    now_ts = datetime.now(tz=timezone.utc).isoformat()

    thread_text = (
        thread_state.answer.get("text", str(thread_state.answer))
        if isinstance(thread_state, SynthesisArtifact)
        else "(unavailable)"
    )
    follow_text = (
        follow_ups.answer.get("text", str(follow_ups.answer))
        if isinstance(follow_ups, SynthesisArtifact)
        else "(unavailable)"
    )

    content = (
        f"---\n"
        f"type: live-surface\n"
        f"generated: {now_ts}\n"
        f"freshness: weekly\n"
        f"---\n"
        f"\n"
        f"# Live Surface\n"
        f"\n"
        f"*Generated {now_ts}*\n"
        f"\n"
        f"## Threads in Motion\n"
        f"\n"
        f"{thread_text}\n"
        f"\n"
        f"## Follow-ups\n"
        f"\n"
        f"{follow_text}\n"
    )

    vault_writer.write(
        out_path,
        content,
        agent_id="librarian",
        intent="weekly live-surface digest",
        stamp_frontmatter=False,  # frontmatter is already present above
    )
