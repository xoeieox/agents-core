"""agents_core.friction_test.driver — Driver Protocol + RadioOpDriver + CockpitDriver.

Each driver is stateless across scenarios (setup / teardown bracket one
scenario set). Exceptions from run_scenario are caught by the orchestrator
and converted to harness_error Observations.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from .observe import Observation
from .scenario import Scenario


@runtime_checkable
class Driver(Protocol):
    """Protocol every concrete driver must satisfy."""

    name: str

    def setup(self) -> None: ...
    def run_scenario(self, s: Scenario) -> Observation: ...
    def teardown(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_record(
    method: str,
    url: str,
    *,
    status: int,
    body: Any,
    latency_ms: float,
    ts: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "url": url,
        "status": status,
        "body": body,
        "latency_ms": latency_ms,
        "ts": ts,
    }


def _read_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    """Read all lines from a JSONL file, ignoring parse errors."""
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return lines


# ---------------------------------------------------------------------------
# RadioOpDriver
# ---------------------------------------------------------------------------

FOYER_BASE = "http://localhost:8402"
CONSULT_LOG_DIR = Path("/srv/lapis/radio/consults")
HARVEST_QUEUE_DIR = Path("/data/foyer/talk_it_out/harvest_queue")


class RadioOpDriver:
    """Drives the foyer talk-it-out flow.

    Flow: session/start → ingest×N → optional harvest → session/end.
    Bypasses Mac whisper.cpp; ingest endpoint is the probe surface.
    """

    name = "radio-op"

    def __init__(self, base_url: str = FOYER_BASE) -> None:
        self._base = base_url.rstrip("/")
        self._client: httpx.Client | None = None
        # Track open sessions so teardown can close them
        self._open_sessions: list[str] = []
        self._lock = threading.Lock()

    def setup(self) -> None:
        self._client = httpx.Client(timeout=30.0)
        # Probe liveness
        try:
            r = self._client.get(f"{self._base}/talk_it_out/panel")
            if r.status_code >= 500:
                raise RuntimeError(f"Foyer /talk_it_out/panel returned {r.status_code}")
        except Exception as exc:
            raise RuntimeError(f"Foyer liveness check failed: {exc}") from exc

    def teardown(self) -> None:
        if self._client is None:
            return
        with self._lock:
            sessions = list(self._open_sessions)
        for sid in sessions:
            try:
                self._client.post(
                    f"{self._base}/talk_it_out/session/end",
                    json={"session_id": sid},
                    timeout=10.0,
                )
            except Exception:
                pass
        self._client.close()
        self._client = None

    def run_scenario(self, s: Scenario) -> Observation:
        assert self._client is not None, "setup() must be called before run_scenario()"
        started_at = _now_iso()
        http_calls: list[dict[str, Any]] = []
        sse_events: list[dict[str, Any]] = []
        log_appends: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        session_id: str | None = None

        try:
            # 1. session/start
            label = f"friction-test-{s.scenario_id}"
            t0 = time.monotonic()
            r = self._client.post(
                f"{self._base}/talk_it_out/session/start",
                json={"label": label},
            )
            latency = (time.monotonic() - t0) * 1000
            body = _safe_json(r)
            http_calls.append(_http_record("POST", f"{self._base}/talk_it_out/session/start",
                                           status=r.status_code, body=body,
                                           latency_ms=latency, ts=_now_iso()))
            if r.status_code != 200:
                errors.append({"stage": "session_start", "status": r.status_code, "body": body})
                return Observation(
                    scenario_id=s.scenario_id,
                    started_at=started_at,
                    finished_at=_now_iso(),
                    http_calls=http_calls,
                    errors=errors,
                    harness_error=True,
                )
            session_id = body.get("session_id") if isinstance(body, dict) else None
            if not session_id:
                errors.append({"stage": "session_start", "error": "no session_id in response", "body": body})
                return Observation(
                    scenario_id=s.scenario_id,
                    started_at=started_at,
                    finished_at=_now_iso(),
                    http_calls=http_calls,
                    errors=errors,
                    harness_error=True,
                )

            with self._lock:
                self._open_sessions.append(session_id)

            # Snapshot consult-log BEFORE ingests
            consult_path = CONSULT_LOG_DIR / f"{session_id}.jsonl"
            lines_before = _read_jsonl_lines(consult_path)

            # 2. SSE stream — open in streaming mode, collect briefly
            sse_events = _collect_sse(
                self._client, f"{self._base}/talk_it_out/stream/{session_id}"
            )

            # 3. ingest segments
            segments = s.inputs.get("segments", [])
            for seg_spec in segments:
                segment_text = seg_spec.get("segment", "")
                ingest_ts = seg_spec.get("ts", _now_iso())
                # Handle delay spec
                delay_ms = seg_spec.get("delay_ms", 0)
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)

                t0 = time.monotonic()
                r = self._client.post(
                    f"{self._base}/talk_it_out/ingest",
                    json={"session_id": session_id, "segment": segment_text, "ts": ingest_ts},
                )
                latency = (time.monotonic() - t0) * 1000
                body = _safe_json(r)
                http_calls.append(_http_record(
                    "POST", f"{self._base}/talk_it_out/ingest",
                    status=r.status_code, body=body,
                    latency_ms=latency, ts=_now_iso(),
                ))
                if r.status_code >= 500:
                    errors.append({"stage": "ingest", "status": r.status_code, "body": body})

            # 4. optional harvest gesture
            if s.inputs.get("harvest", False):
                t0 = time.monotonic()
                r = self._client.post(
                    f"{self._base}/talk_it_out/session/harvest",
                    json={"session_id": session_id},
                )
                latency = (time.monotonic() - t0) * 1000
                body = _safe_json(r)
                http_calls.append(_http_record(
                    "POST", f"{self._base}/talk_it_out/session/harvest",
                    status=r.status_code, body=body,
                    latency_ms=latency, ts=_now_iso(),
                ))

            # 5. session/end
            t0 = time.monotonic()
            r = self._client.post(
                f"{self._base}/talk_it_out/session/end",
                json={"session_id": session_id},
            )
            latency = (time.monotonic() - t0) * 1000
            http_calls.append(_http_record(
                "POST", f"{self._base}/talk_it_out/session/end",
                status=r.status_code, body=_safe_json(r),
                latency_ms=latency, ts=_now_iso(),
            ))
            with self._lock:
                if session_id in self._open_sessions:
                    self._open_sessions.remove(session_id)

            # 6. Snapshot consult-log AFTER, compute new lines
            lines_after = _read_jsonl_lines(consult_path)
            new_lines = lines_after[len(lines_before):]
            for line in new_lines:
                log_appends.append({
                    "file": str(consult_path),
                    "line": line,
                })

            # 7. Check harvest queue marker
            harvest_marker_path = HARVEST_QUEUE_DIR / f"{session_id}.json"
            harvest_marker_exists = harvest_marker_path.exists()
            harvest_marker_data = None
            if harvest_marker_exists:
                try:
                    harvest_marker_data = json.loads(harvest_marker_path.read_text())
                except Exception:
                    harvest_marker_data = {}
            log_appends.append({
                "file": str(harvest_marker_path),
                "harvest_marker_exists": harvest_marker_exists,
                "harvest_marker_data": harvest_marker_data,
                "session_id": session_id,
                "label": label,
            })

        except Exception as exc:
            errors.append({"stage": "run_scenario", "error": str(exc)})
            return Observation(
                scenario_id=s.scenario_id,
                started_at=started_at,
                finished_at=_now_iso(),
                http_calls=http_calls,
                sse_events=sse_events,
                log_appends=log_appends,
                errors=errors,
                harness_error=True,
            )

        return Observation(
            scenario_id=s.scenario_id,
            started_at=started_at,
            finished_at=_now_iso(),
            http_calls=http_calls,
            sse_events=sse_events,
            log_appends=log_appends,
            errors=errors,
            harness_error=False,
        )


# ---------------------------------------------------------------------------
# CockpitDriver
# ---------------------------------------------------------------------------

COCKPIT_BASE = "http://localhost:8400"
MEM_DB_PATH = Path("/srv/agents/mem.db")
VAULT_AUDIT_DIR = Path("/data/vault-audit")
COMMENT_STORE_DIR = Path("/srv/lapis/targets/comments")


class CockpitDriver:
    """Drives the cockpit dashboard HTTP surface + mem.db assertions."""

    name = "cockpit"

    def __init__(self, base_url: str = COCKPIT_BASE) -> None:
        self._base = base_url.rstrip("/")
        self._client: httpx.Client | None = None

    def setup(self) -> None:
        self._client = httpx.Client(timeout=30.0)
        try:
            r = self._client.get(f"{self._base}/api/status", timeout=5.0)
            if r.status_code >= 500:
                raise RuntimeError(f"Cockpit /api/status returned {r.status_code}")
        except Exception as exc:
            raise RuntimeError(f"Cockpit liveness check failed: {exc}") from exc

    def teardown(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def run_scenario(self, s: Scenario) -> Observation:
        assert self._client is not None
        started_at = _now_iso()
        http_calls: list[dict[str, Any]] = []
        mem_writes: list[dict[str, Any]] = []
        vault_writes: list[dict[str, Any]] = []
        log_appends: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        try:
            method = s.inputs.get("method", "GET").upper()
            path = s.inputs.get("path", "/api/status")
            body_in = s.inputs.get("body", None)
            url = f"{self._base}{path}"

            # Snapshot mem.db pm/* keys before
            mem_before = _snapshot_mem_db(MEM_DB_PATH)

            # Snapshot vault audit count before
            vault_before = _snapshot_vault_audit(VAULT_AUDIT_DIR)

            # Snapshot comment-store line counts before (for directive tests)
            tid = s.inputs.get("tid")
            comment_lines_before = 0
            comment_path: Path | None = None
            comment_write_ts: float | None = None
            if tid:
                comment_path = COMMENT_STORE_DIR / f"{tid}.jsonl"
                comment_lines_before = _count_jsonl_lines(comment_path)

            t0 = time.monotonic()
            if method == "GET":
                r = self._client.get(url)
            elif method == "POST":
                r = self._client.post(url, json=body_in)
            elif method == "DELETE":
                r = self._client.delete(url)
            else:
                r = self._client.request(method, url, json=body_in)
            latency = (time.monotonic() - t0) * 1000
            if tid and method == "POST":
                comment_write_ts = latency

            resp_body = _safe_json(r)
            http_calls.append(_http_record(
                method, url,
                status=r.status_code,
                body=resp_body,
                latency_ms=latency,
                ts=_now_iso(),
            ))

            # Settle window for mem.db mutations
            settle_s = s.inputs.get("settle_s", 0)
            if settle_s:
                time.sleep(settle_s)

            # Snapshot mem.db after
            mem_after = _snapshot_mem_db(MEM_DB_PATH)
            mem_diff = _diff_dicts(mem_before, mem_after)
            if mem_diff:
                mem_writes.append({"before": mem_before, "after": mem_after, "diff": mem_diff})

            # Snapshot vault audit after
            vault_after = _snapshot_vault_audit(VAULT_AUDIT_DIR)
            vault_diff = vault_after - vault_before
            if vault_diff:
                vault_writes.append({"new_entries": vault_diff})

            # Comment-store append check
            if comment_path:
                comment_lines_after = _count_jsonl_lines(comment_path)
                new_count = comment_lines_after - comment_lines_before
                log_appends.append({
                    "file": str(comment_path),
                    "lines_before": comment_lines_before,
                    "lines_after": comment_lines_after,
                    "new_count": new_count,
                    "write_latency_ms": comment_write_ts,
                    "tid": tid,
                })

        except Exception as exc:
            errors.append({"stage": "run_scenario", "error": str(exc)})
            return Observation(
                scenario_id=s.scenario_id,
                started_at=started_at,
                finished_at=_now_iso(),
                http_calls=http_calls,
                errors=errors,
                harness_error=True,
            )

        return Observation(
            scenario_id=s.scenario_id,
            started_at=started_at,
            finished_at=_now_iso(),
            http_calls=http_calls,
            mem_writes=mem_writes,
            vault_writes=vault_writes,
            log_appends=log_appends,
            errors=errors,
            harness_error=False,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return r.text


def _collect_sse(client: httpx.Client, url: str, timeout: float = 2.0) -> list[dict[str, Any]]:
    """Open SSE stream briefly; collect events non-blocking."""
    events: list[dict[str, Any]] = []
    try:
        with client.stream("GET", url, timeout=timeout) as stream:
            deadline = time.monotonic() + timeout
            for line in stream.iter_lines():
                if time.monotonic() > deadline:
                    break
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    try:
                        events.append(json.loads(raw))
                    except Exception:
                        events.append({"raw": raw})
    except Exception:
        pass
    return events


def _snapshot_mem_db(db_path: Path) -> dict[str, Any]:
    """Read all pm/* keys from mem.db. Returns empty dict if db absent."""
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path), timeout=3.0)
        rows = conn.execute(
            "SELECT key, value FROM memory WHERE key LIKE 'pm/%'"
        ).fetchall()
        conn.close()
        return {k: v for k, v in rows}
    except Exception:
        return {}


def _snapshot_vault_audit(audit_dir: Path) -> int:
    """Return count of files in vault audit dir."""
    if not audit_dir.exists():
        return 0
    return len(list(audit_dir.iterdir()))


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _diff_dicts(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    diff = {}
    all_keys = set(before) | set(after)
    for k in all_keys:
        if before.get(k) != after.get(k):
            diff[k] = {"before": before.get(k), "after": after.get(k)}
    return diff
