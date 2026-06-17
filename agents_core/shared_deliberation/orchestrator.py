"""Core orchestration logic for shared deliberation service.

Wraps Facets (subprocess) + Mirror Council, runs them concurrently,
unifies results into a DeliberationEnvelope.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from agents_core.shared_deliberation.envelope import DeliberationEnvelope, DeliberationRequest

log = logging.getLogger("shared-deliberation")

# Bounded concurrency for Facets subprocesses (gate against GW lane stampede)
_facets_semaphore: Optional[asyncio.Semaphore] = None


def init_facets_semaphore(max_concurrent: int = 2) -> None:
    """Initialize the Facets concurrency gate. Called once at service startup."""
    global _facets_semaphore
    _facets_semaphore = asyncio.Semaphore(max_concurrent)


async def _facets_subprocess(
    text: str,
    context: dict,
    operator: str = "gravitywell",
) -> tuple[bool, Optional[dict], Optional[str], Optional[str]]:
    """Invoke Facets via subprocess. Returns (ok, deliberation_dict, deliberation_id, errors).

    The subprocess is executed in a thread to avoid blocking the event loop.
    """
    if _facets_semaphore is None:
        raise RuntimeError("Facets semaphore not initialized; call init_facets_semaphore() at startup")

    if os.getenv("SHARED_DELIBERATION_FACETS_STUB") == "1":
        # Stub mode for testing: skip subprocess, return fixture
        return (True, {"stub": True, "methodology": {"synthesis_operator": operator}}, "stub-id", None)

    if os.getenv("FACETS_DISPATCH_DISABLED") == "1":
        return (False, None, None, "Facets dispatch disabled via FACETS_DISPATCH_DISABLED")

    facets_repo = Path("/srv/git/facets-working")
    if not facets_repo.exists():
        return (False, None, None, f"Facets repo not found at {facets_repo}")

    async with _facets_semaphore:
        return await asyncio.to_thread(_run_facets_subprocess, text, context, operator, facets_repo)


def _run_facets_subprocess(
    text: str,
    context: dict,
    operator: str,
    facets_repo: Path,
) -> tuple[bool, Optional[dict], Optional[str], Optional[str]]:
    """Synchronous subprocess invocation (runs in thread)."""
    try:
        # Create temp context file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"text": text, **context}, f)
            context_file = f.name

        try:
            facets_env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(facets_repo), os.environ.get("PYTHONPATH", "")) if p
                ),
            }
            argv = [
                "python3", "-m", "facets.adapter", "deliberate",
                text,
                "--context-file", context_file,
                "--format", "json",
            ]
            if operator and operator != "haiku":
                argv += ["--persona-operator", operator, "--synthesis-operator", operator]

            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=600,
                env=facets_env,
            )

            if result.returncode != 0:
                error_msg = f"Facets subprocess exited {result.returncode}: {result.stderr}"
                log.error(error_msg)
                return (False, None, None, error_msg)

            deliberation_json = json.loads(result.stdout)
            deliberation_id = deliberation_json.get("deliberation_id")
            if not deliberation_id:
                error_msg = "No deliberation_id in Facets output"
                log.error(error_msg)
                return (False, None, None, error_msg)

            log.info(f"Facets complete: deliberation_id={deliberation_id}")
            return (True, deliberation_json, deliberation_id, None)

        finally:
            try:
                os.unlink(context_file)
            except OSError:
                pass

    except subprocess.TimeoutExpired:
        error_msg = "Facets timeout (10 min)"
        log.error(error_msg)
        return (False, None, None, error_msg)
    except Exception as e:
        error_msg = f"Facets error: {e}"
        log.error(error_msg)
        return (False, None, None, error_msg)


async def _council_subprocess(
    text: str,
    voicing: str = "gravitywell",
) -> tuple[bool, Optional[str], Optional[dict], Optional[str]]:
    """Submit council deliberation and poll until terminal.

    Returns (ok, run_id, council_data, errors).
    council_data includes status, landing, confidence, open_questions, positions, etc.
    """
    if os.getenv("SHARED_DELIBERATION_COUNCIL_STUB") == "1":
        # Stub mode for testing
        return (True, "stub-council-id", {"status": "resolved", "positions": []}, None)

    run_id = await asyncio.to_thread(_submit_council, text, voicing)
    if not run_id:
        return (False, None, None, "Failed to submit council")

    council_data, error = await asyncio.to_thread(_poll_council, run_id)
    if error:
        return (False, run_id, None, error)

    return (True, run_id, council_data, None)


def _submit_council(text: str, voicing: str) -> Optional[str]:
    """Submit council deliberation via agents_core.council.cli.cmd_submit."""
    try:
        from agents_core.council.cli import cmd_submit, DEFAULT_TURNS
        import argparse
        import io
        import contextlib
        import re

        args = argparse.Namespace(
            decision=text,
            voicing=voicing,
            mode="deliberation",
            n=None,
            turns=DEFAULT_TURNS,
            with_entity=None,
            narrator=False,
            narrator_voice=None,
            no_queue=False,
            notify=False,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_submit(args)
        output = buf.getvalue()
        m = re.search(r"task_id=(\S+)", output)
        if not m:
            log.error(f"Could not parse run_id from council output: {output!r}")
            return None
        return m.group(1)
    except Exception as e:
        log.error(f"Council submit error: {e}")
        return None


def _poll_council(run_id: str, timeout_s: int = 1800) -> tuple[Optional[dict], Optional[str]]:
    """Poll council run YAML until terminal status or timeout.

    Returns (council_data, error_message).
    council_data extracts status, landing, confidence, open_questions, positions.
    """
    import yaml

    council_dir = Path("/srv/lapis/council")
    start_time = time.time()
    poll_interval = 5

    while time.time() - start_time < timeout_s:
        run_path = council_dir / f"{run_id}.yaml"
        try:
            if not run_path.exists():
                time.sleep(poll_interval)
                continue

            run = yaml.safe_load(run_path.read_text())
            if not isinstance(run, dict):
                error = f"Council YAML is not a dict: {run_path}"
                log.error(error)
                return (None, error)

            status = run.get("status")
            if status in ("resolved", "open", "laid-down", "failed", "closed"):
                # Terminal status reached
                synthesis = run.get("synthesis", {})
                return (
                    {
                        "status": status,
                        "landing": synthesis.get("landing"),
                        "confidence": synthesis.get("confidence"),
                        "open_questions": synthesis.get("open_questions", []),
                        "positions": synthesis.get("positions", []),
                        "voicing_effective": run.get("voicing"),
                        "voicing_degraded": run.get("selection_degraded", False),
                        "voicing_degraded_reason": None,  # TODO: extract from run if available
                    },
                    None,
                )

            time.sleep(poll_interval)
        except Exception as e:
            error = f"Council poll error: {e}"
            log.error(error)
            return (None, error)

    # Timeout
    return (
        {
            "status": "timeout",
            "landing": None,
            "confidence": None,
            "open_questions": [],
            "positions": [],
        },
        None,
    )


async def run_deliberation(request: DeliberationRequest) -> DeliberationEnvelope:
    """Main orchestrator: run Facets and Council concurrently, unify results.

    Partial-failure semantics:
    - If Facets fails but Council succeeds, return HTTP 200 with facets_ok=False
    - If Council fails but Facets succeeds, return HTTP 200 with council_ok=False
    - If both fail, return HTTP 200 with both _ok=False (caller must check flags)
    """
    import uuid

    request_id = f"deliberation-{uuid.uuid4().hex[:12]}"
    triage = request.triage or "full"
    triage_escalated = False
    triage_reason = "caller-requested"

    # Run Facets always
    facets_ok, facets_dict, facets_id, facets_error = await _facets_subprocess(
        request.text,
        request.context,
        request.facets_operator,
    )

    # Run Council only if triage == "full"
    council_ok = False
    council_run_id = None
    council_data = None
    council_error = None

    if triage == "full":
        council_ok, council_run_id, council_data, council_error = await _council_subprocess(
            request.text,
            request.council_voicing,
        )

    # Extract operator info from Facets
    operator_requested = None
    operator_effective = None
    if facets_dict:
        methodology = facets_dict.get("methodology", {})
        operator_requested = methodology.get("operator_requested", request.facets_operator)
        operator_effective = methodology.get("synthesis_operator", request.facets_operator)

    # Extract council info
    council_status = None
    council_landing = None
    council_confidence = None
    council_open_questions = []
    council_positions = []
    council_voicing_effective = None
    council_voicing_degraded = False
    council_voicing_degraded_reason = None

    if council_data and council_ok:
        council_status = council_data.get("status")
        council_landing = council_data.get("landing")
        council_confidence = council_data.get("confidence")
        council_open_questions = council_data.get("open_questions", [])
        council_positions = council_data.get("positions", [])
        council_voicing_effective = council_data.get("voicing_effective")
        council_voicing_degraded = council_data.get("voicing_degraded", False)
        council_voicing_degraded_reason = council_data.get("voicing_degraded_reason")

    # Collect errors
    errors = {}
    if facets_error:
        errors["facets"] = facets_error
    if council_error:
        errors["council"] = council_error

    envelope = DeliberationEnvelope(
        deliberation_request_id=request_id,
        triage=triage,
        triage_reason=triage_reason,
        triage_escalated=triage_escalated,
        # Facets
        facets_ok=facets_ok,
        facets=facets_dict,
        facets_deliberation_id=facets_id,
        operator_requested=operator_requested,
        operator_effective=operator_effective,
        # Council
        council_ok=council_ok,
        council_run_id=council_run_id,
        council_status=council_status,
        council_landing=council_landing,
        council_confidence=council_confidence,
        council_open_questions=council_open_questions,
        council_positions=council_positions,
        council_voicing_requested=request.council_voicing,
        council_voicing_effective=council_voicing_effective,
        council_voicing_degraded=council_voicing_degraded,
        council_voicing_degraded_reason=council_voicing_degraded_reason,
        # Extensions and errors
        extra_modes=[],  # Reserved for jagged-seam tap (v0: empty)
        errors=errors,
    )

    return envelope
