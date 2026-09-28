"""doorman HTTP service — GravityWell power/lease lifecycle manager.

Runs on BRIX (always-on). Owns wake/suspend management for GravityWell so
individual callers never need to shell wake-gravitywell or gw-keepawake.

Entry point:  scripts/doorman_server.py  (thin bootstrap; systemd runs this
                     directly — see systemd/doorman-server.service). This
                     module is logic/API only: create_app() builds the
                     FastAPI app, the process bootstrap lives outside
                     agents_core.
Port:         8407  (DOORMAN_BIND_PORT env var — live-verified free 2026-06-09;
                     8400/8401/8403/8404/8405/8406 are all occupied)
Bind:         127.0.0.1 by default  (BRIX-local; Unit 1 has no off-box clients)

Environment variables:
  DOORMAN_BIND_HOST      — uvicorn bind host (default 127.0.0.1)
  DOORMAN_BIND_PORT      — uvicorn bind port (default 8407)
  DOORMAN_BEARER_TOKEN   — optional shared bearer token
  DOORMAN_IDLE_LOG       — path for structured idle-lifecycle JSONL log
                           (default /var/log/doorman-idle.jsonl)
  DOORMAN_DEFER_TO_CONTROLLER — enable deference to flip-controller (default true);
                                 also kill-switch for non-big flips (REQUIRE_DOORMAN_DEFERENCE gate)
  DOORMAN_CONTROLLER_NAME     — identity of the mode-controller (default flip-controller);
                                reported by /v0/mode-owner
  GW_URL                 — GravityWell base URL (default http://203.0.113.11:8081)
                           NOTE: must match the GW_URL configured for agents_core.llm
                           (the operator reads the same env var for inference POSTs).
                           Also Slot 1 of dual mode; Slot 2 is derived by swapping the
                           port to GW_SLOT2_PORT (default 8082) — see DOORMAN_DEFAULT_SERVE_MODE.
  GW_WAKE_DEADLINE_SEC   — max seconds to wait for GW to serve in big mode (default 180;
                           cold 77GB model load backstop — typical warm wake is ~10s)
  DOORMAN_DEFAULT_SERVE_MODE — cold-wake serving target when no controller owns the mode
                           and GW is not serving: "dual" (default) or "big". "dual" issues
                           `gw-serve dual` (both vLLM slots, :8081 27B + :8082 Devstral);
                           "big" restores the prior byte-identical `gw-serve big` (122B)
                           behavior — set this for an exact rollback to pre-dual-default
                           wake behavior (gw-doorman-wake-to-default-mode-v0).
  GW_DUAL_WAKE_DEADLINE_SEC — max seconds to wait for BOTH dual slots to serve (default
                           720; safely above Devstral's measured ~488s cold-init and
                           under gw-dual's own TimeoutStartSec=900). Only used when
                           DOORMAN_DEFAULT_SERVE_MODE=dual.
  GW_HOLD_TTL_SEC        — keepawake hold TTL in seconds (default 120)
  GW_HOLD_REFRESH_SEC    — refresh interval for the keepawake hold (default 45)
  GW_STOP_GRACE_SEC      — seconds after last-release before the refresh thread
                           issues gw-serve stop (default 600; machine-economics
                           boundary that amortizes the ~25s cold-load against burst
                           gaps — not a human-rhythm value)
  GW_SERVE_STOP_TIMEOUT_SEC — ssh subprocess budget in seconds for `gw-serve stop`,
                           used by both the manual /v0/force-stop endpoint and the
                           deferred idle-stop (default 120). GW's own systemd unit
                           (llama-server.service) is allowed up to TimeoutStopUSec
                           (600s live-read) to shut down cleanly, so a normal-but-
                           slow unload can legitimately outlast this budget — that
                           is by design, not a bug: a timeout here now resolves by
                           observation (still serving -> "stop_in_progress", down ->
                           "stopped") rather than being reported as a hard failure
                           for an unload that is in fact succeeding
                           (agents-core-doorman-force-stop-timeout-truthfulness-v0).
                           Deliberately bounded well below TimeoutStopUSec: sizing
                           it at or above the unit's own allowance would hold
                           state.lock (or, for the deferred path, the background
                           loop) for up to ten minutes and freeze /status.
  GW_FLASHNEXT_URL         — base URL of the flash-next whole-card seat
                           (default http://203.0.113.11:30000; SGLang on
                           GravityWell, flashnext-seat start|stop). The
                           doorman probes /health + /v1/models on this URL to
                           derive the handover window (agents-core-doorman-
                           flashnext-handover-v0, D2/D7) — visibility only:
                           it never starts or stops the seat (D8).
  GW_FLASHNEXT_MODEL_ID    — the seat's EXACT canonical_id (default
                           Qwen3.8-Flash-Next-NVFP4-SSD-Stream). Window
                           admission is identity-exact (D2b): UP_REGISTERED
                           requires /v1/models data[0].id == this value; an
                           identifiable foreign occupant never admits a
                           window (never guess).
  DOORMAN_FLASHNEXT_ACTIVITY_METRICS
                        — comma-separated Prometheus gauge names read from
                           the flash-next seat's /metrics for the D2
                           LEGIBILITY activity clock
                           (gw-doorman-flashnext-idle-awareness-v0; default
                           sglang:num_running_reqs,sglang:num_queue_reqs —
                           DoD-0 live-confirmed against the SGLang source
                           tree 2026-09-26). LEGIBILITY ONLY: never consumed
                           by the stop decision. The seat currently runs
                           with enable_metrics=False, so /metrics answers
                           404 and the probe falls back to /get_load
                           (num_reqs / num_waiting_reqs); a gauge rename in
                           a future SGLang is an env change, not a code
                           change. Any failure on both legs is "unknown",
                           never an idle reading and never a stop
                           authorization.
  GW_SERVE_STOP_GIVEUP_SEC — seconds a stop may report stop_in_progress before the
                           background reconciler gives up and surfaces a genuine
                           failure (last_error + a stop_failed idle-log row),
                           clearing the in-flight flag so a wedged stop cannot
                           permanently block later attempts (default 900 — above
                           TimeoutStopUSec with margin).
  DOORMAN_MODE_AWARE_ADMISSION — enable mode-aware deference before the _is_serving()
                                  fast path (HOLE 1), controller-lease-aware serving_mode
                                  (HOLE 2), and the three-state /v1/models big-model probe.
                                  Default false (lands dark). Set "true" or "1" to activate.
  DOORMAN_PROBE_LLAMA_ACTIVITY — probe for unmediated-caller activity before
                                 dwell-stopping on zero leases (default true; set
                                 "false"/"0" to disable and restore pre-fix
                                 lease-only behavior as a rollback lever). Probes
                                 llama.cpp's /slots (Probe A) and /metrics counter
                                 diff (Probe C, agents-core-doorman-class-aware-
                                 activity-probe-v0 — closes A's once-per-tick
                                 sampling gap) and vLLM's /metrics on both dual-
                                 mode slots (Probe B1/B2, gw-doorman-vllm-activity-
                                 probe-v0), concurrently, per tick. Tri-state per
                                 probe (True/False/None); which probes VOTE is
                                 class-aware once DOORMAN_MODE_AWARE_ADMISSION has
                                 resolved serving_is_big (structurally-absent
                                 sources for the resolved class are dropped, not
                                 read as indeterminate); an indeterminate combine
                                 pauses the grace-period clock rather than either
                                 resetting or advancing it, bounded by
                                 DOORMAN_PROBE_BLINDNESS_SEC — except when
                                 serving_is_big is None (topology unresolved,
                                 flag on), where the clock pauses with NO bound
                                 and a topology_unknown_no_park alarm fires every
                                 tick instead (Erah ruling 2026-08-19).
  DOORMAN_PROBE_BLINDNESS_SEC — extra seconds of benefit-of-the-doubt past
                                 GW_STOP_GRACE_SEC before an indeterminate probe
                                 (both /slots and /metrics unreachable) falls back
                                 to confirmed-idle behavior, so a permanently
                                 broken probe can't pin GW awake forever
                                 (default 900)
  DOORMAN_MAX_HOLD_TIMEOUT_SEC — foreground-priority gate (gw-router-phase1-foreground-
                                 gate): max seconds a `deferrable`-class lease acquire
                                 waits on the pending-defer wait-list while a `protected`
                                 lease (or the brake) is active, anchored to the job's own
                                 enqueue timestamp and never reset by newly-arriving
                                 `protected` leases (default 900 = 15min)
  DOORMAN_RELEASE_JITTER_MAX_SEC — max random backoff (seconds) applied before finalizing
                                 a wait-list release, so a batch of simultaneously-releasable
                                 jobs doesn't thundering-herd the freed slot (default 2.0)
  DOORMAN_PENDING_RELEASE_WARN_SEC — T-minus window (seconds) before a max-hold-timeout
                                 release at which an informational `pending-release-soon`
                                 log event fires once per waiting job (default 120)
  DOORMAN_BRAKE_TTL_SEC       — default TTL (seconds) for the emergency brake set via
                                 POST /v0/brake when the caller omits ttl_s (default 900)

Safety properties (gravitywell-doorman-clean-stop-v0):
  - Doorman crash → GW stays POWERED, not suspended. The host-side guard
    (gw-idle-suspend.sh) blocks suspend while llama-server.service is active.
    A crashed doorman leaves the service running, so the guard keeps GW powered
    (safe, but no power saving). The doorman is a power-saving optimizer layered
    on the guard's safety floor — if the doorman never stops the service, the
    node degrades to "always powered," not to "unsafe suspend."
  - The doorman never causes an unsafe suspend: it can only *enable* suspend by
    first issuing gw-serve stop. A stop failure leaves GW powered (guard holds).
  - Leaked client lease → GC: stale leases (acquired_at + ttl_sec < now) are
    auto-released by the background refresh loop — a crashed operator cannot pin
    GW forever.
  - SSH-refresh failure → logged loudly, last_error set, retried on next tick;
    does NOT crash the thread and does NOT drop live leases.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import agents_core.llm as _llm
from agents_core.llm import gw_serving_state

log = logging.getLogger("doorman-server")

GW_URL_DEFAULT = "http://203.0.113.11:8081"
GW_CREATIVE_URL = os.getenv("GW_CREATIVE_URL", "http://203.0.113.11:8093")
# Flash-Next whole-card seat (agents-core-doorman-flashnext-handover-v0, D1/D7):
# the SGLang seat on :30000 (flashnext-seat start|stop on GravityWell). The
# doorman is this seat's VISIBILITY KEEPER ONLY (D8): it probes the seat, admits
# the handover window, refuses wakes/leases during the window, and reports the
# window on /status — it never starts or stops the seat. :30000 is reachable
# from BRIX over the tailnet (same route the :8081/:8082 probes use).
GW_FLASHNEXT_URL = os.environ.get("GW_FLASHNEXT_URL", "http://203.0.113.11:30000")
# Also defined in llm.py; intentionally not imported to avoid a doorman_server → llm dep.
# GW_WAKE_DEADLINE_SEC coupling: this deadline (default 180s) must be kept in sync
# with the client-side acquire timeout in agents_core.doorman_client._gw_acquire_timeout(),
# which derives the HTTP acquire timeout as GW_WAKE_DEADLINE_SEC + GW_ACQUIRE_MARGIN_SEC.
# The client timeout must be >= this deadline so successful cold wakes (which can take
# up to GW_WAKE_DEADLINE_SEC) are never misread as DoormanUnreachable timeouts.
GW_WAKE_DEADLINE_SEC = int(os.environ.get("GW_WAKE_DEADLINE_SEC", "180"))
GW_HOLD_TTL_SEC = int(os.environ.get("GW_HOLD_TTL_SEC", "120"))
GW_HOLD_REFRESH_SEC = int(os.environ.get("GW_HOLD_REFRESH_SEC", "45"))
# Machine-economics boundary: amortizes the ~25s cold-load against burst gaps.
# Calibrate from /var/log/doorman-idle.jsonl observations — never auto-tuned.
GW_STOP_GRACE_SEC = int(os.environ.get("GW_STOP_GRACE_SEC", "600"))

# agents-core-doorman-force-stop-timeout-truthfulness-v0 (R3): ssh subprocess
# budget for `gw-serve stop`, shared by the manual force-stop endpoint and the
# deferred idle-stop. 120s comfortably covers an ordinary unload, and it is
# deliberately bounded well below GW's own systemd TimeoutStopUSec (600s)
# because a timeout here is no longer a truncation — it now resolves by
# observation (R2), so a slower unload is correctly reported as
# stop_in_progress rather than waited on. Sizing this at or above the unit's
# own allowance would re-introduce the responsiveness problem R1 exists to
# prevent (holding state.lock, or blocking the background loop, for up to
# ten minutes).
GW_SERVE_STOP_TIMEOUT_SEC = int(os.environ.get("GW_SERVE_STOP_TIMEOUT_SEC", "120"))

# R8: bounds how long a force-stop may report stop_in_progress before the
# background reconciler gives up and surfaces a genuine, visible failure.
# Sized above GW's own TimeoutStopUSec (600s) with margin so a normal (if
# slow) unload is never mistaken for a hang.
GW_SERVE_STOP_GIVEUP_SEC = int(os.environ.get("GW_SERVE_STOP_GIVEUP_SEC", "900"))

# Extra benefit-of-the-doubt window past GW_STOP_GRACE_SEC before an indeterminate
# activity probe (gw-doorman-vllm-activity-probe-v0) falls back to confirmed-idle
# behavior — bounds the pause so a permanently broken /slots + /metrics probe can't
# pin GW awake forever (the opposite failure mode from the one this spec fixes).
DOORMAN_PROBE_BLINDNESS_SEC = int(os.environ.get("DOORMAN_PROBE_BLINDNESS_SEC", "900"))

# Cold-wake serving target (gw-doorman-wake-to-default-mode-v0). "dual" is the default —
# GW's boot-default resting posture is now dual (Slot 1 27B :8081 + Slot 2 Devstral :8082,
# see gw-dual-boot.service); "big" restores the exact prior gw-serve big wake path.
DOORMAN_DEFAULT_SERVE_MODE = os.environ.get("DOORMAN_DEFAULT_SERVE_MODE", "dual").strip().lower()

# The declared home posture (agents-core-doorman-wake-honors-declared-posture-v0,
# Part 2) lives in GW_HOME_MODE inside the live conductor.env — NOT in this
# process's own environment. The doorman's systemd unit only loads
# ~/.config/doorman/server.env (EnvironmentFile=-%h/.config/doorman/server.env);
# conductor.env is a different, host-local, non-repo-tracked file, so it must be
# read directly rather than via os.environ (flip_controller.py:56 can read
# GW_HOME_MODE from os.environ only because ITS unit sources conductor.env —
# the doorman's does not). Overridable for tests.
GW_HOME_MODE_ENV_PATH = os.environ.get(
    "GW_HOME_MODE_ENV_PATH", "/srv/agents/config/conductor.env"
)

# Both above Devstral's measured ~488s cold-init and under gw-dual's own TimeoutStartSec=900.
# Only consulted when DOORMAN_DEFAULT_SERVE_MODE == "dual".
GW_DUAL_WAKE_DEADLINE_SEC = int(os.environ.get("GW_DUAL_WAKE_DEADLINE_SEC", "720"))

# Slot 2's port — Slot 2's base host is derived from GW_URL (Slot 1), not re-hardcoded.
GW_SLOT2_PORT = int(os.environ.get("GW_SLOT2_PORT", "8082"))

# ---------------------------------------------------------------------------
# GPU 1 (the berth) awareness — gw-gpu1-berth-standing-seat-v0, leg 2.
#
# The 3090 Ti (GravityWell GPU 1) hosts the standing NInfer Qwen3.8-27B fixer
# seat (the "berth") at host port :8082. It is a SEPARATE card from the GPU 0
# production seat, so its activity is engine-independent of whatever class
# :8081 is serving — the berth's Glances telemetry source votes in ALL
# combine branches, not just one.
#
# The telemetry source is Glances (API v4, /api/4/gpu/), promoted to a
# supervised unit in leg 1. It binds the tailnet interface (NOT 0.0.0.0 —
# the no-unauthenticated-LAN-exposure invariant); the doorman reaches it
# over the tailnet. Env seams for hermetic tests:
#   DOORMAN_GLANCES_URL — base URL of the Glances API (default the tailnet
#                         address; the /api/4/gpu/ path is appended).
#   DOORMAN_GPU1_PROC_THRESHOLD — the proc>threshold test for the True vote
#                         (default 0: ANY proc>0 sample is activity).
#   DOORMAN_GPU1_MEM_THRESHOLD_PCT — DIAGNOSTIC ONLY (distinguishes the
#                         "berth resident" ~77% mem from an ollama baseline
#                         context in the status surface). NEVER part of the
#                         vote: the vote is proc-only (the S2 binding
#                         statement — an idle-but-warm berth MUST vote False
#                         or the box never sleeps).
# ---------------------------------------------------------------------------
DOORMAN_GLANCES_URL = os.environ.get(
    "DOORMAN_GLANCES_URL", "http://203.0.113.11:61208"
)
DOORMAN_GPU1_PROC_THRESHOLD = float(os.environ.get("DOORMAN_GPU1_PROC_THRESHOLD", "0"))
DOORMAN_GPU1_MEM_THRESHOLD_PCT = float(
    os.environ.get("DOORMAN_GPU1_MEM_THRESHOLD_PCT", "50")
)

# The async-initiate ssh call for dual mode only needs to spawn the backgrounded
# `gw-serve dual` remotely and return — it must NOT block for the ~488s bring-up
# (that's what the health-poll loop in _wake_dual is for). A short timeout here
# only bounds the ssh-connect + background-spawn round trip.
GW_DUAL_INITIATE_TIMEOUT_SEC = 20

# Deterministic exponential backoff for the dual readiness poll — a single-consumer
# cold wake has no thundering-herd concern that jitter would address.
GW_DUAL_POLL_INITIAL_SEC = 5.0
GW_DUAL_POLL_BACKOFF_FACTOR = 1.5
GW_DUAL_POLL_MAX_SEC = 30.0

DOORMAN_DEFER_TO_CONTROLLER = os.environ.get("DOORMAN_DEFER_TO_CONTROLLER", "true").lower() == "true"
DOORMAN_CONTROLLER_NAME = os.environ.get("DOORMAN_CONTROLLER_NAME", "flip-controller")

# Mode-aware admission guard — dark / default-OFF. When True:
#   ensure_serving checks deference BEFORE _is_serving() (HOLE 1 fix);
#   status_snapshot.serving_mode is controller-lease-aware (HOLE 2 fix) and reports
#   the actual topology (big/dual), not a hardcoded "big" (agents-core-doorman-
#   serving-mode-topology-truthful-v0);
#   _refresh_serving_cache resolves gw_serving_state() once per tick, feeding
#   serving_mode, the three-state serving_is_big predicate, and big_probe_state.
DOORMAN_MODE_AWARE_ADMISSION = os.environ.get(
    "DOORMAN_MODE_AWARE_ADMISSION", ""
).lower() in ("1", "true", "yes")

# Probe for unmediated-caller activity (doorman-probe-llama-activity-v0, extended by
# gw-doorman-vllm-activity-probe-v0) so dwell-stop doesn't fire out from under a direct
# (non-lease) caller like an interactive OpenCode session hitting a dual-mode vLLM slot
# directly. Probes BOTH llama.cpp's /slots (big mode) and vLLM's /metrics on both dual
# slots, concurrently, each tick — /slots alone silently no-ops against vLLM (vLLM has
# no /slots endpoint), which was the root cause of dual-mode sessions getting evicted.
# Default ON — this is a net-safety fix for a real incident, not a speculative feature.
# Set "false"/"0" as a rollback lever.
DOORMAN_PROBE_LLAMA_ACTIVITY = os.environ.get(
    "DOORMAN_PROBE_LLAMA_ACTIVITY", "true"
).lower() not in ("0", "false")

# Prometheus gauge names read from vLLM's /metrics (plaintext exposition format, not
# JSON) to detect activity on a dual-mode slot (gw-doorman-vllm-activity-probe-v0).
# NOT live-curl-confirmed: GW was in big mode (not dual) throughout implementation
# and at last check, so :8081/:8082 /metrics could not be curled against a running
# vLLM instance (DoD item 0 remains open). These names are instead confirmed by
# reading vLLM 0.22.1's source (vllm/v1/metrics/loggers.py) —
# labelnames = ["model_name", "engine"], e.g.
# `vllm:num_requests_running{model_name="gravitywell-27b",engine="0"} 0.0`.
# Re-verify against a live dual-mode /metrics response before treating DoD item 0
# as satisfied.
_VLLM_ACTIVITY_METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting")

# Prometheus gauge names read from the flash-next seat's (SGLang on :30000)
# /metrics for the D2 legibility activity clock
# (gw-doorman-flashnext-idle-awareness-v0). LEGIBILITY ONLY — never part of
# the stop decision (the stop path consumes the seat-STATE probe).
# DoD-0 LIVE check (2026-09-26, seat UP): the seat runs with
# enable_metrics=False, so GET :30000/metrics answers 404 ({"detail":"Not
# Found"}) and the Prometheus gauges are NOT exposed at all on this
# deployment — the probe therefore falls back to the always-on /get_load
# endpoint below (see _probe_flashnext_activity). When metrics ARE enabled
# the gauge names are the ones the running SGLang builds actually emit,
# confirmed against the live source tree
# (python/sglang/srt/observability/metrics_collector.py, SchedulerMetrics
# Collector.__init__): "sglang:num_running_reqs" and
# "sglang:num_queue_reqs" — NOT the "..._requests" spellings the earlier
# (seat-down) source read recorded; those names exist in no SGLang build and
# would have pinned the substate to "unknown" forever. Overridable via
# DOORMAN_FLASHNEXT_ACTIVITY_METRICS (comma-separated) so a SGLang version
# bump that renames the gauges does not require a code change. A
# missing/unparsable gauge degrades the substate to "unknown" — it can never
# authorize a stop.
_SGLANG_ACTIVITY_METRICS = tuple(
    m.strip()
    for m in os.environ.get(
        "DOORMAN_FLASHNEXT_ACTIVITY_METRICS",
        "sglang:num_running_reqs,sglang:num_queue_reqs",
    ).split(",")
    if m.strip()
)

# Keys read from the flash-next seat's /get_load for the same D2 clock —
# the DoD-0-live-confirmed activity source on THIS deployment (2026-09-26:
# GET :30000/get_load -> [{"dp_rank":0,"num_reqs":0,"num_waiting_reqs":0,
# "num_tokens":0,"num_pending_tokens":0,"ts_tic":...}], HTTP 200 with
# enable_metrics=False). The running/waiting pair maps to the two gauges
# above; the other keys are ignored. As with the gauges, LEGIBILITY ONLY.
_SGLANG_LOAD_RUNNING_KEYS = ("num_reqs", "num_running_reqs")
_SGLANG_LOAD_WAITING_KEYS = ("num_waiting_reqs", "num_queue_reqs")

# ---------------------------------------------------------------------------
# Capacity shadow (agents-core-doorman-capacity-shadow-v0) — instrumentation
# only. Takes a fresh capacity reading on every would-defer deferrable acquire
# and records what was seen alongside what would have been decided.
# Behaviour is byte-identical to today: no acquire that defers today is
# admitted by this unit. Default ON; "false"/"0" restores the exact current
# code path (no scrape, no event) — see DOORMAN_PROBE_LLAMA_ACTIVITY above
# for the repo convention this follows.
# ---------------------------------------------------------------------------
DOORMAN_CAPACITY_SHADOW = os.environ.get(
    "DOORMAN_CAPACITY_SHADOW", "true"
).lower() not in ("0", "false")

# Floor between real capacity scrapes per node — inside this window, a
# would-defer acquire reuses the most recent sample and is marked
# probe_outcome="coalesced" rather than triggering a second /metrics hit.
# Anti-DoS: without this, N deferrable callers refused in a burst would
# fire N concurrent 2.5s scrapes at /metrics (C3, spec 2026-08-06).
DOORMAN_CAPACITY_SHADOW_MIN_INTERVAL_SEC = float(
    os.environ.get("DOORMAN_CAPACITY_SHADOW_MIN_INTERVAL_SEC", "2.0")
)

# Value-preserving vLLM capacity gauge names, read from the same /metrics
# exposition text as _VLLM_ACTIVITY_METRICS but parsed to individual numeric
# values rather than folded into a single boolean (which is all
# _probe_vllm_metrics_activity above can return). Verified live on
# GravityWell slot1, 2026-08-06 — see spec for the raw exposition lines.
# `vllm:num_requests_waiting_by_reason` carries a `reason` label
# ("capacity" | "deferred"); its match test below requires "{" immediately
# after the metric name, so it never collides with the plain
# `vllm:num_requests_waiting{` prefix used for the bare gauge.
_VLLM_CAPACITY_RUNNING_METRIC = "vllm:num_requests_running"
_VLLM_CAPACITY_WAITING_METRIC = "vllm:num_requests_waiting"
_VLLM_CAPACITY_WAITING_BY_REASON_METRIC = "vllm:num_requests_waiting_by_reason"


def _extract_prom_value(line: str, metric: str) -> float | None:
    """Extract a Prometheus plaintext-exposition gauge value from one line,
    for an exact metric name (labels present or not). Mirrors the
    brace-then-value parsing already used by _probe_vllm_metrics_activity —
    duplicated rather than shared so neither parser's behaviour can shift
    under the other's maintenance. Never raises."""
    if line.startswith(metric + "{"):
        brace_end = line.find("}")
        if brace_end == -1:
            return None
        value_str = line[brace_end + 1:].strip().split()
    elif line.startswith(metric + " "):
        value_str = line[len(metric):].strip().split()
    else:
        return None
    if not value_str:
        return None
    try:
        return float(value_str[0])
    except ValueError:
        return None


def _parse_prom_labels(label_str: str) -> dict[str, str]:
    """Parse a Prometheus label-set body (the text between `{` and `}`,
    already stripped of the braces) into a dict. Never raises."""
    labels: dict[str, str] = {}
    for part in label_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        labels[k.strip()] = v.strip().strip('"')
    return labels


def parse_vllm_capacity_gauges(text: str) -> dict[str, float | None]:
    """Value-preserving read of the capacity-shadow gauges from vLLM's
    /metrics exposition text. Unlike _probe_vllm_metrics_activity (which
    sums two gauges into a single `total > 0` boolean, unable to
    distinguish "3 running, 0 waiting" from "0 running, 3 waiting"), this
    returns each gauge's summed numeric value.

    Sums across ALL label sets — deliberately does NOT filter on the
    `model_name` label value. Slot 1 has been flipped across at least six
    models and the label follows --served-model-name; a parser pinned to
    one model name goes permanently indeterminate after any flip, silently.

    A gauge that never appears in `text` maps to None (missing), never 0 —
    a missing reading must be representable as missing, or a probe failure
    silently poisons the capacity-shadow dataset as "0 capacity wait".

    Never raises; unparsable lines are skipped.
    """
    running_total = 0.0
    waiting_total = 0.0
    capacity_wait_total = 0.0
    deferred_wait_total = 0.0
    found: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(_VLLM_CAPACITY_WAITING_BY_REASON_METRIC + "{"):
            brace_end = line.find("}")
            if brace_end == -1:
                continue
            label_str = line[len(_VLLM_CAPACITY_WAITING_BY_REASON_METRIC) + 1:brace_end]
            value_str = line[brace_end + 1:].strip().split()
            if not value_str:
                continue
            try:
                val = float(value_str[0])
            except ValueError:
                continue
            reason = _parse_prom_labels(label_str).get("reason")
            if reason == "capacity":
                capacity_wait_total += val
                found.add("capacity_wait")
            elif reason == "deferred":
                deferred_wait_total += val
                found.add("deferred_wait")
            continue

        val = _extract_prom_value(line, _VLLM_CAPACITY_RUNNING_METRIC)
        if val is not None:
            running_total += val
            found.add("num_requests_running")
            continue

        val = _extract_prom_value(line, _VLLM_CAPACITY_WAITING_METRIC)
        if val is not None:
            waiting_total += val
            found.add("num_requests_waiting")
            continue

    return {
        "num_requests_running": running_total if "num_requests_running" in found else None,
        "num_requests_waiting": waiting_total if "num_requests_waiting" in found else None,
        "capacity_wait": capacity_wait_total if "capacity_wait" in found else None,
        "deferred_wait": deferred_wait_total if "deferred_wait" in found else None,
    }


# The "big" seat's STOCK/DEFAULT member — what the base llama-server.service
# ExecStart falls back to, and the rollback target if a later occupant needs
# reverting (mirrors conductor/scripts/gw_topology.py's STOCK_BIG_MODEL_ID).
# It is NOT the definition of "big": that membership is registry-declared
# (gw_big_seat_members(), below) — a served model other than this one can
# still be a full big-seat member. Must match OPERATOR_DEFAULTS['gravitywell']
# in agents_core.llm (verified: llm.py:58).
GW_BIG_MODEL_ID = "gravitywell-122b"

# The Flash-Next seat's EXACT canonical_id (agents-core-doorman-flashnext-
# handover-v0, D2b): window admission is identity-EXACT — UP_REGISTERED
# requires data[0].id == GW_FLASHNEXT_MODEL_ID. Deliberately NOT
# _gw_registry_lookup(): that resolver matches any alias field of any row
# (every row shares operator_alias "gravitywell"), so a served id of
# "big"/"gravitywell" would have admitted a window. Exact canonical_id match
# closes that; the env override follows the GW_BIG_MODEL_ID constant pattern.
GW_FLASHNEXT_MODEL_ID = os.environ.get(
    "GW_FLASHNEXT_MODEL_ID", "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
)


def gw_big_seat_members() -> frozenset[str]:
    """The registry-declared membership of the "big" seat: every canonical_id
    in gw_models.yaml whose row declares mode_alias == "big"
    (agents-core-doorman-big-seat-membership-v0).

    Reads agents_core.llm's already-loaded registry live on every call — not
    a second parse of gw_models.yaml, not a cached snapshot, not a hardcoded
    list — so a new big-seat occupant is a one-row gw_models.yaml edit with
    no change here. Both comparison sites that used to key on the single
    GW_BIG_MODEL_ID identity (_refresh_serving_cache's big-probe and
    _resolve_live_posture) consult this set instead; council/cli.py's wave
    guard reads the same set for the same reason.
    """
    return frozenset(
        entry.canonical_id
        for entry in _llm._GW_MODEL_REGISTRY
        if entry.mode_alias == "big"
    )


def _topology_models_answered(topology_state) -> bool:
    """Whether the models endpoint answered on the given GwServingState (or
    False if topology_state itself is None — unreachable). Same predicate
    _refresh_serving_cache uses to derive big_probe_state; shared here so the
    park block's unknown-topology alarm (agents-core-doorman-class-aware-
    activity-probe-v0, D3) reports the identical diagnostic field rather than
    a second, possibly-drifting computation of it."""
    if topology_state is None:
        return False
    return (
        topology_state.source_freshness.get("models_endpoint", {}).get("status")
        == "answered"
    )


HOLD_NAME = "doorman"
DOORMAN_IDLE_LOG = os.environ.get("DOORMAN_IDLE_LOG", "/var/log/doorman-idle.jsonl")

# ---------------------------------------------------------------------------
# Foreground-priority gate (gw-router-phase1-foreground-gate)
# ---------------------------------------------------------------------------

# Binary lease classification (Mirror Council, converged 2026-07-17): a third
# tier for measured gates was rejected as needless complexity. Missing `class`
# on /lease/acquire defaults to "deferrable" (safe — never accidentally
# preempts); an invalid value is rejected 400 by the endpoint.
LEASE_CLASSES = ("protected", "deferrable")
DEFAULT_LEASE_CLASS = "deferrable"

# Single source of truth for the requested-serve-mode vocabulary
# (agents-core-doorman-mode-bearing-acquire-v0, AC4). "big" and "dual" both route
# to their existing _wake_big()/_wake_dual() dispatch — no other value is valid.
VALID_SERVE_MODES = frozenset({"big", "dual"})

# The "iron rod": absolute, non-resettable max-hold for a deferrable job
# waiting on the pending-defer wait-list, anchored to its own enqueue
# timestamp — never extended by newly-arriving protected leases.
DOORMAN_MAX_HOLD_TIMEOUT_SEC = int(os.environ.get("DOORMAN_MAX_HOLD_TIMEOUT_SEC", "900"))

# Anti-thundering-herd backoff applied before finalizing a wait-list release.
DOORMAN_RELEASE_JITTER_MAX_SEC = float(os.environ.get("DOORMAN_RELEASE_JITTER_MAX_SEC", "2.0"))

# T-minus window before a max-hold-timeout release at which one informational
# pending-release-soon log event fires (strictly informational, never blocking).
DOORMAN_PENDING_RELEASE_WARN_SEC = int(os.environ.get("DOORMAN_PENDING_RELEASE_WARN_SEC", "120"))

# Default TTL for the emergency brake (POST /v0/brake) when ttl_s is omitted —
# bounded so the brake can never freeze deferrable dispatch indefinitely.
DOORMAN_BRAKE_TTL_SEC = int(os.environ.get("DOORMAN_BRAKE_TTL_SEC", "900"))

# Sentinel for deferred acquire (controller owns the mode)
DEFERRED = object()

# Sentinel for contended acquire (require_drain_clear=True failed: another-principal worker active)
CONTENDED = object()

# Sentinel for creative-occupied acquire (Llama-3.3-70B on :8093 holds the GPU)
CREATIVE_OCCUPIED = object()

# Sentinel for flashnext-occupied acquire (agents-core-doorman-flashnext-
# handover-v0, D4): the flash-next seat holds GPU 0 whole-card during an
# active handover window. Propagated and answered exactly as CREATIVE_OCCUPIED
# (acquire_lease passthrough; 409 flashnext_occupied on /lease/acquire) —
# including for role=mode-controller: during a confirmed window the window
# guard supersedes the controller-deference machinery and no lease registers.
FLASHNEXT_OCCUPIED = object()


class _FlashnextServed:
    """ensure_serving() result for an S2 already-serving grant
    (doorman-flashnext-serving-admission-v0, S2).

    Distinct from the plain ``True`` so acquire_lease can stamp the additive
    ``serve_axis="flashnext"`` field on the lease dict (and log the registration
    audit line with the probe's served_id) WITHOUT a second :30000 probe — the
    value is carried out of the guard's single fresh probe pair. Truthy, so any
    legacy truthiness check reads it as success exactly like ``True``.
    """

    __slots__ = ("served_id",)

    def __init__(self, served_id: str | None = None):
        self.served_id = served_id

    def __bool__(self) -> bool:
        return True

# Sentinel principal for worker leases acquired without an explicit principal.
# Never excluded from drain_count — makes a forgotten-principal diagnosable instead of invisible.
GHOST_PRINCIPAL = "__GHOST_LEASE__"


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _gw_topology_importable() -> bool:
    """Cheap, side-effect-free check of whether `from scripts import
    gw_topology` (the exact form _wake_generic_posture uses — A3) would
    succeed right now, for /status visibility. Never crashes the caller;
    an unexpected error from the import itself just reads as unavailable."""
    try:
        from scripts import gw_topology  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception:
        return False


def _write_idle_log(
    node: str, event: str, lease_count: int, idle_secs: float | None = None, **extra_fields
) -> None:
    """Append one structured entry to the idle-lifecycle observation log.

    Best-effort: a write failure must never crash the caller or block the stop.
    This is an observation substrate for human calibration — not operational
    alerting and not consumed internally for auto-tuning.
    """
    entry: dict[str, Any] = {
        "ts": time.time(),
        "node": node,
        "event": event,
        "lease_count": lease_count,
    }
    if idle_secs is not None:
        entry["idle_secs"] = round(idle_secs, 2)
    entry.update(extra_fields)
    try:
        with open(DOORMAN_IDLE_LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning(f"idle log write failed ({DOORMAN_IDLE_LOG}): {exc}")


# gw-serve's stop_all() prints "  stopping <unit>" (leading whitespace, literal
# tab/space indent) per unit it actually stops. Deliberately rigid — [ \t], not
# \s — so any drift in gw-serve's output format *fails* this match rather than
# loosely capturing something wrong (agents-core-doorman-stop-log-unit-accuracy-v0).
_STOPPING_UNIT_RE = re.compile(r"^[ \t]+stopping[ \t]+(\S+)[ \t]*$", re.MULTILINE)


def _describe_stopped_units(stdout: str | None, node_name: str) -> str:
    """Describe which units gw-serve stop actually stopped, from its stdout.

    A vague-but-true description beats a specific-but-false one: any failure
    to parse stdout with confidence falls back to unit-free wording (and never
    raises — the stop already succeeded, a logging problem must not undo that).
    """
    try:
        if not stdout or not stdout.strip():
            raise ValueError("stdout was empty")
        units = _STOPPING_UNIT_RE.findall(stdout)
        if units:
            return f"{', '.join(units)} stopped"
        if "stopping" not in stdout.lower():
            # No unit-stop lines and nothing even claims to be stopping —
            # distinct from a parse failure: gw-serve ran and had nothing to do.
            return "nothing was serving"
        raise ValueError("stdout did not match the expected 'stopping <unit>' format")
    except Exception as exc:
        try:
            snippet = str(stdout)[:200] if stdout else ""
        except Exception:
            snippet = "<unprintable stdout>"
        log.warning(
            f"[{node_name}] could not parse gw-serve stop stdout ({exc}) — "
            f"falling back to generic stop message. stdout snippet: {snippet!r}"
        )
        return "GW serving stopped"


# ---------------------------------------------------------------------------
# Restore-failure streak + page (attestation-contract-v0, leg 1, D4)
# ---------------------------------------------------------------------------

# D4 constants: the 09-08 incident's WAKE_REFUSED loop ran ~40s then went
# quiet while the seat stayed down (last_error was recorded, nothing paged).
# 3 consecutive restore failures within 10 minutes pages HIGH once per
# episode (I5 page hygiene - no repetition while the streak persists; the
# Forgejo-unreachable 3-strike pattern is the shape precedent).
_RESTORE_FAILURE_STREAK_THRESHOLD = 3
_RESTORE_FAILURE_WINDOW_SEC = 600  # 10 minutes
_RESTORE_FAILURE_PAGE_SOURCE = "lapis-pm-doorman"  # I5: the pinned source


class _RestoreFailureStreak:
    """Consecutive-restore-failure counter (per seat, in the refresh-thread
    state). Reset on any successful serve or successful restore; the streak
    also resets when the window lapses (>10 min since the first failure -
    the episode is over) or the failure reason changes (a new failure
    class is a new episode).

    The page fires exactly once per episode: `_page_fired` latches on the
    threshold-crossing failure and clears on reset (success / window lapse
    / reason change), so a persistent streak pages once, not per failure.
    """

    def __init__(self) -> None:
        self.consecutive: int = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self.last_reason: str = ""
        self.last_detail: str = ""
        self._page_fired: bool = False

    def record_failure(
        self,
        reason: str,
        detail: str,
        now: float | None = None,
        _should_page: bool = True,
        _on_page=None,
    ) -> bool:
        """Record a restore failure. Returns True when the threshold was
        crossed (the page condition holds). `_should_page` / `_on_page`
        are test seams (the wiring seam `_maybe_page_restore_failure`
        supplies them in production).

        Lock note (attestation-contract-v0 rev-4): the PRODUCTION wiring
        (_record_restore_failure) calls this with _should_page=False under
        self.lock and sends the page OUTSIDE the lock - the page's network
        send must never run while holding self.lock. The _on_page seam
        (used by the unit tests) runs inline here and is test-only."""
        if now is None:
            now = time.time()
        reason = (reason or "")[:80]
        detail = (detail or "")[:300]
        # Episode boundaries: window lapse or a new failure class.
        if (
            self.consecutive == 0
            or self.first_ts is None
            or (now - self.first_ts) > _RESTORE_FAILURE_WINDOW_SEC
            or (reason and self.last_reason and reason != self.last_reason)
        ):
            self.consecutive = 0
            self.first_ts = now
            self._page_fired = False
        self.consecutive += 1
        self.last_ts = now
        self.last_reason = reason
        self.last_detail = detail
        if (
            self.consecutive >= _RESTORE_FAILURE_STREAK_THRESHOLD
            and not self._page_fired
            and _should_page
        ):
            self._page_fired = True
            if _should_page and _on_page is not None:
                try:
                    _on_page()
                except Exception:
                    pass  # page-only: a page failure never raises into the wake path
            return True
        return False

    def record_success(self) -> None:
        """A successful serve or restore: the streak (and the page latch)
        reset - the next episode pages again."""
        self.consecutive = 0
        self.first_ts = None
        self.last_ts = None
        self.last_reason = ""
        self.last_detail = ""
        self._page_fired = False


def _build_restore_failure_page(
    seat_id: str,
    reason: str,
    detail: str,
    streak: int,
    last_ts: float | None,
) -> str:
    """The D4 page body (I1: the signal lands on a named human-visible
    surface). Content: seat id, the WAKE_REFUSED/fail reason + detail
    (the recorded string), streak length, last failure ts, and the
    one-line manual-restore command."""
    ts_str = (
        datetime.datetime.fromtimestamp(last_ts, datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        if last_ts
        else "unknown"
    )
    detail = (detail or "")[:300]
    reason = (reason or "unknown")[:80]
    return (
        f"GW seat restore failing: {seat_id} - {streak} consecutive "
        f"restore failures (last {ts_str}). Reason: {reason or 'unknown'}. "
        f"Detail: {detail or 'n/a'}. Manual restore: "
        f"ssh gravitywell 'gw-topology converge --posture slot1-solo' "
        f"(or the declared home posture per /srv/agents/config/conductor.env)."
    )


def _maybe_page_restore_failure(
    *,
    reason: str,
    detail: str,
    streak: int,
    last_ts: float | None,
    seat_id: str,
) -> bool:
    """The wiring seam the wake-failure path calls after recording a
    failure (the streak counter decides WHEN; this decides the page
    CONTENT + delivery). Page-only (Standing ratification 4): never
    restarts or stops the doorman or the seat. Never raises."""
    try:
        from agents_core.notify import Priority, send_notification

        body = _build_restore_failure_page(
            seat_id=seat_id,
            reason=reason,
            detail=detail,
            streak=streak,
            last_ts=last_ts,
        )
        send_notification(
            body,
            title=f"GW seat restore failing: {seat_id}",
            priority=Priority.HIGH,
            source=_RESTORE_FAILURE_PAGE_SOURCE,
        )
        return True
    except Exception as exc:  # page-only: a page failure is a log line, not a raise
        log.error(
            f"[doorman] D4 restore-failure page failed: {exc}"
        )
        return False


# ---------------------------------------------------------------------------
# Node state (per-node; Unit 1 only handles "gravitywell")
# ---------------------------------------------------------------------------

class _NodeState:
    """All mutable state for one node, guarded by self.lock plus a dedicated
    self.wake_lock for the cold-wake path (doorman-acquire-lease-lock-release-during-wake-v0).

    self.lock serializes:
    - every lease-registry mutation (acquire, release, GC)
    - short reads/writes of cached-serving / error / service-lifecycle fields
    - background refresh-thread reads and SSH hold re-issues

    self.wake_lock serializes ensure_serving() calls (prevents parallel
    wake-gravitywell subprocesses) without holding self.lock across the
    minutes-long wake + poll loop, so /status, /lease/release, etc. stay
    responsive for this node while a wake is in flight.
    """

    def __init__(self, gw_url: str, node_name: str = "gravitywell"):
        self.lock = threading.Lock()
        self.wake_lock = threading.Lock()
        self.gw_url = gw_url
        self.node_name = node_name
        # keyed by work_id → {acquired_at: float, ttl_sec: int, reason: str, role: str}
        self.leases: dict[str, dict] = {}
        self.last_wake_at: float | None = None
        self.last_error: str | None = None
        # D4 (attestation-contract-v0, leg 1): the consecutive-restore-
        # failure streak (per seat). Mutated only under self.lock by the
        # wake-failure path (_refuse_wake / _fail_wake) and reset under
        # self.lock on any successful serve/restore.
        self.restore_failure_streak = _RestoreFailureStreak()
        # Service-lifecycle fields (gravitywell-doorman-clean-stop-v0)
        # Seeded at construction (doorman-seed-idle-since-on-startup-v0): leases
        # starts empty, so idle-tracking must begin now, not only on a later
        # empty-transition that may never occur if the process starts at zero leases.
        self.idle_since: float | None = time.time()
        self.service_stopped: bool = False     # True after gw-serve stop confirmed
        # Single-writer stop tracking (agents-core-doorman-force-stop-timeout-
        # truthfulness-v0, R1/R8/R8a). Read and written ONLY while holding
        # self.lock — the lock is the memory barrier, no separate atomic type
        # needed. _stop_in_flight and _stop_in_flight_since are one piece of
        # state: always set together and cleared together (R8), so a reader
        # taking self.lock never observes one without the other.
        self._stop_in_flight: bool = False
        self._stop_in_flight_since: float | None = None
        # Per-tick stop-failure signal (agents-core-doorman-flashnext-
        # handover-v0, rev 3 extraction repair): set by _decide_idle_stop's
        # two stop-failure outcomes (rc!=0-still-serving real failure and
        # the generic exception handler) and read by the refresh loop to
        # apply the stop-failure backoff bump (backoff = min(backoff + 15,
        # GW_HOLD_REFRESH_SEC)) that origin/main carried inline in the loop.
        # Reset at the top of every _decide_idle_stop call; read-only in the
        # loop. Never set on the timeout path (stop_in_progress is not a
        # failure) or on the rc!=0-but-already-down success path.
        self._stop_failed_this_tick: bool = False
        # Monotonic ownership token (R8a): incremented under self.lock every
        # time _stop_in_flight transitions to True (a new stop attempt begins,
        # or the reconciler forces a give-up that ends one). The thread that
        # ran the subprocess captures its epoch and, on completion, only
        # mutates state if the epoch still matches — otherwise the reconciler
        # (or a later stop) already resolved this attempt and it must touch
        # nothing.
        self._stop_epoch: int = 0
        # Cached serving state (doorman-status-cached-serving-v0)
        self._cached_serving: bool | None = None   # None until first refresh
        self._serving_checked_at: float = 0.0      # walltime of last successful probe
        self._cached_creative_serving: bool = False
        # Mode-aware big predicate (populated only when DOORMAN_MODE_AWARE_ADMISSION is True)
        self._serving_is_big: bool | None = None   # None until first refresh with flag ON
        self._big_probe_state: str | None = None   # 'confirmed'|'refuted'|'unknown'
        # Single-resolution topology cache (agents-core-doorman-serving-mode-topology-
        # truthful-v0): set by _refresh_serving_cache (flag ON) from ONE call to
        # agents_core.llm.gw_serving_state(). serving_mode's big/dual branch,
        # _serving_is_big, and _big_probe_state all derive from this one object —
        # never a second independent probe of the same truth.
        self._cached_topology_state = None   # GwServingState | None until first refresh with flag ON
        # llama-server /slots activity probe (doorman-probe-llama-activity-v0)
        self._last_probed_task_by_slot: dict[int, int] = {}
        self._idle_since_source: str | None = None  # 'lease' | 'probe' | 'window_close' | None
        # Probe C: llama.cpp /metrics counter snapshot, keyed by url, then by
        # metric-name+labels (agents-core-doorman-class-aware-activity-probe-v0,
        # D1) — the previous tick's values, diffed to detect activity between
        # probe ticks (closes Probe A's once-per-tick sampling gap). Mutated
        # outside self.lock, same convention as _last_probed_task_by_slot.
        self._llamacpp_metrics_baseline: dict[str, dict[str, float]] = {}
        # Raw per-source tri-state results from the most recent
        # _probe_slot_activity() tick, keyed "A"/"B1"/"B2"/"C"/"GPU1" —
        # diagnostic only (D3), read by the park block to name which sources
        # were indeterminate when a bound-exceeded park fires. Mutated
        # outside self.lock, same convention as _last_probed_task_by_slot.
        self._last_probe_raw: dict[str, bool | None] = {}
        # GPU 1 (berth) Glances diagnostics from the most recent probe tick
        # (gw-gpu1-berth-standing-seat-v0, leg 2): mem_pct/proc are the raw
        # Glances readings (mem is DIAGNOSTIC ONLY — never part of the vote,
        # which is proc-only); seat_health is the berth's :8082 /health 200.
        # Populated by _probe_gpu1_glances() outside self.lock, read by
        # status_snapshot() under the lock (plain reads of immutable scalars).
        self._gpu1_glances_mem_pct: float | None = None
        self._gpu1_glances_proc: float | None = None
        self._gpu1_seat_health: bool | None = None
        # berth_unit: the ninfer-fixer systemd unit state (active/inactive) -
        # DISTINCT from seat_health (the :8082 /health 200). Populated by
        # _probe_gpu1_glances() (same tick, outside self.lock, never raises);
        # None = unknown (pre-probe, ssh timeout/error). Never conflated with
        # the seat health.
        self._gpu1_berth_unit: bool | None = None
        # Flash-next seat (:30000) window bookkeeping (agents-core-doorman-
        # flashnext-handover-v0, D2/D5): the window is PROBE-DERIVED (D1 — no
        # external declaration file, host marker, or lease on the host), so a
        # launcher death cannot leave a stale window; it self-clears when
        # :30000 stops answering. Populated by _refresh_serving_cache() from
        # this tick's own probe + this tick's serving read (under the lock);
        # read by status_snapshot() lock-only (the gpu1 block's precedent) and
        # by the stop path (D3/D9). _flashnext_window: "active" | "none" |
        # None (indeterminate — a blind probe is blindness, never False).
        # _flashnext_window_since: epoch set on the tick the window first
        # reads active, held while active, cleared on close.
        # _flashnext_window_closed_at: epoch set on the active->none
        # transition tick, consumed exactly once by the D9 close re-anchor.
        self._flashnext_state: str | None = None
        self._flashnext_served_id: str | None = None
        self._flashnext_registered: bool | None = None
        self._flashnext_window: str | None = None
        self._flashnext_window_since: float | None = None
        self._flashnext_window_closed_at: float | None = None
        # The ACTUAL exception class name from the most recent blind (or
        # cold) seat probe (e.g. "Timeout", "ConnectionRefusedError",
        # "ConnectionError"), or None on a definitive read — carried by
        # the D1 blind-withhold idle-log rows and the one-shot transition
        # warning so the operator sees the real failure class, not a
        # static placeholder (the D4 BLIND-proceed precedent).
        self._flashnext_error_class: str | None = None
        # Flash-next idle-awareness (gw-doorman-flashnext-idle-awareness-v0,
        # D1/D2): the stop-path partition consumes _flashnext_state
        # (up/blind/down/foreign); these two fields are the bookkeeping the
        # partition needs. _flashnext_blind_since: epoch armed on the first
        # blind (or cold/None) read since the last definitive read, cleared
        # on any definitive read (down / up_registered / up_unverified /
        # up_foreign) — the continuous-blindness clock. DIAGNOSTIC only:
        # it feeds the blind_secs field of the flashnext_blind_hold
        # idle-log rows and the blind-duration text of the withhold
        # warning; the bounded blind-withhold itself is measured on
        # idle_elapsed (the grace clock), not on this field.
        # None = not currently blind
        # (definitive, or never blind since the last definitive read).
        # _flashnext_last_activity_ts: the D2 legibility clock — probe-stamp
        # time of the last successful activity-gauge read that observed
        # running/waiting >= 1; NEVER consumed by the stop decision (D2:
        # legibility only), never set by failed/zero-activity reads, cleared
        # by a definitive "down" probe. None = no observation yet.
        self._flashnext_blind_since: float | None = None
        self._flashnext_last_activity_ts: float | None = None
        # One-shot journal latch for the transition into a new withhold
        # state (D3: one WARNING per transition; steady-state withhold =
        # idle-log rows only). Holds the LAST-WARNED withhold substate
        # ("up:<registered|unverified>:<active|idle>" / "blind" / "cold");
        # _decide_idle_stop warns only when THIS tick's substate differs
        # from it, so a steady-state withhold writes idle-log rows and no
        # journal line. Cleared on a definitive non-withhold seat read
        # (down / up_foreign) so the next transition warns again.
        self._flashnext_withhold_substate: str | None = None
        # Tri-state dual-slot activity probe (gw-doorman-vllm-activity-probe-v0):
        # True when the most recent _probe_slot_activity() tick was indeterminate
        # (at least one probe ambiguous, none confirmed activity) — read by the
        # refresh loop to pause the grace-period clock instead of advancing it.
        self._probe_indeterminate: bool = False
        # Foreground-priority gate (gw-router-phase1-foreground-gate): in-memory
        # pending-defer wait-list, keyed by work_id → {enqueued_at, reason, role,
        # _warned, _release_at, _release_reason}. Ephemeral slot-arbitration state
        # tied to the doorman's own lease lifecycle — deliberately not a table in
        # agents_core.elevator's SQLite work-queue (see spec).
        self.wait_list: dict[str, dict] = {}
        # Emergency brake: a global defer-only flag (not a work_id lease) with a
        # bounded TTL so it auto-expires — never an indefinite freeze.
        self.brake_reason: str | None = None
        self.brake_expires_at: float | None = None
        # Capacity shadow (agents-core-doorman-capacity-shadow-v0): dedicated
        # lock so single-flight scraping never contends with self.lock — the
        # scrape must be able to proceed while self.lock is held elsewhere.
        # _capacity_shadow_last_sample/_last_scraped_at are touched ONLY by
        # capacity_shadow_scrape() under _capacity_shadow_scrape_lock; never
        # read/written under self.lock and never part of doorman state proper.
        self._capacity_shadow_scrape_lock = threading.Lock()
        self._capacity_shadow_last_sample: dict | None = None
        self._capacity_shadow_last_scraped_at: float = 0.0

    # ------------------------------------------------------------------
    # Health poll (lock-free — read-only HTTP, safe to call outside lock)
    # ------------------------------------------------------------------

    def _is_serving(self, timeout: float = 3.0) -> bool:
        try:
            resp = requests.get(f"{self.gw_url}/health", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def _is_creative_serving(self) -> bool:
        """Return True if the Llama-3.3-70B creative server is up on :8093.

        Lock-free HTTP - safe to call outside lock; also called under lock in
        ensure_serving(). Returns False on any error - if :8093 is unreachable,
        the 70B is not actively serving.
        """
        try:
            r = requests.get(f"{GW_CREATIVE_URL}/health", timeout=2.5)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def _probe_llama_slots_activity(self) -> bool:
        """Probe A: llama.cpp's own /slots for unmediated-caller activity (big mode).

        Detects activity from callers that never acquired a doorman lease (e.g. an
        interactive session hitting :8081 directly) so the dwell-stop clock doesn't
        get stopped out from under them. Activity is detected per slot when
        is_processing is True, or when id_task changed since the last probe (a
        generation completed between ticks).

        Best-effort, unchanged since doorman-probe-llama-activity-v0: any error
        (timeout, connection refused, non-200, malformed JSON, empty list) returns
        False — never raises, never returns None (that's Probe B's contract, not
        this one — vLLM's dual slots don't implement /slots at all, so absence
        here is expected in dual mode, not an error).

        Must be called OUTSIDE self.lock (blocking HTTP, ~2.5s timeout).
        """
        try:
            resp = requests.get(f"{self.gw_url}/slots", timeout=2.5)
            if resp.status_code != 200:
                return False
            slots = resp.json()
            if not isinstance(slots, list) or not slots:
                return False
            activity = False
            for slot in slots:
                slot_id = slot.get("id")
                id_task = slot.get("id_task")
                prev_task = self._last_probed_task_by_slot.get(slot_id)
                if slot.get("is_processing") or (
                    prev_task is not None and id_task != prev_task
                ):
                    activity = True
                if slot_id is not None:
                    self._last_probed_task_by_slot[slot_id] = id_task
            return activity
        except Exception as exc:
            log.debug(f"[{self.node_name}] llama activity probe inconclusive: {exc}")
            return False

    def _probe_vllm_metrics_activity(self, url: str) -> bool | None:
        """Probe B: GET {url}/metrics (vLLM's Prometheus plaintext exposition format,
        NOT JSON) and scan for the dual-slot activity gauges.

        Returns:
          True  — a gauge was found and parsed with a nonzero value (confirmed activity)
          False — all target gauges were found and parsed, all zero (confirmed idle)
          None  — the call failed (timeout/connection-refused/non-200) or the body
                  didn't contain either target gauge (malformed, or this port isn't
                  serving vLLM at all — e.g. big mode, or the other slot is down) —
                  indeterminate, never treated as confirmed-idle by the caller.

        Never raises. Must be called OUTSIDE self.lock (blocking HTTP, ~2.5s timeout).
        """
        try:
            resp = requests.get(f"{url}/metrics", timeout=2.5)
            if resp.status_code != 200:
                return None
            total = 0.0
            found = False
            for line in resp.text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for metric in _VLLM_ACTIVITY_METRICS:
                    if line.startswith(metric + "{"):
                        # Labels present: value (and optional timestamp) start
                        # right after the closing brace, not at the last space —
                        # a naive rsplit(" ", 1) would silently take a trailing
                        # Prometheus timestamp field as the value if one is ever
                        # emitted.
                        brace_end = line.find("}")
                        if brace_end == -1:
                            continue
                        value_str = line[brace_end + 1:].strip().split()
                    elif line.startswith(metric + " "):
                        value_str = line[len(metric):].strip().split()
                    else:
                        continue
                    if not value_str:
                        continue
                    try:
                        total += float(value_str[0])
                    except ValueError:
                        continue
                    else:
                        found = True
            if not found:
                return None
            return total > 0
        except Exception as exc:
            log.debug(f"[{self.node_name}] vllm metrics probe ({url}) inconclusive: {exc}")
            return None

    def _probe_llamacpp_metrics_activity(self, url: str) -> bool | None:
        """Probe C: GET {url}/metrics and diff every llamacpp:* counter/gauge
        line against the previous tick's snapshot (agents-core-doorman-class-
        aware-activity-probe-v0, D1) — name-agnostic by design, no pinned
        metric list, so this survives llama.cpp metric names changing across
        builds. Closes Probe A's sampling gap: a generation that finished
        between two ~45s ticks (no is_processing, unchanged id_task at sample
        time) still shows up here as an advanced counter.

        Returns:
          True  — at least one llamacpp:* value increased since the last
                  snapshot (work happened between ticks).
          False — llamacpp:* lines were present and parsed, but nothing
                  advanced (confirmed idle).
          None  — the call failed (timeout/connection-refused/non-200), the
                  body has no llamacpp:* lines at all (structural absence —
                  vLLM or another engine on this port, never confirmed-idle),
                  there is no baseline yet to diff against (first successful
                  sample), or a value DECREASED since the last snapshot (the
                  engine restarted and its counters reset to zero — a reset
                  is never evidence of idleness). The baseline is re-seeded
                  to the fresh snapshot unconditionally in every one of these
                  cases where a snapshot was parsed; only the vote withholds.

        Never raises. Must be called OUTSIDE self.lock (blocking HTTP, ~2.5s timeout).
        """
        try:
            resp = requests.get(f"{url}/metrics", timeout=2.5)
            if resp.status_code != 200:
                return None
            current: dict[str, float] = {}
            for line in resp.text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or not line.startswith("llamacpp:"):
                    continue
                # Same brace-aware parsing as Probe B: a naive rsplit(" ", 1)
                # would silently take a trailing Prometheus timestamp field
                # as the value if one is ever emitted. Key on name+labels so
                # each distinct series is diffed independently — one series
                # resetting can't be masked by another advancing.
                brace_end = line.find("}")
                if brace_end != -1:
                    key = line[:brace_end + 1]
                    value_str = line[brace_end + 1:].strip().split()
                else:
                    parts = line.split(None, 1)
                    if len(parts) != 2:
                        continue
                    key, value_str = parts[0], parts[1].strip().split()
                if not value_str:
                    continue
                try:
                    current[key] = float(value_str[0])
                except ValueError:
                    continue
        except Exception as exc:
            log.debug(f"[{self.node_name}] llamacpp metrics probe ({url}) inconclusive: {exc}")
            return None

        if not current:
            # No llamacpp:* lines at all — this port isn't serving llama.cpp
            # (e.g. vLLM). Structural absence, not evidence of idleness.
            self._llamacpp_metrics_baseline.pop(url, None)
            return None

        previous = self._llamacpp_metrics_baseline.get(url)
        # Baseline update and the vote are separate steps (council ruling,
        # run 2026-08-19-123628): update unconditionally on successful parse.
        self._llamacpp_metrics_baseline[url] = current
        if previous is None:
            return None  # first successful sample — nothing to diff against yet

        reset_detected = False
        advanced = False
        for key, value in current.items():
            prev_value = previous.get(key)
            if prev_value is None:
                continue
            if value < prev_value:
                reset_detected = True
            elif value > prev_value:
                advanced = True

        if reset_detected:
            # Engine restart: counters reset to zero. Baseline is already
            # re-seeded above; the vote stays indeterminate for this tick —
            # a reset is not evidence of idleness.
            return None
        return advanced

    # ------------------------------------------------------------------
    # Capacity shadow (agents-core-doorman-capacity-shadow-v0)
    # ------------------------------------------------------------------

    def _fetch_capacity_gauges(self, timeout: float = 2.5) -> dict:
        """One GET {self.gw_url}/metrics, parsed value-preserving.

        Never raises, never blocks longer than `timeout`, and touches no
        doorman state — pure read. Outside self.lock and outside
        _capacity_shadow_scrape_lock's caller's expectations (this is the
        function that lock wraps for single-flight, not a lock holder
        itself).

        Returns {"probe_outcome": "ok"|"unreachable"|"gauge_absent",
                 "gauges": {...}, "probe_latency_ms": float}.
        `gauge_absent` — 200 but none of the target gauges are present in
        the body (e.g. big mode, where vLLM isn't serving at all).
        `unreachable` — timeout, connection error, or non-200.
        Neither outcome's gauges carry a fabricated 0 — see
        parse_vllm_capacity_gauges.
        """
        start = time.time()
        try:
            resp = requests.get(f"{self.gw_url}/metrics", timeout=timeout)
        except Exception as exc:
            log.debug(f"[{self.node_name}] capacity shadow scrape failed: {exc}")
            return {
                "probe_outcome": "unreachable",
                "gauges": {},
                "probe_latency_ms": (time.time() - start) * 1000,
            }
        latency_ms = (time.time() - start) * 1000
        if resp.status_code != 200:
            return {"probe_outcome": "unreachable", "gauges": {}, "probe_latency_ms": latency_ms}
        gauges = parse_vllm_capacity_gauges(resp.text)
        if all(v is None for v in gauges.values()):
            return {"probe_outcome": "gauge_absent", "gauges": gauges, "probe_latency_ms": latency_ms}
        return {"probe_outcome": "ok", "gauges": gauges, "probe_latency_ms": latency_ms}

    def capacity_shadow_scrape(self) -> dict:
        """Single-flight, floor-respecting capacity scrape (C3, spec
        2026-08-06). At most one real /metrics scrape in flight per node at
        any time; a concurrent or too-soon caller reuses the most recent
        sample, marked probe_outcome="coalesced" with sample_age_ms set to
        how old that reused reading is. Never touches self.lock, self.leases,
        self.wait_list, idle_since, or the serving cache — strict
        no-side-effect contract (C3).

        Must be called OUTSIDE self.lock (blocking HTTP, ~2.5s cap).
        """
        with self._capacity_shadow_scrape_lock:
            now = time.time()
            elapsed = now - self._capacity_shadow_last_scraped_at
            if (
                self._capacity_shadow_last_sample is not None
                and elapsed < DOORMAN_CAPACITY_SHADOW_MIN_INTERVAL_SEC
            ):
                out = dict(self._capacity_shadow_last_sample)
                out["probe_outcome"] = "coalesced"
                out["sample_age_ms"] = elapsed * 1000
                return out

            result = self._fetch_capacity_gauges()
            self._capacity_shadow_last_sample = result
            self._capacity_shadow_last_scraped_at = time.time()
            out = dict(result)
            out["sample_age_ms"] = 0.0
            return out

    def _capacity_shadow_predict_defer(self, work_id: str, principal: str | None) -> bool:
        """Cheap, off-lock approximation of acquire_or_defer's would-defer
        decision — used only to decide whether this acquire is on the
        would-defer path worth scraping for (agents-core-doorman-capacity-
        shadow-v0). Deliberately racy: the real decision happens a moment
        later under self.lock, and disagreement between the two is recorded
        (predicted_defer/actual_defer, shadow-predicate-divergence), not
        prevented. Mirrors _protected_lease_active + _brake_active without
        taking self.lock."""
        if work_id in self.wait_list:
            return True
        if self.brake_expires_at is not None and time.time() < self.brake_expires_at:
            return True
        now = time.time()
        for _wid, info in self.leases.items():
            if not (
                info.get("class", DEFAULT_LEASE_CLASS) == "protected"
                and now <= info["acquired_at"] + info["ttl_sec"]
            ):
                continue
            if principal is not None:
                p = info.get("principal", GHOST_PRINCIPAL)
                if p != GHOST_PRINCIPAL and p == principal:
                    continue
            return True
        return False

    def _gating_protected_lease(self, exclude_principal: str | None = None) -> dict | None:
        """Same matching logic as _protected_lease_active, but returns the
        gating lease's identity (work_id, lease_kind, principal) instead of
        a bool — capacity-shadow event enrichment only (agents-core-doorman-
        capacity-shadow-v0), so the record distinguishes "deferred behind
        real inference" from "deferred behind a coordination hold using no
        GPU". Must be called under self.lock."""
        now = time.time()
        for wid, info in self.leases.items():
            if not (
                info.get("class", DEFAULT_LEASE_CLASS) == "protected"
                and now <= info["acquired_at"] + info["ttl_sec"]
            ):
                continue
            if exclude_principal is not None:
                p = info.get("principal", GHOST_PRINCIPAL)
                if p != GHOST_PRINCIPAL and p == exclude_principal:
                    continue
            return {
                "work_id": wid,
                "lease_kind": info.get("lease_kind"),
                "principal": info.get("principal", GHOST_PRINCIPAL),
            }
        return None

    def _emit_capacity_shadow_event(
        self,
        *,
        work_id: str,
        role: str,
        reason: str,
        principal: str | None,
        lease_class: str,
        scrape: dict,
        predicted_defer: bool,
        actual_defer: bool,
        scrape_to_lock_ms: float,
        gating_lease: dict | None,
    ) -> None:
        """One JSON line per would-defer decision, via the existing
        _write_idle_log machinery (DOORMAN_IDLE_LOG) also used by
        _emit_release_event. Must be called under self.lock — file I/O only,
        same convention as _emit_release_event."""
        gauges = scrape.get("gauges") or {}
        _write_idle_log(
            self.node_name, "capacity-shadow", len(self.leases),
            work_id=work_id,
            role=role,
            reason=reason,
            principal=principal if principal is not None else GHOST_PRINCIPAL,
            lease_class=lease_class,
            num_requests_running=gauges.get("num_requests_running"),
            num_requests_waiting=gauges.get("num_requests_waiting"),
            capacity_wait=gauges.get("capacity_wait"),
            deferred_wait=gauges.get("deferred_wait"),
            probe_outcome=scrape.get("probe_outcome"),
            probe_latency_ms=round(scrape.get("probe_latency_ms", 0.0), 2),
            sample_age_ms=round(scrape.get("sample_age_ms", 0.0), 2),
            predicted_defer=predicted_defer,
            actual_defer=actual_defer,
            scrape_to_lock_ms=round(scrape_to_lock_ms, 2),
            gating_lease_work_id=(gating_lease or {}).get("work_id"),
            gating_lease_kind=(gating_lease or {}).get("lease_kind"),
            gating_lease_principal=(gating_lease or {}).get("principal"),
        )
        if predicted_defer != actual_defer:
            _write_idle_log(
                self.node_name, "shadow-predicate-divergence", len(self.leases),
                work_id=work_id,
                predicted_defer=predicted_defer,
                actual_defer=actual_defer,
            )

    def _probe_gpu1_glances(self) -> bool | None:
        """Probe E: the GPU 1 (berth) Glances activity probe
        (gw-gpu1-berth-standing-seat-v0, leg 2).

        GET {DOORMAN_GLANCES_URL}/api/4/gpu/ (Glances API v4; the API issues
        a 307 the client follows — requests follows redirects by default).
        The response is a JSON LIST of per-GPU objects with `gpu_id`/`mem`/
        `proc` (the T1c contract). The berth is GPU 1 — a SEPARATE card from
        the GPU 0 production seat — so this source is engine-independent of
        whatever class :8081 is serving and votes in ALL combine branches.

        Tri-state, mirroring Probe B's discipline (non-200/exception -> None,
        NEVER a spurious False):
          True  — nvidia1.proc > DOORMAN_GPU1_PROC_THRESHOLD (any proc>0
                  sample resets idle_since, same semantics as the existing
                  probe-activity re-arm).
          False — nvidia1.proc == 0, INCLUDING the idle-but-warm berth
                  (mem ~77%, proc 0). The expected standing state MUST vote
                  False or the box never sleeps (the S2 binding statement).
                  The vote is PROC-ONLY by design: mem is a residency
                  diagnostic (it belongs to the guard's suspend check + the
                  stop's drain verify, not the sleep predicate).
          None  — unreachable/unparsable (dead Glances, timeout, malformed
                  list). A dead Glances maps to the BOUNDED probe-blindness
                  class (the existing grace+900=1500s semantics, then
                  proceed to stop) — NOT the topology-unknown unbounded
                  never-park class, which is reserved for a fundamentally
                  unclassifiable seat.

        Also updates the diagnostic fields _gpu1_glances_mem_pct /
        _gpu1_glances_proc / _gpu1_seat_health (status surface). Never
        raises. Must be called OUTSIDE self.lock (blocking HTTP, ~2.5s cap).
        """
        # Seat health: the berth's own :8082 /health (200 = seat up).
        try:
            resp = requests.get(f"{self._slot2_url()}/health", timeout=2.5)
            self._gpu1_seat_health = resp.status_code == 200
        except Exception:
            self._gpu1_seat_health = False

        # berth_unit: the ninfer-fixer systemd unit state (active/inactive) -
        # distinct from seat_health (the :8082 /health 200). Probed over the
        # doorman's existing gravitywell ssh channel; timeout/error -> None
        # (unknown). Never conflated with the seat health.
        try:
            r = subprocess.run(
                ["ssh", "gravitywell", "systemctl", "is-active", "ninfer-fixer"],
                capture_output=True, text=True, timeout=3.0,
            )
            self._gpu1_berth_unit = r.stdout.strip() == "active"
        except Exception:
            self._gpu1_berth_unit = None

        try:
            resp = requests.get(f"{DOORMAN_GLANCES_URL}/api/4/gpu/", timeout=2.5)
            if resp.status_code != 200:
                return None
            gpus = resp.json()
            if not isinstance(gpus, list):
                return None
            # glances v4 labels gpu_id "nvidia<N>" (N = the CUDA device index);
            # the berth is CUDA device 1 (the 3090 Ti). Match the trailing index
            # - not the whole label - so the live "nvidia1" and the hermetic
            # fixture's bare "1" both resolve to the 3090 Ti. The old exact-match
            # on "1" silently never matched the real "nvidia1" label, so every
            # gpu1 glances reading came back dead. (2026-08-24 live-test finding)
            def _glances_index(entry):
                m = re.search(r"(\d+)$", str(entry.get("gpu_id")))
                return m.group(1) if m else str(entry.get("gpu_id"))

            gpu1 = next(
                (
                    g for g in gpus
                    if isinstance(g, dict) and _glances_index(g) == "1"
                ),
                None,
            )
            if gpu1 is None:
                return None
            proc = gpu1.get("proc")
            mem = gpu1.get("mem")
            if not isinstance(proc, (int, float)):
                return None
            if isinstance(mem, (int, float)):
                self._gpu1_glances_mem_pct = float(mem)
            self._gpu1_glances_proc = float(proc)
            return proc > DOORMAN_GPU1_PROC_THRESHOLD
        except Exception as exc:
            log.debug(f"[{self.node_name}] gpu1 glances probe inconclusive: {exc}")
            return None

    def _probe_flashnext_seat(self, sequential: bool = False) -> tuple[str, str | None, bool | None, str | None]:
        """Probe the flash-next seat on :30000 (agents-core-doorman-flashnext-
        handover-v0, D2/D7). Returns (state, served_id, registered, error_class):

          state:
            "down"          — /health connection refused (host reachable, port
                              closed): no seat listener. Window none.
                              ONLY ConnectionRefusedError classifies as down;
                              every other ConnectionError (DNS failure,
                              unreachable host, reset) is "blind" (Invariant 7:
                              blindness is never a no).
            "blind"         — timeout / connection error (non-refused) /
                              unparseable /health: indeterminate. Window
                              indeterminate (null); the 27B axis behaves
                              exactly as today (a missing probe is blindness,
                              never False).
            "up_registered" — /health answered AND /v1/models returned 200
                              with data[0].id == GW_FLASHNEXT_MODEL_ID (EXACT
                              canonical_id match, D2b — never
                              _gw_registry_lookup). Window active (when the
                              day seat is down).
            "up_unverified" — /health answered but /v1/models was non-200 /
                              unparseable / timed out: a listener exists on
                              the seat port, identity unverified (SGLang
                              mid-load answers health before models). Window
                              ACTIVE — safe direction: the card may be
                              committed, and a refused acquire for a few
                              ticks is cheaper than waking the 27B onto a
                              committed card.
            "up_foreign"    — /v1/models returned 200 with a DIFFERENT
                              data[0].id: an identifiable non-seat occupant
                              on the seat port. Window NONE + structured WARN
                              (a squatter is an operator-visible /status
                              state to clear, not a window; the
                              unregistered-occupant posture of the big-probe
                              DoD-4a gate amendment — never guess).

          served_id: data[0].id of a 200 models response, else None.
          registered: exact canonical_id match (served_id ==
                      GW_FLASHNEXT_MODEL_ID); None when the models endpoint
                      never answered 200.
          error_class: the ACTUAL exception class name when the probe failed
                      (e.g. "Timeout", "ConnectionRefusedError",
                      "ConnectionError"), or None on success. Carried by the
                      D4 BLIND-proceed log.warning so the operator sees the
                      real failure class, not a static placeholder.

        D7 placement/shape: the TICK path (sequential=False) runs the two GETs
        concurrently in a 2-worker pool (mirrors the existing pool style); the
        ACQUIRE path (sequential=True) is a sequential short-circuit — GET
        /health first, and only on an HTTP response GET /v1/models (a refused
        /health stops after exactly one request; no per-acquire pool churn).
        Timeouts 2.5s each. No-redirect pinned: a 3xx on the seat port is
        blindness, not truth (deliberate hardening vs the existing probes).
        Any error -> BLIND, never False. Never raises. Must be called
        OUTSIDE self.lock (blocking HTTP).
        """
        def _health_call():
            try:
                return requests.get(
                    f"{GW_FLASHNEXT_URL}/health",
                    timeout=2.5,
                    allow_redirects=False,
                )
            except Exception as exc:
                return exc

        def _models_call():
            try:
                return requests.get(
                    f"{GW_FLASHNEXT_URL}/v1/models",
                    timeout=2.5,
                    allow_redirects=False,
                )
            except Exception as exc:
                return exc

        if sequential:
            # Acquire path (D4): sequential short-circuit — a refused /health
            # is exactly one HTTP call.
            health = _health_call()
            models = _models_call() if isinstance(health, requests.Response) else None
        else:
            # Tick path (D7): 2-worker pool, both GETs issued concurrently.
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_health = pool.submit(_health_call)
                fut_models = pool.submit(_models_call)
                health = fut_health.result()
                models = fut_models.result()

        # /health half: any HTTP response (any status) = a listener exists on
        # the seat port. A connection-level failure is the DOWN/BLIND split:
        # ONLY ConnectionRefusedError (the host answered RST — no seat
        # listener) classifies as "down". Every other ConnectionError (DNS
        # failure, unreachable host, connection reset) and every Timeout is
        # "blind" (Invariant 7: blindness is never a no).
        if isinstance(health, requests.exceptions.ConnectionError):
            # Walk the exception chain: requests wraps the underlying
            # ConnectionRefusedError in a ConnectionError. Check both the
            # exception itself and its __cause__/__context__ chain.
            _exc = health
            while _exc is not None:
                if isinstance(_exc, ConnectionRefusedError):
                    # Connection refused: the host answered RST — no seat
                    # listener.
                    return ("down", None, None, type(health).__name__)
                _exc = _exc.__cause__ or _exc.__context__
            # Non-refused ConnectionError (DNS, unreachable, reset): blind.
            return ("blind", None, None, type(health).__name__)
        if not isinstance(health, requests.Response):
            # Timeout or any other non-ConnectionError: indeterminate.
            return ("blind", None, None, type(health).__name__)

        # A listener exists. Identity half — only when the models response is
        # an actual HTTP response (a sequential short-circuit on a refused
        # /health never reaches here; a models timeout/exception is BLIND-
        # SHAPED: up but unverified, never down).
        if isinstance(models, requests.Response) and models.status_code == 200:
            try:
                data = models.json().get("data")
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    served_id = data[0].get("id")
                    if isinstance(served_id, str) and served_id:
                        if served_id == GW_FLASHNEXT_MODEL_ID:
                            return ("up_registered", served_id, True, None)
                        # Identifiable non-seat occupant (D2: UP_FOREIGN).
                        log.warning(
                            f"[{self.node_name}] flashnext-seat-foreign-occupant — "
                            f":30000 /v1/models serves {served_id!r} "
                            f"(expected {GW_FLASHNEXT_MODEL_ID!r}); not a window, "
                            f"operator-visible /status state to clear"
                        )
                        return ("up_foreign", served_id, False, None)
            except Exception:
                pass
        # /health answered but /v1/models was non-200 / unparseable / timed
        # out: a listener exists, identity unverified (SGLang mid-load).
        return ("up_unverified", None, None, None)

    def _probe_flashnext_activity(self) -> bool | None:
        """D2 legibility probe (gw-doorman-flashnext-idle-awareness-v0):
        read the flash-next seat's running/waiting request counters, from
        either of two sources on {GW_FLASHNEXT_URL}:

          1. GET /metrics — SGLang's Prometheus plaintext exposition (same
             shape as the vLLM /metrics probes), read for the
             running/queued request gauges (_SGLANG_ACTIVITY_METRICS).
          2. GET /get_load — the DoD-0-live-confirmed fallback for THIS
             deployment (2026-09-26, seat UP): the seat runs with
             enable_metrics=False, so /metrics answers 404 and the gauges
             are never exposed; /get_load answers 200 with one object per
             dp_rank carrying num_reqs (running) and num_waiting_reqs
             (waiting).

        /metrics is tried first and only a NON-200 / unparsable / empty
        gauge read falls through to /get_load — a confirmed reading from
        either source is authoritative for the tick. A 404 (or any other
        failure) on BOTH is "unknown".

        Returns:
          True  — a target gauge/field was found and parsed with a nonzero
                  value (running/waiting >= 1: confirmed activity).
          False — the target gauges/fields were found and parsed, all zero
                  (confirmed idle).
          None  — both calls failed (timeout/connection-refused/non-200),
                  neither body contained a target field, or the values were
                  unparsable: the "unknown" substate. Never a stop
                  authorization, never an idle reading (D2: a 404 /
                  non-200 / unparseable activity gauge is unknown, not idle
                  — the whole reason the 404-on-/metrics deployment needs
                  the /get_load fallback rather than a permanently-unknown
                  clock).

        LEGIBILITY ONLY (D2): this probe NEVER feeds the stop decision — the
        stop path consumes the seat-STATE probe (_probe_flashnext_seat)
        exclusively. It feeds only the _flashnext_last_activity_ts clock
        (rendered as withheld-active vs withheld-up-idle and on /status). A
        clock not refreshed this tick renders "unknown", never idle.

        Never raises. Must be called OUTSIDE self.lock (blocking HTTP,
        2.5s timeout, no-redirect pin — same discipline as the seat probe).
        """
        try:
            resp = requests.get(
                f"{GW_FLASHNEXT_URL}/metrics",
                timeout=2.5,
                allow_redirects=False,
            )
            if resp.status_code == 200:
                total = 0.0
                found = False
                for line in resp.text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    for metric in _SGLANG_ACTIVITY_METRICS:
                        if line.startswith(metric + "{"):
                            # Brace-aware parse (same convention as the vLLM
                            # activity probe): the value starts right after the
                            # closing brace, not at the last space.
                            brace_end = line.find("}")
                            if brace_end == -1:
                                continue
                            value_str = line[brace_end + 1:].strip().split()
                        elif line.startswith(metric + " "):
                            value_str = line[len(metric):].strip().split()
                        else:
                            continue
                        if not value_str:
                            continue
                        try:
                            total += float(value_str[0])
                        except ValueError:
                            continue
                        else:
                            found = True
                if found:
                    return total > 0
        except Exception as exc:
            # Never a stop authorization, never an idle reading — just an
            # inconclusive metrics leg.
            log.debug(
                f"[{self.node_name}] flashnext metrics leg inconclusive "
                f"(falling through to /get_load): {exc}"
            )
        # Metrics leg inconclusive (non-200 — the enable_metrics=False
        # deployment answers 404 — an empty body, no target gauge, or a
        # failed call): fall through to the /get_load leg.
        try:
            return self._probe_flashnext_load()
        except Exception as exc:
            log.debug(
                f"[{self.node_name}] flashnext /get_load leg inconclusive "
                f"too: {exc}"
            )
            return None

    def _probe_flashnext_load(self) -> bool | None:
        """The /get_load leg of the D2 legibility activity probe
        (gw-doorman-flashnext-idle-awareness-v0, DoD-0 live 2026-09-26):
        GET {GW_FLASHNEXT_URL}/get_load and sum the running/waiting request
        counts across the dp_rank objects.

        Same tri-state contract as the /metrics leg: True on any nonzero
        running/waiting count, False when the fields were present and all
        zero, None (unknown) on any failure — non-200, non-JSON, a JSON body
        that is not a list of objects, or objects carrying none of the
        target keys. A malformed response is never an idle reading and never
        a stop authorization. Never raises. Must be called OUTSIDE self.lock.
        """
        resp = requests.get(
            f"{GW_FLASHNEXT_URL}/get_load",
            timeout=2.5,
            allow_redirects=False,
        )
        if resp.status_code != 200:
            return None
        entries = resp.json()
        if not isinstance(entries, list):
            return None
        total = 0.0
        found = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in _SGLANG_LOAD_RUNNING_KEYS + _SGLANG_LOAD_WAITING_KEYS:
                if key not in entry:
                    continue
                try:
                    total += float(entry[key])
                except (TypeError, ValueError):
                    continue
                else:
                    found = True
        if not found:
            return None
        return total > 0

    def _probe_slot_activity(self) -> bool | None:
        """Tri-state unmediated-caller activity probe across all signal sources
        (gw-doorman-vllm-activity-probe-v0, extended by agents-core-doorman-
        class-aware-activity-probe-v0) — Probe A (llama.cpp /slots), Probe B1/B2
        (vLLM /metrics, both dual slots), Probe C (llama.cpp /metrics counter
        diff), and Probe E (the GPU 1 / berth Glances vote,
        gw-gpu1-berth-standing-seat-v0), dispatched concurrently in a 5-worker
        pool so total probe-tick latency stays ~2.5s (the slowest probe)
        rather than growing additively.

        Probe E is engine-independent (the berth is a separate card from
        :8081's class) and therefore votes in ALL combine branches below —
        it is not dropped by the class-aware filter.

        Class-aware combine (D2): which sources VOTE depends on self._serving_is_big,
        resolved earlier THIS tick by _refresh_serving_cache. A source that is
        structurally absent for the resolved class is dropped rather than voting
        None — that was the defect (structural absence read as indeterminate,
        parking a genuinely-idle seat, or blinding a genuinely-busy one):

          - self._serving_is_big is None (flag off, OR flag on but topology
            unresolved this tick): vote-unaware, exactly as before D2 — all
            sources vote. (Unknown topology's never-park behavior lives in the
            park block, not here — this function's output is unchanged for it.)
          - True  (llama.cpp class): B1/B2 are structurally absent — non-voting.
            Combine A + C.
          - False (vLLM class): A is structurally absent — non-voting. Combine
            B1 + B2; if the declared home posture is slot1-solo, B2 (Slot 2,
            parked by design) is non-voting too — combine B1 alone.
          - Probe E (the GPU 1 / berth Glances vote) is NOT dropped by the
            class-aware filter: the berth is a separate card from :8081's
            class, so it votes in ALL three branches (appended to the vote
            list above).

        Combines: True if any voting source confirms activity (a real True
        always wins); False only if every voting source confirms no activity;
        None (indeterminate) otherwise.

        Must be called OUTSIDE self.lock (blocking HTTP via a thread pool).
        """
        with ThreadPoolExecutor(max_workers=5) as pool:
            fut_a = pool.submit(self._probe_llama_slots_activity)
            fut_b1 = pool.submit(self._probe_vllm_metrics_activity, self.gw_url)
            fut_b2 = pool.submit(self._probe_vllm_metrics_activity, self._slot2_url())
            fut_c = pool.submit(self._probe_llamacpp_metrics_activity, self.gw_url)
            # Probe E (gw-gpu1-berth-standing-seat-v0): the GPU 1 (berth)
            # Glances vote. Engine-independent — the berth is a separate card
            # from :8081's class — so it votes in ALL combine branches below.
            fut_gpu1 = pool.submit(self._probe_gpu1_glances)
            (a, b1, b2, c, gpu1) = (
                fut_a.result(), fut_b1.result(), fut_b2.result(),
                fut_c.result(), fut_gpu1.result(),
            )

        self._last_probe_raw = {"A": a, "B1": b1, "B2": b2, "C": c, "GPU1": gpu1}

        serving_is_big = self._serving_is_big
        if serving_is_big is None:
            votes = [a, b1, b2, c, gpu1]
        elif serving_is_big:
            votes = [a, c, gpu1]
        else:
            home_mode = self._read_declared_home_posture()
            votes = ([b1] if home_mode == "slot1-solo" else [b1, b2]) + [gpu1]

        if any(v is True for v in votes):
            return True
        if all(v is False for v in votes):
            return False
        return None

    def _refresh_serving_cache(self) -> None:
        """Refresh the serving cache by probing _is_serving outside the lock.

        This method MUST be called when the lock is NOT held, as it performs
        a blocking network call. It then takes the lock briefly to update the
        cached fields.

        WARNING: This method is non-reentrant — it MUST NOT be called from
        within an already-held self.lock context or it will deadlock
        (threading.Lock is non-reentrant).

        When DOORMAN_MODE_AWARE_ADMISSION is True, also resolves GW's topology via
        ONE call to agents_core.llm.gw_serving_state() (outside the lock, ~4s ×
        up to 4 HTTP calls) — the single resolution that feeds serving_mode's
        big/dual branch, serving_is_big, and big_probe_state (agents-core-doorman-
        serving-mode-topology-truthful-v0). status_snapshot() only ever reads the
        cached result; it never calls gw_serving_state() itself.

        When DOORMAN_PROBE_LLAMA_ACTIVITY is True, also probes llama.cpp's /slots
        and /metrics and vLLM's /metrics (both dual slots) for unmediated-caller
        activity (outside the lock) — see _probe_slot_activity. This runs AFTER
        self._serving_is_big is committed below (a second, brief lock
        acquisition), because the class-aware combine (D2,
        agents-core-doorman-class-aware-activity-probe-v0) needs THIS tick's
        resolved class, not the previous tick's.
        """
        serving = self._is_serving(timeout=2.0)
        creative_serving = self._is_creative_serving()
        # Flash-next seat probe (agents-core-doorman-flashnext-handover-v0,
        # D7): runs unconditionally every tick, beside the serving probe and
        # OUTSIDE the lock. Deliberately NOT in _probe_slot_activity's pool
        # (that pool runs only when DOORMAN_PROBE_LLAMA_ACTIVITY is on, and
        # the window determination must work regardless of that flag) and
        # never joined to the activity vote lists (Invariant 8: :30000
        # activity is irrelevant to the 27B axis's idle clock).
        # Flash-next D2 legibility activity probe
        # (gw-doorman-flashnext-idle-awareness-v0): unconditional, beside
        # the two existing :30000 GETs, same 2.5s timeout and no-redirect
        # pin. Deliberately NOT in _probe_slot_activity's flag-gated pool —
        # the flag's rollback lever must not be able to starve this
        # legibility source. A clock not refreshed this tick renders
        # "unknown", never idle.
        # The two :30000 legs run as CONCURRENT tasks of a small private
        # pool (D2 placement: "widen the worker pool or add a concurrent
        # task") so the legibility leg never adds its 2.5s (metrics leg) +
        # 2.5s (/get_load leg) to the tick's wall time on top of the seat
        # probe's own 2.5s: the whole :30000 probe pass stays ~2.5s, the
        # slowest single leg, never additive.
        with ThreadPoolExecutor(max_workers=2) as _fn_pool:
            _fn_seat_fut = _fn_pool.submit(
                self._probe_flashnext_seat, sequential=False
            )
            _fn_activity_fut = _fn_pool.submit(self._probe_flashnext_activity)
            try:
                flashnext_state, flashnext_served_id, flashnext_registered, flashnext_error_class = (
                    _fn_seat_fut.result()
                )
            except Exception as exc:
                # The seat probe is documented never to raise; a raise is a
                # bug, and the fail-closed reading of a bug on this axis is
                # BLIND (Invariant 7: blindness is never a no), never
                # "down" — and never an exception that takes the refresh
                # tick (and with it the whole idle-stop loop) down.
                log.warning(
                    f"[{self.node_name}] flashnext seat probe raised "
                    f"{type(exc).__name__} ({exc}) — treating as blind"
                )
                flashnext_state, flashnext_served_id = "blind", None
                flashnext_registered, flashnext_error_class = None, type(exc).__name__
            try:
                flashnext_activity: bool | None = _fn_activity_fut.result()
            except Exception as exc:
                # Legibility leg: any failure is "unknown", never idle
                # (D2) — and never allowed to break the tick.
                log.debug(
                    f"[{self.node_name}] flashnext activity probe raised "
                    f"{type(exc).__name__} ({exc}) — unknown substate"
                )
                flashnext_activity = None

        # Single topology resolution — outside the lock (blocking HTTP).
        topology_state = None
        if DOORMAN_MODE_AWARE_ADMISSION:
            try:
                topology_state = gw_serving_state(endpoint=self.gw_url)
            except Exception as exc:
                log.warning(
                    f"[{self.node_name}] gw_serving_state() raised, degrading to "
                    f"unknown topology resolution: {exc}"
                )
                topology_state = None

        with self.lock:
            self._cached_serving = serving
            self._cached_creative_serving = creative_serving
            self._serving_checked_at = time.time()

            # Flash-next window bookkeeping (D2/D5): computed under the lock
            # from THIS tick's probe + THIS tick's serving read. The window is
            # ACTIVE iff (a) the :30000 probe reports a seat listener
            # (up_registered OR up_unverified — the safe direction) AND
            # (b) the :8081 day-seat probe reports down. A blind probe leaves
            # the window indeterminate (None) — the 27B axis behaves exactly
            # as today (Invariant 7: blindness is not down on the :30000
            # axis). window_since is set on the first active read, held while
            # active, cleared on close; window_closed_at is set on the
            # active->none transition tick and consumed by the D9 re-anchor.
            self._flashnext_state = flashnext_state
            self._flashnext_served_id = flashnext_served_id
            self._flashnext_registered = flashnext_registered
            self._flashnext_error_class = flashnext_error_class
            # Window determination (D2): ACTIVE iff the :30000 probe reports
            # a seat listener (up_registered OR up_unverified — the safe
            # direction) AND the :8081 day-seat probe reports down. When the
            # seat is up but the day seat is ALSO up (hand-back overlap,
            # both briefly serving), the window is "none" — the day seat is
            # serving, so no handover is in progress. A blind probe leaves
            # the window indeterminate (None) — the 27B axis behaves exactly
            # as today (Invariant 7: blindness is not down on the :30000
            # axis).
            if flashnext_state in ("up_registered", "up_unverified"):
                window = "active" if not serving else "none"
            elif flashnext_state in ("down", "up_foreign"):
                window = "none"
            else:  # "blind"
                window = None
            if window == "active":
                if self._flashnext_window != "active":
                    # First active read this window (or re-open after a flap):
                    # stamp the open edge.
                    self._flashnext_window_since = time.time()
                self._flashnext_window = "active"
            else:
                if self._flashnext_window == "active":
                    # active -> none (or indeterminate) transition tick: the
                    # D9 close re-anchor consumes this exactly once.
                    self._flashnext_window_closed_at = time.time()
                self._flashnext_window = window
                self._flashnext_window_since = None

            # Flash-next idle-awareness D1 bookkeeping
            # (gw-doorman-flashnext-idle-awareness-v0): the continuous-
            # blindness clock for the stop path's bounded blind-withhold.
            # Bookkeeping lives HERE, in the tick's probe pass under the
            # lock (Council open question 1, resolved): the probe method
            # stays stateless; this block already mutates per-tick state
            # (window_since/window_closed_at) from this tick's own probe
            # result. Armed on the first blind (or cold/None) read since
            # the last definitive read, cleared on any definitive read.
            # The COLD-START rule (D1): before the first definitive probe
            # read the state is None — treated as BLIND here (bounded
            # withhold), never as down/idle-ok.
            if flashnext_state in ("blind", None):
                if self._flashnext_blind_since is None:
                    self._flashnext_blind_since = time.time()
                # A definitive "down" probe clears the D2 activity stamp
                # (a dead seat cannot withhold via a stale stamp); blind
                # and cold reads leave it unchanged.
            else:
                self._flashnext_blind_since = None
                if flashnext_state == "down":
                    self._flashnext_last_activity_ts = None
            # D3 legibility: a steady-state withhold is a normal safety
            # state, not a failure — entering it does not set last_error;
            # a definitive non-withhold seat read (down / up_foreign — the
            # states where this axis does not withhold) re-arms the
            # one-shot transition warning, so the next entry into any
            # withhold substate journals once again.
            if flashnext_state in ("down", "up_foreign"):
                self._flashnext_withhold_substate = None
            # D2 stamp rule (one sentence): the clock is stamped only by a
            # successful probe that observed running/waiting >= 1;
            # successful zero-activity reads, failed reads, and the cold
            # state (None = no observation) leave it unchanged; a
            # definitive "down" probe clears it (handled above). The clock
            # is probe-stamp time, not request-end time (45s sampling — a
            # withhold therefore releases up to ~645-660s after the last
            # confirmed activity, inherent to the tick resolution).
            if flashnext_activity is True:
                self._flashnext_last_activity_ts = time.time()

            if DOORMAN_MODE_AWARE_ADMISSION:
                self._cached_topology_state = topology_state
                controller_owns = self._controller_lease_active()

                # big_probe_state (Council open question 2; membership set added
                # by agents-core-doorman-big-seat-membership-v0): "confirmed" = a
                # served id is a registry-declared big-seat member. "refuted" =
                # the models endpoint answered and none of the served ids is a
                # big-seat member, AND the served id is itself registered — we
                # looked and it is definitively not big. "unknown" = either the
                # resolution itself failed (authority_gap, the models endpoint
                # never answered, or the resolver call raised/returned None), or
                # the served id has no gw_models.yaml row at all — we could not
                # classify it (DoD 4a gate amendment: an unregistered served
                # model must never read as the stronger "refuted"). mode_inferred
                # is never read here.
                models_answered = _topology_models_answered(topology_state)
                if topology_state is None or topology_state.authority_gap or not models_answered:
                    self._big_probe_state = "unknown"
                elif set(topology_state.served_ids) & gw_big_seat_members():
                    self._big_probe_state = "confirmed"
                elif topology_state.unknown_model:
                    self._big_probe_state = "unknown"
                else:
                    self._big_probe_state = "refuted"

                if self._big_probe_state == "refuted":
                    # We looked and no served id is a big-seat member.
                    self._serving_is_big = False
                elif self._big_probe_state == "confirmed":
                    # Controller win takes precedence over probe confirmation (AC6-D)
                    self._serving_is_big = bool(serving and not controller_owns)
                else:  # unknown — could not look; a guess here reproduces the bug this fixes
                    self._serving_is_big = None
                    log.warning(
                        f"[{self.node_name}] topology resolution unknown "
                        f"(authority_gap={None if topology_state is None else topology_state.authority_gap}, "
                        f"models_answered={models_answered}) — serving_is_big left "
                        f"None rather than guessed; topology_resolution_unknown"
                    )

        # Optional slot-activity probe — outside the lock (blocking HTTP), and
        # after self._serving_is_big above so the class-aware combine (D2) votes
        # on THIS tick's resolved class. Tri-state: True (confirmed activity),
        # False (confirmed idle), None (indeterminate — at least one voting
        # source was ambiguous this tick).
        probe_activity: bool | None = None
        if DOORMAN_PROBE_LLAMA_ACTIVITY:
            probe_activity = self._probe_slot_activity()

        with self.lock:
            # Probe-driven idle keepalive (doorman-probe-llama-activity-v0, extended by
            # gw-doorman-vllm-activity-probe-v0): an unmediated caller (e.g. OpenCode
            # hitting a dual-mode vLLM slot directly) never acquires a lease, so
            # detected activity on either signal source re-arms the dwell-stop clock the
            # same way an arriving lease resets idle_since in acquire_lease(). No-op
            # while leases exist — lease-driven bookkeeping already covers that case.
            self._probe_indeterminate = (
                DOORMAN_PROBE_LLAMA_ACTIVITY and probe_activity is None
            )
            if DOORMAN_PROBE_LLAMA_ACTIVITY and probe_activity is True and not self.leases:
                was_idle_unset = self.idle_since is None
                self.idle_since = time.time()
                self._idle_since_source = "probe"
                if was_idle_unset:
                    _write_idle_log(self.node_name, "probe_activity_detected", 0)

    def _controller_lease_active(self) -> bool:
        """Check if a mode-controller lease is currently active (non-expired).

        Must be called under self.lock. Returns True iff some non-expired lease
        has role == "mode-controller".
        """
        now = time.time()
        for lease_info in self.leases.values():
            if (lease_info.get("role") == "mode-controller"
                and now <= lease_info["acquired_at"] + lease_info["ttl_sec"]):
                return True
        return False

    def _foreign_controller_lease_active(self, work_id: str | None) -> bool:
        """Check if a non-expired mode-controller lease is held by a DIFFERENT work_id.

        Must be called under self.lock. Identity-aware counterpart to
        _controller_lease_active() (AC3a, agents-core-doorman-mode-bearing-acquire-v0):
        that method returns a bare bool and cannot distinguish self from other, so a
        controller's own second acquire would defer to itself. This method is used
        only by the mode-bearing acquire path — every other caller of
        _controller_lease_active() is unaffected.
        """
        now = time.time()
        for wid, lease_info in self.leases.items():
            if (lease_info.get("role") == "mode-controller"
                    and wid != work_id
                    and now <= lease_info["acquired_at"] + lease_info["ttl_sec"]):
                return True
        return False

    def _resolve_live_posture(self) -> str:
        """Resolve GW's current serving posture directly on the mode-bearing
        acquire path (agents-core-doorman-warm-box-mode-convergence-v0, AC5) —
        never from _cached_topology_state, independent of
        DOORMAN_MODE_AWARE_ADMISSION (which gates only the background-tick
        cache, and defaults off). Called at most once per mode-bearing
        acquire, outside self.lock (self.wake_lock is held).

        Mirrors _topology_serving_mode's big/dual/unknown rules plus the
        big-probe split-brain check _refresh_serving_cache derives into
        _big_probe_state — computed here from a freshly-resolved state so it
        works even when _big_probe_state was never populated (flag off).
        A resolver exception degrades to "unknown" and never propagates out
        of ensure_serving() (AC5).
        """
        try:
            state = gw_serving_state(endpoint=self.gw_url)
        except Exception as exc:
            log.warning(
                f"[{self.node_name}] gw_serving_state() raised during mode-bearing "
                f"acquire, degrading to unknown topology resolution: {exc}"
            )
            return "unknown"

        if state is None or state.authority_gap or state.unknown_model:
            return "unknown"
        if state.mode == "dual":
            # Half-converged dual: configured dual with only one slot actually
            # up must not report "dual".
            return "dual" if state.distinct_second_model else "unknown"
        if state.mode == "big":
            models_answered = (
                state.source_freshness.get("models_endpoint", {}).get("status")
                == "answered"
            )
            if not models_answered:
                return "unknown"
            if not (set(state.served_ids) & gw_big_seat_members()):
                # We looked and no served id is a big-seat member — split-brain,
                # not a confident "big" (mirrors _refresh_serving_cache's
                # big_probe_state="refuted" branch).
                return "unknown"
            return "big"
        return "unknown"

    def _topology_serving_mode(self) -> str:
        """Derive serving_mode's "what topology is actually serving" branch from
        the single cached GwServingState (agents-core-doorman-serving-mode-
        topology-truthful-v0). Read-only, no network call — must be called under
        self.lock. Never called when controller_owns or not serving (those branches
        are decided in status_snapshot() before reaching here).

        A resolution failure — no resolution yet, authority_gap, or unknown_model —
        always degrades to "unknown", never a confident "big"/"dual" guess.
        mode_inferred is never read here (require_authoritative_mode's contract).
        """
        state = self._cached_topology_state
        if state is None or state.authority_gap or state.unknown_model:
            return "unknown"
        if state.mode == "dual":
            # Half-converged dual (Council open question 1): configured dual with
            # only one slot actually up must not report "dual".
            return "dual" if state.distinct_second_model else "unknown"
        if state.mode == "big":
            # DoD 3 non-contradiction: the models-endpoint half of this SAME
            # resolution already refuted the 122B being served — a flip-controller
            # "big" claim under that condition is a real split-brain, not big.
            if self._big_probe_state == "refuted":
                return "unknown"
            return "big"
        return "unknown"

    # ------------------------------------------------------------------
    # Foreground-priority gate (gw-router-phase1-foreground-gate)
    # ------------------------------------------------------------------

    def _protected_lease_active(self, exclude_principal: str | None = None) -> bool:
        """True iff a non-expired lease with class=='protected' currently exists
        that is not exempted by exclude_principal.

        Must be called under self.lock. A missing class defaults to
        "deferrable" (agents_core.doorman_server.DEFAULT_LEASE_CLASS), so
        pre-gate leases (no `class` field) never count as protected.

        exclude_principal=None (the default) reproduces the original global
        behaviour exactly — any protected lease gates. When a real principal
        is supplied, a protected lease whose stored principal equals it is
        skipped (the caller's own admission group never blocks itself). A
        lease stamped GHOST_PRINCIPAL, or one with no `principal` key at all
        (mirrors _worker_lease_blockers' `info.get("principal", GHOST_PRINCIPAL)`
        fail-safe read), always gates — ghosts are never exempted.
        """
        now = time.time()
        for wid, info in self.leases.items():
            if (info.get("class", DEFAULT_LEASE_CLASS) == "protected"
                    and now <= info["acquired_at"] + info["ttl_sec"]):
                if exclude_principal is not None:
                    p = info.get("principal", GHOST_PRINCIPAL)
                    if p != GHOST_PRINCIPAL and p == exclude_principal:
                        log.info(
                            f"[{self.node_name}] foreground_gate_exempt work_id={wid} "
                            f"principal={exclude_principal}"
                        )
                        continue
                return True
        return False

    def _brake_active(self) -> bool:
        """Check + auto-expire the global emergency brake. Must be called
        under self.lock. Bounded TTL means a forgotten brake never freezes
        deferrable dispatch indefinitely."""
        if self.brake_expires_at is None:
            return False
        if time.time() >= self.brake_expires_at:
            self.brake_expires_at = None
            self.brake_reason = None
            return False
        return True

    def _emit_release_event(self, job_id: str, reason: str, waited_seconds: float) -> None:
        """Structured release event — always emitted, never silent (Council:
        "release signaling stands firm"). Log-level/informational by design
        (anti-thundering-herd, anti-notification-storm for M2M/background
        consumers); more prominent surfacing for an affected interactive
        consumer is a follow-on wrapper concern, not this mechanism's job.
        Release is never gated on an acknowledgment. Must be called under
        self.lock (in-process log/file I/O only, no blocking network calls).
        """
        log.info(
            f"[{self.node_name}] defer_release job_id={job_id} reason={reason} "
            f"waited_seconds={waited_seconds:.1f}"
        )
        _write_idle_log(
            self.node_name, "defer_release", len(self.leases),
            job_id=job_id, reason=reason, waited_seconds=round(waited_seconds, 2),
        )

    def _sweep_wait_list(self) -> list[dict]:
        """Advance the pending-defer wait-list: detect newly-releasable entries,
        finalize jittered releases, and emit release events. Must be called
        under self.lock — this is the "dispatch-layer check" the starvation
        guard is anchored to (gw-router-phase1-foreground-gate spec): the
        max-hold-timeout is computed from each entry's own `enqueued_at` and
        is never reset by a newly-arrived `protected` lease, even across a
        recursive protected-lease chain.

        Returns the list of entries released this call (each a dict with
        job_id/reason/waited_seconds), for callers that want to react
        immediately (e.g. the acquire endpoint completing a wait for its own
        work_id). The background refresh thread calls this too, purely for
        its event-emission side effect, on every tick — release is never
        gated on anyone polling for it.
        """
        now = time.time()
        brake_gated = self._brake_active()
        released: list[dict] = []
        for wid, entry in list(self.wait_list.items()):
            elapsed = now - entry["enqueued_at"]
            timed_out = elapsed >= DOORMAN_MAX_HOLD_TIMEOUT_SEC
            entry_principal = entry.get("principal", GHOST_PRINCIPAL)
            gated = brake_gated or self._protected_lease_active(entry_principal)
            releasable = timed_out or not gated

            if not releasable:
                # Re-gated before a scheduled protected-cleared release finalized:
                # cancel it. The max-hold-timeout anchor is untouched by this —
                # once timed_out flips True it can never flip back, so a
                # timeout-triggered release is never revocable (the iron rod).
                if entry.get("_release_reason") == "protected-cleared":
                    entry.pop("_release_at", None)
                    entry.pop("_release_reason", None)
                remaining = DOORMAN_MAX_HOLD_TIMEOUT_SEC - elapsed
                if remaining <= DOORMAN_PENDING_RELEASE_WARN_SEC and not entry.get("_warned"):
                    entry["_warned"] = True
                    log.info(
                        f"[{self.node_name}] pending-release-soon job_id={wid} "
                        f"in ~{remaining:.0f}s (max-hold-timeout)"
                    )
                continue

            reason = "max-hold-timeout" if timed_out else "protected-cleared"
            if "_release_at" not in entry or entry.get("_release_reason") != reason:
                entry["_release_at"] = now + random.uniform(0, DOORMAN_RELEASE_JITTER_MAX_SEC)
                entry["_release_reason"] = reason
            if now >= entry["_release_at"]:
                del self.wait_list[wid]
                waited = now - entry["enqueued_at"]
                self._emit_release_event(wid, entry["_release_reason"], waited)
                released.append({"job_id": wid, "reason": entry["_release_reason"], "waited_seconds": waited})
        return released

    def acquire_or_defer(self, work_id: str, reason: str, role: str, lease_class: str, principal: str | None = None) -> tuple[dict | None, dict | None]:
        """The dispatch-layer defer-check for a `deferrable`-class acquire.

        Must be called under self.lock. `protected`-class acquires are never
        gated (AC2) — this is a no-op for them, always (None, None).

        For `deferrable`, first advances the wait-list (_sweep_wait_list) so a
        release that becomes due exactly at this call is picked up immediately
        rather than waiting for the next background tick — this call site IS
        one of the "dispatch-layer check" points the starvation guard is
        anchored to. Then:
          - if work_id is (still) on the wait-list → return a pending_defer
            response dict (job must keep waiting), (dict, None).
          - if work_id was just released by the sweep above → (None, release_info)
            so the caller can attach waited_seconds/release_reason to the
            eventual "serving" response.
          - if neither active-protected/brake gating applies (fresh request,
            never enqueued) → (None, None), proceed immediately.
          - otherwise (freshly gated) → enqueue with enqueued_at=now (the
            anchor) and return a pending_defer response, (dict, None).
        """
        if lease_class != "deferrable":
            return None, None

        released = self._sweep_wait_list()
        release_info = next((r for r in released if r["job_id"] == work_id), None)

        if work_id in self.wait_list:
            entry = self.wait_list[work_id]
            return {
                "status": "pending_defer",
                "work_id": work_id,
                "class": lease_class,
                "enqueued_at": entry["enqueued_at"],
                "waited_seconds": time.time() - entry["enqueued_at"],
            }, None

        if release_info is not None:
            return None, release_info

        if self._protected_lease_active(principal) or self._brake_active():
            enqueued_at = time.time()
            self.wait_list[work_id] = {
                "enqueued_at": enqueued_at, "reason": reason, "role": role,
                "principal": principal if principal is not None else GHOST_PRINCIPAL,
            }
            # One warning-level line at the enqueue point (defer/wait-list path
            # only; grants stay visible via the idle-log). Added after the
            # 2026-08-24 berth incident, where the true admission-defer
            # mechanism was invisible in the logs and had to be reconstructed
            # from idle-log resumed/idle_start pairs (shaper-swarm-payload-
            # carry-v0, item 2). Logging only — no decision-logic change.
            gating = self._gating_protected_lease(principal)
            log.warning(
                f"[{self.node_name}] enqueue-defer work_id={work_id} "
                f"principal={principal if principal is not None else GHOST_PRINCIPAL} "
                f"gating_lease={gating['work_id'] if gating else 'brake'} "
                f"gating_principal={gating['principal'] if gating else 'n/a'} "
                f"wait_list_depth={len(self.wait_list)}"
            )
            return {
                "status": "pending_defer",
                "work_id": work_id,
                "class": lease_class,
                "enqueued_at": enqueued_at,
                "waited_seconds": 0.0,
            }, None

        return None, None

    # ------------------------------------------------------------------
    # ensure_serving — serializes wakes via self.wake_lock, not self.lock
    # ------------------------------------------------------------------

    def ensure_serving(self, role: str | None = None, mode: str | None = None, work_id: str | None = None) -> bool | object:
        """Wake GW if needed, start the serving unit, and wait until it serves.

        Returns True on success, DEFERRED if controller owns the mode, False on failure.
        Acquires self.wake_lock internally for the duration of the wake-gravitywell /
        gw-serve subprocess + poll loop — this serializes concurrent wake attempts so
        only one wake-gravitywell subprocess runs at a time, without requiring the
        caller to hold self.lock (which would otherwise freeze every other endpoint
        for this node for the whole wake). self.lock is taken only briefly, internally,
        around each shared-state read/write (controller-lease checks, last_error,
        _cached_serving, service_stopped, last_wake_at).

        Args:
          role: optional role of the caller (e.g., "mode-controller" for flip-controller).
                If role=="mode-controller", this is the controller's own acquire and
                short-circuits to DEFERRED without needing a pre-registered lease.
          mode: optional requested serve mode ("big" or "dual"; already validated against
                VALID_SERVE_MODES by the caller before this method runs — never validated
                here). Only acted on when role=="mode-controller" and no FOREIGN
                mode-controller lease is active (agents-core-doorman-mode-bearing-
                acquire-v0, AC2/AC3/AC3a): then this commands gw-serve <mode> instead of
                deferring, overriding DOORMAN_DEFAULT_SERVE_MODE. Omitted (None) is
                byte-identical to the pre-mode-bearing-acquire behavior (AC1). Ignored
                when role != "mode-controller" (AC2a).
          work_id: the caller's own work_id, used only for the identity-aware foreign-
                   controller check (AC3a) when mode is supplied and role=="mode-controller".

        Flow (gravitywell-doorman-clean-stop-v0 + doorman-mode-deference-v0):
          0. Mode-aware deference (HOLE 1 fix, flag ON only): if controller owns the
             mode, return DEFERRED immediately — before _is_serving() or wake-gravitywell.
             This prevents wrong-model leases when a controller-owned swarm is up.
             Mode-controller's own acquire skips this check and always proceeds.
          1. Fast-path: _is_serving() → return True (service already up).
          2. wake-gravitywell: idempotent host-wake (no-op if already up).
          3. Check deference: if DOORMAN_DEFER_TO_CONTROLLER and (role=="mode-controller"
             or an active mode-controller lease exists), return DEFERRED (no wake issued) —
             UNLESS this is a mode-bearing controller acquire (role=="mode-controller" and
             mode is supplied) and no FOREIGN controller lease is active, in which case it
             commands gw-serve <mode> instead (AC2).
          4. Issue gw-serve ${DOORMAN_DEFAULT_SERVE_MODE} (gw-doorman-wake-to-default-mode-v0),
             or gw-serve <mode> for a mode-bearing controller acquire (step 3 above):
             "big" (_wake_big) — start llama-server.service if stopped (idempotent),
             synchronous ~60s subprocess, poll until serving or GW_WAKE_DEADLINE_SEC.
             "dual" (_wake_dual, default) — async-initiate gw-serve dual (fast-returning
             backgrounded launch; the ~488s Devstral cold-init happens off the subprocess),
             then poll both slots until GW_DUAL_WAKE_DEADLINE_SEC.
        """
        mode = mode or None  # AC1/AC4a: empty string is omission, defensively re-normalized here too
        with self.wake_lock:
            # Block co-load if creative 70B holds the GPU lane
            if self._is_creative_serving():
                return CREATIVE_OCCUPIED

            # HOLE 1 fix (AC2): mode-aware deference before _is_serving() fast path.
            # Worker acquires return DEFERRED immediately when the controller owns the mode,
            # even when _is_serving() would return True (avoids wrong-model leases on a live swarm).
            # Mode-controller's own acquire (role='mode-controller') skips this and always proceeds.
            if DOORMAN_MODE_AWARE_ADMISSION and DOORMAN_DEFER_TO_CONTROLLER:
                with self.lock:
                    controller_owns = self._controller_lease_active()
                if role != "mode-controller" and controller_owns:
                    log.info(
                        f"[{self.node_name}] mode-aware: controller owns mode — "
                        f"deferring before is_serving check (role={role!r})"
                    )
                    return DEFERRED

            # Mode-bearing controller acquire (agents-core-doorman-mode-bearing-
            # acquire-v0, AC2, extended by agents-core-doorman-warm-box-mode-
            # convergence-v0): a caller that BOTH claims mode-controller AND
            # supplies a valid mode wants that posture commanded, not deferred
            # to — and (this unit) not silently accepted via the _is_serving()
            # fast path either when the warm box is in a different posture.
            # mode is already validated (VALID_SERVE_MODES) by the caller
            # before this method runs.
            mode_bearing = role == "mode-controller" and mode is not None

            # Fast path: already awake and serving.
            if self._is_serving():
                if mode_bearing:
                    # AC5: resolve topology directly on this path — never from
                    # _cached_topology_state, never gated on
                    # DOORMAN_MODE_AWARE_ADMISSION (that flag defaults off and
                    # would leave this resolution permanently "unknown").
                    resolved_posture = self._resolve_live_posture()
                    if resolved_posture == "unknown":
                        # AC4/AC4a: an unconfident read never triggers a flip —
                        # take the fast path, but record it as a structured,
                        # machine-parseable WARN so this isn't indistinguishable
                        # from the silent-success defect this unit fixes.
                        log.warning(json.dumps({
                            "reason": "TOPOLOGY_UNRESOLVED_SKIP",
                            "requested_mode": mode,
                            "resolved_posture": "unknown",
                            "action": "fast_path_taken",
                        }))
                    elif resolved_posture != mode:
                        # AC2: warm box serving a confidently-resolved, different
                        # posture — bypass the fast path and converge it.
                        # AC7: the foreign-controller guard still applies, warm
                        # or cold.
                        with self.lock:
                            foreign_controller_owns = self._foreign_controller_lease_active(work_id)
                        if foreign_controller_owns:
                            log.info(
                                f"[{self.node_name}] mode-bearing controller acquire deferring — "
                                f"foreign controller owns mode (work_id={work_id!r})"
                            )
                            return DEFERRED
                        # AC8: legible warm-convergence log line.
                        log.info(
                            f"[{self.node_name}] warm-box convergence: requested mode={mode!r}, "
                            f"resolved posture={resolved_posture!r} — bypassing fast path, "
                            f"commanding gw-serve {mode}"
                        )
                        with self.lock:
                            self.last_error = None
                            self.service_stopped = False
                        return self._wake_big() if mode == "big" else self._wake_dual()
                    # else resolved_posture == mode: AC3, already the requested
                    # posture — fall through to the idempotent True below. No
                    # gw-serve is issued.
                with self.lock:
                    self.last_error = None
                    self.service_stopped = False
                return True

            # Flash-next window guard (agents-core-doorman-flashnext-handover-
            # v0, D4) — the wake collision guard. Placed IMMEDIATELY before the
            # wake-gravitywell subprocess, after the fast path and the
            # deference / mode-bearing returns: normal-day acquires (27B up)
            # return inside the fast path and pay ZERO added probes; only the
            # already-down path (rare) pays one sequential :30000 probe. The
            # :8081 half of the window test is the fast path's own fresh
            # _is_serving() fallthrough above (read milliseconds earlier) —
            # the guard issues ONLY the fresh seat probe, never a second
            # :8081 GET. The probe is FRESH (never the 45s cache): this is the
            # only path that would issue a wake, so the guard must not act on
            # a stale read (Invariant 4).
            seat_state, seat_served_id, _seat_registered, seat_error_class = (
                self._probe_flashnext_seat(sequential=True)
            )
            if seat_state in ("up_registered", "up_unverified"):
                # The seat holds (or is loading onto) GPU 0 whole-card at
                # --mem-fraction-static 0.985: waking the 27B here would put
                # two whole-card occupants on one card = OOM, one of them the
                # drafting session mid-flight. Refuse with the sentinel; no
                # wake-gravitywell, no _wake_*, no lease, idle_since untouched.
                # Applies to ALL acquires, including role=mode-controller —
                # during a confirmed window the window guard supersedes the
                # controller-deference machinery (mirrors CREATIVE_OCCUPIED,
                # which refuses even the controller's own acquire).
                log.info(
                    f"[{self.node_name}] flashnext-window-holding-gpu0 — seat probe "
                    f"{seat_state} (served_id={seat_served_id!r}); refusing wake, "
                    f"no lease registered (FLASHNEXT_OCCUPIED)"
                )
                return FLASHNEXT_OCCUPIED
            if seat_state == "blind":
                # BLIND proceed is TODAY's behavior per D2 (the B2 contract:
                # a missing probe is blindness, never False; and a BRIX->GW
                # partition breaks the wake's own ssh first) — but the blind
                # pass is a log line, not silence (gate binding constraint):
                # the structured warning carries the ACTUAL probe error class
                # (returned by the probe, not a static placeholder).
                # (Deliberate asymmetry with the flip-controller guard's
                # fail-closed BLIND: Invariant 13.)
                log.warning(
                    f"[{self.node_name}] flashnext-seat-probe-blind — :30000 probe "
                    f"indeterminate (error_class={seat_error_class}), "
                    f"proceeding with wake exactly as today"
                )
            elif seat_state == "up_foreign":
                # An identifiable non-seat occupant on the seat port: the
                # window is none (never guess — the probe already WARNed at
                # the probe site with the served id), so the wake proceeds.
                log.warning(
                    f"[{self.node_name}] flashnext-seat-foreign — :30000 probe "
                    f"up_foreign (served_id={seat_served_id!r}); no window, "
                    f"wake proceeds"
                )

            log.info(f"[{self.node_name}] GW not serving — running wake-gravitywell")
            try:
                proc = subprocess.run(
                    ["wake-gravitywell", "doorman-acquire"],
                    capture_output=True, text=True, timeout=60,
                )
                if proc.returncode != 0:
                    err = f"wake-gravitywell failed rc={proc.returncode}: {proc.stderr[:300]}"
                    log.error(f"[{self.node_name}] {err}")
                    with self.lock:
                        self.last_error = err
                    return False
            except Exception as e:
                err = f"wake-gravitywell subprocess error: {e}"
                log.error(f"[{self.node_name}] {err}")
                with self.lock:
                    self.last_error = err
                return False

            # AC6: a supplied mode is honoured independent of
            # DOORMAN_DEFER_TO_CONTROLLER — a requested mode is a property of
            # the acquire, not conditional on the deference policy. AC3/AC3a: a
            # FOREIGN controller lease still wins unconditionally — identity-
            # aware, so a controller's own repeat acquire never defers to itself.
            if mode_bearing:
                with self.lock:
                    foreign_controller_owns = self._foreign_controller_lease_active(work_id)
                if foreign_controller_owns:
                    log.info(
                        f"[{self.node_name}] mode-bearing controller acquire deferring — "
                        f"foreign controller owns mode (work_id={work_id!r})"
                    )
                    return DEFERRED
                log.info(
                    f"[{self.node_name}] mode-bearing controller acquire — "
                    f"commanding gw-serve {mode} (overrides DOORMAN_DEFAULT_SERVE_MODE)"
                )
                return self._wake_big() if mode == "big" else self._wake_dual()

            # Deference guard: if controller owns the mode, don't issue gw-serve big
            if DOORMAN_DEFER_TO_CONTROLLER:
                with self.lock:
                    controller_owns = self._controller_lease_active()

                if role == "mode-controller" or controller_owns:
                    log.info(
                        f"[{self.node_name}] GW not serving but controller owns mode — "
                        f"deferring (no gw-serve big)"
                    )
                    return DEFERRED

            # Resolve, then dispatch, the cold-wake posture (agents-core-doorman-
            # wake-honors-declared-posture-v0, Part 1). The prior two-way branch
            # here fought any declared home posture that wasn't its own literal
            # value — the resolution ladder replaces that with: declared posture
            # (Part 2) > DOORMAN_DEFAULT_SERVE_MODE as the last-resort literal
            # (unchanged fallback semantics — this precedence keeps "big" byte-
            # identical to the pre-dual-default behavior when nothing else
            # resolves). "big"/"dual" dispatch to the unchanged wake functions;
            # any other posture name goes through the generic check-then-act
            # path (Part 1b/A1/A2) — never a blind guess.
            posture = self._resolve_cold_wake_posture()
            if posture == "big":
                return self._wake_big()
            if posture == "dual":
                return self._wake_dual()
            return self._wake_generic_posture(posture)

    def _wake_big(self) -> bool:
        """Synchronous gw-serve big wake — unchanged timings (~60s subprocess,
        poll until GW_WAKE_DEADLINE_SEC). Must be called from ensure_serving()
        while holding self.wake_lock, after wake-gravitywell + deference checks.
        Shared node state is written under a brief self.lock acquisition at
        each write site, not for the duration of the subprocess/poll loop.
        """
        # Ensure the serving unit is up (idempotent — fast no-op if already active)
        log.info(f"[{self.node_name}] running gw-serve big to ensure llama-server.service is up")
        try:
            proc = subprocess.run(
                ["ssh", "gravitywell", "gw-serve big"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                err = (
                    f"gw-serve big failed rc={proc.returncode}: {proc.stderr[:300]}"
                )
                log.error(f"[{self.node_name}] {err}")
                with self.lock:
                    self.last_error = err
                return False
        except Exception as e:
            err = f"gw-serve big subprocess error: {e}"
            log.error(f"[{self.node_name}] {err}")
            with self.lock:
                self.last_error = err
            return False

        # Poll /health until serving or deadline (covers ~25s cold-load)
        deadline = time.time() + GW_WAKE_DEADLINE_SEC
        poll_interval = 3.0
        while time.time() < deadline:
            if self._is_serving():
                elapsed = GW_WAKE_DEADLINE_SEC - (deadline - time.time())
                log.info(f"[{self.node_name}] GW serving after ~{elapsed:.0f}s")
                with self.lock:
                    self.last_wake_at = time.time()
                    self.last_error = None
                    self.service_stopped = False
                    self._cached_serving = True
                    self._serving_checked_at = time.time()
                    # D4: a successful serve resets the restore-failure
                    # streak (the episode is over; the next one pages
                    # again).
                    self.restore_failure_streak.record_success()
                self._place_hold()
                return True
            time.sleep(poll_interval)

        err = f"GW did not serve within {GW_WAKE_DEADLINE_SEC}s after wake"
        log.error(f"[{self.node_name}] {err}")
        with self.lock:
            self.last_error = err
        return False

    def _wake_dual(self) -> bool:
        """Async-initiate gw-serve dual, then poll both slots with backoff.

        gw-serve dual (-> gw-dual up) blocks ~488s for the staggered health-gated
        bring-up (Devstral dense-FP8 init on Slot 2; Slot 1's 27B is fast). Running
        that as one long blocking subprocess would conflict with any short subprocess
        timeout, so instead this fires the launch backgrounded on the remote host
        (fast-returning ssh call) and treats this method's own poll loop as the single
        source of wake-completion truth, bounded by GW_DUAL_WAKE_DEADLINE_SEC.

        gw-dual has its own staggered health-gating + rollback-to-big watchdog on the
        host side — this method does not duplicate that logic, it only detects (a) a
        failed *launch* (caught immediately, distinct from a merely slow init) and
        (b) deadline exhaustion (transient — caller retries/degrades). Must be called
        from ensure_serving() while holding self.wake_lock, after wake-gravitywell +
        deference checks. Shared node state is written under a brief self.lock
        acquisition at each write site, not for the duration of the poll loop.
        """
        log.info(f"[{self.node_name}] issuing async-initiated gw-serve dual")
        try:
            proc = subprocess.run(
                ["ssh", "gravitywell",
                 "nohup gw-serve dual </dev/null >/tmp/gw-serve-dual-wake.log 2>&1 & disown"],
                capture_output=True, text=True, timeout=GW_DUAL_INITIATE_TIMEOUT_SEC,
            )
            if proc.returncode != 0:
                err = f"gw-serve dual initiation failed rc={proc.returncode}: {proc.stderr[:300]}"
                log.error(f"[{self.node_name}] {err}")
                with self.lock:
                    self.last_error = err
                self._cleanup_failed_dual_initiation()
                return False
        except Exception as e:
            err = f"gw-serve dual initiation subprocess error: {e}"
            log.error(f"[{self.node_name}] {err}")
            with self.lock:
                self.last_error = err
            self._cleanup_failed_dual_initiation()
            return False

        # Poll both slots until both show two consecutive healthy 200s, or deadline.
        deadline = time.time() + GW_DUAL_WAKE_DEADLINE_SEC
        poll_interval = GW_DUAL_POLL_INITIAL_SEC
        slot1_streak = 0
        slot2_streak = 0
        while time.time() < deadline:
            slot1_streak = slot1_streak + 1 if self._is_serving() else 0
            slot2_streak = slot2_streak + 1 if self._is_slot2_serving() else 0
            if slot1_streak >= 2 and slot2_streak >= 2:
                elapsed = GW_DUAL_WAKE_DEADLINE_SEC - (deadline - time.time())
                log.info(f"[{self.node_name}] dual serving (both slots) after ~{elapsed:.0f}s")
                with self.lock:
                    self.last_wake_at = time.time()
                    self.last_error = None
                    self.service_stopped = False
                    self._cached_serving = True
                    self._serving_checked_at = time.time()
                    # D4: a successful serve resets the restore-failure
                    # streak (the episode is over; the next one pages
                    # again).
                    self.restore_failure_streak.record_success()
                self._place_hold()
                return True
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * GW_DUAL_POLL_BACKOFF_FACTOR, GW_DUAL_POLL_MAX_SEC)

        # Deadline exhausted — transient (caller retries/degrades via wake_failed).
        # Genuine Slot-2 failure is gw-dual's own rollback-to-big watchdog's concern;
        # this is just the correctly-sized deadline observing it didn't reach dual.
        err = (
            f"GW dual did not reach both-slot readiness within "
            f"{GW_DUAL_WAKE_DEADLINE_SEC}s after wake "
            f"(slot1_ready={slot1_streak >= 2}, slot2_ready={slot2_streak >= 2})"
        )
        log.error(f"[{self.node_name}] {err}")
        with self.lock:
            self.last_error = err
        return False

    def _is_slot2_serving(self, timeout: float = 3.0) -> bool:
        """Slot 2 readiness — ENGINE-AWARE (gw-gpu1-berth-standing-seat-v0,
        leg 2, Fix 2 / port hazard (b)).

        The berth (NInfer Qwen3.8-27B) sits on the SAME host port as Slot 2
        (:8082, GW_SLOT2_PORT). NInfer's own /health answers 200 on :8082, so
        a bare health check would declare "dual ready" against the WRONG
        engine (the false dual-readiness hazard). Slot 2 is ready iff
        :8082/health is 200 AND :8082/v1/models contains a model with
        owned_by == "vllm" (the vLLM seat's shape). A 200 with a non-vLLM
        payload (the berth) = NOT ready — the poll keeps waiting (the
        existing giveup/retry machinery bounds the window; a late wake is the
        safe direction, a false-ready is not).
        """
        try:
            resp = requests.get(f"{self._slot2_url()}/health", timeout=timeout)
            if resp.status_code != 200:
                return False
            models_resp = requests.get(
                f"{self._slot2_url()}/v1/models", timeout=timeout
            )
            if models_resp.status_code != 200:
                return False
            payload = models_resp.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                return False
            return any(
                isinstance(m, dict) and m.get("owned_by") == "vllm"
                for m in models
            )
        except Exception:
            return False

    def _slot2_url(self) -> str:
        """Derive Slot 2's base URL from the configured Slot 1 gw_url — same host,
        GW_SLOT2_PORT instead of Slot 1's port. Not a newly hardcoded IP."""
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(self.gw_url)
        netloc = f"{parts.hostname}:{GW_SLOT2_PORT}"
        return urlunsplit((parts.scheme, netloc, "", "", ""))

    # ------------------------------------------------------------------
    # Declared-posture resolution + generic wake dispatch
    # (agents-core-doorman-wake-honors-declared-posture-v0)
    # ------------------------------------------------------------------

    def _resolve_cold_wake_posture(self) -> str:
        """Part 1 resolution ladder for the cold-wake fallback path, reached
        only after ensure_serving()'s mode-bearing-acquire precedence (line
        ~1076, unchanged, still highest precedence) and controller-deference
        checks have already been resolved:

          1. The declared home posture (GW_HOME_MODE in the live
             conductor.env — Part 2). Consumed, not forked.
          2. DOORMAN_DEFAULT_SERVE_MODE, the last-resort literal — reached
             only when the declaration itself is unreadable (file missing,
             unparseable, or no GW_HOME_MODE key). Unchanged semantics.

        Never guesses: an unreadable declaration degrades loudly to the
        literal (logged inside _read_declared_home_posture), it never
        substitutes a different posture silently.
        """
        declared = self._read_declared_home_posture()
        if declared is not None:
            return declared
        return DOORMAN_DEFAULT_SERVE_MODE

    def _read_declared_home_posture(self) -> str | None:
        """Read GW_HOME_MODE from the live conductor.env (Part 2) — the
        single source of the declared home posture; this method consumes
        it, it does not restate or fork the declaration.

        conductor.env is host-local and NOT sourced into the doorman's own
        process environment (see GW_HOME_MODE_ENV_PATH's comment), so this
        reads the file directly rather than via os.environ.

        Returns None — an "unreadable declaration" — when the file is
        missing/unreadable or carries no GW_HOME_MODE key, logging a loud
        WAKE_DEGRADED WARN. This is deliberately NOT a refusal (A6): an
        unreadable declaration is a weaker signal than a missing actuator,
        and mirrors _topology_serving_mode's existing degrade-to-"unknown"
        contract rather than the check-then-act refusal path used once a
        posture name IS in hand (see _wake_generic_posture).
        """
        try:
            text = Path(GW_HOME_MODE_ENV_PATH).read_text(encoding="utf-8")
        except OSError as e:
            log.warning(
                f'[{self.node_name}] WAKE_DEGRADED reason=POSTURE_UNDECLARED '
                f'detail="conductor.env unreadable at {GW_HOME_MODE_ENV_PATH}: {e}"'
            )
            return None

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.strip() == "GW_HOME_MODE":
                posture = value.strip().strip('"').strip("'")
                if posture:
                    return posture
                break

        log.warning(
            f'[{self.node_name}] WAKE_DEGRADED reason=POSTURE_UNDECLARED '
            f'detail="GW_HOME_MODE not set in {GW_HOME_MODE_ENV_PATH}"'
        )
        return None

    def _record_restore_failure(self, record: str, reason: str, detail: str) -> None:
        """D4 (attestation-contract-v0, leg 1): record a restore failure
        (WAKE_REFUSED / WAKE_FAILED) in the streak counter.

        Lock discipline: the streak mutation happens under self.lock, but
        the page's network send (send_notification -> Pushover POST)
        happens OUTSIDE the lock - the page seam is invoked after the
        `with self.lock` block. Holding self.lock across a network send
        would stall every lease-registry mutation and the refresh thread
        for the duration of a slow/failing Pushover delivery.

        The 09-08 loop ran ~40s then went quiet while the seat stayed
        down - the refusal recorded last_error and paged nothing. The
        streak counter pages HIGH once per episode after 3 consecutive
        failures within 10 minutes (I5: no repetition while the streak
        persists; reset on any successful serve/restore).
        """
        with self.lock:
            self.last_error = record
            crossed = self.restore_failure_streak.record_failure(
                reason, detail, _should_page=False,
            )
            # The page decision + content snapshot is taken UNDER the lock
            # (the streak accounting stays atomic); the send itself runs
            # after the block (see the docstring).
            _streak = self.restore_failure_streak.consecutive
            _last_ts = self.restore_failure_streak.last_ts
        if crossed:
            # The page send runs OUTSIDE self.lock (the lock is already
            # released above - the page's network call must never hold
            # self.lock; the content values were snapshotted under it).
            _maybe_page_restore_failure(
                reason=reason, detail=detail,
                streak=_streak,
                last_ts=_last_ts,
                seat_id=self.node_name,
            )
            log.warning(
                f"[{self.node_name}] D4: restore-failure streak crossed "
                f"the page threshold (see the HIGH page)"
            )

    def _refuse_wake(self, reason: str, detail: str) -> bool:
        """Emit a WAKE_REFUSED record (A6) and set last_error to the same
        structured shape, so the refusal category is readable from /status
        without parsing prose. Refused BEFORE any wake command was issued —
        zero side effects (A1/A2) — and must never be confused with
        _fail_wake's attempted-and-failed shape."""
        record = f'WAKE_REFUSED reason={reason} detail="{detail}"'
        log.error(f"[{self.node_name}] {record}")
        self._record_restore_failure(record, reason, detail)
        return False

    def _fail_wake(self, reason: str, detail: str) -> bool:
        """Emit a WAKE_FAILED record (A6) for an attempted-then-failed
        outcome (TopologyReachBusy/TopologyReachFailed) — a command WAS
        issued. Distinct record shape from _refuse_wake so the Ruling-1
        distinction ("refused before acting" vs "attempted and timed out")
        survives in logs and last_error."""
        record = f'WAKE_FAILED reason={reason} detail="{detail}"'
        log.error(f"[{self.node_name}] {record}")
        # D4: an attempted-then-failed wake is the same restore-failure
        # class (the seat was commanded and did not come up). The page
        # send runs OUTSIDE self.lock (see _record_restore_failure).
        self._record_restore_failure(record, reason, detail)
        return False

    def _wake_generic_posture(self, posture_name: str) -> bool:
        """Cold-wake dispatch for any declared posture other than "big"/
        "dual" (Part 1's third branch). Calls gw_topology.reach() — which
        already implements check-then-act with zero side effects on refusal
        (A1/A2) — rather than re-implementing actuability checking here.
        Must be called from ensure_serving() while holding self.wake_lock.

        Import form is load-bearing (A3): the doorman runs with
        PYTHONPATH=/srv/agents, NOT /srv/agents/scripts, so it must use
        `from scripts import gw_topology`, never the bare `import
        gw_topology` (which raises ModuleNotFoundError under the live
        service's PYTHONPATH — see the poison-pill test). An unimportable
        gw_topology is a REFUSAL (Erah ruling, A3), never a fallback to
        DOORMAN_DEFAULT_SERVE_MODE — this method does not call
        _resolve_cold_wake_posture or touch that constant. force_unproven
        is never set from this cold-wake path (A2).
        """
        try:
            from scripts import gw_topology
        except ImportError as e:
            # The eye is broken — cannot determine what is actuable at all.
            return self._refuse_wake("ACTUATOR_UNAVAILABLE", repr(e))

        try:
            topology = gw_topology.load_topology()
        except Exception as e:
            # Same blindness class as an unimportable module: the registry
            # itself could not be read/parsed, so what's actuable can't be
            # determined either. Refuse, don't guess.
            return self._refuse_wake("ACTUATOR_UNAVAILABLE", repr(e))

        # reach() is the single source of truth on whether posture_name is
        # even a real registered posture (TopologyUnknown) — deliberately
        # NOT pre-checked here via topology.topologies[posture_name], which
        # would just be a second, redundant place that same fact could go
        # stale or be gotten wrong. Slot ports are only read out afterward,
        # once reach() has confirmed the posture is real and actuable.
        try:
            gw_topology.reach(posture_name, topology=topology, ssh_host="gravitywell")
        except (gw_topology.TopologyUnknown, gw_topology.TopologyNotProven,
                gw_topology.TopologyOverCeiling, gw_topology.ForceUnprovenReasonRequired) as e:
            # The road is barred — reach() answered, this specific posture
            # is refused. Zero side effects per reach()'s own contract.
            return self._refuse_wake("POSTURE_INVALID", str(e))
        except gw_topology.PairingApplyError as e:
            # Live-config drift (or apply refused/failed): the posture is
            # VALID; the live /etc/default/gw-dual drifted or the apply
            # refused. reach() raised pre-apply or during apply - per
            # PairingApplyError's own contract the live file is left
            # untouched in every case. This is the confirmed 2026-08-30
            # incident (finding/night-dag-20260830-partial-source-gw-drift-
            # doorman-500-2026-08-30): an uncaught PairingApplyError here
            # propagated out of the handler as an unhandled 500, which the
            # runner read as reason class UNHANDLED - not in the retry
            # allowlist - and silently skipped the night's GW work with no
            # operator-facing reason. A distinct token (not
            # POSTURE_INVALID) so the operator reads "reconcile the seat
            # config", not "posture invalid". Deliberately NOT retried by
            # the runner (never enters the allowlist): a hand-edit drift
            # does not self-heal.
            return self._refuse_wake("DRIFT_REFUSED", str(e))
        except gw_topology.TopologyHelperIncompatible as e:
            # Deployed convergence helper self-hash mismatch (pre-host-touch,
            # zero side effects - the check runs before reach() touches the
            # host). Same 500 class as the drift case if left uncaught
            # (verified reachable through the same reach() call); the
            # operator remediation is re-deploying the helper, so the token
            # says so. Never retried by the runner (not in the allowlist):
            # a helper hash mismatch does not self-heal.
            return self._refuse_wake("HELPER_INCOMPATIBLE", str(e))
        except gw_topology.TopologyReachBusy as e:
            return self._fail_wake("REACH_BUSY", str(e))
        except gw_topology.TopologyReachFailed as e:
            return self._fail_wake("REACH_FAILED", str(e))

        # reach() already confirmed posture_name is a real, registered entry
        # (it would have raised TopologyUnknown above otherwise), so this is
        # a plain read of the same declaration, not a second actuability
        # check — nothing here should ordinarily raise.
        ports = self._posture_slot_ports(topology, posture_name)
        return self._poll_posture_ready(ports)

    def _posture_slot_ports(self, topology, posture_name: str) -> list[int]:
        """The ports the resolved posture's own composition declares —
        read from the same `topologies:` index reach() resolves against
        (Part 2: consume the declaration, never re-derive via GW_SLOT2_PORT
        arithmetic). A mode-kind entry's slots carry their ports directly
        (see gw-topology.yaml's `slot1-solo`); a pairing-kind entry's slots
        are slot1/slot2. Ports are de-duped, order-preserving; a slot with
        no `port` field (e.g. `big`'s single llama-cpp slot) contributes
        nothing here — `big` never reaches this method, it dispatches to
        the unchanged _wake_big() instead."""
        entry = topology.topologies[posture_name]
        if entry["kind"] == "mode":
            slots = topology.modes[entry["ref"]]["slots"]
        else:
            pairing = topology.pairings[entry["ref"]]
            slots = [pairing["slot1"], pairing["slot2"]]

        ports: list[int] = []
        for slot in slots:
            port = slot.get("port")
            if port is not None and int(port) not in ports:
                ports.append(int(port))
        return ports

    def _is_port_serving(self, port: int, timeout: float = 3.0) -> bool:
        """Health-probe an arbitrary GW-hosted port, same host as gw_url."""
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(self.gw_url)
        netloc = f"{parts.hostname}:{port}"
        url = urlunsplit((parts.scheme, netloc, "", "", ""))
        try:
            resp = requests.get(f"{url}/health", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def _poll_posture_ready(self, ports: list[int]) -> bool:
        """Poll exactly the ports the resolved posture declares until each
        shows two consecutive healthy /health 200s, or
        GW_DUAL_WAKE_DEADLINE_SEC elapses — the posture-driven readiness
        gate (Part 1, the doorman_server.py:1190-1204 false-negative this
        unit removes). A single-slot posture is satisfied by that one port
        and never waits on a port it doesn't declare."""
        if not ports:
            # Nothing declared to poll (schema-only/live-composition entry)
            # — reach() already converged synchronously, so there is
            # nothing left to wait on.
            with self.lock:
                self.last_wake_at = time.time()
                self.last_error = None
                self.service_stopped = False
                self._cached_serving = True
                self._serving_checked_at = time.time()
                # D4: a successful serve resets the restore-failure
                # streak (the episode is over; the next one pages again).
                self.restore_failure_streak.record_success()
            self._place_hold()
            return True

        deadline = time.time() + GW_DUAL_WAKE_DEADLINE_SEC
        poll_interval = GW_DUAL_POLL_INITIAL_SEC
        streaks = {p: 0 for p in ports}
        while time.time() < deadline:
            for p in ports:
                streaks[p] = streaks[p] + 1 if self._is_port_serving(p) else 0
            if all(streaks[p] >= 2 for p in ports):
                elapsed = GW_DUAL_WAKE_DEADLINE_SEC - (deadline - time.time())
                log.info(f"[{self.node_name}] posture serving (ports={ports}) after ~{elapsed:.0f}s")
                with self.lock:
                    self.last_wake_at = time.time()
                    self.last_error = None
                    self.service_stopped = False
                    self._cached_serving = True
                    self._serving_checked_at = time.time()
                    # D4: a successful serve resets the restore-failure
                    # streak (the episode is over; the next one pages
                    # again).
                    self.restore_failure_streak.record_success()
                self._place_hold()
                return True
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * GW_DUAL_POLL_BACKOFF_FACTOR, GW_DUAL_POLL_MAX_SEC)

        ready = {p: streaks[p] >= 2 for p in ports}
        err = (
            f"GW posture did not reach readiness on ports {ports} within "
            f"{GW_DUAL_WAKE_DEADLINE_SEC}s after wake (ready={ready})"
        )
        log.error(f"[{self.node_name}] {err}")
        with self.lock:
            self.last_error = err
        return False

    def _cleanup_failed_dual_initiation(self) -> None:
        """Best-effort cleanup after a failed dual-mode launch, so a launch error
        never leaves an orphaned gw-dual/vllm process or half-state behind. Reuses
        the existing gw-serve stop verb (already safe/idempotent per
        gravitywell-doorman-clean-stop-v0) rather than inventing a new host-script
        surface. Errors are swallowed — this is a safety net, not the primary path.

        LEASE GATE (gw-gpu1-berth-standing-seat-v0, leg 2, Fix 1 — critique
        mandatory refinement b): the cleanup fired `gw-serve stop`
        UNCONDITIONALLY. With DOORMAN_DEFAULT_SERVE_MODE=dual (live), a failed
        dual launch while a supervisor lease is held would stop ALL seats
        mid-job. The _worker_lease_blockers guard (the :2407-2434 pattern) now
        blocks the stop when an active role="worker" lease exists; a blocked
        cleanup logs and leaves the half-initialized wake to the existing
        giveup/reconciler path.
        """
        with self.lock:
            self._gc_stale()
            blockers = self._worker_lease_blockers()
        if blockers:
            log.warning(
                f"[{self.node_name}] dual-initiation cleanup: BLOCKED by "
                f"{len(blockers)} active worker lease(s) — leaving the "
                f"half-initialized wake to the giveup/reconciler path"
            )
            return
        try:
            subprocess.run(
                ["ssh", "gravitywell", "gw-serve stop"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as e:
            log.warning(f"[{self.node_name}] dual-initiation cleanup (gw-serve stop) failed: {e}")

    # ------------------------------------------------------------------
    # Keepawake hold helpers — must be called under lock
    # ------------------------------------------------------------------

    def _place_hold(self) -> None:
        try:
            subprocess.run(
                ["ssh", "gravitywell",
                 f"gw-keepawake hold {HOLD_NAME} {GW_HOLD_TTL_SEC} doorman-active"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as e:
            log.warning(f"[{self.node_name}] gw-keepawake hold failed: {e}")

    def _release_hold(self) -> None:
        try:
            subprocess.run(
                ["ssh", "gravitywell", f"gw-keepawake release {HOLD_NAME}"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as e:
            log.warning(f"[{self.node_name}] gw-keepawake release failed: {e}")

    # ------------------------------------------------------------------
    # Lease operations — must be called under lock
    # ------------------------------------------------------------------

    def _gc_stale(self) -> list[str]:
        """Remove expired leases. Returns list of GC'd work_ids.

        Emits an orphan-reclaim scar event if a mode-controller lease is evicted.
        If GC empties the lease set, records idle_since the same way release_lease()
        does on an explicit release -- otherwise the dwell-stop timer never starts
        for leases that expire via TTL rather than an explicit /lease/release call.
        """
        now = time.time()
        expired = [
            wid for wid, info in self.leases.items()
            if now > info["acquired_at"] + info["ttl_sec"]
        ]
        for wid in expired:
            info = self.leases[wid]
            log.info(f"[{self.node_name}] GC stale lease work_id={wid}")
            # Emit scar if a mode-controller lease is being evicted
            if info.get("role") == "mode-controller":
                log.warning(
                    f"[{self.node_name}] mode-controller lease TTL-expired (not released); "
                    f"deference lapsed; legacy wake/serve will auto-recover"
                )
                acquired_at = info.get("acquired_at")
                last_renewed_iso = (
                    datetime.datetime.utcfromtimestamp(acquired_at).isoformat() + "Z"
                    if acquired_at else None
                )
                _write_idle_log(
                    self.node_name,
                    "controller-orphan-reclaim",
                    len(self.leases) - 1,  # count before deletion
                    evicted_lease=wid,
                    evicted_role="mode-controller",
                    last_renewed=last_renewed_iso,
                    ttl_sec=info.get("ttl_sec"),
                    detail="mode-controller lease TTL-expired (not released); deference lapsed; legacy wake/serve",
                )
            del self.leases[wid]
        if expired and not self.leases and self.idle_since is None:
            self.idle_since = time.time()
            self._idle_since_source = "lease"
            _write_idle_log(self.node_name, "idle_start", 0)
        return expired

    def acquire_lease(self, work_id: str, ttl_sec: int, reason: str, role: str = "worker", principal: str | None = None, require_drain_clear: bool = False, lease_kind: str = "inference", lease_class: str = DEFAULT_LEASE_CLASS, mode: str | None = None) -> bool | object:
        """Try to ensure GW is serving, then register the lease.

        Returns True on success, DEFERRED if a foreign caller acquires during controller
        ownership (no lease registered), CONTENDED if require_drain_clear=True and another
        principal's worker lease is active (lease not registered), False on failure.

        Args:
          role: optional role descriptor (default "worker"). E.g., "mode-controller"
                for the flip-controller's keepawake lease. Stored on the lease dict
                for later ownership checks.
          principal: logical admission group for drain-gate exclusion. Worker leases
                     without a principal are stamped GHOST_PRINCIPAL — always counted,
                     never excluded, emits critical log when counted in a drain decision.
                     Non-worker leases are drain-gate-exempt; principal is ignored.
          require_drain_clear: when True and role=="worker", atomically checks for
                               other-principal worker leases before registering this one.
                               Returns CONTENDED without registering if any exist.
                               Defaults False — all existing callers are unchanged.
          lease_kind: discriminator for admission-contention counting. "inference" (default)
                      — the lease holds GPU inference and serializes via the drain-gate.
                      "coordination" — the lease holds no inference (span/keepawake); excluded
                      from drain-gate contention count but still counted by /v0/drain-count
                      for flip-protection. Omitting is byte-identical to "inference".
          lease_class: foreground-priority gate class (gw-router-phase1-foreground-gate):
                      "protected" (never deferred) or "deferrable" (default — yields
                      to an active protected lease/brake via acquire_or_defer, called
                      by the endpoint before this method). Caller (the endpoint) is
                      responsible for the defer-check; this method only stamps the
                      lease with its class.
          mode: optional requested serve mode, already validated by the endpoint
                (agents-core-doorman-mode-bearing-acquire-v0). Passed through to
                ensure_serving(); see that method for when it's acted on vs. ignored.

        self.lock is taken exactly once per call, for the short bookkeeping that spans
        idle-tracking reset, the atomic drain-gate check, and lease registration (see the
        comment on that block below) — NOT across ensure_serving(), which can block for
        minutes on a cold wake. ensure_serving() serializes concurrent wakes internally via
        its own wake_lock, so other node state (reads via /status, /v0/drain-count,
        /lease/release, etc.) stays responsive while a wake is in flight.
        """
        # ensure_serving serializes concurrent wakes internally via its own wake_lock —
        # this call intentionally runs without self.lock held, so a cold wake never freezes
        # the bookkeeping critical section below for other callers.
        ok = self.ensure_serving(role=role, mode=mode, work_id=work_id)
        if ok is CREATIVE_OCCUPIED:
            return CREATIVE_OCCUPIED
        if ok is FLASHNEXT_OCCUPIED:
            # Propagated exactly as CREATIVE_OCCUPIED (D4): the flash-next
            # window guard refused the wake — no lease registered, no hold
            # placed, idle_since untouched. Uniform for all acquires,
            # including role=mode-controller.
            return FLASHNEXT_OCCUPIED
        if ok is DEFERRED:
            # Controller's own acquire (role="mode-controller") registers the lease and hold
            # even though ensure_serving returns DEFERRED (no gw-serve big was issued).
            # Foreign acquires during controller ownership don't register a lease.
            if role == "mode-controller":
                with self.lock:
                    self.leases[work_id] = {
                        "acquired_at": time.time(),
                        "ttl_sec": ttl_sec,
                        "reason": reason,
                        "role": role,
                    }
                    self._place_hold()
                return DEFERRED  # still return DEFERRED so endpoint knows not to issue gw-serve big
            else:
                # Foreign caller during controller ownership — return deferred, no lease
                return DEFERRED
        if not ok:
            return False

        # Single critical section (agents-core-doorman-acquire-lease-nonreentrant-deadlock-v0):
        # idle-tracking reset, the atomic drain-gate check (AC3), and lease registration all
        # happen under ONE self.lock acquisition — previously these were three separate
        # lock/unlock cycles, which left a window where another thread's acquire_lease could
        # interleave between idle-reset and the drain-gate check (idle_since feeds GW's
        # idle-suspend logic elsewhere, so a torn read there isn't cosmetic) and, separately,
        # left the drain-gate check-then-register itself non-atomic despite the intent. This
        # merge closes both gaps in one pass and is also what makes the non-reentrant self.lock
        # safe to acquire in a single call — no caller (production or test) may hold self.lock
        # externally around acquire_lease().
        with self.lock:
            was_idle = self.idle_since is not None
            self.idle_since = None
            self._idle_since_source = None
            lease_count_for_log = len(self.leases)

            contended = False
            if require_drain_clear and role == "worker":
                effective_principal = principal if principal is not None else GHOST_PRINCIPAL
                for _wid, _info in self.leases.items():
                    if _info.get("role") != "worker":
                        continue
                    # Coordination leases hold no inference — not a drain-gate contender (AC2).
                    # /v0/drain-count still counts them for flip-protection (unchanged, AC4).
                    if _info.get("lease_kind", "inference") == "coordination":
                        continue
                    _p = _info.get("principal", GHOST_PRINCIPAL)
                    if _p == GHOST_PRINCIPAL:
                        # Ghost leases always count as contending; never silently excluded (AC7).
                        log.critical(
                            "[doorman] drain_count ghost_lease_counted work_id=%s - "
                            "role=worker lease has no principal; add principal= to "
                            "acquire() call to prevent drain-gate freeze",
                            _wid,
                        )
                        contended = True
                        break
                    if _p != effective_principal:
                        contended = True
                        break

            if not contended:
                lease_entry: dict = {
                    "acquired_at": time.time(),
                    "ttl_sec": ttl_sec,
                    "reason": reason,
                    "role": role,
                    "lease_kind": lease_kind,
                    "class": lease_class,
                }
                if role == "worker":
                    lease_entry["principal"] = principal if principal is not None else GHOST_PRINCIPAL
                self.leases[work_id] = lease_entry
                self._place_hold()

        if was_idle:
            _write_idle_log(self.node_name, "resumed", lease_count_for_log)
        return CONTENDED if contended else True

    def _decide_idle_stop(self) -> bool:
        """The per-node idle-stop decision of the refresh loop (rev 3,
        agents-core-doorman-flashnext-handover-v0): extracted VERBATIM from
        _start_refresh_thread._loop's `if not state.leases:` block — same
        order, same behavior, no second lock. The caller runs with
        self.lock ALREADY held (the refresh loop's critical section); this
        method takes no lock of its own, so D9's re-anchor remains a single
        atomic read-modify-write under state.lock (gate binding
        constraint): read window_closed_at / idle_since / leases / the
        same-tick cached serving, and write idle_since + clear
        window_closed_at, in the same critical section.

        Returns True when the refresh loop must `continue` to the next node
        (a D3 window hold, the topology_unknown_no_park alarm, a
        blind-bounded pause, or the trailing idle-node skip); False when the
        loop falls through to the keepawake-hold refresh (only when
        idle_since is None or a stop is already in flight).

        Stop-failure backoff: this method sets self._stop_failed_this_tick
        on its two stop-failure outcomes (the rc!=0-still-serving real
        failure and the generic exception handler), and the refresh loop
        bumps its `backoff` variable (backoff = min(backoff + 15,
        GW_HOLD_REFRESH_SEC)) when it sees that flag set — the same
        accelerated retry cadence origin/main applied inline in the loop.
        The flag is reset at the top of every call, so the loop observes it
        exactly once per failing tick.
        """
        self._stop_failed_this_tick = False
        # Flash-next window (agents-core-doorman-flashnext-
        # handover-v0, D3): the keeper now KNOWS the nature of
        # the room — the :30000 seat holds GPU 0 whole-card
        # and the 27B is down by design. Never park, never
        # stop, never alarm: the 2026-08-19 ruling is
        # satisfied by knowledge, not by continued refusal.
        # NOT gated by DOORMAN_MODE_AWARE_ADMISSION (D3/
        # Invariant 9): the window is a property of the seat,
        # not of the admission policy — a flag-off doorman
        # must still not park the card the seat holds.
        if self._flashnext_window == "active":
            _write_idle_log(
                self.node_name, "card_held_flashnext", 0,
                idle_secs=(
                    time.time() - self.idle_since
                    if self.idle_since is not None else None
                ),
                served_id=self._flashnext_served_id,
            )
            return True

        # Flash-next close re-anchor (D9): the stop path's
        # grace clock reads idle_since, which the last
        # pre-window lease release (or TTL-GC) set at window
        # open and nothing refreshes during the window.
        # Without a re-anchor, the first tick after close
        # with the restored 27B serving would compute
        # idle_elapsed = the full window duration >=
        # GW_STOP_GRACE_SEC and confirmed-idle-park the seat
        # the launcher just cold-restored. Window-close +
        # restored serving is an operator-intent activity
        # boundary (the same class as the lease-acquire and
        # probe-activity re-anchors): the restored seat
        # receives a full fresh GW_STOP_GRACE_SEC from the
        # doorman's first observation of it serving after
        # the window. Single atomic read-modify-write under
        # state.lock (gate binding constraint): read
        # window_closed_at / idle_since / leases / the
        # same-tick cached serving, and write idle_since +
        # clear window_closed_at, in the same critical
        # section (the caller already holds the lock; no
        # second lock). One-shot (the flag is
        # consumed); fires only on the close transition — a
        # normal wake (no window in the history) is
        # byte-identical to today, and a lease acquired
        # between close and first-serving-observation already
        # owns the clock (acquire sets idle_since = None).
        # NOT gated by DOORMAN_MODE_AWARE_ADMISSION
        # (Invariant 9).
        if (
            self._flashnext_window_closed_at is not None
            and self.idle_since is not None
            and not self.leases
            and self._cached_serving is True
        ):
            _write_idle_log(
                self.node_name, "flashnext_window_closed", 0,
                idle_secs=time.time() - self.idle_since,
            )
            self.idle_since = time.time()
            self._idle_since_source = "window_close"
            self._flashnext_window_closed_at = None

        if (
            self.idle_since is not None
            and not self.service_stopped
            # R5: a manual force-stop already owns the single
            # writer for this node — never launch a second
            # gw-serve stop, and never block on state.lock
            # waiting for it (we already hold the lock; simply
            # skip issuing our own this tick).
            and not self._stop_in_flight
        ):
            idle_elapsed = time.time() - self.idle_since

            # Flash-next idle-awareness (gw-doorman-flashnext-
            # idle-awareness-v0, D1): the stop decision consumes the
            # existing seat-state probe (the D7 probe pass, unmodified).
            # The 2026-09-22 GPU1-wedge incident: the idle-eject fired
            # while the day seat was UP-and-idle and the live brain was
            # the flash-next seat on :30000 — the D3 window guard
            # withholds only the handover case (seat up + day seat
            # DOWN); the overlap case (seat up + day seat up-idle) left
            # the window "none", the guard inert, and the eject enabled
            # a guard-correct suspend that killed the unsupervised seat.
            # Partition (D1):
            #   seat UP (up_registered / up_unverified): withhold
            #   UNCONDITIONALLY — regardless of request activity (a live
            #   run with a >600s request gap must not become stop-
            #   eligible; the gap is exactly the incident's 292k-run
            #   shape). The D3 window guard already withholds unbounded
            #   in the handover case; this extends the same knowledge-
            #   based withhold to the overlap case.
            #   seat BLIND (or cold/None — COLD START, D1): withhold,
            #   BOUNDED, mirroring the vLLM-axis bound below: a
            #   while blind and idle_elapsed < GW_STOP_GRACE_SEC +
            #   DOORMAN_PROBE_BLINDNESS_SEC the grace clock pauses
            #   (idle-log row, no verb) — the bound is measured on
            #   idle_elapsed (the grace clock), exactly like the
            #   vLLM-axis precedent below, NOT on the continuous-
            #   blindness duration: the two clocks coincide only when
            #   the seat went blind at idle_start, and the spec's
            #   grace-pause semantics bind on idle_elapsed. The
            #   continuous-blindness clock (armed on the first blind
            #   read since the last definitive read, cleared on a
            #   definitive read — bookkeeping in the tick's probe
            #   pass, the probe method stays stateless) is DIAGNOSTIC
            #   ONLY: it feeds the blind_secs field of the
            #   flashnext_blind_hold idle-log row and the blind-
            #   duration text of the withhold warning / bound-exceeded
            #   CRITICAL. It is NOT a decision input anywhere. COLD
            #   START (state None, never-observed seat) is withheld by
            #   this same branch and bounded by the SAME idle_elapsed
            #   gate below — a cold start is withheld while idle_since
            #   is unanchored (None) because the stop block above never
            #   runs, not because of any reading of the blindness clock.
            #   Past the bound this axis stops withholding and, where
            #   it is the deciding factor, the stop proceeds with a
            #   distinct stop_reason
            #   (flashnext_probe_blind_bound_exceeded) at CRITICAL.
            #   DOORMAN_PROBE_BLINDNESS_SEC == 0 is the sentinel for
            #   UNBOUNDED (withhold on blind forever — the operator
            #   chooses burning fuel over risking the seat).
            #   seat DOWN or UP_FOREIGN: definitive; this axis does not
            #   withhold (the vLLM axis decides, as today).
            # PRECEDENCE: the stop proceeds only when every withhold
            # condition has fallen through — this block (D1), the
            # topology-unknown no-park, and the vLLM blind-within-bound
            # below. A withheld tick writes no stopped idle-log row and
            # computes no stop_reason for a stop that does not happen.
            # The activity clock (D2) is legibility-only — it selects the
            # substate (withheld-active vs withheld-up-idle) and renders
            # on /status; it is NEVER consumed by the stop decision.
            _fn_state = self._flashnext_state
            if _fn_state in ("up_registered", "up_unverified"):
                _fn_last_activity = self._flashnext_last_activity_ts
                _fn_active = (
                    _fn_last_activity is not None
                    and (time.time() - _fn_last_activity) < GW_STOP_GRACE_SEC
                )
                _write_idle_log(
                    self.node_name,
                    "flashnext_withheld_active" if _fn_active
                    else "flashnext_withheld_up_idle",
                    0,
                    idle_secs=idle_elapsed,
                    seat_state=_fn_state,
                    last_activity_ts=_fn_last_activity,
                )
                # D3 journal: one-shot WARNING on the transition INTO a new
                # withhold substate; a steady-state withhold (the same
                # substate as the previous tick) writes idle-log rows only.
                _fn_substate = (
                    f"up:{_fn_state}:{'active' if _fn_active else 'idle'}"
                )
                if self._flashnext_withhold_substate != _fn_substate:
                    self._flashnext_withhold_substate = _fn_substate
                    log.warning(
                        f"[{self.node_name}] flashnext-seat-up — :30000 "
                        f"seat {_fn_state} (served_id="
                        f"{self._flashnext_served_id!r}); withholding "
                        f"gw-serve stop (idle {idle_elapsed:.0f}s). The "
                        f"seat is the live brain; the stop path must not "
                        f"take a visible seat out of reach."
                    )
                return True
            if _fn_state == "blind" or _fn_state is None:
                # BLIND (or COLD START: state None before the first
                # definitive read — D1: treat as blind, never down/idle-
                # ok). The bound is measured on idle_elapsed (the grace
                # clock — the spec's grace-pause semantics and the
                # vLLM-axis precedent below), NOT on the continuous-
                # blindness duration: the two clocks coincide only when
                # the seat went blind at idle_start; if the seat goes
                # blind after idle has already accumulated, the
                # idle_elapsed bound releases the withhold at the same
                # absolute idle age regardless of when blindness began
                # (matching the vLLM axis, which also keys on
                # idle_elapsed). The blindness clock
                # (_flashnext_blind_since, armed in the tick's probe
                # pass) is DIAGNOSTIC bookkeeping only — it supplies
                # the blind_secs field of the flashnext_blind_hold
                # idle-log row and the "blind Ns" text of the withhold
                # warning, and is never consumed as a decision input.
                # A COLD START (state None) is withheld by this branch
                # and bounded by the SAME idle_elapsed gate below: the
                # cold-start bound is the idle_elapsed gate, not the
                # blindness clock.
                _fn_blind_elapsed = (
                    time.time() - self._flashnext_blind_since
                    if self._flashnext_blind_since is not None
                    else 0.0
                )
                if DOORMAN_PROBE_BLINDNESS_SEC != 0:
                    _fn_blind_bound = (
                        GW_STOP_GRACE_SEC + DOORMAN_PROBE_BLINDNESS_SEC
                    )
                    if idle_elapsed < _fn_blind_bound:
                        _write_idle_log(
                            self.node_name, "flashnext_blind_hold", 0,
                            idle_secs=idle_elapsed,
                            seat_state=_fn_state,
                            blind_secs=round(_fn_blind_elapsed, 2),
                            error_class=self._flashnext_error_class,
                        )
                        # D3 journal: one-shot on the transition into the
                        # blind withhold substate (cold/None is its own
                        # substate, so a cold start that later turns blind
                        # journals once more).
                        _fn_substate = "cold" if _fn_state is None else "blind"
                        if self._flashnext_withhold_substate != _fn_substate:
                            self._flashnext_withhold_substate = _fn_substate
                            log.warning(
                                f"[{self.node_name}] flashnext-seat-blind — "
                                f":30000 probe indeterminate (state="
                                f"{_fn_state}, error_class="
                                f"{self._flashnext_error_class}); "
                                f"withholding gw-serve stop (idle "
                                f"{idle_elapsed:.0f}s, blind "
                                f"{_fn_blind_elapsed:.0f}s < bound "
                                f"{_fn_blind_bound}s). Fail-closed on "
                                f"stop: the probe cannot distinguish a "
                                f"dead seat from an invisible one."
                            )
                        return True
                else:
                    # Sentinel 0: unbounded — withhold on blind forever
                    # (the operator chooses burning fuel over risking the
                    # seat).
                    _write_idle_log(
                        self.node_name, "flashnext_blind_hold", 0,
                        idle_secs=idle_elapsed,
                        seat_state=_fn_state,
                        blind_secs=round(_fn_blind_elapsed, 2),
                        error_class=self._flashnext_error_class,
                    )
                    # D3 journal: one-shot on the transition into the
                    # unbounded-blind withhold substate.
                    _fn_substate = (
                        "cold-unbounded" if _fn_state is None
                        else "blind-unbounded"
                    )
                    if self._flashnext_withhold_substate != _fn_substate:
                        self._flashnext_withhold_substate = _fn_substate
                        log.warning(
                            f"[{self.node_name}] flashnext-seat-blind — "
                            f":30000 probe indeterminate (state="
                            f"{_fn_state}, error_class="
                            f"{self._flashnext_error_class}); "
                            f"withholding gw-serve stop UNBOUNDED "
                            f"(DOORMAN_PROBE_BLINDNESS_SEC=0 sentinel; "
                            f"idle {idle_elapsed:.0f}s, blind "
                            f"{_fn_blind_elapsed:.0f}s)."
                        )
                    return True
                # Bound exceeded: this axis stops withholding. The stop
                # proceeds only if every other withhold has also fallen
                # through (topology-unknown no-park, vLLM blind-within-
                # bound below); where this axis is the deciding factor
                # the stop_reason is distinct (below) and the event logs
                # at CRITICAL (bootstrap basicConfig level=INFO —
                # CRITICAL always renders; no alert hook keys off log
                # level).
                log.critical(
                    f"[{self.node_name}] flashnext_probe_blind_bound_"
                    f"exceeded — :30000 probe blind for "
                    f"{_fn_blind_elapsed:.0f}s, idle "
                    f"{idle_elapsed:.0f}s >= bound "
                    f"{GW_STOP_GRACE_SEC + DOORMAN_PROBE_BLINDNESS_SEC}s "
                    f"(measured on the grace clock, like the vLLM axis); "
                    f"the blind-withhold no longer withholds the stop. "
                    f"A permanently broken probe must not pin the box "
                    f"awake forever."
                )
            # _fn_state in ("down", "up_foreign"): definitive — this axis
            # does not withhold; the vLLM axis decides, as today.

            # Unknown topology (Erah ruling 2026-08-19, agents-core-
            # doorman-class-aware-activity-probe-v0 D2/D3): a keeper
            # who does not know the nature of the room does not close
            # the door. Never park, never guess a class — grace clock
            # stays paused with NO bound (unlike the blindness bound
            # below), and a distinct alarm fires every tick instead of
            # a stop attempt, carrying the same diagnostic fields as
            # the topology_resolution_unknown warning. Only applies
            # when the flag is ON (_serving_is_big is never populated
            # otherwise, so flag-off transparently falls through to
            # the unchanged legacy path below).
            if DOORMAN_MODE_AWARE_ADMISSION and self._serving_is_big is None:
                topo = self._cached_topology_state
                _write_idle_log(
                    self.node_name, "topology_unknown_no_park", 0,
                    idle_secs=idle_elapsed,
                    reason="topology_unknown_no_park",
                    authority_gap=None if topo is None else topo.authority_gap,
                    models_answered=_topology_models_answered(topo),
                    served_ids=None if topo is None else topo.served_ids,
                    unknown_model=None if topo is None else topo.unknown_model,
                )
                log.warning(
                    f"[{self.node_name}] topology_unknown_no_park — idle "
                    f"{idle_elapsed:.0f}s but serving class unresolved; "
                    f"refusing to park, alarming instead"
                )
                return True

            blindness_deadline = (
                GW_STOP_GRACE_SEC + DOORMAN_PROBE_BLINDNESS_SEC
            )
            if self._probe_indeterminate and idle_elapsed < blindness_deadline:
                # Probe genuinely ambiguous this tick (gw-doorman-vllm-
                # activity-probe-v0) — pause the grace clock rather than
                # advance it, same as if leases were held. Bounded below.
                return True
            if self._probe_indeterminate:
                log.warning(
                    f"[{self.node_name}] probe blind for {idle_elapsed:.0f}s — "
                    f"proceeding on stale grace period"
                )
            if idle_elapsed >= GW_STOP_GRACE_SEC:
                # Idle-log reason (D3): distinguishes a genuinely
                # confirmed-idle park from one taken only because the
                # blindness bound was exceeded, so the journal line is
                # diagnosable rather than reading as one undifferentiated
                # "stopped" event. The flash-next idle-awareness axis
                # (gw-doorman-flashnext-idle-awareness-v0, D1) adds a
                # THIRD distinct reason: the seat probe was blind past
                # its bound and the vLLM axis was NOT the deciding factor
                # (vLLM confirmed idle) — the stop fired because the
                # seat's own probe went blind, not because the 27B-axis
                # probe did.
                # The flash-next axis is the deciding factor when it is
                # NOT the vLLM axis that blew its bound (the vLLM axis's
                # own bound-exceeded keeps its distinct reason, as
                # today); a COLD-START state (None, treated as blind per
                # D1) is the same deciding-factor class.
                if (
                    self._flashnext_state in ("blind", None)
                    and not self._probe_indeterminate
                ):
                    stop_reason = "flashnext_probe_blind_bound_exceeded"
                else:
                    stop_reason = (
                        "probe_blind_bound_exceeded"
                        if self._probe_indeterminate else "confirmed_idle"
                    )
                indeterminate_sources = [
                    name for name, v in (self._last_probe_raw or {}).items()
                    if v is None
                ]
                log.warning(
                    f"[{self.node_name}] idle {idle_elapsed:.0f}s >= grace "
                    f"{GW_STOP_GRACE_SEC}s — issuing gw-serve stop. "
                    f"Safety: guard blocks suspend while service active; "
                    f"doorman stop enables suspend, never forces it."
                )
                # Single-writer bookkeeping (R5), shared with
                # _force_stop: bumping the epoch and setting
                # _stop_in_flight here means a concurrent
                # manual force-stop sees this attempt and
                # defers to it, and — if this call times out
                # while GW is still shutting down — the next
                # tick will not re-issue a second gw-serve
                # stop for the same unload.
                self._stop_epoch += 1
                self._stop_in_flight = True
                self._stop_in_flight_since = time.time()
                try:
                    stop_proc = subprocess.run(
                        ["ssh", "gravitywell", "gw-serve stop"],
                        capture_output=True, text=True,
                        timeout=GW_SERVE_STOP_TIMEOUT_SEC,
                    )
                    if stop_proc.returncode == 0:
                        self.service_stopped = True
                        self.idle_since = None
                        self._idle_since_source = None
                        self._cached_serving = False
                        self._serving_checked_at = time.time()
                        self._stop_in_flight = False
                        self._stop_in_flight_since = None
                        stopped_desc = _describe_stopped_units(
                            stop_proc.stdout, self.node_name
                        )
                        log.warning(
                            f"[{self.node_name}] gw-serve stop succeeded — "
                            f"{stopped_desc}, host now "
                            f"suspend-eligible via guard"
                        )
                        _write_idle_log(
                            self.node_name, "stopped", 0,
                            idle_secs=idle_elapsed,
                            reason=stop_reason,
                            indeterminate_sources=indeterminate_sources,
                        )
                    else:
                        # rc != 0: idempotency guard — check if already down
                        if not self._is_serving():
                            # Already stopped — treat as success
                            self.service_stopped = True
                            self.idle_since = None
                            self._idle_since_source = None
                            self._cached_serving = False
                            self._serving_checked_at = time.time()
                            self._stop_in_flight = False
                            self._stop_in_flight_since = None
                            log.warning(
                                f"[{self.node_name}] gw-serve stop "
                                f"rc={stop_proc.returncode} but service "
                                f"already down — treating as success"
                            )
                            _write_idle_log(
                                self.node_name, "stopped", 0,
                                idle_secs=idle_elapsed,
                                reason=stop_reason,
                                indeterminate_sources=indeterminate_sources,
                            )
                        else:
                            # Real failure: still serving
                            err = (
                                f"gw-serve stop failed "
                                f"rc={stop_proc.returncode}: "
                                f"{stop_proc.stderr[:200]}"
                            )
                            log.error(f"[{self.node_name}] {err}")
                            self.last_error = err
                            self._stop_in_flight = False
                            self._stop_in_flight_since = None
                            # Stop-failure backoff signal for the refresh
                            # loop (origin/main's inline bump, restored on
                            # extraction): a real stop failure retries on
                            # the accelerated cadence.
                            self._stop_failed_this_tick = True
                            _write_idle_log(
                                self.node_name, "stop_failed", 0,
                                idle_secs=idle_elapsed,
                                indeterminate_sources=indeterminate_sources,
                            )
                except subprocess.TimeoutExpired:
                    # R2/R5: a timeout is not a failure —
                    # resolve by observation. Confirmed down
                    # -> treat as success; still serving ->
                    # leave service_stopped and
                    # _stop_in_flight alone (do not write
                    # stop_failed, do not re-issue next tick)
                    # so a later tick or the R8 reconciler
                    # resolves it.
                    if not self._is_serving():
                        self.service_stopped = True
                        self.idle_since = None
                        self._idle_since_source = None
                        self._cached_serving = False
                        self._serving_checked_at = time.time()
                        self._stop_in_flight = False
                        self._stop_in_flight_since = None
                        log.warning(
                            f"[{self.node_name}] gw-serve stop timed out "
                            f"after {GW_SERVE_STOP_TIMEOUT_SEC}s but "
                            f"service already down — treating as "
                            f"success"
                        )
                        _write_idle_log(
                            self.node_name, "stopped", 0,
                            idle_secs=idle_elapsed,
                            reason=stop_reason,
                            indeterminate_sources=indeterminate_sources,
                        )
                    else:
                        log.info(
                            f"[{self.node_name}] gw-serve stop still "
                            f"running after "
                            f"{GW_SERVE_STOP_TIMEOUT_SEC}s — "
                            f"deferring to next tick"
                        )
                except Exception as exc:
                    err = f"gw-serve stop exception: {exc}"
                    log.error(f"[{self.node_name}] {err}")
                    self.last_error = err
                    self._stop_in_flight = False
                    self._stop_in_flight_since = None
                    # Stop-failure backoff signal for the refresh loop
                    # (origin/main's inline bump, restored on extraction).
                    self._stop_failed_this_tick = True
                    _write_idle_log(self.node_name, "stop_failed", 0)
            return True

        return False

    def release_lease(self, work_id: str) -> None:
        """Drop a lease. If it was the last, record idle_since and release the hold."""
        self.leases.pop(work_id, None)
        self._gc_stale()
        if not self.leases:
            self.idle_since = time.time()
            self._idle_since_source = "lease"
            self._release_hold()
            _write_idle_log(self.node_name, "idle_start", 0)

    # ------------------------------------------------------------------
    # Status snapshot (for /status endpoint)
    # ------------------------------------------------------------------

    def status_snapshot(self) -> dict:
        with self.lock:
            self._gc_stale()
            self._sweep_wait_list()
            serving = self._cached_serving
            # Check if controller owns the mode
            controller_owns = self._controller_lease_active()
            # HOLE 2 fix (AC5): when flag ON, controller-lease check precedes serving-wins.
            # When flag OFF, preserve today's order (serving wins) for byte-identical behavior.
            if DOORMAN_MODE_AWARE_ADMISSION:
                if controller_owns:
                    serving_mode = "deferred"
                elif serving:
                    serving_mode = self._topology_serving_mode()
                elif self.service_stopped:
                    serving_mode = "stopped"
                else:
                    serving_mode = "unknown"
            else:
                if serving:
                    serving_mode = "big"
                elif controller_owns:
                    serving_mode = "deferred"
                elif self.service_stopped:
                    serving_mode = "stopped"
                else:
                    serving_mode = "unknown"
            # drain_count: worker leases only (mode-controller excluded), AC8
            drain_count = sum(
                1 for info in self.leases.values() if info.get("role") == "worker"
            )
            return {
                "serving": serving,
                "serving_mode": serving_mode,
                "serving_checked_at": self._serving_checked_at,
                "snapshot_mode": "cached",
                "service_stopped": self.service_stopped,
                "idle_since": self.idle_since,
                "lease_count": len(self.leases),
                "leases": [
                    {"work_id": wid, **info}
                    for wid, info in self.leases.items()
                ],
                "last_wake_at": self.last_wake_at,
                "last_error": self.last_error,
                "mode_owner": DOORMAN_CONTROLLER_NAME if controller_owns else None,
                "drain_count": drain_count,
                "worker_lease_count": drain_count,
                # stopped is a KNOWN absence (122B definitively not serving) — False,
                # not None. None is reserved for genuine uncertainty: pre-first-refresh
                # or a resolution failure (Council stand-aside, adopted).
                "serving_is_big": False if self.service_stopped else self._serving_is_big,
                "big_probe_state": self._big_probe_state,
                "creative_serving": self._cached_creative_serving,
                "probe_activity_enabled": DOORMAN_PROBE_LLAMA_ACTIVITY,
                "idle_since_source": self._idle_since_source,
                "protected_active": self._protected_lease_active(),
                "brake_active": self._brake_active(),
                "brake_reason": self.brake_reason,
                "brake_expires_at": self.brake_expires_at,
                "wait_list": [
                    {"job_id": wid, "enqueued_at": entry["enqueued_at"],
                     "waited_seconds": time.time() - entry["enqueued_at"]}
                    for wid, entry in self.wait_list.items()
                ],
                # A3: surface the generic-posture actuator's importability directly
                # in /status, so a broken deploy tree ("the eye is broken") is
                # visible without having to attempt a wake first. Cheap (no
                # execution beyond import machinery) and uses the exact same
                # import form the wake path itself uses (`from scripts import
                # gw_topology`) — a single source of truth about importability.
                "actuator_available": _gw_topology_importable(),
                # GPU 1 (berth) awareness block (gw-gpu1-berth-standing-seat-v0,
                # leg 2): the operator surface for "what does the doorman see on
                # the 3090?". glances=dead + last_vote=None makes the bounded
                # probe-blindness mapping OBSERVABLE (a dead Glances maps to the
                # existing grace+900=1500s blind bound, then a stop with
                # stop_reason="probe_blind_bound_exceeded" — NOT the
                # topology-unknown unbounded never-park path). glances_mem_pct
                # is DIAGNOSTIC ONLY (never part of the proc-only vote); the
                # calibrated +0.65 pp driver-overhead offset (T1c: 90.80% smi vs
                # 91.45% glances) is noted on the field.
                "gpu1": {
                    "berth_unit": (
                        "active" if self._gpu1_berth_unit is True
                        else "inactive" if self._gpu1_berth_unit is False
                        else None
                    ),
                    "seat_health": self._gpu1_seat_health,
                    "glances": (
                        "reachable"
                        if self._gpu1_glances_proc is not None
                        else "dead"
                    ),
                    "glances_mem_pct": self._gpu1_glances_mem_pct,
                    "glances_proc": self._gpu1_glances_proc,
                    "last_vote": self._last_probe_raw.get("GPU1"),
                },
                # Flash-next (:30000) seat awareness block (agents-core-
                # doorman-flashnext-handover-v0, D5): the operator surface
                # for "what does the doorman see on the whole-card seat?".
                # Reads ONLY cached fields (no network, lock-only — the gpu1
                # block's precedent). seat_state is the D2 probe state;
                # window is "active"|"none"|null (null = indeterminate — a
                # blind probe is blindness, never down). window_since is
                # epoch|null (set on the tick the window first reads active,
                # held while active, cleared on close); window_closed_at is
                # epoch|null (set on the active->none transition tick,
                # consumed by the D9 close re-anchor). serving_mode's enum
                # and the 27B-axis fields above are untouched by this block
                # (Invariant 5).
                "flashnext": {
                    "seat_health": (
                        None
                        if self._flashnext_state is None
                        or self._flashnext_state == "blind"
                        else self._flashnext_state
                        in ("up_registered", "up_unverified", "up_foreign")
                    ),
                    "seat_state": self._flashnext_state,
                    "served_id": self._flashnext_served_id,
                    "registered": self._flashnext_registered,
                    "window": self._flashnext_window,
                    "window_since": self._flashnext_window_since,
                    "window_closed_at": self._flashnext_window_closed_at,
                    "last_vote": self._flashnext_state,
                    # Flash-next idle-awareness (gw-doorman-flashnext-
                    # idle-awareness-v0, D3): the operator surface for
                    # "is the stop path withholding on the seat axis, and
                    # why?". last_activity_ts is the D2 legibility clock
                    # (probe-stamp time of the last successful activity
                    # read observing running/waiting >= 1; null = no
                    # observation — a 404 / non-200 / unparseable gauge is
                    # unknown, never idle). Deliberate deviation from the
                    # D2 spec's "a clock not refreshed this tick renders
                    # unknown" wording (confirmed at the gate, cycle-1
                    # review): the last stamp is carried forward — a
                    # stale/absent stamp renders withheld-up-idle, which
                    # is the safer direction (the seat-up withhold is
                    # unconditional regardless of the substate, so a
                    # stale stamp can never authorize a stop; rendering
                    # "unknown" would only lose the active/idle
                    # distinction the operator reads). eject_state is the
                    # stop-path substate: idle-ok (no withhold on the
                    # seat axis — the vLLM axis decides), withheld-active
                    # (seat up + activity within grace), withheld-up-idle
                    # (seat up, stamp stale/absent), withheld-blind (seat
                    # blind or cold, within bound — or unbounded under the
                    # DOORMAN_PROBE_BLINDNESS_SEC=0 sentinel). A withhold
                    # is a normal safety state, not a failure: this block
                    # never sets last_error (the 2026-09-13 cockpit wake-
                    # collapse precedent — last_error stays reserved for
                    # real errors).
                    "last_activity_ts": self._flashnext_last_activity_ts,
                    "eject_state": (
                        "withheld-active"
                        if self._flashnext_state
                        in ("up_registered", "up_unverified")
                        and self._flashnext_last_activity_ts is not None
                        and (time.time() - self._flashnext_last_activity_ts)
                        < GW_STOP_GRACE_SEC
                        else "withheld-up-idle"
                        if self._flashnext_state
                        in ("up_registered", "up_unverified")
                        else "withheld-blind"
                        if self._flashnext_state in ("blind", None)
                        else "idle-ok"
                    ),
                },
            }

    # ------------------------------------------------------------------
    # Manual force-stop (doorman-force-stop-endpoint-v0): bypasses
    # GW_STOP_GRACE_SEC entirely. Mirrors the refresh thread's grace-period
    # stop logic (idempotency guards, state updates) so behavior stays
    # consistent whichever path issues the stop. Manages self.lock
    # internally — the caller (the /v0/force-stop route) must NOT wrap the
    # call in `with state.lock:` (agents-core-doorman-force-stop-timeout-
    # truthfulness-v0, R1): the ssh subprocess runs outside the lock so
    # /status, /lease/acquire and the background loop stay responsive for
    # the full stop duration, guarded instead by the single-writer
    # _stop_in_flight flag (test-and-set under the lock before the
    # subprocess, re-checked under the lock after it returns).
    #
    # Lease guard (gw-force-stop-lease-guard-v0): refuses to stop while any
    # other worker lease is active, so one caller can't evict a model out
    # from under another consumer's in-flight lease. Re-checked after the
    # subprocess returns (R7) so a lease acquired while the ssh call was in
    # flight — which the pre-check could not have seen — is still honoured:
    # service_stopped is only ever set true at a moment when the lease
    # guard passes under the lock.
    # ------------------------------------------------------------------

    def _worker_lease_blockers(self, exclude_principal: str | None = None) -> list[dict]:
        """Active role="worker" leases that should block a drain/force-stop decision.

        Shared filter for drain_count_endpoint and _force_stop: ghost leases (no
        principal) always count; a lease matching exclude_principal (the caller's
        own admission group) is excluded. Caller must hold self.lock and have
        already called self._gc_stale().
        """
        blockers = []
        for wid, info in self.leases.items():
            if info.get("role") != "worker":
                continue
            p = info.get("principal", GHOST_PRINCIPAL)
            if p == GHOST_PRINCIPAL:
                # Ghost leases are always counted; emit critical log when counted in a drain decision
                if exclude_principal is not None:
                    log.critical(
                        "[doorman] drain_count ghost_lease_counted work_id=%s - "
                        "role=worker lease has no principal; add principal= to "
                        "acquire() call to prevent drain-gate freeze",
                        wid,
                    )
                blockers.append({"work_id": wid, "principal": p})
            elif exclude_principal is not None and p == exclude_principal:
                continue  # same admission group — exclude from the check
            else:
                blockers.append({"work_id": wid, "principal": p})
        return blockers

    def _force_stop(self, exclude_principal: str | None = None) -> dict:
        with self.lock:
            if self.service_stopped:
                return {"status": "already_stopped", "node": self.node_name}

            self._gc_stale()
            blocking_leases = self._worker_lease_blockers(exclude_principal)
            if blocking_leases:
                log.info(
                    f"[{self.node_name}] force-stop: blocked by {len(blocking_leases)} "
                    f"active worker lease(s)"
                )
                return {
                    "status": "blocked",
                    "node": self.node_name,
                    "active_leases": blocking_leases,
                }

            if self._stop_in_flight:
                # Single-writer (R1): a duplicate request must never launch a
                # parallel gw-serve stop.
                since = self._stop_in_flight_since or time.time()
                return {
                    "status": "stop_in_progress",
                    "node": self.node_name,
                    "waited_seconds": time.time() - since,
                }

            self._stop_epoch += 1
            my_epoch = self._stop_epoch
            self._stop_in_flight = True
            self._stop_in_flight_since = time.time()

        # Outside self.lock (R1): the ssh subprocess must never block
        # /status, /lease/acquire, or the background loop for the duration
        # of a stop. `resolved` tracks whether a path below has already
        # cleared _stop_in_flight (or determined it doesn't own it) so the
        # `finally` can defensively clear it on any unexpected exception
        # without wedging the endpoint (R1's try/finally requirement).
        resolved = False
        try:
            log.info(
                f"[{self.node_name}] force-stop: issuing gw-serve stop "
                f"(bypassing grace period)"
            )
            timed_out = False
            exc_err: str | None = None
            stop_proc = None
            try:
                stop_proc = subprocess.run(
                    ["ssh", "gravitywell", "gw-serve stop"],
                    capture_output=True, text=True, timeout=GW_SERVE_STOP_TIMEOUT_SEC,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
            except Exception as exc:
                exc_err = f"force-stop exception: {exc}"
                log.error(f"[{self.node_name}] {exc_err}")

            # R2: a timeout is not a failure — resolve by observation, not
            # assumption. Probe outside the lock (same precedent as
            # _refresh_serving_cache: "probing is a blocking network call").
            still_serving: bool | None = None
            if timed_out or (stop_proc is not None and stop_proc.returncode != 0):
                still_serving = self._is_serving()

            with self.lock:
                if self._stop_epoch != my_epoch:
                    # R8a: the reconciler (or a later stop) already resolved
                    # this attempt while we were outside the lock. We do not
                    # own it anymore — mutate nothing, clear nothing.
                    resolved = True
                    return {
                        "status": "stop_in_progress",
                        "node": self.node_name,
                        "waited_seconds": 0.0,
                    }

                if exc_err is not None:
                    self.last_error = exc_err
                    self._stop_in_flight = False
                    self._stop_in_flight_since = None
                    resolved = True
                    return {"status": "error", "node": self.node_name, "error": exc_err}

                if timed_out and still_serving:
                    # Still progressing — leave service_stopped and
                    # _stop_in_flight untouched so the background loop (or a
                    # later force-stop, or the R8 reconciler) reconciles it.
                    waited = time.time() - (self._stop_in_flight_since or time.time())
                    log.info(
                        f"[{self.node_name}] force-stop: gw-serve stop still "
                        f"running after {GW_SERVE_STOP_TIMEOUT_SEC}s — reporting "
                        f"stop_in_progress"
                    )
                    resolved = True  # deliberately not cleared — genuinely still in flight
                    return {
                        "status": "stop_in_progress",
                        "node": self.node_name,
                        "waited_seconds": waited,
                    }

                if timed_out:
                    # Timed out locally but confirmed down (R2): matches the
                    # rc==0 path exactly, "stopped" — not a truncation, this
                    # unload genuinely succeeded within the allowance its own
                    # unit was granted, just past our local budget.
                    exit_code = None
                    status = "stopped"
                elif stop_proc.returncode == 0:
                    exit_code = 0
                    status = "stopped"
                elif not still_serving:
                    # rc != 0: idempotency guard — already down.
                    exit_code = stop_proc.returncode
                    status = "already_stopped"
                else:
                    err = (
                        f"gw-serve stop failed rc={stop_proc.returncode}: "
                        f"{stop_proc.stderr[:200]}"
                    )
                    log.error(f"[{self.node_name}] {err}")
                    self.last_error = err
                    self._stop_in_flight = False
                    self._stop_in_flight_since = None
                    resolved = True
                    return {
                        "status": "error", "node": self.node_name, "error": err,
                        "exit_code": stop_proc.returncode,
                    }

                # R7: re-check the lease guard, under the lock, before ever
                # mutating service_stopped. A worker lease acquired while the
                # subprocess was in flight — invisible to the pre-check —
                # must still be honoured.
                self._gc_stale()
                post_blockers = self._worker_lease_blockers(exclude_principal)
                if post_blockers:
                    self._stop_in_flight = False
                    self._stop_in_flight_since = None
                    resolved = True
                    log.info(
                        f"[{self.node_name}] force-stop: blocked post-subprocess "
                        f"by {len(post_blockers)} active worker lease(s) acquired "
                        f"mid-stop"
                    )
                    return {
                        "status": "blocked",
                        "node": self.node_name,
                        "active_leases": post_blockers,
                    }

                self.service_stopped = True
                self.idle_since = None
                self._idle_since_source = None
                self._cached_serving = False
                self._serving_checked_at = time.time()
                self._stop_in_flight = False
                self._stop_in_flight_since = None
                resolved = True
                verb = "succeeded" if status == "stopped" else f"rc={exit_code} but service already down"
                log.info(f"[{self.node_name}] force-stop: gw-serve stop {verb}")
                _write_idle_log(self.node_name, "force_stopped", 0)
                result = {"status": status, "node": self.node_name}
                if exit_code is not None:
                    result["exit_code"] = exit_code
                return result
        finally:
            if not resolved:
                # An exception escaped somewhere above that no branch caught
                # (R1): never wedge the endpoint — clear the flag iff we
                # still own this attempt.
                with self.lock:
                    if self._stop_epoch == my_epoch:
                        self._stop_in_flight = False
                        self._stop_in_flight_since = None


# ---------------------------------------------------------------------------
# Background refresh thread
# ---------------------------------------------------------------------------

def _start_refresh_thread(nodes: dict[str, _NodeState]) -> threading.Thread:
    """Start the keepawake refresh + stale-lease GC + deferred stop background thread."""

    def _loop():
        backoff = 0.0
        first_iteration = True
        while True:
            if not first_iteration:
                time.sleep(max(GW_HOLD_REFRESH_SEC - backoff, GW_HOLD_REFRESH_SEC // 2))
            first_iteration = False
            backoff = 0.0
            for node_name, state in nodes.items():
                # Refresh serving cache OUTSIDE the lock (probing is a blocking network call).
                # Must never let an exception from a probe (e.g. _probe_slot_activity)
                # escape and kill this daemon thread.
                try:
                    state._refresh_serving_cache()
                except Exception as exc:
                    log.warning(
                        f"[{node_name}] _refresh_serving_cache() raised, skipping this tick: {exc}"
                    )
                with state.lock:
                    state._gc_stale()
                    # Foreground-priority gate (gw-router-phase1-foreground-gate):
                    # advance the pending-defer wait-list every tick so a release
                    # (protected-cleared or max-hold-timeout) and its structured
                    # event fire even if no caller happens to be polling — release
                    # is never gated on an acknowledgment.
                    state._sweep_wait_list()

                    # R8: bound stop_in_progress — a wedged stop (manual or
                    # automatic; the two paths share one writer) cannot report
                    # progress forever. One atomic critical section: reads
                    # service_stopped-adjacent state and, on give-up, mutates
                    # last_error + clears the flag, so no observer sees a
                    # half-applied giveup.
                    if state._stop_in_flight and state._stop_in_flight_since is not None:
                        in_flight_elapsed = time.time() - state._stop_in_flight_since
                        if in_flight_elapsed >= GW_SERVE_STOP_GIVEUP_SEC:
                            err = (
                                f"gw-serve stop unresolved after "
                                f"{in_flight_elapsed:.0f}s (budget "
                                f"{GW_SERVE_STOP_GIVEUP_SEC}s) — giving up"
                            )
                            log.error(f"[{node_name}] {err}")
                            state.last_error = err
                            state._stop_in_flight = False
                            state._stop_in_flight_since = None
                            # R8a: invalidate the owning thread's epoch token —
                            # a late-returning subprocess must not overwrite
                            # this failure with a stale result.
                            state._stop_epoch += 1
                            _write_idle_log(node_name, "stop_failed", 0)

                    if not state.leases:
                        # No active leases: check if deferred service stop is due.
                        # The per-node stop decision (D3 flash-next window
                        # check, D9 close re-anchor, and the legacy
                        # grace/park/stop machinery) is extracted VERBATIM
                        # into _NodeState._decide_idle_stop (behavior-
                        # preserving extraction, rev 3): the loop continues
                        # on its True return, exactly as the inline
                        # `continue` statements did, and the keepawake-hold
                        # refresh below is skipped for an idle node either
                        # way.
                        if state._decide_idle_stop():
                            # Stop-failure backoff (origin/main's inline
                            # bump, restored on extraction): the two stop-
                            # failure outcomes inside _decide_idle_stop set
                            # _stop_failed_this_tick, and the loop applies
                            # the accelerated retry cadence here — a failing
                            # auto gw-serve stop retries at
                            # min(backoff + 15, GW_HOLD_REFRESH_SEC)
                            # instead of the full GW_HOLD_REFRESH_SEC.
                            if state._stop_failed_this_tick:
                                backoff = min(backoff + 15, GW_HOLD_REFRESH_SEC)
                            continue  # no hold refresh needed for idle node
                        # _decide_idle_stop() returned False with no leases:
                        # this node is already stopped (service_stopped=True) or
                        # a stop is in flight, and is suspend-eligible. It must
                        # NOT refresh the keepawake hold -- doing so holds the
                        # box awake forever after a clean stop. (fix 2026-09-06:
                        # the rev-3 extraction returned False for the stopped
                        # case, which previously `continue`d inline.)
                        continue  # stopped idle node: skip keepawake hold refresh

                    # Leases are active: re-issue the keepawake hold to refresh its TTL
                    try:
                        proc = subprocess.run(
                            ["ssh", "gravitywell",
                             f"gw-keepawake hold {HOLD_NAME} {GW_HOLD_TTL_SEC} doorman-refresh"],
                            capture_output=True, text=True, timeout=15,
                        )
                        if proc.returncode != 0:
                            err = (f"keepawake refresh failed rc={proc.returncode}: "
                                   f"{proc.stderr[:200]}")
                            log.error(f"[{node_name}] {err}")
                            state.last_error = err
                            backoff = min(backoff + 15, GW_HOLD_REFRESH_SEC)
                        else:
                            log.debug(f"[{node_name}] keepawake hold refreshed")
                    except Exception as e:
                        err = f"keepawake refresh exception: {e}"
                        log.error(f"[{node_name}] {err}")
                        state.last_error = err
                        backoff = min(backoff + 15, GW_HOLD_REFRESH_SEC)

    t = threading.Thread(target=_loop, daemon=True, name="doorman-refresh")
    t.start()
    return t


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(gw_url: str | None = None) -> FastAPI:
    _gw_url = gw_url or os.environ.get("GW_URL", GW_URL_DEFAULT)

    nodes: dict[str, _NodeState] = {
        "gravitywell": _NodeState(_gw_url, node_name="gravitywell"),
    }

    # Start background refresh thread
    _start_refresh_thread(nodes)

    app = FastAPI(title="doorman-server", version="0")

    # ------------------------------------------------------------------
    # Bearer-token auth middleware
    # ------------------------------------------------------------------
    _token = os.environ.get("DOORMAN_BEARER_TOKEN", "")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if _token:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[len("Bearer "):] != _token:
                return JSONResponse(
                    status_code=401,
                    content=_error("unauthorized", "Missing or invalid bearer token"),
                )
        return await call_next(request)

    # ------------------------------------------------------------------
    # /healthz — doorman's own liveness
    # ------------------------------------------------------------------

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ------------------------------------------------------------------
    # /status — node state snapshot (future Unit 2 UI read surface)
    # ------------------------------------------------------------------

    @app.get("/status")
    def status():
        return {
            "nodes": {
                name: state.status_snapshot()
                for name, state in nodes.items()
            }
        }

    # ------------------------------------------------------------------
    # GET /v0/mode-owner — deference-liveness probe (doorman-mode-deference-v0)
    # ------------------------------------------------------------------

    @app.get("/v0/mode-owner")
    def mode_owner(node: str = "gravitywell"):
        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        with state.lock:
            state._gc_stale()
            owner_lease_held = state._controller_lease_active()
            # Compute owner_lease_age_sec and stale flag
            owner_lease_age_sec = None
            owner_lease_stale = False
            if owner_lease_held:
                now = time.time()
                for lease_info in state.leases.values():
                    if lease_info.get("role") == "mode-controller":
                        age_sec = now - lease_info["acquired_at"]
                        owner_lease_age_sec = age_sec
                        # Past renewal point (60% of TTL)
                        owner_lease_stale = age_sec > lease_info["ttl_sec"] * 0.6
                        break

        return {
            "node": node,
            "controller": DOORMAN_CONTROLLER_NAME,
            "active": DOORMAN_DEFER_TO_CONTROLLER,
            "owner_lease_held": owner_lease_held,
            "owner_lease_age_sec": owner_lease_age_sec,
            "owner_lease_stale": owner_lease_stale,
        }

    # ------------------------------------------------------------------
    # GET /v0/drain-count — in-flight worker-lease count for drain-gate (AC9)
    # ------------------------------------------------------------------

    @app.get("/v0/drain-count")
    def drain_count_endpoint(node: str = "gravitywell", exclude_principal: str | None = None):
        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        with state.lock:
            state._gc_stale()
            count = len(state._worker_lease_blockers(exclude_principal))
        return {"node": node, "drain_count": count}

    # ------------------------------------------------------------------
    # POST /lease/acquire
    # ------------------------------------------------------------------

    @app.post("/lease/acquire")
    def lease_acquire(body: dict[str, Any]):
        node = body.get("node", "")
        work_id = body.get("work_id", "")
        ttl_sec = int(body.get("ttl_sec", 300))
        reason = body.get("reason", "")
        role = body.get("role", "worker")
        principal = body.get("principal") or None  # empty string → None → ghost
        require_drain_clear = bool(body.get("require_drain_clear", False))
        lease_kind = body.get("lease_kind", "inference")
        lease_class = body.get("class", DEFAULT_LEASE_CLASS)  # missing → deferrable (safe)
        mode = body.get("mode") or None  # missing/empty string → None (AC1 — omission)

        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )
        if not work_id:
            return JSONResponse(
                status_code=400,
                content=_error("bad_request", "work_id is required"),
            )
        if lease_class not in LEASE_CLASSES:
            return JSONResponse(
                status_code=400,
                content=_error(
                    "bad_request",
                    f"invalid class {lease_class!r}; must be one of {LEASE_CLASSES}",
                ),
            )
        # AC4/AC6: validated before any lock is taken and before any subprocess
        # could be invoked — a bad mode never reaches gw-serve as a shell argument.
        if mode is not None and mode not in VALID_SERVE_MODES:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "invalid_mode",
                    "detail": f"mode {mode!r} not in accepted set {sorted(VALID_SERVE_MODES)}",
                },
            )

        state = nodes[node]

        # Capacity shadow (agents-core-doorman-capacity-shadow-v0): off-lock,
        # would-defer-path-only instrumentation. Behaviour below this block
        # is byte-identical to today regardless of the outcome here — nothing
        # here feeds acquire_or_defer's decision. Kill switch restores the
        # exact current code path.
        _capacity_shadow_ctx = None
        if lease_class == "deferrable" and DOORMAN_CAPACITY_SHADOW:
            if state._capacity_shadow_predict_defer(work_id, principal):
                _scrape = state.capacity_shadow_scrape()
                _capacity_shadow_ctx = {"scrape": _scrape, "scrape_done_at": time.time()}

        # Dispatch-layer defer-check (gw-router-phase1-foreground-gate): a
        # `deferrable` acquire yields while a `protected` lease or the brake is
        # active. `protected` acquires always return (None, None) here (AC2 —
        # never deferred). release_info is populated only when this call's own
        # wait-list entry finalized its release this call.
        with state.lock:
            state._gc_stale()
            pending_resp, release_info = state.acquire_or_defer(work_id, reason, role, lease_class, principal)
            if _capacity_shadow_ctx is not None:
                _lock_acquired_at = time.time()
                _actual_defer = pending_resp is not None
                _gating_lease = state._gating_protected_lease(principal) if _actual_defer else None
                state._emit_capacity_shadow_event(
                    work_id=work_id,
                    role=role,
                    reason=reason,
                    principal=principal,
                    lease_class=lease_class,
                    scrape=_capacity_shadow_ctx["scrape"],
                    predicted_defer=True,
                    actual_defer=_actual_defer,
                    scrape_to_lock_ms=(_lock_acquired_at - _capacity_shadow_ctx["scrape_done_at"]) * 1000,
                    gating_lease=_gating_lease,
                )
        if pending_resp is not None:
            return pending_resp

        # acquire_lease() manages its own locking internally (narrow self.lock
        # sections around bookkeeping, self.wake_lock around ensure_serving) —
        # it must NOT be wrapped in self.lock here, or a cold wake would once
        # again freeze every other endpoint for this node.
        ok = state.acquire_lease(
            work_id, ttl_sec, reason, role=role, principal=principal,
            require_drain_clear=require_drain_clear, lease_kind=lease_kind,
            lease_class=lease_class, mode=mode,
        )

        if ok is CREATIVE_OCCUPIED:
            return JSONResponse(
                {"ok": False, "creative_occupied": True,
                 "reason": "creative-collider-holding-gpu"},
                status_code=409,
            )
        if ok is FLASHNEXT_OCCUPIED:
            # D4: the flash-next window guard refused the wake — the seat
            # holds GPU 0 whole-card and no lease registered. Mirrors the
            # creative_occupied 409 (same refusal class: another seat holds
            # the lane).
            return JSONResponse(
                {"ok": False, "flashnext_occupied": True,
                 "reason": "flashnext-window-holding-gpu0"},
                status_code=409,
            )
        if ok is CONTENDED:
            return {"ok": False, "contended": True, "node": node}
        if ok is DEFERRED:
            return {
                "status": "deferred",
                "node": node,
                "mode_owner": DOORMAN_CONTROLLER_NAME,
                "detail": "GW serving controller-owned non-big mode; big endpoint unavailable",
            }
        if not ok:
            return {"status": "wake_failed", "detail": state.last_error or "wake failed"}

        resp: dict = {"status": "serving", "node": node, "work_id": work_id, "class": lease_class}
        if require_drain_clear:
            resp["drain_cleared"] = True  # signals to client that drain check was honored (AC5a)
        if release_info is not None:
            resp["waited_seconds"] = release_info["waited_seconds"]
            resp["release_reason"] = release_info["reason"]
        return resp

    # ------------------------------------------------------------------
    # POST /v0/force-stop — manual GW model unload, bypasses GW_STOP_GRACE_SEC
    # ------------------------------------------------------------------

    @app.post("/v0/force-stop")
    def force_stop(body: dict[str, Any]):
        node = body.get("node", "gravitywell")
        exclude_principal = body.get("exclude_principal") or None

        if node != "gravitywell":
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"node {node!r} not supported"),
            )
        if node not in nodes:
            return JSONResponse(
                status_code=404,
                content=_error("not_found", f"node {node!r} not found"),
            )

        state = nodes[node]
        # R1: _force_stop manages state.lock internally (released around the
        # ssh subprocess) — the route must NOT wrap it in `with state.lock:`,
        # or the lock would be held for the full stop duration again.
        result = state._force_stop(exclude_principal=exclude_principal)

        if result["status"] == "error":
            return JSONResponse(result, status_code=500)
        return result

    # ------------------------------------------------------------------
    # POST /lease/release
    # ------------------------------------------------------------------

    @app.post("/lease/release")
    def lease_release(body: dict[str, Any]):
        node = body.get("node", "")
        work_id = body.get("work_id", "")

        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        with state.lock:
            state.release_lease(work_id)

        return {"ok": True}

    # ------------------------------------------------------------------
    # POST /v0/brake, /v0/brake/release — emergency defer-only brake
    # (gw-router-phase1-foreground-gate). Global flag, not a work_id lease;
    # bounded TTL so it always auto-expires. Defers new `deferrable` dispatch
    # via the same wait-list; does NOT clear/kill existing leases. Subordinate
    # to per-job max-hold-timeout — a job already waiting still releases at
    # its own enqueue + max_hold even while the brake is held.
    # ------------------------------------------------------------------

    @app.post("/v0/brake")
    def brake_hold(body: dict[str, Any]):
        node = body.get("node", "gravitywell")
        reason = body.get("reason", "")
        ttl_s = int(body.get("ttl_s", DOORMAN_BRAKE_TTL_SEC))

        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        with state.lock:
            state.brake_reason = reason
            state.brake_expires_at = time.time() + ttl_s
            expires_at = state.brake_expires_at

        log.warning(f"[{node}] brake held reason={reason!r} ttl_s={ttl_s}")
        return {"braked": True, "expires_at": expires_at}

    @app.post("/v0/brake/release")
    def brake_release(body: dict[str, Any]):
        node = body.get("node", "gravitywell")

        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        with state.lock:
            state.brake_reason = None
            state.brake_expires_at = None

        log.warning(f"[{node}] brake released")
        return {"braked": False}

    return app
