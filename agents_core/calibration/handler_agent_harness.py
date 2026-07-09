"""Handler+Agent cross-box smoke harness (agent-designer / Topology Y).

Promoted from a one-off scratchpad script into a config-driven CLI module
(agents-core-handler-agent-harness-promotion-v0). Proves the atomic
batched-worker unit runs across two boxes:
  - ENVIRONMENT          = AgentWorld world-model on GravityWell (BF16)
  - AGENT + HANDLER pair = Qwen3-14B-AWQ on StarHouse

Loop (queue-runner environment, which is what this AgentWorld was trained as):
  AGENT proposes a queue lifecycle action -> AgentWorld simulates state_after
  -> HANDLER observes the transition, steers, decides continue/done.

This is a plumbing + pair-loop smoke, not a scored eval. AgentWorld runs in
with-rules mode at low temp so the environment behaves crisply.

CLI:
  python -m agents_core.calibration.handler_agent_harness \\
    [--gw-agentworld-url URL] [--sh-pair-url URL] \\
    [--agentworld-model NAME] [--pair-model NAME] \\
    [--agent-url URL] [--handler-url URL] \\
    [--agent-model NAME] [--handler-model NAME] \\
    [--max-turns N] [--out PATH]

Env var overrides: --gw-agentworld-url (GW_AGENTWORLD_URL), --sh-pair-url
(SH_PAIR_URL), --agent-url (AGENT_URL), and --handler-url (HANDLER_URL) each
read an env var, and only as their *default* -- an explicit CLI flag always
wins over the env var. The model flags (--agentworld-model, --pair-model,
--agent-model, --handler-model) are CLI-only, no env var.

--sh-pair-url/--pair-model set the Agent+Handler pair default (both roles
resolve to it when unset). --agent-url/--agent-model and --handler-url/
--handler-model independently override a single role, falling back to the
pair default for that role when unset. See _resolve_pair_endpoints for the
exact precedence.

A failed preflight, or a turn that dies mid-run, still writes a report to
--out (with a `preflight_error` field or a per-turn `{"turn", "error"}`
transcript entry, respectively) instead of vanishing with no artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import httpx

from agents_core.room_paths import room_path

log = logging.getLogger("calibration.handler_agent_harness")

GOAL = ("Process BOTH pending jobs to completion. Drive the queue to: "
        "completed=2, pending=0, active=0, failed=0. "
        "Valid actions are queue lifecycle events only: 'submitted', 'claimed', "
        "'completed', 'failed'. A job must be 'claimed' (pending->active) before it "
        "can be 'completed' (active->completed). Use job ids j1 and j2.")

INITIAL_STATE = {
    "counts": {"pending": 2, "active": 0, "completed": 0, "failed": 0},
    "in_flight": [],
    "workers": {"capacity": 2, "utilization": 0.0},
    "stasis_duration": 0.0,
    "stasis_velocity": 0.0,
}

# --- AgentWorld (environment) system prompt: with-rules for crisp behavior ----
AW_SYSTEM = """\
You are AgentWorld - a learned world model of the Lapis StarHouse claude-queue runner.

## Queue-runner state machine
- submitted  -> pending +1
- claimed    -> pending -1; active +1; job enters in_flight
- completed  -> active -1; completed +1; job leaves in_flight
- failed     -> active -1; failed +1; job leaves in_flight
workers.capacity is static (2). workers.utilization = len(in_flight)/capacity.
After any event, state_after.stasis_duration = 0 and stasis_velocity = 0.

## Task
Given state_before and the incoming event, predict the exact state_after JSON.
Output ONLY valid JSON (State schema), no prose, no markdown fences.

## State schema
{"counts":{"pending":<int>,"active":<int>,"completed":<int>,"failed":<int>},
 "in_flight":[{"id":"<str>","task_type":"<str>","model":"<str|null>","claimed_at":"<ISO8601>","stasis_duration":<float>}],
 "workers":{"capacity":<int>,"utilization":<float>},
 "stasis_duration":<float>,"stasis_velocity":<float>}
"""

AGENT_SYSTEM = f"""\
You are the AGENT (the doer) in a Handler+Agent pair operating a job queue.
Your mission: {GOAL}
Each turn you output the SINGLE next action to take, as a queue lifecycle event.
Output ONLY valid JSON, no prose, no markdown:
{{"event":"submitted|claimed|completed|failed","id":"j1|j2","task_type":"test","model":"m1","reason":"<short why>"}}
Rules: never 'completed'/'failed' a job that is not currently active (in_flight).
Claim a pending job before completing it. Work toward the goal state efficiently."""

HANDLER_SYSTEM = f"""\
You are the HANDLER (observer/steerer) in a Handler+Agent pair. You do NOT execute
actions. You watch the Agent act on the queue and keep the mission on track.
Mission: {GOAL}
Given the transition (state_before, the Agent's action, resulting state_after), assess it.
Output ONLY valid JSON, no prose, no markdown:
{{"on_track":true|false,"note":"<one line of provenance>","anomaly":"<oddity to capture, or null>",
  "decision":"continue|done|redirect","redirect":"<instruction if redirecting, else null>"}}
Set decision='done' ONLY when the goal state is reached (completed=2, pending=0, active=0, failed=0).
Capture-don't-chase: if something strange appears you are NOT going to explore, still record it in 'anomaly'."""


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        r = json.loads(text)
        if isinstance(r, dict):
            return r
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group(0))
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    return None


def chat(endpoint: str, model: str, messages: list, temperature: float, max_tokens: int = 1024) -> str:
    payload = {
        "model": model, "messages": messages,
        "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = httpx.post(f"{endpoint}/chat/completions", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_json(endpoint, model, system, user, temp, label) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        raw = chat(endpoint, model, msgs, temp)
        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed
        msgs.append({"role": "assistant", "content": raw})
        msgs.append({"role": "user", "content": "That was not valid JSON. Output ONLY the JSON object."})
    raise RuntimeError(f"{label}: could not parse JSON after 2 attempts. Last raw:\n{raw[:400]}")


def _build_report(
    environment: str,
    agent: str,
    handler: str,
    transcript: list[dict],
    goal_met: bool,
    final_counts: dict | None,
    preflight_error: str | None = None,
) -> dict:
    report = {
        "smoke": "handler-agent-cross-box-topology-y",
        "environment": environment,
        "agent": agent,
        "handler": handler,
        "turns": len(transcript),
        "goal_met": goal_met,
        "final_counts": final_counts,
        "transcript": transcript,
    }
    if preflight_error is not None:
        report["preflight_error"] = preflight_error
    return report


def _write_report(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)


def preflight(
    gw_agentworld_url: str,
    agent_url: str,
    agent_model: str,
    handler_url: str,
    handler_model: str,
    out: Path,
) -> None:
    """Reachability check for the distinct endpoints among environment/agent/handler.

    Probes GW/AgentWorld plus each *distinct* SH endpoint among {agent_url,
    handler_url} (10s timeout each, unchanged) -- up to three probes total.
    When agent_url == handler_url, SH is probed once, not twice, and that
    dedup is logged. On failure, writes a preflight-failure report to `out`
    (§1e), naming which role's endpoint failed, and exits 1.
    """
    if agent_url == handler_url:
        log.info("agent and handler share endpoint %s - probing once", agent_url)

    roles_by_url: dict[str, list[str]] = {}
    for role, ep in (("agent", agent_url), ("handler", handler_url)):
        roles_by_url.setdefault(ep, []).append(role)

    probes = [("GW/AgentWorld", gw_agentworld_url)]
    probes += [(f"SH/{'+'.join(roles)}", ep) for ep, roles in roles_by_url.items()]

    for name, ep in probes:
        try:
            r = httpx.get(f"{ep}/models", timeout=10.0)
            r.raise_for_status()
            served = [m["id"] for m in r.json().get("data", [])]
            log.info("  [%s] reachable, served=%s", name, served)
        except Exception as e:
            log.error("  [%s] UNREACHABLE: %s", name, e)
            report = _build_report(
                gw_agentworld_url,
                agent=f"{agent_url} ({agent_model})",
                handler=f"{handler_url} ({handler_model})",
                transcript=[], goal_met=False,
                final_counts=None, preflight_error=f"{name}: {e}",
            )
            _write_report(report, out)
            sys.exit(1)


def run_smoke(
    gw_agentworld_url: str,
    agentworld_model: str,
    agent_url: str,
    agent_model: str,
    handler_url: str,
    handler_model: str,
    max_turns: int,
    out: Path,
) -> None:
    """Run the Handler+Agent cross-box smoke loop and write the report to `out`."""
    log.info("=== Handler+Agent cross-box smoke (Topology Y) ===")
    log.info("  environment (AgentWorld) : %s", gw_agentworld_url)
    log.info("  agent                    : %s (%s)", agent_url, agent_model)
    log.info("  handler                  : %s (%s)", handler_url, handler_model)
    log.info("--- preflight ---")
    preflight(gw_agentworld_url, agent_url, agent_model, handler_url, handler_model, out)

    agent_descriptor = f"{agent_url} ({agent_model})"
    handler_descriptor = f"{handler_url} ({handler_model})"

    state = json.loads(json.dumps(INITIAL_STATE))
    transcript: list[dict] = []
    goal_met = False
    log.info("--- initial state ---\n%s", json.dumps(state["counts"]))

    for turn in range(1, max_turns + 1):
        log.info("===== TURN %d =====", turn)
        try:
            # 1. AGENT proposes an action (SH)
            t0 = time.monotonic()
            action = ask_json(
                agent_url, agent_model, AGENT_SYSTEM,
                f"Current queue state:\n{json.dumps(state, indent=2)}\n\nYour next action:",
                temp=0.2, label="AGENT",
            )
            ta = time.monotonic() - t0
            log.info("  AGENT (%.1fs): %s", ta,
                      json.dumps({k: action[k] for k in ("event", "id", "reason") if k in action}))

            # 2. ENVIRONMENT simulates state_after (GW AgentWorld)
            env_event = {k: action.get(k) for k in ("event", "id", "task_type", "model")}
            env_event.setdefault("task_type", "test")
            env_event.setdefault("model", "m1")
            t0 = time.monotonic()
            after = ask_json(
                gw_agentworld_url, agentworld_model, AW_SYSTEM,
                f"state_before:\n{json.dumps(state, indent=2)}\n\nevent:\n{json.dumps(env_event, indent=2)}\n\nPredict state_after:",
                temp=0.15, label="AGENTWORLD",
            )
            te = time.monotonic() - t0
            log.info("  ENV/AgentWorld (%.1fs): counts=%s", te, json.dumps(after.get("counts")))

            # 3. HANDLER observes + steers (SH)
            t0 = time.monotonic()
            obs = ask_json(
                handler_url, handler_model, HANDLER_SYSTEM,
                f"state_before:\n{json.dumps(state)}\n\nagent_action:\n{json.dumps(action)}\n\nstate_after:\n{json.dumps(after)}\n\nAssess:",
                temp=0.2, label="HANDLER",
            )
            th = time.monotonic() - t0
            log.info("  HANDLER (%.1fs): on_track=%s decision=%s note=%r",
                      th, obs.get("on_track"), obs.get("decision"), obs.get("note"))
            if obs.get("anomaly"):
                log.info("    [anomaly captured] %s", obs["anomaly"])

            transcript.append({
                "turn": turn, "action": action, "state_after": after, "handler": obs,
                "latency_s": {"agent": round(ta, 1), "env": round(te, 1), "handler": round(th, 1)},
            })
            state = after

            c = after.get("counts", {})
            if c.get("completed") == 2 and c.get("pending") == 0 and c.get("active") == 0 and c.get("failed") == 0:
                goal_met = True
                log.info("  *** GOAL STATE REACHED ***")
                break
            if obs.get("decision") == "done":
                break
        except Exception as e:
            log.error("  turn %d failed: %s", turn, e)
            transcript.append({"turn": turn, "error": str(e)})
            report = _build_report(
                gw_agentworld_url, agent_descriptor, handler_descriptor, transcript,
                goal_met=False, final_counts=state.get("counts"),
            )
            _write_report(report, out)
            sys.exit(1)

    report = _build_report(
        gw_agentworld_url, agent_descriptor, handler_descriptor, transcript,
        goal_met=goal_met, final_counts=state.get("counts"),
    )
    _write_report(report, out)
    log.info("=== RESULT: goal_met=%s, turns=%d, final=%s ===",
              goal_met, len(transcript), json.dumps(state.get("counts")))
    log.info("=== report -> %s ===", out)
    sys.exit(0 if goal_met else 1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Handler+Agent cross-box smoke harness (Topology Y)",
    )
    parser.add_argument(
        "--gw-agentworld-url", dest="gw_agentworld_url",
        default=os.environ.get("GW_AGENTWORLD_URL", "http://203.0.113.11:8090/v1"),
        help="AgentWorld (environment) endpoint on GravityWell "
             "(env: GW_AGENTWORLD_URL; default: %(default)s)",
    )
    parser.add_argument(
        "--sh-pair-url", dest="sh_pair_url",
        default=os.environ.get("SH_PAIR_URL", "http://203.0.113.12:8082/v1"),
        help="Handler+Agent pair endpoint on StarHouse "
             "(env: SH_PAIR_URL; default: %(default)s)",
    )
    parser.add_argument(
        "--agentworld-model", dest="agentworld_model", default="agentworld",
        help="Model name for AgentWorld (default: %(default)s)",
    )
    parser.add_argument(
        "--pair-model", dest="pair_model", default="handler-agent",
        help="Model name for the Handler+Agent pair (default: %(default)s)",
    )
    parser.add_argument(
        "--agent-url", dest="agent_url",
        default=os.environ.get("AGENT_URL"),
        help="Agent-role endpoint override (env: AGENT_URL). Falls back to "
             "--sh-pair-url/SH_PAIR_URL when unset. May be combined with "
             "--sh-pair-url: this flag wins for the Agent role only.",
    )
    parser.add_argument(
        "--handler-url", dest="handler_url",
        default=os.environ.get("HANDLER_URL"),
        help="Handler-role endpoint override (env: HANDLER_URL). Falls back "
             "to --sh-pair-url/SH_PAIR_URL when unset. May be combined with "
             "--sh-pair-url: this flag wins for the Handler role only.",
    )
    parser.add_argument(
        "--agent-model", dest="agent_model", default=None,
        help="Model name override for the Agent role. Falls back to "
             "--pair-model when unset. CLI-only, no env var.",
    )
    parser.add_argument(
        "--handler-model", dest="handler_model", default=None,
        help="Model name override for the Handler role. Falls back to "
             "--pair-model when unset. CLI-only, no env var.",
    )
    parser.add_argument(
        "--max-turns", dest="max_turns", type=int, default=8,
        help="Max turns before giving up (default: %(default)s)",
    )
    parser.add_argument(
        "--out", dest="out", type=Path,
        default=room_path("calibration") / "handler-agent-harness" / "report.json",
        help="Output path for report.json (default: %(default)s)",
    )
    return parser


def _resolve_pair_endpoints(args: argparse.Namespace) -> tuple[str, str, str, str]:
    """Resolve per-role (agent, handler) endpoint + model pairs.

    Precedence per role url: explicit --agent-url/--handler-url flag >
    AGENT_URL/HANDLER_URL env (empty or whitespace-only treated as unset) >
    resolved --sh-pair-url (itself --sh-pair-url flag > SH_PAIR_URL env >
    hardcoded default). Model precedence: explicit --agent-model/
    --handler-model flag > --pair-model.

    Passing both a legacy pair flag (--sh-pair-url/--pair-model) and a role
    flag together is valid, not an error: the role flag wins for that one
    role, and the pair default supplies the other role.

    Caveat: a set-but-empty SH_PAIR_URL="" makes the pair default itself ""
    -- a pre-existing edge in --sh-pair-url's own env resolution that this
    function does not fix (out of scope).
    """
    def _norm(v: str | None) -> str | None:
        return v if (v and v.strip()) else None

    agent_url = _norm(args.agent_url) or args.sh_pair_url
    handler_url = _norm(args.handler_url) or args.sh_pair_url
    agent_model = _norm(args.agent_model) or args.pair_model
    handler_model = _norm(args.handler_model) or args.pair_model
    return agent_url, agent_model, handler_url, handler_model


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _build_parser().parse_args(argv)
    agent_url, agent_model, handler_url, handler_model = _resolve_pair_endpoints(args)

    run_smoke(
        gw_agentworld_url=args.gw_agentworld_url,
        agentworld_model=args.agentworld_model,
        agent_url=agent_url,
        agent_model=agent_model,
        handler_url=handler_url,
        handler_model=handler_model,
        max_turns=args.max_turns,
        out=args.out,
    )


if __name__ == "__main__":
    main()
