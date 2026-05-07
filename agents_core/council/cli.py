"""
agents_core.council.cli — Mirror Council runtime + CLI entry points.

Relocated from /srv/agents/scripts/council.py. Two entry points:

    python -m agents_core.council submit "<decision>"
    python -m agents_core.council run <run_id>

submit:
  Runs the synchronous selection step (mem context → roster → Claude-CLI
  sonnet selector), writes the initial run YAML with status: deliberating
  and a complete selected_entities block, then submits a council.run queue
  task with task_id=run_id.  Use --no-queue to fall back to the legacy
  fire-and-forget fork path.

run:
  Reads the run YAML, drives the engine, writes turns + synthesis + terminal
  status.  Invoked as a subprocess by the queue handler (or directly by the
  legacy fork path).  Does NOT call send_notification — council completion is
  informational per Pushover policy; notification is opt-in via the submitter's
  --notify flag, which sets notify=True on the queue task.

Environment:
    COUNCIL_ENGINE_STUB=1  — skip LLM calls, write fixture turn/synthesis,
                             for smoke / CI use.

Modes:
  deliberation (default) — 2 entities, alternating turns, synthesis turn
    with LANDING / OPEN QUESTIONS / CONFIDENCE.  Uses DeliberationDirector.
  scene — 2-3 entities, rotation through N turns, no synthesis pass.
    Uses SceneDirector.  Prompt text is framed as a scene; final status
    is `closed` after the turn cap is reached.

Internal subcommands:
    council list
    council show <run-id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agents_core.llm import call_claude_cli  # noqa: E402

COUNCIL_DIR = Path("/srv/lapis/council")
CARDS_ROOT = Path(
    "/srv/git/archetypal-intelligence-working/cards/characters"
)
DEFAULT_POOLS = ["personal", "historical", "fiction"]
DEFAULT_TURNS = 8
DEFAULT_VOICING = "sonnet"
DEFAULT_MODE = "deliberation"
VALID_MODES = ("deliberation", "scene")
SCENE_N_RANGE = (2, 3)

# Roles are recorded in the run YAML's `selected_entities` list so the
# runtime subprocess can construct the right Entity type per slot.
ROLE_NARRATOR = "narrator"
LOG_DIR = Path("/srv/lapis/council/logs")

DASHBOARD_BASE = "http://203.0.113.12:8400"


# ---------------------------------------------------------------------------
# Run file I/O
# ---------------------------------------------------------------------------


def run_path(run_id: str) -> Path:
    return COUNCIL_DIR / f"{run_id}.yaml"


def load_run(run_id: str) -> dict:
    return yaml.safe_load(run_path(run_id).read_text())


def save_run(run: dict) -> None:
    run_path(run["run_id"]).write_text(
        yaml.safe_dump(run, sort_keys=False, width=100, allow_unicode=True)
    )


def new_run_id() -> str:
    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    tail = hashlib.sha1(f"{ts}{random.random()}".encode()).hexdigest()[:6]
    return f"{ts}-{tail}"


# ---------------------------------------------------------------------------
# Roster: lightweight card summaries for selection reasoning
# ---------------------------------------------------------------------------


def build_roster(pools: list[str] = DEFAULT_POOLS) -> list[dict]:
    """Return lightweight summaries for every card in the selection pool."""
    roster: list[dict] = []
    for pool in pools:
        pool_dir = CARDS_ROOT / pool
        if not pool_dir.is_dir():
            continue
        for card_path in sorted(pool_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(card_path.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            ctx = (data.get("cultural_context") or "").strip()
            roster.append(
                {
                    "character_id": data.get("character_id", card_path.stem),
                    "character_name": data.get("character_name", card_path.stem),
                    "pool": pool,
                    "cultural_context": ctx[:240],
                    "path": str(card_path),
                }
            )
    return roster


def find_card_path(character_id: str) -> Path | None:
    for pool_dir in CARDS_ROOT.iterdir():
        if not pool_dir.is_dir():
            continue
        candidate = pool_dir / f"{character_id}.yaml"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Receiver: context gathering and entity selection
# ---------------------------------------------------------------------------


def gather_mem_context(decision: str, max_hits: int = 6) -> dict:
    """Run `mem search` on salient tokens from the decision. Best-effort."""
    tokens = _extract_search_terms(decision)
    hits: list[dict] = []
    seen_keys: set[str] = set()
    for term in tokens[:3]:
        try:
            raw = subprocess.run(
                ["mem", "search", term],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout
        except Exception:
            continue
        for key in _parse_mem_keys(raw)[:max_hits]:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hits.append({"key": key, "matched_on": term})
    return {"terms": tokens[:3], "hits": hits[:max_hits]}


def _extract_search_terms(decision: str) -> list[str]:
    """Pull likely-salient nouns/terms from the decision. Heuristic."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", decision.lower())
    stop = {
        "should", "would", "could", "about", "with", "this", "that",
        "have", "from", "into", "them", "they", "then", "than", "when",
        "where", "which", "what", "does", "doing", "just", "some", "more",
        "less", "make", "take", "give", "want", "need", "even", "also",
    }
    terms = [w for w in words if w not in stop]
    seen: set[str] = set()
    ordered: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _parse_mem_keys(raw: str) -> list[str]:
    keys: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"^([a-z]+/[a-z0-9\-_/]+)\s", line)
        if m:
            keys.append(m.group(1))
    return keys


def select_entities(
    decision: str,
    roster: list[dict],
    context: dict,
    n: int = 2,
    with_entity: str | None = None,
    mode: str = DEFAULT_MODE,
    log=None,
) -> dict:
    """LLM-driven entity selection. Returns: {"selected": [...], "reasoning": "..."}"""
    pinned = with_entity or None

    lines = []
    for r in roster:
        label = f"- {r['character_id']} ({r['character_name']}, {r['pool']})"
        cue = r["cultural_context"].split(".")[0][:140]
        if cue:
            label += f" — {cue}"
        lines.append(label)
    roster_text = "\n".join(lines)

    mem_preview = (
        "\n".join(f"- {h['key']}" for h in context.get("hits", []))
        or "(no direct mem hits)"
    )

    remaining = n - 1 if pinned else n
    if mode == "scene":
        header = "You are selecting characters from a roster to inhabit a scene."
        prompt_label = "SCENE"
        guidance = (
            f"Pick the {'remaining character' if pinned else f'{n} characters'} "
            "whose composition, lived context, and habits of mind would make "
            "this scene CRACKLE. Prefer lived, 'muddied' perspectives "
            "(historical/fiction) over crystallized archetypes. Consider:\n"
            "- Who belongs in this scene — whose lived context places them here?\n"
            "- What friction or resonance would emerge between these characters?\n"
            "- Who would speak, act, or observe in a way that surprises the reader?"
        )
        reasoning_hint = (
            f"<2-3 sentences on why these {n} characters together would make "
            "the scene crackle>"
        )
    else:
        header = "You are selecting two characters from a roster to deliberate a decision."
        prompt_label = "DECISION TO DELIBERATE"
        guidance = (
            f"Pick the {'other participant' if pinned else 'two participants'} "
            "whose composition, lived context, and habits of mind would create "
            "the RICHEST friction and complementarity for THIS specific "
            "decision. Avoid participants who would merely agree. Prefer "
            "lived, 'muddied' perspectives (historical/fiction) over "
            "crystallized archetypes. Consider:\n"
            "- What domain is the decision in? Who has lived that domain?\n"
            "- What tensions does the decision hide? Who would surface them?\n"
            "- What blindspot does each candidate have? Would two candidates' "
            "blindspots cover each other?"
        )
        reasoning_hint = (
            "<2-3 sentences on why these two create the richest friction for "
            "this decision>"
        )

    pinned_clause = ""
    if pinned:
        role_word = "participants" if mode == "deliberation" else "characters"
        pinned_clause = (
            f"\n\nCONSTRAINT: The user has pinned `{pinned}` as one of the "
            f"{role_word}. Select {remaining} OTHER "
            f"{'character' if remaining == 1 else 'characters'} to join "
            f"{'them' if remaining > 1 else 'them'}."
        )

    if pinned:
        slots = [f'"{pinned}"'] + [f'"<id_{i+1}>"' for i in range(remaining)]
    else:
        slots = [f'"<id_{i+1}>"' for i in range(n)]
    selected_shape = "[" + ", ".join(slots) + "]"

    prompt = f"""{header}

{prompt_label}:
{decision}

RELATED MEM CONTEXT (from prior decisions/fixes/architecture):
{mem_preview}

ROSTER (character_id, source_pool, and a brief cue):
{roster_text}

{guidance}{pinned_clause}

Respond with ONLY a JSON object:
{{
  "selected": {selected_shape},
  "reasoning": "{reasoning_hint}"
}}
"""

    system_msg = (
        "You are a thoughtful council curator. Select for productive tension, "
        "not consensus."
        if mode == "deliberation"
        else "You are a thoughtful scene curator. Select characters whose "
        "presence together would produce a scene worth reading."
    )

    raw = call_claude_cli(
        prompt=prompt,
        system=system_msg,
        model="sonnet",
        timeout=120,
        json_mode=True,
        log=log,
    )
    if not raw:
        raise RuntimeError("Entity selection failed — Claude CLI returned nothing")

    data = _extract_json(raw)
    selected = data.get("selected") or []
    if not isinstance(selected, list) or len(selected) != n:
        raise RuntimeError(
            f"Selection returned {len(selected) if isinstance(selected, list) else '?'} "
            f"entities, expected {n}. Raw: {raw[:300]}"
        )
    for sid in selected:
        if not isinstance(sid, str) or not find_card_path(sid):
            raise RuntimeError(f"Selected unknown character_id: {sid!r}")
    if len(set(selected)) != len(selected):
        raise RuntimeError(
            f"Selection returned duplicate entities: {selected}"
        )
    return {"selected": selected, "reasoning": data.get("reasoning", "")}


def _extract_json(text: str) -> dict:
    text = text.strip()
    for candidate in (text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Deliberation runtime
# ---------------------------------------------------------------------------


def run_deliberation(run_id: str) -> None:
    """Read /<COUNCIL_DIR>/<run_id>.yaml, drive the engine, write turns +
    synthesis + terminal status back to the same file.  Mutates in place via
    save_run; returns None.

    Does NOT call send_notification.  Council completion is informational per
    Pushover policy; if the caller wants a Pushover, they pass --notify to
    the submitter, which sets notify=True on the queue task and lets the
    runner's notify_completion/notify_failure handle it.

    COUNCIL_ENGINE_STUB=1: skip LLM calls entirely, write fixture turn/synthesis.
    """
    run = load_run(run_id)
    mode = run.get("mode", DEFAULT_MODE)

    if os.environ.get("COUNCIL_ENGINE_STUB") == "1":
        first_id = (run["selected_entities"][0]["id"]
                    if run.get("selected_entities") else "stub-entity")
        run["turns"].append(
            {
                "step": 1,
                "speaker": first_id,
                "type": "dialogue",
                "content": "[STUB] turn output",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if mode == "scene":
            run["status"] = "closed"
        else:
            synth_content = (
                "LANDING: stub-landing\n"
                "OPEN QUESTIONS: -\n"
                "CONFIDENCE: converged"
            )
            run["turns"].append(
                {
                    "step": 2,
                    "speaker": "synthesis",
                    "type": "synthesis",
                    "content": synth_content,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            run["synthesis"] = _parse_synthesis(synth_content)
            run["status"] = _status_from_synthesis(run["synthesis"])
        run["completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_run(run)
        return

    # Late imports: keep lapis-engine load function-local so importing
    # agents_core.council doesn't force-load lapis-engine at queue startup.
    from lapis_engine import (
        ClaudeAdapter,
        DeliberationDirector,
        Engine,
        LlamaAdapter,
        SceneDirector,
        StepData,
    )
    from archetypes.engine.character_entity import CharacterEntity
    from agents_core.council.narrator_entity import NarratorEntity

    try:
        adapter = _build_adapter(run["voicing"], ClaudeAdapter, LlamaAdapter)
        entities = [
            _build_entity(sel, adapter, CharacterEntity, NarratorEntity)
            for sel in run["selected_entities"]
        ]
        director = _build_director(
            mode=mode,
            prompt=run["decision"],
            turns=int(run["turns_cap"]),
            DeliberationDirector=DeliberationDirector,
            SceneDirector=SceneDirector,
        )

        def on_step(sd: "StepData") -> None:
            event = sd.events[0] if sd.events else None
            run["turns"].append(
                {
                    "step": sd.step,
                    "speaker": sd.acting_entity_id,
                    "type": event.type if event else "unknown",
                    "content": sd.response,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            save_run(run)

        Engine().run(director=director, entities=entities, on_step=on_step)

        if mode == "scene":
            run["status"] = "closed"
        else:
            synth_content = run["turns"][-1]["content"] if run["turns"] else ""
            run["synthesis"] = _parse_synthesis(synth_content)
            run["status"] = _status_from_synthesis(run["synthesis"])
        run["completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_run(run)
    except Exception as e:
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        run["completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_run(run)
        raise


def _build_entity(sel: dict, adapter, CharacterEntity, NarratorEntity):
    if sel.get("role") == ROLE_NARRATOR:
        return NarratorEntity(llm=adapter, voice=sel.get("voice"))
    card_path = find_card_path(sel["id"])
    if card_path is None:
        raise RuntimeError(f"No card for entity id: {sel['id']!r}")
    return CharacterEntity.load(card_path, adapter)


def _build_director(mode: str, prompt: str, turns: int,
                    DeliberationDirector, SceneDirector):
    if mode == "scene":
        return SceneDirector(id="council", scene=prompt, turns=turns)
    if mode == "deliberation":
        return DeliberationDirector(id="council", decision=prompt, turns=turns)
    raise ValueError(f"Unknown mode: {mode!r}")


def _build_adapter(voicing: str, ClaudeAdapter, LlamaAdapter):
    if voicing == "local":
        return LlamaAdapter(temperature=0.8, max_tokens=900)
    if voicing in ("haiku", "sonnet", "opus"):
        return ClaudeAdapter(model=voicing, timeout=300)
    raise ValueError(f"Unknown voicing: {voicing!r}")


def _parse_synthesis(text: str) -> dict:
    """Parse LANDING / OPEN QUESTIONS / CONFIDENCE format from synthesis turn."""
    landing_m = re.search(
        r"LANDING:\s*(.+?)(?=\n\s*OPEN QUESTIONS:|$)", text, re.IGNORECASE | re.DOTALL
    )
    questions_m = re.search(
        r"OPEN QUESTIONS:\s*(.+?)(?=\n\s*CONFIDENCE:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    confidence_m = re.search(r"CONFIDENCE:\s*(\w+)", text, re.IGNORECASE)

    questions_raw = (questions_m.group(1).strip() if questions_m else "").strip()
    if questions_raw.lower() in {"none", "none.", "n/a", "-", ""}:
        questions: list[str] = []
    else:
        questions = [
            re.sub(r"^[\-\*\d\.\)]+\s*", "", line).strip()
            for line in questions_raw.splitlines()
            if line.strip()
        ]

    return {
        "landing": (landing_m.group(1).strip() if landing_m else text.strip()),
        "open_questions": questions,
        "confidence": (confidence_m.group(1).lower() if confidence_m else "partial"),
    }


def _status_from_synthesis(synthesis: dict) -> str:
    conf = synthesis.get("confidence", "partial")
    if conf == "converged":
        return "resolved"
    if conf == "diverged":
        return "diverged"
    return "open"


# ---------------------------------------------------------------------------
# CLI: submit / run / list / show
# ---------------------------------------------------------------------------


def _validate_mode_n(
    mode: str,
    n: int,
    narrator: bool = False,
    with_entity: str | None = None,
) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r} (valid: {VALID_MODES})")
    if mode == "deliberation" and n != 2:
        raise ValueError("deliberation mode requires n=2")
    if mode == "scene" and n not in SCENE_N_RANGE:
        raise ValueError(f"scene mode requires n in {SCENE_N_RANGE}, got {n}")
    if narrator:
        if mode != "scene":
            raise ValueError("--narrator requires --mode=scene")
        if n != 3:
            raise ValueError("--narrator requires n=3 (2 characters + narrator)")
        if with_entity and with_entity == ROLE_NARRATOR:
            raise ValueError("cannot pin the narrator with --with")


def _role_assignments(
    selected: list[str],
    mode: str,
    narrator: bool = False,
    narrator_voice: str | None = None,
) -> list[dict]:
    if mode == "deliberation":
        return [
            {"id": selected[0], "role": "first_voice"},
            {"id": selected[1], "role": "second_voice"},
        ]
    if narrator:
        if len(selected) != 2:
            raise ValueError(
                f"narrator scene requires 2 character ids, got {len(selected)}"
            )
        entry: dict = {"id": ROLE_NARRATOR, "role": ROLE_NARRATOR}
        if narrator_voice:
            entry["voice"] = narrator_voice
        return [
            {"id": selected[0], "role": "scene_slot_0"},
            {"id": selected[1], "role": "scene_slot_1"},
            entry,
        ]
    return [{"id": sid, "role": f"scene_slot_{i}"} for i, sid in enumerate(selected)]


def cmd_submit(args: argparse.Namespace) -> int:
    mode = args.mode
    narrator = bool(args.narrator or args.narrator_voice)
    if args.n is None:
        n = 3 if narrator else 2
    else:
        n = args.n
    _validate_mode_n(mode, n, narrator=narrator, with_entity=args.with_entity)

    n_characters = n - 1 if narrator else n

    COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    narrator_tag = " narrator" if narrator else ""
    print(f"[council] run_id={run_id} mode={mode} n={n}{narrator_tag}", flush=True)

    print("[council] gathering mem context...", flush=True)
    context = gather_mem_context(args.decision)
    if context["hits"]:
        print(
            f"[council] mem hits: {len(context['hits'])} from terms {context['terms']}",
            flush=True,
        )
    else:
        print(f"[council] no mem hits for terms {context['terms']}", flush=True)

    print("[council] building roster...", flush=True)
    roster = build_roster()
    print(f"[council] roster size: {len(roster)}", flush=True)

    print("[council] selecting entities (via claude sonnet)...", flush=True)
    selection = select_entities(
        decision=args.decision,
        roster=roster,
        context=context,
        n=n_characters,
        with_entity=args.with_entity,
        mode=mode,
    )
    print(
        f"[council] selected: {' + '.join(selection['selected'])}"
        + (f" + {ROLE_NARRATOR}" if narrator else ""),
        flush=True,
    )

    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "deliberating",
        "mode": mode,
        "decision": args.decision,
        "context_gathered": context,
        "selected_entities": _role_assignments(
            selection["selected"],
            mode,
            narrator=narrator,
            narrator_voice=args.narrator_voice,
        ),
        "selection_reasoning": selection["reasoning"],
        "voicing": args.voicing,
        "turns_cap": args.turns,
        "turns": [],
    }
    save_run(run)

    no_queue = getattr(args, "no_queue", False)
    if no_queue:
        log_file = LOG_DIR / f"{run_id}.log"
        print(
            f"[council] forking deliberation subprocess (log: {log_file})", flush=True
        )
        _fork_runtime(run_id, log_file)
        print(
            f"[council] fire-and-forget. Watch: {DASHBOARD_BASE}/council/{run_id}",
            flush=True,
        )
        return 0

    # Queue path (default)
    from agents_core.claude_queue import ClaudeQueue
    queue = ClaudeQueue()

    timeout_seconds = 2400 if mode == "scene" else 1200
    description = args.decision.replace("\n", " ").replace("\r", " ")
    if len(description) > 80:
        description = description[:80] + "\u2026"

    notify = getattr(args, "notify", False)
    task = {
        "task_type": "council.run",
        "description": description,
        "priority": 5,
        "model": "sonnet",
        "notify": bool(notify),
        "timeout_seconds": timeout_seconds,
        "payload": {
            "mode": mode,
            "_ignore_intention_registry": True,
        },
    }
    submitted_id = queue.submit(task, task_id=run_id)
    print(f"[council] queued task_id={submitted_id}", flush=True)
    print(f"[council] Watch: {DASHBOARD_BASE}/council/{run_id}", flush=True)
    return 0


def _fork_runtime(run_id: str, log_file: Path) -> None:
    """Detached subprocess: `python -m agents_core.council run <run-id>`."""
    with open(log_file, "ab") as f:
        subprocess.Popen(
            [sys.executable, "-m", "agents_core.council", "run", run_id],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )


def cmd_run(args: argparse.Namespace) -> int:
    run_deliberation(args.run_id)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    if not COUNCIL_DIR.exists():
        print("(no runs yet)")
        return 0
    runs = sorted(COUNCIL_DIR.glob("*.yaml"), reverse=True)
    for p in runs[:20]:
        try:
            d = yaml.safe_load(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        status = d.get("status", "?")
        decision = (d.get("decision") or "")[:80]
        ids = " + ".join(e["id"] for e in d.get("selected_entities", []))
        print(f"{d.get('run_id'):<28} {status:<12} {ids:<50} {decision}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        run = load_run(args.run_id)
    except FileNotFoundError:
        print(f"no such run: {args.run_id}")
        return 1
    print(yaml.safe_dump(run, sort_keys=False, width=100, allow_unicode=True))
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="council", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("submit", help="Submit a decision or scene for voicing")
    sp.add_argument(
        "decision",
        help="The decision (deliberation mode) or scene text (scene mode)",
    )
    sp.add_argument(
        "--mode", choices=list(VALID_MODES), default=DEFAULT_MODE,
    )
    sp.add_argument("--n", type=int, default=None)
    sp.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    sp.add_argument(
        "--voicing", choices=["local", "haiku", "sonnet", "opus"],
        default=DEFAULT_VOICING,
    )
    sp.add_argument("--with", dest="with_entity", default=None)
    sp.add_argument("--narrator", action="store_true")
    sp.add_argument("--narrator-voice", dest="narrator_voice", default=None)
    sp.add_argument(
        "--no-queue", dest="no_queue", action="store_true",
        help="Legacy fork: bypass the queue (rollback valve for one release).",
    )
    sp.add_argument(
        "--notify", action="store_true",
        help="Opt-in Pushover on completion. No effect under --no-queue.",
    )
    sp.set_defaults(func=cmd_submit)

    rp = sub.add_parser("run", help="(internal) Run the deliberation for a run_id")
    rp.add_argument("run_id")
    rp.set_defaults(func=cmd_run)

    lp = sub.add_parser("list", help="List recent council runs")
    lp.set_defaults(func=cmd_list)

    shp = sub.add_parser("show", help="Show a run's YAML")
    shp.add_argument("run_id")
    shp.set_defaults(func=cmd_show)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
