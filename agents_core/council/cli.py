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
    COUNCIL_STUB_POSITIONS — comma-separated positions for stub mode
                             (e.g. "agree,stand-aside"). Default: "agree,agree".

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
from datetime import datetime
from pathlib import Path

import yaml

from agents_core.llm import call_operator  # noqa: E402
from agents_core.council.gravitywell_adapter import GravityWellAdapter

COUNCIL_DIR = Path("/srv/lapis/council")
CARDS_ROOT = Path(
    "/srv/git/archetypal-intelligence-working/cards/characters"
)
DEFAULT_POOLS = ["personal", "historical", "fiction"]
DEFAULT_TURNS = 8
DEFAULT_VOICING = "gravitywell"
DEFAULT_MODE = "deliberation"
VALID_MODES = ("deliberation", "scene")
SCENE_N_RANGE = (2, 3)
RECENCY_LOOKBACK = 6

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

    context: dict = {"terms": tokens[:3], "hits": hits[:max_hits]}

    # NEW v0.next: surface prior cohesion findings for voice selection context.
    # find_related returns [] when cache dir is missing (first-deploy safe).
    try:
        from agents_core.council import cache as _cache
        context["cohesion_findings"] = _cache.find_related(
            decision_text=decision, limit=3
        )
    except Exception:
        context["cohesion_findings"] = []

    return context


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


def recent_pair_entities(lookback: int = RECENCY_LOOKBACK) -> list[str]:
    """Entity ids that filled seats 1-2 in the most recent `lookback` deliberation
    runs, newest first, de-duplicated. Far-seat (`third_voice`) excluded."""
    if not COUNCIL_DIR.exists():
        return []
    seen: set[str] = set()
    result: list[str] = []
    deliberation_count = 0
    for run_file in sorted(COUNCIL_DIR.glob("*.yaml"), reverse=True):
        if deliberation_count >= lookback:
            break
        try:
            data = yaml.safe_load(run_file.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("mode") != "deliberation":
            continue
        deliberation_count += 1
        for entity in data.get("selected_entities", []):
            if not isinstance(entity, dict):
                continue
            if entity.get("role") in ("first_voice", "second_voice"):
                eid = entity.get("id")
                if eid and eid not in seen:
                    seen.add(eid)
                    result.append(eid)
    return result


def select_entities(
    decision: str,
    roster: list[dict],
    context: dict,
    n: int = 2,
    with_entity: str | None = None,
    mode: str = DEFAULT_MODE,
    recent_pair_ids: list[str] | None = None,
    log=None,
) -> dict:
    """LLM-driven entity selection. Returns: {"selected": [...], "reasoning": "...", "selection_operator": "...", "selection_degraded": <bool>}"""
    if os.environ.get("COUNCIL_ENGINE_STUB") == "1":
        resolvable_ids = [r["character_id"] for r in roster if find_card_path(r["character_id"])][:n]
        return {
            "selected": resolvable_ids,
            "reasoning": "[stub]",
            "selection_operator": "stub",
            "selection_degraded": False,
        }

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

    # NEW v0.next: render prior cohesion findings section between MEM CONTEXT and ROSTER.
    findings = context.get("cohesion_findings", [])
    if findings:
        finding_lines = []
        for f in findings:
            kv = f.get("kernel_version", "?")
            stale = f.get("stale", False)
            landing = f.get("synthesis", {}).get("landing", "") or ""
            prefix = f"[stale, kernel {kv}]" if stale else f"[prior finding {kv}]"
            truncated = (landing[:300] + "\u2026") if len(landing) > 300 else landing
            finding_lines.append(f"{prefix} {truncated}")
        findings_text = "\n".join(finding_lines)
    else:
        findings_text = "(none)"
    findings_section = (
        f"RELATED PRIOR COHESION FINDINGS:\n{findings_text}\n--- END PRIOR FINDINGS ---"
    )

    effective_recent: list[str] = []
    if recent_pair_ids and mode == "deliberation":
        effective_recent = [rid for rid in recent_pair_ids if rid != pinned]

    recency_clause = ""
    if effective_recent:
        ids_str = ", ".join(effective_recent)
        recency_clause = (
            "\n\nRECENT DIALECTICAL PAIR (these voices filled seats 1-2 in recent councils): "
            f"{ids_str}. Prefer a FRESH pairing - avoid these unless one is *uniquely* required "
            "for THIS specific decision; if you reuse one, your reasoning must say why no fresher "
            "voice could occupy that seat."
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
    elif n == 3:
        header = "You are selecting three characters from a roster to deliberate a decision."
        prompt_label = "DECISION TO DELIBERATE"
        guidance = (
            f"Pick {'the other two participants' if pinned else 'three participants'} "
            "for this deliberation.\n\n"
            "SEATS 1-2 (dialectical pair): Choose for the richest friction and "
            "complementarity — whose lived context and habits of mind create the most "
            "productive tension for this specific decision. Avoid participants who would "
            "merely agree. Prefer lived, 'muddied' perspectives (historical/fiction) over "
            "crystallized archetypes. Consider what domain the decision is in, what "
            "tensions it hides, and how the pair's blindspots might cover each other."
            + recency_clause
            + "\n\nSEAT 3 (FAR SEAT — distant vantage): Choose this voice from a FAR-REACHING "
            "spot in the roster — a different culture, discipline, era, tradition, or kind "
            "of figure (e.g. literary rather than historical). The goal is to MAXIMIZE "
            "VANTAGE DISTANCE from the first two and supply the perspective the dialectical "
            "pair structurally lacks. This is NOT a contrarian, trickster, or devil's "
            "advocate: it widens the vantage, it does not disrupt. Avoid three figures of "
            "the same school, era, or register. Ask: what perspective is the dialectical "
            "pair structurally blind to, and who from a far-reaching cultural, "
            "disciplinary, or temporal origin would supply it?"
        )
        reasoning_hint = (
            "<2-3 sentences: name the dialectical spine the first two bring, then NAME "
            "THE VANTAGE the far seat supplies — the specific perspective the pair "
            "structurally lacks that this third voice contributes>"
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
            + recency_clause
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

{findings_section}

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

    _sel_prov = []
    raw = call_operator(
        "gravitywell",
        prompt,
        system=system_msg,
        temperature=0.4,
        timeout=300,
        json_mode=True,
        on_wake_fail="sonnet",
        log=log,
        _provenance_out=_sel_prov,
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

    # Build resolution map from roster: normalize fuzzy ids to canonical
    _canonical_ids = {r["character_id"] for r in roster}
    _resolution = {}
    for cid in _canonical_ids:
        _resolution[cid] = cid
        parts = cid.split("-")
        for i in range(1, len(parts)):
            suffix = "-".join(parts[i:])
            if suffix not in _resolution:
                _resolution[suffix] = cid

    active_data = data
    resolved = []
    unresolved = []

    # Step 2: fuzzy-resolve before raising
    for sid in selected:
        if not isinstance(sid, str):
            unresolved.append(sid)
            continue
        if find_card_path(sid):
            resolved.append(sid)
            continue
        if sid in _resolution and find_card_path(_resolution[sid]):
            resolved.append(_resolution[sid])
            continue
        unresolved.append(sid)

    # Step 3: retry-once if any ids are still unresolvable
    if unresolved:
        retry_note = (
            f"The following id(s) you returned are NOT in the roster and cannot be "
            f"resolved: {unresolved!r}. Choose ONLY from the character_ids listed in "
            f"the ROSTER. Return the same JSON shape."
        )
        retry_prompt = prompt + f"\n\n--- CORRECTION NEEDED ---\n{retry_note}"
        raw2 = call_operator(
            "gravitywell",
            retry_prompt,
            system=system_msg,
            temperature=0.4,
            timeout=300,
            json_mode=True,
            on_wake_fail="sonnet",
            log=log,
            _provenance_out=_sel_prov,
        )
        data2 = _extract_json(raw2 or "")
        active_data = data2
        selected2 = data2.get("selected") or []
        resolved = []
        for sid in selected2:
            if isinstance(sid, str) and find_card_path(sid):
                resolved.append(sid)
            elif isinstance(sid, str) and sid in _resolution and find_card_path(_resolution[sid]):
                resolved.append(_resolution[sid])
            else:
                raise RuntimeError(
                    f"Selected unknown character_id after retry: {sid!r} "
                    f"(original unresolvable: {unresolved!r})"
                )
        if len(resolved) != n:
            raise RuntimeError(
                f"Retry returned {len(resolved)} entities, expected {n}: {resolved!r}"
            )

    # Step 4: duplicate check and return
    if len(set(resolved)) != len(resolved):
        raise RuntimeError(
            f"Selection returned duplicate entities: {resolved}"
        )

    # Derive selection_operator from provenance (last success entry).
    # call_operator with on_wake_fail="sonnet" always appends a success entry or raises,
    # so selection_operator is guaranteed to be set (never "unknown").
    selection_operator = "unknown"
    if _sel_prov:
        for reason, operator in reversed(_sel_prov):
            if reason == "success":
                selection_operator = operator
                break

    selection_degraded = selection_operator != "gravitywell"

    return {
        "selected": resolved,
        "reasoning": active_data.get("reasoning", ""),
        "selection_operator": selection_operator,
        "selection_degraded": selection_degraded,
    }


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
# Deliberation runtime — v0.next helpers
# ---------------------------------------------------------------------------


def _render_transcript(turns: list[dict]) -> str:
    """Render deliberation_turn entries as transcript text for cast prompts.

    Filters to turns where type == "deliberation_turn" (whitelist per M6 in v3->v4).
    Both _render_transcript and turns_used derivation use this whitelist — they are
    coupled by design; changing one requires changing the other.

    Returns "(no turns recorded)" for empty filtered list (per M7 in v2->v3).
    """
    filtered = [t for t in turns if t.get("type") == "deliberation_turn"]
    if not filtered:
        return "(no turns recorded)"
    return "\n\n".join(
        f"[{t.get('speaker', 'unknown')}] {t.get('content', '')}"
        for t in filtered
    )


def _aggregate_positions(positions: list[dict], turns_used: int, turns_cap: int) -> dict:
    """Compute confidence + derived views from per-voice position list.

    Returns {"confidence": str, "stood_aside": list[dict], "blocks": list[dict]}.

    Aggregator table (per §Scope):
      all agree               -> converged
      agree + stand-aside     -> converged-with-reservation
      any block, turns remain -> partial
      any block, turns at cap -> laid-down
    """
    stood_aside = [
        {"voice": p["voice"], "reason": p.get("reason", "")}
        for p in positions
        if p.get("position") == "stand-aside"
    ]
    blocks = [
        {"voice": p["voice"], "basis": p.get("basis", "")}
        for p in positions
        if p.get("position") == "block"
    ]

    if blocks:
        confidence = "laid-down" if turns_used >= turns_cap else "partial"
    elif stood_aside:
        confidence = "converged-with-reservation"
    else:
        confidence = "converged"

    return {"confidence": confidence, "stood_aside": stood_aside, "blocks": blocks}


def _extract_invariants_implicated(synthesis_text: str, positions: list[dict]) -> list[str]:
    """Extract kernel invariant ids from synthesis text and position bases.

    Three matched forms (per §Scope — regex shapes enumerated):
      kernel.invariant.N  — ascii ref (e.g. kernel.invariant.4)
      Invariant N         — prose ref, case-insensitive
      invariant (N)       — paren-form, case-insensitive

    Returns sorted unique list of ids as strings, e.g. ["4", "8"].
    May return [] — expected for many runs. Empty result does NOT affect actionable.
    """
    patterns = [
        r"kernel\.invariant\.([1-8])\b",
        r"(?i)Invariant\s+([1-8])\b",
        r"(?i)invariant\s+\(([1-8])\)",
    ]
    found: set[int] = set()
    texts = [synthesis_text]
    for p in positions:
        basis = p.get("basis") or ""
        texts.append(str(basis))
    combined = " ".join(texts)
    for pattern in patterns:
        for m in re.finditer(pattern, combined):
            found.add(int(m.group(1)))
    return [str(n) for n in sorted(found)]


def _cast_positions(run: dict, entities, adapter) -> list[dict]:
    """Cast per-voice positions on the synthesis. Returns list of position dicts.

    Handles both stub (COUNCIL_ENGINE_STUB=1) and real paths:

    Stub path: returns fixture positions from COUNCIL_STUB_POSITIONS env var.
      Default "agree,agree". Auto-fills reason/basis for stand-aside/block per
      stub safety contract (L11 in v1->v2). No LLM calls.

    Real path: calls entity.act(cast_prompt, ctx) for each character voice,
      parses JSON response, validates per Invariants 4-5.

    Narrator entities (role=="narrator") are excluded — they don't vote.
    entities parameter is ignored in stub mode (env-var fast-path).
    """
    character_sels = [
        sel for sel in run.get("selected_entities", [])
        if sel.get("role") != ROLE_NARRATOR
    ]

    if os.environ.get("COUNCIL_ENGINE_STUB") == "1":
        stub_env = os.environ.get("COUNCIL_STUB_POSITIONS", "agree,agree")
        raw_positions = [p.strip() for p in stub_env.split(",")]
        result = []
        for i, sel in enumerate(character_sels):
            pos = raw_positions[i] if i < len(raw_positions) else "agree"
            if pos == "stand-aside":
                entry: dict = {
                    "voice": sel["id"],
                    "position": "stand-aside",
                    "reason": "[stub] stand-aside reason",
                    "basis": None,
                }
            elif pos == "block":
                entry = {
                    "voice": sel["id"],
                    "position": "block",
                    "reason": "[stub] block reason",
                    "basis": "kernel.invariant.1",
                }
            else:
                entry = {
                    "voice": sel["id"],
                    "position": "agree",
                    "reason": "",
                    "basis": None,
                }
            result.append(entry)
        return result

    # Real path: build cast prompt per voice, call entity.act, parse+validate.
    synthesis = run.get("synthesis", {})
    synth_landing = synthesis.get("landing", "")
    synth_questions = synthesis.get("open_questions", [])
    transcript = _render_transcript(run.get("turns", []))
    questions_text = (
        "\n".join(f"- {q}" for q in synth_questions) if synth_questions else "(none)"
    )

    try:
        from lapis_engine import RunContext  # type: ignore
    except ImportError:
        RunContext = None  # type: ignore

    result = []
    for i, sel in enumerate(character_sels):
        entity = entities[i] if entities and i < len(entities) else None
        if entity is None:
            raise RuntimeError(
                f"entity is None for {sel['id']!r} in non-stub _cast_positions"
            )

        other_ids = [s["id"] for j, s in enumerate(character_sels) if j != i]
        other_str = ", ".join(other_ids) if other_ids else "the other participant"

        cast_prompt = (
            f"You have just deliberated alongside {other_str}. "
            "Here is the full exchange:\n\n"
            "--- TRANSCRIPT ---\n"
            f"{transcript}\n"
            "--- END TRANSCRIPT ---\n\n"
            "The synthesis of the deliberation reads:\n\n"
            f"{synth_landing}\n\n"
            "Open questions surfaced:\n"
            f"{questions_text}\n\n"
            'Cast your position on this synthesis using the consensus-process taxonomy:\n\n'
            '- "agree" \u2014 you accept the synthesis as it stands.\n'
            '- "stand-aside" \u2014 you don\'t endorse the synthesis but you let the group '
            'proceed; provide a 1-2 sentence reason explaining what you\'d register as '
            'concern. You may cite a kernel invariant (e.g. "kernel.invariant.4") in your '
            'basis when the concern names a specific invariant.\n'
            '- "block" \u2014 you have a firm conviction the synthesis does not serve the '
            'whole; provide a basis (a kernel invariant id like "kernel.invariant.4" or '
            "evidence reference).\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "position": "agree" | "stand-aside" | "block",\n'
            '  "reason": "<your reason, 1-2 sentences; required for stand-aside; '
            'recommended for block; optional for agree>",\n'
            '  "basis": "<kernel invariant id or evidence reference; required for block; '
            'allowed for stand-aside; null for agree>"\n'
            "}"
        )

        ctx = (
            RunContext(
                step=len(run.get("turns", [])),
                entity_ids=[s["id"] for s in character_sels],
            )
            if RunContext is not None
            else None
        )
        raw = entity.act(cast_prompt, ctx)

        data = _extract_json(raw)
        position = data.get("position", "")
        reason = data.get("reason") or ""
        basis = data.get("basis")

        if position not in ("agree", "stand-aside", "block"):
            raise RuntimeError(
                f"_cast_positions: unknown position {position!r} from {sel['id']!r}"
            )
        if position == "stand-aside" and len(reason) < 5:
            raise RuntimeError(
                f"_cast_positions: stand-aside requires non-empty reason (\u22655 chars) "
                f"from {sel['id']!r}; got {reason!r}"
            )
        if position == "block" and (not basis or len(str(basis)) < 5):
            raise RuntimeError(
                f"_cast_positions: block requires non-null basis (\u22655 chars) "
                f"from {sel['id']!r}; got {basis!r}"
            )

        result.append({
            "voice": sel["id"],
            "position": position,
            "reason": reason,
            "basis": basis,
        })

    return result


def run_deliberation(run_id: str) -> None:
    """Read /<COUNCIL_DIR>/<run_id>.yaml, drive the engine, write turns +
    synthesis + terminal status back to the same file.  Mutates in place via
    save_run; returns None.

    Does NOT call send_notification.  Council completion is informational per
    Pushover policy; if the caller wants a Pushover, they pass --notify to
    the submitter, which sets notify=True on the queue task and lets the
    runner's notify_completion/notify_failure handle it.

    COUNCIL_ENGINE_STUB=1: skip LLM calls entirely, write fixture turn/synthesis.
    COUNCIL_STUB_POSITIONS: comma-separated positions for stub cast (default: agree,agree).
    """
    from agents_core.council import cache as _cache

    run = load_run(run_id)
    mode = run.get("mode", DEFAULT_MODE)
    kernel_version = _cache.read_kernel_version()

    if os.environ.get("COUNCIL_ENGINE_STUB") == "1":
        first_id = (run["selected_entities"][0]["id"]
                    if run.get("selected_entities") else "stub-entity")
        run["turns"].append(
            {
                "step": 1,
                "speaker": first_id,
                "type": "deliberation_turn",  # was "dialogue" — H1 fix per v3->v4
                "content": "[STUB] turn output",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if mode == "scene":
            run["status"] = "closed"
            run["completed_at"] = datetime.now().isoformat(timespec="seconds")
            save_run(run)
            return
        # Deliberation stub: write synthesis turn, parse synthesis, then shared tail.
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
        # Stub mode voicing provenance: set effective_voicing to match requested (no actual operation)
        run["effective_voicing"] = run.get("voicing", "sonnet")
        run["voicing_degraded"] = False
        # Common deliberation tail — MUST run in stub mode too (H2 fix per v3->v4).
        # entities=None is safe: _cast_positions uses env-var fast-path in stub mode.
        _apply_position_cast_tail(
            run, entities=None, adapter=None,
            kernel_version=kernel_version, cache=_cache,
        )
        run["completed_at"] = datetime.now().isoformat(timespec="seconds")
        run["paid_spend"] = _calculate_paid_spend(run)
        save_run(run)
        return

    # Late imports: keep lapis-engine load function-local so importing
    # agents_core.council doesn't force-load lapis-engine at queue startup.
    from lapis_engine import (  # type: ignore
        ClaudeAdapter,
        DeliberationDirector,
        Engine,
        LlamaAdapter,
        SceneDirector,
        StepData,
    )
    from archetypes.engine.character_entity import CharacterEntity  # type: ignore
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

        # Record effective voicing from adapter (gravitywell or fallback)
        _apply_voicing_provenance(run, adapter)

        # Emit in-stream voicing-degradation signal (Leg A of gw-voicing-stdout-signal-v0)
        _emit_voicing_signal(run, adapter, run_id)

        if mode == "scene":
            run["status"] = "closed"
        else:
            synth_content = run["turns"][-1]["content"] if run["turns"] else ""
            run["synthesis"] = _parse_synthesis(synth_content)
            # Common deliberation tail — runs in real mode.
            _apply_position_cast_tail(
                run, entities=entities, adapter=adapter,
                kernel_version=kernel_version, cache=_cache,
            )
        run["completed_at"] = datetime.now().isoformat(timespec="seconds")
        run["paid_spend"] = _calculate_paid_spend(run)
        save_run(run)
    except Exception as e:
        run["status"] = "failed"
        run["error"] = f"{type(e).__name__}: {e}"
        run["completed_at"] = datetime.now().isoformat(timespec="seconds")
        run["paid_spend"] = _calculate_paid_spend(run)
        save_run(run)
        raise


def _emit_voicing_signal(run: dict, adapter, run_id: str) -> None:
    """Emit in-stream voicing-degradation signal (Leg A of gw-voicing-stdout-signal-v0).

    Emits a single structured line per run when GW was requested:
    - Degraded: "[council] VOICING DEGRADED run_id=<id> requested=gravitywell effective=<op> reason=<reason>"
    - Clean: "[council] voicing ok run_id=<id> effective=gravitywell" (only if adapter produced voicing_events)

    Does not emit for non-GW runs (noise reduction).
    """
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    if run.get("voicing") == "gravitywell":
        requested = run.get("voicing", "unknown")
        if run["voicing_degraded"] is True:
            reason = run.get("voicing_degraded_reason", "unknown")
            effective = run.get("effective_voicing", "unknown")
            print(f"[council] VOICING DEGRADED run_id={run_id} requested={requested} effective={effective} reason={reason}", flush=True)
        elif run["voicing_degraded"] is False and run.get("effective_voicing") == "gravitywell":
            # SENTINEL guard: only emit positive line if adapter actually produced voicing_events
            # (avoids spurious "voicing ok" from the no-events path in _apply_voicing_provenance)
            if isinstance(adapter, GravityWellAdapter) and adapter.voicing_events:
                print(f"[council] voicing ok run_id={run_id} effective=gravitywell", flush=True)


def _apply_voicing_provenance(run: dict, adapter) -> None:
    """Record effective voicing in the run record based on adapter provenance.

    For GravityWellAdapter, reads voicing_events list and updates run["effective_voicing"]
    and related fields. For other adapters, sets effective_voicing to match requested voicing.

    Mutates run in-place, adding/updating:
    - effective_voicing: the operator that actually answered (e.g., "gravitywell" or "sonnet")
    - voicing_degraded: bool, True if effective != requested
    - voicing_degraded_reason: the reason for degradation (if any)
    - Per-turn effective_voicing keys in run["turns"][]
    """
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    requested_voicing = run.get("voicing", "sonnet")

    if isinstance(adapter, GravityWellAdapter):
        if adapter.voicing_events:
            # Aggregate voicing events: check if all are gravitywell (clean) or mixed
            operators = [e.get("effective_operator") for e in adapter.voicing_events]
            reasons = [e.get("reason") for e in adapter.voicing_events]

            # Determine effective_voicing: if all are gravitywell, it's gravitywell; else fallback
            if all(op == "gravitywell" for op in operators):
                run["effective_voicing"] = "gravitywell"
                run["voicing_degraded"] = False
            else:
                # Mixed or all fallback - pick the first non-gravitywell operator
                non_gw = [op for op in operators if op != "gravitywell"]
                run["effective_voicing"] = non_gw[0] if non_gw else "unknown"
                run["voicing_degraded"] = True
                # Collect unique failure reasons (excluding "success" which indicates clean calls)
                failure_reasons = []
                for r in reasons:
                    if r not in failure_reasons and r != "success":
                        failure_reasons.append(r)
                # Pick the most specific reason (prioritize by specificity)
                if "doorman_unreachable" in failure_reasons:
                    run["voicing_degraded_reason"] = "doorman_unreachable"
                elif "serving_http_error" in failure_reasons:
                    run["voicing_degraded_reason"] = "serving_http_error"
                elif "gw_not_serving" in failure_reasons:
                    run["voicing_degraded_reason"] = "gw_not_serving"
                else:
                    run["voicing_degraded_reason"] = failure_reasons[0] if failure_reasons else "unknown"

            # Add per-turn effective_voicing keys
            for i, turn in enumerate(run.get("turns", [])):
                if i < len(adapter.voicing_events):
                    turn["effective_voicing"] = adapter.voicing_events[i].get("effective_operator")
        else:
            # No voicing events recorded - shouldn't happen in real runs, assume success
            run["effective_voicing"] = "gravitywell"
            run["voicing_degraded"] = False
    elif adapter is not None:
        # Non-GravityWell adapters: effective == requested
        run["effective_voicing"] = requested_voicing
        run["voicing_degraded"] = False
    # If adapter is None, voicing provenance should already be set by caller (stub mode)


def _apply_position_cast_tail(
    run: dict, entities, adapter, kernel_version: str, cache
) -> None:
    """Shared deliberation tail: cast positions, aggregate, update synthesis, write cache.

    Runs in BOTH stub and real branches of run_deliberation (H2 fix per v3->v4).
    Mutates run["synthesis"] in-place with all v0.next keys.
    Aggregator output overrides parsed confidence (Invariant 11).
    Sets run["status"] from the updated synthesis.
    """
    positions = _cast_positions(run, entities, adapter)

    # turns_used: whitelist filter "deliberation_turn" only (per H3 in v2->v3 and M6).
    # Coupled to _render_transcript whitelist — both must stay in sync.
    turns_used = len([
        t for t in run.get("turns", [])
        if t.get("type") == "deliberation_turn"
    ])
    agg = _aggregate_positions(
        positions, turns_used, turns_cap=int(run.get("turns_cap", 8))
    )

    synthesis = run["synthesis"]

    inv_text = (
        synthesis.get("landing", "") + " " +
        " ".join(synthesis.get("open_questions", []))
    )
    synthesis["invariants_implicated"] = _extract_invariants_implicated(inv_text, positions)

    synthesis["evidence"] = [
        {"voice": p["voice"], "claim": p.get("reason", ""), "basis": p.get("basis")}
        for p in positions
        if p.get("reason") or p.get("basis")
    ]

    synthesis["positions"] = positions
    synthesis["stood_aside"] = agg["stood_aside"]
    synthesis["blocks"] = agg["blocks"]

    # Aggregator override of confidence — Invariant 11.
    synthesis["confidence"] = agg["confidence"]
    synthesis["actionable"] = agg["confidence"] in ("converged", "converged-with-reservation")
    synthesis["output_class"] = "cohesion-finding" if synthesis["actionable"] else "none"

    run["status"] = _status_from_synthesis(synthesis)

    if synthesis["output_class"] == "cohesion-finding":
        cache.write_finding(run, kernel_version)


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
    if voicing == "gravitywell":
        return GravityWellAdapter(temperature=0.8)
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


def _calculate_paid_spend(run: dict) -> bool:
    """Derive paid_spend flag from selection and voicing provenance.

    Returns True if the run incurred paid spend:
    - selector fell back to a paid model, or
    - voicing fell back to a paid model, or
    - voicing explicitly requested a paid model, or
    - selection explicitly used a paid model.
    """
    return bool(
        run.get("selection_degraded")
        or run.get("voicing_degraded")
        or run.get("voicing") in ("haiku", "sonnet", "opus")
        or run.get("selection_voicing") in ("haiku", "sonnet", "opus")
    )


def _status_from_synthesis(synthesis: dict) -> str:
    """Map synthesis confidence to run status.

    v0.next extended mapping (per H1 in v1->v2 and §Scope pseudocode):
      converged                  -> resolved
      converged-with-reservation -> resolved
      laid-down                  -> laid-down
      diverged                   -> open  (backward-compat for pre-v0.next run YAMLs)
      partial / <anything else>  -> open
    """
    conf = synthesis.get("confidence", "partial")
    if conf in ("converged", "converged-with-reservation"):
        return "resolved"
    if conf == "laid-down":
        return "laid-down"
    if conf == "diverged":  # backward-compat for pre-v0.next run YAMLs — Invariant 10
        return "open"
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
    if mode == "deliberation" and n not in (2, 3):
        raise ValueError("deliberation mode requires n in (2, 3)")
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
        if len(selected) == 3:
            return [
                {"id": selected[0], "role": "first_voice"},
                {"id": selected[1], "role": "second_voice"},
                {"id": selected[2], "role": "third_voice"},
            ]
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
        if narrator:
            n = 3
        elif mode == "deliberation":
            n = 3
        else:
            n = 2
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

    recent_pair_ids = recent_pair_entities()
    if recent_pair_ids:
        print(
            f"[council] recency penalty: down-weighting {', '.join(recent_pair_ids)}",
            flush=True,
        )

    print("[council] selecting entities (via gravitywell, fallback=sonnet)...", flush=True)
    selection = select_entities(
        decision=args.decision,
        roster=roster,
        context=context,
        n=n_characters,
        with_entity=args.with_entity,
        mode=mode,
        recent_pair_ids=recent_pair_ids,
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
        "selection_voicing": selection.get("selection_operator", "unknown"),
        "selection_degraded": selection.get("selection_degraded", False),
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
        "priority": 50,  # Priority.NORMAL
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
        "--voicing", choices=["local", "gravitywell", "haiku", "sonnet", "opus"],
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
