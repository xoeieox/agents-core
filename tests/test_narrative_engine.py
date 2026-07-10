"""Tests for agents_core.narrative — engine, audiences, CLI emit.

Ten test cases per spec §Deliverables 5:
  1.  Audience profile registration — all four slugs resolve; unknown slug raises KeyError.
  2.  Source resolution — missing extra source raises FileNotFoundError.
  3.  Prompt determinism — identical inputs produce byte-identical prompt; prompt_hash matches.
  4.  --dry-run — ClaudeQueue.submit NOT called; prompt printed; exit 0.
  5.  Front-matter shape — expected YAML keys in expected order, draft body follows.
  6.  Output path defaulting — without --out, resolves to /srv/lapis/narratives/<aud>/<date>-<slug>.md.
  7.  Audience-conditional sources — kyma-marketing loads 3 extra; sustainer-update loads 0 extra.
  8.  Source sha256 stability — engine returns same digest as independent computation.
  9.  Chub silent-empty return becomes loud — mock returns "" → RuntimeError; submit never called.
  10. Front-matter has schema_version and caller at fixed positions.

Live ClaudeQueue calls are NOT exercised here (mocked throughout).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


_CANNED_VAULT: dict[str, str] = {
    "Lapis/Constitution-Kernel.md": "# Constitution Kernel\nInvariant content here.\n",
    "Lapis/Lapis-Philosophy.md": "# Philosophy\nFlame/mirror content.\n",
    "Lapis/Vision-Bidirectional-Mirror.md": "# Vision\nBidirectional mirror.\n",
    "Lapis/Vision-Graduated-Commons.md": "# Graduated Commons\nCommons content.\n",
    "Lapis/Vision-Commons-and-Commerce.md": "# Commons and Commerce\nContent.\n",
    "Lapis/Products/Kyma/MOC-Kyma.md": "# Kyma MOC\nContent.\n",
    "Lapis/Products/Kyma/Documentation/MVP-Scope.md": "# MVP Scope\nContent.\n",
    "Lapis/Products/Kyma/Documentation/Full-Loop-Vision.md": "# Full Loop\nContent.\n",
    "Lapis/State-of-Lapis-2026-04-25.md": "# State of Lapis\nContent.\n",
    "Lapis/Build-Stack-Roadmap.md": "# Build Stack\nContent.\n",
}
_CANNED_CHUB = "# Lapis Ecosystem\nChub bundle content.\n"


def _vault_read(rel_path: str) -> str:
    if rel_path in _CANNED_VAULT:
        return _CANNED_VAULT[rel_path]
    raise FileNotFoundError(f"Vault source not found: {rel_path}")


def _make_emit_result_stub():
    """Return a minimal EmitResult for writer tests (no queue call)."""
    from agents_core.narrative.engine import EmitResult, SourceRef
    return EmitResult(
        draft="This is the draft body.\n",
        audience="grants",
        ask="Draft a 400-word NLNet pitch.",
        length_target="medium",
        sources=[
            SourceRef(path="Lapis/Constitution-Kernel.md", sha256="abc123"),
            SourceRef(path="<chub:conductor/lapis-ecosystem>", sha256="def456"),
        ],
        prompt_hash="deadbeef",
        model="claude-opus-4-7",
        dispatched_at="2026-05-06T19:42:11+00:00",
        returned_at="2026-05-06T19:42:48+00:00",
    )


# ---------------------------------------------------------------------------
# Test 1 — Audience profile registration
# ---------------------------------------------------------------------------

def test_audience_profile_registration_all_four():
    from agents_core.narrative.audiences import AUDIENCE_REGISTRY
    for slug in ("grants", "kyma-marketing", "sustainer-update", "dev-docs"):
        assert slug in AUDIENCE_REGISTRY, f"Missing slug: {slug}"
        profile = AUDIENCE_REGISTRY[slug]
        assert profile.slug == slug


def test_audience_profile_unknown_slug_raises():
    from agents_core.narrative.engine import emit_draft
    with pytest.raises(KeyError):
        with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
             patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
            emit_draft("nonexistent-audience", "some ask")


# ---------------------------------------------------------------------------
# Test 2 — Source resolution: missing extra source raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_source_resolution_missing_extra_source_raises(tmp_path):
    """Stub a profile whose extra_sources points to a non-existent file."""
    from agents_core.narrative.engine import emit_draft

    def vault_read_with_missing(rel_path):
        if rel_path == "Lapis/NonExistent-Source.md":
            raise FileNotFoundError(rel_path)
        return _vault_read(rel_path)

    # Patch audience registry to inject a profile with a bad extra source
    from agents_core.narrative.audiences import AudienceProfile, AUDIENCE_REGISTRY
    bad_profile = AudienceProfile(
        slug="grants",
        title="Test",
        frame="f",
        register="r",
        emphasize=["e"],
        deemphasize=["d"],
        example_callouts=["c"],
        extra_sources=["Lapis/NonExistent-Source.md"],
    )
    patched_registry = {**AUDIENCE_REGISTRY, "grants": bad_profile}

    with patch("agents_core.narrative.engine.AUDIENCE_REGISTRY", patched_registry), \
         patch("agents_core.narrative.engine._read_vault", side_effect=vault_read_with_missing), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        with pytest.raises(FileNotFoundError):
            emit_draft("grants", "some ask")


# ---------------------------------------------------------------------------
# Test 3 — Prompt determinism
# ---------------------------------------------------------------------------

def test_prompt_determinism():
    from agents_core.narrative.engine import emit_draft

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        r1 = emit_draft("grants", "Draft a 400-word pitch.", dry_run=True)
        r2 = emit_draft("grants", "Draft a 400-word pitch.", dry_run=True)

    assert r1._prompt == r2._prompt, "Prompts must be byte-identical"
    assert r1.prompt_hash == r2.prompt_hash, "prompt_hash must match"
    assert r1.prompt_hash == _sha256(r1._prompt), "prompt_hash must equal sha256 of prompt"


# ---------------------------------------------------------------------------
# Test 4 — --dry-run: ClaudeQueue.submit NOT called; prompt printed; exit 0
# ---------------------------------------------------------------------------

def test_dry_run_does_not_call_submit(capsys):
    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB), \
         patch("agents_core.narrative.engine.ClaudeQueue") as mock_q_cls:
        from agents_core.narrative.emit import main
        ret = main(["--audience", "grants", "--ask", "A dry-run ask.", "--dry-run"])

    assert ret == 0
    mock_q_cls.return_value.submit.assert_not_called()
    captured = capsys.readouterr()
    assert "A dry-run ask." in captured.out or len(captured.out) > 50  # prompt was printed


# ---------------------------------------------------------------------------
# Test 5 — Front-matter shape
# ---------------------------------------------------------------------------

def test_front_matter_shape(tmp_path):
    from agents_core.narrative.emit import _build_front_matter
    result = _make_emit_result_stub()
    fm = _build_front_matter(result)

    assert fm.startswith("---\n")
    assert fm.strip().endswith("---")
    # Parse the YAML between the fences
    inner = fm[4:fm.rindex("---")].strip()
    data = yaml.safe_load(inner)

    assert data["schema_version"] == "narrative-emit-v0"
    assert data["caller"] == "narrative-emit"
    assert data["audience"] == "grants"
    assert data["ask"] == "Draft a 400-word NLNet pitch."
    assert data["length_target"] == "medium"
    assert data["model"] == "claude-opus-4-7"
    assert data["prompt_hash"] == "deadbeef"
    assert isinstance(data["sources"], list)
    assert data["sources"][0]["path"] == "Lapis/Constitution-Kernel.md"
    assert data["sources"][0]["sha256"] == "abc123"


def test_front_matter_followed_by_draft_body(tmp_path):
    from agents_core.narrative.emit import _build_front_matter
    result = _make_emit_result_stub()
    fm = _build_front_matter(result)
    full_output = fm + "\n" + result.draft
    assert "This is the draft body." in full_output


# ---------------------------------------------------------------------------
# Test 6 — Output path defaulting
# ---------------------------------------------------------------------------

def test_output_path_defaulting():
    from agents_core.narrative.emit import _default_out
    out = _default_out("grants", "Draft a 400-word NLNet pitch for open call.")
    today = date.today().isoformat()
    assert str(out).startswith(f"/srv/lapis/narratives/grants/{today}-")
    assert out.suffix == ".md"
    # Slug is at most 40 chars (before .md)
    slug = out.stem.replace(f"{today}-", "")
    assert len(slug) <= 40


def test_output_path_parent_created(tmp_path):
    """CLI creates parent directory if missing."""
    from agents_core.narrative.emit import _write_output
    result = _make_emit_result_stub()
    out_path = tmp_path / "narratives" / "grants" / "2026-05-06-test.md"
    assert not out_path.parent.exists()
    _write_output(result, out_path)
    assert out_path.exists()


# ---------------------------------------------------------------------------
# Test 7 — Audience-conditional sources
# ---------------------------------------------------------------------------

def test_kyma_marketing_loads_three_extra_sources():
    from agents_core.narrative.engine import emit_draft
    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        result = emit_draft("kyma-marketing", "A kyma ask.", dry_run=True)

    # 4 canonical + 3 kyma extras = 7 sources
    paths = [ref.path for ref in result.sources]
    assert len(paths) == 7
    assert "Lapis/Products/Kyma/MOC-Kyma.md" in paths
    assert "Lapis/Products/Kyma/Documentation/MVP-Scope.md" in paths
    assert "Lapis/Products/Kyma/Documentation/Full-Loop-Vision.md" in paths


def test_sustainer_update_loads_only_canonical_four():
    from agents_core.narrative.engine import emit_draft
    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        result = emit_draft("sustainer-update", "A sustainer ask.", dry_run=True)

    assert len(result.sources) == 4
    paths = [ref.path for ref in result.sources]
    assert "Lapis/Constitution-Kernel.md" in paths
    assert "<chub:conductor/lapis-ecosystem>" in paths
    assert "Lapis/Lapis-Philosophy.md" in paths
    assert "Lapis/Vision-Bidirectional-Mirror.md" in paths


# ---------------------------------------------------------------------------
# Test 8 — Source sha256 stability
# ---------------------------------------------------------------------------

def test_source_sha256_stability():
    from agents_core.narrative.engine import emit_draft
    expected_digest = _sha256(_CANNED_VAULT["Lapis/Constitution-Kernel.md"])

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        result = emit_draft("grants", "Test ask.", dry_run=True)

    kernel_ref = next(r for r in result.sources if r.path == "Lapis/Constitution-Kernel.md")
    assert kernel_ref.sha256 == expected_digest


# ---------------------------------------------------------------------------
# Test 9 — Chub silent-empty return becomes loud RuntimeError
# ---------------------------------------------------------------------------

def test_chub_empty_return_raises_runtime_error():
    from agents_core.narrative.engine import emit_draft

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=""), \
         patch("agents_core.narrative.engine.ClaudeQueue") as mock_q_cls:
        with pytest.raises(RuntimeError, match="conductor/lapis-ecosystem"):
            emit_draft("grants", "Some ask.")

    mock_q_cls.return_value.submit.assert_not_called()


def test_chub_loader_empty_triggers_loud_failure_at_engine_level():
    """The engine raises on empty even if _load_chub itself returns ''."""
    from agents_core.narrative.engine import emit_draft

    # Directly mock _load_chub to simulate the silent-fail scenario
    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=""), \
         patch("agents_core.narrative.engine.ClaudeQueue") as mock_q:
        with pytest.raises(RuntimeError) as exc_info:
            emit_draft("dev-docs", "Some ask.")

    assert "conductor/lapis-ecosystem" in str(exc_info.value)
    mock_q.return_value.submit.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10 — Front-matter has schema_version and caller at fixed positions
# ---------------------------------------------------------------------------

def test_front_matter_schema_version_and_caller_present():
    from agents_core.narrative.emit import _build_front_matter
    result = _make_emit_result_stub()
    fm = _build_front_matter(result)

    lines = fm.split("\n")
    # First non-fence line should be schema_version
    content_lines = [l for l in lines if l and l != "---"]
    assert content_lines[0].startswith("schema_version:"), (
        f"schema_version must be first key; got: {content_lines[0]!r}"
    )
    assert "narrative-emit-v0" in content_lines[0]
    # Second line should be caller
    assert content_lines[1].startswith("caller:"), (
        f"caller must be second key; got: {content_lines[1]!r}"
    )
    assert "narrative-emit" in content_lines[1]


# ---------------------------------------------------------------------------
# Atomic spec write / submit / rename (agents-core-narrative-emit-atomic-submit-v0)
# ---------------------------------------------------------------------------

def _fake_queue_cls(submit_fn):
    """Build a ClaudeQueue stand-in whose submit() behavior is injected."""
    class _FakeQueue:
        def __init__(self, _queue_dir):
            pass

        def submit(self, task_dict, task_id=None):
            return submit_fn(task_dict, task_id)

    return _FakeQueue


def _short_circuit_poll(monkeypatch):
    """Make the post-submit poll loop time out on its first check instead of
    waiting out the real 360s deadline. The deadline calc consumes the first
    time.monotonic() call; the while-condition check consumes the second."""
    from agents_core.narrative import engine
    calls = iter([0, 10**9])
    monkeypatch.setattr(engine.time, "monotonic", lambda: next(calls))


def test_submit_failure_cleans_up_and_reraises(tmp_path, monkeypatch):
    from agents_core.narrative import engine
    from agents_core.narrative.engine import emit_draft

    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(
        engine, "ClaudeQueue",
        _fake_queue_cls(lambda task_dict, task_id: (_ for _ in ()).throw(RuntimeError("submit boom"))),
    )

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        with pytest.raises(RuntimeError, match="submit boom"):
            emit_draft("grants", "Some ask.", _queue_dir=queue_dir)

    pending_dir = queue_dir / "pending"
    assert list(pending_dir.iterdir()) == [], (
        "no final json and no stray .tmp file should remain after submit() failure"
    )


def test_rename_failure_after_submit_success_cleans_up_and_reraises(tmp_path, monkeypatch):
    from agents_core.narrative import engine
    from agents_core.narrative.engine import emit_draft

    queue_dir = tmp_path / "queue"
    # submit() returns the local task_id it was passed (a normal, non-deduped
    # submission), but the subsequent os.replace() rename itself raises.
    monkeypatch.setattr(
        engine, "ClaudeQueue",
        _fake_queue_cls(lambda task_dict, task_id: task_id),
    )
    monkeypatch.setattr(engine.os, "replace", MagicMock(side_effect=OSError("replace boom")))

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        with pytest.raises(OSError, match="replace boom"):
            emit_draft("grants", "Some ask.", _queue_dir=queue_dir)

    pending_dir = queue_dir / "pending"
    assert list(pending_dir.iterdir()) == [], (
        "temp file must be cleaned up even when submit() succeeded but the rename failed"
    )


def test_dedup_submit_mismatch_does_not_rename_temp_to_final(tmp_path, monkeypatch):
    """submit() returning a different (shared) task_id must not create an
    orphaned final json - the write/submit/rename block must not rename."""
    from agents_core.narrative import engine
    from agents_core.narrative.engine import emit_draft

    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(
        engine, "ClaudeQueue",
        _fake_queue_cls(lambda task_dict, task_id: "claude_shared_other_task"),
    )
    _short_circuit_poll(monkeypatch)

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        # Not fixed by this spec: the unmodified poll loop still polls this
        # caller's own dispatch_cwd, which a dedup join never writes to, so
        # it still hits the existing 360s (here: short-circuited) TimeoutError.
        with pytest.raises(TimeoutError):
            emit_draft("grants", "Some ask.", _queue_dir=queue_dir)

    pending_dir = queue_dir / "pending"
    assert list(pending_dir.iterdir()) == [], (
        "a dedup-mismatched submit() return must not leave an orphaned final json"
    )


def test_successful_submit_writes_final_json_matching_spec(tmp_path, monkeypatch):
    from agents_core.narrative import engine
    from agents_core.narrative.engine import emit_draft

    queue_dir = tmp_path / "queue"
    captured = {}

    def _submit_fn(task_dict, task_id):
        captured["task_id"] = task_id
        captured["task_dict"] = task_dict
        return task_id

    monkeypatch.setattr(engine, "ClaudeQueue", _fake_queue_cls(_submit_fn))
    _short_circuit_poll(monkeypatch)

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        with pytest.raises(TimeoutError):
            emit_draft("grants", "Some ask.", _queue_dir=queue_dir)

    pending_dir = queue_dir / "pending"
    final_jsons = list(pending_dir.glob("*.json"))
    assert len(final_jsons) == 1
    assert final_jsons[0].name == f"{captured['task_id']}.json"
    spec = json.loads(final_jsons[0].read_text())
    assert spec["task_id"] == captured["task_id"]
    assert spec["task_type"] == "subprocess"
    assert spec["model"] == "opus"
    assert captured["task_dict"]["payload"]["spec_path"] == str(final_jsons[0])


def test_submit_failure_logs_full_traceback(tmp_path, monkeypatch, caplog):
    from agents_core.narrative import engine
    from agents_core.narrative.engine import emit_draft

    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(
        engine, "ClaudeQueue",
        _fake_queue_cls(lambda task_dict, task_id: (_ for _ in ()).throw(RuntimeError("boom-for-log"))),
    )

    with patch("agents_core.narrative.engine._read_vault", side_effect=_vault_read), \
         patch("agents_core.narrative.engine._load_chub", return_value=_CANNED_CHUB):
        with caplog.at_level(logging.ERROR, logger="agents_core.narrative.engine"):
            with pytest.raises(RuntimeError, match="boom-for-log"):
                emit_draft("grants", "Some ask.", _queue_dir=queue_dir)

    assert len(caplog.records) >= 1, "submit() failure must be logged"
    record = caplog.records[-1]
    assert record.exc_info is not None, (
        "log.exception must attach exc_info (full traceback), not just the "
        "exception's string form"
    )
