"""Tests for agents_core.expert — Expert dispatch substrate (expert-substrate-v0).

Test categories:
  Unit:        layer composition, input validation, post-mortem schema
  Integration: dispatch cycle, memory loop, failure paths
  Smoke:       pytest.mark.smoke — real ClaudeQueue dispatch (not run in CI)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_core.expert import (
    RECORD_KINDS,
    ExpertDispatchInput,
    ExpertDispatchResult,
    _compose_system_prompt,
    _run_post_mortem,
    dispatch_expert,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STUB_PERSONA_PATH = (
    Path(__file__).parent / "fixtures" / "experts" / "test-stub" / "persona.md"
)


@pytest.fixture
def experts_root(tmp_path):
    """Tmpdir-rooted experts tree with the test-stub persona installed."""
    root = tmp_path / "experts"
    stub_dir = root / "test-stub"
    stub_dir.mkdir(parents=True)
    shutil.copy(STUB_PERSONA_PATH, stub_dir / "persona.md")
    return root


@pytest.fixture
def queue_root(tmp_path):
    """Tmpdir-rooted claude-queue dir."""
    q = tmp_path / "queue"
    q.mkdir()
    return q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dispatch_with_mock_expert(
    inp: ExpertDispatchInput,
    experts_root: Path,
    queue_root: Path,
    expert_notepad: str = "Some field notes.",
    expert_output: str = "Expert final response.",
    mock_pm_records: str = "records: []",
    simulate_crash: bool = False,
) -> ExpertDispatchResult:
    """Dispatch with the Expert subprocess mocked.

    Patches call_claude_cli (post-mortem) and simulates the Expert by
    writing notepad.md + output.md after the spec JSON appears.
    """
    written_files: list[Path] = []

    def _fake_submit(task, task_id=None):
        # Called by dispatch_expert to enqueue the task.
        # Simulate the Expert subprocess: write output files to dispatch CWD.
        spec_json_path = queue_root / "pending" / f"{task_id}.json"
        spec_data = json.loads(spec_json_path.read_text())
        cwd = Path(spec_data["cwd"])
        if not simulate_crash:
            (cwd / "notepad.md").write_text(expert_notepad)
            (cwd / "output.md").write_text(expert_output)
        # Delete spec to signal shaped_runner completion
        spec_json_path.unlink(missing_ok=True)
        written_files.append(cwd)
        # Return a task_id (matches ClaudeQueue.submit contract)
        return task_id

    with (
        patch("agents_core.expert.ClaudeQueue") as MockQueue,
        patch("agents_core.expert.call_claude_cli", return_value=mock_pm_records),
    ):
        mock_q_instance = MagicMock()
        mock_q_instance.submit.side_effect = _fake_submit
        MockQueue.return_value = mock_q_instance

        result = dispatch_expert(
            inp,
            _experts_root=experts_root,
            _queue_dir=queue_root,
            _poll_interval=0.05,
        )

    return result


# ===========================================================================
# Unit tests — layer composition
# ===========================================================================

class TestLayerComposition:
    """_compose_system_prompt produces correct sections for each layer combo."""

    def test_persona_only(self, experts_root):
        prompt = _compose_system_prompt("test-stub", ("persona",), experts_root)
        assert "test stub Expert" in prompt
        assert "## Your accumulated corpus" not in prompt
        assert "## What you have learned from past dispatches" not in prompt
        assert "field notebook" in prompt
        assert "notepad.md" in prompt
        assert "output.md" in prompt

    def test_persona_corpus_empty(self, experts_root):
        prompt = _compose_system_prompt("test-stub", ("persona", "corpus"), experts_root)
        assert "## Your accumulated corpus" in prompt
        assert "(none yet)" in prompt
        assert "## What you have learned from past dispatches" not in prompt
        assert "field notebook" in prompt

    def test_persona_memory_empty(self, experts_root):
        prompt = _compose_system_prompt("test-stub", ("persona", "memory"), experts_root)
        assert "## Your accumulated corpus" not in prompt
        assert "## What you have learned from past dispatches" in prompt
        assert "(none yet)" in prompt
        assert "field notebook" in prompt

    def test_persona_corpus_memory_empty(self, experts_root):
        prompt = _compose_system_prompt(
            "test-stub", ("persona", "corpus", "memory"), experts_root
        )
        assert "## Your accumulated corpus" in prompt
        assert "## What you have learned from past dispatches" in prompt
        # Both sections have (none yet) because dirs are empty
        assert prompt.count("(none yet)") == 2
        assert "field notebook" in prompt

    def test_corpus_with_seeds(self, experts_root):
        seeds_dir = experts_root / "test-stub" / "seeds"
        seeds_dir.mkdir()
        (seeds_dir / "s1.yaml").write_text("seed: one\n")
        (seeds_dir / "s2.yaml").write_text("seed: two\n")
        prompt = _compose_system_prompt("test-stub", ("persona", "corpus"), experts_root)
        assert "seed: one" in prompt
        assert "seed: two" in prompt
        assert "(none yet)" not in prompt

    def test_memory_with_post_mortems(self, experts_root):
        pm_dir = experts_root / "test-stub" / "post-mortems"
        pm_dir.mkdir()
        (pm_dir / "expert_20260502_120000_0001_test-stub.yaml").write_text(
            "dispatch:\n  outcome: completed\nrecords: []\n"
        )
        # .failed files must be skipped
        (pm_dir / "expert_20260502_120001_0002_test-stub.yaml.failed").write_text(
            "error: something\n"
        )
        prompt = _compose_system_prompt("test-stub", ("persona", "memory"), experts_root)
        assert "outcome: completed" in prompt
        assert "error: something" not in prompt
        assert "(none yet)" not in prompt

    def test_empty_section_renders_as_none_yet_not_omitted(self, experts_root):
        """Invariant: requested but empty layer → '(none yet)', never missing section."""
        prompt_full = _compose_system_prompt(
            "test-stub", ("persona", "corpus", "memory"), experts_root
        )
        prompt_persona = _compose_system_prompt("test-stub", ("persona",), experts_root)
        # Structural difference is ONLY the two empty-body sections
        assert "## Your accumulated corpus" in prompt_full
        assert "## Your accumulated corpus" not in prompt_persona
        assert "(none yet)" in prompt_full
        assert "(none yet)" not in prompt_persona


# ===========================================================================
# Unit tests — input validation
# ===========================================================================

class TestInputValidation:
    def test_invalid_expert_id_uppercase(self, experts_root, queue_root):
        with pytest.raises(ValueError, match="must match"):
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="BadExpert",
                    intent="test",
                    layers=("persona",),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
            )

    def test_invalid_expert_id_starts_with_digit(self, experts_root, queue_root):
        with pytest.raises(ValueError, match="must match"):
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="1invalid",
                    intent="test",
                    layers=("persona",),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
            )

    def test_invalid_expert_id_special_chars(self, experts_root, queue_root):
        with pytest.raises(ValueError, match="must match"):
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test_stub",  # underscores not allowed
                    intent="test",
                    layers=("persona",),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
            )

    def test_valid_expert_id_with_hyphens(self, experts_root, queue_root):
        # Should pass validation (hyphens allowed); will fail on missing persona
        with pytest.raises(ValueError, match="persona.md not found"):
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="my-expert",
                    intent="test",
                    layers=("persona",),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
            )

    def test_persona_not_in_layers_raises(self, experts_root, queue_root):
        with pytest.raises(ValueError, match="persona.*required"):
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub",
                    intent="test",
                    layers=("corpus", "memory"),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
            )

    def test_unknown_layer_raises(self, experts_root, queue_root):
        with pytest.raises(ValueError, match="Unknown layer"):
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub",
                    intent="test",
                    layers=("persona", "unknown-layer"),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
            )

    def test_missing_persona_file_raises(self, experts_root, queue_root):
        # Remove persona.md
        (experts_root / "test-stub" / "persona.md").unlink()
        with pytest.raises(ValueError, match="persona.md not found"):
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub",
                    intent="test",
                    layers=("persona",),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
            )


# ===========================================================================
# Unit tests — post-mortem YAML schema
# ===========================================================================

class TestPostMortemSchema:
    """_run_post_mortem validates record kinds and required fields."""

    def _make_dispatch_cwd(self, tmp_path: Path, notepad: str, output: str) -> Path:
        cwd = tmp_path / "dispatch"
        cwd.mkdir(parents=True, exist_ok=True)
        (cwd / "notepad.md").write_text(notepad)
        (cwd / "output.md").write_text(output)
        return cwd

    def test_valid_records_written(self, tmp_path, experts_root):
        cwd = self._make_dispatch_cwd(tmp_path, "notes", "output text")
        pm_yaml = "records:\n  - kind: concept_encountered\n    essence: |\n      A concept.\n    context: during dispatch\n    confidence: high\n    domain_tags: [test]\n"

        with patch("agents_core.expert.call_claude_cli", return_value=pm_yaml):
            outcome, pm_path = _run_post_mortem(
                expert_id="test-stub",
                task_id="expert_20260502_test",
                intent="test intent",
                layers_loaded=["persona"],
                dispatch_cwd=cwd,
                dispatched_at="2026-05-02T12:00:00+00:00",
                dispatched_by="test",
                duration_seconds=5,
                dispatch_outcome="completed",
                experts_root=experts_root,
            )
        assert outcome == "completed"
        assert pm_path.suffix == ".yaml"
        data = yaml.safe_load(pm_path.read_text())
        assert data["dispatch"]["outcome"] == "completed"
        assert len(data["records"]) == 1
        assert data["records"][0]["kind"] == "concept_encountered"

    def test_all_record_kinds_accepted(self, tmp_path, experts_root):
        for kind in RECORD_KINDS:
            cwd = self._make_dispatch_cwd(tmp_path / kind, "n", "o")
            pm_yaml = (
                f"records:\n"
                f"  - kind: {kind}\n"
                f"    essence: |\n"
                f"      A valid essence for {kind}.\n"
                f"    context: test\n"
                f"    confidence: medium\n"
                f"    domain_tags: []\n"
            )
            with patch("agents_core.expert.call_claude_cli", return_value=pm_yaml):
                outcome, pm_path = _run_post_mortem(
                    expert_id="test-stub",
                    task_id=f"expert_20260502_{kind}",
                    intent="test",
                    layers_loaded=["persona"],
                    dispatch_cwd=cwd,
                    dispatched_at="2026-05-02T12:00:00+00:00",
                    dispatched_by="test",
                    duration_seconds=1,
                    dispatch_outcome="completed",
                    experts_root=experts_root,
                )
            assert outcome == "completed", f"kind={kind} failed"

    def test_unknown_kind_rejected(self, tmp_path, experts_root):
        cwd = self._make_dispatch_cwd(tmp_path, "n", "o")
        bad_yaml = "records:\n  - kind: bad_kind\n    essence: |\n      test\n    context: x\n    confidence: low\n    domain_tags: []\n"
        with patch("agents_core.expert.call_claude_cli", return_value=bad_yaml):
            outcome, pm_path = _run_post_mortem(
                expert_id="test-stub",
                task_id="expert_20260502_badkind",
                intent="test",
                layers_loaded=["persona"],
                dispatch_cwd=cwd,
                dispatched_at="2026-05-02T12:00:00+00:00",
                dispatched_by="test",
                duration_seconds=1,
                dispatch_outcome="completed",
                experts_root=experts_root,
            )
        assert outcome == "post_mortem_failed"
        assert pm_path.name.endswith(".yaml.failed")

    def test_empty_records_is_valid(self, tmp_path, experts_root):
        cwd = self._make_dispatch_cwd(tmp_path, "n", "o")
        with patch("agents_core.expert.call_claude_cli", return_value="records: []"):
            outcome, pm_path = _run_post_mortem(
                expert_id="test-stub",
                task_id="expert_20260502_empty",
                intent="test",
                layers_loaded=["persona"],
                dispatch_cwd=cwd,
                dispatched_at="2026-05-02T12:00:00+00:00",
                dispatched_by="test",
                duration_seconds=1,
                dispatch_outcome="completed",
                experts_root=experts_root,
            )
        assert outcome == "completed"
        data = yaml.safe_load(pm_path.read_text())
        assert data["records"] == []

    def test_missing_records_key_rejected(self, tmp_path, experts_root):
        cwd = self._make_dispatch_cwd(tmp_path, "n", "o")
        bad_yaml = "something_else: true\n"
        with patch("agents_core.expert.call_claude_cli", return_value=bad_yaml):
            outcome, pm_path = _run_post_mortem(
                expert_id="test-stub",
                task_id="expert_20260502_nokey",
                intent="test",
                layers_loaded=["persona"],
                dispatch_cwd=cwd,
                dispatched_at="2026-05-02T12:00:00+00:00",
                dispatched_by="test",
                duration_seconds=1,
                dispatch_outcome="completed",
                experts_root=experts_root,
            )
        assert outcome == "post_mortem_failed"

    def test_llm_returns_none_triggers_failure(self, tmp_path, experts_root):
        cwd = self._make_dispatch_cwd(tmp_path, "n", "o")
        with patch("agents_core.expert.call_claude_cli", return_value=None):
            outcome, pm_path = _run_post_mortem(
                expert_id="test-stub",
                task_id="expert_20260502_none",
                intent="test",
                layers_loaded=["persona"],
                dispatch_cwd=cwd,
                dispatched_at="2026-05-02T12:00:00+00:00",
                dispatched_by="test",
                duration_seconds=1,
                dispatch_outcome="completed",
                experts_root=experts_root,
            )
        assert outcome == "post_mortem_failed"
        failed_data = yaml.safe_load(pm_path.read_text())
        assert "error" in failed_data


# ===========================================================================
# Unit tests — outcome enum semantics
# ===========================================================================

class TestOutcomeSemantics:
    """Dispatch outcome and post-mortem outcome combinations produce correct files."""

    def test_completed_outcome_produces_yaml(self, experts_root, queue_root):
        inp = ExpertDispatchInput(
            expert_id="test-stub", intent="test", layers=("persona",)
        )
        result = _dispatch_with_mock_expert(inp, experts_root, queue_root)
        assert result.outcome == "completed"
        assert result.post_mortem_path.suffix == ".yaml"
        assert not result.post_mortem_path.name.endswith(".yaml.failed")

    def test_abandoned_outcome_produces_yaml(self, experts_root, queue_root):
        """When Expert crashes (no output.md), outcome is 'abandoned'."""
        inp = ExpertDispatchInput(
            expert_id="test-stub", intent="test", layers=("persona",),
            timeout_s=1,
        )
        result = _dispatch_with_mock_expert(
            inp, experts_root, queue_root, simulate_crash=True
        )
        assert result.outcome == "abandoned"
        assert result.output_path is None
        # Post-mortem runs on empty notepad → records: []
        assert result.post_mortem_path.exists()

    def test_post_mortem_failed_produces_yaml_failed(self, experts_root, queue_root):
        """When post-mortem LLM fails, outcome is 'post_mortem_failed'."""
        inp = ExpertDispatchInput(
            expert_id="test-stub", intent="test", layers=("persona",)
        )

        def _fake_submit(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec_data = json.loads(spec_json_path.read_text())
            cwd = Path(spec_data["cwd"])
            (cwd / "notepad.md").write_text("notes")
            (cwd / "output.md").write_text("output")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch("agents_core.expert.call_claude_cli", return_value=None),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit
            MockQueue.return_value = mock_q

            result = dispatch_expert(
                inp,
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert result.outcome == "post_mortem_failed"
        assert result.post_mortem_path.name.endswith(".yaml.failed")
        failed_data = yaml.safe_load(result.post_mortem_path.read_text())
        assert "error" in failed_data
        # Dispatch CWD must never be deleted
        assert result.intent_path.exists()


# ===========================================================================
# Integration tests
# ===========================================================================

class TestFullDispatchCycle:
    """Full dispatch cycle: intent.yaml, notepad.md, output.md, post-mortem."""

    def test_all_artifacts_created(self, experts_root, queue_root):
        inp = ExpertDispatchInput(
            expert_id="test-stub",
            intent="Explain the significance of layered primitives.",
            layers=("persona",),
        )
        result = _dispatch_with_mock_expert(inp, experts_root, queue_root)

        assert result.outcome == "completed"
        assert result.task_id.startswith("expert_")
        assert "test-stub" in result.task_id

        # intent.yaml
        assert result.intent_path.exists()
        intent_data = yaml.safe_load(result.intent_path.read_text())
        assert intent_data["expert_id"] == "test-stub"
        assert intent_data["intent"] == inp.intent
        assert intent_data["task_id"] == result.task_id

        # notepad.md (written by mock Expert)
        assert result.notepad_path.exists()
        assert result.notepad_path.read_text() == "Some field notes."

        # output.md
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.output_path.read_text() == "Expert final response."

        # post-mortem
        assert result.post_mortem_path.exists()
        pm_data = yaml.safe_load(result.post_mortem_path.read_text())
        assert pm_data["dispatch"]["expert_id"] == "test-stub"
        assert pm_data["dispatch"]["task_id"] == result.task_id
        assert "records" in pm_data

    def test_spec_json_written_with_correct_fields(self, experts_root, queue_root):
        """The spec JSON written to pending/ must have all shaped_runner fields."""
        captured_specs = []

        def _fake_submit(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec = json.loads(spec_json_path.read_text())
            captured_specs.append(spec)
            cwd = Path(spec["cwd"])
            (cwd / "notepad.md").write_text("n")
            (cwd / "output.md").write_text("o")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch("agents_core.expert.call_claude_cli", return_value="records: []"),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit
            MockQueue.return_value = mock_q
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub",
                    intent="test",
                    layers=("persona",),
                    model="haiku",
                    timeout_s=300,
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert len(captured_specs) == 1
        spec = captured_specs[0]
        assert spec["task_type"] == "subprocess"
        assert spec["model"] == "haiku"
        assert spec["timeout_s"] == 300
        assert spec["json_mode"] is False
        assert spec["worktree_required"] is False
        assert spec["capture_meta"] is False
        assert spec["permission_mode"] == "bypassPermissions"
        assert "test-stub/dispatches/" in spec["cwd"]
        # system prompt contains persona content
        assert "test stub Expert" in spec["system"]

    def test_dispatch_cwd_is_dispatch_dir(self, experts_root, queue_root):
        """Expert CWD is /experts/<id>/dispatches/<task_id>/."""
        cwd_seen = []

        def _fake_submit(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec = json.loads(spec_json_path.read_text())
            cwd_seen.append(Path(spec["cwd"]))
            cwd = Path(spec["cwd"])
            (cwd / "output.md").write_text("o")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch("agents_core.expert.call_claude_cli", return_value="records: []"),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit
            MockQueue.return_value = mock_q
            result = dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub", intent="test", layers=("persona",)
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert len(cwd_seen) == 1
        cwd = cwd_seen[0]
        assert cwd.name == result.task_id
        assert cwd.parent.name == "dispatches"
        assert cwd.parent.parent.name == "test-stub"


class TestMemoryLoop:
    """Second dispatch with memory layer surfaces first dispatch's post-mortem."""

    def test_memory_layer_loads_prior_post_mortem(self, experts_root, queue_root):
        # First dispatch
        inp1 = ExpertDispatchInput(
            expert_id="test-stub",
            intent="First dispatch intent.",
            layers=("persona",),
        )
        result1 = _dispatch_with_mock_expert(
            inp1, experts_root, queue_root,
            mock_pm_records=(
                "records:\n"
                "  - kind: concept_encountered\n"
                "    essence: |\n"
                "      A layered primitive is one that decomposes cleanly into independent strata.\n"
                "    context: from the first dispatch\n"
                "    confidence: high\n"
                "    domain_tags: [architecture]\n"
            ),
        )
        assert result1.outcome == "completed"
        assert result1.post_mortem_path.exists()

        # Second dispatch with memory layer — system prompt should include first PM
        system_prompts_seen: list[str] = []

        def _fake_submit2(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec = json.loads(spec_json_path.read_text())
            system_prompts_seen.append(spec["system"])
            cwd = Path(spec["cwd"])
            (cwd / "output.md").write_text("second output")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        inp2 = ExpertDispatchInput(
            expert_id="test-stub",
            intent="Second dispatch intent.",
            layers=("persona", "memory"),
        )

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch("agents_core.expert.call_claude_cli", return_value="records: []"),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit2
            MockQueue.return_value = mock_q
            result2 = dispatch_expert(
                inp2,
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert result2.outcome == "completed"
        assert len(system_prompts_seen) == 1
        prompt = system_prompts_seen[0]
        assert "## What you have learned from past dispatches" in prompt
        assert "layered primitive" in prompt  # essence from first PM

    def test_failed_post_mortem_skipped_in_memory_load(self, experts_root, queue_root):
        """Memory layer skips .yaml.failed files."""
        pm_dir = experts_root / "test-stub" / "post-mortems"
        pm_dir.mkdir(parents=True, exist_ok=True)
        (pm_dir / "expert_20260502_bad.yaml.failed").write_text(
            "error: something went wrong\nerror_at: 2026-05-02T12:00:00+00:00\n"
        )

        system_prompts_seen: list[str] = []

        def _fake_submit(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec = json.loads(spec_json_path.read_text())
            system_prompts_seen.append(spec["system"])
            cwd = Path(spec["cwd"])
            (cwd / "output.md").write_text("output")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch("agents_core.expert.call_claude_cli", return_value="records: []"),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit
            MockQueue.return_value = mock_q
            dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub",
                    intent="test",
                    layers=("persona", "memory"),
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert len(system_prompts_seen) == 1
        prompt = system_prompts_seen[0]
        assert "something went wrong" not in prompt
        # Section exists but shows (none yet) because no valid .yaml records
        assert "(none yet)" in prompt


class TestPostMortemFailurePath:
    """When post-mortem LLM call raises, .yaml.failed is written correctly."""

    def test_yaml_failed_written_with_error_field(self, experts_root, queue_root):
        inp = ExpertDispatchInput(
            expert_id="test-stub", intent="test", layers=("persona",)
        )

        def _fake_submit(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec_data = json.loads(spec_json_path.read_text())
            cwd = Path(spec_data["cwd"])
            (cwd / "notepad.md").write_text("notes")
            (cwd / "output.md").write_text("output")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch(
                "agents_core.expert.call_claude_cli",
                side_effect=RuntimeError("LLM backend unavailable"),
            ),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit
            MockQueue.return_value = mock_q

            result = dispatch_expert(
                inp,
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert result.outcome == "post_mortem_failed"
        assert result.post_mortem_path.name.endswith(".yaml.failed")

        failed_data = yaml.safe_load(result.post_mortem_path.read_text())
        assert "error" in failed_data
        assert "LLM backend unavailable" in failed_data["error"]
        assert "error_at" in failed_data
        # dispatch block still present
        assert "dispatch" in failed_data
        assert failed_data["dispatch"]["outcome"] == "post_mortem_failed"

    def test_expert_output_still_accessible_after_pm_failure(
        self, experts_root, queue_root
    ):
        """Dispatch produced output.md even though post-mortem failed."""
        inp = ExpertDispatchInput(
            expert_id="test-stub", intent="test", layers=("persona",)
        )

        def _fake_submit(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec_data = json.loads(spec_json_path.read_text())
            cwd = Path(spec_data["cwd"])
            (cwd / "output.md").write_text("Expert produced this output.")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch("agents_core.expert.call_claude_cli", return_value=None),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit
            MockQueue.return_value = mock_q

            result = dispatch_expert(
                inp,
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert result.output_path is not None
        assert result.output_path.read_text() == "Expert produced this output."


class TestAbandonedDispatchPath:
    """Expert subprocess crashes without writing output.md."""

    def test_abandoned_outcome_set(self, experts_root, queue_root):
        inp = ExpertDispatchInput(
            expert_id="test-stub",
            intent="test",
            layers=("persona",),
            timeout_s=1,
        )
        result = _dispatch_with_mock_expert(
            inp, experts_root, queue_root, simulate_crash=True
        )
        assert result.outcome == "abandoned"
        assert result.output_path is None
        assert result.error is not None

    def test_abandoned_post_mortem_runs_on_partial_notepad(
        self, experts_root, queue_root
    ):
        """Even when abandoned, post-mortem runs on whatever notepad exists."""
        captured_pm_prompts: list[str] = []

        def _fake_submit_crash_with_notepad(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            spec_data = json.loads(spec_json_path.read_text())
            cwd = Path(spec_data["cwd"])
            # Write notepad but NOT output.md (simulate crash after notepad write)
            (cwd / "notepad.md").write_text("Partial notes before crash.")
            spec_json_path.unlink(missing_ok=True)
            return task_id

        def _fake_pm_llm(prompt, **kwargs):
            captured_pm_prompts.append(prompt)
            return "records: []"

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch("agents_core.expert.call_claude_cli", side_effect=_fake_pm_llm),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit_crash_with_notepad
            MockQueue.return_value = mock_q

            result = dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub",
                    intent="test",
                    layers=("persona",),
                    timeout_s=1,
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert result.outcome == "abandoned"
        # Post-mortem LLM was called with the partial notepad
        assert len(captured_pm_prompts) == 1
        assert "Partial notes before crash." in captured_pm_prompts[0]

    def test_dispatch_cwd_persisted_after_abandon(self, experts_root, queue_root):
        """Dispatch CWD is never deleted even on abandon."""
        inp = ExpertDispatchInput(
            expert_id="test-stub",
            intent="test",
            layers=("persona",),
            timeout_s=1,
        )
        result = _dispatch_with_mock_expert(
            inp, experts_root, queue_root, simulate_crash=True
        )
        assert result.intent_path.exists()

    def test_abandoned_with_no_files_writes_empty_records(
        self, experts_root, queue_root
    ):
        """When notepad and output are both absent, post-mortem writes records:[]
        without calling the LLM."""
        pm_llm_called = []

        def _fake_submit_total_crash(task, task_id=None):
            spec_json_path = queue_root / "pending" / f"{task_id}.json"
            # Delete spec to signal shaped_runner finished, but write nothing
            spec_json_path.unlink(missing_ok=True)
            return task_id

        with (
            patch("agents_core.expert.ClaudeQueue") as MockQueue,
            patch(
                "agents_core.expert.call_claude_cli",
                side_effect=lambda *a, **kw: pm_llm_called.append(1) or "records: []",
            ),
        ):
            mock_q = MagicMock()
            mock_q.submit.side_effect = _fake_submit_total_crash
            MockQueue.return_value = mock_q

            result = dispatch_expert(
                ExpertDispatchInput(
                    expert_id="test-stub",
                    intent="test",
                    layers=("persona",),
                    timeout_s=1,
                ),
                _experts_root=experts_root,
                _queue_dir=queue_root,
                _poll_interval=0.05,
            )

        assert result.outcome == "abandoned"
        # LLM should NOT have been called (short-circuit for empty notepad+output)
        assert len(pm_llm_called) == 0
        pm_data = yaml.safe_load(result.post_mortem_path.read_text())
        assert pm_data["records"] == []


# ===========================================================================
# Layout helpers
# ===========================================================================

class TestExpertLayout:
    def test_layout_helpers(self):
        from agents_core.expert_layout import (
            dispatches_root,
            expert_root,
            post_mortems_root,
            seeds_root,
        )
        assert expert_root("aitech") == Path("/srv/lapis/experts/aitech")
        assert seeds_root("aitech") == Path("/srv/lapis/experts/aitech/seeds")
        assert post_mortems_root("aitech") == Path("/srv/lapis/experts/aitech/post-mortems")
        assert dispatches_root("aitech", "expert_20260502_120000_0001_aitech") == Path(
            "/srv/lapis/experts/aitech/dispatches/expert_20260502_120000_0001_aitech"
        )


# ===========================================================================
# Smoke test (real ClaudeQueue dispatch — not run in CI)
# ===========================================================================

@pytest.mark.smoke
def test_smoke_real_dispatch(tmp_path):
    """Real dispatch through live ClaudeQueue against test-stub persona.

    Run manually: pytest -m smoke tests/test_expert_substrate.py::test_smoke_real_dispatch
    Requires the claude-queue-runner daemon to be running.
    """
    stub_fixture = (
        Path(__file__).parent / "fixtures" / "experts" / "test-stub"
    )
    experts_root = tmp_path / "experts"
    queue_root = tmp_path / "queue"

    stub_dir = experts_root / "test-stub"
    stub_dir.mkdir(parents=True)
    shutil.copy(stub_fixture / "persona.md", stub_dir / "persona.md")

    inp = ExpertDispatchInput(
        expert_id="test-stub",
        intent="What is 2 + 2? Write the answer to output.md.",
        layers=("persona",),
        model="haiku",
        requested_by="pytest-smoke",
        timeout_s=120,
    )
    result = dispatch_expert(
        inp, _experts_root=experts_root, _queue_dir=queue_root
    )

    assert result.task_id.startswith("expert_")
    assert result.outcome in ("completed", "abandoned", "post_mortem_failed")

    pm_data = yaml.safe_load(result.post_mortem_path.read_text())
    assert "dispatch" in pm_data
    assert pm_data["dispatch"]["expert_id"] == "test-stub"
