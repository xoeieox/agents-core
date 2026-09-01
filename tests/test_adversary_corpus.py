"""Tests for agents_core.adversary_corpus and the corpus_root() extension.

Test categories:
  Unit:  corpus_root() path helpers, ValueError/RuntimeError guards
  Build: fixture-driven real build, idempotency, removal, source-kind scoping
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from agents_core.expert_layout import corpus_root
from agents_core.adversary_corpus import (
    _MEM_LIMIT,
    _TECH_KAMI_EXPERT_ID,
    build_adversary_corpus,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mem_store(entries: dict[str, dict[str, list[tuple[str, str]]]]) -> MagicMock:
    """Build a mock MemoryStore whose list_all() returns tag-keyed fixture rows.

    entries: {mem_tag: [(key, content), ...]}
    """
    store = MagicMock()

    def _list_all(tag: str = "", limit: int = 50, **kwargs):
        rows = entries.get(tag, [])
        return [{"key": k, "content": c} for k, c in rows[:limit]]

    store.list_all.side_effect = _list_all
    store.close.return_value = None
    return store


def _fixture_store(
    feedback: int = 2,
    ratify_correct: int = 1,
    ratify_override: int = 1,
) -> MagicMock:
    """Return a mock store with configurable row counts per tag."""
    entries = {
        "feedback": [
            (f"feedback/item-{i}", f"Feedback body {i}") for i in range(feedback)
        ],
        "ratify:correct": [
            (f"ratify/correct-{i}", f"Ratify correct body {i}") for i in range(ratify_correct)
        ],
        "ratify:override": [
            (f"ratify/override-{i}", f"Ratify override body {i}") for i in range(ratify_override)
        ],
    }
    return _make_mem_store(entries)


def _arc_dir(tmp_path: Path, count: int = 2) -> Path:
    """Create a fixture arc dir with `count` .md files."""
    arc = tmp_path / "lapis-state"
    arc.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (arc / f"doc-{i:02d}.md").write_text(f"# Arc doc {i}\n\nContent for doc {i}.\n")
    return arc


def _parse_frontmatter(fragment_path: Path) -> dict:
    text = fragment_path.read_text()
    assert text.startswith("---\n"), f"Missing frontmatter in {fragment_path}"
    end = text.find("\n---\n", 4)
    assert end != -1, f"Unclosed frontmatter in {fragment_path}"
    return yaml.safe_load(text[4:end])


# ---------------------------------------------------------------------------
# corpus_root() — layout extension
# ---------------------------------------------------------------------------

class TestCorpusRoot:
    def test_build_mode(self):
        assert corpus_root("tk", "build") == Path("/srv/lapis/experts/tk/corpora/build")

    def test_adversary_mode(self):
        assert corpus_root("tk", "adversary") == Path("/srv/lapis/experts/tk/corpora/adversary")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            corpus_root("tk", "junk")

    def test_default_mode_is_build(self):
        assert corpus_root("any-expert") == Path("/srv/lapis/experts/any-expert/corpora/build")


# ---------------------------------------------------------------------------
# build_adversary_corpus() — guard rails
# ---------------------------------------------------------------------------

class TestGuardRails:
    def test_unknown_expert_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown expert_id"):
            build_adversary_corpus("unknown-expert")

    def test_mem_limit_raises_runtime_error(self, tmp_path):
        """If list_all() returns exactly _MEM_LIMIT rows, raise RuntimeError."""
        entries = {
            "feedback": [(f"feedback/item-{i}", f"body {i}") for i in range(_MEM_LIMIT)],
            "ratify:correct": [],
            "ratify:override": [],
        }
        store = _make_mem_store(entries)
        with pytest.raises(RuntimeError, match="feedback"):
            build_adversary_corpus(
                _TECH_KAMI_EXPERT_ID,
                _corpus_root=tmp_path / "corpus",
                _arc_dir=tmp_path / "arc",
                _mem_store=store,
            )

    def test_mem_limit_plus_one_probe_raises(self, tmp_path):
        """Probe path: _MEM_LIMIT+1 matching entries -> RuntimeError.

        The guard must probe with an explicit limit of _MEM_LIMIT+1 (the real
        MemoryStore.list_all() defaults to limit=50, so a default-limit probe
        can never exceed _MEM_LIMIT and the guard would be dead code).
        """
        entries = {
            "feedback": [
                (f"feedback/item-{i}", f"body {i}") for i in range(_MEM_LIMIT + 1)
            ],
            "ratify:correct": [],
            "ratify:override": [],
        }
        store = _make_mem_store(entries)
        with pytest.raises(RuntimeError, match="feedback"):
            build_adversary_corpus(
                _TECH_KAMI_EXPERT_ID,
                _corpus_root=tmp_path / "corpus",
                _arc_dir=tmp_path / "arc",
                _mem_store=store,
            )
        # The probe must have been issued with an explicit limit of _MEM_LIMIT+1
        # (not the default 50), otherwise it could never return > _MEM_LIMIT rows.
        probe_limits = [
            kw.get("limit")
            for (_, kw) in store.list_all.call_args_list
            if kw.get("limit") == _MEM_LIMIT + 1
        ]
        assert probe_limits, "no list_all() probe with limit=_MEM_LIMIT+1 was issued"

    def test_mem_limit_minus_one_no_raise(self, tmp_path):
        """_MEM_LIMIT-1 matching entries -> probe passes, no RuntimeError."""
        entries = {
            "feedback": [
                (f"feedback/item-{i}", f"body {i}") for i in range(_MEM_LIMIT - 1)
            ],
            "ratify:correct": [],
            "ratify:override": [],
        }
        store = _make_mem_store(entries)
        result = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=tmp_path / "corpus",
            _arc_dir=tmp_path / "arc",
            _mem_store=store,
        )
        assert result["errors"] == []
        assert result["written"] == _MEM_LIMIT - 1


# ---------------------------------------------------------------------------
# build_adversary_corpus() — dry_run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_nothing(self, tmp_path):
        store = _fixture_store()
        arc = _arc_dir(tmp_path)
        corpus = tmp_path / "corpus"

        result = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            dry_run=True,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store,
        )

        # No directories created
        assert not corpus.exists()

        # Plan is non-trivially populated
        assert result["written"] > 0
        assert result["errors"] == []

    def test_dry_run_returns_plan_dict(self, tmp_path):
        store = _fixture_store(feedback=3, ratify_correct=2, ratify_override=1)
        arc = _arc_dir(tmp_path, count=4)
        corpus = tmp_path / "corpus"

        result = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            dry_run=True,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store,
        )

        assert set(result.keys()) == {"written", "unchanged", "removed", "errors"}
        # 3 + 2 + 1 mem entries + 4 arc docs = 10 total would-be-written
        assert result["written"] == 10
        assert result["unchanged"] == 0
        assert result["removed"] == 0


# ---------------------------------------------------------------------------
# build_adversary_corpus() — real build with fixtures
# ---------------------------------------------------------------------------

class TestRealBuild:
    def test_each_source_kind_has_fragments(self, tmp_path):
        store = _fixture_store()
        arc = _arc_dir(tmp_path)
        corpus = tmp_path / "corpus"

        result = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store,
        )

        assert result["errors"] == []
        assert result["written"] >= 1

        for subdir_name in ("feedback", "ratify-correct", "ratify-override", "arc-doc"):
            subdir = corpus / subdir_name
            assert subdir.is_dir(), f"Expected subdir: {subdir}"
            frags = list(subdir.glob("*.md"))
            assert len(frags) >= 1, f"No fragments in {subdir}"

    def test_fragments_have_valid_frontmatter(self, tmp_path):
        store = _fixture_store()
        arc = _arc_dir(tmp_path)
        corpus = tmp_path / "corpus"

        build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store,
        )

        for subdir_name in ("feedback", "ratify-correct", "ratify-override", "arc-doc"):
            for frag in (corpus / subdir_name).glob("*.md"):
                fm = _parse_frontmatter(frag)
                assert "source_key" in fm, f"Missing source_key in {frag}"
                assert "source_kind" in fm, f"Missing source_kind in {frag}"
                assert "captured_at" in fm, f"Missing captured_at in {frag}"
                assert "source_sha256" in fm, f"Missing source_sha256 in {frag}"
                assert fm["source_kind"] == subdir_name

    def test_fragment_body_verbatim(self, tmp_path):
        """Body must be source content verbatim (no LLM rewrite)."""
        store = _fixture_store(feedback=1)
        arc = _arc_dir(tmp_path, count=0)
        corpus = tmp_path / "corpus"

        build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store,
        )

        frags = list((corpus / "feedback").glob("*.md"))
        assert len(frags) == 1
        text = frags[0].read_text()
        # Body section after the closing ---
        body = text.split("---\n", 2)[2]
        assert "Feedback body 0" in body

    def test_stable_ids_deterministic(self, tmp_path):
        """Two independent builds produce identical filenames."""
        store1 = _fixture_store()
        arc1 = _arc_dir(tmp_path / "a")
        corpus1 = tmp_path / "corpus1"
        build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus1,
            _arc_dir=arc1,
            _mem_store=store1,
        )

        store2 = _fixture_store()
        arc2 = _arc_dir(tmp_path / "b")
        corpus2 = tmp_path / "corpus2"
        build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus2,
            _arc_dir=arc2,
            _mem_store=store2,
        )

        for subdir_name in ("feedback", "ratify-correct", "ratify-override", "arc-doc"):
            names1 = {f.name for f in (corpus1 / subdir_name).glob("*.md")}
            names2 = {f.name for f in (corpus2 / subdir_name).glob("*.md")}
            assert names1 == names2, f"Unstable ids in {subdir_name}: {names1} vs {names2}"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_second_run_reports_zero_written(self, tmp_path):
        store = _fixture_store()
        arc = _arc_dir(tmp_path)
        corpus = tmp_path / "corpus"

        r1 = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=_fixture_store(),
        )
        total = r1["written"]
        assert total > 0

        r2 = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=_fixture_store(),
        )

        assert r2["written"] == 0
        assert r2["unchanged"] == total
        assert r2["removed"] == 0
        assert r2["errors"] == []

    def test_changed_source_triggers_rewrite(self, tmp_path):
        arc = _arc_dir(tmp_path, count=0)
        corpus = tmp_path / "corpus"

        store1 = _make_mem_store({"feedback": [("feedback/k1", "original body")]})
        r1 = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store1,
        )
        assert r1["written"] >= 1

        store2 = _make_mem_store({"feedback": [("feedback/k1", "updated body")]})
        r2 = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store2,
        )
        assert r2["written"] >= 1
        assert r2["unchanged"] == 0


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class TestRemoval:
    def test_disappeared_source_removes_fragment(self, tmp_path):
        arc = _arc_dir(tmp_path, count=0)
        corpus = tmp_path / "corpus"

        # First run: 2 feedback entries
        store1 = _make_mem_store({
            "feedback": [("feedback/a", "body a"), ("feedback/b", "body b")],
        })
        r1 = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store1,
        )
        assert r1["written"] == 2

        # Second run: one entry gone
        store2 = _make_mem_store({
            "feedback": [("feedback/a", "body a")],
        })
        r2 = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store2,
        )

        assert r2["removed"] >= 1
        # Only one fragment remains in feedback/
        remaining = list((corpus / "feedback").glob("*.md"))
        assert len(remaining) == 1

    def test_removal_scoped_to_enumerated_source_kinds(self, tmp_path):
        """Fragments in a subdir that is NOT in the current source-kind list
        must be left untouched even if the subdir exists."""
        arc = _arc_dir(tmp_path, count=0)
        corpus = tmp_path / "corpus"

        # Manually plant a stale-looking subdir for a hypothetical future kind
        phantom_dir = corpus / "phantom-kind"
        phantom_dir.mkdir(parents=True)
        (phantom_dir / "aabbccdd11223344.md").write_text("orphan fragment\n")

        store = _fixture_store(feedback=1, ratify_correct=0, ratify_override=0)
        r = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store,
        )

        # Phantom subdir must be untouched
        assert (phantom_dir / "aabbccdd11223344.md").exists()
        # Removal count only reflects enumerated source kinds
        assert r["removed"] == 0


# ---------------------------------------------------------------------------
# Arc-doc (file-based source)
# ---------------------------------------------------------------------------

class TestArcDoc:
    def test_arc_docs_become_fragments(self, tmp_path):
        arc = tmp_path / "lapis-state"
        arc.mkdir()
        (arc / "my-doc.md").write_text("# My doc\n\nContent.\n")
        corpus = tmp_path / "corpus"
        store = _make_mem_store({})

        build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=arc,
            _mem_store=store,
        )

        frags = list((corpus / "arc-doc").glob("*.md"))
        assert len(frags) == 1
        fm = _parse_frontmatter(frags[0])
        assert fm["source_kind"] == "arc-doc"
        assert "my-doc.md" in fm["source_key"]

    def test_missing_arc_dir_is_graceful(self, tmp_path):
        corpus = tmp_path / "corpus"
        store = _make_mem_store({})

        result = build_adversary_corpus(
            _TECH_KAMI_EXPERT_ID,
            _corpus_root=corpus,
            _arc_dir=tmp_path / "nonexistent",
            _mem_store=store,
        )

        assert result["errors"] == []
        # arc-doc subdir may not even be created if no files found
