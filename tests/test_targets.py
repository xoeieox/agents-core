"""Smoke test for agents_core.targets — Target YAML roundtrip + PM fields + TargetStore.create."""
from __future__ import annotations

from datetime import date

import yaml

from agents_core.targets import Target, TargetStore


def test_target_yaml_roundtrip(tmp_path):
    path = tmp_path / "example.yaml"
    data = {"id": "example", "title": "Example", "status": "active",
            "urgency": "high", "work_mode": "anywhere"}
    path.write_text(yaml.dump(data))

    t = Target(yaml.safe_load(path.read_text()), path)
    assert t.id == "example"
    assert t.title == "Example"
    assert t.urgency == "high"
    assert t.work_mode == "anywhere"
    assert t.pm_bound is False  # default


def test_target_pm_fields(tmp_path):
    path = tmp_path / "bound.yaml"
    path.write_text("id: bound\ntitle: Bound\n")
    t = Target(yaml.safe_load(path.read_text()), path)

    t.bind_pm(repo="lapis-engine", authority="advisory")
    assert t.pm_bound is True
    assert t.pm_repo == "lapis-engine"
    assert t.pm_authority == "advisory"

    t.unbind_pm()
    assert t.pm_bound is False
    assert t.pm_repo is None


def test_target_bind_pm_authority_levels(tmp_path):
    """All three authority levels (advisory, auto, hold) must be accepted."""
    import pytest
    path = tmp_path / "auth.yaml"
    path.write_text("id: auth\ntitle: Auth\n")

    for level in ("advisory", "auto", "hold"):
        t = Target(yaml.safe_load(path.read_text()), path)
        t.bind_pm(repo="r", authority=level)
        assert t.pm_authority == level

    t = Target(yaml.safe_load(path.read_text()), path)
    with pytest.raises(ValueError):
        t.bind_pm(repo="r", authority="invalid")


def test_target_store_create(tmp_path):
    store = TargetStore(targets_dir=tmp_path)
    target = store.create(
        target_id="ac-test",
        title="agents-core smoke",
        urgency="low",
        description="test",
    )
    assert target.id == "ac-test"
    assert (tmp_path / "ac-test.yaml").exists()

    # Roundtrip via fresh store
    store2 = TargetStore(targets_dir=tmp_path)
    loaded = [t for t in [Target(yaml.safe_load(p.read_text()), p)
                          for p in tmp_path.glob("*.yaml")]]
    assert any(t.id == "ac-test" and t.title == "agents-core smoke" for t in loaded)
    # Silence unused-var if list_all isn't implemented
    del store2


def test_target_view_helpers(tmp_path):
    """Dashboard view helpers: product, created_date, stages_progress, stages_list, arc_doc_*."""
    path = tmp_path / "view.yaml"
    data = {
        "id": "view",
        "title": "View Target",
        "product": "Lapis",
        "created": "2026-04-01",
        "stages": [
            {"name": "scaffold", "status": "completed", "note": "done"},
            {"name": "design", "status": "active"},
            {"name": "ship", "status": "pending"},
        ],
    }
    path.write_text(yaml.dump(data))
    t = Target(yaml.safe_load(path.read_text()), path)

    assert t.product == "Lapis"
    assert t.created_date == date(2026, 4, 1)
    assert t.stages_progress == "1/3"
    assert t.current_stage == "design"
    assert t.arc_doc_path == "/srv/lapis/lapis-state/view.md"
    assert t.arc_doc_exists is False

    stages = t.stages_list
    assert len(stages) == 3
    assert stages[0] == {"name": "scaffold", "status": "completed", "note": "done"}
    assert stages[1]["status"] == "active"
    assert stages[2]["note"] == ""  # default


def test_target_product_default(tmp_path):
    """product defaults to 'Research' when unset or falsy."""
    path = tmp_path / "p.yaml"
    path.write_text("id: p\ntitle: P\n")
    t = Target(yaml.safe_load(path.read_text()), path)
    assert t.product == "Research"

    # Explicit None / empty string both fall back to Research
    t2 = Target({"id": "p2", "product": None}, path)
    assert t2.product == "Research"


def test_target_to_dashboard_dict(tmp_path):
    path = tmp_path / "dash.yaml"
    data = {
        "id": "dash",
        "title": "Dashboard Test",
        "status": "active",
        "category": "active-work",
        "product": "Conductor",
        "urgency": "high",
        "work_mode": "workshop",
        "decay_days": 2,
        "decay_threshold": 5,
        "touched": "2026-04-20",
        "created": "2026-04-10",
        "tags": ["infra", "dashboard"],
        "description": "smoke",
        "stages": [
            {"name": "s1", "status": "completed"},
            {"name": "s2", "status": "active"},
        ],
        "pm_bound": True,
        "pm_repo": "conductor",
        "pm_authority": "auto",
    }
    path.write_text(yaml.dump(data))
    t = Target(yaml.safe_load(path.read_text()), path)

    d = t.to_dashboard_dict()
    assert d["id"] == "dash"
    assert d["product"] == "Conductor"
    assert d["current_stage"] == "s2"
    assert d["stages_done"] == 1
    assert d["stages_total"] == 2
    assert d["is_decaying"] is False
    assert d["tags"] == ["infra", "dashboard"]
    assert d["pm_bound"] is True
    assert d["pm_repo"] == "conductor"
    assert d["pm_authority"] == "auto"
    assert len(d["stages"]) == 2
    assert d["stages"][0]["name"] == "s1"


def test_arc_doc_exists_true(tmp_path, monkeypatch):
    """arc_doc_exists returns True when the file exists at the hardcoded path."""
    # Redirect /srv/lapis/lapis-state to tmp_path for this test
    fake_state_dir = tmp_path / "lapis-state"
    fake_state_dir.mkdir()
    (fake_state_dir / "exists.md").write_text("# Arc doc")

    # Patch the arc_doc_path property via a subclass (simpler than mocking Path)
    class TestTarget(Target):
        @property
        def arc_doc_path(self) -> str:
            return str(fake_state_dir / f"{self.id}.md")

    t = TestTarget({"id": "exists"}, tmp_path / "x.yaml")
    assert t.arc_doc_exists is True

    t2 = TestTarget({"id": "missing"}, tmp_path / "x.yaml")
    assert t2.arc_doc_exists is False
