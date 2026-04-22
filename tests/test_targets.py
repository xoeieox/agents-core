"""Smoke test for agents_core.targets — Target YAML roundtrip + PM fields + TargetStore.create."""
from __future__ import annotations

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
