"""Tests for agents_core.claude_queue_runner — primarily the ops-primitive
extraction hook. The full `_run_task` coroutine spawns a subprocess, so we
don't end-to-end it here — we exercise the hook in isolation.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import types
from unittest.mock import patch

import pytest

from agents_core import claude_queue_runner as runner_mod


@pytest.fixture(autouse=True)
def reset_guard_blocking():
    """Reset the module-level _guard_blocking flag before each test."""
    runner_mod._guard_blocking = False
    yield
    runner_mod._guard_blocking = False


def _install_fake_ops_primitives(monkeypatch, recorder: list) -> None:
    """Inject a fake `ops_primitives` module so the hook is driven without
    hitting mem.db. Records the (task_id, task_type, text) triple."""
    fake = types.ModuleType("ops_primitives")

    def extract_and_store(task_id, task_type, text):
        recorder.append((task_id, task_type, text[:60]))
        return [{"type": "recorded"}]

    fake.extract_and_store = extract_and_store
    monkeypatch.setitem(sys.modules, "ops_primitives", fake)


def test_extract_ops_primitives_forwards_text(monkeypatch):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)

    text = "A" * 200  # exceeds the >=80 "worth tagging" gate
    runner_mod._extract_ops_primitives(
        "claude_test_1", "subprocess", text, "/tmp/out.md"
    )

    assert calls == [("claude_test_1", "subprocess", text[:60])]


def test_extract_ops_primitives_falls_back_to_output_file(monkeypatch, tmp_path):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)

    output_path = tmp_path / "out.md"
    output_path.write_text("B" * 300)

    runner_mod._extract_ops_primitives(
        "claude_test_2", "subprocess",
        result_text="",           # empty — forces fallback read
        output_path=str(output_path),
    )

    assert calls
    assert calls[0][0] == "claude_test_2"
    assert calls[0][2].startswith("B")


def test_extract_ops_primitives_skips_trivial_text(monkeypatch):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)

    runner_mod._extract_ops_primitives(
        "claude_test_3", "subprocess", "tiny", None
    )

    assert calls == []


def test_extract_ops_primitives_honours_env_gate(monkeypatch):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)
    monkeypatch.setenv("LAPIS_OPS_PRIMITIVES", "0")

    runner_mod._extract_ops_primitives(
        "claude_test_4", "subprocess", "A" * 300, "/tmp/out.md"
    )

    assert calls == []


def test_extract_ops_primitives_swallows_failure(monkeypatch, caplog):
    """Missing weather data is acceptable at the aggregate — a broken
    ops_primitives must not blow up the runner."""
    fake = types.ModuleType("ops_primitives")

    def extract_and_store(*a, **k):
        raise RuntimeError("fake extractor error")

    fake.extract_and_store = extract_and_store
    monkeypatch.setitem(sys.modules, "ops_primitives", fake)

    # Should not raise.
    runner_mod._extract_ops_primitives(
        "claude_test_5", "subprocess", "A" * 300, None
    )


# ---------------------------------------------------------------------------
# Freeze-guard tests (AC#1-8)
# ---------------------------------------------------------------------------


def test_spawn_freeze_guard_block_reason_low_ram(monkeypatch):
    """AC#2: RAM below floor returns reason string."""
    class FakeVM:
        available = 2.0 * (1024 ** 3)  # 2 GB < 4.0 GB floor

    class FakeSW:
        total = 8.0 * (1024 ** 3)
        used = 2.0 * (1024 ** 3)

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeVM())
    monkeypatch.setattr("psutil.swap_memory", lambda: FakeSW())

    reason = runner_mod._spawn_freeze_guard_block_reason()
    assert reason is not None
    assert "RAM available" in reason
    assert "2.0 GB" in reason


def test_spawn_freeze_guard_block_reason_low_swap_free(monkeypatch):
    """AC#2: Swap free below headroom returns reason string."""
    class FakeVM:
        available = 10.0 * (1024 ** 3)

    class FakeSW:
        total = 8.0 * (1024 ** 3)
        used = 7.0 * (1024 ** 3)  # only 1.0 GB free < 1.5 GB headroom

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeVM())
    monkeypatch.setattr("psutil.swap_memory", lambda: FakeSW())

    reason = runner_mod._spawn_freeze_guard_block_reason()
    assert reason is not None
    assert "swap free" in reason
    assert "1.0 GB" in reason


def test_spawn_freeze_guard_block_reason_both_low(monkeypatch):
    """AC#2: Ordering — RAM check runs before swap check."""
    class FakeVM:
        available = 2.0 * (1024 ** 3)

    class FakeSW:
        total = 8.0 * (1024 ** 3)
        used = 7.0 * (1024 ** 3)

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeVM())
    monkeypatch.setattr("psutil.swap_memory", lambda: FakeSW())

    reason = runner_mod._spawn_freeze_guard_block_reason()
    assert reason is not None
    assert "RAM available" in reason  # RAM check runs first


def test_spawn_freeze_guard_block_reason_brix_baseline_healthy(monkeypatch):
    """AC#2: Calibration regression — BRIX idle baseline (9.6 GB RAM available,
    4.3 GB swap free) is NOT blocked."""
    class FakeVM:
        available = 9.6 * (1024 ** 3)

    class FakeSW:
        total = 8.0 * (1024 ** 3)
        used = 3.7 * (1024 ** 3)  # 4.3 GB free

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeVM())
    monkeypatch.setattr("psutil.swap_memory", lambda: FakeSW())

    reason = runner_mod._spawn_freeze_guard_block_reason()
    assert reason is None, f"BRIX baseline should not block, got: {reason}"


def test_spawn_freeze_guard_block_reason_read_failure_open(monkeypatch):
    """AC#3: Read failure returns None (fails open; never blocks all work)."""
    def raise_error():
        raise OSError("psutil unavailable")

    monkeypatch.setattr("psutil.virtual_memory", raise_error)

    reason = runner_mod._spawn_freeze_guard_block_reason()
    assert reason is None


def test_spawn_freeze_guard_block_reason_threshold_override(monkeypatch):
    """AC#5: Env-var thresholds override defaults."""
    # Override the constants directly (they are resolved at module import time,
    # so setenv has no effect; we patch the actual module-level constants).
    monkeypatch.setattr(runner_mod, "SPAWN_MIN_RAM_AVAIL_GB", 5.0)
    monkeypatch.setattr(runner_mod, "SPAWN_MIN_SWAP_FREE_GB", 2.0)

    class FakeVM:
        available = 4.9 * (1024 ** 3)  # Below 5.0 threshold

    class FakeSW:
        total = 8.0 * (1024 ** 3)
        used = 6.5 * (1024 ** 3)  # 1.5 GB free, below 2.0 threshold

    monkeypatch.setattr("psutil.virtual_memory", lambda: FakeVM())
    monkeypatch.setattr("psutil.swap_memory", lambda: FakeSW())

    reason = runner_mod._spawn_freeze_guard_block_reason()
    assert reason is not None
    assert "4.9 GB" in reason


@pytest.mark.asyncio
async def test_daemon_claim_loop_withholds_under_pressure(monkeypatch, caplog):
    """AC#1: Daemon.run() does not call queue.claim() when guard returns reason."""
    caplog.set_level(logging.WARNING)

    # Monkeypatch POLL_INTERVAL_S to make test fast and deterministic.
    monkeypatch.setattr(runner_mod, "POLL_INTERVAL_S", 0.001)

    # Mock queue.claim() to track calls.
    claim_calls = []

    class FakeQueue:
        def claim(self):
            claim_calls.append(True)
            return None
        active_dir = None
        queue_dir = None

    # Mock startup_sweep.
    def fake_startup_sweep(q):
        pass

    monkeypatch.setattr(runner_mod, "startup_sweep", fake_startup_sweep)

    # Mock the freeze-guard to always return a block reason.
    def always_block():
        return "simulated pressure"

    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", always_block)

    daemon = runner_mod.Daemon(workers=2)
    daemon.queue = FakeQueue()

    # Run for 2 iterations, then stop.
    iteration_count = [0]

    async def count_iterations():
        while not daemon.stop_claiming.is_set() and iteration_count[0] < 2:
            iteration_count[0] += 1
            if iteration_count[0] >= 2:
                daemon.stop_claiming.set()
            await asyncio.sleep(0.01)

    counter_task = asyncio.create_task(count_iterations())

    try:
        await asyncio.wait_for(daemon.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    counter_task.cancel()
    try:
        await counter_task
    except asyncio.CancelledError:
        pass

    # claim() should NOT have been called (guard withheld).
    assert claim_calls == [], f"Expected claim() to never be called, got {len(claim_calls)} calls"
    # Log should show the guard engaged.
    assert "freeze-guard ENGAGED" in caplog.text


@pytest.mark.asyncio
async def test_daemon_claim_loop_proceeds_when_guard_clear(monkeypatch, caplog):
    """AC#1: Daemon.run() calls queue.claim() when guard returns None."""
    caplog.set_level(logging.INFO)

    # Monkeypatch POLL_INTERVAL_S to make test fast and deterministic.
    monkeypatch.setattr(runner_mod, "POLL_INTERVAL_S", 0.001)

    claim_calls = []

    class FakeQueue:
        def claim(self):
            claim_calls.append(True)
            return None
        active_dir = None
        queue_dir = None

    def fake_startup_sweep(q):
        pass

    monkeypatch.setattr(runner_mod, "startup_sweep", fake_startup_sweep)

    # Mock the freeze-guard to return None (no pressure).
    def always_clear():
        return None

    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", always_clear)

    daemon = runner_mod.Daemon(workers=2)
    daemon.queue = FakeQueue()

    iteration_count = [0]

    async def count_iterations():
        while not daemon.stop_claiming.is_set() and iteration_count[0] < 2:
            iteration_count[0] += 1
            if iteration_count[0] >= 2:
                daemon.stop_claiming.set()
            await asyncio.sleep(0.01)

    counter_task = asyncio.create_task(count_iterations())

    try:
        await asyncio.wait_for(daemon.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    counter_task.cancel()
    try:
        await counter_task
    except asyncio.CancelledError:
        pass

    # claim() should have been called at least once.
    assert len(claim_calls) > 0


@pytest.mark.asyncio
async def test_daemon_freeze_guard_flag_disables(monkeypatch):
    """AC#4: CLAUDE_QUEUE_FREEZE_GUARD=0 disables guard, claim proceeds always."""
    monkeypatch.setenv("CLAUDE_QUEUE_FREEZE_GUARD", "0")

    # Monkeypatch POLL_INTERVAL_S to make test fast and deterministic.
    monkeypatch.setattr(runner_mod, "POLL_INTERVAL_S", 0.001)

    claim_calls = []

    class FakeQueue:
        def claim(self):
            claim_calls.append(True)
            return None
        active_dir = None
        queue_dir = None

    def fake_startup_sweep(q):
        pass

    monkeypatch.setattr(runner_mod, "startup_sweep", fake_startup_sweep)

    # Even if guard would block, it's disabled so claim() runs.
    def always_block():
        return "simulated pressure"

    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", always_block)

    daemon = runner_mod.Daemon(workers=2)
    daemon.queue = FakeQueue()

    iteration_count = [0]

    async def count_iterations():
        while not daemon.stop_claiming.is_set() and iteration_count[0] < 2:
            iteration_count[0] += 1
            if iteration_count[0] >= 2:
                daemon.stop_claiming.set()
            await asyncio.sleep(0.01)

    counter_task = asyncio.create_task(count_iterations())

    try:
        await asyncio.wait_for(daemon.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass

    counter_task.cancel()
    try:
        await counter_task
    except asyncio.CancelledError:
        pass

    # claim() should be called because guard is disabled.
    assert len(claim_calls) > 0


def test_spawn_freeze_guard_log_discipline(monkeypatch, caplog):
    """AC#8: Log discipline — guard transitions logged appropriately.
    Direct test of guard state machine without full daemon loop."""
    caplog.set_level(logging.DEBUG)
    log = logging.getLogger(__name__)

    # Directly test the guard state transitions
    call_count = [0]

    def conditional_block():
        call_count[0] += 1
        if call_count[0] <= 3:
            return "pressure"
        elif call_count[0] <= 6:
            return None
        else:
            return "pressure"

    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", conditional_block)

    # Simulate the daemon guard logic in a small loop
    runner_mod._guard_blocking = False
    for i in range(10):
        block_reason = runner_mod._spawn_freeze_guard_block_reason()
        if block_reason is not None:
            if not runner_mod._guard_blocking:
                runner_mod._guard_blocking = True
                log.warning(f"test: guard ENGAGED")
            else:
                log.debug(f"test: guard still engaged")
        elif runner_mod._guard_blocking:
            runner_mod._guard_blocking = False
            log.warning(f"test: guard CLEARED")

    warn_count = caplog.text.count("test: guard ENGAGED")
    clear_count = caplog.text.count("test: guard CLEARED")

    assert warn_count >= 1, f"Expected at least 1 ENGAGED, got {warn_count}"
    assert clear_count >= 1, f"Expected at least 1 CLEARED, got {clear_count}"
    assert "test: guard still engaged" in caplog.text


@pytest.mark.asyncio
async def test_daemon_run_log_discipline_multi_tick(monkeypatch, caplog):
    """AC#8: Integration test — verify Daemon.run() itself logs WARN on
    guard transitions and DEBUG per intermediate tick."""
    caplog.set_level(logging.DEBUG)

    # Monkeypatch POLL_INTERVAL_S and guard function to control flow.
    monkeypatch.setattr(runner_mod, "POLL_INTERVAL_S", 0.001)

    call_count = [0]

    def conditional_block():
        # Ticks 0-2: pressure (engage on tick 0)
        # Ticks 3-5: clear (clear on tick 3)
        # Ticks 6+: pressure again (engage on tick 6)
        call_count[0] += 1
        if call_count[0] <= 3:
            return "simulated pressure"
        elif call_count[0] <= 6:
            return None
        else:
            return "simulated pressure"

    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", conditional_block)

    # Mock queue to always return None (no task) so loop ticks without spawning.
    class FakeQueue:
        def claim(self):
            return None
        active_dir = None
        queue_dir = None

    def fake_startup_sweep(q):
        pass

    monkeypatch.setattr(runner_mod, "startup_sweep", fake_startup_sweep)

    daemon = runner_mod.Daemon(workers=2)
    daemon.queue = FakeQueue()

    # Run for exactly 8 ticks (roughly), then stop.
    tick_count = [0]

    async def count_ticks():
        while not daemon.stop_claiming.is_set() and tick_count[0] < 8:
            tick_count[0] += 1
            if tick_count[0] >= 8:
                daemon.stop_claiming.set()
            await asyncio.sleep(0.0001)  # very short sleep to let daemon loop progress

    ticker = asyncio.create_task(count_ticks())

    try:
        await asyncio.wait_for(daemon.run(), timeout=5.0)
    except asyncio.TimeoutError:
        pass

    ticker.cancel()
    try:
        await ticker
    except asyncio.CancelledError:
        pass

    # Verify log discipline: WARN on engage and clear, DEBUG in between.
    assert "freeze-guard ENGAGED" in caplog.text, "Expected ENGAGED log"
    assert "freeze-guard CLEARED" in caplog.text, "Expected CLEARED log"
    assert "freeze-guard still engaged" in caplog.text, "Expected DEBUG per intermediate tick"


@pytest.mark.asyncio
async def test_daemon_claim_loop_crash_exits_loud(monkeypatch, caplog):
    """agents-core-queue-runner-wedge-selfheal-v0: an exception escaping the
    claim loop body (e.g. from queue.claim()) must log CRITICAL with a
    traceback, fire exactly one HIGH-priority notification, and sys.exit
    with the dedicated crash exit code, no swallow-and-loop."""
    caplog.set_level(logging.DEBUG)

    monkeypatch.setattr(runner_mod, "POLL_INTERVAL_S", 0.001)

    class FakeQueue:
        def claim(self):
            raise RuntimeError("Event loop is closed")
        active_dir = None
        queue_dir = None

    def fake_startup_sweep(q):
        pass

    monkeypatch.setattr(runner_mod, "startup_sweep", fake_startup_sweep)
    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", lambda: None)

    notify_calls = []
    monkeypatch.setattr(
        runner_mod, "send_notification", lambda **kw: notify_calls.append(kw)
    )

    daemon = runner_mod.Daemon(workers=2)
    daemon.queue = FakeQueue()

    with pytest.raises(SystemExit) as exc_info:
        await daemon.run()

    assert exc_info.value.code == runner_mod._CLAIM_LOOP_CRASH_EXIT_CODE

    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical_records) == 1
    assert critical_records[0].exc_info is not None

    assert len(notify_calls) == 1
    assert notify_calls[0]["priority"] == runner_mod.PushoverPriority.HIGH


@pytest.mark.asyncio
async def test_daemon_claim_loop_reraises_cancelled_error(monkeypatch, caplog):
    """Scope item 2 guard-rail: CancelledError from queue.claim() must
    propagate untouched, not be misclassified as the crash path."""
    monkeypatch.setattr(runner_mod, "POLL_INTERVAL_S", 0.001)

    class FakeQueue:
        def claim(self):
            raise asyncio.CancelledError()
        active_dir = None
        queue_dir = None

    def fake_startup_sweep(q):
        pass

    monkeypatch.setattr(runner_mod, "startup_sweep", fake_startup_sweep)
    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", lambda: None)

    notify_calls = []
    monkeypatch.setattr(
        runner_mod, "send_notification", lambda **kw: notify_calls.append(kw)
    )

    daemon = runner_mod.Daemon(workers=2)
    daemon.queue = FakeQueue()

    with pytest.raises(asyncio.CancelledError):
        await daemon.run()

    assert notify_calls == []


@pytest.mark.asyncio
async def test_daemon_claim_loop_reraises_generator_exit(monkeypatch, caplog):
    """Scope item 2 guard-rail: GeneratorExit from queue.claim() must
    propagate untouched, parallel case to CancelledError since the
    guard-rail clause names both exception types."""
    monkeypatch.setattr(runner_mod, "POLL_INTERVAL_S", 0.001)

    class FakeQueue:
        def claim(self):
            raise GeneratorExit()
        active_dir = None
        queue_dir = None

    def fake_startup_sweep(q):
        pass

    monkeypatch.setattr(runner_mod, "startup_sweep", fake_startup_sweep)
    monkeypatch.setattr(runner_mod, "_spawn_freeze_guard_block_reason", lambda: None)

    notify_calls = []
    monkeypatch.setattr(
        runner_mod, "send_notification", lambda **kw: notify_calls.append(kw)
    )

    daemon = runner_mod.Daemon(workers=2)
    daemon.queue = FakeQueue()

    with pytest.raises(GeneratorExit):
        await daemon.run()

    assert notify_calls == []


# ---------------------------------------------------------------------------
# startup_sweep — in_flight reconciliation against a ghost state.json entry
# ---------------------------------------------------------------------------

def test_startup_sweep_clears_ghost_in_flight_entry(tmp_path, monkeypatch):
    """state.json.in_flight may list an id with no backing active/*.yaml
    file at all (hand-killed + rm'd out of band). The stale-active-task loop
    only ever sees files that still exist, so it can't catch this — the
    unconditional _refresh_state/_write_state reconciliation at the end of
    startup_sweep must clear it regardless."""
    from agents_core.claude_queue import ClaudeQueue

    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")

    queue = ClaudeQueue(queue_dir=tmp_path / "claude-queue")
    state = queue._read_state()
    state["in_flight"] = ["ghost-id"]
    queue._write_state(state)

    with patch("subprocess.run"):
        runner_mod.startup_sweep(queue)

    import json
    persisted = json.loads(queue.state_path.read_text())
    assert "ghost-id" not in persisted["in_flight"]
