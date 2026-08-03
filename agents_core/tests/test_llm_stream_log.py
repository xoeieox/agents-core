"""Unit tests for agents-core-shaped-runner-stream-log-v0.

Covers call_claude_cli's opt-in stream_log_path mode: regression-safety of
the default (None) path, incremental JSONL writes, terminal-event
reconstruction, kill-mid-stream partial-file safety, the flush-not-fsync
write discipline, and non-UTF-8 byte tolerance.
"""
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _fake_completed_process(stdout_obj, returncode=0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = json.dumps(stdout_obj)
    proc.stderr = ""
    return proc


# ---------------------------------------------------------------------------
# DoD 1 — stream_log_path=None reproduces current behavior byte-for-byte
# ---------------------------------------------------------------------------

def test_stream_log_path_none_matches_current_argv_and_return(monkeypatch):
    from agents_core.llm import call_claude_cli

    envelope = {"result": "hello", "total_cost_usd": 0.01}
    with patch("subprocess.run", return_value=_fake_completed_process(envelope)) as mock_run:
        result = call_claude_cli("hi", model="haiku")

    assert result == "hello"
    argv = mock_run.call_args[0][0]
    assert argv == [
        "claude", "-p",
        "--model", "haiku",
        "--no-session-persistence",
        "--output-format", "json",
    ]
    assert mock_run.call_args.kwargs["capture_output"] is True


# ---------------------------------------------------------------------------
# Streaming fixture helper: a fake Popen whose .stdout iterates lines with a
# small delay between them, capturing whether the log file had content
# before the (simulated) process exit.
# ---------------------------------------------------------------------------

class _FakeStreamProc:
    def __init__(self, lines, on_line=None):
        self._lines = lines
        self._on_line = on_line
        self.stdin = MagicMock()
        self.returncode = 0
        self._waited = False

    @property
    def stdout(self):
        return self._iter_lines()

    def _iter_lines(self):
        for line in self._lines:
            if self._on_line:
                self._on_line(line)
            yield line + "\n"

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self._waited = True
        return self.returncode


TERMINAL_RESULT = {
    "type": "result",
    "result": "final answer",
    "total_cost_usd": 0.02,
    "duration_ms": 1234,
    "is_error": False,
}

STREAM_LINES = [
    json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "step1"}]}}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "step2"}]}}),
    json.dumps(TERMINAL_RESULT),
]


# ---------------------------------------------------------------------------
# DoD 2 — stream_log_path=<path> uses stream-json and grows incrementally
# ---------------------------------------------------------------------------

def test_stream_json_flag_used_and_file_grows_incrementally(tmp_path):
    from agents_core.llm import call_claude_cli

    log_path = tmp_path / "stream" / "t-1.jsonl"
    sizes_seen_mid_stream = []

    def _on_line(_line):
        # Called as each line is about to be yielded by the fake stdout
        # iterator — i.e. *before* the mocked process has "exited". The
        # previously-written lines should already be flushed to disk.
        if log_path.exists():
            sizes_seen_mid_stream.append(log_path.stat().st_size)

    fake_proc = _FakeStreamProc(STREAM_LINES, on_line=_on_line)

    with patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
        result = call_claude_cli("hi", model="haiku", stream_log_path=str(log_path))

    assert result == "final answer"
    argv = mock_popen.call_args[0][0]
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"

    # By the time the 3rd (terminal) line was about to be produced, the
    # first two lines must already have been written to disk.
    assert sizes_seen_mid_stream[-1] > 0
    assert sizes_seen_mid_stream == sorted(sizes_seen_mid_stream)

    written = log_path.read_text().splitlines()
    assert len(written) == 3


# ---------------------------------------------------------------------------
# DoD 3 — normal completion reconstructs (result, envelope) from the stream
# ---------------------------------------------------------------------------

def test_stream_reconstructs_result_and_envelope_matching_json_path(tmp_path):
    from agents_core.llm import call_claude_cli

    log_path = tmp_path / "t-2.jsonl"
    fake_proc = _FakeStreamProc(STREAM_LINES)

    with patch("subprocess.Popen", return_value=fake_proc):
        result, envelope = call_claude_cli(
            "hi", model="haiku", stream_log_path=str(log_path), return_envelope=True,
        )

    assert result == "final answer"
    assert envelope["result"] == "final answer"
    assert envelope["total_cost_usd"] == 0.02
    assert envelope["type"] == "result"


# ---------------------------------------------------------------------------
# DoD 3a — reconstructibility: a standalone parser recovers a coherent,
# ordered sequence purely from the file, no in-memory state.
# ---------------------------------------------------------------------------

def test_stream_log_file_independently_reconstructible(tmp_path):
    from agents_core.llm import call_claude_cli

    log_path = tmp_path / "t-3.jsonl"
    fake_proc = _FakeStreamProc(STREAM_LINES)

    with patch("subprocess.Popen", return_value=fake_proc):
        call_claude_cli("hi", model="haiku", stream_log_path=str(log_path))

    lines = log_path.read_text().splitlines()
    assert len(lines) == len(STREAM_LINES)
    events = [json.loads(l) for l in lines]
    assert [e["type"] for e in events] == ["assistant", "assistant", "result"]
    # No corrupt/partial trailing line.
    assert log_path.read_text().endswith("\n")


# ---------------------------------------------------------------------------
# DoD 4 — kill mid-stream leaves only complete lines, returns partial signal
# ---------------------------------------------------------------------------

def test_kill_mid_stream_leaves_complete_lines_only_and_returns_partial(tmp_path):
    from agents_core.llm import call_claude_cli

    log_path = tmp_path / "t-4.jsonl"

    class _KilledProc(_FakeStreamProc):
        def _iter_lines(self):
            # Only the first two (non-terminal) lines ever arrive before the
            # process is killed — simulates a timeout mid-run.
            for line in self._lines[:2]:
                yield line + "\n"
            # No terminal "result" event ever arrives.

    fake_proc = _KilledProc(STREAM_LINES)

    with patch("subprocess.Popen", return_value=fake_proc):
        result, envelope = call_claude_cli(
            "hi", model="haiku", stream_log_path=str(log_path), return_envelope=True,
        )

    assert result is None
    assert envelope is None

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    for l in lines:
        json.loads(l)  # every line parses cleanly, nothing truncated


# ---------------------------------------------------------------------------
# DoD 4a — write helper flushes but never fsyncs
# ---------------------------------------------------------------------------

def test_write_stream_line_flushes_not_fsyncs(tmp_path):
    from agents_core.llm import _write_stream_line

    calls = []
    fake_file = MagicMock()
    fake_file.write.side_effect = lambda s: calls.append(("write", s))
    fake_file.flush.side_effect = lambda: calls.append(("flush",))
    fake_file.close.side_effect = lambda: calls.append(("close",))

    with patch("builtins.open", return_value=fake_file), \
         patch("os.fsync") as mock_fsync:
        _write_stream_line(tmp_path / "x.jsonl", '{"a": 1}')

    mock_fsync.assert_not_called()
    kinds = [c[0] for c in calls]
    assert "flush" in kinds
    assert "close" in kinds
    assert kinds.index("flush") < kinds.index("close")


# ---------------------------------------------------------------------------
# DoD 4b — room_paths key registered, resolves, dir is mkdir'd before write
# ---------------------------------------------------------------------------

def test_stream_logs_room_path_key_registered_and_resolves(monkeypatch, tmp_path):
    from agents_core.room_paths import room_path

    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    p = room_path("claude_queue.stream_logs")
    assert p == tmp_path / "claude-queue" / "stream-logs"
    assert not p.exists()  # room_path never creates the directory


def test_stream_mode_mkdirs_missing_parent_before_first_write(tmp_path):
    from agents_core.llm import call_claude_cli

    log_path = tmp_path / "does" / "not" / "exist" / "t-5.jsonl"
    assert not log_path.parent.exists()
    fake_proc = _FakeStreamProc(STREAM_LINES)

    with patch("subprocess.Popen", return_value=fake_proc):
        call_claude_cli("hi", model="haiku", stream_log_path=str(log_path))

    assert log_path.exists()


# ---------------------------------------------------------------------------
# DoD 4c — a non-UTF-8 byte doesn't raise out of the read loop or corrupt
# other valid lines
# ---------------------------------------------------------------------------

def test_non_utf8_byte_does_not_raise_or_corrupt_other_lines(tmp_path):
    from agents_core.llm import call_claude_cli

    log_path = tmp_path / "t-6.jsonl"
    # Simulate a line that already went through errors="replace" decoding
    # (a stray non-UTF-8 byte becomes U+FFFD) sandwiched between two valid
    # lines — the read loop must not raise and must keep both good lines.
    bad_line = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "�"}]}})
    lines = [STREAM_LINES[0], bad_line, TERMINAL_RESULT and json.dumps(TERMINAL_RESULT)]
    fake_proc = _FakeStreamProc(lines)

    with patch("subprocess.Popen", return_value=fake_proc):
        result = call_claude_cli("hi", model="haiku", stream_log_path=str(log_path))

    assert result == "final answer"
    written = log_path.read_text().splitlines()
    assert len(written) == 3
    for l in written:
        json.loads(l)
