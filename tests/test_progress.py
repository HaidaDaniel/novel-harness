"""Read-only progress snapshot, monitor, launch and liveness tests (offline)."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from typer.testing import CliRunner

from novel_harness import progress as progress_module
from novel_harness.artifacts import write_json
from novel_harness.cli import app as cli_app
from novel_harness.config import load_config
from novel_harness.launch import launch_run
from novel_harness.monitor import (
    is_terminal,
    manual_watch_argv,
    manual_watch_command,
    spawn_monitor,
    watch,
)
from novel_harness.progress import (
    _pid_alive,
    _pid_alive_windows,
    _timing_samples,
    render_snapshot,
    snapshot,
    worker_state,
)
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService

ROOT = Path(__file__).resolve().parents[1]


class CountingRunner:
    def __init__(self):
        self.calls = 0

    def run(self, job):
        self.calls += 1
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


def _make_project(tmp_path, *, paragraphs=14, chunk_chars=200):
    (tmp_path / "original").mkdir(exist_ok=True)
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts", dirs_exist_ok=True)
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["translation"].update(
        chunk_chars=chunk_chars, overlap_chars=0, retries=0, retry_delay_seconds=0
    )
    data["runners"] = {
        "test": {
            "type": "openai",
            "model": "fake-model",
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:9999/v1",
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    (tmp_path / "original" / "0001_first.txt").write_text(
        "第一章\n\n" + "彼は森を歩いていた。\n\n" * paragraphs, "utf-8"
    )
    return path


def _make_multi_chapter_project(tmp_path, numbers):
    (tmp_path / "original").mkdir(exist_ok=True)
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts", dirs_exist_ok=True)
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["translation"].update(
        chunk_chars=6000, overlap_chars=0, retries=0, retry_delay_seconds=0
    )
    data["runners"] = {
        "test": {
            "type": "openai",
            "model": "fake-model",
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:9999/v1",
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    for number in numbers:
        (tmp_path / "original" / f"{number:04d}_chapter.txt").write_text(
            f"第{number}章\n\n" + "彼は森を歩いていた。\n\n" * 3, "utf-8"
        )
    return path


def _service(path):
    runner = CountingRunner()
    return TranslationService(load_config(path), lambda name: runner), runner


def _run_lock(service, run_id):
    lock = service._lock(run_id)
    lock.acquire()
    return lock


# --------------------------------------------------------------------------- liveness


def test_worker_state_reads_metadata_and_liveness(tmp_path, monkeypatch):
    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    write_json(service.run_dir(run_id) / "worker.json", {"pid": os.getpid(), "status": "running"})
    monkeypatch.setattr(progress_module, "_pid_alive", lambda pid, **kwargs: True)

    state = worker_state(service.run_dir(run_id))

    assert state["status"] == "running"
    assert state["alive"] is True


def test_pid_alive_posix_running_missing_and_unknown():
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(2**31 - 1) is False
    assert _pid_alive(None) is None
    assert _pid_alive(-5) is None


class _FakeKernel32:
    def __init__(self, handle, exit_code, last_error):
        self.handle = handle
        self.exit_code = exit_code
        self.last_error = last_error
        self.closed = []
        self.OpenProcess = _Callable()
        self.GetExitCodeProcess = _Callable()
        self.CloseHandle = _Callable(lambda handle: self.closed.append(handle) or 1)


class _Callable:
    """Attribute holder that behaves like a callable WinAPI function."""

    def __init__(self, function=None):
        self.function = function

    def __call__(self, *args):
        if self.function is None:
            raise AssertionError("unexpected call")
        return self.function(*args)


def test_pid_alive_windows_open_process_probe():
    def kernel(handle, exit_code, last_error, *, fail_exit=False):
        fake = _FakeKernel32(handle, exit_code, last_error)
        fake.OpenProcess.function = lambda *args: handle
        if fail_exit:
            fake.GetExitCodeProcess.function = lambda *args: 0
        else:

            def exit_code_process(_handle, pointer):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ulong))[0] = exit_code
                return 1

            fake.GetExitCodeProcess.function = exit_code_process
        return fake

    running = kernel(1, 259, 0)
    assert _pid_alive_windows(123, kernel32=running, get_last_error=lambda: 0) is True
    assert running.closed == [1]

    stopped = kernel(1, 0, 0)
    assert _pid_alive_windows(123, kernel32=stopped, get_last_error=lambda: 0) is False

    missing = kernel(0, 0, 87)
    assert _pid_alive_windows(123, kernel32=missing, get_last_error=lambda: 87) is False

    denied = kernel(0, 0, 5)
    assert _pid_alive_windows(123, kernel32=denied, get_last_error=lambda: 5) is None

    unknown = kernel(1, 0, 0, fail_exit=True)
    assert _pid_alive_windows(123, kernel32=unknown, get_last_error=lambda: 0) is None


def test_pid_alive_dispatches_to_windows_probe(monkeypatch):
    recorded = []

    def fake_windows(pid, **kwargs):
        recorded.append(pid)
        return True

    monkeypatch.setattr(progress_module, "_pid_alive_windows", fake_windows)
    assert _pid_alive(4321, platform="nt") is True
    assert recorded == [4321]


# --------------------------------------------------------------------------- snapshots


def test_snapshot_is_read_only_and_reports_progress(tmp_path):
    path = _make_project(tmp_path)
    service, runner = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id)
    calls = runner.calls

    data = snapshot(service, run_id)

    assert runner.calls == calls
    assert data["status"] == "completed"
    assert data["progress"] == 1.0
    assert data["total_chunks"] > 0
    assert data["current"] is None
    assert data["chapters"]["total"] == 1
    assert "Run:" in render_snapshot(data)


def test_snapshot_pending_run_exposes_current_chunk(tmp_path):
    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")

    data = snapshot(service, run_id)

    assert data["status"] != "completed"
    assert data["current"] is not None
    assert data["current"]["chapter"] == 1
    assert data["worker"] is None


def test_snapshot_running_state_reports_stage_attempt_runner_model(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    manifest = service.repository.run(run_id)["manifest"]
    chapter = manifest["chapters"][0]
    chunk = chapter["chunks"][0]
    service.repository.set_run(run_id, "running", now())
    service.repository.set_chunk(run_id, chapter["number"], chunk["id"], "running")
    service.repository.save_attempt(
        {
            "run_id": run_id,
            "chapter": chapter["number"],
            "chunk_id": chunk["id"],
            "stage": "translation",
            "attempt": 3,
            "status": "running",
            "input_sha256": "x",
        }
    )
    lock = _run_lock(service, run_id)
    try:
        data = snapshot(service, run_id)
    finally:
        lock.release()

    assert data["status"] == "running"
    assert data["active_count"] == 1
    assert data["current"]["stage"] == "translation"
    assert data["current"]["attempt"] == 3
    assert data["current"]["runner"] == "test"
    assert data["current"]["model"] == "fake-model"
    rendered = render_snapshot(data)
    assert "Stage: translation" in rendered
    assert "test / fake-model" in rendered
    assert "Attempt: 3" in rendered


def test_snapshot_paused_failed_interrupted_and_waiting_retry(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")

    service.repository.set_run(run_id, "paused", now())
    assert snapshot(service, run_id)["status"] == "paused"

    service.repository.set_run(run_id, "failed", now(), "boom")
    failed = snapshot(service, run_id)
    assert failed["status"] == "failed"
    assert failed["error"] == "boom"

    service.repository.set_run(run_id, "running", now())
    write_json(service.run_dir(run_id) / "worker.json", {"pid": 2**31 - 1, "status": "running"})
    interrupted = snapshot(service, run_id)
    assert interrupted["status"] == "interrupted"
    assert interrupted["worker"]["alive"] is False

    retry_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    write_json(
        service.run_dir(run_id) / "retry.json",
        {"retry_at": retry_at, "reason": "quota", "deadline_source": "provider"},
    )
    service.repository.set_run(run_id, "failed", now())
    waiting = snapshot(service, run_id)
    assert waiting["status"] == "waiting_retry"
    assert waiting["retry"]["reason"] == "quota"
    assert "Retry:" in render_snapshot(waiting)


def test_snapshot_elapsed_includes_active_worker_session(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.repository.set_run(run_id, "paused", now(), duration=100.0)
    started = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
    write_json(
        service.run_dir(run_id) / "worker.json",
        {"pid": os.getpid(), "status": "running", "started_at": started},
    )

    data = snapshot(service, run_id)

    assert data["elapsed_seconds"] >= 129


def test_elapsed_fresh_worker_uses_frozen_baseline_plus_session(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.repository.set_run(run_id, "paused", now(), duration=0.0)
    started = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()
    write_json(
        service.run_dir(run_id) / "worker.json",
        {
            "pid": os.getpid(),
            "status": "running",
            "started_at": started,
            "base_duration_seconds": 100.0,
        },
    )

    data = snapshot(service, run_id)

    assert 129 <= data["elapsed_seconds"] <= 132


def test_elapsed_does_not_double_count_persisted_duration_after_recovery(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    # An earlier execute() already committed 60s before this worker session.
    service.repository.set_run(run_id, "paused", now(), duration=60.0)
    started = (datetime.now(UTC) - timedelta(seconds=70)).isoformat()
    write_json(
        service.run_dir(run_id) / "worker.json",
        {
            "pid": os.getpid(),
            "status": "running",
            "started_at": started,
            "base_duration_seconds": 0.0,
        },
    )

    first = snapshot(service, run_id)
    assert 69 <= first["elapsed_seconds"] <= 73  # 0 + 70, not 60 + 70

    # A later execute() commits more persisted duration during the same session.
    service.repository.set_run(run_id, "running", now(), duration=30.0)
    second = snapshot(service, run_id)
    assert 69 <= second["elapsed_seconds"] <= 74  # still ~70, no re-added 90


def test_elapsed_new_worker_baselines_previous_persisted_duration(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.repository.set_run(run_id, "paused", now(), duration=200.0)
    started = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
    write_json(
        service.run_dir(run_id) / "worker.json",
        {
            "pid": os.getpid(),
            "status": "running",
            "started_at": started,
            "base_duration_seconds": 200.0,
        },
    )

    data = snapshot(service, run_id)

    assert 209 <= data["elapsed_seconds"] <= 212


def test_snapshot_running_attempt_uses_actual_routed_runner(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, runner = _service(path)
    run_id = service.create_run(runner="test")
    manifest = service.repository.run(run_id)["manifest"]
    chapter = manifest["chapters"][0]
    chunk = chapter["chunks"][0]
    service.repository.set_run(run_id, "running", now())
    service.repository.set_chunk(run_id, chapter["number"], chunk["id"], "running")
    service.repository.save_attempt(
        {
            "run_id": run_id,
            "chapter": chapter["number"],
            "chunk_id": chunk["id"],
            "stage": "translation",
            "attempt": 1,
            "status": "running",
            "input_sha256": "x",
            "runner": "escalation",
            "model": "gpt-5.6-sol",
            "harness": "codex",
        }
    )
    calls = runner.calls
    lock = _run_lock(service, run_id)
    try:
        data = snapshot(service, run_id)
    finally:
        lock.release()

    assert data["current"]["runner"] == "escalation"
    assert data["current"]["model"] == "gpt-5.6-sol"
    assert data["current"]["harness"] == "codex"
    assert "escalation / gpt-5.6-sol" in render_snapshot(data)
    assert runner.calls == calls


def test_snapshot_legacy_running_attempt_falls_back_to_manifest_runner(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    manifest = service.repository.run(run_id)["manifest"]
    chapter = manifest["chapters"][0]
    chunk = chapter["chunks"][0]
    service.repository.set_run(run_id, "running", now())
    service.repository.set_chunk(run_id, chapter["number"], chunk["id"], "running")
    service.repository.save_attempt(
        {
            "run_id": run_id,
            "chapter": chapter["number"],
            "chunk_id": chunk["id"],
            "stage": "translation",
            "attempt": 2,
            "status": "running",
            "input_sha256": "x",
        }
    )
    lock = _run_lock(service, run_id)
    try:
        data = snapshot(service, run_id)
    finally:
        lock.release()

    assert data["current"]["runner"] == "test"
    assert data["current"]["model"] == "fake-model"


def test_snapshot_chapters_separate_number_and_position(tmp_path):
    path = _make_multi_chapter_project(tmp_path, [117, 118, 119])
    service, _ = _service(path)
    run_id = service.create_run(runner="test", chapters=[117, 118, 119])

    data = snapshot(service, run_id)

    assert data["chapters"] == {
        "current_number": 117,
        "current_position": 1,
        "total": 3,
    }
    assert data["current"]["chapter"] == 117
    rendered = render_snapshot(data)
    assert "Chapter: 1 / 3 (#117)" in rendered


def test_snapshot_chapter_position_noncontiguous(tmp_path):
    path = _make_multi_chapter_project(tmp_path, [5, 10, 27])
    service, _ = _service(path)
    run_id = service.create_run(runner="test", chapters=[5, 10, 27])
    service.execute(run_id, max_chunks=1)

    data = snapshot(service, run_id)

    assert data["chapters"]["current_number"] == 10
    assert data["chapters"]["current_position"] == 2
    assert data["chapters"]["total"] == 3
    rendered = render_snapshot(data)
    assert "Chapter: 2 / 3 (#10)" in rendered
    assert "10 / 3" not in rendered


def test_throughput_and_eta_after_enough_samples(tmp_path):
    path = _make_project(tmp_path, paragraphs=60, chunk_chars=60)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id, max_chunks=3)

    data = snapshot(service, run_id)

    assert data["chunks"]["completed"] == 3
    assert data["eta_basis_chunks"] >= 3
    assert data["throughput_chunks_per_minute"] is not None
    assert data["eta_seconds"] is not None


def test_eta_unknown_before_enough_samples(tmp_path):
    path = _make_project(tmp_path, paragraphs=60, chunk_chars=60)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id, max_chunks=1)

    data = snapshot(service, run_id)

    assert data["eta_basis_chunks"] == 1
    assert data["throughput_chunks_per_minute"] is None
    assert data["eta_seconds"] is None
    assert "ETA: unknown" in render_snapshot(data)


def test_inherited_chunks_do_not_corrupt_timings():
    manifest = {"inherited": [{"chapter": 1, "chunk_id": "a"}]}
    rows = [
        {
            "chapter": 1,
            "chunk_id": "a",
            "status": "completed",
            "metadata": json.dumps({"stages": [{"duration_seconds": 1000.0}]}),
        },
        {
            "chapter": 1,
            "chunk_id": "b",
            "status": "completed",
            "metadata": json.dumps({"stages": [{"duration_seconds": 2.0}]}),
        },
    ]

    assert _timing_samples(rows, manifest) == [2.0]


# --------------------------------------------------------------------------- monitor states


def _worker(run_dir, *, status, alive_pid=None, started_at=None):
    write_json(
        run_dir / "worker.json",
        {
            "pid": os.getpid() if alive_pid is None else alive_pid,
            "status": status,
            "started_at": started_at or datetime.now(UTC).isoformat(),
        },
    )


def test_is_terminal_starting_overrides_old_paused_state(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.repository.set_run(run_id, "paused", now())
    _worker(service.run_dir(run_id), status="starting")

    assert is_terminal(snapshot(service, run_id)) is False

    stale = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    _worker(service.run_dir(run_id), status="starting", started_at=stale)
    assert is_terminal(snapshot(service, run_id)) is True


def test_is_terminal_live_worker_overrides_interrupted_state(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.repository.set_run(run_id, "running", now())
    _worker(service.run_dir(run_id), status="running")
    assert is_terminal(snapshot(service, run_id)) is False

    _worker(service.run_dir(run_id), status="running", alive_pid=2**31 - 1)
    assert is_terminal(snapshot(service, run_id)) is True


def test_watch_waits_through_starting_race_and_makes_no_runner_calls(tmp_path):
    from novel_harness.run_factory import now

    path = _make_project(tmp_path)
    service, runner = _service(path)
    run_id = service.create_run(runner="test")
    service.repository.set_run(run_id, "paused", now())
    _worker(service.run_dir(run_id), status="starting")
    calls = runner.calls
    slept = []

    data = watch(
        service,
        run_id,
        interval=0.1,
        iterations=2,
        sleep=slept.append,
        renderer=render_snapshot,
    )

    assert runner.calls == calls
    assert slept == [0.1]
    assert data["status"] == "paused"


def test_watch_iteration_bound_makes_no_runner_calls(tmp_path):
    path = _make_project(tmp_path)
    service, runner = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id)
    calls = runner.calls

    data = watch(service, run_id, interval=0.1, iterations=1, sleep=lambda _seconds: None)

    assert runner.calls == calls
    assert data["status"] == "completed"


def test_watch_disables_unconditional_rich_redraw(tmp_path, monkeypatch):
    import novel_harness.monitor as monitor_module

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    options = {}
    updates = []

    class Live:
        def __init__(self, renderable, **kwargs):
            options.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, renderable, **kwargs):
            updates.append(kwargs)

    monkeypatch.setattr(monitor_module, "Live", Live)
    watch(
        service,
        run_id,
        interval=0.1,
        iterations=1,
        sleep=lambda _seconds: None,
        renderer=render_snapshot,
    )

    assert options["auto_refresh"] is False
    assert updates == [{"refresh": True}]


def test_two_monitors_do_not_duplicate_or_mutate_work(tmp_path):
    path = _make_project(tmp_path, paragraphs=60, chunk_chars=60)
    service, runner = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id, max_chunks=2)
    calls = runner.calls
    worker_file = service.run_dir(run_id) / "worker.json"
    _worker(service.run_dir(run_id), status="running")
    before = worker_file.read_text("utf-8")

    first = watch(service, run_id, interval=0.1, iterations=1, sleep=lambda _: None)
    second = watch(service, run_id, interval=0.1, iterations=1, sleep=lambda _: None)

    assert runner.calls == calls
    assert first["chunks"] == second["chunks"]
    assert worker_file.read_text("utf-8") == before


# --------------------------------------------------------------------------- spawn


def test_spawn_monitor_headless_fallback(tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("must not spawn on headless POSIX")

    result = spawn_monitor(
        tmp_path / "config.yaml",
        "run",
        platform="posix",
        env={},
        which=lambda name: None,
        popen=fail,
    )

    assert result["started"] is False
    assert result["reason"] == "headless_or_no_terminal"
    assert result["watch_argv"][:3] == ["novel", "watch", "--config"]


def test_spawn_monitor_desktop_detaches_controller_streams(tmp_path):
    recorded = {}

    class FakeProcess:
        pid = 4321

    def popen(args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return FakeProcess()

    result = spawn_monitor(
        tmp_path / "config.yaml",
        "run",
        platform="posix",
        env={"DISPLAY": ":0"},
        which=lambda name: "/usr/bin/xterm" if name == "xterm" else None,
        popen=popen,
    )

    assert result["started"] is True
    assert result["pid"] == 4321
    assert recorded["args"][0] == "/usr/bin/xterm"
    assert recorded["kwargs"]["stdin"] is subprocess.DEVNULL
    assert recorded["kwargs"]["stdout"] is subprocess.DEVNULL
    assert recorded["kwargs"]["stderr"] is subprocess.DEVNULL
    assert recorded["kwargs"]["start_new_session"] is True


def test_spawn_monitor_failure_is_reported_and_leaves_worker_unchanged(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "worker.json", {"pid": os.getpid(), "status": "running"})
    before = (run_dir / "worker.json").read_text("utf-8")

    def popen(*args, **kwargs):
        raise OSError("no terminal")

    result = spawn_monitor(
        tmp_path / "config.yaml",
        "run",
        platform="posix",
        env={"DISPLAY": ":0"},
        which=lambda name: "/usr/bin/xterm" if name == "xterm" else None,
        popen=popen,
    )

    assert result["started"] is False
    assert result["error"] == "OSError"
    assert result["watch_command"]
    assert (run_dir / "worker.json").read_text("utf-8") == before


def test_spawn_monitor_windows_uses_console(tmp_path):
    recorded = {}

    class FakeProcess:
        pid = 99

    def popen(args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return FakeProcess()

    result = spawn_monitor(
        tmp_path / "config.yaml", "run", platform="nt", popen=popen
    )

    assert result["started"] is True
    assert result["terminal"] == "console"
    assert recorded["args"][0] == "cmd"
    assert recorded["kwargs"]["stdin"] is subprocess.DEVNULL


def test_watch_argv_and_command_are_shell_safe():
    config = Path("/tmp/with spaces/config.yaml")
    argv = manual_watch_argv(config, "run-1")
    assert argv == ["novel", "watch", "--config", str(config), "--run", "run-1"]

    import shlex

    command = manual_watch_command(config, "run-1", platform="posix")
    assert shlex.split(command) == argv

    windows = manual_watch_command(config, "run-1", platform="nt")
    assert '"' in windows


# --------------------------------------------------------------------------- launch


def test_launch_run_returns_immediately_with_watch_info(tmp_path, monkeypatch):
    from novel_harness.worker import now

    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    spawned = []

    def fake_start(config, run_id_arg, **kwargs):
        spawned.append(run_id_arg)
        directory = service.run_dir(run_id_arg)
        write_json(
            directory / "worker.json",
            {"pid": os.getpid(), "status": "running", "started_at": now()},
        )
        return {
            "run_id": run_id_arg,
            "started": True,
            "already_running": False,
            "pid": os.getpid(),
            "worker_state": str(directory / "worker.json"),
        }

    monkeypatch.setattr("novel_harness.worker.start_worker", fake_start)

    result = launch_run(service, path, run_id)

    assert spawned == [run_id]
    assert result["started"] is True
    assert result["watch_argv"] == ["novel", "watch", "--config", str(path.resolve()), "--run", run_id]
    assert "watch" in result["watch_command"]


def test_launch_run_monitor_opens_for_existing_worker(tmp_path, monkeypatch):
    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    monitors = []

    def fake_start(config, run_id_arg, **kwargs):
        return {
            "run_id": run_id_arg,
            "started": False,
            "already_running": True,
            "completed": False,
            "pid": 777,
            "worker_status": "running",
            "worker_state": "stub",
            "retry_requested": True,
        }

    def fake_monitor(config, run_id_arg, **kwargs):
        monitors.append(run_id_arg)
        return {"started": True, "pid": 555}

    monkeypatch.setattr("novel_harness.worker.start_worker", fake_start)
    monkeypatch.setattr("novel_harness.monitor.spawn_monitor", fake_monitor)

    result = launch_run(service, path, run_id, monitor=True)

    assert result["already_running"] is True
    assert result["retry_requested"] is True
    assert monitors == [run_id]
    assert result["monitor"] == {"started": True, "pid": 555}


def test_launch_run_monitor_skipped_after_start_failure(tmp_path, monkeypatch):
    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")

    def fake_start(config, run_id_arg, **kwargs):
        return {
            "run_id": run_id_arg,
            "started": False,
            "already_running": False,
            "completed": False,
            "error": "worker_exited_during_startup",
        }

    def fail(*args, **kwargs):
        raise AssertionError("must not open a monitor for a dead worker")

    monkeypatch.setattr("novel_harness.worker.start_worker", fake_start)
    monkeypatch.setattr("novel_harness.monitor.spawn_monitor", fail)

    result = launch_run(service, path, run_id, monitor=True)

    assert result["monitor"]["started"] is False
    assert result["monitor"]["reason"] == "worker_exited_during_startup"
    assert result["monitor"]["watch_command"]


def test_launch_run_monitor_skipped_for_completed_run(tmp_path, monkeypatch):
    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id)

    def fail(*args, **kwargs):
        raise AssertionError("must not spawn a worker for a completed run")

    monkeypatch.setattr("novel_harness.worker._spawn_worker", fail)

    result = launch_run(service, path, run_id, monitor=True)

    assert result["completed"] is True
    assert result["started"] is False
    assert result["monitor"]["reason"] == "run_completed"


# --------------------------------------------------------------------------- CLI


def _create_run(path):
    service, runner = _service(path)
    run_id = service.create_run(runner="test")
    return service, runner, run_id


def test_cli_watch_once_json_is_clean_json(tmp_path):
    path = _make_project(tmp_path)
    _, _, run_id = _create_run(path)

    result = CliRunner().invoke(
        cli_app, ["watch", "--config", str(path), "--run", run_id, "--once", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run_id"] == run_id
    assert "Run:" not in result.stdout


def test_cli_watch_json_without_once_is_an_error(tmp_path):
    path = _make_project(tmp_path)
    _, _, run_id = _create_run(path)

    result = CliRunner().invoke(
        cli_app, ["watch", "--config", str(path), "--run", run_id, "--json"]
    )

    assert result.exit_code != 0
    assert "once" in (result.output or "").lower()


def test_cli_watch_rejects_zero_interval(tmp_path):
    path = _make_project(tmp_path)
    _, _, run_id = _create_run(path)

    result = CliRunner().invoke(
        cli_app,
        ["watch", "--config", str(path), "--run", run_id, "--once", "--json", "--interval", "0"],
    )

    assert result.exit_code != 0
    assert "interval" in (result.output or "").lower()


def test_cli_start_with_monitor_uses_shared_orchestration(tmp_path, monkeypatch):
    path = _make_project(tmp_path)
    service, _, run_id = _create_run(path)
    monitors = []

    def fake_start(config, run_id_arg, **kwargs):
        write_json(
            service.run_dir(run_id_arg) / "worker.json",
            {"pid": os.getpid(), "status": "running"},
        )
        return {"run_id": run_id_arg, "started": True, "already_running": False, "pid": os.getpid()}

    def fake_monitor(config, run_id_arg, **kwargs):
        monitors.append(run_id_arg)
        return {"started": True, "pid": 888}

    monkeypatch.setattr("novel_harness.worker.start_worker", fake_start)
    monkeypatch.setattr("novel_harness.monitor.spawn_monitor", fake_monitor)

    result = CliRunner().invoke(
        cli_app, ["start", "--config", str(path), "--run", run_id, "--monitor", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["started"] is True
    assert payload["monitor"]["started"] is True
    assert monitors == [run_id]


def test_cli_start_monitor_for_already_running_worker(tmp_path, monkeypatch):
    path = _make_project(tmp_path)
    _, _, run_id = _create_run(path)
    monitors = []

    def fake_start(config, run_id_arg, **kwargs):
        return {
            "run_id": run_id_arg,
            "started": False,
            "already_running": True,
            "completed": False,
            "pid": 42,
            "worker_status": "running",
            "worker_state": "stub",
        }

    def fake_monitor(config, run_id_arg, **kwargs):
        monitors.append(run_id_arg)
        return {"started": True, "pid": 999}

    monkeypatch.setattr("novel_harness.worker.start_worker", fake_start)
    monkeypatch.setattr("novel_harness.monitor.spawn_monitor", fake_monitor)

    result = CliRunner().invoke(
        cli_app, ["start", "--config", str(path), "--run", run_id, "--monitor", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["already_running"] is True
    assert monitors == [run_id]
