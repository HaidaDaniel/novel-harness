import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
from conftest import FakeRunner
from filelock import FileLock

from novel_harness.artifacts import write_json
from novel_harness.run_factory import now
from novel_harness.runners.base import RunnerDeferred, RunnerExecutionError, RunnerResult
from novel_harness.worker import (
    RETRY_NOW_FILENAME,
    _bounded_build_epub,
    _bounded_execute,
    _request_retry_now,
    _worker_is_active,
    run_worker,
    start_worker,
    worker_lock_held,
)


class FakeProcess:
    def __init__(self, pid, exit_code=None):
        self.pid = pid
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code


def test_v19_worker_recovers_without_repeating_saved_stage(service):
    svc, _ = service
    run_id = svc.create_run(runner="test", editor="test", pipeline="translate-edit")
    fake = FakeRunner()

    class RecoveringRunner:
        failures = 0

        def run(self, job):
            if len(fake.calls) == 1 and self.failures == 0:
                self.failures += 1
                raise RunnerExecutionError("temporary outage")
            return fake.run(job)

    runner = RecoveringRunner()
    svc.runner_factory = lambda _: runner
    result = run_worker(svc, run_id, restart_delay_seconds=0)
    assert result["status"] == "completed"
    assert len(fake.calls) == svc.status(run_id)["total_chunks"] * 2
    first_translation = next(j for j in fake.calls if j.stage == "translation")
    assert (
        sum(
            j.stage == "translation" and j.chunk_id == first_translation.chunk_id
            for j in fake.calls
        )
        == 1
    )
    metadata = json.loads((svc.run_dir(run_id) / "worker.json").read_text("utf-8"))
    assert metadata["recoveries"] == 1
    assert metadata["finished_at"]


def test_v19_pause_during_recovery_and_bounded_permanent_failure(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    calls = []

    class BrokenRunner:
        def run(self, job):
            calls.append(job)
            raise RunnerExecutionError("offline")

    svc.runner_factory = lambda _: BrokenRunner()

    def pause(_):
        svc.pause(run_id)

    result = run_worker(svc, run_id, restart_delay_seconds=1, sleep=pause)
    assert result["status"] == "paused"
    assert len(calls) == 1
    assert (svc.run_dir(run_id) / "pause.requested").exists()
    result = run_worker(svc, run_id, restart_delay_seconds=0, max_restarts=2)
    assert result["status"] == "failed"
    assert len(calls) == 4  # original failure + initial attempt + two recoveries
    assert not (svc.run_dir(run_id) / "book.json").exists()


def test_quota_worker_probes_before_provider_deadline(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    calls = []

    class QuotaOnce:
        def run(self, job):
            calls.append(job)
            if len(calls) == 1:
                raise RunnerDeferred("2099-01-01T00:00:00+00:00")
            return RunnerResult("Русский текст для проверки. " * 6, finish_reason="stop")

    svc.runner_factory = lambda _: QuotaOnce()
    waits = []
    result = run_worker(
        svc,
        run_id,
        quota_retry_poll_seconds=2,
        sleep=waits.append,
    )

    assert result["status"] == "completed"
    assert len(calls) == svc.status(run_id)["total_chunks"] + 1
    assert waits == [1, 1]
    assert not (svc.run_dir(run_id) / "retry.json").exists()
    metadata = json.loads((svc.run_dir(run_id) / "worker.json").read_text("utf-8"))
    assert metadata["retry_probe_interval_seconds"] == 2
    assert metadata["retry_probe_reason"] == "interval"


def test_quota_probe_uses_fixed_interval_near_provider_deadline(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    calls = []

    class QuotaOnce:
        def run(self, job):
            calls.append(job)
            if len(calls) == 1:
                raise RunnerDeferred((datetime.now(UTC) + timedelta(seconds=0.1)).isoformat())
            return RunnerResult("Русский текст для проверки. " * 6, finish_reason="stop")

    svc.runner_factory = lambda _: QuotaOnce()
    waits = []
    result = run_worker(svc, run_id, quota_retry_poll_seconds=2, sleep=waits.append)

    assert result["status"] == "completed"
    assert waits == [1, 1]


def test_http429_worker_repeated_deferrals_remain_bounded(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    calls = []

    class QuotaTwice:
        def run(self, job):
            calls.append(job)
            if len(calls) <= 2:
                raise RunnerDeferred("2000-01-01T00:00:00+00:00", reason="http429")
            return RunnerResult("Русский текст для проверки. " * 6, finish_reason="stop")

    svc.runner_factory = lambda _: QuotaTwice()
    result = run_worker(svc, run_id, quota_retry_poll_seconds=1, sleep=lambda _: None)

    assert result["status"] == "completed"
    assert len(calls) == svc.status(run_id)["total_chunks"] + 2
    assert not (svc.run_dir(run_id) / "retry.json").exists()


def test_retry_now_wakes_quota_worker(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    calls = []
    entered_wait = threading.Event()
    release_wait = threading.Event()

    class QuotaOnce:
        def run(self, job):
            calls.append(job)
            if len(calls) == 1:
                raise RunnerDeferred("2099-01-01T00:00:00+00:00")
            return RunnerResult("Русский текст для проверки. " * 6, finish_reason="stop")

    def controlled_sleep(_):
        entered_wait.set()
        assert release_wait.wait(5)

    svc.runner_factory = lambda _: QuotaOnce()
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(
            run_worker(svc, run_id, sleep=controlled_sleep, quota_retry_poll_seconds=900)
        )
    )
    thread.start()
    assert entered_wait.wait(5)
    _request_retry_now(svc.run_dir(run_id))
    release_wait.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert result["status"] == "completed"
    assert len(calls) == svc.status(run_id)["total_chunks"] + 1
    assert not (svc.run_dir(run_id) / RETRY_NOW_FILENAME).exists()


def test_pause_during_quota_backoff_stops_before_next_probe(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    calls = []

    class QuotaAlways:
        def run(self, job):
            calls.append(job)
            raise RunnerDeferred("2099-01-01T00:00:00+00:00")

    def pause(_):
        svc.pause(run_id)

    svc.runner_factory = lambda _: QuotaAlways()
    result = run_worker(svc, run_id, quota_retry_poll_seconds=900, sleep=pause)

    assert result["status"] == "paused"
    assert len(calls) == 1
    assert (svc.run_dir(run_id) / "pause.requested").exists()


def test_start_worker_requests_retry_now_for_live_waiting_worker(project, source, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    directory = svc.run_dir(run_id)
    write_json(
        directory / "retry.json",
        {
            "retry_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "reason": "quota",
            "deadline_source": "provider",
        },
    )
    write_json(
        directory / "worker.json",
        {"run_id": run_id, "pid": os.getpid(), "status": "running", "started_at": now()},
    )
    monkeypatch.setattr("novel_harness.progress._pid_alive", lambda pid, **kwargs: True)
    worker_lock = FileLock(str(directory / ".worker.lock"))
    worker_lock.acquire()
    try:
        result = start_worker(project, run_id)
    finally:
        worker_lock.release()

    assert result["started"] is False
    assert result["already_running"] is True
    assert result["retry_requested"] is True
    assert (directory / RETRY_NOW_FILENAME).is_file()


def test_start_worker_does_not_wake_nonquota_backoff(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    directory = svc.run_dir(run_id)
    write_json(
        directory / "worker.json",
        {
            "run_id": run_id,
            "pid": os.getpid(),
            "status": "waiting_retry",
            "started_at": now(),
            "current_failure_kind": "runner_error",
        },
    )
    monkeypatch.setattr("novel_harness.progress._pid_alive", lambda pid, **kwargs: True)
    worker_lock = FileLock(str(directory / ".worker.lock"))
    worker_lock.acquire()
    try:
        result = start_worker(project, run_id)
    finally:
        worker_lock.release()

    assert result["already_running"] is True
    assert result.get("retry_requested") is not True
    assert not (directory / RETRY_NOW_FILENAME).exists()


def test_v19_duplicate_worker_does_not_replace_metadata(service):
    import pytest
    from filelock import Timeout

    svc, _ = service
    run_id = svc.create_run(runner="test")
    path = svc.run_dir(run_id) / "worker.json"
    path.write_text('{"pid": 123}', "utf-8")
    from filelock import FileLock

    with FileLock(str(svc.run_dir(run_id) / ".worker.lock")), pytest.raises((Timeout, ValueError)):
        run_worker(svc, run_id)
    assert json.loads(path.read_text("utf-8")) == {"pid": 123}


def _fake_running_spawn(run_dir, counter=None, lock=None, held=None):
    """Simulate a real running worker: publish metadata and hold `.worker.lock`."""
    from filelock import FileLock

    def spawn(config, run_id, directory):
        if lock is not None and counter is not None:
            with lock:
                counter.append(run_id)
        worker_lock = FileLock(str(directory / ".worker.lock"))
        worker_lock.acquire()
        if held is not None:
            held.append(worker_lock)
        write_json(
            directory / "worker.json",
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "started_at": now(),
                "status": "running",
            },
        )
        return FakeProcess(os.getpid())

    return spawn


def test_start_worker_returns_while_fake_worker_stays_alive(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    held = []
    monkeypatch.setattr(
        "novel_harness.worker._spawn_worker",
        _fake_running_spawn(svc.run_dir(run_id), held=held),
    )

    started = time.monotonic()
    result = start_worker(project, run_id, startup_timeout=0.5)
    elapsed = time.monotonic() - started

    assert result["started"] is True
    assert result["already_running"] is False
    assert result["pid"] == os.getpid()
    assert elapsed < 1.5


def test_start_worker_is_idempotent_when_worker_active(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    spawns = []
    lock = threading.Lock()
    held = []
    monkeypatch.setattr(
        "novel_harness.worker._spawn_worker",
        _fake_running_spawn(svc.run_dir(run_id), counter=spawns, lock=lock, held=held),
    )

    first = start_worker(project, run_id, startup_timeout=0.5)
    second = start_worker(project, run_id, startup_timeout=0.5)

    assert first["started"] is True
    assert second["started"] is False
    assert second["already_running"] is True
    assert second["pid"] == os.getpid()
    assert len(spawns) == 1


def test_concurrent_double_start_creates_exactly_one_worker(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    spawns = []
    lock = threading.Lock()
    held = []
    monkeypatch.setattr(
        "novel_harness.worker._spawn_worker",
        _fake_running_spawn(svc.run_dir(run_id), counter=spawns, lock=lock, held=held),
    )

    results = []

    def launch():
        results.append(start_worker(project, run_id, startup_timeout=0.5))

    threads = [threading.Thread(target=launch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(spawns) == 1
    assert sum(bool(item["started"]) for item in results) == 1
    assert sum(bool(item.get("already_running")) for item in results) == 1


def test_start_worker_while_worker_is_executing_is_not_an_error(project, source):
    from filelock import FileLock

    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    run_lock = svc._lock(run_id)
    run_lock.acquire()
    worker_lock = FileLock(str(svc.run_dir(run_id) / ".worker.lock"))
    worker_lock.acquire()
    try:
        result = start_worker(project, run_id)
    finally:
        worker_lock.release()
        run_lock.release()

    assert result["started"] is False
    assert result["already_running"] is True


def test_completed_run_is_not_respawned(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    svc.execute(run_id)
    write_json(
        svc.run_dir(run_id) / "worker.json",
        {"pid": os.getpid(), "status": "completed", "epub": "/books/book.epub"},
    )

    def fail(*args, **kwargs):
        raise AssertionError("must not spawn for a completed run")

    monkeypatch.setattr("novel_harness.worker._spawn_worker", fail)

    result = start_worker(project, run_id)

    assert result["completed"] is True
    assert result["started"] is False
    assert result["epub"] == "/books/book.epub"


def test_worker_survives_transient_readonly_run_lock_probe(project, source):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    probe = svc._lock(run_id)
    probe.acquire()
    result = {}

    def work():
        result["value"] = run_worker(svc, run_id, restart_delay_seconds=0)

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.1)
    probe.release()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert result["value"]["status"] == "completed"


def test_bounded_execute_retries_transient_readonly_probe(project, source):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    probe = svc._lock(run_id)
    probe.acquire()
    result = {}

    def work():
        result["value"] = _bounded_execute(svc, run_id)

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.1)
    probe.release()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert result["value"]["status"] == "completed"


def test_bounded_build_epub_retries_transient_readonly_probe(project, source):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    svc.execute(run_id)
    probe = svc._lock(run_id)
    probe.acquire()
    result = {}

    def work():
        result["value"] = _bounded_build_epub(svc, run_id)

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.1)
    probe.release()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert result["value"].is_file()


def test_start_worker_reports_fast_child_completion(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")

    def fast_spawn(config, run_id_arg, directory):
        write_json(
            directory / "worker.json",
            {
                "run_id": run_id_arg,
                "pid": os.getpid(),
                "status": "completed",
                "epub": "/books/instant.epub",
            },
        )
        return FakeProcess(os.getpid(), exit_code=0)

    monkeypatch.setattr("novel_harness.worker._spawn_worker", fast_spawn)

    result = start_worker(project, run_id, startup_timeout=0.5)

    assert result["started"] is False
    assert result["completed"] is True
    assert result["worker_status"] == "completed"
    assert result["epub"] == "/books/instant.epub"


def test_immediate_child_failure_is_not_reported_as_started(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")

    def dead_spawn(config, run_id_arg, directory):
        return FakeProcess(os.getpid(), exit_code=1)

    monkeypatch.setattr("novel_harness.worker._spawn_worker", dead_spawn)

    result = start_worker(project, run_id, startup_timeout=0.2)

    assert result["started"] is False
    assert result["error"] == "worker_exited_during_startup"
    assert result["exit_code"] == 1
    state = json.loads((svc.run_dir(run_id) / "worker.json").read_text("utf-8"))
    assert state["status"] == "failed"


def test_run_worker_persists_base_duration_baseline(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    svc.repository.set_run(run_id, "paused", now(), duration=50.0)

    run_worker(svc, run_id, restart_delay_seconds=0)

    metadata = json.loads((svc.run_dir(run_id) / "worker.json").read_text("utf-8"))
    assert metadata["base_duration_seconds"] == 50.0


def test_worker_lock_held_probe_is_read_only(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    assert worker_lock_held(directory) is False

    lock = FileLock(str(directory / ".worker.lock"))
    lock.acquire()
    try:
        assert worker_lock_held(directory) is True
    finally:
        lock.release()

    assert worker_lock_held(directory) is False


def test_stale_running_metadata_with_free_lock_spawns_new_worker(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    write_json(
        svc.run_dir(run_id) / "worker.json",
        {"pid": os.getpid(), "status": "running", "started_at": now()},
    )
    # PID liveness says alive, but the lock is free: metadata is stale.
    monkeypatch.setattr("novel_harness.progress._pid_alive", lambda pid, **kwargs: True)
    spawns = []
    monkeypatch.setattr(
        "novel_harness.worker._spawn_worker",
        _fake_running_spawn(svc.run_dir(run_id), counter=spawns, lock=threading.Lock(), held=[]),
    )

    result = start_worker(project, run_id, startup_timeout=0.5)

    assert result["started"] is True
    assert result["already_running"] is False
    assert len(spawns) == 1


def test_live_worker_with_held_lock_is_active_and_not_respawned(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    write_json(
        svc.run_dir(run_id) / "worker.json",
        {"pid": os.getpid(), "status": "running", "started_at": now()},
    )
    monkeypatch.setattr("novel_harness.progress._pid_alive", lambda pid, **kwargs: True)
    worker_lock = FileLock(str(svc.run_dir(run_id) / ".worker.lock"))
    worker_lock.acquire()

    def fail(*args, **kwargs):
        raise AssertionError("must not spawn while the worker lock is held")

    monkeypatch.setattr("novel_harness.worker._spawn_worker", fail)
    try:
        result = start_worker(project, run_id)
    finally:
        worker_lock.release()

    assert result["started"] is False
    assert result["already_running"] is True


def test_fresh_starting_metadata_is_active_without_child_lock(project, source, monkeypatch):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    fresh = {"pid": os.getpid(), "status": "starting", "started_at": now()}
    write_json(svc.run_dir(run_id) / "worker.json", fresh)
    monkeypatch.setattr("novel_harness.progress._pid_alive", lambda pid, **kwargs: True)

    assert _worker_is_active(svc.run_dir(run_id))[0] is True

    stale = dict(fresh, started_at="2020-01-01T00:00:00+00:00")
    write_json(svc.run_dir(run_id) / "worker.json", stale)
    assert _worker_is_active(svc.run_dir(run_id))[0] is False


def test_dead_pid_running_metadata_is_not_active(project, source):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    svc = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    write_json(
        svc.run_dir(run_id) / "worker.json",
        {"pid": 2**31 - 1, "status": "running", "started_at": now()},
    )

    assert _worker_is_active(svc.run_dir(run_id))[0] is False


def test_incomplete_ceiling_survives_restart_and_other_progress(project, source):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    second = source.parent / "0002_second.txt"
    second.write_text("第二章\n\n" + "彼は森を歩いていた。\n\n" * 14, "utf-8")
    calls = []

    class IncompleteFirst:
        def run(self, job):
            calls.append((job.chapter, job.stage))
            if job.chapter == 1:
                raise RunnerExecutionError(
                    "incomplete", failure_kind="incomplete_response", finish_reason="length"
                )
            return RunnerResult("Русский текст для проверки. " * 6, finish_reason="stop")

    svc = TranslationService(load_config(project), lambda name: IncompleteFirst())
    run_id = svc.create_run(runner="test")

    first = run_worker(svc, run_id, restart_delay_seconds=0, max_restarts=2)
    assert first["status"] == "failed"
    restarted = TranslationService(load_config(project), lambda name: IncompleteFirst())
    result = run_worker(restarted, run_id, restart_delay_seconds=0, max_restarts=4)

    assert result["status"] == "failed"
    # Ceiling is ((retries + 1) * 2) = 2 for the retries:0 fixture and is durable
    # across the process restart; progress in chapter 2 never resets it.
    assert sum(chapter == 1 for chapter, _ in calls) == 2
    metadata = json.loads((svc.run_dir(run_id) / "worker.json").read_text("utf-8"))
    assert metadata["exhausted_chunks"] >= 1
    assert "recovery budget" in metadata["guidance"]
    second_chapter = [row for row in svc.repository.chunks(run_id) if row["chapter"] == 2]
    assert second_chapter and all(row["status"] == "completed" for row in second_chapter)


def test_worker_distinguishes_current_error_from_history(project, source):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    class FailOnce:
        def __init__(self):
            self.calls = 0

        def run(self, job):
            self.calls += 1
            if self.calls == 1:
                raise RunnerExecutionError("temporary outage", failure_kind="runner_error")
            return RunnerResult("Русский текст для проверки. " * 6, finish_reason="stop")

    runner = FailOnce()
    svc = TranslationService(load_config(project), lambda name: runner)
    run_id = svc.create_run(runner="test")
    result = run_worker(svc, run_id, restart_delay_seconds=0, max_restarts=3)

    assert result["status"] == "completed"
    metadata = json.loads((svc.run_dir(run_id) / "worker.json").read_text("utf-8"))
    assert "current_error" not in metadata
    assert metadata["recovered_from"] == "temporary outage"
    assert metadata["error_history"][-1]["failure_kind"] == "runner_error"


def test_incomplete_failure_records_reported_finish_and_usage(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")

    class Incomplete:
        def run(self, job):
            raise RunnerExecutionError(
                "incomplete",
                failure_kind="length",
                finish_reason="length",
                usage={"prompt_tokens": 10, "completion_tokens": 3},
            )

    svc.runner_factory = lambda _: Incomplete()
    with pytest.raises(RunnerExecutionError):
        svc.execute(run_id)
    chunk = svc.repository.chunks(run_id)[0]
    attempts = svc.repository.attempts(run_id, chunk["chapter"], chunk["chunk_id"], "translation")
    last = attempts[-1]
    assert last["failure_kind"] == "length"
    assert last["finish_reason"] == "length"
    assert last["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}


def test_quota_deferral_does_not_consume_incomplete_budget(service):
    from datetime import UTC, datetime, timedelta

    svc, _ = service
    run_id = svc.create_run(runner="test")

    class Quota:
        def run(self, job):
            raise RunnerDeferred((datetime.now(UTC) + timedelta(hours=1)).isoformat())

    svc.runner_factory = lambda _: Quota()
    with pytest.raises(RunnerDeferred):
        svc.execute(run_id)
    chunk = svc.repository.chunks(run_id)[0]
    attempts = svc.repository.attempts(run_id, chunk["chapter"], chunk["chunk_id"], "translation")
    assert attempts[-1]["failure_kind"] == "quota"
    assert not any(item.get("failure_kind") == "recovery_exhausted" for item in attempts)


def test_reviewer_failure_reuses_valid_translator_draft(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    calls = {"translation": [], "review": []}
    failed_once = {"value": False}

    class FlakyReview:
        def run(self, job):
            if job.output_mode == "review":
                calls["review"].append(job.chunk_id)
                if not failed_once["value"]:
                    failed_once["value"] = True
                    raise RunnerExecutionError("reviewer offline", failure_kind="runner_error")
                value = {"checked_ids": [s["id"] for s in job.segments], "issues": []}
            else:
                calls["translation"].append(job.chunk_id)
                value = {
                    "segments": [
                        {"id": s["id"], "text": f"Русский абзац номер {i}."}
                        for i, s in enumerate(job.segments)
                    ]
                }
            return RunnerResult(json.dumps(value), finish_reason="stop")

    svc.runner_factory = lambda _: FlakyReview()
    run_id = svc.create_run(translator="test", editor="test", pipeline="translate-review")
    with pytest.raises(RunnerExecutionError):
        svc.execute(run_id)
    first_chunk = calls["translation"][0]
    assert calls["translation"].count(first_chunk) == 1
    assert svc.resume(run_id)["status"] == "completed"
    # The valid translator draft is reused instead of being retranslated.
    assert calls["translation"].count(first_chunk) == 1
    assert calls["review"].count(first_chunk) == 2
