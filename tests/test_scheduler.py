"""Bounded scheduler: capacity, ordering, error isolation and cancellations."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest
import yaml

from novel_harness.config import load_config
from novel_harness.runners.base import RunnerExecutionError, RunnerResult
from novel_harness.scheduler import BoundedScheduler
from novel_harness.service import TranslationService

ROOT = Path(__file__).resolve().parents[1]


def test_rejects_nonpositive_workers():
    with pytest.raises(ValueError):
        BoundedScheduler(0)
    with pytest.raises(ValueError):
        BoundedScheduler(-1)


def test_capacity_is_bounded():
    scheduler = BoundedScheduler(2)
    lock = threading.Lock()
    active = 0
    peak = 0

    def work(item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return item

    results = scheduler.run(list(range(8)), work)

    assert [result.value for result in results] == list(range(8))
    assert peak <= 2


def test_results_preserve_input_order():
    scheduler = BoundedScheduler(4)

    def work(item):
        time.sleep((10 - item) * 0.003)
        return item

    results = scheduler.run(list(range(10)), work)

    assert [result.value for result in results] == list(range(10))


def test_errors_are_isolated_per_unit():
    scheduler = BoundedScheduler(3)

    def work(item):
        if item == 1:
            raise RuntimeError("boom")
        return item

    results = scheduler.run([0, 1, 2], work)

    assert results[0].value == 0 and results[0].error is None
    assert isinstance(results[1].error, RuntimeError)
    assert results[2].value == 2 and results[2].error is None


def test_should_stop_initially_prevents_all_submissions():
    scheduler = BoundedScheduler(2)
    calls = []

    results = scheduler.run([1, 2, 3], lambda item: calls.append(item), should_stop=lambda: True)

    assert results == []
    assert calls == []


def test_should_stop_limits_submissions():
    scheduler = BoundedScheduler(2)
    calls = []
    counter = {"n": 0}

    def should_stop():
        counter["n"] += 1
        return counter["n"] > 2

    results = scheduler.run(
        [1, 2, 3, 4, 5], lambda item: calls.append(item), should_stop=should_stop
    )

    # Only the initial capacity is submitted before the stop flag is observed.
    assert [result.item for result in results] == [1, 2]


def test_backpressure_does_not_invoke_queued_units_after_stop():
    scheduler = BoundedScheduler(2)
    release = threading.Event()
    started = []
    lock = threading.Lock()
    two_started = threading.Event()

    def work(item):
        with lock:
            started.append(item)
            if len(started) == 2:
                two_started.set()
        # Both initial workers block until the caller requests a stop.
        assert release.wait(timeout=5)
        return item

    results = []
    runner = threading.Thread(
        target=lambda: results.extend(
            scheduler.run([0, 1, 2, 3], work, should_stop=release.is_set)
        )
    )
    runner.start()
    assert two_started.wait(timeout=5)
    release.set()
    runner.join(timeout=5)

    assert not runner.is_alive()
    assert sorted(started) == [0, 1]
    assert [result.item for result in results] == [0, 1]


def test_caller_can_skip_completed_units_for_idempotency():
    scheduler = BoundedScheduler(2)
    completed = {1}
    pending = [item for item in range(4) if item not in completed]

    results = scheduler.run(pending, lambda item: item)

    assert [result.value for result in results] == [0, 2, 3]


class TrackingRunner:
    def __init__(self, fail_calls=0, fail_chunk_id=None):
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.fail_calls = fail_calls
        self.fail_chunk_id = fail_chunk_id
        self.lock = threading.Lock()

    def run(self, job):
        with self.lock:
            self.calls += 1
            active_call = self.calls
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        if active_call <= self.fail_calls or (
            self.fail_chunk_id is not None and job.chunk_id == self.fail_chunk_id
        ):
            raise RunnerExecutionError("synthetic runner failure")
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


def _make_project(tmp_path, *, context_strategy="none", chunk_chars=60):
    (tmp_path / "original").mkdir()
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["translation"].update(
        chunk_chars=chunk_chars,
        overlap_chars=0,
        context_strategy=context_strategy,
        retries=0,
        retry_delay_seconds=0,
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
        "第一章\n\n" + "彼は森を歩いていた。\n\n" * 14, "utf-8"
    )
    return path


def test_parallel_execution_bounds_concurrency_and_is_idempotent(tmp_path):
    path = _make_project(tmp_path)
    runner = TrackingRunner()
    service = TranslationService(load_config(path), lambda name: runner)

    run_id = service.create_run(runner="test")
    result = service.execute(run_id, max_parallel=2)

    rows = service.repository.chunks(run_id)
    assert result["status"] == "completed"
    assert runner.calls == len(rows)
    assert 1 <= runner.peak <= 2
    assert [row["ordinal"] for row in rows] == sorted(row["ordinal"] for row in rows)

    calls = runner.calls
    service.resume(run_id)
    assert runner.calls == calls


def test_pause_stops_queued_parallel_chunks(tmp_path):
    path = _make_project(tmp_path)
    # Two workers plus this test thread rendezvous, so both workers are
    # provably in flight and blocked before the pause is requested.
    both_started = threading.Barrier(3)
    release = threading.Event()
    calls = []
    lock = threading.Lock()

    class PausingRunner:
        def run(self, job):
            with lock:
                calls.append(job.chunk_id)
                order = len(calls)
            if order <= 2:
                assert both_started.wait(timeout=5)
                assert release.wait(timeout=5)
            text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
                1, len(job.source) // 20
            )
            return RunnerResult(text, model="fake-model", finish_reason="stop")

    service = TranslationService(load_config(path), lambda name: PausingRunner())
    run_id = service.create_run(runner="test")
    assert len(service.repository.chunks(run_id)) == 3

    result = {}
    worker = threading.Thread(
        target=lambda: result.update(service.execute(run_id, max_parallel=2))
    )
    worker.start()
    both_started.wait(timeout=5)
    (service.run_dir(run_id) / "pause.requested").write_text("", "utf-8")
    release.set()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert len(calls) == 2
    assert result["status"] == "paused"
    states = sorted(row["status"] for row in service.repository.chunks(run_id))
    assert states == ["completed", "completed", "pending"]
    assert (service.run_dir(run_id) / "pause.requested").exists()


@pytest.mark.parametrize("failed_ordinal", [1, 2, 3])
def test_parallel_failure_commits_all_successful_siblings(tmp_path, failed_ordinal):
    path = _make_project(tmp_path)
    setup = TranslationService(load_config(path), lambda name: TrackingRunner())
    run_id = setup.create_run(runner="test")
    chunks = setup.repository.chunks(run_id)
    assert len(chunks) == 3
    target = next(row for row in chunks if row["ordinal"] == failed_ordinal)

    failing = TrackingRunner(fail_chunk_id=target["chunk_id"])
    service = TranslationService(load_config(path), lambda name: failing)
    result = service.execute(run_id, max_parallel=2)

    rows = service.repository.chunks(run_id)
    assert result["status"] in {"failed", "incomplete"}
    assert sorted(row["status"] for row in rows) == ["completed", "completed", "failed"]
    failed = next(row for row in rows if row["status"] == "failed")
    assert failed["chunk_id"] == target["chunk_id"]
    # The failed chunk keeps the real persisted diagnostic rather than a bare flag.
    assert failed["metadata"] and "synthetic runner failure" in failed["metadata"]

    calls = failing.calls
    healthy = TrackingRunner()
    recovered = TranslationService(load_config(path), lambda name: healthy)
    assert recovered.resume(run_id)["status"] == "completed"
    assert healthy.calls == 1
    assert failing.calls == calls


def test_parallel_failure_leaves_recoverable_state(tmp_path):
    path = _make_project(tmp_path)
    failing = TrackingRunner(fail_calls=1)
    service = TranslationService(load_config(path), lambda name: failing)

    run_id = service.create_run(runner="test")
    result = service.execute(run_id, max_parallel=2)
    assert result["status"] in {"failed", "incomplete"}
    assert any(row["status"] == "failed" for row in service.repository.chunks(run_id))

    healthy = TrackingRunner()
    recovered = TranslationService(load_config(path), lambda name: healthy)
    assert recovered.resume(run_id)["status"] == "completed"


@pytest.mark.parametrize("failure_location", ["runner", "context"])
def test_parallel_requeue_preserves_current_failure_instead_of_old_editor_success(
    service, monkeypatch, failure_location
):
    import json

    svc, fake = service
    svc.config.translation.context_strategy = "none"
    svc.config.translation.overlap_chars = 0
    rid = svc.create_run(translator="test", editor="test", pipeline="translate-edit")
    assert svc.execute(rid, max_parallel=2)["status"] == "completed"
    row = svc.repository.chunks(rid)[0]
    svc.requeue_chunk(rid, row["chapter"], row["chunk_id"],
                      expected_machine_sha256=row["output_sha256"])

    class FailingRunner:
        def run(self, job):
            raise RunnerExecutionError(
                "fresh translation failure", usage={"prompt_tokens": 17, "completion_tokens": 3}
            )

    if failure_location == "runner":
        monkeypatch.setattr(svc, "runner_factory", lambda _name: FailingRunner())
    else:
        def broken_context(*args):
            raise ValueError("CANARY-secret-context")
        monkeypatch.setattr(svc, "_chunk_context", broken_context)
    result = svc.execute(rid, max_parallel=2)
    assert result["status"] == "failed"
    rows = svc.repository.chunks(rid)
    failed = next(chunk for chunk in rows if chunk["chunk_id"] == row["chunk_id"])
    metadata = json.loads(failed["metadata"])
    assert failed["status"] == "failed"
    assert metadata["status"] == "failed"
    if failure_location == "runner":
        assert metadata["stage"] == "translation"
        assert metadata["error"] == "fresh translation failure"
        assert metadata["usage"] == {"prompt_tokens": 17, "completion_tokens": 3}
    else:
        assert metadata["error"] == "ValueError"
        assert metadata.get("stage") != "editing"
        assert "CANARY" not in failed["metadata"]
    assert all(chunk["status"] == "completed" for chunk in rows if chunk is not failed)
    monkeypatch.undo()
    previous_calls = len(fake.calls)
    assert svc.execute(rid, max_parallel=2)["status"] == "completed"
    assert len(fake.calls) - previous_calls == 2
