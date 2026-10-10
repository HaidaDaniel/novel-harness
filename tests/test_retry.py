import json
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from novel_harness.runners.base import RunnerDeferred, RunnerResult
from novel_harness.runners.retry import http_deferred, quota_event
from novel_harness.service import TranslationService
from novel_harness.worker import run_worker


def test_v26_real_codex_calendar_and_safe_event_scope():
    message = "You've hit your usage limit. Try again at Sep 7th, 2026 2:00 AM."
    current = datetime(2026, 9, 6, 20, tzinfo=UTC)
    deferred = quota_event(
        {"type": "error", "message": message},
        current=current,
        local_timezone=timezone(timedelta(hours=3)),
    )
    assert deferred.retry_at == "2026-09-06T23:00:00+00:00"
    assert quota_event({"type": "text", "message": message}) is None
    assert (
        quota_event(
            {
                "type": "item.completed",
                "item": {"type": "error", "message": "Code Mode is unavailable"},
            }
        )
        is None
    )


def test_v26_http_retry_after():
    current = datetime(2026, 9, 8, tzinfo=UTC)
    error = SimpleNamespace(
        status_code=429, response=SimpleNamespace(headers={"retry-after": "120"})
    )
    assert http_deferred(error, current=current).retry_at == "2026-09-08T00:02:00+00:00"
    assert http_deferred(SimpleNamespace(status_code=500)) is None


def test_nonterminal_finish_reason_does_not_complete_chunk(service):
    svc, _ = service

    class ToolCalls:
        def run(self, job):
            text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
                1, len(job.source) // 20
            )
            return RunnerResult(text, model="fake-model", finish_reason="tool_calls")

    svc.runner_factory = lambda _: ToolCalls()
    rid = svc.create_run(runner="test")
    result = svc.execute(rid)

    rows = svc.repository.chunks(rid)
    # No nonterminal turn may be committed as a completed translation.
    assert not any(row["status"] == "completed" for row in rows)
    assert any(row["status"] == "failed" for row in rows)
    assert result["status"] == "failed"


def test_absent_finish_reason_follows_legacy_contract(service):
    svc, _ = service

    class NoReason:
        def run(self, job):
            text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
                1, len(job.source) // 20
            )
            return RunnerResult(text, model="fake-model", finish_reason=None)

    svc.runner_factory = lambda _: NoReason()
    rid = svc.create_run(runner="test")
    assert svc.execute(rid)["status"] == "completed"


def test_v26_deadline_survives_restart_and_stage_retries(service):
    svc, fake = service
    svc.config.translation.retries = 5
    calls = []

    class Limited:
        def run(self, job):
            calls.append(job)
            raise RunnerDeferred((datetime.now(UTC) + timedelta(hours=3)).isoformat())

    svc.runner_factory = lambda _: Limited()
    rid = svc.create_run(runner="test")
    with pytest.raises(RunnerDeferred):
        svc.execute(rid)
    restarted = TranslationService(svc.config, svc.runner_factory)
    with pytest.raises(RunnerDeferred):
        restarted.execute(rid)
    assert len(calls) == 1
    assert restarted.status(rid)["status"] == "waiting_retry"
    result = run_worker(restarted, rid, sleep=lambda _: restarted.pause(rid))
    assert result["status"] == "paused" and len(calls) == 1
    worker = json.loads((svc.run_dir(rid) / "worker.json").read_text("utf-8"))
    assert worker["recoveries"] == 0
    assert worker["retry_at"]
    retry = svc.run_dir(rid) / "retry.json"
    saved = json.loads(retry.read_text("utf-8"))
    saved["retry_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    retry.write_text(json.dumps(saved), encoding="utf-8")
    restarted.runner_factory = lambda _: fake
    assert restarted.execute(rid)["status"] == "completed"
    assert not retry.exists()
