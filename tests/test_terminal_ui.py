"""Offline tests for explicit terminal worker controls."""

from __future__ import annotations

import time
from threading import Event

from novel_harness import terminal_ui
from novel_harness.terminal_ui import ActionExecutor


def _take(executor):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        result = executor.take()
        if result[2] is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("action did not complete")


def test_action_executor_captures_full_id_and_rejects_overlap():
    calls = []
    started = Event()
    release = Event()

    class Service:
        def pause(self, run_id):
            calls.append(run_id)
            started.set()
            release.wait(timeout=2)
            return {"run_id": run_id, "pause_requested": True}

    executor = ActionExecutor()
    assert executor.dispatch(Service(), "config.yaml", "space", "full-run-id", active=True)
    assert started.wait(timeout=1)
    assert not executor.dispatch(Service(), "config.yaml", "space", "other-run", active=True)
    release.set()
    action, target, result = _take(executor)
    executor.close()

    assert (action, target) == ("space", "full-run-id")
    assert result["pause_requested"] is True
    assert calls == ["full-run-id"]


def test_start_resume_and_retry_share_existing_launcher(monkeypatch):
    calls = []

    def fake_launch(service, config, run_id):
        calls.append((config, run_id))
        return {"run_id": run_id, "started": True}

    monkeypatch.setattr(terminal_ui, "launch_run", fake_launch)
    executor = ActionExecutor()
    for action in ("s", "u", "t"):
        assert executor.dispatch(object(), "config.yaml", action, "captured-run", active=False)
        assert _take(executor)[2]["started"] is True
    executor.close()

    assert calls == [("config.yaml", "captured-run")] * 3


def test_library_action_creates_source_only_run_with_effective_preset(monkeypatch):
    calls = []

    class Service:
        def launch_remaining_novel(self, novel_id, config):
            calls.append((novel_id, config))
            return {"run_id": "new-full-run", "started": True}

    executor = ActionExecutor()
    row = {
        "novel_id": "novel-source",
        "continue_run_id": None,
        "effective_preset": "local-fast",
    }
    assert executor.dispatch_book(Service(), "config.yaml", row)
    action, target, result = _take(executor)
    executor.close()

    assert (action, target) == ("launch", "novel-source")
    assert result["run_id"] == "new-full-run"
    assert calls == [("novel-source", "config.yaml")]


def test_library_action_continues_captured_run_without_creating_new_run(monkeypatch):
    calls = []

    class Service:
        def launch_remaining_novel(self, novel_id, config):
            calls.append((novel_id, config))
            return {"run_id": "remaining-run", "started": True}

    executor = ActionExecutor()
    assert executor.dispatch_book(
        Service(),
        "config.yaml",
        {
            "novel_id": "novel",
            "continue_run_id": "full-saved-run",
            "effective_preset": "local-fast",
            "preset_available": False,
        },
    )
    action, target, result = _take(executor)
    executor.close()

    assert (action, target) == ("continue", "novel")
    assert result["run_id"] == "remaining-run"
    assert calls == [("novel", "config.yaml")]


def test_library_space_launches_remaining_chapters_after_completed_sample(project, monkeypatch):
    from conftest import FakeRunner

    from novel_harness.config import load_config
    from novel_harness.run_dashboard import library_rows
    from novel_harness.service import TranslationService

    book = project.parent / "original" / "partial-book"
    book.mkdir()
    for number in (1, 2):
        (book / f"{number:04d}.txt").write_text(
            f"第{number}章\n\n" + "合成本文。" * 20,
            "utf-8",
        )
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    service.rescan_library()
    novel_id = next(
        row["novel_id"] for row in service.library_novels()["novels"] if row["title"] == "partial-book"
    )
    sample = service.create_sample_run(runner="test", chapters=[1], novel_id=novel_id)
    service.execute(sample)
    row = next(item for item in library_rows(service) if item["novel_id"] == novel_id)
    assert row["status"] == "remaining"
    assert row["continue_run_id"] == sample

    launched = []
    monkeypatch.setattr(
        "novel_harness.launch.launch_run",
        lambda service_arg, config, run_id: launched.append(run_id)
        or {"run_id": run_id, "started": True},
    )
    executor = ActionExecutor()
    assert executor.dispatch_book(service, project, row)
    _action, _target, result = _take(executor)
    executor.close()

    assert result["run_id"] != sample
    assert launched == [result["run_id"]]
    manifest = service.repository.run(result["run_id"])["manifest"]
    assert [chapter["number"] for chapter in manifest["chapters"]] == [2]
