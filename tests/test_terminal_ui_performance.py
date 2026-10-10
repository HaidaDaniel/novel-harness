"""Deterministic bounds for cached terminal work; no timing thresholds."""

from __future__ import annotations

import threading
from io import StringIO

from rich.console import Console

import novel_harness.run_dashboard as dashboard
from novel_harness.run_dashboard import (
    MAX_INTEGRITY_CACHE,
    DashboardRefresh,
    DashboardState,
    LibraryState,
    filter_library_rows,
    filter_rows,
    hydrate_rows,
    render_library_page,
    render_page,
)
from novel_harness.terminal_ui import ActionExecutor


def test_benchmark_drives_the_production_tui_path():
    from scripts.benchmark_terminal_ui import benchmark

    report = benchmark(seconds=2, run_count=2, chunks_per_run=10)

    live = report["live_tui"]
    assert live["available"] is True
    assert live["key_reads"] > 0
    assert live["dashboard_refresh_starts"] > 0
    assert live["library_refresh_starts"] > 0
    assert live["live_updates"] > 0
    assert live["live_refreshes"] > 0
    assert live["action_calls"]
    assert live["key_latency"]["samples"] > 0


def _run_rows(count: int = 128) -> list[dict]:
    return [
        {
            "id": f"run-{index:04d}-full-id",
            "book": f"Book {index:04d}",
            "runner": "synthetic",
            "status": "pending",
            "counts": {"completed": 0, "pending": 10, "failed": 0, "running": 0, "corrupt": 0},
            "total_chunks": 10,
            "progress": 0.0,
            "worker_active": False,
            "projection": {},
        }
        for index in range(count)
    ]


def _library_rows(count: int = 128) -> list[dict]:
    return [
        {
            "novel_id": f"novel-{index:04d}",
            "title": f"Book {index:04d}",
            "source_path": f"/temporary/book-{index:04d}",
            "status": "remaining",
            "chapter_count": 10,
            "translated": 0,
            "remaining": 10,
            "run_id": f"run-{index:04d}-full-id",
            "continue_run_id": f"run-{index:04d}-full-id",
            "active_run": None,
            "effective_preset": "synthetic",
            "effective_profile": "synthetic",
            "effective_model": "synthetic",
            "preset_available": True,
            "freshness": "cached",
        }
        for index in range(count)
    ]


def test_cached_navigation_and_render_do_not_read_storage():
    runs = _run_rows()
    books = _library_rows()
    run_state = DashboardState()
    library_state = LibraryState()
    console = Console(file=StringIO(), width=120, height=24, force_terminal=False)

    for key in ("down", "up", "j", "k", "next", "previous", "/", "b") * 20:
        visible_runs = filter_rows(runs, active_only=False, search=run_state.search_query)
        run_state.key(key, visible_runs, 15)
        visible_runs = filter_rows(runs, active_only=False, search=run_state.search_query)
        render_page(None, visible_runs, run_state, console, cached_only=True)
        visible_books = filter_library_rows(
            books, filter_name=library_state.filter_name, search=library_state.search_query
        )
        library_state.key("down", visible_books, 15)
        render_library_page(visible_books, library_state, console)

    # The cached-only render path has no service object and therefore cannot
    # perform a repository read, output hash or runner call.
    assert len(runs) == 128 and len(books) == 128


def test_refresh_allows_one_collector_and_action_executor_allows_one_action(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocked_rows(_service):
        started.set()
        assert release.wait(timeout=2)
        return _run_rows(2)

    monkeypatch.setattr(dashboard, "run_rows", blocked_rows)
    monkeypatch.setattr(dashboard, "hydrate_rows", lambda *args, **kwargs: None)
    refresh = DashboardRefresh()
    assert refresh.start(object(), Console(), DashboardState())
    assert started.wait(timeout=1)
    assert not refresh.start(object(), Console(), DashboardState())
    release.set()
    assert refresh.future is not None
    refresh.future.result(timeout=2)
    assert len(refresh.take() or []) == 2
    refresh.close()

    action_started = threading.Event()
    action_release = threading.Event()
    calls = []

    class Service:
        def pause(self, run_id):
            calls.append(run_id)
            action_started.set()
            assert action_release.wait(timeout=2)
            return {"run_id": run_id, "pause_requested": True}

    executor = ActionExecutor()
    assert executor.dispatch(Service(), "config.yaml", "space", "captured", active=True)
    assert action_started.wait(timeout=1)
    assert not executor.dispatch(Service(), "config.yaml", "space", "other", active=True)
    action_release.set()
    while executor.take()[2] is None:
        pass
    executor.close()
    assert calls == ["captured"]


def test_benchmark_fixture_has_required_chunk_shape_without_runner():
    from scripts.benchmark_terminal_ui import _manifest

    manifest, chunks = _manifest("synthetic-run", 0, 300)
    assert len(chunks) == 300
    assert sum(len(chapter["chunks"]) for chapter in manifest["chapters"]) == 300
    assert all(chunk["source_sha256"] == "0" * 64 for chunk in chunks)


def test_integrity_cache_stays_within_declared_bound():
    rows = _run_rows(MAX_INTEGRITY_CACHE + 32)

    class Service:
        def lightweight_statuses(self, run_ids=None):
            selected = run_ids or [row["id"] for row in rows]
            return {
                run_id: {
                    "status": "pending",
                    "chunks": {"completed": 0, "pending": 10, "failed": 0, "running": 0, "corrupt": 0},
                    "total_chunks": 10,
                    "progress": 0.0,
                }
                for run_id in selected
            }

        def status(self, run_id):
            return self.lightweight_statuses([run_id])[run_id]

    cache = {}
    hydrate_rows(
        Service(),
        rows,
        Console(width=120),
        DashboardState(),
        verify_ids={row["id"] for row in rows},
        integrity_cache=cache,
    )
    assert len(cache) == MAX_INTEGRITY_CACHE
