"""Read-only paginated terminal dashboard tests; no model calls."""

from __future__ import annotations

import json
import os

from rich.console import Console
from typer.testing import CliRunner

from novel_harness.cli import app
from novel_harness.run_dashboard import (
    DashboardState,
    LibraryState,
    _next_refresh_delay,
    _read_posix,
    filter_library_rows,
    filter_rows,
    group_runs,
    hydrate_rows,
    library_rows,
    render_page,
    run_rows,
)


def test_dashboard_lists_saved_runs_and_pages_without_inference(service):
    translation, fake = service
    first = translation.create_run(runner="test", chapters=[1])
    second = translation.create_run(runner="test", chapters=[1], force=True)
    rows = run_rows(translation)
    assert [row["id"] for row in rows] == [second, first]

    console = Console(width=120, height=11, record=True, force_terminal=False)
    state = DashboardState()
    console.print(render_page(translation, rows, state, console, once=True))
    rendered = console.export_text(clear=False)
    assert "page 1/1" in rendered
    assert second in rendered
    assert fake.calls == []


def test_dashboard_navigation_selects_pages_details_and_filter():
    rows = [{"id": str(i)} for i in range(7)]
    state = DashboardState()
    assert state.key("next", rows, 3)
    assert state.selected == 3
    assert state.key("down", rows, 3)
    assert state.selected == 4
    assert state.key("enter", rows, 3)
    assert state.detail_run_id == "4"
    assert state.key("right", rows, 3)
    assert state.detail_run_id == "5"
    assert state.selected == 5
    assert state.key("left", rows, 3)
    assert state.detail_run_id == "4"
    assert state.selected == 4
    assert state.key("p", rows, 3)
    assert state.detail_run_id == "3"
    assert state.key("n", rows, 3)
    assert state.detail_run_id == "4"
    # Live sort order can change while details are open; navigate by run ID.
    reordered = [rows[4], *rows[:4], *rows[5:]]
    assert state.key("left", reordered, 3)
    assert state.selected == 0 and state.detail_run_id == "4"
    assert state.key("right", reordered, 3)
    assert state.selected == 1 and state.detail_run_id == "0"
    assert state.key("b", rows, 3)
    assert state.detail_run_id is None
    assert state.key("a", rows, 3)
    assert state.active_only is True and state.selected == 0
    assert not state.key("q", rows, 3)


def test_library_projection_caches_coverage_and_settings_without_scan(project):
    from conftest import FakeRunner

    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    root = project.parent / "original"
    book = root / "source-only"
    book.mkdir()
    for number in (1, 2):
        (book / f"{number:04d}.txt").write_text(
            f"第{number}章\n\n" + "合成本文。" * 20, "utf-8"
        )
    partial_book = root / "partial"
    partial_book.mkdir()
    for number in (1, 2):
        (partial_book / f"{number:04d}.txt").write_text(
            f"第{number}章\n\n" + "部分本文。" * 20, "utf-8"
        )
    unavailable = root / "unavailable"
    unavailable.mkdir()
    (unavailable / "0001.txt").write_text("第一章\n\n失效来源。" * 20, "utf-8")
    (unavailable / "novel.json").write_text('{"version": 1, "unknown": true}', "utf-8")
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    rows = library_rows(service)
    source = next(row for row in rows if row["title"] == "source-only")
    assert source["translated"] == 0
    assert source["remaining"] == 2
    assert source["continue_run_id"] is None
    assert source["effective_preset"] == "local-fast"
    assert source["effective_profile"]

    partial = next(row for row in rows if row["title"] == "partial")
    partial_run = service.create_run(runner="test", chapters=[1], novel_id=partial["novel_id"])
    service.execute(partial_run)
    partial = next(row for row in library_rows(service) if row["novel_id"] == partial["novel_id"])
    assert partial["status"] == "remaining"
    assert (partial["translated"], partial["remaining"]) == (1, 1)
    unavailable_row = next(row for row in rows if row["title"] == "unavailable")
    assert unavailable_row["status"] == "unavailable"
    assert unavailable_row["reason"]

    novel_id = source["novel_id"]
    run_id = service.create_run(runner="test", novel_id=novel_id)
    service.execute(run_id)
    complete = next(row for row in library_rows(service) if row["novel_id"] == novel_id)
    assert complete["status"] == "complete"
    assert complete["translated"] == complete["chapter_count"] == 2
    assert complete["continue_run_id"] == run_id


def test_library_search_and_filters_preserve_novel_identity():
    rows = [
        {"novel_id": "novel-a", "title": "Alpha", "source_path": "/a", "status": "remaining", "active_run": None},
        {"novel_id": "novel-b", "title": "Beta", "source_path": "/b", "status": "complete", "active_run": {"run_id": "run-b"}},
    ]
    state = LibraryState(selected_novel_id="novel-b")
    visible = filter_library_rows(rows, filter_name="active")
    state.sync(visible)
    assert state.selected_novel_id == "novel-b"
    assert state.key("/", visible, 10)
    assert state.key("b", visible, 10)
    assert filter_library_rows(rows, search=state.search_query)[0]["novel_id"] == "novel-b"


def test_dashboard_detail_navigation_clamps_at_boundaries():
    rows = [{"id": "first"}, {"id": "last"}]
    state = DashboardState()
    assert state.key("enter", rows, 3)
    assert state.key("left", rows, 3)
    assert state.selected == 0 and state.detail_run_id == "first"
    assert state.key("right", rows, 3)
    assert state.key("right", rows, 3)
    assert state.selected == 1 and state.detail_run_id == "last"
    assert state.key("esc", rows, 3)
    assert state.detail_run_id is None and state.selected == 1


def test_manual_refresh_is_coalesced_and_selection_tracks_run_identity():
    rows = [
        {"id": "old", "worker_active": False},
        {"id": "other", "worker_active": False},
    ]
    state = DashboardState()
    state.key("down", rows, 10)
    assert state.selected_run_id == "other"
    assert state.key("r", rows, 10)
    assert state.refresh_requested is True
    assert state.key("r", rows, 10)
    assert state.refresh_requested is True
    state.detail_run_id = "other"
    state.refresh_requested = False
    assert state.key("r", rows, 10)
    assert state.refresh_requested is True

    reordered = [{"id": "new-active", "worker_active": True}, *rows]
    state.sync(reordered)
    assert state.selected == 2
    assert state.selected_run_id == "other"

    state.sync([{"id": "old", "worker_active": False}])
    assert state.selected == 0
    assert state.selected_run_id == "old"


def test_idle_refresh_backoff_is_longer_than_active_refresh():
    assert _next_refresh_delay([], 5.0) == 15.0
    assert _next_refresh_delay([{"worker_active": False}], 5.0) == 15.0
    assert _next_refresh_delay([{"worker_active": True}], 5.0) == 5.0


def test_cli_status_is_human_table_and_json_is_stable(project, source):
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    translation = TranslationService(load_config(project))
    run_id = translation.create_run(runner="test", chapters=[1])
    cli = CliRunner()

    human = cli.invoke(app, ["status", "--config", str(project)])
    assert human.exit_code == 0, human.output
    assert "Translations" in human.output
    assert run_id in human.output

    table = cli.invoke(app, ["dashboard", "--once", "--config", str(project)])
    assert table.exit_code == 0, table.output
    assert run_id in table.output

    detail = cli.invoke(app, ["status", "--run", run_id, "--config", str(project)])
    assert detail.exit_code == 0, detail.output
    assert run_id in detail.output

    machine = cli.invoke(app, ["status", "--config", str(project), "--json"])
    assert machine.exit_code == 0, machine.output
    assert json.loads(machine.output)["runs"][0]["run_id"] == run_id


def test_interactive_dashboard_requires_tty(project):
    result = CliRunner().invoke(app, ["dashboard", "--config", str(project)])
    assert result.exit_code == 2
    assert "use --once" in result.output


def test_navigation_reuses_cached_rows_without_rescan_or_hash(service):
    translation, fake = service
    for _ in range(6):
        run_id = translation.create_run(runner="test", chapters=[1], force=True)
        translation.execute(run_id)

    console = Console(width=120, height=24, record=True, force_terminal=False)
    state = DashboardState()
    all_rows = run_rows(translation)
    rows = filter_rows(all_rows, active_only=state.active_only)
    hydrate_rows(translation, rows, console, state)

    calls = {"summaries": 0, "valid": 0}
    original_summaries = translation.repository.list_run_summaries
    original_valid = translation.stages.valid_output

    def counting_summaries(*args, **kwargs):
        calls["summaries"] += 1
        return original_summaries(*args, **kwargs)

    def counting_valid(*args, **kwargs):
        calls["valid"] += 1
        return original_valid(*args, **kwargs)

    translation.repository.list_run_summaries = counting_summaries
    translation.stages.valid_output = counting_valid

    for key in ["down", "down", "up", "down", "next", "previous", "a", "a"]:
        rows = filter_rows(all_rows, active_only=state.active_only)
        console.print(render_page(translation, rows, state, console))
        assert state.key(key, rows, 10)

    assert calls == {"summaries": 0, "valid": 0}
    assert fake.calls


def test_lightweight_refresh_avoids_hashes_until_bounded_integrity_check(service):
    translation, _ = service
    run_ids = []
    for _ in range(3):
        run_id = translation.create_run(runner="test", chapters=[1], force=True)
        translation.execute(run_id)
        run_ids.append(run_id)

    calls = {"status": [], "valid": 0}
    original_status = translation.status
    original_valid = translation.stages.valid_output

    def counting_status(run_id):
        calls["status"].append(run_id)
        return original_status(run_id)

    def counting_valid(row):
        calls["valid"] += 1
        return original_valid(row)

    translation.status = counting_status
    translation.stages.valid_output = counting_valid
    rows = run_rows(translation)
    console = Console(width=120, height=24, record=True, force_terminal=False)
    state = DashboardState()
    integrity_cache = {}

    hydrate_rows(translation, rows, console, state, integrity_cache=integrity_cache)
    assert calls == {"status": [], "valid": 0}
    assert all(row["integrity_stale"] for row in rows)
    assert all(
        row["status"]["chunks"]["completed"] == row["status"]["total_chunks"]
        for row in rows
    )

    hydrate_rows(
        translation,
        rows,
        console,
        state,
        verify_ids={run_ids[0]},
        integrity_cache=integrity_cache,
    )
    assert calls["status"] == [run_ids[0]]
    assert calls["valid"] > 0
    checked = next(row for row in rows if row["id"] == run_ids[0])
    assert checked["integrity_stale"] is False
    assert checked["last_verified_at"]
    assert sum(row["integrity_stale"] for row in rows) == 2
    refreshed_rows = run_rows(translation)
    hydrate_rows(
        translation, refreshed_rows, console, state, integrity_cache=integrity_cache
    )
    checked_again = next(row for row in refreshed_rows if row["id"] == run_ids[0])
    assert checked_again["last_verified_at"] == checked["last_verified_at"]
    assert checked_again["integrity_stale"] is False


def test_selected_detail_reuses_lightweight_status_without_full_verification(service):
    translation, _ = service
    run_id = translation.create_run(runner="test", chapters=[1])
    rows = run_rows(translation)
    state = DashboardState(detail_run_id=run_id)
    console = Console(width=120, height=24, record=True, force_terminal=False)

    def forbidden(_run_id):
        raise AssertionError("selected detail repeated full status verification")

    translation.status = forbidden
    hydrate_rows(translation, rows, console, state)

    assert rows[0]["snapshot"]["run_id"] == run_id
    assert rows[0]["status"]["integrity"]["stale"] is True


def _pipe_events(data: bytes, count: int):
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, data)
        events = []
        pending = b""
        for _ in range(count):
            event, pending = _read_posix(read_fd, 0.02, pending)
            events.append(event)
        return events
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_posix_decoder_preserves_buffered_sequences_and_following_keys():
    assert _pipe_events(b"\x1b[B\x1b[B", 2) == ["down", "down"]
    assert _pipe_events(b"\x1b[Bq", 2) == ["down", "q"]
    assert _pipe_events(b"\x1b[A", 1) == ["up"]
    assert _pipe_events(b"\x1b[6~", 1) == ["next"]
    assert _pipe_events(b"\x1b[5~", 1) == ["previous"]
    assert _pipe_events(b"\x1bq", 2) == ["esc", "q"]
    assert _pipe_events(b"j\x1b[B", 2) == ["j", "down"]


def test_posix_decoder_handles_standalone_escape_and_buffered_bytes():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\x1b")
        assert _read_posix(read_fd, 0.02)[0] == "esc"
        # Bytes carried from an earlier read decode without touching the file descriptor.
        assert _read_posix(read_fd, 0.02, b"\x1b[B")[0] == "down"
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_periodic_refresh_keeps_corrupt_output_observable(service):
    translation, _ = service
    run_id = translation.create_run(runner="test", chapters=[1])
    translation.execute(run_id)

    chunk = translation.repository.chunks(run_id)[0]
    output = translation.run_dir(run_id) / chunk["output_path"]
    output.write_bytes(output.read_bytes() + b"corrupt")

    console = Console(width=120, height=24, record=True, force_terminal=False)
    state = DashboardState()
    rows = filter_rows(run_rows(translation), active_only=state.active_only)
    hydrate_rows(translation, rows, console, state, verify_ids={run_id})

    assert rows[0]["status"]["status"] == "incomplete"
    console.print(render_page(translation, rows, state, console, once=True))
    assert "incomplete" in console.export_text(clear=False)


def test_dashboard_quit_does_not_wait_for_blocked_storage(monkeypatch):
    import threading
    from contextlib import contextmanager
    from io import StringIO
    from types import SimpleNamespace

    import novel_harness.run_dashboard as module

    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    errors = []
    row = {"id": "run", "book": "Book", "runner": "test", "worker_active": False}

    def status(_run_id):
        started.set()
        assert release.wait(timeout=5)

    @contextmanager
    def keys():
        def read(_timeout):
            assert started.wait(timeout=2)
            return "q"
        yield read

    class Live:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args, **kwargs):
            pass

    monkeypatch.setattr(module, "run_rows", lambda _service: [dict(row)])
    monkeypatch.setattr(module, "_keys", keys)
    monkeypatch.setattr(module, "Live", Live)
    monkeypatch.setattr(module, "Console", lambda: Console(width=120, file=StringIO()))
    monkeypatch.setattr(module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(module.sys.stdout, "isatty", lambda: True)

    def browse():
        try:
            module.dashboard(SimpleNamespace(status=status))
        except Exception as error:  # noqa: BLE001 - report failures from the test UI thread
            errors.append(error)
        finally:
            finished.set()

    ui = threading.Thread(target=browse)
    ui.start()
    try:
        assert started.wait(timeout=2)
        quit_before_storage_release = finished.wait(timeout=1)
    finally:
        release.set()
        ui.join(timeout=3)
    assert quit_before_storage_release
    assert errors == []


def test_interactive_detail_render_never_reads_storage(monkeypatch):
    from types import SimpleNamespace

    import novel_harness.run_dashboard as module

    def forbidden(*args, **kwargs):
        raise AssertionError("storage read in key render")

    monkeypatch.setattr(module, "snapshot", forbidden)
    service = SimpleNamespace(status=forbidden)
    rows = [{"id": str(i), "book": "Book", "snapshot": None} for i in range(3)]
    state = DashboardState()
    live = SimpleNamespace(update=lambda *args, **kwargs: None)
    console = Console(width=120)
    for key in ("enter", "right", "left", "right"):
        state.key(key, rows, 10)
        module._render_live(live, service, rows, state, console)


def test_group_runs_orders_book_groups_and_annotates_re_runs():
    rows = [
        {
            "id": "b-old",
            "book": "Beta",
            "novel_id": "n-b",
            "worker_active": False,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "b-new",
            "book": "Beta",
            "novel_id": "n-b",
            "worker_active": False,
            "created_at": "2026-01-03T00:00:00+00:00",
        },
        {
            "id": "a-old",
            "book": "Alpha",
            "novel_id": "n-a",
            "worker_active": False,
            "created_at": "2026-01-02T00:00:00+00:00",
        },
        {
            "id": "z-active",
            "book": "Zeta",
            "novel_id": "n-z",
            "worker_active": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    group_runs(rows)
    assert [row["id"] for row in rows] == ["z-active", "a-old", "b-new", "b-old"]
    assert rows[0]["book_group"] == "n-z"
    assert rows[0]["group_index"] == 0 and rows[0]["older_runs"] == 0
    assert rows[2]["group_index"] == 0 and rows[2]["older_runs"] == 1
    assert rows[3]["group_index"] == 1 and rows[3]["older_runs"] == 1
    assert rows[1]["book_group"] == "n-a"


def test_group_runs_falls_back_to_title_for_legacy_runs():
    rows = [
        {
            "id": "old",
            "book": "Legacy",
            "novel_id": None,
            "worker_active": False,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "new",
            "book": "Legacy",
            "novel_id": None,
            "worker_active": False,
            "created_at": "2026-01-02T00:00:00+00:00",
        },
        {
            "id": "other",
            "book": "Other",
            "novel_id": None,
            "worker_active": False,
            "created_at": "2026-01-03T00:00:00+00:00",
        },
    ]
    group_runs(rows)
    # Groups sort alphabetically; within a group the newest run is first.
    assert [row["id"] for row in rows] == ["new", "old", "other"]
    assert rows[0]["book_group"] == "Legacy"
    assert rows[0]["group_index"] == 0 and rows[0]["older_runs"] == 1
    assert rows[1]["group_index"] == 1


def test_latest_per_book_filter_keeps_current_run_per_group():
    rows = [
        {
            "id": "b-new",
            "book": "Beta",
            "novel_id": "n-b",
            "worker_active": False,
            "group_index": 0,
            "older_runs": 1,
        },
        {
            "id": "b-old",
            "book": "Beta",
            "novel_id": "n-b",
            "worker_active": False,
            "group_index": 1,
            "older_runs": 1,
        },
        {"id": "a", "book": "Alpha", "novel_id": "n-a", "worker_active": False, "group_index": 0, "older_runs": 0},
    ]
    assert [row["id"] for row in filter_rows(rows, active_only=False)] == [
        "b-new",
        "b-old",
        "a",
    ]
    assert [row["id"] for row in filter_rows(rows, active_only=False, latest_only=True)] == [
        "b-new",
        "a",
    ]
    # Unannotated rows from legacy callers survive the latest-per-book filter.
    legacy = [{"id": "legacy", "worker_active": False}]
    assert [row["id"] for row in filter_rows(legacy, active_only=False, latest_only=True)] == [
        "legacy"
    ]


def test_latest_per_book_search_keeps_group_when_older_run_matches():
    rows = [
        {
            "id": "b-new",
            "book": "Beta",
            "novel_id": "n-b",
            "worker_active": False,
            "group_index": 0,
            "older_runs": 1,
        },
        {
            "id": "b-old",
            "book": "Beta",
            "novel_id": "n-b",
            "worker_active": False,
            "group_index": 1,
            "older_runs": 1,
        },
        {"id": "a", "book": "Alpha", "novel_id": "n-a", "worker_active": False, "group_index": 0, "older_runs": 0},
    ]
    # The query matches only the older run; the group head stays visible.
    visible = filter_rows(rows, active_only=False, search="b-old", latest_only=True)
    assert [row["id"] for row in visible] == ["b-new"]
    # Full list mode still shows only the matching row itself.
    assert [row["id"] for row in filter_rows(rows, active_only=False, search="b-old")] == [
        "b-old"
    ]
    # A query matching no run of a group hides the whole group.
    assert filter_rows(rows, active_only=False, search="gamma", latest_only=True) == []


def test_dashboard_g_toggles_latest_per_book_view():
    rows = [{"id": "one"}, {"id": "two"}]
    state = DashboardState()
    assert state.latest_only is False
    state.selected = 1
    state.selected_run_id = "two"
    assert state.key("g", rows, 10)
    assert state.latest_only is True
    assert state.selected == 0 and state.selected_run_id == "one"
    assert state.key("g", rows, 10)
    assert state.latest_only is False


def test_render_page_shows_latest_per_book_mode_and_older_count():
    status = {
        "chunks": {"completed": 1, "pending": 0, "failed": 0, "running": 0, "corrupt": 0},
        "total_chunks": 1,
        "progress": 1.0,
        "status": "completed",
    }
    rows = [
        {
            "id": "a-new",
            "book": "Alpha",
            "novel_id": "n-a",
            "runner": "test",
            "worker_active": False,
            "group_index": 0,
            "older_runs": 1,
            "status": status,
        },
    ]
    state = DashboardState(latest_only=True)
    console = Console(width=120, height=11, record=True, force_terminal=False)
    console.print(render_page(None, rows, state, console, once=True))
    rendered = console.export_text(clear=False)
    assert "latest per book" in rendered
    assert "Alpha (+1)" in rendered


def test_run_rows_group_re_runs_of_the_same_book(project):
    from conftest import FakeRunner

    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    root = project.parent / "original"
    for name in ("book-a", "book-b"):
        book = root / name
        book.mkdir()
        for number in (1, 2):
            (book / f"{number:04d}.txt").write_text(f"第{number}章\n\n{name}本文。" * 20, "utf-8")
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    service.rescan_library()
    cards = {card["title"]: card for card in service.library_novels()["novels"]}
    run_a1 = service.create_run(runner="test", chapters=[1], novel_id=cards["book-a"]["novel_id"])
    run_a2 = service.create_run(runner="test", chapters=[1], novel_id=cards["book-a"]["novel_id"], force=True)
    run_b = service.create_run(runner="test", chapters=[1], novel_id=cards["book-b"]["novel_id"])

    rows = run_rows(service)
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {run_a1, run_a2, run_b}
    assert by_id[run_a1]["novel_id"] == cards["book-a"]["novel_id"]
    assert by_id[run_b]["novel_id"] == cards["book-b"]["novel_id"]
    # Same-book re-runs form one contiguous block, newest first.
    groups = [row["book_group"] for row in rows]
    assert len(set(groups)) == 2
    a_positions = [i for i, group in enumerate(groups) if group == by_id[run_a1]["book_group"]]
    assert a_positions == list(range(min(a_positions), max(a_positions) + 1))
    a_block = [row["id"] for row in rows if row["book_group"] == by_id[run_a1]["book_group"]]
    assert a_block == [run_a2, run_a1]
    assert by_id[run_a2]["group_index"] == 0 and by_id[run_a2]["older_runs"] == 1
    assert by_id[run_a1]["group_index"] == 1 and by_id[run_a1]["older_runs"] == 1
    assert by_id[run_b]["group_index"] == 0 and by_id[run_b]["older_runs"] == 0


def test_background_refresh_is_bounded_and_publishes_new_projections(service):
    import threading

    from novel_harness.run_dashboard import DashboardRefresh

    translation, _ = service
    run_id = translation.create_run(runner="test", chapters=[1])
    translation.execute(run_id)
    refresh = DashboardRefresh(integrity_every=1)
    console, state = Console(width=120), DashboardState()
    entered, release = threading.Event(), threading.Event()
    original = translation.status

    def blocked_status(*args):
        entered.set()
        assert release.wait(timeout=3)
        return original(*args)

    translation.status = blocked_status
    assert refresh.start(translation, console, state)
    try:
        assert entered.wait(timeout=2)
        assert refresh.take() is None
        assert not refresh.start(translation, console, state)
    finally:
        release.set()
    refresh.future.result(timeout=3)
    first = refresh.take()
    assert first[0]["status"]["status"] == "completed"
    row = translation.repository.chunks(run_id)[0]
    (translation.run_dir(run_id) / row["output_path"]).write_text("corrupt", "utf-8")
    assert refresh.start(translation, console, state)
    refresh.future.result(timeout=3)
    second = refresh.take()
    assert second[0]["status"]["chunks"]["corrupt"] == 1
    assert first[0]["status"]["status"] == "completed"
