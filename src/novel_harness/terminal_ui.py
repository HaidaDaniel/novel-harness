"""Explicit terminal controls over the persisted worker lifecycle."""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future
from threading import Lock, Thread

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from .launch import launch_run
from .run_dashboard import (
    DashboardRefresh,
    DashboardState,
    LibraryRefresh,
    LibraryState,
    _keys,
    _next_refresh_delay,
    _page_size,
    filter_library_rows,
    filter_rows,
    render_detail,
    render_library_page,
    render_page,
)

ACTION_KEYS = {"space", "s", "u", "t"}
ACTION_LABELS = {
    "space": "pause/start",
    "s": "start",
    "u": "resume",
    "t": "retry",
    "launch": "launch",
    "continue": "continue",
}


class ActionExecutor:
    """Run at most one control action in a daemon thread without joining it."""

    def __init__(self):
        self.future: Future[dict] | None = None
        self.target_run_id: str | None = None
        self.action: str | None = None
        self._lock = Lock()

    def dispatch(self, service, config, action: str, run_id: str, *, active: bool) -> bool:
        with self._lock:
            if self.future is not None and not self.future.done():
                return False
            future: Future[dict] = Future()
            self.future = future
            self.target_run_id = run_id
            self.action = action
        if action == "space" and active:
            operation = lambda: service.pause(run_id)
        else:
            operation = lambda: launch_run(service, config, run_id)

        def run() -> None:
            try:
                future.set_result(operation())
            except Exception as error:  # noqa: BLE001 - report action failure in the TUI
                future.set_exception(error)

        Thread(target=run, name=f"tui-{action}", daemon=True).start()
        return True

    def dispatch_book(self, service, config, row: dict) -> bool:
        """Capture a stable novel/run identity and use the existing launch path."""
        with self._lock:
            if self.future is not None and not self.future.done():
                return False
            future: Future[dict] = Future()
            self.future = future
            self.target_run_id = row["novel_id"]
            self.action = "continue" if row.get("continue_run_id") else "launch"
        novel_id = row["novel_id"]

        def operation():
            return service.launch_remaining_novel(novel_id, config)

        def run() -> None:
            try:
                future.set_result(operation())
            except Exception as error:  # noqa: BLE001 - report action failure in the TUI
                future.set_exception(error)

        Thread(target=run, name="tui-library-launch", daemon=True).start()
        return True

    def take(self) -> tuple[str | None, str | None, dict | Exception | None]:
        with self._lock:
            future = self.future
            if future is None or not future.done():
                return None, None, None
            action, target = self.action, self.target_run_id
            self.future = None
            self.action = None
            self.target_run_id = None
        try:
            return action, target, future.result()
        except Exception as error:  # noqa: BLE001 - convert to visible action state
            return action, target, error

    def close(self) -> None:
        # The daemon action is deliberately not joined while storage is blocked.
        with self._lock:
            future = self.future
            self.future = None
        if future is not None:
            future.cancel()


def _action_target(rows: list[dict], state: DashboardState) -> dict | None:
    state.sync(rows)
    if state.selected_run_id is None:
        return None
    return next((row for row in rows if row["id"] == state.selected_run_id), None)


def _action_message(action: str | None, target: str | None, result) -> str:
    if target is None:
        return "No run selected."
    if isinstance(result, Exception):
        return f"{ACTION_LABELS.get(action or '', 'action').capitalize()} failed for {target}: {result}"
    if action in {"launch", "continue"} and result.get("run_id"):
        return (
            f"{ACTION_LABELS[action].capitalize()} accepted for {target}; "
            f"worker target is {result['run_id']}."
        )
    if action == "space" and result.get("pause_requested"):
        return f"Pause requested for {target}; it will take effect at the next chunk boundary."
    if result.get("started"):
        return f"Worker start accepted for {target}; translation continues in the background."
    if result.get("already_running"):
        return f"Worker for {target} is already running."
    if result.get("completed"):
        return f"Run {target} is already complete."
    if result.get("error"):
        return f"Worker start failed for {target}: {result['error']}"
    return f"{ACTION_LABELS.get(action or '', 'Action').capitalize()} completed for {target}."


def _render_tui(live, service, rows, state, console, message: str):
    if state.detail_run_id is None:
        body = render_page(service, rows, state, console, cached_only=True)
    else:
        selected = next((row for row in rows if row["id"] == state.detail_run_id), None)
        body = (
            render_detail(service, selected, cached_only=True)
            if selected is not None
            else render_page(service, rows, state, console, cached_only=True)
        )
    controls = Text(
        "Tab Library · Space pause/start · s start · u resume · t retry · "
        "g latest/book · r refresh · q quit"
        + (f" · {message}" if message else ""),
        style="yellow",
    )
    live.update(Group(body, controls), refresh=True)


def _render_library_tui(live, rows, state, console, message: str):
    body = render_library_page(rows, state, console)
    controls = Text(
        "Tab runs · ↑↓ select · f filter · / search · Space launch/continue · "
        "r rescan · q quit"
        + (f" · {message}" if message else ""),
        style="yellow",
    )
    live.update(Group(body, controls), refresh=True)


def tui(service, config):
    """Run the explicit interactive control mode without inline inference."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError("Interactive TUI needs a terminal")
    console = Console()
    state = DashboardState()
    library_state = LibraryState()
    refresh = DashboardRefresh()
    library_refresh = LibraryRefresh()
    actions = ActionExecutor()
    all_rows, rows = [], []
    all_library_rows, library_view_rows = [], []
    view = "runs"
    message = ""
    last_size = (console.size.width, console.size.height)
    with _keys() as read_key, Live(
        Text("Loading saved runs… · q quit"),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        refresh.start(service, console, state)
        library_refresh.start(service)
        refresh_at = time.monotonic() + 5.0
        library_refresh_at = time.monotonic() + 5.0
        try:
            while True:
                try:
                    key = read_key(0.05)
                except KeyboardInterrupt:
                    break
                search_active = (
                    state.search_active if view == "runs" else library_state.search_active
                )
                if key == "q" and not search_active:
                    break
                if key == "\t":
                    view = "library" if view == "runs" else "runs"
                    state.search_active = False
                    library_state.search_active = False
                    message = ""
                elif view == "library" and key == "space":
                    selected = (
                        library_view_rows[library_state.selected]
                        if library_view_rows and library_state.selected < len(library_view_rows)
                        else None
                    )
                    if selected is None or selected.get("status") == "unavailable":
                        message = "Selected source is unavailable; no launch dispatched."
                    elif (
                        not selected.get("continue_run_id")
                        and not selected.get("preset_available", True)
                    ):
                        message = (
                            f"Preset {selected['effective_preset']} is unavailable; "
                            "no launch dispatched."
                        )
                    elif not actions.dispatch_book(service, config, selected):
                        message = "An action is already in progress."
                    else:
                        message = f"Launch requested for {selected['novel_id']}."
                elif view == "runs" and key in ACTION_KEYS:
                    target = _action_target(rows, state)
                    if target is None:
                        message = _action_message(key, None, None)
                    elif not actions.dispatch(
                        service,
                        config,
                        key,
                        target["id"],
                        active=target.get("worker_active", False),
                    ):
                        message = "An action is already in progress."
                    else:
                        message = f"{ACTION_LABELS[key].capitalize()} requested for {target['id']}."
                elif key is not None:
                    if view == "runs":
                        state.key(key, rows, _page_size(console))
                    else:
                        library_state.key(key, library_view_rows, _page_size(console))
                action, target, result = actions.take()
                if result is not None:
                    message = _action_message(action, target, result)
                    if action in {"launch", "continue"}:
                        library_state.refresh_requested = True
                    else:
                        state.refresh_requested = True
                published = refresh.take()
                if published is not None:
                    all_rows = published
                    refresh_at = time.monotonic() + _next_refresh_delay(all_rows, 5.0)
                library_published = library_refresh.take()
                if library_published is not None:
                    all_library_rows = library_published
                    library_refresh_at = time.monotonic() + 15.0
                rows = filter_rows(
                    all_rows,
                    active_only=state.active_only,
                    search=state.search_query,
                    latest_only=state.latest_only,
                )
                library_view_rows = filter_library_rows(
                    all_library_rows,
                    filter_name=library_state.filter_name,
                    search=library_state.search_query,
                )
                state.sync(rows)
                library_state.sync(library_view_rows)
                now = time.monotonic()
                if (state.refresh_requested or now >= refresh_at) and refresh.start(
                    service, console, state
                ):
                    state.refresh_requested = False
                if (library_state.refresh_requested or now >= library_refresh_at) and library_refresh.start(
                    service, rescan=library_state.refresh_requested
                ):
                    library_state.refresh_requested = False
                size = (console.size.width, console.size.height)
                resized = size != last_size
                last_size = size
                if (
                    key is not None
                    or published is not None
                    or library_published is not None
                    or result is not None
                    or resized
                ):
                    if view == "runs":
                        _render_tui(live, service, rows, state, console, message)
                    else:
                        _render_library_tui(
                            live, library_view_rows, library_state, console, message
                        )
        finally:
            actions.close()
            refresh.close()
            library_refresh.close()
