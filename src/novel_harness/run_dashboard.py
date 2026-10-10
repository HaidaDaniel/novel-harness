"""Read-only terminal dashboard for persisted translation runs."""

from __future__ import annotations

import os
import select
import sys
import time
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .progress import _format_duration, render_rich, snapshot, worker_state

ACTIVE_STATES = {"starting", "running", "waiting_retry"}
MAX_INTEGRITY_CACHE = 256
MAX_LIBRARY_ROWS = 2000
STATE_STYLES = {
    "completed": "green",
    "starting": "cyan bold",
    "running": "cyan bold",
    "waiting_retry": "yellow",
    "paused": "yellow",
    "failed": "red bold",
    "interrupted": "red",
    "incomplete": "red",
}


def _book_name(manifest: dict) -> str:
    title = (manifest.get("novel") or {}).get("title")
    if title and title != "Japanese web novel":
        return title
    return Path(manifest.get("source_dir") or "").name or title or "-"


def _book_group(row: dict) -> str:
    """Group key for runs of one source book; legacy runs group by title."""
    return row.get("novel_id") or row.get("book") or "-"


def group_runs(runs: list[dict]) -> None:
    """Sort runs into per-book blocks and annotate each row's position.

    Groups with an active worker come first, then alphabetically; within a
    group the active run is first, then newest first. Rows keep ``group_index``
    (zero for the group's current run) and ``older_runs`` (the other runs of
    the same book) for the latest-per-book view.
    """
    runs.sort(
        key=lambda run: (run["worker_active"], run["created_at"], run["id"]),
        reverse=True,
    )
    group_active: dict[str, bool] = {}
    for run in runs:
        group = _book_group(run)
        run["book_group"] = group
        group_active[group] = group_active.get(group, False) or run["worker_active"]
    runs.sort(
        key=lambda run: (not group_active[run["book_group"]], run["book_group"].casefold())
    )
    seen: dict[str, int] = {}
    for run in runs:
        seen[run["book_group"]] = seen.get(run["book_group"], 0) + 1
        run["group_index"] = seen[run["book_group"]] - 1
    for run in runs:
        run["older_runs"] = seen[run["book_group"]] - 1


def run_rows(service, *, active_only: bool = False) -> list[dict]:
    """List narrow saved-run metadata; never decode manifests or inspect chunks."""
    if hasattr(service, "lightweight_statuses"):
        projections = service.lightweight_statuses()
        runs = []
        for run in projections.values():
            title = run["title"]
            source_dir = run["source_dir"] or ""
            if not title or title == "Japanese web novel":
                title = Path(source_dir).name or title or "-"
            runs.append(
                {
                    "id": run["run_id"],
                    "book": title,
                    "novel_id": run.get("novel_id"),
                    "runner": (run["runners"] or [{}])[0].get("name") or "-",
                    "status": run["status"],
                    "counts": run["chunks"],
                    "total_chunks": run["total_chunks"],
                    "progress": run["progress"],
                    "last_verified_at": None,
                    "integrity_stale": True,
                    "projection": run,
                    "created_at": run["created_at"],
                    "worker_active": run["active"],
                }
            )
    else:
        runs = []
        for run in service.repository.list_run_summaries():
            worker = worker_state(service.run_dir(run["id"]))
            runs.append(
                {
                    "id": run["id"],
                    "book": run["title"] or Path(run["source_dir"] or "").name or "-",
                    "novel_id": run.get("novel_id"),
                    "runner": run["runner"] or "-",
                    "status": run["status"],
                    "counts": run["counts"],
                    "total_chunks": run["total_chunks"],
                    "progress": (
                        run["counts"]["completed"] / run["total_chunks"]
                        if run["total_chunks"]
                        else 0
                    ),
                    "last_verified_at": None,
                    "integrity_stale": True,
                    "created_at": run["created_at"],
                    "worker_active": bool(
                        worker
                        and worker.get("alive") is True
                        and worker.get("status") in {"starting", "running", "waiting_retry"}
                    ),
                }
            )
    if active_only:
        runs = [run for run in runs if run["worker_active"]]
    group_runs(runs)
    for run in runs:
        run.pop("created_at", None)
    return runs


def _matches_search(row: dict, query: str, *fields: str) -> bool:
    if not query:
        return True
    needle = query.casefold()
    return any(needle in str(row.get(field) or "").casefold() for field in fields)


def filter_rows(
    rows: list[dict], *, active_only: bool, search: str = "", latest_only: bool = False
) -> list[dict]:
    """Filter cached rows without touching the repository.

    In latest-per-book mode a book stays visible when any of its runs matches
    the search, so querying an older run's ID does not hide the whole group.
    """
    group_matches = None
    if latest_only and search:
        group_matches = {
            _book_group(row) for row in rows if _matches_search(row, search, "book", "id")
        }
    filtered = []
    for row in rows:
        if active_only and not row["worker_active"]:
            continue
        if latest_only and row.get("group_index", 0) != 0:
            continue
        if (
            search
            and not _matches_search(row, search, "book", "id")
            and not (latest_only and _book_group(row) in group_matches)
        ):
            continue
        filtered.append(row)
    return filtered


def library_rows(service) -> list[dict]:
    """Build a bounded source projection without rescanning the filesystem."""
    cards = service.library_novels()
    remaining = {
        item["novel_id"]: item for item in service.library_remaining()["novels"]
    }
    try:
        default_preset = service.default_preset()
        presets = {item["id"]: item for item in service.presets()}
    except (AttributeError, OSError, ValueError, KeyError, TypeError):
        default_preset, presets = None, {}
    selected_preset = presets.get(default_preset or "")
    preset_id = default_preset or "raw/direct"
    profile_id = (selected_preset or {}).get("runner") or "configured runner"
    profile_model = (selected_preset or {}).get("model")
    projected = []
    ordered_cards = sorted(
        cards.get("novels", []),
        key=lambda card: ((card.get("title") or "-").casefold(), card.get("novel_id") or ""),
    )
    for card in ordered_cards[:MAX_LIBRARY_ROWS]:
        novel_id = card.get("novel_id")
        coverage = remaining.get(novel_id)
        ready = card.get("status") == "ready"
        if not ready:
            translated = None
            untranslated = None
            state = "unavailable"
        elif coverage is None:
            translated = card.get("chapter_count", 0)
            untranslated = 0
            state = "complete"
        else:
            translated = coverage["translated"]
            untranslated = coverage["remaining"]
            state = "remaining" if untranslated else "complete"
        projected.append(
            {
                "novel_id": novel_id,
                "title": card.get("title") or "-",
                "source_path": card.get("source_path") or "-",
                "status": state,
                "source_status": card.get("status"),
                "reason": card.get("error"),
                "chapter_count": card.get("chapter_count", 0),
                "translated": translated,
                "remaining": untranslated,
                "run_id": (coverage or {}).get("run_id"),
                "continue_run_id": card.get("continue_run_id"),
                "active_run": card.get("active_run"),
                "effective_preset": preset_id,
                "effective_profile": profile_id,
                "effective_model": profile_model,
                "preset_available": (
                    True
                    if selected_preset is None
                    else selected_preset.get("available", False)
                ),
                "freshness": "cached until explicit Library refresh",
            }
        )
    projected.sort(key=lambda row: (row["title"].casefold(), row["novel_id"] or ""))
    return projected


def hydrate_rows(
    service,
    rows: list[dict],
    console: Console,
    state: DashboardState,
    *,
    wide: bool | None = None,
    verify_ids: set[str] | None = None,
    integrity_cache: dict[str, str] | None = None,
) -> None:
    """Attach lightweight status and selected snapshots to cached rows.

    The normal path uses one aggregate projection query and reads no output
    bodies. ``verify_ids`` is the explicit bounded integrity budget; those runs
    use the full status contract and record when corruption was last checked.
    """
    wide = console.size.width >= 135 if wide is None else wide
    ids = [row["id"] for row in rows]
    if all("projection" in row for row in rows):
        statuses = {row["id"]: row["projection"] for row in rows}
    elif hasattr(service, "lightweight_statuses"):
        statuses = service.lightweight_statuses(ids)
    else:
        statuses = {run_id: service.status(run_id) for run_id in ids}
    verify_ids = verify_ids or set()
    for row in rows:
        try:
            status = statuses[row["id"]]
            if row["id"] in verify_ids:
                status = service.status(row["id"])
                verified_at = datetime.now(UTC).isoformat()
                row["last_verified_at"] = verified_at
                row["integrity_stale"] = False
                if integrity_cache is not None:
                    integrity_cache.pop(row["id"], None)
                    integrity_cache[row["id"]] = verified_at
                    while len(integrity_cache) > MAX_INTEGRITY_CACHE:
                        oldest = next(iter(integrity_cache))
                        integrity_cache.pop(oldest)
            else:
                verified_at = row.get("last_verified_at") or (
                    integrity_cache or {}
                ).get(row["id"])
                row["last_verified_at"] = verified_at
                row["integrity_stale"] = verified_at is None
            status["integrity"] = {
                "last_verified_at": verified_at,
                "stale": row["integrity_stale"],
            }
        except (OSError, ValueError, KeyError, TypeError):
            status = None
        row["status"] = status
        needs_snapshot = (
            status is not None
            and (
                state.detail_run_id == row["id"]
                or (
                    wide
                    and status["status"] in ACTIVE_STATES | {"paused", "interrupted"}
                )
            )
        )
        if needs_snapshot:
            try:
                row["snapshot"] = snapshot(service, row["id"], status=status)
            except (OSError, ValueError, KeyError):
                row["snapshot"] = None


@dataclass
class DashboardState:
    selected: int = 0
    selected_run_id: str | None = None
    detail_run_id: str | None = None
    active_only: bool = False
    latest_only: bool = False
    refresh_requested: bool = False
    search_query: str = ""
    search_active: bool = False

    def sync(self, rows: list[dict]) -> None:
        """Keep the selected identity stable while rows are reordered or filtered."""
        if not rows:
            self.selected = 0
            self.selected_run_id = None
            self.detail_run_id = None
            return
        if self.selected_run_id is not None:
            current = next(
                (index for index, row in enumerate(rows) if row["id"] == self.selected_run_id),
                None,
            )
            if current is not None:
                self.selected = current
                return
        self.selected = min(max(0, self.selected), len(rows) - 1)
        self.selected_run_id = rows[self.selected]["id"]

    def _remember_selection(self, rows: list[dict]) -> None:
        if rows:
            self.selected = min(max(0, self.selected), len(rows) - 1)
            self.selected_run_id = rows[self.selected]["id"]

    def key(self, key: str, rows: list[dict], page_size: int) -> bool:
        """Update navigation state; return false only when the user exits."""
        if key == "/":
            self.search_active = True
            return True
        if self.search_active:
            if key in {"enter", "esc"}:
                self.search_active = False
            elif key == "backspace":
                self.search_query = self.search_query[:-1]
            elif len(key) == 1 and key.isprintable():
                self.search_query += key
            self.selected = 0
            self.selected_run_id = None
            return True
        if key in {"q", "ctrl_c"}:
            return False
        self.sync(rows)
        if self.detail_run_id is not None:
            current = next((i for i, row in enumerate(rows) if row["id"] == self.detail_run_id), None)
            if current is None:
                self.detail_run_id = None
                self.selected = min(self.selected, max(0, len(rows) - 1))
                self._remember_selection(rows)
                return True
            self.selected = current
            if key == "r":
                self.refresh_requested = True
            elif key in {"b", "esc"}:
                self.detail_run_id = None
            elif key in {"right", "n", "left", "p"}:
                step = 1 if key in {"right", "n"} else -1
                self.selected = min(len(rows) - 1, max(0, current + step))
                self.detail_run_id = rows[self.selected]["id"]
            self._remember_selection(rows)
            return True
        if key == "r":
            self.refresh_requested = True
        elif key == "a":
            self.active_only = not self.active_only
            self.selected = 0
            self.selected_run_id = None
        elif key == "g":
            self.latest_only = not self.latest_only
            self.selected = 0
            self.selected_run_id = None
        elif key in {"down", "j"}:
            self.selected = min(max(0, len(rows) - 1), self.selected + 1)
        elif key in {"up", "k"}:
            self.selected = max(0, self.selected - 1)
        elif key in {"next", "right", "n"}:
            self.selected = min(max(0, len(rows) - 1), self.selected + page_size)
        elif key in {"previous", "left", "p"}:
            self.selected = max(0, self.selected - page_size)
        elif key == "enter" and rows:
            self.selected = min(self.selected, len(rows) - 1)
            self.detail_run_id = rows[self.selected]["id"]
        self._remember_selection(rows)
        return True


LIBRARY_FILTERS = ("all", "remaining", "active", "complete", "unavailable")


@dataclass
class LibraryState:
    selected: int = 0
    selected_novel_id: str | None = None
    filter_name: str = "all"
    search_query: str = ""
    search_active: bool = False
    refresh_requested: bool = False

    def sync(self, rows: list[dict]) -> None:
        if not rows:
            self.selected = 0
            self.selected_novel_id = None
            return
        if self.selected_novel_id is not None:
            current = next(
                (index for index, row in enumerate(rows)
                 if row["novel_id"] == self.selected_novel_id),
                None,
            )
            if current is not None:
                self.selected = current
                return
        self.selected = min(max(0, self.selected), len(rows) - 1)
        self.selected_novel_id = rows[self.selected]["novel_id"]

    def key(self, key: str, rows: list[dict], page_size: int) -> bool:
        if key == "/":
            self.search_active = True
            return True
        if self.search_active:
            if key in {"enter", "esc"}:
                self.search_active = False
            elif key == "backspace":
                self.search_query = self.search_query[:-1]
            elif len(key) == 1 and key.isprintable():
                self.search_query += key
            self.selected = 0
            self.selected_novel_id = None
            return True
        if key in {"q", "ctrl_c"}:
            return False
        self.sync(rows)
        if key == "r":
            self.refresh_requested = True
        elif key == "f":
            self.filter_name = LIBRARY_FILTERS[
                (LIBRARY_FILTERS.index(self.filter_name) + 1) % len(LIBRARY_FILTERS)
            ]
            self.selected = 0
            self.selected_novel_id = None
        elif key in {"down", "j"}:
            self.selected = min(len(rows) - 1, self.selected + 1) if rows else 0
        elif key in {"up", "k"}:
            self.selected = max(0, self.selected - 1)
        elif key in {"next", "right", "n"}:
            self.selected = min(len(rows) - 1, self.selected + page_size) if rows else 0
        elif key in {"previous", "left", "p"}:
            self.selected = max(0, self.selected - page_size)
        self.sync(rows)
        return True


def filter_library_rows(
    rows: list[dict], *, filter_name: str = "all", search: str = ""
) -> list[dict]:
    """Filter the cached Library projection without storage or filesystem work."""
    filtered = [
        row
        for row in rows
        if filter_name == "all"
        or (filter_name == "active" and row.get("active_run") is not None)
        or row.get("status") == filter_name
    ]
    return [
        row
        for row in filtered
        if _matches_search(
            row, search, "title", "novel_id", "source_path", "run_id", "continue_run_id"
        )
    ]


class DashboardRefresh:
    """One read-only collector; completed projections are adopted by the UI thread.

    A daemon thread lets quitting restore the terminal without waiting for slow
    storage. No thread touches Rich Live or the displayed rows after publication.
    """

    def __init__(self, *, integrity_every: int = 5):
        self.future: Future[list[dict]] | None = None
        self.detail_run_id: str | None = None
        self.refresh_count = 0
        self.integrity_cursor = 0
        self.integrity_every = max(1, integrity_every)
        self.integrity_cache: OrderedDict[str, str] = OrderedDict()

    def start(self, service, console: Console, state: DashboardState) -> bool:
        if self.future is not None:
            return False
        frozen_state = replace(state)
        wide = console.size.width >= 135
        detail_changed = frozen_state.detail_run_id != self.detail_run_id
        future = Future()
        self.future = future
        self.detail_run_id = frozen_state.detail_run_id

        def collect():
            if not future.set_running_or_notify_cancel():
                return
            try:
                rows = run_rows(service)
                verify_ids = (
                    {frozen_state.detail_run_id}
                    if detail_changed and frozen_state.detail_run_id is not None
                    else set()
                )
                if rows and self.refresh_count % self.integrity_every == 0:
                    verify_ids.add(rows[self.integrity_cursor % len(rows)]["id"])
                    self.integrity_cursor += 1
                hydrate_rows(
                    service,
                    rows,
                    console,
                    frozen_state,
                    wide=wide,
                    verify_ids=verify_ids,
                    integrity_cache=self.integrity_cache,
                )
                self.refresh_count += 1
                future.set_result(rows)
            except Exception as error:  # noqa: BLE001 - propagate storage failures to the UI
                future.set_exception(error)

        Thread(target=collect, name="dashboard-refresh", daemon=True).start()
        return True

    def take(self) -> list[dict] | None:
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return future.result()

    def close(self) -> None:
        if self.future is not None:
            self.future.cancel()


class LibraryRefresh:
    """Collect the source projection off the input/render thread."""

    def __init__(self):
        self.future: Future[list[dict]] | None = None

    def start(self, service, *, rescan: bool = False) -> bool:
        if self.future is not None:
            return False
        future: Future[list[dict]] = Future()
        self.future = future

        def collect():
            if not future.set_running_or_notify_cancel():
                return
            try:
                if rescan:
                    service.rescan_library()
                future.set_result(library_rows(service))
            except Exception as error:  # noqa: BLE001 - publish storage failures to UI
                future.set_exception(error)

        Thread(target=collect, name="library-refresh", daemon=True).start()
        return True

    def take(self) -> list[dict] | None:
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return future.result()

    def close(self) -> None:
        if self.future is not None:
            self.future.cancel()


def _page_size(console: Console) -> int:
    return max(3, console.size.height - 9)


def _short_id(run_id: str, wide: bool) -> str:
    return run_id if wide else f"{run_id[:13]}…{run_id[-4:]}"


def render_library_page(
    rows: list[dict], state: LibraryState, console: Console, *, once: bool = False
):
    """Render cached source cards, including coverage and launch provenance."""
    state.sync(rows)
    page_size = _page_size(console)
    state.selected = min(max(0, state.selected), max(0, len(rows) - 1))
    page = state.selected // page_size
    visible = rows[page * page_size : (page + 1) * page_size]
    pages = max(1, (len(rows) + page_size - 1) // page_size)
    table = Table(expand=True, show_header=True, header_style="bold cyan")
    table.add_column("", width=1, no_wrap=True)
    table.add_column("Book", max_width=28, no_wrap=True, overflow="ellipsis")
    table.add_column("State", width=12, no_wrap=True)
    table.add_column("Coverage", width=15, no_wrap=True)
    table.add_column("Run", width=22, no_wrap=True)
    for index, row in enumerate(visible, start=page * page_size):
        coverage = (
            "unavailable"
            if row["status"] == "unavailable"
            else f"{row['translated']}/{row['chapter_count']}"
        )
        run_id = row.get("continue_run_id") or "source-only"
        table.add_row(
            "▶" if index == state.selected else " ",
            row["title"],
            row["status"],
            coverage,
            _short_id(run_id, console.size.width >= 135) if run_id != "source-only" else run_id,
            style="on dark_blue" if index == state.selected else None,
        )
    if not rows:
        table.add_row("", "No source novels", "-", "-", "-")
    selected = rows[state.selected] if rows else None
    details = "No book selected."
    if selected is not None:
        remaining = (
            "unavailable"
            if selected["status"] == "unavailable"
            else f"{selected['translated']} translated, {selected['remaining']} remaining"
        )
        action = (
            "continue saved run"
            if selected.get("continue_run_id")
            else "launch source-only book"
        )
        reason = f" · reason: {selected['reason']}" if selected.get("reason") else ""
        details = (
            f"Selected: {selected['title']} · {remaining} · {action}\n"
            f"Projection: {selected['freshness']}\n"
            f"Settings: preset={selected['effective_preset']} "
            f"profile={selected['effective_profile']}"
            + (f" model={selected['effective_model']}" if selected.get("effective_model") else "")
            + reason
        )
    heading = (
        f"Library · {state.filter_name} · {len(rows)} book(s) · page {page + 1}/{pages}"
        + (f" · search={state.search_query!r}" if state.search_query else "")
    )
    help_text = (
        "↑↓ select  f filter  / search  Tab runs  Space launch/continue  r rescan  q quit"
        if not once
        else "Open the interactive view: novel tui --config <config>"
    )
    return Group(
        Panel(table, title=heading, subtitle=help_text, border_style="green"),
        Text(details, style="dim"),
    )


def render_page(
    service, rows: list[dict], state: DashboardState, console: Console, *, once=False,
    cached_only=False,
):
    """Render a page; interactive callers exclusively use published projections."""
    state.sync(rows)
    page_size = _page_size(console)
    state.selected = min(max(0, state.selected), max(0, len(rows) - 1))
    page = state.selected // page_size
    visible = rows[page * page_size : (page + 1) * page_size]
    pages = max(1, (len(rows) + page_size - 1) // page_size)
    wide = console.size.width >= 135
    table = Table(expand=True, show_header=True, header_style="bold cyan")
    table.add_column("", width=1, no_wrap=True)
    table.add_column("Book", max_width=24 if wide else 12, no_wrap=True, overflow="ellipsis")
    table.add_column("Run", width=32 if wide else 19, no_wrap=True)
    table.add_column("State", width=11, no_wrap=True)
    if wide:
        table.add_column("Runner", max_width=17, overflow="ellipsis")
    table.add_column("Chunks", width=17 if wide else 16, no_wrap=True)
    if wide:
        table.add_column("Chapter", width=9, no_wrap=True)
        table.add_column("ETA", width=11, no_wrap=True)
    for index, row in enumerate(visible, start=page * page_size):
        try:
            status = row.get("status") if cached_only else (
                row["status"] if "status" in row else service.status(row["id"])
            )
            counts = status["chunks"]
            completed, total = counts["completed"], status["total_chunks"]
            chunk_label = f"{completed}/{total} ({status['progress']:.0%})"
            state_label = status["status"]
            chapter, eta = "-", "-"
            if wide and state_label in ACTIVE_STATES | {"paused", "interrupted"}:
                detail = row.get("snapshot") if "snapshot" in row else None
                if detail is None and not cached_only:
                    detail = snapshot(service, row["id"])
                position = detail["chapters"]["current_position"] if detail else None
                if position is not None:
                    chapter = f"{position}/{detail['chapters']['total']}"
                if detail and detail["eta_seconds"] is not None:
                    eta = f"~{_format_duration(detail['eta_seconds'])}"
        except (OSError, ValueError, KeyError, TypeError):
            state_label, chunk_label, chapter, eta = "unavailable", "-", "-", "-"
        book_label = row["book"]
        if state.latest_only and row.get("older_runs"):
            book_label = f"{row['book']} (+{row['older_runs']})"
        cells = [
            "▶" if index == state.selected else " ",
            book_label,
            _short_id(row["id"], wide),
            Text(state_label, style=STATE_STYLES.get(state_label, "dim")),
        ]
        if wide:
            cells.append(row["runner"])
        cells.append(chunk_label)
        if wide:
            cells.extend((chapter, eta))
        if index == state.selected:
            row_style = "on dark_blue"
        elif row.get("group_index", 0) > 0:
            row_style = "dim"
        else:
            row_style = None
        table.add_row(*cells, style=row_style)
    if not rows:
        table.add_row("", "No saved runs", "-", "-", *("-",) * (4 if wide else 1))
    if state.active_only and state.latest_only:
        mode = "running, latest per book"
    elif state.active_only:
        mode = "running only"
    elif state.latest_only:
        mode = "latest per book"
    else:
        mode = "all saved runs"
    heading = (
        f"Translations · {mode} · {len(rows)} run(s) · page {page + 1}/{pages}"
        + (f" · search={state.search_query!r}" if state.search_query else "")
    )
    help_text = (
        "↑↓ select  PgUp/PgDn page  Enter details  a active/all  g latest/book  / search  q quit"
        if not once
        else "Open the interactive view: novel dashboard --config <config>"
    )
    selected_id = rows[state.selected]["id"] if rows else "-"
    return Group(
        Panel(table, title=heading, subtitle=help_text, border_style="cyan"),
        Text(f"Selected run: {selected_id}", style="dim"),
    )


def render_detail(service, row: dict, *, cached_only=False):
    detail = row.get("snapshot") if "snapshot" in row else None
    if detail is None:
        if cached_only:
            return Text(f"{row['book']} · Loading details… · b back · q quit", style="cyan")
        detail = snapshot(service, row["id"])
    return Group(
        Text(f"{row['book']} · b back · ←/→ previous/next · q quit", style="bold cyan"),
        render_rich(detail),
    )


_ESCAPE_SEQUENCES = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    b"\x1b[5~": "previous",
    b"\x1b[6~": "next",
}


def _plain_key(value: bytes) -> str:
    if value in {b"\r", b"\n"}:
        return "enter"
    if value in {b"\x08", b"\x7f"}:
        return "backspace"
    if value == b" ":
        return "space"
    if value == b"\x03":
        return "ctrl_c"
    return value.decode("ascii", "ignore").lower()


def _read_posix(fd: int, timeout: float, pending: bytes = b"") -> tuple[str | None, bytes]:
    """Decode exactly one key event and return any trailing bytes unread.

    ``pending`` holds bytes buffered by a previous call. A known escape sequence
    is matched byte by byte with a bounded wait, so a second key that arrived in
    the same read is preserved for the next call instead of being swallowed.
    """
    buffer = pending
    if not buffer:
        if not select.select([fd], [], [], timeout)[0]:
            return None, b""
        buffer = os.read(fd, 1)
    if buffer[:1] != b"\x1b":
        return _plain_key(buffer[:1]), buffer[1:]
    while True:
        if buffer in _ESCAPE_SEQUENCES:
            return _ESCAPE_SEQUENCES[buffer], buffer[len(buffer):]
        if not any(sequence.startswith(buffer) for sequence in _ESCAPE_SEQUENCES):
            # Unsupported sequence: report Escape, keep the remaining bytes.
            return "esc", buffer[1:]
        if not select.select([fd], [], [], 0.015)[0]:
            # Standalone or truncated Escape and the bounded wait expired.
            return "esc", buffer[1:]
        buffer += os.read(fd, 1)


def _read_windows(timeout: float) -> str | None:
    import msvcrt

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            value = msvcrt.getwch()
            if value in {"\x00", "\xe0"}:
                return {
                    "H": "up",
                    "P": "down",
                    "K": "left",
                    "M": "right",
                    "I": "previous",
                    "Q": "next",
                }.get(msvcrt.getwch())
            if value in {"\r", "\n"}:
                return "enter"
            if value in {"\b", "\x7f"}:
                return "backspace"
            if value == " ":
                return "space"
            return value.lower()
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return None


@contextmanager
def _keys():
    if os.name == "nt":
        yield _read_windows
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    pending = b""

    def read_key(timeout: float) -> str | None:
        nonlocal pending
        event, pending = _read_posix(fd, timeout, pending)
        return event

    try:
        tty.setcbreak(fd)
        yield read_key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def _next_refresh_delay(rows: list[dict], interval: float) -> float:
    """Poll active workers normally and idle historical views less often."""
    return interval if any(row.get("worker_active") for row in rows) else max(interval, 15.0)


def dashboard(service, *, interval: float = 5.0, once: bool = False, page: int = 1):
    """Browse runs without touching the translation worker or model endpoint."""
    if interval < 1:
        raise ValueError("interval must be at least 1 second")
    if page < 1:
        raise ValueError("page must be at least 1")
    console = Console()
    state = DashboardState(selected=(page - 1) * _page_size(console))
    if once:
        rows = run_rows(service)
        hydrate_rows(service, rows, console, state, verify_ids={row["id"] for row in rows})
        console.print(render_page(service, rows, state, console, once=True))
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError("Interactive dashboard needs a terminal; use --once for a table")
    refresh = DashboardRefresh()
    all_rows, rows = [], []
    last_size = (console.size.width, console.size.height)
    with _keys() as read_key, Live(
        Text("Loading saved runs… · q quit"),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        refresh.start(service, console, state)
        refresh_at = time.monotonic() + interval
        try:
            while True:
                try:
                    key = read_key(0.05)
                except KeyboardInterrupt:
                    break
                if key and not state.key(key, rows, _page_size(console)):
                    break
                published = refresh.take()
                if published is not None:
                    all_rows = published
                    refresh_at = time.monotonic() + _next_refresh_delay(all_rows, interval)
                rows = filter_rows(
                    all_rows,
                    active_only=state.active_only,
                    search=state.search_query,
                    latest_only=state.latest_only,
                )
                state.sync(rows)
                now = time.monotonic()
                needs_detail = (
                    state.detail_run_id is not None
                    and state.detail_run_id != refresh.detail_run_id
                )
                manual_refresh = state.refresh_requested
                if (
                    manual_refresh or now >= refresh_at or needs_detail
                ) and refresh.start(service, console, state):
                    state.refresh_requested = False
                size = (console.size.width, console.size.height)
                resized = size != last_size
                last_size = size
                if key or published is not None or resized:
                    _render_live(live, service, rows, state, console)
        finally:
            refresh.close()


def _render_live(live, service, rows, state, console):
    """Render only cached rows and snapshots, including during detail loading."""
    if state.detail_run_id is None:
        live.update(render_page(service, rows, state, console, cached_only=True), refresh=True)
        return
    selected = next((row for row in rows if row["id"] == state.detail_run_id), None)
    if selected is None:
        state.detail_run_id = None
        live.update(render_page(service, rows, state, console, cached_only=True), refresh=True)
    else:
        live.update(render_detail(service, selected, cached_only=True), refresh=True)
