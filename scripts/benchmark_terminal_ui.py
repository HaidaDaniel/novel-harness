"""Measure the production terminal loop against a synthetic large saved-run store.

The fixture is temporary, contains no source text or model settings, and drives
the real ``tui()`` entrypoint through a POSIX PTY. The default command is
intentionally long enough to expose warm idle behavior::

    python scripts/benchmark_terminal_ui.py --seconds 30
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import select
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import novel_harness.progress as progress_module
import novel_harness.run_dashboard as dashboard
from novel_harness import terminal_ui
from novel_harness.persistence import Repository
from novel_harness.reporting import Reporter


def _now(index: int) -> str:
    return (datetime.now(UTC) - timedelta(seconds=index)).isoformat()


def _manifest(run_id: str, index: int, chunks_per_run: int) -> tuple[dict, list[dict]]:
    chapters = []
    rows = []
    chapter_count, remainder = divmod(chunks_per_run, 10)
    if remainder:
        raise ValueError("chunks-per-run must be divisible by 10")
    for chapter_number in range(1, chapter_count + 1):
        chapter_chunks = []
        for ordinal in range(10):
            chunk_id = f"c{chapter_number:03d}-{ordinal:02d}"
            source_sha256 = "0" * 64
            chapter_chunks.append({"id": chunk_id, "index": ordinal, "text": ""})
            rows.append(
                {
                    "chapter": chapter_number,
                    "id": chunk_id,
                    "index": ordinal,
                    "source_sha256": source_sha256,
                }
            )
        chapters.append(
            {
                "number": chapter_number,
                "title": f"Synthetic chapter {chapter_number}",
                "source_sha256": "1" * 64,
                "chunks": chapter_chunks,
            }
        )
    created_at = _now(index)
    manifest = {
        "run_id": run_id,
        "fingerprint": f"fingerprint-{index}",
        "created_at": created_at,
        "source_dir": "<temporary-synthetic-source>",
        "workflow": "direct",
        "novel": {"title": f"Synthetic book {index:03d}"},
        "chapters": chapters,
        "stages": [
            {
                "name": "translation",
                "runner": "synthetic",
                "runner_config": {"model": "synthetic"},
            }
        ],
    }
    return manifest, rows


class SyntheticService:
    """The smallest service surface used by the read-only projections."""

    def __init__(self, root: Path, repository: Repository):
        self.root = root
        self.repository = repository
        self.reporter = Reporter(self)
        self.run_ids: list[str] = []
        self.action_calls: list[str] = []

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def lightweight_statuses(self, run_ids=None):
        return self.reporter.lightweight_statuses(run_ids)

    def status(self, run_id):
        return self.reporter.lightweight_status(run_id)

    def rescan_library(self):
        return self.library_novels()

    def library_novels(self):
        return {
            "novels": [
                {
                    "novel_id": f"synthetic-novel-{index}",
                    "title": f"Synthetic book {index:03d}",
                    "source_path": str(self.root / f"book-{index}"),
                    "status": "ready",
                    "chapter_count": 30,
                    "chapters": [],
                    "active_run": None,
                    "latest_run": {"run_id": self.run_ids[index], "status": "running"},
                    "continue_run_id": self.run_ids[index],
                }
                for index in range(len(self.run_ids))
            ],
            "diagnostics": [],
        }

    def library_remaining(self):
        return {
            "novels": [
                {
                    "novel_id": f"synthetic-novel-{index}",
                    "title": f"Synthetic book {index:03d}",
                    "source_path": str(self.root / f"book-{index}"),
                    "chapter_count": 30,
                    "translated": 0,
                    "remaining": 30,
                    "run_id": self.run_ids[index],
                }
                for index in range(len(self.run_ids))
            ],
            "excluded": [],
        }

    def default_preset(self):
        return "synthetic"

    def presets(self):
        return [{"id": "synthetic", "runner": "synthetic", "available": True}]

    def launch_remaining_novel(self, novel_id, config):
        self.action_calls.append(novel_id)
        return {"run_id": self.run_ids[0], "started": True}


def _rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text("utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if platform.system() != "Darwin" else 1))
    except (ImportError, AttributeError, OSError):
        return None


def _wait(future, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.001)
    if not future.done():
        raise TimeoutError("synthetic collector did not finish")


def _run_live_tui_child(service, config: Path, metrics_path: Path) -> None:
    metrics = {
        "key_reads": 0,
        "dashboard_refresh_starts": 0,
        "library_refresh_starts": 0,
        "live_updates": 0,
        "live_refreshes": 0,
        "action_calls": [],
        "rss_samples": [],
        "key_render_ms": [],
        "error": None,
    }
    original_update = terminal_ui.Live.update
    original_refresh = terminal_ui.Live.refresh
    original_dashboard_start = terminal_ui.DashboardRefresh.start
    original_library_start = terminal_ui.LibraryRefresh.start
    original_read_posix = dashboard._read_posix
    last_key_ns = None

    def update(live, *args, **kwargs):
        nonlocal last_key_ns
        metrics["live_updates"] += 1
        if last_key_ns is not None:
            metrics["key_render_ms"].append((time.perf_counter_ns() - last_key_ns) / 1_000_000)
            last_key_ns = None
        return original_update(live, *args, **kwargs)

    def refresh(live, *args, **kwargs):
        metrics["live_refreshes"] += 1
        return original_refresh(live, *args, **kwargs)

    def dashboard_start(refresh_obj, *args, **kwargs):
        metrics["dashboard_refresh_starts"] += 1
        return original_dashboard_start(refresh_obj, *args, **kwargs)

    def library_start(refresh_obj, *args, **kwargs):
        metrics["library_refresh_starts"] += 1
        return original_library_start(refresh_obj, *args, **kwargs)

    def read_posix(*args, **kwargs):
        nonlocal last_key_ns
        metrics["key_reads"] += 1
        event = original_read_posix(*args, **kwargs)
        if event is not None:
            last_key_ns = time.perf_counter_ns()
        return event

    terminal_ui.Live.update = update
    terminal_ui.Live.refresh = refresh
    terminal_ui.DashboardRefresh.start = dashboard_start
    terminal_ui.LibraryRefresh.start = library_start
    dashboard._read_posix = read_posix
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    metrics["rss_samples"].append(_rss_bytes())
    try:
        # Pytest and other harnesses may replace sys.stdin/stdout objects after
        # fork; bind the real PTY descriptors back before entering tui().
        if not sys.stdin.isatty():
            sys.stdin = os.fdopen(os.dup(0), "r", buffering=1)
        if not sys.stdout.isatty():
            sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)
        terminal_ui.tui(service, config)
    except BaseException as error:  # noqa: BLE001 - publish child diagnostics
        metrics["error"] = f"{type(error).__name__}: {error}"
    finally:
        metrics["cpu_fraction_one_core"] = (time.process_time() - cpu_start) / max(
            time.monotonic() - wall_start, 0.001
        )
        metrics["wall_seconds"] = time.monotonic() - wall_start
        metrics["rss_samples"].append(_rss_bytes())
        ordered = sorted(metrics.pop("key_render_ms"))
        metrics["key_latency"] = {
            "samples": len(ordered),
            "p50_ms": round(statistics.median(ordered), 3) if ordered else None,
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3)
            if ordered
            else None,
            "max_ms": round(max(ordered), 3) if ordered else None,
        }
        metrics["action_calls"] = list(getattr(service, "action_calls", []))
        metrics_path.write_text(json.dumps(metrics), "utf-8")
        terminal_ui.Live.update = original_update
        terminal_ui.Live.refresh = original_refresh
        terminal_ui.DashboardRefresh.start = original_dashboard_start
        terminal_ui.LibraryRefresh.start = original_library_start
        dashboard._read_posix = original_read_posix
        os._exit(0)


def _measure_live_tui(service, config: Path, root: Path, seconds: float) -> dict:
    """Measure the production TUI loop through a real POSIX PTY."""
    try:
        import pty
    except ImportError:
        return {"available": False, "reason": "pty unavailable"}
    metrics_path = root / "live-tui-metrics.json"
    pid, master = pty.fork()
    if pid == 0:
        _run_live_tui_child(service, config, metrics_path)
    start = time.monotonic()
    sent = set()
    events = [
        (min(0.5, max(0.1, seconds * 0.1)), b"\t"),
        (min(0.8, max(0.2, seconds * 0.2)), b" "),
        (min(1.0, max(0.3, seconds * 0.3)), b"j"),
    ]
    output_bytes = 0
    while time.monotonic() - start < seconds:
        elapsed = time.monotonic() - start
        for index, (at, value) in enumerate(events):
            if index not in sent and elapsed >= at:
                os.write(master, value)
                sent.add(index)
        readable, _, _ = select.select([master], [], [], 0.02)
        if readable:
            try:
                output_bytes += len(os.read(master, 65536))
            except OSError:
                break
    try:
        os.write(master, b"q")
    except OSError:
        pass
    deadline = time.monotonic() + 10
    status = None
    while time.monotonic() < deadline:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            break
        try:
            readable, _, _ = select.select([master], [], [], 0.05)
            if readable:
                output_bytes += len(os.read(master, 65536))
        except (OSError, ValueError):
            break
    if status is None:
        os.kill(pid, 9)
        _, status = os.waitpid(pid, 0)
    try:
        os.close(master)
    except OSError:
        pass
    if not metrics_path.is_file():
        return {"available": False, "reason": "tui child did not publish metrics", "status": status}
    metrics = json.loads(metrics_path.read_text("utf-8"))
    metrics.update({"available": True, "pty_output_bytes": output_bytes, "exit_status": status})
    return metrics


def _build_fixture(root: Path, run_count: int, chunks_per_run: int):
    repository = Repository(root / "state.sqlite3")
    service = SyntheticService(root, repository)
    run_ids = []
    for index in range(run_count):
        run_id = f"20261004T{index:06d}-synthetic-{index:032x}"
        manifest, chunks = _manifest(run_id, index, chunks_per_run)
        repository.create(manifest, chunks)
        run_ids.append(run_id)
    service.run_ids = run_ids
    active = run_ids[-1]
    service.run_dir(active).mkdir(parents=True, exist_ok=True)
    (service.run_dir(active) / "worker.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        "utf-8",
    )
    return service, run_ids


def benchmark(*, seconds: float, run_count: int, chunks_per_run: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="novel-terminal-benchmark-") as temporary:
        root = Path(temporary)
        rss_before_import = _rss_bytes()
        service, _run_ids = _build_fixture(root, run_count, chunks_per_run)
        rss_after_import = _rss_bytes()
        storage_counts = {
            "list_run_summaries": 0,
            "run_rows": 0,
            "chunk_rows": 0,
            "worker_state_reads": 0,
            "output_hash_checks": 0,
        }
        repository = service.repository
        original_summaries = repository.list_run_summaries
        original_run = repository.run
        original_chunks = repository.chunks
        original_worker_state = progress_module.worker_state

        def counted_summaries(run_ids=None):
            storage_counts["list_run_summaries"] += 1
            return original_summaries(run_ids)

        def counted_run(run_id):
            storage_counts["run_rows"] += 1
            return original_run(run_id)

        def counted_chunks(run_id):
            storage_counts["chunk_rows"] += 1
            return original_chunks(run_id)

        def counted_worker_state(path):
            storage_counts["worker_state_reads"] += 1
            return original_worker_state(path)

        repository.list_run_summaries = counted_summaries
        repository.run = counted_run
        repository.chunks = counted_chunks
        dashboard.worker_state = counted_worker_state
        progress_module.worker_state = counted_worker_state
        live_tui = _measure_live_tui(service, root / "config.yaml", root, seconds)
        live_tui["rss_before_import_bytes"] = rss_before_import
        live_tui["rss_after_import_bytes"] = rss_after_import
        repository.list_run_summaries = original_summaries
        repository.run = original_run
        repository.chunks = original_chunks
        dashboard.worker_state = original_worker_state
        progress_module.worker_state = original_worker_state
        return {
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "terminal": os.environ.get("TERM") or "unknown",
                "pid": os.getpid(),
            },
            "dataset": {
                "runs": run_count,
                "chunks": run_count * chunks_per_run,
                "chunks_per_run": chunks_per_run,
                "temporary_root": True,
                "runner_calls": 0,
            },
            "live_tui": live_tui,
            "storage": storage_counts,
            "targets": {
                "live_tui_path_exercised": (
                    live_tui.get("available") is True
                    and live_tui.get("key_reads", 0) > 0
                    and live_tui.get("dashboard_refresh_starts", 0) > 0
                    and live_tui.get("library_refresh_starts", 0) > 0
                    and live_tui.get("live_updates", 0) > 0
                ),
                "warm_idle_cpu_at_or_below_1_percent": (
                    live_tui.get("cpu_fraction_one_core", 1) * 100 <= 1
                ),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--chunks-per-run", type=int, default=300)
    args = parser.parse_args()
    if args.seconds < 0 or args.runs < 1 or args.chunks_per_run < 1:
        parser.error("seconds must be non-negative and runs/chunks must be positive")
    print(
        json.dumps(
            benchmark(
                seconds=args.seconds,
                run_count=args.runs,
                chunks_per_run=args.chunks_per_run,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
