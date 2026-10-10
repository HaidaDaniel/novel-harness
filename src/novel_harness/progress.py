"""Read-only progress snapshots derived from persisted run state.

The monitor and this module never call a runner or a model: they read SQLite
rows, artifact metadata and ``worker.json``. Unknown values are reported as
``None`` rather than guessed.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path

ACTIVE_WORKER_STATES = {"starting", "running", "waiting_retry"}
MIN_INTERVAL = 0.1
DEFAULT_ETA_MIN_SAMPLES = 3
RECENT_WINDOW = 10
STARTING_STALE_SECONDS = 120.0


def _pid_alive_posix(pid: int) -> bool | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _pid_alive_windows(pid: int, *, kernel32=None, get_last_error=None) -> bool | None:
    """Probe without terminating: OpenProcess + GetExitCodeProcess only."""
    import ctypes
    from ctypes import wintypes

    if kernel32 is None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if get_last_error is None:
        get_last_error = ctypes.get_last_error

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87

    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = get_last_error()
        if error == error_invalid_parameter:
            return False
        # Access denied (or an unknown error) means the process may exist but is
        # not probed; report unknown rather than guessing liveness.
        if error == error_access_denied:
            return None
        return None
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid, *, platform=None, kernel32=None, get_last_error=None) -> bool | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    platform = platform or os.name
    if platform == "nt":
        return _pid_alive_windows(pid, kernel32=kernel32, get_last_error=get_last_error)
    return _pid_alive_posix(pid)


def worker_state(run_dir: Path) -> dict | None:
    path = run_dir / "worker.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {"status": "unknown", "alive": None}
    if not isinstance(data, dict):
        return {"status": "unknown", "alive": None}
    data["alive"] = _pid_alive(data.get("pid"))
    return data


def _stage_entries(status: dict, manifest: dict) -> dict:
    """Manifest-stage defaults, enriched with the stage harness for fallback."""
    manifest_types = {
        stage["name"]: (stage.get("runner_config") or {}).get("type")
        for stage in manifest.get("stages", [])
    }
    entries = {}
    for entry in status.get("runners", []):
        value = dict(entry)
        value.setdefault("harness", manifest_types.get(entry["stage"]))
        entries[entry["stage"]] = value
    return entries


def _running_attempt(service, run_id: str, chapter: int, chunk_id: str, stage_names):
    """Return ``(stage_name, attempt_metadata)`` for the running attempt, if any."""
    for name in stage_names:
        attempts = service.repository.attempts(run_id, chapter, chunk_id, name)
        running = [item for item in attempts if item.get("status") == "running"]
        if running:
            return name, running[-1]
    return None, None


def _entry(chapter, chunk, row, *, stage, attempt, runner, model, harness):
    return {
        "chapter": chapter["number"],
        "chunk_id": chunk["id"],
        "position_in_chapter": row["ordinal"],
        "chunks_in_chapter": len(chapter["chunks"]),
        "status": row["status"],
        "stage": stage,
        "attempt": attempt,
        "runner": runner,
        "model": model,
        "harness": harness,
    }


def _pending_entry(chapter, chunk, row):
    return _entry(
        chapter, chunk, row, stage=None, attempt=None, runner=None, model=None, harness=None
    )


def _positions(service, run_id, rows, manifest, status):
    stage_entries = _stage_entries(status, manifest)
    stage_names = [entry["stage"] for entry in status.get("runners", [])]
    manifest_stage_order = [s["name"] for s in manifest["stages"]]
    # Preserve manifest stage order for attempt probing even if status order differs.
    probe_order = [name for name in manifest_stage_order if name in stage_names] or stage_names
    chapters = {chapter["number"]: chapter for chapter in manifest["chapters"]}
    chunks = {
        (chapter["number"], chunk["id"]): chunk
        for chapter in manifest["chapters"]
        for chunk in chapter["chunks"]
    }
    by_key = {(row["chapter"], row["chunk_id"]): row for row in rows}
    active = []
    for row in rows:
        if row["status"] != "running":
            continue
        key = (row["chapter"], row["chunk_id"])
        chapter, chunk = chapters.get(row["chapter"]), chunks.get(key)
        if chapter is None or chunk is None:
            continue
        active.append(
            _entry_probe(service, run_id, chapter, chunk, row, stage_entries, probe_order)
        )
    current = active[0] if active else None
    if current is None:
        for chapter in manifest["chapters"]:
            for chunk in chapter["chunks"]:
                row = by_key.get((chapter["number"], chunk["id"]))
                if row is None or row["status"] == "completed":
                    continue
                current = _pending_entry(chapter, chunk, row)
                break
            if current is not None:
                break
    return current, active


def _entry_probe(service, run_id, chapter, chunk, row, stage_entries, probe_order):
    stage, attempt = _running_attempt(
        service, run_id, chapter["number"], chunk["id"], probe_order
    )
    selected = stage_entries.get(stage) if stage else None
    metadata = attempt or {}
    # Adaptive routing records the effective runner/model on the running attempt;
    # legacy attempts fall back to the manifest/default stage runner.
    return _entry(
        chapter,
        chunk,
        row,
        stage=stage,
        attempt=metadata.get("attempt"),
        runner=metadata.get("runner") or (selected["name"] if selected else None),
        model=metadata.get("model") or (selected["model"] if selected else None),
        harness=metadata.get("harness") or (selected.get("harness") if selected else None),
    )


def _elapsed_seconds(run: dict, worker: dict | None) -> float:
    if (
        not worker
        or worker.get("status") not in ACTIVE_WORKER_STATES
        or worker.get("alive") is not True
        or not worker.get("started_at")
    ):
        # No active worker: persisted duration is authoritative.
        return max(0.0, float(run.get("duration_seconds") or 0.0))
    base = worker.get("base_duration_seconds")
    if base is None:
        # Legacy/launcher record without a frozen baseline: persisted duration is
        # the best available baseline at this instant.
        base = float(run.get("duration_seconds") or 0.0)
    try:
        started = datetime.fromisoformat(worker["started_at"])
    except (TypeError, ValueError):
        return max(0.0, float(base))
    return max(0.0, float(base) + max(0.0, (datetime.now(UTC) - started).total_seconds()))


def _timing_samples(rows: list[dict], manifest: dict) -> list[float]:
    inherited = {(item["chapter"], item["chunk_id"]) for item in manifest.get("inherited", [])}
    samples = []
    for row in rows:
        if row["status"] != "completed":
            continue
        if (row["chapter"], row["chunk_id"]) in inherited:
            continue
        raw = row.get("metadata")
        if not raw:
            continue
        try:
            metadata = json.loads(raw)
        except (TypeError, ValueError):
            continue
        stages = metadata.get("stages") or []
        seconds = sum(
            stage["duration_seconds"]
            for stage in stages
            if stage.get("duration_seconds") is not None
        )
        if seconds > 0:
            samples.append(seconds)
    return samples


def _rate(samples: list[float], pending: int, min_samples: int):
    if pending <= 0 or len(samples) < max(1, min_samples):
        return None, None, len(samples)
    window = samples[-RECENT_WINDOW:]
    median = statistics.median(window)
    if median <= 0:
        return None, None, len(window)
    return round(60.0 / median, 3), round(median * pending, 1), len(window)


def snapshot(
    service,
    run_id: str,
    *,
    min_samples: int = DEFAULT_ETA_MIN_SAMPLES,
    status: dict | None = None,
) -> dict:
    """Build a progress snapshot, optionally from a no-hash status projection.

    The default remains the full verification contract used by status/watch and
    one-shot output. Interactive dashboard hydration supplies a lightweight
    status so this selected-run detail read does not hash completed outputs.
    """
    status = service.status(run_id) if status is None else status
    run = service.repository.run(run_id)
    manifest = run["manifest"]
    rows = service.repository.chunks(run_id)
    worker = worker_state(service.run_dir(run_id))
    current, active = _positions(service, run_id, rows, manifest, status)
    total = status["total_chunks"]
    completed = status["chunks"]["completed"]
    pending = max(0, total - completed)
    samples = _timing_samples(rows, manifest)
    throughput, eta, basis = _rate(samples, pending, min_samples)
    chapter_positions = {
        chapter["number"]: index + 1 for index, chapter in enumerate(manifest["chapters"])
    }
    return {
        "run_id": run_id,
        "status": status["status"],
        "workflow": status["workflow"],
        "runners": status["runners"],
        "chunks": status["chunks"],
        "total_chunks": total,
        "progress": status["progress"],
        "chapters": {
            "current_number": current["chapter"] if current else None,
            "current_position": chapter_positions.get(current["chapter"]) if current else None,
            "total": len(manifest["chapters"]),
        },
        "current": current,
        "active": active,
        "active_count": len(active),
        "worker": worker,
        "elapsed_seconds": round(_elapsed_seconds(run, worker), 1),
        "throughput_chunks_per_minute": throughput,
        "eta_seconds": eta,
        "eta_basis_chunks": basis,
        "retry": status["retry"],
        "error": status["error"],
    }


def _format_duration(seconds) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {rest}s"
    return f"{minutes}m {rest}s"


def _bar(progress: float, width: int = 30) -> str:
    clamped = min(1.0, max(0.0, progress))
    filled = round(clamped * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {clamped * 100:5.1f}%"


def _worker_line(worker: dict | None) -> str:
    if not worker:
        return "Worker PID: unknown (unknown)"
    alive = worker.get("alive")
    label = "alive" if alive is True else "dead" if alive is False else "unknown"
    return f"Worker PID: {worker.get('pid')} ({label})"


def render_snapshot(data: dict) -> str:
    lines = [
        f"Run: {data['run_id']}",
        f"State: {data['status']}",
        f"Pipeline: {data.get('workflow') or '-'}",
    ]
    current = data.get("current")
    if current:
        stage = current.get("stage") or "-"
        runner = current.get("runner") or "-"
        model = current.get("model") or "-"
        lines.append(f"Stage: {stage}")
        lines.append(f"Runner: {runner} / {model}")
        lines.append(f"Attempt: {current.get('attempt') or '-'}")
    else:
        lines.append("Stage: -")
        lines.append("Runner: -")
    lines.append(_worker_line(data.get("worker")))
    chapters = data.get("chapters") or {}
    number = chapters.get("current_number")
    position = chapters.get("current_position")
    total = chapters.get("total")
    if number is None:
        lines.append(f"Chapter: - / {total if total is not None else '-'}")
    else:
        lines.append(f"Chapter: {position} / {total} (#{number})")
    if current:
        lines.append(
            f"Chunk: {current.get('position_in_chapter')}/{current.get('chunks_in_chapter')}"
        )
    if data.get("active_count", 0) > 1:
        for entry in data.get("active", []):
            lines.append(
                f"Active: chapter {entry['chapter']} chunk {entry['chunk_id']} "
                f"stage {entry.get('stage') or '-'} attempt {entry.get('attempt') or '-'}"
            )
    counts = data["chunks"]
    lines.append(
        f"Counts: completed {counts['completed']} pending {counts['pending']} "
        f"running {counts['running']} failed {counts['failed']} corrupt {counts['corrupt']}"
    )
    lines.append(_bar(data.get("progress") or 0.0))
    lines.append(f"Elapsed: {_format_duration(data.get('elapsed_seconds'))}")
    if data.get("throughput_chunks_per_minute") is not None:
        lines.append(f"Throughput: {data['throughput_chunks_per_minute']} chunks/min")
    else:
        lines.append("Throughput: unknown")
    if data.get("eta_seconds") is not None:
        lines.append(
            f"ETA: {_format_duration(data['eta_seconds'])} "
            f"({data.get('eta_basis_chunks')} samples)"
        )
    else:
        lines.append(f"ETA: unknown ({data.get('eta_basis_chunks', 0)} samples)")
    retry = data.get("retry")
    worker = data.get("worker") or {}
    if retry:
        lines.append(f"Retry: {retry.get('reason')} until {retry.get('retry_at')}")
    elif worker.get("status") == "waiting_retry":
        lines.append(f"Retry: backoff {worker.get('retry_delay_seconds')}s")
    if data.get("error"):
        lines.append(f"Last error: {data['error']}")
    return "\n".join(lines)


def render_rich(data: dict):
    """Rich renderable with a real progress bar; falls back to plain text data."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.text import Text

    progress = min(1.0, max(0.0, data.get("progress") or 0.0))
    body = Group(
        Text(render_snapshot(data)),
        ProgressBar(total=1.0, completed=progress, width=48),
    )
    return Panel(body, title=f"Run {data['run_id']}", border_style="cyan")
