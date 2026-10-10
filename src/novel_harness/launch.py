"""Shared controller-side launch orchestration for CLI and future adapters.

The controller prepares and launches, then returns immediately. The independent
worker owns execution and the monitor is a separate read-only process.
"""

from __future__ import annotations

from pathlib import Path

from .monitor import manual_watch_argv, manual_watch_command


def watch_info(config, run_id) -> dict:
    return {
        "watch_argv": manual_watch_argv(config, run_id),
        "watch_command": manual_watch_command(config, run_id),
    }


def launch_run(service, config_path, run_id: str, *, monitor: bool = False) -> dict:
    """Start/reuse a worker and optionally open a separate monitor; never waits."""
    from .monitor import spawn_monitor
    from .worker import start_worker

    config = Path(config_path).resolve()
    info = start_worker(config, run_id)
    service.activate(run_id, config)
    result = {"run_id": run_id, **info, **watch_info(config, run_id)}
    if not monitor:
        return result
    if info.get("completed"):
        result["monitor"] = {
            "started": False,
            "reason": "run_completed",
            **watch_info(config, run_id),
        }
    elif info.get("started") or info.get("already_running"):
        result["monitor"] = spawn_monitor(config, run_id)
    else:
        # The child failed during startup; do not open a monitor on a dead run.
        result["monitor"] = {
            "started": False,
            "reason": info.get("error", "worker_not_started"),
            **watch_info(config, run_id),
        }
    return result
