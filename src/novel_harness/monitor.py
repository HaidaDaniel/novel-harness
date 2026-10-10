"""Live read-only progress monitor.

Polls persisted state at a bounded interval in an ordinary Python process. It
never calls a runner/model, never mutates translation state, and never requires a
controller agent loop. Monitor failure never stops the translation worker.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime

from rich.console import Console
from rich.live import Live

from .progress import (
    ACTIVE_WORKER_STATES,
    MIN_INTERVAL,
    STARTING_STALE_SECONDS,
    render_rich,
    snapshot,
)

TERMINAL_STATES = {"completed", "failed", "paused", "interrupted", "incomplete"}
#: Terminal emulators that accept the command as a trailing argv (no shell string).
POSIX_TERMINALS = (
    ("gnome-terminal", ["--"]),
    ("konsole", ["-e"]),
    ("x-terminal-emulator", ["-e"]),
    ("xterm", ["-e"]),
    ("alacritty", ["-e"]),
    ("kitty", []),
)


def manual_watch_argv(config, run_id, *, program="novel") -> list[str]:
    return [program, "watch", "--config", str(config), "--run", run_id]


def manual_watch_command(config, run_id, *, program="novel", platform=None) -> str:
    argv = manual_watch_argv(config, run_id, program=program)
    platform = platform or os.name
    if platform == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _starting_is_fresh(worker: dict) -> bool:
    started = worker.get("started_at")
    if not started:
        return False
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        return False
    return age <= STARTING_STALE_SECONDS


def is_terminal(data: dict) -> bool:
    """Terminal only when the run state belongs to the current worker lifecycle."""
    worker = data.get("worker") or {}
    status = worker.get("status")
    if status == "starting":
        return not _starting_is_fresh(worker) and data["status"] in TERMINAL_STATES
    if status in ACTIVE_WORKER_STATES and worker.get("alive") is True:
        return False
    return data["status"] in TERMINAL_STATES


def watch(
    service,
    run_id,
    *,
    interval=2.0,
    iterations=None,
    sleep=time.sleep,
    console=None,
    renderer=render_rich,
):
    """Poll and render progress until a terminal state, an iteration bound, or Ctrl-C."""
    if interval < MIN_INTERVAL:
        raise ValueError(f"interval must be at least {MIN_INTERVAL} seconds")
    console = console or Console()
    data = snapshot(service, run_id)
    count = 1
    with Live(renderer(data), console=console, auto_refresh=False) as live:
        live.update(renderer(data), refresh=True)
        while not is_terminal(data):
            if iterations is not None and count >= iterations:
                break
            sleep(interval)
            data = snapshot(service, run_id)
            live.update(renderer(data), refresh=True)
            count += 1
    return data


def _monitor_argv(config, run_id, python) -> list[str]:
    return [python, "-m", "novel_harness", "watch", "--config", str(config), "--run", run_id]


def _find_terminal(which, env):
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return None
    custom = env.get("TERMINAL")
    if custom:
        path = which(custom)
        if path:
            return [path, "-e"]
    for name, prefix in POSIX_TERMINALS:
        path = which(name)
        if path:
            return [path, *prefix]
    return None


def _windows_launch(argv):
    """Run the monitor in its own console window via cmd's detached ``start``."""
    return ["cmd", "/c", "start", "", *argv]


def spawn_monitor(
    config,
    run_id,
    *,
    python=sys.executable,
    platform=None,
    env=None,
    which=None,
    popen=None,
) -> dict:
    """Best-effort separate non-LLM monitor process; never raises into the caller."""
    platform = platform or os.name
    env = os.environ if env is None else env
    which = shutil.which if which is None else which
    popen = subprocess.Popen if popen is None else popen
    fallback = {
        "watch_argv": manual_watch_argv(config, run_id),
        "watch_command": manual_watch_command(config, run_id, platform=platform),
    }
    argv = _monitor_argv(config, run_id, python)
    try:
        if platform == "nt":
            process = popen(
                _windows_launch(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return {"started": True, "pid": process.pid, "terminal": "console", **fallback}
        terminal = _find_terminal(which, env)
        if terminal is None:
            # Never bind a long-lived Rich process to the controller's captured
            # pipes on a headless machine: return the manual watch command instead.
            return {"started": False, "reason": "headless_or_no_terminal", **fallback}
        process = popen(
            [*terminal, *argv],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"started": False, "error": type(error).__name__, **fallback}
    return {
        "started": True,
        "pid": process.pid,
        "terminal": terminal[0],
        "watch_argv": fallback["watch_argv"],
        "watch_command": fallback["watch_command"],
    }
