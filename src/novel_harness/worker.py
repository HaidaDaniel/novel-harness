"""Local process entry point. SQLite holds progress, process files hold diagnostics."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout

from .artifacts import write_json
from .runners.base import RunnerExecutionError
from .service import TranslationService, now

LAUNCHER_TIMEOUT_SECONDS = 10.0
STARTUP_TIMEOUT_SECONDS = 1.5
STARTUP_POLL_SECONDS = 0.05


LOCK_RETRY_ATTEMPTS = 10
LOCK_RETRY_DELAY_SECONDS = 0.2
QUOTA_RETRY_POLL_SECONDS = 15 * 60
RETRY_NOW_FILENAME = "retry-now.requested"


def _retry_now_path(directory: Path) -> Path:
    return directory / RETRY_NOW_FILENAME


def _request_retry_now(directory: Path) -> None:
    """Atomically ask the single live worker to bypass its current retry gate."""
    write_json(_retry_now_path(directory), {"requested_at": now(), "reason": "explicit_start"})


def _consume_retry_now(directory: Path) -> bool:
    """Consume one retry-now request without treating a stale request as an error."""
    try:
        _retry_now_path(directory).unlink()
    except FileNotFoundError:
        return False
    return True


def _clear_pending_retry(directory: Path) -> None:
    """Remove the durable provider gate immediately before a bounded probe."""
    (directory / "retry.json").unlink(missing_ok=True)


def _quota_retry_metadata(directory: Path) -> dict | None:
    """Read a quota marker even after its provider timestamp has expired.

    service.pending_retry intentionally hides expired provider deadlines for
    read-only status projections. The worker needs the durable marker itself to
    keep quota failures out of the ordinary restart budget until a probe is made.
    """
    try:
        data = json.loads((directory / "retry.json").read_text("utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("reason") not in {"quota", "http429"}:
        return None
    if not data.get("retry_at"):
        return None
    return data


def _bounded_run_lock(service, run_id, *, attempts=LOCK_RETRY_ATTEMPTS, delay=LOCK_RETRY_DELAY_SECONDS, sleep=time.sleep):
    """Take ``.run.lock`` despite sub-millisecond read-only ``status`` probes.

    ``Reporter.status`` briefly acquires the execution lock to detect activity.
    A worker's non-blocking acquire must tolerate that transient contention
    instead of treating a read-only probe as a competing executor.
    """
    for attempt in range(attempts):
        lock = service._lock(run_id)
        try:
            lock.acquire()
        except Timeout:
            if attempt == attempts - 1:
                raise
            sleep(delay)
            continue
        return lock


def _bounded_execute(
    service,
    run_id,
    *,
    attempts=LOCK_RETRY_ATTEMPTS,
    delay=LOCK_RETRY_DELAY_SECONDS,
    sleep=time.sleep,
    **kwargs,
):
    """Run ``service.execute``; retry only short-lived read-only lock probes."""
    for attempt in range(attempts):
        try:
            return service.execute(run_id, **kwargs)
        except ValueError as error:
            if "already executing" not in str(error) or attempt == attempts - 1:
                raise
            sleep(delay)


def _bounded_build_epub(
    service,
    run_id,
    *,
    attempts=LOCK_RETRY_ATTEMPTS,
    delay=LOCK_RETRY_DELAY_SECONDS,
    sleep=time.sleep,
):
    """Build the EPUB despite transient read-only lock probes at completion."""
    for attempt in range(attempts):
        try:
            return service.build_epub(run_id)
        except ValueError as error:
            if "currently executing" not in str(error) or attempt == attempts - 1:
                raise
            sleep(delay)


def _exhausted_chunks(service, run_id) -> int:
    """Count chunks whose stage exhausted its durable incomplete-output budget."""
    count = 0
    for row in service.repository.chunks(run_id):
        if row["status"] != "failed" or not row.get("metadata"):
            continue
        try:
            metadata = json.loads(row["metadata"])
        except ValueError:
            continue
        if metadata.get("failure_kind") == "recovery_exhausted":
            count += 1
    return count


def _record_worker_error(metadata: dict, service, run_id, error, result) -> None:
    kind = getattr(error, "failure_kind", "runner_error")
    metadata["current_error"] = str(error)
    metadata["current_failure_kind"] = kind
    history = metadata.setdefault("error_history", [])
    history.append(
        {
            "at": now(),
            "error": str(error),
            "failure_kind": kind,
            "completed_chunks": result["chunks"]["completed"],
        }
    )
    metadata["error_history"] = history[-20:]


def run_worker(
    service,
    run_id,
    *,
    max_chunks=None,
    max_restarts=5,
    restart_delay_seconds=30,
    sleep=time.sleep,
    max_parallel=1,
    quota_retry_poll_seconds=QUOTA_RETRY_POLL_SECONDS,
):
    """Resume after bounded outages; progress resets the consecutive failure budget."""
    if max_restarts < 0 or restart_delay_seconds < 0:
        raise ValueError("Worker recovery settings must be nonnegative")
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    if quota_retry_poll_seconds <= 0:
        raise ValueError("Quota retry poll interval must be positive")
    directory = service.run_dir(run_id)
    with FileLock(str(directory / ".worker.lock"), timeout=0):
        # Do not overwrite an existing foreground/legacy worker's metadata; a
        # transient read-only probe releases the lock immediately, so retry.
        barrier = _bounded_run_lock(service, run_id, sleep=sleep)
        barrier.release()
        # Persisted duration already accumulated by earlier sessions. Freeze it
        # as this worker session's baseline so live elapsed does not add the
        # whole worker wall time on top of durations committed by recovery.
        base_duration_seconds = float(service.repository.run(run_id).get("duration_seconds") or 0.0)
        metadata = {
            "run_id": run_id,
            "pid": os.getpid(),
            "started_at": now(),
            "finished_at": None,
            "status": "running",
            "base_duration_seconds": base_duration_seconds,
            "recoveries": 0,
            "consecutive_failures": 0,
            "max_restarts": max_restarts,
        }
        path = directory / "worker.json"
        write_json(path, metadata)
        best = service.status(run_id)["chunks"]["completed"]
        initial_completed = best
        first = True
        try:
            while True:
                try:
                    # A start/resume request can arrive while a deferred worker
                    # is waiting. Remove the execution gate once, immediately
                    # before the next attempt; execution_operations recreates
                    # it if the provider defers again.
                    if _consume_retry_now(directory):
                        _clear_pending_retry(directory)
                    remaining = (
                        None if max_chunks is None else max_chunks - (best - initial_completed)
                    )
                    if remaining is not None and remaining <= 0:
                        service.repository.set_run(run_id, "paused", now())
                        result = service.status(run_id)
                    else:
                        result = _bounded_execute(
                            service,
                            run_id,
                            sleep=sleep,
                            max_chunks=remaining,
                            clear_pause=first,
                            max_parallel=max_parallel,
                        )
                except RunnerExecutionError as error:
                    result = service.status(run_id)
                    _record_worker_error(metadata, service, run_id, error, result)
                first = False
                completed = result["chunks"]["completed"]
                metadata.update(
                    status=result["status"], completed_chunks=completed, updated_at=now()
                )
                exhausted = _exhausted_chunks(service, run_id)
                if exhausted:
                    metadata["exhausted_chunks"] = exhausted
                    metadata["guidance"] = (
                        "One or more chunk/stage failures exhausted their incomplete-output "
                        "recovery budget. Create a fresh continuation or change model/profile; "
                        "retrying this run will not call the model again for them."
                    )
                if completed > best:
                    metadata["consecutive_failures"] = 0
                    if metadata.get("current_error"):
                        metadata["recovered_from"] = metadata["current_error"]
                    metadata.pop("current_error", None)
                    metadata.pop("current_failure_kind", None)
                best = completed
                if result["status"] == "completed":
                    metadata.pop("current_error", None)
                    metadata.pop("current_failure_kind", None)
                    metadata["epub"] = str(_bounded_build_epub(service, run_id, sleep=sleep))
                    return service.status(run_id)
                if result["status"] == "paused":
                    metadata.pop("current_error", None)
                    metadata.pop("current_failure_kind", None)
                    return result
                if (directory / "pause.requested").exists():
                    service.repository.set_run(run_id, "paused", now())
                    metadata["status"] = "paused"
                    metadata.pop("current_error", None)
                    metadata.pop("current_failure_kind", None)
                    return service.status(run_id)
                deferred = _quota_retry_metadata(directory)
                if not deferred and metadata["consecutive_failures"] >= max_restarts:
                    metadata["status"] = "failed"
                    return result
                if deferred:
                    # retry_at is diagnostic provider guidance. A fixed
                    # probe interval must not collapse to an early retry
                    # merely because the suggested timestamp is near.
                    delay = quota_retry_poll_seconds
                    metadata.update(deferred)
                    metadata["failure_kind"] = "quota"
                    metadata["retry_probe_interval_seconds"] = quota_retry_poll_seconds
                else:
                    metadata.pop("failure_kind", None)
                    metadata["consecutive_failures"] += 1
                    metadata["recoveries"] += 1
                    delay = min(
                        restart_delay_seconds * 2 ** (metadata["consecutive_failures"] - 1), 300
                    )
                metadata.update(status="waiting_retry", retry_delay_seconds=delay)
                write_json(path, metadata)
                print(json.dumps(metadata, ensure_ascii=False), flush=True)
                # Small waits allow a pause request to interrupt backoff promptly.
                waited = 0
                retry_requested = _consume_retry_now(directory)
                while (
                    waited < delay
                    and not (directory / "pause.requested").exists()
                    and not retry_requested
                ):
                    interval = min(1, delay - waited)
                    sleep(interval)
                    waited += interval
                    retry_requested = _consume_retry_now(directory)
                if (directory / "pause.requested").exists():
                    service.repository.set_run(run_id, "paused", now())
                    metadata["status"] = "paused"
                    metadata.pop("current_error", None)
                    metadata.pop("current_failure_kind", None)
                    return service.status(run_id)
                if deferred:
                    _clear_pending_retry(directory)
                    metadata["retry_probe_reason"] = "explicit" if retry_requested else "interval"
                metadata.update(status="running", updated_at=now())
                write_json(path, metadata)
        except BaseException as error:
            metadata.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=type(error).__name__,
            )
            raise
        finally:
            metadata["finished_at"] = now()
            write_json(path, metadata)


def worker_lock_held(directory: Path) -> bool | None:
    """Read-only probe: ``True`` held, ``False`` free, ``None`` undetermined.

    Never used while a child is ``starting``: a transient parent acquire could
    make the child's non-blocking ``.worker.lock`` acquisition fail. When the
    probe does acquire the lock it releases it immediately.
    """
    lock = FileLock(str(directory / ".worker.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return True
    except OSError:
        return None
    try:
        lock.release()
    except OSError:
        return None
    return False


def _starting_is_active(state: dict) -> bool:
    from .progress import STARTING_STALE_SECONDS

    if state.get("alive") is not True:
        return False
    started = state.get("started_at")
    if not started:
        return False
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(started)).total_seconds()
    except (TypeError, ValueError):
        return False
    return age <= STARTING_STALE_SECONDS


def _worker_is_active(directory: Path) -> tuple[bool, dict | None]:
    from .progress import worker_state

    # A fresh launcher-published ``starting`` record is trusted from metadata
    # because the child may not hold ``.worker.lock`` yet. For ``running`` and
    # ``waiting_retry`` the lock is the authoritative execution-ownership signal,
    # so a reused PID with a free lock is not mistaken for an active worker.
    state = worker_state(directory)
    if not state:
        return False, state
    status = state.get("status")
    if status == "starting":
        return _starting_is_active(state), state
    if status in {"running", "waiting_retry"}:
        if state.get("alive") is not True:
            return False, state
        held = worker_lock_held(directory)
        if held is None:
            # Lock state undetermined: trust metadata plus safe PID liveness.
            return True, state
        return held, state
    return False, state


def _publish_starting(directory: Path, run_id: str, base_duration_seconds: float) -> None:
    write_json(
        directory / "worker.json",
        {
            "run_id": run_id,
            "pid": None,
            "started_at": now(),
            "finished_at": None,
            "status": "starting",
            "base_duration_seconds": base_duration_seconds,
            "recoveries": 0,
            "consecutive_failures": 0,
            "published_by": "launcher",
        },
    )


def _publish_failed(directory: Path, run_id: str, pid: int, exit_code) -> None:
    from .progress import worker_state

    state = worker_state(directory) or {}
    state.pop("alive", None)
    state.update(
        {
            "run_id": run_id,
            "pid": pid,
            "status": "failed",
            "error": "worker_exited_during_startup",
            "exit_code": exit_code,
            "finished_at": now(),
        }
    )
    write_json(directory / "worker.json", state)


def _spawn_worker(config: Path, run_id: str, directory: Path):
    args = [sys.executable, "-m", "novel_harness.worker", "--config", str(config), "--run", run_id]
    with (
        (directory / "worker.stdout.log").open("ab") as stdout,
        (directory / "worker.stderr.log").open("ab") as stderr,
    ):
        return subprocess.Popen(
            args,
            cwd=config.parent,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )


def _confirm_started(directory: Path, process, timeout: float, sleep) -> str:
    """Bounded handshake: ``running``/``finished`` once confirmed, else ``starting``/``dead``."""
    from .progress import ACTIVE_WORKER_STATES, worker_state

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        state = worker_state(directory)
        if state and state.get("pid") == process.pid and state.get("alive") is not False:
            # The child published its own metadata: a terminal status means it
            # finished inside the handshake (e.g. a tiny all-cached resume).
            if state.get("status") in ACTIVE_WORKER_STATES:
                return "running"
            return "finished"
        if process.poll() is not None:
            return "dead"
        if time.monotonic() >= deadline:
            latest = worker_state(directory)
            if latest and latest.get("status") == "starting" and latest.get("pid") is None:
                latest.pop("alive", None)
                latest["pid"] = process.pid
                write_json(directory / "worker.json", latest)
            return "starting"
        sleep(STARTUP_POLL_SECONDS)


def start_worker(
    config: str | Path,
    run_id: str,
    *,
    startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
    launcher_timeout: float = LAUNCHER_TIMEOUT_SECONDS,
    sleep=time.sleep,
) -> dict:
    """Idempotently launch an independent worker for ``run_id``.

    A launcher lock serializes check, spawn and startup publication so two
    concurrent launches cannot both report success. The parent never holds
    ``.worker.lock``; the child takes it and becomes the single execution owner.
    """
    config = Path(config).resolve()
    service = TranslationService.from_config(config)
    directory = service.run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    launcher = FileLock(str(directory / ".launcher.lock"), timeout=launcher_timeout)
    try:
        launcher.acquire()
    except Timeout as error:
        raise ValueError(f"Another launch is already in progress for run {run_id}") from error
    try:
        status = service.status(run_id)
        from .progress import worker_state

        if status["status"] == "completed":
            state = worker_state(directory) or {}
            return {
                "run_id": run_id,
                "started": False,
                "already_running": False,
                "completed": True,
                "pid": None,
                "worker_state": str(directory / "worker.json"),
                "epub": state.get("epub"),
            }
        active, state = _worker_is_active(directory)
        if active:
            result = {
                "run_id": run_id,
                "started": False,
                "already_running": True,
                "completed": False,
                "pid": (state or {}).get("pid"),
                "worker_status": (state or {}).get("status"),
                "worker_state": str(directory / "worker.json"),
            }
            state = state or {}
            known_failure_kind = state.get("failure_kind") or state.get("current_failure_kind")
            quota_marker = _quota_retry_metadata(directory)
            quota_wait = known_failure_kind == "quota" or (
                known_failure_kind is None
                and (
                    state.get("reason") == "quota"
                    or quota_marker is not None
                )
            )
            waiting_for_retry = (
                state.get("status") == "waiting_retry"
                or status["status"] == "waiting_retry"
                or quota_marker is not None
            )
            if waiting_for_retry and quota_wait:
                _request_retry_now(directory)
                result["retry_requested"] = True
            return result
        try:
            verify_lock = _bounded_run_lock(service, run_id, sleep=sleep)
        except Timeout:
            # A real executor (not a transient read-only probe) holds the lock.
            state = worker_state(directory) or {}
            return {
                "run_id": run_id,
                "started": False,
                "already_running": True,
                "completed": False,
                "pid": state.get("pid"),
                "worker_status": state.get("status"),
                "worker_state": str(directory / "worker.json"),
            }
        try:
            service._verify_inputs(service.repository.run(run_id)["manifest"])
        finally:
            verify_lock.release()
        retry_requested = _quota_retry_metadata(directory) is not None
        if retry_requested:
            _request_retry_now(directory)
        base_duration = float(service.repository.run(run_id).get("duration_seconds") or 0.0)
        _publish_starting(directory, run_id, base_duration)
        process = _spawn_worker(config, run_id, directory)
        outcome = _confirm_started(directory, process, startup_timeout, sleep)
        if outcome == "dead":
            exit_code = process.poll()
            _publish_failed(directory, run_id, process.pid, exit_code)
            return {
                "run_id": run_id,
                "started": False,
                "already_running": False,
                "completed": False,
                "error": "worker_exited_during_startup",
                "exit_code": exit_code,
                "worker_state": str(directory / "worker.json"),
            }
        if outcome == "finished":
            # The child already reached a terminal state inside the handshake.
            state = worker_state(directory) or {}
            return {
                "run_id": run_id,
                "started": False,
                "already_running": False,
                "completed": state.get("status") == "completed",
                "pid": process.pid,
                "worker_status": state.get("status"),
                "epub": state.get("epub"),
                "worker_state": str(directory / "worker.json"),
            }
        result = {
            "run_id": run_id,
            "started": True,
            "already_running": False,
            "completed": False,
            "pid": process.pid,
            "worker_status": "running" if outcome == "running" else "starting",
            "worker_state": str(directory / "worker.json"),
        }
        if retry_requested:
            result["retry_requested"] = True
        return result
    finally:
        launcher.release()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description="Execute a persisted run; build EPUB on success")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--restart-delay-seconds", type=float, default=30)
    parser.add_argument(
        "--quota-retry-poll-seconds",
        type=float,
        default=QUOTA_RETRY_POLL_SECONDS,
        help="Maximum wait before probing a deferred provider quota again",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Bounded parallel chunks for independent work (context_strategy: none)",
    )
    args = parser.parse_args()
    service = TranslationService.from_config(args.config)
    result = run_worker(
        service,
        args.run,
        max_chunks=args.max_chunks,
        max_restarts=args.max_restarts,
        restart_delay_seconds=args.restart_delay_seconds,
        max_parallel=args.parallel,
        quota_retry_poll_seconds=args.quota_retry_poll_seconds,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"completed", "paused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
