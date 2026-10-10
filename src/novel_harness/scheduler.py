"""Bounded parallel scheduling for independent work units.

Concurrency never exceeds ``max_workers``. Results are returned in input order so
callers can apply side effects (commits) deterministically. Errors are captured
per unit instead of aborting the pool, so an all-fail run still leaves
recoverable per-unit state. Scheduling stops submitting new units as soon as
``should_stop`` returns true; already-running units finish.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnitResult[T, R]:
    item: T
    value: R | None
    error: BaseException | None


class BoundedScheduler:
    def __init__(self, max_workers: int):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    def run[T, R](
        self,
        items: Sequence[T],
        work: Callable[[T], R],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[UnitResult[T, R]]:
        """Run units with at most ``max_workers`` in flight.

        Submission is back-pressured: a new unit is only submitted after one
        in-flight unit completes, and ``should_stop`` is checked immediately
        before every submission. A stop therefore prevents units that have not
        started from running, while units already in flight finish normally.
        Completed results are returned in input order.
        """
        queue = list(items)
        results: list[UnitResult[T, R]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            pending: deque[Any] = deque()
            index = 0
            stopped = False

            def stopped_now() -> bool:
                return should_stop is not None and should_stop()

            def fill_window():
                nonlocal index, stopped
                while not stopped and len(pending) < self.max_workers and index < len(queue):
                    if stopped_now():
                        stopped = True
                        return
                    item = queue[index]
                    index += 1
                    pending.append((item, pool.submit(work, item)))

            fill_window()
            while pending:
                item, future = pending.popleft()
                try:
                    results.append(UnitResult(item, future.result(), None))
                except BaseException as error:  # noqa: BLE001 - isolate per-unit failure
                    results.append(UnitResult(item, None, error))
                fill_window()
        return results
