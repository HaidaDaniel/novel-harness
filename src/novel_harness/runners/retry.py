"""Provider error parsing only: assistant prose never triggers quota handling."""

import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

from .base import RunnerDeferred


def quota_event(event: dict, *, current: datetime | None = None, local_timezone=None):
    current = current or datetime.now(UTC)
    if event.get("type") in {"error", "turn.failed"}:
        error = event.get("error") or event
    elif event.get("type") == "item.completed" and event.get("item", {}).get("type") == "error":
        error = event["item"]
    else:
        return None
    if not isinstance(error, dict):
        return None
    data = error.get("data") or error
    if not isinstance(data, dict):
        return None
    message = str(data.get("message", ""))
    if not re.search(
        r"usage limit|rate.?limit|insufficient.quota|too many requests", message, re.IGNORECASE
    ):
        return None
    deadline, source = None, "fallback_1h"
    for key in ("retry_at", "reset_at", "resets_at"):
        value = data.get(key)
        if value is not None:
            try:
                deadline = (
                    datetime.fromtimestamp(float(value), UTC)
                    if isinstance(value, (int, float))
                    else datetime.fromisoformat(value)
                )
                if deadline.tzinfo is None:
                    deadline = None
                else:
                    source = "provider_timestamp"
            except (ValueError, TypeError, OverflowError, OSError):
                pass
            if deadline:
                break
    relative = re.search(
        r"(?:retry|try again) (?:after|in) (\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?)",
        message,
        re.IGNORECASE,
    )
    if deadline is None and relative:
        unit = relative[2].lower()
        seconds = float(relative[1]) * (
            3600 if unit.startswith("hour") else 60 if unit.startswith("minute") else 1
        )
        deadline, source = current + timedelta(seconds=seconds), "provider_relative"
    calendar = re.search(
        r"try again at ([A-Za-z]{3}) (\d{1,2})(?:st|nd|rd|th)?, (\d{4}) (\d{1,2}):(\d{2}) (AM|PM)",
        message,
        re.IGNORECASE,
    )
    if deadline is None and calendar:
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        try:
            month, day, year, hour, minute, ampm = calendar.groups()
            hour = int(hour) % 12 + (12 if ampm.upper() == "PM" else 0)
            zone = local_timezone or datetime.now().astimezone().tzinfo
            deadline = datetime(
                int(year), months.index(month.title()) + 1, int(day), hour, int(minute), tzinfo=zone
            ).astimezone(UTC)
            source = f"cli_local_calendar:{zone}"
        except (ValueError, OverflowError):
            pass
    if deadline is None or deadline <= current:
        deadline, source = current + timedelta(hours=1), "fallback_1h"
    return RunnerDeferred(deadline.astimezone(UTC).isoformat(), "quota", source)


def http_deferred(error, *, current: datetime | None = None):
    if getattr(error, "status_code", None) != 429:
        return None
    current = current or datetime.now(UTC)
    headers = getattr(getattr(error, "response", None), "headers", {})
    value = headers.get("retry-after")
    deadline, source = current + timedelta(minutes=1), "http429_fallback_1m"
    if value:
        try:
            deadline = current + timedelta(seconds=max(1, float(value)))
            source = "retry_after_seconds"
        except (ValueError, OverflowError):
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is not None and parsed > current:
                    deadline, source = parsed, "retry_after_date"
            except (ValueError, TypeError, OverflowError):
                pass
    return RunnerDeferred(deadline.astimezone(UTC).isoformat(), "http429", source)
