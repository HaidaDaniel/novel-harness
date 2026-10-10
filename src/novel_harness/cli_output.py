"""Human CLI rendering kept separate from the stable JSON contract."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_TABLE_COLUMNS = {
    "novels": ("novel_id", "title", "chapter_count", "translation_state", "progress"),
    "runners": ("id", "kind", "available", "model", "reason"),
    "presets": ("id", "label", "runner", "pipeline", "description"),
    "profiles": ("id", "kind", "available", "model_policy", "resolved_model"),
}
_SECRET_VALUE = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|password|secret|authorization)\b\s*[:=]\s*)([^\s,;]+)"
)
_AUTHORIZATION_BEARER = re.compile(r"(?i)(\bauthorization\b\s*[:=]\s*bearer\s+)([^\s,;]+)")
_BEARER_VALUE = re.compile(r"(?i)(\bbearer\s+)([^\s,;]+)")


def _clip(value: str, limit: int = 120) -> str:
    value = value.replace("\n", " ").replace("\r", " ")
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _clip_multiline(value: str, limit: int = 4000) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return _clip(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    if value is None:
        return "-"
    return _clip(str(value))


def _table(rows: list[dict[str, Any]], preferred: tuple[str, ...] = ()) -> str:
    if not rows:
        return "No entries."
    columns = [column for column in preferred if any(column in row for row in rows)]
    if not columns:
        columns = list(rows[0])[:5]
    labels = [column.replace("_", " ").title() for column in columns]
    values = [[_value(row.get(column)) for column in columns] for row in rows]
    widths = [max(len(label), *(len(row[index]) for row in values)) for index, label in enumerate(labels)]
    lines = [" | ".join(label.ljust(width) for label, width in zip(labels, widths))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in values)
    return "\n".join(lines)


def render(value: Any) -> str:
    """Render a bounded human summary without terminal control sequences."""
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return _table(value)
    if not isinstance(value, Mapping):
        return _value(value)
    lines = []
    for key, item in value.items():
        label = str(key).replace("_", " ").capitalize()
        if key in _TABLE_COLUMNS and isinstance(item, list) and all(isinstance(row, dict) for row in item):
            lines.extend((f"{label}:", _table(item, _TABLE_COLUMNS[key])))
        elif key == "pause_requested" and item:
            lines.append("Pause requested; it will take effect at the next chunk boundary.")
        elif key == "started" and item:
            lines.append("Worker start accepted; translation continues in the background.")
        elif key == "already_running" and item:
            lines.append("Worker is already running; no second worker was started.")
        elif key == "completed" and item:
            lines.append("Run is already complete; no worker was started.")
        elif key == "error" and item:
            lines.append(f"Error: {_value(item)}")
        elif isinstance(item, str) and "\n" in item:
            lines.extend((f"{label}:", _clip_multiline(item)))
        else:
            lines.append(f"{label}: {_value(item)}")
    return "\n".join(lines)


def sanitize_error(error: object, *, limit: int = 240) -> str:
    """Return a bounded error message without credential-shaped values."""
    def redact_secret(match: re.Match[str]) -> str:
        value = match.group(2)
        if value.casefold() == "bearer":
            return f"{match.group(1)}Bearer <redacted>"
        return f"{match.group(1)}<redacted>"

    message = _AUTHORIZATION_BEARER.sub(r"\1<redacted>", str(error))
    message = _SECRET_VALUE.sub(redact_secret, message)
    message = _BEARER_VALUE.sub(r"\1<redacted>", message)
    return _clip(message, limit)


def render_error(error: Exception) -> str:
    return f"Error: {sanitize_error(error)}\nTry the command with --help for usage."
