"""Bounded semantic-plan-v1: explicit current-chunk fields only, no free-form reasoning.

The plan is a small, host-validated structure (actor/action/object, polarity,
question/statement, speaker/reference, terminology, unresolved ambiguity). It is
built from the current chunk only and is injected into the translation request as
bounded context. It is never a chain-of-thought transcript and carries no chapter
numbers, hashes, offsets, quotes or dependency fields.
"""

from __future__ import annotations

import json

from ..artifacts import canonical, sha256

PLAN_VERSION = "semantic-plan-v1"

POLARITY_VALUES = ("affirmative", "negative", "unknown")
MOOD_VALUES = ("statement", "question", "command", "unknown")

#: The only permitted item fields. Any other key is rejected, so a model cannot
#: smuggle free-form reasoning or unbounded text into the artifact.
PLAN_ITEM_FIELDS = (
    "actor",
    "action",
    "object",
    "polarity",
    "mood",
    "speaker",
    "reference",
    "terminology",
    "unresolved",
)

_STRING_MAX = 120
_UNRESOLVED_MAX = 160
_MAX_ITEMS = 24
_MAX_TERMS = 8
_MAX_UNRESOLVED = 8


class PlanError(ValueError):
    """The model response is not a valid bounded semantic plan."""


def planner_context(ordinary_context: dict) -> dict:
    """Bounded planner input: current chunk only.

    The chunk source and glossary are already separate ``TranslationJob`` fields, so
    the planner context is empty. Previous source/translation, generated book context
    and any previous semantic plan are deliberately excluded, matching the frozen
    Candidate B hypothesis ("semantic analysis of the current chunk only").
    """
    return {}


def plan_schema() -> dict:
    """JSON schema for the bounded plan; used for structured decoding."""
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "actor": {"type": "string", "maxLength": _STRING_MAX},
            "action": {"type": "string", "maxLength": _STRING_MAX},
            "object": {"type": "string", "maxLength": _STRING_MAX},
            "polarity": {"enum": list(POLARITY_VALUES)},
            "mood": {"enum": list(MOOD_VALUES)},
            "speaker": {"type": "string", "maxLength": _STRING_MAX},
            "reference": {"type": "string", "maxLength": _STRING_MAX},
            "terminology": {
                "type": "array",
                "maxItems": _MAX_TERMS,
                "items": {"type": "string", "maxLength": _STRING_MAX},
            },
            "unresolved": {
                "type": "array",
                "maxItems": _MAX_UNRESOLVED,
                "items": {"type": "string", "maxLength": _UNRESOLVED_MAX},
            },
        },
        "required": ["polarity", "mood"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "plan_version": {"const": PLAN_VERSION},
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": _MAX_ITEMS,
                "items": item,
            },
        },
        "required": ["plan_version", "items"],
    }


def _bounded_string(value, *, maximum: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"plan field {field!r} must be a non-empty string")
    if len(value) > maximum:
        raise PlanError(f"plan field {field!r} exceeds {maximum} characters")
    if "\ufffd" in value or "??" in value or any(ord(c) < 32 for c in value):
        raise PlanError(f"plan field {field!r} contains damaged text")
    return value.strip()


def parse_plan(text: str | None) -> dict:
    """Strictly parse a model response into the bounded plan or raise PlanError."""
    if not isinstance(text, str) or not text.strip():
        raise PlanError("empty semantic-plan response")
    try:
        raw = json.loads(text)
    except ValueError as error:
        raise PlanError(f"semantic plan is not valid JSON ({error})") from error
    if not isinstance(raw, dict) or set(raw) != {"plan_version", "items"}:
        raise PlanError("semantic plan must contain exactly plan_version and items")
    if raw["plan_version"] != PLAN_VERSION:
        raise PlanError(f"unsupported plan_version: {raw['plan_version']!r}")
    items = raw["items"]
    if not isinstance(items, list) or not 1 <= len(items) <= _MAX_ITEMS:
        raise PlanError(f"plan items must be a list of 1..{_MAX_ITEMS}")
    plan_items = []
    for item in items:
        if not isinstance(item, dict) or set(item) - set(PLAN_ITEM_FIELDS):
            raise PlanError("plan item contains unknown fields")
        if "polarity" not in item or "mood" not in item:
            raise PlanError("plan item requires polarity and mood")
        if item["polarity"] not in POLARITY_VALUES:
            raise PlanError(f"invalid polarity: {item['polarity']!r}")
        if item["mood"] not in MOOD_VALUES:
            raise PlanError(f"invalid mood: {item['mood']!r}")
        clean: dict = {"polarity": item["polarity"], "mood": item["mood"]}
        # Optional fields: an omitted, null or blank value is dropped rather than
        # rejected (models commonly emit ""); a non-string or over-length value is
        # still rejected, so the artifact stays bounded and free-form-free.
        for field in ("actor", "action", "object", "speaker", "reference"):
            if field in item:
                value = item[field]
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                clean[field] = _bounded_string(value, maximum=_STRING_MAX, field=field)
        for field, maximum, cap in (
            ("terminology", _STRING_MAX, _MAX_TERMS),
            ("unresolved", _UNRESOLVED_MAX, _MAX_UNRESOLVED),
        ):
            if field in item:
                values = item[field]
                if values is None:
                    continue
                if not isinstance(values, list) or len(values) > cap:
                    raise PlanError(f"plan field {field!r} must be a bounded list")
                cleaned = [
                    _bounded_string(v, maximum=maximum, field=field)
                    for v in values
                    if isinstance(v, str) and v.strip()
                ]
                if cleaned:
                    clean[field] = cleaned
        plan_items.append(clean)
    return {"plan_version": PLAN_VERSION, "items": plan_items}


def render_plan(plan: dict) -> str:
    """Deterministic bounded text block injected into the translation context."""
    lines = [f"semantic_plan ({PLAN_VERSION}); current chunk only; explicit fields:"]
    for index, item in enumerate(plan["items"], start=1):
        parts = [
            f"polarity={item['polarity']}",
            f"mood={item['mood']}",
        ]
        for field in ("actor", "action", "object", "speaker", "reference"):
            if field in item:
                parts.append(f"{field}={item[field]}")
        if item.get("terminology"):
            parts.append("terminology=" + "|".join(item["terminology"]))
        if item.get("unresolved"):
            parts.append("unresolved=" + "|".join(item["unresolved"]))
        lines.append(f"{index}. " + " ".join(parts))
    return "\n".join(lines)


def plan_sha256(plan: dict) -> str:
    return sha256(canonical(plan))
