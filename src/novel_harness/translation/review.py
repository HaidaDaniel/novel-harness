"""Minimal, source-addressed review patches with compare-and-set preconditions."""

import json
import re

from ..artifacts import sha256
from .segments import OutputContractError
from .validator import foreign_script_count

CATEGORIES = [
    "omission",
    "meaning",
    "speaker",
    "negation",
    "gender",
    "terminology",
    "style",
    "format",
]

# ``semantic-edit-v1`` is a distinct, explicitly versioned contract for the
# bounded same-model source-aware semantic edit. It is not generalized
# ``review-v2``: style and format are not separate categories and style-only
# rewriting is not a target. Categories follow the frozen priority order
# (meaning; omission; negation; actor/patient; speaker; gender;
# terminology/consistency).
SEMANTIC_EDIT_CATEGORIES = [
    "meaning",
    "omission",
    "negation",
    "actor_patient",
    "speaker",
    "gender",
    "terminology",
    "consistency",
]

# Versioned review contracts. ``review-v1`` is the frozen legacy contract: an
# ``issues`` array whose ``replacement`` may be ``null`` (the host then rejects
# it as ``review_unresolved``). ``review-v2`` is the unified contract: fixable
# defects carry a mandatory non-empty replacement; defects the reviewer cannot
# fix are reported in ``unresolved`` and are accepted as open defects instead of
# being discarded, hallucinated or retried to the recovery ceiling.
# ``semantic-edit-v1`` uses the same unified patch shape with its own semantic
# category set and is reported as the distinct ``semantic_edited`` layer.
REVIEW_V1 = "review-v1"
REVIEW_V2 = "review-v2"
SEMANTIC_EDIT_V1 = "semantic-edit-v1"
REVIEW_CONTRACTS = (REVIEW_V1, REVIEW_V2, SEMANTIC_EDIT_V1)
# Contracts that carry ``unresolved`` and forbid a null replacement.
UNIFIED_REVIEW_CONTRACTS = (REVIEW_V2, SEMANTIC_EDIT_V1)


def review_categories(contract: str) -> list[str]:
    """Return the frozen category set for a versioned review contract."""
    if contract not in REVIEW_CONTRACTS:
        raise ValueError(f"Unknown review contract: {contract!r}")
    return list(SEMANTIC_EDIT_CATEGORIES if contract == SEMANTIC_EDIT_V1 else CATEGORIES)


def review_contract(prompt_version: str | None, explicit: str | None = None) -> str:
    """Resolve a review contract from a frozen stage identity.

    An explicit frozen ``explicit`` contract always wins. Otherwise the prompt
    version identifies the contract; any unknown or missing version stays on the
    legacy contract so already-frozen runs cannot silently switch.
    """
    if explicit is not None:
        if explicit not in REVIEW_CONTRACTS:
            raise ValueError(f"Unknown review contract: {explicit!r}")
        return explicit
    if prompt_version in (REVIEW_V2, SEMANTIC_EDIT_V1):
        return prompt_version
    return REVIEW_V1

# Deterministic, meaning-free rejection of patches that the GBNF shape cannot
# rule out. The grammar guarantees well-formed JSON, single-line text and the
# exact patch ID order; it cannot guarantee that a replacement is a translation.
# These guards reject observed non-translations (echoed IDs, JSON sentinels and
# review meta-text) and impossible Russian output (no Cyrillic, Japanese script).
_PATCH_ID = re.compile(r"^p\d{4}-[0-9a-f]{8}$")
_SENTINELS = frozenset({"null", "none", "n/a", "na", "nil", "undefined", "-"})
_META_TEXT = re.compile(
    r"^\s*[\(\[]?\s*(?:no\s+change|no\s+issues?|acceptable|unchanged|checking|"
    r"source\s*:|draft\s*:|ok\b|pass\b|looks\s+good)",
    re.IGNORECASE,
)
_CYRILLIC_LETTER = re.compile(r"[\u0400-\u04ff]")
# A short segment can legitimately condense during correction, so content-loss
# detection only applies to long segments where a large drop proves deletion.
_TRUNCATION_MIN_CHARS = 120
_TRUNCATION_RATIO = 0.6


def patch_text_error(text: str, original: str, segment_ids: set[str]) -> str | None:
    """Return a rejection code for a non-translation patch, else ``None``."""
    stripped = text.strip()
    if stripped in segment_ids or _PATCH_ID.fullmatch(stripped):
        return "review_patch_echoes_id"
    if stripped.lower() in _SENTINELS:
        return "review_patch_sentinel"
    if _META_TEXT.match(stripped):
        return "review_patch_meta"
    if foreign_script_count(stripped):
        return "review_patch_foreign_script"
    if _CYRILLIC_LETTER.search(original) and not _CYRILLIC_LETTER.search(stripped):
        return "review_patch_not_russian"
    if (
        len(original.strip()) >= _TRUNCATION_MIN_CHARS
        and len(stripped) < _TRUNCATION_RATIO * len(original.strip())
    ):
        return "review_patch_truncated"
    return None


def review_schema(segments: list[dict], contract: str = REVIEW_V1) -> dict:
    if contract not in REVIEW_CONTRACTS:
        raise ValueError(f"Unknown review contract: {contract!r}")
    categories = review_categories(contract)
    ids = [s["id"] for s in segments]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["checked_ids", "issues"],
        "properties": {
            "checked_ids": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {"type": "string", "enum": ids},
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "category", "reason", "original_sha256", "replacement"],
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "category": {"type": "string", "enum": categories},
                        "reason": {"type": "string"},
                        "original_sha256": {"type": "string"},
                        "replacement": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }
    if contract in UNIFIED_REVIEW_CONTRACTS:
        # Fixable defects must carry a concrete replacement; the schema no longer
        # permits null anywhere.
        schema["required"] = ["checked_ids", "issues", "unresolved"]
        schema["properties"]["issues"]["items"]["properties"]["replacement"] = {"type": "string"}
        schema["properties"]["unresolved"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "category", "reason"],
                "properties": {
                    "id": {"type": "string", "enum": ids},
                    "category": {"type": "string", "enum": categories},
                    "reason": {"type": "string"},
                },
            },
        }
    return schema


def apply_review(
    raw: str | None, originals: list[dict], draft: list[dict], contract: str = REVIEW_V1
) -> tuple[dict, list[dict]]:
    if contract not in REVIEW_CONTRACTS:
        raise ValueError(f"Unknown review contract: {contract!r}")
    categories = review_categories(contract)
    try:
        report = json.loads(raw or "")
    except (ValueError, TypeError) as error:
        raise OutputContractError("invalid_review_json") from error
    ids = [s["id"] for s in originals]
    expected_keys = (
        {"checked_ids", "issues"}
        if contract == REVIEW_V1
        else {"checked_ids", "issues", "unresolved"}
    )
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise OutputContractError("invalid_review_envelope")
    if report["checked_ids"] != ids or [s["id"] for s in draft] != ids:
        raise OutputContractError("review_coverage_mismatch")
    if not isinstance(report["issues"], list):
        raise OutputContractError("invalid_review_issues")
    values = {s["id"]: s["text"] for s in draft}
    segment_ids = set(values)
    changed = set()
    for issue in report["issues"]:
        if not isinstance(issue, dict) or set(issue) != {
            "id",
            "category",
            "reason",
            "original_sha256",
            "replacement",
        }:
            raise OutputContractError("invalid_review_issue")
        key = issue["id"]
        if not isinstance(key, str) or key not in values or key in changed:
            raise OutputContractError("invalid_or_duplicate_patch_id")
        if issue["original_sha256"] != sha256(values[key]):
            raise OutputContractError("stale_review_patch")
        if (
            issue["category"] not in categories
            or not isinstance(issue["reason"], str)
            or not issue["reason"].strip()
        ):
            raise OutputContractError("review_reason_required")
        text = issue["replacement"]
        if text is None:
            # review-v1 keeps the legacy rejection code; the unified contracts
            # forbid a null replacement outright because unresolved defects
            # belong in the separate unresolved array.
            if contract in UNIFIED_REVIEW_CONTRACTS:
                raise OutputContractError("review_issue_requires_replacement")
            raise OutputContractError("review_unresolved")
        if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text:
            raise OutputContractError("invalid_patch_text")
        rejection = patch_text_error(text, values[key], segment_ids)
        if rejection is not None:
            raise OutputContractError(rejection)
        values[key] = text
        changed.add(key)
    if contract in UNIFIED_REVIEW_CONTRACTS:
        unresolved = report["unresolved"]
        if not isinstance(unresolved, list):
            raise OutputContractError("invalid_review_unresolved")
        for item in unresolved:
            if not isinstance(item, dict) or set(item) != {"id", "category", "reason"}:
                raise OutputContractError("invalid_review_unresolved")
            key = item["id"]
            if not isinstance(key, str) or key not in values or key in changed:
                raise OutputContractError("invalid_or_duplicate_patch_id")
            if (
                item["category"] not in categories
                or not isinstance(item["reason"], str)
                or not item["reason"].strip()
            ):
                raise OutputContractError("review_reason_required")
            # An unresolved defect is reported and counted, but the segment keeps
            # its draft text. It is never treated as corrected.
            changed.add(key)
    return report, [{"id": s["id"], "text": values[s["id"]]} for s in draft]
