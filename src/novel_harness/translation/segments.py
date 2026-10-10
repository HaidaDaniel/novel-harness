"""Source-addressed output contract. Coverage is structural, not proof of meaning."""

import json
import re

from ..artifacts import sha256


class OutputContractError(ValueError):
    pass


def source_segments(source: str, *, heading: bool = False) -> list[dict]:
    result = []
    for match in re.finditer(r"[^\r\n]+", source):
        if not match[0].strip():
            continue
        result.append(
            {
                "id": f"p{len(result) + 1:04d}-{sha256(match[0])[:8]}",
                "source": match[0],
                "kind": "heading" if heading and not result else "body",
                "start": match.start(),
                "end": match.end(),
            }
        )
    return result


def translation_schema(segments: list[dict]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["segments"],
        "properties": {
            "segments": {
                "type": "array",
                "minItems": len(segments),
                "maxItems": len(segments),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text"],
                    "properties": {
                        "id": {"type": "string", "enum": [s["id"] for s in segments]},
                        "text": {"type": "string"},
                    },
                },
            }
        },
    }


def decode_translation(raw: str | None, segments: list[dict]) -> list[dict]:
    try:
        data = json.loads(raw or "")
    except (ValueError, TypeError) as error:
        raise OutputContractError("invalid_translation_json") from error
    if not isinstance(data, dict) or set(data) != {"segments"}:
        raise OutputContractError("invalid_translation_envelope")
    result = data["segments"]
    if not isinstance(result, list) or any(
        not isinstance(s, dict) or set(s) != {"id", "text"} for s in result
    ):
        raise OutputContractError("invalid_translation_segments")
    if [s["id"] for s in result] != [s["id"] for s in segments]:
        raise OutputContractError("source_coverage_mismatch")
    if any(
        not isinstance(s["text"], str)
        or not s["text"].strip()
        or "\n" in s["text"]
        or "\r" in s["text"]
        for s in result
    ):
        raise OutputContractError("empty_or_multiline_segment")
    return result


def render_segments(source: str, segments: list[dict], translations: list[dict]) -> str:
    pieces, end = [], 0
    for original, translated in zip(segments, translations, strict=True):
        pieces.extend([source[end : original["start"]], translated["text"].strip()])
        end = original["end"]
    pieces.append(source[end:])
    return "".join(pieces)
