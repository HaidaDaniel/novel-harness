"""Verified positional mapping and lossless rendering of an immutable draft."""

import json
import re

from ..artifacts import sha256
from .segments import OutputContractError, decode_translation, render_segments, source_segments


def freeze_draft(source: str, draft: str, saved_segments=None, *, heading=False) -> dict:
    originals = source_segments(source, heading=heading)
    lines = [m for m in re.finditer(r"[^\r\n]+", draft) if m[0].strip()]
    if len(lines) != len(originals):
        raise OutputContractError("direct_draft_mapping_mismatch")
    if saved_segments is not None:
        segments = decode_translation(json.dumps({"segments": saved_segments}), originals)
        if render_segments(source, originals, segments) != draft:
            raise OutputContractError("direct_draft_segments_hash_mismatch")
        kind = "segments-v1"
    else:
        segments = [
            {"id": s["id"], "text": m[0].strip()} for s, m in zip(originals, lines, strict=True)
        ]
        kind = "text-lines-v1"
    spans = []
    for segment, match in zip(segments, lines, strict=True):
        if segment["text"].strip() != match[0].strip():
            raise OutputContractError("direct_draft_mapping_mismatch")
        left = len(match[0]) - len(match[0].lstrip())
        right = len(match[0].rstrip())
        spans.append(
            {"id": segment["id"], "start": match.start() + left, "end": match.start() + right}
        )
    return {
        "version": 1,
        "kind": kind,
        "source_sha256": sha256(source),
        "draft_sha256": sha256(draft),
        "segments": segments,
        "spans": spans,
    }


def verify_mapping(source: str, draft: str, mapping: dict, *, heading=False):
    if mapping.get("version") != 1 or mapping.get("kind") not in {"segments-v1", "text-lines-v1"}:
        raise OutputContractError("unsupported_frozen_draft_mapping")
    saved = mapping["segments"] if mapping["kind"] == "segments-v1" else None
    if freeze_draft(source, draft, saved, heading=heading) != mapping:
        raise OutputContractError("frozen_draft_mapping_changed")


def render_frozen(draft: str, mapping: dict, translated: list[dict]) -> str:
    if sha256(draft) != mapping["draft_sha256"]:
        raise OutputContractError("frozen_draft_hash_mismatch")
    if [s["id"] for s in translated] != [s["id"] for s in mapping["segments"]]:
        raise OutputContractError("review_coverage_mismatch")
    pieces, end = [], 0
    for original, span, replacement in zip(
        mapping["segments"], mapping["spans"], translated, strict=True
    ):
        pieces.append(draft[end : span["start"]])
        pieces.append(
            draft[span["start"] : span["end"]]
            if replacement == original
            else replacement["text"].strip()
        )
        end = span["end"]
    pieces.append(draft[end:])
    return "".join(pieces)
