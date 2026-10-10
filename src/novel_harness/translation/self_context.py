"""Frozen source-only self-context-lite artifacts.

A candidate model may derive compact context for itself from Japanese source and
previously frozen source-derived context. Model claims carry a source anchor and a
semantic class; the anchor proves quote existence only, never entailment. This
module defines the strict schema, deterministic entry IDs and relevance selection,
fail-closed budgets, and the freeze/verify path with recursive prior provenance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ..artifacts import canonical, sha256
from ..config import StrictModel
from .discovery import discover

VERSION = "self-context-lite-v1"
CONTEXT_RESPONSE_VERSION = "context-generation-v4"
SOURCE_ADDRESSING_VERSION = "quote-ref-addressing-v1"
QUOTE_UNIT_SEGMENTATION_VERSION = "quote-unit-segmentation-v1"
# Translation-consumption budget policy: bound the exact rendered projection, not
# the canonical selected-entry JSON. See `select_relevant`.
CONSUMPTION_BUDGET_POLICY = "rendered-context-budget-v1"
SOURCE_REFS = ("current_chapter", "previous_window")
SOURCE_REF_PREFIX = {"current_chapter": "cur", "previous_window": "prev"}
QUOTE_UNIT_MAX_CHARS = 200
SENTENCE_ENDS = "。！？!?．"
SENTENCE_CLOSERS = "」』）)】〉》｝}”’\"'"
PRIOR_PROJECTION_VERSION = "prior-context-projection-v1"
TOKEN_ESTIMATOR = "utf8-codepoints-div-2-ceil-v1"

MAX_SNAPSHOTS = 100
MAX_ENTRIES_PER_SNAPSHOT = 64
# Generation policy for the intermediate context-generation-v4 response only.
# The final self-context-lite-v1 artifact keeps its independent 64-entry ceiling.
MAX_GENERATED_CONTEXT_ENTRIES = 12
MAX_SNAPSHOT_CHARS = 24000
MAX_SELECTED_ENTRIES = 64
MAX_CONTEXT_CHARS = 30000
MAX_ESTIMATED_CONTEXT_TOKENS = 15000
MAX_CONTINUITY_WINDOW_CHARS = 4000
MAX_ANCHORS = 6
MAX_SOURCE_FORMS = 12
SET_FIELDS = ("source_forms", "source_names", "source_terms", "aliases", "participant_ids")

ENTRY_MODELS = {
    "character": "CharacterEntry",
    "relationship": "RelationshipEntry",
    "term": "TermEntry",
    "speech": "SpeechEntry",
    "continuity": "ContinuityEntry",
    "chapter_preflight_fact": "ContinuityEntry",
}


def default_budget() -> dict:
    return {
        "max_entries_per_snapshot": MAX_ENTRIES_PER_SNAPSHOT,
        "max_snapshot_chars": 24000,
        "max_selected_entries": 24,
        "max_context_chars": 4000,
        "token_estimator": TOKEN_ESTIMATOR,
        "chars_per_token": 2.0,
        "max_estimated_context_tokens": 2000,
        "continuity_window_chars": 800,
    }


def estimate_tokens(text: str, chars_per_token: float = 2.0) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return math.ceil(len(text) / chars_per_token)


Hex64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Anchor(StrictModel):
    chapter: int = Field(gt=0)
    source_sha256: Hex64
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=500)
    quote_sha256: Hex64

    @model_validator(mode="after")
    def coherent(self):
        if self.end <= self.start:
            raise ValueError("anchor end must exceed start")
        if len(self.quote) != self.end - self.start:
            raise ValueError("anchor quote length must equal end - start")
        if sha256(self.quote) != self.quote_sha256:
            raise ValueError("anchor quote_sha256 mismatch")
        return self


class _EntryBase(StrictModel):
    id: str
    semantic_class: Literal["source_anchored_claim", "model_hypothesis", "unresolved"]
    source_forms: list[str] = Field(min_length=1, max_length=MAX_SOURCE_FORMS)
    anchors: list[Anchor] = Field(min_length=1, max_length=MAX_ANCHORS)
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def bounds(self):
        if len(set(self.source_forms)) != len(self.source_forms):
            raise ValueError("duplicate source_forms")
        for form in self.source_forms:
            if not form or len(form) > 80 or form != form.strip():
                raise ValueError("invalid source form")
        if len({(a.chapter, a.start, a.end, a.quote_sha256) for a in self.anchors}) != len(
            self.anchors
        ):
            raise ValueError("duplicate anchors")
        if self.semantic_class == "source_anchored_claim":
            if self.confidence is not None or self.uncertainty is not None:
                raise ValueError("anchored claims carry no confidence/uncertainty")
        elif self.confidence is None or not self.uncertainty:
            raise ValueError("hypothesis/unresolved require confidence and uncertainty")
        return self


class CharacterEntry(_EntryBase):
    kind: Literal["character"]
    claim: str = Field(min_length=1, max_length=400)
    source_names: list[str] = Field(min_length=1, max_length=8)
    rendering: str | None = Field(default=None, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=12)
    gender: Literal["female", "male", "unknown"] = "unknown"

    @model_validator(mode="after")
    def names(self):
        for name in [*self.source_names, *self.aliases]:
            if not name or len(name) > 80 or name != name.strip():
                raise ValueError("invalid character name/alias")
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError("duplicate source_names")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("duplicate aliases")
        return self


class RelationshipEntry(_EntryBase):
    kind: Literal["relationship"]
    subject_id: str
    object_id: str
    relation: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def distinct_endpoints(self):
        if self.subject_id == self.object_id:
            raise ValueError("relationship endpoints must be distinct")
        return self


class TermEntry(_EntryBase):
    kind: Literal["term"]
    source_terms: list[str] = Field(min_length=1, max_length=12)
    rendering: str | None = Field(default=None, max_length=160)
    meaning: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def terms(self):
        for term in self.source_terms:
            if not term or len(term) > 80 or term != term.strip():
                raise ValueError("invalid source term")
        if len(set(self.source_terms)) != len(self.source_terms):
            raise ValueError("duplicate source_terms")
        return self


class SpeechEntry(_EntryBase):
    kind: Literal["speech"]
    speaker_id: str
    pattern: str = Field(min_length=1, max_length=200)


class ContinuityEntry(_EntryBase):
    kind: Literal["continuity", "chapter_preflight_fact"]
    participant_ids: list[str] = Field(default_factory=list, max_length=8)
    claim: str = Field(min_length=1, max_length=300)


Entry = Annotated[
    CharacterEntry | RelationshipEntry | TermEntry | SpeechEntry | ContinuityEntry,
    Field(discriminator="kind"),
]


# Host-owned exact quote unit ids: `<source_ref prefix>-<4 digits>`.
WireQuoteRef = Annotated[
    str, Field(min_length=1, max_length=24, pattern=r"^(cur|prev)-[0-9]{4}$")
]


class AnchoredSemantic(StrictModel):
    """The cited quote supports the claim; certainty fields are structurally absent."""

    semantic_class: Literal["source_anchored_claim"] = Field(alias="class")


class HypothesisSemantic(StrictModel):
    """A model inference; confidence and uncertainty are structurally required."""

    semantic_class: Literal["model_hypothesis"] = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    uncertainty: str = Field(min_length=1, max_length=300)


class UnresolvedSemantic(StrictModel):
    """The source does not answer the question; confidence and uncertainty are required."""

    semantic_class: Literal["unresolved"] = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    uncertainty: str = Field(min_length=1, max_length=300)


Semantic = Annotated[
    AnchoredSemantic | HypothesisSemantic | UnresolvedSemantic,
    Field(discriminator="semantic_class"),
]

# Explicit negated character class (no `\s`/`\S`): the llama.cpp JSON-schema->GBNF
# converter only accepts anchored patterns built from literal classes.
WireForm = Annotated[
    str, Field(min_length=1, max_length=80, pattern=r"^[^ \t\r\n](.*[^ \t\r\n])?$")
]
WireRef = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")]
UNIQUE_FORMS = {"uniqueItems": True}


def _unique_forms(min_length: int, max_length: int) -> Any:
    return Field(min_length=min_length, max_length=max_length, json_schema_extra=UNIQUE_FORMS)


class _WireEntryBase(StrictModel):
    """Host-independent wire fields shared by every kind.

    The wire shape is a discriminated union by ``kind`` and a nested discriminated
    union by semantic ``class``, so every static constraint is encoded in JSON Schema
    and a constrained decoder cannot emit a shape the host rejects.
    """

    local_key: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    semantic: Semantic
    source_forms: list[WireForm] = _unique_forms(1, MAX_SOURCE_FORMS)
    # The model cites host-supplied exact quote units by id; it never emits text,
    # chapter numbers, offsets or hashes. Duplicates are schema-rejected.
    anchor_refs: list[WireQuoteRef] = Field(
        min_length=1, max_length=3, json_schema_extra=UNIQUE_FORMS
    )


class CharacterWireEntry(_WireEntryBase):
    kind: Literal["character"]
    claim: str = Field(min_length=1, max_length=400)
    source_names: list[WireForm] = _unique_forms(1, 8)
    aliases: list[WireForm] = Field(
        default_factory=list, max_length=12, json_schema_extra=UNIQUE_FORMS
    )
    rendering: str | None = Field(default=None, max_length=160)
    gender: Literal["female", "male", "unknown"] = "unknown"


class RelationshipWireEntry(_WireEntryBase):
    kind: Literal["relationship"]
    relation: str = Field(min_length=1, max_length=200)
    subject_ref: WireRef
    object_ref: WireRef


class TermWireEntry(_WireEntryBase):
    kind: Literal["term"]
    meaning: str = Field(min_length=1, max_length=300)
    source_terms: list[WireForm] = _unique_forms(1, 12)
    rendering: str | None = Field(default=None, max_length=160)


class SpeechWireEntry(_WireEntryBase):
    kind: Literal["speech"]
    pattern: str = Field(min_length=1, max_length=200)
    speaker_ref: WireRef


class ContinuityWireEntry(_WireEntryBase):
    kind: Literal["continuity"]
    claim: str = Field(min_length=1, max_length=300)
    participant_refs: list[WireRef] = Field(
        default_factory=list, max_length=8, json_schema_extra=UNIQUE_FORMS
    )


class ChapterPreflightWireEntry(_WireEntryBase):
    kind: Literal["chapter_preflight_fact"]
    claim: str = Field(min_length=1, max_length=300)
    participant_refs: list[WireRef] = Field(
        default_factory=list, max_length=8, json_schema_extra=UNIQUE_FORMS
    )


WireEntry = Annotated[
    CharacterWireEntry
    | RelationshipWireEntry
    | TermWireEntry
    | SpeechWireEntry
    | ContinuityWireEntry
    | ChapterPreflightWireEntry,
    Field(discriminator="kind"),
]


class ModelResponse(StrictModel):
    # Generation policy: cap the intermediate response, not the frozen artifact.
    entries: list[WireEntry] = Field(max_length=MAX_GENERATED_CONTEXT_ENTRIES)


def context_response_schema() -> dict:
    """Strict JSON Schema for the intermediate model-owned response (v4)."""
    schema = ModelResponse.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


class StructureChapter(StrictModel):
    chapter: int = Field(gt=0)
    filename: str
    source_sha256: Hex64


class JapaneseSource(StrictModel):
    kind: Literal["japanese_source"]
    chapter: int = Field(gt=0)
    filename: str
    source_sha256: Hex64
    normalized_text_sha256: Hex64
    scope: Literal["full_chapter"]


class SourceWindow(StrictModel):
    kind: Literal["source_window"]
    chapter: int = Field(gt=0)
    filename: str
    source_sha256: Hex64
    normalized_text_sha256: Hex64
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: Hex64

    @model_validator(mode="after")
    def coherent(self):
        if self.end <= self.start:
            raise ValueError("source window end must exceed start")
        return self


class PriorSelfContext(StrictModel):
    kind: Literal["prior_self_context"]
    artifact_path: str
    artifact_sha256: Hex64
    version: Literal["self-context-lite-v1"]
    context_origin: Literal["self"]
    generator_identity_sha256: Hex64
    dependency_closure_sha256: Hex64
    max_source_chapter: int = Field(ge=0)
    # Provenance of the bounded semantic projection actually sent to the model.
    # The path above is provenance only; the model never reads it.
    projection: PriorProjection | None = None


class SourceStructure(StrictModel):
    kind: Literal["source_structure"]
    source_title: str
    chapters: list[StructureChapter] = Field(min_length=1, max_length=100)
    structure_sha256: Hex64


ContextInput = Annotated[
    JapaneseSource | SourceWindow | PriorSelfContext | SourceStructure,
    Field(discriminator="kind"),
]


class PriorProjection(StrictModel):
    selection: str
    projection_sha256: Hex64
    entry_count: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)


class GenerationCall(StrictModel):
    call_id: str
    attempt: int = Field(gt=0)
    status: Literal["completed", "rejected", "failed"]
    request_sha256: Hex64
    response_sha256: Hex64
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    wall_seconds: float = Field(ge=0)
    source_chars: int = Field(ge=0)
    finish_reason: str | None = None
    failure_kind: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def tokens(self):
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError("token counts must be all null or all integers")
        return self


class Snapshot(StrictModel):
    target_chapter: int = Field(gt=0)
    empty_reason: Literal["no_prior_source"] | None = None
    inputs: list[ContextInput] = Field(max_length=4)
    entries: list[Entry] = Field(max_length=MAX_ENTRIES_PER_SNAPSHOT)
    generation_calls: list[GenerationCall] = Field(default_factory=list)
    accepted_call_id: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.entries:
            if self.empty_reason is not None:
                raise ValueError("non-empty snapshot cannot carry empty_reason")
        elif self.empty_reason != "no_prior_source":
            raise ValueError("empty snapshot requires empty_reason: no_prior_source")
        ids = [e.id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate entry id")
        for entry in self.entries:
            if derive_entry_id(entry.model_dump()) != entry.id:
                raise ValueError("entry id mismatch")
        by_id = {e.id: e for e in self.entries}
        for entry in self.entries:
            if isinstance(entry, RelationshipEntry):
                for ref in (entry.subject_id, entry.object_id):
                    if ref not in by_id or not isinstance(by_id[ref], CharacterEntry):
                        raise ValueError("dangling relationship endpoint")
            if isinstance(entry, SpeechEntry) and (
                entry.speaker_id not in by_id
                or not isinstance(by_id[entry.speaker_id], CharacterEntry)
            ):
                raise ValueError("dangling speech speaker")
            if isinstance(entry, ContinuityEntry):
                for ref in entry.participant_ids:
                    if ref not in by_id or not isinstance(by_id[ref], CharacterEntry):
                        raise ValueError("dangling participant")
        if len(canonical([e.model_dump() for e in self.entries])) > MAX_SNAPSHOT_CHARS:
            raise ValueError("snapshot entries exceed max_snapshot_chars")
        if self.generation_calls:
            attempts = [c.attempt for c in self.generation_calls]
            if len(set(attempts)) != len(attempts) or attempts != sorted(attempts):
                raise ValueError("generation attempts must be unique and increasing")
            completed = [c.call_id for c in self.generation_calls if c.status == "completed"]
            if self.accepted_call_id is not None and completed.count(self.accepted_call_id) != 1:
                raise ValueError("accepted_call_id must name one completed call")
        return self


class Generator(StrictModel):
    provider: str
    configured_model: str | None = None
    resolved_model_id: str
    model_source: Literal["configured", "reported"] = "configured"
    reported_model: str | None = None
    weights_sha256: Hex64 | None = None
    quantization: str | None = None
    harness: str
    profile: str
    identity_sha256: Hex64


class PromptRef(StrictModel):
    version: str
    sha256: Hex64


class Budget(StrictModel):
    max_entries_per_snapshot: int = Field(gt=0, le=MAX_ENTRIES_PER_SNAPSHOT)
    max_snapshot_chars: int = Field(gt=0, le=MAX_SNAPSHOT_CHARS)
    max_selected_entries: int = Field(gt=0, le=MAX_SELECTED_ENTRIES)
    max_context_chars: int = Field(gt=0, le=MAX_CONTEXT_CHARS)
    token_estimator: Literal["utf8-codepoints-div-2-ceil-v1"]
    chars_per_token: float = Field(gt=0, le=10)
    max_estimated_context_tokens: int = Field(gt=0, le=MAX_ESTIMATED_CONTEXT_TOKENS)
    continuity_window_chars: int = Field(ge=0, le=MAX_CONTINUITY_WINDOW_CHARS)


class SelfContext(StrictModel):
    version: Literal["self-context-lite-v1"]
    context_origin: Literal["self"]
    horizon_mode: Literal["history_only", "chapter_preflight"]
    generator: Generator
    prompt: PromptRef
    decoding: dict[str, Any]
    tool_version: str
    generated_at: str
    budget: Budget
    snapshots: list[Snapshot] = Field(min_length=1, max_length=MAX_SNAPSHOTS)
    dependency_closure_sha256: Hex64
    artifact_sha256: Hex64

    @model_validator(mode="after")
    def increasing(self):
        chapters = [s.target_chapter for s in self.snapshots]
        if chapters != sorted(chapters) or len(set(chapters)) != len(chapters):
            raise ValueError("snapshots must have unique increasing target_chapter")
        return self


def sort_set_fields(payload: dict) -> dict:
    for key in SET_FIELDS:
        value = payload.get(key)
        if isinstance(value, list):
            payload[key] = sorted(set(value))
    anchors = payload.get("anchors")
    if isinstance(anchors, list):
        payload["anchors"] = sorted(
            anchors, key=lambda a: (a["chapter"], a["start"], a["end"], a["quote_sha256"])
        )
    return payload


def derive_entry_id(entry: dict) -> str:
    payload = {k: v for k, v in entry.items() if k != "id"}
    sort_set_fields(payload)
    return "ctx-" + sha256(canonical(payload))[:16]


def artifact_hash(artifact: dict) -> str:
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    return sha256(canonical(payload))


def dependency_closure_hash(hashes: list[str]) -> str:
    return sha256(canonical(sorted(set(hashes))))


def compute_identity_sha256(
    *,
    provider: str,
    resolved_model_id: str,
    weights_sha256: str | None,
    quantization: str | None,
    harness: str,
) -> str:
    return sha256(
        canonical(
            {
                "harness": harness,
                "provider": provider,
                "quantization": quantization,
                "resolved_model_id": resolved_model_id,
                "weights_sha256": weights_sha256,
            }
        )
    )


def compute_source_window(previous_text: str, window_chars: int) -> dict | None:
    if not previous_text or window_chars <= 0:
        return None
    start = max(0, len(previous_text) - window_chars)
    text = previous_text[start:]
    return {
        "start": start,
        "end": len(previous_text),
        "text": text,
        "text_sha256": sha256(text),
    }


def _source_map(source_dir: Path) -> dict[int, Any]:
    return {chapter.number: chapter for chapter in discover(source_dir)}


def verify_anchor(anchor: Anchor, sources: dict[int, Any]) -> None:
    chapter = sources.get(anchor.chapter)
    if chapter is None:
        raise ValueError(f"anchor chapter missing: {anchor.chapter}")
    if chapter.source_sha256 != anchor.source_sha256:
        raise ValueError("anchor source hash mismatch")
    if anchor.end > len(chapter.text) or chapter.text[anchor.start : anchor.end] != anchor.quote:
        raise ValueError("anchor quote not found at cited source span")


def _horizon_limit(target_chapter: int, horizon_mode: str) -> int:
    return target_chapter if horizon_mode == "chapter_preflight" else target_chapter - 1


def verify_inputs(
    snapshot: Snapshot, sources: dict[int, Any], *, horizon_mode: str, budget: Budget
) -> None:
    limit = _horizon_limit(snapshot.target_chapter, horizon_mode)
    for item in snapshot.inputs:
        if isinstance(item, SourceStructure):
            payload = item.model_dump()
            payload.pop("structure_sha256")
            if sha256(canonical(payload)) != item.structure_sha256:
                raise ValueError("source structure hash mismatch")
            continue
        if isinstance(item, PriorSelfContext):
            if item.max_source_chapter > limit:
                raise ValueError("prior self-context exceeds declared horizon")
            continue
        chapter = sources.get(item.chapter)
        if chapter is None:
            raise ValueError(f"context input chapter missing: {item.chapter}")
        if item.chapter > limit:
            raise ValueError("context input exceeds declared horizon")
        if chapter.source_sha256 != item.source_sha256:
            raise ValueError("context input source hash mismatch")
        if sha256(chapter.text) != item.normalized_text_sha256:
            raise ValueError("context input normalized text hash mismatch")
        if isinstance(item, SourceWindow):
            if item.end > len(chapter.text):
                raise ValueError("source window out of bounds")
            if item.end - item.start > budget.continuity_window_chars:
                raise ValueError("source window exceeds continuity_window_chars")
            slice_text = chapter.text[item.start : item.end]
            if sha256(slice_text) != item.text_sha256:
                raise ValueError("source window text hash mismatch")
    for entry in snapshot.entries:
        for anchor in entry.anchors:
            if anchor.chapter > limit:
                raise ValueError("anchor exceeds declared horizon")
            verify_anchor(anchor, sources)


def max_source_chapter(artifact: dict) -> int:
    """Actual maximum source chapter referenced directly or through priors."""
    value = 0
    for snapshot in artifact.get("snapshots", []):
        for item in snapshot.get("inputs", []):
            if item.get("kind") in {"japanese_source", "source_window"}:
                value = max(value, int(item["chapter"]))
            elif item.get("kind") == "prior_self_context":
                value = max(value, int(item["max_source_chapter"]))
        for entry in snapshot.get("entries", []):
            for anchor in entry.get("anchors", []):
                value = max(value, int(anchor["chapter"]))
    return value


def transitive_closure(artifact: dict, _seen: set[str] | None = None) -> set[str]:
    """All recursively referenced prior artifact hashes (cycle-safe)."""
    seen = set() if _seen is None else set(_seen)
    result: set[str] = set()
    for snapshot in artifact.get("snapshots", []):
        for item in snapshot.get("inputs", []):
            if item.get("kind") != "prior_self_context":
                continue
            digest = item["artifact_sha256"]
            if digest in result:
                continue
            if digest in seen:
                raise ValueError("prior self-context cycle")
            path = Path(item["artifact_path"])
            if not path.is_file():
                raise ValueError("prior self-context missing")
            prior = json.loads(path.read_text("utf-8"))
            if artifact_hash(prior) != digest:
                raise ValueError("prior self-context hash mismatch")
            result.add(digest)
            result |= transitive_closure(prior, seen | {digest})
    return result


def _normalize_entries(snapshots: list[dict]) -> None:
    for snapshot in snapshots:
        for index, entry in enumerate(snapshot.get("entries", [])):
            kind = entry.get("kind")
            model_name = ENTRY_MODELS.get(kind)
            if model_name is None:
                raise ValueError(f"Unknown entry kind: {kind}")
            model_cls = globals()[model_name]
            draft = dict(entry)
            draft.setdefault("id", "ctx-pending")
            normalized = model_cls.model_validate(draft).model_dump()
            normalized["id"] = derive_entry_id(normalized)
            snapshot["entries"][index] = normalized


def unique_quote_position(text: str, quote: str) -> int:
    """Return the single exact code-point offset, or fail closed.

    No fuzzy matching is used for benchmark artifacts. A quote absent from the
    cited chapter or occurring more than once is rejected rather than resolved
    arbitrarily.
    """
    first = text.find(quote)
    if first == -1:
        raise ValueError("anchor quote not found in referenced source")
    if text.find(quote, first + 1) != -1:
        raise ValueError("anchor quote is ambiguous in referenced source")
    return first


def _finalize_entry(kind: str, payload: dict) -> dict:
    model_name = ENTRY_MODELS.get(kind)
    if model_name is None:
        raise ValueError(f"Unknown entry kind: {kind}")
    model_cls = globals()[model_name]
    draft = {**payload, "id": "ctx-pending"}
    normalized = model_cls.model_validate(draft).model_dump()
    normalized["id"] = derive_entry_id(normalized)
    return normalized


def _anchor_payload(chapter, start: int, quote: str) -> dict:
    return {
        "chapter": chapter.number,
        "source_sha256": chapter.source_sha256,
        "start": start,
        "end": start + len(quote),
        "quote": quote,
        "quote_sha256": sha256(quote),
    }


def _segment_line(line: str, max_chars: int) -> list[tuple[int, int]]:
    if len(line) <= max_chars:
        return [(0, len(line))]
    spans: list[tuple[int, int]] = []
    seg_start = 0
    index = 0
    while index < len(line):
        if line[index] in SENTENCE_ENDS:
            end = index + 1
            while end < len(line) and line[end] in SENTENCE_CLOSERS:
                end += 1
            spans.append((seg_start, end))
            seg_start = end
            index = end
            continue
        if index - seg_start + 1 >= max_chars:
            spans.append((seg_start, index + 1))
            seg_start = index + 1
        index += 1
    if seg_start < len(line):
        spans.append((seg_start, len(line)))
    return spans


def segment_source_text(
    text: str, max_chars: int = QUOTE_UNIT_MAX_CHARS
) -> list[tuple[int, int]]:
    """Deterministic quote-unit spans over normalized source text.

    Physical non-empty lines are the primary units. A line longer than ``max_chars``
    is subdivided after Japanese sentence-ending punctuation (closing punctuation
    retained), with a deterministic hard split as a final fallback. Every span is a
    contiguous exact slice; whitespace, ruby readings and punctuation are preserved.
    """
    spans: list[tuple[int, int]] = []
    line_start = 0
    for line in text.split("\n"):
        if line:
            for start, end in _segment_line(line, max_chars):
                spans.append((line_start + start, line_start + end))
        line_start += len(line) + 1
    return spans


def build_quote_units(
    text: str,
    *,
    source_ref: str,
    chapter_number: int,
    offset: int = 0,
    max_chars: int = QUOTE_UNIT_MAX_CHARS,
) -> list[dict]:
    """Build host-owned exact quote units with stable ids for one source document.

    Ids are deterministic: ``<source_ref prefix>-<4 digits>`` by unit order.
    """
    if source_ref not in SOURCE_REFS:
        raise ValueError(f"unknown source reference: {source_ref}")
    prefix = SOURCE_REF_PREFIX[source_ref]
    units = []
    for number, (start, end) in enumerate(segment_source_text(text, max_chars), start=1):
        units.append(
            {
                "quote_id": f"{prefix}-{number:04d}",
                "source_ref": source_ref,
                "chapter": chapter_number,
                "start": offset + start,
                "end": offset + end,
                "text": text[start:end],
            }
        )
    return units


def resolve_quote_units(
    quote_units: dict[str, dict] | None,
    *,
    sources: dict[int, Any],
    target_chapter: int,
) -> dict[str, dict]:
    """Return a ``quote_id`` -> unit table; derive current-chapter units if omitted."""
    if quote_units is None:
        chapter = sources.get(target_chapter)
        if chapter is None:
            raise ValueError("target chapter missing")
        derived = build_quote_units(
            chapter.text, source_ref="current_chapter", chapter_number=target_chapter
        )
        return {unit["quote_id"]: unit for unit in derived}
    return quote_units


def normalize_model_response(
    response: dict,
    *,
    sources: dict[int, Any],
    horizon_mode: str,
    target_chapter: int,
    quote_units: dict[str, dict] | None = None,
) -> list[dict]:
    """Resolve a v4 model-owned response into strict final entries.

    The wire shape is already structurally constrained by `context-generation-v4`.
    The model only cites host-owned exact quote units by ``quote_id``; here the host
    resolves each id to its frozen chapter and absolute span, derives the final
    `self-context-lite-v1` anchors, and applies the dynamic checks JSON Schema cannot
    know (unknown/duplicate refs, horizon, local-reference resolution, hashes/offsets/
    final ids). The model can never supply quote text or source metadata.
    """
    parsed = ModelResponse.model_validate(response)
    keys = [entry.local_key for entry in parsed.entries]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate local key")
    limit = _horizon_limit(target_chapter, horizon_mode)
    table = resolve_quote_units(
        quote_units, sources=sources, target_chapter=target_chapter
    )

    def anchors_for(entry) -> list[dict]:
        refs = list(entry.anchor_refs)
        if len(set(refs)) != len(refs):
            raise ValueError("duplicate quote reference")
        anchors: list[dict] = []
        for ref in refs:
            unit = table.get(ref)
            if unit is None:
                raise ValueError(f"unknown quote reference for this request: {ref}")
            chapter = sources.get(unit["chapter"])
            if chapter is None:
                raise ValueError(f"quote reference chapter missing: {ref}")
            if chapter.number > limit:
                raise ValueError("model anchor exceeds declared horizon")
            if chapter.text[unit["start"] : unit["end"]] != unit["text"]:
                raise ValueError(f"quote unit does not match frozen chapter text: {ref}")
            anchors.append(_anchor_payload(chapter, unit["start"], unit["text"]))
        return anchors

    resolved: list[tuple] = [(entry, anchors_for(entry)) for entry in parsed.entries]

    def semantic_fields(entry) -> dict:
        semantic = entry.semantic
        return {
            "semantic_class": semantic.semantic_class,
            "confidence": getattr(semantic, "confidence", None),
            "uncertainty": getattr(semantic, "uncertainty", None),
        }

    id_by_key: dict[str, str] = {}
    final: list[dict] = []
    for entry, anchors in resolved:
        if entry.kind != "character":
            continue
        payload = {
            "kind": "character",
            **semantic_fields(entry),
            "claim": entry.claim,
            "source_forms": entry.source_forms,
            "source_names": entry.source_names,
            "rendering": entry.rendering,
            "aliases": entry.aliases,
            "gender": entry.gender,
            "anchors": anchors,
        }
        normalized = _finalize_entry("character", payload)
        id_by_key[entry.local_key] = normalized["id"]
        final.append(normalized)

    def resolve_ref(ref: str, kind: str) -> str:
        if ref not in id_by_key:
            raise ValueError(f"dangling {kind} reference: {ref}")
        return id_by_key[ref]

    for entry, anchors in resolved:
        if entry.kind == "character":
            continue
        payload: dict = {
            "kind": entry.kind,
            **semantic_fields(entry),
            "source_forms": entry.source_forms,
            "anchors": anchors,
        }
        if entry.kind == "relationship":
            payload.update(
                subject_id=resolve_ref(entry.subject_ref, "subject"),
                object_id=resolve_ref(entry.object_ref, "object"),
                relation=entry.relation,
            )
        elif entry.kind == "term":
            payload.update(
                source_terms=entry.source_terms,
                meaning=entry.meaning,
                rendering=entry.rendering,
            )
        elif entry.kind == "speech":
            payload.update(
                speaker_id=resolve_ref(entry.speaker_ref, "speaker"),
                pattern=entry.pattern,
            )
        else:
            payload.update(
                participant_ids=[resolve_ref(r, "participant") for r in entry.participant_refs],
                claim=entry.claim,
            )
        final.append(_finalize_entry(entry.kind, payload))

    final.sort(key=lambda item: item["id"])
    return final


def reported_model_consistent(expected: str, reported: str | None) -> bool:
    """Runtime consistency guard for provider-reported model identity.

    The configured resolved model path/id remains the comparison identity. A
    provider may report the same model under a harmless alias (for example a
    llama.cpp basename or a provider-prefixed path); that is accepted. A reported
    identity that contradicts the frozen expected model fails closed. Normalization
    is explicit and documented: lowercase, drop a trailing path basename, and ignore
    surrounding whitespace; no other aliases are accepted.
    """
    if not reported:
        return True

    def normalize(value: str) -> str:
        value = value.strip().lower()
        return value.rsplit("/", 1)[-1]

    return normalize(expected) == normalize(reported)


def build_artifact(
    raw: dict,
    *,
    source_dir: Path,
    horizon_mode: str,
    generator: dict,
    prompt: dict,
    decoding: dict,
    tool_version: str,
    budget: dict,
    generated_at: str,
    prior_closure: list[str] | None = None,
    candidate_identity_sha256: str | None = None,
) -> dict:
    """Fill host-owned fields (ids, closure, artifact hash) then verify strictly."""
    artifact = json.loads(json.dumps(raw))
    artifact.setdefault("snapshots", [])
    _normalize_entries(artifact["snapshots"])
    gen = dict(generator)
    gen["identity_sha256"] = gen.get("identity_sha256") or compute_identity_sha256(
        provider=gen["provider"],
        resolved_model_id=gen["resolved_model_id"],
        weights_sha256=gen.get("weights_sha256"),
        quantization=gen.get("quantization"),
        harness=gen["harness"],
    )
    artifact.update(
        version=VERSION,
        context_origin="self",
        horizon_mode=horizon_mode,
        generator=gen,
        prompt=prompt,
        decoding=decoding,
        tool_version=tool_version,
        generated_at=generated_at,
        budget=budget,
        dependency_closure_sha256=dependency_closure_hash(prior_closure or []),
    )
    # First validation normalizes defaults; the normalized dump is what is stored
    # and hashed so a reload produces identical bytes.
    normalized = SelfContext.model_validate({**artifact, "artifact_sha256": "0" * 64}).model_dump()
    normalized["artifact_sha256"] = artifact_hash(normalized)
    verify_self_context(
        normalized,
        source_dir=source_dir,
        horizon_mode=horizon_mode,
        candidate_identity_sha256=candidate_identity_sha256 or gen["identity_sha256"],
    )
    return normalized


def verify_self_context(
    artifact: dict,
    *,
    source_dir: Path,
    horizon_mode: str | None = None,
    candidate_identity_sha256: str | None = None,
    prior_artifacts: dict[str, dict] | None = None,
) -> None:
    model = SelfContext.model_validate(artifact)
    if horizon_mode is not None and model.horizon_mode != horizon_mode:
        raise ValueError("horizon mode mismatch")
    if artifact_hash(artifact) != artifact["artifact_sha256"]:
        raise ValueError("artifact_sha256 mismatch")
    if candidate_identity_sha256 is not None and (
        model.generator.identity_sha256 != candidate_identity_sha256
    ):
        raise ValueError("self-context generator identity differs from candidate")
    sources = _source_map(source_dir)
    for snapshot in model.snapshots:
        if len(snapshot.entries) > model.budget.max_entries_per_snapshot:
            raise ValueError("snapshot exceeds max_entries_per_snapshot")
        serialized_entries = canonical([e.model_dump() for e in snapshot.entries])
        if len(serialized_entries) > model.budget.max_snapshot_chars:
            raise ValueError("snapshot exceeds max_snapshot_chars")
        verify_inputs(
            snapshot, sources, horizon_mode=model.horizon_mode, budget=model.budget
        )
        if not snapshot.entries:
            if model.horizon_mode != "history_only" or snapshot.target_chapter != 1:
                raise ValueError(
                    "empty snapshot is legal only for history_only with no prior source"
                )
            if snapshot.inputs:
                raise ValueError("empty history snapshot cannot carry source inputs")
        elif not snapshot.inputs:
            raise ValueError("non-empty snapshot requires at least one typed input")
        for item in snapshot.inputs:
            if not isinstance(item, PriorSelfContext):
                continue
            path = Path(item.artifact_path)
            if not path.is_file():
                raise ValueError("prior self-context missing")
            prior = json.loads(path.read_text("utf-8"))
            if artifact_hash(prior) != item.artifact_sha256:
                raise ValueError("prior self-context hash mismatch")
            if prior.get("generator", {}).get("identity_sha256") != item.generator_identity_sha256:
                raise ValueError("prior self-context generator identity mismatch")
            if prior.get("dependency_closure_sha256") != item.dependency_closure_sha256:
                raise ValueError("prior self-context closure mismatch")
            if max_source_chapter(prior) != item.max_source_chapter:
                raise ValueError("prior self-context max_source_chapter mismatch")
            if prior_artifacts is not None:
                prior_artifacts[item.artifact_sha256] = prior
            verify_self_context(
                prior,
                source_dir=source_dir,
                candidate_identity_sha256=candidate_identity_sha256,
                prior_artifacts=prior_artifacts,
            )
    if dependency_closure_hash(sorted(transitive_closure(artifact))) != (
        model.dependency_closure_sha256
    ):
        raise ValueError("dependency closure hash mismatch")


KIND_RANK = {
    "character": 0,
    "relationship": 1,
    "term": 2,
    "speech": 3,
    "continuity": 4,
    "chapter_preflight_fact": 5,
}


def select_relevant(
    artifact: dict,
    *,
    target_chapter: int,
    chunk_text: str,
    chunk_start: int,
    chunk_end: int,
    max_selected: int | None = None,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    chars_per_token: float | None = None,
    continuity_window_chars: int | None = None,
) -> list[dict]:
    """Deterministic per-chunk projection. Raises on budget overflow, never truncates."""
    model = SelfContext.model_validate(artifact)
    budget = model.budget
    snapshot = next((s for s in model.snapshots if s.target_chapter == target_chapter), None)
    if snapshot is None:
        return []
    max_selected = min(
        budget.max_selected_entries if max_selected is None else max_selected,
        budget.max_selected_entries,
    )
    max_chars = min(
        budget.max_context_chars if max_chars is None else max_chars, budget.max_context_chars
    )
    max_tokens = min(
        budget.max_estimated_context_tokens if max_tokens is None else max_tokens,
        budget.max_estimated_context_tokens,
    )
    chars_per_token = budget.chars_per_token if chars_per_token is None else chars_per_token
    window_chars = min(
        budget.continuity_window_chars
        if continuity_window_chars is None
        else continuity_window_chars,
        budget.continuity_window_chars,
    )
    window = (max(0, chunk_start - window_chars), chunk_start)

    direct_ids = {
        entry.id
        for entry in snapshot.entries
        if any(form and form in chunk_text for form in entry.source_forms)
    }
    ranked: list[tuple] = []
    for entry in snapshot.entries:
        if entry.semantic_class != "source_anchored_claim":
            continue
        direct = entry.id in direct_ids
        included = direct
        endpoint_rank = 2
        if isinstance(entry, RelationshipEntry):
            endpoints = {entry.subject_id, entry.object_id}
            direct_endpoints = endpoints & direct_ids
            if direct_endpoints:
                endpoint_rank = 0 if direct_endpoints == endpoints else 1
            if direct or direct_endpoints:
                included = True
        if isinstance(entry, ContinuityEntry):
            overlap = any(
                a.chapter == target_chapter
                and not (a.end <= window[0] or a.start >= window[1])
                for a in entry.anchors
            )
            included = included or overlap
        if not included:
            continue
        match_rank = 0 if direct else 1
        anchors = sorted(entry.anchors, key=lambda a: (a.chapter, a.start, a.end, a.quote_sha256))
        first = anchors[0]
        ranked.append(
            (
                (
                    match_rank,
                    endpoint_rank if isinstance(entry, RelationshipEntry) else 2,
                    KIND_RANK[entry.kind],
                    first.chapter,
                    first.start,
                    first.end,
                    entry.id,
                ),
                entry.model_dump(),
            )
        )
    if len(ranked) > max_selected:
        raise ValueError("matched entries exceed max_selected_entries")
    ranked.sort(key=lambda item: item[0])
    selected = [payload for _, payload in ranked]
    # rendered-context-budget-v1: the budget bounds the exact deterministic text
    # injected into the translation request, not the canonical entry JSON. The
    # canonical selected-entry serialization stays observable provenance.
    rendered = render_projection(selected)
    if len(rendered) > max_chars:
        raise ValueError("relevant self-context exceeds max_context_chars")
    if estimate_tokens(rendered, chars_per_token) > max_tokens:
        raise ValueError("relevant self-context exceeds max_estimated_context_tokens")
    return selected


def render_projection(entries: list[dict]) -> str:
    lines = []
    for entry in entries:
        text = (
            entry.get("claim")
            or entry.get("relation")
            or entry.get("meaning")
            or entry.get("pattern")
        )
        forms = ", ".join(entry.get("source_forms", []))
        lines.append(f"- [{entry['kind']}] {forms}: {text}")
    return "\n".join(lines)


def _model_view(entry: dict) -> dict:
    """Compact semantic view for the model; never includes host-only fields."""
    anchor = entry["anchors"][0]
    payload = {
        "kind": entry["kind"],
        "semantic_class": entry["semantic_class"],
        "text": entry.get("claim")
        or entry.get("relation")
        or entry.get("meaning")
        or entry.get("pattern"),
        "source_forms": entry.get("source_forms", []),
        "chapter": anchor["chapter"],
        "quote": anchor["quote"],
    }
    for key in ("source_names", "aliases", "source_terms", "gender", "rendering"):
        if entry.get(key):
            payload[key] = entry[key]
    return payload


def project_prior(
    prior: dict, *, target_chapter: int, source_text: str, budget: dict
) -> tuple[list[dict], dict]:
    """Deterministic bounded semantic projection of eligible prior self-context.

    The model receives content, not a path. Only source-anchored prior claims are
    eligible, ordered deterministically; overflow fails closed rather than
    truncating. Returns the payload entries and their provenance metadata.
    """
    sel = SelfContext.model_validate(prior)
    limit = target_chapter - 1
    matched: list[tuple] = []
    for snapshot in sel.snapshots:
        if snapshot.target_chapter > limit:
            continue
        for entry in snapshot.entries:
            if entry.semantic_class != "source_anchored_claim":
                continue
            direct = any(form and form in source_text for form in entry.source_forms)
            anchors = sorted(
                entry.anchors, key=lambda a: (a.chapter, a.start, a.end, a.quote_sha256)
            )
            first = anchors[0]
            matched.append(
                (
                    0 if direct else 1,
                    KIND_RANK[entry.kind],
                    first.chapter,
                    entry.id,
                    entry.model_dump(),
                )
            )
    matched.sort(key=lambda item: item[:4])
    entries = [_model_view(item[4]) for item in matched]
    serialized = canonical(entries)
    if len(matched) > budget["max_selected_entries"]:
        raise ValueError("prior projection exceeds max_selected_entries")
    if len(serialized) > budget["max_context_chars"]:
        raise ValueError("prior projection exceeds max_context_chars")
    tokens = estimate_tokens(serialized, budget["chars_per_token"])
    if tokens > budget["max_estimated_context_tokens"]:
        raise ValueError("prior projection exceeds max_estimated_context_tokens")
    metadata = {
        "selection": PRIOR_PROJECTION_VERSION,
        "projection_sha256": sha256(serialized),
        "entry_count": len(entries),
        "estimated_tokens": tokens,
    }
    return entries, metadata
