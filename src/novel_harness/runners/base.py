from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..artifacts import canonical


@dataclass(frozen=True)
class TranslationJob:
    run_id: str
    chapter: int
    chunk_id: str
    stage: str
    prompt: str
    source: str
    context: dict[str, Any]
    glossary: dict
    workspace: Path
    draft: str | None = None
    temperature: float | None = 0.2
    max_tokens: int = 8192
    segments: list[dict] | None = None
    response_schema: dict | None = None
    draft_segments: list[dict] | None = None
    output_mode: str = "text"
    # Frozen review contract identity; unknown/missing stays on the legacy
    # ``review-v1`` contract so already-frozen runs cannot silently switch.
    review_contract: str = "review-v1"

    @property
    def instructions(self) -> str:
        prompt = self.prompt
        if "book_title" in self.context:
            prompt += (
                "\ncontext.book_title is fixed book metadata. Reuse its Russian wording for "
                "references to the work; do not translate that title again. Do not prepend "
                "the book title to chapters or insert it where the source does not mention it."
            )
        if self.response_schema is None:
            return prompt
        if self.output_mode == "context":
            return prompt + (
                "\nReturn only one JSON object matching response_schema, without Markdown "
                "fences or commentary. Emit semantic entries only: never chapter numbers, "
                "hashes, offsets, quote text, ctx-* ids or dependency fields. Each entry "
                "must cite one or more 'anchor_refs' naming quote_id values supplied in "
                "context.source_documents[].quote_units; never emit quote text."
            )
        if self.output_mode == "review":
            return prompt + (
                "\nReturn only a JSON review matching response_schema. Inspect every segment, "
                "return checked_ids in source order, and use the supplied draft hashes for patches. "
                "Do not return a full rewritten translation. No Markdown fences."
            )
        return prompt + (
            "\nResponse format overrides prose-only output instructions: return one JSON object "
            "matching response_schema, without Markdown fences. Each source segment must be "
            "translated completely exactly once, under its unchanged ID, in source order. "
            "Do not merge, split, omit, duplicate or add segments. Each text is one line. "
            "Translate heading segments once; retain author notes and all other source content. "
            "The host restores spacing and builds the book."
        )

    def messages(self) -> list[dict[str, str]]:
        payload = {"context": self.context, "glossary": self.glossary}
        if self.segments is None:
            payload["source"] = self.source
        if self.draft is not None and self.draft_segments is None:
            payload["draft"] = self.draft
        if self.segments is not None:
            payload["segments"] = self.segments
            payload["response_schema"] = self.response_schema
        if self.draft_segments is not None:
            payload["draft_segments"] = self.draft_segments
        return [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": canonical(payload)},
        ]


@dataclass
class RunnerResult:
    text: str | None
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int | None] = field(
        default_factory=lambda: {"prompt_tokens": None, "completion_tokens": None}
    )
    raw_response: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    effective_settings: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    def run(self, job: TranslationJob) -> RunnerResult: ...


class RunnerUnavailable(ValueError):
    pass


# Failure kinds that represent an incomplete or malformed model turn. They are
# deterministic enough to count against a durable per-stage recovery ceiling;
# quota/deferred, timeout and endpoint-busy stay separate (transient) classes.
INCOMPLETE_FAILURE_KINDS = frozenset(
    {
        "length",
        "max_tokens",
        "content_filter",
        "incomplete_response",
        "reasoning_only",
        "malformed_output",
        "empty_output",
    }
)


def incomplete_finish_kind(reason: str | None) -> str:
    """Map a reported finish reason to a bounded incomplete-turn failure kind."""
    if reason in {"length", "max_tokens"}:
        return "length"
    if reason == "content_filter":
        return "content_filter"
    return "incomplete_response"


def usage_from_tokens(tokens: dict | None) -> dict:
    """Normalized usage; never invents values the provider did not report."""
    tokens = tokens or {}
    cache = tokens.get("cache") or {}
    prompt = tokens.get("input")
    usage: dict = {
        "prompt_tokens": (
            prompt + (cache.get("read") or 0) + (cache.get("write") or 0)
            if prompt is not None
            else None
        ),
        "completion_tokens": tokens.get("output"),
    }
    for key, value in {
        "reasoning_tokens": tokens.get("reasoning"),
        "cached_input_tokens": cache.get("read"),
        "cache_write_input_tokens": cache.get("write"),
    }.items():
        if value is not None:
            usage[key] = value
    return usage


class RunnerExecutionError(RuntimeError):
    """Safe runner failure carrying only non-secret diagnostics.

    ``failure_kind`` classifies the failure so callers can separate a repeated
    incomplete turn from a transient outage. ``finish_reason``/``usage`` are
    retained only when the provider actually reported them; otherwise they stay
    unknown instead of being fabricated.
    """

    failure_kind = "runner_error"

    def __init__(
        self,
        message: str = "",
        *,
        failure_kind: str | None = None,
        finish_reason: str | None = None,
        usage: dict | None = None,
        artifact: str | None = None,
    ):
        super().__init__(message)
        if failure_kind is not None:
            self.failure_kind = failure_kind
        self.finish_reason = finish_reason
        self.usage = usage
        self.artifact = artifact

    def to_dict(self) -> dict:
        data: dict = {"failure_kind": self.failure_kind}
        if self.finish_reason is not None:
            data["reported_finish_reason"] = self.finish_reason
        if self.usage is not None:
            data["usage"] = self.usage
        if self.artifact is not None:
            data["artifact"] = self.artifact
        return data


class RunnerDeferred(RunnerExecutionError):
    failure_kind = "quota"

    def __init__(self, retry_at: str, reason: str = "quota", deadline_source: str = "provider"):
        self.retry_at = retry_at
        self.reason = reason
        self.deadline_source = deadline_source
        super().__init__(f"Runner deferred ({reason}) until {retry_at}")

    def to_dict(self):
        return {
            "retry_at": self.retry_at,
            "reason": self.reason,
            "deadline_source": self.deadline_source,
        }
