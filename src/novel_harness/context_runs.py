"""Prepare, generate and consume frozen source-only self-context artifacts.

Generation reads Japanese source inside the declared horizon only. The artifact is
frozen before any measured translation. Translation runs may reference one frozen
artifact and record its hash, origin and horizon in the run manifest.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import canonical, sha256, write_json
from .runners.base import RunnerExecutionError, RunnerResult, TranslationJob
from .translation import self_context as sc
from .translation.discovery import Chapter, discover

CONTEXT_COMPOSITIONS = (
    "none",
    "self_only",
    "canonical_human_only",
    "self_plus_shared_baseline",
)


def identity_from_runner(runner_config: dict) -> tuple[str, str]:
    """Comparison identity from a runner configuration.

    The configured resolved model id is the comparison identity on both the
    generation and translation sides, so a translation run (which cannot know a
    provider-reported id at preparation time) still matches. A provider-reported id
    is recorded as extra `reported_model` provenance only, never as the identity.
    """
    harness = runner_config.get("type", "openai")
    resolved = runner_config.get("model")
    if not resolved:
        raise ValueError("Runner has no resolved model identity")
    quantization = (runner_config.get("extra_params") or {}).get("quantization")
    identity = sc.compute_identity_sha256(
        provider=harness,
        resolved_model_id=resolved,
        weights_sha256=None,
        quantization=quantization,
        harness=harness,
    )
    return resolved, identity


def _allowed_inputs(
    chapter: Chapter,
    chapters: dict[int, Chapter],
    horizon_mode: str,
    window_chars: int,
) -> tuple[list[dict], list[dict]]:
    """Return the frozen allowed inputs and explicit source documents with quote units.

    Every document carries a host-generated ``source_ref``, its chapter, a role, the
    exact text and deterministic exact ``quote_units``. The previous chapter is never
    concatenated anonymously onto the current chapter.
    """
    limit = chapter.number if horizon_mode == "chapter_preflight" else chapter.number - 1
    inputs: list[dict] = []
    documents: list[dict] = []
    if horizon_mode == "chapter_preflight":
        inputs.append(
            {
                "kind": "japanese_source",
                "chapter": chapter.number,
                "filename": chapter.filename,
                "source_sha256": chapter.source_sha256,
                "normalized_text_sha256": sha256(chapter.text),
                "scope": "full_chapter",
            }
        )
        documents.append(
            {
                "source_ref": "current_chapter",
                "chapter": chapter.number,
                "role": "current_chapter",
                "start": 0,
                "text": chapter.text,
                "units": sc.build_quote_units(
                    chapter.text,
                    source_ref="current_chapter",
                    chapter_number=chapter.number,
                ),
            }
        )
    previous = chapters.get(chapter.number - 1)
    if previous is not None and previous.number <= limit:
        window = sc.compute_source_window(previous.text, window_chars)
        if window is not None:
            inputs.append(
                {
                    "kind": "source_window",
                    "chapter": previous.number,
                    "filename": previous.filename,
                    "source_sha256": previous.source_sha256,
                    "normalized_text_sha256": sha256(previous.text),
                    "start": window["start"],
                    "end": window["end"],
                    "text_sha256": window["text_sha256"],
                }
            )
            documents.insert(
                0,
                {
                    "source_ref": "previous_window",
                    "chapter": previous.number,
                    "role": "previous_chapter_window",
                    "start": window["start"],
                    "text": window["text"],
                    "units": sc.build_quote_units(
                        window["text"],
                        source_ref="previous_window",
                        chapter_number=previous.number,
                        offset=window["start"],
                    ),
                },
            )
    return inputs, documents


def _parse_response(text: str) -> dict:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.removeprefix("json")
    try:
        value = json.loads(payload)
    except (ValueError, TypeError) as error:
        raise ValueError("Context generation did not return valid JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
        raise ValueError("Context generation JSON must be an object with an entries list")  # noqa: TRY004
    return value


def _failure_kind(error: Exception) -> str:
    message = str(error).lower()
    if "valid json" in message:
        return "malformed_json"
    if "ambiguous" in message:
        return "ambiguous_anchor"
    if "quote not found" in message:
        return "anchor_not_found"
    if "horizon" in message:
        return "anchor_out_of_horizon"
    if "quote reference" in message or "quote unit" in message:
        return "invalid_quote_ref"
    if "source reference" in message or "source_ref" in message or "source document" in message:
        return "invalid_source_ref"
    if "reference" in message or "local key" in message:
        return "invalid_reference"
    if "validation error" in message or "value error" in message:
        return "schema_invalid"
    return "invalid_response"


def _token_triple(usage: dict) -> tuple[int | None, int | None, int | None]:
    """Return all-null or all-int token counts; derive a missing total arithmetically."""
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if prompt is None or completion is None or total is None:
        return None, None, None
    return prompt, completion, total


def generate_artifact(
    source_dir: Path,
    *,
    chapters: list[int] | None,
    horizon_mode: str,
    call: Callable[[TranslationJob], RunnerResult],
    generator: dict,
    prompt: str,
    prompt_version: str,
    budget: dict | None = None,
    generated_at: str | None = None,
    prior_artifact: tuple[dict, str] | None = None,
    retry_ceiling: int = 2,
    runner_decoding: dict | None = None,
) -> dict:
    """Generate and freeze one source-only self-context artifact.

    ``call`` is the same Runner-style boundary used by translation; tests inject a
    deterministic fake. The model emits only the intermediate model-owned response;
    anchors, hashes, offsets and ids are resolved by the host. ``prior_artifact`` is
    an optional already-frozen artifact (path is provenance only) whose eligible
    prior claims are projected deterministically into the model request.
    """
    if retry_ceiling < 0:
        raise ValueError("retry_ceiling must be non-negative")
    budget = dict(sc.default_budget() if budget is None else budget)
    discovered = discover(source_dir)
    available = {c.number: c for c in discovered}
    selected = (
        [c for c in discovered if c.number in set(chapters)]
        if chapters is not None
        else list(discovered)
    )
    if not selected:
        raise ValueError("No chapters selected for context generation")
    missing = {c for c in (chapters or [])} - set(available)
    if missing:
        raise ValueError(f"Chapters not found: {sorted(missing)}")
    if horizon_mode not in {"history_only", "chapter_preflight"}:
        raise ValueError(f"Unknown horizon mode: {horizon_mode}")
    generated_at = generated_at or _now()
    schema = sc.context_response_schema()
    # Records the caller's actual structured-decoding mode; never claims json-schema
    # unless the caller reports it. The CLI enforces json-schema for real runs.
    requested_mode = (runner_decoding or {}).get("mode") or "unconstrained"
    decoding = {
        "mode": requested_mode,
        "response_schema_version": sc.CONTEXT_RESPONSE_VERSION,
        "response_schema_sha256": sha256(canonical(schema)),
        "source_addressing_version": sc.SOURCE_ADDRESSING_VERSION,
        "quote_unit_segmentation_version": sc.QUOTE_UNIT_SEGMENTATION_VERSION,
        "retry_ceiling": retry_ceiling,
        "temperature": None,
        "max_tokens": None,
    }
    if runner_decoding:
        decoding.update(
            {
                key: value
                for key, value in runner_decoding.items()
                if value is not None and key != "mode"
            }
        )
    snapshots = []
    prior_hashes: set[str] = set()
    reported_ids: set[str] = set()
    for chapter in selected:
        if horizon_mode == "history_only" and chapter.number == 1:
            snapshots.append(
                {
                    "target_chapter": 1,
                    "empty_reason": "no_prior_source",
                    "inputs": [],
                    "entries": [],
                    "generation_calls": [],
                    "accepted_call_id": None,
                }
            )
            continue
        inputs, source_documents = _allowed_inputs(
            chapter, available, horizon_mode, budget["continuity_window_chars"]
        )
        quote_table = {
            unit["quote_id"]: unit
            for document in source_documents
            for unit in document["units"]
        }
        # Relevance text for prior projection: the current chapter when present,
        # otherwise the referenced previous-chapter window (never a blind concat).
        projection_text = next(
            (
                document["text"]
                for document in source_documents
                if document["source_ref"] == "current_chapter"
            ),
            source_documents[0]["text"] if source_documents else "",
        )
        prior_payload = None
        if prior_artifact is not None and chapter.number > 1:
            artifact, path = prior_artifact
            try:
                projected, metadata = sc.project_prior(
                    artifact,
                    target_chapter=chapter.number,
                    source_text=projection_text,
                    budget=budget,
                )
            except ValueError as error:
                raise ValueError(f"prior context projection failed: {error}") from error
            inputs.append(
                {
                    "kind": "prior_self_context",
                    "artifact_path": str(path),
                    "artifact_sha256": artifact["artifact_sha256"],
                    "version": sc.VERSION,
                    "context_origin": "self",
                    "generator_identity_sha256": artifact["generator"]["identity_sha256"],
                    "dependency_closure_sha256": artifact["dependency_closure_sha256"],
                    "max_source_chapter": sc.max_source_chapter(artifact),
                    "projection": metadata,
                }
            )
            prior_hashes.add(artifact["artifact_sha256"])
            prior_hashes |= sc.transitive_closure(artifact)
            prior_payload = {
                "selection": metadata["selection"],
                "entries": projected,
                "entry_count": metadata["entry_count"],
                "estimated_tokens": metadata["estimated_tokens"],
            }
        attempts: list[dict] = []
        accepted = None
        entries: list[dict] | None = None
        for attempt in range(1, retry_ceiling + 2):
            job = TranslationJob(
                run_id="context",
                chapter=chapter.number,
                # Distinct per attempt: a real runner creates its workspace from
                # run/chapter/chunk/stage/workspace, so reusing the id would make a
                # bounded retry collide with the previous attempt instead of running.
                chunk_id=f"context-{chapter.number:04d}-attempt-{attempt}",
                stage="context",
                prompt=prompt,
                # No anonymous/concatenated source text: the model reasons only from
                # the explicit source documents and their host-owned quote units.
                source="",
                context={
                    "horizon_mode": horizon_mode,
                    "inputs": inputs,
                    # Host-generated source documents with exact quote units. Offsets,
                    # hashes and chapter spans stay host-side.
                    "source_documents": [
                        {
                            "source_ref": document["source_ref"],
                            "chapter": document["chapter"],
                            "role": document["role"],
                            "quote_units": [
                                {"quote_id": unit["quote_id"], "text": unit["text"]}
                                for unit in document["units"]
                            ],
                        }
                        for document in source_documents
                    ],
                    "quote_unit_segmentation_version": sc.QUOTE_UNIT_SEGMENTATION_VERSION,
                    "prior_self_context": prior_payload,
                },
                glossary={},
                workspace=source_dir,
                response_schema=schema,
                output_mode="context",
            )
            call_id = f"context-{chapter.number:04d}-attempt-{attempt}"
            request_sha256 = sha256(prompt + canonical(job.messages()))
            started = time.monotonic()
            try:
                result = call(job)
            except RunnerExecutionError as error:
                # A runner-classified failure (for example an incomplete/truncated
                # turn) counts against the frozen retry ceiling instead of aborting
                # the whole artifact.
                attempts.append(
                    {
                        "call_id": call_id,
                        "attempt": attempt,
                        "status": "failed",
                        "request_sha256": request_sha256,
                        "response_sha256": sha256(""),
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "wall_seconds": max(0.0, time.monotonic() - started),
                        "source_chars": sum(
                            len(document["text"]) for document in source_documents
                        ),
                        "finish_reason": getattr(error, "finish_reason", None),
                        "failure_kind": getattr(error, "failure_kind", None) or "runner_error",
                    }
                )
                entries = None
                continue
            wall = time.monotonic() - started
            raw = result.text or ""
            prompt_tokens, completion_tokens, total_tokens = _token_triple(result.usage or {})
            record = {
                "call_id": call_id,
                "attempt": attempt,
                "status": "completed",
                "request_sha256": request_sha256,
                "response_sha256": sha256(raw),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "wall_seconds": max(0.0, wall),
                "source_chars": sum(len(document["text"]) for document in source_documents),
                "finish_reason": result.finish_reason,
                "failure_kind": None,
            }
            reported = (result.effective_settings or {}).get("reported_model") or result.model
            if reported:
                if not sc.reported_model_consistent(
                    generator["resolved_model_id"], str(reported)
                ):
                    raise ValueError(
                        "provider-reported model contradicts the frozen expected model: "
                        f"{reported!r}"
                    )
                reported_ids.add(str(reported))
            if result.effective_settings:
                for key in ("temperature", "max_tokens"):
                    value = result.effective_settings.get(key)
                    if value is not None:
                        decoding[key] = value
            failure = None
            if result.finish_reason not in (None, "stop"):
                failure = "incomplete"
            elif not raw.strip():
                failure = "empty_response"
            else:
                try:
                    entries = sc.normalize_model_response(
                        _parse_response(raw),
                        sources=available,
                        horizon_mode=horizon_mode,
                        target_chapter=chapter.number,
                        quote_units=quote_table,
                    )
                except ValueError as error:
                    failure = _failure_kind(error)
            if failure is not None:
                record["status"] = "failed"
                record["failure_kind"] = failure
                attempts.append(record)
                entries = None
                continue
            attempts.append(record)
            accepted = call_id
            break
        if accepted is None or entries is None:
            raise ValueError(
                f"context generation failed for chapter {chapter.number} after "
                f"{len(attempts)} attempt(s)"
            )
        snapshots.append(
            {
                "target_chapter": chapter.number,
                "inputs": inputs,
                "entries": entries,
                "generation_calls": attempts,
                "accepted_call_id": accepted,
            }
        )
    gen = dict(generator)
    if reported_ids:
        if len(reported_ids) != 1:
            raise ValueError("Inconsistent provider-reported model across context calls")
        # The provider-reported id is extra provenance only. The comparison identity
        # stays the configured resolved identity because a translation run cannot
        # know its reported id at preparation time; both sides derive it the same way.
        gen["reported_model"] = next(iter(reported_ids))
        gen["model_source"] = "reported"
    return sc.build_artifact(
        {"snapshots": snapshots},
        source_dir=source_dir,
        horizon_mode=horizon_mode,
        generator=gen,
        prompt={"version": prompt_version, "sha256": sha256(prompt)},
        decoding=decoding,
        tool_version=__version__,
        budget=budget,
        generated_at=generated_at,
        prior_closure=sorted(prior_hashes),
    )


def write_artifact(path: Path, artifact: dict) -> None:
    write_json(path, artifact)


def load_artifact(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _now() -> str:
    from .run_factory import now

    return now()


def verify_consumable(
    artifact: dict,
    *,
    source_dir: Path,
    candidate_identity_sha256: str | None = None,
    prior_artifacts: dict[str, dict] | None = None,
) -> None:
    sc.verify_self_context(
        artifact,
        source_dir=source_dir,
        candidate_identity_sha256=candidate_identity_sha256,
        prior_artifacts=prior_artifacts,
    )


def validate_composition(
    *,
    composition: str,
    self_context: dict | None,
    canonical_human: dict | None,
) -> None:
    if composition not in CONTEXT_COMPOSITIONS:
        raise ValueError(f"Unknown context composition: {composition}")
    if composition == "none" and (self_context or canonical_human):
        raise ValueError("none composition cannot carry context blocks")
    if composition == "self_only" and (not self_context or canonical_human):
        raise ValueError("self_only requires only self_context")
    if composition == "canonical_human_only" and (self_context or not canonical_human):
        raise ValueError("canonical_human_only requires only canonical_human")
    if composition == "self_plus_shared_baseline" and not (self_context and canonical_human):
        raise ValueError("self_plus_shared_baseline requires both blocks")
    if composition == "self_only":
        # A pure-self result may not carry canonical-human/human-answer inputs.
        if self_context is None or self_context.get("context_origin") != "self":
            raise ValueError("self_only requires context_origin: self")
        if canonical_human is not None:
            raise ValueError("self_only rejects canonical-human context")


def projection_for_chunk(
    artifact: dict,
    *,
    chapter: int,
    chunk: dict,
    budget: dict | None = None,
) -> tuple[list[dict], str, dict]:
    """Deterministic per-chunk projection, the exact rendered text and its metadata.

    The rendered text is the exact string injected into the translation request; the
    frozen character/token budget is enforced against it (rendered-context-budget-v1).
    The canonical selected-entry size stays observable provenance.
    """
    bounds = _budget_bounds(artifact, budget)
    entries = sc.select_relevant(
        artifact,
        target_chapter=chapter,
        chunk_text=chunk["text"],
        chunk_start=chunk["start"],
        chunk_end=chunk["end"],
        **bounds,
    )
    projection = sc.render_projection(entries)
    chars_per_token = bounds.get("chars_per_token", artifact["budget"]["chars_per_token"])
    metadata = {
        "consumption_budget_policy": sc.CONSUMPTION_BUDGET_POLICY,
        "selected_entry_count": len(entries),
        "canonical_selected_chars": len(canonical(entries)),
        "rendered_context_chars": len(projection),
        "rendered_estimated_tokens": sc.estimate_tokens(projection, chars_per_token),
        "projection_sha256": sha256(projection),
    }
    return entries, projection, metadata


def _budget_bounds(artifact: dict, budget: dict | None) -> dict[str, Any]:
    frozen = artifact["budget"]
    if not budget:
        return {}
    # Config/manifest keys differ from the select_relevant parameter names.
    mapping = {
        "max_selected_entries": "max_selected",
        "max_context_chars": "max_chars",
        "max_estimated_context_tokens": "max_tokens",
        "continuity_window_chars": "continuity_window_chars",
    }
    lowered: dict[str, Any] = {}
    for key, parameter in mapping.items():
        if key in budget:
            lowered[parameter] = min(int(budget[key]), int(frozen[key]))
    if "chars_per_token" in budget:
        lowered["chars_per_token"] = min(
            float(budget["chars_per_token"]), float(frozen["chars_per_token"])
        )
    return lowered
