"""Per-stage runner execution, attempt persistence and retry bookkeeping.

Extracted from :class:`TranslationService` so chapter traversal stays in the
service while stage execution is independently testable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .artifacts import canonical, sha256, write_json, write_text
from .cache import semantic_cache_key
from .config import ValidationConfig
from .persistence import Repository
from .run_factory import attempt_execution_provenance, now, update_execution_provenance
from .runners.base import (
    INCOMPLETE_FAILURE_KINDS,
    Runner,
    RunnerDeferred,
    RunnerExecutionError,
    RunnerResult,
    TranslationJob,
    incomplete_finish_kind,
)
from .runners.retry import http_deferred
from .translation.frozen_draft import render_frozen
from .translation.review import (
    UNIFIED_REVIEW_CONTRACTS,
    apply_review,
    review_contract,
    review_schema,
)
from .translation.segments import (
    OutputContractError,
    decode_translation,
    render_segments,
    source_segments,
    translation_schema,
)
from .translation.semantic_plan import (
    PlanError,
    parse_plan,
    plan_schema,
    plan_sha256,
    planner_context,
    render_plan,
)
from .translation.validator import (
    TERMINAL_FINISH_REASONS,
    ValidationResult,
    validate,
    validate_segments,
)

# A stage gets this many full per-invocation retry budgets before its durable
# quality/incomplete recovery is declared exhausted. Restart-surviving and
# independent of progress in other chunks/stages. Transient kinds (quota, timeout,
# endpoint-busy, runner_error) do not consume this budget.
INCOMPLETE_RECOVERY_CYCLES = 2
QUALITY_RETRY_FAILURE_KINDS = INCOMPLETE_FAILURE_KINDS | {"validation_failed"}


def _quality_retry_count(attempts: list[dict]) -> int:
    return sum(
        1 for attempt in attempts if attempt.get("failure_kind") in QUALITY_RETRY_FAILURE_KINDS
    )


def _validation_failure_kind(finish_reason: str | None, issues: list[str]) -> str:
    """Classify a deterministic validation rejection for recovery bookkeeping."""
    if finish_reason in {"length", "max_tokens"}:
        return "length"
    if finish_reason == "content_filter":
        return "content_filter"
    if finish_reason is not None and finish_reason not in {"stop", "end_turn"}:
        return incomplete_finish_kind(finish_reason)
    if {"too_short", "empty_output", "missing_output_file"} & set(issues):
        return "incomplete_response"
    return "validation_failed"


class StageExecutor:
    """Run one stage against a chunk and persist attempt metadata/artifacts."""

    def __init__(
        self, repository: Repository, run_dir: Callable[[str], Path], cache=None, tm=None
    ):
        self.repository = repository
        self.run_dir = run_dir
        self.cache = cache
        self.tm = tm

    def valid_output(self, row: dict) -> bool:
        if row["status"] != "completed" or not row.get("output_path"):
            return False
        path = self.run_dir(row["run_id"]) / row["output_path"]
        try:
            return sha256(path.read_bytes()) == row["output_sha256"]
        except OSError:
            return False

    def save_attempt(self, metadata: dict):
        path = self.run_dir(metadata["run_id"]) / metadata["artifact_dir"]
        write_json(path / "metadata.json", metadata)
        self.repository.save_attempt(metadata)

    def run_stage(
        self,
        manifest: dict,
        chapter: dict,
        chunk: dict,
        stage: dict,
        context: dict,
        draft: str | None,
        runner: Runner,
        bypass_cache: bool = False,
        draft_segments: list[dict] | None = None,
        frozen_draft: dict | None = None,
    ):
        run_id = manifest["run_id"]
        run_dir = self.run_dir(run_id)
        settings = manifest["translation"]
        if stage["role"] == "planner":
            # Bounded semantic-plan-v1: one structured current-chunk plan per attempt.
            # The planner sees only the current chunk source plus the glossary; the
            # ordinary previous-source/translation context is deliberately excluded.
            return self._run_planner_stage(
                manifest, chapter, chunk, stage, planner_context(context), runner
            )
        segments = (
            source_segments(chunk["text"], heading=chunk["start"] == 0)
            if settings.get("output_format") == "segments-v1"
            else None
        )
        reviewing = stage["role"] == "reviewer"
        if reviewing and (segments is None or draft_segments is None):
            raise ValueError("Reviewer requires verified draft segments")
        contract = review_contract(stage.get("prompt_version"), stage.get("review_contract"))
        input_hash = sha256(
            canonical(
                {
                    "source": chunk["text"],
                    "context": context,
                    "draft": draft,
                    "stage": stage,
                    "glossary": manifest["glossary"],
                    **({"frozen_draft": frozen_draft, "draft_segments": draft_segments,
                        "validation": manifest["validation"]} if frozen_draft is not None else {}),
                }
            )
        )
        # Cross-run reuse key: the attempt input hash plus the explicit
        # semantics-changing inputs. The attempt input hash itself is unchanged, so
        # resume idempotency for existing runs is preserved.
        runner_config = stage.get("runner_config", {})
        cache_key = semantic_cache_key(
            input_hash,
            {
                "name_rules": settings.get("name_rules", []),
                "book_bible_sha256": manifest.get("book_bible_sha256"),
                "glossary_sha256": manifest.get("glossary_sha256"),
                "prompt_sha256": stage.get("prompt_sha256"),
                "runner_config": runner_config,
                # Effective decoding: the profile may omit temperature/max_tokens, in
                # which case the translation settings supply them, so both are part
                # of the key to avoid a false hit on a decoding change.
                "decoding": {
                    "profile_temperature": runner_config.get("temperature"),
                    "profile_max_tokens": runner_config.get("max_tokens"),
                    "settings_temperature": settings.get("temperature"),
                    "settings_max_tokens": settings.get("max_tokens"),
                    "output_format": settings.get("output_format"),
                },
            },
        )
        attempts = self.repository.attempts(run_id, chapter["number"], chunk["id"], stage["name"])
        for attempt in attempts:
            if attempt["status"] == "running":
                attempt.update(
                    status="interrupted",
                    finished_at=now(),
                    error="Process interrupted before attempt completion",
                )
                self.save_attempt(attempt)
        if not bypass_cache:
            for attempt in reversed(attempts):
                path = run_dir / attempt["artifact_dir"] / "translation.md"
                if (
                    attempt["status"] == "completed"
                    and attempt["input_sha256"] == input_hash
                    and path.is_file()
                    and sha256(path.read_bytes()) == attempt["output_sha256"]
                ):
                    return path.read_text("utf-8"), attempt
        quality_prior = _quality_retry_count(attempts)
        # Durable, restart-surviving ceiling for quality/incomplete failures. Two
        # recovery cycles per stage: the per-invocation retry budget plus one further
        # cycle, after which the same failing stage stops consuming model calls and
        # must be re-planned.
        recovery_ceiling = (settings["retries"] + 1) * INCOMPLETE_RECOVERY_CYCLES
        if quality_prior >= recovery_ceiling:
            previous = attempts[-1] if attempts else {}
            exhausted = {
                **previous,
                "run_id": run_id,
                "chapter": chapter["number"],
                "chunk": chunk["index"],
                "chunk_id": chunk["id"],
                "stage": stage["name"],
                "attempt": previous.get("attempt", len(attempts)),
                "status": "failed",
                "failure_kind": "recovery_exhausted",
                "recovery_exhausted": True,
                "error": (
                    f"Quality/incomplete recovery budget ({recovery_ceiling}) exhausted for "
                    f"stage {stage['name']}; create a fresh continuation or change model/profile"
                ),
                "finished_at": now(),
            }
            self.repository.set_chunk(run_id, chapter["number"], chunk["id"], "failed", exhausted)
            return None, exhausted
        previous_reason = (
            (attempts[-1].get("error") or "incomplete_or_corrupt_artifact") if attempts else None
        )
        allowed_attempts = min(settings["retries"] + 1, recovery_ceiling - quality_prior)
        for retry in range(allowed_attempts):
            number = len(attempts) + retry + 1
            relative = (
                Path("chapters")
                / f"{chapter['number']:04d}"
                / "chunks"
                / chunk["id"]
                / stage["name"]
                / f"attempt-{number:04d}"
            )
            artifact_dir = run_dir / relative
            job = TranslationJob(
                run_id=run_id,
                chapter=chapter["number"],
                chunk_id=chunk["id"],
                stage=stage["name"],
                prompt=stage["prompt"],
                source=chunk["text"],
                context=context,
                glossary=manifest["glossary"],
                workspace=artifact_dir,
                draft=draft,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                segments=segments,
                response_schema=translation_schema(segments) if segments is not None else None,
                draft_segments=[{**s, "sha256": sha256(s["text"])} for s in draft_segments]
                if draft_segments is not None
                else None,
                output_mode="review"
                if reviewing
                else "segments"
                if segments is not None
                else "text",
            )
            if reviewing:
                job = replace(
                    job,
                    response_schema=review_schema(segments, contract),
                    review_contract=contract,
                )
            started = now()
            tick = time.perf_counter()
            metadata = {
                "measurement_version": 1,
                "run_id": run_id,
                "chapter": chapter["number"],
                "chunk": chunk["index"],
                "chunk_id": chunk["id"],
                "stage": stage["name"],
                "attempt": number,
                "runner": stage["runner"],
                "harness": stage["runner_config"]["type"],
                "model": stage["runner_config"]["model"],
                "execution_provenance": attempt_execution_provenance(
                    stage.get("execution_provenance", {})
                ),
                "reasoning_effort": stage["runner_config"].get("reasoning_effort"),
                "workflow": manifest["workflow"],
                "prompt_version": stage["prompt_version"],
                "prompt_sha256": stage["prompt_sha256"],
                "source_sha256": chunk["source_sha256"],
                "input_sha256": input_hash,
                "glossary_sha256": manifest["glossary_sha256"],
                "status": "running",
                "started_at": started,
                "finished_at": None,
                "duration_seconds": None,
                "input_chars": len(chunk["text"]),
                "output_chars": 0,
                "finish_reason": None,
                "usage": {"prompt_tokens": None, "completion_tokens": None},
                "retry_reason": previous_reason,
                "failed_prior": quality_prior + retry,
                "selection_reason": None,
                **({"draft_sha256": frozen_draft["draft_sha256"],
                    "mapping_sha256": sha256(canonical(frozen_draft)),
                    "parent_run_id": manifest["review_input"]["parent_run_id"]}
                   if frozen_draft is not None else {}),
                "artifact_dir": relative.as_posix(),
            }
            write_text(artifact_dir / "source.txt", chunk["text"])
            write_json(artifact_dir / "request.json", job.messages())
            self.save_attempt(metadata)
            process_error = None
            cache_hit = False
            report = None
            try:
                # Consult the cache only on the first attempt. If a cached candidate
                # fails the current validation policy it must not pin every retry to
                # the same rejected text; later attempts regenerate from the runner.
                # Reuse is disabled for segments-v1/reviewer stages: their stored
                # text is the rendered plain text, not the raw structured envelope,
                # so it cannot be re-decoded. The default text output is unaffected.
                lookup = (
                    frozen_draft is None
                    and segments is None
                    and not bypass_cache
                    and retry == 0
                )
                # User-approved translation memory takes precedence over the
                # automatic machine cache; both are exact-fingerprint and re-validated.
                tm_entry = self.tm.get(cache_key) if lookup and self.tm is not None else None
                cached = (
                    self.cache.get(cache_key)
                    if tm_entry is None and lookup and self.cache is not None
                    else None
                )
                if tm_entry is not None:
                    cache_hit = True
                    metadata["cache"] = "tm_hit"
                    result = RunnerResult(
                        text=tm_entry["text"],
                        model=None,
                        finish_reason="stop",
                        usage={"prompt_tokens": None, "completion_tokens": None},
                        effective_settings={},
                    )
                elif cached is not None:
                    cache_hit = True
                    metadata["cache"] = "hit"
                    result = RunnerResult(
                        text=cached["text"],
                        model=cached.get("model"),
                        finish_reason=cached.get("finish_reason"),
                        usage=cached.get("usage")
                        or {"prompt_tokens": None, "completion_tokens": None},
                        effective_settings=cached.get("effective_settings") or {},
                    )
                else:
                    result = runner.run(job)
                if not isinstance(result, RunnerResult):
                    raise TypeError("Runner must return RunnerResult")
                # A cache/TM hit performs no inference in this run: its usage is the
                # original generation's, kept as provenance rather than charged as
                # current usage. Retain measured usage even if validation rejects the
                # returned candidate below.
                reused_usage = result.usage if cache_hit else None
                if cache_hit:
                    metadata["reused_usage"] = reused_usage
                current_usage = (
                    {"prompt_tokens": None, "completion_tokens": None}
                    if cache_hit
                    else result.usage
                )
                metadata.update(
                    model=result.model or metadata["model"],
                    finish_reason=result.finish_reason,
                    usage=current_usage,
                    timings=result.timings,
                    effective_settings=result.effective_settings,
                    execution_provenance=update_execution_provenance(
                        metadata["execution_provenance"],
                        model=result.model,
                        runtime=(result.effective_settings or {}).get("execution"),
                    ),
                )
                if result.text is not None:
                    write_text(artifact_dir / "model-output.txt", result.text)
                if result.raw_response is not None:
                    write_json(artifact_dir / "response.json", result.raw_response)
                if segments is not None:
                    if reviewing:
                        report, translated = apply_review(
                            result.text, segments, draft_segments, contract
                        )
                        metadata["review"] = report
                    else:
                        translated = decode_translation(result.text, segments)
                    result.text = (render_frozen(draft, frozen_draft, translated)
                                   if frozen_draft is not None
                                   else render_segments(chunk["text"], segments, translated))
                    metadata["segments"] = translated
                policy = ValidationConfig.model_validate(manifest["validation"])
                validation = validate(
                    chunk["text"],
                    result.text,
                    result.finish_reason,
                    policy,
                )
                if segments is not None:
                    segment_validation = validate_segments(chunk["text"], translated, policy)
                    if segment_validation.issues:
                        # validate_segments reports FAIL or PASS; any issue is fatal.
                        validation = ValidationResult(
                            "FAIL",
                            list(dict.fromkeys([*validation.issues, *segment_validation.issues])),
                        )
                if result.text is not None:
                    write_text(artifact_dir / "translation.md", result.text)
                failed = validation.level == "FAIL"
                if failed and cache_hit:
                    # A reused candidate that fails current validation is not a hit.
                    metadata["cache"] = "rejected"
                metadata.update(
                    model=result.model or metadata["model"],
                    finish_reason=result.finish_reason,
                    usage=current_usage,
                    validation=validation.to_dict(),
                    output_chars=len(result.text or ""),
                    output_sha256=sha256(result.text or ""),
                    status="failed" if failed else "completed",
                    error=", ".join(validation.issues) if failed else None,
                    cache_key=cache_key,
                    selection_reason=None if failed else "first_passing_attempt",
                    failure_kind=(
                        _validation_failure_kind(result.finish_reason, validation.issues)
                        if failed
                        else None
                    ),
                )
                if (
                    reviewing
                    and contract in UNIFIED_REVIEW_CONTRACTS
                    and report is not None
                    and report.get("unresolved")
                ):
                    # Unresolved defects are accepted and counted, never silently
                    # treated as corrected. The chunk stays lifecycle-completed but
                    # is marked non-clean with an observable open-defect count.
                    metadata["review_open_defects"] = len(report["unresolved"])
                    metadata["review_status"] = "open_defects"
                    if metadata["status"] == "completed":
                        metadata["selection_reason"] = "accepted_with_open_defects"
                if (
                    not cache_hit
                    and self.cache is not None
                    and frozen_draft is None
                    and segments is None
                    and metadata["status"] == "completed"
                    and result.text
                ):
                    self.cache.put(
                        cache_key,
                        text=result.text,
                        model=result.model,
                        finish_reason=result.finish_reason,
                        usage=result.usage,
                        effective_settings=result.effective_settings,
                    )
                # Observable cache state: hit / tm_hit / regenerated.
                metadata.setdefault("cache", "regenerated")
            except Exception as error:  # noqa: BLE001 - isolate third-party runner failures per attempt
                # Provider exceptions can contain request data/credentials. Persist safe diagnostics.
                error = http_deferred(error) or error
                status_code = getattr(error, "status_code", None)
                reason = type(error).__name__ + (f" (HTTP {status_code})" if status_code else "")
                if isinstance(error, RunnerExecutionError):
                    reason = str(error)
                    process_error = error
                if isinstance(error, OutputContractError):
                    reason = str(error)
                if isinstance(error, RunnerDeferred):
                    metadata.update(error.to_dict())
                    write_json(run_dir / "retry.json", error.to_dict())
                # Preserve only reported, non-secret diagnostics; unknown stays unknown.
                diagnostics = error.to_dict() if isinstance(error, RunnerExecutionError) else {}
                if isinstance(error, RunnerDeferred):
                    failure_kind = "quota"
                elif isinstance(error, OutputContractError):
                    failure_kind = "malformed_output"
                else:
                    failure_kind = diagnostics.get("failure_kind", "runner_error")
                safe = {
                    key: diagnostics[key]
                    for key in ("reported_finish_reason", "usage", "artifact")
                    if diagnostics.get(key) is not None
                }
                metadata.update(
                    status="failed",
                    error=reason,
                    failure_kind=failure_kind,
                    validation={
                        "level": "FAIL",
                        "issues": [
                            reason if isinstance(error, OutputContractError) else "runner_error"
                        ],
                    },
                )
                if "reported_finish_reason" in safe:
                    metadata["finish_reason"] = safe["reported_finish_reason"]
                if "usage" in safe:
                    metadata["usage"] = safe["usage"]
                if "artifact" in safe:
                    metadata["failure_artifact"] = safe["artifact"]
                write_text(artifact_dir / "error.txt", reason + "\n")
            metadata.update(finished_at=now(), duration_seconds=time.perf_counter() - tick)
            self.save_attempt(metadata)
            if metadata["status"] == "completed":
                return result.text, metadata
            previous_reason = metadata["error"]
            if process_error is not None and (
                isinstance(process_error, RunnerDeferred) or retry == allowed_attempts - 1
            ):
                self.repository.set_chunk(
                    run_id, chapter["number"], chunk["id"], "failed", metadata
                )
                raise process_error
            if retry < allowed_attempts - 1:
                time.sleep(min(settings["retry_delay_seconds"] * (2**retry), 300))
        return None, metadata

    def _run_planner_stage(
        self,
        manifest: dict,
        chapter: dict,
        chunk: dict,
        stage: dict,
        context: dict,
        runner: Runner,
    ):
        """Bounded semantic-plan-v1: structured current-chunk plan, no free-form text."""
        run_id = manifest["run_id"]
        run_dir = self.run_dir(run_id)
        settings = manifest["translation"]
        attempts = self.repository.attempts(
            run_id, chapter["number"], chunk["id"], stage["name"]
        )
        allowed_attempts = settings["retries"] + 1
        metadata = {}
        for retry in range(allowed_attempts):
            number = len(attempts) + retry + 1
            relative = (
                Path("chapters")
                / f"{chapter['number']:04d}"
                / "chunks"
                / chunk["id"]
                / stage["name"]
                / f"attempt-{number:04d}"
            )
            artifact_dir = run_dir / relative
            artifact_dir.mkdir(parents=True, exist_ok=True)
            input_hash = sha256(
                canonical(
                    {
                        "source": chunk["text"],
                        "context": context,
                        "stage": stage,
                        "glossary": manifest["glossary"],
                        "plan_version": plan_schema()["properties"]["plan_version"]["const"],
                    }
                )
            )
            job = TranslationJob(
                run_id=run_id,
                chapter=chapter["number"],
                chunk_id=chunk["id"],
                stage=stage["name"],
                prompt=stage["prompt"],
                source=chunk["text"],
                context=context,
                glossary=manifest["glossary"],
                workspace=artifact_dir,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                response_schema=plan_schema(),
                output_mode="context",
            )
            write_text(artifact_dir / "source.txt", chunk["text"])
            write_json(artifact_dir / "request.json", job.messages())
            started = now()
            tick = time.perf_counter()
            metadata = {
                "measurement_version": 1,
                "run_id": run_id,
                "chapter": chapter["number"],
                "chunk": chunk["index"],
                "chunk_id": chunk["id"],
                "stage": stage["name"],
                "attempt": number,
                "runner": stage["runner"],
                "harness": stage["runner_config"]["type"],
                "model": stage["runner_config"]["model"],
                "execution_provenance": attempt_execution_provenance(
                    stage.get("execution_provenance", {})
                ),
                "workflow": manifest["workflow"],
                "prompt_version": stage["prompt_version"],
                "prompt_sha256": stage["prompt_sha256"],
                "source_sha256": chunk["source_sha256"],
                "input_sha256": input_hash,
                "glossary_sha256": manifest["glossary_sha256"],
                "status": "running",
                "started_at": started,
                "artifact_dir": relative.as_posix(),
                "plan_version": plan_schema()["properties"]["plan_version"]["const"],
            }
            self.save_attempt(metadata)
            try:
                result = runner.run(job)
                if not isinstance(result, RunnerResult):
                    raise TypeError("Runner must return RunnerResult")
                metadata.update(
                    model=result.model or metadata["model"],
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                    timings=result.timings,
                    effective_settings=result.effective_settings,
                    execution_provenance=update_execution_provenance(
                        metadata["execution_provenance"],
                        model=result.model,
                        runtime=(result.effective_settings or {}).get("execution"),
                    ),
                )
                if result.text is not None:
                    write_text(artifact_dir / "model-output.txt", result.text)
                if (
                    result.finish_reason is not None
                    and result.finish_reason not in TERMINAL_FINISH_REASONS
                ):
                    metadata["failure_kind"] = _validation_failure_kind(result.finish_reason, [])
                    raise PlanError(f"finish_reason:{result.finish_reason}")
                plan = parse_plan(result.text)
                write_json(artifact_dir / "plan.json", plan)
                metadata.update(
                    status="completed",
                    semantic_plan=plan,
                    plan_sha256=plan_sha256(plan),
                    finished_at=now(),
                    duration_seconds=time.perf_counter() - tick,
                )
                self.save_attempt(metadata)
                return render_plan(plan), metadata
            except (PlanError, ValueError, TypeError) as error:
                metadata.update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    failure_kind=metadata.get("failure_kind", "invalid_semantic_plan"),
                    finished_at=now(),
                    duration_seconds=time.perf_counter() - tick,
                )
                self.save_attempt(metadata)
                if retry < allowed_attempts - 1:
                    time.sleep(min(settings["retry_delay_seconds"] * (2**retry), 300))
        return None, metadata
