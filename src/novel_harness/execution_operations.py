"""Run input verification and execution orchestration for TranslationService."""

from __future__ import annotations

import json
import time
from pathlib import Path

from filelock import Timeout

from .artifacts import canonical, sha256, write_json, write_text
from .book_metadata import verify_novel
from .config import TranslationConfig, load_glossary, validate_glossary
from .routing import choose_runner
from .run_factory import now
from .runners.base import RunnerDeferred
from .scheduler import BoundedScheduler
from .translation.assembly import assemble
from .translation.book_context import bible_context_for, context_for, verify_book_context
from .translation.context import build_context
from .translation.semantic_plan import render_plan


def _verify_inputs(self, manifest: dict):
    """Explicit resume refuses silent drift. A new create_run snapshots changed inputs."""
    verify_novel(manifest)
    TranslationConfig.model_validate(manifest["translation"])
    glossary = validate_glossary(manifest["glossary"])
    if sha256(canonical(glossary)) != manifest["glossary_sha256"]:
        raise ValueError("Run glossary hash mismatch")
    snapshot = load_glossary(self.run_dir(manifest["run_id"]) / "glossary.yaml")
    if snapshot != glossary:
        raise ValueError("Run glossary snapshot changed")
    if "book_context" in manifest:
        book = manifest["book_context"]
        path = self.run_dir(manifest["run_id"]) / "book-context.json"
        if sha256(canonical(book)) != manifest["book_context_sha256"] or (
            not path.is_file() or json.loads(path.read_text("utf-8")) != book
        ):
            raise ValueError("Run book context snapshot changed")
        verify_book_context(book, Path(manifest["source_dir"]))
    if "book_bible" in manifest:
        bible = manifest["book_bible"]
        path = self.run_dir(manifest["run_id"]) / "book-bible.json"
        if sha256(canonical(bible)) != manifest["book_bible_sha256"] or (
            not path.is_file() or json.loads(path.read_text("utf-8")) != bible
        ):
            raise ValueError("Run book bible snapshot changed")
    if "context" in manifest:
        from .context_runs import (
            identity_from_runner,
            validate_composition,
        )
        from .context_runs import verify_consumable as verify_self_context

        block = manifest["context"]
        self_block = block.get("self_context")
        human_block = block.get("canonical_human")
        validate_composition(
            composition=block["context_composition"],
            self_context=self_block,
            canonical_human=human_block,
        )
        if human_block:
            if human_block["book_context_sha256"] != (
                sha256(canonical(manifest["book_context"]))
                if manifest.get("book_context")
                else None
            ):
                raise ValueError("Run canonical-human book context hash changed")
            if human_block["glossary_sha256"] != manifest["glossary_sha256"]:
                raise ValueError("Run canonical-human glossary hash changed")
            if human_block["name_rules_sha256"] != sha256(
                canonical(manifest["translation"].get("name_rules", []))
            ):
                raise ValueError("Run canonical-human name-rule hash changed")
        if self_block:
            artifact_path = self.run_dir(manifest["run_id"]) / self_block["artifact_path"]
            if not artifact_path.is_file():
                raise ValueError("Run self-context snapshot missing")
            artifact = json.loads(artifact_path.read_text("utf-8"))
            if artifact.get("artifact_sha256") != self_block["artifact_sha256"]:
                raise ValueError("Run self-context snapshot changed")
            if artifact["generator"]["identity_sha256"] != self_block["generator_identity_sha256"]:
                raise ValueError("Run self-context generator identity changed")
            if artifact["dependency_closure_sha256"] != self_block["dependency_closure_sha256"]:
                raise ValueError("Run self-context dependency closure changed")
            _resolved, identity = identity_from_runner(manifest["stages"][0]["runner_config"])
            verify_self_context(
                artifact,
                source_dir=Path(manifest["source_dir"]),
                candidate_identity_sha256=identity,
            )
    for stage in manifest["stages"]:
        name = stage["runner"]
        if name not in self.config.runners or (
            self.config.runners[name].public_settings() != stage["runner_config"]
        ):
            raise ValueError(f"Runner configuration changed: {name}; start a new run")
    folder = self.run_dir(manifest["run_id"])
    for chapter in manifest["chapters"]:
        source = Path(manifest["source_dir"]) / chapter["filename"]
        if not source.is_file() or sha256(source.read_bytes()) != chapter["source_sha256"]:
            raise ValueError(f"Source changed or missing: {chapter['filename']}; start a new run")
        original = folder / "chapters" / f"{chapter['number']:04d}" / "original.txt"
        if not original.is_file() or sha256(original.read_bytes()) != chapter["source_sha256"]:
            raise ValueError("Run source snapshot missing or corrupt")

    if "review_input" in manifest:
        from .review_runs import verify_review_inputs

        verify_review_inputs(self, manifest)


def execute(
    self,
    run_id: str,
    *,
    max_chunks: int | None = None,
    clear_pause: bool = True,
    max_parallel: int = 1,
) -> dict:
    """Run/resume synchronously. Call from an external worker for background execution."""
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    try:
        with self._lock(run_id):
            if max_chunks is not None and max_chunks < 1:
                raise ValueError("max_chunks must be positive")
            if clear_pause:
                (self.run_dir(run_id) / "pause.requested").unlink(missing_ok=True)
            self._execute_locked(run_id, max_chunks=max_chunks, max_parallel=max_parallel)
    except Timeout as error:
        raise ValueError(f"Run already executing: {run_id}") from error
    return self.status(run_id)


def _execute_locked(
    self, run_id: str, *, max_chunks: int | None = None, max_parallel: int = 1, run_paused
) -> dict:
    manifest = self.repository.run(run_id)["manifest"]
    routing = manifest.get("routing")
    self._verify_inputs(manifest)
    deferred = self.pending_retry(run_id)
    if deferred:
        raise RunnerDeferred(**deferred)
    (self.run_dir(run_id) / "retry.json").unlink(missing_ok=True)
    rows = self.repository.chunks(run_id)
    lookup = {(r["chapter"], r["chunk_id"]): r for r in rows}
    # Explicitly requeued chunks must bypass the idempotent stage-attempt cache,
    # otherwise resume would just return the previous machine output again.
    requeued = self._requeued(run_id)
    # Construct runners only if execution is necessary; reading completed runs needs no credentials.
    runners = {}
    processed = 0
    start = time.monotonic()
    self.repository.set_run(run_id, "running", now())
    parallel = (
        max_parallel
        if max_parallel > 1
        and manifest["translation"]["context_strategy"] == "none"
        and not manifest["translation"].get("cross_chapter_context", False)
        else 1
    )
    try:
        carry_source, carry_translation, carry_chapter, carry_changed = "", "", None, False
        for chapter in manifest["chapters"]:
            use_carry = (
                manifest["translation"].get("cross_chapter_context", False)
                and carry_chapter == chapter["number"] - 1
            )
            previous_source = carry_source if use_carry else ""
            previous_translation = carry_translation if use_carry else ""
            changed_context = carry_changed if use_carry else False
            if parallel > 1 and max_chunks is None:
                pending = [
                    (chunk, lookup[(chapter["number"], chunk["id"])])
                    for chunk in chapter["chunks"]
                    if not (
                        self.stages.valid_output(lookup[(chapter["number"], chunk["id"])])
                        and not changed_context
                    )
                ]
                if pending:
                    if (self.run_dir(run_id) / "pause.requested").exists():
                        raise run_paused()
                    processed = self._run_parallel_chapter(
                        run_id,
                        manifest,
                        chapter,
                        pending,
                        routing,
                        runners,
                        processed,
                        parallel,
                    )
            else:
                for chunk in chapter["chunks"]:
                    row = lookup[(chapter["number"], chunk["id"])]
                    if self.stages.valid_output(row) and not changed_context:
                        previous_source = chunk["text"]
                        previous_translation = (
                            self.run_dir(run_id) / row["output_path"]
                        ).read_text("utf-8")
                        continue
                    if (max_chunks is not None and processed >= max_chunks) or (
                        self.run_dir(run_id) / "pause.requested"
                    ).exists():
                        raise run_paused()
                    self.repository.set_chunk(run_id, chapter["number"], chunk["id"], "running")
                    context = self._chunk_context(
                        manifest, chapter, chunk, previous_source, previous_translation
                    )
                    draft, stage_results = self._translate_chunk(
                        manifest,
                        chapter,
                        chunk,
                        context,
                        routing,
                        runners,
                        bypass_cache=(chapter["number"], chunk["id"]) in requeued,
                    )
                    if draft is None:
                        self.repository.set_chunk(
                            run_id,
                            chapter["number"],
                            chunk["id"],
                            "failed",
                            stage_results[-1] if stage_results else None,
                        )
                        # Later chunks require the missing context; leave them pending.
                        break
                    changed_context = self._commit_chunk(
                        run_id, manifest, chapter, chunk, row, draft, stage_results
                    )
                    previous_source, previous_translation = chunk["text"], draft
                    processed += 1
            self._assemble_chapter(run_id, chapter)
            chapter_rows = [
                r for r in self.repository.chunks(run_id) if r["chapter"] == chapter["number"]
            ]
            if all(self.stages.valid_output(r) for r in chapter_rows):
                carry_source, carry_translation = previous_source, previous_translation
                carry_chapter, carry_changed = chapter["number"], changed_context
            else:
                carry_chapter, carry_changed = None, False
        all_rows = self.repository.chunks(run_id)
        state = "completed" if all(self.stages.valid_output(r) for r in all_rows) else "failed"
        self.repository.set_run(run_id, state, now(), duration=time.monotonic() - start)
    except run_paused:
        self.repository.set_run(run_id, "paused", now(), duration=time.monotonic() - start)
    except BaseException as error:
        self.repository.set_run(
            run_id,
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            now(),
            type(error).__name__,
            time.monotonic() - start,
        )
        raise
    finally:
        for runner in runners.values():
            close = getattr(runner, "close", None)
            if callable(close):
                close()
    return self.status(run_id)


def _chunk_context(
    self,
    manifest: dict,
    chapter: dict,
    chunk: dict,
    previous_source: str,
    previous_translation: str,
) -> dict:
    if "review_input" in manifest:
        from .review_runs import unit_for

        return unit_for(manifest, chapter, chunk)["context"]
    context = build_context(
        previous_source,
        previous_translation,
        manifest["translation"]["overlap_chars"],
        manifest["translation"]["context_strategy"],
    )
    book = context_for(
        manifest.get("book_context", {}),
        chapter["number"],
        chunk["text"] + previous_source,
        manifest["translation"].get("book_context_chars", 6000),
    )
    if book:
        context["book"] = book
    bible = bible_context_for(manifest.get("book_bible", {}), chunk["text"] + previous_source)
    if bible:
        context["book_bible"] = bible
    if "source_title" in manifest["novel"]:
        context["book_title"] = {
            "source": manifest["novel"]["source_title"],
            "russian": manifest["novel"]["title"],
        }
    self_block = manifest.get("context", {}).get("self_context")
    if self_block:
        from .context_runs import projection_for_chunk

        artifact_path = self.run_dir(manifest["run_id"]) / self_block["artifact_path"]
        artifact = json.loads(artifact_path.read_text("utf-8"))
        entries, projection, metadata = projection_for_chunk(
            artifact,
            chapter=chapter["number"],
            chunk=chunk,
            budget=manifest["translation"].get("self_context_budget"),
        )
        if entries:
            # Inject exactly the deterministic rendered projection. The canonical
            # entry JSON is provenance only, never budgeted-then-swapped
            # (rendered-context-budget-v1).
            context["self_context"] = {"text": projection}
            record_dir = (
                self.run_dir(manifest["run_id"])
                / "self-context-projections"
                / f"{chapter['number']:04d}"
            )
            record_dir.mkdir(parents=True, exist_ok=True)
            (record_dir / f"{chunk['id']}.json").write_text(
                json.dumps(
                    {
                        **metadata,
                        "chapter": chapter["number"],
                        "chunk_id": chunk["id"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "utf-8",
            )
    return context


def _translate_chunk(
    self,
    manifest: dict,
    chapter: dict,
    chunk: dict,
    context: dict,
    routing,
    runners: dict,
    *,
    bypass_cache: bool = False,
):
    draft, stage_results, draft_segments = None, [], None
    frozen = None
    if "review_input" in manifest:
        from .review_runs import checked_bytes, unit_for

        unit = unit_for(manifest, chapter, chunk)
        draft = checked_bytes(
            self.run_dir(manifest["run_id"]), unit["draft_path"], unit["draft_sha256"]
        ).decode("utf-8")
        frozen = unit["mapping"]
        draft_segments = frozen["segments"]
    for stage in manifest["stages"]:
        selected = choose_runner(stage, chunk, manifest.get("glossary", {}), routing)
        if selected.runner not in runners:
            runners[selected.runner] = self.runner_factory(selected.runner)
        effective_stage = stage
        if selected.escalated:
            effective_stage = {
                **stage,
                "runner": selected.runner,
                "runner_config": selected.runner_config,
            }
        draft, metadata = self.stages.run_stage(
            manifest,
            chapter,
            chunk,
            effective_stage,
            context,
            draft,
            runners[selected.runner],
            bypass_cache=bypass_cache,
            draft_segments=draft_segments,
            frozen_draft=frozen,
        )
        if selected.escalated:
            metadata["routing"] = {"runner": selected.runner, "reason": selected.reason}
        stage_results.append(metadata)
        if stage["role"] == "planner":
            # Inject the bounded plan into the following translator's context;
            # the rendered plan is not a translation draft.
            plan = metadata.get("semantic_plan")
            if plan is None:
                break
            context = {**context, "semantic_plan": render_plan(plan)}
            draft = None
            continue
        draft_segments = metadata.get("segments")
        if draft is None:
            break
    return draft, stage_results


def _commit_chunk(
    self,
    run_id: str,
    manifest: dict,
    chapter: dict,
    chunk: dict,
    row: dict,
    draft: str,
    stage_results: list[dict],
) -> bool:
    final_dir = Path("chapters") / f"{chapter['number']:04d}" / "chunks" / chunk["id"]
    final_path = final_dir / "translation.md"
    final_meta = {
        **stage_results[-1],
        "stages": stage_results,
        "output_path": final_path.as_posix(),
    }
    write_text(self.run_dir(run_id) / final_path, draft)
    write_json(self.run_dir(run_id) / final_dir / "metadata.json", final_meta)
    self.repository.set_chunk(
        run_id,
        chapter["number"],
        chunk["id"],
        "completed",
        final_meta,
        final_path.as_posix(),
        sha256(draft),
    )
    # The explicit regeneration request is satisfied once a new machine
    # output is committed; later resumes return to the idempotent path.
    self._clear_requeued(run_id, chapter["number"], chunk["id"])
    return (
        "review_input" not in manifest
        and row.get("output_sha256") != sha256(draft)
        and manifest["translation"]["context_strategy"] == "previous"
        and manifest["translation"]["overlap_chars"] > 0
    )


def _run_parallel_chapter(
    self,
    run_id: str,
    manifest: dict,
    chapter: dict,
    pending: list,
    routing,
    runners: dict,
    processed: int,
    max_parallel: int,
    run_paused,
) -> int:
    for stage in manifest["stages"]:
        if stage["runner"] not in runners:
            runners[stage["runner"]] = self.runner_factory(stage["runner"])
    if routing and routing.get("escalation_runner"):
        escalation = routing["escalation_runner"]
        if escalation not in runners:
            runners[escalation] = self.runner_factory(escalation)

    requeued = self._requeued(run_id)

    def work(item):
        chunk, _row = item
        # A chunk becomes running only when its unit actually starts, so units
        # skipped by an early stop stay pending.
        self.repository.set_chunk(run_id, chapter["number"], chunk["id"], "running")
        context = self._chunk_context(manifest, chapter, chunk, "", "")
        return self._translate_chunk(
            manifest,
            chapter,
            chunk,
            context,
            routing,
            runners,
            bypass_cache=(chapter["number"], chunk["id"]) in requeued,
        )

    results = BoundedScheduler(max_parallel).run(
        pending,
        work,
        should_stop=lambda: (self.run_dir(run_id) / "pause.requested").exists(),
    )
    for (chunk, row), result in zip(pending, results):
        if result.error is not None:
            # The stage executor settles this invocation's failed chunk before
            # raising. Historical later-stage attempts may belong to an older
            # successful translation, so they cannot supply this failure.
            failed_row = self._chunk_row(run_id, chapter["number"], chunk["id"])
            metadata = json.loads(failed_row["metadata"] or "{}")
            if failed_row["status"] != "failed" or metadata.get("status") != "failed":
                # Errors before a stage starts have no measured attempt. Keep
                # the exception type, never arbitrary context/credential text.
                metadata = {
                    "status": "failed",
                    "error": type(result.error).__name__,
                    "failure_kind": "execution_error",
                }
            self.repository.set_chunk(
                run_id,
                chapter["number"],
                chunk["id"],
                "failed",
                metadata,
            )
            continue
        draft, stage_results = result.value
        if draft is None:
            self.repository.set_chunk(
                run_id,
                chapter["number"],
                chunk["id"],
                "failed",
                stage_results[-1] if stage_results else None,
            )
            continue
        self._commit_chunk(run_id, manifest, chapter, chunk, row, draft, stage_results)
        processed += 1
    for chunk, _row in pending[len(results):]:
        # Units never submitted by the back-pressured scheduler stay pending.
        self.repository.set_chunk(run_id, chapter["number"], chunk["id"], "pending")
    if len(results) < len(pending):
        raise run_paused()
    return processed


def _assemble_chapter(self, run_id: str, chapter: dict):
    rows = [r for r in self.repository.chunks(run_id) if r["chapter"] == chapter["number"]]
    if all(self.stages.valid_output(row) for row in rows):
        text = assemble(
            [c["text"] for c in chapter["chunks"]],
            [(self.run_dir(run_id) / r["output_path"]).read_text("utf-8") for r in rows],
        )
        write_text(
            self.run_dir(run_id) / "chapters" / f"{chapter['number']:04d}" / "translation.md",
            text,
        )
