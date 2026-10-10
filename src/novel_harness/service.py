"""Shared application API for CLI now, HTTP/UI and agents later."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from . import (
    execution_operations,
    library_operations,
    preset_operations,
    review_operations,
    run_operations,
)
from .artifacts import sha256, write_json, write_text
from .cache import TranslationCache, TranslationMemory
from .config import (
    AppConfig,
    load_config,
    safe_name,
)
from .connections import (
    import_codex_profile,
    import_direct_profile,
    import_opencode_profile,
    list_connections,
    refresh_execution_profiles,
)
from .exporting import Exporter
from .persistence import Repository
from .presets import (
    default_preset_id,
    preset_entries,
    resolve_preset,
)
from .reporting import Reporter
from .run_factory import now
from .runners import make_runner, runner_info
from .runners.base import Runner
from .stage_executor import StageExecutor
from .translation.assembly import assemble


class RunPaused(Exception):
    """Stop at a committed chunk boundary without losing completed work."""


class ConflictError(ValueError):
    """A compare-and-set precondition failed; maps to HTTP 409 at the API edge."""


class TranslationService:
    def __init__(self, config: AppConfig, runner_factory: Callable[[str], Runner] | None = None):
        self.config = config
        self.root = config.novel.output_dir
        self.repository = Repository(self.root / "state.sqlite3")
        cache = TranslationCache(self.root / "cache") if config.translation.cache_enabled else None
        # The user-approved translation memory is always available; it is distinct
        # from the opt-in automatic machine cache.
        self.tm = TranslationMemory(self.root / "tm")
        self.stages = StageExecutor(self.repository, self.run_dir, cache, self.tm)
        self.reporter = Reporter(self)
        self.exporter = Exporter(self)
        self.runner_factory = runner_factory or (lambda name: make_runner(config, name))
        self._library_diagnostics: list[str] = []
        self._library_rows: list[dict] = []
        self._scan_library()

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> TranslationService:
        return cls(load_config(path))

    def runners(self) -> list[dict]:
        refresh_execution_profiles(self.config)
        return [runner_info(self.config, name) for name in self.config.runners]

    def connections(self) -> dict:
        refresh_execution_profiles(self.config)
        return list_connections(self.config)

    def import_opencode_connection(
        self,
        profile_id: str,
        agent_id: str,
        *,
        advanced: bool = False,
        model_override: str | None = None,
    ) -> dict:
        if model_override is not None and not advanced:
            raise ValueError("Model override is available only through Advanced settings")
        return import_opencode_profile(
            profile_id,
            agent_id,
            model_policy="override" if model_override is not None else "inherit",
            model_override=model_override,
        )

    def import_codex_connection(self, profile_id: str, codex_profile: str | None = None) -> dict:
        return import_codex_profile(profile_id, codex_profile)

    def import_direct_connection(
        self,
        profile_id: str,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 180,
    ) -> dict:
        return import_direct_profile(
            profile_id,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
        )

    def presets(self) -> list[dict]:
        """Resolved user-facing translation modes; never calls a model endpoint."""
        refresh_execution_profiles(self.config)
        return preset_entries(self.config)

    def wizard_options(self) -> dict:
        """Project the preset's actual stage connections for the Web wizard."""
        profiles = []
        for item in self.connections()["profiles"]:
            profiles.append(
                {
                    key: item.get(key)
                    for key in (
                        "id", "kind", "harness", "available", "reason",
                        "model_policy", "requested_model", "resolved_model",
                        "model_source", "agent_id",
                    )
                }
                | {"configured_profile": item["id"] in self.config.execution_profiles}
            )
        presets = []
        for item in self.presets():
            stages = []
            for stage in self.config.pipelines[item["pipeline"]].stages:
                profile_id = stage.runner or (
                    item["translator"] or item["runner"]
                    if stage.role in {"translator", "planner"}
                    else item["editor"]
                )
                stages.append({"name": stage.name, "role": stage.role, "profile_id": profile_id})
            primary = next(
                (stage["profile_id"] for stage in stages if stage["role"] in {"translator", "planner"}),
                None,
            )
            presets.append(item | {"execution_profile_id": primary, "stages": stages})
        return {
            "presets": presets,
            "default": self.default_preset(),
            "profiles": profiles,
        }

    def default_preset(self) -> str | None:
        return default_preset_id(self.config)

    def resolve_preset(self, preset_id: str) -> dict:
        return resolve_preset(self.config, preset_id)

    def _auto_default_selection(
        self,
        *,
        execution_profile: str | None,
        preset: str | None,
        runner: str | None,
        pipeline: str | None,
        translator: str | None,
        editor: str | None,
        prompt_version: str | None,
    ) -> tuple[str | None, str | None]:
        """Apply an opted-in default only to otherwise unselected new runs."""
        return preset_operations._auto_default_selection(
            self,
            execution_profile=execution_profile,
            preset=preset,
            runner=runner,
            pipeline=pipeline,
            translator=translator,
            editor=editor,
            prompt_version=prompt_version,
        )

    def _execution_profile_selection(
        self,
        execution_profile: str | None,
        runner: str | None,
        translator: str | None,
        editor: str | None,
        preset: str | None,
    ) -> str | None:
        return preset_operations._execution_profile_selection(
            self, execution_profile, runner, translator, editor, preset
        )

    def _validate_execution_profile_stage(
        self,
        execution_profile: str | None,
        pipeline: str,
        runner: str | None,
        translator: str | None,
    ) -> None:
        return preset_operations._validate_execution_profile_stage(
            self, execution_profile, pipeline, runner, translator
        )

    def _resolved_preset(
        self,
        preset: str | None,
        runner: str | None,
        pipeline: str | None,
        translator: str | None,
        editor: str | None,
        prompt_version: str | None,
    ) -> tuple[str | None, str, str | None, str | None, dict | None]:
        """Fail closed when a preset is combined with conflicting raw selections."""
        return preset_operations._resolved_preset(
            self, preset, runner, pipeline, translator, editor, prompt_version
        )

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / safe_name(run_id)

    def _scan_library(self) -> list[dict]:
        """Compute a fresh in-memory projection without writing the index."""
        return library_operations._scan_library(self)

    def rescan_library(self) -> dict:
        """Explicitly persist a fresh index projection; never writes sources."""
        return library_operations.rescan_library(self)

    def import_narou_source(
        self, source_dir: str | Path, *, ncode: str, title: str, expected_chapters: tuple[int, ...]
    ) -> dict:
        """Import a completed local Narou output set into the first Library root."""
        return library_operations.import_narou_source(
            self, source_dir, ncode=ncode, title=title, expected_chapters=expected_chapters
        )

    def import_epub_source(self, source_path: str | Path) -> dict:
        """Import a local Japanese source EPUB into the configured Novel Library."""
        return library_operations.import_epub_source(self, source_path)

    def invoke_narou_downloader(
        self,
        command: list[str] | tuple[str, ...],
        *,
        ncode: str,
        source_url: str,
        title: str,
        expected_chapters: tuple[int, ...],
        output_dir: str | Path,
        timeout_seconds: float = 1800,
        cwd: str | Path | None = None,
    ) -> dict:
        """Run the external protocol-v1 downloader, then update the library."""
        return library_operations.invoke_narou_downloader(
            self,
            command,
            ncode=ncode,
            source_url=source_url,
            title=title,
            expected_chapters=expected_chapters,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            cwd=cwd,
        )

    @staticmethod
    def _run_novel_id(manifest: dict, rows: list[dict]) -> str | None:
        return library_operations._run_novel_id(manifest, rows)

    def library_novels(self) -> dict:
        """Read-only card projection over the index and existing run state."""
        return library_operations.library_novels(self)

    def library_remaining(self) -> dict:
        """Return ready novels with at least one untranslated source chapter."""
        return library_operations.library_remaining(self)

    def launch_remaining_novel(self, novel_id: str, config_path: str | Path, **kwargs) -> dict:
        """Launch exactly the current untranslated coverage for one novel."""
        return library_operations.launch_remaining_novel(self, novel_id, config_path, **kwargs)

    def _library_final_summary(self, run_id: str) -> dict:
        """Return final-layer counts without exposing chapter or final text."""
        return library_operations._library_final_summary(self, run_id)

    def library_show(self, novel_id: str) -> dict:
        """Read a unique indexed source without rescanning or changing state."""
        matches = [row for row in self.library_novels()["novels"] if row["novel_id"] == novel_id]
        if not matches:
            raise ValueError(f"Unknown novel: {novel_id}")
        ready = [row for row in matches if row["status"] == "ready"]
        if len(ready) == 1 and not any(row["status"] == "conflict" for row in matches):
            return ready[0]
        if len(matches) != 1 or matches[0]["status"] == "conflict":
            raise ConflictError(f"Ambiguous novel identity: {novel_id}")
        return matches[0]

    def _selected_config(self, novel_id: str | None) -> tuple[AppConfig, dict | None]:
        if novel_id is None:
            return self.config, None
        self.rescan_library()
        row = self.library_show(novel_id)
        if row["status"] != "ready":
            raise ValueError(f"Novel {novel_id} is {row['status']}: {row['error']}")
        selected = self.config.model_copy(deep=True)
        selected.novel.source_dir = Path(row["source_path"])
        selected.novel.chapter_pattern = row["chapter_pattern"]
        if selected.novel.source_dir != self.config.novel.source_dir:
            selected.novel.metadata_path = None
        if selected.novel.metadata_path is None:
            selected.novel.title = row["title"]
            selected.novel.author = row["author"]
        identity = {
            "version": 1,
            "novel_id": novel_id,
            "identity_kind": row["identity_kind"],
            "inventory_sha256": row["inventory_sha256"],
            "source_type": row["source_type"],
            "source_id": row["source_id"],
            "chapter_pattern": row["chapter_pattern"],
        }
        return selected, identity

    @staticmethod
    def _prepare_source_epub(
        selected_config: AppConfig, identity: dict | None
    ) -> tuple[dict | None, bytes | None]:
        return run_operations._prepare_source_epub(selected_config, identity)

    def _store_source_epub(self, run_id: str, descriptor: dict | None, data: bytes | None) -> None:
        return run_operations._store_source_epub(self, run_id, descriptor, data)

    def create_run(
        self,
        *,
        execution_profile: str | None = None,
        runner: str | None = None,
        pipeline: str | None = None,
        translator: str | None = None,
        editor: str | None = None,
        chapters: list[int] | None = None,
        force: bool = False,
        chunk_chars: int | None = None,
        temperature: float | None = None,
        prompt_version: str | None = None,
        preset: str | None = None,
        sample: dict | None = None,
        novel_id: str | None = None,
    ) -> str:
        """Identical input/settings reuse latest run. Force creates a new experiment."""
        run_id = run_operations.create_run(
            self,
            execution_profile=execution_profile,
            runner=runner,
            pipeline=pipeline,
            translator=translator,
            editor=editor,
            chapters=chapters,
            force=force,
            chunk_chars=chunk_chars,
            temperature=temperature,
            prompt_version=prompt_version,
            preset=preset,
            sample=sample,
            novel_id=novel_id,
        )
        self._persist_config_identity(run_id)
        return run_id

    def _persist_config_identity(self, run_id: str) -> None:
        """Attach only non-secret config provenance to a newly materialized run."""
        path = self.config.config_path
        if path is None:
            return
        identity = {
            "version": 1,
            "path": str(path),
            "sha256": sha256(path.read_bytes()),
        }
        run = self.repository.run(run_id)
        if run["manifest"].get("config_identity") == identity:
            return
        manifest = run["manifest"]
        manifest["config_identity"] = identity
        write_json(self.run_dir(run_id) / "manifest.json", manifest)
        self.repository.replace_manifest(run_id, manifest)

    #: A sample is the first chapter or one or two explicit user chapters.
    SAMPLE_MAX_CHAPTERS = 2

    def create_sample_run(
        self,
        *,
        execution_profile: str | None = None,
        runner: str | None = None,
        pipeline: str | None = None,
        translator: str | None = None,
        editor: str | None = None,
        chapters: list[int] | None = None,
        chunk_chars: int | None = None,
        temperature: float | None = None,
        prompt_version: str | None = None,
        preset: str | None = None,
        novel_id: str | None = None,
    ) -> str:
        """Create a normal frozen run over one or two chapters as a sample."""
        run_id = run_operations.create_sample_run(
            self,
            execution_profile=execution_profile,
            runner=runner,
            pipeline=pipeline,
            translator=translator,
            editor=editor,
            chapters=chapters,
            chunk_chars=chunk_chars,
            temperature=temperature,
            prompt_version=prompt_version,
            preset=preset,
            novel_id=novel_id,
        )
        self._persist_config_identity(run_id)
        return run_id

    def sample_approval(self, run_id: str) -> dict | None:
        return run_operations.sample_approval(self, run_id)

    def sample_status(self, run_id: str) -> dict:
        """Read sample state without treating an ordinary run as an unapproved sample."""
        run = self.repository.run(run_id)
        if (run["manifest"].get("sample") or {}).get("role") != "sample":
            raise ValueError("Run is not a sample run")
        return {"status": self.status(run_id), "approval": self.sample_approval(run_id)}

    def approve_sample(self, run_id: str, *, actor: str = "user") -> dict:
        """Record explicit human approval of a completed frozen sample run.

        Approval fails closed until the sample translation has completed
        successfully: a new, running, paused-incomplete or failed sample cannot be
        approved. This is enforced server-side so CLI/API/UI cannot bypass it.
        """
        with self._sample_gate(run_id):
            return run_operations.approve_sample(self, run_id, actor=actor)

    #: Frozen manifest fields that materially affect translation. A whole-book run is
    #: refused when the current configuration would produce any different value than
    #: the approved sample, so approval is never silently inherited across drift.
    _MATERIAL_KEYS = (
        "engine_version",
        "chunker_version",
        "workflow",
        "translation",
        "validation",
        "novel_sha256",
        "source_dir",
        "source_identity",
        "stages",
        "glossary_sha256",
        "book_context_sha256",
        "book_bible_sha256",
    )

    @staticmethod
    def _sample_effective_settings(manifest: dict) -> dict:
        return run_operations._sample_effective_settings(manifest)

    def _material_mismatches(self, frozen_manifest: dict, current_inputs: dict) -> list[str]:
        return run_operations._material_mismatches(self, frozen_manifest, current_inputs)

    @staticmethod
    def _stages_match(frozen: object, current: object) -> bool:
        return run_operations._stages_match(frozen, current)

    def create_whole_book_run(
        self,
        sample_run_id: str,
        *,
        execution_profile: str | None = None,
        preset: str | None = None,
        runner: str | None = None,
        pipeline: str | None = None,
        translator: str | None = None,
        editor: str | None = None,
        chapters: list[int] | None = None,
        chunk_chars: int | None = None,
        temperature: float | None = None,
        prompt_version: str | None = None,
        novel_id: str | None = None,
    ) -> str:
        """Create the whole-book run, gated on an approved frozen sample.

        The whole-book run reuses the approved sample's effective settings. Any
        explicit override that differs from the sample, a different preset, or any
        translation-relevant configuration drift since approval refuses the launch and
        requires a new sample. The CLI batch path (:meth:`create_run`) is unchanged and
        needs no approval.
        """
        with self._sample_gate(sample_run_id):
            run_id = run_operations.create_whole_book_run(
                self,
                sample_run_id,
                execution_profile=execution_profile,
                preset=preset,
                runner=runner,
                pipeline=pipeline,
                translator=translator,
                editor=editor,
                chapters=chapters,
                chunk_chars=chunk_chars,
                temperature=temperature,
                prompt_version=prompt_version,
                novel_id=novel_id,
            )
            self._persist_config_identity(run_id)
            return run_id

    def create_review_run(
        self,
        parent_run_id: str,
        *,
        reviewer: str,
        chapters: list[int] | None = None,
        force: bool = False,
        prompt_version: str = "review-v1",
    ) -> str:
        """Freeze compatible direct chapter drafts; preparation makes no model calls."""
        from .review_runs import prepare_review

        run_id = prepare_review(
            self,
            parent_run_id,
            reviewer=reviewer,
            chapters=chapters,
            force=force,
            prompt_version=prompt_version,
        )
        self._persist_config_identity(run_id)
        return run_id

    def _chapter_draft(self, run_id: str, chapter: dict) -> str:
        rows = [r for r in self.repository.chunks(run_id) if r["chapter"] == chapter["number"]]
        return assemble(
            [c["text"] for c in chapter["chunks"]],
            [(self.run_dir(run_id) / r["output_path"]).read_text("utf-8") for r in rows],
        )

    def _valid_output(self, row: dict) -> bool:
        return self.stages.valid_output(row)

    def _lock(self, run_id: str):
        self.repository.run(run_id)
        return FileLock(str(self.run_dir(run_id) / ".run.lock"), timeout=0)

    @contextmanager
    def _sample_gate(self, run_id: str):
        """Serialize approval, linked creation and requeue with sample execution."""
        self.repository.run(run_id)
        try:
            with FileLock(str(self.run_dir(run_id) / ".run.lock"), timeout=10):
                yield
        except Timeout as error:
            raise ValueError("Sample run is busy; retry after execution") from error

    def _verify_inputs(self, manifest: dict):
        """Explicit resume refuses silent drift. A new create_run snapshots changed inputs."""
        return execution_operations._verify_inputs(self, manifest)

    def execute(
        self,
        run_id: str,
        *,
        max_chunks: int | None = None,
        clear_pause: bool = True,
        max_parallel: int = 1,
    ) -> dict:
        """Run/resume synchronously. Call from an external worker for background execution."""
        return execution_operations.execute(
            self, run_id, max_chunks=max_chunks, clear_pause=clear_pause, max_parallel=max_parallel
        )

    def _execute_locked(
        self, run_id: str, *, max_chunks: int | None = None, max_parallel: int = 1
    ) -> dict:
        return execution_operations._execute_locked(
            self, run_id, max_chunks=max_chunks, max_parallel=max_parallel, run_paused=RunPaused
        )

    def _chunk_context(
        self,
        manifest: dict,
        chapter: dict,
        chunk: dict,
        previous_source: str,
        previous_translation: str,
    ) -> dict:
        return execution_operations._chunk_context(
            self, manifest, chapter, chunk, previous_source, previous_translation
        )

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
        return execution_operations._translate_chunk(
            self, manifest, chapter, chunk, context, routing, runners, bypass_cache=bypass_cache
        )

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
        return execution_operations._commit_chunk(
            self, run_id, manifest, chapter, chunk, row, draft, stage_results
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
    ) -> int:
        return execution_operations._run_parallel_chapter(
            self,
            run_id,
            manifest,
            chapter,
            pending,
            routing,
            runners,
            processed,
            max_parallel,
            run_paused=RunPaused,
        )

    def _assemble_chapter(self, run_id: str, chapter: dict):
        return execution_operations._assemble_chapter(self, run_id, chapter)

    def resume(self, run_id: str) -> dict:
        return self.execute(run_id)

    def continue_run(
        self,
        run_id: str,
        *,
        runner: str,
        chunk_chars: int = 1500,
        editor: str | None = None,
        pipeline: str | None = None,
    ) -> str:
        """New run/settings, inheriting only verified completed prefixes with provenance."""
        from .continuation import create_continuation

        new_id = create_continuation(
            self, run_id, runner, chunk_chars, editor=editor, pipeline=pipeline
        )
        self._persist_config_identity(new_id)
        return new_id

    def pause(self, run_id: str) -> dict:
        self.repository.run(run_id)
        write_text(self.run_dir(run_id) / "pause.requested", now())
        return {"run_id": run_id, "pause_requested": True, "boundary": "after current chunk"}

    def retry_failed(self, run_id: str) -> dict:
        """Retry failed/interrupted/missing work and dependent pending chunks; retain successes."""
        return self.execute(run_id)

    def _chunk_row(self, run_id: str, chapter: int, chunk_id: str) -> dict:
        for row in self.repository.chunks(run_id):
            if row["chapter"] == chapter and row["chunk_id"] == chunk_id:
                return row
        raise ValueError(f"Unknown chunk {chunk_id} in chapter {chapter}")

    def _requeued_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "requeued.json"

    def _requeued(self, run_id: str) -> set[tuple[int, str]]:
        path = self._requeued_path(run_id)
        if not path.is_file():
            return set()
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return set()
        return {(int(item[0]), str(item[1])) for item in data}

    def _write_requeued(self, run_id: str, entries: set[tuple[int, str]]) -> None:
        path = self._requeued_path(run_id)
        if not entries:
            path.unlink(missing_ok=True)
            return
        write_json(path, [list(entry) for entry in sorted(entries)])

    def _mark_requeued(self, run_id: str, chapter: int, chunk_id: str) -> None:
        entries = self._requeued(run_id)
        entries.add((chapter, chunk_id))
        self._write_requeued(run_id, entries)

    def _clear_requeued(self, run_id: str, chapter: int, chunk_id: str) -> None:
        entries = self._requeued(run_id)
        if (chapter, chunk_id) in entries:
            entries.discard((chapter, chunk_id))
            self._write_requeued(run_id, entries)

    def translation_memory(self) -> dict:
        """User-approved translation-memory entries with provenance."""
        return review_operations.translation_memory(self)

    def promote_final_to_tm(
        self, run_id: str, chapter: int, chunk_id: str, *, actor: str = "user"
    ) -> dict:
        """Promote an approved human final into the translation memory.

        This is an explicit action: the machine cache is never promoted
        automatically, and a valid existing entry is never overwritten.
        """
        return review_operations.promote_final_to_tm(self, run_id, chapter, chunk_id, actor=actor)

    def book_bible(self) -> dict:
        """Editable entries plus the deterministic authoritative projection."""
        return review_operations.book_bible(self)

    def propose_bible_entries(self, entries: list[dict], *, proposer: str) -> dict:
        """Proposals (for example from a model) are always ``proposed``.

        A proposal never auto-confirms and never overwrites or downgrades an
        existing entry; existing ids are returned under ``skipped``.
        """
        return review_operations.propose_bible_entries(self, entries, proposer=proposer)

    def set_bible_trust(
        self,
        entry_id: str,
        *,
        action: str,
        actor: str,
        target_form: str | None = None,
        aliases: list[str] | None = None,
        notes: str | None = None,
    ) -> dict:
        """Human review actions: confirm, lock, edit or reject an entry."""
        return review_operations.set_bible_trust(
            self,
            entry_id,
            action=action,
            actor=actor,
            target_form=target_form,
            aliases=aliases,
            notes=notes,
        )

    def final_layer(self, run_id: str) -> list[dict]:
        return review_operations.final_layer(self, run_id)

    def set_final(
        self,
        run_id: str,
        chapter: int,
        chunk_id: str,
        *,
        action: str,
        actor: str,
        text: str | None = None,
        expected_machine_sha256: str,
        expected_final_sha256: str | None = None,
        reason: str | None = None,
    ) -> dict:
        """Accept/edit/needs-review a chunk or revert it to the machine candidate.

        The machine candidate is never overwritten: the human layer is a separate
        record keyed by the chunk, guarded by a compare-and-set on the current
        machine ``output_sha256`` (and on the previous final hash when updating).
        """
        return review_operations.set_final(
            self,
            run_id,
            chapter,
            chunk_id,
            action=action,
            actor=actor,
            text=text,
            expected_machine_sha256=expected_machine_sha256,
            expected_final_sha256=expected_final_sha256,
            reason=reason,
            conflict_error=ConflictError,
        )

    def requeue_chunk(
        self, run_id: str, chapter: int, chunk_id: str, *, expected_machine_sha256: str
    ) -> dict:
        """Mark the chunk for regeneration without destroying the human final.

        Requeue only returns the machine chunk to the state the worker needs; the
        human final and the previous machine output stay addressable. Once the
        worker commits a new machine output, the final is stale by machine-hash
        mismatch and requires an explicit review-desk decision. No model call.
        """
        sample = (self.repository.run(run_id)["manifest"].get("sample") or {}).get("role") == "sample"
        if sample:
            with self._sample_gate(run_id):
                result = review_operations.requeue_chunk(
                    self, run_id, chapter, chunk_id,
                    expected_machine_sha256=expected_machine_sha256,
                    conflict_error=ConflictError,
                )
                (self.run_dir(run_id) / "sample-approval.json").unlink(missing_ok=True)
                return result
        return review_operations.requeue_chunk(
            self, run_id, chapter, chunk_id,
            expected_machine_sha256=expected_machine_sha256,
            conflict_error=ConflictError,
        )

    def status(self, run_id: str | None = None):
        return self.reporter.status(run_id)

    def lightweight_statuses(self, run_ids: list[str] | None = None) -> dict[str, dict]:
        return self.reporter.lightweight_statuses(run_ids)

    def lightweight_status(self, run_id: str) -> dict:
        return self.reporter.lightweight_status(run_id)

    def inspect_chapter(self, run_id: str, chapter: int) -> dict:
        return self.reporter.inspect_chapter(run_id, chapter)

    def compare(self, chapter: int, runs: list[str]) -> dict:
        return self.exporter.compare(chapter, runs)

    def compare_html(self, chapter: int, runs: list[str]) -> Path:
        return self.exporter.compare_html(chapter, runs)

    def revise_names(self, run_id: str) -> dict:
        return self.exporter.revise_names(run_id)

    def build_epub(self, run_id: str, *, metadata_path: Path | None = None) -> Path:
        return self.exporter.build_epub(run_id, metadata_path=metadata_path)

    def audit(self, run_id: str) -> dict:
        return self.reporter.audit(run_id)

    def pending_retry(self, run_id: str) -> dict | None:
        return self.reporter.pending_retry(run_id)

    def metrics(self, run_id: str) -> dict:
        return self.reporter.metrics(run_id)

    def judge_run(self, run_id: str, *, runner: str, chapters: list[int] | None = None) -> dict:
        """Advisory structured judging; runs only when explicitly requested."""
        from .judge import JudgeError, build_judge_job, parse_judge_response

        if runner not in self.config.runners:
            raise ValueError(f"Unknown runner: {runner}")
        manifest = self.repository.run(run_id)["manifest"]
        judge_runner = self.runner_factory(runner)
        results = []
        for chapter in manifest["chapters"]:
            if chapters is not None and chapter["number"] not in chapters:
                continue
            chunk_text = {c["id"]: c["text"] for c in chapter["chunks"]}
            rows = [r for r in self.repository.chunks(run_id) if r["chapter"] == chapter["number"]]
            for row in rows:
                if not self.stages.valid_output(row):
                    continue
                translation = (self.run_dir(run_id) / row["output_path"]).read_text("utf-8")
                workspace = (
                    self.run_dir(run_id) / "judge" / f"{chapter['number']:04d}" / row["chunk_id"]
                )
                workspace.mkdir(parents=True, exist_ok=True)
                job = build_judge_job(
                    run_id=run_id,
                    chapter=chapter["number"],
                    chunk_id=row["chunk_id"],
                    source=chunk_text[row["chunk_id"]],
                    translation=translation,
                    workspace=workspace,
                )
                try:
                    verdict = parse_judge_response(judge_runner.run(job).text)
                    status = "scored"
                except JudgeError as error:
                    verdict = {"score": None, "issues": [], "notes": str(error)}
                    status = "rejected"
                results.append(
                    {
                        "chapter": chapter["number"],
                        "chunk_id": row["chunk_id"],
                        "status": status,
                        **verdict,
                    }
                )
        payload = {
            "run_id": run_id,
            "runner": runner,
            "scored": sum(r["status"] == "scored" for r in results),
            "rejected": sum(r["status"] == "rejected" for r in results),
            "results": results,
        }
        write_json(self.run_dir(run_id) / "judge.json", payload)
        return payload

    def activate(self, run_id: str, config_path: str | Path) -> dict:
        """Only update the matching harness pointer; never execute a translation."""
        manifest = self.repository.run(run_id)["manifest"]
        stage = manifest["stages"][0]
        harness = stage["runner_config"]["type"]
        names = {
            "codex": "active-run.json",
            "opencode": "active-opencode-run.json",
            "openai": "active-direct-run.json",
        }
        if harness not in names:
            raise ValueError("Cannot activate an unsupported harness")
        path = Path(config_path).resolve()
        if not path.is_file():
            raise ValueError("Active config does not exist")
        config = load_config(path)
        if config.novel.output_dir != self.root:
            raise ValueError("Active config must reference this run's output directory")
        if any(
            s["runner"] not in config.runners
            or config.runners[s["runner"]].public_settings() != s["runner_config"]
            for s in manifest["stages"]
        ):
            raise ValueError("Active config must preserve the run's frozen runner profiles")
        pointer = {
            "config": str(path),
            "run_id": run_id,
            "runner": stage["runner"],
            "model": stage["runner_config"]["model"],
            "harness": harness,
            "continued_from": manifest.get("continued_from"),
        }
        target = self.root / names[harness]
        write_json(target, pointer)
        return {"path": str(target), **pointer}
