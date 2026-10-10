"""Run creation and sample approval operations over the existing run factory."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import canonical, sha256, write_bytes, write_json
from .config import AppConfig
from .connections import refresh_execution_profiles
from .presets import resolve_preset
from .run_factory import build_inputs, now
from .run_factory import create_run as create_run_factory
from .translation.discovery import discover


def _prepare_source_epub(
    selected_config: AppConfig, identity: dict | None
) -> tuple[dict | None, bytes | None]:
    if not identity or identity.get("source_type") != "epub":
        return identity, None
    from .epub import inspect_source_epub

    directory = selected_config.novel.source_dir
    receipt_path = directory / "source-import.json"
    source_path = directory / "source.epub"
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or source_path.is_symlink()
        or not source_path.is_file()
    ):
        raise ValueError("Imported EPUB source receipt or file is missing")
    if source_path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Imported EPUB source exceeds the run snapshot size limit")
    try:
        receipt = json.loads(receipt_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Imported EPUB source receipt is invalid") from error
    if not isinstance(receipt, dict):
        raise TypeError("Imported EPUB source receipt must be an object")
    data = source_path.read_bytes()
    source_hash = sha256(data)
    if (
        receipt.get("version") != 1
        or receipt.get("provider") != "epub"
        or receipt.get("source_sha256") != source_hash
        or receipt.get("source_id") != source_hash
        or receipt.get("inventory_sha256") != identity.get("inventory_sha256")
        or identity.get("source_id") != source_hash
    ):
        raise ValueError("Imported EPUB source no longer matches its recorded identity")
    imported = inspect_source_epub(data)
    expected_documents = [
        {
            "number": document.number,
            "href": document.href,
            "filename": document.filename,
        }
        for document in imported.documents
    ]
    if receipt.get("documents") != expected_documents:
        raise ValueError("Imported EPUB document map no longer matches its source")
    descriptor = {
        "version": 1,
        "file": "source-epub.epub",
        "sha256": source_hash,
    }
    return {**identity, "source_epub": descriptor}, data


def _store_source_epub(self, run_id: str, descriptor: dict | None, data: bytes | None) -> None:
    if descriptor is None or data is None:
        return
    if sha256(data) != descriptor.get("sha256"):
        raise ValueError("Source EPUB changed before its run snapshot was written")
    run_dir = self.run_dir(run_id)
    path = run_dir / "source-epub.epub"
    if path.is_symlink():
        raise ValueError("Run source EPUB snapshot must not be a symlink")
    if path.exists():
        if not path.is_file() or sha256(path.read_bytes()) != descriptor["sha256"]:
            raise ValueError("Existing run source EPUB snapshot does not match its input")
        return
    write_bytes(path, data)


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
    refresh_execution_profiles(self.config)
    preset, runner = self._auto_default_selection(
        execution_profile=execution_profile,
        preset=preset,
        runner=runner,
        pipeline=pipeline,
        translator=translator,
        editor=editor,
        prompt_version=prompt_version,
    )
    runner = self._execution_profile_selection(
        execution_profile, runner, translator, editor, preset
    )
    runner, pipeline, translator, editor, preset_block = self._resolved_preset(
        preset, runner, pipeline, translator, editor, prompt_version
    )
    self._validate_execution_profile_stage(execution_profile, pipeline, runner, translator)
    selected_config, identity = self._selected_config(novel_id)
    identity, source_epub = self._prepare_source_epub(selected_config, identity)
    run_id = create_run_factory(
        selected_config,
        self.repository,
        self.root,
        self.run_dir,
        runner=runner,
        pipeline=pipeline,
        translator=translator,
        editor=editor,
        chapters=chapters,
        force=force,
        chunk_chars=chunk_chars,
        temperature=temperature,
        prompt_version=prompt_version,
        sample=sample,
        preset=preset_block,
        source_identity=identity,
    )
    self._store_source_epub(run_id, (identity or {}).get("source_epub"), source_epub)
    return run_id


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
    refresh_execution_profiles(self.config)
    preset, runner = self._auto_default_selection(
        execution_profile=execution_profile,
        preset=preset,
        runner=runner,
        pipeline=pipeline,
        translator=translator,
        editor=editor,
        prompt_version=prompt_version,
    )
    runner = self._execution_profile_selection(
        execution_profile, runner, translator, editor, preset
    )
    selected_config, identity = self._selected_config(novel_id)
    identity, source_epub = self._prepare_source_epub(selected_config, identity)
    if chapters is not None:
        selected = sorted(chapters)
    elif novel_id:
        inventory = discover(
            selected_config.novel.source_dir, selected_config.novel.chapter_pattern
        )
        selected = [inventory[0].number] if inventory else []
    else:
        selected = [1]
    if not selected:
        raise ValueError("A sample requires at least one chapter")
    if len(selected) > self.SAMPLE_MAX_CHAPTERS:
        raise ValueError(
            f"A sample is at most {self.SAMPLE_MAX_CHAPTERS} chapters; "
            "launch the whole book after approving the sample"
        )
    runner, pipeline, translator, editor, preset_block = self._resolved_preset(
        preset, runner, pipeline, translator, editor, prompt_version
    )
    self._validate_execution_profile_stage(execution_profile, pipeline, runner, translator)
    run_id = create_run_factory(
        selected_config,
        self.repository,
        self.root,
        self.run_dir,
        runner=runner,
        pipeline=pipeline,
        translator=translator,
        editor=editor,
        chapters=selected,
        chunk_chars=chunk_chars,
        temperature=temperature,
        prompt_version=prompt_version,
        sample={"role": "sample", "chapters": selected},
        preset=preset_block,
        source_identity=identity,
    )
    self._store_source_epub(run_id, (identity or {}).get("source_epub"), source_epub)
    return run_id


def sample_approval(self, run_id: str) -> dict | None:
    path = self.run_dir(run_id) / "sample-approval.json"
    if not path.is_file():
        return None
    approval = json.loads(path.read_text("utf-8"))
    if (
        not isinstance(approval, dict)
        or not _sample_complete(self, run_id)
        or approval.get("sample_output_sha256") != _sample_output_sha256(self, run_id)
    ):
        return None
    return approval


def _sample_complete(self, run_id: str) -> bool:
    status = self.reporter.status(run_id)
    counts = status["chunks"]
    return (
        status["status"] == "completed"
        and status["generation_complete"]
        and counts["completed"] == status["total_chunks"]
        and status["total_chunks"] > 0
        and counts["pending"] == 0
        and counts["running"] == 0
        and counts["failed"] == 0
        and counts["corrupt"] == 0
    )


def _sample_output_sha256(self, run_id: str) -> str:
    """Fingerprint the machine text selected at the moment of human approval."""
    return sha256(canonical([
        (row["chapter"], row["chunk_id"], row["output_sha256"])
        for row in self.repository.chunks(run_id)
    ]))


def approve_sample(self, run_id: str, *, actor: str = "user") -> dict:
    """Record explicit human approval of a completed frozen sample run.

    Approval fails closed until the sample translation has completed
    successfully: a new, running, paused-incomplete or failed sample cannot be
    approved. This is enforced server-side so CLI/API/UI cannot bypass it.
    """
    run = self.repository.run(run_id)
    block = run["manifest"].get("sample")
    if not block or block.get("role") != "sample":
        raise ValueError("Run is not a sample run")
    if not _sample_complete(self, run_id):
        raise ValueError("Sample translation must complete successfully before approval.")
    approval = {
        "run_id": run_id,
        "sample_fingerprint": run["fingerprint"],
        "sample_output_sha256": _sample_output_sha256(self, run_id),
        "actor": actor,
        "approved_at": now(),
    }
    write_json(self.run_dir(run_id) / "sample-approval.json", approval)
    return approval


def _sample_effective_settings(manifest: dict) -> dict:
    stages = manifest.get("stages") or []
    translator = next(
        (s["runner"] for s in stages if s.get("role") in {"translator", "planner"}),
        None,
    )
    translator_stage_for_profile = next(
        (s for s in stages if s.get("role") in {"translator", "planner"}), None
    )
    editor = next((s["runner"] for s in stages if s.get("role") == "editor"), None)
    translation = manifest.get("translation") or {}
    # The effective prompt is the frozen stage prompt, not the legacy global
    # `translation.prompt_version` (which stays untouched for raw-run compatibility).
    translator_stage = next((s for s in stages if s.get("role") in {"translator", "planner"}), None)
    prompt_version = (
        translator_stage.get("prompt_version") if translator_stage else None
    ) or translation.get("prompt_version")
    return {
        "runner": translator,
        "execution_profile": (
            translator_stage_for_profile.get("execution", {}).get("profile_id")
            if translator_stage_for_profile
            and translator_stage_for_profile.get("execution", {}).get("kind")
            in {"direct", "harness-managed"}
            and translator_stage_for_profile.get("execution", {}).get("configured_profile")
            else None
        ),
        "pipeline": manifest.get("workflow"),
        "translator": translator,
        "editor": editor,
        "chunk_chars": translation.get("chunk_chars"),
        "temperature": translation.get("temperature"),
        "prompt_version": prompt_version,
        "preset": manifest.get("preset"),
    }


def _material_mismatches(self, frozen_manifest: dict, current_inputs: dict) -> list[str]:
    mismatches = []
    for key in self._MATERIAL_KEYS:
        frozen_value = frozen_manifest.get(key)
        current_value = current_inputs.get(key)
        if key == "stages":
            matches = self._stages_match(frozen_value, current_value)
        else:
            matches = frozen_value == current_value
        if not matches:
            mismatches.append(key)
    return mismatches


def _stages_match(frozen: object, current: object) -> bool:
    if not isinstance(frozen, list) or not isinstance(current, list):
        return frozen == current
    if len(frozen) != len(current):
        return False
    for frozen_stage, current_stage in zip(frozen, current, strict=True):
        if not isinstance(frozen_stage, dict) or not isinstance(current_stage, dict):
            if frozen_stage != current_stage:
                return False
            continue
        frozen_provenance = frozen_stage.get("execution_provenance")
        if frozen_provenance is not None:
            current_provenance = current_stage.get("execution_provenance")
            if (
                not isinstance(frozen_provenance, dict)
                or not isinstance(current_provenance, dict)
                or not isinstance(frozen_provenance.get("resolved"), dict)
                or not isinstance(current_provenance.get("resolved"), dict)
                or frozen_provenance.get("completeness") != "complete"
                or current_provenance.get("completeness") != "complete"
                or (frozen_provenance.get("resolved") or {}).get("status") == "unknown"
                or (current_provenance.get("resolved") or {}).get("status") == "unknown"
            ):
                return False
        comparable = dict(current_stage)
        # Execution identity is part of each stage. Preserve approvals made
        # by older manifests while still checking all fields those manifests froze.
        if "execution" not in frozen_stage:
            comparable.pop("execution", None)
        if "execution_provenance" not in frozen_stage:
            comparable.pop("execution_provenance", None)
        if frozen_stage != comparable:
            return False
    return True


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
    refresh_execution_profiles(self.config)
    sample = self.repository.run(sample_run_id)
    sample_identity = sample["manifest"].get("source_identity")
    if novel_id is not None and novel_id != (sample_identity or {}).get("novel_id"):
        raise ValueError("Whole-book novel differs from the approved sample")
    selected_config, identity = self._selected_config((sample_identity or {}).get("novel_id"))
    identity, source_epub = self._prepare_source_epub(selected_config, identity)
    if str(selected_config.novel.source_dir) != sample["manifest"].get("source_dir"):
        raise ValueError("Source moved after sample approval; create a new sample")
    block = sample["manifest"].get("sample")
    if not block or block.get("role") != "sample":
        raise ValueError("Run is not a sample run")
    if not (self.run_dir(sample_run_id) / "sample-approval.json").is_file():
        raise ValueError("Whole-book launch requires an approved sample")
    if not _sample_complete(self, sample_run_id):
        raise ValueError("Sample translation must complete successfully before whole-book creation")
    approval = self.sample_approval(sample_run_id)
    if approval is None or approval.get("sample_fingerprint") != sample["fingerprint"]:
        raise ValueError("Whole-book launch requires an approved sample")
    for chapter in sample["manifest"]["chapters"]:
        source = Path(sample["manifest"]["source_dir"]) / chapter["filename"]
        if not source.is_file() or sha256(source.read_bytes()) != chapter["source_sha256"]:
            raise ValueError("Approved sample source changed; create a new sample")
    frozen = self._sample_effective_settings(sample["manifest"])

    requested = {
        "execution_profile": execution_profile,
        "runner": runner,
        "pipeline": pipeline,
        "translator": translator,
        "editor": editor,
        "chunk_chars": chunk_chars,
        "temperature": temperature,
        "prompt_version": prompt_version,
    }
    for key, value in requested.items():
        if value is not None and value != frozen.get(key):
            raise ValueError(
                "Whole-book settings differ from the approved sample; "
                "create and review a new sample"
            )

    preset_block = frozen["preset"]
    if preset is not None:
        resolved = resolve_preset(self.config, preset)
        if resolved["preset_id"] != (preset_block or {}).get("preset_id"):
            raise ValueError(
                "Whole-book preset differs from the approved sample; create and review a new sample"
            )
        preset_block = preset_block or resolved

    sample_block = {
        "role": "whole-book",
        "sample_run_id": sample_run_id,
        "sample_fingerprint": sample["fingerprint"],
        "approved_at": approval["approved_at"],
    }
    # The effective prompt is carried by the frozen `stages`. Do NOT translate it
    # back into the legacy global `translation.prompt_version`: a normal
    # preset/pipeline sample has no explicit override, so the whole-book run must
    # also leave the global value untouched and let the pipeline stage supply the
    # prompt. The `stages` material comparison below is what guarantees the
    # effective prompt cannot silently change (e.g. draft-light -> translate-v1).
    stage_prompt = frozen["prompt_version"]
    global_prompt = (sample["manifest"].get("translation") or {}).get("prompt_version")
    explicit_prompt = None
    if stage_prompt is not None and stage_prompt == global_prompt:
        # The sample explicitly overrode the prompt; re-apply that override.
        explicit_prompt = stage_prompt
    preview = build_inputs(
        selected_config,
        self.repository,
        runner=frozen["runner"],
        pipeline=frozen["pipeline"],
        translator=frozen["translator"],
        editor=frozen["editor"],
        chapters=chapters,
        chunk_chars=frozen["chunk_chars"],
        temperature=frozen["temperature"],
        prompt_version=explicit_prompt,
        sample=sample_block,
        preset=preset_block,
        source_identity=identity,
    )
    mismatches = self._material_mismatches(sample["manifest"], preview.inputs)
    if mismatches:
        raise ValueError(
            "Translation-relevant settings changed after sample approval "
            f"({', '.join(mismatches)}); create and review a new sample"
        )
    run_id = create_run_factory(
        selected_config,
        self.repository,
        self.root,
        self.run_dir,
        runner=frozen["runner"],
        pipeline=frozen["pipeline"],
        translator=frozen["translator"],
        editor=frozen["editor"],
        chapters=chapters,
        chunk_chars=frozen["chunk_chars"],
        temperature=frozen["temperature"],
        prompt_version=explicit_prompt,
        sample=sample_block,
        preset=preset_block,
        source_identity=identity,
    )
    self._store_source_epub(run_id, (identity or {}).get("source_epub"), source_epub)
    return run_id
