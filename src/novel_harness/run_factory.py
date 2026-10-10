"""Run creation, input freezing and fingerprinting.

Kept separate from :class:`TranslationService` so orchestration stays thin and
run/input provenance can be tested and reused independently.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from filelock import FileLock

from . import __version__
from .artifacts import canonical, sha256, write_bytes, write_json, write_text
from .book_metadata import freeze_novel
from .budget import source_budget
from .config import AppConfig, TranslationConfig, load_glossary, safe_name
from .connections import execution_snapshot, refresh_execution_profiles
from .persistence import Repository
from .translation.book_context import book_bible_block, freeze_book_context
from .translation.chunker import CHUNKER_VERSION, chunk_text
from .translation.discovery import discover


def now() -> str:
    return datetime.now(UTC).isoformat()


def _model_descriptor(value: object, *, source: str) -> dict:
    """Normalize a reported model without treating a missing report as success."""
    if not isinstance(value, str) or not value.strip():
        return {"status": "unknown", "reason": "runner did not report a model"}
    value = value.strip()
    if "/" in value:
        provider, model = value.split("/", 1)
        if provider and model:
            return {"provider": provider, "model": model, "source": source}
    return {"id": value, "source": source}


def _provenance_completeness(resolved: dict) -> tuple[str, list[str]]:
    if not isinstance(resolved, dict):
        return "unknown", ["resolved_model"]
    if resolved.get("status") == "unknown":
        return "unknown", ["resolved_model"]
    if resolved.get("id") or resolved.get("model"):
        return "complete", []
    return "unknown", ["resolved_model"]


def execution_provenance(
    execution: dict,
    *,
    pipeline: str,
    runner: str,
    runner_config: dict,
    prompt_version: str,
    prompt_sha256: str,
    translation: dict,
) -> dict:
    """Build the versioned, secret-free execution comparison snapshot."""
    requested_model = execution.get("requested_model")
    if requested_model is None:
        requested_model = (
            {"id": runner_config["model"]}
            if runner_config.get("model")
            else {"policy": execution.get("model_policy", "unknown")}
        )
    resolved = execution.get("resolved_model") or {
        "status": "unknown",
        "reason": "execution identity was not resolved",
    }
    if not isinstance(resolved, dict):
        resolved = {"status": "unknown", "reason": "execution identity is malformed"}
    completeness, missing = _provenance_completeness(resolved)
    return {
        "schema_version": 1,
        "requested": {
            "profile_id": execution.get("profile_id"),
            "agent_id": execution.get("agent_id"),
            "model_policy": execution.get("model_policy"),
            "model": requested_model,
        },
        "resolved": resolved,
        "harness": {
            "name": execution.get("harness") or runner_config.get("type"),
            "version": execution.get("harness_version"),
        },
        "profile": {
            "id": execution.get("profile_id"),
            "agent_id": execution.get("agent_id"),
            "config_sha256": execution.get("config_sha256"),
        },
        "material": {
            "pipeline": pipeline,
            "runner": runner,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "temperature": translation.get("temperature"),
            "max_tokens": translation.get("max_tokens"),
            "profile_temperature": runner_config.get("temperature"),
            "profile_max_tokens": runner_config.get("max_tokens"),
            "reasoning_effort": runner_config.get("reasoning_effort"),
            "structured_decoding": runner_config.get("structured_decoding"),
            "transport": runner_config.get("opencode_transport"),
        },
        "completeness": completeness,
        "missing": missing,
    }


def attempt_execution_provenance(manifest_provenance: dict) -> dict:
    """Start an attempt as unresolved until its runner returns evidence."""
    result = deepcopy(manifest_provenance)
    result["resolved"] = {
        "status": "unknown",
        "reason": "runner has not reported execution identity",
    }
    result["completeness"] = "unknown"
    result["missing"] = ["resolved_model"]
    return result


def update_execution_provenance(
    attempt_provenance: dict,
    *,
    model: str | None = None,
    runtime: dict | None = None,
) -> dict:
    """Merge safe adapter evidence into an attempt snapshot."""
    result = deepcopy(attempt_provenance)
    runtime = runtime if isinstance(runtime, dict) else {}
    resolved = runtime.get("resolved_model")
    if not isinstance(resolved, dict):
        resolved = _model_descriptor(model, source="runner-reported")
    if resolved.get("status") != "unknown" and not (resolved.get("id") or resolved.get("model")):
        resolved = {"status": "unknown", "reason": "runner returned an invalid model identity"}
    result["resolved"] = resolved
    if runtime.get("harness_version") is not None:
        result.setdefault("harness", {})["version"] = runtime["harness_version"]
    result["completeness"], result["missing"] = _provenance_completeness(resolved)
    return result


def _chunk_chars_bounds() -> tuple[int, int]:
    """The configured ``TranslationConfig.chunk_chars`` lower and upper bounds."""
    field = TranslationConfig.model_fields["chunk_chars"]
    minimum = next(
        (meta.ge for meta in field.metadata if getattr(meta, "ge", None) is not None), 1
    )
    maximum = next(
        (meta.le for meta in field.metadata if getattr(meta, "le", None) is not None), 100000
    )
    return minimum, maximum


@dataclass(frozen=True)
class FrozenRunInputs:
    """Deterministic, side-effect-free run inputs before any file is written."""

    inputs: dict
    selected: list
    chunks: list
    stages: list
    glossary: dict
    book_context: dict
    bible_block: dict
    self_context_artifact: dict | None


def build_inputs(
    config: AppConfig,
    repository: Repository,
    *,
    runner: str | None = None,
    pipeline: str = "direct",
    translator: str | None = None,
    editor: str | None = None,
    chapters: list[int] | None = None,
    chunk_chars: int | None = None,
    temperature: float | None = None,
    prompt_version: str | None = None,
    sample: dict | None = None,
    preset: dict | None = None,
    source_identity: dict | None = None,
) -> FrozenRunInputs:
    """Compute the frozen inputs for a run without writing anything.

    ``sample`` is the frozen sample-first provenance block; ``preset`` is the frozen
    preset provenance block. Both are part of the fingerprint, so changing
    them (or editing a preset definition later) produces a new run rather than mutating
    an existing one.
    """
    cfg = config
    refresh_execution_profiles(cfg)
    if pipeline not in cfg.pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline}")
    settings = cfg.translation.model_dump(mode="json")
    if chunk_chars is not None:
        settings["chunk_chars"] = chunk_chars
    if temperature is not None:
        settings["temperature"] = temperature
    if prompt_version is not None:
        settings["prompt_version"] = safe_name(prompt_version)
    cfg.translation.__class__.model_validate(settings)
    selected = discover(cfg.novel.source_dir, cfg.novel.chapter_pattern)
    if chapters is not None:
        if not chapters:
            raise ValueError("Select at least one chapter")
        missing = set(chapters) - {chapter.number for chapter in selected}
        if missing:
            raise ValueError(f"Chapters not found: {sorted(missing)}")
        selected = [chapter for chapter in selected if chapter.number in chapters]
    if not selected:
        raise ValueError("No .txt chapters found in source_dir")
    stages = []
    for index, stage in enumerate(cfg.pipelines[pipeline].stages):
        if stage.role == "reviewer" and settings.get("output_format") != "segments-v1":
            raise ValueError("Reviewer stages require translation.output_format: segments-v1")
        # An explicit per-stage runner profile wins; otherwise translator/planner
        # fall back to the selected runner and other roles to the editor.
        if stage.runner:
            name = stage.runner
        elif stage.role in {"translator", "planner"}:
            name = translator or runner
        else:
            name = editor
        if not name or name not in cfg.runners:
            raise ValueError(f"Configure runner for stage {stage.name} ({stage.role})")
        if name in cfg.execution_profiles:
            resolution = cfg._profile_resolution.get(name, {})
            if not resolution.get("available"):
                raise ValueError(
                    f"Execution profile {name} is unavailable: "
                    f"{resolution.get('reason') or 'profile resolution failed'}"
                )
            if cfg.execution_profiles[name].kind == "direct":
                # Direct profiles own an env reference and cannot silently launch
                # without their Novel Harness credential.
                cfg.key_for(name)
        version = (
            prompt_version
            if index == 0 and prompt_version
            else stage.prompt_version or settings["prompt_version"]
        )
        prompt = (cfg.translation.prompts_dir / f"{safe_name(version)}.md").read_text("utf-8")
        if not prompt.strip():
            raise ValueError(f"Empty prompt: {version}")
        runner_config = cfg.runners[name].public_settings()
        execution = execution_snapshot(cfg, name)
        prompt_hash = sha256(prompt)
        stages.append(
            {
                "name": stage.name,
                "role": stage.role,
                "runner": name,
                "runner_config": runner_config,
                "execution": execution,
                "execution_provenance": execution_provenance(
                    execution,
                    pipeline=pipeline,
                    runner=name,
                    runner_config=runner_config,
                    prompt_version=version,
                    prompt_sha256=prompt_hash,
                    translation=settings,
                ),
                "prompt_version": version,
                "prompt_sha256": prompt_hash,
                "prompt": prompt,
            }
        )
    glossary = load_glossary(cfg.translation.glossary_path)
    novel = freeze_novel(cfg.novel)
    book_context = freeze_book_context(cfg.translation.book_context_path, cfg.novel.source_dir)
    # Only confirmed/locked book-bible entries are authoritative; the deterministic
    # projection hash is frozen into run provenance.
    bible_block = book_bible_block(repository.bible_entries())
    self_context_artifact = None
    context_block = None
    composition = settings.get("context_composition", "none")
    if composition != "none" or settings.get("self_context_path"):
        from .context_runs import identity_from_runner, load_artifact, validate_composition
        from .context_runs import verify_consumable as _verify_self_context

        self_block = None
        if settings.get("self_context_path"):
            _resolved, identity = identity_from_runner(stages[0]["runner_config"])
            self_context_artifact = load_artifact(Path(settings["self_context_path"]))
            _verify_self_context(
                self_context_artifact,
                source_dir=cfg.novel.source_dir,
                candidate_identity_sha256=identity,
            )
            self_block = {
                "context_origin": "self",
                "artifact_path": "self-context.json",
                "artifact_sha256": self_context_artifact["artifact_sha256"],
                "dependency_closure_sha256": self_context_artifact[
                    "dependency_closure_sha256"
                ],
                "generator_identity_sha256": self_context_artifact["generator"][
                    "identity_sha256"
                ],
            }
        human_block = {
            "context_origin": "canonical-human",
            "book_context_sha256": sha256(canonical(book_context)) if book_context else None,
            "glossary_sha256": sha256(canonical(glossary)),
            "name_rules_sha256": sha256(canonical(settings.get("name_rules", []))),
            # Both the canonical_human_only baseline and the matching
            # self_plus_shared_baseline arm denote the frozen shared baseline inputs.
            "shared_with_baseline": composition
            in {"canonical_human_only", "self_plus_shared_baseline"},
        }
        if composition == "self_only":
            if book_context or any(glossary.values()) or settings.get("name_rules"):
                raise ValueError("self_only rejects canonical-human context inputs")
            human_block = None
        validate_composition(
            composition=composition, self_context=self_block, canonical_human=human_block
        )
        context_block = {
            "context_composition": composition,
            "self_context": self_block,
            "canonical_human": human_block,
        }
    if chunk_chars is None and settings.get("context_limit"):
        # Derive the source budget from declared context minus fixed overhead.
        # An explicit chunk_chars argument always wins for reproducibility.
        min_chunk, max_chunk = _chunk_chars_bounds()
        settings["chunk_chars"] = source_budget(
            context_limit=settings["context_limit"],
            reserved_output=settings["max_tokens"],
            safety_margin=settings["context_safety_margin"],
            chars_per_token=settings["chars_per_token"],
            overhead_texts=(
                "".join(stage["prompt"] for stage in stages),
                canonical(glossary),
                canonical(book_context),
                canonical(bible_block),
            ),
            reserved_chars=2 * settings["overlap_chars"],
            min_chars=min_chunk,
            max_chars=max_chunk,
        )
        # A derived setting must satisfy the same contract as an explicit one; a
        # derived budget that fails its own validation must never be persisted.
        cfg.translation.__class__.model_validate(settings)
    chapter_data, chunks = [], []
    for chapter in selected:
        chapter_chunks = chunk_text(chapter.text, settings["chunk_chars"])
        chapter_data.append(
            {
                "number": chapter.number,
                "filename": chapter.filename,
                "title": chapter.title,
                "source_sha256": chapter.source_sha256,
                "text_sha256": sha256(chapter.text),
                "chunks": [asdict(c) for c in chapter_chunks],
            }
        )
        chunks.extend({**asdict(c), "chapter": chapter.number} for c in chapter_chunks)
    routing = None
    policy = settings.get("routing") or {}
    if policy.get("enabled"):
        escalation = policy.get("escalation_runner")
        if not escalation or escalation not in cfg.runners:
            raise ValueError("routing.escalation_runner must name a configured runner")
        if escalation in cfg.execution_profiles:
            resolution = cfg._profile_resolution.get(escalation, {})
            if not resolution.get("available"):
                raise ValueError(
                    f"Routing profile {escalation} is unavailable: "
                    f"{resolution.get('reason') or 'profile resolution failed'}"
                )
            if cfg.execution_profiles[escalation].kind == "direct":
                cfg.key_for(escalation)
        routing = {
            **policy,
            "runner_config": cfg.runners[escalation].public_settings(),
            "execution": execution_snapshot(cfg, escalation),
        }
    inputs = {
        "schema_version": 1,
        "routing": routing,
        "engine_version": __version__,
        "chunker_version": CHUNKER_VERSION,
        "workflow": pipeline,
        "translation": settings,
        "validation": cfg.validation.model_dump(),
        "novel": novel,
        "novel_sha256": sha256(canonical(novel)),
        "source_dir": str(cfg.novel.source_dir),
        "stages": stages,
        "glossary": glossary,
        "glossary_sha256": sha256(canonical(glossary)),
        "book_context": book_context,
        "book_context_sha256": sha256(canonical(book_context)),
        "book_bible": bible_block,
        "book_bible_sha256": sha256(canonical(bible_block)),
        "chapters": chapter_data,
    }
    if sample is not None:
        inputs["sample"] = sample
    if preset is not None:
        inputs["preset"] = preset
    if source_identity is not None:
        inputs["source_identity"] = source_identity
    if context_block is not None:
        inputs["context"] = context_block
    return FrozenRunInputs(
        inputs=inputs,
        selected=selected,
        chunks=chunks,
        stages=stages,
        glossary=glossary,
        book_context=book_context,
        bible_block=bible_block,
        self_context_artifact=self_context_artifact,
    )


def create_run(
    config: AppConfig,
    repository: Repository,
    root: Path,
    run_dir: Callable[[str], Path],
    *,
    runner: str | None = None,
    pipeline: str = "direct",
    translator: str | None = None,
    editor: str | None = None,
    chapters: list[int] | None = None,
    force: bool = False,
    chunk_chars: int | None = None,
    temperature: float | None = None,
    prompt_version: str | None = None,
    sample: dict | None = None,
    preset: dict | None = None,
    source_identity: dict | None = None,
) -> str:
    """Identical input/settings reuse latest run. Force creates a new experiment."""
    frozen = build_inputs(
        config,
        repository,
        runner=runner,
        pipeline=pipeline,
        translator=translator,
        editor=editor,
        chapters=chapters,
        chunk_chars=chunk_chars,
        temperature=temperature,
        prompt_version=prompt_version,
        sample=sample,
        preset=preset,
        source_identity=source_identity,
    )
    inputs = frozen.inputs
    fingerprint = sha256(canonical(inputs))
    with FileLock(str(root / ".create.lock"), timeout=30):
        existing = repository.matching(fingerprint)
        if existing and not force:
            return existing
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        manifest = {**inputs, "run_id": run_id, "fingerprint": fingerprint, "created_at": now()}
        folder = run_dir(run_id)
        write_json(folder / "manifest.json", manifest)
        write_text(
            folder / "glossary.yaml",
            yaml.safe_dump(frozen.glossary, allow_unicode=True),
        )
        write_json(folder / "book-context.json", frozen.book_context)
        write_json(folder / "book-bible.json", frozen.bible_block)
        if sample is not None:
            write_json(folder / "sample.json", sample)
        if frozen.self_context_artifact is not None:
            write_json(folder / "self-context.json", frozen.self_context_artifact)
        for stage in frozen.stages:
            write_text(folder / "prompts" / f"{stage['name']}.md", stage["prompt"])
        for chapter in frozen.selected:
            chapter_dir = folder / "chapters" / f"{chapter.number:04d}"
            write_bytes(chapter_dir / "original.txt", chapter.raw)
            write_text(chapter_dir / "source.txt", chapter.text)
        repository.create(manifest, frozen.chunks)
        return run_id
