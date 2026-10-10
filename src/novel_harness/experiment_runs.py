"""Durable, bounded generation-matrix control over existing translation runs.

This module only freezes plans, starts the existing worker, and projects saved
attempt evidence. It never calls a runner directly.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from .artifacts import canonical, sha256, write_json
from .config import StrictModel, _GlossaryLoader, safe_name
from .runners.opencode import OpenCodeRunner
from .translation.discovery import discover
from .worker import start_worker

MATRIX_VERSION = 1
AXES = frozenset({"model", "harness", "pipeline", "prompt", "chunking", "decoding"})
AXIS_SIGNATURE_FIELDS = {
    "model": "resolved_model",
    "harness": "harness",
    "pipeline": "pipeline",
    "prompt": "prompt",
    "chunking": "chunking",
    "decoding": "decoding",
}
FAILURE_POLICIES = {
    "endpoint_busy": ("transient", "bounded_worker_retry"),
    "timeout": ("transient", "bounded_worker_retry"),
    "quota": ("transient", "deferred_until_deadline"),
    "length": ("partial", "preserve_cap_and_sampler"),
    "max_tokens": ("partial", "preserve_cap_and_sampler"),
    "malformed_output": ("output_validation", "bounded_incomplete_recovery"),
    "incomplete_response": ("output_validation", "bounded_incomplete_recovery"),
    "reasoning_only": ("output_validation", "bounded_incomplete_recovery"),
    "empty_output": ("output_validation", "bounded_incomplete_recovery"),
    "source_drift": ("invariant_drift", "stop_arm"),
    "model_drift": ("invariant_drift", "stop_arm"),
    "wire_drift": ("invariant_drift", "stop_arm"),
}


class MatrixArm(StrictModel):
    id: str
    runner: str
    editor: str | None = None
    translator: str | None = None
    pipeline: str = "direct"
    prompt_version: str | None = None
    chunk_chars: int | None = Field(default=None, ge=32, le=100000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    # These fields are assertions over the configured runner, not alternate
    # credentials or hidden per-arm config. The service remains the executor.
    model: str | None = None
    harness: str | None = None
    endpoint: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    retries: int | None = Field(default=None, ge=0, le=20)
    seed: int | None = None
    sampler: dict[str, Any] | None = None


class MatrixSpec(StrictModel):
    version: int = MATRIX_VERSION
    name: str
    chapters: list[int] = Field(min_length=1)
    vary: list[str] = Field(default_factory=lambda: ["model"])
    arms: list[MatrixArm] = Field(min_length=2, max_length=32)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_spec(path: Path) -> tuple[MatrixSpec, str]:
    raw = path.read_bytes()
    value = yaml.load(raw.decode("utf-8"), Loader=_GlossaryLoader)
    spec = MatrixSpec.model_validate(value)
    if spec.version != MATRIX_VERSION:
        raise ValueError(f"Unsupported matrix version: {spec.version}")
    if len(set(spec.chapters)) != len(spec.chapters) or any(c < 0 for c in spec.chapters):
        raise ValueError("Matrix chapters must be unique and nonnegative")
    if not set(spec.vary) <= AXES or len(set(spec.vary)) != len(spec.vary):
        raise ValueError(f"Matrix vary must use unique axes from {sorted(AXES)}")
    if not spec.vary:
        raise ValueError("Matrix must declare at least one varying axis")
    for arm in spec.arms:
        safe_name(arm.id)
    if len({arm.id for arm in spec.arms}) != len(spec.arms):
        raise ValueError("Matrix arm IDs must be unique")
    return spec, sha256(raw)


def _stage(manifest: dict) -> dict:
    return next(
        stage for stage in manifest["stages"] if stage.get("role") in {"translator", "planner"}
    )


def _resolved_id(provenance: dict) -> str | None:
    resolved = provenance.get("resolved") or {}
    if resolved.get("status") == "unknown":
        return None
    if resolved.get("provider") and resolved.get("model"):
        return f"{resolved['provider']}/{resolved['model']}"
    return resolved.get("id")


def _arm_signature(manifest: dict, *, vary: set[str]) -> dict:
    stage = _stage(manifest)
    translation = manifest["translation"]
    provenance = stage.get("execution_provenance") or {}
    material = provenance.get("material") or {}
    signature = {
        "source": [[c["number"], c["source_sha256"]] for c in manifest["chapters"]],
        "glossary_sha256": manifest.get("glossary_sha256"),
        "context": {
            "strategy": translation.get("context_strategy"),
            "composition": translation.get("context_composition"),
            "self_context_budget": translation.get("self_context_budget"),
        },
        "pipeline": manifest.get("workflow"),
        "prompt": stage.get("prompt_sha256"),
        "chunking": {
            "chunk_chars": translation.get("chunk_chars"),
            "overlap_chars": translation.get("overlap_chars"),
        },
        "resolved_model": _resolved_id(provenance),
        "harness": (provenance.get("harness") or {}).get("name"),
        "decoding": {
            "temperature": material.get("temperature"),
            "max_tokens": material.get("max_tokens"),
            "reasoning_effort": material.get("reasoning_effort"),
            "structured_decoding": material.get("structured_decoding"),
        },
    }
    for axis in vary:
        signature.pop(AXIS_SIGNATURE_FIELDS[axis], None)
    return signature


def _check_arm_assertions(arm: MatrixArm, manifest: dict) -> None:
    stage = _stage(manifest)
    config = stage["runner_config"]
    checks = {
        "model": config.get("model"),
        "harness": config.get("type"),
        "endpoint": config.get("base_url"),
        "max_tokens": config.get("max_tokens") or manifest["translation"].get("max_tokens"),
        "timeout_seconds": config.get("timeout_seconds"),
        "retries": manifest["translation"].get("retries"),
        "pipeline": manifest.get("workflow"),
        "prompt_version": stage.get("prompt_version"),
        "chunk_chars": manifest["translation"].get("chunk_chars"),
        "temperature": manifest["translation"].get("temperature"),
    }
    for field, expected in checks.items():
        declared = getattr(arm, field, None)
        if declared is not None and declared != expected:
            raise ValueError(f"Arm {arm.id} {field} does not match frozen run settings")
    if arm.model is not None and arm.model != _resolved_id(stage.get("execution_provenance") or {}):
        raise ValueError(f"Arm {arm.id} model does not match resolved model identity")
    if arm.seed is not None and config.get("extra_params", {}).get("seed") != arm.seed:
        raise ValueError(f"Arm {arm.id} seed is not present in the runner request settings")
    if arm.sampler is not None and config.get("extra_params", {}).get("sampler") != arm.sampler:
        raise ValueError(f"Arm {arm.id} sampler is not present in the runner request settings")


def _plan_dir(service, plan_id: str) -> Path:
    return service.root / "experiment-plans" / plan_id


def _state_path(service, plan_id: str) -> Path:
    return _plan_dir(service, plan_id) / "state.json"


def _set_state(service, plan_id: str, state: str) -> None:
    write_json(
        _state_path(service, plan_id),
        {"plan_id": plan_id, "state": state, "updated_at": _now()},
    )


def _read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Matrix plan is missing or invalid") from error
    if record.get("schema_version") != MATRIX_VERSION:
        raise ValueError("Unsupported frozen matrix plan version")
    return record


def plan_matrix(service, matrix: Path) -> dict:
    """Freeze one source sample and create one ordinary persisted run per arm."""
    spec, source_plan_sha256 = _load_spec(matrix)
    population = {chapter.number for chapter in discover(service.config.novel.source_dir)}
    if not set(spec.chapters) <= population:
        raise ValueError("Matrix chapter is not present in the configured source")
    source_hashes = [
        [chapter.number, chapter.source_sha256]
        for chapter in discover(service.config.novel.source_dir)
        if chapter.number in spec.chapters
    ]
    plan_id = sha256(
        canonical(
            {
                "source_plan_sha256": source_plan_sha256,
                "source_hashes": source_hashes,
                "name": spec.name,
            }
        )
    )[:20]
    path = _plan_dir(service, plan_id) / "plan.json"
    if path.is_file():
        return _read_record(path)
    arms = []
    for arm in spec.arms:
        run_id = service.create_run(
            runner=arm.runner,
            translator=arm.translator,
            editor=arm.editor,
            pipeline=arm.pipeline,
            chapters=sorted(spec.chapters),
            chunk_chars=arm.chunk_chars,
            temperature=arm.temperature,
            prompt_version=arm.prompt_version,
            force=True,
        )
        manifest = service.repository.run(run_id)["manifest"]
        _check_arm_assertions(arm, manifest)
        stage = _stage(manifest)
        arms.append(
            {
                "id": arm.id,
                "run_id": run_id,
                "state": "planned",
                "manifest_sha256": sha256(canonical(manifest)),
                "manifest_fingerprint": manifest["fingerprint"],
                "manifest": {"fingerprint": manifest["fingerprint"]},
                "execution_provenance": stage.get("execution_provenance"),
                "settings": arm.model_dump(mode="json"),
            }
        )
    signatures = [_arm_signature(service.repository.run(a["run_id"])["manifest"], vary=set(spec.vary)) for a in arms]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Matrix arms differ on an undeclared fixed axis")
    record = {
        "schema_version": MATRIX_VERSION,
        "plan_id": plan_id,
        "path": str(path),
        "name": spec.name,
        "source_plan_sha256": source_plan_sha256,
        "created_at": _now(),
        "vary": spec.vary,
        "chapters": sorted(spec.chapters),
        "source_dir": str(service.config.novel.source_dir),
        "source_hashes": source_hashes,
        "arms": arms,
        "frozen_signatures": signatures,
        "state": "planned",
    }
    write_json(path, record)
    return record


def preflight_matrix(service, plan: Path, *, allow_started: bool = False) -> dict:
    """Prove frozen settings before any matrix worker is started."""
    record = _read_record(plan)
    checks = []
    for arm in record["arms"]:
        run = service.repository.run(arm["run_id"])
        manifest = run["manifest"]
        if manifest.get("fingerprint") != arm.get("manifest_fingerprint"):
            raise ValueError(f"Matrix arm {arm['id']} changed after freeze")
        if sha256(canonical(manifest)) != arm.get("manifest_sha256"):
            raise ValueError(f"Matrix arm {arm['id']} manifest drifted after freeze")
        manifest_file = service.run_dir(arm["run_id"]) / "manifest.json"
        try:
            saved_manifest = json.loads(manifest_file.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Matrix arm {arm['id']} manifest artifact is unreadable") from error
        if saved_manifest != manifest:
            raise ValueError(f"Matrix arm {arm['id']} manifest artifact drifted after freeze")
        if _arm_signature(manifest, vary=set(record["vary"])) != record["frozen_signatures"][
            record["arms"].index(arm)
        ]:
            raise ValueError(f"Matrix arm {arm['id']} settings drifted after freeze")
        if not allow_started and any(
            row["status"] != "pending" for row in service.repository.chunks(arm["run_id"])
        ):
            raise ValueError(f"Matrix arm {arm['id']} has already started")
        stage = _stage(manifest)
        provenance = stage.get("execution_provenance") or {}
        resolved = _resolved_id(provenance)
        if not resolved:
            raise ValueError(f"Matrix arm {arm['id']} has unknown resolved model identity")
        config = stage["runner_config"]
        output_cap = config.get("max_tokens") or manifest["translation"]["max_tokens"]
        proof = {"source": "frozen_runner_config", "output_cap": output_cap}
        if config.get("type") == "opencode":
            runner_name = stage["runner"]
            runner_config = service.config.runners[runner_name]
            proof = OpenCodeRunner(
                runner_config, service.root / "jobs"
            ).preflight_effective_settings(output_cap)
            proof["output_cap"] = output_cap
        checks.append(
            {
                "arm": arm["id"],
                "run_id": arm["run_id"],
                "resolved_model": resolved,
                "provenance_completeness": provenance.get("completeness"),
                "effective_settings": {"temperature": manifest["translation"].get("temperature"), **proof},
            }
        )
    result = {"plan_id": record["plan_id"], "status": "ready", "checks": checks}
    write_json(_plan_dir(service, record["plan_id"]) / "preflight.json", result)
    return result


def _arm_status(service, arm: dict) -> dict:
    status = service.status(arm["run_id"])
    return {
        "id": arm["id"],
        "run_id": arm["run_id"],
        "status": status["status"],
        "completed": status["chunks"]["completed"],
        "total": status["total_chunks"],
        "progress": status["progress"],
        "error": status["error"],
    }


def status_matrix(service, plan: Path) -> dict:
    record = _read_record(plan)
    arms = [_arm_status(service, arm) for arm in record["arms"]]
    state = record["state"]
    state_path = _state_path(service, record["plan_id"])
    if state_path.is_file():
        state = json.loads(state_path.read_text("utf-8")).get("state", state)
    return {"plan_id": record["plan_id"], "state": state, "arms": arms}


def run_matrix(
    service,
    config: Path,
    plan: Path,
    *,
    arm_ids: set[str] | None = None,
    allow_started: bool = False,
) -> dict:
    preflight = preflight_matrix(service, plan, allow_started=allow_started)
    record = _read_record(plan)
    selected = [arm for arm in record["arms"] if arm_ids is None or arm["id"] in arm_ids]
    if not selected:
        raise ValueError("No matching matrix arms")
    launches = []
    for arm in selected:
        try:
            launches.append(start_worker(config, arm["run_id"]))
        except Exception as error:  # noqa: BLE001 - isolate one arm from its siblings
            launches.append(
                {
                    "run_id": arm["run_id"],
                    "started": False,
                    "error": type(error).__name__,
                }
            )
    _set_state(service, record["plan_id"], "running")
    return {"preflight": preflight, "launches": launches}


def resume_matrix(service, config: Path, plan: Path) -> dict:
    record = _read_record(plan)
    selected = {
        arm["id"]
        for arm in record["arms"]
        if service.status(arm["run_id"])["status"] != "completed"
    }
    if not selected:
        return {"plan_id": record["plan_id"], "launches": [], "status": "completed"}
    return run_matrix(service, config, plan, arm_ids=selected, allow_started=True)


def export_matrix(service, plan: Path) -> dict:
    record = _read_record(plan)
    arms = []
    for arm in record["arms"]:
        run_id = arm["run_id"]
        run = service.repository.run(run_id)
        attempts = []
        problems = []
        for row in service.repository.chunks(run_id):
            if row["status"] != "completed":
                problems.append(row["chunk_id"])
            for stage in run["manifest"]["stages"]:
                attempts.extend(service.repository.attempts(run_id, row["chapter"], row["chunk_id"], stage["name"]))
        problems.extend(
            attempt["chunk_id"]
            for attempt in attempts
            if attempt.get("status") != "completed"
        )
        finish_reasons = Counter(a.get("finish_reason") or "unknown" for a in attempts)
        outcome_classes = Counter()
        arms.append(
            {
                "id": arm["id"],
                "run_id": run_id,
                "status": service.status(run_id)["status"],
                "attempts": len(attempts),
                "attempt_details": [],
                "finish_reasons": dict(finish_reasons),
                "outcome_classes": dict(outcome_classes),
                "problems": sorted(set(problems)),
                "metrics": service.metrics(run_id),
                "execution_provenance": arm.get("execution_provenance"),
            }
        )
        details = []
        for attempt in attempts:
            failure_kind = attempt.get("failure_kind") or "unknown"
            outcome, retry_policy = FAILURE_POLICIES.get(
                failure_kind, ("failed", "stop_arm")
            )
            if attempt.get("status") == "completed":
                outcome, retry_policy = "completed", "none"
            outcome_classes[outcome] += 1
            details.append(
                {
                    key: attempt.get(key)
                    for key in (
                        "chapter",
                        "chunk_id",
                        "stage",
                        "attempt",
                        "status",
                        "finish_reason",
                        "usage",
                        "output_sha256",
                        "input_sha256",
                        "failure_kind",
                        "execution_provenance",
                    )
                }
                | {"outcome": outcome, "retry_policy": retry_policy}
            )
        arms[-1]["attempt_details"] = details
        arms[-1]["outcome_classes"] = dict(outcome_classes)
    result = {"schema_version": MATRIX_VERSION, "plan_id": record["plan_id"], "arms": arms}
    path = _plan_dir(service, record["plan_id"]) / "export.json"
    write_json(path, result)
    return {"path": str(path), **result}
