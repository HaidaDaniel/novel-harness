"""User-facing translation presets.

A preset is a bounded, config-backed name for a runner + pipeline pair. It is a
run-creation convenience only: the resolved runner/pipeline and the preset identity
are frozen into the run manifest, so editing a preset later can never reinterpret an
existing run.

No credentials or model identities are stored in preset metadata; the model shown in
listings is read from the resolved runner configuration. Availability is a
configuration-only check (the existing :func:`runner_info` contract); it never probes
or calls a model endpoint.
"""

from __future__ import annotations

from .config import AppConfig
from .runners import runner_info

#: Provenance block keys copied into the frozen run manifest.
_PRESET_BLOCK_KEYS = (
    "preset_id",
    "label",
    "version",
    "category",
    "cloud",
    "runner",
    "pipeline",
    "translator",
    "editor",
)


def _pipeline_stage_runner(config: AppConfig, preset_id: str, stage) -> str | None:
    """Resolve the runner a pipeline stage would actually use at run creation.

    Mirrors :func:`run_factory.build_inputs`: an explicit ``stage.runner`` wins;
    translator/planner stages fall back to the preset translator (else the primary
    runner); every other role (editor/reviewer) falls back to the preset editor.
    """
    preset = config.translation_presets.presets[preset_id]
    if stage.runner is not None:
        return stage.runner
    if stage.role in {"translator", "planner"}:
        return preset.translator or preset.runner
    return preset.editor


def _required_runners(config: AppConfig, preset_id: str) -> list[str]:
    """Every runner the resolved preset/pipeline needs, in deterministic order."""
    preset = config.translation_presets.presets[preset_id]
    required: list[str] = [preset.runner]
    for name in (preset.translator, preset.editor):
        if name is not None and name not in required:
            required.append(name)
    for stage in config.pipelines[preset.pipeline].stages:
        name = _pipeline_stage_runner(config, preset_id, stage)
        if name is not None:
            required.append(name)
    # Preserve first-seen order without duplicating.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in required:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def preset_availability(config: AppConfig, preset_id: str) -> tuple[bool, str | None]:
    """Resolve whether a preset can be used with the current configuration.

    Checks EVERY runner the resolved preset/pipeline requires (primary runner,
    optional translator/editor and the effective runner of every pipeline stage,
    including implicit editor/reviewer fallbacks). Never substitutes another runner;
    availability is configuration-only. A required runner with no configured model is
    treated as unavailable so a placeholder cannot masquerade as a measured identity.
    """
    preset = config.translation_presets.presets[preset_id]
    if preset.pipeline not in config.pipelines:
        return False, "configured pipeline missing"
    for stage in config.pipelines[preset.pipeline].stages:
        if _pipeline_stage_runner(config, preset_id, stage) is None:
            return False, f"required stage '{stage.name}' has no runner configured"
    for name in _required_runners(config, preset_id):
        role = _runner_role(config, preset_id, name)
        if name not in config.runners:
            if name == preset.runner:
                return False, "configured runner missing"
            return False, f"required {role} runner '{name}' is not configured"
        if not config.runners[name].model:
            reason = "Model ID not configured"
            if name == preset.runner:
                return False, reason
            return False, f"required {role} runner '{name}' unavailable: {reason}"
        info = runner_info(config, name)
        if not info["available"]:
            reason = info.get("reason") or "runner unavailable"
            if name == preset.runner:
                return False, reason
            return False, f"required {role} runner '{name}' unavailable: {reason}"
    return True, None


def _runner_role(config: AppConfig, preset_id: str, name: str) -> str:
    preset = config.translation_presets.presets[preset_id]
    if name == preset.translator:
        return "translator"
    if name == preset.editor:
        return "editor"
    for stage in config.pipelines[preset.pipeline].stages:
        if stage.runner == name:
            return f"stage '{stage.name}'"
    return "primary"


def preset_entries(config: AppConfig) -> list[dict]:
    """Resolved preset listings; never calls a model endpoint."""
    default_id = config.translation_presets.default
    entries: list[dict] = []
    for preset_id, preset in config.translation_presets.presets.items():
        available, reason = preset_availability(config, preset_id)
        model = None
        if preset.runner in config.runners:
            model = config.runners[preset.runner].model
        entries.append(
            {
                "id": preset_id,
                "label": preset.label,
                "description": preset.description,
                "runner": preset.runner,
                "model": model,
                "pipeline": preset.pipeline,
                "translator": preset.translator,
                "editor": preset.editor,
                "category": preset.category,
                "cloud": preset.cloud,
                "recommended": preset.recommended,
                "default": preset_id == default_id,
                "available": available,
                "reason": reason,
            }
        )
    return entries


def default_preset_id(config: AppConfig) -> str | None:
    return config.translation_presets.default


def resolve_preset(config: AppConfig, preset_id: str) -> dict:
    """Resolve a preset to concrete runner/pipeline settings, failing closed.

    The returned mapping is also the frozen provenance block recorded in the run
    manifest (with ``preset_id``/``label``/``version`` added).
    """
    preset = config.translation_presets.presets.get(preset_id)
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_id}")
    available, reason = preset_availability(config, preset_id)
    if not available:
        raise ValueError(f"Preset {preset_id} is unavailable: {reason}")
    resolved = {
        "preset_id": preset_id,
        "label": preset.label,
        "version": 1,
        "category": preset.category,
        "cloud": preset.cloud,
        "runner": preset.runner,
        "pipeline": preset.pipeline,
        "translator": preset.translator,
        "editor": preset.editor,
    }
    return {key: resolved[key] for key in _PRESET_BLOCK_KEYS}
