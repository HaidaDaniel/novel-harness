"""Preset and execution selection for the public service facade."""

from __future__ import annotations

from .presets import resolve_preset


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
    presets = self.config.translation_presets
    if (
        not presets.auto_select_default
        or preset is not None
        or execution_profile is not None
        or pipeline is not None
        or runner is not None
        or translator is not None
        or editor is not None
        or prompt_version is not None
    ):
        return preset, runner
    default_id = presets.default
    if default_id is None:
        return preset, runner
    # Resolve through the preset so its pipeline and provenance are frozen
    # together. A named runner, including an equivalent alias, is explicit.
    return default_id, None


def _execution_profile_selection(
    self,
    execution_profile: str | None,
    runner: str | None,
    translator: str | None,
    editor: str | None,
    preset: str | None,
) -> str | None:
    if execution_profile is None:
        return runner
    if preset is not None or runner is not None or translator is not None:
        raise ValueError(
            "execution_profile cannot be combined with a preset or explicit runner/translator"
        )
    if execution_profile not in self.config.execution_profiles:
        raise ValueError("Unknown execution profile")
    return execution_profile


def _validate_execution_profile_stage(
    self,
    execution_profile: str | None,
    pipeline: str,
    runner: str | None,
    translator: str | None,
) -> None:
    if execution_profile is None:
        return
    pipeline_config = self.config.pipelines.get(pipeline)
    if pipeline_config is None:
        return  # build_inputs returns the established unknown-pipeline error.
    used = any(
        stage.role in {"translator", "planner"}
        and (stage.runner or translator or runner) == execution_profile
        for stage in pipeline_config.stages
    )
    if not used:
        raise ValueError("execution_profile is not selected by a translator or planner stage")


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
    if preset is None:
        return (
            runner,
            pipeline if pipeline is not None else "direct",
            translator,
            editor,
            None,
        )
    resolved = resolve_preset(self.config, preset)
    if runner is not None and runner != resolved["runner"]:
        raise ValueError("preset conflicts with an explicit runner")
    if pipeline is not None and pipeline != resolved["pipeline"]:
        raise ValueError("preset conflicts with an explicit pipeline")
    if translator is not None and translator != resolved["translator"]:
        raise ValueError("preset conflicts with an explicit translator")
    if editor is not None and editor != resolved["editor"]:
        raise ValueError("preset conflicts with an explicit editor")
    if prompt_version is not None:
        raise ValueError("preset conflicts with an explicit prompt_version")
    return (
        resolved["runner"],
        resolved["pipeline"],
        resolved["translator"],
        resolved["editor"],
        resolved,
    )
