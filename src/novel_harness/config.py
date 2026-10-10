"""Strict configuration. No environment mutation and no persisted credentials."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)


_SECRET_PARAM_NAMES = {
    "auth",
    "authorization",
    "credential",
    "credentials",
    "key",
    "password",
    "passwd",
    "secret",
    "token",
}


def _validate_public_extra_params(value: object) -> None:
    """Reject secret-named fields before extra parameters enter public artifacts."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise TypeError("extra_params keys must be strings")
                separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
                separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
                if _SECRET_PARAM_NAMES.intersection(re.split(r"[^A-Za-z0-9]+", separated.lower())):
                    raise ValueError("extra_params must not contain secret-bearing fields")
                pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)


class NovelConfig(StrictModel):
    title: str = "Japanese web novel"
    author: str | None = None
    metadata_path: Path | None = None
    source_dir: Path = Path("original")
    library_roots: list[Path] | None = None
    chapter_pattern: str = "*.txt"
    output_dir: Path = Path("outputs")
    books_dir: Path = Path("books")
    jobs_dir: Path = Path("jobs")
    # Export-only settings. `spread: auto` is the standard default;
    # `none` only asks the reading system not to form a synthetic spread (the EPUB
    # stays reflowable; readers may still use their own columns/pagination). The
    # cover is optional.
    spread: Literal["auto", "none"] = "auto"
    cover_path: Path | None = None

    @field_validator("chapter_pattern")
    @classmethod
    def safe_chapter_pattern(cls, value: str) -> str:
        # Kept local to config to avoid importing the library scanner here.
        if not re.fullmatch(r"[A-Za-z0-9_*?.-]+\.txt", value, re.IGNORECASE) or (
            ".." in value or "**" in value
        ):
            raise ValueError("chapter_pattern must be a safe non-recursive .txt glob")
        return value


class RoutingConfig(StrictModel):
    """Deterministic escalation policy; disabled by default (local-first)."""

    enabled: bool = False
    escalation_runner: str | None = None
    escalate_over_chars: int = Field(default=3000, ge=1)
    escalate_over_names: int = Field(default=8, ge=1)


class TranslationConfig(StrictModel):
    source_language: Literal["ja"] = "ja"
    target_language: Literal["ru"] = "ru"
    chunk_chars: int = Field(default=6000, ge=32, le=100000)
    overlap_chars: int = Field(default=1000, ge=0, le=10000)
    context_strategy: Literal["none", "previous"] = "previous"
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=8192, gt=0)
    prompt_version: str = "translate-v1"
    prompts_dir: Path = Path("prompts")
    glossary_path: Path = Path("glossary.yaml")
    retries: int = Field(default=2, ge=0, le=20)
    retry_delay_seconds: float = Field(default=1, ge=0, le=300)
    name_rules: list[dict[str, Any]] = Field(default_factory=list)
    name_rules_path: Path | None = None
    output_format: Literal["text", "segments-v1"] = "text"
    book_context_path: Path | None = None
    book_context_chars: int = Field(default=6000, ge=256, le=30000)
    self_context_path: Path | None = None
    context_composition: Literal[
        "none", "self_only", "canonical_human_only", "self_plus_shared_baseline"
    ] = "none"
    self_context_budget: dict[str, Any] = Field(default_factory=dict)
    cross_chapter_context: bool = False
    context_limit: int | None = Field(default=None, ge=256)
    context_safety_margin: float = Field(default=0.15, ge=0, le=0.9)
    chars_per_token: float = Field(default=2.0, gt=0, le=10)
    cache_enabled: bool = False
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @model_validator(mode="after")
    def valid_name_rules(self):
        from .translation.names import validate_name_rules

        validate_name_rules(self.name_rules)
        return self

    @model_validator(mode="after")
    def valid_context_composition(self):
        allowed = {
            "none",
            "self_only",
            "canonical_human_only",
            "self_plus_shared_baseline",
        }
        if self.context_composition not in allowed:
            raise ValueError("Unknown context_composition")
        if self.self_context_path is None:
            if self.context_composition in {"self_only", "self_plus_shared_baseline"}:
                raise ValueError("self context composition requires self_context_path")
        elif self.context_composition not in {"self_only", "self_plus_shared_baseline"}:
            raise ValueError("self_context_path requires self_only/self_plus_shared_baseline")
        return self


class ValidationConfig(StrictModel):
    min_output_chars: int = Field(default=10, ge=1)
    min_length_ratio: float = Field(default=0.3, gt=0)
    max_length_ratio: float = Field(default=4, gt=0)
    max_japanese_ratio: float = Field(default=0.05, ge=0, le=1)
    repetition_count: int = Field(default=3, ge=2)
    # Explicit opt-in strict JA->RU policy. ``None``/``False`` keep legacy behavior
    # for old manifests that lack these fields.
    max_foreign_script_chars: int | None = Field(default=None, ge=0)
    reject_malformed_dialogue: bool = False
    # Corruption-aware deterministic gate (opt-in). False keeps legacy behavior for
    # manifests that lack the fields. These detect observable corruption only, not
    # semantic/pseudo-Russian degeneration.
    reject_service_artifacts: bool = False
    reject_segment_collapse: bool = False
    reject_missing_cyrillic: bool = False

    @model_validator(mode="after")
    def ratios(self):
        if self.max_length_ratio <= self.min_length_ratio:
            raise ValueError("max_length_ratio must exceed min_length_ratio")
        return self


class CapabilityConfig(StrictModel):
    """Optional declared overrides; unset fields are inferred from the runner type.

    ``strict`` makes malformed declared data (for example an integer where a bool
    is expected) fail closed instead of being coerced.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    context_limit: int | None = Field(default=None, ge=256)
    supports_temperature: bool | None = None
    supports_output_limit: bool | None = None
    supports_structured_output: bool | None = None
    enforces_output_schema: bool | None = None
    supports_tools: bool | None = None
    supports_reasoning_effort: bool | None = None


class RunnerConfig(StrictModel):
    type: str = "openai"
    model: str | None = None
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr | None = Field(default=None, exclude=True)
    api_key_env: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=180, gt=0, le=86400)
    extra_params: dict[str, Any] = Field(default_factory=dict)
    opencode_variant: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
    )
    profile: str | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    opencode_transport: Literal["cli", "server"] = "cli"
    harness_managed: bool = False
    opencode_agent_id: str | None = None
    opencode_config_path: Path | None = Field(default=None, exclude=True)
    opencode_config_sha256: str | None = None
    execution_profile_id: str | None = None
    requested_model_policy: Literal["inherit", "override"] | None = None
    capabilities: CapabilityConfig | None = None
    structured_decoding: Literal["prompt", "json-schema", "llama-grammar"] = "prompt"

    def public_settings(self) -> dict:
        value = self.model_dump(mode="json")
        # Omitted temperature inherits the translation default; explicit null suppresses it.
        if "temperature" not in self.model_fields_set:
            value.pop("temperature")
        if "opencode_transport" not in self.model_fields_set:
            value.pop("opencode_transport")
        if self.opencode_variant is None:
            value.pop("opencode_variant")
        value.pop("opencode_config_path", None)
        if not self.harness_managed:
            value.pop("harness_managed", None)
        for field in (
            "opencode_agent_id",
            "opencode_config_sha256",
            "execution_profile_id",
            "requested_model_policy",
        ):
            if getattr(self, field) is None:
                value.pop(field, None)
        if self.harness_managed:
            # Harness-owned credentials and config paths stay in OpenCode. These
            # fields describe only the safe execution reference.
            value.pop("base_url", None)
            value.pop("api_key_env", None)
            value.pop("api_key", None)
        return value

    @model_validator(mode="after")
    def safe_request(self):
        if self.structured_decoding != "prompt" and self.type not in {"openai", "opencode"}:
            raise ValueError("structured_decoding requires an openai or opencode runner")
        if self.opencode_variant is not None:
            if self.type != "opencode":
                raise ValueError("opencode_variant requires an OpenCode runner")
            if self.reasoning_effort is not None:
                raise ValueError("Choose either opencode_variant or reasoning_effort")
        if not self.harness_managed:
            parts = urlsplit(self.base_url)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError("base_url must be an HTTP(S) URL")
            if parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("base_url must not contain credentials, query or fragment")
        reserved = {
            "messages",
            "model",
            "stream",
            "n",
            "tools",
            "tool_choice",
            "api_key",
            "authorization",
            "headers",
            "temperature",
            "max_tokens",
        }
        if reserved.intersection(self.extra_params):
            raise ValueError("extra_params contains reserved request fields")
        _validate_public_extra_params(self.extra_params)
        return self


class DirectExecutionProfile(StrictModel):
    """Versioned direct-provider profile; its only secret reference is an env name."""

    version: Literal[1] = 1
    kind: Literal["direct"]
    model: str = Field(min_length=1)
    base_url: str
    timeout_seconds: float = Field(default=180, gt=0, le=86400)
    api_key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_direct_runner(self):
        # Reuse the legacy runner's endpoint and reserved-parameter validation.
        RunnerConfig(
            type="openai",
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_params=self.extra_params,
        )
        return self


class HarnessExecutionProfile(StrictModel):
    """Secret-free pointer to a locally configured harness agent/profile."""

    version: Literal[1] = 1
    kind: Literal["harness"]
    harness: Literal["opencode", "codex"]
    agent_id: str | None = None
    profile: str | None = None
    model_policy: Literal["inherit", "override"] = "inherit"
    model_override: str | None = None

    @model_validator(mode="after")
    def valid_harness_reference(self):
        if self.harness == "opencode":
            if self.agent_id is None:
                raise ValueError("OpenCode profile requires agent_id")
            safe_name(self.agent_id)
            if self.profile is not None:
                raise ValueError("OpenCode profile does not accept a CLI profile")
        else:
            if self.agent_id is not None:
                raise ValueError("Codex profile does not accept agent_id")
            if self.profile is not None:
                safe_name(self.profile)
        if self.model_policy == "override":
            if not self.model_override:
                raise ValueError("model_override is required for override policy")
            if self.harness == "opencode":
                _validate_model_reference(self.model_override)
            elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,199}", self.model_override):
                raise ValueError("Codex model override is invalid")
        elif self.model_override is not None:
            raise ValueError("model_override requires model_policy: override")
        return self


def _validate_model_reference(value: str) -> tuple[str, str, str | None]:
    """Validate an explicit provider/model ID without probing a provider."""
    if not isinstance(value, str) or value.count("/") < 1:
        raise ValueError("Model must use provider/model syntax")
    provider, model = value.split("/", 1)
    variant = None
    if "#" in model:
        model, variant = model.rsplit("#", 1)
        if not variant or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", variant):
            raise ValueError("Model variant is invalid")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", provider)
        or not model
        or len(model) > 200
        or any(char.isspace() or ord(char) < 32 for char in model)
        or model in {".", ".."}
        or "\\" in model
    ):
        raise ValueError("Model must use a valid provider/model ID")
    return provider, model, variant


ExecutionProfile = Annotated[
    DirectExecutionProfile | HarnessExecutionProfile,
    Field(discriminator="kind"),
]


class StageConfig(StrictModel):
    name: str
    #: ``planner`` is the bounded semantic-plan-v1 pre-translation stage; it must
    #: be the first stage and is followed by exactly the translator stage.
    role: Literal["planner", "translator", "editor", "reviewer"]
    prompt_version: str | None = None
    runner: str | None = None


class PipelineConfig(StrictModel):
    stages: list[StageConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_stages(self):
        roles = [stage.role for stage in self.stages]
        if "translator" not in roles:
            raise ValueError("Pipeline requires a translator stage")
        if roles[0] not in {"planner", "translator"}:
            raise ValueError("First stage must be translator or planner")
        if "planner" in roles and roles[0] != "planner":
            raise ValueError("A planner stage must be first")
        if roles.count("planner") > 1:
            raise ValueError("At most one planner stage")
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError("Stage names must be unique")
        for stage in self.stages:
            safe_name(stage.name)
            if stage.prompt_version is not None:
                safe_name(stage.prompt_version)
        return self


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def default_pipelines():
    return {
        "direct": PipelineConfig(
            stages=[
                StageConfig(name="translation", role="translator", prompt_version="translate-v1")
            ]
        )
    }


class TranslationPreset(StrictModel):
    """A named, config-backed translation mode over raw runner/pipeline settings.

    Presets are run-creation convenience only: no credentials or model identities are
    stored here, and the resolved runner/pipeline are frozen into each run manifest.
    """

    label: str
    description: str = ""
    runner: str
    pipeline: str
    translator: str | None = None
    editor: str | None = None
    category: str = "custom"
    cloud: bool = False
    recommended: bool = False


class TranslationPresetConfig(StrictModel):
    default: str | None = None
    auto_select_default: bool = False
    presets: dict[str, TranslationPreset] = Field(default_factory=dict)


class AppConfig(StrictModel):
    novel: NovelConfig = Field(default_factory=NovelConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    runners: dict[str, RunnerConfig]
    execution_profiles: dict[str, ExecutionProfile] = Field(default_factory=dict)
    pipelines: dict[str, PipelineConfig] = Field(default_factory=default_pipelines)
    #: Optional user-facing translation modes. Empty by default so a
    #: config without presets keeps raw runner/pipeline behavior.
    translation_presets: TranslationPresetConfig = Field(
        default_factory=TranslationPresetConfig
    )
    _env: dict[str, str | None] = PrivateAttr(default_factory=dict)
    _profile_resolution: dict[str, dict] = PrivateAttr(default_factory=dict)
    _config_path: Path | None = PrivateAttr(default=None)

    @property
    def config_path(self) -> Path | None:
        """Absolute source path when this config was loaded from disk."""
        return self._config_path

    @model_validator(mode="after")
    def valid_translation_presets(self):
        # Pipeline references are structural config errors; runner references are
        # deliberately not validated here so an optional preset whose runner is absent
        # is reported as unavailable instead of crashing the whole server.
        for preset_id, preset in self.translation_presets.presets.items():
            safe_name(preset_id)
            safe_name(preset.pipeline)
            if preset.pipeline not in self.pipelines:
                raise ValueError(
                    f"translation_presets.{preset_id} references unknown pipeline: "
                    f"{preset.pipeline}"
                )
            for field in ("runner", "translator", "editor"):
                value = getattr(preset, field)
                if value is not None:
                    safe_name(value)
        default = self.translation_presets.default
        if self.translation_presets.auto_select_default and default is None:
            raise ValueError("translation_presets.auto_select_default requires a default preset")
        if default is not None and default not in self.translation_presets.presets:
            raise ValueError("translation_presets.default must name a configured preset")
        for profile_id in self.execution_profiles:
            safe_name(profile_id)
            if profile_id in self.runners:
                raise ValueError(
                    f"execution_profiles.{profile_id} conflicts with a runner name"
                )
        return self

    def key_for(self, name: str) -> str:
        runner = self.runners[name]
        if runner.api_key_env:
            key = os.environ.get(runner.api_key_env, self._env.get(runner.api_key_env))
            if not key:
                raise ValueError(f"Set {runner.api_key_env} in environment or .env")
            return key
        return runner.api_key.get_secret_value() if runner.api_key else "EMPTY"


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        profiles = raw.get("execution_profiles", {})
        runners = raw.get("runners", {})
        if isinstance(profiles, dict) and isinstance(runners, dict):
            # AppConfig.model_dump includes compiled runner aliases. Drop only an
            # alias carrying its matching generated profile ID; its source profile
            # is revalidated and materialized below.
            for profile_id in profiles:
                alias = runners.get(profile_id)
                if (
                    isinstance(alias, dict)
                    and alias.get("execution_profile_id") == profile_id
                ):
                    runners.pop(profile_id)
    config = AppConfig.model_validate(raw)
    for runner in config.runners.values():
        if (
            runner.harness_managed
            or runner.opencode_agent_id is not None
            or runner.opencode_config_path is not None
            or runner.opencode_config_sha256 is not None
            or runner.execution_profile_id is not None
            or runner.requested_model_policy is not None
        ):
            raise ValueError(
                "Harness-managed runtime fields are generated from execution_profiles"
            )
    root = path.parent
    for name in ("source_dir", "output_dir", "books_dir", "jobs_dir"):
        setattr(config.novel, name, (root / getattr(config.novel, name)).resolve())
    if config.novel.library_roots is not None:
        if not config.novel.library_roots:
            raise ValueError("novel.library_roots must contain at least one root")
        config.novel.library_roots = [
            (root / item).resolve() for item in config.novel.library_roots
        ]
        if len(set(config.novel.library_roots)) != len(config.novel.library_roots):
            raise ValueError("novel.library_roots contains duplicate roots")
    if config.novel.metadata_path is not None:
        config.novel.metadata_path = (root / config.novel.metadata_path).resolve()
    if config.novel.cover_path is not None:
        config.novel.cover_path = (root / config.novel.cover_path).resolve()
    for name in ("prompts_dir", "glossary_path"):
        setattr(config.translation, name, (root / getattr(config.translation, name)).resolve())
    for name in ("book_context_path", "name_rules_path", "self_context_path"):
        value = getattr(config.translation, name)
        if value is not None:
            setattr(config.translation, name, (root / value).resolve())
    if config.translation.name_rules_path is not None:
        from .translation.names import validate_name_rules

        if config.translation.name_rules:
            raise ValueError("Choose name_rules or name_rules_path, not both")
        rules = yaml.load(
            config.translation.name_rules_path.read_text("utf-8"), Loader=_GlossaryLoader
        )
        validate_name_rules(rules)
        config.translation.name_rules = rules
    source = config.novel.source_dir
    writes = [config.novel.output_dir, config.novel.books_dir, config.novel.jobs_dir]
    reads = [
        source,
        *(config.novel.library_roots or []),
        config.translation.prompts_dir,
        config.translation.glossary_path,
        path,
    ]
    if config.novel.metadata_path is not None:
        reads.append(config.novel.metadata_path)
    if config.novel.cover_path is not None:
        reads.append(config.novel.cover_path)
    if config.translation.book_context_path is not None:
        reads.append(config.translation.book_context_path)
    if config.translation.name_rules_path is not None:
        reads.append(config.translation.name_rules_path)
    if config.translation.self_context_path is not None:
        reads.append(config.translation.self_context_path)
    for target in writes:
        for read in reads:
            if target.is_relative_to(read) or read.is_relative_to(target):
                raise ValueError("Output/jobs/books paths must be separate from input paths")
    for i, target in enumerate(writes):
        for other in writes[i + 1 :]:
            if target.is_relative_to(other) or other.is_relative_to(target):
                raise ValueError("Output/jobs/books directories must not overlap")
    config._env = dict(dotenv_values(root / ".env"))
    if config.execution_profiles:
        from .connections import materialize_execution_profiles

        materialize_execution_profiles(config)
    for name in [*config.runners, *config.pipelines, config.translation.prompt_version]:
        safe_name(name)
    escalation = config.translation.routing.escalation_runner
    if escalation and escalation not in config.runners:
        raise ValueError("routing.escalation_runner must name a configured runner")
    config._config_path = path
    return config


def validate_glossary(value: object) -> dict:
    """Reject lossy text before it becomes a frozen model input."""
    sections = {"characters", "locations", "organizations", "terms", "titles"}
    if not isinstance(value, dict) or set(value) - sections:
        raise ValueError("Glossary must contain characters/locations/organizations/terms/titles")
    if any(not isinstance(v, dict) for v in value.values()):
        raise ValueError("Each glossary section must be a mapping")
    for section, entries in value.items():
        for key, translation in entries.items():
            for text in (key, translation):
                if (
                    not isinstance(text, str)
                    or not any(c.isalnum() for c in text)
                    or "\ufffd" in text
                    or "??" in text
                    or any(ord(c) < 32 for c in text)
                ):
                    raise ValueError(f"Invalid glossary text in {section}; repair UTF-8 entries")
    return {name: value.get(name, {}) for name in sorted(sections)}


class _GlossaryLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(not isinstance(key, str) for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Glossary keys must be unique strings")
        return super().construct_mapping(node, deep=deep)


def load_glossary(path: Path) -> dict:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_GlossaryLoader)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError("Glossary must be valid UTF-8 YAML") from error
    return validate_glossary({} if value is None else value)
