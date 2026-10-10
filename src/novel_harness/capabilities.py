"""Model/provider capability records.

Capabilities are declared in config and otherwise inferred from the runner type.
Nothing here performs network I/O or model inference; an unsupported parameter is
rejected before any request is dispatched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import RunnerConfig

_BASE = {
    "context_limit": None,
    "supports_temperature": True,
    "supports_output_limit": True,
    # The adapter can carry and host-validate the structured job contract.
    "supports_structured_output": True,
    # The provider/CLI enforces the schema itself (for example Codex --output-schema).
    "enforces_output_schema": False,
    "supports_tools": True,
    "supports_reasoning_effort": False,
}

_TYPE_DEFAULTS: dict[str, dict] = {
    "openai": dict(_BASE),
    "codex": {
        **_BASE,
        "supports_temperature": False,
        "supports_output_limit": False,
        "enforces_output_schema": True,
        "supports_tools": False,
        "supports_reasoning_effort": True,
    },
    "opencode": {
        **_BASE,
        "supports_reasoning_effort": True,
    },
    # Unknown/typo'd runner types fail closed: no optional parameter is assumed.
    "_unknown": {
        **_BASE,
        "supports_temperature": False,
        "supports_output_limit": False,
        "supports_structured_output": False,
        "supports_tools": False,
        "supports_reasoning_effort": False,
    },
}


@dataclass(frozen=True)
class ModelCapabilities:
    context_limit: int | None
    supports_temperature: bool
    supports_output_limit: bool
    supports_structured_output: bool
    enforces_output_schema: bool
    supports_tools: bool
    supports_reasoning_effort: bool
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_capabilities(profile: RunnerConfig) -> ModelCapabilities:
    """Merge declared overrides over runner-type defaults."""
    values = dict(_TYPE_DEFAULTS.get(profile.type, _TYPE_DEFAULTS["_unknown"]))
    source = "inferred"
    if profile.structured_decoding != "prompt":
        values["enforces_output_schema"] = True
    if profile.capabilities is not None:
        declared = profile.capabilities.model_dump(exclude_none=True)
        if declared:
            source = "declared"
            values.update(declared)
    return ModelCapabilities(source=source, **values)


def check_runner_compatibility(profile: RunnerConfig) -> None:
    """Reject config-level parameters a provider does not support."""
    capabilities = resolve_capabilities(profile)
    unsupported = []
    if profile.temperature is not None and not capabilities.supports_temperature:
        unsupported.append("temperature")
    if profile.max_tokens is not None and not capabilities.supports_output_limit:
        unsupported.append("max_tokens")
    if profile.reasoning_effort is not None and not capabilities.supports_reasoning_effort:
        unsupported.append("reasoning_effort")
    if unsupported:
        raise ValueError(
            f"Runner type '{profile.type}' does not support: {', '.join(sorted(unsupported))}"
        )


def check_job_compatibility(profile: RunnerConfig, job) -> None:
    """Reject job-level parameters a provider does not support.

    Job temperature is intentionally not checked here: adapters such as Codex
    accept the field at the job level and ignore it, while output format is a
    hard contract that must be rejected before dispatch.
    """
    capabilities = resolve_capabilities(profile)
    if job.response_schema is not None and not capabilities.supports_structured_output:
        raise ValueError(f"Runner type '{profile.type}' does not support structured output")
