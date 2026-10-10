"""Capability records, inference and compatibility rejection. No provider call."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from novel_harness.capabilities import (
    check_job_compatibility,
    check_runner_compatibility,
    resolve_capabilities,
)
from novel_harness.config import RunnerConfig
from novel_harness.runners.base import TranslationJob


def job_with(**overrides):
    values = {
        "run_id": "r",
        "chapter": 1,
        "chunk_id": "c",
        "stage": "translation",
        "prompt": "p",
        "source": "s",
        "context": {},
        "glossary": {},
        "workspace": Path("."),
    }
    values.update(overrides)
    return TranslationJob(**values)


def test_type_defaults_are_inferred():
    openai = resolve_capabilities(RunnerConfig(type="openai"))
    assert openai.source == "inferred"
    assert openai.supports_structured_output is True
    assert openai.enforces_output_schema is False
    assert openai.supports_reasoning_effort is False

    codex = resolve_capabilities(RunnerConfig(type="codex"))
    assert codex.supports_temperature is False
    assert codex.supports_output_limit is False
    assert codex.enforces_output_schema is True
    assert codex.supports_reasoning_effort is True

    opencode = resolve_capabilities(RunnerConfig(type="opencode"))
    assert opencode.supports_structured_output is True
    assert opencode.enforces_output_schema is False
    assert opencode.supports_reasoning_effort is True


def test_unknown_runner_type_is_conservative():
    capabilities = resolve_capabilities(RunnerConfig(type="gemini"))
    assert capabilities.source == "inferred"
    assert capabilities.supports_temperature is False
    assert capabilities.supports_structured_output is False
    assert capabilities.supports_reasoning_effort is False


def test_declared_overrides_inferred_values():
    profile = RunnerConfig(
        type="openai",
        capabilities={"supports_tools": False, "context_limit": 4096},
    )
    capabilities = resolve_capabilities(profile)
    assert capabilities.source == "declared"
    assert capabilities.supports_tools is False
    assert capabilities.context_limit == 4096
    assert capabilities.supports_structured_output is True


def test_empty_declaration_stays_inferred():
    capabilities = resolve_capabilities(RunnerConfig(type="openai", capabilities={}))
    assert capabilities.source == "inferred"


def test_runner_compatibility_rejects_unsupported_parameters():
    with pytest.raises(ValueError, match="temperature"):
        check_runner_compatibility(RunnerConfig(type="codex", temperature=0.5))
    with pytest.raises(ValueError, match="max_tokens"):
        check_runner_compatibility(RunnerConfig(type="codex", max_tokens=100))
    with pytest.raises(ValueError, match="reasoning_effort"):
        check_runner_compatibility(RunnerConfig(type="openai", reasoning_effort="high"))

    check_runner_compatibility(RunnerConfig(type="codex", reasoning_effort="high"))
    check_runner_compatibility(RunnerConfig(type="openai", temperature=0.2, max_tokens=100))


def test_job_structured_output_rejected_only_when_declared_unsupported():
    job = job_with(response_schema={"type": "object"})
    check_job_compatibility(RunnerConfig(type="openai"), job)
    check_job_compatibility(RunnerConfig(type="codex"), job)
    check_job_compatibility(RunnerConfig(type="opencode"), job)

    with pytest.raises(ValueError, match="structured output"):
        check_job_compatibility(
            RunnerConfig(type="opencode", capabilities={"supports_structured_output": False}),
            job,
        )


def test_malformed_capability_data_fails_closed():
    with pytest.raises(ValidationError):
        RunnerConfig(type="openai", capabilities={"context_limit": -5})
    with pytest.raises(ValidationError):
        RunnerConfig(type="openai", capabilities={"supports_temperature": 1})
    with pytest.raises(ValidationError):
        RunnerConfig(type="openai", capabilities={"supports_temperature": "not-a-bool"})
    with pytest.raises(ValidationError):
        RunnerConfig(type="openai", capabilities={"enforces_output_schema": "yes"})


@pytest.mark.parametrize("mode", ["json-schema", "llama-grammar"])
def test_explicit_structured_decoding_reports_enforcement(mode):
    for kind in ["openai", "opencode"]:
        profile = RunnerConfig(type=kind, structured_decoding=mode)
        assert resolve_capabilities(profile).enforces_output_schema is True
    with pytest.raises(ValueError, match="requires an openai or opencode"):
        RunnerConfig(type="codex", structured_decoding=mode)
