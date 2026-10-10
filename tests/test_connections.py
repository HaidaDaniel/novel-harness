"""Offline, redacted connection-profile discovery tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from conftest import FakeRunner

from novel_harness.config import (
    DirectExecutionProfile,
    HarnessExecutionProfile,
    RunnerConfig,
    load_config,
)
from novel_harness.connections import (
    discover_opencode,
    import_opencode_profile,
    opencode_config_unchanged,
    resolve_opencode_agent,
)
from novel_harness.runners.codex import CodexRunner
from novel_harness.runners.opencode import OpenCodeRunner
from novel_harness.service import TranslationService


def _opencode_fixture(tmp_path: Path, monkeypatch, document: str) -> Path:
    config_dir = tmp_path / "opencode"
    config_dir.mkdir(exist_ok=True)
    config = config_dir / "opencode.jsonc"
    config.write_text(document, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("novel_harness.connections.shutil.which", lambda _: "/usr/bin/opencode")
    monkeypatch.setattr(
        "novel_harness.connections.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "1.18.31\n", ""),
    )
    return config


def test_opencode_import_resolves_inherited_model_and_redacts_config(tmp_path, monkeypatch):
    config = _opencode_fixture(
        tmp_path,
        monkeypatch,
        '''{
          // Provider secrets are owned by OpenCode and never projected.
          "provider": {"private": {"options": {"apiKey": "CANARY-SECRET"},
                                    "models": {"model-a": {"variants": {"high": {}}},
                                               "model-fallback": {}}}},
          "model": "private/model-fallback",
          "agent": {
            "translator": {"mode": "primary", "model": "private/model-a",},
            "inheritor": {"mode": "primary",},
          },
        }''',
    )

    discovered = discover_opencode()
    assert discovered["available"] is True
    agents = {item["agent_id"]: item for item in discovered["agents"]}
    assert agents["translator"]["model"] == {
        "provider": "private",
        "model": "model-a",
        "variant": None,
        "source": "agent",
    }
    assert agents["inheritor"]["model"]["source"] == "global_default"
    imported = import_opencode_profile("opencode-translator", "translator")
    assert imported["profile"]["model_policy"] == "inherit"
    assert imported["model_source"] == "inherited from OpenCode agent"
    from novel_harness.connections import _profile_runner

    inherited_profile = HarnessExecutionProfile.model_validate(
        {"kind": "harness", "harness": "opencode", "agent_id": "inheritor"}
    )
    inherited_runner, inherited_snapshot = _profile_runner(
        "opencode-inherited", inherited_profile
    )
    assert inherited_runner.model == "private/model-fallback"
    assert inherited_snapshot["requested_model"] == {"policy": "inherit"}
    assert inherited_snapshot["resolved_model"] == {
        "provider": "private",
        "model": "model-fallback",
        "variant": None,
        "source": "global_default",
    }
    assert inherited_snapshot["model_source"] == "inherited from OpenCode global default"
    assert "CANARY-SECRET" not in json.dumps(discovered)
    assert "CANARY-SECRET" not in json.dumps(imported)
    assert opencode_config_unchanged(config, imported["config_sha256"])
    config.write_text(config.read_text("utf-8").replace("model-a", "model-b"), "utf-8")
    assert not opencode_config_unchanged(config, imported["config_sha256"])


def test_advanced_override_freezes_requested_and_resolved_model_without_secrets(
    project, source, tmp_path, monkeypatch
):
    _opencode_fixture(
        tmp_path,
        monkeypatch,
        json.dumps(
            {
                "provider": {
                    "private": {
                        "options": {"apiKey": "CANARY-SECRET"},
                        "models": {"model-a": {}, "model-b": {}},
                    }
                },
                "agent": {"translator": {"mode": "primary", "model": "private/model-a"}},
            }
        ),
    )
    imported = import_opencode_profile(
        "opencode-override", "translator",
        model_policy="override", model_override="private/model-b",
    )
    from novel_harness.connections import _profile_runner

    profile = HarnessExecutionProfile.model_validate(imported["profile"])
    runner, snapshot = _profile_runner("opencode-override", profile)
    assert runner.model == "private/model-b"
    assert snapshot["model_policy"] == "override"
    assert snapshot["requested_model"] == {"id": "private/model-b"}
    assert snapshot["resolved_model"] == {
        "provider": "private", "model": "model-b", "variant": None,
        "source": "advanced_override",
    }
    assert snapshot["model_source"] == "explicit advanced override"
    assert "CANARY-SECRET" not in json.dumps({"import": imported, "snapshot": snapshot})
    config_data = yaml.safe_load(project.read_text("utf-8"))
    config_data["execution_profiles"] = {"opencode-override": imported["profile"]}
    project.write_text(yaml.safe_dump(config_data, allow_unicode=True), "utf-8")
    service = TranslationService(load_config(project), lambda _name: FakeRunner())
    run_id = service.create_run(execution_profile="opencode-override", chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    frozen_stage = manifest["stages"][0]
    assert frozen_stage["runner"] == "opencode-override"
    assert frozen_stage["execution"]["requested_model"] == snapshot["requested_model"]
    assert frozen_stage["execution"]["resolved_model"] == snapshot["resolved_model"]
    assert frozen_stage["execution"]["model_source"] == snapshot["model_source"]
    assert frozen_stage["execution"]["config_sha256"] == imported["config_sha256"]
    assert "CANARY-SECRET" not in json.dumps(manifest)
    with pytest.raises(ValueError, match="not declared"):
        import_opencode_profile(
            "opencode-override", "translator",
            model_policy="override", model_override="private/unknown",
        )


def test_opencode_fails_closed_for_unknown_ambiguous_and_symlink_paths(tmp_path, monkeypatch):
    config = _opencode_fixture(
        tmp_path,
        monkeypatch,
        '{"model":"private/fallback","agent":{"translator":{"mode":"primary"}}}',
    )
    assert resolve_opencode_agent("unknown")["reason"] == "OpenCode agent is unknown"
    config.write_text(
        '{"model":"private/a","model":"private/b",'
        '"agent":{"translator":{"mode":"primary"}}}',
        "utf-8",
    )
    assert discover_opencode()["reason"] == "OpenCode config is invalid or unreadable"
    config.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text('{"agent":{}}', "utf-8")
    config.symlink_to(outside)
    assert discover_opencode()["reason"] == "OpenCode config path contains a symlink"


def test_codex_import_checks_named_profile_file_without_reading_credentials(
    monkeypatch, tmp_path
):
    from novel_harness.connections import import_codex_profile

    monkeypatch.setattr("novel_harness.connections.shutil.which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(
        "novel_harness.connections.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not invoke Codex")),
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="not present in CODEX_HOME"):
        import_codex_profile("codex-local", "missing-profile")
    (tmp_path / "local.config.toml").write_text('model = "private-model"\n', "utf-8")
    imported = import_codex_profile("codex-local", "local")
    assert imported["profile"]["profile"] == "local"
    assert imported["model_identity"] == "unknown until Codex reports it"


def test_codex_named_profile_loads_user_config_with_runner_safety_overrides(tmp_path):
    named = CodexRunner(
        RunnerConfig(type="codex", harness_managed=True, profile="local"), tmp_path
    ).command("codex", tmp_path)
    assert "--profile" in named
    assert named[named.index("--profile") + 1] == "local"
    assert "--ignore-user-config" not in named
    assert named[named.index("--sandbox") + 1] == "read-only"
    assert "--disable" in named

    isolated = CodexRunner(RunnerConfig(type="codex", harness_managed=True), tmp_path).command(
        "codex", tmp_path
    )
    assert "--ignore-user-config" in isolated


def test_opencode_unknown_model_is_not_available_or_listed(tmp_path, monkeypatch):
    _opencode_fixture(
        tmp_path,
        monkeypatch,
        '{"provider":{"known":{"models":{"model-a":{}}}},'
        '"agent":{"translator":{"mode":"primary",'
        '"model":"ghost-provider/model-404"}}}',
    )
    discovered = discover_opencode()
    agent = discovered["agents"][0]
    assert agent["available"] is False
    assert "not declared" in agent["reason"]
    assert discovered["models"] == [
        {
            "provider": "known",
            "model": "model-a",
            "variant": None,
            "sources": ["provider_config"],
        }
    ]
    assert resolve_opencode_agent("translator")["available"] is False


def test_opencode_model_variant_is_separate_from_model_id(tmp_path, monkeypatch):
    _opencode_fixture(
        tmp_path,
        monkeypatch,
        json.dumps(
            {
                "provider": {"private": {"models": {"model-a": {"variants": {"high": {}}}}}},
                "agent": {
                    "translator": {"mode": "primary", "model": "private/model-a#high"}
                },
            }
        ),
    )
    from novel_harness.config import HarnessExecutionProfile
    from novel_harness.connections import _profile_runner

    profile = HarnessExecutionProfile.model_validate(
        {"kind": "harness", "harness": "opencode", "agent_id": "translator"}
    )
    runner, snapshot = _profile_runner("translator-profile", profile)
    assert runner.model == "private/model-a"
    assert runner.opencode_variant == "high"
    assert snapshot["resolved_model"]["variant"] == "high"


def test_unsupported_opencode_schema_and_unresolved_agent_are_not_imported(
    tmp_path, monkeypatch
):
    _opencode_fixture(
        tmp_path,
        monkeypatch,
        '{"agents":{"translator":{"model":{"providerID":"private",'
        '"model":"model-a"}}}}',
    )
    assert "v2" in discover_opencode()["reason"]
    with pytest.raises(ValueError, match="v2"):
        import_opencode_profile("unsupported", "translator")

    _opencode_fixture(
        tmp_path,
        monkeypatch,
        '{"agent":{"translator":{"mode":"primary","model":"not-a-model"}}}',
    )
    assert discover_opencode()["agents"][0]["reason"] == (
        "OpenCode agent model is unresolved or unsupported"
    )


@pytest.mark.parametrize(
    "agent, reason",
    [
        ({"mode": "primary", "disable": True, "model": "private/model-a"}, "disabled"),
        ({"mode": "subagent", "model": "private/model-a"}, "not primary-capable"),
        ({"mode": "primary", "model": 17}, "unresolved or unsupported"),
    ],
)
def test_opencode_unsupported_agent_bindings_fail_closed(
    tmp_path, monkeypatch, agent, reason
):
    _opencode_fixture(
        tmp_path,
        monkeypatch,
        json.dumps(
            {
                "provider": {"private": {"models": {"model-a": {}}}},
                "agent": {"translator": agent},
            }
        ),
    )
    result = resolve_opencode_agent("translator")
    assert result["available"] is False
    assert reason in result["reason"]


def test_profile_schemas_and_managed_runtime_keep_credentials_in_harness():
    legacy_opencode = RunnerConfig(type="opencode", model="local/qwen")
    reloaded_opencode = RunnerConfig.model_validate(legacy_opencode.model_dump(mode="json"))
    assert "opencode_variant" not in legacy_opencode.public_settings()
    assert "opencode_variant" not in reloaded_opencode.public_settings()

    direct = DirectExecutionProfile.model_validate(
        {
            "kind": "direct",
            "model": "model-a",
            "base_url": "https://provider.example/v1",
            "api_key_env": "NOVEL_PROVIDER_KEY",
        }
    )
    assert direct.version == 1
    with pytest.raises(ValueError, match="api_key_env"):
        DirectExecutionProfile.model_validate(
            {
                "kind": "direct",
                "model": "model-a",
                "base_url": "https://provider.example/v1",
                "api_key_env": "literal-secret-value",
            }
        )
    with pytest.raises(ValueError, match="secret-bearing") as error:
        DirectExecutionProfile.model_validate(
            {
                "kind": "direct",
                "model": "model-a",
                "base_url": "https://provider.example/v1",
                "api_key_env": "NOVEL_PROVIDER_KEY",
                "extra_params": {"nested": [{"authToken": "CANARY-SECRET"}]},
            }
        )
    assert "CANARY-SECRET" not in str(error.value)
    with pytest.raises(ValueError, match="secret-bearing"):
        RunnerConfig(type="openai", extra_params={"provider": {"private_key": "value"}})
    with pytest.raises(ValueError, match="Advanced|override"):
        HarnessExecutionProfile.model_validate(
            {
                "kind": "harness",
                "harness": "opencode",
                "agent_id": "translator",
                "model_override": "private/model-a",
            }
        )

    runner_config = RunnerConfig(
        type="opencode",
        model="private/model-a",
        harness_managed=True,
        opencode_agent_id="translator",
        opencode_config_sha256="a" * 64,
    )
    runtime = OpenCodeRunner(runner_config, Path("jobs")).runtime_config(
        SimpleNamespace(instructions="Translate safely", max_tokens=100, temperature=None)
    )
    assert runtime["default_agent"] == "translator"
    assert runtime["agent"]["translator"]["model"] == "private/model-a"
    assert runtime["permission"] == {"*": "deny"}
    assert runtime["tools"] == {"*": False}
    assert "provider" not in runtime
    assert "apiKey" not in json.dumps(runtime)


def test_user_runner_cannot_inject_an_opencode_config_path(project):
    config = yaml.safe_load(project.read_text("utf-8"))
    config["runners"]["test"]["opencode_config_path"] = (
        "../../.local/share/opencode/auth.json"
    )
    project.write_text(yaml.safe_dump(config), "utf-8")
    with pytest.raises(ValueError, match="generated from execution_profiles") as error:
        load_config(project)
    assert "auth.json" not in str(error.value)


def test_compiled_profile_alias_round_trips_as_source_config(project, tmp_path, monkeypatch):
    monkeypatch.setenv("NOVEL_PROFILE_KEY", "unused-in-test")
    data = yaml.safe_load(project.read_text("utf-8"))
    data["execution_profiles"] = {
        "direct-profile": {
            "kind": "direct",
            "model": "provider-model",
            "base_url": "https://provider.example/v1",
            "api_key_env": "NOVEL_PROFILE_KEY",
        }
    }
    project.write_text(yaml.safe_dump(data), "utf-8")
    compiled = load_config(project)
    dumped = tmp_path / "round-trip.yaml"
    dumped.write_text(yaml.safe_dump(compiled.model_dump(mode="json")), "utf-8")
    reloaded = load_config(dumped)
    assert "direct-profile" in reloaded.execution_profiles
    assert reloaded.runners["direct-profile"].model == "provider-model"
