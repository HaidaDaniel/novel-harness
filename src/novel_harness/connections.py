"""Secret-free discovery and resolution for connection/execution profiles.

OpenCode import intentionally accepts only the installed v1 JSON/JSONC shape in
the user's global config. It never reads OpenCode's auth store or project config,
and it does not extract or return provider credentials. The OpenCode process
continues to own its own credentials.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from .artifacts import canonical, sha256
from .config import (
    DirectExecutionProfile,
    RunnerConfig,
    _validate_model_reference,
    safe_name,
)

if TYPE_CHECKING:
    from .config import AppConfig

_MAX_CONFIG_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _strip_jsonc(text: str) -> str:
    """Remove JSONC comments and trailing commas without changing string data."""
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "*":
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("OpenCode config is invalid")
            index = end + 2
            continue
        output.append(char)
        index += 1

    text = "".join(output)
    output = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _no_symlink_path(path: Path) -> bool:
    """Reject any existing symlink component before opening a config file."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                return False
        except OSError:
            return False
    return True


def opencode_config_directory() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME")
    if configured:
        base = Path(configured).expanduser()
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".config"
    if not base.is_absolute():
        raise ValueError("OpenCode config directory must be absolute")
    if ".." in base.parts:
        raise ValueError("OpenCode config directory cannot traverse parent paths")
    return base / "opencode"


def _read_opencode_config(config_dir: Path | None = None) -> tuple[dict, Path, str]:
    directory = config_dir or opencode_config_directory()
    candidates = [directory / "opencode.json", directory / "opencode.jsonc"]
    if not _no_symlink_path(directory):
        raise ValueError("OpenCode config path contains a symlink")
    present = [path for path in candidates if path.exists()]
    if len(present) != 1:
        if not present:
            raise ValueError("OpenCode global config is missing")
        raise ValueError("OpenCode global config is ambiguous")
    path = present[0]
    if not _no_symlink_path(path):
        raise ValueError("OpenCode config path contains a symlink")
    if not path.is_file():
        raise ValueError("OpenCode config path is not a regular file")
    try:
        size = path.stat().st_size
        if size > _MAX_CONFIG_BYTES:
            raise ValueError("OpenCode config exceeds the supported size")
        raw = path.read_bytes()
        decoded = json.loads(
            _strip_jsonc(raw.decode("utf-8")), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("OpenCode config"):
            raise
        raise ValueError("OpenCode config is invalid or unreadable") from None
    if not isinstance(decoded, dict):
        raise TypeError("OpenCode config must be an object")
    return decoded, path, sha256(raw)


def _model_parts(value: object) -> tuple[str, str, str | None] | None:
    if not isinstance(value, str):
        return None
    try:
        return _validate_model_reference(value)
    except ValueError:
        return None


def _configured_model(
    document: dict, provider: str, model: str, variant: str | None = None
) -> bool:
    providers = document.get("provider", {})
    provider_config = providers.get(provider) if isinstance(providers, dict) else None
    models = provider_config.get("models") if isinstance(provider_config, dict) else None
    model_config = models.get(model) if isinstance(models, dict) else None
    if not isinstance(model_config, dict):
        return False
    if variant is None:
        return True
    variants = model_config.get("variants") if isinstance(model_config, dict) else None
    return isinstance(variants, dict) and variant in variants


def _agent_markdown_exists(config_dir: Path, agent_id: str) -> bool:
    # Markdown frontmatter is a distinct OpenCode agent source. It is deliberately
    # not imported in this version, so the same ID cannot resolve ambiguously.
    for folder in ("agents", "agent"):
        path = config_dir / folder / f"{agent_id}.md"
        if path.exists():
            return True
    return False


def _opencode_version() -> str | None:
    executable = shutil.which("opencode")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=2, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = (result.stdout or result.stderr).strip()
    if result.returncode == 0 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", value):
        return value
    return None


def _codex_version() -> str | None:
    executable = shutil.which("codex")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=2, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = (result.stdout or result.stderr).strip()
    if result.returncode == 0 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", value):
        return value
    return None


def codex_profile_available(profile: str | None) -> bool:
    """Check a Codex profile by its local config file name without reading it."""
    if profile is None:
        return True
    try:
        safe_name(profile)
    except ValueError:
        return False
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    return (home / f"{profile}.config.toml").is_file()


def discover_opencode(config_dir: Path | None = None) -> dict:
    """Return a redacted read-only projection; never reads auth or probes models."""
    executable = shutil.which("opencode")
    try:
        directory = config_dir or opencode_config_directory()
        document, _path, config_hash = _read_opencode_config(directory)
    except (TypeError, ValueError) as error:
        return {
            "available": False,
            "reason": str(error),
            "executable": bool(executable),
            "version": _opencode_version() if executable else None,
            "agents": [],
            "models": [],
            "config_sha256": None,
        }
    if not executable:
        reason = "OpenCode CLI is not on PATH"
    elif "agents" in document:
        reason = "OpenCode v2 agent schema is not supported for import"
    elif not isinstance(document.get("agent", {}), dict):
        reason = "OpenCode agent configuration is unsupported"
    else:
        reason = None

    agents = []
    configured_agents = document.get("agent", {})
    if isinstance(configured_agents, dict):
        for agent_id, agent in sorted(configured_agents.items()):
            if not isinstance(agent_id, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", agent_id
            ):
                continue
            state, model_data = _resolve_agent(document, directory, agent_id, agent)
            agents.append(
                {
                    "agent_id": agent_id,
                    "available": state is None and bool(executable),
                    "reason": state or (None if executable else "OpenCode CLI is not on PATH"),
                    "model": model_data,
                }
            )
    return {
        "available": reason is None,
        "reason": reason,
        "executable": bool(executable),
        "version": _opencode_version() if executable else None,
        "agents": agents,
        "models": _configured_models(document),
        "config_sha256": config_hash,
    }


def _configured_models(document: dict) -> list[dict]:
    """List only declared model IDs; discard all provider options and credentials."""
    models: dict[tuple[str, str, str | None], set[str]] = {}

    def add(value: object, source: str) -> None:
        parts = _model_parts(value)
        if parts is not None and _configured_model(document, *parts):
            models.setdefault(parts, set()).add(source)

    add(document.get("model"), "global_default")
    agents = document.get("agent", {})
    if isinstance(agents, dict):
        for agent in agents.values():
            if isinstance(agent, dict):
                add(agent.get("model"), "agent")
    providers = document.get("provider", {})
    if isinstance(providers, dict):
        for provider, config in providers.items():
            if not isinstance(provider, str) or not isinstance(config, dict):
                continue
            provider_models = config.get("models", {})
            if not isinstance(provider_models, dict):
                continue
            for model_id in provider_models:
                if isinstance(model_id, str):
                    add(f"{provider}/{model_id}", "provider_config")
    return [
        {
            "provider": provider,
            "model": model,
            "variant": variant,
            "sources": sorted(sources),
        }
        for (provider, model, variant), sources in sorted(
            models.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
        )
    ]


def _resolve_agent(document: dict, config_dir: Path, agent_id: str, agent: object):
    if not isinstance(agent, dict):
        return "OpenCode agent entry is unsupported", None
    if _agent_markdown_exists(config_dir, agent_id):
        return "OpenCode agent has an unsupported Markdown definition", None
    if agent.get("disable") is True:
        return "OpenCode agent is disabled", None
    mode = agent.get("mode")
    if mode not in (None, "primary", "all"):
        return "OpenCode agent is not primary-capable", None
    binding_source = "agent"
    model_value = agent.get("model")
    if "model" not in agent:
        model_value = document.get("model")
        binding_source = "global"
    model_parts = _model_parts(model_value)
    if model_parts is None:
        return "OpenCode agent model is unresolved or unsupported", None
    provider, model, variant = model_parts
    if not _configured_model(document, provider, model, variant):
        return "OpenCode agent model is not declared in local provider configuration", None
    return None, {
        "provider": provider,
        "model": model,
        "variant": variant,
        "source": "agent" if binding_source == "agent" else "global_default",
    }


def resolve_opencode_agent(
    agent_id: str,
    *,
    model_policy: str = "inherit",
    model_override: str | None = None,
    config_dir: Path | None = None,
) -> dict:
    """Resolve one agent and return safe provenance fields or a generic failure."""
    try:
        safe_name(agent_id)
    except ValueError:
        return {"available": False, "reason": "OpenCode agent ID is invalid"}
    found = discover_opencode(config_dir)
    if not found["available"]:
        return {"available": False, "reason": found["reason"] or "OpenCode unavailable"}
    summary = next((item for item in found["agents"] if item["agent_id"] == agent_id), None)
    if summary is None:
        return {"available": False, "reason": "OpenCode agent is unknown"}
    if not summary["available"]:
        return {"available": False, "reason": summary["reason"]}
    directory = config_dir or opencode_config_directory()
    try:
        document, path, config_hash = _read_opencode_config(directory)
    except (TypeError, ValueError) as error:
        return {"available": False, "reason": str(error)}
    resolution_error, resolved = _resolve_agent(
        document, directory, agent_id, document.get("agent", {}).get(agent_id)
    )
    if resolution_error or resolved is None:
        return {
            "available": False,
            "reason": resolution_error or "OpenCode agent is unresolved",
        }
    source = (
        "inherited from OpenCode global default"
        if resolved.get("source") == "global_default"
        else "inherited from OpenCode agent"
    )
    if model_policy == "override":
        override = _model_parts(model_override)
        if override is None:
            return {"available": False, "reason": "Explicit model override is invalid"}
        if not _configured_model(document, *override):
            return {
                "available": False,
                "reason": "Explicit model override is not declared in local provider configuration",
            }
        resolved = {
            "provider": override[0],
            "model": override[1],
            "variant": override[2],
            "source": "advanced_override",
        }
        source = "explicit advanced override"
    elif model_policy != "inherit" or model_override is not None:
        return {"available": False, "reason": "OpenCode model policy is invalid"}
    return {
        "available": True,
        "agent_id": agent_id,
        "model": resolved,
        "model_source": source,
        "config_sha256": config_hash,
        "config_path": str(path),
        "harness_version": found["version"],
    }


def opencode_config_unchanged(path: Path | None, expected_hash: str | None) -> bool:
    """Verify the imported config identity without exposing its contents."""
    if path is None or expected_hash is None or not _no_symlink_path(path):
        return False
    try:
        if path.stat().st_size > _MAX_CONFIG_BYTES:
            return False
        raw = path.read_bytes()
    except OSError:
        return False
    return len(raw) <= _MAX_CONFIG_BYTES and sha256(raw) == expected_hash


def _profile_runner(profile_id: str, profile, config_dir: Path | None = None):
    if isinstance(profile, DirectExecutionProfile):
        values = {
            "type": "openai",
            "model": profile.model,
            "base_url": profile.base_url,
            "api_key_env": profile.api_key_env,
            "timeout_seconds": profile.timeout_seconds,
            "extra_params": profile.extra_params,
            "execution_profile_id": profile_id,
        }
        for field in ("temperature", "max_tokens"):
            if field in profile.model_fields_set:
                values[field] = getattr(profile, field)
        runner = RunnerConfig(**values)
        settings = profile.model_dump(mode="json")
        return runner, {
            "version": 1,
            "configured_profile": True,
            "profile_id": profile_id,
            "kind": "direct",
            "credential_owner": "Novel Harness",
            "model_policy": "explicit",
            "requested_model": {"id": profile.model},
            "resolved_model": {"id": profile.model, "source": "direct profile"},
            "config_sha256": sha256(canonical(settings)),
            "available": True,
        }

    if profile.harness == "opencode":
        resolution = resolve_opencode_agent(
            profile.agent_id,
            model_policy=profile.model_policy,
            model_override=profile.model_override,
            config_dir=config_dir,
        )
        resolved = resolution.get("model") if resolution.get("available") else None
        values = {
            "type": "opencode",
            "model": f"{resolved['provider']}/{resolved['model']}" if resolved else None,
            "harness_managed": True,
            "opencode_agent_id": profile.agent_id,
            "opencode_config_path": Path(resolution["config_path"])
            if resolution.get("config_path")
            else None,
            "opencode_config_sha256": resolution.get("config_sha256"),
            "execution_profile_id": profile_id,
            "requested_model_policy": profile.model_policy,
        }
        if resolved and resolved.get("variant"):
            values["opencode_variant"] = resolved["variant"]
        runner = RunnerConfig(**values)
        snapshot = {
            "version": 1,
            "configured_profile": True,
            "profile_id": profile_id,
            "kind": "harness-managed",
            "credential_owner": "OpenCode",
            "harness": "opencode",
            "harness_version": resolution.get("harness_version"),
            "agent_id": profile.agent_id,
            "model_policy": profile.model_policy,
            "requested_model": (
                {"id": profile.model_override}
                if profile.model_policy == "override"
                else {"policy": "inherit"}
            ),
            "model_source": resolution.get("model_source"),
            "resolved_model": resolved or {"status": "unknown"},
            "config_sha256": resolution.get("config_sha256"),
            "available": bool(resolution.get("available")),
            "reason": resolution.get("reason"),
        }
        return runner, snapshot

    runner = RunnerConfig(
        type="codex",
        model=profile.model_override if profile.model_policy == "override" else None,
        profile=profile.profile,
        harness_managed=True,
        execution_profile_id=profile_id,
        requested_model_policy=profile.model_policy,
    )
    executable = shutil.which("codex")
    profile_available = codex_profile_available(profile.profile)
    available = bool(executable) and profile_available
    reason = (
        "Codex CLI is not on PATH"
        if not executable
        else None
        if profile_available
        else "Codex profile config is not present in CODEX_HOME"
    )
    return runner, {
        "version": 1,
        "configured_profile": True,
        "profile_id": profile_id,
        "kind": "harness-managed",
        "credential_owner": "Codex",
        "harness": "codex",
        "harness_version": _codex_version() if executable else None,
        "profile": profile.profile,
        "model_policy": profile.model_policy,
        "requested_model": (
            {"id": profile.model_override}
            if profile.model_policy == "override"
            else {"policy": "inherit"}
        ),
        "resolved_model": {"status": "unknown"},
        "available": available,
        "reason": reason,
    }


def materialize_execution_profiles(config: AppConfig) -> None:
    """Compile configured profiles to runner aliases and store safe provenance."""
    for profile_id, profile in config.execution_profiles.items():
        runner, resolution = _profile_runner(profile_id, profile)
        config.runners[profile_id] = runner
        config._profile_resolution[profile_id] = resolution


def refresh_execution_profiles(config: AppConfig) -> None:
    """Refresh harness bindings before status, run preparation and sample gates."""
    if config.execution_profiles:
        materialize_execution_profiles(config)


def execution_snapshot(config: AppConfig, runner_name: str) -> dict:
    """Return the safe immutable execution identity for a configured runner."""
    if runner_name in config._profile_resolution:
        return dict(config._profile_resolution[runner_name])
    profile = config.runners[runner_name]
    if profile.type == "openai":
        return {
            "version": 1,
            "configured_profile": False,
            "profile_id": runner_name,
            "kind": "direct",
            "credential_owner": "Novel Harness",
            "model_policy": "explicit" if profile.model else "unresolved",
            "resolved_model": {"id": profile.model} if profile.model else {"status": "unknown"},
            "config_sha256": sha256(canonical(profile.public_settings())),
            "available": bool(profile.model),
        }
    if profile.type == "opencode" and not profile.harness_managed:
        model = None
        if profile.model and "/" in profile.model:
            provider, model_id = profile.model.split("/", 1)
            model = {"provider": provider, "model": model_id, "source": "configured"}
        return {
            "version": 1,
            "configured_profile": False,
            "profile_id": runner_name,
            "kind": "direct",
            "credential_owner": "Novel Harness",
            "harness": "opencode",
            "harness_version": _opencode_version(),
            "model_policy": "explicit" if model else "unresolved",
            "resolved_model": model or {"status": "unknown"},
            "config_sha256": sha256(canonical(profile.public_settings())),
            "available": bool(model),
        }
    if profile.type == "codex":
        return {
            "version": 1,
            "configured_profile": False,
            "profile_id": runner_name,
            "kind": "harness-managed",
            "credential_owner": "Codex",
            "harness": "codex",
            "harness_version": _codex_version(),
            "requested_model": {"id": profile.model} if profile.model else {"policy": "inherit"},
            "resolved_model": {"status": "unknown"},
            "profile": profile.profile,
            "config_sha256": sha256(canonical(profile.public_settings())),
            "available": bool(_codex_version()) and codex_profile_available(profile.profile),
        }
    return {
        "version": 1,
        "configured_profile": False,
        "profile_id": runner_name,
        "kind": "harness-managed" if profile.type in {"opencode", "codex"} else "unknown",
        "credential_owner": profile.type,
        "harness": profile.type if profile.type in {"opencode", "codex"} else None,
        "model_policy": "explicit" if profile.model else "inherit",
        "resolved_model": {"id": profile.model} if profile.model else {"status": "unknown"},
        "config_sha256": sha256(canonical(profile.public_settings())),
        "available": True,
    }


def list_connections(config: AppConfig) -> dict:
    from .runners import runner_info

    discovery = discover_opencode()
    profiles = []
    configured_profile_ids = set(config.execution_profiles)
    for profile_id, definition in config.execution_profiles.items():
        snapshot = config._profile_resolution.get(profile_id, {})
        status = runner_info(config, profile_id)
        profiles.append(
            {
                "id": profile_id,
                "kind": definition.kind,
                "harness": getattr(definition, "harness", "direct"),
                "available": status["available"],
                "reason": status["reason"] or snapshot.get("reason"),
                "model_policy": snapshot.get("model_policy"),
                "requested_model": snapshot.get("requested_model"),
                "resolved_model": snapshot.get("resolved_model"),
                "model_source": snapshot.get("model_source"),
                "agent_id": snapshot.get("agent_id"),
            }
        )
    for runner_name, runner_config in config.runners.items():
        if runner_name in configured_profile_ids:
            continue
        status = runner_info(config, runner_name)
        snapshot = execution_snapshot(config, runner_name)
        profiles.append(
            {
                "id": runner_name,
                "kind": snapshot["kind"],
                "harness": runner_config.type,
                "available": status["available"],
                "reason": status["reason"],
                "model_policy": snapshot.get("model_policy"),
                "requested_model": snapshot.get("requested_model"),
                "resolved_model": snapshot.get("resolved_model"),
                "model_source": snapshot.get("model_source"),
                "agent_id": runner_config.opencode_agent_id,
            }
        )
    public_opencode = {
        key: discovery.get(key)
        for key in ("available", "executable", "version")
        if key in discovery
    }
    public_opencode["agent_count"] = len(discovery.get("agents", []))
    public_opencode["model_count"] = len(discovery.get("models", []))
    return {
        "version": 1,
        "profiles": profiles,
        "opencode": public_opencode,
        "codex": {
            "available": shutil.which("codex") is not None,
            "version": _codex_version() if shutil.which("codex") else None,
            "authentication": "owned by Codex; not inspected",
        },
    }


def import_opencode_profile(
    profile_id: str,
    agent_id: str,
    *,
    model_policy: str = "inherit",
    model_override: str | None = None,
) -> dict:
    """Return an explicit, redacted YAML config proposal; do not write user config."""
    try:
        safe_name(profile_id)
    except ValueError:
        raise ValueError("Execution profile ID is invalid") from None
    resolution = resolve_opencode_agent(
        agent_id, model_policy=model_policy, model_override=model_override
    )
    if not resolution.get("available"):
        raise ValueError(resolution.get("reason") or "OpenCode profile is unavailable")
    profile = {
        "version": 1,
        "kind": "harness",
        "harness": "opencode",
        "agent_id": agent_id,
        "model_policy": model_policy,
    }
    if model_policy == "override":
        profile["model_override"] = model_override
    return {
        "profile_id": profile_id,
        "profile": profile,
        "resolved_model": resolution["model"],
        "model_source": resolution["model_source"],
        "config_sha256": resolution["config_sha256"],
        "config_yaml": yaml.safe_dump(
            {"execution_profiles": {profile_id: profile}},
            allow_unicode=True,
            sort_keys=False,
        ),
    }


def import_direct_profile(
    profile_id: str,
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout_seconds: float = 180,
) -> dict:
    """Validate a direct profile proposal containing an env reference, never a key."""
    try:
        safe_name(profile_id)
        profile = DirectExecutionProfile.model_validate(
            {
                "version": 1,
                "kind": "direct",
                "model": model,
                "base_url": base_url,
                "api_key_env": api_key_env,
                "timeout_seconds": timeout_seconds,
            }
        )
    except (ValidationError, ValueError):
        raise ValueError("Invalid direct execution profile") from None
    value = profile.model_dump(mode="json")
    return {
        "profile_id": profile_id,
        "profile": value,
        "config_yaml": yaml.safe_dump(
            {"execution_profiles": {profile_id: value}},
            allow_unicode=True,
            sort_keys=False,
        ),
    }


def import_codex_profile(profile_id: str, codex_profile: str | None = None) -> dict:
    try:
        safe_name(profile_id)
    except ValueError:
        raise ValueError("Execution profile ID is invalid") from None
    if not shutil.which("codex"):
        raise ValueError("Codex CLI is not on PATH")
    profile = {
        "version": 1,
        "kind": "harness",
        "harness": "codex",
        "model_policy": "inherit",
    }
    if codex_profile is not None:
        try:
            profile["profile"] = safe_name(codex_profile)
        except ValueError:
            raise ValueError("Codex profile reference is invalid") from None
        if not codex_profile_available(codex_profile):
            raise ValueError("Codex profile is not present in CODEX_HOME")
    return {
        "profile_id": profile_id,
        "profile": profile,
        "model_identity": "unknown until Codex reports it",
        "config_yaml": yaml.safe_dump(
            {"execution_profiles": {profile_id: profile}},
            allow_unicode=True,
            sort_keys=False,
        ),
    }
