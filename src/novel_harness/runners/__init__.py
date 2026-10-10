import shutil

from ..capabilities import check_runner_compatibility, resolve_capabilities
from ..config import AppConfig
from .base import Runner, RunnerUnavailable
from .codex import CodexRunner
from .openai import DirectOpenAIRunner
from .opencode import OpenCodeRunner, find_executable


def runner_info(config: AppConfig, name: str) -> dict:
    profile = config.runners[name]
    reason = None
    executable = None
    resolved_profile = config._profile_resolution.get(name)
    if resolved_profile and not resolved_profile.get("available"):
        reason = resolved_profile.get("reason") or "Execution profile is unavailable"
    if profile.type == "openai":
        if reason:
            pass
        elif not profile.model:
            reason = "Model ID not configured"
        else:
            try:
                config.key_for(name)
            except ValueError as error:
                reason = str(error)
    elif profile.type == "opencode":
        try:
            executable = find_executable()
            if reason is None:
                key = "EMPTY" if profile.harness_managed else config.key_for(name)
                OpenCodeRunner(profile, config.novel.jobs_dir, key)
        except ValueError as error:
            reason = str(error)
    elif profile.type == "codex":
        executable = shutil.which("codex")
        if not executable:
            reason = "Codex CLI is not on PATH"
        elif reason is None and profile.profile is not None:
            from ..connections import codex_profile_available

            if not codex_profile_available(profile.profile):
                reason = "Codex profile config is not present in CODEX_HOME"
    else:
        executable = shutil.which(profile.type)
        reason = (
            "CLI detected; subprocess integration scheduled for Milestone 3"
            if executable
            else "Integration unavailable: executable not found"
        )
    return {
        "name": name,
        "type": profile.type,
        "model": profile.model,
        "available": reason is None,
        "reason": reason,
        "executable": executable,
        "availability_check": "configuration only; endpoint not probed",
        "capabilities": resolve_capabilities(profile).to_dict(),
    }


def make_runner(config: AppConfig, name: str) -> Runner:
    if name not in config.runners:
        raise ValueError(f"Unknown runner: {name}")
    info = runner_info(config, name)
    if not info["available"]:
        raise RunnerUnavailable(f"{name}: {info['reason']}")
    check_runner_compatibility(config.runners[name])
    if config.runners[name].type == "codex":
        return CodexRunner(config.runners[name], config.novel.jobs_dir)
    if config.runners[name].type == "opencode":
        key = "EMPTY" if config.runners[name].harness_managed else config.key_for(name)
        return OpenCodeRunner(config.runners[name], config.novel.jobs_dir, key)
    return DirectOpenAIRunner(config.runners[name], config.key_for(name))
