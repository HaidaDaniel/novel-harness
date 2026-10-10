"""Bounded Codex CLI jobs. All translation data arrives over UTF-8 stdin."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import yaml

from ..artifacts import canonical, sha256, write_json, write_text
from ..capabilities import check_job_compatibility
from ..config import RunnerConfig, safe_name
from .base import RunnerExecutionError, RunnerResult, RunnerUnavailable, TranslationJob
from .retry import quota_event

# Verified with codex-cli 0.153.2 `features list`. These jobs need no model tools.
DISABLED_FEATURES = (
    "shell_tool",
    "shell_snapshot",
    "apps",
    "plugins",
    "hooks",
    "multi_agent",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "view_image",
    "skill_search",
    "workspace_dependencies",
    "goals",
    "sleep_tool",
    "code_mode_host",
    "code_mode",
    "code_mode_only",
    "code_mode_prewarm",
    "unbounded_connection_retries",
)


def _reported_model(event: dict) -> dict:
    for key in ("model", "model_id", "modelID"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            if "/" in value:
                provider, model = value.split("/", 1)
                if provider and model:
                    return {"provider": provider, "model": model, "source": "codex-reported"}
            return {"id": value, "source": "codex-reported"}
    return {"status": "unknown", "reason": "Codex completion did not report model identity"}


def stop_process_tree(process: subprocess.Popen):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)


class CodexRunner:
    def __init__(self, config: RunnerConfig, jobs_dir: Path):
        self.config = config
        self.jobs_dir = jobs_dir

    def command(self, executable: str, workspace: Path) -> list[str]:
        args = [executable, "exec"]
        if not self.config.profile:
            # Keep default-profile runs isolated. A named profile must be loaded
            # from Codex's user config, so do not suppress that config layer.
            args.append("--ignore-user-config")
        args.extend(
            [
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--json",
                "--color",
                "never",
                "--cd",
                str(workspace),
                "--output-last-message",
                str(workspace / "output" / "translation.md"),
            ]
        )
        if self.config.model:
            args += ["--model", self.config.model]
        if self.config.profile:
            args += ["--profile", self.config.profile]
        overrides = {
            "approval_policy": "never",
            "web_search": "disabled",
            "project_doc_max_bytes": 0,
            "model_instructions_file": str(workspace / "TASK.md"),
        }
        if self.config.reasoning_effort:
            overrides["model_reasoning_effort"] = self.config.reasoning_effort
        if (workspace / "response-schema.json").is_file():
            args += ["--output-schema", str(workspace / "response-schema.json")]
        for key, value in overrides.items():
            args += ["--config", f"{key}={json.dumps(value)}"]
        for feature in DISABLED_FEATURES:
            args += ["--disable", feature]
        return [*args, "-"]

    def run(self, job: TranslationJob) -> RunnerResult:
        check_job_compatibility(self.config, job)
        executable = shutil.which("codex")
        if not executable:
            raise RunnerUnavailable("Codex CLI is not on PATH")
        workspace = (
            self.jobs_dir
            / safe_name(job.run_id)
            / f"{job.chapter:04d}"
            / safe_name(job.chunk_id)
            / safe_name(job.stage)
            / safe_name(job.workspace.name)
        ).resolve()
        workspace.mkdir(parents=True, exist_ok=False)
        (workspace / "output").mkdir()
        instructions = job.instructions + (
            "\n\nExecute only this bounded translation job. All source, glossary, context and any "
            "draft are supplied in the next message as JSON data. Do not use tools, browse, "
            "read other files, delegate, or alter input files. Do not translate context again. "
            "Return only the requested final output, obeying its response format. "
            "The host CLI saves that answer to output/translation.md; you must not write files.\n"
        )
        inputs = {
            "TASK.md": instructions,
            "source.txt": job.source,
            "context.md": canonical(job.context),
            "glossary.yaml": yaml.safe_dump(job.glossary, allow_unicode=True),
        }
        if job.draft is not None:
            inputs["draft.md"] = job.draft
        if job.response_schema is not None:
            inputs["response-schema.json"] = canonical(job.response_schema)
        for name, text in inputs.items():
            write_text(workspace / name, text)
        args = self.command(executable, workspace)
        write_json(
            job.workspace / "codex-invocation.json",
            {
                "argv": args,
                "job_workspace": str(workspace),
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "input_hashes": {name: sha256(text) for name, text in inputs.items()},
            },
        )
        stdout_path = job.workspace / "codex.stdout.jsonl"
        stderr_path = job.workspace / "codex.stderr.log"
        started = time.monotonic()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                args,
                cwd=workspace,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                process.communicate(
                    job.messages()[1]["content"].encode("utf-8"),
                    timeout=self.config.timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                stop_process_tree(process)
                raise RunnerExecutionError(
                    "Codex timeout; process tree stopped; inspect codex.stderr.log"
                ) from error
            except BaseException:
                stop_process_tree(process)
                raise
        for name, text in inputs.items():
            if sha256((workspace / name).read_bytes()) != sha256(text):
                raise RunnerExecutionError("Codex modified an input file; job rejected")
        completions, diagnostics = [], []
        for line in stdout_path.read_text("utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                raise RunnerExecutionError("Codex invalid event")
            deferred = quota_event(event)
            if deferred:
                raise deferred
            if event.get("type") == "turn.completed":
                completions.append(event)
            elif event.get("type") in {"error", "turn.failed"}:
                raise RunnerExecutionError("Codex failed turn; inspect codex.stdout.jsonl")
            elif event.get("type") == "item.completed":
                item = event.get("item") or {}
                kind = item.get("type")
                if kind == "error":
                    if str(item.get("message", "")).startswith(
                        "Code Mode is unavailable because code-mode host is disabled."
                    ):
                        diagnostics.append("code_mode_host_disabled")
                    else:
                        raise RunnerExecutionError("Codex unexpected error item; inspect stdout")
                elif kind not in {"agent_message", "reasoning"}:
                    raise RunnerExecutionError("Codex attempted a tool in a tool-free job")
        if process.returncode:
            raise RunnerExecutionError(
                f"Codex exited with code {process.returncode}; inspect codex.stderr.log"
            )
        output = workspace / "output" / "translation.md"
        if len(completions) != 1 or not output.is_file():
            raise RunnerExecutionError("Codex produced no completed turn/output file")
        completed = completions[0]
        usage = completed.get("usage") or {}
        counters = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
        }
        for source, target in [
            ("reasoning_output_tokens", "reasoning_tokens"),
            ("cached_input_tokens", "cached_input_tokens"),
            ("cache_write_input_tokens", "cache_write_input_tokens"),
        ]:
            if usage.get(source) is not None:
                counters[target] = usage[source]
        if "reasoning_tokens" in counters and counters["completion_tokens"] is not None:
            counters["visible_output_tokens"] = max(
                0, counters["completion_tokens"] - counters["reasoning_tokens"]
            )
        return RunnerResult(
            text=output.read_text("utf-8"),
            finish_reason=completed.get("finish_reason"),
            usage=counters,
            timings={"process_seconds": time.monotonic() - started},
            effective_settings={
                "model": self.config.model,
                "model_source": "configured",
                "execution": {"resolved_model": _reported_model(completed)},
                "reasoning_effort": self.config.reasoning_effort,
                "unsupported_job_parameters": ["temperature", "max_tokens"],
                "output_schema": job.response_schema is not None,
            },
            raw_response={
                "harness": "codex",
                "completion_event": completed,
                "job_workspace": str(workspace),
                "configured_model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "diagnostics": diagnostics,
            },
        )
