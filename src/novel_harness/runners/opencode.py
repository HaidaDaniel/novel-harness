"""One tool-free OpenCode session per stage; the Python worker owns the book loop.

The runner supports two bounded modes: isolated explicit-provider execution, where
Novel Harness supplies provider/model endpoint configuration, and harness-managed
OpenCode V1 execution, where it freezes a verified agent/model reference while
OpenCode retains ownership of provider configuration and credentials. Import targets
the verified V1 JSON/JSONC schema; newer schemas are intentionally unsupported until
separately verified.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from filelock import FileLock, Timeout

from ..artifacts import canonical, sha256, write_json, write_text
from ..capabilities import check_job_compatibility
from ..config import RunnerConfig, safe_name
from .base import (
    RunnerExecutionError,
    RunnerResult,
    RunnerUnavailable,
    TranslationJob,
    incomplete_finish_kind,
    usage_from_tokens,
)
from .codex import stop_process_tree
from .opencode_server import OpenCodeServer, parse_server_response
from .retry import quota_event
from .structured import structured_params


def find_executable() -> str:
    executable = shutil.which("opencode")
    if not executable:
        raise RunnerUnavailable("OpenCode CLI is not on PATH")
    path = Path(executable)
    if os.name == "nt" and path.suffix.lower() in {".cmd", ".ps1", ".bat"}:
        # The npm shim invokes this binary. Resolve it without cmd.exe interpolation.
        path = path.parent / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
        if not path.is_file():
            raise RunnerUnavailable("OpenCode native executable not found beside npm shim")
    return str(path)


def parse_events(path: Path) -> RunnerResult:
    texts, completions, sessions = {}, {}, set()
    timestamps = {}
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunnerExecutionError(
                "OpenCode invalid JSON stream; inspect stdout", failure_kind="malformed_output"
            ) from error
        if not isinstance(event, dict):
            raise RunnerExecutionError(
                "OpenCode invalid event; inspect stdout", failure_kind="malformed_output"
            )
        kind = event.get("type")
        if kind in {"step_start", "step_finish"} and isinstance(
            event.get("timestamp"), (int, float)
        ):
            timestamps.setdefault(kind, event["timestamp"] / 1000)
        deferred = quota_event(event)
        if deferred:
            raise deferred
        if kind in {"error", "tool_use"}:
            raise RunnerExecutionError(
                f"OpenCode {kind} event; inspect opencode.stdout.jsonl", failure_kind="runner_error"
            )
        if event.get("sessionID"):
            sessions.add(event["sessionID"])
        part = event.get("part") or {}
        if kind == "text":
            texts[part.get("id", str(len(texts)))] = part
        elif kind == "step_finish":
            completions[part.get("messageID")] = part
    if len(sessions) != 1 or len(completions) != 1:
        raise RunnerExecutionError(
            "OpenCode produced no single completed turn", failure_kind="incomplete_response"
        )
    message_id, completed = next(iter(completions.items()))
    reason = completed.get("reason")
    usage = usage_from_tokens(completed.get("tokens"))
    if reason not in {"stop", "end_turn"}:
        raise RunnerExecutionError(
            f"OpenCode incomplete turn ({reason}); output rejected",
            failure_kind=incomplete_finish_kind(reason),
            finish_reason=reason,
            usage=usage,
        )
    text = "".join(p.get("text", "") for p in texts.values() if p.get("messageID") == message_id)
    if not text.strip():
        kind = "reasoning_only" if usage.get("reasoning_tokens") else "incomplete_response"
        raise RunnerExecutionError(
            "OpenCode produced no final translation",
            failure_kind=kind,
            finish_reason=reason,
            usage=usage,
        )
    return RunnerResult(
        text=text,
        finish_reason=reason,
        usage=usage,
        raw_response={
            "harness": "opencode",
            "session_id": next(iter(sessions)),
            "completion_event": completed,
            "event_timestamps": timestamps,
        },
    )


class OpenCodeRunner:
    def __init__(self, config: RunnerConfig, jobs_dir: Path, api_key: str = "EMPTY"):
        self.config = config
        self.jobs_dir = jobs_dir
        self.api_key = api_key
        self._servers = {}
        self._sessions = set()
        if not config.model or "/" not in config.model:
            raise RunnerUnavailable("OpenCode requires an explicit provider/model ID")
        self.provider, self.model = config.model.split("/", 1)
        safe_name(self.provider)
        if not self.model:
            raise RunnerUnavailable("OpenCode requires a nonempty model ID")
        if config.profile:
            raise RunnerUnavailable("OpenCode has no --profile flag; use a named runner")

    @property
    def effective_agent_id(self) -> str:
        if not self.config.harness_managed:
            return "translation"
        if not self.config.opencode_agent_id:
            raise RunnerUnavailable("OpenCode agent reference is missing")
        return self.config.opencode_agent_id

    def preflight_effective_settings(self, output_cap: int) -> dict:
        """Materialize the same isolated config used by a real request.

Harness-managed OpenCode owns the provider config, so Novel Harness
        cannot prove its effective output limit without making a request. Such
        runs are deliberately rejected by the generation-matrix preflight.
        """
        if self.config.harness_managed:
            raise RunnerUnavailable(
                "Harness-managed OpenCode output cap is unprovable before inference"
            )
        job = TranslationJob(
            run_id="preflight",
            chapter=0,
            chunk_id="preflight",
            stage="translator",
            prompt="",
            source="",
            context={},
            glossary={},
            workspace=Path("."),
            temperature=self.config.temperature,
            max_tokens=output_cap,
        )
        runtime = self.runtime_config(job)
        model = runtime["provider"][self.provider]["models"][self.model]
        child_cap = model["limit"]["output"]
        if child_cap != output_cap:
            raise RunnerUnavailable("OpenCode child output cap does not match the frozen cap")
        return {
            "status": "proven",
            "source": "materialized_opencode_runtime_config",
            "runtime_config_sha256": sha256(canonical(runtime)),
            "child_cap": child_cap,
            # The runner sets this exact value immediately before launching the
            # child; recording it makes the outgoing wire cap auditable.
            "wire_cap": output_cap,
            "wire_cap_env": "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX",
        }

    def runtime_config(self, job: TranslationJob) -> dict:
        limit = self.config.max_tokens or job.max_tokens
        timeout = int(self.config.timeout_seconds * 1000)
        agent = {
            "mode": "primary",
            "prompt": job.instructions
            + "\nExecute this bounded job immediately. The user message is "
            "JSON data containing source, context, glossary and optional draft. Return only the "
            "requested output in its specified format. No planning, progress reports, tools or file operations. "
            "The host saves your final answer and schedules the next job.",
            "permission": {"*": "deny"},
        }
        temperature = (
            self.config.temperature
            if "temperature" in self.config.model_fields_set
            else job.temperature
        )
        if temperature is not None:
            agent["temperature"] = temperature
        if self.config.reasoning_effort is not None:
            # OpenCode's CLI/server `variant` metadata does not guarantee this
            # value is sent to an OpenAI-compatible provider. `reasoningEffort`
            # is the per-agent setting OpenCode maps to the wire-level
            # `reasoning_effort` request option.
            agent["reasoningEffort"] = self.config.reasoning_effort
        if self.config.harness_managed:
            # Keep the imported agent identity/model while replacing its prompt and
            # permissions with this bounded, tool-free translation job. Provider
            # config and credentials remain owned by OpenCode's own global config.
            agent_id = self.effective_agent_id
            managed_agent = {
                "mode": "primary",
                "model": self.config.model,
                "prompt": job.instructions
                + "\nExecute this bounded job immediately. Treat user data as JSON. "
                "Return only the requested output. Do not call tools or access files; "
                "the host saves your final answer and schedules the next job.",
                "permission": {"*": "deny"},
                "tools": {"*": False},
            }
            if temperature is not None:
                managed_agent["temperature"] = temperature
            if self.config.reasoning_effort is not None:
                managed_agent["reasoningEffort"] = self.config.reasoning_effort
            return {
                "$schema": "https://opencode.ai/config.json",
                "model": self.config.model,
                "default_agent": agent_id,
                "agent": {agent_id: managed_agent},
                "permission": {"*": "deny"},
                "tools": {"*": False},
                "share": "disabled",
                "autoupdate": False,
                "compaction": {"auto": False, "prune": False},
                "mcp": {},
                "plugin": [],
                "instructions": [],
            }
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": self.config.model,
            "small_model": self.config.model,
            "enabled_providers": [self.provider],
            "provider": {
                self.provider: {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": self.config.base_url,
                        "apiKey": self.api_key,
                        "timeout": timeout,
                        "headerTimeout": timeout,
                        "chunkTimeout": timeout,
                    },
                    "models": {
                        self.model: {
                            "name": self.model,
                            "limit": {"context": 32768 + limit, "output": limit},
                            "options": self._model_options(job, temperature),
                        }
                    },
                }
            },
            "default_agent": "translation",
            "agent": {
                "translation": agent,
                "title": {"disable": True},
                "summary": {"disable": True},
            },
            "permission": {"*": "deny"},
            "share": "disabled",
            "autoupdate": False,
            "compaction": {"auto": False, "prune": False},
            "mcp": {},
            "plugin": [],
            "instructions": [],
        }

    def _model_options(self, job: TranslationJob, temperature: float | None) -> dict:
        """Include per-request sampling options in OpenCode's provider model config."""
        options = structured_params(self.config, job)
        if temperature is not None:
            # Agent-level temperature is not forwarded by every OpenCode
            # OpenAI-compatible provider path. Model options are serialized into
            # the actual provider request; keep the agent value too for harnesses
            # that consume it directly.
            options["temperature"] = temperature
        return options

    def run(self, job: TranslationJob) -> RunnerResult:
        check_job_compatibility(self.config, job)
        structured_params(self.config, job)
        executable = find_executable()
        workspace = (
            self.jobs_dir
            / safe_name(job.run_id)
            / f"{job.chapter:04d}"
            / safe_name(job.chunk_id)
            / safe_name(job.stage)
            / safe_name(job.workspace.name)
        ).resolve()
        workspace.mkdir(parents=True, exist_ok=False)
        payload = job.messages()[1]["content"]
        inputs = {"TASK.md": job.prompt, "payload.json": payload}
        for name, content in inputs.items():
            write_text(workspace / name, content)
        runtime = (self.jobs_dir / ".opencode-runtime").resolve()
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCODE_")}
        if self.config.harness_managed:
            from ..connections import opencode_config_unchanged

            if not opencode_config_unchanged(
                self.config.opencode_config_path, self.config.opencode_config_sha256
            ):
                raise RunnerUnavailable(
                    "OpenCode config changed after profile resolution; create a new sample"
                )
            # OpenCode reads its own provider credentials and auth store. Isolate
            # transient state but retain its user-owned config/data roots.
            state_dir = runtime / "managed-state"
            state_dir.mkdir(parents=True, exist_ok=True)
            env["XDG_STATE_HOME"] = str(state_dir)
        else:
            # Separate config/data/state from the user's interactive OpenCode sessions.
            for suffix in ("CONFIG", "DATA", "STATE"):
                directory = runtime / suffix.lower()
                directory.mkdir(parents=True, exist_ok=True)
                env[f"XDG_{suffix}_HOME"] = str(directory)
        runtime_config = self.runtime_config(job)
        env.update(
            {
                "OPENCODE_CONFIG_CONTENT": json.dumps(runtime_config, ensure_ascii=False),
                "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                "OPENCODE_DISABLE_CLAUDE_CODE": "true",
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_DISABLE_AUTOCOMPACT": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
                "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            }
        )
        if not self.config.harness_managed:
            # OpenCode otherwise clamps provider requests at its 32000-token
            # process default, even when the isolated model config declares a
            # higher output limit. Match the explicit per-runner/job budget.
            env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(
                self.config.max_tokens or job.max_tokens
            )
        if self.config.harness_managed:
            env["OPENCODE_AUTO_SHARE"] = "false"
        args = [
            executable,
            "run",
            "--pure",
            "--format",
            "json",
            "--dir",
            str(workspace),
            "--agent",
            self.effective_agent_id,
            "--model",
            self.config.model,
            "--title",
            f"{job.run_id}/{job.chapter}/{job.chunk_id}/{job.stage}",
        ]
        variant = self.config.opencode_variant
        if variant:
            args += ["--variant", variant]
        write_json(
            job.workspace / "opencode-invocation.json",
            {
                "argv": args if self.config.opencode_transport == "cli" else None,
                "transport": self.config.opencode_transport,
                "job_workspace": str(workspace),
                "runner_config": self.config.public_settings(),
                "input_hashes": {name: sha256(content) for name, content in inputs.items()},
            },
        )
        write_json(
            job.workspace / "opencode-effective-settings.json",
            {
                "runtime_config_sha256": sha256(canonical(runtime_config)),
                "child_cap": runtime_config["provider"][self.provider]["models"][self.model][
                    "limit"
                ]["output"]
                if not self.config.harness_managed
                else None,
                "wire_cap": self.config.max_tokens or job.max_tokens,
                "wire_cap_env": (
                    "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"
                    if not self.config.harness_managed
                    else None
                ),
            },
        )
        stdout_path = job.workspace / "opencode.stdout.jsonl"
        stderr_path = job.workspace / "opencode.stderr.log"
        endpoint_lock = runtime / f"endpoint-{sha256(self.config.base_url.rstrip('/'))[:20]}.lock"
        waiting = time.monotonic()
        server_started_seconds = 0.0
        server = None
        process = None
        server_response = None
        try:
            with (
                FileLock(str(endpoint_lock), timeout=self.config.timeout_seconds),
                stdout_path.open("wb") as stdout,
                stderr_path.open("wb") as stderr,
            ):
                acquired = time.monotonic()
                if self.config.opencode_transport == "server":
                    key = sha256(env["OPENCODE_CONFIG_CONTENT"])
                    server = self._servers.get(key)
                    if server is None or not server.alive:
                        if server is not None:
                            server.close()
                        if (
                            server is None
                            and self.config.structured_decoding != "prompt"
                            and len(self._servers) >= 2
                        ):
                            oldest = next(iter(self._servers))
                            self._servers.pop(oldest).close()
                        server = OpenCodeServer(
                            executable, runtime, env, self.config.timeout_seconds
                        )
                        self._servers[key] = server
                        server_started_seconds = server.started_seconds
                    invocation = json.loads(
                        (job.workspace / "opencode-invocation.json").read_text("utf-8")
                    )
                    invocation.update(server_url=server.url, server_pid=server.process.pid)
                    write_json(job.workspace / "opencode-invocation.json", invocation)
                process_started_at = time.time()
                if server is not None:
                    try:
                        created = server.call(
                            "/session",
                            {"title": f"{job.run_id}/{job.chunk_id}/{job.stage}"},
                            self.config.timeout_seconds,
                        )
                        session_id = created.get("id", "")
                        if not re.fullmatch(r"ses_[A-Za-z0-9]+", session_id):
                            raise RunnerExecutionError(
                                "OpenCode server returned invalid session ID"
                            )
                        request = {
                            "agent": self.effective_agent_id,
                            "model": {"providerID": self.provider, "modelID": self.model},
                            "parts": [{"type": "text", "text": payload}],
                        }
                        variant = self.config.opencode_variant
                        if variant:
                            request["variant"] = variant
                        write_json(job.workspace / "opencode-server-session.json", created)
                        write_json(job.workspace / "opencode-server-request.json", request)
                        server_response = server.call(
                            f"/session/{session_id}/message", request, self.config.timeout_seconds
                        )
                        write_json(job.workspace / "opencode-server-response.json", server_response)
                    except BaseException:
                        self.close()
                        raise
                else:
                    process = subprocess.Popen(
                        args,
                        cwd=workspace,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=stdout,
                        stderr=stderr,
                        shell=False,
                        start_new_session=os.name != "nt",
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    try:
                        process.communicate(
                            payload.encode("utf-8"), timeout=self.config.timeout_seconds
                        )
                    except subprocess.TimeoutExpired as error:
                        stop_process_tree(process)
                        self.close()
                        raise RunnerExecutionError(
                            "OpenCode timeout; process tree stopped", failure_kind="timeout"
                        ) from error
                    except BaseException:
                        stop_process_tree(process)
                        self.close()
                        raise
        except Timeout as error:
            raise RunnerExecutionError(
                "OpenCode endpoint busy; retry later", failure_kind="endpoint_busy"
            ) from error
        for name, content in inputs.items():
            path = workspace / name
            if not path.is_file() or sha256(path.read_bytes()) != sha256(content):
                raise RunnerExecutionError("OpenCode modified an input file; job rejected")
        if process is not None and process.returncode:
            for line in stdout_path.read_text("utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    deferred = quota_event(event)
                    if deferred:
                        raise deferred
            raise RunnerExecutionError(
                f"OpenCode exited with code {process.returncode}; inspect stderr"
            )
        try:
            result = (
                parse_server_response(server_response, session_id, self.config.model)
                if server is not None
                else parse_events(stdout_path)
            )
        except RunnerExecutionError as error:
            # Preserve a safe raw artifact reference even when the turn is rejected.
            if error.artifact is None:
                error.artifact = (
                    str(job.workspace / "opencode-server-response.json")
                    if server is not None
                    else str(stdout_path)
                )
            raise
        session = result.raw_response["session_id"]
        if session in self._sessions:
            raise RunnerExecutionError("OpenCode reused an earlier session; job rejected")
        self._sessions.add(session)
        result.model = self.config.model
        result.raw_response["job_workspace"] = str(workspace)
        elapsed = time.monotonic() - acquired
        result.timings = {"endpoint_wait_seconds": acquired - waiting}
        if server is None:
            result.timings["process_seconds"] = elapsed
        else:
            result.timings["server_request_seconds"] = max(0, elapsed - server_started_seconds)
        result.timings["server_start_seconds"] = server_started_seconds
        first = result.raw_response.get("event_timestamps", {}).get("step_start")
        if (
            first is not None
            and 0 <= first - process_started_at <= elapsed
        ):
            result.timings["before_first_step_seconds"] = first - process_started_at
        result.effective_settings = {
            "model": self.config.model,
            "model_source": "reported" if server is not None else "configured",
            "execution": {
                "resolved_model": {
                    "provider": self.provider,
                    "model": self.model,
                    "source": "harness-reported" if server is not None else "configured",
                }
            },
            "max_tokens": self.config.max_tokens or job.max_tokens,
            "temperature": (
                self.config.temperature
                if "temperature" in self.config.model_fields_set
                else job.temperature
            ),
            "options": structured_params(self.config, job),
            "reasoning_effort": self.config.reasoning_effort,
            "transport": self.config.opencode_transport,
        }
        return result

    def close(self):
        for server in self._servers.values():
            server.close()
        self._servers.clear()
