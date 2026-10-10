"""Owned, authenticated loopback OpenCode backend; sessions remain independent."""

import base64
import json
import os
import re
import secrets
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from ..artifacts import write_json
from .base import RunnerExecutionError, RunnerResult, incomplete_finish_kind, usage_from_tokens
from .codex import stop_process_tree
from .retry import quota_event


class OpenCodeServer:
    def __init__(self, executable: str, root: Path, env: dict, timeout: float):
        self.directory = root / "servers" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.password = secrets.token_urlsafe(32)
        self.url = None
        self.process = None
        self.started_seconds = 0.0
        self.opener = build_opener(ProxyHandler({}))
        started = time.monotonic()
        stdout_path = self.directory / "server.stdout.log"
        args = [executable, "serve", "--pure", "--hostname", "127.0.0.1", "--port", "0"]
        try:
            with (
                stdout_path.open("wb") as stdout,
                (self.directory / "server.stderr.log").open("wb") as stderr,
            ):
                self.process = subprocess.Popen(
                    args,
                    cwd=self.directory,
                    env={**env, "OPENCODE_SERVER_PASSWORD": self.password},
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            while time.monotonic() - started < min(timeout, 60):
                if self.process.poll() is not None:
                    raise RunnerExecutionError(
                        "OpenCode server exited during startup; inspect server logs"
                    )
                match = re.search(
                    r"http://127\.0\.0\.1:\d+", stdout_path.read_text("utf-8", errors="replace")
                )
                if match:
                    self.url = match[0]
                    try:
                        with self.opener.open(
                            self.request("/global/health"), timeout=1
                        ) as response:
                            if json.load(response).get("healthy") is True:
                                break
                    except (URLError, OSError, ValueError):
                        pass
                time.sleep(0.1)
            else:
                raise RunnerExecutionError("OpenCode server startup timed out")
            self.started_seconds = time.monotonic() - started
            write_json(
                self.directory / "server.json",
                {
                    "pid": self.process.pid,
                    "url": self.url,
                    "argv": args,
                    "auth": "ephemeral; not persisted",
                },
            )
        except BaseException:
            self.close()
            raise

    def request(self, path: str) -> Request:
        auth = base64.b64encode(f"opencode:{self.password}".encode()).decode()
        return Request(self.url + path, headers={"Authorization": f"Basic {auth}"})

    def call(self, path: str, data: dict, timeout: float):
        request = self.request(path)
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                deferred = quota_event({"type": "error", "error": json.load(error)})
                if deferred:
                    raise deferred
            except (ValueError, TypeError):
                pass
            raise RunnerExecutionError(f"OpenCode server HTTP {error.code}") from error
        except (URLError, OSError, ValueError) as error:
            timed_out = isinstance(error, TimeoutError) or isinstance(
                getattr(error, "reason", None), TimeoutError
            )
            raise RunnerExecutionError(
                "OpenCode server request failed or timed out",
                failure_kind="timeout" if timed_out else "endpoint_busy",
            ) from error

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.process is not None:
            stop_process_tree(self.process)


def parse_server_response(value: dict, session: str, expected_model: str) -> RunnerResult:
    info = value.get("info") or {}
    if info.get("error"):
        deferred = quota_event({"type": "error", "error": info["error"]})
        if deferred:
            raise deferred
        raise RunnerExecutionError(
            "OpenCode server returned a model error; inspect response", failure_kind="runner_error"
        )
    tokens = info.get("tokens") or {}
    usage = usage_from_tokens(tokens)
    if (
        info.get("sessionID") != session
        or info.get("role") != "assistant"
        or not (info.get("time") or {}).get("completed")
        or info.get("finish") not in {"stop", "end_turn"}
    ):
        raise RunnerExecutionError(
            "OpenCode server returned no completed assistant turn",
            failure_kind=incomplete_finish_kind(info.get("finish")),
            finish_reason=info.get("finish"),
            usage=usage,
        )
    model = f"{info.get('providerID')}/{info.get('modelID')}"
    if model != expected_model:
        raise RunnerExecutionError(
            "OpenCode server reported an unexpected model", failure_kind="runner_error"
        )
    parts = value.get("parts") or []
    if any(p.get("type") == "tool" for p in parts):
        raise RunnerExecutionError(
            "OpenCode server attempted a tool in a tool-free job", failure_kind="runner_error"
        )
    if any(p.get("sessionID") != session or p.get("messageID") != info.get("id") for p in parts):
        raise RunnerExecutionError(
            "OpenCode server response contains unrelated parts", failure_kind="runner_error"
        )
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    if not text.strip():
        kind = "reasoning_only" if usage.get("reasoning_tokens") else "incomplete_response"
        raise RunnerExecutionError(
            "OpenCode server returned no final text",
            failure_kind=kind,
            finish_reason=info.get("finish"),
            usage=usage,
        )
    return RunnerResult(
        text=text,
        model=model,
        finish_reason=info["finish"],
        usage=usage,
        raw_response={
            "harness": "opencode",
            "transport": "server",
            "session_id": session,
            "response": value,
        },
    )
