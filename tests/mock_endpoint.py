"""Deterministic loopback OpenAI-compatible fixtures.

The small ``endpoint`` context manager at the bottom is kept for existing unit
tests. ``MockProvider`` is the process-capable acceptance fixture: it records
bounded request metadata, supports explicit fault modes, and never calls a
real model or provider.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

MOCK_MODEL = "novel-harness-acceptance-mock"
MAX_LEDGER_REQUESTS = 256
SUPPORTED_MODES = frozenset(
    {"success", "delay", "delay-once", "429", "500", "disconnect", "length"}
)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _source_payload(body: dict[str, Any]) -> str:
    for message in body.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("source"), str):
            return payload["source"]
    return ""


def _synthetic_translation(source: str) -> str:
    """Return deterministic Cyrillic text sized for the production validator."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    target = max(36, min(320, int(len(source.strip()) * 0.8)))
    parts = []
    for index in range(32):
        marker = digest[index * 2 : index * 2 + 2]
        parts.append(
            f"Синтетический фрагмент {marker} содержит проверочный русский текст {index}."
        )
    return " ".join(parts)[:target].rstrip()


class MockProvider:
    """Threaded, loopback-only fake for the production OpenAI runner."""

    def __init__(
        self,
        *,
        mode: str = "success",
        delay_seconds: float = 0.05,
        failures_before_success: int = 0,
        failure_status: int | None = None,
        model: str = MOCK_MODEL,
        ledger_path: Path | None = None,
        legacy_output: bool = False,
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mock mode: {mode}")
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be nonnegative")
        if failures_before_success < 0:
            raise ValueError("failures_before_success must be nonnegative")
        if (
            failure_status is not None
            and (
                not isinstance(failure_status, int)
                or isinstance(failure_status, bool)
                or not 500 <= failure_status <= 599
            )
        ):
            raise ValueError("failure_status must be a 5xx status")
        self.mode = mode
        self.delay_seconds = delay_seconds
        self.failures_before_success = failures_before_success
        self.failure_status = failure_status
        self.model = model
        self.ledger_path = ledger_path
        self.legacy_output = legacy_output
        self.calls: list[dict[str, Any]] = []
        self._total_requests = 0
        self._delay_once_pending = mode == "delay-once"
        self._lock = threading.Lock()
        self._ledger_lock = threading.Lock()
        self._success_after_failures = False
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("MockProvider is not running")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def ledger_url(self) -> str:
        if self._server is None:
            raise RuntimeError("MockProvider is not running")
        return f"http://127.0.0.1:{self._server.server_port}/__status"

    def set_mode(self, mode: str, *, failures_before_success: int = 0) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mock mode: {mode}")
        if failures_before_success < 0:
            raise ValueError("failures_before_success must be nonnegative")
        with self._lock:
            self.mode = mode
            self.failures_before_success = failures_before_success
            self._delay_once_pending = mode == "delay-once"
            self._success_after_failures = False

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
            return {
                "schema_version": 1,
                "provider": "deterministic-loopback-mock",
                "model": self.model,
                "mock_model_requests": self._total_requests,
                "external_llm_requests": 0,
                "external_llm_tokens": 0,
                "requests": calls,
                "requests_truncated": max(0, self._total_requests - len(calls)),
            }

    def _write_ledger(self) -> None:
        if self.ledger_path is None:
            return
        with self._ledger_lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self._snapshot(), ensure_ascii=False, indent=2) + "\n", "utf-8"
            )
            temporary.replace(self.ledger_path)

    def _record(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.calls.append(entry)
            self._total_requests += 1
            if len(self.calls) > MAX_LEDGER_REQUESTS:
                del self.calls[: len(self.calls) - MAX_LEDGER_REQUESTS]
        self._write_ledger()

    def _response(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        return status, payload, headers or {}

    def _handle_models(self):
        return self._response(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "novel-harness-acceptance",
                    }
                ],
            },
        )

    def _handle_chat(self, body: dict[str, Any]):
        source = _source_payload(body)
        with self._lock:
            mode = self.mode
            if self._success_after_failures:
                selected_mode = "success"
            elif mode == "delay-once" and self._delay_once_pending:
                self._delay_once_pending = False
                selected_mode = mode
            elif mode == "delay-once":
                selected_mode = "success"
            elif self.failures_before_success > 0:
                self.failures_before_success -= 1
                selected_mode = mode
                if self.failures_before_success == 0:
                    self._success_after_failures = True
            else:
                selected_mode = mode
        if mode in {"delay", "delay-once"} and selected_mode == mode:
            time.sleep(self.delay_seconds)
        if selected_mode == "429":
            return self._response(
                429,
                {"error": {"message": "Synthetic quota response", "type": "rate_limit_error"}},
                headers={"Retry-After": "1"},
            )
        if selected_mode == "500":
            status = self.failure_status or 500
            return self._response(
                status,
                {"error": {"message": "Synthetic provider outage", "type": "server_error"}},
            )
        if selected_mode == "disconnect":
            return None
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        content = (
            "Тестовая глава\n\n" + "Проверочный русский текст. " * max(1, len(source) // 20)
            if self.legacy_output
            else _synthetic_translation(source)
        )
        finish_reason = "length" if selected_mode == "length" else "stop"
        if finish_reason == "length":
            content = content[: max(10, len(content) // 3)]
        prompt_tokens = 17 if self.legacy_output else 40 + len(source) // 4
        completion_tokens = 23 if self.legacy_output else max(8, len(content) // 4)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return self._response(
            200,
            {
                "id": f"mock-{digest[:16]}",
                "object": "chat.completion",
                "created": 0,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": usage,
            },
        )

    def _handler_class(self):
        provider = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "NovelHarnessAcceptanceMock/1"

            def log_message(self, *_args):
                pass

            def _send(self, status: int, payload: dict[str, Any], headers: dict[str, str]):
                raw = _json_bytes(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(raw)

            def _reject(self, status: int, message: str):
                provider._record(
                    {"path": self.path, "method": self.command, "status": status, "mode": "rejected"}
                )
                self._send(status, {"error": {"message": message, "type": "invalid_request_error"}}, {})

            def do_GET(self):
                if self.path == "/v1/models":
                    status, payload, headers = provider._handle_models()
                    provider._record(
                        {"path": self.path, "method": "GET", "status": status, "mode": "models"}
                    )
                    self._send(status, payload, headers)
                    return
                if self.path == "/__status":
                    self._send(200, provider._snapshot(), {})
                    return
                self._reject(404, "Unsupported mock endpoint path")

            def do_POST(self):
                if self.path == "/__control":
                    try:
                        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        provider.set_mode(
                            body["mode"],
                            failures_before_success=int(body.get("failures_before_success", 0)),
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                        self._reject(400, f"Invalid mock control: {type(error).__name__}")
                        return
                    self._send(200, {"mode": provider.mode}, {})
                    return
                if self.path != "/v1/chat/completions":
                    self._reject(404, "Unsupported mock endpoint path")
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    self._reject(400, "Invalid JSON request")
                    return
                source = _source_payload(body)
                source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
                result = provider._handle_chat(body)
                if result is None:
                    provider._record(
                        {
                            "path": self.path,
                            "method": "POST",
                            "status": "disconnect",
                            "mode": provider.mode,
                            "model": body.get("model"),
                            "source_sha256": source_hash,
                            "source_chars": len(source),
                        }
                    )
                    self.close_connection = True
                    return
                status, payload, headers = result
                usage = payload.get("usage") if isinstance(payload, dict) else None
                provider._record(
                    {
                        "path": self.path,
                        "method": "POST",
                        "status": status,
                        "mode": provider.mode,
                        "model": body.get("model"),
                        "source_sha256": source_hash,
                        "source_chars": len(source),
                        "usage": usage,
                    }
                )
                self._send(status, payload, headers)

            def do_PUT(self):
                self._reject(405, "Unsupported mock endpoint method")

            def do_DELETE(self):
                self._reject(405, "Unsupported mock endpoint method")

        return Handler

    def start(self) -> MockProvider:
        if self._server is not None:
            raise RuntimeError("MockProvider is already running")
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="novel-harness-acceptance-mock",
            daemon=True,
        )
        self._thread.start()
        self._write_ledger()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_args) -> None:
        self.stop()


@contextmanager
def endpoint(fail_first=False):
    """Backward-compatible fixture for existing tests."""
    provider = MockProvider(
        mode="500" if fail_first else "success",
        failures_before_success=1 if fail_first else 0,
        failure_status=503,
        legacy_output=True,
    )
    with provider:
        yield provider.base_url, provider.calls
