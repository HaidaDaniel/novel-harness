"""Offline readiness-doctor tests. No model or provider inference is performed."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from novel_harness.config import AppConfig
from novel_harness.doctor import diagnose


@contextmanager
def models_endpoint():
    """Minimal local /v1/models endpoint; answers GET only, never inference."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_reports_reachable_and_unreachable_endpoints():
    with models_endpoint() as base_url:
        config = AppConfig(
            runners={
                "local": {
                    "type": "openai",
                    "model": "fake-model",
                    "base_url": base_url,
                    "api_key": "EMPTY",
                },
                "down": {
                    "type": "openai",
                    "model": "fake-model",
                    "base_url": "http://127.0.0.1:1/v1",
                    "api_key": "EMPTY",
                },
            }
        )
        report = diagnose(config, endpoint_timeout=0.5)

    by_name = {runner["name"]: runner for runner in report["runners"]}
    assert by_name["local"]["ready"] is True
    assert by_name["down"]["ready"] is False
    assert next(c for c in by_name["local"]["checks"] if c["id"] == "endpoint")["status"] == "pass"
    assert next(c for c in by_name["down"]["checks"] if c["id"] == "endpoint")["status"] == "fail"
    assert report["inference"] == "none"
    json.dumps(report)


def test_missing_model_is_not_ready():
    config = AppConfig(
        runners={
            "nokey": {"type": "openai", "base_url": "http://127.0.0.1:1/v1", "api_key": "EMPTY"}
        }
    )
    report = diagnose(config, endpoint_timeout=0.5)
    runner = report["runners"][0]
    assert runner["ready"] is False
    configuration = next(c for c in runner["checks"] if c["id"] == "configuration")
    assert configuration["status"] == "fail"
    assert "Model" in configuration["detail"]


def test_missing_cli_is_not_ready(monkeypatch):
    monkeypatch.setattr("novel_harness.runners.shutil.which", lambda name: None)
    config = AppConfig(runners={"codex": {"type": "codex", "model": "gpt-5.6-sol"}})
    report = diagnose(config)
    runner = report["runners"][0]
    assert runner["ready"] is False
    cli = next(c for c in runner["checks"] if c["id"] == "cli")
    assert cli["status"] == "fail"


def test_report_never_contains_api_key(monkeypatch):
    monkeypatch.setenv("NOVEL_DOCTOR_SECRET", "sk-secret-value-123")
    config = AppConfig(
        runners={
            "keyed": {
                "type": "openai",
                "model": "fake-model",
                "base_url": "http://127.0.0.1:1/v1",
                "api_key_env": "NOVEL_DOCTOR_SECRET",
            }
        }
    )
    report = diagnose(config, endpoint_timeout=0.5)
    assert "sk-secret-value-123" not in json.dumps(report)
