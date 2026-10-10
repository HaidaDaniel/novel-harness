"""Agent-facing REST and optional MCP adapter tests (offline)."""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from novel_harness.api import create_app
from novel_harness.config import load_config
from novel_harness.mcp import handle_request, serve_stdio
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService


class FakeRunner:
    def __init__(self):
        self.calls = 0

    def run(self, job):
        self.calls += 1
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


def _client(project, runner):
    app = create_app(project, runner_factory=lambda name: runner)
    return TestClient(app), app


def test_rest_resume_delegates_to_worker(project, source, monkeypatch):
    client, _ = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]

    monkeypatch.setattr(
        "novel_harness.api.start_worker",
        lambda config, rid: {"run_id": rid, "started": True, "pid": 1},
    )

    response = client.post(f"/runs/{run_id}/resume")
    assert response.status_code == 202
    assert response.json()["run_id"] == run_id


def test_rest_audit_uses_service(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)

    response = client.get(f"/runs/{run_id}/audit")

    assert response.status_code == 200
    assert response.json()["run_id"] == run_id
    assert "blocking_chapters" in response.json()


def _service(project, runner):
    return TranslationService(load_config(project), lambda name: runner)


def test_mcp_initialize_and_tools_list(project):
    service = _service(project, FakeRunner())

    initialized = handle_request(service, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert initialized["result"]["protocolVersion"]

    listed = handle_request(service, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"list_runners", "create_run", "run_status", "inspect_chapter", "build_run"} <= names


def test_mcp_tool_call_maps_to_service(project, source):
    service = _service(project, FakeRunner())

    created = handle_request(
        service,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "create_run", "arguments": {"runner": "test", "chapters": [1]}},
        },
    )
    run_id = json.loads(created["result"]["content"][0]["text"])["run_id"]
    service.execute(run_id)

    status = handle_request(
        service,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "run_status", "arguments": {"run_id": run_id}},
        },
    )
    payload = json.loads(status["result"]["content"][0]["text"])
    assert payload["status"] == "completed"


def test_mcp_reports_unknown_method_and_tool(project):
    service = _service(project, FakeRunner())

    assert handle_request(service, {"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"] == -32601
    bad_tool = handle_request(
        service,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "nope", "arguments": {}}},
    )
    assert bad_tool["error"]["code"] == -32602


def test_mcp_stdio_handles_parse_error_and_requests(project):
    service = _service(project, FakeRunner())
    stdin = io.StringIO(
        "not json\n"
        + json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"})
        + "\n"
    )
    stdout = io.StringIO()

    serve_stdio(service, stdin=stdin, stdout=stdout)

    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["error"]["code"] == -32700
    assert lines[1]["id"] == 7
    assert "tools" in lines[1]["result"]


def test_mcp_is_independent_of_rest():
    import novel_harness.api as api_module
    import novel_harness.mcp as mcp_module

    assert not hasattr(api_module, "mcp")
    assert "fastapi" not in mcp_module.__dict__
