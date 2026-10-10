"""Offline typed MCP workflow tests; no provider or real worker process."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from conftest import FakeRunner

from novel_harness.config import load_config
from novel_harness.mcp import TOOLS, handle_request
from novel_harness.service import ConflictError, TranslationService


def _request(service, name, arguments=None, *, config_path=None):
    return handle_request(service, {
        "jsonrpc": "2.0", "id": 17, "method": "tools/call",
        "params": {"name": name, "arguments": arguments if arguments is not None else {}},
    }, config_path=config_path)


def _value(response):
    assert "result" in response, response
    return json.loads(response["result"]["content"][0]["text"])


def _reason(response):
    assert "error" in response, response
    return response["error"]["data"]["reason"]


def _service(project):
    return TranslationService(load_config(project), lambda _name: FakeRunner())


def test_tools_have_strict_schemas_and_keep_the_five_legacy_names(project):
    names = {item["name"] for item in TOOLS}
    assert {"list_runners", "create_run", "run_status", "inspect_chapter", "build_run"} <= names
    assert {"list_books", "get_book", "rescan_books", "list_presets", "list_connections",
            "create_sample", "sample_status", "approve_sample", "create_whole_book",
            "start_run", "pause_run", "resume_run", "retry_run", "run_progress",
            "set_final", "requeue_chunk"} <= names
    assert len(names) == len(TOOLS)
    assert all(tool["inputSchema"]["additionalProperties"] is False for tool in TOOLS)
    service = _service(project)
    for arguments in ({"run_id": 1}, {"run_id": True}, {"run_id": "x", "secret": "CANARY"}):
        response = _request(service, "run_status", arguments)
        assert _reason(response) == "invalid_arguments"
        assert "CANARY" not in json.dumps(response)
    assert _reason(_request(service, "create_sample", {
        "runner": "test", "chapters": [True],
    })) == "invalid_arguments"
    assert _reason(_request(service, "create_run", {
        "temperature": math.nan,
    })) == "invalid_arguments"
    for approval_value in (False, 1, "true"):
        response = _request(service, "approve_sample", {
            "run_id": "fake", "actor": "user", "user_approved": approval_value,
        })
        assert _reason(response) == "invalid_arguments"
    assert _reason(_request(service, "create_sample", {
        "preset": "quality", "runner": "test",
    })) == "invalid_arguments"
    assert _reason(_request(service, "set_final", {
        "run_id": "x", "chapter": 1, "chunk_id": "c", "action": "accept",
        "actor": "user", "expected_machine_sha256": "wrong",
    })) == "invalid_arguments"
    assert _reason(_request(service, "set_final", {
        "run_id": "x", "chapter": 1, "chunk_id": "c", "action": "edit",
        "actor": "user", "expected_machine_sha256": "0" * 64,
    })) == "invalid_arguments"


def test_read_projections_omit_raw_source_and_secret_bearing_fields():
    canary = "CANARY-SECRET-SOURCE"
    fake = SimpleNamespace(
        runners=lambda: [{"name": "test", "type": "openai", "model": "model",
                          "available": False, "reason": canary, "executable": canary,
                          "capabilities": {"supports_tools": True, "secret": canary}}],
        presets=lambda: [{"id": "preset", "available": False, "reason": canary,
                          "runner": "test", "pipeline": "direct"}],
        default_preset=lambda: None,
        library_novels=lambda: {"novels": [{
            "novel_id": "book", "title": "Book", "status": "ready",
            "source_path": canary, "body": canary, "error": canary,
        }]},
        library_show=lambda _id: {
            "novel_id": "book", "title": "Book", "status": "ready",
            "chapters": [{"number": 1, "body": canary}], "source_path": canary,
        },
        connections=lambda: {"profiles": [{
            "id": "profile", "available": False, "configuration": {"apiKey": canary},
            "reason": canary,
        }]},
        status=lambda run_id=None: {
            "runs": [{"run_id": "run", "status": "failed", "error": canary}]
        } if run_id is None else {"run_id": run_id, "status": "failed", "error": canary},
    )
    for name, arguments in (
        ("list_runners", {}), ("list_presets", {}),
        ("list_books", {}), ("get_book", {"novel_id": "book"}),
        ("list_connections", {}), ("list_runs", {}),
        ("run_status", {"run_id": "run"}),
    ):
        response = _request(fake, name, arguments)
        assert canary not in json.dumps(response)
        assert "result" in response
    assert _value(_request(fake, "get_book", {"novel_id": "book"}))["chapters"] == [1]
    assert _value(_request(fake, "list_runs"))["runs"][0]["error_present"] is True
    assert _reason(_request(SimpleNamespace(), "list_books")) == "capability_unavailable"


def test_book_sample_whole_book_worker_and_export_flow(project, monkeypatch):
    root = project.parent / "original"
    for title in ("alpha", "beta"):
        folder = root / title
        folder.mkdir()
        for number in (1, 2):
            (folder / f"{number:04d}_chapter.txt").write_text(
                f"第{number}章\n\n" + f"{title}の本文。" * 20, "utf-8"
            )
    service = _service(project)
    books = _value(_request(service, "list_books"))["novels"]
    assert {book["title"] for book in books} == {"alpha", "beta"}
    ids = {book["title"]: book["novel_id"] for book in books}
    assert ids["alpha"] != ids["beta"]
    detail = _value(_request(service, "get_book", {"novel_id": ids["alpha"]}))
    assert detail["chapters"] == [1, 2]
    assert "source_path" not in detail
    assert _reason(_request(service, "get_book", {"novel_id": "missing"})) == "novel_unavailable"
    assert {item["novel_id"] for item in _value(_request(service, "rescan_books"))["novels"]} == set(ids.values())

    sample = _value(_request(service, "create_sample", {
        "novel_id": ids["alpha"], "runner": "test",
    }))["run_id"]
    assert _value(_request(service, "sample_status", {"run_id": sample}))["approval"]["approved"] is False
    assert _reason(_request(service, "approve_sample", {
        "run_id": sample, "actor": "controller", "user_approved": True,
    })) == "sample_incomplete"
    assert _reason(_request(service, "create_whole_book", {
        "sample_run_id": sample, "approved": True,
    })) == "approval_required"
    assert _reason(_request(service, "create_whole_book", {
        "sample_run_id": sample, "approved": False,
    })) == "invalid_arguments"
    assert _reason(_request(service, "start_run", {"run_id": sample})) == "worker_unavailable"

    launches = []

    def fake_worker(config_path, run_id):
        launches.append((str(config_path), run_id))
        return {"run_id": run_id, "started": True, "already_running": False,
                "completed": False, "worker_status": "running",
                "worker_state": "/secret/local/path", "pid": 1234}

    monkeypatch.setattr("novel_harness.mcp.start_worker", fake_worker)
    started = _value(_request(service, "start_run", {"run_id": sample}, config_path=project))
    assert started["started"] is True
    assert "/secret/local/path" not in json.dumps(started)
    assert "pid" not in started
    assert _value(_request(service, "pause_run", {"run_id": sample}))["pause_requested"] is True
    assert _value(_request(service, "resume_run", {"run_id": sample}, config_path=project))["started"]
    assert _value(_request(service, "retry_run", {"run_id": sample}, config_path=project))["started"]
    assert launches == [(str(project), sample)] * 3
    assert _value(_request(service, "run_progress", {"run_id": sample}))["run_id"] == sample
    assert _value(_request(service, "run_status", {"run_id": sample}))["novel_id"] == ids["alpha"]

    service.execute(sample)
    assert _value(_request(service, "sample_status", {"run_id": sample}))["status"]["status"] == "completed"
    approval = _value(_request(service, "approve_sample", {
        "run_id": sample, "actor": "controller", "user_approved": True,
    }))
    assert approval["run_id"] == sample
    assert _value(_request(service, "sample_status", {"run_id": sample}))["approval"]["approved"]
    assert _reason(_request(service, "create_whole_book", {
        "sample_run_id": sample, "novel_id": ids["beta"], "approved": True,
    })) == "approval_drift"
    whole = _value(_request(service, "create_whole_book", {
        "sample_run_id": sample, "novel_id": ids["alpha"], "approved": True,
    }))["run_id"]
    assert service.repository.run(whole)["manifest"]["sample"]["sample_run_id"] == sample
    assert _value(_request(service, "start_run", {"run_id": whole}, config_path=project))["started"]
    service.execute(whole)
    epub = Path(_value(_request(service, "build_run", {"run_id": whole}))["path"])
    assert epub.is_file()


def test_final_cas_and_errors_are_safe(project, source, monkeypatch):
    service = _service(project)
    created = _value(_request(service, "create_run", {"runner": "test", "chapters": [1]}))
    run_id = created["run_id"]
    assert _reason(_request(service, "sample_status", {"run_id": run_id})) == "service_rejected"
    service.execute(run_id)
    chapter = _value(_request(service, "inspect_chapter", {"run_id": run_id, "chapter": 1}))
    assert "第一章" in chapter["source"]
    assert "chunks" not in chapter
    first = chapter["final_status"][0]
    payload = {
        "run_id": run_id, "chapter": 1, "chunk_id": first["chunk_id"],
        "action": "accept", "actor": "user",
        "expected_machine_sha256": first["machine_sha256"],
    }
    accepted = _value(_request(service, "set_final", payload))
    assert accepted["status"] == "accepted"
    assert "final_text" not in accepted
    assert _reason(_request(service, "set_final", payload)) == "cas_conflict"
    assert _reason(_request(service, "requeue_chunk", {
        "run_id": run_id, "chapter": 1, "chunk_id": first["chunk_id"],
        "expected_machine_sha256": "0" * 64,
    })) == "cas_conflict"
    assert _value(_request(service, "requeue_chunk", {
        "run_id": run_id, "chapter": 1, "chunk_id": first["chunk_id"],
        "expected_machine_sha256": first["machine_sha256"],
    }))["status"] == "pending"
    assert "error" in _request(service, "build_run", {"run_id": run_id})

    def broken_worker(_config, _run_id):
        raise OSError("CANARY-SECRET-PATH")

    monkeypatch.setattr("novel_harness.mcp.start_worker", broken_worker)
    response = _request(service, "retry_run", {"run_id": run_id}, config_path=project)
    assert _reason(response) == "worker_unavailable"
    assert "CANARY-SECRET-PATH" not in json.dumps(response)


def test_missing_capability_and_bad_rpc_never_echo_input(project):
    service = _service(project)
    assert _reason(_request(SimpleNamespace(), "list_connections")) == "capability_unavailable"
    bad = handle_request(service, {"jsonrpc": "1.0", "id": 4, "method": "tools/list"})
    assert _reason(bad) == "invalid_request"
    unknown = handle_request(service, {"jsonrpc": "2.0", "id": 5, "method": "CANARY-SECRET-METHOD"})
    assert unknown["error"]["code"] == -32601
    assert "CANARY" not in json.dumps(unknown)
    assert handle_request(service, {
        "jsonrpc": "2.0", "method": "notifications/initialized",
    }) is None


def test_ambiguous_book_conflict_is_not_reported_as_final_layer_cas():
    def ambiguous(_id):
        raise ConflictError("Ambiguous novel identity: CANARY-SECRET-BOOK")

    fake = SimpleNamespace(library_show=ambiguous)
    response = _request(fake, "get_book", {"novel_id": "book"})
    assert _reason(response) == "novel_unavailable"
    assert "CANARY-SECRET-BOOK" not in json.dumps(response)


def test_busy_sample_gate_has_bounded_worker_error():
    def busy(_run_id, **_kwargs):
        raise ValueError("Sample run is busy; retry after execution")

    response = _request(SimpleNamespace(approve_sample=busy), "approve_sample", {
        "run_id": "sample", "actor": "user", "user_approved": True,
    })
    assert _reason(response) == "worker_unavailable"


def test_profile_and_worker_failures_use_fixed_reasons_without_exception_text(project, monkeypatch):
    canary = "CANARY-AUTH-VALUE"

    def unavailable(**_kwargs):
        raise ValueError(f"Execution profile unavailable: {canary}")

    fake = SimpleNamespace(create_sample_run=unavailable)
    response = _request(fake, "create_sample", {"execution_profile": "bad"})
    assert _reason(response) == "profile_unavailable"
    assert canary not in json.dumps(response)

    monkeypatch.setattr("novel_harness.mcp.start_worker", lambda _config, run_id: {
        "run_id": run_id, "started": False, "already_running": True,
        "completed": False, "worker_status": canary, "worker_state": canary,
    })
    result = _value(_request(fake, "start_run", {"run_id": "existing"}, config_path=project))
    assert result["already_running"] is True
    assert result["worker_status"] == "unknown"
    assert canary not in json.dumps(result)


def test_cli_passes_its_config_path_to_stdio_worker_boundary(project, monkeypatch):
    from novel_harness import cli

    recorded = {}
    service = object()
    monkeypatch.setattr(cli.TranslationService, "from_config", lambda path: service)
    monkeypatch.setattr(
        "novel_harness.mcp.serve_stdio",
        lambda value, **kwargs: recorded.update(service=value, **kwargs),
    )
    cli.mcp(project)
    assert recorded == {"service": service, "config_path": project}


def test_invalid_mcp_envelopes_never_dispatch():
    from types import SimpleNamespace

    calls = []
    service = SimpleNamespace(runners=lambda: calls.append("dispatch") or [])
    valid = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "list_runners"}}
    invalid = [
        {key: value for key, value in valid.items() if key != "jsonrpc"},
        valid | {"jsonrpc": "1.0"},
        *[valid | {"id": value} for value in (False, True, [], {}, None, 1.5, float("nan"))],
        *[valid | {"method": value} for value in (None, 1, [], {})],
        *[valid | {"params": value} for value in (None, 1, "secret", [])],
        [], None,
    ]
    for request in invalid:
        response = handle_request(service, request)
        assert response["error"]["code"] == -32600, request
        assert response["id"] is None
    assert calls == []
    for request_id in (0, -1, "request-id"):
        response = handle_request(service, valid | {"id": request_id})
        assert response["id"] == request_id and "result" in response


def test_mcp_notifications_never_write_responses_even_on_dispatch_errors():
    from io import StringIO
    from types import SimpleNamespace

    from novel_harness.mcp import serve_stdio

    calls = []
    service = SimpleNamespace(runners=lambda: calls.append("dispatch") or [])
    notifications = [
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "unknown"},
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "list_runners"}},
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "run_status"}},
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "list_books"}},
    ]
    for request in notifications:
        assert handle_request(service, request) is None
    stdin = StringIO("\n".join(json.dumps(request) for request in notifications))
    stdout = StringIO()
    serve_stdio(service, stdin, stdout)
    assert stdout.getvalue() == ""
    assert calls == ["dispatch", "dispatch"]
    response = handle_request(service, {
        "jsonrpc": "2.0", "id": 4, "method": "notifications/initialized",
    })
    assert response["error"]["code"] == -32601
