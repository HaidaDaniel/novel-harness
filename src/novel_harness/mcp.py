"""Typed JSON-RPC stdio adapter over TranslationService and the persisted worker."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from .progress import snapshot as progress_snapshot
from .service import ConflictError
from .worker import start_worker

MCP_PROTOCOL_VERSION = "2024-11-05"
_ID = {"type": "string", "minLength": 1, "maxLength": 160}
_ACTOR = {"type": "string", "minLength": 1, "maxLength": 100}
_CHAPTER = {"type": "integer", "minimum": 1}
_SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_RUN_ID = {"run_id": _ID}
_CHAPTER_REF = _RUN_ID | {"chapter": _CHAPTER}
_SELECTION = {
    "novel_id": _ID, "preset": _ID, "execution_profile": _ID,
    "runner": _ID, "pipeline": _ID, "translator": _ID, "editor": _ID,
    "chapters": {"type": "array", "items": _CHAPTER, "minItems": 1},
    "chunk_chars": {"type": "integer", "minimum": 1},
    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
    "prompt_version": _ID,
}


def _tool(name: str, description: str, properties: dict | None = None,
          required: tuple[str, ...] = ()) -> dict:
    return {"name": name, "description": description, "inputSchema": {
        "type": "object", "properties": properties or {}, "required": list(required),
        "additionalProperties": False,
    }}


TOOLS: list[dict[str, Any]] = [
    _tool("list_runners", "List configured runner availability."),
    _tool("create_run", "Create or reuse a frozen run without starting inference.",
          _SELECTION | {"force": {"type": "boolean"}}),
    _tool("run_status", "Read persisted run status.", _RUN_ID, ("run_id",)),
    _tool("inspect_chapter", "Read chapter source, translation and final CAS hashes.",
          _CHAPTER_REF, ("run_id", "chapter")),
    _tool("build_run", "Build a completed run's EPUB.", _RUN_ID, ("run_id",)),
    _tool("list_books", "List Library books without source bodies."),
    _tool("get_book", "Read one Library book's identity and chapter numbers.",
          {"novel_id": _ID}, ("novel_id",)),
    _tool("rescan_books", "Explicitly refresh the Library index."),
    _tool("list_presets", "List translation modes and availability."),
    _tool("list_connections", "List redacted configured execution profiles."),
    _tool("list_runs", "List saved runs without raw error text."),
    _tool("create_sample", "Create a frozen sample; start it separately.", _SELECTION),
    _tool("sample_status", "Read sample status and existing approval.", _RUN_ID, ("run_id",)),
    _tool("approve_sample", "Record an explicitly user-approved completed sample.",
          _RUN_ID | {"actor": _ACTOR, "user_approved": {"const": True}},
          ("run_id", "actor", "user_approved")),
    _tool("create_whole_book", "Create a run from an already approved sample.",
          _SELECTION | {"sample_run_id": _ID, "approved": {"const": True}},
          ("sample_run_id", "approved")),
    _tool("start_run", "Start through the persisted worker.", _RUN_ID, ("run_id",)),
    _tool("pause_run", "Pause at the next chunk boundary.", _RUN_ID, ("run_id",)),
    _tool("resume_run", "Resume through the persisted worker.", _RUN_ID, ("run_id",)),
    _tool("retry_run", "Retry through the persisted worker.", _RUN_ID, ("run_id",)),
    _tool("run_progress", "Read bounded worker progress without paths.", _RUN_ID, ("run_id",)),
    _tool("set_final", "Apply a human final-layer action with CAS hashes.",
          _CHAPTER_REF | {
              "chunk_id": _ID, "action": {"enum": ["accept", "edit", "needs_review", "revert"]},
              "actor": _ACTOR, "text": {"type": "string", "maxLength": 1_000_000},
              "expected_machine_sha256": _SHA, "expected_final_sha256": _SHA,
              "reason": {"type": "string", "maxLength": 500},
          }, ("run_id", "chapter", "chunk_id", "action", "actor", "expected_machine_sha256")),
    _tool("requeue_chunk", "Requeue a chunk with its expected machine CAS hash.",
          _CHAPTER_REF | {"chunk_id": _ID, "expected_machine_sha256": _SHA},
          ("run_id", "chapter", "chunk_id", "expected_machine_sha256")),
]
_TOOLS = {tool["name"]: tool for tool in TOOLS}
_MESSAGES = {
    "invalid_request": "Invalid JSON-RPC request",
    "invalid_arguments": "Invalid tool arguments",
    "unknown_tool": "Unknown tool",
    "capability_unavailable": "Service capability is unavailable",
    "worker_unavailable": "Persisted worker launch is unavailable",
    "novel_unavailable": "Book identity is unavailable",
    "profile_unavailable": "Execution profile is unavailable",
    "run_unavailable": "Run is unavailable",
    "resource_unavailable": "Required resource is unavailable",
    "sample_incomplete": "Sample must complete successfully before approval",
    "approval_required": "Explicit sample approval is required",
    "approval_drift": "Approved sample settings changed; create a new sample",
    "cas_conflict": "Final-layer CAS precondition failed; refresh before editing",
    "service_rejected": "Service rejected the operation",
    "internal_error": "MCP operation failed",
}


class MCPError(ValueError):
    """Safe bounded error reason; never embeds an argument value."""

    def __init__(self, reason: str, field: str | None = None):
        self.reason = reason
        self.field = field
        super().__init__(_MESSAGES[reason])


def _validate(value: Any, schema: dict, field: str) -> None:
    if "const" in schema and (
        value != schema["const"] or type(value) is not type(schema["const"])
    ):
        raise MCPError("invalid_arguments", field)
    if "enum" in schema and value not in schema["enum"]:
        raise MCPError("invalid_arguments", field)
    kind = schema.get("type")
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
    }
    if kind and not checks[kind]():
        raise MCPError("invalid_arguments", field)
    if kind == "object":
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                raise MCPError("invalid_arguments", name)
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    raise MCPError("invalid_arguments", "unknown_field")
        for name, item in value.items():
            _validate(item, props[name], name)
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > 10000:
            raise MCPError("invalid_arguments", field)
        for item in value:
            _validate(item, schema["items"], field)
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 1000000):
            raise MCPError("invalid_arguments", field)
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise MCPError("invalid_arguments", field)
    elif kind in {"integer", "number"}:
        if kind == "number" and not math.isfinite(value):
            raise MCPError("invalid_arguments", field)
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            raise MCPError("invalid_arguments", field)


def _method(service, name: str):
    value = getattr(service, name, None)
    if not callable(value):
        raise MCPError("capability_unavailable")
    return value


def _selection(arguments: dict) -> None:
    if arguments.get("preset") and any(key in arguments for key in (
        "runner", "execution_profile", "pipeline", "translator", "editor", "prompt_version"
    )):
        raise MCPError("invalid_arguments", "preset")


def _book(item: dict, detail: bool = False) -> dict:
    keys = ("novel_id", "title", "author", "status", "chapter_count", "source_type",
            "source_id", "identity_kind", "translation_state", "progress",
            "continue_run_id", "final_layer")
    result = {key: item.get(key) for key in keys}
    if detail:
        result["chapters"] = [chapter.get("number") for chapter in item.get("chapters", [])]
    return result


def _status(item: dict) -> dict:
    keys = ("run_id", "status", "generation_complete", "active", "workflow", "novel_id",
            "created_at", "updated_at", "duration_seconds", "chapters", "runners",
            "chunks", "total_chunks", "progress", "continued_from", "parent_run_id",
            "candidate_layer", "inherited_chunks")
    return {key: item.get(key) for key in keys} | {"error_present": bool(item.get("error"))}


def _runner(item: dict) -> dict:
    keys = ("name", "type", "model", "available")
    capability_keys = (
        "context_limit", "supports_temperature", "supports_output_limit",
        "supports_structured_output", "enforces_output_schema", "supports_tools",
        "supports_reasoning_effort", "source",
    )
    capabilities = item.get("capabilities") or {}
    return {key: item.get(key) for key in keys} | {
        "capabilities": {key: capabilities.get(key) for key in capability_keys},
        "availability_check": "configuration only; endpoint not probed",
        "reason": None if item.get("available") else "runner unavailable",
    }


def _preset(item: dict) -> dict:
    keys = ("id", "label", "description", "runner", "model", "pipeline",
            "translator", "editor", "category", "cloud", "recommended",
            "default", "available")
    return {key: item.get(key) for key in keys} | {
        "reason": None if item.get("available") else "preset unavailable",
    }


def _inspect(service, run_id: str, chapter: int) -> dict:
    data = _method(service, "inspect_chapter")(run_id, chapter)
    keys = ("run_id", "chapter", "title", "source", "source_sha256", "translation",
            "final", "complete")
    final_keys = ("ordinal", "chunk_id", "source", "status", "stale", "final_sha256",
                  "machine_sha256", "machine", "final")
    return {key: data.get(key) for key in keys} | {
        "final_status": [
            {key: item.get(key) for key in final_keys} for item in data.get("final_status", [])
        ],
        "validation": [
            {key: item.get(key) for key in ("ordinal", "status", "level", "issues")}
            for item in data.get("validation", [])
        ],
    }


def _launch(run_id: str, config_path: str | Path | None) -> dict:
    if config_path is None:
        raise MCPError("worker_unavailable")
    try:
        value = start_worker(config_path, run_id)
    except OSError as error:
        raise MCPError("worker_unavailable") from error
    worker_status = value.get("worker_status")
    if worker_status not in {"starting", "running", "completed", "failed", "paused", "interrupted"}:
        worker_status = "unknown"
    return {key: value.get(key) for key in
            ("run_id", "started", "already_running", "completed")} | {
        "worker_status": worker_status,
        "error_code": "worker_start_failed" if value.get("error") else None,
    }


def call_tool(service, name: str, arguments: dict, *, config_path: str | Path | None = None) -> Any:
    """Validate and delegate a tool call; never run translation inline."""
    tool = _TOOLS.get(name)
    if tool is None:
        raise MCPError("unknown_tool")
    _validate(arguments, tool["inputSchema"], "arguments")
    if name == "list_runners":
        return [_runner(item) for item in _method(service, "runners")()]
    if name in {"list_books", "rescan_books"}:
        method = "library_novels" if name == "list_books" else "rescan_library"
        return {"novels": [_book(item) for item in _method(service, method)()["novels"]]}
    if name == "get_book":
        return _book(_method(service, "library_show")(arguments["novel_id"]), detail=True)
    if name == "list_presets":
        return {"presets": [_preset(item) for item in _method(service, "presets")()],
                "default": _method(service, "default_preset")()}
    if name == "list_connections":
        data = _method(service, "connections")()
        keys = ("id", "kind", "harness", "available", "model_policy", "requested_model",
                "resolved_model", "model_source", "agent_id")
        return {"profiles": [
            {key: item.get(key) for key in keys} |
            {"reason": "profile unavailable" if not item.get("available") else None}
            for item in data.get("profiles", [])
        ]}
    if name == "list_runs":
        return {"runs": [_status(item) for item in _method(service, "status")()["runs"]]}
    if name in {"create_run", "create_sample"}:
        _selection(arguments)
        method = "create_run" if name == "create_run" else "create_sample_run"
        return {"run_id": _method(service, method)(**arguments)}
    if name == "create_whole_book":
        _selection(arguments)
        payload = {key: value for key, value in arguments.items() if key != "approved"}
        return {"run_id": _method(service, "create_whole_book_run")(
            payload.pop("sample_run_id"), **payload
        )}
    if name == "run_status":
        return _status(_method(service, "status")(arguments["run_id"]))
    if name == "sample_status":
        data = _method(service, "sample_status")(arguments["run_id"])
        approval = data["approval"]
        return {"status": _status(data["status"]), "approval": {
            "approved": approval is not None,
            "approved_at": approval.get("approved_at") if approval else None,
        }}
    if name == "approve_sample":
        return _method(service, "approve_sample")(
            arguments["run_id"], actor=arguments["actor"]
        )
    if name in {"start_run", "resume_run", "retry_run"}:
        return _launch(arguments["run_id"], config_path)
    if name == "pause_run":
        return _method(service, "pause")(arguments["run_id"])
    if name == "run_progress":
        _method(service, "status")(arguments["run_id"])
        progress = progress_snapshot(service, arguments["run_id"])
        keys = ("run_id", "status", "workflow", "chunks", "total_chunks", "progress",
                "chapters", "current", "active_count", "elapsed_seconds",
                "throughput_chunks_per_minute", "eta_seconds")
        return {key: progress.get(key) for key in keys}
    if name == "inspect_chapter":
        return _inspect(service, arguments["run_id"], arguments["chapter"])
    if name == "set_final":
        if arguments["action"] == "edit" and not arguments.get("text"):
            raise MCPError("invalid_arguments", "text")
        if arguments["action"] == "revert" and "text" in arguments:
            raise MCPError("invalid_arguments", "text")
        value = _method(service, "set_final")(
            arguments["run_id"], arguments["chapter"], arguments["chunk_id"],
            **{key: val for key, val in arguments.items()
               if key not in {"run_id", "chapter", "chunk_id"}},
        )
        return {key: value.get(key) for key in (
            "run_id", "chapter", "chunk_id", "status", "final_sha256", "machine_sha256",
            "updated_at",
        )}
    if name == "requeue_chunk":
        return _method(service, "requeue_chunk")(
            arguments["run_id"], arguments["chapter"], arguments["chunk_id"],
            expected_machine_sha256=arguments["expected_machine_sha256"],
        )
    if name == "build_run":
        return {"path": str(_method(service, "build_epub")(arguments["run_id"]))}
    raise MCPError("unknown_tool")


def _service_reason(error: Exception) -> str:
    if isinstance(error, ConflictError):
        if "ambiguous novel" in str(error).lower():
            return "novel_unavailable"
        return "cas_conflict"
    if isinstance(error, OSError):
        return "resource_unavailable"
    message = str(error).lower()
    if "unknown novel" in message or "ambiguous novel" in message:
        return "novel_unavailable"
    if "sample translation must complete" in message:
        return "sample_incomplete"
    if "changed" in message or "differ" in message or "drift" in message or "source moved" in message:
        return "approval_drift"
    if "approved sample" in message or "requires an approved sample" in message:
        return "approval_required"
    if "profile" in message or "runner unavailable" in message or "unknown runner" in message:
        return "profile_unavailable"
    if "unknown run" in message or ("run" in message and "not found" in message):
        return "run_unavailable"
    if "worker" in message or "another launch" in message or "sample run is busy" in message:
        return "worker_unavailable"
    return "service_rejected"


def _error(request_id: Any, code: int, reason: str, field: str | None = None) -> dict:
    data = {"reason": reason}
    if field is not None:
        data["field"] = field
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": _MESSAGES[reason], "data": data}}


def handle_request(service, request: Any, *, config_path: str | Path | None = None) -> dict | None:
    """Validate the MCP envelope before dispatch; notifications never reply."""
    if (
        not isinstance(request, dict)
        or request.get("jsonrpc") != "2.0"
        or not isinstance(request.get("method"), str)
        or ("params" in request and not isinstance(request["params"], dict))
        or ("id" in request and type(request["id"]) not in {str, int})
    ):
        return _error(None, -32600, "invalid_request")
    response = _dispatch_request(service, request, config_path=config_path)
    return response if "id" in request else None


def _dispatch_request(service, request: dict, *, config_path: str | Path | None) -> dict | None:
    """Return a bounded result/error for an already validated message."""
    request_id = request.get("id")
    method = request.get("method")
    if method == "notifications/initialized" and "id" not in request:
        return None
    try:
        if method == "initialize":
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "novel_harness", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = request.get("params")
            if not isinstance(params, dict):
                raise MCPError("invalid_arguments", "params")
            value = call_tool(
                service, params.get("name"), params.get("arguments", {}),
                config_path=config_path,
            )
            result = {"content": [
                {"type": "text", "text": json.dumps(value, ensure_ascii=False, default=str)}
            ]}
        else:
            return {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"}}
    except MCPError as error:
        return _error(request_id, -32602, error.reason, error.field)
    except (ValueError, OSError) as error:
        return _error(request_id, -32000, _service_reason(error))
    except Exception:  # noqa: BLE001 - never expose unexpected text on stdio
        return _error(request_id, -32603, "internal_error")
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve_stdio(service, stdin=None, stdout=None, *, config_path: str | Path | None = None) -> None:
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            response = _error(None, -32700, "invalid_request")
            response["error"]["message"] = "Parse error"
        else:
            response = handle_request(service, request, config_path=config_path)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()
