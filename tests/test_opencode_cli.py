"""Opt-in contract test against installed OpenCode; loopback SSE only, no model calls."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from novel_harness.config import RunnerConfig
from novel_harness.runners.base import TranslationJob
from novel_harness.runners.opencode import OpenCodeRunner


@pytest.mark.skipif(
    os.environ.get("TEST_OPENCODE_CLI") != "1", reason="Opt-in installed CLI contract"
)
@pytest.mark.parametrize("transport", ["cli", "server"])
@pytest.mark.parametrize("decoding", ["prompt", "json-schema", "llama-grammar"])
def test_v17_installed_cli_sends_single_tool_free_bounded_request(tmp_path, transport, decoding):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            text = "Глава первая\n\nПроверочный русский текст."
            for delta, reason in [({"role": "assistant", "content": text}, None), ({}, "stop")]:
                event = {
                    "id": "mock",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "qwen-test",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
                }
                if reason:
                    event["usage"] = {
                        "prompt_tokens": 17,
                        "completion_tokens": 23,
                        "total_tokens": 40,
                    }
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    workspace = tmp_path / "attempt-0001"
    workspace.mkdir()
    job = TranslationJob(
        "contract",
        1,
        "0001-hash",
        "translation",
        "Translate Japanese to Russian.",
        "第一章\n\n彼は森へ行った。",
        {},
        {},
        workspace,
    )
    if decoding != "prompt":
        from dataclasses import replace

        from novel_harness.translation.segments import source_segments, translation_schema

        segments = source_segments(job.source)
        job = replace(
            job,
            segments=segments,
            response_schema=translation_schema(segments),
            output_mode="segments",
        )
    config = RunnerConfig(
        structured_decoding=decoding,
        type="opencode",
        model="local/qwen-test",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        timeout_seconds=60,
        max_tokens=12000,
        extra_params={"chat_template_kwargs": {"enable_thinking": False}},
        opencode_transport=transport,
    )
    runner = OpenCodeRunner(config, tmp_path / "jobs")
    try:
        result = runner.run(job)
        if transport == "server":
            from dataclasses import replace

            second_workspace = tmp_path / "attempt-0002"
            second_workspace.mkdir()
            second_job = replace(job, source="別の章。", workspace=second_workspace)
            if decoding != "prompt":
                second_segments = source_segments(second_job.source)
                second_job = replace(
                    second_job,
                    segments=second_segments,
                    response_schema=translation_schema(second_segments),
                )
            second = runner.run(second_job)
            assert second.raw_response["session_id"] != result.raw_response["session_id"]
            first_call = json.loads((workspace / "opencode-invocation.json").read_text("utf-8"))
            second_call = json.loads(
                (second_workspace / "opencode-invocation.json").read_text("utf-8")
            )
            if decoding == "prompt":
                assert first_call["server_pid"] == second_call["server_pid"]
                assert second.timings["server_start_seconds"] == 0
            else:
                assert first_call["server_pid"] != second_call["server_pid"]
                second_payload = json.loads(calls[1]["messages"][-1]["content"])
                assert second_payload["segments"][0]["source"] == "別の章。"
                if decoding == "llama-grammar":
                    assert calls[0]["grammar"] != calls[1]["grammar"]
            assert job.source not in json.dumps(calls[1], ensure_ascii=False)
    finally:
        owned = list(runner._servers.values())
        runner.close()
        assert all(not s.alive for s in owned)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.text == "Глава первая\n\nПроверочный русский текст."
    assert result.finish_reason == "stop"
    assert result.usage["prompt_tokens"] == 17
    assert result.usage["completion_tokens"] == 23
    assert len(calls) == (2 if transport == "server" else 1)
    request = calls[0]
    assert request["model"] == "qwen-test"
    assert not request.get("tools")
    assert request["max_tokens"] == 12000
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    payload = json.loads(request["messages"][-1]["content"])
    if decoding == "prompt":
        assert payload["source"] == job.source
        assert "grammar" not in request and "response_format" not in request
    elif decoding == "json-schema":
        assert request["response_format"]["json_schema"]["schema"] == job.response_schema
    else:
        from novel_harness.runners.structured import llama_grammar

        assert request["grammar"] == llama_grammar(job)
        assert len(runner._servers) == 0
