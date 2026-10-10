import json

import httpx
import pytest
from openai import OpenAI

from novel_harness.config import RunnerConfig
from novel_harness.runners.base import Runner, TranslationJob
from novel_harness.runners.openai import DirectOpenAIRunner


@pytest.mark.parametrize("decoding", ["prompt", "json-schema", "llama-grammar"])
def test_openai_wire_contract(tmp_path, decoding):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "reported-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Русский перевод."},
                    }
                ],
            },
        )

    cfg = RunnerConfig(
        model="selected-model",
        structured_decoding=decoding,
        base_url="http://local.test/v1",
        extra_params={"seed": 7, "chat_template_kwargs": {"enable_thinking": False}},
    )
    with OpenAI(
        api_key="EMPTY",
        base_url=cfg.base_url,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        runner = DirectOpenAIRunner(cfg, "EMPTY", client)
        assert isinstance(runner, Runner)
        job = TranslationJob(
            "run", 1, "chunk", "translation", "Translate.", "原文。", {}, {}, tmp_path
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
        result = runner.run(job)
    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "http://local.test/v1/chat/completions"
    assert body["model"] == "selected-model"
    assert body["seed"] == 7
    assert body["max_tokens"] == 8192
    payload = json.loads(body["messages"][1]["content"])
    if decoding == "prompt":
        assert payload["source"] == "原文。"
        assert "grammar" not in body and "response_format" not in body
    elif decoding == "json-schema":
        assert body["response_format"]["json_schema"]["schema"] == job.response_schema
    else:
        assert "root ::=" in body["grammar"]
        assert job.segments[0]["id"] in body["grammar"]
    assert result.effective_settings["extra_params"] == {
        k: body[k]
        for k in ["seed", "chat_template_kwargs", "grammar", "response_format"]
        if k in body
    }
    assert result.model == "reported-model"
    assert result.usage == {"prompt_tokens": None, "completion_tokens": None}
    assert result.effective_settings["model"] == "selected-model"
    assert result.effective_settings["model_source"] == "configured"
    assert result.effective_settings["reported_model"] == "reported-model"


def test_openai_runner_surfaces_nonterminal_finish_reason(tmp_path):
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "reported-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": "Русский перевод."},
                    }
                ],
            },
        )

    cfg = RunnerConfig(model="selected-model", base_url="http://local.test/v1")
    with OpenAI(
        api_key="EMPTY",
        base_url=cfg.base_url,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        result = DirectOpenAIRunner(cfg, "EMPTY", client).run(
            TranslationJob(
                "run", 1, "chunk", "translation", "Translate.", "原文。", {}, {}, tmp_path
            )
        )

    assert result.finish_reason == "tool_calls"


def test_reasoning_profile_omits_temperature_and_legacy_token_limit(tmp_path):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "reasoning-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "Перевод."},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )

    cfg = RunnerConfig(
        model="reasoning-model",
        temperature=None,
        extra_params={"max_completion_tokens": 1234, "reasoning_effort": "low"},
    )
    with OpenAI(
        api_key="EMPTY", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ) as client:
        result = DirectOpenAIRunner(cfg, "EMPTY", client).run(
            TranslationJob(
                "run", 1, "chunk", "translation", "Translate.", "原文。", {}, {}, tmp_path
            )
        )
    assert "temperature" not in bodies[0]
    assert "max_tokens" not in bodies[0]
    assert bodies[0]["max_completion_tokens"] == 1234
    assert result.finish_reason == "length"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}
