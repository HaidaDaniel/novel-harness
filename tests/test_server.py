import pytest

from novel_harness.config import RunnerConfig
from novel_harness.runners.base import RunnerExecutionError
from novel_harness.runners.opencode_server import parse_server_response


def test_v28_transport_is_explicit_and_legacy_profiles_stable():
    legacy = RunnerConfig(type="opencode", model="local/model")
    assert "opencode_transport" not in legacy.public_settings()
    configured = RunnerConfig(type="opencode", model="local/model", opencode_transport="server")
    assert configured.public_settings()["opencode_transport"] == "server"


def test_v28_service_closes_owned_runner_after_pause_or_error(service):
    svc, fake = service
    closed = []
    fake.close = lambda: closed.append(True)
    rid = svc.create_run(runner="test")
    svc.execute(rid, max_chunks=1)
    assert closed == [True]
    fake.interrupt_at = len(fake.calls) + 1
    with pytest.raises(KeyboardInterrupt):
        svc.resume(rid)
    assert closed == [True, True]


@pytest.mark.parametrize(
    "defect, kind",
    [
        ("incomplete", "incomplete_response"),
        ("wrong_model", "runner_error"),
        ("tool", "runner_error"),
        ("foreign_part", "runner_error"),
        ("length", "length"),
    ],
)
def test_v28_server_rejects_invalid_or_unrelated_completion(defect, kind):
    value = {
        "info": {
            "id": "msg",
            "sessionID": "ses_a",
            "role": "assistant",
            "time": {"completed": 10},
            "finish": "stop",
            "providerID": "local",
            "modelID": "qwen",
        },
        "parts": [
            {"type": "text", "text": "Русский текст.", "sessionID": "ses_a", "messageID": "msg"}
        ],
    }
    if defect == "incomplete":
        value["info"]["time"] = {}
    if defect == "wrong_model":
        value["info"]["modelID"] = "other"
    if defect == "tool":
        value["parts"][0]["type"] = "tool"
    if defect == "foreign_part":
        value["parts"][0]["sessionID"] = "ses_other"
    if defect == "length":
        value["info"]["finish"] = "length"
    with pytest.raises(RunnerExecutionError) as info:
        parse_server_response(value, "ses_a", "local/qwen")
    assert info.value.failure_kind == kind


def test_v28_server_length_preserves_reported_finish_reason():
    value = {
        "info": {
            "id": "msg",
            "sessionID": "ses_a",
            "role": "assistant",
            "time": {"completed": 10},
            "finish": "length",
            "providerID": "local",
            "modelID": "qwen",
            "tokens": {"input": 10, "output": 4},
        },
        "parts": [],
    }
    with pytest.raises(RunnerExecutionError) as info:
        parse_server_response(value, "ses_a", "local/qwen")
    assert info.value.finish_reason == "length"
    assert info.value.usage["completion_tokens"] == 4


def test_v28_server_accepts_valid_completion():
    value = {
        "info": {
            "id": "msg",
            "sessionID": "ses_a",
            "role": "assistant",
            "time": {"completed": 10},
            "finish": "stop",
            "providerID": "local",
            "modelID": "qwen",
        },
        "parts": [
            {"type": "text", "text": "Русский текст.", "sessionID": "ses_a", "messageID": "msg"}
        ],
    }
    result = parse_server_response(value, "ses_a", "local/qwen")
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": None, "completion_tokens": None}


def test_v28_server_reasoning_only_is_classified():
    value = {
        "info": {
            "id": "msg",
            "sessionID": "ses_a",
            "role": "assistant",
            "time": {"completed": 10},
            "finish": "stop",
            "providerID": "local",
            "modelID": "qwen",
            "tokens": {"input": 10, "output": 0, "reasoning": 7},
        },
        "parts": [],
    }
    with pytest.raises(RunnerExecutionError) as info:
        parse_server_response(value, "ses_a", "local/qwen")
    assert info.value.failure_kind == "reasoning_only"
