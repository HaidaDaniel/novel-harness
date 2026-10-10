import json

import pytest

from novel_harness.runners.base import RunnerResult
from novel_harness.translation.segments import (
    OutputContractError,
    decode_translation,
    render_segments,
    source_segments,
)


def test_v23_lossless_boundaries_and_heading():
    source = "題名\n\n　本文。\n次の文。\n\n"
    segments = source_segments(source, heading=True)
    assert [s["kind"] for s in segments] == ["heading", "body", "body"]
    translated = [
        {"id": s["id"], "text": t}
        for s, t in zip(segments, ["Заголовок", "Первый абзац.", "Второй абзац."], strict=True)
    ]
    assert decode_translation(json.dumps({"segments": translated}), segments) == translated
    assert (
        render_segments(source, segments, translated)
        == "Заголовок\n\nПервый абзац.\nВторой абзац.\n\n"
    )
    assert source_segments(source, heading=True) == segments


@pytest.mark.parametrize("change", ["missing", "duplicate", "reorder", "invented", "empty"])
def test_v23_rejects_bad_coverage(change):
    segments = source_segments("一文。\n\n二文。")
    translated = [{"id": s["id"], "text": "Перевод."} for s in segments]
    if change == "missing":
        translated.pop()
    if change == "duplicate":
        translated.append(translated[0])
    if change == "reorder":
        translated.reverse()
    if change == "invented":
        translated[0]["id"] = "invented"
    if change == "empty":
        translated[0]["text"] = ""
    with pytest.raises(OutputContractError):
        decode_translation(json.dumps({"segments": translated}), segments)


def test_v23_contract_failure_retries_and_preserves_raw_response(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    svc.config.translation.retries = 1
    calls = []

    class StructuredRunner:
        def run(self, job):
            calls.append(job)
            segments = [
                {"id": s["id"], "text": f"Русский перевод абзаца {i}."}
                for i, s in enumerate(job.segments)
            ]
            if len(calls) == 1:
                segments.pop()
            return RunnerResult(json.dumps({"segments": segments}), finish_reason="stop")

    svc.runner_factory = lambda _: StructuredRunner()
    rid = svc.create_run(runner="test")
    assert svc.execute(rid)["status"] == "completed"
    first = calls[0]
    attempts = svc.repository.attempts(rid, first.chapter, first.chunk_id, first.stage)
    assert attempts[0]["error"] == "source_coverage_mismatch"
    assert (first.workspace / "model-output.txt").is_file()
    assert calls[1].source == first.source
    assert "segments" in attempts[-1]
    assert svc.build_epub(rid).is_file()
    count = len(calls)
    svc.resume(rid)
    assert len(calls) == count
