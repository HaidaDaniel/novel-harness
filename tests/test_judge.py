"""Structured judge parsing and opt-in service integration (offline)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from novel_harness.config import load_config
from novel_harness.judge import JudgeError, build_judge_job, parse_judge_response
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService

ROOT = Path(__file__).resolve().parents[1]


def test_parse_judge_response_accepts_valid_json():
    verdict = parse_judge_response('{"score": 4, "issues": ["punctuation"], "notes": "ok"}')
    assert verdict == {"score": 4, "issues": ["punctuation"], "notes": "ok"}


def test_parse_judge_response_strips_markdown_fence():
    verdict = parse_judge_response('```json\n{"score": 5, "issues": []}\n```')
    assert verdict["score"] == 5


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        "not json",
        '{"score": 9, "issues": []}',
        '{"score": true, "issues": []}',
        '{"score": "4", "issues": []}',
        '{"score": 4, "issues": "oops"}',
        '{"score": 4, "issues": [], "notes": 5}',
    ],
)
def test_parse_judge_response_rejects_malformed_payloads(payload):
    with pytest.raises(JudgeError):
        parse_judge_response(payload)


def test_build_judge_job_carries_source_and_schema(tmp_path):
    job = build_judge_job(
        run_id="r",
        chapter=1,
        chunk_id="c",
        source="原文",
        translation="Перевод",
        workspace=tmp_path,
    )
    assert job.stage == "judge"
    assert job.context == {"source": "原文"}
    assert job.source == "Перевод"
    assert job.response_schema is not None


class TextRunner:
    def __init__(self, model):
        self.model = model

    def run(self, job):
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model=self.model, finish_reason="stop")


class JudgeRunner:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def run(self, job):
        self.calls += 1
        return RunnerResult(self.payload, model="judge-model", finish_reason="stop")


def _make_project(tmp_path):
    (tmp_path / "original").mkdir()
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["translation"].update(
        chunk_chars=200, overlap_chars=0, retries=0, retry_delay_seconds=0
    )
    data["runners"] = {
        "test": {
            "type": "openai",
            "model": "fake-model",
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:9999/v1",
        },
        "judge": {
            "type": "openai",
            "model": "judge-model",
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:9999/v1",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    (tmp_path / "original" / "0001_first.txt").write_text(
        "第一章\n\n" + "彼は森を歩いていた。\n\n" * 14, "utf-8"
    )
    return path


def _service(path, payload='{"score": 4, "issues": []}'):
    agents = {"test": TextRunner("fake-model"), "judge": JudgeRunner(payload)}
    return TranslationService(load_config(path), lambda name: agents[name]), agents


def test_judge_run_scores_completed_chunks_without_touching_translation(tmp_path):
    path = _make_project(tmp_path)
    service, agents = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id)

    translation_paths = [
        service.run_dir(run_id) / row["output_path"] for row in service.repository.chunks(run_id)
    ]
    before = {path: path.read_bytes() for path in translation_paths}

    report = service.judge_run(run_id, runner="judge")

    assert report["scored"] > 0
    assert report["rejected"] == 0
    assert agents["judge"].calls == report["scored"]
    assert all(result["score"] == 4 for result in report["results"])
    assert (service.run_dir(run_id) / "judge.json").is_file()
    assert {path: path.read_bytes() for path in translation_paths} == before


def test_judge_run_rejects_malformed_judge_output(tmp_path):
    path = _make_project(tmp_path)
    service, _ = _service(path, payload="not json")
    run_id = service.create_run(runner="test")
    service.execute(run_id)

    report = service.judge_run(run_id, runner="judge")

    assert report["scored"] == 0
    assert report["rejected"] > 0
    assert all(result["score"] is None for result in report["results"])


def test_judge_run_requires_known_runner(tmp_path):
    path = _make_project(tmp_path)
    service, _ = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id)

    with pytest.raises(ValueError):
        service.judge_run(run_id, runner="missing")
