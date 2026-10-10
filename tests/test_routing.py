"""Deterministic routing selection and create_run/execute integration (offline)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from novel_harness.config import load_config
from novel_harness.routing import choose_runner, matched_names
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService

ROOT = Path(__file__).resolve().parents[1]


def _stage(role="translator", runner="test"):
    return {"role": role, "runner": runner}


def _routing(**overrides):
    values = {
        "escalation_runner": "strong",
        "escalate_over_chars": 10,
        "escalate_over_names": 1000,
        "runner_config": {"model": "strong-model"},
    }
    values.update(overrides)
    return values


def test_matched_names_counts_glossary_hits():
    glossary = {"characters": {"ルー": "Ру", "ミナ": "Мина"}, "terms": {"魔石": "магический камень"}}
    assert matched_names("ルーは魔石を持っていた", glossary) == 2
    assert matched_names("", glossary) == 0


def test_choose_runner_disabled_returns_default():
    decision = choose_runner(_stage(), {"text": "x" * 100}, {}, None)
    assert decision.escalated is False
    assert decision.runner == "test"


def test_choose_runner_only_escalates_translator_stage():
    decision = choose_runner(_stage(role="editor"), {"text": "x" * 100}, {}, _routing())
    assert decision.escalated is False


def test_choose_runner_length_trigger():
    decision = choose_runner(_stage(), {"text": "x" * 10}, {}, _routing())
    assert decision.escalated is True
    assert decision.runner == "strong"
    assert "source_chars" in decision.reason
    assert decision.runner_config == {"model": "strong-model"}


def test_choose_runner_glossary_trigger():
    glossary = {"characters": {"ルー": "Ру"}}
    decision = choose_runner(
        _stage(),
        {"text": "ルーは歩く"},
        glossary,
        _routing(escalate_over_names=1, escalate_over_chars=100000),
    )
    assert decision.escalated is True
    assert "glossary_names" in decision.reason


def test_choose_runner_no_trigger_returns_default():
    decision = choose_runner(
        _stage(),
        {"text": "short"},
        {},
        _routing(escalate_over_chars=100000, escalate_over_names=100000),
    )
    assert decision.escalated is False
    assert decision.runner == "test"


class CountingRunner:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def run(self, job):
        self.calls += 1
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model=self.model, finish_reason="stop")


def _make_project(tmp_path, *, routing=None):
    (tmp_path / "original").mkdir()
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["translation"].update(
        chunk_chars=200, overlap_chars=0, retries=0, retry_delay_seconds=0
    )
    if routing is not None:
        data["translation"]["routing"] = routing
    data["runners"] = {
        "test": {
            "type": "openai",
            "model": "fake-model",
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:9999/v1",
        },
        "strong": {
            "type": "openai",
            "model": "strong-model",
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


def _service(path):
    agents = {"test": CountingRunner("fake-model"), "strong": CountingRunner("strong-model")}
    return TranslationService(load_config(path), lambda name: agents[name]), agents


def test_execute_escalates_triggered_chunks(tmp_path):
    path = _make_project(
        tmp_path,
        routing={
            "enabled": True,
            "escalation_runner": "strong",
            "escalate_over_chars": 10,
            "escalate_over_names": 1000,
        },
    )
    service, agents = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id)

    assert agents["strong"].calls > 0
    assert agents["test"].calls == 0
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["routing"]["runner_config"]["model"] == "strong-model"
    rows = service.repository.chunks(run_id)
    assert any('"routing"' in (row["metadata"] or "") for row in rows)


def test_execute_keeps_default_when_no_trigger(tmp_path):
    path = _make_project(
        tmp_path,
        routing={
            "enabled": True,
            "escalation_runner": "strong",
            "escalate_over_chars": 100000,
            "escalate_over_names": 100000,
        },
    )
    service, agents = _service(path)
    run_id = service.create_run(runner="test")
    service.execute(run_id)

    assert agents["test"].calls > 0
    assert agents["strong"].calls == 0


def test_unknown_escalation_runner_is_rejected(tmp_path):
    path = _make_project(
        tmp_path,
        routing={"enabled": True, "escalation_runner": "missing"},
    )
    with pytest.raises(ValueError):
        load_config(path)
