"""Offline tests for bounded semantic-plan-v1 and the plan-translate pipeline."""

from __future__ import annotations

import json

import pytest
import yaml

from novel_harness.config import load_config
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService
from novel_harness.translation.semantic_plan import (
    PLAN_VERSION,
    PlanError,
    parse_plan,
    plan_sha256,
    render_plan,
)

VALID = {
    "plan_version": PLAN_VERSION,
    "items": [
        {
            "actor": "он",
            "action": "идти",
            "object": "в лес",
            "polarity": "negative",
            "mood": "question",
            "speaker": "рассказчик",
            "terminology": ["森"],
            "unresolved": ["кто именно говорит"],
        }
    ],
}


def test_parse_plan_accepts_bounded_fields():
    plan = parse_plan(json.dumps(VALID, ensure_ascii=False))
    assert plan["plan_version"] == PLAN_VERSION
    assert plan["items"][0]["polarity"] == "negative"
    assert plan["items"][0]["terminology"] == ["森"]


def test_parse_plan_rejects_free_form_and_unknown_fields():
    bad = {
        "plan_version": PLAN_VERSION,
        "items": [{"polarity": "affirmative", "mood": "statement", "reasoning": "let me think"}],
    }
    with pytest.raises(PlanError, match="unknown fields"):
        parse_plan(json.dumps(bad))


def test_parse_plan_rejects_invalid_variants():
    with pytest.raises(PlanError, match="not valid JSON"):
        parse_plan("not json")
    with pytest.raises(PlanError, match="plan_version"):
        parse_plan(json.dumps({"plan_version": "other", "items": VALID["items"]}))
    with pytest.raises(PlanError, match="polarity and mood"):
        parse_plan(json.dumps({"plan_version": PLAN_VERSION, "items": [{"actor": "он"}]}))
    with pytest.raises(PlanError, match="invalid polarity"):
        parse_plan(
            json.dumps(
                {
                    "plan_version": PLAN_VERSION,
                    "items": [{"polarity": "maybe", "mood": "statement"}],
                }
            )
        )
    with pytest.raises(PlanError, match="1..24"):
        parse_plan(json.dumps({"plan_version": PLAN_VERSION, "items": []}))
    with pytest.raises(PlanError, match="exceeds"):
        parse_plan(
            json.dumps(
                {
                    "plan_version": PLAN_VERSION,
                    "items": [{"polarity": "unknown", "mood": "unknown", "actor": "x" * 500}],
                }
            )
        )


def test_render_plan_is_deterministic_and_bounded():
    plan = parse_plan(json.dumps(VALID, ensure_ascii=False))
    first = render_plan(plan)
    assert first == render_plan(plan)
    assert "semantic_plan" in first
    assert "polarity=negative" in first
    assert plan_sha256(plan) == plan_sha256(parse_plan(json.dumps(VALID, ensure_ascii=False)))


#: Frozen P27_pipeline translator prompt SHA256 (translate-draft-v1).
BASELINE_TRANSLATOR_PROMPT_SHA = (
    "21609c2107264587636a4aaf792efe31dd41acdc62a52458994211bbf208dda2"
)


def _project_with_plan_pipeline(project):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["pipelines"]["plan-translate-v2"] = {
        "stages": [
            {"name": "planning", "role": "planner", "prompt_version": "translate-plan-v1"},
            {"name": "translation", "role": "translator", "prompt_version": "translate-draft-v1"},
        ]
    }
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")


class PlanRunner:
    def __init__(self, plan_text):
        self.plan_text = plan_text
        self.jobs = []

    def run(self, job):
        self.jobs.append(job)
        if job.stage == "planning":
            return RunnerResult(self.plan_text, model="fake-model", finish_reason="stop")
        return RunnerResult(
            f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(1, len(job.source) // 20),
            model="fake-model",
            finish_reason="stop",
        )


def test_plan_translate_injects_bounded_plan_and_stays_offline(project, source):
    _project_with_plan_pipeline(project)
    runner = PlanRunner(json.dumps(VALID, ensure_ascii=False))
    svc = TranslationService(load_config(project), lambda name: runner)

    run_id = svc.create_run(runner="test", pipeline="plan-translate-v2", chapters=[1])
    assert svc.execute(run_id)["status"] == "completed"

    stages = [job.stage for job in runner.jobs]
    assert "planning" in stages and "translation" in stages
    translator = next(job for job in runner.jobs if job.stage == "translation")
    assert "semantic_plan" in translator.context
    assert "polarity=negative" in translator.context["semantic_plan"]
    # The plan is never passed as a translation draft.
    assert translator.draft is None

    plan_path = next(svc.run_dir(run_id).rglob("planning/attempt-*/plan.json"))
    assert json.loads(plan_path.read_text("utf-8"))["plan_version"] == PLAN_VERSION
    manifest = svc.repository.run(run_id)["manifest"]
    assert manifest["stages"][0]["role"] == "planner"
    # Test A: the translator prompt is byte-identical to the frozen baseline prompt.
    assert manifest["stages"][1]["prompt_version"] == "translate-draft-v1"
    assert manifest["stages"][1]["prompt_sha256"] == BASELINE_TRANSLATOR_PROMPT_SHA
    # Test B: the planner sees the current chunk only, never previous context.
    planner_jobs = [job for job in runner.jobs if job.stage == "planning"]
    assert planner_jobs
    for job in planner_jobs:
        assert job.context == {}


def test_planner_excludes_previous_context_but_translator_keeps_it(project, source):
    _project_with_plan_pipeline(project)
    runner = PlanRunner(json.dumps(VALID, ensure_ascii=False))
    svc = TranslationService(load_config(project), lambda name: runner)

    run_id = svc.create_run(runner="test", pipeline="plan-translate-v2", chapters=[1])
    svc.execute(run_id)

    # Test B: no planner request contains previous source/translation.
    for job in (j for j in runner.jobs if j.stage == "planning"):
        assert "previous_japanese" not in job.context
        assert "previous_russian" not in job.context
        assert job.context == {}

    # Test C: later translator requests keep the baseline previous context and add
    # only the semantic plan.
    later = [
        j for j in runner.jobs
        if j.stage == "translation" and ("previous_japanese" in j.context or "previous_russian" in j.context)
    ]
    assert later, "fixture must yield a chunk with previous context"
    for job in later:
        assert "semantic_plan" in job.context


def test_plan_translate_fails_closed_on_free_form_plan(project, source):
    _project_with_plan_pipeline(project)
    bad = json.dumps(
        {
            "plan_version": PLAN_VERSION,
            "items": [{"polarity": "unknown", "mood": "unknown", "notes": "chain of thought"}],
        }
    )
    runner = PlanRunner(bad)
    svc = TranslationService(load_config(project), lambda name: runner)

    run_id = svc.create_run(runner="test", pipeline="plan-translate-v2", chapters=[1])
    assert svc.execute(run_id)["status"] != "completed"
    assert not any(job.stage == "translation" for job in runner.jobs)


def test_parse_plan_drops_blank_optional_fields():
    plan = parse_plan(
        json.dumps(
            {
                "plan_version": PLAN_VERSION,
                "items": [
                    {
                        "polarity": "unknown",
                        "mood": "unknown",
                        "actor": "",
                        "object": None,
                        "terminology": [],
                        "unresolved": ["", "  "],
                    }
                ],
            }
        )
    )
    item = plan["items"][0]
    assert "actor" not in item and "object" not in item
    assert "terminology" not in item and "unresolved" not in item


@pytest.mark.parametrize("finish_reason", [
    "length", "max_tokens", "content_filter", "tool_calls", "function_call", "paused", "unknown", "",
])
def test_nonterminal_planner_never_reaches_translation(project, source, finish_reason):
    _project_with_plan_pipeline(project)

    class NonterminalPlanner(PlanRunner):
        def run(self, job):
            result = super().run(job)
            if job.stage == "planning":
                return RunnerResult(result.text, model="fake-model", finish_reason=finish_reason,
                                    usage={"prompt_tokens": 17, "completion_tokens": 3})
            return result

    runner = NonterminalPlanner(json.dumps(VALID, ensure_ascii=False))
    svc = TranslationService(load_config(project), lambda _name: runner)
    rid = svc.create_run(runner="test", pipeline="plan-translate-v2", chapters=[1])
    assert svc.execute(rid)["status"] == "failed"
    assert not any(job.stage == "translation" for job in runner.jobs)
    row = svc.repository.chunks(rid)[0]
    attempt = svc.repository.attempts(rid, row["chapter"], row["chunk_id"], "planning")[-1]
    assert attempt["status"] == "failed"
    assert attempt["finish_reason"] == finish_reason
    assert attempt["usage"] == {"prompt_tokens": 17, "completion_tokens": 3}
    assert attempt["failure_kind"] in {"length", "content_filter", "incomplete_response"}
    assert not list(svc.run_dir(rid).rglob("planning/attempt-*/plan.json"))


@pytest.mark.parametrize("finish_reason", ["stop", "end_turn", None])
def test_terminal_or_legacy_planner_results_remain_compatible(project, source, finish_reason):
    _project_with_plan_pipeline(project)

    class CompatiblePlanner(PlanRunner):
        def run(self, job):
            result = super().run(job)
            if job.stage == "planning":
                return RunnerResult(result.text, finish_reason=finish_reason)
            return result

    runner = CompatiblePlanner(json.dumps(VALID, ensure_ascii=False))
    svc = TranslationService(load_config(project), lambda _name: runner)
    rid = svc.create_run(runner="test", pipeline="plan-translate-v2", chapters=[1])
    assert svc.execute(rid)["status"] == "completed"


@pytest.mark.parametrize("recover", [True, False])
def test_planner_incomplete_retry_is_bounded_and_keeps_failed_measurements(project, source, recover):
    _project_with_plan_pipeline(project)
    cfg = load_config(project)
    cfg.translation.retries = 1

    class RetryingPlanner(PlanRunner):
        def __init__(self):
            super().__init__(json.dumps(VALID, ensure_ascii=False))
            self.planning_calls = 0

        def run(self, job):
            result = super().run(job)
            if job.stage == "planning":
                self.planning_calls += 1
                if not recover or self.planning_calls == 1:
                    return RunnerResult(result.text, finish_reason="tool_calls",
                                        usage={"prompt_tokens": 17, "completion_tokens": 3})
            return result

    runner = RetryingPlanner()
    svc = TranslationService(cfg, lambda _name: runner)
    rid = svc.create_run(runner="test", pipeline="plan-translate-v2", chapters=[1])
    result = svc.execute(rid)
    assert result["status"] == ("completed" if recover else "failed")
    row = svc.repository.chunks(rid)[0]
    attempts = svc.repository.attempts(rid, row["chapter"], row["chunk_id"], "planning")
    assert len(attempts) == 2
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["usage"] == {"prompt_tokens": 17, "completion_tokens": 3}
    if not recover:
        assert runner.planning_calls == 2
        assert not any(job.stage == "translation" for job in runner.jobs)
