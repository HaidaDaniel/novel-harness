import copy
import json
from datetime import UTC, datetime, timedelta

import pytest
from filelock import Timeout
from typer.testing import CliRunner

from novel_harness.artifacts import sha256
from novel_harness.cache import TranslationCache
from novel_harness.cli import app
from novel_harness.config import RunnerConfig
from novel_harness.progress import snapshot
from novel_harness.runners.base import RunnerDeferred, RunnerExecutionError, RunnerResult
from novel_harness.service import TranslationService
from novel_harness.translation.segments import source_segments
from novel_harness.worker import run_worker


class ParentTranslator:
    def __init__(self, mode="text"):
        self.jobs = []
        self.mode = mode

    def run(self, job):
        self.jobs.append(job)
        originals = source_segments(job.source, heading=job.source.startswith("第一章"))
        translated = [
            {"id": s["id"], "text": f"Русский абзац номер {i}, глава {job.chapter}."}
            for i, s in enumerate(originals)
        ]
        if self.mode == "segments-v1":
            text = json.dumps({"segments": translated})
        else:
            # Real direct paragraph layout can differ from Japanese blank lines.
            text = "\r\n\r\n\r\n".join("\t" + s["text"] + "  " for s in translated) + "\r\n"
        return RunnerResult(text, model="translator-reported", finish_reason="stop")


def parent(service, mode="text", second=False):
    svc, _ = service
    if second:
        (svc.config.novel.source_dir / "0002_second.txt").write_text(
            "第二章\n\n彼は帰った。\n\n" * 8, "utf-8"
        )
    svc.config.translation.output_format = mode
    translator = ParentTranslator(mode)
    svc.runner_factory = lambda name: translator
    rid = svc.create_run(runner="test")
    assert svc.execute(rid)["status"] == "completed"
    svc.config.runners["reviewer"] = RunnerConfig(model="review-model")
    return svc, rid, translator


def inventory(svc, rid):
    return {
        str(p.relative_to(svc.run_dir(rid))): sha256(p.read_bytes())
        for p in svc.run_dir(rid).rglob("*")
        if p.is_file() and not p.name.endswith(".lock")
    }


class ReviewRunner:
    def __init__(self, fail=None, correction=False):
        self.jobs = []
        self.fail = fail
        self.correction = correction

    def run(self, job):
        self.jobs.append(copy.deepcopy(job))
        assert job.output_mode == "review" and job.draft_segments
        assert [s["id"] for s in job.segments] == [s["id"] for s in job.draft_segments]
        assert all(s["sha256"] == sha256(s["text"]) for s in job.draft_segments)
        if self.fail and len(self.jobs) == 1:
            if self.fail == "runner":
                raise RunnerExecutionError("offline", failure_kind="runner_error")
            if self.fail == "quota":
                raise RunnerDeferred(
                    retry_at=(datetime.now(UTC) + timedelta(seconds=3)).isoformat(), reason="quota"
                )
            if self.fail == "length":
                return RunnerResult("", finish_reason="length")
            return RunnerResult("{broken", finish_reason="stop")
        report = {"checked_ids": [s["id"] for s in job.segments], "issues": []}
        if self.correction:
            s = job.draft_segments[0]
            report["issues"] = [
                {
                    "id": s["id"],
                    "category": "meaning",
                    "reason": "Correct an actor",
                    "original_sha256": s["sha256"],
                    "replacement": "Исправленный русский абзац.",
                }
            ]
        return RunnerResult(json.dumps(report), finish_reason="stop", model="review-reported")


def review_factory(svc, runner):
    constructed = []

    def factory(name):
        constructed.append(name)
        assert name == "reviewer", "Translator construction forbidden in derived review"
        return runner

    svc.runner_factory = factory
    return constructed


@pytest.mark.parametrize("mode", ["text", "segments-v1"])
@pytest.mark.parametrize("cached", [False, True])
def test_direct_then_review_zero_translator_and_immutable(service, mode, cached):
    svc, rid, translator = parent(service, mode)
    before = inventory(svc, rid)
    parent_state = svc.repository.run(rid)
    original_rows = svc.repository.chunks(rid)
    count = len(translator.jobs)
    reviewer = ReviewRunner()
    constructed = review_factory(svc, reviewer)
    if cached:

        class ForbiddenCrossRunCache(TranslationCache):
            def get(self, key):
                pytest.fail("Derived review must not read rendered cross-run cache")

            def put(self, *args, **kwargs):
                pytest.fail("Derived review must not publish rendered cross-run cache")

        svc.stages.cache = ForbiddenCrossRunCache(svc.root / "cache")
    child = svc.create_review_run(rid, reviewer="reviewer")
    assert not constructed and not reviewer.jobs
    assert svc.status(child)["chunks"]["completed"] == 0
    assert svc.execute(child)["status"] == "completed"
    assert len(translator.jobs) == count
    assert len(reviewer.jobs) == len(original_rows)
    assert set(constructed) == {"reviewer"}
    for a, b in zip(original_rows, svc.repository.chunks(child), strict=True):
        assert a["output_sha256"] == b["output_sha256"]
    assert svc.inspect_chapter(child, 1)["parent_run_id"] == rid
    assert svc.create_review_run(rid, reviewer="reviewer") == child
    assert svc.create_review_run(rid, reviewer="reviewer", force=True) != child
    svc.resume(child)
    assert len(reviewer.jobs) == len(original_rows)
    assert svc.repository.run(rid) == parent_state
    assert svc.repository.chunks(rid) == original_rows
    assert inventory(svc, rid) == before
    assert svc.build_epub(child).is_file()


@pytest.mark.parametrize("failure", ["runner", "json", "length", "quota"])
def test_review_failure_resume_same_frozen_draft(service, failure):
    svc, rid, translator = parent(service)
    reviewer = ReviewRunner(fail=failure)
    review_factory(svc, reviewer)
    child = svc.create_review_run(rid, reviewer="reviewer")
    if failure in {"runner", "quota"}:
        with pytest.raises(RunnerExecutionError):
            svc.execute(child)
    else:
        assert svc.execute(child)["status"] == "failed"
    if failure == "quota":
        # Deadline is a distinct persisted policy; move test clock via service stub.
        svc.pending_retry = lambda _: None
    first = reviewer.jobs[0]
    svc2 = TranslationService(svc.config)
    if failure == "quota":
        svc2.pending_retry = lambda _: None
    review_factory(svc2, reviewer)
    assert svc2.resume(child)["status"] == "completed"
    second = reviewer.jobs[1]
    assert (first.source, first.draft, first.draft_segments, first.context) == (
        second.source,
        second.draft,
        second.draft_segments,
        second.context,
    )
    assert len(translator.jobs) == len(svc.repository.chunks(rid))


def test_derived_review_forces_strict_script_policy(service):
    svc, rid, _ = parent(service)
    child = svc.create_review_run(rid, reviewer="reviewer")
    policy = svc.repository.run(child)["manifest"]["validation"]
    assert policy["max_foreign_script_chars"] == 0
    assert policy["reject_malformed_dialogue"] is True


def test_minimal_correction_and_frozen_previous_context(service):
    svc, rid, _ = parent(service)
    reviewer = ReviewRunner(correction=True)
    review_factory(svc, reviewer)
    child = svc.create_review_run(rid, reviewer="reviewer")
    assert svc.execute(child)["status"] == "completed"
    previous = ""
    for job, row in zip(reviewer.jobs, svc.repository.chunks(child), strict=True):
        assert job.context.get("previous_russian", "") == previous[-20:]
        result = (svc.run_dir(child) / row["output_path"]).read_bytes().decode("utf-8")
        original = job.draft_segments[0]["text"]
        assert result == job.draft.replace(original, "Исправленный русский абзац.", 1)
        previous = job.draft
    n = len(reviewer.jobs)
    svc.resume(child)
    assert len(reviewer.jobs) == n


@pytest.mark.parametrize("defect", ["source", "draft", "mapping", "segments", "snapshot"])
def test_corrupt_inputs_refused_before_inference(service, defect):
    svc, rid, _ = parent(service)
    reviewer = ReviewRunner()
    calls = review_factory(svc, reviewer)
    if defect == "segments":
        row = svc.repository.chunks(rid)[0]
        meta = json.loads(row["metadata"])
        meta["segments"] = [
            {"id": s["id"], "text": "Иной текст."} for s in source_segments("第一章")
        ]
        svc.repository.set_chunk(
            rid,
            row["chapter"],
            row["chunk_id"],
            "completed",
            meta,
            row["output_path"],
            row["output_sha256"],
        )
        with pytest.raises(ValueError):
            svc.create_review_run(rid, reviewer="reviewer")
    else:
        child = svc.create_review_run(rid, reviewer="reviewer")
        m = svc.repository.run(child)["manifest"]
        u = m["review_input"]["units"][0]
        if defect == "source":
            p = svc.config.novel.source_dir / m["chapters"][0]["filename"]
        elif defect == "draft":
            p = svc.run_dir(rid) / u["parent_output_path"]
        elif defect == "snapshot":
            p = svc.run_dir(child) / u["draft_path"]
        else:
            p = svc.run_dir(child) / u["draft_path"].replace("draft.md", "frozen-draft.json")
        p.write_bytes(b"tampered")
        with pytest.raises(ValueError):
            svc.resume(child)
    assert not calls and not reviewer.jobs


def test_partial_selected_chapter_refused_other_completed_selectable(service):
    svc, rid, _ = parent(service, second=True)
    row = next(r for r in svc.repository.chunks(rid) if r["chapter"] == 2)
    svc.repository.set_chunk(rid, 2, row["chunk_id"], "failed")
    reviewer = ReviewRunner()
    calls = review_factory(svc, reviewer)
    with pytest.raises(ValueError, match="incomplete/corrupt"):
        svc.create_review_run(rid, reviewer="reviewer")
    child = svc.create_review_run(rid, reviewer="reviewer", chapters=[1])
    assert not calls
    assert svc.execute(child)["status"] == "completed"


def test_changed_current_context_not_adopted_and_strict_validation_refused(service):
    svc, rid, _ = parent(service)
    svc.config.translation.glossary_path.write_text("names:\n  新: Новое\n", "utf-8")
    svc.config.translation.book_context_chars = 256
    reviewer = ReviewRunner()
    review_factory(svc, reviewer)
    child = svc.create_review_run(rid, reviewer="reviewer")
    parent_manifest = svc.repository.run(rid)["manifest"]
    assert svc.repository.run(child)["manifest"]["glossary"] == parent_manifest["glossary"]
    assert svc.execute(child)["status"] == "completed"
    # The frozen drafts expand Japanese to Russian by roughly 3.1-3.4x, so a
    # threshold above that must reject the parent draft under the new policy.
    svc.config.validation.min_length_ratio = 5
    with pytest.raises(ValueError, match="validation failed"):
        svc.create_review_run(rid, reviewer="reviewer", force=True)


def test_continuation_preserves_actual_translator_origin(service):
    svc, rid, _ = parent(service)
    svc.config.runners["new-translator"] = RunnerConfig(model="new-model")
    cont = svc.continue_run(rid, runner="new-translator", chunk_chars=32)
    reviewer = ReviewRunner()
    review_factory(svc, reviewer)
    child = svc.create_review_run(cont, reviewer="reviewer")
    origins = svc.repository.run(child)["manifest"]["review_input"]["units"]
    assert all(u["origin"]["run_id"] == rid and u["origin"]["runner"] == "test" for u in origins)
    assert svc.execute(child)["status"] == "completed"
    with pytest.raises(ValueError, match="Derived review"):
        svc.continue_run(child, runner="new-translator")


def test_no_translator_config_needed_and_worker_monitor_correct(service):
    svc, rid, _ = parent(service)
    reviewer = ReviewRunner()
    review_factory(svc, reviewer)
    child = svc.create_review_run(rid, reviewer="reviewer")
    del svc.config.runners["test"]
    assert run_worker(svc, child, sleep=lambda _: None)["status"] == "completed"
    data = snapshot(svc, child)
    assert [s["stage"] for s in data["runners"]] == ["review"]
    assert svc.status(child)["parent_run_id"] == rid


class UnresolvedReviewRunner:
    def __init__(self):
        self.jobs = []

    def run(self, job):
        self.jobs.append(copy.deepcopy(job))
        assert job.output_mode == "review" and job.draft_segments
        assert job.review_contract == "review-v2"
        draft = job.draft_segments[0]
        report = {
            "checked_ids": [s["id"] for s in job.segments],
            "issues": [],
            "unresolved": [
                {"id": draft["id"], "category": "negation", "reason": "Cannot fix reliably."}
            ],
        }
        return RunnerResult(json.dumps(report), finish_reason="stop", model="review-reported")


def test_derived_review_v2_accepts_open_defects_without_retry(service):
    from novel_harness.quality import readiness

    svc, rid, _ = parent(service)
    reviewer = UnresolvedReviewRunner()
    review_factory(svc, reviewer)
    child = svc.create_review_run(rid, reviewer="reviewer", prompt_version="review-v2")
    assert svc.execute(child)["status"] == "completed"
    rows = svc.repository.chunks(child)
    # One accepting attempt per chunk: an unresolved defect is not a failure and
    # does not trigger the recovery loop.
    assert len(reviewer.jobs) == len(rows)
    assert all(row["status"] == "completed" for row in rows)
    meta = json.loads(rows[0]["metadata"])
    assert meta["review"]["unresolved"]
    assert meta["review_status"] == "open_defects"
    assert meta["review_open_defects"] == 1
    assert meta["selection_reason"] == "accepted_with_open_defects"
    report = readiness(svc.run_dir(child), rows, complete=True)
    assert report["open_review_defects"] == len(rows)
    assert report["status"] == "needs_review"


def draft_light_parent(service):
    from novel_harness.config import PipelineConfig, StageConfig

    svc, _ = service
    svc.config.pipelines["draft-light"] = PipelineConfig(
        stages=[
            StageConfig(
                name="translation", role="translator", prompt_version="translate-draft-v1"
            )
        ]
    )
    svc.config.translation.output_format = "segments-v1"
    translator = ParentTranslator("segments-v1")
    svc.runner_factory = lambda name: translator
    rid = svc.create_run(runner="test", pipeline="draft-light")
    assert svc.execute(rid)["status"] == "completed"
    svc.config.runners["reviewer"] = RunnerConfig(model="review-model")
    return svc, rid, translator


def test_draft_light_parent_allows_derived_review(service):
    svc, rid, translator = draft_light_parent(service)
    reviewer = UnresolvedReviewRunner()
    constructed = review_factory(svc, reviewer)
    child = svc.create_review_run(rid, reviewer="reviewer", prompt_version="review-v2")
    assert not constructed and not reviewer.jobs
    manifest = svc.repository.run(child)["manifest"]
    assert manifest["review_input"]["parent_workflow"] == "draft-light"
    assert manifest["review_input"]["candidate_layer"] == "draft-light-translator"
    stage = manifest["stages"][0]
    assert stage["role"] == "reviewer"
    assert stage["review_contract"] == "review-v2"
    calls = len(translator.jobs)
    assert svc.execute(child)["status"] == "completed"
    assert len(translator.jobs) == calls
    assert len(reviewer.jobs) == len(svc.repository.chunks(rid))


class LeakyDraftRunner:
    def run(self, job):
        originals = source_segments(job.source, heading=job.source.startswith("第一章"))
        segments = [
            {"id": s["id"], "text": f"Длинный русский абзац номер {i} с одной утечкой少."}
            for i, s in enumerate(originals)
        ]
        return RunnerResult(json.dumps({"segments": segments}), finish_reason="stop")


def test_draft_light_parent_admits_known_corruption_but_output_stays_strict(service):
    from novel_harness.config import PipelineConfig, StageConfig

    svc, _ = service
    svc.config.pipelines["draft-light"] = PipelineConfig(
        stages=[
            StageConfig(
                name="translation", role="translator", prompt_version="translate-draft-v1"
            )
        ]
    )
    svc.config.translation.output_format = "segments-v1"
    svc.runner_factory = lambda name: LeakyDraftRunner()
    rid = svc.create_run(runner="test", pipeline="draft-light")
    assert svc.execute(rid)["status"] == "completed"
    svc.config.runners["reviewer"] = RunnerConfig(model="review-model")
    reviewer = ReviewRunner()
    review_factory(svc, reviewer)
    # The frozen leaky draft is admitted as review input...
    child = svc.create_review_run(rid, reviewer="reviewer", prompt_version="review-v2")
    policy = svc.repository.run(child)["manifest"]["validation"]
    # ...but the reviewed output is still re-checked with the strict policy.
    assert policy["max_foreign_script_chars"] == 0
    assert policy["reject_malformed_dialogue"] is True
    assert not reviewer.jobs


def test_parent_busy_and_cli_prepare_without_inference(service, monkeypatch):
    svc, rid, _ = parent(service)
    with svc._lock(rid):
        other = TranslationService(svc.config)
        with pytest.raises(Timeout):
            other.create_review_run(rid, reviewer="reviewer")
    reviewer = ReviewRunner()
    calls = review_factory(svc, reviewer)
    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: svc))
    result = CliRunner().invoke(
        app, ["review", "--run", rid, "--reviewer", "reviewer", "--chapters", "1", "--json"]
    )
    assert result.exit_code == 0, result.output
    child = json.loads(result.output)
    assert child["parent_run_id"] == rid and child["chunks"]["completed"] == 0
    assert not calls
