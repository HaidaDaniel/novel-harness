import json

from novel_harness.cache import TranslationCache
from novel_harness.runners.base import RunnerResult


class _MeteredRunner:
    def __init__(self):
        self.calls = 0

    def run(self, job):
        self.calls += 1
        return RunnerResult(
            "Глава\n\nРусский текст. " * 5,
            model="fake",
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


def _enable_cache(svc):
    svc.stages.cache = TranslationCache(svc.root / "cache")


def test_v27_usage_retained_on_invalid_structured_output(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"

    class Invalid:
        def run(self, job):
            return RunnerResult(
                "not JSON",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "reasoning_tokens": 18,
                    "cached_input_tokens": 2,
                },
                effective_settings={"model_source": "configured"},
                timings={"process_seconds": 0.1},
            )

    svc.runner_factory = lambda _: Invalid()
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    metrics = svc.metrics(rid)["stages"][0]
    assert metrics["usage"]["reasoning_tokens"] == 18
    assert metrics["usage"]["cache_write_input_tokens"] is None
    assert metrics["usage_observed_attempts"]["cache_write_input_tokens"] == 0
    assert metrics["statuses"] == {"failed": 1}
    assert metrics["measured_attempts"] == 1
    row = svc.repository.chunks(rid)[0]
    assert json.loads(row["metadata"])["finish_reason"] == "stop"


def test_cache_only_run_does_not_charge_original_tokens(service):
    svc, _ = service
    _enable_cache(svc)
    runner = _MeteredRunner()
    svc.runner_factory = lambda _: runner

    first = svc.create_run(runner="test", chapters=[1])
    svc.execute(first)
    calls_after_first = runner.calls

    second = svc.create_run(runner="test", chapters=[1], force=True)
    svc.execute(second)
    chunks = len(svc.repository.chunks(second))
    assert runner.calls == calls_after_first  # served entirely from cache

    stage = svc.metrics(second)["stages"][0]
    assert stage["usage"]["prompt_tokens"] is None
    assert stage["usage"]["completion_tokens"] is None
    assert stage["reused_usage"]["prompt_tokens"] == 100 * chunks
    assert stage["reused_usage"]["completion_tokens"] == 50 * chunks
    assert stage["inference_attempts"] == 0
    assert stage["cache_reused_attempts"] == chunks


def test_mixed_cache_misses_and_hits_report_separate_totals(service):
    svc, _ = service
    _enable_cache(svc)
    runner = _MeteredRunner()
    svc.runner_factory = lambda _: runner

    first = svc.create_run(runner="test", chapters=[1])
    svc.execute(first)
    rows = svc.repository.chunks(first)
    keys = [json.loads(row["metadata"])["cache_key"] for row in rows]
    svc.stages.cache._path(keys[0]).unlink()
    calls_after_first = runner.calls

    second = svc.create_run(runner="test", chapters=[1], force=True)
    svc.execute(second)
    assert runner.calls - calls_after_first == 1  # only the evicted chunk misses

    stage = svc.metrics(second)["stages"][0]
    assert stage["usage"]["prompt_tokens"] == 100
    assert stage["reused_usage"]["prompt_tokens"] == 100 * (len(rows) - 1)
    assert stage["inference_attempts"] == 1
    assert stage["cache_reused_attempts"] == len(rows) - 1


def test_tm_hit_run_reports_unknown_current_usage(service):
    svc, _ = service
    _enable_cache(svc)
    runner = _MeteredRunner()
    svc.runner_factory = lambda _: runner

    first = svc.create_run(runner="test", chapters=[1])
    svc.execute(first)
    for row in svc.repository.chunks(first):
        svc.set_final(
            first, row["chapter"], row["chunk_id"], action="accept", actor="ui",
            expected_machine_sha256=row["output_sha256"],
        )
        svc.promote_final_to_tm(first, row["chapter"], row["chunk_id"], actor="ui")
    calls_after_first = runner.calls

    second = svc.create_run(runner="test", chapters=[1], force=True)
    svc.execute(second)
    chunks = len(svc.repository.chunks(second))
    assert runner.calls == calls_after_first

    stage = svc.metrics(second)["stages"][0]
    assert stage["usage"]["prompt_tokens"] is None
    assert stage["reused_usage"]["prompt_tokens"] is None
    assert stage["inference_attempts"] == 0
    assert stage["cache_reused_attempts"] == chunks


def test_v27_legacy_save_timing_not_reported_as_generation(service):
    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid, max_chunks=1)
    row = svc.repository.chunks(rid)[0]
    meta = json.loads(row["metadata"])
    attempt = svc.repository.attempts(rid, row["chapter"], row["chunk_id"], meta["stage"])[0]
    attempt.pop("measurement_version")
    attempt["duration_seconds"] = 0.016
    svc.repository.save_attempt(attempt)
    metrics = svc.metrics(rid)["stages"][0]
    assert metrics["seconds"] is None
    assert metrics["legacy_unverified_attempts"] == 1


def test_v27_readiness_checks_actual_epub_and_never_claims_human_review(service):
    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    assert not svc.status(rid)["quality"]["epub_matches_outputs"]
    path = svc.build_epub(rid)
    assert svc.status(rid)["quality"]["epub_matches_outputs"]
    assert svc.status(rid)["quality"]["human_review"] == "not_recorded"
    path.write_bytes(b"corrupt")
    assert not svc.status(rid)["quality"]["epub_matches_outputs"]
