import copy
import json
from pathlib import Path

import pytest
import yaml

from novel_harness.evaluation import (
    ERRORS,
    build_blind_human_package,
    export_evaluation,
    prepare_plan,
    read_candidate,
    score_evaluation,
    select_chapters,
)
from novel_harness.runners.base import RunnerResult


def test_v30_seed_and_held_out_selection():
    sample = select_chapters(set(range(1, 226)), 5, 20260908, [])
    assert sample == [33, 49, 63, 91, 217]
    held_out = select_chapters(set(range(1, 226)), 20, 20260909, sample)
    assert not set(sample) & set(held_out)


def test_v30_prepare_calls_no_models_and_keeps_pointers(service, tmp_path):
    svc, fake = service
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "count": 1,
                "variants": [
                    {"name": "small", "runner": "test", "chunk_chars": 40},
                    {"name": "large", "runner": "test", "chunk_chars": 80},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = prepare_plan(svc, matrix)
    assert result["chapters"] == [1] and not fake.calls
    assert len({v["run_id"] for v in result["variants"]}) == 2
    assert not list(svc.root.glob("active*.json"))
    assert prepare_plan(svc, matrix) == result


def test_v30_blind_export_and_human_scores(service):
    svc, _ = service
    runs = [svc.create_run(runner="test", force=True) for _ in range(2)]
    for rid in runs:
        svc.execute(rid)
    result = export_evaluation(svc, runs, count=1)
    directory = Path(result["directory"])
    html = Path(result["path"]).read_text("utf-8")
    assert all(rid not in html for rid in runs)
    with pytest.raises(ValueError):
        score_evaluation(directory, directory / "scorecard.json")
    data = json.loads((directory / "scorecard.json").read_text("utf-8"))
    data["cases"][0]["preferred"] = "A"
    for candidate in data["cases"][0]["candidates"]:
        candidate.update({k: 0 for k in ERRORS})
        candidate["manual_minutes"] = 2.5
    path = directory / "human-scores.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    scored = score_evaluation(directory, path)
    assert sum(r["preferred_chapters"] for r in scored["runs"].values()) == 1
    assert all(r["manual_minutes"] == 2.5 for r in scored["runs"].values())
    for invalid in (None, -1, True, 1.5, float("nan")):
        bad = copy.deepcopy(data)
        bad["cases"][0]["candidates"][0]["meaning"] = invalid
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            score_evaluation(directory, path)
    bad = copy.deepcopy(data)
    bad["cases"][0]["candidates"][1] = bad["cases"][0]["candidates"][0]
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        score_evaluation(directory, path)
    path.write_text(json.dumps(data), encoding="utf-8")
    assert export_evaluation(svc, runs, count=1) == result
    assert path.is_file()


def test_v30_export_rejects_different_sources(service, source):
    svc, _ = service
    first = svc.create_run(runner="test")
    svc.execute(first)
    source.write_text(source.read_text("utf-8") + "追加の文章。", "utf-8")
    second = svc.create_run(runner="test")
    svc.execute(second)
    with pytest.raises(ValueError, match="snapshots differ"):
        export_evaluation(svc, [first, second], count=1)


def test_read_candidate_labels_translator_and_review_layers(service, tmp_path):
    from novel_harness.config import (
        PipelineConfig,
        RunnerConfig,
        StageConfig,
    )

    svc, _ = service
    # A segment-aware translator so a segments-v1 parent is valid, and a reviewer
    # that emits the review-v2 contract.
    class SegmentTranslator:
        def run(self, job):
            from novel_harness.translation.segments import source_segments

            originals = source_segments(job.source, heading=job.source.startswith("第一章"))
            segments = [
                {"id": s["id"], "text": f"Русский абзац номер {i}."}
                for i, s in enumerate(originals)
            ]
            return RunnerResult(json.dumps({"segments": segments}), finish_reason="stop")

    class EmptyReviewer:
        def run(self, job):
            report = {"checked_ids": [s["id"] for s in job.segments], "issues": [], "unresolved": []}
            return RunnerResult(json.dumps(report), finish_reason="stop")

    svc.runner_factory = lambda name: SegmentTranslator()
    svc.config.pipelines["draft-light"] = PipelineConfig(
        stages=[
            StageConfig(
                name="translation", role="translator", prompt_version="translate-draft-v1"
            )
        ]
    )
    svc.config.translation.output_format = "segments-v1"
    draft = svc.create_run(runner="test", pipeline="draft-light")
    assert svc.execute(draft)["status"] == "completed"
    svc.config.runners["reviewer"] = RunnerConfig(model="review-model")
    svc.runner_factory = lambda name: EmptyReviewer()
    review = svc.create_review_run(draft, reviewer="reviewer", prompt_version="review-v2")
    assert svc.execute(review)["status"] == "completed"
    edit = svc.create_review_run(draft, reviewer="reviewer", prompt_version="semantic-edit-v1")
    assert svc.execute(edit)["status"] == "completed"

    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(svc.config.model_dump(mode="json")), encoding="utf-8")
    reviewed = read_candidate(cfg_path, review, 1)
    assert reviewed["candidate_layer"] == "reviewed"
    assert reviewed["workflow"] == "review-direct"
    edited = read_candidate(cfg_path, edit, 1)
    assert edited["candidate_layer"] == "semantic_edited"
    assert edited["prompt_version"] == "semantic-edit-v1"
    translator = read_candidate(cfg_path, draft, 1)
    assert translator["candidate_layer"] == "translator_raw"

    bundle = build_blind_human_package(
        [
            {"arm": "draft", "config": cfg_path, "run_id": draft},
            {"arm": "reviewed", "config": cfg_path, "run_id": review},
        ],
        chapters=[1],
        seed=7,
        directory=tmp_path / "blind",
    )
    key = json.loads((Path(bundle["directory"]) / "answer-key.json").read_text("utf-8"))
    assert key["candidate_layers"] == {"draft": "translator_raw", "reviewed": "reviewed"}
    assert key["candidate_layer"] == "mixed"
