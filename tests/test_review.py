import json

import pytest

from novel_harness.artifacts import sha256
from novel_harness.runners.base import RunnerResult
from novel_harness.translation.review import (
    REVIEW_V1,
    REVIEW_V2,
    SEMANTIC_EDIT_V1,
    apply_review,
    review_categories,
    review_contract,
    review_schema,
)
from novel_harness.translation.segments import OutputContractError, source_segments


def example():
    segments = source_segments("背中に乗せてくれ。\n騒がしい。")
    draft = [
        {"id": s["id"], "text": text}
        for s, text in zip(segments, ["Садись на меня.", "Здесь шумно."], strict=True)
    ]
    issue = {
        "id": draft[0]["id"],
        "category": "meaning",
        "reason": "Reversed actors",
        "original_sha256": sha256(draft[0]["text"]),
        "replacement": "Подвези меня.",
    }
    return segments, draft, {"checked_ids": [s["id"] for s in segments], "issues": [issue]}


def test_v25_minimal_patch_preserves_unaffected_text():
    segments, draft, report = example()
    saved, result = apply_review(json.dumps(report), segments, draft)
    assert result[0]["text"] == "Подвези меня."
    assert result[1] == draft[1] and saved == report
    assert draft[0]["text"] == "Садись на меня."


def _single_patch(replacement, original="Достаточно длинный русский абзац для проверки замены."):
    segments = source_segments("原文。")
    draft = [{"id": segments[0]["id"], "text": original}]
    report = {
        "checked_ids": [segments[0]["id"]],
        "issues": [
            {
                "id": segments[0]["id"],
                "category": "meaning",
                "reason": "correct the segment",
                "original_sha256": sha256(original),
                "replacement": replacement,
            }
        ],
    }
    return segments, draft, report


@pytest.mark.parametrize(
    ("replacement", "original", "code"),
    [
        ("p0001-00000000", None, "review_patch_echoes_id"),
        ("null", None, "review_patch_sentinel"),
        ("(Acceptable)", None, "review_patch_meta"),
        ("(No change needed, but checking the draft)", None, "review_patch_meta"),
        ("икуいく", None, "review_patch_foreign_script"),
        ("或许", None, "review_patch_foreign_script"),
        ("looks fine now", None, "review_patch_not_russian"),
        ("Короткий ответ.", "Д" * 200, "review_patch_truncated"),
    ],
)
def test_v26_non_translation_patch_fails_closed(replacement, original, code):
    segments, draft, report = _single_patch(
        replacement, **({"original": original} if original is not None else {})
    )
    before = draft[0]["text"]
    with pytest.raises(OutputContractError, match=code):
        apply_review(json.dumps(report), segments, draft)
    assert draft[0]["text"] == before


def test_v26_valid_patch_and_short_segment_condensation_still_apply():
    segments, draft, report = _single_patch("Исправленный русский абзац.")
    _, result = apply_review(json.dumps(report), segments, draft)
    assert result[0]["text"] == "Исправленный русский абзац."

    segments, draft, report = _single_patch(
        "Да-да, понял.", original="«Да ладно тебе, я ведь сказал — не трогаться с места же!»"
    )
    _, result = apply_review(json.dumps(report), segments, draft)
    assert result[0]["text"] == "Да-да, понял."



@pytest.mark.parametrize("defect", ["hash", "duplicate", "coverage", "unresolved", "empty"])
def test_v25_bad_review_never_applied(defect):
    segments, draft, report = example()
    if defect == "hash":
        report["issues"][0]["original_sha256"] = "stale"
    if defect == "duplicate":
        report["issues"].append(report["issues"][0])
    if defect == "coverage":
        report["checked_ids"].pop()
    if defect == "unresolved":
        report["issues"][0]["replacement"] = None
    if defect == "empty":
        report["issues"][0]["replacement"] = ""
    with pytest.raises(OutputContractError):
        apply_review(json.dumps(report), segments, draft)
    assert draft[0]["text"] == "Садись на меня."


def test_v25_service_review_and_resume(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    calls = []

    class ReviewingRunner:
        def run(self, job):
            calls.append(job)
            if job.output_mode == "review":
                assert job.draft_segments
                value = {"checked_ids": [s["id"] for s in job.segments], "issues": []}
            else:
                value = {
                    "segments": [
                        {"id": s["id"], "text": f"Русский абзац номер {i}."}
                        for i, s in enumerate(job.segments)
                    ]
                }
            return RunnerResult(json.dumps(value), finish_reason="stop")

    svc.runner_factory = lambda _: ReviewingRunner()
    rid = svc.create_run(translator="test", editor="test", pipeline="translate-review")
    assert svc.execute(rid)["status"] == "completed"
    row = svc.repository.chunks(rid)[0]
    meta = json.loads(row["metadata"])
    assert meta["review"]["issues"] == []
    assert meta["stages"][0]["output_sha256"] == meta["output_sha256"]
    count = len(calls)
    svc.resume(rid)
    assert len(calls) == count
    assert svc.build_epub(rid).is_file()


def test_v25_reviewer_receives_source_ids_and_draft_hashes(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    captured = {}

    class Capturing:
        def run(self, job):
            if job.output_mode == "review":
                captured["segments"] = job.segments
                captured["draft_segments"] = job.draft_segments
                value = {"checked_ids": [s["id"] for s in job.segments], "issues": []}
            else:
                value = {
                    "segments": [
                        {"id": s["id"], "text": f"Русский абзац {i}."}
                        for i, s in enumerate(job.segments)
                    ]
                }
            return RunnerResult(json.dumps(value), finish_reason="stop")

    svc.runner_factory = lambda _: Capturing()
    rid = svc.create_run(translator="test", editor="test", pipeline="translate-review")
    assert svc.execute(rid)["status"] == "completed"
    assert all(s["id"] and s["source"] for s in captured["segments"])
    assert any(ord(c) > 0x3000 for s in captured["segments"] for c in s["source"])
    assert all(s["id"] and "sha256" in s for s in captured["draft_segments"])
    assert all(s["sha256"] == sha256(s["text"]) for s in captured["draft_segments"])


def test_v25_reviewed_text_goes_through_strict_validation(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    svc.config.validation.max_foreign_script_chars = 0
    svc.config.validation.reject_malformed_dialogue = True

    class LeakyReview:
        def run(self, job):
            if job.output_mode == "review":
                seg = job.segments[0]
                draft = next(s for s in job.draft_segments if s["id"] == seg["id"])
                value = {
                    "checked_ids": [s["id"] for s in job.segments],
                    "issues": [
                        {
                            "id": seg["id"],
                            "category": "meaning",
                            "reason": "leak",
                            "original_sha256": draft["sha256"],
                            "replacement": "данジョンa",
                        }
                    ],
                }
            else:
                value = {
                    "segments": [
                        {"id": s["id"], "text": f"Чистый русский {i}."}
                        for i, s in enumerate(job.segments)
                    ]
                }
            return RunnerResult(json.dumps(value), finish_reason="stop")

    svc.runner_factory = lambda _: LeakyReview()
    rid = svc.create_run(translator="test", editor="test", pipeline="translate-review")
    assert svc.execute(rid)["status"] == "failed"
    assert all(row["status"] != "completed" for row in svc.repository.chunks(rid))


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"checked_ids": [], "issues": [], "extra": 1}),
        json.dumps(["checked_ids", "issues"]),
    ],
)
def test_v25_malformed_review_envelope_rejected(raw):
    segments, draft, _ = example()
    with pytest.raises(OutputContractError):
        apply_review(raw, segments, draft)
    assert draft[0]["text"] == "Садись на меня."


def _v2_report(segments, draft, *, issues=(), unresolved=()):
    return {
        "checked_ids": [s["id"] for s in segments],
        "issues": list(issues),
        "unresolved": list(unresolved),
    }


def _v2_issue(draft, index, replacement):
    return {
        "id": draft[index]["id"],
        "category": "meaning",
        "reason": "Correct this segment.",
        "original_sha256": sha256(draft[index]["text"]),
        "replacement": replacement,
    }


def _v2_unresolved(draft, index):
    return {
        "id": draft[index]["id"],
        "category": "negation",
        "reason": "Detected but not reliably fixable from source.",
    }


def test_v2_fixable_applied_and_unresolved_reported_unchanged():
    segments, draft, _ = example()
    report = _v2_report(
        segments,
        draft,
        issues=[_v2_issue(draft, 0, "Подвези меня.")],
        unresolved=[_v2_unresolved(draft, 1)],
    )
    saved, patched = apply_review(json.dumps(report), segments, draft, REVIEW_V2)
    assert saved == report
    assert patched[0]["text"] == "Подвези меня."
    assert patched[1] == draft[1]
    assert draft[0]["text"] == "Садись на меня."


def test_v2_schema_and_legacy_schema_differ_on_null_and_unresolved():
    segments, _, _ = example()
    legacy = review_schema(segments, REVIEW_V1)
    unified = review_schema(segments, REVIEW_V2)
    assert legacy["properties"]["issues"]["items"]["properties"]["replacement"] == {
        "type": ["string", "null"]
    }
    assert "unresolved" not in legacy["properties"]
    assert unified["properties"]["issues"]["items"]["properties"]["replacement"] == {
        "type": "string"
    }
    assert unified["required"] == ["checked_ids", "issues", "unresolved"]
    assert unified["properties"]["unresolved"]["items"]["required"] == [
        "id",
        "category",
        "reason",
    ]


def test_explicit_unknown_review_contract_fails_closed():
    with pytest.raises(ValueError, match="Unknown review contract"):
        review_contract("review-v2", "review-v3")


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda r: r["issues"][0].__setitem__("replacement", None), "review_issue_requires_replacement"),
        (lambda r: r.pop("unresolved"), "invalid_review_envelope"),
        (lambda r: r.__setitem__("unresolved", [r["issues"][0]]), "invalid_review_unresolved"),
        (lambda r: r.__setitem__("unresolved", [r["unresolved"][0], r["unresolved"][0]]),
         "invalid_or_duplicate_patch_id"),
        (lambda r: r["unresolved"][0].__setitem__("replacement", "x"), "invalid_review_unresolved"),
    ],
)
def test_v2_contract_violations_fail_closed(mutate, code):
    segments, draft, _ = example()
    report = _v2_report(
        segments,
        draft,
        issues=[_v2_issue(draft, 0, "Подвези меня.")],
        unresolved=[_v2_unresolved(draft, 1)],
    )
    mutate(report)
    with pytest.raises(OutputContractError, match=code):
        apply_review(json.dumps(report), segments, draft, REVIEW_V2)
    assert draft[0]["text"] == "Садись на меня."


def test_v2_same_segment_cannot_be_fixed_and_unresolved():
    segments, draft, _ = example()
    report = _v2_report(
        segments,
        draft,
        issues=[_v2_issue(draft, 0, "Подвези меня.")],
        unresolved=[_v2_unresolved(draft, 0)],
    )
    with pytest.raises(OutputContractError, match="invalid_or_duplicate_patch_id"):
        apply_review(json.dumps(report), segments, draft, REVIEW_V2)


def test_semantic_edit_contract_resolution_and_categories():
    assert review_contract(SEMANTIC_EDIT_V1) == SEMANTIC_EDIT_V1
    assert review_categories(SEMANTIC_EDIT_V1) == [
        "meaning",
        "omission",
        "negation",
        "actor_patient",
        "speaker",
        "gender",
        "terminology",
        "consistency",
    ]
    # Style is not a semantic-edit category, and the unified shape is used.
    assert "style" not in review_categories(SEMANTIC_EDIT_V1)
    segments, _, _ = example()
    schema = review_schema(segments, SEMANTIC_EDIT_V1)
    assert schema["required"] == ["checked_ids", "issues", "unresolved"]


def test_semantic_edit_applies_fix_and_reports_unresolved_unchanged():
    segments, draft, _ = example()
    report = _v2_report(
        segments,
        draft,
        issues=[_v2_issue(draft, 0, "Подвези меня.")],
        unresolved=[_v2_unresolved(draft, 1)],
    )
    saved, patched = apply_review(json.dumps(report), segments, draft, SEMANTIC_EDIT_V1)
    assert saved == report
    assert patched[0]["text"] == "Подвези меня."
    assert patched[1] == draft[1]
    assert draft[0]["text"] == "Садись на меня."


def test_semantic_edit_rejects_style_only_category_and_null():
    segments, draft, _ = example()
    style = _v2_issue(draft, 0, "Более гладкий русский.")
    style["category"] = "style"
    report = _v2_report(segments, draft, issues=[style])
    with pytest.raises(OutputContractError, match="review_reason_required"):
        apply_review(json.dumps(report), segments, draft, SEMANTIC_EDIT_V1)

    null_issue = _v2_issue(draft, 0, None)
    report = _v2_report(segments, draft, issues=[null_issue])
    with pytest.raises(OutputContractError, match="review_issue_requires_replacement"):
        apply_review(json.dumps(report), segments, draft, SEMANTIC_EDIT_V1)


def test_v1_legacy_still_rejects_null_and_ignores_unresolved():
    segments, draft, _ = example()
    legacy = _v2_report(
        segments, draft, issues=[_v2_issue(draft, 0, "Подвези меня.")], unresolved=[]
    )
    with pytest.raises(OutputContractError, match="invalid_review_envelope"):
        apply_review(json.dumps(legacy), segments, draft, REVIEW_V1)
    null_issue = _v2_issue(draft, 0, None)
    report = {"checked_ids": [s["id"] for s in segments], "issues": [null_issue]}
    with pytest.raises(OutputContractError, match="review_unresolved"):
        apply_review(json.dumps(report), segments, draft, REVIEW_V1)


def _executed_chunk(translation):
    run_id = translation.create_run(runner="test", chapters=[1])
    translation.execute(run_id)
    row = translation.repository.chunks(run_id)[0]
    return run_id, row


def test_promote_rejects_needs_review_and_unsettled_candidates(service):
    translation, _ = service
    run_id, row = _executed_chunk(translation)
    chapter, chunk_id, machine_sha = row["chapter"], row["chunk_id"], row["output_sha256"]

    translation.set_final(
        run_id, chapter, chunk_id, action="needs_review", actor="ui",
        expected_machine_sha256=machine_sha,
    )
    needs = translation.final_layer(run_id)[0]
    assert needs["status"] == "needs_review"
    with pytest.raises(ValueError, match="accepted"):
        translation.promote_final_to_tm(run_id, chapter, chunk_id, actor="ui")

    # An accepted but unsettled (requeued) chunk has no completed candidate.
    translation.set_final(
        run_id, chapter, chunk_id, action="accept", actor="ui",
        expected_machine_sha256=machine_sha, expected_final_sha256=needs["final_sha256"],
    )
    translation.requeue_chunk(run_id, chapter, chunk_id, expected_machine_sha256=machine_sha)
    with pytest.raises(ValueError, match="valid completed"):
        translation.promote_final_to_tm(run_id, chapter, chunk_id, actor="ui")

    assert translation.translation_memory()["count"] == 0


def test_promote_rejects_stale_final_after_machine_change(service):
    translation, _ = service
    run_id, row = _executed_chunk(translation)
    chapter, chunk_id, machine_sha = row["chapter"], row["chunk_id"], row["output_sha256"]
    translation.set_final(
        run_id, chapter, chunk_id, action="accept", actor="ui",
        expected_machine_sha256=machine_sha,
    )

    path = translation.run_dir(run_id) / row["output_path"]
    path.write_text("Изменённый машинный перевод", "utf-8")
    new_sha = sha256("Изменённый машинный перевод")
    translation.repository.set_chunk(
        run_id, chapter, chunk_id, "completed",
        json.loads(row["metadata"]), row["output_path"], new_sha,
    )

    assert translation.final_layer(run_id)[0]["stale"] is True
    with pytest.raises(ValueError, match="stale"):
        translation.promote_final_to_tm(run_id, chapter, chunk_id, actor="ui")
    assert translation.translation_memory()["count"] == 0


def test_promote_rejects_corrupt_machine_candidate(service):
    translation, _ = service
    run_id, row = _executed_chunk(translation)
    chapter, chunk_id, machine_sha = row["chapter"], row["chunk_id"], row["output_sha256"]
    translation.set_final(
        run_id, chapter, chunk_id, action="accept", actor="ui",
        expected_machine_sha256=machine_sha,
    )

    # The recorded hash still matches the final, but the machine file is corrupt.
    path = translation.run_dir(run_id) / row["output_path"]
    path.write_bytes(path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="valid completed"):
        translation.promote_final_to_tm(run_id, chapter, chunk_id, actor="ui")
    assert translation.translation_memory()["count"] == 0


def test_promote_accepts_current_final_with_exact_provenance(service):
    translation, _ = service
    run_id, row = _executed_chunk(translation)
    chapter, chunk_id, machine_sha = row["chapter"], row["chunk_id"], row["output_sha256"]
    translation.set_final(
        run_id, chapter, chunk_id, action="accept", actor="ui",
        expected_machine_sha256=machine_sha,
    )
    final = translation.final_layer(run_id)[0]

    result = translation.promote_final_to_tm(run_id, chapter, chunk_id, actor="ui")

    assert result["inserted"] is True
    assert result["provenance"]["final_sha256"] == final["final_sha256"]
    assert result["provenance"]["machine_sha256"] == machine_sha
    assert translation.translation_memory()["count"] == 1
