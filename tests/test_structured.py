"""Recognize generated GBNF independently and exercise the existing strict validators."""

import json
import re
from dataclasses import replace
from functools import cache

import pytest

from novel_harness.artifacts import sha256
from novel_harness.config import RunnerConfig
from novel_harness.runners.base import TranslationJob
from novel_harness.runners.structured import llama_grammar, structured_params
from novel_harness.translation.review import (
    REVIEW_V2,
    SEMANTIC_EDIT_V1,
    apply_review,
    review_schema,
)
from novel_harness.translation.segments import (
    decode_translation,
    source_segments,
    translation_schema,
)


def recognizes(grammar, raw):
    # This grammar subset is regular (no recursive rules). Expand its terminals
    # into a Python recognizer, independent of the builder's field/array logic.
    rules = dict(line.split(" ::= ", 1) for line in grammar.splitlines())
    tokenizer = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]\\])*\]|[A-Za-z][A-Za-z0-9-]*|[()|*+?]')

    @cache
    def expand(name):
        pieces, end = [], 0
        for token in tokenizer.finditer(rules[name]):
            assert not rules[name][end : token.start()].strip()
            value = token[0]
            if value.startswith('"'):
                pieces.append(re.escape(json.loads(value)))
            elif value.startswith("[") or value in "()|*+?":
                pieces.append(value)
            else:
                pieces.append("(?:" + expand(value) + ")")
            end = token.end()
        assert not rules[name][end:].strip()
        return "".join(pieces)

    return re.fullmatch(expand("root"), raw) is not None


def jobs(tmp_path):
    source = "題名\n「彼女は誰？」\n「知らない」"
    segments = source_segments(source)
    translated = [
        {"id": s["id"], "text": t}
        for s, t in zip(segments, ["Заголовок", 'Он спросил: "Кто она?"', "«Не знаю»"], strict=True)
    ]
    translation = TranslationJob(
        "r",
        1,
        "c",
        "translation",
        "Translate.",
        source,
        {},
        {},
        tmp_path,
        segments=segments,
        response_schema=translation_schema(segments),
        output_mode="segments",
    )
    review = replace(
        translation,
        stage="review",
        output_mode="review",
        response_schema=review_schema(segments),
        draft_segments=[{**s, "sha256": sha256(s["text"])} for s in translated],
    )
    return translation, review, translated


def test_translation_grammar_accepts_dialogue_and_enforces_exact_coverage(tmp_path):
    job, _, draft = jobs(tmp_path)
    grammar = llama_grammar(job)
    raw = json.dumps({"segments": draft}, ensure_ascii=False)
    assert recognizes(grammar, raw)
    assert decode_translation(raw, job.segments) == draft
    for damaged in [draft[:-1], draft[::-1], [draft[0], draft[0], draft[2]], draft + [draft[0]]]:
        assert not recognizes(grammar, json.dumps({"segments": damaged}, ensure_ascii=False))
    assert not recognizes(grammar, raw.replace('"text": "Заголовок"', '"text": «Заголовок»'))
    assert not recognizes(grammar, raw.replace('"text": "Заголовок"', '"text": "Он сказал: "Нет""'))


@pytest.mark.parametrize("text", ["", "Первая\nвторая", "Первая\rвторая"])
def test_translation_grammar_rejects_empty_or_multiline_strings(tmp_path, text):
    job, _, draft = jobs(tmp_path)
    draft[0]["text"] = text
    assert not recognizes(llama_grammar(job), json.dumps({"segments": draft}))


@pytest.mark.parametrize("escape", [r"\n", r"\r", r"\u000a", r"\u000A", r"\u000d", r"\u000D"])
def test_line_grammar_rejects_all_encoded_newlines(tmp_path, escape):
    job, _, draft = jobs(tmp_path)
    raw = json.dumps({"segments": draft}, ensure_ascii=False).replace(
        "Заголовок", "До" + escape + "после"
    )
    assert not recognizes(llama_grammar(job), raw)


def test_review_grammar_preserves_patch_meaning_hash_pair_and_coverage(tmp_path):
    _, job, draft = jobs(tmp_path)
    issue = {
        "id": draft[1]["id"],
        "category": "meaning",
        "reason": 'Wrong question: "who"',
        "original_sha256": sha256(draft[1]["text"]),
        "replacement": "«Кто она?»",
    }
    report = {"checked_ids": [s["id"] for s in job.segments], "issues": [issue]}
    grammar = llama_grammar(job)
    raw = json.dumps(report, ensure_ascii=False)
    assert recognizes(grammar, raw)
    _, patched = apply_review(raw, job.segments, draft)
    assert patched[1]["text"] == "«Кто она?»" and patched[0] == draft[0]
    report["checked_ids"][1] = report["checked_ids"][0]
    assert not recognizes(grammar, json.dumps(report))
    report["checked_ids"] = [s["id"] for s in job.segments]
    issue["original_sha256"] = sha256(draft[0]["text"])
    assert not recognizes(grammar, json.dumps(report))
    issue["original_sha256"] = sha256(draft[1]["text"])
    issue["replacement"] = None
    assert not recognizes(grammar, json.dumps(report))
    with pytest.raises(ValueError, match="review_unresolved"):
        apply_review(json.dumps(report), job.segments, draft)
    issue["replacement"] = "Первая\nвторая"
    assert not recognizes(grammar, json.dumps(report))


def test_structured_options_are_explicit_per_job_and_do_not_mutate_profile(tmp_path):
    translation, review, _ = jobs(tmp_path)
    profile = RunnerConfig(extra_params={"seed": 7}, structured_decoding="llama-grammar")
    first = structured_params(profile, translation)
    second = structured_params(profile, review)
    assert first["grammar"] != second["grammar"]
    assert first["seed"] == 7 and profile.extra_params == {"seed": 7}
    assert structured_params(profile, replace(translation, response_schema=None)) == {"seed": 7}
    assert structured_params(RunnerConfig(), translation) == {}
    schema = structured_params(RunnerConfig(structured_decoding="json-schema"), translation)
    assert schema["response_format"]["json_schema"]["schema"] == translation.response_schema


@pytest.mark.parametrize("field", ["grammar", "json_schema", "response_format"])
def test_conflicting_constraints_fail_before_dispatch(tmp_path, field):
    job, _, _ = jobs(tmp_path)
    profile = RunnerConfig(structured_decoding="llama-grammar", extra_params={field: "explicit"})
    with pytest.raises(ValueError, match="conflicts"):
        structured_params(profile, job)


def test_unsupported_grammar_job_fails_closed(tmp_path):
    translation, review, _ = jobs(tmp_path)
    with pytest.raises(ValueError, match="segments-v1"):
        llama_grammar(replace(translation, output_mode="judge"))
    with pytest.raises(ValueError, match="verified ordered draft"):
        llama_grammar(replace(review, draft_segments=review.draft_segments[::-1]))


def test_review_grammar_allows_every_unique_ordered_subset_and_forbids_repeats(tmp_path):
    from itertools import combinations

    _, job, draft = jobs(tmp_path)
    patches = [
        {
            "id": s["id"],
            "category": "meaning",
            "reason": "Correct this segment.",
            "original_sha256": sha256(s["text"]),
            "replacement": "Исправленный перевод.",
        }
        for s in draft
    ]
    grammar = llama_grammar(job)
    checked = [s["id"] for s in job.segments]
    for count in range(len(patches) + 1):
        for subset in combinations(patches, count):
            raw = json.dumps({"checked_ids": checked, "issues": list(subset)}, ensure_ascii=False)
            assert recognizes(grammar, raw)
            apply_review(raw, job.segments, draft)
            if len(subset) > 1:
                assert not recognizes(
                    grammar, json.dumps({"checked_ids": checked, "issues": list(subset[::-1])})
                )
    for patch in patches:
        for invalid in [[patch, patch], patches + [patch]]:
            assert not recognizes(grammar, json.dumps({"checked_ids": checked, "issues": invalid}))


def test_review_grammar_rejects_null_replacements_that_the_host_always_rejects(tmp_path):
    _, job, draft = jobs(tmp_path)
    grammar = llama_grammar(job)
    checked = [s["id"] for s in job.segments]

    def issue(segment, replacement):
        return {
            "id": segment["id"],
            "category": "meaning",
            "reason": "Review this segment.",
            "original_sha256": sha256(segment["text"]),
            "replacement": replacement,
        }

    concrete = issue(draft[0], "Исправлено.")
    assert not recognizes(grammar, json.dumps({"checked_ids": checked, "issues": [issue(draft[1], None)]}))
    assert not recognizes(
        grammar, json.dumps({"checked_ids": checked, "issues": [concrete, issue(draft[1], None)]})
    )
    assert recognizes(grammar, json.dumps({"checked_ids": checked, "issues": [concrete]}))
    assert recognizes(grammar, json.dumps({"checked_ids": checked, "issues": []}))
    with pytest.raises(ValueError, match="review_unresolved"):
        apply_review(
            json.dumps({"checked_ids": checked, "issues": [issue(draft[1], None)]}),
            job.segments,
            draft,
        )


def _v2_review(review):
    return replace(
        review,
        review_contract=REVIEW_V2,
        response_schema=review_schema(review.segments, REVIEW_V2),
    )


def _v2_entries(job, draft):
    checked = [s["id"] for s in job.segments]
    fix = {
        "id": draft[0]["id"],
        "category": "meaning",
        "reason": "Fix segment.",
        "original_sha256": sha256(draft[0]["text"]),
        "replacement": "Исправлено.",
    }
    open_defect = {
        "id": draft[1]["id"],
        "category": "negation",
        "reason": "Cannot resolve reliably.",
    }
    return checked, fix, open_defect


def test_review_v2_grammar_allows_fixable_and_unresolved_without_null(tmp_path):
    _, legacy, draft = jobs(tmp_path)
    job = _v2_review(legacy)
    grammar = llama_grammar(job)
    checked, fix, open_defect = _v2_entries(job, draft)
    raw = json.dumps(
        {"checked_ids": checked, "issues": [fix], "unresolved": [open_defect]},
        ensure_ascii=False,
    )
    assert recognizes(grammar, raw)
    _, patched = apply_review(raw, job.segments, draft, REVIEW_V2)
    assert patched[0]["text"] == "Исправлено." and patched[1] == draft[1]
    # null is not a valid replacement in the unified contract, and the grammar
    # cannot emit it anywhere.
    assert not recognizes(
        grammar,
        json.dumps(
            {
                "checked_ids": checked,
                "issues": [{**fix, "replacement": None}],
                "unresolved": [],
            }
        ),
    )
    # unresolved entries carry no replacement field.
    assert not recognizes(
        grammar,
        json.dumps(
            {
                "checked_ids": checked,
                "issues": [],
                "unresolved": [{**open_defect, "replacement": "x"}],
            }
        ),
    )
    # empty arrays and full ordered subsets are valid.
    assert recognizes(grammar, json.dumps({"checked_ids": checked, "issues": [], "unresolved": []}))
    assert not recognizes(
        grammar,
        json.dumps(
            {"checked_ids": checked, "issues": [fix], "unresolved": [open_defect, open_defect]}
        ),
    )


def test_review_v2_cross_array_duplicate_is_rejected_by_host(tmp_path):
    _, legacy, draft = jobs(tmp_path)
    job = _v2_review(legacy)
    checked, fix, _ = _v2_entries(job, draft)
    duplicated = {"id": draft[0]["id"], "category": "negation", "reason": "Also open."}
    raw = json.dumps({"checked_ids": checked, "issues": [fix], "unresolved": [duplicated]})
    # The linear grammar cannot express disjointness across two ordered arrays;
    # the authoritative host validator rejects the overlap.
    with pytest.raises(ValueError, match="invalid_or_duplicate_patch_id"):
        apply_review(raw, job.segments, draft, REVIEW_V2)


def test_sparse_review_subsets_support_many_ids_with_linear_grammar(tmp_path):
    _, example, _ = jobs(tmp_path)

    def large_job(count):
        segments = source_segments("\n".join(f"原文{i}。" for i in range(count)))
        return replace(
            example,
            segments=segments,
            draft_segments=[
                {"id": s["id"], "text": "Черновик.", "sha256": sha256("Черновик.")}
                for s in segments
            ],
        )

    small, large = large_job(25), large_job(51)
    grammar = llama_grammar(large)
    assert len(grammar) < 2.2 * len(llama_grammar(small))
    chosen = [large.draft_segments[i] for i in [0, 25, 50]]
    issues = [
        {
            "id": s["id"],
            "category": "meaning",
            "reason": "Fix.",
            "original_sha256": s["sha256"],
            "replacement": "Исправлено.",
        }
        for s in chosen
    ]
    checked = [s["id"] for s in large.segments]
    raw = json.dumps({"checked_ids": checked, "issues": issues})
    assert recognizes(grammar, raw)
    issues.append(issues[0])
    assert not recognizes(grammar, json.dumps({"checked_ids": checked, "issues": issues}))


def test_semantic_edit_v1_grammar_uses_semantic_categories_and_unified_shape(tmp_path):
    _, review, draft = jobs(tmp_path)
    edit = replace(
        review,
        review_contract=SEMANTIC_EDIT_V1,
        response_schema=review_schema(review.segments, SEMANTIC_EDIT_V1),
    )
    grammar = llama_grammar(edit)
    checked = [s["id"] for s in edit.segments]
    fix = {
        "id": draft[0]["id"],
        "category": "actor_patient",
        "reason": "Wrong actor.",
        "original_sha256": sha256(draft[0]["text"]),
        "replacement": "Подвези меня.",
    }
    open_defect = {
        "id": draft[1]["id"],
        "category": "negation",
        "reason": "Cannot resolve reliably.",
    }
    raw = json.dumps(
        {"checked_ids": checked, "issues": [fix], "unresolved": [open_defect]},
        ensure_ascii=False,
    )
    assert recognizes(grammar, raw)
    _, patched = apply_review(raw, edit.segments, draft, SEMANTIC_EDIT_V1)
    assert patched[0]["text"] == "Подвези меня." and patched[1] == draft[1]
    # Style is not a semantic-edit category and the grammar cannot emit it.
    style_only = {**fix, "category": "style"}
    assert not recognizes(
        grammar,
        json.dumps({"checked_ids": checked, "issues": [style_only], "unresolved": []}),
    )
