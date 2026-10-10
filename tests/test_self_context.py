"""Synthetic tests for the frozen source-only self-context contract."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from novel_harness.artifacts import canonical, sha256
from novel_harness.context_runs import (
    generate_artifact,
    load_artifact,
    validate_composition,
    write_artifact,
)
from novel_harness.runners.base import RunnerResult, TranslationJob
from novel_harness.translation import self_context as sc


def make_source(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "original"
    source.mkdir()
    (source / "0001.txt").write_text(
        "第一章\n\nユキは剣を取った。ユキは笑った。\n", encoding="utf-8"
    )
    (source / "0002.txt").write_text("第二章\n\nユキは走った。\n", encoding="utf-8")
    (source / "0003.txt").write_text("第三章\n\nユキは立った。\n", encoding="utf-8")
    chapters = {c.number: c for c in sc.discover(source)}
    return source, chapters


def source_anchor(chapter, quote: str, occurrence: int = 0) -> dict:
    start = chapter.text.index(quote, occurrence)
    return {
        "chapter": chapter.number,
        "source_sha256": chapter.source_sha256,
        "start": start,
        "end": start + len(quote),
        "quote": quote,
        "quote_sha256": sha256(quote),
    }


def character_entry(chapter, quote: str, claim: str = "ユキ appears") -> dict:
    return {
        "kind": "character",
        "semantic_class": "source_anchored_claim",
        "claim": claim,
        "source_forms": ["ユキ"],
        "source_names": ["ユキ"],
        "anchors": [source_anchor(chapter, quote)],
    }


def generator() -> dict:
    return {
        "provider": "openai",
        "configured_model": "fake-model",
        "resolved_model_id": "fake-model",
        "weights_sha256": None,
        "quantization": None,
        "harness": "openai",
        "profile": "context-lite-9b-v1",
    }


def build(source: Path, chapter, *, entries, target=None, horizon="chapter_preflight", budget=None):
    target = chapter.number if target is None else target
    raw = {
        "snapshots": [
            {
                "target_chapter": target,
                "inputs": [
                    {
                        "kind": "japanese_source",
                        "chapter": chapter.number,
                        "filename": chapter.filename,
                        "source_sha256": chapter.source_sha256,
                        "normalized_text_sha256": sha256(chapter.text),
                        "scope": "full_chapter",
                    }
                ],
                "entries": entries,
                "generation_calls": [],
                "accepted_call_id": None,
            }
        ]
    }
    return sc.build_artifact(
        raw,
        source_dir=source,
        horizon_mode=horizon,
        generator=generator(),
        prompt={"version": "context-v1", "sha256": sha256("prompt")},
        decoding={},
        tool_version="test",
        budget=budget or sc.default_budget(),
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_valid_artifact_freezes_reloads_and_ids_are_reproducible(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    artifact = build(source, chapter, entries=[character_entry(chapter, "ユキは剣を取った。")])
    entry_id = artifact["snapshots"][0]["entries"][0]["id"]
    assert entry_id.startswith("ctx-")
    reloaded = json.loads(canonical(artifact))
    assert reloaded["snapshots"][0]["entries"][0]["id"] == entry_id
    path = tmp_path / "self-context.json"
    write_artifact(path, artifact)
    sc.verify_self_context(load_artifact(path), source_dir=source)
    assert sc.select_relevant(
        artifact,
        target_chapter=1,
        chunk_text=chapter.text,
        chunk_start=0,
        chunk_end=len(chapter.text),
    )


def test_hallucinated_quote_fails(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    entry = character_entry(chapter, "ユキは剣を取った。")
    entry["anchors"][0]["quote"] = "存在しない文"
    entry["anchors"][0]["quote_sha256"] = sha256("存在しない文")
    with pytest.raises(ValueError, match="anchor|quote"):
        build(source, chapter, entries=[entry])


def test_correct_quote_with_wrong_claim_never_becomes_confirmed(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    artifact = build(
        source, chapter, entries=[character_entry(chapter, "ユキは剣を取った。", claim="ユキ is male")]
    )
    entry = artifact["snapshots"][0]["entries"][0]
    assert entry["semantic_class"] == "source_anchored_claim"
    assert entry["semantic_class"] != "confirmed"
    assert "confirmed" not in canonical(artifact)


def test_other_resolved_model_identity_rejected(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    artifact = build(source, chapter, entries=[character_entry(chapter, "ユキは剣を取った。")])
    other = sc.compute_identity_sha256(
        provider="opencode",
        resolved_model_id="/abs/other.gguf",
        weights_sha256=None,
        quantization=None,
        harness="opencode",
    )
    with pytest.raises(ValueError, match="identity"):
        sc.verify_self_context(
            artifact, source_dir=source, candidate_identity_sha256=other
        )


def test_translated_russian_input_is_not_representable(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    raw = {
        "snapshots": [
            {
                "target_chapter": 1,
                "inputs": [
                    {
                        "kind": "previous_translation",
                        "chapter": 1,
                        "path": "chapters/0001/translation.md",
                        "sha256": sha256("russian"),
                    }
                ],
                "entries": [character_entry(chapter, "ユキは剣を取った。")],
            }
        ]
    }
    with pytest.raises(ValueError):
        sc.build_artifact(
            raw,
            source_dir=source,
            horizon_mode="chapter_preflight",
            generator=generator(),
            prompt={"version": "context-v1", "sha256": sha256("p")},
            decoding={},
            tool_version="test",
            budget=sc.default_budget(),
            generated_at="2026-01-01T00:00:00+00:00",
        )


def test_out_of_horizon_evidence_fails(tmp_path):
    source, chapters = make_source(tmp_path)
    first, second = chapters[1], chapters[2]
    entry = character_entry(second, "ユキは走った。")
    with pytest.raises(ValueError, match="horizon"):
        build(source, first, entries=[entry])


def test_history_only_forbids_current_chapter_input(tmp_path):
    source, chapters = make_source(tmp_path)
    second = chapters[2]
    with pytest.raises(ValueError, match="horizon"):
        build(
            source,
            second,
            entries=[character_entry(second, "ユキは走った。")],
            horizon="history_only",
        )


def test_pure_self_composition_rejects_human_context():
    validate_composition(
        composition="self_only",
        self_context={"context_origin": "self"},
        canonical_human=None,
    )
    validate_composition(
        composition="self_plus_shared_baseline",
        self_context={"context_origin": "self"},
        canonical_human={"context_origin": "canonical-human", "glossary_sha256": "a" * 64},
    )
    with pytest.raises(ValueError):
        validate_composition(
            composition="self_only",
            self_context={"context_origin": "self"},
            canonical_human={"glossary_sha256": "a" * 64},
        )
    with pytest.raises(ValueError):
        validate_composition(composition="none", self_context=None, canonical_human={"x": 1})


def test_budget_overflow_fails_closed(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    entry = character_entry(chapter, "ユキは剣を取った。")
    entry["claim"] = "あ" * 400
    artifact = build(source, chapter, entries=[entry])
    with pytest.raises(ValueError, match="max_context_chars"):
        sc.select_relevant(
            artifact,
            target_chapter=1,
            chunk_text=chapter.text,
            chunk_start=0,
            chunk_end=len(chapter.text),
            max_chars=10,
        )


def _request_quote_id(job: TranslationJob, source_ref="current_chapter") -> str:
    for document in job.context.get("source_documents", []):
        if document["source_ref"] == source_ref and document["quote_units"]:
            return document["quote_units"][0]["quote_id"]
    raise AssertionError(f"no quote units for {source_ref}")


class ContextFake:
    def __init__(self, chapter, quote, source_form=None):
        self.chapter = chapter
        self.quote = quote
        self.source_form = source_form or quote[:2]
        self.calls = []

    def __call__(self, job: TranslationJob) -> RunnerResult:
        self.calls.append(job)
        name = self.source_form
        entry = {
            "local_key": "char_1",
            "kind": "character",
            "semantic": {"class": "source_anchored_claim"},
            "claim": f"{name} appears",
            "source_forms": [name],
            "source_names": [name],
            "anchor_refs": [_request_quote_id(job)],
        }
        return RunnerResult(
            json.dumps({"entries": [entry]}),
            model="fake-model",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            effective_settings={"reported_model": "fake-model"},
        )


def _auto_context_call(job: TranslationJob) -> RunnerResult:
    """Cite the first host-owned current-chapter quote unit of this request."""
    quote_id = _request_quote_id(job)
    entry = {
        "local_key": "char_1",
        "kind": "character",
        "semantic": {"class": "source_anchored_claim"},
        "claim": "auto appears",
        "source_forms": ["ユキ"],
        "source_names": ["ユキ"],
        "anchor_refs": [quote_id],
    }
    return RunnerResult(
        json.dumps({"entries": [entry]}),
        model="fake-model",
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        effective_settings={"reported_model": "fake-model"},
    )


def test_prior_self_context_is_recursively_verified(tmp_path):
    source, chapters = make_source(tmp_path)
    gen = {
        "provider": "openai",
        "configured_model": "fake-model",
        "resolved_model_id": "fake-model",
        "weights_sha256": None,
        "quantization": None,
        "harness": "openai",
        "profile": "context-lite-v1",
    }

    first = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[1], "ユキは剣を取った。"),
        generator=gen,
        prompt="Return JSON context.",
        prompt_version="context-v1",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    first_path = tmp_path / "prior.json"
    write_artifact(first_path, first)
    second = generate_artifact(
        source,
        chapters=[2],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[2], "ユキは走った。"),
        generator=gen,
        prompt="Return JSON context.",
        prompt_version="context-v1",
        generated_at="2026-01-02T00:00:00+00:00",
        prior_artifact=(first, first_path),
    )
    sc.verify_self_context(second, source_dir=source)
    assert second["dependency_closure_sha256"] == sc.dependency_closure_hash(
        [first["artifact_sha256"]]
    )
    tampered = dict(first)
    tampered["generator"] = {**first["generator"], "profile": "tampered"}
    write_artifact(first_path, tampered)
    with pytest.raises(ValueError, match="prior self-context"):
        sc.verify_self_context(second, source_dir=source)


def _artifact_for(
    source_dir, chapter, quote, *, source_form="彼", resolved="fake-model", harness="openai"
):
    fake = ContextFake(chapter, quote, source_form=source_form)
    return generate_artifact(
        source_dir,
        chapters=[chapter.number],
        horizon_mode="chapter_preflight",
        call=fake,
        generator={
            "provider": harness,
            "configured_model": resolved,
            "resolved_model_id": resolved,
            "weights_sha256": None,
            "quantization": None,
            "harness": harness,
            "profile": "context-lite-v1",
        },
        prompt="Return JSON context.",
        prompt_version="context-v2",
        generated_at="2026-01-01T00:00:00+00:00",
    )


def test_translation_run_records_and_consumes_self_context(service, tmp_path):
    svc, runner = service
    source_dir = svc.config.novel.source_dir
    chapter = sc.discover(source_dir)[0]
    artifact = _artifact_for(source_dir, chapter, "第一章\n\n彼は森")
    path = tmp_path / "self-context.json"
    write_artifact(path, artifact)
    svc.config.translation.self_context_path = path
    svc.config.translation.context_composition = "self_only"
    rid = svc.create_run(runner="test")
    block = svc.repository.run(rid)["manifest"]["context"]
    assert block["context_composition"] == "self_only"
    assert block["self_context"]["artifact_sha256"] == artifact["artifact_sha256"]
    assert block["canonical_human"] is None
    assert (svc.run_dir(rid) / "self-context.json").is_file()
    assert svc.execute(rid)["status"] == "completed"
    consumed = [j.context["self_context"] for j in runner.calls if j.context.get("self_context")]
    assert consumed
    # rendered-context-budget-v1: exactly the rendered text is injected, never the
    # canonical entry JSON.
    assert all(set(projected) == {"text"} for projected in consumed)
    assert all(projected["text"] for projected in consumed)
    records = list((svc.run_dir(rid) / "self-context-projections").rglob("*.json"))
    assert records
    record = json.loads(records[0].read_text("utf-8"))
    assert record["consumption_budget_policy"] == sc.CONSUMPTION_BUDGET_POLICY
    assert record["rendered_context_chars"] >= 0
    assert record["canonical_selected_chars"] > record["rendered_context_chars"]
    # The measured string is exactly the injected string.
    record_hashes = {
        json.loads(path.read_text("utf-8"))["projection_sha256"] for path in records
    }
    assert {sc.sha256(projected["text"]) for projected in consumed} <= record_hashes


def test_shared_baseline_holds_human_hashes_constant(service, tmp_path):
    import yaml

    from novel_harness.config import load_glossary

    svc, _ = service
    source_dir = svc.config.novel.source_dir
    chapter = sc.discover(source_dir)[0]
    glossary_path = tmp_path / "glossary.yaml"
    glossary_path.write_text(
        yaml.safe_dump({"characters": {"彼": "Он"}}), encoding="utf-8"
    )
    svc.config.translation.glossary_path = glossary_path
    artifact = _artifact_for(source_dir, chapter, "第一章\n\n彼は森")
    path = tmp_path / "self-context.json"
    write_artifact(path, artifact)
    svc.config.translation.self_context_path = path
    svc.config.translation.context_composition = "self_plus_shared_baseline"
    rid = svc.create_run(runner="test")
    block = svc.repository.run(rid)["manifest"]["context"]
    assert block["context_composition"] == "self_plus_shared_baseline"
    assert block["canonical_human"]["shared_with_baseline"] is True
    assert block["canonical_human"]["glossary_sha256"] == sha256(
        canonical(load_glossary(glossary_path))
    )
    assert block["self_context"]["artifact_sha256"] == artifact["artifact_sha256"]
    # self_only is invalid with human semantic inputs.
    svc.config.translation.context_composition = "self_only"
    with pytest.raises(ValueError, match="self_only"):
        svc.create_run(runner="test", force=True)


def test_generation_path_freezes_artifact_with_provenance(tmp_path):
    source, chapters = make_source(tmp_path)
    fake = ContextFake(chapters[1], "ユキは剣を取った。")
    artifact = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=fake,
        generator=generator(),
        prompt="Return JSON context.",
        prompt_version="context-v2",
        generated_at="2026-01-01T00:00:00+00:00",
        runner_decoding={"mode": "json-schema"},
    )
    assert artifact["version"] == sc.VERSION
    assert artifact["context_origin"] == "self"
    assert artifact["horizon_mode"] == "chapter_preflight"
    assert artifact["generator"]["resolved_model_id"] == "fake-model"
    assert artifact["generator"]["reported_model"] == "fake-model"
    assert artifact["generator"]["model_source"] == "reported"
    assert artifact["prompt"]["version"] == "context-v2"
    assert artifact["decoding"]["mode"] == "json-schema"
    assert artifact["decoding"]["response_schema_version"] == sc.CONTEXT_RESPONSE_VERSION
    snapshot = artifact["snapshots"][0]
    assert snapshot["accepted_call_id"]
    call = snapshot["generation_calls"][0]
    assert call["status"] == "completed"
    assert call["total_tokens"] == 30
    assert call["prompt_tokens"] == 10 and call["completion_tokens"] == 20
    assert call["finish_reason"] == "stop"
    assert fake.calls[0].stage == "context"
    assert fake.calls[0].output_mode == "context"
    assert fake.calls[0].response_schema is not None
    sc.verify_self_context(artifact, source_dir=source)


def test_multi_snapshot_prior_is_not_a_cycle(tmp_path):
    source, chapters = make_source(tmp_path)
    first = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[1], "ユキは剣を取った。"),
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    path = tmp_path / "prior.json"
    write_artifact(path, first)
    multi = generate_artifact(
        source,
        chapters=[2, 3],
        horizon_mode="chapter_preflight",
        call=_auto_context_call,
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-02T00:00:00+00:00",
        prior_artifact=(first, path),
    )
    sc.verify_self_context(multi, source_dir=source)
    assert [s["target_chapter"] for s in multi["snapshots"]] == [2, 3]


def test_transitive_closure_includes_grandparent(tmp_path):
    source, chapters = make_source(tmp_path)
    a1 = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[1], "ユキは剣を取った。"),
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    p1 = tmp_path / "a1.json"
    write_artifact(p1, a1)
    a2 = generate_artifact(
        source,
        chapters=[2],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[2], "ユキは走った。"),
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-02T00:00:00+00:00",
        prior_artifact=(a1, p1),
    )
    p2 = tmp_path / "a2.json"
    write_artifact(p2, a2)
    a3 = generate_artifact(
        source,
        chapters=[3],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[3], "ユキは立った。"),
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-03T00:00:00+00:00",
        prior_artifact=(a2, p2),
    )
    sc.verify_self_context(a3, source_dir=source)
    assert a3["dependency_closure_sha256"] == sc.dependency_closure_hash(
        sorted([a1["artifact_sha256"], a2["artifact_sha256"]])
    )


def test_source_window_length_is_bounded(tmp_path):
    source, chapters = make_source(tmp_path)
    first, second = chapters[1], chapters[2]
    raw = {
        "snapshots": [
            {
                "target_chapter": 2,
                "inputs": [
                    {
                        "kind": "japanese_source",
                        "chapter": 2,
                        "filename": second.filename,
                        "source_sha256": second.source_sha256,
                        "normalized_text_sha256": sha256(second.text),
                        "scope": "full_chapter",
                    },
                    {
                        "kind": "source_window",
                        "chapter": 1,
                        "filename": first.filename,
                        "source_sha256": first.source_sha256,
                        "normalized_text_sha256": sha256(first.text),
                        "start": 0,
                        "end": 10,
                        "text_sha256": sha256(first.text[:10]),
                    },
                ],
                "entries": [character_entry(second, "ユキは走った。")],
            }
        ]
    }
    budget = sc.default_budget()
    budget["continuity_window_chars"] = 5
    with pytest.raises(ValueError, match="continuity_window"):
        sc.build_artifact(
            raw,
            source_dir=source,
            horizon_mode="chapter_preflight",
            generator=generator(),
            prompt={"version": "context-v1", "sha256": sha256("p")},
            decoding={},
            tool_version="t",
            budget=budget,
            generated_at="2026-01-01T00:00:00+00:00",
        )


def test_empty_chapter_preflight_snapshot_is_rejected(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    raw = {
        "snapshots": [
            {
                "target_chapter": 1,
                "empty_reason": "no_prior_source",
                "inputs": [
                    {
                        "kind": "japanese_source",
                        "chapter": 1,
                        "filename": chapter.filename,
                        "source_sha256": chapter.source_sha256,
                        "normalized_text_sha256": sha256(chapter.text),
                        "scope": "full_chapter",
                    }
                ],
                "entries": [],
            }
        ]
    }
    with pytest.raises(ValueError, match="empty snapshot"):
        sc.build_artifact(
            raw,
            source_dir=source,
            horizon_mode="chapter_preflight",
            generator=generator(),
            prompt={"version": "context-v1", "sha256": sha256("p")},
            decoding={},
            tool_version="t",
            budget=sc.default_budget(),
            generated_at="2026-01-01T00:00:00+00:00",
        )


def test_history_only_chapter_one_generates_empty_snapshot(tmp_path):
    source, _chapters = make_source(tmp_path)

    def never(_job):
        raise AssertionError("history_only chapter 1 must not call the model")

    artifact = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="history_only",
        call=never,
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    snapshot = artifact["snapshots"][0]
    assert snapshot["entries"] == []
    assert snapshot["empty_reason"] == "no_prior_source"
    assert snapshot["inputs"] == []
    sc.verify_self_context(artifact, source_dir=source)


def test_continuity_window_can_be_lowered(tmp_path):
    from novel_harness.context_runs import _budget_bounds

    source, chapters = make_source(tmp_path)
    artifact = build(source, chapters[1], entries=[character_entry(chapters[1], "ユキは剣を取った。")])
    assert _budget_bounds(artifact, {"continuity_window_chars": 1})[
        "continuity_window_chars"
    ] == 1


def test_canonical_human_only_is_recorded(service):
    svc, _ = service
    svc.config.translation.context_composition = "canonical_human_only"
    rid = svc.create_run(runner="test")
    block = svc.repository.run(rid)["manifest"]["context"]
    assert block["context_composition"] == "canonical_human_only"
    assert block["self_context"] is None
    assert block["canonical_human"]["glossary_sha256"]


def test_resume_rejects_tampered_canonical_human_hash(service, tmp_path):
    svc, _ = service
    import yaml

    source_dir = svc.config.novel.source_dir
    chapter = sc.discover(source_dir)[0]
    glossary_path = tmp_path / "glossary.yaml"
    glossary_path.write_text(yaml.safe_dump({"characters": {"彼": "Он"}}), encoding="utf-8")
    svc.config.translation.glossary_path = glossary_path
    artifact = _artifact_for(source_dir, chapter, "第一章\n\n彼は森")
    path = tmp_path / "self-context.json"
    write_artifact(path, artifact)
    svc.config.translation.self_context_path = path
    svc.config.translation.context_composition = "self_plus_shared_baseline"
    rid = svc.create_run(runner="test")
    manifest = json.loads(canonical(svc.repository.run(rid)["manifest"]))
    manifest["context"]["canonical_human"]["glossary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="glossary"):
        svc._verify_inputs(manifest)


def _alias_fake(chapter, quote, reported):
    class AliasFake(ContextFake):
        def __call__(self, job):
            self.calls.append(job)
            name = self.source_form
            entry = {
                "local_key": "char_1",
                "kind": "character",
                "semantic": {"class": "source_anchored_claim"},
                "claim": f"{name} appears",
                "source_forms": [name],
                "source_names": [name],
                "anchor_refs": [_request_quote_id(job)],
            }
            return RunnerResult(
                json.dumps({"entries": [entry]}),
                model=reported,
                finish_reason="stop",
                effective_settings={"reported_model": reported},
            )

    return AliasFake(chapter, quote)


def _configured_generator():
    return {
        "provider": "openai",
        "configured_model": "llamacpp99//mnt/models/configured.gguf",
        "resolved_model_id": "llamacpp99//mnt/models/configured.gguf",
        "weights_sha256": None,
        "quantization": None,
        "harness": "openai",
        "profile": "context-lite-v1",
    }


def test_reported_model_alias_is_accepted_without_changing_identity(tmp_path):
    from novel_harness.context_runs import load_artifact, write_artifact

    source, chapters = make_source(tmp_path)
    artifact = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=_alias_fake(chapters[1], "ユキは剣を取った。", "configured.gguf"),
        generator=_configured_generator(),
        prompt="p",
        prompt_version="context-v2",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert artifact["generator"]["resolved_model_id"] == "llamacpp99//mnt/models/configured.gguf"
    assert artifact["generator"]["reported_model"] == "configured.gguf"
    assert artifact["generator"]["model_source"] == "reported"
    path = tmp_path / "alias.json"
    write_artifact(path, artifact)
    sc.verify_self_context(load_artifact(path), source_dir=source)


def test_contradicting_reported_model_fails_closed(tmp_path):
    source, chapters = make_source(tmp_path)
    with pytest.raises(ValueError, match="contradicts"):
        generate_artifact(
            source,
            chapters=[1],
            horizon_mode="chapter_preflight",
            call=_alias_fake(chapters[1], "ユキは剣を取った。", "a-different-model"),
            generator=_configured_generator(),
            prompt="p",
            prompt_version="context-v2",
            generated_at="2026-01-01T00:00:00+00:00",
        )


def test_empty_history_artifact_can_be_referenced_as_prior(tmp_path):
    source, chapters = make_source(tmp_path)

    def never(_job):
        raise AssertionError("empty history snapshot must not call the model")

    empty = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="history_only",
        call=never,
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    path = tmp_path / "empty.json"
    write_artifact(path, empty)
    second = generate_artifact(
        source,
        chapters=[2],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[2], "ユキは走った。"),
        generator=generator(),
        prompt="p",
        prompt_version="context-v1",
        generated_at="2026-01-02T00:00:00+00:00",
        prior_artifact=(empty, path),
    )
    prior_input = next(
        item for item in second["snapshots"][0]["inputs"] if item["kind"] == "prior_self_context"
    )
    assert prior_input["max_source_chapter"] == 0
    sc.verify_self_context(second, source_dir=source)


def _units(chapter, source_ref="current_chapter", offset=0):
    return sc.build_quote_units(
        chapter.text, source_ref=source_ref, chapter_number=chapter.number, offset=offset
    )


def _ref(chapter, needle, source_ref="current_chapter"):
    for unit in _units(chapter, source_ref):
        if needle in unit["text"]:
            return unit["quote_id"]
    raise AssertionError(f"no quote unit containing {needle!r}")


def _model_entry(chapter, quote, *, semantic=None, source_ref="current_chapter", **overrides):
    entry = {
        "local_key": "char_1",
        "kind": "character",
        "semantic": semantic or {"class": "source_anchored_claim"},
        "claim": "ユキ appears",
        "source_forms": ["ユキ"],
        "source_names": ["ユキ"],
        "anchor_refs": [_ref(chapter, quote, source_ref)],
    }
    entry.update(overrides)
    return entry


def test_host_resolves_quote_units_to_exact_offsets_and_hashes(tmp_path):
    _source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    unit = _units(chapter)[0]
    entries = sc.normalize_model_response(
        {"entries": [_model_entry(chapter, unit["text"])]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=1,
    )
    anchor = entries[0]["anchors"][0]
    assert anchor["start"] == unit["start"]
    assert anchor["end"] == unit["end"]
    assert anchor["quote"] == unit["text"]
    assert anchor["source_sha256"] == chapter.source_sha256
    assert anchor["quote_sha256"] == sha256(unit["text"])


def test_unknown_quote_reference_fails_closed(tmp_path):
    _source, chapters = make_source(tmp_path)
    entry = _model_entry(chapters[1], "ユキは剣を取った。")
    entry["anchor_refs"] = ["cur-9999"]
    with pytest.raises(ValueError, match="quote reference"):
        sc.normalize_model_response(
            {"entries": [entry]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
        )


def test_out_of_horizon_quote_reference_fails(tmp_path):
    _source, chapters = make_source(tmp_path)
    units = sc.build_quote_units(
        chapters[2].text, source_ref="previous_window", chapter_number=2
    )
    table = {unit["quote_id"]: unit for unit in units}
    entry = _model_entry(chapters[2], "ユキは走った。", source_ref="previous_window")
    with pytest.raises(ValueError, match="horizon"):
        sc.normalize_model_response(
            {"entries": [entry]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
            quote_units=table,
        )


def test_local_references_resolve_to_deterministic_final_ids(tmp_path):
    _source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    response = {
        "entries": [
            {
                "local_key": "rel_1",
                "kind": "relationship",
                "semantic": {"class": "source_anchored_claim"},
                "relation": "knows",
                "source_forms": ["ユキ"],
                "subject_ref": "char_1",
                "object_ref": "char_2",
                "anchor_refs": [_ref(chapter, "ユキは剣を取った。")],
            },
            _model_entry(chapter, "ユキは剣を取った。", local_key="char_1"),
            _model_entry(
                chapter,
                "ユキは笑った。",
                local_key="char_2",
                claim="ユキは笑う",
                source_forms=["ユキ"],
                source_names=["ユキ"],
            ),
            {
                "local_key": "term_1",
                "kind": "term",
                "semantic": {"class": "source_anchored_claim"},
                "meaning": "a sword",
                "source_forms": ["剣"],
                "source_terms": ["剣"],
                "anchor_refs": [_ref(chapter, "ユキは剣を取った。")],
            },
            {
                "local_key": "speech_1",
                "kind": "speech",
                "semantic": {"class": "source_anchored_claim"},
                "pattern": "plain",
                "source_forms": ["ユキ"],
                "speaker_ref": "char_1",
                "anchor_refs": [_ref(chapter, "ユキは笑った。")],
            },
            {
                "local_key": "cont_1",
                "kind": "continuity",
                "semantic": {"class": "source_anchored_claim"},
                "claim": "they acted",
                "source_forms": ["ユキ"],
                "participant_refs": ["char_1", "char_2"],
                "anchor_refs": [_ref(chapter, "ユキは剣を取った。")],
            },
        ]
    }
    entries = sc.normalize_model_response(
        response, sources=chapters, horizon_mode="chapter_preflight", target_chapter=1
    )
    by_kind = {e["kind"]: e for e in entries}
    char_ids = {e["id"] for e in entries if e["kind"] == "character"}
    assert by_kind["relationship"]["subject_id"] in char_ids
    assert by_kind["relationship"]["object_id"] in char_ids
    assert by_kind["speech"]["speaker_id"] in char_ids
    assert set(by_kind["continuity"]["participant_ids"]) <= char_ids
    assert [e["id"] for e in entries] == sorted(e["id"] for e in entries)


def test_local_reference_errors_fail_closed(tmp_path):
    _source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    base = {
        "local_key": "rel_1",
        "kind": "relationship",
        "semantic": {"class": "source_anchored_claim"},
        "relation": "knows",
        "source_forms": ["ユキ"],
        "subject_ref": "char_1",
        "object_ref": "char_2",
        "anchor_refs": [_ref(chapter, "ユキは剣を取った。")],
    }
    with pytest.raises(ValueError, match="reference"):
        sc.normalize_model_response(
            {"entries": [base, _model_entry(chapter, "ユキは剣を取った。")]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
        )
    duplicate = [
        _model_entry(chapter, "ユキは剣を取った。", local_key="char_1"),
        _model_entry(chapter, "ユキは笑った。", local_key="char_1"),
    ]
    with pytest.raises(ValueError, match="duplicate local key"):
        sc.normalize_model_response(
            {"entries": duplicate},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
        )
    wrong_kind = [
        _model_entry(chapter, "ユキは剣を取った。", local_key="char_1"),
        {
            "local_key": "rel_1",
            "kind": "relationship",
            "semantic": {"class": "source_anchored_claim"},
            "relation": "knows",
            "source_forms": ["ユキ"],
            "subject_ref": "char_1",
            "object_ref": "term_1",
            "anchor_refs": [_ref(chapter, "ユキは剣を取った。")],
        },
        {
            "local_key": "term_1",
            "kind": "term",
            "semantic": {"class": "source_anchored_claim"},
            "meaning": "x",
            "source_forms": ["剣"],
            "source_terms": ["剣"],
            "anchor_refs": [_ref(chapter, "ユキは剣を取った。")],
        },
    ]
    with pytest.raises(ValueError, match="reference"):
        sc.normalize_model_response(
            {"entries": wrong_kind},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
        )


def test_deterministic_result_is_order_independent(tmp_path):
    _source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    a = _model_entry(chapter, "ユキは剣を取った。", local_key="char_a")
    b = _model_entry(chapter, "ユキは笑った。", local_key="char_b", claim="ユキは笑う")
    rel = {
        "local_key": "rel_1",
        "kind": "relationship",
        "semantic": {"class": "source_anchored_claim"},
        "relation": "knows",
        "source_forms": ["ユキ"],
        "subject_ref": "char_a",
        "object_ref": "char_b",
        "anchor_refs": [_ref(chapter, "ユキは剣を取った。")],
    }
    first = sc.normalize_model_response(
        {"entries": [a, b, rel]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=1,
    )
    second = sc.normalize_model_response(
        {"entries": [rel, b, a]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=1,
    )
    assert [e["id"] for e in first] == [e["id"] for e in second]


def test_hypothesis_and_unresolved_require_quote_and_uncertainty(tmp_path):
    _source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    hypothesis = _model_entry(
        chapter,
        "ユキは剣を取った。",
        semantic={
            "class": "model_hypothesis",
            "confidence": 0.4,
            "uncertainty": "gender inferred",
        },
    )
    entries = sc.normalize_model_response(
        {"entries": [hypothesis]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=1,
    )
    assert entries[0]["semantic_class"] == "model_hypothesis"
    unresolved = _model_entry(
        chapter,
        "ユキは剣を取った。",
        local_key="char_2",
        semantic={"class": "unresolved", "confidence": 0.5, "uncertainty": "unknown"},
    )
    sc.normalize_model_response(
        {"entries": [unresolved]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=1,
    )
    without_uncertainty = _model_entry(
        chapter,
        "ユキは剣を取った。",
        semantic={"class": "model_hypothesis", "confidence": 0.4},
    )
    with pytest.raises(ValueError):
        sc.normalize_model_response(
            {"entries": [without_uncertainty]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
        )
    without_anchor = dict(hypothesis)
    without_anchor.pop("anchor_refs")
    with pytest.raises(ValueError):
        sc.normalize_model_response(
            {"entries": [without_anchor]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
        )


def _schema_property_names(node, found):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                found.update(value.keys())
            _schema_property_names(value, found)
    elif isinstance(node, list):
        for value in node:
            _schema_property_names(value, found)
    return found


def test_context_response_schema_has_no_host_owned_fields():
    found = _schema_property_names(sc.context_response_schema(), set())
    host_fields = {
        "id",
        "source_sha256",
        "quote_sha256",
        "artifact_sha256",
        "dependency_closure_sha256",
        "start",
        "end",
    }
    assert not (host_fields & found)
    assert {"local_key", "kind", "semantic", "anchor_refs"} <= found
    defs = sc.context_response_schema()["$defs"]
    assert defs["CharacterWireEntry"]["additionalProperties"] is False
    assert defs["AnchoredSemantic"]["additionalProperties"] is False
    assert defs["HypothesisSemantic"]["additionalProperties"] is False


class RetryFake(ContextFake):
    def __init__(self, chapter, quote, failures):
        super().__init__(chapter, quote)
        self.failures = failures
        self.attempt = 0

    def __call__(self, job):
        self.attempt += 1
        self.calls.append(job)
        if self.attempt in self.failures:
            return RunnerResult("not json", finish_reason="stop", model="fake-model")
        return super().__call__(job)


def test_bounded_retry_recovers_malformed_then_valid(tmp_path):
    source, chapters = make_source(tmp_path)
    fake = RetryFake(chapters[1], "ユキは剣を取った。", failures={1})
    artifact = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=fake,
        generator=generator(),
        prompt="p",
        prompt_version="context-v2",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    calls = artifact["snapshots"][0]["generation_calls"]
    assert len(calls) == 2
    assert calls[0]["status"] == "failed" and calls[0]["failure_kind"] == "malformed_json"
    assert calls[1]["status"] == "completed"
    assert artifact["snapshots"][0]["accepted_call_id"] == calls[1]["call_id"]


def test_generation_record_derives_missing_total_tokens(tmp_path):
    source, _chapters = make_source(tmp_path)

    def call(job):
        entry = {
            "local_key": "char_1",
            "kind": "character",
            "semantic": {"class": "source_anchored_claim"},
            "claim": "c",
            "source_forms": ["ユキ"],
            "source_names": ["ユキ"],
            "anchor_refs": [_request_quote_id(job)],
        }
        return RunnerResult(
            json.dumps({"entries": [entry]}),
            model="fake-model",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    artifact = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=call,
        generator=generator(),
        prompt="p",
        prompt_version="context-v5",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    record = artifact["snapshots"][0]["generation_calls"][0]
    assert record["prompt_tokens"] == 10
    assert record["completion_tokens"] == 20
    assert record["total_tokens"] == 30


def test_runner_incomplete_turn_counts_against_retry_ceiling(tmp_path):
    from novel_harness.runners.base import RunnerExecutionError

    source, chapters = make_source(tmp_path)
    good = ContextFake(chapters[1], "ユキは剣を取った。")
    seen = {"count": 0}

    def call(job):
        seen["count"] += 1
        if seen["count"] == 1:
            raise RunnerExecutionError(
                "incomplete turn", failure_kind="incomplete", finish_reason="length"
            )
        return good(job)

    artifact = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=call,
        generator=generator(),
        prompt="p",
        prompt_version="context-v5",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    calls = artifact["snapshots"][0]["generation_calls"]
    assert calls[0]["status"] == "failed" and calls[0]["failure_kind"] == "incomplete"
    assert calls[1]["status"] == "completed"


def test_retry_attempts_use_distinct_runner_workspace_ids(tmp_path):
    source, chapters = make_source(tmp_path)
    good = ContextFake(chapters[1], "ユキは剣を取った。")
    seen: list[str] = []

    def call(job):
        seen.append(job.chunk_id)
        if len(seen) == 1:
            return RunnerResult("not json", finish_reason="stop", model="fake-model")
        return good(job)

    generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=call,
        generator=generator(),
        prompt="p",
        prompt_version="context-v2",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert len(seen) == 2
    assert len(set(seen)) == len(seen)


def test_bounded_retry_exhaustion_fails_closed(tmp_path):
    source, chapters = make_source(tmp_path)
    fake = RetryFake(chapters[1], "ユキは剣を取った。", failures={1, 2, 3})
    with pytest.raises(ValueError, match="after 3 attempt"):
        generate_artifact(
            source,
            chapters=[1],
            horizon_mode="chapter_preflight",
            call=fake,
            generator=generator(),
            prompt="p",
            prompt_version="context-v2",
            retry_ceiling=2,
            generated_at="2026-01-01T00:00:00+00:00",
        )


def test_prior_context_is_projected_into_the_request(tmp_path):
    from novel_harness.context_runs import write_artifact

    source, chapters = make_source(tmp_path)
    first = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[1], "ユキは剣を取った。"),
        generator=generator(),
        prompt="p",
        prompt_version="context-v2",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    path = tmp_path / "prior.json"
    write_artifact(path, first)
    fake = ContextFake(chapters[2], "ユキは走った。")
    second = generate_artifact(
        source,
        chapters=[2],
        horizon_mode="chapter_preflight",
        call=fake,
        generator=generator(),
        prompt="p",
        prompt_version="context-v2",
        generated_at="2026-01-02T00:00:00+00:00",
        prior_artifact=(first, path),
    )
    job = fake.calls[0]
    projected = job.context["prior_self_context"]
    assert projected["entries"]
    assert all("quote" in e and "id" not in e for e in projected["entries"])
    prior_input = next(
        i for i in second["snapshots"][0]["inputs"] if i["kind"] == "prior_self_context"
    )
    assert prior_input["projection"]["selection"] == sc.PRIOR_PROJECTION_VERSION
    assert prior_input["projection"]["projection_sha256"] == sc.project_prior(
        first,
        target_chapter=2,
        source_text=chapters[2].text,
        budget=sc.default_budget(),
    )[1]["projection_sha256"]
    sc.verify_self_context(second, source_dir=source)


def test_tampered_prior_fails_projection_or_verification(tmp_path):
    from novel_harness.context_runs import write_artifact

    source, chapters = make_source(tmp_path)
    first = generate_artifact(
        source,
        chapters=[1],
        horizon_mode="chapter_preflight",
        call=ContextFake(chapters[1], "ユキは剣を取った。"),
        generator=generator(),
        prompt="p",
        prompt_version="context-v2",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    path = tmp_path / "prior.json"
    tampered = json.loads(json.dumps(first))
    tampered["generator"]["profile"] = "tampered"
    write_artifact(path, tampered)
    with pytest.raises(ValueError, match="prior self-context"):
        generate_artifact(
            source,
            chapters=[2],
            horizon_mode="chapter_preflight",
            call=ContextFake(chapters[2], "ユキは走った。"),
            generator=generator(),
            prompt="p",
            prompt_version="context-v2",
            generated_at="2026-01-02T00:00:00+00:00",
            prior_artifact=(tampered, path),
        )


def test_prior_projection_excludes_non_anchored_entries(tmp_path):
    source, chapters = make_source(tmp_path)
    hypothesis = {
        "kind": "character",
        "semantic_class": "model_hypothesis",
        "claim": "maybe ユキ",
        "source_forms": ["ユキ"],
        "source_names": ["ユキ"],
        "anchors": [source_anchor(chapters[1], "ユキは剣を取った。")],
        "confidence": 0.5,
        "uncertainty": "gender inferred",
    }
    prior = build(
        source,
        chapters[1],
        entries=[character_entry(chapters[1], "ユキは剣を取った。"), hypothesis],
    )
    payload, metadata = sc.project_prior(
        prior,
        target_chapter=2,
        source_text=chapters[2].text,
        budget=sc.default_budget(),
    )
    assert metadata["entry_count"] == 1
    assert payload[0]["semantic_class"] == "source_anchored_claim"


# --- context-generation-v4 structural wire contract with host quote refs -----------


def _v4_validator():
    return Draft202012Validator(sc.context_response_schema())


def _wire():
    return ["cur-0001"]


def _base_entry():
    return {
        "local_key": "x_1",
        "semantic": {"class": "source_anchored_claim"},
        "source_forms": ["ユキ"],
        "anchor_refs": _wire(),
    }


def test_v4_schema_rejects_kinds_missing_their_required_fields():
    validator = _v4_validator()
    invalid = [
        {**_base_entry(), "kind": "term", "source_terms": ["剣"]},
        {**_base_entry(), "kind": "term", "meaning": "a sword"},
        {**_base_entry(), "kind": "term", "meaning": "a sword", "source_terms": []},
        {**_base_entry(), "kind": "relationship", "relation": "knows", "object_ref": "b"},
        {**_base_entry(), "kind": "relationship", "relation": "knows", "subject_ref": "a"},
        {**_base_entry(), "kind": "speech", "pattern": "plain"},
        {**_base_entry(), "kind": "character", "source_names": ["ユキ"]},
        {**_base_entry(), "kind": "continuity"},
        {**_base_entry(), "kind": "chapter_preflight_fact"},
    ]
    for entry in invalid:
        assert list(validator.iter_errors({"entries": [entry]})), entry


def test_v4_schema_rejects_semantic_class_shape_violations():
    validator = _v4_validator()
    base = {
        "local_key": "x_1",
        "kind": "character",
        "claim": "c",
        "source_forms": ["ユキ"],
        "source_names": ["ユキ"],
        "anchor_refs": _wire(),
    }
    invalid = [
        {**base, "semantic": {"class": "source_anchored_claim", "confidence": 0.5}},
        {**base, "semantic": {"class": "source_anchored_claim", "uncertainty": "x"}},
        {**base, "semantic": {"class": "model_hypothesis", "confidence": 0.5}},
        {**base, "semantic": {"class": "model_hypothesis", "uncertainty": "x"}},
        {**base, "semantic": {"class": "unresolved", "confidence": 0.5}},
        {**base, "semantic": {"class": "unresolved"}},
        {**base, "semantic": {"class": "confirmed"}},
    ]
    for entry in invalid:
        assert list(validator.iter_errors({"entries": [entry]})), entry


def test_v4_schema_rejects_forbidden_extra_fields():
    validator = _v4_validator()
    term = {
        **_base_entry(),
        "kind": "term",
        "meaning": "a sword",
        "source_terms": ["剣"],
    }
    assert list(validator.iter_errors({"entries": [{**term, "subject_ref": "a"}]}))
    assert list(validator.iter_errors({"entries": [{**term, "speaker_ref": "a"}]}))
    assert list(validator.iter_errors({"entries": [{**term, "confidence": 0.5}]}))


def test_v4_schema_and_host_accept_every_kind_and_semantic_shape(tmp_path):
    validator = _v4_validator()
    _source, chapters = make_source(tmp_path)
    entries = [
        {
            "local_key": "char_1",
            "kind": "character",
            "semantic": {"class": "source_anchored_claim"},
            "claim": "c1",
            "source_forms": ["ユキ"],
            "source_names": ["ユキ"],
            "anchor_refs": _wire(),
        },
        {
            "local_key": "char_2",
            "kind": "character",
            "semantic": {
                "class": "model_hypothesis",
                "confidence": 0.4,
                "uncertainty": "inferred",
            },
            "claim": "c2",
            "source_forms": ["ユキ"],
            "source_names": ["ユキ"],
            "anchor_refs": _wire(),
        },
        {
            "local_key": "char_3",
            "kind": "character",
            "semantic": {"class": "unresolved", "confidence": 0.3, "uncertainty": "unknown"},
            "claim": "c3",
            "source_forms": ["ユキ"],
            "source_names": ["ユキ"],
            "anchor_refs": _wire(),
        },
        {
            "local_key": "rel_1",
            "kind": "relationship",
            "semantic": {"class": "source_anchored_claim"},
            "relation": "knows",
            "source_forms": ["ユキ"],
            "subject_ref": "char_1",
            "object_ref": "char_2",
            "anchor_refs": _wire(),
        },
        {
            "local_key": "term_1",
            "kind": "term",
            "semantic": {"class": "source_anchored_claim"},
            "meaning": "a sword",
            "source_terms": ["剣"],
            "source_forms": ["剣"],
            "anchor_refs": _wire(),
        },
        {
            "local_key": "speech_1",
            "kind": "speech",
            "semantic": {"class": "source_anchored_claim"},
            "pattern": "plain",
            "speaker_ref": "char_1",
            "source_forms": ["ユキ"],
            "anchor_refs": _wire(),
        },
        {
            "local_key": "cont_1",
            "kind": "continuity",
            "semantic": {"class": "source_anchored_claim"},
            "claim": "they acted",
            "participant_refs": ["char_1", "char_2"],
            "source_forms": ["ユキ"],
            "anchor_refs": _wire(),
        },
        {
            "local_key": "cpre_1",
            "kind": "chapter_preflight_fact",
            "semantic": {"class": "source_anchored_claim"},
            "claim": "fact",
            "source_forms": ["ユキ"],
            "anchor_refs": _wire(),
        },
    ]
    response = {"entries": entries}
    assert not list(validator.iter_errors(response))
    normalized = sc.normalize_model_response(
        response, sources=chapters, horizon_mode="chapter_preflight", target_chapter=1
    )
    kinds = {entry["kind"] for entry in normalized}
    assert kinds == {
        "character",
        "relationship",
        "term",
        "speech",
        "continuity",
        "chapter_preflight_fact",
    }
    semantic_classes = {entry["semantic_class"] for entry in normalized}
    assert semantic_classes == {"source_anchored_claim", "model_hypothesis", "unresolved"}


# --- context-generation-v4 host-owned quote refs -----------------------------------


def _quote_table(previous, current, *, start=None, end=None):
    table = {}
    if previous is not None:
        text = previous.text
        begin = 0 if start is None else start
        stop = len(text) if end is None else end
        for unit in sc.build_quote_units(
            text[begin:stop],
            source_ref="previous_window",
            chapter_number=previous.number,
            offset=begin,
        ):
            table[unit["quote_id"]] = unit
    if current is not None:
        for unit in sc.build_quote_units(
            current.text, source_ref="current_chapter", chapter_number=current.number
        ):
            table[unit["quote_id"]] = unit
    return table


def test_v4_current_chapter_quote_resolves_to_current_chapter(tmp_path):
    _source, chapters = make_source(tmp_path)
    unit = _units(chapters[2])[0]
    entries = sc.normalize_model_response(
        {"entries": [_model_entry(chapters[2], unit["text"])]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=2,
        quote_units=_quote_table(chapters[1], chapters[2]),
    )
    anchor = entries[0]["anchors"][0]
    assert anchor["chapter"] == 2
    assert anchor["start"] == unit["start"]
    assert anchor["quote"] == unit["text"]
    assert anchor["source_sha256"] == chapters[2].source_sha256


def test_v4_previous_window_subdivision_and_absolute_offsets(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    long_line = "あ" * 150 + "。" + "い" * 150 + "。"
    (source / "0001.txt").write_text("第一章\n\n" + long_line + "\n", encoding="utf-8")
    (source / "0002.txt").write_text("第二章\n\n現在の章。\n", encoding="utf-8")
    chapters = {chapter.number: chapter for chapter in sc.discover(source)}
    previous = chapters[1]
    units = sc.build_quote_units(
        previous.text, source_ref="previous_window", chapter_number=previous.number
    )
    assert len(units) >= 2
    second = units[1]
    assert second["start"] > 0
    table = {unit["quote_id"]: unit for unit in units}
    table.update(
        {
            unit["quote_id"]: unit
            for unit in sc.build_quote_units(
                chapters[2].text, source_ref="current_chapter", chapter_number=2
            )
        }
    )
    entries = sc.normalize_model_response(
        {"entries": [_model_entry(previous, second["text"], source_ref="previous_window")]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=2,
        quote_units=table,
    )
    anchor = entries[0]["anchors"][0]
    assert anchor["chapter"] == 1
    assert anchor["start"] == second["start"]
    assert anchor["end"] == second["end"]
    assert anchor["quote"] == second["text"]
    assert anchor["source_sha256"] == previous.source_sha256


def test_v4_previous_unit_is_rejected_when_not_supplied(tmp_path):
    _source, chapters = make_source(tmp_path)
    entry = _model_entry(chapters[1], "ユキは笑った。", source_ref="previous_window")
    with pytest.raises(ValueError, match="quote reference"):
        sc.normalize_model_response(
            {"entries": [entry]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=2,
            quote_units=_quote_table(None, chapters[2]),
        )


def test_v4_quote_unit_slice_mismatch_fails(tmp_path):
    _source, chapters = make_source(tmp_path)
    unit = _units(chapters[2])[0]
    table = {unit["quote_id"]: {**unit, "text": "tampered"}}
    entry = _model_entry(chapters[2], unit["text"])
    with pytest.raises(ValueError, match="quote unit"):
        sc.normalize_model_response(
            {"entries": [entry]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=2,
            quote_units=table,
        )


def test_v4_segmentation_preserves_ruby_whitespace_and_lines(tmp_path):
    text = (
        "第一章\n\n満身創痍(まんしんそうい)だ\u3000その通り。\n"
        "A B は半角空白。\n「次は」\n「これだ」\n"
    )
    units = [text[start:end] for start, end in sc.segment_source_text(text)]
    assert "満身創痍(まんしんそうい)だ\u3000その通り。" in units
    assert "A B は半角空白。" in units
    assert "「次は」" in units
    assert "「これだ」" in units
    assert "「次は」「これだ」" not in units
    for unit in units:
        assert unit in text


def test_v4_clipped_previous_window_units_map_to_absolute_offsets(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    long_line = "あ" * 300 + "。" + "い" * 300 + "。"
    (source / "0001.txt").write_text("第一章\n\n" + long_line + "\n", encoding="utf-8")
    (source / "0002.txt").write_text("第二章\n\n現在の章。\n", encoding="utf-8")
    chapters = {chapter.number: chapter for chapter in sc.discover(source)}
    previous = chapters[1]
    window_start = 100
    window_text = previous.text[window_start:]
    previous_units = sc.build_quote_units(
        window_text,
        source_ref="previous_window",
        chapter_number=previous.number,
        offset=window_start,
    )
    assert previous_units[0]["start"] == window_start
    target = previous_units[-1]
    assert target["start"] > window_start
    table = {unit["quote_id"]: unit for unit in previous_units}
    table.update(
        {
            unit["quote_id"]: unit
            for unit in sc.build_quote_units(
                chapters[2].text, source_ref="current_chapter", chapter_number=2
            )
        }
    )
    entry = {
        "local_key": "char_1",
        "kind": "character",
        "semantic": {"class": "source_anchored_claim"},
        "claim": "c",
        "source_forms": ["ユキ"],
        "source_names": ["ユキ"],
        "anchor_refs": [target["quote_id"]],
    }
    entries = sc.normalize_model_response(
        {"entries": [entry]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=2,
        quote_units=table,
    )
    anchor = entries[0]["anchors"][0]
    assert anchor["chapter"] == previous.number
    assert anchor["start"] == target["start"]
    assert anchor["end"] == target["end"]
    assert anchor["quote"] == previous.text[target["start"] : target["end"]]


def test_v4_schema_rejects_bad_refs_and_host_owned_anchor_fields():
    validator = _v4_validator()
    base = {
        "local_key": "char_1",
        "kind": "character",
        "semantic": {"class": "source_anchored_claim"},
        "claim": "c",
        "source_forms": ["ユキ"],
        "source_names": ["ユキ"],
        "anchor_refs": ["cur-0001"],
    }
    assert list(validator.iter_errors({"entries": [{**base, "anchor_refs": ["invented"]}]}))
    assert list(validator.iter_errors({"entries": [{**base, "anchor_refs": []}]}))
    assert list(validator.iter_errors({"entries": [{**base, "anchor_refs": ["cur-0001", "cur-0001"]}]}))
    assert list(
        validator.iter_errors(
            {"entries": [{**base, "anchor_refs": ["cur-0001", "cur-0002", "cur-0003", "cur-0004"]}]}
        )
    )
    # The model can never inject a quote text / chapter / offset / hash anchor object.
    assert list(validator.iter_errors({"entries": [{**base, "anchor": {"quote": "x"}}]}))
    assert list(validator.iter_errors({"entries": [{**base, "quote": "x"}]}))
    assert list(validator.iter_errors({"entries": [{**base, "chapter": 1}]}))
    assert list(validator.iter_errors({"entries": [{**base, "source_sha256": "0" * 64}]}))


def test_v4_duplicate_anchor_refs_fail_closed(tmp_path):
    _source, chapters = make_source(tmp_path)
    unit = _units(chapters[1])[0]
    entry = _model_entry(chapters[1], unit["text"], anchor_refs=[unit["quote_id"], unit["quote_id"]])
    with pytest.raises(ValueError):
        sc.normalize_model_response(
            {"entries": [entry]},
            sources=chapters,
            horizon_mode="chapter_preflight",
            target_chapter=1,
        )


def test_v4_generated_artifact_remains_self_context_lite_v1(tmp_path):
    from novel_harness.context_runs import generate_artifact, load_artifact, write_artifact

    source, _chapters = make_source(tmp_path)
    artifact = generate_artifact(
        source,
        chapters=[2],
        horizon_mode="chapter_preflight",
        call=_auto_context_call,
        generator=generator(),
        prompt="p",
        prompt_version="context-v5",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert artifact["version"] == sc.VERSION
    assert artifact["horizon_mode"] == "chapter_preflight"
    assert artifact["decoding"]["source_addressing_version"] == sc.SOURCE_ADDRESSING_VERSION
    assert artifact["decoding"]["response_schema_version"] == "context-generation-v4"
    anchor = artifact["snapshots"][0]["entries"][0]["anchors"][0]
    assert set(anchor) == {"chapter", "source_sha256", "start", "end", "quote", "quote_sha256"}
    write_artifact(tmp_path / "artifact.json", artifact)
    sc.verify_self_context(load_artifact(tmp_path / "artifact.json"), source_dir=source)


def test_v4_schema_rejects_duplicate_and_invalid_forms_and_refs():
    validator = _v4_validator()

    def character(**overrides):
        entry = {
            "local_key": "c1",
            "kind": "character",
            "semantic": {"class": "source_anchored_claim"},
            "claim": "c",
            "source_forms": ["ユキ"],
            "source_names": ["ユキ"],
            "anchor_refs": _wire(),
        }
        entry.update(overrides)
        return {"entries": [entry]}

    assert list(validator.iter_errors(character(source_forms=["ユキ", "ユキ"])))
    assert list(validator.iter_errors(character(source_forms=[" ユキ"])))
    assert list(validator.iter_errors(character(source_forms=["ユキ "])))
    assert list(validator.iter_errors(character(source_forms=[""])))
    assert list(validator.iter_errors(character(source_names=["ユキ", "ユキ"])))
    assert list(validator.iter_errors(character(aliases=["x", "x"])))
    assert list(validator.iter_errors(character(anchor_refs=[])))
    assert list(validator.iter_errors(character(anchor_refs=["cur-0001", "cur-0001"])))
    assert list(validator.iter_errors(character(anchor_refs=["invented"])))
    term = {
        "entries": [
            {
                "local_key": "t",
                "kind": "term",
                "semantic": {"class": "source_anchored_claim"},
                "meaning": "m",
                "source_terms": ["剣", "剣"],
                "source_forms": ["剣"],
                "anchor_refs": _wire(),
            }
        ]
    }
    assert list(validator.iter_errors(term))
    relationship = {
        "entries": [
            {
                "local_key": "r",
                "kind": "relationship",
                "semantic": {"class": "source_anchored_claim"},
                "relation": "knows",
                "subject_ref": "a b",
                "object_ref": "c",
                "source_forms": ["ユキ"],
                "anchor_refs": _wire(),
            }
        ]
    }
    assert list(validator.iter_errors(relationship))
    continuity = {
        "entries": [
            {
                "local_key": "n",
                "kind": "continuity",
                "semantic": {"class": "source_anchored_claim"},
                "claim": "c",
                "participant_refs": ["a", "a"],
                "source_forms": ["ユキ"],
                "anchor_refs": _wire(),
            }
        ]
    }
    assert list(validator.iter_errors(continuity))


def test_v4_host_rejects_relationship_self_endpoint(tmp_path):
    _source, chapters = make_source(tmp_path)
    response = {
        "entries": [
            _model_entry(chapters[1], "ユキは剣を取った。", local_key="char_1"),
            {
                "local_key": "rel_1",
                "kind": "relationship",
                "semantic": {"class": "source_anchored_claim"},
                "relation": "knows",
                "subject_ref": "char_1",
                "object_ref": "char_1",
                "source_forms": ["ユキ"],
                "anchor_refs": [_ref(chapters[1], "ユキは剣を取った。")],
            },
        ]
    }
    with pytest.raises(ValueError, match="distinct"):
        sc.normalize_model_response(
            response, sources=chapters, horizon_mode="chapter_preflight", target_chapter=1
        )


def test_v4_multiple_anchor_refs_resolve_to_multiple_anchors(tmp_path):
    _source, chapters = make_source(tmp_path)
    table = _quote_table(None, chapters[1])
    ids = list(table)
    assert len(ids) >= 2
    entry = _model_entry(chapters[1], _units(chapters[1])[0]["text"])
    entry["anchor_refs"] = ids[:2]
    entries = sc.normalize_model_response(
        {"entries": [entry]},
        sources=chapters,
        horizon_mode="chapter_preflight",
        target_chapter=1,
        quote_units=table,
    )
    anchors = entries[0]["anchors"]
    assert len(anchors) == 2
    assert {anchor["quote"] for anchor in anchors} == {table[i]["text"] for i in ids[:2]}


def test_v4_generation_request_exposes_explicit_quote_units(tmp_path):
    from novel_harness.context_runs import generate_artifact

    source, _chapters = make_source(tmp_path)
    captured: dict = {}

    def capture(job):
        captured["source"] = job.source
        captured["documents"] = job.context.get("source_documents")
        captured["segmentation"] = job.context.get("quote_unit_segmentation_version")
        return _auto_context_call(job)

    generate_artifact(
        source,
        chapters=[2],
        horizon_mode="chapter_preflight",
        call=capture,
        generator=generator(),
        prompt="p",
        prompt_version="context-v5",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    documents = {doc["source_ref"]: doc for doc in captured["documents"]}
    assert documents["current_chapter"]["chapter"] == 2
    assert documents["previous_window"]["chapter"] == 1
    assert captured["segmentation"] == sc.QUOTE_UNIT_SEGMENTATION_VERSION
    # No anonymous concatenated source text is sent alongside the explicit units.
    assert captured["source"] == ""
    for document in captured["documents"]:
        assert "text" not in document
        for unit in document["quote_units"]:
            assert set(unit) == {"quote_id", "text"}


def test_generated_response_is_capped_at_twelve_entries():
    assert sc.MAX_GENERATED_CONTEXT_ENTRIES == 12
    assert sc.MAX_ENTRIES_PER_SNAPSHOT == 64
    schema = sc.context_response_schema()
    assert schema["properties"]["entries"]["maxItems"] == 12
    assert sc.Snapshot.model_json_schema()["properties"]["entries"]["maxItems"] == 64
    validator = _v4_validator()

    def entry(index):
        return {
            "local_key": f"c{index}",
            "kind": "character",
            "semantic": {"class": "source_anchored_claim"},
            "claim": "c",
            "source_forms": [f"f{index}"],
            "source_names": [f"n{index}"],
            "anchor_refs": _wire(),
        }

    assert not list(validator.iter_errors({"entries": [entry(i) for i in range(12)]}))
    assert list(validator.iter_errors({"entries": [entry(i) for i in range(13)]}))
    with pytest.raises(ValueError):
        sc.ModelResponse.model_validate({"entries": [entry(i) for i in range(13)]})


def test_context_generation_policy_defaults_are_preserved():
    import inspect

    import yaml

    from novel_harness.context_runs import generate_artifact

    signature = inspect.signature(generate_artifact)
    assert signature.parameters["retry_ceiling"].default == 2
    root = Path(__file__).resolve().parents[1]
    example = yaml.safe_load(
        (root / "resources/example-book/evaluation-config.example.yaml").read_text("utf-8")
    )
    assert example["runners"]["qwen-context"]["max_tokens"] == 8000


def test_context_v5_prompt_caps_entries_orders_priorities_and_uses_quote_refs():
    prompt = (Path(__file__).resolve().parents[1] / "prompts/context-v5.md").read_text("utf-8")
    assert "at most 12 entries" in prompt
    assert "Do not exhaustively catalog" in prompt
    assert "anchor_refs" in prompt
    assert "quote_id" in prompt
    for marker in (
        "character identity / speaker / gender",
        "actor-patient and relationship",
        "ambiguous names, aliases and important terms",
        "speech/register",
        "immediate continuity / chapter-preflight",
    ):
        assert marker in prompt


def _projection_artifact(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    entries = [
        character_entry(chapter, "ユキは剣を取った。", claim=f"claim {index}")
        for index in range(12)
    ]
    artifact = build(source, chapter, entries=entries)
    chunk = {"text": chapter.text, "start": 0, "end": len(chapter.text)}
    return artifact, chunk


def test_rendered_budget_accepts_canonical_over_budget(tmp_path):
    from novel_harness.context_runs import projection_for_chunk

    artifact, chunk = _projection_artifact(tmp_path)
    _, projection, meta = projection_for_chunk(artifact, chapter=1, chunk=chunk)
    # The operator case: canonical selected JSON exceeds 4000, rendered is far below.
    assert meta["canonical_selected_chars"] > 4000
    assert meta["rendered_context_chars"] < 4000
    _, projection2, meta2 = projection_for_chunk(
        artifact,
        chapter=1,
        chunk=chunk,
        budget={"max_context_chars": meta["rendered_context_chars"]},
    )
    assert meta2["canonical_selected_chars"] > meta2["rendered_context_chars"]
    assert projection2 == projection
    assert meta2["consumption_budget_policy"] == sc.CONSUMPTION_BUDGET_POLICY


def test_rendered_budget_rejects_over_chars_and_tokens(tmp_path):
    from novel_harness.context_runs import projection_for_chunk

    artifact, chunk = _projection_artifact(tmp_path)
    _, _, meta = projection_for_chunk(artifact, chapter=1, chunk=chunk)
    with pytest.raises(ValueError, match="max_context_chars"):
        projection_for_chunk(
            artifact,
            chapter=1,
            chunk=chunk,
            budget={"max_context_chars": max(1, meta["rendered_context_chars"] - 1)},
        )
    with pytest.raises(ValueError, match="max_estimated_context_tokens"):
        projection_for_chunk(
            artifact, chapter=1, chunk=chunk, budget={"max_estimated_context_tokens": 0}
        )


def test_rendered_projection_is_deterministic_and_measured_exactly(tmp_path):
    from novel_harness.context_runs import projection_for_chunk

    artifact, chunk = _projection_artifact(tmp_path)
    entries, projection, meta = projection_for_chunk(artifact, chapter=1, chunk=chunk)
    assert projection == sc.render_projection(entries)
    assert meta["rendered_context_chars"] == len(projection)
    assert meta["projection_sha256"] == sc.sha256(projection)
    assert meta["selected_entry_count"] == len(entries)
    _, projection2, meta2 = projection_for_chunk(artifact, chapter=1, chunk=chunk)
    assert projection2 == projection
    assert meta2["projection_sha256"] == meta["projection_sha256"]


def test_rendered_budget_selected_count_and_lowering_only(tmp_path):
    from novel_harness.context_runs import _budget_bounds, projection_for_chunk

    artifact, chunk = _projection_artifact(tmp_path)
    with pytest.raises(ValueError, match="max_selected_entries"):
        projection_for_chunk(
            artifact, chapter=1, chunk=chunk, budget={"max_selected_entries": 1}
        )
    assert _budget_bounds(artifact, {"max_context_chars": 100}) == {"max_chars": 100}
    assert _budget_bounds(artifact, {"max_context_chars": 999999}) == {"max_chars": 4000}
    assert _budget_bounds(artifact, {"max_estimated_context_tokens": 999999}) == {
        "max_tokens": 2000
    }


def test_snapshot_char_limit_still_enforced(tmp_path):
    source, chapters = make_source(tmp_path)
    chapter = chapters[1]
    budget = {**sc.default_budget(), "max_snapshot_chars": 50}
    with pytest.raises(ValueError, match="max_snapshot_chars"):
        build(
            source,
            chapter,
            entries=[character_entry(chapter, "ユキは剣を取った。")],
            budget=budget,
        )


def test_prior_projection_policy_is_unchanged():
    assert sc.PRIOR_PROJECTION_VERSION == "prior-context-projection-v1"
    assert sc.CONSUMPTION_BUDGET_POLICY == "rendered-context-budget-v1"


def test_historical_context_prompts_are_byte_identical():
    root = Path(__file__).resolve().parents[1]
    pinned = {
        "context-v2.md": "2e976298568561ef753aa8ac973650bb50c6fef978928acfb5e2a34474b77820",
        "context-v3.md": "b54f4dbd65a899ddac0b0de97a6f2ea0043a6a7f781761b419ecd9264537f900",
        "context-v4.md": "40f815e6f3eb15649a552fb21b0424abb52f24459047982881384ae62357f3e5",
    }
    for name, digest in pinned.items():
        assert sha256((root / "prompts" / name).read_text("utf-8")) == digest
    assert sc.CONTEXT_RESPONSE_VERSION == "context-generation-v4"
    assert sc.SOURCE_ADDRESSING_VERSION == "quote-ref-addressing-v1"
    assert sc.QUOTE_UNIT_SEGMENTATION_VERSION == "quote-unit-segmentation-v1"
