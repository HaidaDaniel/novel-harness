import json

import pytest

from novel_harness.artifacts import sha256
from novel_harness.cache import TranslationCache
from novel_harness.config import ValidationConfig
from novel_harness.runners.base import RunnerResult
from novel_harness.translation.segments import source_segments
from novel_harness.translation.validator import (
    foreign_script_count,
    mixed_script_tokens,
    validate,
    validate_segments,
)

# Passes the legacy policy: the short katakana leak is neither a 16+ run nor a
# dominant Japanese share, so only the explicit strict profile catches it.
SHORT_LEAK = "Он вошёл в данジョンa и остановился, чтобы внимательно осмотреться вокруг."
CLEAN_RUSSIAN = (
    "Он осторожно вошёл в комнату и остановился, внимательно осмотревшись вокруг себя."
)


def strict_config():
    return ValidationConfig(max_foreign_script_chars=0, reject_malformed_dialogue=True)


def test_v22_chinese_loop_rejected_and_natural_dialogue_allowed():
    result = validate("マリオンなら直せる！", "Марион " + "一定能" * 78, "stop", ValidationConfig())
    assert result.level == "FAIL"
    assert {"foreign_script_run", "runaway_repetition"} <= set(result.issues)
    assert validate("「はい！」" * 20, "Да! " * 20, "stop", ValidationConfig()).level != "FAIL"
    assert validate("原文。", "Пауза............", "stop", ValidationConfig()).level != "FAIL"
    assert (
        "unbalanced_quotes"
        in validate("原文。", "«Незакрытая реплика.", None, ValidationConfig()).issues
    )


@pytest.mark.parametrize("source_copies,output_copies", [(5, 12), (9, 21), (10, 12)])
def test_v33_short_source_cries_allow_bounded_russian_expansion(source_copies, output_copies):
    source = "「うお" + "あ" * source_copies + "！」"
    output = "«Уо" + "-а" * output_copies + "!»"
    assert "runaway_repetition" not in validate(source, output, "stop", ValidationConfig()).issues
    excessive = "«Уо" + "-а" * (source_copies * 3 + 1) + "!»"
    assert "runaway_repetition" in validate(source, excessive, "stop", ValidationConfig()).issues


def test_v33_thousands_of_syllables_still_fail():
    source = "「" + "あ" * 14 + "！」"
    output = "«А" + "-а" * 2776 + "!»"
    assert "runaway_repetition" in validate(source, output, "stop", ValidationConfig()).issues


def test_v22_retry_uses_clean_source_before_editor(service):
    svc, fake = service
    svc.config.translation.retries = 1
    calls = []

    class CorruptOnce:
        def run(self, job):
            calls.append(job)
            if len(calls) == 1:
                return RunnerResult("Марион " + "一定能" * 78, finish_reason="stop")
            return fake.run(job)

    svc.runner_factory = lambda _: CorruptOnce()
    rid = svc.create_run(translator="test", editor="test", pipeline="translate-edit")
    assert svc.execute(rid)["status"] == "completed"
    assert [j.stage for j in calls[:3]] == ["translation", "translation", "editing"]
    assert calls[0].source == calls[1].source and calls[1].draft is None
    assert "一定能" not in calls[2].draft


def test_v22_historical_corruption_blocks_export_without_rewriting(service):
    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    row = svc.repository.chunks(rid)[0]
    path = svc.run_dir(rid) / row["output_path"]
    bad = "Марион " + "一定能" * 78
    path.write_text(bad, encoding="utf-8")
    svc.repository.set_chunk(
        rid,
        row["chapter"],
        row["chunk_id"],
        "completed",
        json.loads(row["metadata"]),
        row["output_path"],
        sha256(bad),
    )
    assert svc.audit(rid)["status"] == "blocked"
    with pytest.raises(ValueError, match="quality blocked"):
        svc.build_epub(rid)
    assert path.read_text("utf-8") == bad
    assert not list(svc.config.novel.books_dir.rglob("*.epub"))


def test_strict_short_mixed_script_leak_blocked_legacy_compatible():
    legacy = validate("原文。", SHORT_LEAK, "stop", ValidationConfig())
    assert legacy.level != "FAIL"
    assert "foreign_script_leak" not in legacy.issues
    assert "mixed_script_token" not in legacy.issues
    strict = validate("原文。", SHORT_LEAK, "stop", strict_config())
    assert strict.level == "FAIL"
    assert {"foreign_script_leak", "mixed_script_token"} <= set(strict.issues)


def test_strict_clean_russian_dialogue_and_punctuation_only_pass():
    clean = "— Привет, — сказал он.\n— «Название» здесь, — ответила она."
    result = validate("原文。", clean, "stop", strict_config())
    assert "foreign_script_leak" not in result.issues
    assert "mixed_script_token" not in result.issues
    punctuation = "Он сказал: 「」、、。・ — и замолчал, глядя в окно на закат."
    assert foreign_script_count(punctuation) == 0
    assert "foreign_script_leak" not in validate(
        "原文。", punctuation, "stop", strict_config()
    ).issues


def test_strict_counts_halfwidth_kana_and_supplementary_ideographs_not_punctuation():
    assert foreign_script_count("ｱｲｳ") == 3
    assert foreign_script_count("\U00020000\U00020001") == 2
    assert foreign_script_count("ア・イ") == 2
    assert "foreign_script_leak" in validate(
        "原文。", "Он сказал ｱｲｳ и ушёл.", "stop", strict_config()
    ).issues


def test_strict_mixed_token_detection():
    assert mixed_script_tokens("Он вошёл в данジョンa.") == ["данジョンa."]
    assert mixed_script_tokens("Чистый русский текст.") == []


def test_strict_detached_dash_line_and_dangling_comma():
    output = "Он посмотрел на неё,\n—\nи вышел."
    issues = validate("原文。", output, "stop", strict_config()).issues
    assert "runaway_dash_line" in issues
    assert "dangling_dialogue_comma" in issues
    clean = "Он посмотрел на неё,\n— Полная реплика.\nи вышел."
    clean_issues = validate("原文。", clean, "stop", strict_config()).issues
    assert "runaway_dash_line" not in clean_issues
    assert "dangling_dialogue_comma" not in clean_issues
    legacy_issues = validate("原文。", output, "stop", ValidationConfig()).issues
    assert "runaway_dash_line" not in legacy_issues


def test_strict_guillemets_and_nested_quotes_not_malformed():
    output = (
        "— Он сказал «слово» и замолчал, — произнесла она.\n"
        "— Здесь «название» и „вложенная речь“ допустимы."
    )
    result = validate("原文。", output, "stop", strict_config())
    assert "runaway_dash_line" not in result.issues
    assert "dangling_dialogue_comma" not in result.issues


def test_strict_policy_rejects_cached_malformed_candidate_and_regenerates(service):
    svc, _ = service
    svc.stages.cache = TranslationCache(svc.root / "cache")
    svc.config.translation.retries = 1

    class LeakThenClean:
        def __init__(self):
            self.calls = 0

        def run(self, job):
            self.calls += 1
            if self.calls == 1:
                return RunnerResult(SHORT_LEAK, model="fake-model", finish_reason="stop")
            return RunnerResult(CLEAN_RUSSIAN * 4, model="fake-model", finish_reason="stop")

    runner = LeakThenClean()
    svc.runner_factory = lambda _: runner
    relaxed = svc.create_run(runner="test", chapters=[1])
    assert svc.execute(relaxed)["status"] == "completed"

    svc.config.validation.max_foreign_script_chars = 0
    svc.config.validation.reject_malformed_dialogue = True
    strict = svc.create_run(runner="test", chapters=[1])
    assert strict != relaxed
    assert svc.execute(strict)["status"] == "completed"
    row = svc.repository.chunks(strict)[0]
    text = (svc.run_dir(strict) / row["output_path"]).read_text("utf-8")
    assert "ジョン" not in text
    assert runner.calls >= 2


def test_strict_ideographic_content_marks_are_foreign_script_leaks():
    for mark in ("々", "〆", "〇", "〻"):
        assert foreign_script_count(f"Он снова {mark}.") == 1
        result = validate("原文。", f"Он снова повторил {mark}.", "stop", strict_config())
        assert result.level == "FAIL"
        assert "foreign_script_leak" in result.issues


def test_strict_cjk_punctuation_alone_is_not_a_foreign_script_leak():
    punctuation = "«Текст»・\nРусский текст。\n— «Русская реплика»."
    assert foreign_script_count("、。「」『』・") == 0
    assert foreign_script_count(punctuation) == 0
    result = validate("原文。" * 6, punctuation, "stop", strict_config())
    assert "foreign_script_leak" not in result.issues
    assert "mixed_script_token" not in result.issues
    assert "runaway_dash_line" not in result.issues
    assert "dangling_dialogue_comma" not in result.issues
    assert result.level == "PASS"


def corruption_config():
    return ValidationConfig(
        max_foreign_script_chars=0,
        reject_malformed_dialogue=True,
        reject_service_artifacts=True,
        reject_segment_collapse=True,
        reject_missing_cyrillic=True,
    )


def test_corruption_service_artifact_gate_is_opt_in():
    output = 'Он вошёл и увидел "},{" вокруг.'
    assert "service_artifact" not in validate("原文。", output, "stop", ValidationConfig()).issues
    strict = validate("原文。", output, "stop", corruption_config())
    assert strict.level == "FAIL"
    assert "service_artifact" in strict.issues


def test_corruption_segment_checks():
    source = (
        "これはとても長い日本語の文章で、四十文字をはるかに超えるように意図的に書かれています。\n"
        "彼は本を読んだ。"
    )
    ids = [s["id"] for s in source_segments(source)]
    collapse = [
        {"id": ids[0], "text": "Коротко."},
        {"id": ids[1], "text": "Он читал книгу."},
    ]
    assert "segment_collapse" in validate_segments(source, collapse, corruption_config()).issues
    artifact = [
        {"id": ids[0], "text": "Он читал.},{"},
        {"id": ids[1], "text": "Он читал книгу."},
    ]
    assert "service_artifact" in validate_segments(source, artifact, corruption_config()).issues


def test_corruption_segment_id_errors():
    source = "一行目です。\n二行目です。\n三行目です。"
    ids = [s["id"] for s in source_segments(source)]
    ok = [{"id": key, "text": "Строка."} for key in ids]
    assert validate_segments(source, ok[:2], corruption_config()).issues == ["segment_id_mismatch"]
    duplicate = [
        {"id": ids[0], "text": "Строка."},
        {"id": ids[0], "text": "Строка."},
        {"id": ids[2], "text": "Строка."},
    ]
    assert validate_segments(source, duplicate, corruption_config()).issues == [
        "segment_id_mismatch"
    ]
    reordered = [ok[1], ok[0], ok[2]]
    assert validate_segments(source, reordered, corruption_config()).issues == [
        "segment_id_mismatch"
    ]
    assert validate_segments(source, [{"id": ids[0]}, *ok[1:]], corruption_config()).issues == [
        "invalid_segment_shape"
    ]
    assert validate_segments(
        source, [{"id": ids[0], "text": None}, *ok[1:]], corruption_config()
    ).issues == ["invalid_segment_shape"]


def test_corruption_missing_cyrillic_ignores_latin_and_numbers():
    source = "グーグルクロームの設定を開いた。"
    ids = [s["id"] for s in source_segments(source)]
    for text in ("Google Chrome", "1234567890"):
        assert (
            "missing_cyrillic"
            not in validate_segments(
                source, [{"id": ids[0], "text": text}], corruption_config()
            ).issues
        )
    assert (
        "missing_cyrillic"
        in validate_segments(
            source, [{"id": ids[0], "text": "……"}], corruption_config()
        ).issues
    )


def test_corruption_service_artifact_ignores_plain_prose():
    prose = [
        "В отчёте было несколько issues: сбой и текст: продолжение.",
        "Он сказал про id: это шутка.",
    ]
    for text in prose:
        assert "service_artifact" not in validate("原文。", text, "stop", corruption_config()).issues
    assert "service_artifact" in validate(
        "原文。", 'Он ответил: "segments": []', "stop", corruption_config()
    ).issues


def test_corruption_clean_segments_pass():
    source = "「おはよう」\n彼は本を読んだ。"
    ids = [s["id"] for s in source_segments(source)]
    translations = [
        {"id": ids[0], "text": "— Доброе утро."},
        {"id": ids[1], "text": "Он читал книгу."},
    ]
    assert validate_segments(source, translations, corruption_config()).level == "PASS"


def test_corruption_checks_off_by_default():
    source = "これはとても長い日本語の文章で、四十文字をはるかに超えるように意図的に書かれています。"
    ids = [s["id"] for s in source_segments(source)]
    translations = [{"id": ids[0], "text": "Коротко."}]
    assert validate_segments(source, translations, ValidationConfig()).level == "PASS"


def test_documented_false_negative_pseudorussian_passes_deterministic_gate():
    # Measured 2026-09-15 from run 20260914T203716-99fc6a3f3fc1 chunk
    # 0003-bc18aebaf12babaa: the real chapter-63 collapse passes every deterministic
    # segment check, so it remains a documented false-negative handled by selective
    # retry (0036) and Review V2.
    source = (
        "\u3000ボーパルバニーたちは嬉しそうにきゅうきゅうと鳴く。\n"
        "「おーい、シャーロット。こっちに来て……」\n"
        "\u3000その場でぴしりと凍り付いてしまった。\n"
        "\u3000アレンはひとり、呆然とするしかなかった。"
    )
    captured = [
        "Крепы занимались хотельциво.",
        "— Че, Шарлотта. Пойде.”",
        "Она мжелайно замерзла.",
        "В целом, Виль.",
    ]
    ids = [s["id"] for s in source_segments(source)]
    translations = [{"id": key, "text": text} for key, text in zip(ids, captured, strict=True)]
    assert validate_segments(source, translations, corruption_config()).issues == []



def test_quality_retry_recovers_and_records_selection(service):
    svc, _ = service
    svc.config.validation.max_foreign_script_chars = 0
    svc.config.translation.retries = 1
    calls = []

    class CorruptThenClean:
        def run(self, job):
            calls.append(job)
            if len(calls) == 1:
                return RunnerResult(SHORT_LEAK, model="fake-model", finish_reason="stop")
            return RunnerResult(CLEAN_RUSSIAN * 4, model="fake-model", finish_reason="stop")

    svc.runner_factory = lambda _: CorruptThenClean()
    rid = svc.create_run(runner="test", chapters=[1])
    assert svc.execute(rid)["status"] == "completed"
    chapter = svc.repository.run(rid)["manifest"]["chapters"][0]
    chunk_id = chapter["chunks"][0]["id"]
    attempts = svc.repository.attempts(rid, chapter["number"], chunk_id, "translation")
    assert attempts[0]["failure_kind"] == "validation_failed"
    assert attempts[0]["validation"]["level"] == "FAIL"
    assert attempts[-1]["selection_reason"] == "first_passing_attempt"
    assert attempts[-1]["failed_prior"] == 1


def test_quality_retry_budget_is_durable_across_restarts(service):
    svc, _ = service
    svc.config.validation.max_foreign_script_chars = 0
    svc.config.translation.retries = 0
    calls = []

    class AlwaysCorrupt:
        def run(self, job):
            calls.append(job)
            return RunnerResult(SHORT_LEAK, model="fake-model", finish_reason="stop")

    svc.runner_factory = lambda _: AlwaysCorrupt()
    rid = svc.create_run(runner="test", chapters=[1])
    svc.execute(rid)
    assert len(calls) == 1
    svc.execute(rid)
    assert len(calls) == 2
    svc.execute(rid)
    assert len(calls) == 2  # durable ceiling: no further model calls
    chapter = svc.repository.run(rid)["manifest"]["chapters"][0]
    chunk_id = chapter["chunks"][0]["id"]
    row = next(
        r
        for r in svc.repository.chunks(rid)
        if r["chapter"] == chapter["number"] and r["chunk_id"] == chunk_id
    )
    assert json.loads(row["metadata"])["recovery_exhausted"] is True
    assert svc.metrics(rid)["stages"][0]["recovery_exhausted"] == 1


def test_passing_first_attempt_not_retried(service):
    svc, _ = service
    svc.config.translation.retries = 2
    rid = svc.create_run(runner="test", chapters=[1])
    svc.execute(rid)
    chapter = svc.repository.run(rid)["manifest"]["chapters"][0]
    for chunk in chapter["chunks"]:
        attempts = svc.repository.attempts(rid, chapter["number"], chunk["id"], "translation")
        assert len(attempts) == 1
        assert attempts[0]["selection_reason"] == "first_passing_attempt"


def test_metrics_surface_validation_failures(service):
    svc, _ = service
    svc.config.validation.max_foreign_script_chars = 0
    svc.config.translation.retries = 0

    class AlwaysCorrupt:
        def run(self, job):
            return RunnerResult(SHORT_LEAK, model="fake-model", finish_reason="stop")

    svc.runner_factory = lambda _: AlwaysCorrupt()
    rid = svc.create_run(runner="test", chapters=[1])
    svc.execute(rid)
    stage = svc.metrics(rid)["stages"][0]
    assert stage["validation_failures"] == 1
    assert stage["statuses"].get("failed") == 1


def test_epub_freshness_tracks_human_final_layer(service):
    translation, _ = service
    run_id = translation.create_run(runner="test", chapters=[1])
    translation.execute(run_id)
    translation.build_epub(run_id)
    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is True

    row = translation.repository.chunks(run_id)[0]
    chapter, chunk_id, machine_sha = row["chapter"], row["chunk_id"], row["output_sha256"]
    first = translation.set_final(
        run_id, chapter, chunk_id, action="edit", actor="ui", text="Правка",
        expected_machine_sha256=machine_sha,
    )
    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is False

    translation.build_epub(run_id)
    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is True

    replaced = translation.set_final(
        run_id, chapter, chunk_id, action="edit", actor="ui", text="Правка-2",
        expected_machine_sha256=machine_sha, expected_final_sha256=first["final_sha256"],
    )
    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is False

    translation.set_final(
        run_id, chapter, chunk_id, action="revert", actor="ui",
        expected_machine_sha256=machine_sha, expected_final_sha256=replaced["final_sha256"],
    )
    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is False

    translation.build_epub(run_id)
    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is True


def test_legacy_book_record_does_not_claim_final_layer_freshness(service):
    from novel_harness.artifacts import canonical

    translation, _ = service
    run_id = translation.create_run(runner="test", chapters=[1])
    translation.execute(run_id)
    translation.build_epub(run_id)

    book_path = translation.run_dir(run_id) / "book.json"
    book = json.loads(book_path.read_text("utf-8"))
    rows = translation.repository.chunks(run_id)
    # Pre-human-final fingerprint format: machine hashes only.
    legacy = sha256(
        canonical([(r["chapter"], r["chunk_id"], r["output_sha256"]) for r in rows])
    )
    book["inputs_sha256"] = legacy
    book_path.write_text(json.dumps(book), "utf-8")

    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is False


def test_stale_final_invalidates_epub_freshness(service):
    translation, _ = service
    run_id = translation.create_run(runner="test", chapters=[1])
    translation.execute(run_id)
    row = translation.repository.chunks(run_id)[0]
    chapter, chunk_id, machine_sha = row["chapter"], row["chunk_id"], row["output_sha256"]
    translation.set_final(
        run_id, chapter, chunk_id, action="accept", actor="ui",
        expected_machine_sha256=machine_sha,
    )
    translation.build_epub(run_id)
    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is True

    # Simulate a later machine regeneration that stales the accepted final.
    path = translation.run_dir(run_id) / row["output_path"]
    revised = "Обновлённый машинный перевод"
    path.write_text(revised, "utf-8")
    translation.repository.set_chunk(
        run_id, chapter, chunk_id, "completed",
        json.loads(row["metadata"]), row["output_path"], sha256(revised),
    )

    assert translation.status(run_id)["quality"]["epub_matches_outputs"] is False


def test_machine_only_run_rebuild_keeps_freshness(service):
    translation, _ = service
    run_id = translation.create_run(runner="test", chapters=[1])
    translation.execute(run_id)
    translation.build_epub(run_id)
    for _ in range(2):
        assert translation.status(run_id)["quality"]["epub_matches_outputs"] is True


@pytest.mark.parametrize("action", ["edit", "revert"])
def test_export_receipt_uses_the_final_snapshot_actually_rendered(service, monkeypatch, action):
    import zipfile

    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    row = svc.repository.chunks(rid)[0]
    ref = (rid, row["chapter"], row["chunk_id"])
    old_text = "До одновременной правки был именно этот русский текст. " * 4
    new_text = "После одновременной правки появился новый русский текст. " * 4
    entry = svc.set_final(*ref, action="edit", actor="user", text=old_text,
                          expected_machine_sha256=row["output_sha256"])
    inspect = svc.reporter.inspect_chapter
    changed = False

    def edit_after_inspection(*args, **kwargs):
        nonlocal changed
        chapter = inspect(*args, **kwargs)
        if not changed:
            changed = True
            svc.set_final(*ref, action=action, actor="user",
                          text=new_text if action == "edit" else None,
                          expected_machine_sha256=row["output_sha256"],
                          expected_final_sha256=entry["final_sha256"])
        return chapter

    monkeypatch.setattr(svc.reporter, "inspect_chapter", edit_after_inspection)
    path = svc.build_epub(rid)
    with zipfile.ZipFile(path) as archive:
        contents = "\n".join(archive.read(name).decode("utf-8", "ignore")
                             for name in archive.namelist())
    assert "До одновременной правки" in contents
    assert not svc.status(rid)["quality"]["epub_matches_outputs"]
    receipt = json.loads((svc.run_dir(rid) / "book.json").read_text("utf-8"))
    assert receipt["final_layer"][0]["final_sha256"] == entry["final_sha256"]
    svc.build_epub(rid)
    assert svc.status(rid)["quality"]["epub_matches_outputs"]
