"""Clean-room Novel Library fixtures; no downloader or model endpoint."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
import yaml
from conftest import FakeRunner

from novel_harness.config import load_config
from novel_harness.service import ConflictError, TranslationService


def _book(root: Path, name: str, numbers: tuple[int, ...] = (1, 2)) -> Path:
    path = root / name
    path.mkdir()
    for number in numbers:
        (path / f"{number:04d}_chapter.txt").write_text(
            f"第{number}章 {name}\n\n" + f"{name}の物語が続く。" * 18,
            "utf-8",
        )
    return path


def _service(project) -> TranslationService:
    return TranslationService(load_config(project), lambda name: FakeRunner())


def _by_title(service: TranslationService) -> dict[str, dict]:
    return {item["title"]: item for item in service.library_novels()["novels"]}


def test_multiple_books_read_only_listing_idempotent_rescan_and_no_body_in_index(project):
    root = project.parent / "original"
    first = _book(root, "alpha")
    _book(root, "beta")
    service = _service(project)
    cards = _by_title(service)
    assert set(cards) == {"alpha", "beta"}
    assert all(item["status"] == "ready" and item["chapter_count"] == 2 for item in cards.values())
    before_rows = service.repository.library_rows()
    assert before_rows == []
    before_source = {path: path.read_bytes() for path in root.rglob("*.txt")}
    alpha_id = cards["alpha"]["novel_id"]
    assert service.library_show(alpha_id)["source_path"] == str(first)
    assert service.repository.library_rows() == before_rows
    assert service.rescan_library()["novels"] == service.library_novels()["novels"]
    indexed_rows = service.repository.library_rows()
    service.rescan_library()
    assert service.repository.library_rows() == indexed_rows
    assert {path: path.read_bytes() for path in root.rglob("*.txt")} == before_source
    with sqlite3.connect(service.repository.path) as db:
        stored = "".join(str(row) for row in db.execute("SELECT * FROM library_sources"))
    assert "の物語が続く" not in stored


def test_legacy_id_refresh_move_index_loss_and_missing_source(project):
    root = project.parent / "original"
    book = _book(root, "alpha")
    service = _service(project)
    service.rescan_library()
    original_id = _by_title(service)["alpha"]["novel_id"]
    (book / "0003_chapter.txt").write_text("第三章\n\n新しい章。" * 5, "utf-8")
    service.rescan_library()
    assert _by_title(service)["alpha"]["novel_id"] == original_id
    moved = book.rename(root / "renamed")
    service.rescan_library()
    assert service.library_show(original_id)["source_path"] == str(moved)
    with service.repository.connection() as db:
        db.execute("DELETE FROM library_sources")
    service.rescan_library()
    unchanged_id = _by_title(service)["renamed"]["novel_id"]
    assert unchanged_id != original_id  # The earlier same-path refresh cannot be recovered.
    with service.repository.connection() as db:
        db.execute("DELETE FROM library_sources")
    (moved / "0004_chapter.txt").write_text("第四章\n\n変化。" * 5, "utf-8")
    service.rescan_library()
    assert _by_title(service)["renamed"]["novel_id"] != unchanged_id
    missing_id = _by_title(service)["renamed"]["novel_id"]
    shutil.rmtree(moved)
    service.rescan_library()
    assert service.library_show(missing_id)["status"] == "missing"


def test_ambiguous_legacy_move_never_transfers_an_id_to_first_copy(project):
    root = project.parent / "original"
    book = _book(root, "alpha")
    service = _service(project)
    service.rescan_library()
    original_id = _by_title(service)["alpha"]["novel_id"]

    # The same-path refresh preserves the old indexed identity while changing inventory.
    (book / "0003_chapter.txt").write_text("第三章\n\n新しい章。" * 5, "utf-8")
    service.rescan_library()
    assert _by_title(service)["alpha"]["novel_id"] == original_id

    first_copy = root / "copy-one"
    second_copy = root / "copy-two"
    shutil.copytree(book, first_copy)
    shutil.copytree(book, second_copy)
    shutil.rmtree(book)
    cards = service.rescan_library()["novels"]

    copies = [row for row in cards if row["source_path"] in {str(first_copy), str(second_copy)}]
    missing = next(row for row in cards if row["novel_id"] == original_id)
    assert len(copies) == 2
    assert all(row["status"] == "conflict" for row in copies)
    assert all("Ambiguous legacy source move" in row["error"] for row in copies)
    assert all(row["novel_id"] != original_id for row in copies)
    assert missing["status"] == "missing"


def test_strict_manifest_conflict_and_bad_candidate_isolation(project):
    root = project.parent / "original"
    one = _book(root, "one")
    two = _book(root, "two")
    healthy = _book(root, "healthy")
    broken = _book(root, "broken")
    ignored = root / "ignored" / "deep"
    ignored.mkdir(parents=True)
    (ignored / "novel.json").write_text("{\"version\": 1}", "utf-8")
    for path in (one, two):
        (path / "novel.json").write_text(json.dumps({
            "version": 1,
            "title": path.name,
            "source": {"type": "narou", "id": "n123"},
        }), "utf-8")
    (broken / "novel.json").write_text(json.dumps({
        "version": 1, "id": "bad", "title": "Bad", "unknown": "not allowed"
    }), "utf-8")
    service = _service(project)
    cards = _by_title(service)
    assert cards["healthy"]["status"] == "ready"
    assert cards["broken"]["status"] == "invalid"
    assert cards["one"]["status"] == cards["two"]["status"] == "conflict"
    assert cards["one"]["novel_id"] == cards["two"]["novel_id"]
    with pytest.raises(ConflictError, match="Ambiguous"):
        service.library_show(cards["one"]["novel_id"])
    with pytest.raises((ConflictError, ValueError)):
        service.create_run(runner="test", novel_id=cards["one"]["novel_id"])
    assert all(item["source_path"] != str(ignored) for item in service.library_novels()["novels"])
    assert healthy.is_dir()


def test_manifest_validation_diagnostics_do_not_leak_user_values(project):
    root = project.parent / "original"
    secret = "not-a-real-api-key-47d2"
    schema_error = _book(root, "schema-error")
    (schema_error / "novel.json").write_text(json.dumps({
        "version": 1, "id": "bad", "title": "Bad", "api_key": secret,
    }), "utf-8")
    json_error = _book(root, "json-error")
    (json_error / "novel.json").write_text(
        '{"version": 1, "token": "' + secret,
        "utf-8",
    )

    service = _service(project)
    cards = service.rescan_library()["novels"]
    assert all(item["status"] == "invalid" for item in cards)
    assert secret not in repr(cards)
    with sqlite3.connect(service.repository.path) as db:
        stored = "".join(str(row) for row in db.execute("SELECT * FROM library_sources"))
    assert secret not in stored


def test_configured_roots_keep_direct_source_title(project):
    root = project.parent / "original"
    extra = project.parent / "completed-books"
    extra.mkdir()
    for number in (1, 2):
        (root / f"{number:04d}_source.txt").write_text(
            f"第{number}章\n\nSource text {number}.", "utf-8"
        )
        (extra / f"{number:04d}_extra.txt").write_text(
            f"第{number}章\n\nExtra text {number}.", "utf-8"
        )
    data = yaml.safe_load(project.read_text("utf-8"))
    data["novel"]["title"] = "Configured direct title"
    data["novel"]["library_roots"] = ["./original", "./completed-books"]
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")

    cards = _service(project).library_novels()["novels"]
    direct = next(item for item in cards if item["source_path"] == str(root))
    additional = next(item for item in cards if item["source_path"] == str(extra))
    assert direct["title"] == "Configured direct title"
    assert additional["title"] == "completed-books"


def test_web_library_contract_uses_stable_ids_and_reports_states():
    app_js = Path(__file__).resolve().parents[1] / "src/novel_harness/web/app.js"
    script = app_js.read_text("utf-8")
    assert 'Loading source novels…' in script
    assert 'No source novels found.' in script
    assert 'Library unavailable:' in script
    assert 'data-translate="${escapeHtml(novel.novel_id || "")}"' in script
    assert 'data-novel-open="${escapeHtml(novel.novel_id || "")}"' in script
    assert 'api(`/library/${encodeURIComponent(button.dataset.novelOpen)}`)' in script
    assert 'api("/library/rescan", { method: "POST" })' in script
    assert 'data-continue="${escapeHtml(novel.continue_run_id || "")}"' in script
    assert "Final: ${final.accepted}/${final.total_chunks} accepted" in script
    assert "${final.needs_review} review" in script
    assert "${final.stale} stale" in script
    assert "${final.machine_only} machine-only" in script


def test_library_card_progress_and_final_layer_counts_follow_latest_run(project):
    root = project.parent / "original"
    _book(root, "alpha")
    service = _service(project)
    novel_id = _by_title(service)["alpha"]["novel_id"]
    run_id = service.create_run(runner="test", novel_id=novel_id)
    service.execute(run_id)

    card = next(row for row in service.library_novels()["novels"] if row["novel_id"] == novel_id)
    summary = card["final_layer"]
    chunks = service.repository.chunks(run_id)
    assert card["progress"] == 1
    assert summary == {
        "run_id": run_id,
        "total_chunks": len(chunks),
        "accepted": 0,
        "needs_review": 0,
        "stale": 0,
        "machine_only": len(chunks),
    }
    assert len(chunks) >= 2

    first, second = chunks[:2]
    service.set_final(
        run_id, first["chapter"], first["chunk_id"], action="accept", actor="tester",
        expected_machine_sha256=first["output_sha256"],
    )
    service.set_final(
        run_id, second["chapter"], second["chunk_id"], action="needs_review", actor="tester",
        expected_machine_sha256=second["output_sha256"], reason="Check this translation",
    )
    card = next(row for row in service.library_novels()["novels"] if row["novel_id"] == novel_id)
    assert card["final_layer"]["accepted"] == 1
    assert card["final_layer"]["needs_review"] == 1
    assert card["final_layer"]["machine_only"] == len(chunks) - 2
    assert "final_text" not in card["final_layer"]

    service.repository.set_chunk(
        run_id, first["chapter"], first["chunk_id"], "completed",
        json.loads(first["metadata"] or "{}"), first["output_path"], "0" * 64,
    )
    card = next(row for row in service.library_novels()["novels"] if row["novel_id"] == novel_id)
    assert card["final_layer"]["accepted"] == 0
    assert card["final_layer"]["stale"] == 1
    assert card["final_layer"]["needs_review"] == 1
    assert card["final_layer"]["machine_only"] == len(chunks) - 2


def test_remaining_coverage_unions_complete_chapters_across_runs(project):
    root = project.parent / "original"
    _book(root, "alpha", (1, 2))
    _book(root, "beta", (1, 2))
    service = _service(project)
    cards = _by_title(service)
    alpha_id = cards["alpha"]["novel_id"]

    initial = service.library_remaining()
    assert [item["title"] for item in initial["novels"]] == ["alpha", "beta"]
    assert all(item["translated"] == 0 and item["remaining"] == 2 for item in initial["novels"])

    first_run = service.create_run(runner="test", chapters=[1], novel_id=alpha_id)
    service.execute(first_run)
    partial = next(item for item in service.library_remaining()["novels"] if item["title"] == "alpha")
    assert partial["translated"] == 1
    assert partial["remaining"] == 1
    assert partial["run_id"] == first_run

    second_run = service.create_run(runner="test", chapters=[2], novel_id=alpha_id)
    service.execute(second_run)
    remaining_titles = [item["title"] for item in service.library_remaining()["novels"]]
    assert remaining_titles == ["beta"]


def test_remaining_failed_or_uncommitted_chunk_stays_untranslated(project):
    root = project.parent / "original"
    _book(root, "alpha", (1, 2))
    service = _service(project)
    alpha_id = _by_title(service)["alpha"]["novel_id"]
    run_id = service.create_run(runner="test", chapters=[1], novel_id=alpha_id)
    row = service.repository.chunks(run_id)[0]
    service.repository.set_chunk(run_id, row["chapter"], row["chunk_id"], "failed", {"error": "test"})
    result = service.library_remaining()
    alpha = next(item for item in result["novels"] if item["novel_id"] == alpha_id)
    assert alpha["translated"] == 0
    assert alpha["remaining"] == 2


def test_launch_remaining_preserves_approved_sample_settings_and_skips_sample_chapter(
    project, monkeypatch
):
    root = project.parent / "original"
    _book(root, "alpha")
    service = _service(project)
    alpha_id = _by_title(service)["alpha"]["novel_id"]
    sample = service.create_sample_run(runner="test", chapters=[1], novel_id=alpha_id)
    service.execute(sample)
    service.approve_sample(sample)
    launched = []
    monkeypatch.setattr(
        "novel_harness.launch.launch_run",
        lambda service_arg, config, run_id: launched.append(run_id)
        or {"run_id": run_id, "started": True},
    )

    result = service.launch_remaining_novel(alpha_id, project)

    assert result["run_id"] == launched[0]
    assert result["run_id"] != sample
    manifest = service.repository.run(result["run_id"])["manifest"]
    assert [chapter["number"] for chapter in manifest["chapters"]] == [2]
    assert manifest["sample"]["role"] == "whole-book"
    assert manifest["sample"]["sample_run_id"] == sample


def test_remaining_excludes_invalid_entries_with_reason(project):
    root = project.parent / "original"
    _book(root, "healthy", (1, 2))
    broken = _book(root, "broken", (1, 2))
    (broken / "novel.json").write_text('{"version": 1, "unknown": true}', "utf-8")
    service = _service(project)
    result = service.library_remaining()
    assert [item["title"] for item in result["novels"]] == ["healthy"]
    assert any(item["title"] == "broken" and item["reason"] for item in result["excluded"])


def test_selected_raw_sample_and_whole_book_freeze_source(project, source):
    root = project.parent / "original"
    other = _book(root, "other", (7, 8))
    service = _service(project)
    cards = service.library_novels()["novels"]
    direct = next(item for item in cards if item["source_path"] == str(root))
    selected = next(item for item in cards if item["source_path"] == str(other))
    old_run = service.create_run(runner="test", chapters=[1])
    old_manifest = service.repository.run(old_run)["manifest"]
    assert "source_identity" not in old_manifest
    selected_run = service.create_run(runner="test", novel_id=selected["novel_id"])
    manifest = service.repository.run(selected_run)["manifest"]
    assert manifest["source_dir"] == str(other)
    assert manifest["source_identity"]["novel_id"] == selected["novel_id"]
    assert [chapter["number"] for chapter in manifest["chapters"]] == [7, 8]
    assert service.repository.run(old_run)["manifest"] == old_manifest
    sample = service.create_sample_run(runner="test", novel_id=selected["novel_id"])
    assert service.repository.run(sample)["manifest"]["sample"]["chapters"] == [7]
    assert service.execute(sample)["status"] == "completed"
    service.approve_sample(sample)
    with pytest.raises(ValueError, match="differs"):
        service.create_whole_book_run(sample, novel_id=direct["novel_id"])
    whole = service.create_whole_book_run(sample, novel_id=selected["novel_id"])
    assert service.repository.run(whole)["manifest"]["source_dir"] == str(other)
    assert service.repository.run(old_run)["manifest"]["source_dir"] == str(root)
    assert source.is_file()


def test_legacy_single_direct_source_and_manifest_pattern(project, source):
    nested = project.parent / "original" / "patterned"
    nested.mkdir()
    (nested / "0001_keep.txt").write_text("第一章\n\n本文。" * 8, "utf-8")
    (nested / "0002_skip.txt").write_text("第二章\n\n本文。" * 8, "utf-8")
    (nested / "novel.json").write_text(json.dumps({
        "version": 1, "id": "patterned", "title": "Patterned",
        "chapter_pattern": "*_keep.txt",
    }), "utf-8")
    service = _service(project)
    direct = next(
        item for item in service.library_novels()["novels"]
        if item["source_path"] == str(source.parent)
    )
    assert direct["chapter_count"] == 1
    selected = service.library_show(_by_title(service)["Patterned"]["novel_id"])
    assert selected["chapter_count"] == 1
    selected_run = service.create_run(runner="test", novel_id=selected["novel_id"])
    assert [chapter["number"] for chapter in service.repository.run(selected_run)["manifest"]["chapters"]] == [1]
    run_id = service.create_run(runner="test", novel_id=direct["novel_id"])
    assert service.repository.run(run_id)["manifest"]["source_dir"] == str(source.parent)


def test_direct_source_chapter_pattern_applies_to_library_and_run_selection(project):
    root = project.parent / "original"
    (root / "0001_keep.txt").write_text("第一章\n\n選択する本文。", "utf-8")
    (root / "0001_excluded.txt").write_text("第一章\n\n除外する本文。", "utf-8")
    data = yaml.safe_load(project.read_text("utf-8"))
    data["novel"]["chapter_pattern"] = "*_keep.txt"
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")

    service = _service(project)
    direct = next(
        item for item in service.library_novels()["novels"]
        if item["source_path"] == str(root)
    )
    assert direct["status"] == "ready"
    assert [chapter["filename"] for chapter in direct["chapters"]] == ["0001_keep.txt"]
    selected = service.create_run(runner="test", novel_id=direct["novel_id"])
    legacy = service.create_run(runner="test")
    assert [
        chapter["filename"] for chapter in service.repository.run(selected)["manifest"]["chapters"]
    ] == ["0001_keep.txt"]
    assert [
        chapter["filename"] for chapter in service.repository.run(legacy)["manifest"]["chapters"]
    ] == ["0001_keep.txt"]


def test_index_recreation_preserves_existing_runs(project):
    book = _book(project.parent / "original", "alpha")
    service = _service(project)
    novel_id = _by_title(service)["alpha"]["novel_id"]
    run_id = service.create_run(runner="test", novel_id=novel_id)
    with service.repository.connection() as db:
        db.execute("DROP TABLE library_sources")
    restarted = _service(project)
    assert restarted.library_show(novel_id)["source_path"] == str(book)
    assert restarted.repository.run(run_id)["manifest"]["source_identity"]["novel_id"] == novel_id


def test_configured_roots_and_manifest_identity_survive_inventory_changes(project):
    external = project.parent / "completed-books"
    external.mkdir()
    book = _book(external, "narou")
    (book / "novel.json").write_text(json.dumps({
        "version": 1,
        "id": "local-alias",
        "source": {"type": "narou", "id": "n5727fr"},
        "title": "Source title",
        "language": "ja",
        "format": "txt-chapters",
    }), "utf-8")
    data = yaml.safe_load(project.read_text("utf-8"))
    data["novel"]["library_roots"] = ["./original", "./completed-books"]
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    service = _service(project)
    card = _by_title(service)["Source title"]
    novel_id = card["novel_id"]
    assert card["source_type"] == "narou"
    assert card["source_id"] == "n5727fr"
    assert card["source_path"] == str(book)
    assert not service.repository.library_rows()  # Listing is a read-only projection.
    service.rescan_library()
    (book / "0003_chapter.txt").write_text("第三章\n\n続き。" * 5, "utf-8")
    service.rescan_library()
    assert service.library_show(novel_id)["chapter_count"] == 3
    with service.repository.connection() as db:
        db.execute("DELETE FROM library_sources")
    restarted = _service(project)
    assert restarted.library_show(novel_id)["source_path"] == str(book)


def test_unsafe_manifest_and_changed_move_do_not_merge(project):
    root = project.parent / "original"
    original = _book(root, "legacy")
    unsafe = _book(root, "unsafe")
    (unsafe / "novel.json").write_text(json.dumps({
        "version": 1, "id": "unsafe", "title": "Unsafe",
        "chapter_pattern": "../*.txt",
    }), "utf-8")
    service = _service(project)
    service.rescan_library()
    old_id = _by_title(service)["legacy"]["novel_id"]
    assert _by_title(service)["unsafe"]["status"] == "invalid"
    moved = original.rename(root / "moved")
    (moved / "0003_chapter.txt").write_text("第三章\n\n変更。" * 5, "utf-8")
    service.rescan_library()
    assert _by_title(service)["moved"]["novel_id"] != old_id
    assert service.library_show(old_id)["status"] == "missing"
