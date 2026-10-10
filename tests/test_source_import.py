from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from conftest import FakeRunner
from typer.testing import CliRunner

from novel_harness.cli import app
from novel_harness.config import load_config
from novel_harness.service import TranslationService
from novel_harness.source_import import parse_chapter_numbers


def _output(directory: Path, *, numbers=(1, 2), body="下載本文") -> dict[str, bytes]:
    directory.mkdir(parents=True)
    files = {}
    for number in numbers:
        name = f"{number:04d}_episode-{number}.txt"
        content = f"作品名 第{number}話 第{number}話題\n\n{body}{number}\n".encode()
        (directory / name).write_bytes(content)
        files[name] = content
    return files


def _service(project) -> TranslationService:
    return TranslationService(load_config(project), lambda _name: FakeRunner())


def _import(service, source: Path, *, expected=(1, 2), ncode="n5727fr"):
    return service.import_narou_source(
        source,
        ncode=ncode,
        title="作品名",
        expected_chapters=expected,
    )


def test_import_copies_exact_bytes_publishes_manifest_and_rescans_library(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    originals = _output(source)
    service = _service(project)

    result = _import(service, source)

    destination = project.parent / "original" / "n5727fr"
    assert result["status"] == "imported"
    assert result["destination"] == str(destination)
    assert result["source_identity"] == {"type": "narou", "id": "n5727fr"}
    assert result["chapter_numbers"] == [1, 2]
    assert {path.name: path.read_bytes() for path in source.glob("*.txt")} == originals
    assert {path.name: path.read_bytes() for path in destination.glob("*.txt")} == originals
    assert json.loads((destination / "novel.json").read_text("utf-8")) == {
        "version": 1,
        "source": {"type": "narou", "id": "n5727fr"},
        "title": "作品名",
        "language": "ja",
        "format": "txt-chapters",
        "chapter_pattern": "*.txt",
    }
    assert json.loads((destination / "source-import.json").read_text("utf-8")) == {
        "version": 1,
        "provider": "narou",
        "source_id": "n5727fr",
        "expected_chapters": [1, 2],
        "inventory_sha256": result["inventory_sha256"],
    }
    book = service.library_show(result["novel_id"])
    assert book["status"] == "ready"
    assert book["source_path"] == str(destination)
    assert book["chapter_count"] == 2
    assert list((project.parent / "original").glob(".novel-import-*")) == []
    with sqlite3.connect(service.repository.path) as db:
        indexed = "".join(str(row) for row in db.execute("SELECT * FROM library_sources"))
    assert "下載本文" not in indexed
    with service.repository.connection() as db:
        db.execute("DELETE FROM library_sources")
    rebuilt = service.rescan_library()["novels"]
    assert [(row["novel_id"], row["status"]) for row in rebuilt] == [
        (result["novel_id"], "ready")
    ]


def test_repeated_import_is_idempotent_and_keeps_existing_library_record(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    _output(source)
    service = _service(project)

    first = _import(service, source)
    before = {
        path.relative_to(first["destination"]).as_posix(): path.read_bytes()
        for path in Path(first["destination"]).iterdir()
        if path.is_file()
    }
    second = _import(service, source)

    assert second["status"] == "already_present"
    assert second["novel_id"] == first["novel_id"]
    assert second["destination"] == first["destination"]
    assert {
        path.relative_to(first["destination"]).as_posix(): path.read_bytes()
        for path in Path(first["destination"]).iterdir()
        if path.is_file()
    } == before


def test_import_uses_first_configured_library_root(project):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["novel"]["library_roots"] = ["./imported-books", "./original"]
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    source = project.parent / "downloader" / "output" / "n5727fr"
    _output(source)
    service = _service(project)

    result = _import(service, source)

    assert result["destination"] == str(project.parent / "imported-books" / "n5727fr")
    assert service.library_show(result["novel_id"])["status"] == "ready"


def test_interrupted_staging_directory_is_not_visible_in_library(project):
    staging = project.parent / "original" / ".novel-import-interrupted"
    _output(staging)
    (staging / "novel.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source": {"type": "narou", "id": "n5727fr"},
                "title": "作品名",
                "language": "ja",
                "format": "txt-chapters",
                "chapter_pattern": "*.txt",
            }
        ),
        "utf-8",
    )

    service = _service(project)

    assert service.library_novels()["novels"] == []


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [((1, 2), (1, 2, 3)), ((1, 2, 3), (1, 2))],
)
def test_incomplete_or_unexpected_chapter_inventory_publishes_nothing(
    project, numbers, expected
):
    source = project.parent / "downloader" / "output" / "n5727fr"
    source_files = _output(source, numbers=numbers)
    service = _service(project)

    with pytest.raises(ValueError, match="incomplete or unexpected"):
        _import(service, source, expected=expected)

    assert not (project.parent / "original" / "n5727fr").exists()
    assert {path.name: path.read_bytes() for path in source.glob("*.txt")} == source_files
    assert service.library_novels()["novels"] == []


def test_duplicate_episode_numbers_are_rejected_before_publication(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    source.mkdir(parents=True)
    for name in ("0001_first.txt", "0001_duplicate.txt"):
        (source / name).write_text("作品名 第1話 題名\n\n本文\n", "utf-8")
    service = _service(project)

    with pytest.raises(ValueError, match="Duplicate chapter number"):
        _import(service, source, expected=(1,))

    assert not (project.parent / "original" / "n5727fr").exists()


def test_invalid_text_symlink_and_unsafe_identity_fail_before_publication(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    source.mkdir(parents=True)
    (source / "0001_bad.txt").write_bytes(b"\xff")
    service = _service(project)
    with pytest.raises(ValueError, match="UTF-8"):
        _import(service, source, expected=(1,))
    assert not (project.parent / "original" / "n5727fr").exists()

    (source / "0001_bad.txt").write_text("第1話\n\n本文", "utf-8")
    with pytest.raises(ValueError, match="letters and digits"):
        _import(service, source, expected=(1,), ncode="../escape")
    assert not (project.parent / "escape").exists()


def test_truncated_episode_fails_even_when_expected_numbers_match(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    _output(source)
    last_chapter = source / "0002_episode-2.txt"
    last_chapter.write_text("作品名 第2話 第2話題\n\n途中で切れた本文", "utf-8")
    service = _service(project)

    with pytest.raises(ValueError, match="Incomplete Narou episode"):
        _import(service, source)

    assert not (project.parent / "original" / "n5727fr").exists()


def test_symlinked_chapter_is_rejected_without_reading_through_it(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    _output(source, numbers=(1,))
    external = project.parent / "external.txt"
    external.write_text("作品名 第2話 第2話題\n\n外部本文\n", "utf-8")
    linked = source / "0002_episode-2.txt"
    try:
        linked.symlink_to(external)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable: {error}")
    service = _service(project)

    with pytest.raises(ValueError, match="regular file"):
        _import(service, source, expected=(1, 2))

    assert not (project.parent / "original" / "n5727fr").exists()
    assert external.read_text("utf-8").startswith("作品名 第2話")


def test_existing_destination_is_never_overwritten(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    source_files = _output(source)
    destination = project.parent / "original" / "n5727fr"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep this collision", "utf-8")
    service = _service(project)

    with pytest.raises(ValueError, match="destination already exists"):
        _import(service, source)

    assert (destination / "keep.txt").read_text("utf-8") == "keep this collision"
    assert {path.name: path.read_bytes() for path in source.glob("*.txt")} == source_files


def test_source_inside_library_root_is_rejected_without_mutating_it(project):
    source = project.parent / "original" / "n5727fr"
    source_files = _output(source)
    service = _service(project)

    with pytest.raises(ValueError, match="already inside the configured Library root"):
        _import(service, source)

    assert {path.name: path.read_bytes() for path in source.glob("*.txt")} == source_files
    assert not (source / "novel.json").exists()


def test_duplicate_provider_identity_with_different_content_fails_closed(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    source_files = _output(source, body="new body")
    existing = project.parent / "original" / "other-name"
    _output(existing, body="old body")
    (existing / "novel.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source": {"type": "narou", "id": "n5727fr"},
                "title": "作品名",
                "language": "ja",
                "format": "txt-chapters",
                "chapter_pattern": "*.txt",
            }
        ),
        "utf-8",
    )
    service = _service(project)
    existing_before = {
        path.name: path.read_bytes() for path in existing.iterdir() if path.is_file()
    }

    with pytest.raises(ValueError, match="already exists with different"):
        _import(service, source)

    assert {path.name: path.read_bytes() for path in existing.iterdir()} == existing_before
    assert not (project.parent / "original" / "n5727fr").exists()
    assert {path.name: path.read_bytes() for path in source.glob("*.txt")} == source_files


def test_chapter_list_parser_requires_exact_unique_positive_values():
    assert parse_chapter_numbers("1, 3-5, 9") == (1, 3, 4, 5, 9)
    for malformed in ("", "1,,2", "0", "3-1", "1-3,3"):
        with pytest.raises(ValueError):
            parse_chapter_numbers(malformed)


def test_cli_import_uses_same_service_adapter_and_returns_json(project):
    source = project.parent / "downloader" / "output" / "n5727fr"
    _output(source)
    result = CliRunner().invoke(
        app,
        [
            "library",
            "import-narou",
            "--source",
            str(source),
            "--ncode",
            "n5727fr",
            "--title",
            "作品名",
            "--expected-chapters",
            "1-2",
            "--config",
            str(project),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "imported"
    assert payload["source_identity"] == {"type": "narou", "id": "n5727fr"}
