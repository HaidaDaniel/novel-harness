import copy
import json
import zipfile
from xml.etree import ElementTree as ET

import pytest
import yaml

from novel_harness.artifacts import sha256
from novel_harness.book_metadata import BookMetadata, load_book_metadata
from novel_harness.config import load_config

NOVEL = {"source_title": "原作の名前", "title": "Название произведения", "author": None}


def write_metadata(path, value=NOVEL):
    path.write_text(yaml.safe_dump(value, allow_unicode=True), "utf-8")
    return path


def test_v31_shared_title_is_frozen_for_runners_resume_and_continuation(service, tmp_path):
    svc, fake = service
    path = write_metadata(tmp_path / "book.yaml")
    svc.config.novel.metadata_path = path
    svc.config.runners["other"] = copy.deepcopy(svc.config.runners["test"])
    first = svc.create_run(runner="test")
    second = svc.create_run(runner="other")
    assert not fake.calls
    assert svc.repository.run(first)["manifest"]["novel"] == NOVEL
    assert svc.repository.run(second)["manifest"]["novel"] == NOVEL
    svc.execute(first, max_chunks=1)
    write_metadata(path, {**NOVEL, "title": "Изменённое название"})
    svc.resume(first)
    assert all(j.context["book_title"]["russian"] == NOVEL["title"] for j in fake.calls)
    assert all("do not translate that title again" in j.instructions for j in fake.calls)
    assert len(fake.calls) == svc.status(first)["total_chunks"]
    child = svc.continue_run(first, runner="other")
    assert svc.repository.run(child)["manifest"]["novel"] == NOVEL
    epub = svc.build_epub(first)
    with zipfile.ZipFile(epub) as archive:
        root = ET.fromstring(archive.read("EPUB/content.opf"))
        assert root.find(".//{http://purl.org/dc/elements/1.1/}title").text == NOVEL["title"]
    manifest = svc.repository.run(first)["manifest"]
    manifest["novel"]["title"] = "Подмена"
    with svc.repository.connection() as db:
        db.execute("UPDATE runs SET manifest=? WHERE id=?", (json.dumps(manifest), first))
    with pytest.raises(ValueError, match="metadata hash"):
        svc.resume(first)


def test_v31_explicit_legacy_export_preserves_originals_and_reuses_title(service, tmp_path):
    svc, _ = service
    svc.config.novel.title = NOVEL["source_title"]
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    old_epub = svc.build_epub(rid)
    old_hash = sha256(old_epub.read_bytes())
    old_record = (svc.run_dir(rid) / "book.json").read_bytes()
    old_manifest = svc.repository.run(rid)["manifest"]
    path = write_metadata(tmp_path / "book.yaml")
    new_epub = svc.build_epub(rid, metadata_path=path)
    assert new_epub != old_epub and sha256(old_epub.read_bytes()) == old_hash
    assert svc.repository.run(rid)["manifest"] == old_manifest
    assert (
        svc.run_dir(rid) / "book-history" / f"{sha256(old_record)}.json"
    ).read_bytes() == old_record
    path.unlink()
    assert svc.build_epub(rid) == new_epub
    write_metadata(path, {**NOVEL, "source_title": "別の作品"})
    with pytest.raises(ValueError, match="another source"):
        svc.build_epub(rid, metadata_path=path)


@pytest.mark.parametrize(
    "title", ["日本語", "???", "Название\ufffd", "Название\nкниги", " Название"]
)
def test_v31_bad_or_untranslated_title_rejected(title):
    with pytest.raises(ValueError):
        BookMetadata.model_validate({**NOVEL, "title": title})


def test_v31_metadata_path_resolution_and_duplicate_keys(project):
    path = write_metadata(project.parent / "book.yaml")
    data = yaml.safe_load(project.read_text("utf-8"))
    data["novel"]["metadata_path"] = "book.yaml"
    project.write_text(yaml.safe_dump(data), "utf-8")
    assert load_config(project).novel.metadata_path == path
    path.write_text("title: Название\ntitle: Другое\nsource_title: 原作", "utf-8")
    with pytest.raises(ValueError):
        load_book_metadata(path)
