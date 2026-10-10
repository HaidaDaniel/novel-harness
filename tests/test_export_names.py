import copy
import io
import json
import re
import zipfile
from xml.etree import ElementTree as ET

import pytest

from novel_harness.artifacts import sha256, write_text
from novel_harness.epub import OPF_NS, render_epub, validate_epub
from novel_harness.export_names import export_identity, export_path, safe_piece, utf16_units

# A 1x1 PNG; the builder only embeds the bytes, it does not re-encode the image.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000154a24f9e0000000049454e44ae426082"
)
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


def _opf_meta(data: bytes, prop: str):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        opf = ET.fromstring(archive.read("EPUB/content.opf"))
    ns = {"opf": OPF_NS}
    for meta in opf.findall("opf:metadata/opf:meta", ns):
        if meta.get("property") == prop:
            return meta.text
    return None


def test_v32_readable_name_uses_translator_not_reviewer(service):
    svc, _ = service
    svc.config.novel.title = "Название произведения"
    rid = svc.create_run(runner="test", editor="test", pipeline="translate-edit")
    manifest = svc.repository.run(rid)["manifest"]
    manifest["stages"][0]["runner_config"].update(
        type="opencode", model="local//mnt/models/Qwen-Q4.gguf"
    )
    manifest["stages"][1]["runner_config"].update(type="codex", model="gpt-luna")
    identity = export_identity(manifest, manifest["novel"])
    assert identity["model"] == "local//mnt/models/Qwen-Q4.gguf"
    path = export_path(svc.config.novel.books_dir, manifest, manifest["novel"], "a" * 64)
    assert path.name == "Название произведения - opencode - Qwen-Q4.gguf.epub"
    assert path.parent.parent.name == rid
    assert path.is_relative_to(svc.config.novel.books_dir)


def test_v32_versions_keep_previous_files_and_do_not_collide(service):
    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    first = svc.build_epub(rid)
    original = first.read_bytes()
    assert svc.build_epub(rid) == first
    row = svc.repository.chunks(rid)[0]
    path = svc.run_dir(rid) / row["output_path"]
    revised = path.read_text("utf-8") + " Проверенная правка."
    write_text(path, revised)
    svc.repository.set_chunk(
        rid,
        row["chapter"],
        row["chunk_id"],
        "completed",
        json.loads(row["metadata"]),
        row["output_path"],
        sha256(revised),
    )
    second = svc.build_epub(rid)
    assert first != second and first.name == second.name
    assert first.read_bytes() == original
    other = svc.create_run(runner="test", force=True)
    svc.execute(other)
    third = svc.build_epub(other)
    assert third.name == first.name and third != first and third != second
    record = json.loads((svc.run_dir(rid) / "book.json").read_text("utf-8"))
    assert record["export_identity"]["model"] == "fake-model"
    second.write_bytes(b"externally modified")
    with pytest.raises(ValueError, match="previous file preserved"):
        svc.build_epub(rid)
    assert second.read_bytes() == b"externally modified"


def test_v32_windows_safety_long_unicode_and_sanitization_collisions(service, tmp_path):
    svc, _ = service
    rid = svc.create_run(runner="test")
    manifest = svc.repository.run(rid)["manifest"]
    novel = {"title": 'Очень длинное название 😺 /:*? "<>|' * 30}
    path = export_path(tmp_path, manifest, novel, "b" * 64)
    assert utf16_units(str(path)) <= 240 and len(path.name.encode("utf-8")) <= 240
    assert not re.search(r'[<>:"/\\|?*\x00-\x1f]', path.name)
    assert "~" in path.name and path.suffix == ".epub"
    assert safe_piece("CON.txt") == "_CON.txt"
    assert safe_piece(".. / ..") == "untitled"
    assert safe_piece("Имя. ") == "Имя"
    other = copy.deepcopy(manifest)
    other["run_id"] += "-other"
    assert export_path(tmp_path, other, novel, "b" * 64) != path
    with pytest.raises(ValueError, match="too long"):
        export_path(tmp_path / ("x" * 160), manifest, novel, "b" * 64)


def test_epub_structure_validates_and_metadata_is_complete(service):
    svc, _ = service
    svc.config.novel.author = "Test Author"
    rid = svc.create_run(runner="test")
    svc.execute(rid)

    data = svc.build_epub(rid).read_bytes()

    validate_epub(data)  # does not raise
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert names[0] == "mimetype"
        assert archive.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        assert archive.read("mimetype") == b"application/epub+zip"
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(f"{{{CONTAINER_NS}}}rootfiles/{{{CONTAINER_NS}}}rootfile")
        assert rootfile is not None and rootfile.get("full-path") == "EPUB/content.opf"
        opf = ET.fromstring(archive.read("EPUB/content.opf"))
        dc = {"dc": "http://purl.org/dc/elements/1.1/"}
        title = opf.find("opf:metadata/dc:title", {**dc, "opf": OPF_NS})
        language = opf.find("opf:metadata/dc:language", {**dc, "opf": OPF_NS})
        creator = opf.find("opf:metadata/dc:creator", {**dc, "opf": OPF_NS})
        assert title is not None and title.text
        assert language is not None and language.text == "ru"
        assert creator is not None and creator.text == "Test Author"
        nav_items = [
            item
            for item in opf.findall(f"{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
            if "nav" in (item.get("properties") or "").split()
        ]
        assert nav_items, "an EPUB3 nav document must be declared"
        assert opf.find(f"{{{OPF_NS}}}spine/{{{OPF_NS}}}itemref") is not None
    assert _opf_meta(data, "rendition:spread") == "auto"


def test_epub_spread_none_setting_and_rejects_unknown(service):
    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    default = svc.build_epub(rid).read_bytes()
    assert _opf_meta(default, "rendition:spread") == "auto"

    svc.config.novel.spread = "none"
    none = svc.build_epub(rid).read_bytes()
    assert _opf_meta(none, "rendition:spread") == "none"
    assert none != default

    manifest = svc.repository.run(rid)["manifest"]
    chapters = [svc.inspect_chapter(rid, c["number"]) for c in manifest["chapters"]]
    with pytest.raises(ValueError, match="spread must be one of"):
        render_epub({**manifest, "novel": manifest["novel"]}, chapters, spread="both")


def test_epub_cover_is_optional_and_embedded(service, tmp_path):
    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    cover = tmp_path / "cover.png"
    cover.write_bytes(PNG_1X1)

    svc.config.novel.cover_path = cover
    data = svc.build_epub(rid).read_bytes()

    validate_epub(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert any(name.endswith("cover.png") for name in archive.namelist())
        opf = ET.fromstring(archive.read("EPUB/content.opf"))
        assert any(
            (item.get("href") or "").endswith("cover.png")
            for item in opf.findall(f"{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
        )

    svc.config.novel.cover_path = tmp_path / "missing.png"
    with pytest.raises(ValueError, match="Cover image not found"):
        svc.build_epub(rid)


def test_validate_epub_rejects_broken_archives(service):
    svc, _ = service
    rid = svc.create_run(runner="test")
    svc.execute(rid)
    good = svc.build_epub(rid).read_bytes()

    with pytest.raises(ValueError, match="not a Zip"):
        validate_epub(b"not an epub")

    def rezip(entries: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as target:
            for name, payload in entries.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = (
                    zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
                )
                target.writestr(info, payload)
        return buffer.getvalue()

    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}

    no_mimetype = dict(entries)
    del no_mimetype["mimetype"]
    with pytest.raises(ValueError, match="mimetype"):
        validate_epub(rezip(no_mimetype))

    chapter = next(name for name in entries if re.search(r"chapter-\d+\.xhtml$", name))
    missing_chapter = {name: payload for name, payload in entries.items() if name != chapter}
    with pytest.raises(ValueError, match="missing from the archive"):
        validate_epub(rezip(missing_chapter))


def test_name_revision_composes_with_human_final_per_chunk(project, source):
    from conftest import FakeRunner

    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    fake = FakeRunner()
    svc = TranslationService(load_config(project), lambda name: fake)
    svc.config.translation.name_rules = [{"source": "彼は", "forms": {"Русский": "Иван"}}]
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)

    rows = [r for r in svc.repository.chunks(run_id) if r["chapter"] == 1]
    assert len(rows) >= 2, "fixture must yield a multi-chunk chapter"
    target = rows[0]
    svc.set_final(
        run_id, target["chapter"], target["chunk_id"],
        action="edit", actor="t", text="Правка человека уникальная.",
        expected_machine_sha256=target["output_sha256"],
    )
    svc.build_epub(run_id)

    record = json.loads((svc.run_dir(run_id) / "book.json").read_text("utf-8"))
    assert record["name_revision"]["replacement_count"] > 0
    entry = svc.repository.final_entry(run_id, target["chapter"], target["chunk_id"])
    # The human final is authoritative and is never name-revised.
    assert entry["final_text"] == "Правка человека уникальная."
    assert "Иван" not in entry["final_text"]

    with zipfile.ZipFile(io.BytesIO(svc.build_epub(run_id).read_bytes())) as archive:
        chapter_html = next(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.endswith("chapter-0001.xhtml")
        )
    # One edited chunk must not disable name revision for the whole chapter.
    assert "Правка человека уникальная." in chapter_html
    assert "Иван" in chapter_html


def test_epub_terminology_is_not_misleading():
    from novel_harness import epub as epub_module

    module_doc = (epub_module.__doc__ or "").lower()
    assert "reflowable" in module_doc
    assert "non-paginated rendition" not in module_doc

    validate_doc = (epub_module.validate_epub.__doc__ or "").lower()
    assert "structural" in validate_doc
    assert "not a full" in validate_doc
