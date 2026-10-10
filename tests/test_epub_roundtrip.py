import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import yaml

from novel_harness.artifacts import sha256
from novel_harness.config import load_config
from novel_harness.epub import inspect_source_epub, render_source_epub
from novel_harness.library import candidate
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService
from novel_harness.source_import import import_epub_source
from novel_harness.translation.chunker import chunk_text

OPF = "http://www.idpf.org/2007/opf"
DC = "http://purl.org/dc/elements/1.1/"
XHTML = "http://www.w3.org/1999/xhtml"


def _source_epub() -> bytes:
    files = {
        "mimetype": b"application/epub+zip",
        "META-INF/container.xml": (
            b'<?xml version="1.0"?>'
            b'<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
            b'version="1.0"><rootfiles><rootfile full-path="OEBPS/book.opf" '
            b'media-type="application/oebps-package+xml"/></rootfiles></container>'
        ),
        "OEBPS/book.opf": (
            f'<package xmlns="{OPF}" xmlns:dc="{DC}" version="3.0" '
            'unique-identifier="book-id"><metadata>'
            '<dc:identifier id="book-id">urn:source:book-17</dc:identifier>'
            '<dc:title>原作タイトル</dc:title><dc:creator>著者</dc:creator>'
            '<dc:language>ja</dc:language><dc:publisher>出版社</dc:publisher>'
            '<meta property="dcterms:modified">2026-09-25T00:00:00Z</meta>'
            '</metadata><manifest>'
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
            'properties="nav"/><item id="css" href="styles/book.css" '
            'media-type="text/css"/><item id="cover" href="images/cover.png" '
            'media-type="image/png" properties="cover-image"/>'
            '<item id="front" href="text/front.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="chapter" href="text/chapter.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="notes" href="text/notes.xhtml" media-type="application/xhtml+xml"/>'
            '</manifest><spine><itemref idref="front"/><itemref idref="chapter"/>'
            '<itemref idref="notes" linear="no"/></spine></package>'
        ).encode(),
        "OEBPS/nav.xhtml": (
            f'<html xmlns="{XHTML}"><head><title>目次</title></head><body>'
            '<nav><ol><li><a href="text/chapter.xhtml#chapter-title">第一話</a></li>'
            '</ol></nav></body></html>'
        ).encode(),
        "OEBPS/styles/book.css": b"p { font-style: italic }\n",
        "OEBPS/images/cover.png": b"fixture-image-bytes",
        "OEBPS/text/front.xhtml": (
            f'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>'
            f'<html xmlns="{XHTML}"><head><title>前書き</title>'
            '<link rel="stylesheet" href="../styles/book.css"/></head><body>'
            '<h1 id="front-title">序章</h1><p>物語の始まり。</p>'
            '</body></html>'
        ).encode(),
        "OEBPS/text/chapter.xhtml": (
            f'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>'
            f'<html xmlns="{XHTML}"><head><title>第一話</title>'
            '<link rel="stylesheet" href="../styles/book.css"/></head><body>'
            '<h1 id="chapter-title">第一話</h1>'
            '<p id="p1">私<a id="fnref1" href="notes.xhtml#note1">は</a>'
            '<em>森の中</em>を歩いた。</p>'
            '<p>風が吹いた。</p></body></html>'
        ).encode(),
        "OEBPS/text/notes.xhtml": (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f'<html xmlns="{XHTML}"><head><title>注</title></head><body>'
            '<aside epub:type="footnote" xmlns:epub="http://www.idpf.org/2007/ops" '
            'id="note1"><p>注釈<a href="chapter.xhtml#fnref1">戻る</a></p></aside>'
            '</body></html>'
        ).encode(),
    }
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        for name, contents in files.items():
            info = zipfile.ZipInfo(name, (2024, 1, 2, 3, 4, 6))
            info.compress_type = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)
    return result.getvalue()


def _translated(value: str) -> str:
    pattern = re.compile(r"(⟦EPUB_[^⟧]+_S⟧).*?(⟦EPUB_[^⟧]+_E⟧)", re.DOTALL)
    return pattern.sub(r"\1Русский перевод.\2", value)


def _zip_contents(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _replace_entry(data: bytes, path: str, replacement: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(output, "w") as target:
        target.comment = source.comment
        for info in source.infolist():
            target.writestr(info, replacement if info.filename == path else source.read(info.filename))
    return output.getvalue()


def test_roundtrip_translates_spine_text_and_preserves_package_resources():
    original = _source_epub()
    parsed = inspect_source_epub(original)
    assert parsed.title == "原作タイトル"
    assert parsed.author == "著者"
    assert parsed.language == "ja"
    assert [document.href for document in parsed.documents] == [
        "OEBPS/text/front.xhtml",
        "OEBPS/text/chapter.xhtml",
        "OEBPS/text/notes.xhtml",
    ]
    assert all("⟦EPUB_" in document.text for document in parsed.documents)

    translations = {document.number: _translated(document.text) for document in parsed.documents}
    translated = render_source_epub(original, translations)
    before, after = _zip_contents(original), _zip_contents(translated)
    assert list(before) == list(after)
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {document.href for document in parsed.documents}
    for name in (
        "mimetype",
        "META-INF/container.xml",
        "OEBPS/book.opf",
        "OEBPS/nav.xhtml",
        "OEBPS/styles/book.css",
        "OEBPS/images/cover.png",
    ):
        assert after[name] == before[name]
    assert "Русский перевод.".encode() in after["OEBPS/text/chapter.xhtml"]
    chapter = ET.fromstring(after["OEBPS/text/chapter.xhtml"])
    assert chapter.find(f".//{{{XHTML}}}h1").get("id") == "chapter-title"
    assert chapter.find(f".//{{{XHTML}}}a").get("href") == "notes.xhtml#note1"
    with zipfile.ZipFile(io.BytesIO(translated)) as archive:
        assert archive.namelist()[0] == "mimetype"
        assert archive.getinfo("mimetype").compress_type == zipfile.ZIP_STORED


def test_roundtrip_rejects_lost_or_reordered_text_markers():
    parsed = inspect_source_epub(_source_epub())
    first = parsed.documents[0]
    broken = first.text.replace("⟦EPUB_", "[EPUB_", 1)
    with pytest.raises(ValueError, match="markers"):
        render_source_epub(_source_epub(), {first.number: broken})

    with pytest.raises(ValueError, match="outside protected slots"):
        render_source_epub(_source_epub(), {first.number: _translated(first.text) + " extra"})

    translated = _translated(first.text)
    pairs = list(
        re.finditer(r"(⟦EPUB_[^⟧]+_S⟧).*?(⟦EPUB_[^⟧]+_E⟧)", translated, re.DOTALL)
    )
    assert len(pairs) > 1
    first_pair, second_pair = pairs[:2]
    reordered = (
        translated[: first_pair.start()]
        + translated[second_pair.start() : second_pair.end()]
        + translated[first_pair.end() : second_pair.start()]
        + translated[first_pair.start() : first_pair.end()]
        + translated[second_pair.end() :]
    )
    with pytest.raises(ValueError, match="markers"):
        render_source_epub(_source_epub(), {first.number: reordered})


def test_chunker_keeps_epub_markers_whole_with_small_chunk_size():
    source = "x" * 25 + "⟦EPUB_0123456789abcdef_0001_000001_S⟧" + "y" * 40
    chunks = chunk_text(source, max_chars=32)
    assert "".join(chunk.text for chunk in chunks) == source
    assert any("⟦EPUB_0123456789abcdef_0001_000001_S⟧" in chunk.text for chunk in chunks)


def test_import_rejects_package_path_escape_and_non_japanese_epub():
    source = _source_epub()
    package = _zip_contents(source)["OEBPS/book.opf"].replace(
        b'href="text/front.xhtml"', b'href="../../../outside.xhtml"'
    )
    package_escape = _replace_entry(source, "OEBPS/book.opf", package)
    with pytest.raises(ValueError, match="outside its archive"):
        inspect_source_epub(package_escape)

    package = _zip_contents(source)["OEBPS/book.opf"].replace(
        b"<dc:language>ja</dc:language>", b"<dc:language>en</dc:language>"
    )
    wrong_language = _replace_entry(source, "OEBPS/book.opf", package)
    with pytest.raises(ValueError, match="Japanese language"):
        inspect_source_epub(wrong_language)


def test_roundtrip_rejects_dangling_internal_fragment_links():
    source = _source_epub()
    chapter = _zip_contents(source)["OEBPS/text/chapter.xhtml"].replace(
        b"notes.xhtml#note1", b"notes.xhtml#missing-note"
    )
    broken = _replace_entry(source, "OEBPS/text/chapter.xhtml", chapter)

    with pytest.raises(ValueError, match="fragment #missing-note"):
        inspect_source_epub(broken)
    with pytest.raises(ValueError, match="fragment #missing-note"):
        render_source_epub(broken, {})


def test_import_rejects_unsupported_xhtml_base_references():
    source = _source_epub()
    chapter = _zip_contents(source)["OEBPS/text/chapter.xhtml"].replace(
        b"<body>", b'<body xml:base="../">'
    )
    based = _replace_entry(source, "OEBPS/text/chapter.xhtml", chapter)

    with pytest.raises(ValueError, match="xml:base references are unsupported"):
        inspect_source_epub(based)


def test_epub_import_is_atomic_and_idempotent(tmp_path):
    source = tmp_path / "source.epub"
    source.write_bytes(_source_epub())
    root = tmp_path / "library"
    result = import_epub_source(source, root, existing_sources=[], library_roots=[root])
    assert result.status == "imported"
    imported = Path(result.destination)
    assert (imported / "source.epub").read_bytes() == source.read_bytes()
    assert candidate(imported, direct=False)["status"] == "ready"
    assert sha256((imported / "source.epub").read_bytes()) == result.source_id
    same = import_epub_source(
        source,
        root,
        existing_sources=[candidate(imported, direct=False)],
        library_roots=[root],
    )
    assert same.status == "already_present"
    assert same.novel_id == result.novel_id


def test_service_freezes_epub_snapshot_into_run_and_exports_it(project, tmp_path):
    project_data = yaml.safe_load(project.read_text("utf-8"))
    library_root = tmp_path / "library"
    project_data["novel"]["library_roots"] = [str(library_root)]
    project_data["translation"]["chunk_chars"] = 20_000
    project.write_text(yaml.safe_dump(project_data, allow_unicode=True), "utf-8")

    class MarkerRunner:
        def run(self, job):
            return RunnerResult(_translated(job.source), model="fixture-model")

    source = tmp_path / "source.epub"
    source.write_bytes(_source_epub())
    service = TranslationService(load_config(project), lambda _name: MarkerRunner())
    imported = service.import_epub_source(source)
    run_id = service.create_run(runner="test", novel_id=imported["novel_id"], force=True)
    run = service.repository.run(run_id)
    source_record = run["manifest"]["source_identity"]["source_epub"]
    snapshot = service.run_dir(run_id) / source_record["file"]
    assert snapshot.read_bytes() == source.read_bytes()
    assert source_record["sha256"] == sha256(source.read_bytes())
    assert service.execute(run_id)["status"] == "completed"

    output = service.build_epub(run_id)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("OEBPS/book.opf") == _zip_contents(source.read_bytes())["OEBPS/book.opf"]
        assert "Русский перевод.".encode() in archive.read(
            "OEBPS/text/chapter.xhtml"
        )

    sample_id = service.create_sample_run(
        runner="test", novel_id=imported["novel_id"], chapters=[1]
    )
    assert service.execute(sample_id)["status"] == "completed"
    service.approve_sample(sample_id)
    whole_book_id = service.create_whole_book_run(sample_id)
    whole_book = service.repository.run(whole_book_id)
    descriptor = whole_book["manifest"]["source_identity"]["source_epub"]
    assert (service.run_dir(whole_book_id) / descriptor["file"]).read_bytes() == source.read_bytes()
