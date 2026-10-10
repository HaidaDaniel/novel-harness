"""Deterministic EPUB3 rendering, structural validation and normalized ZIP metadata.

The output is a reflowable EPUB3. A reader's one/two-column display is a reader
preference; this module does not force single or dual columns with CSS hacks. The
``spread`` setting maps to the standard ``rendition:spread`` metadata property:
``auto`` by default, and ``none`` merely asks the reading system not to form a
synthetic two-page spread. ``none`` does not make the EPUB non-paginated and does
not disable a reader's own multi-column or paginated layout; those stay reader
settings. It is never a column override.
"""

import html.entities
import io
import posixpath
import re
import stat
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from ebooklib import epub

from .artifacts import sha256

CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
SPREAD_VALUES = ("auto", "none")

_STYLESHEET = (
    "html{font-family:serif;line-height:1.5}"
    "body{margin:0 1em}"
    "p{margin:0.7em 0;text-align:justify}"
    "h1{font-size:1.5em;line-height:1.3;margin:1.2em 0 0.8em}"
)

_COVER_MEDIA_TYPES = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
}

_XHTML_NS = "http://www.w3.org/1999/xhtml"
_DC_TITLE = f"{{{DC_NS}}}title"
_DC_CREATOR = f"{{{DC_NS}}}creator"
_DC_LANGUAGE = f"{{{DC_NS}}}language"
_PROTECTED_TEXT_TAGS = {"code", "math", "pre", "rp", "rt", "script", "style", "svg"}
_BLOCK_TAGS = {
    "article", "blockquote", "dd", "div", "dl", "dt", "figcaption", "figure",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ol", "p", "section", "table",
    "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}
_MAX_SOURCE_EPUB_BYTES = 512 * 1024 * 1024
_MAX_EPUB_ENTRIES = 20_000
_MAX_EPUB_EXPANDED_BYTES = 1024 * 1024 * 1024
_MAX_EPUB_ENTRY_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class SourceEpubDocument:
    number: int
    href: str
    filename: str
    text: str


@dataclass(frozen=True)
class SourceEpub:
    source_sha256: str
    title: str
    author: str | None
    language: str
    documents: tuple[SourceEpubDocument, ...]


@dataclass(frozen=True)
class _TextSlot:
    owner: ET.Element
    field: str
    original: str
    leading: str
    trailing: str
    content: str
    start: str
    end: str
    block: int


def _local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _safe_archive_name(name: str) -> str:
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError("EPUB contains an unsafe archive path")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts[:-1]) or parts[-1] in {".", ".."}:
        raise ValueError("EPUB contains an unsafe archive path")
    return name


def _safe_package_path(path: str) -> str:
    decoded = unquote(path)
    if not decoded or "\\" in decoded or decoded.startswith("/"):
        raise ValueError("EPUB package references an unsafe resource path")
    normalized = posixpath.normpath(decoded)
    if normalized in {".", ".."} or normalized.startswith("../"):
        raise ValueError("EPUB package references a resource outside its archive")
    return normalized


def _resolve_manifest_href(package_path: str, href: str) -> str:
    parsed = urlparse(href)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("EPUB manifest contains a non-local resource reference")
    return _safe_package_path(posixpath.join(posixpath.dirname(package_path), parsed.path))


def _check_source_archive(archive: zipfile.ZipFile, names: list[str]) -> None:
    if len(names) > _MAX_EPUB_ENTRIES or len(names) != len(set(names)):
        raise ValueError("EPUB has too many or duplicate archive entries")
    total = 0
    for info in archive.infolist():
        _safe_archive_name(info.filename)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError("EPUB may not contain symbolic links")
        if info.flag_bits & 0x1:
            raise ValueError("Encrypted EPUB entries are unsupported")
        if info.file_size > _MAX_EPUB_ENTRY_BYTES:
            raise ValueError("EPUB contains an oversized resource")
        total += info.file_size
        if total > _MAX_EPUB_EXPANDED_BYTES:
            raise ValueError("EPUB expands beyond the import size limit")


def _parse_xhtml(data: bytes) -> ET.Element:
    if len(data) > _MAX_EPUB_ENTRY_BYTES:
        raise ValueError("EPUB XHTML document exceeds the import size limit")
    try:
        text = _decode_xml(data)
        doctype = _xml_doctype_text(text)
        if doctype:
            text = text.replace(doctype, "", 1)
        text = re.sub(r"<\?xml\b[^?]*\?>", "", text, count=1, flags=re.IGNORECASE)
        text = _decode_named_entities(text)
        parser = ET.XMLParser(
            target=ET.TreeBuilder(insert_comments=True, insert_pis=True)
        )
        root = ET.fromstring(text.encode("utf-8"), parser=parser)
    except (ET.ParseError, LookupError, UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("EPUB content document is not supported well-formed XHTML") from error
    if root.tag != f"{{{_XHTML_NS}}}html":
        raise ValueError("EPUB content document must use the XHTML namespace")
    return root


def _decode_xml(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith(b"\x00<"):
        return data.decode("utf-16-be")
    if data.startswith(b"<\x00"):
        return data.decode("utf-16-le")
    return data.decode("utf-8")


def _xml_doctype_text(text: str) -> str:
    """Return the source DOCTYPE so serializing translated text keeps it intact."""
    match = re.search(r"<!DOCTYPE\b", text, re.IGNORECASE)
    if not match:
        return ""
    quote = None
    subset = 0
    for index in range(match.start(), len(text)):
        char = text[index]
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "[":
            subset += 1
        elif char == "]" and subset:
            subset -= 1
        elif char == ">" and subset == 0:
            return text[match.start() : index + 1]
    raise ValueError("EPUB content document has an incomplete DOCTYPE")


def _decode_named_entities(text: str) -> str:
    predefined = {"amp", "lt", "gt", "quot", "apos"}

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name in predefined:
            return match.group(0)
        value = html.entities.html5.get(name + ";")
        if value is None:
            raise ValueError("EPUB content document uses an unsupported named entity")
        return "".join(f"&#x{ord(char):x};" for char in value)

    return re.sub(r"&([A-Za-z][A-Za-z0-9]+);", replace, text)


def _validate_source_internal_links(
    archive: zipfile.ZipFile, package_path: str, package: ET.Element
) -> None:
    """Reject local XHTML links whose resources or fragment targets are missing."""
    manifest = package.find(f"{{{OPF_NS}}}manifest")
    if manifest is None:
        return
    names = set(archive.namelist())
    documents: dict[str, ET.Element] = {}
    manifest_paths: set[str] = set()
    for item in manifest.findall(f"{{{OPF_NS}}}item"):
        href = item.get("href") or ""
        parsed = urlparse(href)
        if parsed.scheme or parsed.netloc:
            continue
        path = _resolve_manifest_href(package_path, href)
        manifest_paths.add(path)
        if item.get("media-type") != "application/xhtml+xml":
            continue
        try:
            documents[path] = _parse_xhtml(archive.read(path))
        except KeyError as error:
            raise ValueError("EPUB XHTML manifest resource is missing") from error

    anchors: dict[str, set[str]] = {}
    for path, root in documents.items():
        id_nodes: dict[str, ET.Element] = {}
        for element in root.iter():
            if element.get("{http://www.w3.org/XML/1998/namespace}base") is not None:
                raise ValueError("EPUB XHTML xml:base references are unsupported")
            if _local_name(element.tag) == "base" and element.get("href") is not None:
                raise ValueError("EPUB XHTML base href references are unsupported")
            for key in ("id", "{http://www.w3.org/XML/1998/namespace}id"):
                value = element.get(key)
                if value:
                    previous = id_nodes.get(value)
                    if previous is not None and previous is not element:
                        raise ValueError(f"EPUB XHTML has duplicate anchor ID {value!r}")
                    id_nodes[value] = element
            if _local_name(element.tag) == "a" and element.get("name"):
                value = element.get("name")
                previous = id_nodes.get(value)
                if previous is not None and previous is not element:
                    raise ValueError(f"EPUB XHTML has duplicate anchor ID {value!r}")
                id_nodes[value] = element
        anchors[path] = set(id_nodes)

    for path, root in documents.items():
        for element in root.iter():
            href_values = (
                value
                for key, value in element.attrib.items()
                if _local_name(key) == "href"
            )
            for href in href_values:
                if not href:
                    continue
                reference = urlparse(href)
                if reference.scheme or reference.netloc:
                    continue
                target = (
                    path
                    if not reference.path
                    else _safe_package_path(
                        posixpath.join(posixpath.dirname(path), reference.path)
                    )
                )
                if target not in names or target not in manifest_paths:
                    raise ValueError("EPUB XHTML link references a missing local resource")
                if reference.fragment and target in anchors:
                    fragment = unquote(reference.fragment)
                    if fragment not in anchors[target]:
                        raise ValueError(
                            f"EPUB XHTML link fragment #{fragment} does not resolve"
                        )


def _source_slots(
    body: ET.Element, source_sha256: str, document_number: int
) -> list[_TextSlot]:
    slots: list[_TextSlot] = []
    block_number = 0
    marker_prefix = f"⟦EPUB_{source_sha256[:16]}_{document_number:04d}_"

    def add(owner: ET.Element, field: str, value: str | None, block: int) -> None:
        if not value:
            return
        if not value.strip():
            # Preserve inline spaces; layout-only indentation remains in the source tree.
            if "\n" not in value and "\r" not in value:
                slots.append(
                    _TextSlot(owner, field, value, "", "", "", "", "", block)
                )
            return
        leading = value[: len(value) - len(value.lstrip())]
        trailing = value[len(value.rstrip()) :]
        content_end = len(value) - len(trailing) if trailing else len(value)
        content = value[len(leading) : content_end]
        index = len(slots) + 1
        marker = f"{marker_prefix}{index:06d}"
        if marker in content or "⟦EPUB_" in content:
            raise ValueError("EPUB text collides with a protected round-trip marker")
        slots.append(
            _TextSlot(
                owner,
                field,
                value,
                leading,
                trailing,
                content,
                f"{marker}_S⟧",
                f"{marker}_E⟧",
                block,
            )
        )

    def visit(element: ET.Element, inherited_block: int, protected: bool = False) -> None:
        nonlocal block_number
        tag = _local_name(element.tag)
        if not tag:
            return
        if protected or tag in _PROTECTED_TEXT_TAGS:
            return
        if tag in _BLOCK_TAGS:
            block_number += 1
            inherited_block = block_number
        add(element, "text", element.text, inherited_block)
        for child in list(element):
            visit(child, inherited_block)
            add(child, "tail", child.tail, inherited_block)

    visit(body, 0)
    return slots


def _document_text(slots: list[_TextSlot]) -> str:
    parts: list[str] = []
    previous_block: int | None = None
    for slot in slots:
        if slot.block != previous_block and parts:
            parts.append("\n\n")
        previous_block = slot.block
        parts.append(slot.leading)
        if slot.start:
            parts.extend((slot.start, slot.content, slot.end))
        else:
            parts.append(slot.original)
        parts.append(slot.trailing)
    body = "".join(parts)
    guide = (
        "⟦EPUB_GUIDANCE_S⟧Preserve every EPUB marker exactly once and in order; "
        "translate only marked text.⟦EPUB_GUIDANCE_E⟧"
    )
    return guide + "\n\n" + body


def inspect_source_epub(data: bytes) -> SourceEpub:
    """Validate an input EPUB and expose its spine text as protected TXT chapters."""
    if not isinstance(data, bytes) or not data or len(data) > _MAX_SOURCE_EPUB_BYTES:
        raise ValueError("Source EPUB is empty or exceeds the import size limit")
    source_sha = sha256(data)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ValueError("Source EPUB is not a valid ZIP archive") from error
    with archive:
        names = archive.namelist()
        _check_source_archive(archive, names)
        try:
            package_path = _safe_package_path(
                _find_rootfile(archive.read("META-INF/container.xml"))
            )
            package = ET.fromstring(archive.read(package_path))
        except (KeyError, ET.ParseError, ValueError) as error:
            raise ValueError("Source EPUB package document is unusable") from error
        metadata = package.find(f"{{{OPF_NS}}}metadata")
        title = (metadata.findtext(_DC_TITLE, default="") if metadata is not None else "").strip()
        language = (
            metadata.findtext(_DC_LANGUAGE, default="") if metadata is not None else ""
        ).strip()
        author = (
            metadata.findtext(_DC_CREATOR, default="") if metadata is not None else ""
        ).strip() or None
        if not title or not language or language.lower() != "ja" and not language.lower().startswith("ja-"):
            raise ValueError("Source EPUB must declare a title and Japanese language")
        manifest = package.find(f"{{{OPF_NS}}}manifest")
        spine = package.find(f"{{{OPF_NS}}}spine")
        items = {
            item.get("id"): item
            for item in (manifest.findall(f"{{{OPF_NS}}}item") if manifest is not None else [])
        }
        item_ids = [item.get("id") for item in (manifest.findall(f"{{{OPF_NS}}}item") if manifest is not None else [])]
        if None in item_ids or len(item_ids) != len(set(item_ids)):
            raise ValueError("Source EPUB manifest has missing or duplicate item IDs")
        nav_ids = {
            item.get("id")
            for item in items.values()
            if "nav" in (item.get("properties") or "").split()
        }
        documents: list[SourceEpubDocument] = []
        seen_documents: set[str] = set()
        refs = spine.findall(f"{{{OPF_NS}}}itemref") if spine is not None else []
        for itemref in refs:
            item_id = itemref.get("idref")
            item = items.get(item_id)
            if item is None or item_id in nav_ids or item.get("media-type") != "application/xhtml+xml":
                continue
            href = item.get("href") or ""
            archive_path = _resolve_manifest_href(package_path, href)
            if archive_path in seen_documents:
                raise ValueError("Source EPUB repeats a content document in its spine")
            seen_documents.add(archive_path)
            try:
                raw = archive.read(archive_path)
            except KeyError as error:
                raise ValueError("Source EPUB spine references a missing XHTML document") from error
            root = _parse_xhtml(raw)
            body = root.find(f".//{{{_XHTML_NS}}}body")
            if body is None:
                raise ValueError("Source EPUB content document has no XHTML body")
            number = len(documents) + 1
            try:
                slots = _source_slots(body, source_sha, number)
            except RecursionError as error:
                raise ValueError("Source EPUB XHTML nesting is too deep") from error
            if not any(slot.start for slot in slots):
                continue
            basename = posixpath.basename(urlparse(href).path)
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", posixpath.splitext(basename)[0]).strip("_")
            filename = f"{number:04d}_{(stem or 'document')[:64]}.txt"
            documents.append(
                SourceEpubDocument(
                    number=number,
                    href=archive_path,
                    filename=filename,
                    text=_document_text(slots),
                )
            )
        if not documents:
            raise ValueError("Source EPUB has no translatable XHTML documents in its spine")
        _validate_source_internal_links(archive, package_path, package)
    validate_epub(data)
    return SourceEpub(source_sha, title, author, language, tuple(documents))


def _translated_slot_text(slots: list[_TextSlot], translated: str) -> list[str]:
    """Extract complete protected text slots; malformed model output fails closed."""
    translatable = [slot for slot in slots if slot.start]
    marker_pattern = re.compile(r"⟦EPUB_[^⟧]+⟧")
    markers = list(marker_pattern.finditer(translated))
    expected = ["⟦EPUB_GUIDANCE_S⟧", "⟦EPUB_GUIDANCE_E⟧"] + [
        marker for slot in translatable for marker in (slot.start, slot.end)
    ]
    if [match.group() for match in markers] != expected:
        raise ValueError("Translated EPUB chapter has missing, duplicated, or reordered markers")
    results: list[str] = []
    previous = 0
    for index, slot in enumerate([None, *translatable]):
        start, end = markers[index * 2 : index * 2 + 2]
        if translated[previous : start.start()].strip():
            raise ValueError("Translated EPUB chapter contains text outside protected slots")
        value = translated[start.end() : end.start()]
        if slot is None:
            previous = end.end()
            continue
        if not value.strip():
            raise ValueError("Translated EPUB chapter dropped a protected text slot")
        results.append(value)
        previous = end.end()
    if translated[previous:].strip():
        raise ValueError("Translated EPUB chapter contains text outside protected slots")
    return results


def _structure_signature(element: ET.Element) -> tuple:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        tuple(_structure_signature(child) for child in list(element)),
    )


def _serialize_xhtml(root: ET.Element, original: bytes) -> bytes:
    ET.register_namespace("", _XHTML_NS)
    doctype = _xml_doctype_text(_decode_xml(original))
    body = ET.tostring(root, encoding="utf-8", short_empty_elements=True)
    return (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        + (doctype.encode("utf-8") + b"\n" if doctype else b"")
        + body
    )


def render_source_epub(
    data: bytes,
    translations: dict[int, str],
    *,
    require_complete: bool = True,
) -> bytes:
    """Replace mapped XHTML text, optionally requiring every source document."""
    source = inspect_source_epub(data)
    if (
        not isinstance(translations, dict)
        or any(
            type(number) is not int
            or number < 1
            or number > len(source.documents)
            or not isinstance(value, str)
            for number, value in translations.items()
        )
    ):
        raise ValueError("Translated EPUB chapter selection is invalid")
    replacements: dict[str, bytes] = {}
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ValueError("Source EPUB is not a valid ZIP archive") from error
    with archive:
        for document in source.documents:
            translated = translations.get(document.number)
            if translated is None:
                continue
            raw = archive.read(document.href)
            root = _parse_xhtml(raw)
            body = root.find(f".//{{{_XHTML_NS}}}body")
            if body is None:
                raise ValueError("Source EPUB content document has no XHTML body")
            try:
                slots = _source_slots(body, source.source_sha256, document.number)
            except RecursionError as error:
                raise ValueError("Source EPUB XHTML nesting is too deep") from error
            results = _translated_slot_text(slots, translated)
            slot_index = 0
            try:
                before = _structure_signature(root)
            except RecursionError as error:
                raise ValueError("Source EPUB XHTML nesting is too deep") from error
            for slot in slots:
                if not slot.start:
                    continue
                value = slot.leading + results[slot_index] + slot.trailing
                setattr(slot.owner, slot.field, value)
                slot_index += 1
            try:
                after = _structure_signature(root)
            except RecursionError as error:
                raise ValueError("Translated EPUB XHTML nesting is too deep") from error
            if before != after:
                raise ValueError("Translated EPUB changed XHTML structure or attributes")
            replacements[document.href] = _serialize_xhtml(root, raw)
    if require_complete:
        missing = sorted(
            {document.number for document in source.documents} - set(translations)
        )
        if missing:
            chapters = ", ".join(str(number) for number in missing)
            raise ValueError(
                "Cannot build EPUB: source EPUB translation is incomplete; "
                f"missing translations for source chapters: {chapters}"
            )
    output = io.BytesIO()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as source_archive, zipfile.ZipFile(
            output, "w"
        ) as target:
            _check_source_archive(source_archive, source_archive.namelist())
            target.comment = source_archive.comment
            for info in source_archive.infolist():
                contents = replacements.get(info.filename)
                if contents is None:
                    contents = source_archive.read(info.filename)
                target.writestr(info, contents)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("Could not preserve source EPUB archive resources") from error
    rendered = output.getvalue()
    validate_epub(rendered)
    try:
        with zipfile.ZipFile(io.BytesIO(rendered)) as rendered_archive:
            rendered_package_path = _safe_package_path(
                _find_rootfile(rendered_archive.read("META-INF/container.xml"))
            )
            rendered_package = ET.fromstring(rendered_archive.read(rendered_package_path))
            _validate_source_internal_links(
                rendered_archive, rendered_package_path, rendered_package
            )
    except (KeyError, ET.ParseError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("Translated EPUB has an unresolved internal link") from error
    return rendered


def strip_epub_roundtrip_markers(text: str) -> str:
    """Return readable text for checks and projections that must not see sentinels."""
    text = re.sub(
        r"⟦EPUB_GUIDANCE_S⟧.*?⟦EPUB_GUIDANCE_E⟧", "", text, flags=re.DOTALL
    )
    return re.sub(r"⟦EPUB_[^⟧]+⟧", "", text)


def _find_rootfile(container_xml: bytes) -> str:
    root = ET.fromstring(container_xml)
    rootfile = root.find(f"{{{CONTAINER_NS}}}rootfiles/{{{CONTAINER_NS}}}rootfile")
    full_path = rootfile.get("full-path") if rootfile is not None else None
    if not full_path:
        raise ValueError("container.xml has no rootfile")
    return full_path


def validate_epub(data: bytes) -> None:
    """Deterministic structural EPUB validation.

    This is a bounded structural check, not a full ``epubcheck``-equivalent
    conformance check: ``epubcheck`` is a Java tool and is not available offline
    here, so full EPUB conformance validation remains a separate/optional
    capability. This check verifies the parts that break real readers: the
    mimetype entry, the container/OPF wiring, metadata (title/language), a
    non-empty spine whose idrefs exist, an EPUB3 nav document, and that every
    local manifest resource is actually present in the archive. Semantic quality
    and full spec conformance are out of scope.
    """
    problems: list[str] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ValueError(f"Invalid EPUB: not a Zip archive ({error})") from error
    with archive:
        names = archive.namelist()
        infos = {info.filename: info for info in archive.infolist()}
        if not names or names[0] != "mimetype":
            problems.append("mimetype must be the first archive entry")
        elif infos["mimetype"].compress_type != zipfile.ZIP_STORED:
            problems.append("mimetype must be stored uncompressed")
        if "mimetype" in names and archive.read("mimetype") != b"application/epub+zip":
            problems.append("mimetype content must be application/epub+zip")
        if "META-INF/container.xml" not in names:
            problems.append("META-INF/container.xml is missing")
            raise ValueError("Invalid EPUB: " + "; ".join(problems))
        try:
            opf_path = _find_rootfile(archive.read("META-INF/container.xml"))
        except (ET.ParseError, ValueError) as error:
            raise ValueError(f"Invalid EPUB: container.xml is unusable ({error})") from error
        if opf_path not in names:
            raise ValueError(f"Invalid EPUB: package document {opf_path!r} is missing")
        try:
            package = ET.fromstring(archive.read(opf_path))
        except ET.ParseError as error:
            raise ValueError(f"Invalid EPUB: package document does not parse ({error})") from error
        metadata = package.find(f"{{{OPF_NS}}}metadata")
        if metadata is None:
            problems.append("package metadata is missing")
        else:
            if metadata.find(f"{{{DC_NS}}}title") is None:
                problems.append("dc:title is missing")
            if metadata.find(f"{{{DC_NS}}}language") is None:
                problems.append("dc:language is missing")
        manifest_items = {
            item.get("id"): item
            for item in package.findall(f"{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
        }
        if not manifest_items:
            problems.append("manifest is empty")
        for item in manifest_items.values():
            href = item.get("href") or ""
            if urlparse(href).scheme:
                continue
            try:
                resolved = _resolve_manifest_href(opf_path, href)
            except ValueError:
                problems.append(f"manifest resource {href!r} is not a safe local path")
                continue
            if resolved not in names:
                problems.append(f"manifest resource {href!r} is missing from the archive")
        has_nav = any("nav" in (item.get("properties") or "").split() for item in manifest_items.values())
        if not has_nav:
            problems.append("no EPUB3 nav document is declared")
        spine = package.find(f"{{{OPF_NS}}}spine")
        idrefs = [ref.get("idref") for ref in (spine.findall(f"{{{OPF_NS}}}itemref") if spine is not None else [])]
        if not idrefs:
            problems.append("spine is empty")
        for idref in idrefs:
            if idref not in manifest_items:
                problems.append(f"spine idref {idref!r} is not declared in the manifest")
    if problems:
        raise ValueError("Invalid EPUB: " + "; ".join(problems))


def render_epub(
    manifest: dict,
    chapters: list[dict],
    *,
    spread: str = "auto",
    cover_path: Path | None = None,
) -> bytes:
    if spread not in SPREAD_VALUES:
        raise ValueError(f"spread must be one of {SPREAD_VALUES}, got {spread!r}")
    if not chapters or any(not c["complete"] for c in chapters):
        raise ValueError("Cannot build EPUB: failed, incomplete or corrupt chunks")
    book = epub.EpubBook()
    book.set_identifier(f"urn:novel-harness:{manifest['run_id']}")
    book.set_title(manifest["novel"]["title"])
    book.set_language("ru")
    if manifest["novel"]["author"]:
        book.add_author(manifest["novel"]["author"])
    book.add_metadata("OPF", "meta", spread, {"property": "rendition:spread"})
    if cover_path is not None:
        cover = Path(cover_path)
        if not cover.is_file():
            raise ValueError(f"Cover image not found: {cover}")
        media_type = _COVER_MEDIA_TYPES.get(cover.suffix.lower())
        if media_type is None:
            raise ValueError(f"Unsupported cover image type: {cover.suffix}")
        book.set_cover(f"cover{cover.suffix.lower()}", cover.read_bytes(), create_page=True)
    css = epub.EpubItem(
        uid="style",
        file_name="style.css",
        media_type="text/css",
        content=_STYLESHEET,
    )
    book.add_item(css)
    documents = []
    for chapter in sorted(chapters, key=lambda c: c["chapter"]):
        text = chapter["translation"].strip()
        # Input convention: first nonempty line is the chapter title, translated with chunk 1.
        lines = text.splitlines()
        heading = lines[0].strip().lstrip("# ")
        body = "\n".join(lines[1:]).strip()
        paragraphs = re.split(r"\n\s*\n", body) if body else []
        content = "<h1>" + escape(heading) + "</h1>"
        content += "".join("<p>" + escape(p).replace("\n", "<br/>") + "</p>" for p in paragraphs)
        document = epub.EpubHtml(
            uid=f"chapter-{chapter['chapter']}",
            title=heading,
            file_name=f"chapter-{chapter['chapter']:04d}.xhtml",
            lang="ru",
        )
        document.content = content
        document.add_item(css)
        book.add_item(document)
        documents.append(document)
    book.toc = tuple(documents)
    book.spine = ["nav", *documents]
    book.add_item(epub.EpubNcx())
    nav = epub.EpubNav(title="Оглавление")
    nav.add_item(css)
    book.add_item(nav)
    generated = io.BytesIO()
    epub.write_epub(
        generated,
        book,
        {
            "raise_exceptions": True,
            "mtime": datetime.fromisoformat(manifest["created_at"]),
            "epub3_pages": False,
        },
    )
    normalized = io.BytesIO()
    with zipfile.ZipFile(generated) as source, zipfile.ZipFile(normalized, "w") as target:
        for name in ["mimetype", *sorted(n for n in source.namelist() if n != "mimetype")]:
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o644 << 16
            target.writestr(entry, source.read(name), compresslevel=6)
    data = normalized.getvalue()
    validate_epub(data)
    return data
