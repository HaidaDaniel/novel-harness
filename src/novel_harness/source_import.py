"""Offline, provider-neutral import of validated source chapter directories."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .artifacts import canonical, sha256, write_bytes, write_json
from .epub import inspect_source_epub
from .library import candidate
from .translation.discovery import Chapter, discover

_NAROU_ID = re.compile(r"[A-Za-z0-9]{1,32}\Z")


@dataclass(frozen=True)
class SourceImportRequest:
    """Versioned, human- or adapter-supplied source identity and completeness."""

    provider: str
    source_id: str
    title: str
    expected_chapters: tuple[int, ...]
    version: int = 1

    def validate(self) -> SourceImportRequest:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported source import request version")
        if not isinstance(self.provider, str) or self.provider != "narou":
            raise ValueError(f"Unsupported source provider: {self.provider}")
        if not isinstance(self.source_id, str) or not _NAROU_ID.fullmatch(self.source_id):
            raise ValueError("Narou source ID must contain only letters and digits")
        if (
            not isinstance(self.title, str)
            or not self.title
            or self.title != self.title.strip()
            or any(ord(char) < 32 for char in self.title)
            or "\ufffd" in self.title
            or len(self.title) > 500
        ):
            raise ValueError("Source title must be intact single-line text")
        if (
            not isinstance(self.expected_chapters, tuple)
            or not self.expected_chapters
            or len(self.expected_chapters) > 100_000
            or any(type(number) is not int or number < 1 for number in self.expected_chapters)
            or len(set(self.expected_chapters)) != len(self.expected_chapters)
            or tuple(sorted(self.expected_chapters)) != self.expected_chapters
        ):
            raise ValueError(
                "Expected chapter numbers must be sorted, unique positive integers"
            )
        return self


@dataclass(frozen=True)
class ValidatedSource:
    source_dir: Path
    request: SourceImportRequest
    chapters: tuple[Chapter, ...]
    inventory: tuple[dict, ...]
    inventory_sha256: str


@dataclass(frozen=True)
class ImportResult:
    """Stable version-1 result returned by every registered source adapter."""

    version: int
    status: Literal["imported", "already_present"]
    novel_id: str
    provider: str
    source_id: str
    destination: Path
    chapter_numbers: tuple[int, ...]
    inventory_sha256: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "status": self.status,
            "novel_id": self.novel_id,
            "source_identity": {"type": self.provider, "id": self.source_id},
            "destination": str(self.destination),
            "chapter_numbers": list(self.chapter_numbers),
            "inventory_sha256": self.inventory_sha256,
        }


def parse_chapter_numbers(value: str) -> tuple[int, ...]:
    """Parse an explicit comma-separated list or inclusive range list."""
    if not value or not value.strip():
        raise ValueError("Expected chapter list is required")
    numbers: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            raise ValueError("Expected chapter list contains an empty item")
        if "-" in item:
            match = re.fullmatch(r"([1-9]\d*)-([1-9]\d*)", item)
            if not match:
                raise ValueError("Chapter ranges must use positive N-M syntax")
            start, end = (int(match.group(1)), int(match.group(2)))
            if start > end:
                raise ValueError("Chapter range starts after it ends")
            if end - start + 1 > 100_000:
                raise ValueError("Chapter range exceeds the import limit")
            numbers.extend(range(start, end + 1))
        else:
            if not re.fullmatch(r"[1-9]\d*", item):
                raise ValueError("Chapter numbers must be positive integers")
            numbers.append(int(item))
    if len(numbers) > 100_000:
        raise ValueError("Expected chapter list exceeds the import limit")
    if len(numbers) != len(set(numbers)):
        raise ValueError("Expected chapter list contains duplicates")
    return tuple(sorted(numbers))


class NarouFilesystemAdapter:
    """Validate the downloader's flat ``NNNN_title.txt`` output without network."""

    provider = "narou"

    def validate(self, source_dir: Path, request: SourceImportRequest) -> ValidatedSource:
        request.validate()
        if source_dir.is_symlink():
            raise ValueError("Source directory must not be a symlink")
        path = source_dir.resolve(strict=True)
        if not path.is_dir():
            raise ValueError("Source directory does not exist")
        for item in path.iterdir():
            if item.is_symlink() and item.suffix.lower() == ".txt":
                raise ValueError(f"Chapter must be a regular file: {item.name}")
        chapters = tuple(discover(path, "*.txt"))
        for chapter in chapters:
            sections = chapter.text.split("\n\n", maxsplit=1)
            if (
                f"第{chapter.number}話" not in chapter.title
                or len(sections) != 2
                or not sections[1].strip()
                or not chapter.raw.endswith(b"\n")
            ):
                raise ValueError(f"Incomplete Narou episode file: {chapter.filename}")
        actual = tuple(chapter.number for chapter in chapters)
        if actual != request.expected_chapters:
            raise ValueError(
                "Chapter inventory is incomplete or unexpected: "
                f"expected {list(request.expected_chapters)}, found {list(actual)}"
            )
        inventory = tuple(
            {
                "number": chapter.number,
                "filename": chapter.filename,
                "source_sha256": chapter.source_sha256,
            }
            for chapter in chapters
        )
        return ValidatedSource(
            source_dir=path,
            request=request,
            chapters=chapters,
            inventory=inventory,
            inventory_sha256=sha256(canonical(inventory)),
        )


_ADAPTERS = {NarouFilesystemAdapter.provider: NarouFilesystemAdapter()}


def _manifest(validated: ValidatedSource) -> dict:
    return {
        "version": 1,
        "source": {
            "type": validated.request.provider,
            "id": validated.request.source_id.lower(),
        },
        "title": validated.request.title,
        "language": "ja",
        "format": "txt-chapters",
        "chapter_pattern": "*.txt",
    }


def _same_source(existing: dict, validated: ValidatedSource) -> bool:
    return (
        existing.get("status") == "ready"
        and existing.get("title") == validated.request.title
        and existing.get("inventory_sha256") == validated.inventory_sha256
        and existing.get("chapters") == list(validated.inventory)
    )


def import_source(
    source_dir: Path,
    request: SourceImportRequest,
    destination_root: Path,
    *,
    existing_sources: list[dict],
    library_roots: list[Path],
) -> ImportResult:
    """Validate, stage and atomically publish into one configured Library root."""
    request.validate()
    try:
        adapter = _ADAPTERS[request.provider]
    except KeyError:
        raise ValueError(f"Unsupported source provider: {request.provider}") from None
    validated = adapter.validate(source_dir, request)
    provider_id = request.source_id.lower()
    root = destination_root.resolve()
    target = root / provider_id
    matching = [
        entry
        for entry in existing_sources
        if entry.get("source_type") == request.provider
        and isinstance(entry.get("source_id"), str)
        and entry["source_id"].lower() == provider_id
    ]
    present = [entry for entry in matching if entry.get("status") != "missing"]
    if present:
        if len(present) != 1 or not _same_source(present[0], validated):
            raise ValueError(
                "Source identity already exists with different metadata/content or is ambiguous"
            )
        return ImportResult(
            version=1,
            status="already_present",
            novel_id=present[0]["novel_id"],
            provider=request.provider,
            source_id=provider_id,
            destination=Path(present[0]["source_path"]),
            chapter_numbers=request.expected_chapters,
            inventory_sha256=validated.inventory_sha256,
        )
    if len(matching) > 1:
        raise ValueError("Source identity has multiple missing Library records")

    if any(
        validated.source_dir == root_path.resolve()
        or validated.source_dir.is_relative_to(root_path.resolve())
        for root_path in library_roots
    ):
        raise ValueError("Source directory is already inside the configured Library root")
    if destination_root.is_symlink():
        raise ValueError("Library root must not be a symlink")
    if target.exists() or target.is_symlink():
        raise ValueError(f"Library destination already exists: {target}")

    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".novel-import-", dir=root))
    try:
        for chapter in validated.chapters:
            raw = chapter.raw
            if sha256(raw) != chapter.source_sha256:
                raise ValueError(f"Validated source snapshot changed: {chapter.filename}")
            write_bytes(stage / chapter.filename, raw)
        write_json(stage / "novel.json", _manifest(validated))
        staged_files = adapter.validate(
            stage,
            SourceImportRequest(
                provider=request.provider,
                source_id=request.source_id,
                title=request.title,
                expected_chapters=request.expected_chapters,
            ),
        )
        staged = candidate(stage, direct=False)
        if (
            staged_files.inventory_sha256 != validated.inventory_sha256
            or staged["status"] != "ready"
            or staged.get("source_type") != request.provider
            or staged.get("source_id", "").lower() != provider_id
            or staged.get("title") != request.title
            or staged.get("inventory_sha256") != validated.inventory_sha256
        ):
            raise ValueError("Staged source does not match the validated import request")
        current = adapter.validate(validated.source_dir, request)
        if current.inventory_sha256 != validated.inventory_sha256:
            raise ValueError("Source changed during import; nothing was published")
        write_json(
            stage / "source-import.json",
            {
                "version": 1,
                "provider": request.provider,
                "source_id": provider_id,
                "expected_chapters": list(request.expected_chapters),
                "inventory_sha256": validated.inventory_sha256,
            },
        )
        if target.exists() or target.is_symlink():
            raise ValueError(f"Library destination already exists: {target}")
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    published = candidate(target, direct=False)
    if (
        published["status"] != "ready"
        or published.get("source_type") != request.provider
        or published.get("source_id", "").lower() != provider_id
        or not _same_source(published, validated)
    ):
        raise ValueError("Published source failed Library validation")
    return ImportResult(
        version=1,
        status="imported",
        novel_id=published["novel_id"],
        provider=request.provider,
        source_id=provider_id,
        destination=target,
        chapter_numbers=request.expected_chapters,
        inventory_sha256=validated.inventory_sha256,
    )


def import_epub_source(
    source_path: Path,
    destination_root: Path,
    *,
    existing_sources: list[dict],
    library_roots: list[Path],
) -> ImportResult:
    """Stage a source EPUB and its protected text projection as one Library book."""
    if source_path.is_symlink():
        raise ValueError("Source EPUB must not be a symlink")
    resolved_source = source_path.resolve(strict=True)
    if not resolved_source.is_file():
        raise ValueError("Source EPUB is not a regular file")
    if resolved_source.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("Source EPUB exceeds the import size limit")
    if any(
        resolved_source == root.resolve() or resolved_source.is_relative_to(root.resolve())
        for root in library_roots
    ):
        raise ValueError("Source EPUB must be outside the configured Library roots")
    data = resolved_source.read_bytes()
    epub = inspect_source_epub(data)
    provider_id = epub.source_sha256
    root = destination_root.resolve()
    target = root / provider_id
    matching = [
        entry
        for entry in existing_sources
        if entry.get("source_type") == "epub"
        and isinstance(entry.get("source_id"), str)
        and entry["source_id"].lower() == provider_id
    ]
    present = [entry for entry in matching if entry.get("status") != "missing"]
    if present:
        expected = _validated_epub_inventory(epub)
        if len(present) != 1 or not _same_epub_source(present[0], expected):
            raise ValueError("Source EPUB identity already exists with different or ambiguous content")
        existing_path = Path(present[0]["source_path"])
        receipt = existing_path / "source-import.json"
        original = existing_path / "source.epub"
        try:
            receipt_data = json.loads(receipt.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Existing EPUB import receipt is invalid") from error
        if (
            not isinstance(receipt_data, dict)
            or receipt.is_symlink()
            or original.is_symlink()
            or not original.is_file()
        ):
            raise ValueError("Existing EPUB import receipt or source copy is invalid")
        if (
            receipt_data.get("version") != 1
            or receipt_data.get("provider") != "epub"
            or receipt_data.get("source_id") != provider_id
            or receipt_data.get("source_sha256") != provider_id
            or receipt_data.get("inventory_sha256") != expected["inventory_sha256"]
            or sha256(original.read_bytes()) != provider_id
        ):
            raise ValueError("Existing EPUB import receipt or source copy is invalid")
        return ImportResult(
            version=1,
            status="already_present",
            novel_id=present[0]["novel_id"],
            provider="epub",
            source_id=provider_id,
            destination=existing_path,
            chapter_numbers=tuple(document.number for document in epub.documents),
            inventory_sha256=present[0]["inventory_sha256"],
        )
    if len(matching) > 1:
        raise ValueError("Source EPUB identity has multiple missing Library records")
    if destination_root.is_symlink():
        raise ValueError("Library root must not be a symlink")
    if target.exists() or target.is_symlink():
        raise ValueError(f"Library destination already exists: {target}")

    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".novel-import-", dir=root))
    try:
        for document in epub.documents:
            write_bytes(stage / document.filename, document.text.encode("utf-8"))
        write_bytes(stage / "source.epub", data)
        write_json(
            stage / "novel.json",
            {
                "version": 1,
                "source": {"type": "epub", "id": provider_id},
                "title": epub.title,
                "author": epub.author,
                "language": "ja",
                "format": "txt-chapters",
                "chapter_pattern": "*.txt",
            },
        )
        staged = candidate(stage, direct=False)
        if (
            staged["status"] != "ready"
            or staged.get("source_type") != "epub"
            or staged.get("source_id") != provider_id
            or staged.get("title") != epub.title
            or sha256((stage / "source.epub").read_bytes()) != provider_id
        ):
            raise ValueError("Staged EPUB import does not match the validated source")
        write_json(
            stage / "source-import.json",
            {
                "version": 1,
                "provider": "epub",
                "source_id": provider_id,
                "source_sha256": provider_id,
                "original_file": "source.epub",
                "expected_chapters": [document.number for document in epub.documents],
                "inventory_sha256": staged["inventory_sha256"],
                "documents": [
                    {
                        "number": document.number,
                        "href": document.href,
                        "filename": document.filename,
                    }
                    for document in epub.documents
                ],
            },
        )
        if target.exists() or target.is_symlink():
            raise ValueError(f"Library destination already exists: {target}")
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    published = candidate(target, direct=False)
    if (
        published["status"] != "ready"
        or published.get("source_type") != "epub"
        or published.get("source_id") != provider_id
        or published.get("inventory_sha256") != staged.get("inventory_sha256")
        or sha256((target / "source.epub").read_bytes()) != provider_id
    ):
        raise ValueError("Published EPUB import failed Library or source validation")
    return ImportResult(
        version=1,
        status="imported",
        novel_id=published["novel_id"],
        provider="epub",
        source_id=provider_id,
        destination=target,
        chapter_numbers=tuple(document.number for document in epub.documents),
        inventory_sha256=published["inventory_sha256"],
    )


def _validated_epub_inventory(epub) -> dict:
    """Build the narrow source projection used to check idempotent imports."""
    inventory = tuple(
        {
            "number": document.number,
            "filename": document.filename,
            "source_sha256": sha256(document.text.encode("utf-8")),
        }
        for document in epub.documents
    )
    return {
        "status": "ready",
        "title": epub.title,
        "inventory_sha256": sha256(canonical(inventory)),
        "chapters": list(inventory),
    }


def _same_epub_source(existing: dict, validated: dict) -> bool:
    return (
        existing.get("status") == "ready"
        and existing.get("title") == validated["title"]
        and existing.get("inventory_sha256") == validated["inventory_sha256"]
        and existing.get("chapters") == validated["chapters"]
    )
