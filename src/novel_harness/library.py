"""Read-only, one-level source discovery for the filesystem-backed Novel Library."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, ValidationError, model_validator

from .artifacts import canonical, sha256
from .config import StrictModel
from .translation.discovery import discover

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAFE_PATTERN = re.compile(r"[A-Za-z0-9_*?.-]+\.txt\Z", re.IGNORECASE)


def valid_pattern(pattern: str) -> str:
    """A chapter glob may select names in one directory, never traverse paths."""
    if not _SAFE_PATTERN.fullmatch(pattern) or ".." in pattern or "**" in pattern:
        raise ValueError("chapter_pattern must be a safe non-recursive .txt glob")
    return pattern


class SourceIdentity(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: str
    id: str

    @model_validator(mode="after")
    def valid_identity(self):
        if not _SAFE_ID.fullmatch(self.type) or not _SAFE_ID.fullmatch(self.id):
            raise ValueError("source type/id must be safe identifiers")
        return self


class SourceManifest(StrictModel):
    """Versioned optional ``novel.json`` contract; unknown fields fail closed."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    id: str | None = None
    source: SourceIdentity | None = None
    title: str = Field(min_length=1, max_length=500)
    author: str | None = Field(default=None, max_length=500)
    language: Literal["ja"] = "ja"
    format: Literal["txt-chapters"] = "txt-chapters"
    chapter_pattern: str = "*.txt"

    @model_validator(mode="after")
    def valid_metadata(self):
        if self.source is None and self.id is None:
            raise ValueError("novel.json requires id or source type/id")
        if self.id is not None and not _SAFE_ID.fullmatch(self.id):
            raise ValueError("manifest id must be a safe identifier")
        for value in (self.title, self.author):
            if value is not None and (
                value != value.strip()
                or "\ufffd" in value
                or any(ord(char) < 32 for char in value)
            ):
                raise ValueError("manifest title/author must be intact single-line text")
        valid_pattern(self.chapter_pattern)
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate novel.json field: {key}")
        value[key] = item
    return value


def _read_manifest(path: Path) -> SourceManifest | None:
    file = path / "novel.json"
    if not file.exists() and not file.is_symlink():
        return None
    if file.is_symlink() or not file.is_file():
        raise ValueError("novel.json must be a regular file")
    if file.stat().st_size > 64_000:
        raise ValueError("novel.json exceeds 64 KB")
    value = json.loads(file.read_text("utf-8"), object_pairs_hook=_unique_object)
    return SourceManifest.model_validate(value)


def candidate(
    path: Path,
    *,
    direct: bool,
    fallback_title: str | None = None,
    fallback_pattern: str = "*.txt",
) -> dict:
    """Scan exactly one chapter directory; errors are per-book diagnostics."""
    result = {
        "source_path": str(path),
        "novel_id": None,
        "identity_kind": None,
        "inventory_sha256": None,
        "title": fallback_title or path.name,
        "author": None,
        "language": "ja",
        "format": "txt-chapters",
        "source_type": None,
        "source_id": None,
        "chapter_pattern": "*.txt",
        "chapters": [],
        "status": "ready",
        "error": None,
    }
    try:
        if path.is_symlink():
            raise ValueError("Source directory must not be a symlink")
        manifest = _read_manifest(path)
        pattern = manifest.chapter_pattern if manifest else fallback_pattern
        for item in path.iterdir():
            if item.is_symlink() and item.suffix.lower() == ".txt":
                raise ValueError(f"Chapter must be a regular file: {item.name}")
        chapters = discover(path, pattern)
        if len(chapters) < (1 if manifest or direct else 2):
            raise ValueError("Legacy nested novel requires at least two valid chapters")
        inventory = [
            {"number": c.number, "filename": c.filename, "source_sha256": c.source_sha256}
            for c in chapters
        ]
        inventory_sha256 = sha256(canonical(inventory))
        if manifest and manifest.source:
            identity_kind = "source"
            key = f"{manifest.source.type}:{manifest.source.id}"
            novel_id = f"source-{sha256(key)[:24]}"
        elif manifest:
            identity_kind = "manifest"
            novel_id = f"manifest-{sha256(manifest.id)[:24]}"
        else:
            identity_kind = "legacy"
            novel_id = f"legacy-{inventory_sha256[:24]}"
        result.update(
            novel_id=novel_id,
            identity_kind=identity_kind,
            inventory_sha256=inventory_sha256,
            title=manifest.title if manifest else result["title"],
            author=manifest.author if manifest else None,
            language=manifest.language if manifest else "ja",
            format=manifest.format if manifest else "txt-chapters",
            source_type=manifest.source.type if manifest and manifest.source else None,
            source_id=manifest.source.id if manifest and manifest.source else None,
            chapter_pattern=pattern,
            chapters=inventory,
        )
    except ValidationError:
        # Pydantic includes input values in its default message. Manifests may
        # contain accidental credentials in unknown fields, so keep diagnostics
        # useful without copying user-provided values into cards or SQLite.
        result.update(status="invalid", error="novel.json does not match schema version 1")
    except json.JSONDecodeError as error:
        result.update(
            status="invalid",
            error=f"novel.json has invalid JSON at line {error.lineno}, column {error.colno}",
        )
    except UnicodeError:
        result.update(status="invalid", error="novel.json must be valid UTF-8")
    except (OSError, ValueError) as error:
        result.update(status="invalid", error=str(error))
    return result


def scan_roots(
    roots: list[Path],
    *,
    direct_titles: dict[str, str] | None = None,
    direct_patterns: dict[str, str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Inspect configured roots and immediate child directories only."""
    found: dict[str, dict] = {}
    diagnostics: list[str] = []
    for root in roots:
        if not root.is_dir():
            diagnostics.append(f"Library root does not exist: {root}")
            continue
        try:
            entries = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError as error:
            diagnostics.append(f"Cannot read library root {root}: {error}")
            continue
        if any(item.suffix.lower() == ".txt" or item.name == "novel.json" for item in entries):
            found[str(root)] = candidate(
                root,
                direct=True,
                fallback_title=(direct_titles or {}).get(str(root)),
                fallback_pattern=(direct_patterns or {}).get(str(root), "*.txt"),
            )
        for item in entries:
            if item.is_symlink():
                diagnostics.append(f"Skipped symlinked library entry: {item}")
                continue
            if item.name.startswith(".novel-import-"):
                continue
            if not item.is_dir():
                continue
            try:
                names = {child.name for child in item.iterdir()}
            except OSError as error:
                diagnostics.append(f"Cannot read library entry {item}: {error}")
                continue
            if "novel.json" in names or any(name.lower().endswith(".txt") for name in names):
                found[str(item)] = candidate(item, direct=False)
    return [found[key] for key in sorted(found)], diagnostics


def project_library(previous: dict[str, dict], candidates: list[dict]) -> tuple[list[dict], set[str]]:
    """Pure reconciliation shared by read-only listing and explicit index writes."""
    scanned = {entry["source_path"] for entry in candidates}
    moved: set[str] = set()
    legacy_new: dict[str, list[dict]] = {}
    legacy_current: dict[str, list[dict]] = {}
    legacy_missing: dict[str, list[dict]] = {}
    for entry in candidates:
        old = previous.get(entry["source_path"])
        if entry["status"] == "ready" and entry["identity_kind"] == "legacy":
            inventory = entry["inventory_sha256"]
            legacy_current.setdefault(inventory, []).append(entry)
            if not (old and old["identity_kind"] == "legacy"):
                legacy_new.setdefault(inventory, []).append(entry)
    for path, item in previous.items():
        if (
            path not in scanned
            and item["identity_kind"] == "legacy"
            and item.get("inventory_sha256")
        ):
            legacy_missing.setdefault(item["inventory_sha256"], []).append(item)

    legacy_moves: dict[str, dict] = {}
    ambiguous_legacy_paths: set[str] = set()
    for inventory, new_entries in legacy_new.items():
        old_entries = legacy_missing.get(inventory, [])
        if not old_entries:
            continue
        if (
            len(new_entries) == len(old_entries) == 1
            and len(legacy_current.get(inventory, [])) == 1
        ):
            legacy_moves[new_entries[0]["source_path"]] = old_entries[0]
            moved.add(old_entries[0]["source_path"])
        else:
            ambiguous_legacy_paths.update(entry["source_path"] for entry in new_entries)

    other_new: dict[str, list[dict]] = {}
    other_current: dict[str, list[dict]] = {}
    other_missing: dict[str, list[dict]] = {}
    for entry in candidates:
        old = previous.get(entry["source_path"])
        if (
            entry["status"] == "ready"
            and entry["identity_kind"] != "legacy"
            and entry["novel_id"]
        ):
            other_current.setdefault(entry["novel_id"], []).append(entry)
            if not old:
                other_new.setdefault(entry["novel_id"], []).append(entry)
    for path, item in previous.items():
        if path not in scanned and item["identity_kind"] != "legacy" and item.get("novel_id"):
            other_missing.setdefault(item["novel_id"], []).append(item)
    for novel_id, new_entries in other_new.items():
        old_entries = other_missing.get(novel_id, [])
        if (
            len(new_entries) == len(old_entries) == 1
            and len(other_current.get(novel_id, [])) == 1
        ):
            moved.add(old_entries[0]["source_path"])

    current: list[dict] = []
    for raw in candidates:
        entry = dict(raw)
        old = previous.get(entry["source_path"])
        if entry["status"] == "invalid" and old:
            entry["novel_id"] = old["novel_id"]
            entry["identity_kind"] = old["identity_kind"]
        elif entry["identity_kind"] == "legacy":
            if old and old["identity_kind"] == "legacy":
                entry["novel_id"] = old["novel_id"]
            elif entry["source_path"] in legacy_moves:
                entry["novel_id"] = legacy_moves[entry["source_path"]]["novel_id"]
        if entry["source_path"] in ambiguous_legacy_paths:
            entry["status"] = "conflict"
            entry["error"] = "Ambiguous legacy source move; identity was not transferred"
        current.append(entry)

    legacy_counts: dict[str, int] = {}
    for entry in current:
        if entry["status"] == "ready" and entry["identity_kind"] == "legacy":
            inventory = entry["inventory_sha256"]
            legacy_counts[inventory] = legacy_counts.get(inventory, 0) + 1
    for entry in current:
        if (
            entry["status"] == "ready"
            and entry["identity_kind"] == "legacy"
            and legacy_counts.get(entry["inventory_sha256"], 0) > 1
        ):
            entry["status"] = "conflict"
            entry["error"] = "Duplicate legacy chapter inventory in multiple directories"

    counts: dict[str, int] = {}
    for entry in current:
        if entry["status"] == "ready" and entry["novel_id"]:
            key = entry["novel_id"]
            counts[key] = counts.get(key, 0) + 1
    for entry in current:
        if entry["status"] == "ready" and counts.get(entry["novel_id"], 0) > 1:
            entry["status"] = "conflict"
            entry["error"] = "Duplicate source identity in multiple directories"
    for path, old in previous.items():
        if path not in scanned and path not in moved:
            current.append({
                **old,
                "status": "missing",
                "error": "Source directory no longer discovered",
            })
    return sorted(current, key=lambda item: (item["title"].lower(), item["source_path"])), moved
