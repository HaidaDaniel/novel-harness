"""A translated title is a shared book input, never a per-chunk model task."""

import json
import re
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from .artifacts import canonical, sha256
from .config import NovelConfig, StrictModel, _GlossaryLoader


class BookMetadata(StrictModel):
    source_title: str = Field(min_length=1, max_length=2000)
    title: str = Field(min_length=1, max_length=2000)
    author: str | None = None

    @model_validator(mode="after")
    def valid_text(self):
        for value in (self.source_title, self.title, self.author):
            if value is not None and (
                not value.strip()
                or value != value.strip()
                or "\ufffd" in value
                or "??" in value
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("Book metadata must contain intact, nonempty single-line text")
        if not re.search(r"[А-Яа-яЁё]", self.title) or re.search(
            r"[\u3040-\u30ff\u3400-\u9fff]", self.title
        ):
            raise ValueError("Translate the book title into Russian once before preparing runs")
        return self


def load_book_metadata(path: Path) -> dict:
    try:
        value = yaml.load(path.read_text("utf-8"), Loader=_GlossaryLoader)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError("Book metadata must be valid UTF-8 YAML") from error
    return BookMetadata.model_validate(value).model_dump()


def freeze_novel(config: NovelConfig) -> dict:
    if config.metadata_path is not None:
        return load_book_metadata(config.metadata_path)
    return {"title": config.title, "author": config.author}


def verify_novel(manifest: dict):
    if "novel_sha256" in manifest:
        if sha256(canonical(manifest["novel"])) != manifest["novel_sha256"]:
            raise ValueError("Run book metadata hash mismatch")
        if "source_title" in manifest["novel"]:
            BookMetadata.model_validate(manifest["novel"])


def export_novel(
    manifest: dict, metadata_path: Path | None = None, previous: dict | None = None
) -> dict:
    verify_novel(manifest)
    if metadata_path is None:
        saved = (previous or {}).get("novel", {})
        if "source_title" in manifest["novel"] or "source_title" not in saved:
            return dict(manifest["novel"])
        if sha256(canonical(saved)) != previous.get("novel_sha256"):
            raise ValueError("Saved export book metadata hash mismatch")
        novel = BookMetadata.model_validate(saved).model_dump()
    else:
        novel = load_book_metadata(metadata_path)
    frozen = manifest["novel"]
    if frozen.get("source_title", frozen["title"]) not in {novel["source_title"], novel["title"]}:
        raise ValueError("Book metadata belongs to another source title")
    return novel


def preserve_book_record(directory: Path):
    """Keep old export metadata when moving the latest-export pointer."""
    path = directory / "book.json"
    if path.is_file():
        from .artifacts import write_bytes

        raw = path.read_bytes()
        json.loads(raw)  # Refuse to silently replace an unreadable previous record.
        history = directory / "book-history" / f"{sha256(raw)}.json"
        if not history.exists():
            write_bytes(history, raw)
