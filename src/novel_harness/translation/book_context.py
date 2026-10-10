"""Explicit source-backed book facts; never infer facts from a model's translation."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from ..artifacts import canonical, sha256
from ..config import StrictModel, _GlossaryLoader
from .discovery import discover


class Evidence(StrictModel):
    chapter: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=2000)


class Card(StrictModel):
    source_terms: list[str] = Field(min_length=1, max_length=20)
    name: str = Field(min_length=1, max_length=200)
    gender: Literal["female", "male", "unknown"] = "unknown"
    notes: str = Field(default="", max_length=1500)
    forms: list[str] = Field(default_factory=list, max_length=30)
    available_from: int = Field(default=1, gt=0)
    evidence: list[Evidence] = Field(min_length=1, max_length=20)


class ChapterSummary(StrictModel):
    before_chapter: int = Field(gt=1)
    text: str = Field(min_length=1, max_length=3000)
    evidence: list[Evidence] = Field(min_length=1, max_length=30)


class BookContext(StrictModel):
    version: Literal[1] = 1
    style: str = Field(default="", max_length=3000)
    cards: list[Card] = Field(default_factory=list, max_length=500)
    summaries: list[ChapterSummary] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def meaningful(self):
        def visit(value):
            if isinstance(value, str) and ("\ufffd" in value or "??" in value):
                raise ValueError("Damaged book context text")
            if isinstance(value, dict):
                for v in value.values():
                    visit(v)
            if isinstance(value, list):
                for v in value:
                    visit(v)

        visit(self.model_dump())
        names = [c.name for c in self.cards]
        chapters = [s.before_chapter for s in self.summaries]
        if len(names) != len(set(names)) or len(chapters) != len(set(chapters)):
            raise ValueError("Duplicate book card/summary")
        for card in self.cards:
            if any(not t.strip() for t in card.source_terms):
                raise ValueError("Empty card source term")
            if any(e.chapter > card.available_from for e in card.evidence):
                raise ValueError("Card contains future evidence")
        for summary in self.summaries:
            if any(e.chapter >= summary.before_chapter for e in summary.evidence):
                raise ValueError("Summary contains future evidence")
        return self


#: Book-bible trust model. Only ``confirmed``/``locked`` are authoritative; a
#: ``proposed`` (for example model-generated) entry never reaches a translation
#: prompt on its own.
TRUST_STATES = ("proposed", "confirmed", "locked")
AUTHORITATIVE_TRUST = ("confirmed", "locked")
ENTRY_KINDS = (
    "character",
    "place",
    "organization",
    "term",
    "title",
    "name",
    "expression",
    "relationship",
)


def authoritative_projection(entries: list[dict]) -> list[dict]:
    """Deterministic projection of only the authoritative trust levels."""
    rows = [
        {
            "kind": entry["kind"],
            "source_form": entry["source_form"],
            "target_form": entry["target_form"],
            "aliases": sorted(entry.get("aliases", [])),
        }
        for entry in entries
        if entry["trust"] in AUTHORITATIVE_TRUST
    ]
    rows.sort(key=lambda row: (row["kind"], row["source_form"], row["target_form"]))
    return rows


def projection_sha256(projection: list[dict]) -> str:
    return sha256(canonical(projection))


def book_bible_block(entries: list[dict]) -> dict:
    """Frozen, per-run snapshot of the authoritative book-bible context."""
    projection = authoritative_projection(entries)
    return {
        "version": 1,
        "policy": "confirmed+locked",
        "authoritative_trust": list(AUTHORITATIVE_TRUST),
        "entries": projection,
        "projection_sha256": projection_sha256(projection),
    }


def bible_context_for(bible: dict, source: str) -> dict:
    """Terms whose source form or alias occurs in this chunk's source text."""
    if not bible:
        return {}
    terms = [
        {
            "kind": entry["kind"],
            "source_form": entry["source_form"],
            "target_form": entry["target_form"],
        }
        for entry in bible.get("entries", [])
        if entry["source_form"] in source
        or any(alias in source for alias in entry.get("aliases", []))
    ]
    return {"terms": terms} if terms else {}


def freeze_book_context(path: Path | None, source_dir: Path) -> dict:
    if path is None:
        return {}
    book = BookContext.model_validate(yaml.load(path.read_text("utf-8"), Loader=_GlossaryLoader))
    sources = {c.number: c for c in discover(source_dir)}
    used = {}
    for entry in [*book.cards, *book.summaries]:
        for evidence in entry.evidence:
            chapter = sources.get(evidence.chapter)
            if chapter is None or evidence.quote not in chapter.text:
                raise ValueError(f"Book evidence not found in source chapter {evidence.chapter}")
            used[chapter.number] = {"filename": chapter.filename, "sha256": chapter.source_sha256}
        if isinstance(entry, Card) and not any(
            term in e.quote for term in entry.source_terms for e in entry.evidence
        ):
            raise ValueError("Card evidence must identify its source term")
    return {**book.model_dump(), "evidence_sources": list(used.values())}


def verify_book_context(book: dict, source_dir: Path):
    if not book:
        return
    BookContext.model_validate({k: v for k, v in book.items() if k != "evidence_sources"})
    for item in book["evidence_sources"]:
        path = source_dir / item["filename"]
        if not path.is_file() or sha256(path.read_bytes()) != item["sha256"]:
            raise ValueError("Book context evidence source changed")


def context_for(book: dict, chapter: int, source: str, max_chars: int) -> dict:
    if not book:
        return {}
    cards = [
        {k: c[k] for k in ("name", "source_terms", "gender", "forms", "notes")}
        for c in book["cards"]
        if c["available_from"] <= chapter and any(term in source for term in c["source_terms"])
    ]
    result = {
        "style": book["style"],
        "cards": cards,
        "previous_chapter_summary": next(
            (s["text"] for s in book["summaries"] if s["before_chapter"] == chapter), ""
        ),
    }
    if len(canonical(result)) > max_chars:
        raise ValueError(
            "Relevant book context exceeds budget; shorten cards or increase book_context_chars"
        )
    return result
