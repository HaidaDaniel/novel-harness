"""Cross-run content-addressed cache for validated stage output.

Entries are keyed by the stage input hash, which already covers the prompt,
source, context, glossary and runner profile (including model), so a model or
prompt change yields a different key. Corrupt, partial or hash-mismatched
entries are treated as misses. Cached text is re-validated by the caller; the
cache never bypasses deterministic validation and never overwrites a valid
entry.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock

from .artifacts import canonical, sha256

SCHEMA_VERSION = 1


def _publish(path: Path, entry: dict) -> None:
    """Atomically write ``entry`` to ``path`` via a temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _lock(path: Path) -> FileLock:
    return FileLock(str(path) + ".lock")


def semantic_cache_key(input_hash: str, semantic_inputs: dict) -> str:
    """Exact reuse key covering the attempt input hash plus explicit semantics.

    ``input_hash`` already covers source, context (including the book bible),
    glossary, prompt and the runner profile/model. The explicit fields make the
    remaining semantics-changing inputs (name rules, book-bible/glossary/prompt
    hashes and decoding settings) part of the key, so any change forces a miss.
    """
    return sha256(canonical({"input_hash": input_hash, **semantic_inputs}))


class TranslationCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        try:
            raw = self._path(key).read_text("utf-8")
        except OSError:
            return None
        try:
            entry = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(entry, dict) or entry.get("schema_version") != SCHEMA_VERSION:
            return None
        text = entry.get("text")
        if not isinstance(text, str) or not text:
            return None
        if entry.get("output_sha256") != sha256(text):
            return None
        return entry

    def put(
        self,
        key: str,
        *,
        text: str,
        model: str | None,
        finish_reason: str | None,
        usage: dict,
        effective_settings: dict,
    ) -> None:
        if not text:
            return
        path = self._path(key)
        entry = {
            "schema_version": SCHEMA_VERSION,
            "text": text,
            "output_sha256": sha256(text),
            "model": model,
            "finish_reason": finish_reason,
            "usage": usage,
            "effective_settings": effective_settings,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock(path):
            # Insert-only: a valid existing entry is never overwritten. A corrupt
            # or partial entry is recovered by this write.
            if self.get(key) is not None:
                return
            _publish(path, entry)


class TranslationMemory:
    """Durable, user-approved translation memory with stronger provenance.

    Distinct from the automatic machine cache: entries are written only by an
    explicit promote action, a valid entry is never overwritten (a corrupt or
    partial entry is recovered by the next write), and entries carry the
    approving actor, the source run and the approved final hash. Exact-fingerprint
    reuse only; there is no fuzzy/CAT matching.
    """

    SCHEMA_VERSION = 1

    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        try:
            raw = self._path(key).read_text("utf-8")
        except OSError:
            return None
        try:
            entry = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(entry, dict) or entry.get("schema_version") != self.SCHEMA_VERSION:
            return None
        text = entry.get("text")
        if not isinstance(text, str) or not text:
            return None
        if entry.get("output_sha256") != sha256(text):
            return None
        return entry

    def put(self, key: str, *, text: str, provenance: dict) -> bool:
        """Insert an approved entry; a valid existing entry is never overwritten."""
        if not text:
            return False
        path = self._path(key)
        entry = {
            "schema_version": self.SCHEMA_VERSION,
            "fingerprint": key,
            "text": text,
            "output_sha256": sha256(text),
            "provenance": provenance,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock(path):
            # Cross-process serialized check-and-write: exactly one concurrent
            # insertion of an absent key returns True; a valid existing entry is
            # byte-identical and never overwritten. A corrupt/partial entry is
            # recovered by this write.
            if self.get(key) is not None:
                return False
            _publish(path, entry)
            return True

    def entries(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        found = []
        for path in sorted(self.root.glob("*/*.json")):
            entry = self.get(path.stem)
            if entry is not None:
                found.append(entry)
        return found
