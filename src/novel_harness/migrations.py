"""Ordered, idempotent SQLite schema migrations.

The on-disk schema is versioned with SQLite's ``PRAGMA user_version``. Each
migration upgrades exactly one step and runs inside a transaction, so a failed
or interrupted migration rolls back and can be retried. A database whose
``user_version`` is newer than this code understands fails closed and is never
mutated.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

SCHEMA_VERSION = 4

Migration = tuple[int, Callable[[sqlite3.Connection], None]]


class SchemaVersionError(RuntimeError):
    """The database schema is newer than this code understands."""


def _to_v1(db: sqlite3.Connection) -> None:
    """Initial schema: runs, chunks and attempts."""
    for statement in (
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, manifest TEXT NOT NULL,
            error TEXT, duration_seconds REAL NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS runs_fingerprint ON runs(fingerprint)",
        """
        CREATE TABLE IF NOT EXISTS chunks (
            run_id TEXT NOT NULL REFERENCES runs(id), chapter INTEGER NOT NULL,
            chunk_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            source_sha256 TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            output_path TEXT, output_sha256 TEXT, metadata TEXT,
            PRIMARY KEY(run_id, chapter, chunk_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS attempts (
            run_id TEXT NOT NULL REFERENCES runs(id), chapter INTEGER NOT NULL,
            chunk_id TEXT NOT NULL, stage TEXT NOT NULL, number INTEGER NOT NULL,
            status TEXT NOT NULL, input_sha256 TEXT NOT NULL, metadata TEXT NOT NULL,
            PRIMARY KEY(run_id, chapter, chunk_id, stage, number)
        )
        """,
    ):
        db.execute(statement)


def default_migrations() -> list[Migration]:
    return [(1, _to_v1), (2, _to_v2), (3, _to_v3), (4, _to_v4)]


def _to_v4(db: sqlite3.Connection) -> None:
    """Filesystem-backed Novel Library index; never stores chapter bodies."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS library_sources (
            source_path TEXT PRIMARY KEY,
            novel_id TEXT,
            identity_kind TEXT,
            inventory_sha256 TEXT,
            title TEXT NOT NULL,
            author TEXT,
            language TEXT,
            format TEXT,
            source_type TEXT,
            source_id TEXT,
            chapter_pattern TEXT NOT NULL,
            chapters_json TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS library_sources_id ON library_sources(novel_id)")


def ensure_library_index(db: sqlite3.Connection) -> None:
    """Recreate the disposable library projection if its table was removed."""
    _to_v4(db)


def _to_v3(db: sqlite3.Connection) -> None:
    """Durable, editable Book Bible with trust states and provenance."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS book_bible (
            entry_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            source_form TEXT NOT NULL,
            target_form TEXT NOT NULL,
            aliases TEXT NOT NULL,
            trust TEXT NOT NULL,
            notes TEXT,
            source_anchors TEXT NOT NULL,
            provenance TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS book_bible_trust ON book_bible(trust)")


def _to_v2(db: sqlite3.Connection) -> None:
    """Durable human final layer, separate from the raw machine candidate."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS final_layer (
            run_id TEXT NOT NULL REFERENCES runs(id), chapter INTEGER NOT NULL,
            chunk_id TEXT NOT NULL, status TEXT NOT NULL,
            final_text TEXT, final_sha256 TEXT, machine_sha256 TEXT NOT NULL,
            actor TEXT NOT NULL, reason TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY(run_id, chapter, chunk_id)
        )
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS final_layer_run ON final_layer(run_id)")


def current_version(db: sqlite3.Connection) -> int:
    row = db.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def migrate(path: Path, migrations: Sequence[Migration] | None = None) -> int:
    """Bring ``path`` up to the latest known schema version and return it.

    Every pending step is applied in its own transaction; a failing step rolls
    back so ``user_version`` and the schema stay consistent. Repeating the call
    after success is a no-op.
    """
    steps = sorted(migrations if migrations is not None else default_migrations(), key=lambda item: item[0])
    db = sqlite3.connect(path, timeout=30)
    try:
        # Read and guard the version before any mutating pragma so an unknown
        # newer database is refused without touching its file state.
        version = current_version(db)
        if version > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Database schema version {version} is newer than supported "
                f"version {SCHEMA_VERSION}; refusing to migrate."
            )
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        for target, step in steps:
            if target <= version:
                continue
            db.execute("BEGIN")
            try:
                step(db)
                db.execute(f"PRAGMA user_version={target}")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
            version = target
        return version
    finally:
        db.close()
