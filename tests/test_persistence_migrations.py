"""Ordered, idempotent SQLite schema migration tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from novel_harness.migrations import (
    SCHEMA_VERSION,
    SchemaVersionError,
    current_version,
    migrate,
)


def _tables(path: Path) -> set[str]:
    db = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        db.close()


def _version(path: Path) -> int:
    db = sqlite3.connect(path)
    try:
        return current_version(db)
    finally:
        db.close()


def test_fresh_database_migrates_to_latest(tmp_path):
    path = tmp_path / "state.sqlite3"

    assert migrate(path) == SCHEMA_VERSION
    assert {"runs", "chunks", "attempts"}.issubset(_tables(path))
    assert _version(path) == SCHEMA_VERSION


def test_existing_current_database_is_not_rewritten(tmp_path):
    path = tmp_path / "state.sqlite3"
    migrate(path)

    db = sqlite3.connect(path)
    try:
        db.execute(
            "INSERT INTO runs(id,fingerprint,status,created_at,updated_at,manifest) "
            "VALUES('run-1','fp','pending','t','t','{}')"
        )
        db.commit()
    finally:
        db.close()

    assert migrate(path) == SCHEMA_VERSION

    db = sqlite3.connect(path)
    try:
        rows = db.execute("SELECT id, status FROM runs").fetchall()
    finally:
        db.close()
    assert rows == [("run-1", "pending")]


def test_upgrade_preserves_existing_rows(tmp_path):
    path = tmp_path / "state.sqlite3"
    db = sqlite3.connect(path)
    try:
        db.executescript(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, manifest TEXT NOT NULL,
                error TEXT, duration_seconds REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX runs_fingerprint ON runs(fingerprint);
            CREATE TABLE chunks (
                run_id TEXT NOT NULL REFERENCES runs(id), chapter INTEGER NOT NULL,
                chunk_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                source_sha256 TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                output_path TEXT, output_sha256 TEXT, metadata TEXT,
                PRIMARY KEY(run_id, chapter, chunk_id)
            );
            CREATE TABLE attempts (
                run_id TEXT NOT NULL REFERENCES runs(id), chapter INTEGER NOT NULL,
                chunk_id TEXT NOT NULL, stage TEXT NOT NULL, number INTEGER NOT NULL,
                status TEXT NOT NULL, input_sha256 TEXT NOT NULL, metadata TEXT NOT NULL,
                PRIMARY KEY(run_id, chapter, chunk_id, stage, number)
            );
            INSERT INTO runs(id,fingerprint,status,created_at,updated_at,manifest)
                VALUES('run-1','fp','completed','t','t','{}');
            INSERT INTO chunks(run_id,chapter,chunk_id,ordinal,source_sha256)
                VALUES('run-1',1,'0001-abc',1,'sha');
            INSERT INTO attempts(run_id,chapter,chunk_id,stage,number,status,input_sha256,metadata)
                VALUES('run-1',1,'0001-abc','translate',1,'completed','in','{}');
            """
        )
        db.commit()
    finally:
        db.close()

    assert _version(path) == 0
    assert migrate(path) == SCHEMA_VERSION

    db = sqlite3.connect(path)
    try:
        counts = (
            db.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
        )
        status = db.execute("SELECT status FROM runs WHERE id='run-1'").fetchone()[0]
    finally:
        db.close()

    assert counts == (1, 1, 1)
    assert status == "completed"


def test_unversioned_database_gains_schema(tmp_path):
    path = tmp_path / "state.sqlite3"
    sqlite3.connect(path).close()

    assert _version(path) == 0
    assert migrate(path) == SCHEMA_VERSION
    assert {"runs", "chunks", "attempts"}.issubset(_tables(path))


def test_newer_unknown_version_fails_closed(tmp_path):
    path = tmp_path / "state.sqlite3"

    db = sqlite3.connect(path)
    try:
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
        db.commit()
    finally:
        db.close()

    with pytest.raises(SchemaVersionError):
        migrate(path)

    assert _version(path) == SCHEMA_VERSION + 1
    assert _tables(path) == set()


def test_failed_migration_rolls_back_and_can_retry(tmp_path):
    path = tmp_path / "state.sqlite3"

    def boom(db):
        db.execute("CREATE TABLE rollback_probe (id INTEGER)")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        migrate(path, [(1, boom)])

    assert _version(path) == 0
    assert "rollback_probe" not in _tables(path)

    assert migrate(path) == SCHEMA_VERSION
    assert "rollback_probe" not in _tables(path)


def test_repeated_migration_is_idempotent(tmp_path):
    path = tmp_path / "state.sqlite3"

    first = migrate(path)
    second = migrate(path)

    assert first == second == SCHEMA_VERSION


def test_book_bible_migration_adds_trust_table(tmp_path):
    path = tmp_path / "state.sqlite3"

    assert migrate(path) == SCHEMA_VERSION == 4
    assert {
        "runs", "chunks", "attempts", "final_layer", "book_bible", "library_sources"
    }.issubset(_tables(path))

    db = sqlite3.connect(path)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(book_bible)")}
    finally:
        db.close()
    assert {
        "entry_id", "kind", "source_form", "target_form", "aliases", "trust",
        "notes", "source_anchors", "provenance", "created_at", "updated_at",
    }.issubset(columns)


def test_final_layer_migration_adds_durable_table(tmp_path):
    path = tmp_path / "state.sqlite3"

    assert migrate(path) == SCHEMA_VERSION == 4
    assert {"runs", "chunks", "attempts", "final_layer", "library_sources"}.issubset(
        _tables(path)
    )

    db = sqlite3.connect(path)
    try:
        columns = {
            row[1] for row in db.execute("PRAGMA table_info(final_layer)")
        }
    finally:
        db.close()
    assert {"run_id", "chapter", "chunk_id", "status", "final_text",
            "final_sha256", "machine_sha256", "actor", "reason", "updated_at"}.issubset(
        columns
    )
