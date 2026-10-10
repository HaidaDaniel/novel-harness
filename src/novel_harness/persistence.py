"""SQLite is the authoritative state index; large payloads remain regular files."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from .artifacts import canonical
from .library import project_library
from .migrations import ensure_library_index, migrate


def _selector_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _reject_path_like_selector(value: str) -> None:
    if (
        not value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or value.startswith((".", "~"))
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError("Run selector must be an ID or a non-path prefix")


def _selector_matches(db, selector: str, *, limit: int) -> list[str]:
    escaped = _selector_like(selector)
    patterns = [f"{escaped}%"]
    parameters: list[str | int] = [patterns[0]]
    if len(selector) >= 4:
        patterns.append(f"%-{escaped}%")
        parameters.append(patterns[1])
    placeholders = " OR ".join("id LIKE ? ESCAPE '\\'" for _ in patterns)
    parameters.append(limit)
    rows = db.execute(
        f"SELECT id FROM runs WHERE {placeholders} ORDER BY id LIMIT ?", parameters
    ).fetchall()
    return [row[0] for row in rows]


def _resolve_run_selector(db, selector: str, *, ambiguity_limit: int = 6) -> str:
    _reject_path_like_selector(selector)
    exact = db.execute("SELECT id FROM runs WHERE id=?", (selector,)).fetchone()
    if exact is not None:
        return exact[0]
    if len(selector) < 4:
        raise ValueError("Run selector must be a full ID or at least four characters")
    matches = _selector_matches(db, selector, limit=ambiguity_limit)
    if not matches:
        raise ValueError(f"Unknown run selector: {selector}")
    if len(matches) > 1:
        shown = ", ".join(matches[:5])
        suffix = ", and more" if len(matches) > 5 else ""
        raise ValueError(f"Ambiguous run selector '{selector}'; choose: {shown}{suffix}")
    return matches[0]


def completion_rows(path: Path, incomplete: str, *, limit: int = 20) -> list[dict]:
    """Read bounded run suggestions from an existing SQLite store without migration."""
    if not path.is_file():
        return []
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            escaped = _selector_like(incomplete)
            patterns = [f"{escaped}%"]
            parameters: list[str | int] = [patterns[0]]
            if len(incomplete) >= 4:
                patterns.append(f"%-{escaped}%")
                parameters.append(patterns[1])
            placeholders = " OR ".join("r.id LIKE ? ESCAPE '\\'" for _ in patterns)
            parameters.append(limit)
            rows = db.execute(
                "SELECT r.id, r.status, "
                "json_extract(r.manifest,'$.novel.title') AS title "
                "FROM runs r "
                f"WHERE {placeholders} ORDER BY r.id LIMIT ?",
                parameters,
            ).fetchall()
            return [dict(row) for row in rows]
    except (OSError, sqlite3.Error, ValueError):
        return []


class Repository:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        migrate(path)
        with sqlite3.connect(path) as db:
            ensure_library_index(db)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, manifest: dict, chunks: list[dict], *, initial_outputs: Sequence[dict] = ()):
        with self.connection() as db:
            db.execute(
                "INSERT INTO runs(id,fingerprint,status,created_at,updated_at,manifest) "
                "VALUES(?,?,'pending',?,?,?)",
                (
                    manifest["run_id"],
                    manifest["fingerprint"],
                    manifest["created_at"],
                    manifest["created_at"],
                    canonical(manifest),
                ),
            )
            db.executemany(
                "INSERT INTO chunks(run_id,chapter,chunk_id,ordinal,source_sha256) "
                "VALUES(?,?,?,?,?)",
                [
                    (manifest["run_id"], c["chapter"], c["id"], c["index"], c["source_sha256"])
                    for c in chunks
                ],
            )
            db.executemany(
                "UPDATE chunks SET status='completed',output_path=?,output_sha256=?,metadata=? "
                "WHERE run_id=? AND chapter=? AND chunk_id=?",
                [
                    (
                        r["output_path"],
                        r["output_sha256"],
                        canonical(r["metadata"]),
                        manifest["run_id"],
                        r["chapter"],
                        r["chunk_id"],
                    )
                    for r in initial_outputs
                ],
            )

    def matching(self, fingerprint: str) -> str | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT id FROM runs WHERE fingerprint=? ORDER BY created_at DESC LIMIT 1",
                (fingerprint,),
            ).fetchone()
            return row[0] if row else None

    def run(self, run_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown run: {run_id}")
            result = dict(row)
            result["manifest"] = json.loads(result["manifest"])
            return result

    def replace_manifest(self, run_id: str, manifest: dict) -> None:
        """Persist an additive manifest update without changing run identity."""
        with self.connection() as db:
            updated = db.execute(
                "UPDATE runs SET manifest=? WHERE id=?",
                (canonical(manifest), run_id),
            ).rowcount
            if updated != 1:
                raise ValueError(f"Unknown run: {run_id}")

    def resolve_run_selector(self, selector: str) -> str:
        """Resolve an exact, unique full-ID or hash-component prefix in this store."""
        with self.connection() as db:
            return _resolve_run_selector(db, selector)

    def list_runs(self) -> list[dict]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC,id DESC"
            ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["manifest"] = json.loads(value["manifest"])
            result.append(value)
        return result

    def list_run_summaries(self, run_ids: Sequence[str] | None = None) -> list[dict]:
        """Return narrow run metadata and persisted chunk counters.

        Interactive views must not decode the full manifest or inspect output
        files for every historical run. JSON1 extracts only display metadata;
        the aggregate counters come from the SQLite index.
        """
        where = ""
        parameters: tuple[str, ...] = ()
        if run_ids is not None:
            if not run_ids:
                return []
            where = "WHERE r.id IN (" + ",".join("?" for _ in run_ids) + ")"
            parameters = tuple(run_ids)
        with self.connection() as db:
            rows = db.execute(
                "SELECT r.id,r.created_at,r.updated_at,r.status,r.error,"
                "r.duration_seconds,"
                "json_extract(r.manifest,'$.novel.title') AS title,"
                "json_extract(r.manifest,'$.source_dir') AS source_dir,"
                "json_extract(r.manifest,'$.stages[0].runner') AS runner,"
                "json_extract(r.manifest,'$.stages[0].runner_config.model') AS model,"
                "json_extract(r.manifest,'$.workflow') AS workflow,"
                "json_extract(r.manifest,'$.source_identity.novel_id') AS novel_id,"
                "COUNT(c.chunk_id) AS total_chunks,"
                "SUM(CASE WHEN c.status='completed' THEN 1 ELSE 0 END) AS completed,"
                "SUM(CASE WHEN c.status='pending' THEN 1 ELSE 0 END) AS pending,"
                "SUM(CASE WHEN c.status='failed' THEN 1 ELSE 0 END) AS failed,"
                "SUM(CASE WHEN c.status='running' THEN 1 ELSE 0 END) AS running "
                "FROM runs r LEFT JOIN chunks c ON c.run_id=r.id "
                f"{where} GROUP BY r.id ORDER BY r.created_at DESC,r.id DESC",
                parameters,
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "status": row["status"],
                "error": row["error"],
                "duration_seconds": row["duration_seconds"],
                "title": row["title"],
                "source_dir": row["source_dir"],
                "runner": row["runner"],
                "model": row["model"],
                "workflow": row["workflow"],
                "novel_id": row["novel_id"],
                "counts": {
                    "completed": row["completed"] or 0,
                    "pending": row["pending"] or 0,
                    "failed": row["failed"] or 0,
                    "running": row["running"] or 0,
                    "corrupt": 0,
                },
                "total_chunks": row["total_chunks"] or 0,
            }
            for row in rows
        ]

    @staticmethod
    def _library_row(row) -> dict:
        value = dict(row)
        value["chapters"] = json.loads(value.pop("chapters_json"))
        value["chapter_count"] = len(value["chapters"])
        return value

    def library_rows(self) -> list[dict]:
        with self.connection() as db:
            return [
                self._library_row(row)
                for row in db.execute(
                    "SELECT * FROM library_sources ORDER BY lower(title), source_path"
                )
            ]

    def reconcile_library(self, candidates: list[dict], *, at: str) -> list[dict]:
        """Project a complete one-level scan transactionally; repeated scans are no-ops."""
        columns = (
            "source_path", "novel_id", "identity_kind", "inventory_sha256", "title",
            "author", "language", "format", "source_type", "source_id",
            "chapter_pattern", "chapters_json", "status", "error", "updated_at",
        )
        with self.connection() as db:
            previous = {
                row["source_path"]: self._library_row(row)
                for row in db.execute("SELECT * FROM library_sources")
            }
            current, moved = project_library(previous, candidates)
            for path in moved:
                db.execute("DELETE FROM library_sources WHERE source_path=?", (path,))
            for entry in current:
                row = {
                    **entry,
                    "chapters_json": canonical(entry["chapters"]),
                    "updated_at": at,
                }
                old = previous.get(entry["source_path"])
                if old:
                    old_row = {**old, "chapters_json": canonical(old["chapters"])}
                    if all(row[column] == old_row[column] for column in columns if column != "updated_at"):
                        continue
                db.execute(
                    "INSERT OR REPLACE INTO library_sources (" + ",".join(columns) + ") "
                    "VALUES (" + ",".join("?" for _ in columns) + ")",
                    tuple(row[column] for column in columns),
                )
        return self.library_rows()

    def set_run(
        self, run_id: str, status: str, at: str, error: str | None = None, duration: float = 0
    ):
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET status=?,updated_at=?,error=?,"
                "duration_seconds=duration_seconds+? WHERE id=?",
                (status, at, error, duration, run_id),
            )

    def chunks(self, run_id: str) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM chunks WHERE run_id=? ORDER BY chapter,ordinal", (run_id,)
                )
            ]

    def set_chunk(
        self,
        run_id: str,
        chapter: int,
        chunk_id: str,
        status: str,
        metadata: dict | None = None,
        output_path: str | None = None,
        output_sha256: str | None = None,
    ):
        with self.connection() as db:
            db.execute(
                "UPDATE chunks SET status=?,metadata=?,output_path=?,output_sha256=? "
                "WHERE run_id=? AND chapter=? AND chunk_id=?",
                (
                    status,
                    canonical(metadata) if metadata else None,
                    output_path,
                    output_sha256,
                    run_id,
                    chapter,
                    chunk_id,
                ),
            )

    def requeue_chunk_cas(
        self, run_id: str, chapter: int, chunk_id: str, expected_machine_sha256: str,
        at: str,
    ) -> bool:
        """Requeue the inspected machine candidate and reopen a completed run."""
        with self.connection() as db:
            cursor = db.execute(
                "UPDATE chunks SET status='pending' WHERE run_id=? AND chapter=? "
                "AND chunk_id=? AND status='completed' AND output_sha256=?",
                (run_id, chapter, chunk_id, expected_machine_sha256),
            )
            if cursor.rowcount == 1:
                db.execute(
                    "UPDATE runs SET status='pending',updated_at=?,error=NULL "
                    "WHERE id=? AND status='completed'",
                    (at, run_id),
                )
            return cursor.rowcount == 1

    def attempts(self, run_id: str, chapter: int, chunk_id: str, stage: str) -> list[dict]:
        with self.connection() as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT metadata FROM attempts WHERE run_id=? AND chapter=? AND chunk_id=? "
                    "AND stage=? ORDER BY number",
                    (run_id, chapter, chunk_id, stage),
                )
            ]

    def save_attempt(self, metadata: dict):
        with self.connection() as db:
            db.execute(
                "INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id,chapter,chunk_id,stage,number) DO UPDATE SET "
                "status=excluded.status,metadata=excluded.metadata",
                (
                    metadata["run_id"],
                    metadata["chapter"],
                    metadata["chunk_id"],
                    metadata["stage"],
                    metadata["attempt"],
                    metadata["status"],
                    metadata["input_sha256"],
                    canonical(metadata),
                ),
            )

    def final_entries(self, run_id: str) -> list[dict]:
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM final_layer WHERE run_id=? ORDER BY chapter,chunk_id",
                    (run_id,),
                )
            ]

    def final_entry(self, run_id: str, chapter: int, chunk_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM final_layer WHERE run_id=? AND chapter=? AND chunk_id=?",
                (run_id, chapter, chunk_id),
            ).fetchone()
            return dict(row) if row else None

    def final_snapshot(self, run_id: str, chapter: int, chunk_id: str) -> dict | None:
        """Consistent single-statement snapshot of a final and its machine chunk.

        The human final and the machine chunk live in separate rows. Reading them
        in one SELECT prevents a promotion from mixing a final revision with a
        machine candidate from a different instant.
        """
        with self.connection() as db:
            row = db.execute(
                "SELECT f.status AS final_status, f.final_text, f.final_sha256,"
                " f.machine_sha256 AS final_machine_sha256, f.actor, f.reason, f.updated_at,"
                " c.status AS chunk_status, c.output_sha256, c.output_path, c.metadata "
                "FROM final_layer f JOIN chunks c "
                "ON c.run_id=f.run_id AND c.chapter=f.chapter AND c.chunk_id=f.chunk_id "
                "WHERE f.run_id=? AND f.chapter=? AND f.chunk_id=?",
                (run_id, chapter, chunk_id),
            ).fetchone()
            return dict(row) if row else None

    def write_final_cas(self, entry: dict, expected_final_sha256: str | None) -> bool:
        """Atomically compare the machine and prior final before writing."""
        key = (entry["run_id"], entry["chapter"], entry["chunk_id"])
        values = (
            entry["status"], entry["final_text"], entry["final_sha256"],
            entry["machine_sha256"], entry["actor"], entry["reason"], entry["updated_at"],
        )
        with self.connection() as db:
            if expected_final_sha256 is None:
                cursor = db.execute(
                    "INSERT INTO final_layer(run_id,chapter,chunk_id,status,final_text,"
                    "final_sha256,machine_sha256,actor,reason,updated_at) "
                    "SELECT ?,?,?,?,?,?,?,?,?,? WHERE EXISTS ("
                    "SELECT 1 FROM chunks WHERE run_id=? AND chapter=? AND chunk_id=? "
                    "AND status='completed' AND output_sha256=?) "
                    "ON CONFLICT(run_id,chapter,chunk_id) DO NOTHING",
                    (*key, *values, *key, entry["machine_sha256"]),
                )
            else:
                cursor = db.execute(
                    "UPDATE final_layer SET status=?,final_text=?,final_sha256=?,"
                    "machine_sha256=?,actor=?,reason=?,updated_at=? "
                    "WHERE run_id=? AND chapter=? AND chunk_id=? AND final_sha256=? "
                    "AND EXISTS (SELECT 1 FROM chunks WHERE run_id=? AND chapter=? "
                    "AND chunk_id=? AND status='completed' AND output_sha256=?)",
                    (*values, *key, expected_final_sha256, *key, entry["machine_sha256"]),
                )
            return cursor.rowcount == 1

    def delete_final_cas(
        self, run_id: str, chapter: int, chunk_id: str,
        expected_machine_sha256: str, expected_final_sha256: str,
    ) -> bool:
        with self.connection() as db:
            cursor = db.execute(
                "DELETE FROM final_layer WHERE run_id=? AND chapter=? AND chunk_id=? "
                "AND final_sha256=? AND EXISTS (SELECT 1 FROM chunks WHERE run_id=? "
                "AND chapter=? AND chunk_id=? AND status='completed' AND output_sha256=?)",
                (run_id, chapter, chunk_id, expected_final_sha256,
                 run_id, chapter, chunk_id, expected_machine_sha256),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _bible_row(row) -> dict:
        entry = dict(row)
        entry["aliases"] = json.loads(entry["aliases"])
        entry["source_anchors"] = json.loads(entry["source_anchors"])
        entry["provenance"] = json.loads(entry["provenance"])
        return entry

    def bible_entries(self) -> list[dict]:
        with self.connection() as db:
            return [
                self._bible_row(row)
                for row in db.execute(
                    "SELECT * FROM book_bible ORDER BY kind,source_form,entry_id"
                )
            ]

    def bible_entry(self, entry_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM book_bible WHERE entry_id=?", (entry_id,)
            ).fetchone()
            return self._bible_row(row) if row else None

    def upsert_bible_entry(self, entry: dict):
        with self.connection() as db:
            db.execute(
                "INSERT INTO book_bible(entry_id,kind,source_form,target_form,aliases,trust,"
                "notes,source_anchors,provenance,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(entry_id) DO UPDATE SET "
                "kind=excluded.kind,source_form=excluded.source_form,"
                "target_form=excluded.target_form,aliases=excluded.aliases,trust=excluded.trust,"
                "notes=excluded.notes,source_anchors=excluded.source_anchors,"
                "provenance=excluded.provenance,updated_at=excluded.updated_at",
                (
                    entry["entry_id"],
                    entry["kind"],
                    entry["source_form"],
                    entry["target_form"],
                    json.dumps(entry["aliases"], ensure_ascii=False),
                    entry["trust"],
                    entry.get("notes"),
                    json.dumps(entry["source_anchors"], ensure_ascii=False),
                    json.dumps(entry["provenance"], ensure_ascii=False),
                    entry["created_at"],
                    entry["updated_at"],
                ),
            )

    def insert_bible_entry(self, entry: dict) -> bool:
        """Insert only if absent, never overwriting an existing entry."""
        with self.connection() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO book_bible(entry_id,kind,source_form,target_form,aliases,"
                "trust,notes,source_anchors,provenance,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entry["entry_id"],
                    entry["kind"],
                    entry["source_form"],
                    entry["target_form"],
                    json.dumps(entry["aliases"], ensure_ascii=False),
                    entry["trust"],
                    entry.get("notes"),
                    json.dumps(entry["source_anchors"], ensure_ascii=False),
                    json.dumps(entry["provenance"], ensure_ascii=False),
                    entry["created_at"],
                    entry["updated_at"],
                ),
            )
            return cursor.rowcount > 0

    def delete_bible_entry(self, entry_id: str) -> bool:
        with self.connection() as db:
            cursor = db.execute("DELETE FROM book_bible WHERE entry_id=?", (entry_id,))
            return cursor.rowcount > 0
