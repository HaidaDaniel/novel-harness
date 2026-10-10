"""EPUB build, comparison HTML and export naming.

Extracted from :class:`TranslationService`; reads persisted runs and writes only
export artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from filelock import Timeout

from .artifacts import canonical, sha256, write_bytes, write_json, write_text
from .book_metadata import export_novel, preserve_book_record
from .translation.assembly import assemble


class Exporter:
    """Comparison, name-revision and EPUB export operations."""

    def __init__(self, service):
        self.service = service

    def compare(self, chapter: int, runs: list[str]) -> dict:
        service = self.service
        if len(runs) < 2 or len(set(runs)) != len(runs):
            raise ValueError("Select at least two distinct runs")
        candidates = [service.reporter.inspect_chapter(run_id, chapter) for run_id in runs]
        if len({c["source_sha256"] for c in candidates}) != 1:
            raise ValueError(
                "Runs use different source snapshots; select runs with the same source"
            )
        return {
            "chapter": chapter,
            "title": candidates[0]["title"],
            "source": candidates[0]["source"],
            "candidates": [
                {**candidate, "status": service.reporter.status(candidate["run_id"])}
                for candidate in candidates
            ],
        }

    def compare_html(self, chapter: int, runs: list[str]) -> Path:
        from .compare import render_comparison

        service = self.service
        data = self.compare(chapter, runs)
        path = (
            service.root
            / "comparisons"
            / f"chapter-{chapter:04d}-{sha256(canonical(runs))[:16]}.html"
        )
        write_text(path, render_comparison(data))
        return path

    def revise_names(self, run_id: str) -> dict:
        from .translation.names import create_revision

        service = self.service
        with service._lock(run_id):
            report, _ = create_revision(service, run_id)
            return report

    @staticmethod
    def _compose_final_projection(
        revised_chapter: dict, manifest_chapter: dict, final_text: dict
    ) -> dict:
        """Compose one chapter from name-revised machine chunks and human finals.

        Human-final chunks are authoritative and byte-identical; machine chunks use
        the name-revised text. ``final_text`` is keyed by ``(chapter, chunk_id)``.
        """
        revised = {item["chunk_id"]: item["text"] for item in revised_chapter.get("chunks", [])}
        number = manifest_chapter["number"]
        texts = [
            final_text.get((number, chunk["id"]), revised.get(chunk["id"], ""))
            for chunk in manifest_chapter["chunks"]
        ]
        return {
            **revised_chapter,
            "translation": assemble(
                [c["text"] for c in manifest_chapter["chunks"]], texts
            ),
        }

    def build_epub(self, run_id: str, *, metadata_path: Path | None = None) -> Path:
        from .epub import render_epub
        from .export_names import export_identity, export_path

        service = self.service
        try:
            with service._lock(run_id):
                run = service.repository.run(run_id)
                book_record = service.run_dir(run_id) / "book.json"
                previous = (
                    json.loads(book_record.read_text("utf-8")) if book_record.is_file() else None
                )
                novel = export_novel(run["manifest"], metadata_path, previous)
                # Editors use SQLite CAS without the build's execution lock.
                # Capture once so text, provenance and receipt name one revision.
                final_entries = service.repository.final_entries(run_id)
                chapters = [
                    service.reporter.inspect_chapter(
                        run_id, c["number"], final_entries=final_entries
                    )
                    for c in run["manifest"]["chapters"]
                ]
                chunk_machine = {
                    (row["chapter"], row["chunk_id"]): row["output_sha256"]
                    for row in service.repository.chunks(run_id)
                }
                stale_finals = [
                    (entry["chapter"], entry["chunk_id"])
                    for entry in final_entries
                    if chunk_machine.get((entry["chapter"], entry["chunk_id"]))
                    != entry["machine_sha256"]
                ]
                if stale_finals:
                    listed = ", ".join(
                        f"chapter {chapter} chunk {chunk_id}"
                        for chapter, chunk_id in stale_finals
                    )
                    raise ValueError(
                        "Cannot build EPUB: stale human-final chunks require review: "
                        + listed
                    )
                final_text = {
                    (entry["chapter"], entry["chunk_id"]): entry["final_text"]
                    for entry in final_entries
                    if entry["final_text"] is not None
                }
                name_revision = None
                if run["manifest"]["translation"].get("name_rules"):
                    from .translation.names import create_revision

                    name_revision, revised = create_revision(service, run_id)
                    name_revision = {
                        k: name_revision[k]
                        for k in ("directory", "complete", "replacement_count", "review_count")
                    }
                    manifest_chapters = {
                        c["number"]: c for c in run["manifest"]["chapters"]
                    }
                    # Name revision composes per chunk: machine chunks are revised,
                    # human-final chunks stay authoritative and byte-identical.
                    chapters = [
                        self._compose_final_projection(
                            revised_chapter,
                            manifest_chapters[revised_chapter["chapter"]],
                            final_text,
                        )
                        for revised_chapter in revised
                    ]
                else:
                    # No name revision: the inspected final projection is authoritative.
                    for chapter in chapters:
                        chapter["translation"] = chapter.get(
                            "final", chapter["translation"]
                        )
                final_layer = [
                    {
                        "chapter": entry["chapter"],
                        "chunk_id": entry["chunk_id"],
                        "status": entry["status"],
                        "final_sha256": entry["final_sha256"],
                        "machine_sha256": entry["machine_sha256"],
                        "stale": chunk_machine.get((entry["chapter"], entry["chunk_id"]))
                        != entry["machine_sha256"],
                        "actor": entry["actor"],
                        "reason": entry["reason"],
                        "updated_at": entry["updated_at"],
                    }
                    for entry in final_entries
                ]
                from .quality import audit_chapters, output_fingerprint

                if any(not chapter["complete"] for chapter in chapters):
                    raise ValueError("Cannot build EPUB: incomplete or corrupt chapters")
                manifest = run["manifest"]
                source_epub = (manifest.get("source_identity") or {}).get("source_epub")
                if source_epub:
                    from .epub import (
                        render_source_epub,
                        strip_epub_roundtrip_markers,
                    )

                    if (
                        not isinstance(source_epub, dict)
                        or
                        source_epub.get("version") != 1
                        or source_epub.get("file") != "source-epub.epub"
                        or not isinstance(source_epub.get("sha256"), str)
                    ):
                        raise ValueError("Run has an invalid source EPUB snapshot reference")
                    snapshot = service.run_dir(run_id) / "source-epub.epub"
                    if snapshot.is_symlink() or not snapshot.is_file():
                        raise ValueError("Run source EPUB snapshot is missing or changed")
                    source_bytes = snapshot.read_bytes()
                    if sha256(source_bytes) != source_epub["sha256"]:
                        raise ValueError("Run source EPUB snapshot is missing or changed")
                    quality_manifest = {
                        **manifest,
                        "chapters": [
                            {
                                **chapter,
                                "chunks": [
                                    {
                                        **chunk,
                                        "text": strip_epub_roundtrip_markers(chunk["text"]),
                                    }
                                    for chunk in chapter["chunks"]
                                ],
                            }
                            for chapter in manifest["chapters"]
                        ],
                    }
                    quality_chapters = [
                        {
                            **chapter,
                            "translation": strip_epub_roundtrip_markers(
                                chapter["translation"]
                            ),
                        }
                        for chapter in chapters
                    ]
                    quality = audit_chapters(quality_manifest, quality_chapters)
                    data = render_source_epub(
                        source_bytes,
                        {
                            chapter["chapter"]: chapter["translation"]
                            for chapter in chapters
                        },
                        require_complete=True,
                    )
                else:
                    quality = audit_chapters(manifest, chapters)
                    data = render_epub(
                        {**manifest, "novel": novel},
                        chapters,
                        spread=service.config.novel.spread,
                        cover_path=service.config.novel.cover_path,
                    )
                if quality["blocking_chapters"]:
                    affected = [c["chapter"] for c in quality["checks"] if c["level"] == "FAIL"]
                    raise ValueError(f"EPUB quality blocked; repair chapters {affected}; run audit")
                path = export_path(
                    service.config.novel.books_dir, run["manifest"], novel, sha256(data)
                )
                if path.exists():
                    if path.read_bytes() != data:
                        raise ValueError(
                            "Existing export content mismatch; previous file preserved"
                        )
                else:
                    write_bytes(path, data)
                preserve_book_record(service.run_dir(run_id))
                write_json(
                    service.run_dir(run_id) / "book.json",
                    {
                        "run_id": run_id,
                        "path": str(path),
                        "sha256": sha256(data),
                        "novel": novel,
                        "novel_sha256": sha256(canonical(novel)),
                        "export_identity": export_identity(run["manifest"], novel),
                        "stages": [
                            {k: stage[k] for k in ("name", "role", "runner", "runner_config")}
                            for stage in run["manifest"]["stages"]
                        ],
                        "inherited": run["manifest"].get("inherited", []),
                        "chapters": [c["chapter"] for c in chapters],
                        "name_revision": name_revision,
                        "quality": quality,
                        "final_layer": final_layer,
                        "spread": service.config.novel.spread,
                        "cover": (
                            str(service.config.novel.cover_path)
                            if service.config.novel.cover_path is not None
                            else None
                        ),
                        "inputs_sha256": output_fingerprint(
                            service.repository.chunks(run_id), final_entries
                        ),
                        "warnings": [
                            json.loads(r["metadata"])["validation"]
                            for r in service.repository.chunks(run_id)
                            if json.loads(r["metadata"])["validation"]["level"] == "WARNING"
                        ],
                    },
                )
                return path
        except Timeout as error:
            raise ValueError(f"Run currently executing: {run_id}") from error
