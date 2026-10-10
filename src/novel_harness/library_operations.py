"""Novel Library projection and source import operations."""

from __future__ import annotations

from pathlib import Path

from filelock import FileLock

from .library import project_library, scan_roots
from .run_factory import now
from .source_import import SourceImportRequest, import_source
from .source_import import import_epub_source as import_epub_source_file
from .sources.narou_invocation import invoke_narou_download


def _scan_library(self) -> list[dict]:
    """Compute a fresh in-memory projection without writing the index."""
    roots = self.config.novel.library_roots or [self.config.novel.source_dir]
    direct_titles = (
        {str(self.config.novel.source_dir): self.config.novel.title}
        if self.config.novel.title
        else {}
    )
    direct_patterns = {str(self.config.novel.source_dir): self.config.novel.chapter_pattern}
    candidates, self._library_diagnostics = scan_roots(
        roots,
        direct_titles=direct_titles,
        direct_patterns=direct_patterns,
    )
    indexed = {row["source_path"]: row for row in self.repository.library_rows()}
    self._library_rows, _moved = project_library(indexed, candidates)
    return candidates


def rescan_library(self) -> dict:
    """Explicitly persist a fresh index projection; never writes sources."""
    candidates = self._scan_library()
    self.repository.reconcile_library(candidates, at=now())
    self._library_rows = self.repository.library_rows()
    return self.library_novels()


def import_narou_source(
    self,
    source_dir: str | Path,
    *,
    ncode: str,
    title: str,
    expected_chapters: tuple[int, ...],
) -> dict:
    """Import a completed local Narou output set into the first Library root."""
    roots = self.config.novel.library_roots or [self.config.novel.source_dir]
    destination_root = roots[0]
    request = SourceImportRequest(
        provider="narou",
        source_id=ncode,
        title=title,
        expected_chapters=expected_chapters,
    )
    with FileLock(str(self.root / "source-import.lock"), timeout=30):
        self._scan_library()
        result = import_source(
            Path(source_dir),
            request,
            destination_root,
            existing_sources=self._library_rows,
            library_roots=roots,
        )
        if result.status == "imported":
            self.rescan_library()
            row = self.library_show(result.novel_id)
            if row["status"] != "ready":
                raise ValueError("Imported source is not selectable after Library rescan")
    return result.to_dict()


def import_epub_source(self, source_path: str | Path) -> dict:
    """Import a local Japanese source EPUB into the configured Novel Library."""
    roots = self.config.novel.library_roots or [self.config.novel.source_dir]
    destination_root = roots[0]
    with FileLock(str(self.root / "source-import.lock"), timeout=30):
        self._scan_library()
        result = import_epub_source_file(
            Path(source_path),
            destination_root,
            existing_sources=self._library_rows,
            library_roots=roots,
        )
        if result.status == "imported":
            self.rescan_library()
            row = self.library_show(result.novel_id)
            if row["status"] != "ready":
                raise ValueError("Imported EPUB is not selectable after Library rescan")
    return result.to_dict()


def invoke_narou_downloader(
    self,
    command: list[str] | tuple[str, ...],
    *,
    ncode: str,
    source_url: str,
    title: str,
    expected_chapters: tuple[int, ...],
    output_dir: str | Path,
    timeout_seconds: float = 1800,
    cwd: str | Path | None = None,
) -> dict:
    """Run the external protocol-v1 downloader, then update the library."""
    roots = self.config.novel.library_roots or [self.config.novel.source_dir]

    def import_completed(path: Path, chapters: tuple[int, ...]) -> dict:
        return self.import_narou_source(
            path,
            ncode=ncode,
            title=title,
            expected_chapters=chapters,
        )

    return invoke_narou_download(
        command=command,
        ncode=ncode,
        source_url=source_url,
        title=title,
        expected_chapters=expected_chapters,
        output_dir=Path(output_dir),
        timeout_seconds=timeout_seconds,
        library_roots=roots,
        import_completed=import_completed,
        cwd=Path(cwd) if cwd is not None else None,
    )


def _run_novel_id(manifest: dict, rows: list[dict]) -> str | None:
    identity = manifest.get("source_identity") or {}
    if identity.get("novel_id"):
        return identity["novel_id"]
    matches = [row for row in rows if row["source_path"] == manifest.get("source_dir")]
    if len(matches) == 1:
        return matches[0]["novel_id"]
    chapters = manifest.get("chapters") or []
    if not chapters:
        return None
    matches = [
        row
        for row in rows
        if row["novel_id"]
        and all(
            any(
                item["number"] == chapter["number"]
                and item["source_sha256"] == chapter["source_sha256"]
                for item in row["chapters"]
            )
            for chapter in chapters
        )
    ]
    return matches[0]["novel_id"] if len(matches) == 1 else None


def library_novels(self) -> dict:
    """Read-only card projection over the index and existing run state."""
    rows = self._library_rows
    runs: dict[str, list[dict]] = {}
    for run in self.repository.list_runs():
        novel_id = self._run_novel_id(run["manifest"], rows)
        if novel_id:
            runs.setdefault(novel_id, []).append(self.reporter.status(run["id"]))
    cards = []
    for row in rows:
        card = dict(row)
        card.pop("updated_at", None)
        card["chapter_count"] = len(card["chapters"])
        related = runs.get(row["novel_id"], [])
        latest = related[0] if related else None
        active = next((item for item in related if item["active"]), None)
        card["latest_run"] = latest
        card["active_run"] = active
        card["continue_run_id"] = (active or latest or {}).get("run_id")
        card["translation_state"] = (
            "unavailable"
            if row["status"] != "ready"
            else "active"
            if active
            else latest["status"]
            if latest
            else "not_started"
        )
        progress_run = active or latest
        card["progress"] = progress_run["progress"] if progress_run else 0
        card["final_layer"] = (
            self._library_final_summary(progress_run["run_id"]) if progress_run else None
        )
        cards.append(card)
    return {"novels": cards, "diagnostics": list(self._library_diagnostics)}


def _run_covered_chapters(service, run: dict, novel: dict) -> set[int]:
    """Return source chapters with every persisted chunk output committed."""
    manifest = run["manifest"]
    source_chapters = {
        chapter["number"]: chapter["source_sha256"] for chapter in novel["chapters"]
    }
    covered = set()
    rows = service.repository.chunks(run["id"])
    by_chapter: dict[int, list[dict]] = {}
    for row in rows:
        by_chapter.setdefault(row["chapter"], []).append(row)
    for chapter in manifest.get("chapters", []):
        number = chapter["number"]
        if source_chapters.get(number) != chapter.get("source_sha256"):
            continue
        chapter_rows = by_chapter.get(number, [])
        expected_chunks = {chunk["id"] for chunk in chapter.get("chunks", [])}
        actual_chunks = {row["chunk_id"] for row in chapter_rows}
        if expected_chunks and actual_chunks == expected_chunks and all(
            service.stages.valid_output(row) for row in chapter_rows
        ):
            covered.add(number)
    return covered


def library_remaining(self) -> dict:
    """Project source coverage using the union of all non-canceled runs."""
    rows = sorted(
        self._library_rows,
        key=lambda row: (row["title"].casefold(), row["novel_id"], row["source_path"]),
    )
    runs_by_novel: dict[str, list[tuple[str, set[int]]]] = {}
    for run in self.repository.list_runs():
        if run["status"] == "canceled":
            continue
        novel_id = self._run_novel_id(run["manifest"], rows)
        if novel_id:
            novel = next((row for row in rows if row["novel_id"] == novel_id), None)
            if novel is not None and novel["status"] == "ready":
                runs_by_novel.setdefault(novel_id, []).append(
                    (run["id"], _run_covered_chapters(self, run, novel))
                )
    remaining = []
    excluded = []
    for row in rows:
        if row["status"] != "ready":
            excluded.append(
                {
                    "novel_id": row["novel_id"],
                    "title": row["title"],
                    "source_path": row["source_path"],
                    "reason": row["error"] or row["status"],
                }
            )
            continue
        contributors = runs_by_novel.get(row["novel_id"], [])
        translated = set().union(*(chapters for _, chapters in contributors)) if contributors else set()
        total = len(row["chapters"])
        if len(translated) >= total:
            continue
        governing = sorted(
            ((len(chapters), run_id) for run_id, chapters in contributors),
            key=lambda item: (-item[0], item[1]),
        )
        remaining.append(
            {
                "novel_id": row["novel_id"],
                "title": row["title"],
                "source_path": row["source_path"],
                "chapter_count": total,
                "translated": len(translated),
                "remaining": total - len(translated),
                "run_id": governing[0][1] if governing and governing[0][0] else None,
            }
        )
    return {"novels": remaining, "excluded": excluded}


def _remaining_chapters(self, novel_id: str) -> tuple[dict, list[int], list[dict]]:
    rows = self._library_rows
    novel = next((row for row in rows if row["novel_id"] == novel_id), None)
    if novel is None:
        raise ValueError(f"Unknown novel: {novel_id}")
    if novel["status"] != "ready":
        raise ValueError(f"Novel {novel_id} is {novel['status']}: {novel['error']}")
    related = []
    for run in self.repository.list_runs():
        if run["status"] == "canceled" or self._run_novel_id(run["manifest"], rows) != novel_id:
            continue
        covered = _run_covered_chapters(self, run, novel)
        related.append((run, covered))
    translated = set().union(*(covered for _, covered in related)) if related else set()
    remaining = [chapter["number"] for chapter in novel["chapters"] if chapter["number"] not in translated]
    return novel, remaining, related


def _source_matches_run(run: dict, novel: dict) -> bool:
    current = {chapter["number"]: chapter["source_sha256"] for chapter in novel["chapters"]}
    return all(
        current.get(chapter["number"]) == chapter.get("source_sha256")
        for chapter in run["manifest"].get("chapters", [])
    )


def _parent_launch_kwargs(manifest: dict) -> dict:
    stage = next(
        stage
        for stage in manifest["stages"]
        if stage.get("role") in {"translator", "planner"}
    )
    preset = manifest.get("preset") or {}
    if preset.get("preset_id"):
        return {"preset": preset["preset_id"]}
    values = {
        "pipeline": manifest.get("workflow"),
        "chunk_chars": manifest["translation"].get("chunk_chars"),
        "temperature": manifest["translation"].get("temperature"),
        "prompt_version": manifest["translation"].get("prompt_version"),
    }
    values["runner"] = stage["runner"]
    editor_stage = next((item for item in manifest["stages"] if item.get("role") == "editor"), None)
    if editor_stage is not None:
        values["editor"] = editor_stage["runner"]
    return values


def launch_remaining_novel(
    self,
    novel_id: str,
    config_path: str | Path,
    *,
    preset: str | None = None,
    runner: str | None = None,
    pipeline: str | None = None,
    translator: str | None = None,
    editor: str | None = None,
    force: bool = False,
    chunk_chars: int | None = None,
    temperature: float | None = None,
    prompt_version: str | None = None,
) -> dict:
    """Create/resume and launch only chapters not covered by valid run output."""
    from .launch import launch_run

    self.rescan_library()
    novel, remaining, related = _remaining_chapters(self, novel_id)
    if not remaining:
        raise ValueError(f"Novel {novel_id} is already fully translated")
    for run, _covered in related:
        if not _source_matches_run(run, novel):
            raise ValueError("Source changed since a saved run; create a new sample or run")

    card = next(item for item in self.library_novels()["novels"] if item["novel_id"] == novel_id)
    active = card.get("active_run")
    if active:
        active_run = self.repository.run(active["run_id"])
        active_chapters = {chapter["number"] for chapter in active_run["manifest"].get("chapters", [])}
        if set(remaining) <= active_chapters:
            if any(value is not None for value in (preset, runner, pipeline, translator, editor, chunk_chars, temperature, prompt_version)):
                raise ValueError("Active run settings are frozen; resume without overrides")
            return launch_run(self, config_path, active["run_id"])

    approved_sample = next(
        (
            run
            for run, _covered in reversed(related)
            if (run["manifest"].get("sample") or {}).get("role") == "sample"
            and self.sample_approval(run["id"]) is not None
        ),
        None,
    )
    if approved_sample is not None:
        run_id = self.create_whole_book_run(
            approved_sample["id"],
            preset=preset,
            runner=runner,
            pipeline=pipeline,
            translator=translator,
            editor=editor,
            chapters=remaining,
            chunk_chars=chunk_chars,
            temperature=temperature,
            prompt_version=prompt_version,
            novel_id=novel_id,
        )
    else:
        parent = max(related, key=lambda item: (len(item[1]), item[0]["id"]), default=None)
        inherited = _parent_launch_kwargs(parent[0]["manifest"]) if parent else {}
        if "preset" in inherited and any(
            value is not None for value in (runner, pipeline, translator, editor, chunk_chars, temperature, prompt_version)
        ):
            raise ValueError("Saved run preset is frozen; continue without raw overrides")
        inherited.update(
            {
                key: value
                for key, value in {
                    "preset": preset,
                    "runner": runner,
                    "pipeline": pipeline,
                    "translator": translator,
                    "editor": editor,
                    "chunk_chars": chunk_chars,
                    "temperature": temperature,
                    "prompt_version": prompt_version,
                }.items()
                if value is not None
            }
        )
        run_id = self.create_run(
            chapters=remaining,
            force=True,
            novel_id=novel_id,
            **inherited,
        )
    return launch_run(self, config_path, run_id)


def _library_final_summary(self, run_id: str) -> dict:
    """Return final-layer counts without exposing chapter or final text."""
    chunks = self.repository.chunks(run_id)
    current = {(row["chapter"], row["chunk_id"]): row["output_sha256"] for row in chunks}
    entries = self.repository.final_entries(run_id)
    finals = {(row["chapter"], row["chunk_id"]): row for row in entries}
    accepted = needs_review = stale = 0
    for key, entry in finals.items():
        machine_sha = current.get(key)
        if machine_sha is None or entry["machine_sha256"] != machine_sha:
            stale += 1
        elif entry["status"] == "accepted":
            accepted += 1
        elif entry["status"] == "needs_review":
            needs_review += 1
    machine_only = sum(
        1
        for row in chunks
        if row["output_sha256"] and (row["chapter"], row["chunk_id"]) not in finals
    )
    return {
        "run_id": run_id,
        "total_chunks": len(chunks),
        "accepted": accepted,
        "needs_review": needs_review,
        "stale": stale,
        "machine_only": machine_only,
    }
