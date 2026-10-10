"""Explicit continuation with new settings and verifiable inherited translations."""

import copy
import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from filelock import FileLock

from .artifacts import canonical, sha256, write_bytes, write_json, write_text
from .book_metadata import export_novel, verify_novel
from .config import TranslationConfig, ValidationConfig, load_glossary, safe_name
from .translation.book_context import book_bible_block, freeze_book_context
from .translation.chunker import chunk_text
from .translation.validator import validate


def create_continuation(
    service,
    parent_id: str,
    runner: str,
    chunk_chars: int,
    *,
    editor: str | None = None,
    pipeline: str | None = None,
) -> str:
    if runner not in service.config.runners:
        raise ValueError(f"Unknown runner: {runner}")
    # Hold both locks through artifact copying and the single index transaction.
    with service._lock(parent_id), FileLock(str(service.root / ".create.lock"), timeout=30):
        parent = service.repository.run(parent_id)["manifest"]
        if "review_input" in parent:
            raise ValueError("Derived review uses a frozen direct parent; create_review_run with force or resume")
        verify_novel(parent)
        old_dir = service.run_dir(parent_id)
        manifest = copy.deepcopy(parent)
        if "source_title" not in parent["novel"] and service.config.novel.metadata_path:
            manifest["novel"] = export_novel(parent, service.config.novel.metadata_path)
            manifest["novel_sha256"] = sha256(canonical(manifest["novel"]))
        for key in ("run_id", "fingerprint", "created_at", "inherited"):
            manifest.pop(key, None)
        manifest["continued_from"] = parent_id
        manifest["validation"] = service.config.validation.model_dump()
        manifest["translation"].update(
            chunk_chars=chunk_chars,
            retries=service.config.translation.retries,
            retry_delay_seconds=service.config.translation.retry_delay_seconds,
            max_tokens=service.config.translation.max_tokens,
            glossary_path=str(service.config.translation.glossary_path),
            name_rules=copy.deepcopy(service.config.translation.name_rules),
            name_rules_path=(
                str(service.config.translation.name_rules_path)
                if service.config.translation.name_rules_path
                else None
            ),
            output_format=service.config.translation.output_format,
            book_context_path=(
                str(service.config.translation.book_context_path)
                if service.config.translation.book_context_path
                else None
            ),
            book_context_chars=service.config.translation.book_context_chars,
            cross_chapter_context=service.config.translation.cross_chapter_context,
        )
        TranslationConfig.model_validate(manifest["translation"])
        manifest["glossary"] = load_glossary(service.config.translation.glossary_path)
        manifest["glossary_sha256"] = sha256(canonical(manifest["glossary"]))
        manifest["book_context"] = freeze_book_context(
            service.config.translation.book_context_path, Path(manifest["source_dir"])
        )
        manifest["book_context_sha256"] = sha256(canonical(manifest["book_context"]))
        manifest["book_bible"] = book_bible_block(service.repository.bible_entries())
        manifest["book_bible_sha256"] = sha256(canonical(manifest["book_bible"]))
        if pipeline is not None:
            if pipeline not in service.config.pipelines:
                raise ValueError(f"Unknown pipeline: {pipeline}")
            manifest["workflow"] = pipeline
            manifest["stages"] = []
            for stage in service.config.pipelines[pipeline].stages:
                version = stage.prompt_version or service.config.translation.prompt_version
                prompt = (
                    service.config.translation.prompts_dir / f"{safe_name(version)}.md"
                ).read_text("utf-8")
                if not prompt.strip():
                    raise ValueError(f"Empty prompt: {version}")
                manifest["stages"].append(
                    {
                        "name": stage.name,
                        "role": stage.role,
                        "runner": stage.runner,
                        "prompt": prompt,
                        "prompt_version": version,
                        "prompt_sha256": sha256(prompt),
                    }
                )
        for stage in manifest["stages"]:
            if stage["role"] == "translator":
                name = runner
            else:
                # Same precedence as run_factory: explicit editor, then the
                # requested/frozen stage runner. No silent translator fallback.
                name = editor or stage.get("runner")
            if not name:
                raise ValueError(f"Configure runner for stage {stage['name']} ({stage['role']})")
            if name not in service.config.runners:
                raise ValueError(f"Unknown runner: {name}")
            if (
                stage["role"] == "reviewer"
                and manifest["translation"]["output_format"] != "segments-v1"
            ):
                raise ValueError("Reviewer stages require translation.output_format: segments-v1")
            stage.update(runner=name, runner_config=service.config.runners[name].public_settings())
        rows = {(r["chapter"], r["chunk_id"]): r for r in service.repository.chunks(parent_id)}
        inherited, chunks, rejected = [], [], []
        for chapter in manifest["chapters"]:
            source = Path(manifest["source_dir"]) / chapter["filename"]
            snapshot = old_dir / "chapters" / f"{chapter['number']:04d}" / "original.txt"
            for path in (source, snapshot):
                if not path.is_file() or sha256(path.read_bytes()) != chapter["source_sha256"]:
                    raise ValueError(f"Source changed or missing: {chapter['filename']}")
            original_chunks = chapter["chunks"]
            text = "".join(c["text"] for c in original_chunks)
            prefix = []
            for chunk in original_chunks:
                row = rows[(chapter["number"], chunk["id"])]
                if not service._valid_output(row):
                    break
                quality = validate(
                    chunk["text"],
                    (old_dir / row["output_path"]).read_text("utf-8"),
                    None,
                    ValidationConfig.model_validate(manifest["validation"]),
                )
                if quality.level == "FAIL":
                    rejected.append(
                        {
                            "chapter": chapter["number"],
                            "chunk_id": chunk["id"],
                            "issues": quality.issues,
                            "output_sha256": row["output_sha256"],
                        }
                    )
                    break
                prefix.append(chunk)
                inherited.append(row)
            offset = prefix[-1]["end"] if prefix else 0
            suffix = []
            for chunk in chunk_text(text[offset:], chunk_chars):
                value = asdict(chunk)
                index = len(prefix) + chunk.index
                value.update(
                    index=index,
                    id=f"{index:04d}-{chunk.source_sha256[:16]}",
                    start=offset + chunk.start,
                    end=offset + chunk.end,
                )
                suffix.append(value)
            chapter["chunks"] = prefix + suffix
            chunks.extend({**c, "chapter": chapter["number"]} for c in chapter["chunks"])
        manifest["inherited"] = [
            {
                "chapter": r["chapter"],
                "chunk_id": r["chunk_id"],
                "output_sha256": r["output_sha256"],
                "run_id": parent_id,
            }
            for r in inherited
        ]
        manifest["rejected_inheritance"] = rejected
        fingerprint = sha256(canonical(manifest))
        # Repeating the continuation command resumes this exact continuation.
        existing = service.repository.matching(fingerprint)
        if existing:
            return existing
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        manifest.update(
            run_id=run_id, fingerprint=fingerprint, created_at=datetime.now(UTC).isoformat()
        )
        directory = service.run_dir(run_id)
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "book-context.json", manifest["book_context"])
        write_json(directory / "book-bible.json", manifest["book_bible"])
        write_text(
            directory / "glossary.yaml", yaml.safe_dump(manifest["glossary"], allow_unicode=True)
        )
        for stage in manifest["stages"]:
            write_text(directory / "prompts" / f"{stage['name']}.md", stage["prompt"])
        for chapter in manifest["chapters"]:
            relative = Path("chapters") / f"{chapter['number']:04d}"
            write_bytes(
                directory / relative / "original.txt",
                (old_dir / relative / "original.txt").read_bytes(),
            )
            write_text(
                directory / relative / "source.txt", "".join(c["text"] for c in chapter["chunks"])
            )
        initial_outputs = []
        for row in inherited:
            data = (old_dir / row["output_path"]).read_bytes()
            if sha256(data) != row["output_sha256"]:
                raise ValueError("Inherited output changed while copying")
            relative = Path("chapters") / f"{row['chapter']:04d}" / "chunks" / row["chunk_id"]
            metadata = {
                **json.loads(row["metadata"]),
                "run_id": run_id,
                "output_path": (relative / "translation.md").as_posix(),
                "origin": {
                    "run_id": parent_id,
                    "output_path": row["output_path"],
                    "output_sha256": row["output_sha256"],
                },
            }
            write_bytes(directory / relative / "translation.md", data)
            write_json(directory / relative / "metadata.json", metadata)
            initial_outputs.append(
                {**row, "output_path": metadata["output_path"], "metadata": metadata}
            )
        service.repository.create(manifest, chunks, initial_outputs=initial_outputs)
        for chapter in manifest["chapters"]:
            service._assemble_chapter(run_id, chapter)
        return run_id
