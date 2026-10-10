"""Prepare and verify derived review runs using existing manifests and storage."""

import copy
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from filelock import FileLock

from .artifacts import canonical, sha256, write_bytes, write_json, write_text
from .book_metadata import verify_novel
from .config import ValidationConfig, load_glossary, safe_name
from .run_factory import now
from .translation.book_context import verify_book_context
from .translation.frozen_draft import freeze_draft, verify_mapping
from .translation.review import review_contract
from .translation.validator import validate

# Parent workflows whose accepted candidate is a frozen translator draft.
# ``direct`` is the legacy baseline; ``draft-light`` is the semantic-first
# pipeline whose accepted attempt is the frozen draft.
REVIEWABLE_PARENT_WORKFLOWS = frozenset({"direct", "draft-light"})


def checked_bytes(folder: Path, relative: str, expected: str) -> bytes:
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Draft artifact path escapes run directory")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Missing frozen input: {relative}") from error
    if sha256(data) != expected:
        raise ValueError(f"Frozen input hash mismatch: {relative}")
    return data


def verify_source(service, manifest, chapter):
    folder = service.run_dir(manifest["run_id"])
    relative = f"chapters/{chapter['number']:04d}"
    checked_bytes(folder, relative + "/original.txt", chapter["source_sha256"])
    path = Path(manifest["source_dir"]) / chapter["filename"]
    if not path.is_file() or sha256(path.read_bytes()) != chapter["source_sha256"]:
        raise ValueError(f"Source changed or missing: {chapter['filename']}")
    source = checked_bytes(folder, relative + "/source.txt", chapter["text_sha256"]).decode("utf-8")
    if "".join(c["text"] for c in chapter["chunks"]) != source:
        raise ValueError("Parent source chunk coverage mismatch")
    for c in chapter["chunks"]:
        if sha256(c["text"]) != c["source_sha256"] or source[c["start"] : c["end"]] != c["text"]:
            raise ValueError("Parent source chunk hash/boundary mismatch")


def origin_for(service, row, seen=None):
    seen = set() if seen is None else seen
    identity = (row["run_id"], row["chapter"], row["chunk_id"])
    if identity in seen:
        raise ValueError("Direct draft origin cycle")
    seen.add(identity)
    if not service.stages.valid_output(row):
        raise ValueError(f"Direct draft incomplete/corrupt: {identity}")
    manifest = service.repository.run(row["run_id"])["manifest"]
    meta = json.loads(row["metadata"] or "{}")
    origin = meta.get("origin")
    if origin:
        candidates = [
            r
            for r in service.repository.chunks(origin["run_id"])
            if r["chapter"] == row["chapter"] and r["chunk_id"] == row["chunk_id"]
        ]
        if len(candidates) != 1:
            raise ValueError("Missing direct draft origin")
        parent = candidates[0]
        if (
            parent["source_sha256"] != row["source_sha256"]
            or parent["output_sha256"] != row["output_sha256"]
            or origin["output_sha256"] != row["output_sha256"]
        ):
            raise ValueError("Direct draft origin hash mismatch")
        return origin_for(service, parent, seen)
    stages = manifest.get("stages", [])
    if (
        manifest.get("workflow") not in REVIEWABLE_PARENT_WORKFLOWS
        or len(stages) != 1
        or stages[0]["role"] != "translator"
    ):
        raise ValueError("Parent output is not a translator baseline")
    stage = stages[0]
    attempt = (meta.get("stages") or [meta])[0]
    if attempt.get("review") or attempt.get("stage") not in {None, stage["name"]}:
        raise ValueError("Parent output candidate layer mismatch")
    # Inherited origins are resolved before consulting the current profile.
    return {
        "run_id": row["run_id"],
        "chapter": row["chapter"],
        "chunk_id": row["chunk_id"],
        "output_path": row["output_path"],
        "output_sha256": row["output_sha256"],
        "source_sha256": row["source_sha256"],
        "runner": attempt.get("runner", stage["runner"]),
        "harness": attempt.get("harness", stage["runner_config"].get("type")),
        "configured_model": stage["runner_config"].get("model"),
        "model": attempt.get("model"),
        "reported_model": (attempt.get("effective_settings") or {}).get("reported_model"),
        "prompt_version": attempt.get("prompt_version", stage.get("prompt_version")),
        "prompt_sha256": attempt.get("prompt_sha256", stage.get("prompt_sha256")),
        "runner_config": stage["runner_config"],
        "effective_settings": attempt.get("effective_settings"),
        "segments": meta.get("segments", attempt.get("segments")),
    }


def prepare_review(
    service, parent_id, *, reviewer, chapters=None, force=False, prompt_version="review-v1"
):
    if reviewer not in service.config.runners:
        raise ValueError(f"Unknown reviewer: {reviewer}")
    prompt = (service.config.translation.prompts_dir / f"{safe_name(prompt_version)}.md").read_text(
        "utf-8"
    )
    if not prompt.strip():
        raise ValueError("Empty review prompt")
    with service._lock(parent_id), FileLock(str(service.root / ".create.lock"), timeout=30):
        parent = service.repository.run(parent_id)["manifest"]
        if parent.get("workflow") not in REVIEWABLE_PARENT_WORKFLOWS:
            raise ValueError("Review requires a direct or draft-light parent run")
        verify_novel(parent)
        folder = service.run_dir(parent_id)
        if (
            load_glossary(folder / "glossary.yaml") != parent["glossary"]
            or sha256(canonical(parent["glossary"])) != parent["glossary_sha256"]
        ):
            raise ValueError("Parent glossary snapshot changed")
        if "book_context" in parent:
            book = parent["book_context"]
            if (
                sha256(canonical(book)) != parent["book_context_sha256"]
                or json.loads((folder / "book-context.json").read_text("utf-8")) != book
            ):
                raise ValueError("Parent book context snapshot changed")
            verify_book_context(book, Path(parent["source_dir"]))
        selected = parent["chapters"]
        if chapters is not None:
            if not chapters or set(chapters) - {c["number"] for c in selected}:
                raise ValueError("Requested chapters are not in direct parent")
            selected = [c for c in selected if c["number"] in set(chapters)]
        rows = {(r["chapter"], r["chunk_id"]): r for r in service.repository.chunks(parent_id)}
        units, dependencies = [], []
        # The documented post-review contract re-checks the rendered JA->RU text
        # with the strict 0021 profile. Force it on the derived run so a profile
        # that left script/dialogue checks opt-in cannot publish a reviewed leak.
        policy = service.config.validation.model_copy(
            update={"max_foreign_script_chars": 0, "reject_malformed_dialogue": True}
        )
        # The frozen *input* draft of a ``draft-light`` parent may contain known
        # observable corruption (script/dialogue leaks) that the stronger reviewer
        # is meant to repair, so it is admitted under the parent's own validation
        # policy. The reviewed *output* is still validated with the strict policy
        # above, so a reviewer that leaves the leak in fails closed.
        input_policy = (
            ValidationConfig.model_validate(parent["validation"])
            if parent.get("workflow") == "draft-light"
            else policy
        )

        def draft_row(ch, chunk):
            row = rows[(ch["number"], chunk["id"])]
            origin = origin_for(service, row)
            data = checked_bytes(folder, row["output_path"], row["output_sha256"])
            if row["source_sha256"] != chunk["source_sha256"]:
                raise ValueError("Parent draft source hash mismatch")
            return row, origin, data

        for ch in selected:
            verify_source(service, parent, ch)
            previous_source = previous_draft = ""
            settings = parent["translation"]
            if (
                settings.get("cross_chapter_context")
                and settings.get("context_strategy") == "previous"
                and settings.get("overlap_chars", 0)
            ):
                predecessor = next(
                    (c for c in parent["chapters"] if c["number"] == ch["number"] - 1), None
                )
                if predecessor:
                    verify_source(service, parent, predecessor)
                    for c in predecessor["chunks"]:
                        row, origin, data = draft_row(predecessor, c)
                        dependencies.append(
                            {
                                "chapter": predecessor["number"],
                                "chunk_id": c["id"],
                                "source_sha256": c["source_sha256"],
                                "output_path": row["output_path"],
                                "output_sha256": row["output_sha256"],
                                "origin": origin,
                            }
                        )
                    previous_source = predecessor["chunks"][-1]["text"]
                    previous_draft = data.decode("utf-8")
            for c in ch["chunks"]:
                row, origin, data = draft_row(ch, c)
                draft = data.decode("utf-8")
                quality = validate(c["text"], draft, None, input_policy)
                if quality.level == "FAIL":
                    raise ValueError(
                        f"Frozen draft validation failed: chapter {ch['number']} {c['id']}: {quality.issues}"
                    )
                try:
                    mapping = freeze_draft(
                        c["text"], draft, origin["segments"], heading=c["start"] == 0
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Direct draft mapping failed: chapter {ch['number']} {c['id']}: {error}"
                    ) from error
                relative = f"chapters/{ch['number']:04d}/chunks/{c['id']}"
                context = service._chunk_context(parent, ch, c, previous_source, previous_draft)
                units.append(
                    {
                        "chapter": ch["number"],
                        "chunk_id": c["id"],
                        "source_sha256": c["source_sha256"],
                        "draft_sha256": sha256(data),
                        "parent_output_path": row["output_path"],
                        "draft_path": relative + "/draft.md",
                        "mapping": mapping,
                        "context": context,
                        "origin": origin,
                    }
                )
                previous_source, previous_draft = c["text"], draft
            baseline = service._chapter_draft(parent_id, ch)
            chapter_path = f"chapters/{ch['number']:04d}/translation.md"
            checked_bytes(folder, chapter_path, sha256(baseline))
            quality = validate(
                "".join(c["text"] for c in ch["chunks"]), baseline, None, input_policy
            )
            if quality.level == "FAIL":
                raise ValueError(
                    f"Frozen chapter validation failed: {ch['number']}: {quality.issues}"
                )
        manifest = copy.deepcopy(parent)
        for key in (
            "run_id",
            "created_at",
            "fingerprint",
            "inherited",
            "continued_from",
            "rejected_inheritance",
        ):
            manifest.pop(key, None)
        manifest.update(
            workflow="review-direct",
            chapters=copy.deepcopy(selected),
            routing=None,
            validation=policy.model_dump(),
        )
        manifest["translation"].update(
            output_format="segments-v1",
            retries=service.config.translation.retries,
            retry_delay_seconds=service.config.translation.retry_delay_seconds,
            max_tokens=service.config.translation.max_tokens,
            temperature=service.config.translation.temperature,
        )
        manifest["stages"] = [
            {
                "name": "review",
                "role": "reviewer",
                "runner": reviewer,
                "runner_config": service.config.runners[reviewer].public_settings(),
                "prompt_version": prompt_version,
                "prompt_sha256": sha256(prompt),
                "prompt": prompt,
                "review_contract": review_contract(prompt_version),
            }
        ]
        manifest["review_input"] = {
            "version": 1,
            "parent_run_id": parent_id,
            "parent_fingerprint": parent["fingerprint"],
            "parent_workflow": parent.get("workflow"),
            "candidate_layer": f"{parent.get('workflow')}-translator",
            "translation": parent["translation"],
            "units": units,
            "dependencies": dependencies,
        }
        fingerprint = sha256(canonical(manifest))
        existing = service.repository.matching(fingerprint)
        if existing and not force:
            return existing
        rid = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        manifest.update(run_id=rid, fingerprint=fingerprint, created_at=now())
        target = service.run_dir(rid)
        write_json(target / "manifest.json", manifest)
        write_text(
            target / "glossary.yaml", yaml.safe_dump(manifest["glossary"], allow_unicode=True)
        )
        if "book_context" in manifest:
            write_json(target / "book-context.json", manifest["book_context"])
        if "book_bible" in manifest:
            write_json(target / "book-bible.json", manifest["book_bible"])
        write_text(target / "prompts/review.md", prompt)
        for ch in selected:
            relative = f"chapters/{ch['number']:04d}"
            write_bytes(
                target / relative / "original.txt",
                checked_bytes(folder, relative + "/original.txt", ch["source_sha256"]),
            )
            write_bytes(
                target / relative / "source.txt",
                checked_bytes(folder, relative + "/source.txt", ch["text_sha256"]),
            )
            write_bytes(
                target / relative / "draft.md", (folder / relative / "translation.md").read_bytes()
            )
        for u in units:
            write_bytes(
                target / u["draft_path"],
                checked_bytes(folder, u["parent_output_path"], u["draft_sha256"]),
            )
            write_json(target / Path(u["draft_path"]).parent / "frozen-draft.json", u)
        chunks = [{**c, "chapter": ch["number"]} for ch in selected for c in ch["chunks"]]
        service.repository.create(manifest, chunks)
        return rid


def unit_for(manifest, chapter, chunk):
    return next(
        u
        for u in manifest["review_input"]["units"]
        if u["chapter"] == chapter["number"] and u["chunk_id"] == chunk["id"]
    )


def verify_review_inputs(service, manifest):
    review = manifest["review_input"]
    if (
        review.get("version") != 1
        or manifest["workflow"] != "review-direct"
        or len(manifest["stages"]) != 1
        or manifest["stages"][0]["role"] != "reviewer"
    ):
        raise ValueError("Unsupported derived review contract")
    parent = service.repository.run(review["parent_run_id"])["manifest"]
    if parent["fingerprint"] != review["parent_fingerprint"]:
        raise ValueError("Review parent fingerprint changed")
    folder, old = service.run_dir(manifest["run_id"]), service.run_dir(parent["run_id"])
    if len(review["units"]) != sum(len(ch["chunks"]) for ch in manifest["chapters"]):
        raise ValueError("Frozen draft unit coverage mismatch")
    for ch in manifest["chapters"]:
        verify_source(service, manifest, ch)
        verify_source(service, parent, ch)
        baseline = []
        for c in ch["chunks"]:
            u = unit_for(manifest, ch, c)
            data = checked_bytes(folder, u["draft_path"], u["draft_sha256"])
            checked_bytes(old, u["parent_output_path"], u["draft_sha256"])
            checked_bytes(
                service.run_dir(u["origin"]["run_id"]),
                u["origin"]["output_path"],
                u["origin"]["output_sha256"],
            )
            snapshot = folder / Path(u["draft_path"]).parent / "frozen-draft.json"
            if (
                json.loads(snapshot.read_text("utf-8")) != u
                or u["source_sha256"] != c["source_sha256"]
            ):
                raise ValueError("Frozen draft metadata changed")
            verify_mapping(c["text"], data.decode("utf-8"), u["mapping"], heading=c["start"] == 0)
            baseline.append(data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
        from .translation.assembly import assemble

        text = assemble([c["text"] for c in ch["chunks"]], baseline)
        checked_bytes(folder, f"chapters/{ch['number']:04d}/draft.md", sha256(text))
        checked_bytes(old, f"chapters/{ch['number']:04d}/translation.md", sha256(text))
    for d in review["dependencies"]:
        ch = next(c for c in parent["chapters"] if c["number"] == d["chapter"])
        verify_source(service, parent, ch)
        checked_bytes(old, d["output_path"], d["output_sha256"])
