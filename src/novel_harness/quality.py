"""Read-only quality audit, also applied to the exact text exported to EPUB."""

import json
from pathlib import Path

from .artifacts import canonical, sha256
from .config import ValidationConfig
from .translation.validator import validate


def audit_chapters(manifest: dict, chapters: list[dict]) -> dict:
    checks = []
    config = ValidationConfig.model_validate(manifest["validation"])
    sources = {c["number"]: "".join(p["text"] for p in c["chunks"]) for c in manifest["chapters"]}
    for chapter in chapters:
        if not chapter["complete"]:
            level, issues = "FAIL", ["incomplete_chapter"]
        else:
            result = validate(sources[chapter["chapter"]], chapter["translation"], None, config)
            level, issues = result.level, result.issues
        checks.append({"chapter": chapter["chapter"], "level": level, "issues": issues})
    blocked = sum(c["level"] == "FAIL" for c in checks)
    warnings = sum(c["level"] == "WARNING" for c in checks)
    return {
        "version": 1,
        "status": "blocked" if blocked else "needs_review" if warnings else "checks_passed",
        "blocking_chapters": blocked,
        "warning_chapters": warnings,
        "checks": checks,
        "scope": "automated_text_checks; semantic review still required",
    }


def output_fingerprint(rows: list[dict], final_entries: list[dict] | None = None) -> str:
    """Fingerprint the effective exported layer, not just the machine layer.

    A chunk with a human final contributes its final revision plus the current
    machine candidate hash; a machine-only chunk contributes its machine output
    hash. Any accepted-final edit/replacement/revert, or a machine regeneration
    that stales a final, changes this fingerprint. Name-revised machine text is
    not covered here; only the persisted machine output and final layer are.
    """
    finals = {
        (entry["chapter"], entry["chunk_id"]): entry
        for entry in (final_entries or [])
    }
    items = []
    for row in rows:
        entry = finals.get((row["chapter"], row["chunk_id"]))
        if entry is not None and entry.get("final_text") is not None:
            items.append(
                (row["chapter"], row["chunk_id"], "final", entry["final_sha256"],
                 row["output_sha256"])
            )
        else:
            items.append((row["chapter"], row["chunk_id"], "machine", row["output_sha256"]))
    return sha256(canonical(items))


def readiness(
    directory: Path, rows: list[dict], complete: bool, final_entries: list[dict] | None = None
) -> dict:
    metadata = [json.loads(r["metadata"] or "{}") for r in rows]
    # A reviewer may accept a chunk that still has explicitly reported open
    # defects (review contract v2 ``unresolved``). Such a chunk is not clean and
    # must stay observable instead of being read as a PASS.
    open_defects = sum(
        len(m.get("review", {}).get("unresolved") or []) for m in metadata
    )
    status = (
        "needs_review"
        if open_defects
        or any(m.get("validation", {}).get("level") in {"WARNING", "FAIL"} for m in metadata)
        else "not_audited"
    )
    result = {
        "status": status,
        "epub_present": False,
        "epub_matches_outputs": False,
        "automatically_reviewed_chunks": sum("review" in m for m in metadata),
        "open_review_defects": open_defects,
        "human_review": "not_recorded",
    }
    try:
        book = json.loads((directory / "book.json").read_text("utf-8"))
        path = Path(book["path"])
        result["epub_present"] = path.is_file()
        if (
            complete
            and book.get("inputs_sha256") == output_fingerprint(rows, final_entries)
            and (path.is_file() and sha256(path.read_bytes()) == book["sha256"])
        ):
            result.update(status=book["quality"]["status"], epub_matches_outputs=True)
    except (OSError, ValueError, KeyError):
        pass
    return result
