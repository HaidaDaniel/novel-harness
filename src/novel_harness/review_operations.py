"""Human final, Book Bible, and translation-memory operations."""

from __future__ import annotations

import json
from uuid import uuid4

from .artifacts import canonical, sha256
from .run_factory import now
from .translation.book_context import (
    AUTHORITATIVE_TRUST,
    ENTRY_KINDS,
    TRUST_STATES,
    authoritative_projection,
    projection_sha256,
)


def translation_memory(self) -> dict:
    """User-approved translation-memory entries with provenance."""
    entries = self.tm.entries()
    return {
        "count": len(entries),
        "entries": [
            {
                "fingerprint": entry["fingerprint"],
                "output_sha256": entry["output_sha256"],
                "provenance": entry["provenance"],
            }
            for entry in entries
        ],
    }


def promote_final_to_tm(
    self, run_id: str, chapter: int, chunk_id: str, *, actor: str = "user"
) -> dict:
    """Promote an approved, current human final into the translation memory.

    This is an explicit action: the machine cache is never promoted
    automatically, and a valid existing entry is never overwritten. Only an
    ``accepted`` final whose recorded machine hash still matches the current
    completed machine candidate is promoted; ``needs_review``, stale, failed or
    corrupt states are rejected. Final and machine state are read in one
    statement, so the published provenance cannot mix revisions.
    """
    self.repository.run(run_id)
    snapshot = self.repository.final_snapshot(run_id, chapter, chunk_id)
    if snapshot is None or snapshot.get("final_text") is None:
        raise ValueError("Chunk has no human final to promote")
    if snapshot["final_status"] != "accepted":
        raise ValueError("Only an accepted human final can be promoted")
    machine_row = {
        "run_id": run_id,
        "status": snapshot["chunk_status"],
        "output_path": snapshot["output_path"],
        "output_sha256": snapshot["output_sha256"],
    }
    if snapshot["chunk_status"] != "completed" or not self.stages.valid_output(machine_row):
        raise ValueError("Chunk has no valid completed machine candidate to promote")
    machine_sha = snapshot["output_sha256"]
    if snapshot["final_machine_sha256"] != machine_sha:
        raise ValueError("Human final is stale against the current machine candidate")
    metadata = json.loads(snapshot["metadata"] or "{}")
    key = metadata.get("cache_key")
    if not key:
        raise ValueError("Chunk has no cached fingerprint to promote")
    provenance = {
        "source_run_id": run_id,
        "chapter": chapter,
        "chunk_id": chunk_id,
        "machine_sha256": machine_sha,
        "final_sha256": snapshot["final_sha256"],
        "actor": actor,
        "approved_at": now(),
    }
    inserted = self.tm.put(key, text=snapshot["final_text"], provenance=provenance)
    return {"fingerprint": key, "inserted": inserted, "provenance": provenance}


def book_bible(self) -> dict:
    """Editable entries plus the deterministic authoritative projection."""
    entries = self.repository.bible_entries()
    projection = authoritative_projection(entries)
    return {
        "entries": entries,
        "authoritative_projection": projection,
        "projection_sha256": projection_sha256(projection),
        "trust_states": list(TRUST_STATES),
        "authoritative_trust": list(AUTHORITATIVE_TRUST),
        "entry_kinds": list(ENTRY_KINDS),
    }


def propose_bible_entries(self, entries: list[dict], *, proposer: str) -> dict:
    """Proposals (for example from a model) are always ``proposed``.

    A proposal never auto-confirms and never overwrites or downgrades an
    existing entry; existing ids are returned under ``skipped``.
    """
    created, skipped = [], []
    for raw in entries:
        kind = str(raw.get("kind", "")).strip()
        source_form = str(raw.get("source_form", "")).strip()
        target_form = str(raw.get("target_form", "")).strip()
        if kind not in ENTRY_KINDS:
            raise ValueError(f"Unknown book-bible kind: {kind!r}")
        if not source_form or not target_form:
            raise ValueError("Book-bible entries require source_form and target_form")
        entry_id = sha256(canonical([kind, source_form]))[:24]
        stamp = now()
        entry = {
            "entry_id": entry_id,
            "kind": kind,
            "source_form": source_form,
            "target_form": target_form,
            "aliases": [str(alias) for alias in raw.get("aliases", [])],
            "trust": "proposed",
            "notes": raw.get("notes"),
            "source_anchors": [str(anchor) for anchor in raw.get("source_anchors", [])],
            "provenance": {
                "proposer": proposer,
                "proposal_source": "model",
                "history": [
                    {
                        "action": "propose",
                        "actor": proposer,
                        "from": None,
                        "to": "proposed",
                        "at": stamp,
                    }
                ],
            },
            "created_at": stamp,
            "updated_at": stamp,
        }
        # Insert-if-absent avoids a check-then-write race: a concurrent proposal
        # for the same id cannot overwrite or downgrade an existing entry.
        if self.repository.insert_bible_entry(entry):
            created.append(entry)
        else:
            skipped.append(entry_id)
    return {"created": created, "skipped": skipped}


def set_bible_trust(
    self,
    entry_id: str,
    *,
    action: str,
    actor: str,
    target_form: str | None = None,
    aliases: list[str] | None = None,
    notes: str | None = None,
) -> dict:
    """Human review actions: confirm, lock, edit or reject an entry."""
    entry = self.repository.bible_entry(entry_id)
    if entry is None:
        raise ValueError(f"Unknown book-bible entry: {entry_id}")
    if action not in {"confirm", "lock", "edit", "reject"}:
        raise ValueError(f"Unknown book-bible action: {action}")
    if action == "reject":
        self.repository.delete_bible_entry(entry_id)
        return {"entry_id": entry_id, "status": "rejected"}
    previous = entry["trust"]
    if action == "confirm":
        entry["trust"] = "locked" if previous == "locked" else "confirmed"
    elif action == "lock":
        entry["trust"] = "locked"
    else:
        if target_form is not None:
            if not target_form.strip():
                raise ValueError("target_form must not be empty")
            entry["target_form"] = target_form
        if aliases is not None:
            entry["aliases"] = [str(alias) for alias in aliases]
        if notes is not None:
            entry["notes"] = notes
    stamp = now()
    entry["updated_at"] = stamp
    entry["provenance"].setdefault("history", []).append(
        {
            "action": action,
            "actor": actor,
            "from": previous,
            "to": entry["trust"],
            "at": stamp,
        }
    )
    self.repository.upsert_bible_entry(entry)
    return entry


def final_layer(self, run_id: str) -> list[dict]:
    self.repository.run(run_id)
    machine = {
        (row["chapter"], row["chunk_id"]): row["output_sha256"]
        for row in self.repository.chunks(run_id)
    }
    return [
        {
            **entry,
            "stale": machine.get((entry["chapter"], entry["chunk_id"])) != entry["machine_sha256"],
        }
        for entry in self.repository.final_entries(run_id)
    ]


def set_final(
    self,
    run_id: str,
    chapter: int,
    chunk_id: str,
    *,
    action: str,
    actor: str,
    text: str | None = None,
    expected_machine_sha256: str,
    expected_final_sha256: str | None = None,
    reason: str | None = None,
    conflict_error,
) -> dict:
    """Accept/edit/needs-review a chunk or revert it to the machine candidate.

    The machine candidate is never overwritten: the human layer is a separate
    record keyed by the chunk, guarded by a compare-and-set on the current
    machine ``output_sha256`` (and on the previous final hash when updating).
    """
    if action not in {"accept", "edit", "needs_review", "revert"}:
        raise ValueError(f"Unknown final-layer action: {action}")
    self.repository.run(run_id)
    row = self._chunk_row(run_id, chapter, chunk_id)
    machine_sha = row["output_sha256"]
    if not machine_sha or not row["output_path"]:
        raise ValueError("Chunk has no machine candidate to edit")
    if expected_machine_sha256 != machine_sha:
        raise conflict_error("Stale machine candidate; refresh before editing")
    existing = self.repository.final_entry(run_id, chapter, chunk_id)
    current = existing["final_sha256"] if existing else None
    if existing is not None:
        # Any mutation/deletion of an existing final must be a compare-and-set:
        # omitting the hash is not a way to bypass it.
        if expected_final_sha256 is None:
            raise conflict_error("expected_final_sha256 is required to modify an existing final")
        if expected_final_sha256 != current:
            raise conflict_error("Stale final edit; refresh before saving")
    elif expected_final_sha256 is not None and expected_final_sha256 != current:
        raise conflict_error("Stale final edit; refresh before saving")
    if action == "revert":
        if existing is None or not self.repository.delete_final_cas(
            run_id, chapter, chunk_id, machine_sha, expected_final_sha256
        ):
            raise conflict_error("Stale final edit; refresh before saving")
        return {"run_id": run_id, "chapter": chapter, "chunk_id": chunk_id, "status": "machine"}
    if text is not None and not text.strip():
        raise ValueError("Final text must not be empty")
    machine_text = (self.run_dir(run_id) / row["output_path"]).read_text("utf-8")
    # A final always records a concrete text and hash so CAS is unambiguous:
    # without explicit text, ``accept``/``needs_review`` snapshot the machine
    # candidate as the current final.
    final_text = text if text is not None else machine_text
    status = "needs_review" if action == "needs_review" else "accepted"
    entry = {
        "run_id": run_id,
        "chapter": chapter,
        "chunk_id": chunk_id,
        "status": status,
        "final_text": final_text,
        # This field is the public CAS token. A content-only hash would stay
        # unchanged for a status/reason update with identical text, allowing a
        # stale reviewer to overwrite that decision.
        "final_sha256": sha256(canonical([sha256(final_text), uuid4().hex])),
        "machine_sha256": machine_sha,
        "actor": actor,
        "reason": reason,
        "updated_at": now(),
    }
    if not self.repository.write_final_cas(entry, expected_final_sha256):
        raise conflict_error("Stale final edit; refresh before saving")
    return entry


def requeue_chunk(
    self, run_id: str, chapter: int, chunk_id: str, *, expected_machine_sha256: str, conflict_error
) -> dict:
    """Mark the chunk for regeneration without destroying the human final.

    Requeue only returns the machine chunk to the state the worker needs; the
    human final and the previous machine output stay addressable. Once the
    worker commits a new machine output, the final is stale by machine-hash
    mismatch and requires an explicit review-desk decision. No model call.
    """
    self.repository.run(run_id)
    row = self._chunk_row(run_id, chapter, chunk_id)
    if expected_machine_sha256 != row["output_sha256"]:
        raise conflict_error("Stale machine candidate; refresh before requeue")
    if not self.repository.requeue_chunk_cas(
        run_id, chapter, chunk_id, expected_machine_sha256, now()
    ):
        raise conflict_error("Stale machine candidate; refresh before requeue")
    self._mark_requeued(run_id, chapter, chunk_id)
    return {
        "run_id": run_id,
        "chapter": chapter,
        "chunk_id": chunk_id,
        "status": "pending",
    }
