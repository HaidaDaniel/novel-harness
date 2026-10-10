"""Read-only run status, chapter inspection and metrics reporting.

These projections read persisted run/chunk state and delegate to the service
only for storage, locking and output paths.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from filelock import Timeout

from .translation.assembly import assemble
from .translation.discovery import discover


class Reporter:
    """Read-only projections over persisted run and chunk state."""

    def __init__(self, service):
        self.service = service

    def lightweight_statuses(self, run_ids: list[str] | None = None) -> dict[str, dict]:
        """Return dashboard progress without manifest decoding or output hashes."""
        from .progress import ACTIVE_WORKER_STATES, worker_state

        result = {}
        for summary in self.service.repository.list_run_summaries(run_ids):
            run_id = summary["id"]
            worker = worker_state(self.service.run_dir(run_id))
            worker_active = bool(
                worker
                and worker.get("alive") is True
                and worker.get("status") in ACTIVE_WORKER_STATES
            )
            state = summary["status"]
            if worker_active:
                state = worker["status"]
            elif state == "running":
                state = "interrupted"
            retry = self.pending_retry(run_id)
            if retry and state not in {"paused", "completed"} and not worker_active:
                state = "waiting_retry"
            counts = dict(summary["counts"])
            total = summary["total_chunks"]
            result[run_id] = {
                "run_id": run_id,
                "title": summary["title"],
                "source_dir": summary["source_dir"],
                "novel_id": summary["novel_id"],
                "status": state,
                "workflow": summary["workflow"],
                "runners": (
                    [{"name": summary["runner"], "model": summary["model"], "stage": "translation"}]
                    if summary["runner"]
                    else []
                ),
                "chunks": counts,
                "total_chunks": total,
                "progress": counts["completed"] / total if total else 0,
                "active": worker_active,
                "created_at": summary["created_at"],
                "updated_at": summary["updated_at"],
                "duration_seconds": summary["duration_seconds"],
                "retry": retry,
                "error": summary["error"],
                "integrity": {"last_verified_at": None, "stale": True},
            }
        return result

    def lightweight_status(self, run_id: str) -> dict:
        """Return one dashboard projection, failing like the full status lookup."""
        status = self.lightweight_statuses([run_id]).get(run_id)
        if status is None:
            raise ValueError(f"Unknown run: {run_id}")
        return status

    def status(self, run_id: str | None = None):
        service = self.service
        if run_id is None:
            try:
                original_chapters = len(discover(service.config.novel.source_dir))
            except (ValueError, OSError):
                # A missing or malformed source must not hide persisted run history.
                original_chapters = 0
            return {
                "original_chapters": original_chapters,
                "runs": [self.status(run["id"]) for run in service.repository.list_runs()],
            }
        run = service.repository.run(run_id)
        rows = service.repository.chunks(run_id)
        counts = {"completed": 0, "pending": 0, "failed": 0, "running": 0, "corrupt": 0}
        for row in rows:
            state = row["status"]
            if state == "completed" and not service.stages.valid_output(row):
                state = "corrupt"
            counts[state] += 1
        state = run["status"]
        if state == "completed" and counts["corrupt"]:
            state = "incomplete"
        active = False
        try:
            with service._lock(run_id):
                pass
        except Timeout:
            active = True
        if state == "running" and not active:
            state = "interrupted"
        deferred = self.pending_retry(run_id)
        if (
            deferred
            and state != "paused"
            and not (service.run_dir(run_id) / "pause.requested").exists()
        ):
            state = "waiting_retry"
        manifest = run["manifest"]
        from .quality import readiness

        return {
            "run_id": run_id,
            "status": state,
            "generation_complete": counts["completed"] == len(rows),
            "quality": readiness(
                service.run_dir(run_id),
                rows,
                counts["completed"] == len(rows),
                service.repository.final_entries(run_id),
            ),
            "active": active,
            "workflow": manifest["workflow"],
            "novel_id": (manifest.get("source_identity") or {}).get("novel_id"),
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "duration_seconds": run["duration_seconds"],
            "chapters": [c["number"] for c in manifest["chapters"]],
            "runners": [
                {
                    "name": s["runner"],
                    "model": s["runner_config"]["model"],
                    "stage": s["name"],
                    "reasoning_effort": s["runner_config"].get("reasoning_effort"),
                    "execution_provenance": s.get("execution_provenance"),
                }
                for s in manifest["stages"]
            ],
            "chunks": counts,
            "total_chunks": len(rows),
            "progress": counts["completed"] / len(rows) if rows else 0,
            "error": run["error"],
            "retry": deferred,
            "continued_from": manifest.get("continued_from"),
            "parent_run_id": (manifest.get("review_input") or {}).get("parent_run_id"),
            "candidate_layer": (manifest.get("review_input") or {}).get("candidate_layer"),
            "inherited_chunks": len(manifest.get("inherited", [])),
        }

    def inspect_chapter(
        self, run_id: str, chapter: int, *, final_entries: list[dict] | None = None
    ) -> dict:
        service = self.service
        manifest = service.repository.run(run_id)["manifest"]
        source = next((c for c in manifest["chapters"] if c["number"] == chapter), None)
        if source is None:
            raise ValueError(f"Chapter {chapter} is not in run {run_id}")
        rows = [r for r in service.repository.chunks(run_id) if r["chapter"] == chapter]
        complete = all(service.stages.valid_output(row) for row in rows)
        texts = [
            (service.run_dir(run_id) / r["output_path"]).read_text("utf-8").strip()
            if service.stages.valid_output(r)
            else f"[CHUNK {r['ordinal']} {r['status'].upper()}/MISSING]"
            for r in rows
        ]
        validation = []
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            block = metadata.get("validation") or {}
            validation.append(
                {
                    "ordinal": row["ordinal"],
                    "status": row["status"],
                    "level": block.get("level"),
                    "issues": block.get("issues") or [],
                    "error": metadata.get("error"),
                    "cache": metadata.get("cache"),
                }
            )
        captured_finals = {
            (entry["chapter"], entry["chunk_id"]): entry
            for entry in (
                service.repository.final_entries(run_id) if final_entries is None else final_entries
            )
        }
        final_texts = []
        final_status = []
        for row, machine_text in zip(rows, texts):
            entry = captured_finals.get((chapter, row["chunk_id"]))
            machine_sha = row["output_sha256"]
            if entry is not None and entry.get("final_text") is not None:
                final_texts.append(entry["final_text"])
                final_status.append(
                    {
                        "ordinal": row["ordinal"],
                        "chunk_id": row["chunk_id"],
                        "source": "final",
                        "status": entry["status"],
                        "stale": entry["machine_sha256"] != machine_sha,
                        "final_sha256": entry["final_sha256"],
                        "machine_sha256": machine_sha,
                        "actor": entry["actor"],
                        "reason": entry["reason"],
                        "updated_at": entry["updated_at"],
                        "machine": machine_text,
                        "final": entry["final_text"],
                    }
                )
            else:
                final_texts.append(machine_text)
                final_status.append(
                    {
                        "ordinal": row["ordinal"],
                        "chunk_id": row["chunk_id"],
                        "source": "machine",
                        "status": entry["status"] if entry is not None else "machine",
                        "stale": False,
                        "final_sha256": None,
                        "machine_sha256": machine_sha,
                        "actor": entry["actor"] if entry is not None else None,
                        "reason": entry["reason"] if entry is not None else None,
                        "updated_at": entry["updated_at"] if entry is not None else None,
                        "machine": machine_text,
                        "final": machine_text,
                    }
                )
        return {
            "run_id": run_id,
            "chapter": chapter,
            "title": source["title"],
            "source": "".join(c["text"] for c in source["chunks"]),
            "source_sha256": source["source_sha256"],
            "parent_run_id": (manifest.get("review_input") or {}).get("parent_run_id"),
            "draft_provenance": [u["origin"] for u in (manifest.get("review_input") or {}).get("units", [])
                                 if u["chapter"] == chapter],
            "translation": assemble([c["text"] for c in source["chunks"]], texts),
            "final": assemble([c["text"] for c in source["chunks"]], final_texts),
            "final_status": final_status,
            "complete": complete,
            "chunks": rows,
            "validation": validation,
        }

    def audit(self, run_id: str) -> dict:
        from .quality import audit_chapters

        service = self.service
        with service._lock(run_id):
            manifest = service.repository.run(run_id)["manifest"]
            chapters = [
                self.inspect_chapter(run_id, c["number"]) for c in manifest["chapters"]
            ]
            return {"run_id": run_id, **audit_chapters(manifest, chapters)}

    def pending_retry(self, run_id: str) -> dict | None:
        path = self.service.run_dir(run_id) / "retry.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text("utf-8"))
        if datetime.fromisoformat(data["retry_at"]) > datetime.now(UTC):
            return data
        return None

    def metrics(self, run_id: str) -> dict:
        from .metrics import run_metrics

        return run_metrics(self.service, run_id)
