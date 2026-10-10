"""Thin FastAPI adapter over TranslationService.

Endpoints validate inputs and call service methods only; long-running translation
is delegated to the existing process worker. Requires the optional ``web`` extra
(``pip install -e ".[web]"``). Restarting the HTTP process never loses run state
because all state is persisted by the service.

The optional reader/control UI is a static client of the same
endpoints: it never translates, chunks or validates on its own, and it cannot
start translation inline because start/resume/retry all go through
``start_worker`` (the persisted process worker). Its assets are a single source of
truth under ``novel_harness/web/`` and ship as package data, so they resolve after
a normal install without depending on the working directory or a git checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config
from .progress import snapshot as progress_snapshot
from .service import ConflictError, TranslationService
from .worker import start_worker

_FALLBACK_HTML = (
    "<!doctype html><meta charset=\"utf-8\"><title>novel_harness</title>"
    "<h1>novel_harness</h1><p>The web UI assets were not found. The REST API is still "
    "available at <code>/runs</code>, <code>/runners</code> and the run endpoints.</p>"
)


class CreateRunRequest(BaseModel):
    novel_id: str | None = None
    preset: str | None = None
    execution_profile: str | None = None
    runner: str | None = None
    pipeline: str | None = None
    translator: str | None = None
    editor: str | None = None
    chapters: list[int] | None = None
    force: bool = False
    chunk_chars: int | None = None
    temperature: float | None = None
    prompt_version: str | None = None


class FinalActionRequest(BaseModel):
    action: Literal["accept", "edit", "needs_review", "revert"]
    text: str | None = None
    expected_machine_sha256: str
    expected_final_sha256: str | None = None
    actor: str = "user"
    reason: str | None = None


class RequeueRequest(BaseModel):
    expected_machine_sha256: str


class BibleEntryRequest(BaseModel):
    kind: str
    source_form: str
    target_form: str
    aliases: list[str] = []
    notes: str | None = None
    source_anchors: list[str] = []


class BibleProposeRequest(BaseModel):
    entries: list[BibleEntryRequest]
    proposer: str = "api"


class BibleActionRequest(BaseModel):
    action: Literal["confirm", "lock", "edit", "reject"]
    actor: str = "user"
    target_form: str | None = None
    aliases: list[str] | None = None
    notes: str | None = None


class PromoteTmRequest(BaseModel):
    actor: str = "user"


class SampleRunRequest(BaseModel):
    novel_id: str | None = None
    preset: str | None = None
    execution_profile: str | None = None
    runner: str | None = None
    pipeline: str | None = None
    translator: str | None = None
    editor: str | None = None
    chapters: list[int] | None = None
    chunk_chars: int | None = None
    temperature: float | None = None
    prompt_version: str | None = None


class ApproveSampleRequest(BaseModel):
    actor: str = "user"


class ImportOpenCodeProfileRequest(BaseModel):
    profile_id: str
    agent_id: str
    advanced: bool = False
    model_override: str | None = None


class ImportCodexProfileRequest(BaseModel):
    profile_id: str
    codex_profile: str | None = None


class ImportDirectProfileRequest(BaseModel):
    profile_id: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: float = 180


class WholeBookRunRequest(SampleRunRequest):
    sample_run_id: str
    approved: bool = False


#: Raw selection fields that a preset request must not also set.
_RAW_SELECTION_FIELDS = (
    "runner",
    "execution_profile",
    "pipeline",
    "translator",
    "editor",
    "prompt_version",
)


def _reject_preset_conflict(request: BaseModel) -> None:
    """Fail closed when a preset request also names raw runner/pipeline settings."""
    if getattr(request, "preset", None) and request.model_fields_set.intersection(
        _RAW_SELECTION_FIELDS
    ):
        raise HTTPException(
            status_code=400,
            detail="preset cannot be combined with explicit runner/pipeline settings",
        )


def _resolve_web_dir(explicit: str | Path | None = None) -> Path:
    """Locate the packaged static UI without depending on the working directory.

    Assets are a single source of truth under ``novel_harness/web/`` and ship as
    package data, so ``novel serve`` finds them after a normal install. An
    explicit override and ``NOVEL_WEB_DIR`` are honoured for development; the
    legacy repository-root ``web/`` and ``./web`` remain last resorts. The first
    candidate containing ``index.html`` wins; otherwise the packaged directory is
    returned so ``create_app`` can fall back to the inline page.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("NOVEL_WEB_DIR")
    if env:
        candidates.append(Path(env))
    packaged = Path(__file__).resolve().parent / "web"
    candidates.append(packaged)
    candidates.append(Path(__file__).resolve().parents[2] / "web")
    candidates.append(Path.cwd() / "web")
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return packaged


def create_app(
    config_path: str | Path = "config.yaml",
    runner_factory=None,
    web_dir: str | Path | None = None,
) -> FastAPI:
    if runner_factory is None:
        service = TranslationService.from_config(config_path)
    else:
        service = TranslationService(load_config(config_path), runner_factory)

    web_root = _resolve_web_dir(web_dir)
    app = FastAPI(title="novel_harness", version="0.1.0")
    app.state.service = service
    app.state.config_path = Path(config_path)
    app.state.web_dir = web_root

    def call(function):
        try:
            return function()
        except ConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    if web_root.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_root)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = web_root / "index.html"
        if index_path.is_file():
            return HTMLResponse(index_path.read_text("utf-8"))
        return HTMLResponse(_FALLBACK_HTML)

    @app.get("/runners")
    def list_runners():
        return service.runners()

    @app.get("/connections", tags=["connections"])
    def get_connections():
        return service.connections()

    @app.post("/connections/opencode/import", tags=["connections"])
    def import_opencode_connection(request: ImportOpenCodeProfileRequest):
        return call(
            lambda: service.import_opencode_connection(
                request.profile_id,
                request.agent_id,
                advanced=request.advanced,
                model_override=request.model_override,
            )
        )

    @app.post("/connections/codex/import", tags=["connections"])
    def import_codex_connection(request: ImportCodexProfileRequest):
        return call(
            lambda: service.import_codex_connection(
                request.profile_id, request.codex_profile
            )
        )

    @app.post("/connections/direct/import", tags=["connections"])
    def import_direct_connection(request: ImportDirectProfileRequest):
        return call(
            lambda: service.import_direct_connection(
                request.profile_id,
                model=request.model,
                base_url=request.base_url,
                api_key_env=request.api_key_env,
                timeout_seconds=request.timeout_seconds,
            )
        )

    @app.get("/presets")
    def list_presets():
        """Resolved user-facing translation modes; no model endpoint is called."""
        return {"presets": service.presets(), "default": service.default_preset()}

    @app.get("/wizard/options", tags=["wizard"])
    def wizard_options():
        return call(service.wizard_options)

    @app.get("/library", tags=["library"])
    def list_library():
        return call(service.library_novels)

    @app.post("/library/rescan", tags=["library"])
    def rescan_library():
        return call(service.rescan_library)

    @app.get("/library/{novel_id}", tags=["library"])
    def show_library_novel(novel_id: str):
        return call(lambda: service.library_show(novel_id))

    @app.get("/book-bible", tags=["book-bible"])
    def get_book_bible():
        return call(service.book_bible)

    @app.post("/book-bible", status_code=201, tags=["book-bible"])
    def propose_book_bible(request: BibleProposeRequest):
        return call(
            lambda: service.propose_bible_entries(
                [entry.model_dump() for entry in request.entries],
                proposer=request.proposer,
            )
        )

    @app.post("/book-bible/{entry_id}", tags=["book-bible"])
    def act_book_bible(entry_id: str, request: BibleActionRequest):
        return call(lambda: service.set_bible_trust(entry_id, **request.model_dump()))

    @app.get("/translation-memory", tags=["translation-memory"])
    def get_translation_memory():
        return call(service.translation_memory)

    @app.post(
        "/runs/{run_id}/chapters/{chapter}/chunks/{chunk_id}/tm", tags=["translation-memory"]
    )
    def promote_translation_memory(
        run_id: str, chapter: int, chunk_id: str, request: PromoteTmRequest
    ):
        return call(
            lambda: service.promote_final_to_tm(
                run_id, chapter, chunk_id, actor=request.actor
            )
        )

    @app.post("/sample-runs", status_code=201, tags=["sample"])
    def create_sample_run(request: SampleRunRequest):
        _reject_preset_conflict(request)
        payload = request.model_dump()
        return {"run_id": call(lambda: service.create_sample_run(**payload))}

    @app.post("/sample-runs/{run_id}/approve", tags=["sample"])
    def approve_sample(run_id: str, request: ApproveSampleRequest):
        return call(lambda: service.approve_sample(run_id, actor=request.actor))

    @app.get("/sample-runs/{run_id}/approval", tags=["sample"])
    def sample_approval(run_id: str):
        return call(lambda: service.sample_approval(run_id))

    @app.post("/whole-book-runs", status_code=201, tags=["sample"])
    def create_whole_book_run(request: WholeBookRunRequest):
        _reject_preset_conflict(request)
        payload = request.model_dump()
        if not payload.pop("approved"):
            raise HTTPException(
                status_code=409,
                detail="Whole-book launch requires explicit approval of a reviewed sample",
            )
        return {"run_id": call(lambda: service.create_whole_book_run(**payload))}

    @app.get("/runs")
    def list_runs():
        """Run dashboard projection, distinct from the source Novel Library."""
        return call(lambda: service.status())

    @app.post("/runs", status_code=201)
    def create_run(request: CreateRunRequest):
        _reject_preset_conflict(request)
        payload = request.model_dump()
        run_id = call(lambda: service.create_run(**payload))
        return {"run_id": run_id}

    @app.post("/runs/{run_id}/start", status_code=202)
    def start_run(run_id: str):
        return call(lambda: start_worker(app.state.config_path, run_id))

    @app.get("/runs/{run_id}/status")
    def run_status(run_id: str):
        return call(lambda: service.status(run_id))

    @app.post("/runs/{run_id}/pause", status_code=202)
    def pause_run(run_id: str):
        return call(lambda: service.pause(run_id))

    @app.post("/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str):
        return call(lambda: start_worker(app.state.config_path, run_id))

    @app.post("/runs/{run_id}/retry-failed", status_code=202)
    def retry_failed(run_id: str):
        """Retry failed work through the worker boundary, never inline."""
        call(lambda: service.status(run_id))
        return call(lambda: start_worker(app.state.config_path, run_id))

    @app.get("/runs/{run_id}/progress")
    def run_progress(run_id: str):
        return call(lambda: progress_snapshot(service, run_id))

    @app.get("/runs/{run_id}/final")
    def list_final(run_id: str):
        return call(lambda: service.final_layer(run_id))

    @app.post("/runs/{run_id}/chapters/{chapter}/chunks/{chunk_id}/final")
    def set_final(run_id: str, chapter: int, chunk_id: str, request: FinalActionRequest):
        return call(
            lambda: service.set_final(
                run_id, chapter, chunk_id, **request.model_dump()
            )
        )

    @app.post(
        "/runs/{run_id}/chapters/{chapter}/chunks/{chunk_id}/requeue", status_code=202
    )
    def requeue_chunk(run_id: str, chapter: int, chunk_id: str, request: RequeueRequest):
        result = call(
            lambda: service.requeue_chunk(
                run_id, chapter, chunk_id, expected_machine_sha256=request.expected_machine_sha256
            )
        )
        worker = call(lambda: start_worker(app.state.config_path, run_id))
        return {**result, "worker": worker}

    @app.get("/runs/{run_id}/chapters/{chapter}/layers")
    def chapter_layers(run_id: str, chapter: int):
        def projection():
            data = service.inspect_chapter(run_id, chapter)
            return {
                "run_id": run_id,
                "chapter": chapter,
                "title": data["title"],
                "source": data["source"],
                "machine": data["translation"],
                "final": data["final"],
                "final_status": data["final_status"],
            }

        return call(projection)

    @app.get("/runs/{run_id}/chapters/{chapter}")
    def inspect_chapter(run_id: str, chapter: int):
        return call(lambda: service.inspect_chapter(run_id, chapter))

    @app.get("/runs/{run_id}/metrics")
    def run_metrics(run_id: str):
        return call(lambda: service.metrics(run_id))

    @app.get("/compare")
    def compare(chapter: int, runs: str):
        run_ids = [run_id.strip() for run_id in runs.split(",") if run_id.strip()]
        return call(lambda: service.compare(chapter, run_ids))

    @app.get("/runs/{run_id}/audit")
    def audit(run_id: str):
        return call(lambda: service.audit(run_id))

    @app.post("/runs/{run_id}/build")
    def build(run_id: str):
        path = call(lambda: service.build_epub(run_id))
        return {"path": str(path)}

    return app
