"""Offline tests for user-facing translation presets.

No model inference, no network: runner availability is configuration-only and run
creation freezes inputs without executing anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from novel_harness.api import create_app
from novel_harness.config import load_config
from novel_harness.presets import resolve_preset
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService


class FakeRunner:
    def __init__(self):
        self.calls = 0

    def run(self, job):
        self.calls += 1
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


def _draft_light() -> dict:
    return {
        "stages": [
            {"name": "translation", "role": "translator", "prompt_version": "translate-draft-v1"}
        ]
    }


def configure_presets(project: Path, presets: dict, default: str | None = "local-quality") -> Path:
    data = yaml.safe_load(project.read_text("utf-8"))
    data.setdefault("pipelines", {})
    data["pipelines"].setdefault("draft-light", _draft_light())
    data["translation_presets"] = {"default": default, "presets": presets}
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    return project


def standard_presets(runner: str = "test") -> dict:
    return {
        "quality": {
            "label": "Quality",
            "description": "Best measured quality",
            "runner": runner,
            "pipeline": "draft-light",
            "category": "quality",
            "cloud": True,
        },
        "local-quality": {
            "label": "Local Quality",
            "description": "Recommended local mode",
            "runner": runner,
            "pipeline": "draft-light",
            "category": "local-quality",
            "recommended": True,
        },
        "local-fast": {
            "label": "Local Fast",
            "description": "Faster local mode",
            "runner": runner,
            "pipeline": "direct",
            "category": "local-fast",
        },
        "cheap-draft": {
            "label": "Cheap Draft",
            "description": "Small-model rough draft",
            "runner": "missing-9b",
            "pipeline": "draft-light",
            "category": "cheap-draft",
        },
    }


def configure_practical_default(project: Path) -> Path:
    """Offline B/R fixture with the same preset and pipeline split as Qwen95."""
    configure_presets(project, standard_presets(), default="local-fast")
    data = yaml.safe_load(project.read_text("utf-8"))
    data["runners"]["test-b"] = {**data["runners"]["test"], "model": "b-model"}
    data["runners"]["test-b-alias"] = dict(data["runners"]["test-b"])
    data["runners"]["test-r"] = {**data["runners"]["test"], "model": "r-model"}
    data["translation_presets"]["auto_select_default"] = True
    data["translation_presets"]["presets"]["local-fast"].update(
        runner="test-b", pipeline="draft-light", recommended=True
    )
    data["translation_presets"]["presets"]["local-quality"].update(
        runner="test-r", recommended=False
    )
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    return project


def _service(project: Path, runner: FakeRunner | None = None) -> TranslationService:
    runner = runner or FakeRunner()
    return TranslationService(load_config(project), lambda name: runner)


def _client(project: Path, runner: FakeRunner | None = None):
    app = create_app(project, runner_factory=lambda name: runner or FakeRunner())
    return TestClient(app), app


# --------------------------------------------------------------- config


def test_presets_parse_default_and_listing(project):
    configure_presets(project, standard_presets())
    service = _service(project)
    entries = {entry["id"]: entry for entry in service.presets()}
    assert service.default_preset() == "local-quality"
    assert entries["local-quality"]["available"] is True
    assert entries["local-quality"]["default"] is True
    assert entries["local-quality"]["pipeline"] == "draft-light"
    assert entries["local-quality"]["model"] == "fake-model"
    assert entries["quality"]["cloud"] is True
    assert entries["local-fast"]["pipeline"] == "direct"


def test_opted_in_default_applies_to_new_service_and_api_runs(project, source):
    configure_presets(project, standard_presets())
    data = yaml.safe_load(project.read_text("utf-8"))
    data["translation_presets"]["auto_select_default"] = True
    data["runners"]["test-alias"] = dict(data["runners"]["test"])
    data["runners"]["other"] = {**data["runners"]["test"], "model": "other-model"}
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")

    service = _service(project)
    run_id = service.create_run(chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["preset"]["preset_id"] == "local-quality"
    assert manifest["workflow"] == "draft-light"
    assert manifest["stages"][0]["runner"] == "test"
    alias_id = service.create_run(runner="test-alias", chapters=[1])
    alias = service.repository.run(alias_id)["manifest"]
    assert alias["workflow"] == "direct"
    assert alias["stages"][0]["runner"] == "test-alias"
    assert alias.get("preset") is None

    sample_id = service.create_sample_run(chapters=[1])
    sample = service.repository.run(sample_id)["manifest"]
    assert sample["preset"]["preset_id"] == "local-quality"
    assert sample["workflow"] == "draft-light"

    explicit_id = service.create_run(runner="other", chapters=[1])
    explicit = service.repository.run(explicit_id)["manifest"]
    assert explicit.get("preset") is None
    assert explicit["workflow"] == "direct"

    client, _ = _client(project)
    body = client.post("/runs", json={"chapters": [1]}).json()
    api_manifest = service.repository.run(body["run_id"])["manifest"]
    assert api_manifest["preset"]["preset_id"] == "local-quality"


def test_practical_default_and_explicit_direct_service_and_sample(project, source):
    configure_practical_default(project)
    service = _service(project)

    default_id = service.create_run(chapters=[1])
    default = service.repository.run(default_id)["manifest"]
    assert default["preset"]["preset_id"] == "local-fast"
    assert default["workflow"] == "draft-light"
    assert default["stages"][0]["runner"] == "test-b"

    for runner in ("test-b", "test-b-alias"):
        raw_id = service.create_run(runner=runner, pipeline="direct", chapters=[1])
        raw = service.repository.run(raw_id)["manifest"]
        assert raw["workflow"] == "direct"
        assert raw["stages"][0]["runner"] == runner
        assert raw.get("preset") is None
    with pytest.raises(ValueError, match="Configure runner"):
        service.create_run(pipeline="direct", chapters=[1])

    for preset, runner in (("local-quality", "test-r"), ("local-fast", "test-b")):
        run_id = service.create_run(preset=preset, chapters=[1])
        manifest = service.repository.run(run_id)["manifest"]
        assert manifest["preset"]["preset_id"] == preset
        assert manifest["workflow"] == "draft-light"
        assert manifest["stages"][0]["runner"] == runner

    with pytest.raises(ValueError, match="conflicts with an explicit pipeline"):
        service.create_run(preset="local-quality", pipeline="direct", chapters=[1])

    sample_id = service.create_sample_run(chapters=[1])
    sample = service.repository.run(sample_id)["manifest"]
    assert sample["preset"]["preset_id"] == "local-fast"
    raw_sample_id = service.create_sample_run(runner="test-b", pipeline="direct", chapters=[1])
    raw_sample = service.repository.run(raw_sample_id)["manifest"]
    assert raw_sample["workflow"] == "direct" and raw_sample.get("preset") is None


def test_other_explicit_service_selections_do_not_invoke_auto_default(project, source):
    configure_practical_default(project)
    service = _service(project)
    translated = service.create_run(translator="test-b", chapters=[1])
    manifest = service.repository.run(translated)["manifest"]
    assert manifest["workflow"] == "direct" and manifest.get("preset") is None
    for selection in (
        {"pipeline": "draft-light"},
        {"editor": "test-b"},
        {"prompt_version": "translate-v1"},
    ):
        with pytest.raises(ValueError, match="Configure runner"):
            service.create_run(chapters=[1], **selection)
    with pytest.raises(ValueError, match="Unknown execution profile"):
        service.create_run(chapters=[1], execution_profile="missing-profile")


def test_practical_default_api_preserves_explicit_raw_selection(project, source):
    configure_practical_default(project)
    client, app = _client(project)
    listing = client.get("/presets").json()
    assert listing["default"] == "local-fast"
    b_entry = next(entry for entry in listing["presets"] if entry["id"] == "local-fast")
    assert b_entry["default"] is True and b_entry["recommended"] is True
    for path in ("/runs", "/sample-runs"):
        for payload in ({"chapters": [1]}, {"chapters": [1], "pipeline": None}):
            default = client.post(path, json=payload)
            assert default.status_code == 201, default.text
            manifest = app.state.service.repository.run(default.json()["run_id"])["manifest"]
            assert manifest["preset"]["preset_id"] == "local-fast"
            assert manifest["workflow"] == "draft-light"
            assert manifest["stages"][0]["runner"] == "test-b"

        raw = client.post(path, json={"runner": "test-b", "pipeline": "direct", "chapters": [1]})
        assert raw.status_code == 201, raw.text
        manifest = app.state.service.repository.run(raw.json()["run_id"])["manifest"]
        assert manifest["workflow"] == "direct"
        assert manifest.get("preset") is None

        no_runner = client.post(path, json={"pipeline": "direct", "chapters": [1]})
        assert no_runner.status_code == 400
        assert "Configure runner" in no_runner.json()["detail"]

        named_translator = client.post(path, json={"translator": "test-b", "chapters": [1]})
        assert named_translator.status_code == 201, named_translator.text
        manifest = app.state.service.repository.run(named_translator.json()["run_id"])["manifest"]
        assert manifest["workflow"] == "direct" and manifest.get("preset") is None


def test_frozen_quality_sample_survives_default_change(project, source):
    configure_practical_default(project)
    data = yaml.safe_load(project.read_text("utf-8"))
    data["translation_presets"]["default"] = "local-quality"
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    service = _service(project)
    sample_id = service.create_sample_run(chapters=[1])
    before = service.repository.run(sample_id)["manifest"]
    service.execute(sample_id)
    service.approve_sample(sample_id)
    data["translation_presets"]["default"] = "local-fast"
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    service.config = load_config(project)
    service._verify_inputs(before)
    whole_id = service.create_whole_book_run(sample_id)
    whole = service.repository.run(whole_id)["manifest"]
    assert whole["preset"]["preset_id"] == "local-quality"
    assert whole["workflow"] == "draft-light"
    assert whole["stages"][0]["runner"] == "test-r"
    assert service.repository.run(sample_id)["manifest"] == before


def test_auto_default_requires_a_configured_default(project):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["translation_presets"] = {"auto_select_default": True, "presets": {}}
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    with pytest.raises(ValueError, match="requires a default preset"):
        load_config(project)


def test_missing_optional_runner_marks_preset_unavailable(project):
    configure_presets(project, standard_presets())
    service = _service(project)
    entries = {entry["id"]: entry for entry in service.presets()}
    assert entries["cheap-draft"]["available"] is False
    assert entries["cheap-draft"]["reason"] == "configured runner missing"


def test_invalid_pipeline_reference_rejected(project):
    presets = standard_presets()
    presets["local-quality"]["pipeline"] = "not-a-pipeline"
    configure_presets(project, presets)
    with pytest.raises(ValueError, match="unknown pipeline"):
        load_config(project)


def test_invalid_default_reference_rejected(project):
    configure_presets(project, standard_presets(), default="does-not-exist")
    with pytest.raises(ValueError, match="default must name"):
        load_config(project)


def test_invalid_preset_id_rejected(project):
    presets = standard_presets()
    presets["bad id!"] = presets.pop("local-quality")
    configure_presets(project, presets)
    with pytest.raises(ValueError, match="Invalid identifier"):
        load_config(project)


def test_empty_preset_config_keeps_raw_behavior(project, source):
    data = yaml.safe_load(project.read_text("utf-8"))
    data.pop("translation_presets", None)
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    service = _service(project)
    assert service.presets() == []
    assert service.default_preset() is None
    run_id = service.create_run(runner="test", chapters=[1])
    assert service.repository.run(run_id)["manifest"].get("preset") is None


# ----------------------------------------------------------- resolution


def test_resolve_preset_returns_exact_runner_and_pipeline(project):
    configure_presets(project, standard_presets())
    resolved = resolve_preset(load_config(project), "local-quality")
    assert resolved["runner"] == "test"
    assert resolved["pipeline"] == "draft-light"
    assert resolved["preset_id"] == "local-quality"


# ------------------------------------------------- raw manifest compatibility


def _add_draft_light_and_edit(project):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["pipelines"]["draft-light"] = _draft_light()
    data["pipelines"]["translate-edit"] = {
        "stages": [
            {"name": "translation", "role": "translator", "prompt_version": "translate-v1"},
            {"name": "editing", "role": "editor", "prompt_version": "edit-v1"},
        ]
    }
    data["runners"]["editor"] = data["runners"]["test"]
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")


def test_raw_direct_keeps_legacy_translation_prompt_version(project, source):
    """translation.prompt_version keeps the global default; stages carry the prompt."""
    _add_draft_light_and_edit(project)
    service = _service(project)
    run_id = service.create_run(runner="test", pipeline="direct", chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["translation"]["prompt_version"] == "translate-v1"
    assert manifest["stages"][0]["prompt_version"] == "translate-v1"
    assert manifest.get("preset") is None


def test_raw_draft_light_does_not_rewrite_global_prompt_version(project, source):
    """A raw draft-light run must NOT rewrite translation.prompt_version."""
    _add_draft_light_and_edit(project)
    service = _service(project)
    run_id = service.create_run(runner="test", pipeline="draft-light", chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    # Legacy semantics: the global setting stays as configured (translate-v1),
    # while the effective prompt is frozen in the stage.
    assert manifest["translation"]["prompt_version"] == "translate-v1"
    assert manifest["stages"][0]["prompt_version"] == "translate-draft-v1"
    assert manifest["stages"][0]["prompt_sha256"]
    assert manifest.get("preset") is None


def test_raw_multi_stage_preserves_stage_prompts(project, source):
    _add_draft_light_and_edit(project)
    service = _service(project)
    run_id = service.create_run(
        runner="test", pipeline="translate-edit", translator="test", editor="editor", chapters=[1]
    )
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["translation"]["prompt_version"] == "translate-v1"
    assert [s["prompt_version"] for s in manifest["stages"]] == ["translate-v1", "edit-v1"]
    assert [s["runner"] for s in manifest["stages"]] == ["test", "editor"]


def test_raw_explicit_prompt_version_preserved(project, source):
    _add_draft_light_and_edit(project)
    service = _service(project)
    run_id = service.create_run(
        runner="test", pipeline="draft-light", prompt_version="translate-v1", chapters=[1]
    )
    manifest = service.repository.run(run_id)["manifest"]
    # Historical semantics: an explicit override rewrites the global value too.
    assert manifest["translation"]["prompt_version"] == "translate-v1"
    assert manifest["stages"][0]["prompt_version"] == "translate-v1"


def test_preset_run_keeps_legacy_global_prompt_version(project, source):
    """A preset run must not rewrite translation.prompt_version either."""
    _add_draft_light_and_edit(project)
    configure_presets(project, standard_presets())
    service = _service(project)
    run_id = service.create_run(preset="local-quality", chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["translation"]["prompt_version"] == "translate-v1"
    assert manifest["stages"][0]["prompt_version"] == "translate-draft-v1"


# ---------------------------------------------------- preset availability


def test_availability_checks_translator_and_editor(project):
    _add_draft_light_and_edit(project)
    presets = standard_presets()
    presets["with-editor"] = {
        "label": "With Editor",
        "description": "",
        "runner": "test",
        "pipeline": "translate-edit",
        "editor": "missing-editor",
    }
    configure_presets(project, presets)
    entries = {entry["id"]: entry for entry in _service(project).presets()}
    assert entries["with-editor"]["available"] is False
    assert "missing-editor" in entries["with-editor"]["reason"]
    assert entries["local-quality"]["available"] is True


def test_availability_checks_explicit_stage_runner(project):
    _add_draft_light_and_edit(project)
    data = yaml.safe_load(project.read_text("utf-8"))
    data["pipelines"]["draft-light"]["stages"][0]["runner"] = "missing-stage-runner"
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    configure_presets(project, standard_presets())
    entries = {entry["id"]: entry for entry in _service(project).presets()}
    assert entries["local-quality"]["available"] is False
    assert "missing-stage-runner" in entries["local-quality"]["reason"]


def test_availability_unconfigured_primary_runner(project):
    _add_draft_light_and_edit(project)
    entries = {entry["id"]: entry for entry in _service(project).presets()}
    assert entries["cheap-draft"]["available"] is False
    assert entries["cheap-draft"]["reason"] == "configured runner missing"


def test_availability_all_required_runners_present(project):
    _add_draft_light_and_edit(project)
    configure_presets(project, standard_presets())
    entries = {entry["id"]: entry for entry in _service(project).presets()}
    assert entries["local-quality"]["available"] is True
    assert entries["quality"]["available"] is True


def test_availability_checks_implicit_editor_fallback(project):
    """A multi-stage pipeline without an explicit stage runner still needs the editor."""
    _add_draft_light_and_edit(project)
    presets = standard_presets()
    presets["with-editor-pipeline"] = {
        "label": "Editor Pipeline",
        "description": "",
        "runner": "test",
        "pipeline": "translate-edit",
        "editor": "missing-editor",
    }
    configure_presets(project, presets)
    entries = {entry["id"]: entry for entry in _service(project).presets()}
    # translate-edit has an editor stage with no explicit stage.runner; the preset's
    # editor is the effective runner and must be validated.
    assert entries["with-editor-pipeline"]["available"] is False
    assert "missing-editor" in entries["with-editor-pipeline"]["reason"]


def test_availability_rejects_multi_stage_without_editor(project):
    """A pipeline with an editor stage but no editor runner must be unavailable."""
    _add_draft_light_and_edit(project)
    presets = standard_presets()
    presets["editorless"] = {
        "label": "Editorless",
        "description": "",
        "runner": "test",
        "pipeline": "translate-edit",
    }
    configure_presets(project, presets)
    entries = {entry["id"]: entry for entry in _service(project).presets()}
    assert entries["editorless"]["available"] is False
    assert "editing" in entries["editorless"]["reason"]
    assert "no runner configured" in entries["editorless"]["reason"]


def test_availability_rejects_unconfigured_model_placeholder(project):
    """A required runner with no model must not present as available."""
    _add_draft_light_and_edit(project)
    data = yaml.safe_load(project.read_text("utf-8"))
    data["runners"]["test"]["model"] = None
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    configure_presets(project, standard_presets())
    entries = {entry["id"]: entry for entry in _service(project).presets()}
    assert entries["local-quality"]["available"] is False
    assert entries["local-quality"]["reason"] == "Model ID not configured"


def test_create_run_with_preset_freezes_provenance(project, source):
    configure_presets(project, standard_presets())
    service = _service(project)
    run_id = service.create_run(preset="local-quality", chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["preset"]["preset_id"] == "local-quality"
    assert manifest["preset"]["runner"] == "test"
    assert manifest["preset"]["pipeline"] == "draft-light"
    assert manifest["workflow"] == "draft-light"


def test_preset_edit_does_not_reinterpret_existing_run(project, source):
    configure_presets(project, standard_presets())
    service = _service(project)
    run_id = service.create_run(preset="local-quality", chapters=[1])
    before = service.repository.run(run_id)["manifest"]
    # Rewrite the preset to a different pipeline after the run was frozen.
    presets = standard_presets()
    presets["local-quality"]["pipeline"] = "direct"
    configure_presets(project, presets)
    service.config = load_config(project)
    after = service.repository.run(run_id)["manifest"]
    assert after == before
    assert after["workflow"] == "draft-light"


def test_raw_legacy_invocation_still_works(project, source):
    configure_presets(project, standard_presets())
    service = _service(project)
    run_id = service.create_run(runner="test", pipeline="direct", chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["workflow"] == "direct"
    assert manifest.get("preset") is None


def test_preset_conflicting_with_raw_runner_fails_closed(project):
    configure_presets(project, standard_presets())
    service = _service(project)
    with pytest.raises(ValueError, match="conflicts"):
        service.create_run(preset="local-quality", runner="other", chapters=[1])
    with pytest.raises(ValueError, match="conflicts"):
        service.create_run(preset="local-quality", pipeline="translate-review", chapters=[1])
    with pytest.raises(ValueError, match="conflicts"):
        service.create_run(preset="local-quality", prompt_version="translate-v1", chapters=[1])


def test_unavailable_preset_cannot_create_run(project):
    configure_presets(project, standard_presets())
    service = _service(project)
    with pytest.raises(ValueError, match="unavailable"):
        service.create_run(preset="cheap-draft", chapters=[1])


# ------------------------------------------------------------------ API


def test_presets_endpoint_reports_default_and_availability(project):
    configure_presets(project, standard_presets())
    client, _ = _client(project)
    body = client.get("/presets").json()
    assert body["default"] == "local-quality"
    entries = {entry["id"]: entry for entry in body["presets"]}
    assert entries["local-quality"]["available"] is True
    assert entries["cheap-draft"]["available"] is False
    assert entries["cheap-draft"]["reason"] == "configured runner missing"


def test_create_run_with_preset_over_api(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    created = client.post("/runs", json={"preset": "local-quality", "chapters": [1]})
    assert created.status_code == 201
    manifest = app.state.service.repository.run(created.json()["run_id"])["manifest"]
    assert manifest["preset"]["preset_id"] == "local-quality"
    assert manifest["workflow"] == "draft-light"


def test_sample_with_preset_over_api(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    created = client.post("/sample-runs", json={"preset": "local-quality", "chapters": [1]})
    assert created.status_code == 201
    manifest = app.state.service.repository.run(created.json()["run_id"])["manifest"]
    assert manifest["preset"]["preset_id"] == "local-quality"
    assert manifest["sample"]["role"] == "sample"


def test_unavailable_preset_over_api_returns_400(project):
    configure_presets(project, standard_presets())
    client, _ = _client(project)
    response = client.post("/runs", json={"preset": "cheap-draft", "chapters": [1]})
    assert response.status_code == 400
    assert "unavailable" in response.json()["detail"]


def test_legacy_raw_request_still_valid_over_api(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    created = client.post("/runs", json={"runner": "test", "pipeline": "direct", "chapters": [1]})
    assert created.status_code == 201
    manifest = app.state.service.repository.run(created.json()["run_id"])["manifest"]
    assert manifest["workflow"] == "direct"
    assert manifest.get("preset") is None


def test_api_rejects_preset_with_raw_selection(project):
    configure_presets(project, standard_presets())
    client, _ = _client(project)
    response = client.post(
        "/runs", json={"preset": "local-quality", "runner": "test", "chapters": [1]}
    )
    assert response.status_code == 400
    assert "preset cannot be combined" in response.json()["detail"]


def _complete_sample(client, app, sample_id):
    """Execute the frozen sample offline so approval requirements are met."""
    app.state.service.execute(sample_id)
    status = client.get(f"/runs/{sample_id}/status").json()
    assert status["generation_complete"] is True
    assert status["chunks"]["failed"] == 0
    return sample_id


def test_approve_requires_completed_sample(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    sample_id = client.post(
        "/sample-runs", json={"preset": "local-quality", "chapters": [1]}
    ).json()["run_id"]

    # A newly created (incomplete) sample cannot be approved.
    rejected = client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})
    assert rejected.status_code == 400
    assert "must complete successfully before approval" in rejected.json()["detail"]
    assert client.get(f"/sample-runs/{sample_id}/approval").json() is None

    # After offline completion, approval succeeds and whole-book is allowed.
    _complete_sample(client, app, sample_id)
    approved = client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})
    assert approved.status_code == 200
    whole = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample_id, "approved": True, "preset": "local-quality"},
    )
    assert whole.status_code == 201


def test_approve_rejected_for_failed_sample(project, source):
    configure_presets(project, standard_presets())

    class FailingRunner:
        def run(self, job):
            return RunnerResult("", finish_reason="length")

    app = create_app(project, runner_factory=lambda name: FailingRunner())
    client = TestClient(app)
    sample_id = client.post(
        "/sample-runs", json={"preset": "local-quality", "chapters": [1]}
    ).json()["run_id"]
    app.state.service.execute(sample_id)
    status = client.get(f"/runs/{sample_id}/status").json()
    assert status["chunks"]["failed"] > 0

    rejected = client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})
    assert rejected.status_code == 400
    assert "must complete successfully before approval" in rejected.json()["detail"]


def test_approve_rejected_when_run_status_failed_even_with_valid_chunks(project, source):
    """A run marked failed must not be approvable even if every chunk validated."""
    configure_presets(project, standard_presets())
    service = _service(project)
    sample_id = service.create_sample_run(preset="local-quality", chapters=[1])
    service.execute(sample_id)
    # Force the run-level state to failed after all chunks committed.
    service.repository.set_run(sample_id, "failed", "2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="must complete successfully before approval"):
        service.approve_sample(sample_id)


# ---------------------------------------------------------- sample first


def test_whole_book_reuses_sample_preset_and_settings(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    sample = client.post("/sample-runs", json={"preset": "local-quality", "chapters": [1]})
    sample_id = sample.json()["run_id"]
    _complete_sample(client, app, sample_id)
    client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})

    whole = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample_id, "approved": True, "preset": "local-quality"},
    )
    assert whole.status_code == 201
    whole_manifest = app.state.service.repository.run(whole.json()["run_id"])["manifest"]
    sample_manifest = app.state.service.repository.run(sample_id)["manifest"]
    assert whole_manifest["preset"]["preset_id"] == "local-quality"
    assert whole_manifest["workflow"] == sample_manifest["workflow"]
    assert whole_manifest["translation"]["chunk_chars"] == sample_manifest["translation"]["chunk_chars"]
    assert whole_manifest["sample"]["sample_run_id"] == sample_id


def test_whole_book_refuses_different_preset(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    sample_id = client.post(
        "/sample-runs", json={"preset": "local-quality", "chapters": [1]}
    ).json()["run_id"]
    _complete_sample(client, app, sample_id)
    client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})

    response = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample_id, "approved": True, "preset": "local-fast"},
    )
    assert response.status_code == 400
    assert "differs from the approved sample" in response.json()["detail"]


def test_whole_book_refuses_config_drift_after_approval(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    sample_id = client.post(
        "/sample-runs", json={"preset": "local-quality", "chapters": [1]}
    ).json()["run_id"]
    _complete_sample(client, app, sample_id)
    client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})

    # Translation-relevant drift after approval must not silently reuse approval.
    app.state.service.config.translation.max_tokens += 1
    response = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample_id, "approved": True, "preset": "local-quality"},
    )
    assert response.status_code == 400
    assert "new sample" in response.json()["detail"]


def test_whole_book_refuses_book_bible_drift(project, source):
    configure_presets(project, standard_presets())
    client, app = _client(project)
    sample_id = client.post(
        "/sample-runs", json={"preset": "local-quality", "chapters": [1]}
    ).json()["run_id"]
    _complete_sample(client, app, sample_id)
    client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})

    entry = client.post(
        "/book-bible",
        json={
            "entries": [
                {
                    "kind": "character",
                    "source_form": "ユキ",
                    "target_form": "Юки",
                    "aliases": [],
                    "source_anchors": ["ユキは走った"],
                }
            ],
            "proposer": "test",
        },
    ).json()["created"][0]
    client.post(f"/book-bible/{entry['entry_id']}", json={"action": "confirm", "actor": "ui"})

    response = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample_id, "approved": True, "preset": "local-quality"},
    )
    assert response.status_code == 400
    assert "new sample" in response.json()["detail"]


# ----------------------------------------------------------- UI contract


def test_index_exposes_preset_first_ui(project):
    configure_presets(project, standard_presets())
    client, _ = _client(project)
    html = client.get("/").text
    assert "Translation mode" in html
    assert 'id="create-presets"' in html
    assert 'id="sample-presets"' in html
    assert "Custom / Advanced" in html
    assert "Start Sample" in html
    assert "Translate Whole Book" in html


def test_static_app_js_loads_presets_without_hardcoded_runners(project):
    configure_presets(project, standard_presets())
    client, _ = _client(project)
    script = client.get("/static/app.js").text
    assert "loadPresets" in script
    assert "/presets" in script
    assert "selectionPayload" in script
    # The configured default preset (not raw custom/direct) is the checked radio, and
    # an unavailable default falls back to Custom with a visible warning.
    assert "resolveInitialPreset" in script
    assert "Default preset unavailable" in script
    # No hardcoded model identity in the frontend preset layer.
    assert "qwen-server" not in script
    assert "luna-xhigh" not in script


def test_unavailable_default_preset_not_auto_selected_in_ui(project):
    """An unavailable configured default resolves to Custom, never another preset."""
    _add_draft_light_and_edit(project)
    presets = standard_presets()
    presets["local-quality"]["runner"] = "missing-runner"
    configure_presets(project, presets, default="local-quality")
    client, _ = _client(project)

    body = client.get("/presets").json()
    assert body["default"] == "local-quality"
    default_entry = next(entry for entry in body["presets"] if entry["id"] == "local-quality")
    assert default_entry["available"] is False
    assert default_entry["reason"] == "configured runner missing"

    script = client.get("/static/app.js").text
    # Default is only used when available; otherwise the UI falls back to custom.
    assert "if (preset && preset.available) return requested;" in script
    assert 'return "custom";' in script


def test_whole_book_reuses_explicit_prompt_version(project, source):
    configure_presets(project, standard_presets())
    service = _service(project)
    sample_id = service.create_sample_run(
        runner="test", pipeline="draft-light", prompt_version="translate-v1", chapters=[1]
    )
    service.execute(sample_id)
    service.approve_sample(sample_id)
    whole_id = service.create_whole_book_run(sample_id)
    whole = service.repository.run(whole_id)["manifest"]
    sample = service.repository.run(sample_id)["manifest"]
    assert whole["stages"][0]["prompt_version"] == sample["stages"][0]["prompt_version"]
    assert whole["sample"]["sample_run_id"] == sample_id
    # A different explicit prompt version is refused rather than silently accepted.
    with pytest.raises(ValueError, match="new sample"):
        service.create_whole_book_run(sample_id, prompt_version="translate-draft-v1")


def test_default_preset_is_local_quality_for_configured_ui(project):
    configure_presets(project, standard_presets())
    client, _ = _client(project)
    body = client.get("/presets").json()
    assert body["default"] == "local-quality"
    default_entry = next(entry for entry in body["presets"] if entry["default"])
    assert default_entry["pipeline"] == "draft-light"
    assert default_entry["category"] == "local-quality"


def test_service_still_uses_direct_python_default(project, source):
    """The low-level service default must stay pipeline='direct'."""
    configure_presets(project, standard_presets())
    service = _service(project)
    run_id = service.create_run(runner="test", chapters=[1])
    assert service.repository.run(run_id)["manifest"]["workflow"] == "direct"
