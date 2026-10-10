import inspect
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Barrier, Event

import pytest
import yaml
from conftest import FakeRunner
from filelock import FileLock

from novel_harness import run_operations
from novel_harness.artifacts import canonical, sha256, write_json, write_text
from novel_harness.config import RunnerConfig, load_config
from novel_harness.persistence import Repository
from novel_harness.runners.base import RunnerResult
from novel_harness.service import ConflictError, TranslationService
from novel_harness.worker import start_worker


@pytest.mark.parametrize(
    ("method", "positional", "keyword_only", "required"),
    [
        (
            "create_run", "self",
            (
                "execution_profile runner pipeline translator editor chapters force chunk_chars "
                "temperature prompt_version preset sample novel_id"
            ), "",
        ),
        (
            "create_sample_run", "self",
            (
                "execution_profile runner pipeline translator editor chapters chunk_chars "
                "temperature prompt_version preset novel_id"
            ), "",
        ),
        (
            "create_whole_book_run", "self sample_run_id",
            (
                "execution_profile preset runner pipeline translator editor chapters chunk_chars "
                "temperature prompt_version novel_id"
            ), "sample_run_id",
        ),
        ("execute", "self run_id", "max_chunks clear_pause max_parallel", "run_id"),
        ("resume", "self run_id", "", "run_id"),
        ("pause", "self run_id", "", "run_id"),
        ("retry_failed", "self run_id", "", "run_id"),
        ("status", "self run_id", "", ""),
        ("inspect_chapter", "self run_id chapter", "", "run_id chapter"),
        ("build_epub", "self run_id", "metadata_path", "run_id"),
        (
            "set_final", "self run_id chapter chunk_id",
            "action actor text expected_machine_sha256 expected_final_sha256 reason",
            "run_id chapter chunk_id action actor expected_machine_sha256",
        ),
        (
            "requeue_chunk", "self run_id chapter chunk_id", "expected_machine_sha256",
            "run_id chapter chunk_id expected_machine_sha256",
        ),
    ],
)
def test_external_facade_signatures_keep_selected_call_shapes(
    method, positional, keyword_only, required
):
    """Protect selected CLI/API/MCP/worker methods, not private helpers."""
    signature = inspect.signature(getattr(TranslationService, method))
    parameters = signature.parameters
    assert tuple(
        name for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ) == tuple(positional.split())
    assert tuple(
        name for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ) == tuple(keyword_only.split())
    assert len(parameters) == len(positional.split()) + len(keyword_only.split())
    assert {
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.default is inspect.Parameter.empty
    } == set(required.split())
    if method in {"create_run", "create_sample_run", "create_whole_book_run"}:
        for name in ("execution_profile", "runner", "pipeline", "preset", "novel_id"):
            assert parameters[name].default is None
    if method == "status":
        assert parameters["run_id"].default is None
    if method == "execute":
        assert parameters["max_chunks"].default is None
        assert parameters["clear_pause"].default is True
        assert parameters["max_parallel"].default == 1
    if method == "build_epub":
        assert parameters["metadata_path"].default is None


def test_public_facade_read_surfaces_keep_their_shapes_without_execution(service):
    svc, fake = service
    assert isinstance(svc.runners(), list)
    assert isinstance(svc.connections(), dict)
    assert isinstance(svc.presets(), list)
    assert svc.default_preset() == "local-fast"
    assert set(svc.library_novels()) == {"novels", "diagnostics"}
    assert set(svc.rescan_library()) == {"novels", "diagnostics"}
    assert svc.translation_memory() == {"count": 0, "entries": []}
    bible = svc.book_bible()
    assert bible["entries"] == []
    assert bible["authoritative_projection"] == []
    assert set(bible) == {
        "entries", "authoritative_projection", "projection_sha256",
        "trust_states", "authoritative_trust", "entry_kinds",
    }
    assert fake.calls == []


def test_repository_resolves_bounded_store_local_run_selectors(tmp_path):
    repository = Repository(tmp_path / "state.sqlite3")
    for run_id in (
        "20261004T000000-abcd1234",
        "20261005T000000-abcd9999",
        "20261006T000000-feed5678",
    ):
        repository.create(
            {"run_id": run_id, "fingerprint": run_id, "created_at": run_id}, []
        )

    assert repository.resolve_run_selector("20261004T000000") == "20261004T000000-abcd1234"
    assert repository.resolve_run_selector("feed5") == "20261006T000000-feed5678"
    assert repository.resolve_run_selector("20261004T000000-abcd1234") == (
        "20261004T000000-abcd1234"
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        repository.resolve_run_selector("abcd")
    with pytest.raises(ValueError, match="at least four"):
        repository.resolve_run_selector("fee")
    with pytest.raises(ValueError, match="non-path"):
        repository.resolve_run_selector("../feed")


def test_resume_force_and_artifacts(service, source):
    svc, fake = service
    before = source.read_bytes()
    run_id = svc.create_run(runner="test", chapters=[1])
    result = svc.execute(run_id)
    assert result["status"] == "completed"
    assert not result["active"]
    count = len(fake.calls)
    assert count > 1
    assert svc.create_run(runner="test", chapters=[1]) == run_id
    restarted = TranslationService(svc.config, lambda name: fake)
    restarted.resume(run_id)
    assert len(fake.calls) == count
    row = svc.repository.chunks(run_id)[0]
    metadata = json.loads(row["metadata"])
    assert metadata["usage"] == {"prompt_tokens": None, "completion_tokens": None}
    assert metadata["prompt_version"] == "translate-v1"
    assert metadata["duration_seconds"] >= 0
    assert (svc.run_dir(run_id) / metadata["artifact_dir"] / "request.json").is_file()
    assert source.read_bytes() == before
    forced = svc.create_run(runner="test", chapters=[1], force=True)
    assert forced != run_id
    assert svc.status(run_id)["status"] == "completed"


def test_direct_execution_profile_is_selected_and_frozen_without_secret_value(
    project, source, monkeypatch
):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["execution_profiles"] = {
        "direct-profile": {
            "version": 1,
            "kind": "direct",
            "model": "provider-model",
            "base_url": "https://provider.example/v1",
            "api_key_env": "NOVEL_PROFILE_KEY",
        }
    }
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    monkeypatch.setenv("NOVEL_PROFILE_KEY", "CANARY-KEY-VALUE")
    fake = FakeRunner()
    svc = TranslationService(load_config(project), lambda _name: fake)
    run_id = svc.create_run(execution_profile="direct-profile", chapters=[1])
    stage = svc.repository.run(run_id)["manifest"]["stages"][0]
    assert stage["runner"] == "direct-profile"
    assert stage["execution"]["kind"] == "direct"
    assert stage["execution"]["model_policy"] == "explicit"
    serialized = json.dumps(svc.repository.run(run_id)["manifest"])
    assert "CANARY-KEY-VALUE" not in serialized


def test_execution_profile_fails_closed_without_direct_secret(project, source, monkeypatch):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["execution_profiles"] = {
        "direct-profile": {
            "kind": "direct",
            "model": "provider-model",
            "base_url": "https://provider.example/v1",
            "api_key_env": "NOVEL_MISSING_PROFILE_KEY",
        }
    }
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    monkeypatch.delenv("NOVEL_MISSING_PROFILE_KEY", raising=False)
    svc = TranslationService(load_config(project), lambda _name: FakeRunner())
    with pytest.raises(ValueError, match="NOVEL_MISSING_PROFILE_KEY"):
        svc.create_run(execution_profile="direct-profile", chapters=[1])


def test_changed_opencode_binding_invalidates_approved_profile_sample(
    project, source, tmp_path, monkeypatch
):
    config_dir = tmp_path / "xdg" / "opencode"
    config_dir.mkdir(parents=True)
    opencode_config = config_dir / "opencode.json"
    opencode_config.write_text(
        '{"provider":{"private":{"models":{"model-a":{"variants":{"high":{}}},'
        '"model-b":{"variants":{"low":{}}}}}},'
        '"agent":{"translator":{"mode":"primary","model":"private/model-a#high"}}}',
        "utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("novel_harness.connections.shutil.which", lambda _name: "/bin/opencode")
    monkeypatch.setattr(
        "novel_harness.connections.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "1.18.31\n", ""),
    )
    data = yaml.safe_load(project.read_text("utf-8"))
    data["execution_profiles"] = {
        "opencode-profile": {
            "kind": "harness",
            "harness": "opencode",
            "agent_id": "translator",
            "model_policy": "inherit",
        }
    }
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    fake = FakeRunner()
    svc = TranslationService(load_config(project), lambda _name: fake)
    sample_id = svc.create_sample_run(execution_profile="opencode-profile", chapters=[1])
    sample_manifest = svc.repository.run(sample_id)["manifest"]
    frozen_execution = sample_manifest["stages"][0]["execution"]
    assert frozen_execution["agent_id"] == "translator"
    assert frozen_execution["requested_model"] == {"policy": "inherit"}
    assert frozen_execution["resolved_model"] == {
        "provider": "private",
        "model": "model-a",
        "variant": "high",
        "source": "agent",
    }
    assert frozen_execution["model_source"] == "inherited from OpenCode agent"
    assert frozen_execution["config_sha256"]
    assert svc.execute(sample_id)["status"] == "completed"
    svc.approve_sample(sample_id)

    opencode_config.write_text(
        '{"provider":{"private":{"models":{"model-a":{"variants":{"high":{}}},'
        '"model-b":{"variants":{"low":{}}}}}},'
        '"agent":{"translator":{"mode":"primary","model":"private/model-b#low"}}}',
        "utf-8",
    )
    with pytest.raises(ValueError, match="stages"):
        svc.create_whole_book_run(sample_id)
    current_execution = svc.config._profile_resolution["opencode-profile"]
    assert current_execution["config_sha256"] != frozen_execution["config_sha256"]
    assert current_execution["agent_id"] == frozen_execution["agent_id"]
    assert current_execution["requested_model"] == frozen_execution["requested_model"]
    assert current_execution["resolved_model"]["model"] == "model-b"
    assert current_execution["resolved_model"]["variant"] == "low"


def test_legacy_approved_sample_without_execution_metadata_still_passes(
    service, source
):
    svc, _ = service
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    svc.execute(sample_id)
    approval = svc.approve_sample(sample_id)
    manifest = svc.repository.run(sample_id)["manifest"]
    for stage in manifest["stages"]:
        stage.pop("execution", None)
    inputs = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_id", "fingerprint", "created_at"}
    }
    fingerprint = sha256(canonical(inputs))
    manifest["fingerprint"] = fingerprint
    with svc.repository.connection() as db:
        db.execute(
            "UPDATE runs SET manifest=?,fingerprint=? WHERE id=?",
            (canonical(manifest), fingerprint, sample_id),
        )
    write_json(svc.run_dir(sample_id) / "manifest.json", manifest)
    approval["sample_fingerprint"] = fingerprint
    write_json(svc.run_dir(sample_id) / "sample-approval.json", approval)

    whole_id = svc.create_whole_book_run(sample_id, runner="test")
    assert svc.repository.run(whole_id)["manifest"]["sample"]["sample_run_id"] == sample_id


def test_interruption_resume_keeps_success(project, source):
    fake = FakeRunner(interrupt_at=2)
    svc = TranslationService(load_config(project), lambda name: fake)
    run_id = svc.create_run(runner="test", chapters=[1])
    with pytest.raises(KeyboardInterrupt):
        svc.execute(run_id)
    assert svc.status(run_id)["chunks"]["completed"] == 1
    new_fake = FakeRunner()
    restarted = TranslationService(load_config(project), lambda name: new_fake)
    assert restarted.resume(run_id)["status"] == "completed"
    assert all(job.chunk_id != fake.calls[0].chunk_id for job in new_fake.calls)
    attempts = restarted.repository.attempts(run_id, 1, fake.calls[1].chunk_id, "translation")
    assert [a["status"] for a in attempts] == ["interrupted", "completed"]
    assert attempts[-1]["retry_reason"]


def test_missing_final_recovers_from_stage_without_call(service):
    svc, fake = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    (svc.run_dir(run_id) / row["output_path"]).unlink()
    assert svc.status(run_id)["status"] == "incomplete"
    count = len(fake.calls)
    assert svc.resume(run_id)["status"] == "completed"
    assert len(fake.calls) == count


def test_corrupt_stage_and_final_reruns(service):
    svc, fake = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    metadata = json.loads(row["metadata"])
    (svc.run_dir(run_id) / row["output_path"]).write_text("broken", "utf-8")
    (svc.run_dir(run_id) / metadata["artifact_dir"] / "translation.md").write_text(
        "broken", "utf-8"
    )
    count = len(fake.calls)
    svc.resume(run_id)
    assert len(fake.calls) == count + 1


def test_source_and_settings_drift(service, source):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    source.write_text(source.read_text("utf-8") + "追加。", "utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        svc.resume(run_id)
    assert svc.create_run(runner="test", chapters=[1]) != run_id
    assert "追加。" not in svc.inspect_chapter(run_id, 1)["source"]


def test_config_and_prompt_changes_new_identity(service):
    svc, _ = service
    first = svc.create_run(runner="test", chapters=[1])
    path = svc.config.translation.prompts_dir / "translate-v1.md"
    path.write_text(path.read_text("utf-8") + "\nMore instructions.", "utf-8")
    second = svc.create_run(runner="test", chapters=[1])
    assert first != second
    svc.config.runners["test"].model = "other-model"
    with pytest.raises(ValueError, match="configuration changed"):
        svc.resume(first)


def test_lock_prevents_duplicate_execution(service):
    svc, fake = service
    run_id = svc.create_run(runner="test", chapters=[1])
    with (
        FileLock(str(svc.run_dir(run_id) / ".run.lock")),
        pytest.raises(ValueError, match="already executing"),
    ):
        svc.execute(run_id)
    assert not fake.calls


def test_editor_failure_resumes_only_editor(project, source):
    translator = FakeRunner()
    editor = FakeRunner(fail_at=1)
    cfg = load_config(project)
    cfg.runners["editor"] = cfg.runners["test"].model_copy()
    svc = TranslationService(cfg, lambda name: translator if name == "test" else editor)
    run_id = svc.create_run(
        pipeline="translate-edit", translator="test", editor="editor", chapters=[1]
    )
    assert svc.execute(run_id)["status"] == "failed"
    first_chunk = translator.calls[0].chunk_id
    assert svc.resume(run_id)["status"] == "completed"
    assert sum(job.chunk_id == first_chunk for job in translator.calls) == 1
    assert all(job.draft for job in editor.calls)


def test_v13_explicit_null_temperature_changes_identity(service):
    svc, _ = service
    first = svc.create_run(runner="test")
    assert "temperature" not in svc.repository.run(first)["manifest"]["stages"][0]["runner_config"]
    svc.config.runners["test"].temperature = None
    second = svc.create_run(runner="test")
    assert first != second
    assert (
        svc.repository.run(second)["manifest"]["stages"][0]["runner_config"]["temperature"] is None
    )


def test_pipeline_prompt_precedence_and_global_fallback(service):
    svc, _ = service
    svc.config.translation.prompt_version = "review-v1"
    first = svc.create_run(runner="test")
    assert svc.repository.run(first)["manifest"]["stages"][0]["prompt_version"] == "translate-v1"
    second = svc.create_run(runner="test", prompt_version="edit-v1")
    assert svc.repository.run(second)["manifest"]["stages"][0]["prompt_version"] == "edit-v1"
    svc.config.pipelines["direct"].stages[0].prompt_version = None
    svc.config.translation.prompt_version = "translate-v1"
    third = svc.create_run(runner="test")
    assert svc.repository.run(third)["manifest"]["stages"][0]["prompt_version"] == "translate-v1"


def test_strict_continuation_revalidates_inherited_chunks(service):
    svc, _ = service
    parent = svc.create_run(runner="test", chapters=[1])
    svc.execute(parent)
    rows = svc.repository.chunks(parent)
    assert len(rows) >= 2
    good_row, bad_row = rows[0], rows[1]
    bad = "Он вошёл в данジョンa и остановился, чтобы внимательно осмотреться вокруг."
    bad_path = svc.run_dir(parent) / bad_row["output_path"]
    bad_path.write_text(bad, encoding="utf-8")
    svc.repository.set_chunk(
        parent,
        bad_row["chapter"],
        bad_row["chunk_id"],
        "completed",
        json.loads(bad_row["metadata"]),
        bad_row["output_path"],
        sha256(bad),
    )
    parent_before = svc.repository.run(parent)
    parent_rows_before = svc.repository.chunks(parent)
    bad_bytes = bad_path.read_bytes()

    svc.config.validation.max_foreign_script_chars = 0
    svc.config.validation.reject_malformed_dialogue = True
    child = svc.continue_run(parent, runner="test", chunk_chars=40)
    manifest = svc.repository.run(child)["manifest"]
    inherited = {c["chunk_id"]: c for c in manifest["inherited"]}
    assert good_row["chunk_id"] in inherited
    assert inherited[good_row["chunk_id"]]["output_sha256"] == good_row["output_sha256"]
    assert bad_row["chunk_id"] not in inherited
    assert manifest["rejected_inheritance"][0]["chunk_id"] == bad_row["chunk_id"]
    assert manifest["validation"]["max_foreign_script_chars"] == 0
    child_chunks = svc.repository.run(child)["manifest"]["chapters"][0]["chunks"]
    assert all(c["id"] != bad_row["chunk_id"] for c in child_chunks)
    assert svc.repository.run(parent) == parent_before
    assert svc.repository.chunks(parent) == parent_rows_before
    assert bad_path.read_bytes() == bad_bytes


def test_v25_same_runner_review_allowed_and_cross_model_frozen(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    svc.config.runners["stronger"] = RunnerConfig(model="stronger-model")
    same = svc.create_run(
        translator="test", editor="test", pipeline="translate-review", force=True
    )
    same_stages = svc.repository.run(same)["manifest"]["stages"]
    assert [s["runner"] for s in same_stages] == ["test", "test"]
    cross = svc.create_run(
        translator="test", editor="stronger", pipeline="translate-review", force=True
    )
    cross_stages = svc.repository.run(cross)["manifest"]["stages"]
    assert [s["runner"] for s in cross_stages] == ["test", "stronger"]
    assert cross_stages[1]["runner_config"]["model"] == "stronger-model"
    assert cross_stages[0]["runner_config"]["model"] == "fake-model"


def test_v25_fresh_stage_runner_fallback(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    svc.config.runners["reviewer"] = RunnerConfig(model="reviewer-model")
    svc.config.pipelines["translate-review"].stages[1].runner = "reviewer"
    rid = svc.create_run(translator="test", pipeline="translate-review", force=True)
    stages = svc.repository.run(rid)["manifest"]["stages"]
    assert [s["runner"] for s in stages] == ["test", "reviewer"]


def test_draft_light_pipeline_prepares_and_freezes_prompt(service):
    from pathlib import Path

    import yaml

    from novel_harness.config import PipelineConfig, StageConfig

    example = Path(__file__).resolve().parents[1] / "config.quality.example.yaml"
    tracked = yaml.safe_load(example.read_text("utf-8"))
    assert (
        tracked["pipelines"]["draft-light"]["stages"][0]["prompt_version"]
        == "translate-draft-v1"
    )

    svc, _ = service
    svc.config.pipelines["draft-light"] = PipelineConfig(
        stages=[
            StageConfig(
                name="translation", role="translator", prompt_version="translate-draft-v1"
            )
        ]
    )
    svc.config.translation.output_format = "segments-v1"
    run_id = svc.create_run(runner="test", pipeline="draft-light", chapters=[1])
    manifest = svc.repository.run(run_id)["manifest"]
    assert manifest["workflow"] == "draft-light"
    assert manifest["translation"]["output_format"] == "segments-v1"
    stage = manifest["stages"][0]
    assert stage["prompt_version"] == "translate-draft-v1"
    assert "as a rough draft" in stage["prompt"]
    assert "do not spend effort on literary polish" in stage["prompt"]
    direct_prompt = (
        Path(svc.config.translation.prompts_dir) / "translate-v1.md"
    ).read_text("utf-8")
    assert stage["prompt"] != direct_prompt
    # The default direct pipeline and prompt are unchanged.
    direct = svc.create_run(runner="test", chapters=[1])
    direct_stage = svc.repository.run(direct)["manifest"]["stages"][0]
    assert direct_stage["prompt_version"] == "translate-v1"
    assert "as a rough draft" not in direct_stage["prompt"]


class VariantRunner:
    """Deterministic fake whose output changes when the test flips ``variant``."""

    def __init__(self):
        self.variant = "A"
        self.calls = 0

    def run(self, job):
        self.calls += 1
        text = f"Глава {job.chapter}\n\n" + f"{self.variant} вариант. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


def test_requeue_preserves_final_and_marks_stale_after_regeneration(project, source):
    runner = VariantRunner()
    svc = TranslationService(load_config(project), lambda name: runner)
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    machine_a = row["output_sha256"]
    assert "A вариант" in (svc.run_dir(run_id) / row["output_path"]).read_text("utf-8")

    svc.set_final(
        run_id, row["chapter"], row["chunk_id"],
        action="edit", actor="tester", text="Правка человека",
        expected_machine_sha256=machine_a,
    )

    result = svc.requeue_chunk(
        run_id, row["chapter"], row["chunk_id"], expected_machine_sha256=machine_a
    )
    assert result["status"] == "pending"
    entry = svc.repository.final_entry(run_id, row["chapter"], row["chunk_id"])
    assert entry is not None and entry["final_text"] == "Правка человека"

    # Full regeneration to machine B through the fake runner.
    runner.variant = "B"
    svc.execute(run_id)

    entry = svc.repository.final_entry(run_id, row["chapter"], row["chunk_id"])
    assert entry is not None and entry["final_text"] == "Правка человека"
    updated = next(r for r in svc.repository.chunks(run_id) if r["chunk_id"] == row["chunk_id"])
    machine_b = updated["output_sha256"]
    assert machine_b != machine_a
    assert "B вариант" in (svc.run_dir(run_id) / updated["output_path"]).read_text("utf-8")
    assert entry["machine_sha256"] == machine_a
    listed = svc.final_layer(run_id)
    assert listed[0]["stale"] is True

    # The requeue marker is cleared after the new machine output commits, so a
    # later resume is idempotent and does not call the runner again.
    calls = runner.calls
    svc.execute(run_id)
    assert runner.calls == calls
    row = next(r for r in svc.repository.chunks(run_id) if r["chunk_id"] == updated["chunk_id"])
    assert row["output_sha256"] == machine_b


def test_final_cas_is_mandatory_for_existing_final(service):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    machine = row["output_sha256"]
    first = svc.set_final(
        run_id, row["chapter"], row["chunk_id"],
        action="edit", actor="t", text="Правка", expected_machine_sha256=machine,
    )

    with pytest.raises(ValueError, match="expected_final_sha256 is required"):
        svc.set_final(
            run_id, row["chapter"], row["chunk_id"],
            action="edit", actor="t", text="Правка-2", expected_machine_sha256=machine,
        )
    with pytest.raises(ValueError, match="Stale final"):
        svc.set_final(
            run_id, row["chapter"], row["chunk_id"],
            action="edit", actor="t", text="Правка-3", expected_machine_sha256=machine,
            expected_final_sha256="nope",
        )
    ok = svc.set_final(
        run_id, row["chapter"], row["chunk_id"],
        action="accept", actor="t", expected_machine_sha256=machine,
        expected_final_sha256=first["final_sha256"],
    )
    assert ok["status"] == "accepted"


def test_same_text_review_decision_rotates_final_cas_token(service):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    args = (run_id, row["chapter"], row["chunk_id"])
    machine_sha = row["output_sha256"]
    first = svc.set_final(
        *args, action="edit", actor="reviewer-a", text="Та же правка",
        expected_machine_sha256=machine_sha,
    )
    changed = svc.set_final(
        *args, action="needs_review", actor="reviewer-b", text="Та же правка",
        reason="needs another look", expected_machine_sha256=machine_sha,
        expected_final_sha256=first["final_sha256"],
    )
    assert changed["final_sha256"] != first["final_sha256"]
    with pytest.raises(ConflictError, match="Stale final edit"):
        svc.set_final(
            *args, action="accept", actor="reviewer-c", text="Та же правка",
            expected_machine_sha256=machine_sha,
            expected_final_sha256=first["final_sha256"],
        )
    saved = svc.repository.final_entry(*args)
    assert saved["status"] == "needs_review"
    assert saved["reason"] == "needs another look"


def test_concurrent_first_final_writes_have_one_cas_winner(service, monkeypatch):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    original = svc.repository.final_entry
    barrier = Barrier(2)

    def synchronized_read(*args):
        value = original(*args)
        barrier.wait(timeout=5)
        return value

    monkeypatch.setattr(svc.repository, "final_entry", synchronized_read)

    def edit(text, expected_final=None):
        try:
            return svc.set_final(
                run_id, row["chapter"], row["chunk_id"], action="edit",
                actor="tester", text=text,
                expected_machine_sha256=row["output_sha256"],
                expected_final_sha256=expected_final,
            )
        except ConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, ("Первый вариант", "Второй вариант")))
    winners = [result for result in results if isinstance(result, dict)]
    conflicts = [result for result in results if isinstance(result, ConflictError)]
    assert len(winners) == len(conflicts) == 1
    assert original(run_id, row["chapter"], row["chunk_id"])[
        "final_text"
    ] == winners[0]["final_text"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(edit, text, winners[0]["final_sha256"])
                   for text in ("Третья правка", "Четвёртая правка")]
        updated = [future.result() for future in futures]
    winners = [result for result in updated if isinstance(result, dict)]
    conflicts = [result for result in updated if isinstance(result, ConflictError)]
    assert len(winners) == len(conflicts) == 1
    assert original(run_id, row["chapter"], row["chunk_id"])[
        "final_text"
    ] == winners[0]["final_text"]

    def revert():
        try:
            return svc.set_final(
                run_id, row["chapter"], row["chunk_id"], action="revert",
                actor="tester", expected_machine_sha256=row["output_sha256"],
                expected_final_sha256=winners[0]["final_sha256"],
            )
        except ConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        reverted = list(pool.map(lambda _index: revert(), range(2)))
    assert sum(isinstance(result, dict) for result in reverted) == 1
    assert sum(isinstance(result, ConflictError) for result in reverted) == 1
    assert original(run_id, row["chapter"], row["chunk_id"]) is None


def test_concurrent_requeue_has_one_machine_cas_winner(service, monkeypatch):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    original = svc._chunk_row
    barrier = Barrier(2)

    def synchronized_read(*args):
        value = original(*args)
        barrier.wait(timeout=5)
        return value

    monkeypatch.setattr(svc, "_chunk_row", synchronized_read)

    def requeue(_index):
        try:
            return svc.requeue_chunk(
                run_id, row["chapter"], row["chunk_id"],
                expected_machine_sha256=row["output_sha256"],
            )
        except ConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(requeue, range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, ConflictError) for result in results) == 1


def test_requeue_reopens_completed_run_for_persisted_worker(project, source, monkeypatch):
    svc = TranslationService(load_config(project), lambda _name: FakeRunner())
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    assert svc.status(run_id)["status"] == "completed"
    row = svc.repository.chunks(run_id)[0]
    svc.requeue_chunk(
        run_id, row["chapter"], row["chunk_id"],
        expected_machine_sha256=row["output_sha256"],
    )
    assert svc.repository.run(run_id)["status"] == "pending"
    assert svc.status(run_id)["status"] == "pending"

    def reached_spawn(_config, _run_id, _directory):
        raise RuntimeError("worker spawn reached")

    monkeypatch.setattr("novel_harness.worker._spawn_worker", reached_spawn)
    with pytest.raises(RuntimeError, match="worker spawn reached"):
        start_worker(project, run_id, startup_timeout=0.1)


def test_stale_final_blocks_epub_until_resolved(service):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    machine_a = row["output_sha256"]
    final = svc.set_final(
        run_id, row["chapter"], row["chunk_id"],
        action="edit", actor="t", text="Правка", expected_machine_sha256=machine_a,
    )
    svc.build_epub(run_id)  # fresh final builds

    # Simulate a later machine regeneration (machine B) on the same chunk.
    path = svc.run_dir(run_id) / row["output_path"]
    revised = path.read_text("utf-8") + " Обновлённый машинный текст."
    write_text(path, revised)
    machine_b = sha256(revised)
    svc.repository.set_chunk(
        run_id, row["chapter"], row["chunk_id"], "completed",
        json.loads(row["metadata"]), row["output_path"], machine_b,
    )

    with pytest.raises(ValueError, match="stale human-final"):
        svc.build_epub(run_id)

    svc.set_final(
        run_id, row["chapter"], row["chunk_id"],
        action="accept", actor="t", expected_machine_sha256=machine_b,
        expected_final_sha256=final["final_sha256"],
    )
    svc.build_epub(run_id)  # resolved: builds again


def test_run_freezes_only_authoritative_book_bible(service):
    svc, _ = service
    created = svc.propose_bible_entries(
        [{"kind": "term", "source_form": "魔術", "target_form": "магия"}],
        proposer="model",
    )
    assert created["created"][0]["trust"] == "proposed"
    entry_id = created["created"][0]["entry_id"]

    proposed_run = svc.create_run(runner="test", chapters=[1])
    proposed_block = svc.repository.run(proposed_run)["manifest"]["book_bible"]
    assert proposed_block["entries"] == []  # proposed is not authoritative

    svc.set_bible_trust(entry_id, action="confirm", actor="human")
    confirmed_run = svc.create_run(runner="test", chapters=[1], force=True)
    confirmed_block = svc.repository.run(confirmed_run)["manifest"]["book_bible"]
    assert [entry["target_form"] for entry in confirmed_block["entries"]] == ["магия"]
    assert confirmed_block["projection_sha256"] != proposed_block["projection_sha256"]
    # The frozen snapshot is written next to the run manifest.
    assert (svc.run_dir(confirmed_run) / "book-bible.json").is_file()


def test_run_rejects_tampered_book_bible_snapshot(service):
    svc, _ = service
    entry_id = svc.propose_bible_entries(
        [{"kind": "character", "source_form": "ルゥ", "target_form": "Ру"}],
        proposer="model",
    )["created"][0]["entry_id"]
    svc.set_bible_trust(entry_id, action="lock", actor="human")
    run_id = svc.create_run(runner="test", chapters=[1])

    write_json(
        svc.run_dir(run_id) / "book-bible.json",
        {
            "version": 1,
            "policy": "tampered",
            "authoritative_trust": [],
            "entries": [],
            "projection_sha256": "0",
        },
    )
    with pytest.raises(ValueError, match="book bible snapshot changed"):
        svc.execute(run_id)


def test_sample_first_gate_and_provenance(service, source):
    svc, _ = service
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    sample_manifest = svc.repository.run(sample_id)["manifest"]
    assert sample_manifest["sample"] == {"role": "sample", "chapters": [1]}
    assert (svc.run_dir(sample_id) / "sample.json").is_file()

    with pytest.raises(ValueError, match="approved sample"):
        svc.create_whole_book_run(sample_id, runner="test")

    # Approval fails closed until the sample translation has completed.
    with pytest.raises(ValueError, match="must complete successfully before approval"):
        svc.approve_sample(sample_id, actor="tester")
    svc.execute(sample_id)

    approval = svc.approve_sample(sample_id, actor="tester")
    assert approval["sample_fingerprint"] == svc.repository.run(sample_id)["fingerprint"]

    whole_id = svc.create_whole_book_run(sample_id, runner="test")
    whole = svc.repository.run(whole_id)["manifest"]
    assert whole["sample"]["role"] == "whole-book"
    assert whole["sample"]["sample_run_id"] == sample_id
    assert whole["sample"]["sample_fingerprint"] == approval["sample_fingerprint"]


@pytest.mark.parametrize("invalidate", ["requeue", "corrupt"])
def test_approved_sample_must_still_be_complete_when_creating_whole_book(
    service, source, invalidate
):
    svc, _ = service
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    svc.execute(sample_id)
    svc.approve_sample(sample_id)
    row = svc.repository.chunks(sample_id)[0]
    if invalidate == "requeue":
        svc.requeue_chunk(
            sample_id, row["chapter"], row["chunk_id"],
            expected_machine_sha256=row["output_sha256"],
        )
    else:
        (svc.run_dir(sample_id) / row["output_path"]).write_text("corrupt", "utf-8")
    assert svc.sample_approval(sample_id) is None
    expected = "approved sample" if invalidate == "requeue" else "Sample translation must complete successfully"
    with pytest.raises(ValueError, match=expected):
        svc.create_whole_book_run(sample_id, runner="test")
    if invalidate == "requeue":
        svc.execute(sample_id)
        with pytest.raises(ValueError, match="approved sample"):
            svc.create_whole_book_run(sample_id, runner="test")


@pytest.mark.parametrize("operation", ["approve", "whole_book"])
def test_sample_requeue_cannot_pass_approval_or_whole_book_gate(
    service, source, monkeypatch, operation
):
    svc, _ = service
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    svc.execute(sample_id)
    if operation == "whole_book":
        svc.approve_sample(sample_id)
    row = svc.repository.chunks(sample_id)[0]
    entered = Event()
    release = Event()
    original = run_operations._sample_complete

    def paused_completion(*args):
        result = original(*args)
        entered.set()
        assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(run_operations, "_sample_complete", paused_completion)
    with ThreadPoolExecutor(max_workers=2) as pool:
        if operation == "approve":
            action = pool.submit(svc.approve_sample, sample_id)
        else:
            action = pool.submit(svc.create_whole_book_run, sample_id, runner="test")
        assert entered.wait(timeout=5)
        requeue = pool.submit(
            svc.requeue_chunk, sample_id, row["chapter"], row["chunk_id"],
            expected_machine_sha256=row["output_sha256"],
        )
        with pytest.raises(FutureTimeout):
            requeue.result(timeout=0.1)
        release.set()
        assert action.result(timeout=5)
        assert requeue.result(timeout=5)["status"] == "pending"
    assert svc.sample_approval(sample_id) is None
    svc.execute(sample_id)
    with pytest.raises(ValueError, match="approved sample"):
        svc.create_whole_book_run(sample_id, runner="test")


def test_approval_is_bound_to_machine_output_after_corruption_recovery(project, source):
    runner = VariantRunner()
    svc = TranslationService(load_config(project), lambda _name: runner)
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    svc.execute(sample_id)
    approval = svc.approve_sample(sample_id)
    row = svc.repository.chunks(sample_id)[0]
    original_sha = row["output_sha256"]
    stage = svc.repository.run(sample_id)["manifest"]["stages"][0]["name"]
    attempt = svc.repository.attempts(sample_id, row["chapter"], row["chunk_id"], stage)[-1]
    (svc.run_dir(sample_id) / row["output_path"]).write_text("corrupt", "utf-8")
    (svc.run_dir(sample_id) / attempt["artifact_dir"] / "translation.md").write_text(
        "corrupt", "utf-8"
    )
    runner.variant = "B"
    svc.execute(sample_id)
    assert svc.repository.chunks(sample_id)[0]["output_sha256"] != original_sha
    assert svc.sample_approval(sample_id) is None
    assert (svc.run_dir(sample_id) / "sample-approval.json").is_file()
    with pytest.raises(ValueError, match="approved sample"):
        svc.create_whole_book_run(sample_id, runner="test")
    renewed = svc.approve_sample(sample_id)
    assert renewed["sample_output_sha256"] != approval["sample_output_sha256"]
    assert svc.create_whole_book_run(sample_id, runner="test")


def test_legacy_sample_approval_without_output_digest_requires_renewal(service, source):
    svc, _ = service
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    svc.execute(sample_id)
    approval = svc.approve_sample(sample_id)
    approval.pop("sample_output_sha256")
    write_json(svc.run_dir(sample_id) / "sample-approval.json", approval)
    assert svc.sample_approval(sample_id) is None
    with pytest.raises(ValueError, match="approved sample"):
        svc.create_whole_book_run(sample_id, runner="test")
    svc.approve_sample(sample_id)
    assert svc.create_whole_book_run(sample_id, runner="test")


def test_sample_is_at_most_two_chapters(service, source):
    svc, _ = service
    with pytest.raises(ValueError, match="at most 2 chapters"):
        svc.create_sample_run(runner="test", chapters=[1, 2, 3])


def test_cli_batch_path_needs_no_sample_approval(service, source):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    assert svc.repository.run(run_id)["manifest"].get("sample") is None
    assert svc.execute(run_id)["status"] == "completed"


def test_whole_book_launch_does_not_mutate_the_sample(service, source):
    svc, _ = service
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    before = svc.repository.run(sample_id)["manifest"]
    svc.execute(sample_id)
    svc.approve_sample(sample_id)
    whole_id = svc.create_whole_book_run(sample_id, runner="test")
    assert whole_id != sample_id
    assert svc.repository.run(sample_id)["manifest"] == before


def test_settings_change_after_sample_refuses_whole_book_launch(service, source):
    svc, _ = service
    sample_id = svc.create_sample_run(runner="test", chapters=[1])
    svc.execute(sample_id)
    svc.approve_sample(sample_id)
    with pytest.raises(ValueError, match="new sample"):
        svc.create_whole_book_run(sample_id, runner="test", chunk_chars=120)
    # The matching request still launches and keeps the sample provenance.
    whole_id = svc.create_whole_book_run(sample_id, runner="test")
    assert svc.repository.run(whole_id)["manifest"]["sample"]["sample_run_id"] == sample_id


def _enable_cache(svc):
    from novel_harness.cache import TranslationCache

    svc.stages.cache = TranslationCache(svc.root / "cache")


def test_machine_cache_hit_then_book_bible_invalidation(service, source):
    svc, fake = service
    _enable_cache(svc)

    run_a = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_a)
    calls = len(fake.calls)

    run_b = svc.create_run(runner="test", chapters=[1], force=True)
    svc.execute(run_b)
    assert len(fake.calls) == calls  # exact fingerprint reuse
    assert any(
        json.loads(row["metadata"]).get("cache") == "hit"
        for row in svc.repository.chunks(run_b)
    )

    entry_id = svc.propose_bible_entries(
        [{"kind": "term", "source_form": "魔術", "target_form": "магия"}], proposer="model"
    )["created"][0]["entry_id"]
    svc.set_bible_trust(entry_id, action="confirm", actor="human")

    run_c = svc.create_run(runner="test", chapters=[1], force=True)
    svc.execute(run_c)
    assert len(fake.calls) > calls  # no false reuse after book-bible change
    assert all(
        json.loads(row["metadata"]).get("cache") == "regenerated"
        for row in svc.repository.chunks(run_c)
    )


def test_translation_memory_promotion_and_reuse(service, source):
    svc, fake = service
    _enable_cache(svc)

    run_a = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_a)
    row = svc.repository.chunks(run_a)[0]
    # A promoted final must still pass the deterministic validator on reuse.
    machine_text = (svc.run_dir(run_a) / row["output_path"]).read_text("utf-8")
    svc.set_final(
        run_a, row["chapter"], row["chunk_id"],
        action="edit", actor="human", text=machine_text + "\n\nПравка человека.",
        expected_machine_sha256=row["output_sha256"],
    )

    promoted = svc.promote_final_to_tm(run_a, row["chapter"], row["chunk_id"], actor="human")
    assert promoted["inserted"] is True
    again = svc.promote_final_to_tm(run_a, row["chapter"], row["chunk_id"], actor="human")
    assert again["inserted"] is False  # valid entry never overwritten

    calls = len(fake.calls)
    run_b = svc.create_run(runner="test", chapters=[1], force=True)
    svc.execute(run_b)

    rows_b = {r["chunk_id"]: r for r in svc.repository.chunks(run_b)}
    promoted_row = rows_b[row["chunk_id"]]
    assert json.loads(promoted_row["metadata"]).get("cache") == "tm_hit"
    assert "Правка человека." in (
        svc.run_dir(run_b) / promoted_row["output_path"]
    ).read_text("utf-8")
    # The promoted edit changes the next chunk's context, so downstream reuse is
    # correctly invalidated (no false reuse).
    assert len(fake.calls) > calls
    assert svc.translation_memory()["count"] == 1


def test_machine_cache_invalidates_on_decoding_change(service, source):
    svc, fake = service
    _enable_cache(svc)

    run_a = svc.create_run(runner="test", chapters=[1])
    svc.execute(run_a)
    calls = len(fake.calls)

    svc.config.translation.temperature = 0.9
    run_b = svc.create_run(runner="test", chapters=[1], force=True)
    svc.execute(run_b)
    assert len(fake.calls) > calls  # no false reuse after a decoding change
    assert all(
        json.loads(row["metadata"]).get("cache") == "regenerated"
        for row in svc.repository.chunks(run_b)
    )
