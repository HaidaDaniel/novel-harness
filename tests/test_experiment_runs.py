import json
from copy import deepcopy

import pytest
import yaml

from novel_harness.config import RunnerConfig
from novel_harness.experiment_runs import (
    FAILURE_POLICIES,
    _arm_signature,
    export_matrix,
    plan_matrix,
    preflight_matrix,
    resume_matrix,
    run_matrix,
    status_matrix,
)
from novel_harness.runners.opencode import OpenCodeRunner


def _matrix(path):
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "offline-matrix",
                "chapters": [1],
                "vary": ["decoding"],
                "arms": [
                    {"id": "cold", "runner": "test", "temperature": 0.1},
                    {"id": "warm", "runner": "test", "temperature": 0.8},
                ],
            }
        ),
        "utf-8",
    )


def test_plan_preflight_status_and_export_are_offline(service, source, tmp_path):
    svc, fake = service
    matrix = tmp_path / "matrix.yaml"
    _matrix(matrix)

    record = plan_matrix(svc, matrix)
    plan = svc.root / "experiment-plans" / record["plan_id"] / "plan.json"
    assert plan.is_file()
    assert [arm["id"] for arm in record["arms"]] == ["cold", "warm"]
    assert fake.calls == []

    ready = preflight_matrix(svc, plan)
    assert ready["status"] == "ready"
    assert all(check["resolved_model"] == "fake-model" for check in ready["checks"])
    expected_cap = svc.repository.run(record["arms"][0]["run_id"])["manifest"]["translation"]["max_tokens"]
    assert all(check["effective_settings"]["output_cap"] == expected_cap for check in ready["checks"])

    status = status_matrix(svc, plan)
    assert [arm["status"] for arm in status["arms"]] == ["pending", "pending"]
    exported = export_matrix(svc, plan)
    assert json.loads((plan.parent / "export.json").read_text("utf-8")) == {
        "schema_version": 1,
        "plan_id": record["plan_id"],
        "arms": exported["arms"],
    }
    assert all(arm["attempts"] == 0 for arm in exported["arms"])
    assert fake.calls == []


def test_plan_rejects_undeclared_fixed_axis(service, source, tmp_path):
    svc, _ = service
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "name": "invalid-matrix",
                "chapters": [1],
                "vary": ["model"],
                "arms": [
                    {"id": "a", "runner": "test", "chunk_chars": 80},
                    {"id": "b", "runner": "test", "chunk_chars": 120},
                ],
            }
        ),
        "utf-8",
    )

    with pytest.raises(ValueError, match="undeclared fixed axis"):
        plan_matrix(svc, matrix)


def test_plan_accepts_different_resolved_models_when_model_varies(service, source, tmp_path):
    svc, _ = service
    svc.config.runners["model-a"] = RunnerConfig(
        type="openai", model="model-a", api_key="EMPTY", base_url="http://127.0.0.1:9999/v1"
    )
    svc.config.runners["model-b"] = RunnerConfig(
        type="openai", model="model-b", api_key="EMPTY", base_url="http://127.0.0.1:9999/v1"
    )
    matrix = tmp_path / "models.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "name": "different-models",
                "chapters": [1],
                "vary": ["model"],
                "arms": [
                    {"id": "a", "runner": "model-a", "model": "model-a"},
                    {"id": "b", "runner": "model-b", "model": "model-b"},
                ],
            }
        ),
        "utf-8",
    )

    record = plan_matrix(svc, matrix)
    models = [arm["execution_provenance"]["resolved"]["id"] for arm in record["arms"]]
    assert models == ["model-a", "model-b"]
    plan = svc.root / "experiment-plans" / record["plan_id"] / "plan.json"
    assert preflight_matrix(svc, plan)["status"] == "ready"


@pytest.mark.parametrize("axis", sorted({"model", "harness", "pipeline", "prompt", "chunking", "decoding"}))
def test_matrix_axis_mapping_ignores_only_declared_internal_field(service, source, axis):
    svc, _ = service
    manifest = deepcopy(svc.repository.run(svc.create_run(runner="test", chapters=[1]))["manifest"])
    changed = deepcopy(manifest)
    stage = changed["stages"][0]
    provenance = stage["execution_provenance"]
    if axis == "model":
        provenance["resolved"] = {"id": "other-model"}
    elif axis == "harness":
        provenance["harness"]["name"] = "other-harness"
    elif axis == "pipeline":
        changed["workflow"] = "other-pipeline"
    elif axis == "prompt":
        stage["prompt_sha256"] = "other-prompt"
    elif axis == "chunking":
        changed["translation"]["chunk_chars"] += 1
    elif axis == "decoding":
        provenance["material"]["temperature"] = 1.25

    assert _arm_signature(manifest, vary={axis}) == _arm_signature(changed, vary={axis})
    assert _arm_signature(manifest, vary=set()) != _arm_signature(changed, vary=set())


def test_run_matrix_isolates_one_arm_start_failure(service, source, tmp_path, monkeypatch):
    svc, _ = service
    matrix = tmp_path / "matrix.yaml"
    _matrix(matrix)
    record = plan_matrix(svc, matrix)
    plan = svc.root / "experiment-plans" / record["plan_id"] / "plan.json"
    failed_id = record["arms"][0]["run_id"]
    started = []

    def fake_start(config, run_id):
        started.append(run_id)
        if run_id == failed_id:
            raise RuntimeError("synthetic arm start failure")
        return {"run_id": run_id, "started": True}

    monkeypatch.setattr("novel_harness.experiment_runs.start_worker", fake_start)
    result = run_matrix(svc, tmp_path / "config.yaml", plan)

    assert started == [arm["run_id"] for arm in record["arms"]]
    assert result["launches"][0]["started"] is False
    assert result["launches"][1] == {"run_id": record["arms"][1]["run_id"], "started": True}


def test_resume_matrix_launches_only_noncompleted_arms(service, source, tmp_path, monkeypatch):
    svc, _ = service
    matrix = tmp_path / "matrix.yaml"
    _matrix(matrix)
    record = plan_matrix(svc, matrix)
    plan = svc.root / "experiment-plans" / record["plan_id"] / "plan.json"
    states = {record["arms"][0]["run_id"]: "completed", record["arms"][1]["run_id"]: "pending"}
    monkeypatch.setattr(svc, "status", lambda run_id: {"status": states[run_id]})
    started = []
    monkeypatch.setattr(
        "novel_harness.experiment_runs.start_worker",
        lambda config, run_id: started.append(run_id) or {"run_id": run_id, "started": True},
    )

    result = resume_matrix(svc, tmp_path / "config.yaml", plan)

    assert started == [record["arms"][1]["run_id"]]
    assert [item["run_id"] for item in result["launches"]] == started


def test_existing_completed_chunks_are_not_regenerated_after_resume(service, source):
    svc, fake = service
    run_id = svc.create_run(runner="test", chapters=[1])
    first = svc.execute(run_id)
    calls = len(fake.calls)
    second = svc.execute(run_id)

    assert first["status"] == second["status"] == "completed"
    assert len(fake.calls) == calls


def test_preflight_rejects_manifest_drift(service, source, tmp_path):
    svc, _ = service
    matrix = tmp_path / "matrix.yaml"
    _matrix(matrix)
    record = plan_matrix(svc, matrix)
    plan = svc.root / "experiment-plans" / record["plan_id"] / "plan.json"
    run_id = record["arms"][0]["run_id"]
    manifest_path = svc.run_dir(run_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["translation"]["temperature"] = 1.1
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match="artifact drifted after freeze"):
        preflight_matrix(svc, plan)


def test_opencode_preflight_materializes_both_caps(tmp_path):
    config = RunnerConfig(
        type="opencode",
        model="local/test-model",
        base_url="http://127.0.0.1:9000/v1",
        max_tokens=4096,
    )
    proof = OpenCodeRunner(config, tmp_path).preflight_effective_settings(4096)
    assert proof["status"] == "proven"
    assert proof["child_cap"] == proof["wire_cap"] == 4096
    assert proof["runtime_config_sha256"]
    with pytest.raises(ValueError, match="unprovable"):
        OpenCodeRunner(
            config.model_copy(update={"harness_managed": True, "opencode_agent_id": "agent"}),
            tmp_path,
        ).preflight_effective_settings(4096)


def test_export_failure_policies_keep_recovery_classes_distinct():
    assert FAILURE_POLICIES["endpoint_busy"][0] == "transient"
    assert FAILURE_POLICIES["length"][0] == "partial"
    assert FAILURE_POLICIES["malformed_output"][0] == "output_validation"
    assert FAILURE_POLICIES["source_drift"][0] == "invariant_drift"
