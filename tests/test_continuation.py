import json
from pathlib import Path

import pytest
from filelock import Timeout

from novel_harness.artifacts import sha256
from novel_harness.config import RunnerConfig


def test_v18_continuation_preserves_prefix_and_rechunks_only_remaining(service):
    svc, fake = service
    original = svc.create_run(runner="test")
    svc.execute(original, max_chunks=1)
    before = svc.repository.run(original)
    rows_before = svc.repository.chunks(original)
    files_before = {
        str(p): sha256(p.read_bytes()) for p in svc.run_dir(original).rglob("*") if p.is_file()
    }
    svc.config.runners["new"] = RunnerConfig(model="different-model")
    continued = svc.continue_run(original, runner="new", chunk_chars=32)
    assert svc.continue_run(original, runner="new", chunk_chars=32) == continued
    manifest = svc.repository.run(continued)["manifest"]
    assert manifest["continued_from"] == original
    old_chunks = before["manifest"]["chapters"][0]["chunks"]
    new_chunks = manifest["chapters"][0]["chunks"]
    assert new_chunks[0] == old_chunks[0]
    assert "".join(c["text"] for c in new_chunks) == "".join(c["text"] for c in old_chunks)
    assert all(len(c["text"]) <= 32 for c in new_chunks[1:])
    assert [c["index"] for c in new_chunks] == list(range(1, len(new_chunks) + 1))
    imported = svc.repository.chunks(continued)[0]
    assert imported["output_sha256"] == rows_before[0]["output_sha256"]
    assert json.loads(imported["metadata"])["origin"]["run_id"] == original
    assert svc.status(continued)["chunks"]["completed"] == 1
    fake.calls.clear()
    assert svc.execute(continued, max_chunks=1)["status"] == "paused"
    assert svc.resume(continued)["status"] == "completed"
    assert len(fake.calls) == len(new_chunks) - 1
    assert all(j.chunk_id != old_chunks[0]["id"] for j in fake.calls)
    assert svc.repository.run(original) == before
    assert svc.repository.chunks(original) == rows_before
    assert all(sha256(Path(p).read_bytes()) == h for p, h in files_before.items())
    assert svc.build_epub(continued).is_file()


def test_v18_corrupt_output_not_inherited(service):
    svc, _ = service
    original = svc.create_run(runner="test")
    svc.execute(original, max_chunks=1)
    row = svc.repository.chunks(original)[0]
    (svc.run_dir(original) / row["output_path"]).write_text("bad", "utf-8")
    continued = svc.continue_run(original, runner="test", chunk_chars=32)
    assert svc.status(continued)["chunks"]["completed"] == 0


def test_v18_changed_source_and_active_run_rejected(service, source):
    svc, _ = service
    original = svc.create_run(runner="test")
    with svc._lock(original), pytest.raises((ValueError, Timeout)):
        svc.continue_run(original, runner="test")
    source.write_text("changed", "utf-8")
    with pytest.raises(ValueError, match="Source"):
        svc.continue_run(original, runner="test")


def test_v22_continuation_rejects_hashed_but_damaged_prefix(service):
    svc, _ = service
    parent = svc.create_run(runner="test")
    svc.execute(parent)
    rows = svc.repository.chunks(parent)
    row = rows[1]
    bad = "Марион " + "一定能" * 78
    path = svc.run_dir(parent) / row["output_path"]
    path.write_text(bad, encoding="utf-8")
    svc.repository.set_chunk(
        parent,
        row["chapter"],
        row["chunk_id"],
        "completed",
        json.loads(row["metadata"]),
        row["output_path"],
        sha256(bad),
    )
    before = svc.repository.run(parent)
    child = svc.continue_run(parent, runner="test", chunk_chars=32)
    assert svc.status(child)["inherited_chunks"] == 1
    manifest = svc.repository.run(child)["manifest"]
    assert manifest["rejected_inheritance"][0]["chunk_id"] == row["chunk_id"]
    assert svc.repository.run(parent) == before and path.read_text("utf-8") == bad


def test_v25_continuation_selects_separate_reviewer_profile(service):
    svc, _ = service
    parent = svc.create_run(runner="test")
    svc.config.translation.output_format = "segments-v1"
    svc.config.runners["reviewer"] = RunnerConfig(model="stronger-model")
    child = svc.continue_run(parent, runner="test", editor="reviewer", pipeline="translate-review")
    stages = svc.repository.run(child)["manifest"]["stages"]
    assert [s["runner"] for s in stages] == ["test", "reviewer"]
    assert stages[1]["role"] == "reviewer" and stages[1]["prompt_version"] == "review-v1"


def test_v29_active_pointers_follow_frozen_harness_without_running(service, project):
    import yaml

    svc, fake = service
    svc.config.runners["codex"] = RunnerConfig(type="codex", model="luna")
    svc.config.runners["opencode"] = RunnerConfig(type="opencode", model="local/qwen")
    codex = svc.create_run(runner="codex")
    opencode = svc.create_run(runner="opencode")
    config = yaml.safe_load(project.read_text("utf-8"))
    config["runners"].update(
        {name: svc.config.runners[name].public_settings() for name in ("codex", "opencode")}
    )
    project.write_text(yaml.safe_dump(config), encoding="utf-8")
    svc.activate(codex, project)
    before = (svc.root / "active-run.json").read_bytes()
    svc.activate(opencode, project)
    assert (svc.root / "active-run.json").read_bytes() == before
    assert (
        json.loads((svc.root / "active-opencode-run.json").read_text("utf-8"))["run_id"] == opencode
    )
    assert not fake.calls


def test_v25_continuation_keeps_frozen_reviewer_without_editor(service):
    svc, _ = service
    svc.config.translation.output_format = "segments-v1"
    svc.config.runners["reviewer"] = RunnerConfig(model="reviewer-model")
    parent = svc.create_run(translator="test", editor="reviewer", pipeline="translate-review")
    child = svc.continue_run(parent, runner="test")
    stages = svc.repository.run(child)["manifest"]["stages"]
    assert [s["runner"] for s in stages] == ["test", "reviewer"]
    assert stages[1]["runner_config"]["model"] == "reviewer-model"


def test_v25_continuation_pipeline_stage_runner_fallback(service):
    svc, _ = service
    svc.config.runners["reviewer"] = RunnerConfig(model="reviewer-model")
    svc.config.pipelines["translate-review"].stages[1].runner = "reviewer"
    parent = svc.create_run(runner="test")
    svc.config.translation.output_format = "segments-v1"
    child = svc.continue_run(parent, runner="test", pipeline="translate-review")
    stages = svc.repository.run(child)["manifest"]["stages"]
    assert [s["runner"] for s in stages] == ["test", "reviewer"]
