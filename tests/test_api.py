"""Offline FastAPI adapter tests (TestClient; no network or inference)."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from novel_harness.api import create_app
from novel_harness.runners.base import RunnerResult


class FakeRunner:
    def __init__(self):
        self.calls = 0

    def run(self, job):
        self.calls += 1
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


def _client(project, runner):
    app = create_app(project, runner_factory=lambda name: runner)
    return TestClient(app), app


def test_runners_endpoint(project):
    client, _ = _client(project, FakeRunner())
    response = client.get("/runners")
    assert response.status_code == 200
    assert response.json()[0]["name"] == "test"


def test_connections_endpoint_and_advanced_import_surface(project, monkeypatch):
    monkeypatch.setattr(
        "novel_harness.connections.discover_opencode",
        lambda: {
            "available": False,
            "reason": "OpenCode config is missing",
            "executable": False,
            "version": None,
            "agents": [{"id": "local", "model": "synthetic/model.gguf"}],
            "models": [{"provider": "synthetic", "model": "synthetic/model.gguf"}],
            "config_sha256": None,
        },
    )
    monkeypatch.setattr("novel_harness.connections.shutil.which", lambda _name: None)
    client, _ = _client(project, FakeRunner())
    response = client.get("/connections")
    assert response.status_code == 200
    assert response.json()["version"] == 1
    assert any(item["id"] == "test" for item in response.json()["profiles"])
    test_profile = next(item for item in response.json()["profiles"] if item["id"] == "test")
    assert test_profile["resolved_model"]["id"] == "fake-model"
    assert "configuration" not in test_profile
    assert "api_key" not in json.dumps(response.json())
    assert "agents" not in response.json()["opencode"]
    assert "models" not in response.json()["opencode"]
    assert "synthetic/model.gguf" not in json.dumps(response.json())
    assert "EMPTY" not in json.dumps(response.json())

    rejected = client.post(
        "/connections/opencode/import",
        json={
            "profile_id": "opencode-test",
            "agent_id": "translator",
            "model_override": "provider/model",
        },
    )
    assert rejected.status_code == 400
    assert "Advanced" in rejected.json()["detail"]

    monkeypatch.setattr(
        "novel_harness.service.import_opencode_profile",
        lambda profile_id, agent_id, **kwargs: {
            "profile_id": profile_id,
            "profile": {"agent_id": agent_id, **kwargs},
        },
    )
    imported = client.post(
        "/connections/opencode/import",
        json={
            "profile_id": "opencode-test",
            "agent_id": "translator",
            "advanced": True,
            "model_override": "provider/model",
        },
    )
    assert imported.status_code == 200
    assert imported.json()["profile_id"] == "opencode-test"

    direct = client.post(
        "/connections/direct/import",
        json={
            "profile_id": "direct-cloud",
            "model": "provider/model",
            "base_url": "https://provider.example/v1",
            "api_key_env": "PROVIDER_API_KEY",
        },
    )
    assert direct.status_code == 200
    assert direct.json()["profile"]["api_key_env"] == "PROVIDER_API_KEY"
    assert '"api_key":' not in json.dumps(direct.json()).lower()

    invalid_direct = client.post(
        "/connections/direct/import",
        json={
            "profile_id": "direct-cloud",
            "model": "provider/model",
            "base_url": "https://user:secret@provider.example/v1",
            "api_key_env": "PROVIDER_API_KEY",
        },
    )
    assert invalid_direct.status_code == 400
    assert "secret" not in invalid_direct.text


def test_wizard_options_show_inherited_and_explicit_override_without_secrets(
    project, source, monkeypatch
):
    config_dir = project.parent / "xdg" / "opencode"
    config_dir.mkdir(parents=True)
    (config_dir / "opencode.json").write_text(json.dumps({
        "provider": {"private": {
            "options": {"apiKey": "CANARY-SECRET"},
            "models": {"model-a": {}, "model-b": {}},
        }},
        "agent": {"translator": {"mode": "primary", "model": "private/model-a"}},
    }), "utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(project.parent / "xdg"))
    monkeypatch.setattr("novel_harness.connections.shutil.which", lambda _name: "/bin/opencode")
    monkeypatch.setattr(
        "novel_harness.connections.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "1.18.31\n", ""),
    )
    data = yaml.safe_load(project.read_text("utf-8"))
    data["execution_profiles"] = {
        "oc-inherit": {"kind": "harness", "harness": "opencode", "agent_id": "translator"},
        "oc-override": {
            "kind": "harness", "harness": "opencode", "agent_id": "translator",
            "model_policy": "override", "model_override": "private/model-b",
        },
        "oc-unknown": {"kind": "harness", "harness": "opencode", "agent_id": "unknown"},
    }
    data["translation_presets"] = {
        "default": "inherit",
        "presets": {
            "inherit": {"label": "Inherited", "runner": "oc-inherit", "pipeline": "direct"},
            "override": {"label": "Override", "runner": "oc-override", "pipeline": "direct"},
        },
    }
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    client, app = _client(project, FakeRunner())
    response = client.get("/wizard/options")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["default"] == "inherit"
    presets = {item["id"]: item for item in body["presets"]}
    profiles = {item["id"]: item for item in body["profiles"]}
    assert presets["inherit"]["execution_profile_id"] == "oc-inherit"
    assert presets["override"]["execution_profile_id"] == "oc-override"
    assert profiles["oc-inherit"]["model_policy"] == "inherit"
    assert profiles["oc-inherit"]["resolved_model"]["model"] == "model-a"
    assert profiles["oc-inherit"]["model_source"] == "inherited from OpenCode agent"
    assert profiles["oc-override"]["model_policy"] == "override"
    assert profiles["oc-override"]["resolved_model"]["model"] == "model-b"
    assert profiles["oc-unknown"]["available"] is False
    assert "CANARY-SECRET" not in response.text
    assert all("configuration" not in profile for profile in body["profiles"])
    advanced = client.post("/sample-runs", json={"execution_profile": "oc-override"})
    assert advanced.status_code == 201, advanced.text
    manifest = app.state.service.repository.run(advanced.json()["run_id"])["manifest"]
    execution = manifest["stages"][0]["execution"]
    assert execution["requested_model"] == {"id": "private/model-b"}
    assert execution["resolved_model"]["model"] == "model-b"
    assert execution["model_policy"] == "override"
    assert "CANARY-SECRET" not in json.dumps(manifest)


def test_wizard_book_samples_keep_distinct_sources_and_approval_gate(project):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["translation_presets"] = {
        "default": "test-mode",
        "presets": {"test-mode": {"label": "Test", "runner": "test", "pipeline": "direct"}},
    }
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    root = project.parent / "original"
    for title in ("book-a", "book-b"):
        folder = root / title
        folder.mkdir()
        for number in (1, 2):
            (folder / f"{number:04d}_chapter.txt").write_text(
                f"第{number}章\n\n" + f"{title}の本文。" * 20, "utf-8"
            )
    client, app = _client(project, FakeRunner())
    options = client.get("/wizard/options").json()
    assert options["presets"][0]["execution_profile_id"] == "test"
    books = {item["title"]: item for item in client.get("/library").json()["novels"]}
    assert books["book-a"]["novel_id"] != books["book-b"]["novel_id"]
    samples = {}
    for title in ("book-a", "book-b"):
        novel_id = books[title]["novel_id"]
        response = client.post("/sample-runs", json={"preset": "test-mode", "novel_id": novel_id})
        assert response.status_code == 201, response.text
        sample = response.json()["run_id"]
        samples[title] = sample
        assert app.state.service.repository.run(sample)["manifest"]["source_identity"]["novel_id"] == novel_id
        assert client.post(f"/sample-runs/{sample}/approve", json={"actor": "ui"}).status_code == 400
        assert client.post("/whole-book-runs", json={
            "sample_run_id": sample, "novel_id": novel_id, "approved": True,
        }).status_code == 400
        app.state.service.execute(sample)
        assert client.post(f"/sample-runs/{sample}/approve", json={"actor": "ui"}).status_code == 200
        wrong_id = books["book-b" if title == "book-a" else "book-a"]["novel_id"]
        assert client.post("/whole-book-runs", json={
            "sample_run_id": sample, "novel_id": wrong_id, "approved": True,
        }).status_code == 400
        whole = client.post("/whole-book-runs", json={
            "sample_run_id": sample, "novel_id": novel_id,
            "preset": "test-mode", "approved": True,
        })
        assert whole.status_code == 201, whole.text
        manifest = app.state.service.repository.run(whole.json()["run_id"])["manifest"]
        assert manifest["source_identity"]["novel_id"] == novel_id
        assert manifest["sample"]["sample_run_id"] == sample
    changed = root / "book-a" / "0001_chapter.txt"
    changed.write_text(changed.read_text("utf-8") + "\n変更", "utf-8")
    stale = client.post("/whole-book-runs", json={
        "sample_run_id": samples["book-a"], "novel_id": books["book-a"]["novel_id"],
        "preset": "test-mode", "approved": True,
    })
    assert stale.status_code == 400
    assert "new sample" in stale.text


def test_library_api_selection_and_missing_source(project):
    root = project.parent / "original"
    first = root / "book-a"
    second = root / "book-b"
    for path in (first, second):
        path.mkdir()
        for number in (1, 2):
            (path / f"{number:04d}.txt").write_text(
                f"第{number}章\n\n" + f"{path.name}の本文。" * 20, "utf-8"
            )
    client, app = _client(project, FakeRunner())
    listed = client.get("/library")
    assert listed.status_code == 200
    cards = {item["title"]: item for item in listed.json()["novels"]}
    assert set(cards) == {"book-a", "book-b"}
    book_id = cards["book-b"]["novel_id"]
    assert client.get(f"/library/{book_id}").json()["chapter_count"] == 2
    run_id = client.post("/runs", json={"runner": "test", "novel_id": book_id}).json()["run_id"]
    assert app.state.service.repository.run(run_id)["manifest"]["source_dir"] == str(second)
    sample = client.post(
        "/sample-runs", json={"runner": "test", "novel_id": book_id}
    ).json()["run_id"]
    app.state.service.execute(sample)
    assert client.post(f"/sample-runs/{sample}/approve", json={"actor": "test"}).status_code == 200
    whole = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample, "approved": True, "novel_id": book_id},
    )
    assert whole.status_code == 201, whole.text
    assert app.state.service.repository.run(whole.json()["run_id"])["manifest"]["source_dir"] == str(second)
    for path in second.iterdir():
        path.unlink()
    second.rmdir()
    rescanned = client.post("/library/rescan")
    assert rescanned.status_code == 200
    assert client.get(f"/library/{book_id}").json()["status"] == "missing"
    assert client.get("/runs").status_code == 200


def test_create_status_inspect_and_build_flow(project, source):
    app_client, app = _client(project, FakeRunner())

    created = app_client.post("/runs", json={"runner": "test", "chapters": [1]})
    assert created.status_code == 201
    run_id = created.json()["run_id"]

    pending = app_client.get(f"/runs/{run_id}/status")
    assert pending.status_code == 200
    assert pending.json()["status"] != "completed"

    app.state.service.execute(run_id)

    chapter = app_client.get(f"/runs/{run_id}/chapters/1")
    assert chapter.status_code == 200
    assert chapter.json()["complete"] is True

    metrics = app_client.get(f"/runs/{run_id}/metrics")
    assert metrics.status_code == 200

    built = app_client.post(f"/runs/{run_id}/build")
    assert built.status_code == 200
    assert Path(built.json()["path"]).is_file()


def test_start_delegates_to_worker(project, source, monkeypatch):
    client, _ = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]

    recorded = {}

    def fake_start_worker(config, run_id_arg):
        recorded["config"] = str(config)
        recorded["run_id"] = run_id_arg
        return {"run_id": run_id_arg, "pid": 123, "started": True, "worker_state": "stub"}

    monkeypatch.setattr("novel_harness.api.start_worker", fake_start_worker)

    response = client.post(f"/runs/{run_id}/start")

    assert response.status_code == 202
    assert response.json()["worker_state"] == "stub"
    assert recorded["run_id"] == run_id


def test_unknown_run_returns_400(project):
    client, _ = _client(project, FakeRunner())
    response = client.get("/runs/missing/status")
    assert response.status_code == 400


def test_library_lists_runs(project, source):
    client, _ = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]

    response = client.get("/runs")

    assert response.status_code == 200
    body = response.json()
    assert body["original_chapters"] == 1
    assert run_id in [run["run_id"] for run in body["runs"]]


def test_pause_endpoint_requests_boundary(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]

    response = client.post(f"/runs/{run_id}/pause")

    assert response.status_code == 202
    assert response.json()["pause_requested"] is True
    assert (app.state.service.run_dir(run_id) / "pause.requested").is_file()


def test_retry_failed_delegates_to_worker(project, source, monkeypatch):
    client, _ = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]

    recorded = {}

    def fake_start_worker(config, run_id_arg):
        recorded["run_id"] = run_id_arg
        return {"run_id": run_id_arg, "pid": 1, "started": True, "worker_state": "stub"}

    monkeypatch.setattr("novel_harness.api.start_worker", fake_start_worker)

    response = client.post(f"/runs/{run_id}/retry-failed")

    assert response.status_code == 202
    assert response.json()["worker_state"] == "stub"
    assert recorded["run_id"] == run_id


def test_progress_endpoint_reports_chunk_counts(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)

    response = client.get(f"/runs/{run_id}/progress")

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"]["completed"] == body["total_chunks"]
    assert body["chunks"]["pending"] == 0


def test_chapter_includes_validation_projection(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)

    body = client.get(f"/runs/{run_id}/chapters/1").json()

    assert body["complete"] is True
    assert isinstance(body["validation"], list)
    assert [entry["ordinal"] for entry in body["validation"]] == [
        row["ordinal"] for row in body["chunks"]
    ]


def test_index_serves_ui(project):
    client, _ = _client(project, FakeRunner())

    response = client.get("/")

    assert response.status_code == 200
    assert "Novel Harness" in response.text


def test_static_assets_served(project):
    client, _ = _client(project, FakeRunner())

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "loadLibrary" in response.text


def _first_chunk_layer(client, run_id, chapter=1):
    layers = client.get(f"/runs/{run_id}/chapters/{chapter}/layers").json()
    return layers, layers["final_status"][0]


def test_final_layer_edit_cas_epub_and_machine_preserved(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)
    _, chunk = _first_chunk_layer(client, run_id)
    chunk_id = chunk["chunk_id"]
    machine_sha = chunk["machine_sha256"]
    endpoint = f"/runs/{run_id}/chapters/1/chunks/{chunk_id}/final"

    stale = client.post(
        endpoint,
        json={"action": "edit", "text": "Правка", "expected_machine_sha256": "deadbeef"},
    )
    assert stale.status_code == 409

    edited = client.post(
        endpoint,
        json={"action": "edit", "text": "Правка человека", "expected_machine_sha256": machine_sha},
    )
    assert edited.status_code == 200
    final_sha = edited.json()["final_sha256"]
    assert final_sha

    # An existing final cannot be mutated by omitting the CAS hash.
    missing_cas = client.post(
        endpoint,
        json={"action": "edit", "text": "Правка без CAS", "expected_machine_sha256": machine_sha},
    )
    assert missing_cas.status_code == 409

    stale_final = client.post(
        endpoint,
        json={
            "action": "edit",
            "text": "Правка-2",
            "expected_machine_sha256": machine_sha,
            "expected_final_sha256": "nope",
        },
    )
    assert stale_final.status_code == 409

    layers = client.get(f"/runs/{run_id}/chapters/1/layers").json()
    assert "Правка человека" in layers["final"]
    assert layers["final"] != layers["machine"]

    # The raw machine candidate file is untouched.
    row = next(r for r in app.state.service.repository.chunks(run_id) if r["chunk_id"] == chunk_id)
    machine_text = (app.state.service.run_dir(run_id) / row["output_path"]).read_text("utf-8")
    assert "Правка человека" not in machine_text

    # Resume/execute preserves the human layer.
    app.state.service.execute(run_id)
    assert "Правка человека" in client.get(f"/runs/{run_id}/chapters/1/layers").json()["final"]

    # Evidence for /final listing.
    final_rows = client.get(f"/runs/{run_id}/final").json()
    assert [entry["chunk_id"] for entry in final_rows] == [chunk_id]

    # EPUB is built from the final layer.
    built = client.post(f"/runs/{run_id}/build")
    assert built.status_code == 200
    epub = Path(built.json()["path"]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(epub)) as archive:
        chapter_html = next(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.endswith(f"chapter-{1:04d}.xhtml")
        )
    assert "Правка человека" in chapter_html
    record = json.loads((app.state.service.run_dir(run_id) / "book.json").read_text("utf-8"))
    assert any(entry["final_sha256"] == final_sha for entry in record["final_layer"])

    needs_review = client.post(
        endpoint,
        json={
            "action": "needs_review",
            "expected_machine_sha256": machine_sha,
            "expected_final_sha256": final_sha,
        },
    )
    assert needs_review.json()["status"] == "needs_review"
    review_sha = needs_review.json()["final_sha256"]
    assert review_sha

    # Revert of an existing final also requires the current final hash.
    missing_revert = client.post(
        endpoint, json={"action": "revert", "expected_machine_sha256": machine_sha}
    )
    assert missing_revert.status_code == 409

    reverted = client.post(
        endpoint,
        json={
            "action": "revert",
            "expected_machine_sha256": machine_sha,
            "expected_final_sha256": review_sha,
        },
    )
    assert reverted.json()["status"] == "machine"
    assert client.get(f"/runs/{run_id}/final").json() == []


def test_final_layer_accept_uses_machine_candidate(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)
    _, chunk = _first_chunk_layer(client, run_id)

    accepted = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk['chunk_id']}/final",
        json={"action": "accept", "expected_machine_sha256": chunk["machine_sha256"]},
    )

    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"
    assert len(accepted.json()["final_sha256"]) == 64
    assert accepted.json()["final_sha256"] != hashlib.sha256(
        accepted.json()["final_text"].encode("utf-8")
    ).hexdigest()


def test_final_layer_requeue_preserves_edit_and_delegates(project, source, monkeypatch):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)
    _, chunk = _first_chunk_layer(client, run_id)
    chunk_id = chunk["chunk_id"]
    machine_sha = chunk["machine_sha256"]
    edited = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk_id}/final",
        json={"action": "edit", "text": "Правка", "expected_machine_sha256": machine_sha},
    )
    assert edited.status_code == 200

    recorded = {}

    def fake_start_worker(config, run_id_arg):
        recorded["run_id"] = run_id_arg
        return {"run_id": run_id_arg, "pid": 1, "started": True, "worker_state": "stub"}

    monkeypatch.setattr("novel_harness.api.start_worker", fake_start_worker)

    stale = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk_id}/requeue",
        json={"expected_machine_sha256": "deadbeef"},
    )
    assert stale.status_code == 409

    response = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk_id}/requeue",
        json={"expected_machine_sha256": machine_sha},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert response.json()["worker"]["worker_state"] == "stub"
    assert recorded["run_id"] == run_id
    # Non-destructive: the human final survives and the machine output stays
    # addressable until the worker commits a replacement.
    final_rows = client.get(f"/runs/{run_id}/final").json()
    assert len(final_rows) == 1
    assert final_rows[0]["final_text"] == "Правка"
    assert final_rows[0]["stale"] is False
    row = next(r for r in app.state.service.repository.chunks(run_id) if r["chunk_id"] == chunk_id)
    assert row["status"] == "pending"
    assert row["output_sha256"] == machine_sha


def _bible_entry(kind="character", source="ルゥ", target="Ру"):
    return {
        "kind": kind,
        "source_form": source,
        "target_form": target,
        "aliases": ["ルー"],
        "source_anchors": ["ルゥは女の子"],
    }


def test_book_bible_proposals_never_auto_confirm(project):
    client, _ = _client(project, FakeRunner())

    created = client.post(
        "/book-bible", json={"entries": [_bible_entry()], "proposer": "model-x"}
    )
    assert created.status_code == 201
    entry = created.json()["created"][0]
    assert entry["trust"] == "proposed"
    assert entry["provenance"]["proposal_source"] == "model"
    entry_id = entry["entry_id"]

    listed = client.get("/book-bible").json()
    assert listed["entries"][0]["trust"] == "proposed"
    assert listed["authoritative_projection"] == []
    assert listed["projection_sha256"]

    confirmed = client.post(
        f"/book-bible/{entry_id}", json={"action": "confirm", "actor": "human"}
    )
    assert confirmed.json()["trust"] == "confirmed"
    projection = client.get("/book-bible").json()["authoritative_projection"]
    assert [row["target_form"] for row in projection] == ["Ру"]

    locked = client.post(
        f"/book-bible/{entry_id}", json={"action": "lock", "actor": "human"}
    )
    assert locked.json()["trust"] == "locked"

    rejected = client.post(
        f"/book-bible/{entry_id}", json={"action": "reject", "actor": "human"}
    )
    assert rejected.json()["status"] == "rejected"
    assert client.get("/book-bible").json()["entries"] == []


def test_book_bible_duplicate_proposal_is_skipped_not_overwritten(project):
    client, _ = _client(project, FakeRunner())
    client.post("/book-bible", json={"entries": [_bible_entry()]})

    again = client.post("/book-bible", json={"entries": [_bible_entry(target="Ру-2")]})

    assert again.json()["created"] == []
    assert len(again.json()["skipped"]) == 1
    assert client.get("/book-bible").json()["entries"][0]["target_form"] == "Ру"


def test_book_bible_edit_changes_projection_hash(project):
    client, _ = _client(project, FakeRunner())
    entry = client.post("/book-bible", json={"entries": [_bible_entry()]}).json()["created"][0]
    client.post(f"/book-bible/{entry['entry_id']}", json={"action": "confirm", "actor": "human"})
    before = client.get("/book-bible").json()["projection_sha256"]

    client.post(
        f"/book-bible/{entry['entry_id']}",
        json={"action": "edit", "actor": "human", "target_form": "Ру-2"},
    )
    after = client.get("/book-bible").json()

    assert after["projection_sha256"] != before
    assert after["authoritative_projection"][0]["target_form"] == "Ру-2"
    assert after["entries"][0]["provenance"]["history"][-1]["action"] == "edit"


def test_book_bible_rejects_unknown_entry_and_unknown_action(project):
    client, _ = _client(project, FakeRunner())
    assert client.post("/book-bible/missing", json={"action": "confirm"}).status_code == 400
    entry = client.post("/book-bible", json={"entries": [_bible_entry()]}).json()["created"][0]
    assert client.post(
        f"/book-bible/{entry['entry_id']}", json={"action": "bogus"}
    ).status_code == 422


def test_sample_first_gate_over_api(project, source):
    client, app = _client(project, FakeRunner())
    sample = client.post("/sample-runs", json={"runner": "test", "chapters": [1]})
    assert sample.status_code == 201
    sample_id = sample.json()["run_id"]

    # Explicit approval flag is required for the UI whole-book path.
    no_flag = client.post(
        "/whole-book-runs", json={"sample_run_id": sample_id, "runner": "test"}
    )
    assert no_flag.status_code == 409

    # Even with the flag, an unapproved sample is refused.
    unapproved = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample_id, "approved": True, "runner": "test"},
    )
    assert unapproved.status_code == 400

    assert client.get(f"/sample-runs/{sample_id}/approval").json() is None

    # An incomplete sample cannot be approved either (server-side gate).
    incomplete = client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})
    assert incomplete.status_code == 400
    assert "must complete successfully before approval" in incomplete.json()["detail"]

    # Complete the sample offline, then approval and whole-book launch succeed.
    app.state.service.execute(sample_id)
    approved = client.post(f"/sample-runs/{sample_id}/approve", json={"actor": "ui"})
    assert approved.status_code == 200

    whole = client.post(
        "/whole-book-runs",
        json={"sample_run_id": sample_id, "approved": True, "runner": "test"},
    )
    assert whole.status_code == 201
    manifest = app.state.service.repository.run(whole.json()["run_id"])["manifest"]
    assert manifest["sample"]["sample_run_id"] == sample_id


def test_translation_memory_endpoints_and_cache_state(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)

    chapter = client.get(f"/runs/{run_id}/chapters/1").json()
    assert "cache" in chapter["validation"][0]

    chunk = chapter["final_status"][0]
    edited = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk['chunk_id']}/final",
        json={
            "action": "edit",
            "text": "Правка",
            "expected_machine_sha256": chunk["machine_sha256"],
        },
    )
    assert edited.status_code == 200

    promoted = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk['chunk_id']}/tm", json={"actor": "ui"}
    )
    assert promoted.status_code == 200
    assert promoted.json()["inserted"] is True

    memory = client.get("/translation-memory").json()
    assert memory["count"] == 1
    assert memory["entries"][0]["provenance"]["source_run_id"] == run_id


def test_promote_endpoint_rejects_unaccepted_and_stale_finals(project, source):
    client, app = _client(project, FakeRunner())
    run_id = client.post("/runs", json={"runner": "test", "chapters": [1]}).json()["run_id"]
    app.state.service.execute(run_id)
    chunk = client.get(f"/runs/{run_id}/chapters/1").json()["final_status"][0]

    needs = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk['chunk_id']}/final",
        json={
            "action": "needs_review",
            "expected_machine_sha256": chunk["machine_sha256"],
        },
    )
    assert needs.status_code == 200
    rejected = client.post(
        f"/runs/{run_id}/chapters/1/chunks/{chunk['chunk_id']}/tm", json={"actor": "ui"}
    )
    assert rejected.status_code == 400
    assert client.get("/translation-memory").json()["count"] == 0
