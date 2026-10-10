import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click import Command, Context
from conftest import FakeRunner
from mock_endpoint import endpoint
from typer.testing import CliRunner

from novel_harness.cli import app as cli_app
from novel_harness.cli import parse_chapter_list, selection
from novel_harness.cli_output import render
from novel_harness.cli_selection import complete_run_ids, resolve_run_id
from novel_harness.config import load_config
from novel_harness.context_runs import generate_artifact
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService

FROZEN_SAMPLE = [33, 49, 63, 91, 217]


def make_sparse_source(tmp_path: Path, numbers: list[int]) -> Path:
    """Deterministic synthetic source, independent of any gitignored real data."""
    source = tmp_path / "original"
    source.mkdir()
    for number in numbers:
        body = (
            f"第{number}章 合成\n\n"
            f"これは合成された第{number}章の固有の一文です。ユキは走った。\n"
        )
        (source / f"{number:04d}_synthetic.txt").write_text(body, encoding="utf-8")
    return source


def test_v12_cli_help_under_legacy_windows_encoding():
    result = subprocess.run(
        [sys.executable, "-m", "novel_harness", "--help"],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "cp1251"},
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "Russian" in result.stdout.decode("utf-8")


def test_library_admin_json_commands_and_read_only_list_show(project):
    root = project.parent / "original"
    for name in ("alpha", "beta"):
        path = root / name
        path.mkdir()
        for number in (1, 2):
            (path / f"{number:04d}.txt").write_text(
                f"第{number}章\n\n{name}の本文。" * 10, "utf-8"
            )
    bad = root / "bad"
    bad.mkdir()
    (bad / "not-numbered.txt").write_text("壊れた候補", "utf-8")
    first = invoke_cli("library", "rescan", "--config", project, "--json")
    rows = {item["title"]: item for item in first["novels"]}
    assert rows["alpha"]["status"] == rows["beta"]["status"] == "ready"
    assert rows["bad"]["status"] == "invalid"
    source_before = {path: path.read_bytes() for path in root.rglob("*.txt")}
    listed = invoke_cli("library", "list", "--config", project, "--json")
    shown = invoke_cli(
        "library", "show", rows["alpha"]["novel_id"], "--config", project, "--json"
    )
    assert shown["source_path"] == str(root / "alpha")
    assert listed == first
    assert invoke_cli("library", "rescan", "--config", project, "--json") == first
    assert {path: path.read_bytes() for path in root.rglob("*.txt")} == source_before
    shutil.rmtree(root / "alpha")
    after_delete = invoke_cli("library", "rescan", "--config", project, "--json")
    assert next(
        item for item in after_delete["novels"] if item["novel_id"] == rows["alpha"]["novel_id"]
    )["status"] == "missing"


def test_connections_cli_lists_and_imports_opencode_without_credentials(
    project, tmp_path, monkeypatch
):
    config_home = tmp_path / "xdg"
    opencode_dir = config_home / "opencode"
    opencode_dir.mkdir(parents=True)
    (opencode_dir / "opencode.jsonc").write_text(
        '{"provider":{"private":{"options":{"apiKey":"CANARY-CONFIG-SECRET"},'
        '"models":{"model-a":{}}}},'
        '"agent":{"translator":{"mode":"primary",'
        '"model":"private/model-a"}}}',
        "utf-8",
    )
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    executable = binary_dir / "opencode"
    executable.write_text("#!/bin/sh\nprintf '1.18.31\\n'\n", "utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("PATH", f"{binary_dir}:{Path(sys.executable).parent}")

    imported = invoke_cli(
        "connections",
        "import-opencode",
        "--profile-id",
        "oc-cli",
        "--agent",
        "translator",
        "--config",
        project,
        "--json",
    )
    assert imported["profile_id"] == "oc-cli"
    assert imported["profile"]["model_policy"] == "inherit"
    assert "CANARY-CONFIG-SECRET" not in json.dumps(imported)

    listed = invoke_cli("connections", "list", "--config", project, "--json")
    assert listed["opencode"]["available"] is True
    assert "CANARY-CONFIG-SECRET" not in json.dumps(listed)


def invoke_cli(*args):
    result = subprocess.run(
        [sys.executable, "-m", "novel_harness", *map(str, args)],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8") + result.stdout.decode("utf-8")
    return json.loads(result.stdout.decode("utf-8"))


def test_library_remaining_cli_and_launch_delegate_without_inline_inference(project, monkeypatch):
    root = project.parent / "original"
    for name in ("alpha", "beta"):
        path = root / name
        path.mkdir()
        for number in (1, 2):
            (path / f"{number:04d}.txt").write_text(
                f"第{number}章\n\n{name}の本文。" * 8, "utf-8"
            )
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    service.rescan_library()
    cards = {item["title"]: item for item in service.library_novels()["novels"]}
    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: service))
    launched = []

    def fake_launch(service_arg, config, run_id, *, monitor=False):
        launched.append((service_arg, config, run_id, monitor))
        return {"run_id": run_id, "started": True, "worker_state": "fake"}

    monkeypatch.setattr("novel_harness.launch.launch_run", fake_launch)
    runner = CliRunner()
    listed = runner.invoke(cli_app, ["library", "remaining", "--config", str(project), "--json"])
    assert listed.exit_code == 0, listed.stdout
    payload = json.loads(listed.stdout)
    assert [item["title"] for item in payload["novels"]] == ["alpha", "beta"]

    launched_result = runner.invoke(
        cli_app,
        [
            "library",
            "launch-remaining",
            "--novel-id",
            cards["alpha"]["novel_id"],
            "--runner",
            "test",
            "--config",
            str(project),
            "--json",
        ],
    )
    assert launched_result.exit_code == 0, (launched_result.stdout, launched_result.stderr, launched_result.exception)
    assert len(launched) == 1
    assert json.loads(launched_result.stdout)["novels"][0]["run_id"] == launched[0][2]


def test_one_chapter_over_local_http_and_cli_resume(project, source):
    before = source.read_bytes()
    with endpoint() as (url, calls):
        data = yaml.safe_load(project.read_text("utf-8"))
        data["runners"]["test"]["base_url"] = url
        project.write_text(yaml.safe_dump(data), "utf-8")
        command = ["translate", "--runner", "test", "--chapter", "1", "--config", project, "--json"]
        result = invoke_cli(*command)
        assert result["status"] == "completed"
        assert result["total_chunks"] == len(calls)
        assert all(c["path"] == "/v1/chat/completions" for c in calls)
        count = len(calls)
        repeated = invoke_cli(*command)
        assert repeated["run_id"] == result["run_id"]
        assert len(calls) == count
        chapter = invoke_cli(
            "chapter", "--run", result["run_id"], "--chapter", "1", "--config", project, "--json"
        )
        assert chapter["complete"]
        assert "第一章" in chapter["source"]
        assert "Тестовая глава" in chapter["translation"]
    assert source.read_bytes() == before


def test_sparse_chapter_list_parsing_preserves_exact_sample():
    assert parse_chapter_list("33,49,63,91,217") == FROZEN_SAMPLE
    assert parse_chapter_list(" 33 , 49 ") == [33, 49]
    assert selection(None, None, False, "33,49,63,91,217") == FROZEN_SAMPLE


def test_run_target_resolver_accepts_equal_selectors_and_rejects_conflicts():
    assert resolve_run_id("run-a", "run-a") == "run-a"
    assert resolve_run_id(None, "run-a") == "run-a"
    assert resolve_run_id(None, None, required=False) is None
    with pytest.raises(ValueError, match="same run"):
        resolve_run_id("run-a", "run-b")
    with pytest.raises(ValueError, match="Provide RUN_ID"):
        resolve_run_id(None, None)


def test_cli_completion_reads_only_the_selected_store(project, source, service):
    svc, fake = service
    run_id = svc.create_run(runner="test", chapters=[1])
    root = Context(Command("novel"))
    root.params["config"] = None
    command = Context(Command("pause"), parent=root)
    command.params["config"] = project

    suggestions = complete_run_ids(command, None, run_id[:8])

    assert [item.value for item in suggestions] == [run_id]
    assert "pending" in suggestions[0].help
    assert fake.calls == []


def test_cli_completion_returns_no_suggestions_for_missing_or_invalid_config(tmp_path):
    root = Context(Command("novel"))
    root.params["config"] = None
    command = Context(Command("pause"), parent=root)
    command.params["config"] = tmp_path / "missing.yaml"

    assert complete_run_ids(command, None, "2026") == []


def test_cli_pause_resolves_unique_short_selector_before_mutation(project, source, service, monkeypatch):
    svc, fake = service
    run_id = svc.create_run(runner="test", chapters=[1])
    monkeypatch.setattr(
        TranslationService, "from_config", classmethod(lambda cls, path: svc)
    )

    result = CliRunner().invoke(
        cli_app, ["pause", run_id[:8], "--config", str(project), "--json"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["run_id"] == run_id
    assert (svc.run_dir(run_id) / "pause.requested").is_file()
    assert fake.calls == []


def test_pause_positional_and_legacy_run_select_once_before_mutation(monkeypatch):
    class StubService:
        def __init__(self):
            self.calls = []

        def pause(self, run_id):
            self.calls.append(run_id)
            return {"run_id": run_id, "pause_requested": True}

        def status(self, run_id):
            self.calls.append(("status", run_id))
            return {"runs": []} if run_id is None else {"run_id": run_id}

    service = StubService()
    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: service))
    runner = CliRunner()
    equal = runner.invoke(cli_app, ["pause", "run-a", "--run", "run-a", "--json"])
    assert equal.exit_code == 0, equal.stdout
    assert service.calls == ["run-a"]

    service.calls.clear()
    conflict = runner.invoke(cli_app, ["pause", "run-a", "--run", "run-b", "--json"])
    assert conflict.exit_code == 2
    assert service.calls == []

    listing = runner.invoke(cli_app, ["status", "--json"])
    assert listing.exit_code == 0, listing.stdout
    assert service.calls == [("status", None)]


def test_resume_background_uses_launcher_and_foreground_stays_synchronous(monkeypatch):
    class StubService:
        def __init__(self):
            self.resume_calls = []

        def resume(self, run_id):
            self.resume_calls.append(run_id)
            return {"run_id": run_id, "status": "completed"}

    service = StubService()
    launches = []
    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: service))

    def fake_launch(service_arg, config, run_id, *, monitor=False):
        launches.append((service_arg, config, run_id, monitor))
        return {
            "run_id": run_id,
            "started": False,
            "already_running": True,
            "retry_requested": True,
            "worker_state": "worker.json",
        }

    monkeypatch.setattr("novel_harness.launch.launch_run", fake_launch)
    runner = CliRunner()
    background = runner.invoke(cli_app, ["resume", "run-a", "--background", "--json"])
    assert background.exit_code == 0, background.stdout
    assert json.loads(background.stdout)["retry_requested"] is True
    assert launches == [(service, Path("config.yaml").resolve(), "run-a", False)]
    assert service.resume_calls == []

    background_retry = runner.invoke(
        cli_app, ["retry-failed", "--run", "run-b", "--background", "--json"]
    )
    assert background_retry.exit_code == 0, background_retry.stdout
    assert json.loads(background_retry.stdout)["retry_requested"] is True
    assert launches[-1] == (service, Path("config.yaml").resolve(), "run-b", False)
    assert service.resume_calls == []

    foreground = runner.invoke(cli_app, ["retry-failed", "--run", "run-c", "--json"])
    assert foreground.exit_code == 0, foreground.stdout
    assert service.resume_calls == ["run-c"]

    monkeypatch.setattr(
        "novel_harness.launch.launch_run",
        lambda *args, **kwargs: {"run_id": "run-c", "started": False, "error": "dead"},
    )
    failed = runner.invoke(cli_app, ["resume", "run-c", "--background", "--json"])
    assert failed.exit_code == 1


def test_quota_cli_commands_wake_one_live_worker_without_spawning(project, source, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from filelock import FileLock

    from novel_harness.artifacts import write_json
    from novel_harness.run_factory import now

    path = project
    service = TranslationService(load_config(path), lambda name: FakeRunner())
    run_id = service.create_run(runner="test")
    directory = service.run_dir(run_id)
    write_json(
        directory / "retry.json",
        {
            "retry_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "reason": "quota",
            "deadline_source": "provider",
        },
    )
    write_json(
        directory / "worker.json",
        {"run_id": run_id, "pid": os.getpid(), "status": "running", "started_at": now()},
    )
    monkeypatch.setattr("novel_harness.progress._pid_alive", lambda pid, **kwargs: True)
    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: service))
    spawns = []
    monkeypatch.setattr(
        "novel_harness.worker._spawn_worker",
        lambda *args, **kwargs: spawns.append(args),
    )
    runner = CliRunner()
    worker_lock = FileLock(str(directory / ".worker.lock"))
    worker_lock.acquire()
    try:
        traces = []
        for command in (
            ["start", run_id, "--config", str(path), "--json"],
            ["resume", run_id, "--background", "--config", str(path), "--json"],
            ["retry-failed", run_id, "--background", "--config", str(path), "--json"],
        ):
            result = runner.invoke(cli_app, command)
            assert result.exit_code == 0, result.stdout
            payload = json.loads(result.stdout)
            traces.append((command[0], payload["pid"], payload["retry_requested"]))
            assert payload["already_running"] is True
    finally:
        worker_lock.release()

    assert spawns == []
    assert (directory / "retry-now.requested").is_file()
    print(f"CLI_TRACE run={run_id} traces={traces} spawns={len(spawns)}")


def test_cli_config_precedence_covers_root_local_env_and_ancestor(tmp_path, monkeypatch):
    class StubService:
        def runners(self):
            return []

        def connections(self):
            return {}

    seen = []
    service = StubService()
    monkeypatch.setattr(
        TranslationService,
        "from_config",
        classmethod(lambda cls, path: seen.append(Path(path)) or service),
    )
    project = tmp_path / "project"
    nested = project / "deep" / "inside"
    nested.mkdir(parents=True)
    root_config = tmp_path / "root config.yaml"
    local_config = nested / "конфиг с пробелом.yaml"
    env_config = tmp_path / "env.yaml"
    nearest_config = project / "config.yaml"
    for path in (root_config, local_config, env_config, nearest_config):
        path.write_text("# fake config path for the stub\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    runner = CliRunner()

    local = runner.invoke(
        cli_app,
        [
            "-c",
            str(root_config),
            "connections",
            "list",
            "--config",
            local_config.name,
            "--json",
        ],
    )
    assert local.exit_code == 0, local.output
    assert seen[-1] == local_config.resolve()

    root = runner.invoke(cli_app, ["-c", str(root_config), "runners", "--json"])
    assert root.exit_code == 0, root.output
    assert seen[-1] == root_config.resolve()

    monkeypatch.setenv("NOVEL_CONFIG", str(env_config))
    environment = runner.invoke(cli_app, ["runners", "--json"])
    assert environment.exit_code == 0, environment.output
    assert seen[-1] == env_config.resolve()

    monkeypatch.delenv("NOVEL_CONFIG")
    ancestor = runner.invoke(cli_app, ["runners", "--json"])
    assert ancestor.exit_code == 0, ancestor.output
    assert seen[-1] == nearest_config.resolve()


def test_cli_config_resolution_rejects_invalid_selected_path_without_fallback(tmp_path, monkeypatch):
    cwd = tmp_path / "outside"
    cwd.mkdir()
    fallback = tmp_path / "fallback.yaml"
    fallback.write_text("# fallback\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("NOVEL_CONFIG", str(fallback))
    calls = []
    monkeypatch.setattr(
        TranslationService,
        "from_config",
        classmethod(lambda cls, path: calls.append(path)),
    )

    result = CliRunner().invoke(cli_app, ["runners", "--config", "missing.yaml", "--json"])

    assert result.exit_code == 2
    assert calls == []
    assert "does not exist" in result.stderr
    assert "missing.yaml" in result.stderr


def test_cli_config_resolution_reports_actionable_error_without_match(tmp_path, monkeypatch):
    cwd = tmp_path / "isolated"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("NOVEL_CONFIG", raising=False)

    result = CliRunner().invoke(cli_app, ["runners", "--json"])

    assert result.exit_code == 2
    assert "No config.yaml found" in result.stderr
    assert "--config PATH" in result.stderr


def test_new_run_persists_secret_free_config_identity(project, source):
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = service.create_run(runner="test", chapters=[1])

    manifest = service.repository.run(run_id)["manifest"]
    identity = manifest["config_identity"]
    assert identity["version"] == 1
    assert identity["path"] == str(project.resolve())
    assert len(identity["sha256"]) == 64
    saved = (service.run_dir(run_id) / "manifest.json").read_text("utf-8")
    assert json.loads(saved)["config_identity"] == identity
    assert "api_key" not in json.dumps(identity).lower()


def test_run_only_status_resolves_unique_project_config(project, source, monkeypatch):
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = service.create_run(runner="test", chapters=[1])
    monkeypatch.chdir(project.parent)

    result = CliRunner().invoke(cli_app, ["status", run_id, "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["run_id"] == run_id


def test_run_only_background_resume_hands_worker_exact_config(project, source, monkeypatch):
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = service.create_run(runner="test", chapters=[1])
    monkeypatch.chdir(project.parent)
    calls = []

    def fake_launch(service_arg, config, selected, *, monitor=False):
        calls.append((service_arg, config, selected, monitor))
        return {"run_id": selected, "started": True}

    monkeypatch.setattr("novel_harness.launch.launch_run", fake_launch)
    result = CliRunner().invoke(cli_app, ["resume", run_id, "--background", "--json"])

    assert result.exit_code == 0, result.output
    assert calls[0][1:] == (project.resolve(), run_id, False)


def test_legacy_run_only_resolution_rejects_ambiguous_matching_configs(project, source, monkeypatch):
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = service.create_run(runner="test", chapters=[1])
    manifest = service.repository.run(run_id)["manifest"]
    manifest.pop("config_identity", None)
    service.repository.replace_manifest(run_id, manifest)
    (service.run_dir(run_id) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    alternate = project.parent / "config.копия.yaml"
    alternate.write_bytes(project.read_bytes())
    monkeypatch.chdir(project.parent)

    result = CliRunner().invoke(cli_app, ["status", run_id, "--json"])

    assert result.exit_code == 2
    assert "multiple project configs" in result.stderr
    assert run_id not in result.stdout


def test_explicit_config_cannot_switch_identified_run_to_another_path(project, source):
    service = TranslationService(load_config(project), lambda name: FakeRunner())
    run_id = service.create_run(runner="test", chapters=[1])
    alternate = project.parent / "config.альтернативный.yaml"
    alternate.write_bytes(project.read_bytes())

    result = CliRunner().invoke(
        cli_app, ["status", run_id, "--config", str(alternate), "--json"]
    )

    assert result.exit_code == 2
    assert "compatible saved run" in result.stderr


def test_run_only_resolution_does_not_scan_outside_project(project, source, tmp_path, monkeypatch):
    outside = tmp_path.parent / "outside-project"
    outside.mkdir()
    monkeypatch.chdir(outside)

    result = CliRunner().invoke(cli_app, ["status", "missing-run", "--json"])

    assert result.exit_code == 2
    assert "does not scan unrelated directories" in result.stderr


def test_human_pause_and_json_pause_keep_distinct_contracts(monkeypatch):
    class StubService:
        def pause(self, run_id):
            return {"run_id": run_id, "pause_requested": True}

    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: StubService()))
    runner = CliRunner()

    human = runner.invoke(cli_app, ["pause", "run-a"])
    assert human.exit_code == 0, human.output
    assert "Pause requested" in human.output
    assert "{" not in human.output
    assert "\x1b[" not in human.output

    machine = runner.invoke(cli_app, ["pause", "run-a", "--json"])
    assert machine.exit_code == 0, machine.output
    assert json.loads(machine.stdout) == {"run_id": "run-a", "pause_requested": True}


def test_human_errors_are_bounded_and_preserve_json_error_contract():
    runner = CliRunner()
    human = runner.invoke(cli_app, ["pause"])
    assert human.exit_code == 2
    assert "Error:" in human.output
    assert "--help" in human.output

    machine = runner.invoke(cli_app, ["pause", "--json"])
    assert machine.exit_code == 2
    assert json.loads(machine.stderr) == {"error": "Provide RUN_ID or --run RUN_ID"}


def test_human_errors_redact_credential_shaped_values_without_changing_json(monkeypatch):
    class StubService:
        def pause(self, run_id):
            raise ValueError("api_key=canary-token password:secret Bearer abc123")

    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: StubService()))
    runner = CliRunner()
    human = runner.invoke(cli_app, ["pause", "run-a"])
    assert human.exit_code == 2
    assert "canary-token" not in human.output
    assert "secret" not in human.output
    assert "abc123" not in human.output
    assert "<redacted>" in human.output

    machine = runner.invoke(cli_app, ["pause", "run-a", "--json"])
    assert machine.exit_code == 2
    payload = json.loads(machine.stderr)
    assert set(payload) == {"error"}
    assert "canary-token" not in payload["error"]
    assert "secret" not in payload["error"]
    assert "abc123" not in payload["error"]
    assert "<redacted>" in payload["error"]


@pytest.mark.parametrize(
    "message",
    [
        "api_key=canary-token",
        "api-key: canary-token",
        "token=canary-token",
        "password: secret-value",
        "secret=secret-value",
        "Authorization: Bearer abc123",
        "Bearer abc123",
        "line one\napi_key=multiline-secret\nline three",
        "api_key=" + "x" * 10_000,
    ],
)
def test_json_errors_redact_and_bound_credential_shaped_values(monkeypatch, message):
    class StubService:
        def pause(self, run_id):
            raise ValueError(message)

    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: StubService()))
    result = CliRunner().invoke(cli_app, ["pause", "run-a", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert set(payload) == {"error"}
    assert len(payload["error"]) <= 240
    assert "canary-token" not in payload["error"]
    assert "secret-value" not in payload["error"]
    assert "abc123" not in payload["error"]


def test_safe_json_errors_remain_informative():
    result = CliRunner().invoke(cli_app, ["pause", "missing-run", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stderr) == {"error": "Unknown run selector: missing-run"}


def test_generic_human_rendering_bounds_multiline_values():
    rendered = render({"diagnostic": "line\n" * 10_000})
    assert len(rendered) <= 4_020
    assert rendered.endswith("…")


def test_human_list_commands_use_bounded_tables_and_grouped_help(monkeypatch):
    class StubService:
        def runners(self):
            return [{"id": "local", "kind": "direct", "available": True, "model": "test"}]

        def presets(self):
            return [{"id": "fast", "label": "Fast", "runner": "local"}]

        def default_preset(self):
            return "fast"

        def connections(self):
            return {"profiles": [{"id": "local", "kind": "direct", "available": True}]}

    monkeypatch.setattr(TranslationService, "from_config", classmethod(lambda cls, path: StubService()))
    runner = CliRunner()
    runners = runner.invoke(cli_app, ["runners"])
    presets = runner.invoke(cli_app, ["presets"])
    connections = runner.invoke(cli_app, ["connections", "list"])
    assert all(result.exit_code == 0 for result in (runners, presets, connections))
    assert "Id" in runners.output and "local" in runners.output
    assert "Id" in presets.output and "fast" in presets.output
    assert "Id" in connections.output and "local" in connections.output
    assert all("\x1b[" not in result.output for result in (runners, presets, connections))

    help_result = runner.invoke(cli_app, ["--help"])
    assert help_result.exit_code == 0
    for panel in ("Translation control", "Library", "Configuration", "Research and review"):
        assert panel in help_result.output
    for command in ("start", "resume", "retry-failed", "watch"):
        assert command in help_result.output
    assert "tui" in help_result.output


def test_sparse_chapter_list_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate chapter"):
        parse_chapter_list("33,33,49")


def test_sparse_chapter_list_rejects_invalid_and_negative_values():
    for bad in ("", "33,", ",33", "a,1", "33,,49", "-1", "1,-2", "1.5"):
        with pytest.raises(ValueError):
            parse_chapter_list(bad)


def test_selection_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        selection(None, None, False)
    with pytest.raises(ValueError, match="exactly one"):
        selection(33, None, True)
    with pytest.raises(ValueError, match="exactly one"):
        selection(None, "1-2", False, "33")
    with pytest.raises(ValueError, match="exactly one"):
        selection(1, "1-2", False, "33")
    with pytest.raises(ValueError, match="exactly one"):
        selection(1, None, False, "33")


def _fake_context_call(job):
    quote_id = None
    for document in job.context.get("source_documents", []):
        if document["source_ref"] == "current_chapter" and document["quote_units"]:
            quote_id = document["quote_units"][0]["quote_id"]
            break
    if quote_id is None:
        raise AssertionError("no current-chapter quote units in the request")
    payload = {
        "entries": [
            {
                "local_key": "c1",
                "kind": "character",
                "semantic": {"class": "source_anchored_claim"},
                "claim": "test claim",
                "source_forms": ["x"],
                "source_names": ["x"],
                "anchor_refs": [quote_id],
            }
        ]
    }
    return RunnerResult(text=json.dumps(payload, ensure_ascii=False), finish_reason="stop")


def _context_generator() -> dict:
    return {
        "provider": "openai",
        "configured_model": "fake-model",
        "resolved_model_id": "fake-model",
        "weights_sha256": None,
        "quantization": None,
        "harness": "openai",
        "profile": "context-chapter_preflight-v1",
    }


def test_context_generation_freezes_exact_sparse_sample_snapshots(tmp_path):
    source = make_sparse_source(tmp_path, FROZEN_SAMPLE)
    artifact = generate_artifact(
        source,
        chapters=FROZEN_SAMPLE,
        horizon_mode="chapter_preflight",
        call=_fake_context_call,
        generator=_context_generator(),
        prompt="prompt",
        prompt_version="context-v2",
    )
    assert [s["target_chapter"] for s in artifact["snapshots"]] == FROZEN_SAMPLE


def test_context_generation_excludes_unselected_and_rejects_missing(tmp_path):
    source = make_sparse_source(tmp_path, [33, 34, 49, 63, 91, 100, 217])
    subset = generate_artifact(
        source,
        chapters=[33, 49],
        horizon_mode="chapter_preflight",
        call=_fake_context_call,
        generator=_context_generator(),
        prompt="prompt",
        prompt_version="context-v2",
    )
    assert [s["target_chapter"] for s in subset["snapshots"]] == [33, 49]
    with pytest.raises(ValueError, match="Chapters not found"):
        generate_artifact(
            source,
            chapters=[33, 999999],
            horizon_mode="chapter_preflight",
            call=_fake_context_call,
            generator=_context_generator(),
            prompt="prompt",
            prompt_version="context-v2",
        )


def test_sample_command_creates_a_sample_without_a_gate(project, source):
    result = invoke_cli(
        "sample", "--runner", "test", "--chapters", "1", "--config", project, "--json"
    )
    assert result["sample"] == {"role": "sample", "chapters": [1]}
    assert result["run_id"]


@pytest.mark.parametrize("command", ["run", "sample", "launch"])
def test_cli_unspecified_uses_practical_preset_but_direct_stays_raw(
    project, source, monkeypatch, command
):
    from test_presets import configure_practical_default

    from novel_harness import launch as launch_module
    from novel_harness.config import load_config
    from novel_harness.service import TranslationService

    configure_practical_default(project)
    monkeypatch.setattr(
        TranslationService,
        "execute",
        lambda self, run_id: {"run_id": run_id, "status": "completed"},
    )
    monkeypatch.setattr(
        launch_module,
        "launch_run",
        lambda service, config, run_id, **kwargs: {"run_id": run_id},
    )
    selection_option = ["--chapters", "1"] if command == "sample" else ["--chapter", "1"]
    cli = CliRunner()
    repo = TranslationService(load_config(project)).repository

    normal = cli.invoke(
        cli_app, [command, *selection_option, "--config", str(project), "--json"]
    )
    assert normal.exit_code == 0, normal.output
    manifest = repo.run(json.loads(normal.output)["run_id"])["manifest"]
    assert manifest["preset"]["preset_id"] == "local-fast"
    assert manifest["workflow"] == "draft-light"
    assert manifest["stages"][0]["runner"] == "test-b"

    explicit = cli.invoke(
        cli_app,
        [
            command,
            *selection_option,
            "--runner",
            "test-b",
            "--pipeline",
            "direct",
            "--config",
            str(project),
            "--json",
        ],
    )
    assert explicit.exit_code == 0, explicit.output
    manifest = repo.run(json.loads(explicit.output)["run_id"])["manifest"]
    assert manifest["workflow"] == "direct"
    assert manifest.get("preset") is None


def test_presets_command_lists_configured_modes(project):
    from test_presets import configure_presets, standard_presets

    configure_presets(project, standard_presets())
    result = invoke_cli("presets", "--config", project, "--json")
    assert result["default"] == "local-quality"
    entries = {entry["id"]: entry for entry in result["presets"]}
    assert entries["local-quality"]["available"] is True
    assert entries["cheap-draft"]["available"] is False
    assert entries["cheap-draft"]["reason"] == "configured runner missing"


def test_sample_command_with_preset(project, source):
    from test_presets import configure_presets, standard_presets

    configure_presets(project, standard_presets())
    result = invoke_cli(
        "sample", "--preset", "local-quality", "--chapters", "1", "--config", project, "--json"
    )
    assert result["run_id"]
    assert result["sample"] == {"role": "sample", "chapters": [1]}


def test_run_command_with_preset_against_mock_endpoint(project, source):
    from test_presets import configure_presets, standard_presets

    configure_presets(project, standard_presets())
    with endpoint() as (url, calls):
        data = yaml.safe_load(project.read_text("utf-8"))
        data["runners"]["test"]["base_url"] = url
        project.write_text(yaml.safe_dump(data), "utf-8")
        result = invoke_cli(
            "run", "--preset", "local-quality", "--chapter", "1", "--config", project, "--json"
        )
        assert result["status"] == "completed"
        assert calls


def test_preset_conflicting_cli_options_rejected(project, source):
    from test_presets import configure_presets, standard_presets

    configure_presets(project, standard_presets())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "novel_harness",
            "sample",
            "--preset",
            "local-quality",
            "--runner",
            "test",
            "--chapters",
            "1",
            "--config",
            str(project),
            "--json",
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert b"cannot be combined" in result.stderr
