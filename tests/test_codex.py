import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import FakeRunner
from mock_endpoint import endpoint

from novel_harness.config import RunnerConfig, load_config
from novel_harness.runners import make_runner
from novel_harness.runners.base import Runner, RunnerExecutionError, TranslationJob
from novel_harness.runners.codex import CodexRunner
from novel_harness.service import TranslationService


@pytest.fixture
def codex_job(tmp_path):
    workspace = tmp_path / "artifacts" / "attempt-0001"
    workspace.mkdir(parents=True)
    return TranslationJob(
        "test-run",
        1,
        "0001-abcd",
        "translation",
        "Translate into Russian.",
        "原文。 $(malicious) `command`",
        {},
        {},
        workspace,
    )


def fake_cli(
    monkeypatch, *, code=0, write_output=True, usage=True, timeout=False, event_override=None
):
    captured = {}
    monkeypatch.setattr("novel_harness.runners.codex.shutil.which", lambda _: "codex.exe")

    class Process:
        pid = 999
        returncode = code

        def __init__(self, args, **kwargs):
            captured.update(args=args, kwargs=kwargs)

        def communicate(self, data, timeout):
            captured["stdin"] = data
            if captured.get("timeout_requested"):
                raise subprocess.TimeoutExpired("codex", timeout)
            kwargs = captured["kwargs"]
            event = {"type": "turn.completed"}
            if usage:
                event["usage"] = {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 20}
            if event_override is not None:
                event = event_override
            events = event if isinstance(event, list) else [event]
            kwargs["stdout"].write(("\n".join(json.dumps(e) for e in events) + "\n").encode())
            kwargs["stderr"].write(b"CLI diagnostic\n")
            if write_output:
                output = Path(captured["args"][captured["args"].index("--output-last-message") + 1])
                output.write_text("Полный русский перевод.", encoding="utf-8")

        def poll(self):
            return self.returncode

    monkeypatch.setattr("novel_harness.runners.codex.subprocess.Popen", Process)
    captured["timeout_requested"] = timeout
    return captured


def test_v15_codex_bounded_argv_files_usage(codex_job, tmp_path, monkeypatch):
    captured = fake_cli(monkeypatch)
    runner = CodexRunner(
        RunnerConfig(type="codex", model="gpt-5.6-luna", reasoning_effort="xhigh"),
        tmp_path / "jobs",
    )
    assert isinstance(runner, Runner)
    result = runner.run(codex_job)
    args = captured["args"]
    assert args[args.index("--model") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="xhigh"' in args
    assert "--ignore-user-config" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "shell_tool" in args and "multi_agent" in args
    assert captured["kwargs"]["shell"] is False
    assert codex_job.source not in " ".join(args)
    assert json.loads(captured["stdin"].decode("utf-8"))["source"] == codex_job.source
    assert (captured["kwargs"]["cwd"] / "source.txt").read_text("utf-8") == codex_job.source
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "cached_input_tokens": 2}
    assert result.finish_reason is None
    assert (codex_job.workspace / "codex.stderr.log").read_text() == "CLI diagnostic\n"
    assert (codex_job.workspace / "codex-invocation.json").is_file()


def test_codex_missing_usage_is_null(codex_job, tmp_path, monkeypatch):
    fake_cli(monkeypatch, usage=False)
    result = CodexRunner(RunnerConfig(type="codex"), tmp_path / "jobs").run(codex_job)
    assert result.usage == {"prompt_tokens": None, "completion_tokens": None}


def test_v23_codex_passes_schema_without_duplicate_source(codex_job, tmp_path, monkeypatch):
    from dataclasses import replace

    from novel_harness.translation.segments import source_segments, translation_schema

    captured = fake_cli(monkeypatch)
    segments = source_segments(codex_job.source)
    schema = translation_schema(segments)
    job = replace(codex_job, segments=segments, response_schema=schema)
    CodexRunner(RunnerConfig(type="codex"), tmp_path / "jobs").run(job)
    args = captured["args"]
    path = Path(args[args.index("--output-schema") + 1])
    assert json.loads(path.read_text("utf-8")) == schema
    payload = json.loads(captured["stdin"].decode("utf-8"))
    assert payload["segments"] == segments and "source" not in payload


@pytest.mark.parametrize("code,write_output", [(1, True), (0, False)])
def test_codex_rejects_exit_error_and_missing_output(
    codex_job, tmp_path, monkeypatch, code, write_output
):
    fake_cli(monkeypatch, code=code, write_output=write_output)
    with pytest.raises(RunnerExecutionError):
        CodexRunner(RunnerConfig(type="codex"), tmp_path / "jobs").run(codex_job)
    assert (codex_job.workspace / "codex.stderr.log").is_file()


def test_v26_codex_quota_parsed_even_when_cli_exits_nonzero(codex_job, tmp_path, monkeypatch):
    from novel_harness.runners.base import RunnerDeferred

    fake_cli(
        monkeypatch,
        code=1,
        write_output=False,
        event_override={
            "type": "turn.failed",
            "error": {"message": "Usage limit; try again in 30 minutes"},
        },
    )
    with pytest.raises(RunnerDeferred) as caught:
        CodexRunner(RunnerConfig(type="codex"), tmp_path / "jobs").run(codex_job)
    assert caught.value.deadline_source == "provider_relative"


def test_v27_codex_reasoning_and_known_diagnostic(codex_job, tmp_path, monkeypatch):
    fake_cli(
        monkeypatch,
        event_override=[
            {
                "type": "item.completed",
                "item": {
                    "type": "error",
                    "message": "Code Mode is unavailable because code-mode host is disabled. Enable host.",
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 18,
                    "cached_input_tokens": 2,
                },
            },
        ],
    )
    result = CodexRunner(RunnerConfig(type="codex"), tmp_path / "jobs").run(codex_job)
    assert result.usage["reasoning_tokens"] == 18
    assert result.usage["visible_output_tokens"] == 2
    assert result.raw_response["diagnostics"] == ["code_mode_host_disabled"]
    assert result.finish_reason is None
    assert result.effective_settings["unsupported_job_parameters"] == ["temperature", "max_tokens"]


@pytest.mark.parametrize(
    "event",
    [
        {"type": "turn.failed", "error": {"message": "fatal"}},
        {"type": "item.completed", "item": {"type": "command_execution"}},
    ],
)
def test_v27_codex_rejects_failed_turn_or_tool_even_with_output(
    codex_job, tmp_path, monkeypatch, event
):
    fake_cli(monkeypatch, event_override=[event, {"type": "turn.completed"}])
    with pytest.raises(RunnerExecutionError):
        CodexRunner(RunnerConfig(type="codex"), tmp_path / "jobs").run(codex_job)


def test_v15_timeout_stops_process_tree(codex_job, tmp_path, monkeypatch):
    fake_cli(monkeypatch, timeout=True)
    stopped = []
    monkeypatch.setattr(
        "novel_harness.runners.codex.stop_process_tree", lambda p: stopped.append(p.pid)
    )
    with pytest.raises(RunnerExecutionError, match="timeout"):
        CodexRunner(RunnerConfig(type="codex"), tmp_path / "jobs").run(codex_job)
    assert stopped == [999]


def test_codex_registered_without_api_key(project, monkeypatch):
    cfg = load_config(project)
    cfg.runners["luna"] = RunnerConfig(type="codex", model="gpt-5.6-luna", reasoning_effort="xhigh")
    monkeypatch.setattr("novel_harness.runners.shutil.which", lambda _: "codex.exe")
    assert isinstance(make_runner(cfg, "luna"), CodexRunner)


def test_v16_pause_after_one_then_resume(service):
    svc, fake = service
    run_id = svc.create_run(runner="test")
    result = svc.execute(run_id, max_chunks=1)
    assert result["status"] == "paused"
    assert result["chunks"]["completed"] == 1
    first_chunk = fake.calls[0].chunk_id
    assert svc.resume(run_id)["status"] == "completed"
    assert sum(j.chunk_id == first_chunk for j in fake.calls) == 1


def test_v16_pause_request_during_chunk(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    fake = FakeRunner()

    class PausingRunner:
        def run(self, job):
            svc.pause(run_id)
            return fake.run(job)

    svc.runner_factory = lambda _: PausingRunner()
    result = svc.execute(run_id)
    assert result["status"] == "paused"
    assert result["chunks"]["completed"] == 1


def test_v16_process_failure_aborts_run_after_retries(service):
    svc, _ = service
    svc.config.translation.retries = 1
    calls = []

    class BrokenRunner:
        def run(self, job):
            calls.append(job)
            raise RunnerExecutionError("CLI unavailable")

    svc.runner_factory = lambda _: BrokenRunner()
    run_id = svc.create_run(runner="test")
    with pytest.raises(RunnerExecutionError, match="CLI unavailable"):
        svc.execute(run_id)
    assert len(calls) == 2
    assert svc.status(run_id)["status"] == "failed"
    assert svc.status(run_id)["chunks"]["failed"] == 1


def test_worker_executes_and_builds_epub(project, source):
    with endpoint() as (url, calls):
        data = yaml.safe_load(project.read_text("utf-8"))
        data["runners"]["test"]["base_url"] = url
        project.write_text(yaml.safe_dump(data), "utf-8")
        svc = TranslationService.from_config(project)
        run_id = svc.create_run(runner="test")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "novel_harness.worker",
                "--config",
                str(project),
                "--run",
                run_id,
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8")
        assert svc.status(run_id)["status"] == "completed"
        book = json.loads((svc.run_dir(run_id) / "book.json").read_text("utf-8"))
        assert Path(book["path"]).is_file()
        assert " - openai - fake-model.epub" in Path(book["path"]).name
        assert json.loads((svc.run_dir(run_id) / "worker.json").read_text("utf-8"))["finished_at"]
        assert len(calls) == svc.status(run_id)["total_chunks"]
