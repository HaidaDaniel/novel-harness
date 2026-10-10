from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from novel_harness.cli import app
from novel_harness.config import load_config
from novel_harness.service import TranslationService
from novel_harness.sources.narou_invocation import NarouInvocationError, invoke_narou_download

_FAKE_DOWNLOADER = r'''
import hashlib
import json
import os
import pathlib
import sys
import time

request_path = pathlib.Path(os.environ["NOVEL_HARNESS_NAROU_REQUEST"])
completion_path = pathlib.Path(os.environ["NOVEL_HARNESS_NAROU_COMPLETION"])
request = json.loads(request_path.read_text(encoding="utf-8"))
args = sys.argv[1:]
output_dir = pathlib.Path(args[args.index("--out") + 1])
assert str(output_dir) == request["output_dir"]
mode = os.environ.get("FAKE_NAROU_MODE", "complete")
if mode == "timeout":
    time.sleep(20)
if mode == "exit":
    print("fixture downloader failed", file=sys.stderr)
    raise SystemExit(4)

chapter_rows = []
numbers = request["expected_chapters"][:1] if mode == "partial" else request["expected_chapters"]
for number in numbers:
    filename = f"{number:04d}_episode-{number}.txt"
    content = f"作品名 第{number}話 第{number}話題\n\n本文{number}\n".encode("utf-8")
    (output_dir / filename).write_bytes(content)
    chapter_rows.append({
        "number": number,
        "filename": filename,
        "sha256": "0" * 64 if mode == "bad-hash" else hashlib.sha256(content).hexdigest(),
    })

if mode == "no-marker":
    raise SystemExit(0)
completion = {
    "version": 1,
    "request_sha256": "0" * 64 if mode == "stale" else request["request_sha256"],
    "status": "partial" if mode == "partial" else "complete",
    "source": request["source"],
    "output_dir": request["output_dir"],
    "expected_chapters": request["expected_chapters"],
    "chapters": chapter_rows,
    "diagnostics": ["chapter 2 returned 404"] if mode == "partial" else [],
}
completion_path.write_text(json.dumps(completion), encoding="utf-8")
'''


def _fake_command(path: Path) -> list[str]:
    path.write_text(_FAKE_DOWNLOADER, encoding="utf-8")
    return [sys.executable, str(path), "--url", "{url}", "--out", "{output_dir}"]


def _invoke(command, output_dir: Path, importer, *, expected=(1,), roots=(), timeout=5):
    return invoke_narou_download(
        command=command,
        ncode="n5727fr",
        source_url="https://ncode.syosetu.com/n5727fr/1/",
        title="作品名",
        expected_chapters=expected,
        output_dir=output_dir,
        timeout_seconds=timeout,
        library_roots=roots,
        import_completed=importer,
    )


def test_valid_completion_calls_task_0080_importer_once(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_NAROU_MODE", raising=False)
    command = _fake_command(tmp_path / "fake-downloader.py")
    output_dir = tmp_path / "downloads" / "n5727fr"
    calls = []

    result = _invoke(
        command,
        output_dir,
        lambda path, chapters: calls.append((path, chapters)) or {"status": "imported"},
    )

    assert result["protocol_version"] == 1
    assert result["process_exit_code"] == 0
    assert result["status"] == "imported"
    assert result["chapter_numbers"] == [1]
    assert len(calls) == 1
    assert calls[0] == (output_dir.resolve(), (1,))


@pytest.mark.parametrize(
    ("mode", "message", "expected", "timeout"),
    [
        ("exit", "exited with status 4", (1,), 5),
        ("timeout", "exceeded timeout", (1,), 0.1),
        ("no-marker", "fresh completion manifest", (1,), 5),
        ("stale", "does not match this request", (1,), 5),
        ("bad-hash", "hash differs from completion manifest", (1,), 5),
        ("partial", "reported 'partial'", (1, 2), 5),
    ],
)
def test_failed_or_partial_invocation_never_calls_importer(
    tmp_path, monkeypatch, mode, message, expected, timeout
):
    monkeypatch.setenv("FAKE_NAROU_MODE", mode)
    command = _fake_command(tmp_path / "fake-downloader.py")
    calls = []

    with pytest.raises(NarouInvocationError, match=message):
        _invoke(
            command,
            tmp_path / "downloads" / "n5727fr",
            lambda *args: calls.append(args) or {},
            expected=expected,
            timeout=timeout,
        )

    assert calls == []


def test_unsafe_or_stale_output_path_is_rejected_before_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_NAROU_MODE", raising=False)
    script = tmp_path / "should-not-run.py"
    command = _fake_command(script)
    library = tmp_path / "library"
    library.mkdir()
    calls = []

    with pytest.raises(NarouInvocationError, match="separate from configured Library"):
        _invoke(
            command,
            library / "n5727fr",
            lambda *args: calls.append(args) or {},
            roots=(library,),
        )
    assert calls == []

    stale_output = tmp_path / "stale-output"
    stale_output.mkdir()
    (stale_output / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(NarouInvocationError, match="must be empty"):
        _invoke(command, stale_output, lambda *args: calls.append(args) or {})
    assert calls == []


def test_service_import_is_idempotent_across_fresh_retry_directories(
    project, tmp_path, monkeypatch
):
    monkeypatch.delenv("FAKE_NAROU_MODE", raising=False)
    service = TranslationService(load_config(project))
    command = _fake_command(tmp_path / "fake-downloader.py")

    first = service.invoke_narou_downloader(
        command,
        ncode="n5727fr",
        source_url="https://ncode.syosetu.com/n5727fr/1/",
        title="作品名",
        expected_chapters=(1,),
        output_dir=tmp_path / "attempt-1",
    )
    second = service.invoke_narou_downloader(
        command,
        ncode="n5727fr",
        source_url="https://ncode.syosetu.com/n5727fr/1/",
        title="作品名",
        expected_chapters=(1,),
        output_dir=tmp_path / "attempt-2",
    )

    assert first["status"] == "imported"
    assert second["status"] == "already_present"
    assert second["import"]["novel_id"] == first["import"]["novel_id"]
    assert service.library_show(first["import"]["novel_id"])["status"] == "ready"


def test_cli_invocation_uses_protocol_and_json_argv(project, tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_NAROU_MODE", raising=False)
    script = tmp_path / "fake-downloader.py"
    command_json = json.dumps(_fake_command(script), ensure_ascii=False)
    output_dir = tmp_path / "cli-output" / "n5727fr"
    result = CliRunner().invoke(
        app,
        [
            "library",
            "invoke-narou",
            "--ncode",
            "n5727fr",
            "--url",
            "https://ncode.syosetu.com/n5727fr/1/",
            "--title",
            "作品名",
            "--expected-chapters",
            "1",
            "--output",
            str(output_dir),
            "--command-json",
            command_json,
            "--config",
            str(project),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["protocol_version"] == 1
    assert payload["import"]["status"] == "imported"


def test_service_rejects_url_for_another_ncode_before_import(tmp_path):
    calls = []
    with pytest.raises(NarouInvocationError, match="matching-ncode"):
        invoke_narou_download(
            command=[sys.executable, "unused.py", "{url}", "{output_dir}"],
            ncode="n5727fr",
            source_url="https://ncode.syosetu.com/n3289ds/1/",
            title="作品名",
            expected_chapters=(1,),
            output_dir=tmp_path / "output",
            timeout_seconds=2,
            library_roots=(),
            import_completed=lambda *args: calls.append(args) or {},
        )
    assert calls == []
