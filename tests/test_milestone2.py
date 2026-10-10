import io
import json
import subprocess
import sys
import zipfile
from xml.etree import ElementTree as ET

import pytest
import yaml
from conftest import FakeRunner
from mock_endpoint import endpoint
from test_cli import invoke_cli

from novel_harness.config import load_config
from novel_harness.service import TranslationService


def test_epub_numeric_order_metadata_and_byte_determinism(project, source):
    for n in (10, 2):
        (source.parent / f"{n}_chapter.txt").write_text(f"第{n}章\n\n本文です。", "utf-8")
    cfg = load_config(project)
    cfg.novel.author = "Test Author"
    svc = TranslationService(cfg, lambda name: FakeRunner())
    run_id = svc.create_run(runner="test")
    svc.execute(run_id)
    path = svc.build_epub(run_id)
    first = path.read_bytes()
    assert svc.build_epub(run_id).read_bytes() == first
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist()[0] == "mimetype"
        assert archive.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        assert archive.read("mimetype") == b"application/epub+zip"
        opf = ET.fromstring(archive.read("EPUB/content.opf"))
        ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
        spine = opf.find("opf:spine", ns)
        assert [item.attrib["idref"] for item in spine] == [
            "nav",
            "chapter-1",
            "chapter-2",
            "chapter-10",
        ]
        assert opf.find("opf:metadata/dc:language", ns).text == "ru"
        assert opf.find("opf:metadata/dc:creator", ns).text == "Test Author"
        toc = archive.read("EPUB/nav.xhtml").decode("utf-8")
        assert toc.index("chapter-0001") < toc.index("chapter-0002") < toc.index("chapter-0010")
        for name in archive.namelist():
            if name.endswith((".xhtml", ".opf", ".ncx", ".xml")):
                ET.fromstring(archive.read(name))


def test_epub_rejects_incomplete_and_corrupt(service):
    svc, _ = service
    run_id = svc.create_run(runner="test")
    with pytest.raises(ValueError, match="incomplete"):
        svc.build_epub(run_id)
    svc.execute(run_id)
    row = svc.repository.chunks(run_id)[0]
    (svc.run_dir(run_id) / row["output_path"]).write_text("broken", "utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        svc.build_epub(run_id)


def test_html_compare_escapes_text_and_checks_sources(service, source):
    svc, _ = service
    source.write_text("<script>alert('x')</script>\n\n原文。", "utf-8")
    a = svc.create_run(runner="test")
    b = svc.create_run(runner="test", force=True)
    svc.execute(a)
    svc.execute(b)
    html = svc.compare_html(1, [a, b]).read_text("utf-8")
    assert html.count("<section>") == 3
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert a in html and b in html
    source.write_text("違う原文。", "utf-8")
    changed = svc.create_run(runner="test")
    with pytest.raises(ValueError, match="different source"):
        svc.compare(1, [a, changed])


def test_http_retry_budget_reason_usage_and_cli_build_compare(project, source):
    with endpoint(fail_first=True) as (url, calls):
        data = yaml.safe_load(project.read_text("utf-8"))
        data["runners"]["test"]["base_url"] = url
        data["translation"]["retries"] = 1
        project.write_text(yaml.safe_dump(data), "utf-8")
        result = invoke_cli(
            "run", "--runner", "test", "--range", "1-1", "--config", project, "--json"
        )
        assert result["status"] == "completed"
        svc = TranslationService.from_config(project)
        row = svc.repository.chunks(result["run_id"])[0]
        attempts = svc.repository.attempts(result["run_id"], 1, row["chunk_id"], "translation")
        assert len(attempts) == 2
        assert attempts[1]["retry_reason"] == "InternalServerError (HTTP 503)"
        assert attempts[1]["usage"] == {"prompt_tokens": 17, "completion_tokens": 23}
        assert len(calls) == result["total_chunks"] + 1
        second = invoke_cli(
            "run", "--runner", "test", "--all", "--force", "--config", project, "--json"
        )
        assert second["run_id"] != result["run_id"]
        book = invoke_cli("build", "--run", result["run_id"], "--config", project, "--json")
        assert book["path"].endswith(".epub")
        compare = invoke_cli(
            "compare",
            "--chapter",
            "1",
            "--runs",
            result["run_id"] + "," + second["run_id"],
            "--config",
            project,
            "--json",
        )
        assert compare["path"].endswith(".html")


def test_hard_process_death_releases_lock_and_resumes(project, source):
    cfg = load_config(project)
    svc = TranslationService(cfg)
    run_id = svc.create_run(runner="test")
    # No finally handlers, no clean SQLite/lock teardown: simulate abrupt process death.
    script = """
import os, sys
from novel_harness.config import load_config
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService
class CrashRunner:
    calls = 0
    def run(self, job):
        self.calls += 1
        if self.calls == 2:
            os._exit(73)
        return RunnerResult('Russian translation for recovery test. ' * 3, finish_reason='stop')
runner = CrashRunner()
TranslationService(load_config(sys.argv[1]), lambda name: runner).execute(sys.argv[2])
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(project), run_id],
        check=False,
        capture_output=True,
        timeout=20,
    )
    assert child.returncode == 73, child.stderr.decode("utf-8", errors="replace")
    assert svc.status(run_id)["status"] == "interrupted"
    assert svc.status(run_id)["chunks"]["completed"] == 1
    fake = FakeRunner()
    restarted = TranslationService(cfg, lambda name: fake)
    assert restarted.resume(run_id)["status"] == "completed"
    assert len(fake.calls) == restarted.status(run_id)["total_chunks"] - 1


def test_failure_budget_and_retry_preserve_success(project, source):
    cfg = load_config(project)
    fake = FakeRunner(fail_at=2)
    svc = TranslationService(cfg, lambda name: fake)
    run_id = svc.create_run(runner="test")
    assert svc.execute(run_id)["status"] == "failed"
    count = len(fake.calls)
    first = fake.calls[0].chunk_id
    assert svc.retry_failed(run_id)["status"] == "completed"
    assert all(job.chunk_id != first for job in fake.calls[count:])


def test_cli_selections_and_unavailable_runners(project, source):
    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "novel_harness",
            "run",
            "--runner",
            "test",
            "--all",
            "--chapter",
            "1",
            "--config",
            str(project),
            "--json",
        ],
        check=False,
        capture_output=True,
        timeout=20,
    )
    assert bad.returncode == 2
    assert "exactly one" in json.loads(bad.stderr.decode("utf-8"))["error"]
    cfg = load_config(project)
    cfg.runners["test"].type = "pi"
    svc = TranslationService(cfg)
    assert not svc.runners()[0]["available"]
