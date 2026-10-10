import shutil
from pathlib import Path

import pytest
import yaml

from novel_harness.config import load_config
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService

ROOT = Path(__file__).resolve().parents[1]


class FakeRunner:
    def __init__(self, fail_at=None, interrupt_at=None):
        self.calls = []
        self.fail_at = fail_at
        self.interrupt_at = interrupt_at

    def run(self, job):
        self.calls.append(job)
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt()
        if len(self.calls) == self.fail_at:
            return RunnerResult("", finish_reason="length")
        # Synthetic output deliberately scales to source; never claims translation quality.
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


@pytest.fixture
def project(tmp_path):
    (tmp_path / "original").mkdir()
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    shutil.copytree(ROOT / "resources/example-book", tmp_path / "resources/example-book")
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["novel"]["source_dir"] = "./original"
    data["translation"]["glossary_path"] = "./glossary.yaml"
    data["translation"].pop("book_context_path", None)
    data["translation"].pop("name_rules_path", None)
    data["translation"].pop("output_format", None)
    for key in (
        "max_foreign_script_chars",
        "reject_malformed_dialogue",
        "reject_service_artifacts",
        "reject_segment_collapse",
        "reject_missing_cyrillic",
    ):
        data["validation"].pop(key, None)
    data["translation"].update(chunk_chars=80, overlap_chars=20, retries=0, retry_delay_seconds=0)
    data["runners"] = {
        "test": {
            "type": "openai",
            "model": "fake-model",
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:9999/v1",
        }
    }
    data["translation_presets"] = {
        "default": "local-fast",
        "auto_select_default": True,
        "presets": {
            "quality": {"label": "Quality", "runner": "test", "pipeline": "draft-light", "category": "quality", "cloud": True},
            "local-quality": {"label": "Local Quality", "runner": "test", "pipeline": "draft-light", "category": "local-quality"},
            "local-pr": {"label": "Local PR", "runner": "test", "pipeline": "draft-light", "category": "local-quality"},
            "local-t-xhigh": {"label": "Local T-xhigh", "runner": "test", "pipeline": "draft-light", "category": "local-quality"},
            "local-fast": {"label": "Local Fast", "runner": "test", "pipeline": "draft-light", "category": "local-fast", "recommended": True},
            "cheap-draft": {"label": "Cheap Draft", "runner": "missing-9b", "pipeline": "draft-light", "category": "cheap-draft"},
        },
    }
    data["pipelines"].update(
        {
            "translate-edit": {
                "stages": [
                    {"name": "translation", "role": "translator", "prompt_version": "translate-v1"},
                    {"name": "editing", "role": "editor", "prompt_version": "edit-v1"},
                ]
            },
            "translate-review": {
                "stages": [
                    {"name": "translation", "role": "translator", "prompt_version": "translate-v1"},
                    {"name": "review", "role": "reviewer", "prompt_version": "review-v1"},
                ]
            },
        }
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    return path


@pytest.fixture
def source(project):
    path = project.parent / "original" / "0001_first.txt"
    path.write_text("第一章\n\n" + "彼は森を歩いていた。\n\n" * 14, "utf-8")
    return path


@pytest.fixture
def service(project, source):
    fake = FakeRunner()
    return TranslationService(load_config(project), lambda name: fake), fake
