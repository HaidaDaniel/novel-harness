"""Source-budget estimation and its create_run integration (offline)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
import yaml

from novel_harness.budget import estimate_tokens, source_budget
from novel_harness.config import load_config
from novel_harness.service import TranslationService
from novel_harness.translation.chunker import chunk_text

ROOT = Path(__file__).resolve().parents[1]


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("", 2.0) == 0
    assert estimate_tokens("abcd", 2.0) == 2
    assert estimate_tokens("abcde", 2.0) == 3
    with pytest.raises(ValueError):
        estimate_tokens("x", 0)


def test_source_budget_subtracts_reserved_output_and_overhead():
    base = source_budget(
        context_limit=8192, reserved_output=2048, safety_margin=0.1, chars_per_token=2.0
    )
    assert base == (8192 - 2048 - math.ceil(8192 * 0.1)) * 2

    with_overhead = source_budget(
        context_limit=8192,
        reserved_output=2048,
        safety_margin=0.1,
        chars_per_token=2.0,
        overhead_texts=["x" * 4000],
    )
    assert with_overhead < base


def test_source_budget_fails_loudly_when_overhead_exceeds_context():
    with pytest.raises(ValueError):
        source_budget(
            context_limit=1024, reserved_output=2048, safety_margin=0.1, chars_per_token=2.0
        )
    with pytest.raises(ValueError):
        source_budget(
            context_limit=2048,
            reserved_output=512,
            safety_margin=0.1,
            chars_per_token=2.0,
            overhead_texts=["x" * 100000],
        )


def test_source_budget_minimum_chunk_boundary():
    # usable = 256 - 240 = 16 tokens -> 32 chars, exactly the minimum.
    assert (
        source_budget(
            context_limit=256, reserved_output=240, safety_margin=0, chars_per_token=2.0
        )
        == 32
    )
    # One extra overhead token drops below the minimum and must fail loudly.
    with pytest.raises(ValueError, match="minimum supported chunk"):
        source_budget(
            context_limit=256,
            reserved_output=240,
            safety_margin=0,
            chars_per_token=2.0,
            overhead_texts=["aa"],
        )


def test_source_budget_caps_result_at_max_chars():
    assert (
        source_budget(
            context_limit=163840,
            reserved_output=1024,
            safety_margin=0,
            chars_per_token=2.0,
            max_chars=100000,
        )
        == 100000
    )
    # A result already below the cap is returned unchanged.
    assert (
        source_budget(
            context_limit=8192,
            reserved_output=2048,
            safety_margin=0,
            chars_per_token=2.0,
            max_chars=100000,
        )
        == (8192 - 2048) * 2
    )


def test_derived_budget_keeps_chunking_lossless():
    source = "第一章\n\n" + "彼は森を歩いていた。\n\n" * 40
    budget = source_budget(
        context_limit=4096, reserved_output=512, safety_margin=0.1, chars_per_token=2.0
    )
    chunks = chunk_text(source, budget)
    assert "".join(chunk.text for chunk in chunks) == source


def _make_project(tmp_path, *, context_limit, max_tokens, chunk_chars=6000):
    (tmp_path / "original").mkdir()
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["translation"].update(
        chunk_chars=chunk_chars,
        overlap_chars=20,
        retries=0,
        retry_delay_seconds=0,
        max_tokens=max_tokens,
        context_limit=context_limit,
    )
    data["runners"] = {
        "test": {
            "type": "openai",
            "model": "fake-model",
            "api_key": "EMPTY",
            "base_url": "http://127.0.0.1:9999/v1",
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    (tmp_path / "original" / "0001_first.txt").write_text(
        "第一章\n\n" + "彼は森を歩いていた。\n\n" * 14, "utf-8"
    )
    return path


def test_create_run_derives_budget_and_respects_explicit_override(tmp_path):
    path = _make_project(tmp_path, context_limit=16384, max_tokens=1024)
    service = TranslationService(load_config(path))

    derived_id = service.create_run(runner="test")
    derived = service.repository.run(derived_id)["manifest"]["translation"]["chunk_chars"]
    usable_chars = int((16384 - 1024 - math.ceil(16384 * 0.15)) * 2.0)
    assert 0 < derived < usable_chars
    assert derived != 6000

    override_id = service.create_run(runner="test", chunk_chars=1500)
    override = service.repository.run(override_id)["manifest"]["translation"]["chunk_chars"]
    assert override == 1500


def test_create_run_fails_when_overhead_exceeds_context(tmp_path):
    path = _make_project(tmp_path, context_limit=1024, max_tokens=512)
    service = TranslationService(load_config(path))

    with pytest.raises(ValueError):
        service.create_run(runner="test")


def test_derived_budget_respects_config_upper_bound_and_validates(tmp_path):
    path = _make_project(tmp_path, context_limit=163840, max_tokens=1024)
    service = TranslationService(load_config(path))

    run_id = service.create_run(runner="test")
    manifest = service.repository.run(run_id)["manifest"]
    assert manifest["translation"]["chunk_chars"] == 100000
    service._verify_inputs(manifest)  # the persisted derived manifest must validate


def test_create_run_fails_before_creation_when_capacity_is_insufficient(tmp_path):
    path = _make_project(tmp_path, context_limit=256, max_tokens=240)
    service = TranslationService(load_config(path))

    with pytest.raises(ValueError):
        service.create_run(runner="test")
    assert service.repository.list_runs() == []
