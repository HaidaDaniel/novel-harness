"""Cross-run content-addressed cache: round-trip, corruption and integration."""

from __future__ import annotations

import json
import multiprocessing
import shutil
from pathlib import Path

import pytest
import yaml

from novel_harness import cache as cache_module
from novel_harness.cache import TranslationCache, TranslationMemory
from novel_harness.config import load_config
from novel_harness.runners.base import RunnerResult
from novel_harness.service import TranslationService

ROOT = Path(__file__).resolve().parents[1]
KEY = "a" * 64


def test_cache_round_trip_and_no_overwrite(tmp_path):
    cache = TranslationCache(tmp_path / "cache")
    cache.put(
        KEY, text="hello", model="m", finish_reason="stop", usage={}, effective_settings={}
    )
    assert cache.get(KEY)["text"] == "hello"

    # A valid existing entry is never overwritten.
    cache.put(
        KEY, text="changed", model="m", finish_reason="stop", usage={}, effective_settings={}
    )
    assert cache.get(KEY)["text"] == "hello"


def test_cache_treats_corrupt_and_mismatched_entries_as_misses(tmp_path):
    cache = TranslationCache(tmp_path / "cache")
    assert cache.get(KEY) is None

    path = cache._path(KEY)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("{not json", encoding="utf-8")
    assert cache.get(KEY) is None

    path.write_text(
        json.dumps({"schema_version": 1, "text": "x", "output_sha256": "deadbeef"}),
        encoding="utf-8",
    )
    assert cache.get(KEY) is None

    path.write_text(
        json.dumps({"schema_version": 99, "text": "x", "output_sha256": "deadbeef"}),
        encoding="utf-8",
    )
    assert cache.get(KEY) is None


def test_cache_rejects_empty_text(tmp_path):
    cache = TranslationCache(tmp_path / "cache")
    cache.put(KEY, text="", model=None, finish_reason=None, usage={}, effective_settings={})
    assert cache.get(KEY) is None


class CountingRunner:
    def __init__(self):
        self.calls = 0

    def run(self, job):
        self.calls += 1
        text = f"Глава {job.chapter}\n\n" + "Русский текст для проверки. " * max(
            1, len(job.source) // 20
        )
        return RunnerResult(text, model="fake-model", finish_reason="stop")


def _make_project(tmp_path):
    (tmp_path / "original").mkdir()
    shutil.copytree(ROOT / "prompts", tmp_path / "prompts")
    shutil.copy(ROOT / "glossary.yaml", tmp_path / "glossary.yaml")
    data = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    data["translation"].update(
        chunk_chars=200,
        overlap_chars=0,
        retries=0,
        retry_delay_seconds=0,
        cache_enabled=True,
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


def _service(path):
    runner = CountingRunner()
    return TranslationService(load_config(path), lambda name: runner), runner


def test_cache_reuses_validated_output_across_runs(tmp_path):
    path = _make_project(tmp_path)
    service, runner = _service(path)

    first = service.create_run(runner="test")
    service.execute(first)
    assert runner.calls > 0
    calls_after_first = runner.calls

    second = service.create_run(runner="test", force=True)
    assert second != first
    service.execute(second)

    assert runner.calls == calls_after_first
    assert service.status(second)["status"] == "completed"
    rows = service.repository.chunks(second)
    assert any('"cache":"hit"' in (row["metadata"] or "") for row in rows)


def test_cache_key_isolates_model_change(tmp_path):
    path = _make_project(tmp_path)
    service, runner = _service(path)

    first = service.create_run(runner="test")
    service.execute(first)
    calls_after_first = runner.calls

    data = yaml.safe_load(path.read_text("utf-8"))
    data["runners"]["test"]["model"] = "other-model"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")

    service2, runner2 = _service(path)
    second = service2.create_run(runner="test")
    service2.execute(second)

    assert runner2.calls > 0
    assert calls_after_first > 0


def test_semantic_cache_key_changes_with_each_semantic_input():
    from novel_harness.cache import semantic_cache_key

    base = {
        "name_rules": [],
        "book_bible_sha256": "b",
        "glossary_sha256": "g",
        "prompt_sha256": "p",
        "runner_config": {"model": "m"},
        "decoding": {"temperature": 0.2},
    }
    key = semantic_cache_key("input", base)
    assert key == semantic_cache_key("input", dict(base))
    assert semantic_cache_key("other-input", base) != key
    for field, value in (
        ("name_rules", [{"source": "x"}]),
        ("book_bible_sha256", "b2"),
        ("glossary_sha256", "g2"),
        ("prompt_sha256", "p2"),
        ("runner_config", {"model": "m2"}),
        ("decoding", {"temperature": 0.9}),
    ):
        assert semantic_cache_key("input", {**base, field: value}) != key, field


def test_translation_memory_provenance_and_no_overwrite(tmp_path):
    from novel_harness.cache import TranslationMemory

    tm = TranslationMemory(tmp_path / "tm")
    key = "a" * 64
    assert tm.get(key) is None

    assert tm.put(key, text="Привет", provenance={"actor": "human"}) is True
    entry = tm.get(key)
    assert entry["text"] == "Привет"
    assert entry["provenance"]["actor"] == "human"

    assert tm.put(key, text="Другое", provenance={"actor": "x"}) is False
    assert tm.get(key)["text"] == "Привет"
    assert len(tm.entries()) == 1


def test_translation_memory_rejects_corrupt_entry(tmp_path):
    from novel_harness.cache import TranslationMemory

    tm = TranslationMemory(tmp_path / "tm")
    key = "b" * 64
    path = tm._path(key)
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"text":"x","output_sha256":"bad"}', "utf-8")
    assert tm.get(key) is None


def test_translation_memory_put_blocks_while_entry_lock_is_held(tmp_path):
    import threading

    from filelock import FileLock

    tm = TranslationMemory(tmp_path / "tm")
    key = "f" * 64
    path = tm._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)

    started = threading.Event()
    done = threading.Event()

    def worker():
        started.set()
        tm.put(key, text="Первый финал", provenance={"actor": "a"})
        done.set()

    held = FileLock(str(path) + ".lock")
    held.acquire()
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert started.wait(timeout=5)
        # While the entry lock is held the writer cannot publish.
        assert not done.is_set()
    finally:
        held.release()
    assert done.wait(timeout=5)
    thread.join(timeout=5)

    winner = tm.get(key)
    assert winner["text"] == "Первый финал"
    assert winner["provenance"] == {"actor": "a"}
    assert tm.put(key, text="Второй финал", provenance={"actor": "b"}) is False
    # The winning entry and provenance stay byte-identical.
    assert tm.get(key) == winner


def test_translation_memory_repairs_corrupt_entry_on_put(tmp_path):
    tm = TranslationMemory(tmp_path / "tm")
    key = "g" * 64
    path = tm._path(key)
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"text":"x","output_sha256":"bad"}', "utf-8")
    assert tm.get(key) is None

    assert tm.put(key, text="Восстановлено", provenance={"actor": "human"}) is True

    assert tm.get(key)["text"] == "Восстановлено"


def test_translation_cache_put_blocks_while_entry_lock_is_held(tmp_path, monkeypatch):
    import threading

    from filelock import FileLock

    cache = TranslationCache(tmp_path / "cache")
    published = []
    original = cache_module._publish

    def counting(path, entry):
        published.append(entry["text"])
        original(path, entry)

    monkeypatch.setattr(cache_module, "_publish", counting)
    path = cache._path(KEY)
    path.parent.mkdir(parents=True, exist_ok=True)

    started = threading.Event()
    done = threading.Event()

    def worker():
        started.set()
        cache.put(
            KEY, text="one", model="m", finish_reason="stop",
            usage={}, effective_settings={},
        )
        done.set()

    held = FileLock(str(path) + ".lock")
    held.acquire()
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert started.wait(timeout=5)
        assert not done.is_set()
        assert published == []
    finally:
        held.release()
    assert done.wait(timeout=5)
    thread.join(timeout=5)

    assert published == ["one"]
    cache.put(
        KEY, text="two", model="m", finish_reason="stop", usage={}, effective_settings={}
    )
    assert cache.get(KEY)["text"] == "one"
    assert published == ["one"]


def test_cache_repairs_corrupt_entry_on_put(tmp_path):
    cache = TranslationCache(tmp_path / "cache")
    path = cache._path(KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version":1,"text":"x","output_sha256":"bad"}', "utf-8")

    cache.put(KEY, text="recovered", model="m", finish_reason="stop", usage={}, effective_settings={})

    assert cache.get(KEY)["text"] == "recovered"


def _process_tm_put(root: str, key: str, text: str, queue) -> None:
    from pathlib import Path as _Path

    from novel_harness.cache import TranslationMemory as _TranslationMemory

    queue.put(_TranslationMemory(_Path(root)).put(key, text=text, provenance={"actor": text}))


def test_translation_memory_cross_process_insert_is_insert_only(tmp_path):
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX platforms
        pytest.skip("fork start method is unavailable")

    root = tmp_path / "tm"
    key = "d" * 64
    queue = context.Queue()
    processes = [
        context.Process(target=_process_tm_put, args=(str(root), key, text, queue))
        for text in ("Первый финал", "Второй финал")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    results = [queue.get(timeout=5) for _ in processes]
    assert results.count(True) == 1
    assert results.count(False) == 1
    assert len(TranslationMemory(root).entries()) == 1
