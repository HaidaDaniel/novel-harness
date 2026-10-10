import copy
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from novel_harness.artifacts import sha256, write_text
from novel_harness.config import TranslationConfig, load_config
from novel_harness.service import TranslationService
from novel_harness.translation.names import normalize_names

RULES = [{"source": "アレン", "forms": {"Ален": "Аллен", "Алену": "Аллену"}, "review": ["Аллэн"]}]


def test_v21_names_keep_boundaries_case_forms_and_ambiguous_text():
    text = "Ален сказал Алену. Аленушка и Аллэн.\nАлен-сама."
    result, changes, review = normalize_names(text, "アレン", RULES)
    assert result == "Аллен сказал Аллену. Аленушка и Аллэн.\nАллен-сама."
    assert len(changes) == 3
    assert [x["text"] for x in review] == ["Аллэн"]
    assert normalize_names(result, "アレン", RULES)[0] == result
    unchanged, changes, review = normalize_names(text, "別の人", RULES)
    assert unchanged == text
    assert not changes
    assert any(x["reason"] == "source_anchor_missing" for x in review)


@pytest.mark.parametrize(
    "rules",
    [
        [{"source": "アレン", "forms": {"A": "B", "B": "C"}}],
        [{"source": "アレン", "forms": {"A": "B"}}, {"source": "別人", "forms": {"A": "C"}}],
        [{"source": "???", "forms": {"A": "B"}}],
    ],
)
def test_v21_conflicting_or_lossy_name_rules_rejected(rules):
    with pytest.raises(ValueError):
        TranslationConfig(name_rules=rules)


def test_v21_revision_preserves_artifacts_and_epub_uses_fixed_text(service):
    svc, fake = service
    svc.config.translation.name_rules = copy.deepcopy(RULES)
    # Match the actual fixture source; this test targets export, not translation quality.
    svc.config.translation.name_rules[0]["source"] = "彼"
    parent = svc.create_run(runner="test")
    svc.execute(parent, max_chunks=1)
    row = svc.repository.chunks(parent)[0]
    path = svc.run_dir(parent) / row["output_path"]
    text = "Глава 1\r\n\r\nАлен сказал Алену. Аллэн."
    write_text(path, text)
    svc.repository.set_chunk(
        parent,
        row["chapter"],
        row["chunk_id"],
        "completed",
        json.loads(row["metadata"]),
        row["output_path"],
        sha256(text),
    )
    before = {
        str(p): p.read_bytes() for p in (svc.run_dir(parent) / "chapters").rglob("*") if p.is_file()
    }
    calls = len(fake.calls)
    report = svc.revise_names(parent)
    assert not report["complete"]
    assert report["replacement_count"] == 2
    assert report["review_count"] == 1
    assert report["covered_chunks"] == 1
    assert (Path(report["directory"]) / "report.md").is_file()
    assert report["chunks"][0]["input_sha256"] == sha256(path.read_bytes())
    assert report == svc.revise_names(parent)
    assert len(fake.calls) == calls
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    with pytest.raises(ValueError, match="incomplete"):
        svc.build_epub(parent)
    svc.resume(parent)
    epub = svc.build_epub(parent)
    with zipfile.ZipFile(epub) as book:
        chapter = book.read("EPUB/chapter-0001.xhtml").decode()
    assert "Аллен сказал Аллену" in chapter
    assert path.read_bytes().decode("utf-8") == text
    metadata = json.loads((svc.run_dir(parent) / "book.json").read_text("utf-8"))
    assert metadata["name_revision"]["review_count"] == 1


def test_v21_continuation_freezes_new_rules(service):
    svc, _ = service
    parent = svc.create_run(runner="test")
    svc.config.translation.name_rules = RULES
    continued = svc.continue_run(parent, runner="test")
    assert svc.repository.run(continued)["manifest"]["translation"]["name_rules"] == RULES
    assert svc.repository.run(parent)["manifest"]["translation"]["name_rules"] == []


def test_v21_book_rules_preserve_ambiguous_cases():
    path = Path(__file__).resolve().parents[1] / "resources/example-book/name_rules.yaml"
    rules = yaml.safe_load(path.read_text("utf-8"))
    TranslationConfig(name_rules=rules)
    text, changes, review = normalize_names(
        "Ален подошёл к Наталии и Шарлотт.", "アレン ナタリア シャーロット", rules
    )
    assert text == "Аллен подошёл к Наталии и Шарлотт."
    assert len(changes) == 1
    assert [x["text"] for x in review] == ["Наталии", "Шарлотт"]


def test_v21_external_rules_resolve_and_freeze(project, source):
    data = yaml.safe_load(project.read_text("utf-8"))
    rules_path = project.parent / "names.yaml"
    rules_path.write_text(yaml.safe_dump(RULES, allow_unicode=True), "utf-8")
    data["translation"]["name_rules_path"] = "names.yaml"
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    config = load_config(project)
    assert config.translation.name_rules_path == rules_path
    svc = TranslationService(config)
    rid = svc.create_run(runner="test")
    rules_path.write_text("[]", "utf-8")
    assert load_config(project).translation.name_rules == []
    frozen = svc.repository.run(rid)["manifest"]["translation"]
    assert frozen["name_rules"] == RULES
    assert TranslationConfig.model_validate(frozen).name_rules == RULES
    data["translation"]["name_rules"] = RULES
    project.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    with pytest.raises(ValueError, match="not both"):
        load_config(project)


def test_v21_ru_ryu_replacement_only_with_source_anchor():
    path = Path(__file__).resolve().parents[1] / "resources/example-book/name_rules.yaml"
    rules = yaml.safe_load(path.read_text("utf-8"))
    TranslationConfig(name_rules=rules)
    text, changes, _ = normalize_names("Рю и Руу и Ру.", "ルゥ", rules)
    assert text == "Ру и Ру и Ру."
    assert len(changes) == 2
    unchanged, changes2, _ = normalize_names("Рю ушёл.", "別の人", rules)
    assert unchanged == "Рю ушёл."
    assert not changes2
