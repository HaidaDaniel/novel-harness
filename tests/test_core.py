import pytest
import yaml
from pydantic import ValidationError

from novel_harness.artifacts import sha256
from novel_harness.config import RunnerConfig, ValidationConfig, load_config
from novel_harness.translation.assembly import assemble
from novel_harness.translation.chunker import chunk_text
from novel_harness.translation.context import build_context
from novel_harness.translation.discovery import discover
from novel_harness.translation.validator import validate, validate_file


def test_discovery_numeric_sort_hash_and_read_only(tmp_path):
    for n in (10, 2, 1):
        (tmp_path / f"{n}_chapter.txt").write_bytes("題名\r\n\r\n本文。".encode())
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    chapters = discover(tmp_path)
    assert [c.number for c in chapters] == [1, 2, 10]
    assert chapters[0].source_sha256 == sha256(before["1_chapter.txt"])
    assert "\r" not in chapters[0].text
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    (tmp_path / "0001_duplicate.txt").write_text("本文", "utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        discover(tmp_path)


def test_discovery_invalid_utf8(tmp_path):
    (tmp_path / "1_bad.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        discover(tmp_path)


def test_discovery_allows_literal_replacement_marker(tmp_path):
    (tmp_path / "1_literal_marker.txt").write_text(
        "題名\n\n名前を『譁�ュ怜喧縺�』と文字化けさせた。", "utf-8"
    )
    chapters = discover(tmp_path)
    assert chapters[0].number == 1
    assert "\ufffd" in chapters[0].text


@pytest.mark.parametrize("raw", [b"", "題名\n\n本文\x00。".encode()])
def test_discovery_rejects_empty_or_nul_chapters(tmp_path, raw):
    (tmp_path / "1_invalid.txt").write_bytes(raw)
    with pytest.raises(ValueError, match="Empty chapter or broken encoding"):
        discover(tmp_path)


@pytest.mark.parametrize("text", ["", "あ" * 201, "一文。二文。\n\n" * 30, "「はい！」\n" * 20])
def test_chunks_lossless_stable_bounded(text):
    chunks = chunk_text(text, 32)
    assert "".join(c.text for c in chunks) == text
    assert chunks == chunk_text(text, 32)
    assert all(0 < len(c.text) <= 32 for c in chunks)
    assert all(c.source_sha256 == sha256(c.text) for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_boundary_and_hash_changes():
    text = "あ" * 15 + "\n\n" + "い" * 25
    assert chunk_text(text, 32)[0].text.endswith("\n\n")
    assert chunk_text(text, 32)[0].id != chunk_text("う" + text[1:], 32)[0].id
    text = "あ" * 15 + "。" + "い" * 25
    assert chunk_text(text, 32)[0].text.endswith("。")


def test_v14_assembly_preserves_source_chunk_boundaries():
    assert assemble(["日本語。", "次の文。"], ["Первое.", "Второе."]) == "Первое. Второе."
    assert assemble(["日本語。\n", "次の文。"], ["Первое.", "Второе."]) == "Первое.\nВторое."
    assert assemble(["日本語。\n", "\n次の文。"], ["Первое.", "Второе."]) == "Первое.\n\nВторое."


def test_context_is_bounded_and_zero_means_empty():
    assert build_context("123456", "abcdef", 3, "previous") == {
        "previous_japanese": "456",
        "previous_russian": "def",
    }
    assert build_context("abc", "abc", 0, "previous") == {}
    assert build_context("abc", "abc", 3, "none") == {}


@pytest.mark.parametrize(
    "output,reason,level,issue",
    [
        (None, None, "FAIL", "missing_output_file"),
        ("", "stop", "FAIL", "empty_output"),
        ("Да", "stop", "FAIL", "too_short"),
        ("Нормальный русский текст.", "length", "FAIL", "finish_reason:length"),
        ("Нормальный русский текст.", "tool_calls", "FAIL", "finish_reason:tool_calls"),
        ("Нормальный русский текст.", "function_call", "FAIL", "finish_reason:function_call"),
        ("Испорченный \ufffd текст.", "stop", "FAIL", "broken_encoding"),
        ("日本語だけの文章です。日本語。", "stop", "WARNING", "japanese_characters"),
        ("Повторяющаяся длинная фраза.\n" * 3, "stop", "WARNING", "repetition"),
        ("а" * 200, "stop", "FAIL", "runaway_repetition"),
    ],
)
def test_validation(output, reason, level, issue):
    result = validate("あ" * 30, output, reason, ValidationConfig())
    assert result.level == level
    assert issue in result.issues


def test_validation_allows_source_replacement_markers_but_rejects_added_markers():
    source = "Имя героя: 『譁�ュ怜喧縺�』."
    preserved = "Перевод сохраняет авторское имя: «譁�ュ怜喧縺�»."
    result = validate(source, preserved, "stop", ValidationConfig())
    assert "broken_encoding" not in result.issues

    added = "Перевод сохраняет авторское имя: «譁�ュ怜喧縺��»."
    result = validate(source, added, "stop", ValidationConfig())
    assert result.level == "FAIL"
    assert "broken_encoding" in result.issues


@pytest.mark.parametrize("reason", ["stop", "end_turn", None])
def test_terminal_and_absent_finish_reasons_are_not_rejected(reason):
    result = validate("あ" * 30, "Нормальный русский текст.", reason, ValidationConfig())
    assert not any(issue.startswith("finish_reason:") for issue in result.issues)


def test_validation_pass_and_file(tmp_path):
    assert validate("原文。", "Текст.", "stop", ValidationConfig()).level == "PASS"
    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff")
    assert validate_file("原文", path, None, ValidationConfig()).issues == ["broken_encoding"]
    assert validate_file("原文", tmp_path / "missing", None, ValidationConfig()).level == "FAIL"


def test_config_paths_dotenv_secrets(project, monkeypatch):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["runners"]["test"]["api_key_env"] = "NOVEL_TEST_SECRET"
    project.write_text(yaml.safe_dump(data), "utf-8")
    (project.parent / ".env").write_text("NOVEL_TEST_SECRET=local-secret", "utf-8")
    cfg = load_config(project)
    assert cfg.novel.source_dir == project.parent / "original"
    assert cfg.key_for("test") == "local-secret"
    monkeypatch.setenv("NOVEL_TEST_SECRET", "override-secret")
    assert cfg.key_for("test") == "override-secret"
    assert "secret" not in cfg.model_dump_json()
    assert "api_key" not in cfg.runners["test"].model_dump()


def test_config_rejects_overlap_unknown_fields_and_bad_params(project):
    data = yaml.safe_load(project.read_text("utf-8"))
    data["novel"]["output_dir"] = "./original/out"
    project.write_text(yaml.safe_dump(data), "utf-8")
    with pytest.raises(ValueError, match="separate"):
        load_config(project)
    with pytest.raises(ValidationError):
        RunnerConfig(extra_params={"messages": []})
    with pytest.raises(ValidationError):
        RunnerConfig(temperatur=0.2)
    with pytest.raises(ValidationError):
        RunnerConfig(base_url="https://secret@example.com/v1")
