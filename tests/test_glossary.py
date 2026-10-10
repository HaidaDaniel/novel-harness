import json

import pytest

from novel_harness.config import load_glossary


@pytest.mark.parametrize(
    "content",
    [
        "characters:\n  '???': '????'\n",
        "characters:\n  アレン: '??? ???'\n",
        "characters:\n  アレン: Алл\ufffdен\n",
        "characters:\n  アレン: Алл??ен\n",
        "characters:\n  アレン: 123\n",
        "characters:\n  '': Аллен\n",
        "characters:\n  アレン: Аллен\n  アレン: Ален\n",
        "characters: {}\ncharacters: {}\n",
        "false\n",
    ],
)
def test_v20_glossary_rejects_damaged_or_duplicate_entries(tmp_path, content):
    path = tmp_path / "glossary.yaml"
    path.write_text(content, "utf-8")
    with pytest.raises(ValueError, match="[Gg]lossary"):
        load_glossary(path)


def test_v20_glossary_unicode_roundtrip_and_empty_sections(tmp_path):
    path = tmp_path / "glossary.yaml"
    path.write_text("characters:\n  アレン: Аллен\n", "utf-8")
    assert load_glossary(path)["characters"] == {"アレン": "Аллен"}
    path.write_text("{}", "utf-8")
    assert all(section == {} for section in load_glossary(path).values())


def test_v20_bad_glossary_rejected_before_run_or_model(service):
    svc, fake = service
    svc.config.translation.glossary_path.write_text("characters: {'???': '???'}", "utf-8")
    with pytest.raises(ValueError, match="[Gg]lossary"):
        svc.create_run(runner="test")
    assert not svc.repository.list_runs()
    assert not fake.calls


def test_v20_frozen_glossary_validated_before_resume(service):
    svc, fake = service
    rid = svc.create_run(runner="test")
    manifest = svc.repository.run(rid)["manifest"]
    manifest["glossary"]["characters"] = {"???": "???"}
    with svc.repository.connection() as db:
        db.execute("UPDATE runs SET manifest=? WHERE id=?", (json.dumps(manifest), rid))
    with pytest.raises(ValueError, match="[Gg]lossary"):
        svc.execute(rid)
    assert not fake.calls


def test_v20_continuation_uses_repaired_glossary_preserving_parent(service, tmp_path):
    svc, fake = service
    parent = svc.create_run(runner="test")
    svc.execute(parent, max_chunks=1)
    old_manifest = svc.repository.run(parent)["manifest"]
    path = tmp_path / "repaired.yaml"
    path.write_text("characters:\n  アレン: Аллен\n", "utf-8")
    svc.config.translation.glossary_path = path
    continued = svc.continue_run(parent, runner="test")
    manifest = svc.repository.run(continued)["manifest"]
    assert manifest["glossary"]["characters"] == {"アレン": "Аллен"}
    assert manifest["translation"]["glossary_path"] == str(path)
    assert svc.repository.run(parent)["manifest"] == old_manifest
    assert svc.status(continued)["inherited_chunks"] == 1
    svc.execute(continued, max_chunks=1)
    assert fake.calls[-1].glossary["characters"] == {"アレン": "Аллен"}
