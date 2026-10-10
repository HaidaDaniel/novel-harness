import pytest
import yaml

from novel_harness.artifacts import canonical, sha256
from novel_harness.translation.book_context import context_for, freeze_book_context


def configure(svc, source):
    text = "第一章\n\nルゥは女の子。"
    source.write_text(text, encoding="utf-8")
    data = {
        "style": "Пиши естественным литературным русским.",
        "cards": [
            {
                "source_terms": ["ルゥ"],
                "name": "Ру",
                "gender": "female",
                "evidence": [{"chapter": 1, "quote": "ルゥは女の子"}],
            }
        ],
    }
    path = source.parent.parent / "book.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    svc.config.translation.book_context_path = path
    svc.config.translation.cross_chapter_context = True
    return path


def test_v24_frozen_cards_and_adjacent_context(service, source):
    svc, fake = service
    path = configure(svc, source)
    (source.parent / "0002_next.txt").write_text("次章\n\nルゥは笑った。", encoding="utf-8")
    rid = svc.create_run(runner="test")
    path.write_text("cards: []", encoding="utf-8")  # Frozen run keeps reviewed facts.
    assert svc.execute(rid)["status"] == "completed"
    second = next(j for j in fake.calls if j.chapter == 2)
    assert second.context["book"]["cards"][0]["gender"] == "female"
    assert second.context["previous_japanese"] == source.read_text("utf-8")[-20:]
    assert second.context["previous_russian"]
    snapshot = svc.run_dir(rid) / "book-context.json"
    snapshot.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot changed"):
        svc.resume(rid)


def test_v24_evidence_and_budget(service, source):
    svc, _ = service
    path = configure(svc, source)
    book = freeze_book_context(path, source.parent)
    assert context_for(book, 1, "ルゥ", 1000)["cards"]
    with pytest.raises(ValueError, match="budget"):
        context_for(book, 1, "ルゥ", 10)
    data = yaml.safe_load(path.read_text("utf-8"))
    data["cards"][0]["evidence"][0]["quote"] = "not in original"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence not found"):
        svc.create_run(runner="test")


def test_v24_continuation_snapshots_new_cards(service, source):
    svc, _ = service
    path = configure(svc, source)
    svc.config.translation.book_context_path = None
    parent = svc.create_run(runner="test")
    svc.execute(parent)
    svc.config.translation.book_context_path = path
    child = svc.continue_run(parent, runner="test")
    assert svc.repository.run(parent)["manifest"]["book_context"] == {}
    assert svc.repository.run(child)["manifest"]["book_context"]["cards"][0]["gender"] == "female"
    assert svc.resume(child)["status"] == "completed"


def test_v24_nonadjacent_sample_does_not_inherit_context(service, source):
    svc, fake = service
    configure(svc, source)
    (source.parent / "0003_later.txt").write_text("後の章\n\nルゥは笑った。", encoding="utf-8")
    rid = svc.create_run(runner="test", chapters=[1, 3])
    svc.execute(rid)
    assert not next(j for j in fake.calls if j.chapter == 3).context["previous_japanese"]


def test_v24_future_facts_excluded_and_duplicate_keys_rejected(service, source):
    svc, _ = service
    path = configure(svc, source)
    book = freeze_book_context(path, source.parent)
    book["cards"][0]["available_from"] = 2
    assert not context_for(book, 1, "ルゥ", 1000)["cards"]
    path.write_text("cards: []\ncards: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        freeze_book_context(path, source.parent)


def test_v24_global_style_present_without_matched_card(service, source):
    svc, _ = service
    path = configure(svc, source)
    book = freeze_book_context(path, source.parent)
    without_card = context_for(book, 1, "здесь нет имени", 1000)
    assert without_card["style"].startswith("Пиши")
    assert without_card["cards"] == []


def test_v24_card_requires_term_and_chapter_gate(service, source):
    svc, _ = service
    path = configure(svc, source)
    book = freeze_book_context(path, source.parent)
    assert context_for(book, 1, "ルゥ пришла", 1000)["cards"]
    assert context_for(book, 1, "без имени", 1000)["cards"] == []
    book["cards"][0]["available_from"] = 2
    assert context_for(book, 1, "ルゥ пришла", 1000)["cards"] == []


def test_v24_frozen_context_top_level_with_hash(service, source):
    svc, _ = service
    configure(svc, source)
    rid = svc.create_run(runner="test")
    manifest = svc.repository.run(rid)["manifest"]
    assert manifest["book_context"]["style"].startswith("Пиши")
    assert manifest["book_context_sha256"] == sha256(canonical(manifest["book_context"]))
    assert (svc.run_dir(rid) / "book-context.json").is_file()


def test_v24_damaged_context_rejected(service, source):
    svc, _ = service
    path = configure(svc, source)
    data = yaml.safe_load(path.read_text("utf-8"))
    data["style"] = "Плохой ?? текст"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Damaged"):
        freeze_book_context(path, source.parent)


def _bible_entries():
    return [
        {
            "kind": "character", "source_form": "ルゥ", "target_form": "Ру",
            "aliases": ["ルー"], "trust": "proposed",
        },
        {
            "kind": "term", "source_form": "魔術", "target_form": "магия",
            "aliases": [], "trust": "confirmed",
        },
        {
            "kind": "place", "source_form": "森", "target_form": "лес",
            "aliases": ["森の中"], "trust": "locked",
        },
    ]


def test_projection_includes_only_authoritative_trust_and_is_deterministic():
    from novel_harness.translation.book_context import (
        authoritative_projection,
        projection_sha256,
    )

    first = authoritative_projection(_bible_entries())
    second = authoritative_projection(list(reversed(_bible_entries())))

    assert first == second
    assert {row["source_form"] for row in first} == {"魔術", "森"}
    assert projection_sha256(first) == projection_sha256(second)


def test_book_bible_block_records_policy_and_hash():
    from novel_harness.translation.book_context import (
        book_bible_block,
        projection_sha256,
    )

    block = book_bible_block(_bible_entries())

    assert block["policy"] == "confirmed+locked"
    assert block["authoritative_trust"] == ["confirmed", "locked"]
    assert block["projection_sha256"] == projection_sha256(block["entries"])
    assert all(row["source_form"] != "ルゥ" for row in block["entries"])


def test_bible_context_matches_source_form_or_alias_only():
    from novel_harness.translation.book_context import bible_context_for, book_bible_block

    block = book_bible_block(_bible_entries())

    assert bible_context_for(block, "魔術を使う")["terms"][0]["target_form"] == "магия"
    assert bible_context_for(block, "森を歩く")["terms"][0]["target_form"] == "лес"
    # A proposed entry is never authoritative context.
    assert bible_context_for(block, "ルーが来た") == {}


def test_bible_context_matches_alias_when_source_form_absent():
    from novel_harness.translation.book_context import bible_context_for, book_bible_block

    block = book_bible_block(
        [
            {
                "kind": "term", "source_form": "魔法", "target_form": "магия",
                "aliases": ["魔術"], "trust": "confirmed",
            }
        ]
    )

    haystack = "魔術を使う"
    assert "魔法" not in haystack
    assert bible_context_for(block, haystack)["terms"][0]["target_form"] == "магия"
    assert bible_context_for(block, "剣を使う") == {}
