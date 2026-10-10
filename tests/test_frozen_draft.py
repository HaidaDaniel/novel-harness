import copy

import pytest

from novel_harness.translation.frozen_draft import freeze_draft, render_frozen, verify_mapping
from novel_harness.translation.segments import OutputContractError, source_segments


@pytest.mark.parametrize(
    "draft", ["  Заголовок\r\n\r\n\t— «Реплика».  \r\n", "Заголовок\n\n\n— «Реплика».\n\n"]
)
def test_lossless_layout_and_minimal_patch(draft):
    mapping = freeze_draft("題名\n本文。\n", draft, heading=True)
    verify_mapping("題名\n本文。\n", draft, mapping, heading=True)
    assert render_frozen(draft, mapping, mapping["segments"]) == draft
    patched = copy.deepcopy(mapping["segments"])
    patched[1]["text"] = "— Исправлено."
    assert render_frozen(draft, mapping, patched) == draft.replace("— «Реплика».", "— Исправлено.")


@pytest.mark.parametrize("draft", ["Один абзац", "Раз\nДва\nТри"])
def test_ambiguous_text_mapping_refused(draft):
    with pytest.raises(OutputContractError, match="mapping_mismatch"):
        freeze_draft("一\n二", draft)


def test_saved_segments_must_render_exact_baseline():
    source = "題名\n\n本文。\n"
    saved = [
        {"id": s["id"], "text": t}
        for s, t in zip(source_segments(source), ["Название", "Текст."], strict=True)
    ]
    mapping = freeze_draft(source, "Название\n\nТекст.\n", saved)
    assert mapping["kind"] == "segments-v1"
    with pytest.raises(OutputContractError, match="hash_mismatch"):
        freeze_draft(source, "Название\nТекст.\n", saved)
    mapping["spans"][0]["start"] = 1
    with pytest.raises(OutputContractError, match="mapping_changed"):
        verify_mapping(source, "Название\n\nТекст.\n", mapping)
