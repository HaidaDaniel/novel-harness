def build_context(
    previous_source: str, previous_translation: str, overlap_chars: int, strategy: str
) -> dict[str, str]:
    if strategy == "none" or overlap_chars == 0:
        return {}
    return {
        "previous_japanese": previous_source[-overlap_chars:],
        "previous_russian": previous_translation[-overlap_chars:],
    }
