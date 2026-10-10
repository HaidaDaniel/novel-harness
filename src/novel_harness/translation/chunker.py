import re
from dataclasses import dataclass

from ..artifacts import sha256

CHUNKER_VERSION = "paragraph-v1"
_PROTECTED_MARKER = re.compile(r"⟦EPUB_[^⟧]+⟧")


@dataclass(frozen=True)
class Chunk:
    index: int
    id: str
    text: str
    source_sha256: str
    start: int
    end: int


def chunk_text(text: str, max_chars: int = 6000) -> list[Chunk]:
    """Lossless partitions; prefer blank lines, then newlines, then sentence ends."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks = []
    start = 0
    marker = _PROTECTED_MARKER.search(text)
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            for boundary in (r"\n[ \t]*\n", r"\n", r"[。！？.!?][」』）\"']?\s*"):
                candidates = [
                    m.end() for m in re.finditer(boundary, window) if m.end() >= max_chars // 3
                ]
                if candidates:
                    end = start + candidates[-1]
                    break
            # EPUB round-trip sentinels are an export wire format. Keep each
            # sentinel whole even when the configured chunk size is very small.
            while marker is not None and marker.start() < end:
                if marker.start() < end < marker.end():
                    end = marker.end()
                marker = _PROTECTED_MARKER.search(text, marker.end())
        value = text[start:end]
        digest = sha256(value)
        index = len(chunks) + 1
        chunks.append(Chunk(index, f"{index:04d}-{digest[:16]}", value, digest, start, end))
        start = end
    return chunks
