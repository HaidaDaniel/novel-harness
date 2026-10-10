"""Readable final filenames; immutable versions live in separate directories."""

import re
import unicodedata
from pathlib import Path

from .artifacts import sha256
from .config import safe_name


def _translator_profile(manifest: dict) -> dict:
    """Translator identity for a direct run or its frozen derived review."""
    stage = next((s for s in manifest["stages"] if s["role"] == "translator"), None)
    if stage is not None:
        return stage["runner_config"]
    # A derived review has only a reviewer stage; naming still reflects the frozen
    # direct translator baseline, never the reviewer, and adds no fake stage.
    for unit in (manifest.get("review_input") or {}).get("units") or []:
        origin = unit.get("origin") or {}
        model = origin.get("configured_model") or origin.get("model")
        if model:
            return {"type": origin.get("harness") or "unknown", "model": model}
    raise ValueError("Manifest has no translator or derived review identity")


def export_identity(manifest: dict, novel: dict) -> dict:
    profile = _translator_profile(manifest)
    model = profile.get("model") or "unspecified-model"
    # A provider/model ID or a GGUF path is provenance, not a filesystem path here.
    display_model = re.split(r"[/\\]", model.rstrip("/\\"))[-1] or "unspecified-model"
    return {
        "title": novel["title"],
        "harness": profile["type"],
        "model": model,
        "display_model": display_model,
    }


def safe_piece(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .") or "untitled"
    if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", text):
        text = "_" + text
    return text


def utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def export_path(books_dir: Path, manifest: dict, novel: dict, content_hash: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", content_hash):
        raise ValueError("Export requires a SHA256 content hash")
    directory = books_dir.resolve() / safe_name(manifest["run_id"]) / content_hash[:16]
    # Keep below legacy Windows MAX_PATH and portable component byte limits.
    max_units = min(240, 240 - utf16_units(str(directory)) - 1)
    if max_units < 60:
        raise ValueError("books_dir path is too long; select a shorter output directory")
    identity = export_identity(manifest, novel)
    originals = [safe_piece(identity[k]) for k in ("title", "harness", "display_model")]
    parts = list(originals)

    def filename():
        values = [
            p.rstrip(" .") + ("~" + sha256(original)[:8] if p != original else "")
            for p, original in zip(parts, originals, strict=True)
        ]
        return " - ".join(values) + ".epub"

    name = filename()
    while utf16_units(name) > max_units or len(name.encode("utf-8")) > 240:
        eligible = [i for i, p in enumerate(parts) if len(p) > 12]
        if not eligible:
            raise ValueError("Cannot fit export filename; select a shorter books_dir")
        longest = max(eligible, key=lambda i: len(parts[i].encode("utf-8")))
        parts[longest] = parts[longest][:-1]
        name = filename()
    return directory / name
