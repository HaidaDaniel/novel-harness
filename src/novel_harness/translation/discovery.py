import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from ..artifacts import sha256


@dataclass(frozen=True)
class Chapter:
    number: int
    filename: str
    title: str
    text: str
    source_sha256: str
    raw: bytes


def discover(source_dir: Path, pattern: str = "*.txt") -> list[Chapter]:
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    chapters = []
    seen = set()
    for path in source_dir.iterdir():
        if (
            path.suffix.lower() != ".txt"
            or not fnmatchcase(path.name.lower(), pattern.lower())
            or not path.is_file()
        ):
            continue
        match = re.match(r"^(\d+)(?:\D|$)", path.stem)
        if not match:
            raise ValueError(f"Chapter filename must start with a number: {path.name}")
        number = int(match[1])
        if number in seen:
            raise ValueError(f"Duplicate chapter number: {number}")
        seen.add(number)
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as error:
            raise ValueError(f"Chapter must be UTF-8: {path.name}") from error
        if not text.strip() or "\x00" in text:
            raise ValueError(f"Empty chapter or broken encoding: {path.name}")
        title = next(line.strip() for line in text.splitlines() if line.strip())
        chapters.append(Chapter(number, path.name, title, text, sha256(raw), raw))
    return sorted(chapters, key=lambda c: c.number)
