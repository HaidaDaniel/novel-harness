import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ValidationConfig


@dataclass(frozen=True)
class ValidationResult:
    level: str
    issues: list[str]

    def to_dict(self):
        return asdict(self)


# Explicit terminal provider turns. ``stop`` is the OpenAI-compatible terminal
# reason; ``end_turn`` is the documented legacy adapter terminal reason. An
# absent reason (None) is a separate legacy/export-audit case and is not treated
# as a failure here.
TERMINAL_FINISH_REASONS = frozenset({"stop", "end_turn"})


# Japanese letters only: kana, CJK ideographs (including extension A, compatibility
# and supplementary planes), halfwidth kana, and the ideographic content marks
# (U+3005, U+3006, U+3007, U+303B). Ordinary CJK punctuation such as the katakana
# middle dot (U+30FB), brackets, comma and full stop (U+3000-U+303F) and halfwidth
# punctuation is intentionally excluded so a stray mark is not reported as a
# script leak.
FOREIGN_SCRIPT_LETTERS = re.compile(
    "["
    "\u3005\u3006\u3007\u303b"  # ideographic iteration, closing, zero and vertical marks
    "\u3041-\u3096"  # hiragana
    "\u309d-\u309f"  # hiragana iteration marks and digraph yori
    "\u30a1-\u30fa"  # katakana, excluding the middle dot U+30FB
    "\u30fc-\u30ff"  # prolonged sound mark, iteration marks, digraph koto
    "\u3400-\u4dbf"  # CJK unified ideographs extension A
    "\u4e00-\u9fff"  # CJK unified ideographs
    "\uf900-\ufaff"  # CJK compatibility ideographs
    "\uff66-\uff9d"  # halfwidth katakana
    "\U00020000-\U0002ffff"  # supplementary ideographic planes
    "]"
)
_CYRILLIC_OR_LATIN = re.compile(
    "[\u0400-\u04ff\u0500-\u052fA-Za-z\u00c0-\u024f]"
)
_CYRILLIC = re.compile("[\u0400-\u04ff\u0500-\u052f]")
_DASH_ONLY_LINE = re.compile(r"[ \t]*[\u2014\u2013]+[ \t]*$")
# Leaked JSON/markup envelope fragments observed in saved runs. Quoted JSON keys
# only: bare English words like `issues:`/`text:` in prose must not match.
_SERVICE_ARTIFACT = re.compile(
    r'\}\s*,\s*\{|\[\s*\{|\}\s*\]|```|\\u[0-9a-fA-F]{4}|"[a-z_]+"\s*:\s*'
)
_MIN_COLLAPSE_SOURCE_CHARS = 40
_MAX_COLLAPSE_RATIO = 0.4
_MIN_CYRILLIC_SOURCE_CHARS = 8


def foreign_script_count(text: str) -> int:
    return len(FOREIGN_SCRIPT_LETTERS.findall(text))


def service_artifact_issues(output: str, config: ValidationConfig) -> list[str]:
    if not config.reject_service_artifacts:
        return []
    return ["service_artifact"] if _SERVICE_ARTIFACT.search(output) else []


def validate_segments(
    source: str, translations: list[dict], config: ValidationConfig
) -> ValidationResult:
    """Per-segment corruption checks on the source-addressed contract.

    These detect observable, non-semantic corruption: exact ID coverage, a segment
    translated to implausibly little text, a Japanese-bearing segment rendered with
    no Cyrillic, script leaks/mixes and leaked service envelopes. They do not and
    cannot detect semantic degeneration.
    """
    from .segments import source_segments

    originals = source_segments(source)
    if any(
        not isinstance(s, dict)
        or set(s) != {"id", "text"}
        or not isinstance(s["id"], str)
        or not isinstance(s["text"], str)
        for s in translations
    ):
        return ValidationResult("FAIL", ["invalid_segment_shape"])
    ids = [s["id"] for s in translations]
    if ids != [s["id"] for s in originals]:
        return ValidationResult("FAIL", ["segment_id_mismatch"])
    failures: list[str] = []
    for original, translated in zip(originals, translations, strict=True):
        source_text, text = original["source"], translated["text"]
        if config.reject_segment_collapse and (
            len(source_text) >= _MIN_COLLAPSE_SOURCE_CHARS
            and len(text) < _MAX_COLLAPSE_RATIO * len(source_text)
        ):
            failures.append("segment_collapse")
        # Conservative: only a segment rendered as pure symbols/whitespace (no
        # letter or digit in any script) is flagged, so Latin names, loanwords and
        # numeric-only output do not false-positive.
        if (
            config.reject_missing_cyrillic
            and len(source_text) >= _MIN_CYRILLIC_SOURCE_CHARS
            and FOREIGN_SCRIPT_LETTERS.search(source_text)
            and not _CYRILLIC.search(text)
            and not re.search(r"[^\W_]", text)
        ):
            failures.append("missing_cyrillic")
        failures.extend(service_artifact_issues(text, config))
        failures.extend(strict_script_issues(text, config))
    return ValidationResult("FAIL" if failures else "PASS", sorted(set(failures)))


def mixed_script_tokens(text: str) -> list[str]:
    """Whitespace-delimited tokens mixing a Japanese letter with Cyrillic/Latin."""
    tokens = []
    for match in re.finditer(r"\S+", text):
        token = match.group(0)
        if FOREIGN_SCRIPT_LETTERS.search(token) and _CYRILLIC_OR_LATIN.search(token):
            tokens.append(token)
    return tokens


def strict_script_issues(output: str, config: ValidationConfig) -> list[str]:
    if config.max_foreign_script_chars is None:
        return []
    issues = []
    if foreign_script_count(output) > config.max_foreign_script_chars:
        issues.append("foreign_script_leak")
    if mixed_script_tokens(output):
        issues.append("mixed_script_token")
    return issues


def strict_dialogue_issues(output: str, config: ValidationConfig) -> list[str]:
    if not config.reject_malformed_dialogue:
        return []
    issues = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if not _DASH_ONLY_LINE.fullmatch(line):
            continue
        issues.append("runaway_dash_line")
        previous = next(
            (lines[candidate] for candidate in range(index - 1, -1, -1) if lines[candidate].strip()),
            "",
        )
        if previous.rstrip().endswith(","):
            issues.append("dangling_dialogue_comma")
    return issues



def repeated_copies(text: str, *, minimum: int = 12) -> int:
    """Contiguous loops; source baseline also counts short intentional sound effects."""
    pattern = r"(.{1,80}?)\1" + f"{{{minimum - 1},}}"
    return max(
        (
            len(m[0]) // len(m[1])
            for m in re.finditer(pattern, text)
            if any(c.isalnum() for c in m[1])
        ),
        default=0,
    )


def _has_unexpected_replacement_marker(source: str, output: str) -> bool:
    """Reject replacement markers that exceed the source's literal markers.

    U+FFFD can be intentional source prose (for example, a character's
    deliberately mojibake profile name).  The validator still fails closed when
    a model adds more markers than the source chunk contains.
    """
    return output.count("\ufffd") > source.count("\ufffd")


def validate(
    source: str, output: str | None, finish_reason: str | None, config: ValidationConfig
) -> ValidationResult:
    failures, warnings = [], []
    if output is None:
        return ValidationResult("FAIL", ["missing_output_file"])
    if not output.strip():
        failures.append("empty_output")
    if _has_unexpected_replacement_marker(source, output) or any(
        ord(c) < 32 and c not in "\n\r\t" for c in output
    ):
        failures.append("broken_encoding")
    # Reject any explicit nonterminal turn (length, content_filter, tool_calls,
    # function_call and unknown reasons) rather than treating its text as a
    # finished translation. A missing reason is not rejected here.
    if finish_reason is not None and finish_reason not in TERMINAL_FINISH_REASONS:
        failures.append(f"finish_reason:{finish_reason}")
    source_length = len(source.strip())
    output_length = len(output.strip())
    if output_length < min(config.min_output_chars, max(source_length, 1)):
        failures.append("too_short")
    ratio = output_length / max(source_length, 1)
    if ratio < config.min_length_ratio:
        failures.append("length_ratio_low")
    if ratio > config.max_length_ratio:
        warnings.append("length_ratio_high")
    japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", output))
    if japanese / max(output_length, 1) > config.max_japanese_ratio:
        warnings.append("japanese_characters")
    failures.extend(strict_script_issues(output, config))
    failures.extend(strict_dialogue_issues(output, config))
    failures.extend(service_artifact_issues(output, config))
    units = [s.strip() for s in re.split(r"\n+|(?<=[.!?。！？])\s+", output) if len(s.strip()) > 12]
    if any(n >= config.repetition_count for n in Counter(units).values()):
        warnings.append("repetition")
    if re.search(r"(.{12,80}?)\1{2,}", output) and "repetition" not in warnings:
        warnings.append("repetition")
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]{16,}", output):
        failures.append("foreign_script_run")
    # Source-aware allowance for deliberately repeated dialogue/sound effects.
    copies = repeated_copies(output)
    if copies >= 12 and copies > max(3 * repeated_copies(source, minimum=2), 11):
        failures.append("runaway_repetition")
    if output.count("«") != output.count("»"):
        warnings.append("unbalanced_quotes")
    return ValidationResult(
        "FAIL" if failures else "WARNING" if warnings else "PASS", failures + warnings
    )


def validate_file(
    source: str, path: Path, finish_reason: str | None, config: ValidationConfig
) -> ValidationResult:
    if not path.is_file():
        return validate(source, None, finish_reason, config)
    try:
        return validate(source, path.read_text(encoding="utf-8"), finish_reason, config)
    except UnicodeDecodeError:
        return ValidationResult("FAIL", ["broken_encoding"])
