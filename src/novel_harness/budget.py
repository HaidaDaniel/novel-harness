"""Conservative source-budget estimation from model context and overhead.

Provider tokenizers differ, so this uses a single configurable characters-per-
token ratio and deliberately over-estimates overhead rather than under-estimate
it. Exceeding the available context raises instead of truncating silently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def estimate_tokens(text: str, chars_per_token: float) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return math.ceil(len(text) / chars_per_token)


def source_budget(
    *,
    context_limit: int,
    reserved_output: int,
    safety_margin: float,
    chars_per_token: float,
    overhead_texts: Sequence[str] = (),
    reserved_chars: int = 0,
    min_chars: int = 32,
    max_chars: int | None = None,
) -> int:
    """Largest source size (characters) that keeps the request inside context.

    Reserves the configured output tokens plus a safety margin, then subtracts
    prompt/glossary/book/previous-context overhead. Raises when the reserved
    overhead leaves no room, so callers fail loudly instead of truncating. A
    result below ``min_chars`` also raises rather than rounding up past the
    available context; ``max_chars`` caps the result to a valid configured bound.
    """
    if context_limit < 1:
        raise ValueError("context_limit must be positive")
    if reserved_output < 0:
        raise ValueError("reserved_output must not be negative")
    if not 0 <= safety_margin < 1:
        raise ValueError("safety_margin must be in [0, 1)")
    usable = context_limit - reserved_output - math.ceil(context_limit * safety_margin)
    if usable <= 0:
        raise ValueError("Reserved output and safety margin leave no context for source")
    overhead = sum(estimate_tokens(text, chars_per_token) for text in overhead_texts)
    overhead += math.ceil(max(0, reserved_chars) / chars_per_token)
    source_tokens = usable - overhead
    if source_tokens <= 0:
        raise ValueError("Prompt/glossary/context overhead exceeds the context budget")
    chars = int(source_tokens * chars_per_token)
    if chars < min_chars:
        raise ValueError(
            f"Available context fits only {chars} source characters, below the "
            f"minimum supported chunk of {min_chars}; increase context_limit or "
            f"lower reserved output/overhead"
        )
    if max_chars is not None:
        chars = min(chars, max_chars)
    return chars
