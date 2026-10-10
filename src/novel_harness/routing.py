"""Deterministic, explainable per-chunk routing to an escalation runner.

Routing is opt-in and only ever escalates the translator stage. Signals are
local and deterministic (source size, glossary/name density); a chunk with no
trigger keeps the stage's default runner. The escalation runner profile is
frozen in the run manifest so provenance stays reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    runner: str
    runner_config: dict | None
    escalated: bool
    reason: str | None


def matched_names(text: str, glossary: dict) -> int:
    if not isinstance(glossary, dict):
        return 0
    return sum(
        1
        for section in glossary.values()
        if isinstance(section, dict)
        for name in section
        if name and name in text
    )


def choose_runner(
    stage: dict,
    chunk: dict,
    glossary: dict,
    routing: dict | None,
) -> RoutingDecision:
    default = RoutingDecision(stage["runner"], None, False, None)
    if stage.get("role") != "translator" or not routing:
        return default
    escalation = routing.get("escalation_runner")
    if not escalation:
        return default
    text = chunk["text"]
    reasons = []
    if len(text) >= routing["escalate_over_chars"]:
        reasons.append(f"source_chars>={routing['escalate_over_chars']}")
    names = matched_names(text, glossary)
    if names >= routing["escalate_over_names"]:
        reasons.append(f"glossary_names>={routing['escalate_over_names']}")
    if not reasons:
        return default
    return RoutingDecision(
        escalation,
        routing.get("runner_config"),
        True,
        "; ".join(reasons),
    )
