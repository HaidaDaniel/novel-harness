"""Measured attempts and usage; absent counters remain unknown, never zero-filled."""

import json
import statistics
from collections import Counter, defaultdict

COUNTERS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "visible_output_tokens",
)

# Attempts served from the machine cache or translation memory did no inference
# in this run; their usage is the original generation's provenance.
CACHE_REUSE_STATES = {"hit", "tm_hit", "rejected"}


def _reused_usage(attempt: dict) -> dict:
    if attempt.get("cache") not in CACHE_REUSE_STATES:
        return {}
    return attempt.get("reused_usage") or attempt.get("usage") or {}


def run_metrics(service, run_id: str) -> dict:
    run = service.repository.run(run_id)
    with service.repository.connection() as db:
        attempts = [
            json.loads(r[0])
            for r in db.execute(
                "SELECT metadata FROM attempts WHERE run_id=? ORDER BY chapter,chunk_id,stage,number",
                (run_id,),
            )
        ]
    groups = defaultdict(list)
    for attempt in attempts:
        groups[(attempt["stage"], attempt["runner"], attempt.get("model"))].append(attempt)
    # Exhaustion is recorded on the chunk row, not as an attempt, so count it from
    # chunk metadata and attribute it to the stage that produced it.
    exhausted = defaultdict(int)
    for row in service.repository.chunks(run_id):
        meta = json.loads(row["metadata"] or "{}")
        if meta.get("recovery_exhausted"):
            exhausted[(meta.get("stage"), meta.get("runner"), meta.get("model"))] += 1
    stages = []
    for (stage, runner, model), items in groups.items():
        measured = [a for a in items if a.get("measurement_version") == 1]
        times = [a["duration_seconds"] for a in measured if a.get("duration_seconds") is not None]
        inferred = [a for a in items if a.get("cache") not in CACHE_REUSE_STATES]
        reuse_hits = [a for a in items if a.get("cache") in CACHE_REUSE_STATES]
        usage, observed, reused, reused_observed = {}, {}, {}, {}
        for key in COUNTERS:
            values = [(a.get("usage") or {}).get(key) for a in inferred]
            values = [v for v in values if v is not None]
            usage[key], observed[key] = (sum(values) if values else None), len(values)
            reused_values = [_reused_usage(a).get(key) for a in reuse_hits]
            reused_values = [v for v in reused_values if v is not None]
            reused[key], reused_observed[key] = (
                sum(reused_values) if reused_values else None,
                len(reused_values),
            )
        timing_keys = {k for a in measured for k in a.get("timings", {})}
        stages.append(
            {
                "stage": stage,
                "runner": runner,
                "model": model,
                "attempts": len(items),
                "statuses": dict(Counter(a["status"] for a in items)),
                "validation_failures": sum(
                    1 for a in items if a.get("failure_kind") == "validation_failed"
                ),
                "recovery_exhausted": exhausted.get((stage, runner, model), 0),
                "measured_attempts": len(measured),
                "legacy_unverified_attempts": len(items) - len(measured),
                "seconds": sum(times) if times else None,
                "median_seconds": statistics.median(times) if times else None,
                "usage": usage,
                "usage_observed_attempts": observed,
                "reused_usage": reused,
                "reused_usage_observed_attempts": reused_observed,
                "inference_attempts": len(inferred),
                "cache_reused_attempts": len(reuse_hits),
                "timings": {
                    k: sum(a.get("timings", {}).get(k, 0) for a in measured) for k in timing_keys
                },
                "effective_settings": [
                    a["effective_settings"] for a in items[-1:] if a.get("effective_settings")
                ],
            }
        )
    return {
        "run_id": run_id,
        "stages": stages,
        "source_chars": sum(
            len(c["text"]) for ch in run["manifest"]["chapters"] for c in ch["chunks"]
        ),
        "inherited_chunks": len(run["manifest"].get("inherited", [])),
        "measurement_scope": "this run's attempts, including failures; inherited work excluded",
        "legacy_note": "Older timings lack measurement_version; inspect raw CLI artifacts before comparison.",
    }
