"""Opt-in structured translation quality judge.

The judge is advisory only: it never modifies the translation, never replaces
deterministic validation, and only runs when explicitly invoked with a
configured runner. Malformed judge output is rejected rather than trusted.
"""

from __future__ import annotations

import json

from .runners.base import TranslationJob

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
        "issues": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["score", "issues"],
    "additionalProperties": False,
}

JUDGE_PROMPT = (
    "You are a translation quality judge. Compare the translation with its source. "
    "Return only a JSON object matching response_schema: an integer score from 0 "
    "(unusable) to 5 (excellent), a list of concrete issues, and short notes. "
    "Do not rewrite or return the translation itself."
)


class JudgeError(ValueError):
    """Malformed or unusable judge output."""


def parse_judge_response(text: str | None) -> dict:
    if not text or not text.strip():
        raise JudgeError("Judge returned no output")
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[1] if "\n" in payload else payload
        if payload.rstrip().endswith("```"):
            payload = payload.rsplit("```", 1)[0]
    try:
        data = json.loads(payload)
    except ValueError as error:
        raise JudgeError("Judge output is not valid JSON") from error
    if not isinstance(data, dict):
        raise JudgeError("Judge output must be a JSON object")
    score = data.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 5:
        raise JudgeError("Judge score must be an integer between 0 and 5")
    issues = data.get("issues", [])
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise JudgeError("Judge issues must be a list of strings")
    notes = data.get("notes", "")
    if not isinstance(notes, str):
        raise JudgeError("Judge notes must be a string")
    return {"score": score, "issues": issues, "notes": notes}


def build_judge_job(
    *,
    run_id: str,
    chapter: int,
    chunk_id: str,
    source: str,
    translation: str,
    workspace,
) -> TranslationJob:
    return TranslationJob(
        run_id=run_id,
        chapter=chapter,
        chunk_id=chunk_id,
        stage="judge",
        prompt=JUDGE_PROMPT,
        source=translation,
        context={"source": source},
        glossary={},
        workspace=workspace,
        temperature=0.0,
        response_schema=JUDGE_SCHEMA,
        output_mode="review",
    )
