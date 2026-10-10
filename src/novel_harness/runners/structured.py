"""Explicit generation constraints; deterministic host validation remains authoritative."""

import json

from ..config import RunnerConfig
from .base import TranslationJob


def _literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_literal(value) -> str:
    return _literal(json.dumps(value, ensure_ascii=False))


def llama_grammar(job: TranslationJob) -> str:
    """GBNF for the built-in contracts, including ordered coverage and patch hashes."""
    if job.segments is None or job.output_mode not in {"segments", "review"}:
        raise ValueError("llama-grammar requires a segments-v1 translation or review job")
    ids = [s["id"] for s in job.segments]
    rules = [
        r"ws ::= [ \t\n\r]*",
        r"hex ::= [0-9a-fA-F]",
        r'char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)',
        r'string ::= "\"" char* "\""',
        # Exclude decoded CR/LF, including \u000a and \u000d spellings.
        r'line-unicode ::= [1-9a-fA-F] hex hex hex | "0" [1-9a-fA-F] hex hex | "00" [1-9a-fA-F] hex | "000" [0-9bBcCeEfF]',
        r'line-char ::= [^"\\\x00-\x1F] | "\\" (["\\/bft] | "u" line-unicode)',
        r'line ::= "\"" line-char+ "\""',
    ]

    def field(name: str, value: str) -> str:
        return f'{_json_literal(name)} ws ":" ws {value}'

    def obj(fields: list[str]) -> str:
        return '"{" ws ' + ' "," ws '.join(fields) + ' ws "}"'

    if job.output_mode == "segments":
        entries = []
        for i, key in enumerate(ids):
            name = f"segment-{i}"
            rules.append(
                f"{name} ::= " + obj([field("id", _json_literal(key)), field("text", "line")])
            )
            entries.append(name)
        array = '"[" ws ' + ' ws "," ws '.join(entries) + ' ws "]"'
        rules.append("root ::= " + obj([field("segments", array)]) + " ws")
    else:
        if job.draft_segments is None or [s["id"] for s in job.draft_segments] != ids:
            raise ValueError("llama-grammar review requires verified ordered draft segments")
        from ..translation.review import UNIFIED_REVIEW_CONTRACTS, review_categories

        rules.append(
            "category ::= "
            + " | ".join(_json_literal(c) for c in review_categories(job.review_contract))
        )

        def subset_array(entries: list[str], name: str) -> str:
            # Nonempty source-ordered subsets. Each suffix appears once per rule,
            # so both the grammar and an expanded recognizer stay linear in IDs.
            # Optional prefixes skip acceptable segments without inventing entries.
            if not entries:
                return '"[" ws "]"'
            for i in reversed(range(len(entries))):
                tail = f"{name}-list-{i}"
                if i == len(entries) - 1:
                    body = entries[i]
                else:
                    body = f'({entries[i]} ws "," ws)? {name}-list-{i + 1} | {entries[i]}'
                rules.append(f"{tail} ::= {body}")
            return f'"[" ws {name}-list-0? ws "]"'

        patches = []
        for i, draft in enumerate(job.draft_segments):
            name = f"patch-{i}"
            rules.append(
                f"{name} ::= "
                + obj(
                    [
                        field("id", _json_literal(draft["id"])),
                        field("category", "category"),
                        field("reason", "string"),
                        field("original_sha256", _json_literal(draft["sha256"])),
                        # A null replacement is always rejected by apply_review, so
                        # it can never produce a valid review. Generation may only
                        # emit a concrete correction or omit the patch for that
                        # segment. Unresolved defects belong in ``unresolved``.
                        field("replacement", "line"),
                    ]
                )
            )
            patches.append(name)
        issues = subset_array(patches, "patch")
        checked = '"[" ws ' + ' ws "," ws '.join(_json_literal(key) for key in ids) + ' ws "]"'
        if job.review_contract in UNIFIED_REVIEW_CONTRACTS:
            unresolved_entries = []
            for i, draft in enumerate(job.draft_segments):
                name = f"unresolved-{i}"
                rules.append(
                    f"{name} ::= "
                    + obj(
                        [
                            field("id", _json_literal(draft["id"])),
                            field("category", "category"),
                            field("reason", "string"),
                        ]
                    )
                )
                unresolved_entries.append(name)
            unresolved = subset_array(unresolved_entries, "unresolved")
            rules.append(
                "root ::= "
                + obj(
                    [
                        field("checked_ids", checked),
                        field("issues", issues),
                        field("unresolved", unresolved),
                    ]
                )
                + " ws"
            )
        else:
            rules.append(
                "root ::= " + obj([field("checked_ids", checked), field("issues", issues)]) + " ws"
            )
    return "\n".join(rules) + "\n"


def structured_params(profile: RunnerConfig, job: TranslationJob) -> dict:
    """Copy provider options and inject per-job constraints only when selected."""
    options = dict(profile.extra_params)
    if job.response_schema is None or profile.structured_decoding == "prompt":
        return options
    if {"grammar", "json_schema", "response_format"}.intersection(options):
        raise ValueError("structured_decoding conflicts with explicit provider constraints")
    if profile.structured_decoding == "llama-grammar":
        options["grammar"] = llama_grammar(job)
    else:
        options["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "stage_output", "strict": True, "schema": job.response_schema},
        }
    return options
