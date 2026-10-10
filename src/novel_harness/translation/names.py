"""Deterministic, source-anchored name corrections; raw model outputs stay immutable."""

import re
from collections import Counter

from ..artifacts import canonical, sha256, write_json, write_text
from .assembly import assemble

VERSION = "names-v1"


def validate_name_rules(rules: list[dict]) -> None:
    aliases, targets = set(), set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) - {"source", "forms", "review"}:
            raise ValueError("Name rule requires source, forms and optional review")
        forms, review = rule.get("forms", {}), rule.get("review", [])
        if not isinstance(forms, dict) or not isinstance(review, list):
            # Pydantic and CLI expose malformed configuration through ValueError.
            raise ValueError("Name forms must be a mapping; review must be a list")  # noqa: TRY004
        values = [rule.get("source"), *forms.keys(), *forms.values(), *review]
        if any(
            not isinstance(v, str)
            or not any(c.isalpha() for c in v)
            or "??" in v
            or "\ufffd" in v
            or any(ord(c) < 32 for c in v)
            for v in values
        ):
            raise ValueError("Name rules require meaningful UTF-8 text")
        keys = [*forms.keys(), *review]
        if len(set(keys)) != len(keys) or aliases.intersection(keys):
            raise ValueError("Name rule aliases must be unique")
        aliases.update(keys)
        targets.update(forms.values())
    if aliases.intersection(targets):
        raise ValueError("Name rule targets must not be aliases; chained edits are ambiguous")


def normalize_names(text: str, source: str, rules: list[dict]) -> tuple[str, list, list]:
    validate_name_rules(rules)
    aliases = {}
    for rule in rules:
        for alias in [*rule.get("forms", {}), *rule.get("review", [])]:
            aliases[alias] = rule
    if not aliases:
        return text, [], []
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(alias) for alias in sorted(aliases, key=lambda x: (-len(x), x)))
        + r")(?!\w)"
    )
    changes, review = [], []

    def replace(match):
        alias = match.group()
        rule = aliases[alias]
        entry = {
            "start": match.start(),
            "end": match.end(),
            "text": alias,
            "source_anchor": rule["source"],
            "context": text[max(0, match.start() - 70) : match.end() + 70],
        }
        if rule["source"] not in source or alias not in rule.get("forms", {}):
            entry["reason"] = (
                "source_anchor_missing" if rule["source"] not in source else "ambiguous_form"
            )
            review.append(entry)
            return alias
        replacement = rule["forms"][alias]
        changes.append({**entry, "replacement": replacement})
        return replacement

    return pattern.sub(replace, text), changes, review


def create_revision(service, run_id: str) -> tuple[dict, list[dict]]:
    """Caller holds run lock. Incomplete runs yield explicit partial previews only."""
    manifest = service.repository.run(run_id)["manifest"]
    rules = manifest["translation"].get("name_rules", [])
    validate_name_rules(rules)
    if not rules:
        raise ValueError("No frozen name_rules configured; create an explicit continuation")
    rows = {(r["chapter"], r["chunk_id"]): r for r in service.repository.chunks(run_id)}
    records, chapters = [], []
    for chapter in manifest["chapters"]:
        texts, complete = [], True
        chunk_texts = []
        for chunk in chapter["chunks"]:
            row = rows[(chapter["number"], chunk["id"])]
            if not service._valid_output(row):
                complete = False
                placeholder = f"[CHUNK {chunk['index']} {row['status'].upper()}/MISSING]"
                texts.append(placeholder)
                chunk_texts.append({"chunk_id": chunk["id"], "text": placeholder})
                continue
            raw = (service.run_dir(run_id) / row["output_path"]).read_bytes().decode("utf-8")
            fixed, changes, review = normalize_names(raw, chunk["text"], rules)
            texts.append(fixed)
            chunk_texts.append({"chunk_id": chunk["id"], "text": fixed})
            records.append(
                {
                    "chapter": chapter["number"],
                    "chunk_id": chunk["id"],
                    "input_path": row["output_path"],
                    "input_sha256": sha256(raw),
                    "source_sha256": sha256(chunk["text"]),
                    "output_sha256": sha256(fixed),
                    "changes": changes,
                    "review": review,
                }
            )
        chapters.append(
            {
                "chapter": chapter["number"],
                "complete": complete,
                "translation": assemble([c["text"] for c in chapter["chunks"]], texts),
                # Per-chunk revised text lets the caller compose name revision with
                # the human final layer without flattening the chapter.
                "chunks": chunk_texts,
            }
        )
    report = {
        "run_id": run_id,
        "method": VERSION,
        "rules": rules,
        "complete": all(c["complete"] for c in chapters),
        "covered_chunks": len(records),
        "total_chunks": len(rows),
        "replacement_count": sum(len(r["changes"]) for r in records),
        "review_count": sum(len(r["review"]) for r in records),
        "chunks": records,
        "chapters": [
            {"chapter": c["chapter"], "complete": c["complete"], "sha256": sha256(c["translation"])}
            for c in chapters
        ],
    }
    revision_id = sha256(canonical(report))[:24]
    directory = service.run_dir(run_id) / "name-revisions" / revision_id
    for chapter in chapters:
        write_text(directory / f"{chapter['chapter']:04d}.md", chapter["translation"])
    report["directory"] = str(directory.resolve())
    write_json(directory / "report.json", report)
    lines = [
        "# Редакция имён",
        "",
        f"Проверено готовых фрагментов: {len(records)} из {len(rows)}.",
        f"Применено замен: {report['replacement_count']}.",
        f"Оставлено без изменений для проверки: {report['review_count']} вхождений.",
        "",
        "Исходные переводы сохранены. Эта редакция исправляет только перечисленные формы имён.",
        "Спорные вхождения ниже ещё не исправлены; контекст приведён из исходного перевода.",
        "",
        "## Применённые замены",
        "",
        "Было | Стало | Вхождений",
        "--- | --- | ---",
    ]
    counts = Counter((x["text"], x["replacement"]) for r in records for x in r["changes"])

    def cell(value):
        return str(value).replace("|", "\\|").replace("\r", "").replace("\n", " ")

    lines.extend(f"{cell(a)} | {cell(b)} | {n}" for (a, b), n in counts.most_common())
    lines.extend(
        [
            "",
            "## Требуют проверки",
            "",
            "Глава / фрагмент | Форма | Причина | Контекст",
            "--- | --- | --- | ---",
        ]
    )
    for record in records:
        for item in record["review"]:
            reason = (
                "нет имени в японском фрагменте"
                if item["reason"] == "source_anchor_missing"
                else "неоднозначное написание или падеж"
            )
            lines.append(
                f"{record['chapter']} / {record['chunk_id']} | {cell(item['text'])} | "
                f"{reason} | {cell(item['context'])}"
            )
    write_text(directory / "report.md", "\n".join(lines) + "\n")
    return report, chapters
