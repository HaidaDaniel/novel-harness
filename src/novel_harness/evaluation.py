"""Explicit experiment plans and blind human comparison; no model calls here."""

import json
import math
import random
from html import escape
from pathlib import Path

import yaml
from pydantic import Field

from .artifacts import canonical, sha256, write_json, write_text
from .config import StrictModel, _GlossaryLoader, load_config, safe_name
from .runners import runner_info
from .translation.discovery import discover
from .translation.review import SEMANTIC_EDIT_V1

ERRORS = ("meaning", "omission", "speaker_gender", "terminology", "style", "corruption")


class Variant(StrictModel):
    name: str
    runner: str
    editor: str | None = None
    pipeline: str = "direct"
    chunk_chars: int = Field(default=1500, ge=32, le=100000)


class Plan(StrictModel):
    seed: int = 20260908
    count: int = Field(default=5, ge=1, le=50)
    chapters: list[int] | None = None
    exclude: list[int] = Field(default_factory=list)
    variants: list[Variant] = Field(min_length=2, max_length=20)


def select_chapters(population: set[int], count: int, seed: int, exclude: list[int]) -> list[int]:
    available = sorted(population - set(exclude))
    if count < 1 or count > len(available):
        raise ValueError("Not enough common chapters for the requested sample")
    return sorted(random.Random(seed).sample(available, count))


def prepare_plan(service, matrix: Path) -> dict:
    plan = Plan.model_validate(yaml.load(matrix.read_text("utf-8"), Loader=_GlossaryLoader))
    if len({v.name for v in plan.variants}) != len(plan.variants):
        raise ValueError("Variant names must be unique")
    for variant in plan.variants:
        safe_name(variant.name)
        if variant.pipeline not in service.config.pipelines:
            raise ValueError(f"Unknown pipeline: {variant.pipeline}")
        for name in [variant.runner, *([variant.editor] if variant.editor else [])]:
            if (
                name not in service.config.runners
                or not runner_info(service.config, name)["available"]
            ):
                raise ValueError(f"Configure an available runner: {name}")
    population = {c.number for c in discover(service.config.novel.source_dir)}
    chapters = (
        plan.chapters
        if plan.chapters is not None
        else select_chapters(population, plan.count, plan.seed, plan.exclude)
    )
    if (
        not chapters
        or len(chapters) != len(set(chapters))
        or not set(chapters) <= population
        or set(chapters) & set(plan.exclude)
    ):
        raise ValueError("Invalid explicit evaluation chapters")
    variants = []
    for variant in plan.variants:
        rid = service.create_run(
            runner=variant.runner,
            editor=variant.editor,
            pipeline=variant.pipeline,
            chapters=chapters,
            chunk_chars=variant.chunk_chars,
        )
        variants.append({**variant.model_dump(), "run_id": rid})
    result = {
        "seed": plan.seed,
        "chapters": sorted(chapters),
        "variants": variants,
        "execution": "prepared_only; start selected runs sequentially",
    }
    path = service.root / "evaluation-plans" / f"{sha256(canonical(result))[:20]}.json"
    write_json(path, result)
    return {"path": str(path), **result}


def final_candidate(service, run_id: str, chapter: int) -> dict:
    candidate = service.inspect_chapter(run_id, chapter)
    if not candidate["complete"]:
        raise ValueError("Evaluation requires complete, intact chapter artifacts")
    book_path = service.run_dir(run_id) / "book.json"
    if book_path.is_file():
        revision = json.loads(book_path.read_text("utf-8")).get("name_revision")
        if revision:
            directory = Path(revision["directory"])
            report = json.loads((directory / "report.json").read_text("utf-8"))
            entry = next(c for c in report["chapters"] if c["chapter"] == chapter)
            text = (directory / f"{chapter:04d}.md").read_bytes().decode("utf-8")
            if sha256(text) != entry["sha256"]:
                raise ValueError("Name revision hash mismatch")
            candidate["translation"] = text
    return candidate


def export_evaluation(
    service, runs: list[str], *, count: int = 5, seed: int = 20260908, exclude: list[int] = ()
) -> dict:
    if not 2 <= len(runs) <= 4 or len(set(runs)) != len(runs):
        raise ValueError("Choose 2–4 distinct runs for a blind comparison")
    manifests = [service.repository.run(r)["manifest"] for r in runs]
    common = set.intersection(*[{c["number"] for c in m["chapters"]} for m in manifests])
    chapters = select_chapters(common, count, seed, exclude)
    cases, texts, key = [], [], []
    rng = random.Random(seed)
    for number in chapters:
        candidates = [final_candidate(service, rid, number) for rid in runs]
        if len({c["source_sha256"] for c in candidates}) != 1:
            raise ValueError("Evaluation source snapshots differ")
        rng.shuffle(candidates)
        case = {"chapter": number, "preferred": None, "candidates": []}
        panels = [
            f"<section><h3>Оригинал</h3><pre>{escape(candidates[0]['source'])}</pre></section>"
        ]
        for i, candidate in enumerate(candidates):
            label = chr(65 + i)
            case["candidates"].append(
                {"label": label, **{k: None for k in ERRORS}, "manual_minutes": None}
            )
            key.append(
                {
                    "chapter": number,
                    "label": label,
                    "run_id": candidate["run_id"],
                    "source_chars": len(candidate["source"]),
                    "source_sha256": candidate["source_sha256"],
                    "translation_sha256": sha256(candidate["translation"]),
                }
            )
            panels.append(
                f"<section><h3>Вариант {label}</h3><pre>{escape(candidate['translation'])}</pre></section>"
            )
        texts.append((number, "".join(panels)))
        cases.append(case)
    identity = sha256(canonical({"seed": seed, "key": key}))[:20]
    directory = service.root / "evaluations" / identity
    path = directory / "index.html"
    if not path.exists():
        scorecard = {"bundle_id": identity, "cases": cases}
        write_json(directory / "scorecard.json", scorecard)
        write_json(directory / "answer-key.json", {"bundle_id": identity, "seed": seed, "key": key})
        write_text(path, render_blind(scorecard, texts))
    return {
        "path": str(path),
        "directory": str(directory),
        "chapters": chapters,
        "bundle_id": identity,
        "scoring": "human; blank scorecard is not a result",
    }


def render_blind(scorecard: dict, texts: list[tuple[int, str]]) -> str:
    labels = {
        "meaning": "Ошибки смысла",
        "omission": "Пропуски",
        "speaker_gender": "Говорящий / род",
        "terminology": "Имена / термины",
        "style": "Стиль / грамматика",
        "corruption": "Повреждённые фрагменты",
        "manual_minutes": "Минуты ручной правки",
    }
    body = []
    for case, (chapter, panels) in zip(scorecard["cases"], texts, strict=True):
        forms = []
        for candidate in case["candidates"]:
            label = candidate["label"]
            inputs = "".join(
                f'<label>{caption}<input required type="number" min="0" step="{0.1 if field == "manual_minutes" else 1}" name="{chapter}:{label}:{field}"></label>'
                for field, caption in labels.items()
            )
            forms.append(f"<fieldset><legend>Вариант {label}</legend>{inputs}</fieldset>")
        options = "".join(f"<option>{c['label']}</option>" for c in case["candidates"])
        body.append(
            f'<article><h2>Глава {chapter}</h2><div class="panels">{panels}</div><div class="scores">{"".join(forms)}</div><label>Предпочтение<select required name="{chapter}:preferred"><option value="">Выбрать</option>{options}<option value="tie">Равны</option></select></label></article>'
        )
    template = json.dumps(scorecard, ensure_ascii=False).replace("<", "\\u003c")
    return (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><title>Слепое сравнение переводов</title>
<style>body{font:16px/1.5 system-ui;margin:24px;background:#f4f6fa;color:#182332}article{margin:30px 0;padding:20px;background:white;border-radius:12px}.panels{display:flex;gap:20px;overflow:auto}.panels section{flex:1;min-width:300px}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}.scores{display:flex;flex-wrap:wrap;gap:16px}fieldset{flex:1;min-width:250px}label{display:block;margin:8px 0}input,select{display:block;padding:6px;max-width:140px}button{font:inherit;padding:12px 22px}h3{position:sticky;top:0;background:white}</style>
<h1>Слепое сравнение переводов</h1><p>Названия моделей находятся отдельно в answer-key.json. Сначала оцените текст. Числа — количество отдельных ошибок; 0 означает, что вы проверили и не нашли ошибок. Один дефект учитывайте в одной основной категории. Минуты правки измеряйте отдельно.</p><form id="scores">"""
        + "".join(body)
        + """<button>Скачать оценки JSON</button></form><script>
const template="""
        + template
        + """;
document.querySelector('form').addEventListener('submit',event=>{event.preventDefault();const values=new FormData(event.target);const result=structuredClone(template);for(const c of result.cases){c.preferred=values.get(`${c.chapter}:preferred`);for(const v of c.candidates){for(const k of Object.keys(v)){if(k!=='label')v[k]=Number(values.get(`${c.chapter}:${v.label}:${k}`));}}}const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='scores-'+result.bundle_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});</script></html>"""
    )


def score_evaluation(directory: Path, scores: Path) -> dict:
    key = json.loads((directory / "answer-key.json").read_text("utf-8"))
    result = json.loads(scores.read_text("utf-8"))
    if (
        not isinstance(result, dict)
        or set(result) != {"bundle_id", "cases"}
        or result["bundle_id"] != key["bundle_id"]
    ):
        raise ValueError("Scorecard belongs to another evaluation")
    expected = {(k["chapter"], k["label"]) for k in key["key"]}
    lookup = {(k["chapter"], k["label"]): k for k in key["key"]}
    seen, seen_chapters, summaries = set(), set(), {}
    if not isinstance(result["cases"], list):
        raise ValueError("Invalid scorecard cases")  # noqa: TRY004 - invalid JSON input value
    for case in result["cases"]:
        if (
            not isinstance(case, dict)
            or set(case) != {"chapter", "preferred", "candidates"}
            or not isinstance(case["candidates"], list)
        ):
            raise ValueError("Invalid scorecard case")
        if type(case["chapter"]) is not int or case["chapter"] in seen_chapters:
            raise ValueError("Invalid or duplicate chapter score")
        seen_chapters.add(case["chapter"])
        labels = [v.get("label") for v in case["candidates"] if isinstance(v, dict)]
        if case["preferred"] not in ["tie", *labels]:
            raise ValueError("Choose a preference for every chapter")
        for candidate in case["candidates"]:
            if not isinstance(candidate, dict) or set(candidate) != {
                "label",
                "manual_minutes",
                *ERRORS,
            }:
                raise ValueError("Invalid candidate score fields")
            if not isinstance(candidate["label"], str):
                raise ValueError("Invalid candidate label")  # noqa: TRY004 - invalid JSON input value
            pair = (case["chapter"], candidate["label"])
            if pair not in expected or pair in seen:
                raise ValueError("Duplicate or unknown evaluation candidate")
            seen.add(pair)
            for field in [*ERRORS, "manual_minutes"]:
                value = candidate[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    or (field in ERRORS and not isinstance(value, int))
                ):
                    raise ValueError("Complete every human score with a nonnegative number")
            entry = lookup[pair]
            summary = summaries.setdefault(
                entry["run_id"],
                {
                    **{k: 0 for k in ERRORS},
                    "manual_minutes": 0,
                    "source_chars": 0,
                    "preferred_chapters": 0,
                },
            )
            for field in [*ERRORS, "manual_minutes"]:
                summary[field] += candidate[field]
            summary["source_chars"] += entry["source_chars"]
            summary["preferred_chapters"] += case["preferred"] == candidate["label"]
    if seen != expected:
        raise ValueError("Scorecard is incomplete")
    for summary in summaries.values():
        summary["errors_per_1000_source_chars"] = {
            k: summary[k] * 1000 / max(1, summary["source_chars"]) for k in ERRORS
        }
    return {"bundle_id": key["bundle_id"], "assessment": "user_supplied", "runs": summaries}


# --- Blind human scoring over isolated run roots (read-only, no inference) ---

HUMAN_COUNT_FIELDS = (
    "meaning",
    "omission",
    "negation",
    "actor_patient",
    "speaker_gender",
    "terminology",
    "corruption",
)
# Severe semantic defects, countable per chapter against the frozen acceptance bars.
HUMAN_SEVERE_FIELDS = (
    "meaning",
    "omission",
    "negation",
    "actor_patient",
    "speaker_gender",
    "terminology",
)
HUMAN_NATURALNESS = "naturalness"
HUMAN_FLAG = "catastrophic"
HUMAN_TEXT_FIELDS = ("notes", "evidence")
HUMAN_LABELS = ("X", "Y", "Z", "W")


def read_candidate(config_path: Path | str, run_id: str, chapter: int) -> dict:
    """Read the frozen raw chapter candidate from isolated run roots.

    Reads ``manifest.json`` and the chapter ``source.txt``/``translation.md`` only;
    it never opens the run database or writes into a run directory.
    """
    config = load_config(config_path)
    run_dir = config.novel.output_dir / "runs" / safe_name(run_id)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    entry = next((c for c in manifest["chapters"] if c["number"] == chapter), None)
    if entry is None:
        raise ValueError(f"Chapter {chapter} is not in run {run_id}")
    stages = manifest["stages"]
    # A candidate layer is either a single raw translator stage, or a single
    # reviewer stage whose frozen parent is a translator baseline (derived
    # review). Both read the same rendered chapter ``translation.md``; the layer
    # is reported so callers can label ``translator_raw`` vs ``reviewed``.
    translator = len(stages) == 1 and stages[0]["role"] == "translator"
    reviewer = (
        len(stages) == 1
        and stages[0]["role"] == "reviewer"
        and manifest.get("workflow") == "review-direct"
        and "review_input" in manifest
    )
    if not translator and not reviewer:
        raise ValueError(f"Run {run_id} is not a single raw translator candidate layer")
    stage = stages[0]
    if translator:
        candidate_layer = "translator_raw"
    elif stage.get("review_contract") == SEMANTIC_EDIT_V1:
        # The bounded same-model source-aware semantic edit is its own layer,
        # never silently mixed with the generalized ``reviewed`` layer.
        candidate_layer = "semantic_edited"
    else:
        candidate_layer = "reviewed"
    source_path = run_dir / "chapters" / f"{chapter:04d}" / "source.txt"
    translation_path = run_dir / "chapters" / f"{chapter:04d}" / "translation.md"
    if not source_path.is_file() or not translation_path.is_file():
        raise ValueError(f"Chapter {chapter} artifacts are incomplete in run {run_id}")
    source = source_path.read_text("utf-8")
    translation = translation_path.read_text("utf-8")
    if sha256(source.encode("utf-8")) != entry["text_sha256"]:
        raise ValueError(f"Frozen source hash mismatch for chapter {chapter} in run {run_id}")
    return {
        "chapter": chapter,
        "source": source,
        "source_sha256": entry["source_sha256"],
        "text_sha256": entry["text_sha256"],
        "translation": translation,
        "translation_sha256": sha256(translation),
        "candidate_layer": candidate_layer,
        "workflow": manifest["workflow"],
        "prompt_version": stages[0]["prompt_version"],
        "runner": stages[0]["runner"],
        "model": stages[0]["runner_config"].get("model"),
        "harness": stages[0]["runner_config"].get("type"),
        "structured_decoding": stages[0]["runner_config"].get("structured_decoding", "prompt"),
        "retries": manifest["translation"]["retries"],
        "output_format": manifest["translation"]["output_format"],
    }


def build_blind_human_package(
    arms: list[dict], *, chapters: list[int], seed: int, directory: Path | str,
    source_hashes: dict[int, str] | None = None,
) -> dict:
    """Write a blind human-scoring package referencing isolated run roots.

    ``arms`` entries are ``{"arm": <hidden key>, "config": <path>, "run_id": ...}``.
    Candidate order is shuffled per chapter with the frozen ``seed`` and neutral
    labels are assigned by position; the label->arm mapping is written only to the
    hidden ``answer-key.json``.
    """
    if not 2 <= len(arms) <= len(HUMAN_LABELS):
        raise ValueError("Need 2-4 arms for a blind human comparison")
    if len({entry["arm"] for entry in arms}) != len(arms):
        raise ValueError("Arm keys must be unique")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    readings, identities = {}, {}
    for entry in arms:
        for chapter in chapters:
            candidate = read_candidate(entry["config"], entry["run_id"], chapter)
            readings[(entry["arm"], chapter)] = candidate
            identities[entry["arm"]] = {
                "run_id": entry["run_id"],
                "config": str(entry["config"]),
                "candidate_layer": candidate["candidate_layer"],
                "workflow": candidate["workflow"],
                "prompt_version": candidate["prompt_version"],
                "runner": candidate["runner"],
                "model": candidate["model"],
                "harness": candidate["harness"],
                "structured_decoding": candidate["structured_decoding"],
                "retries": candidate["retries"],
                "output_format": candidate["output_format"],
            }
    rng = random.Random(seed)
    cases, blocks, key_units, public_units = [], [], [], []
    for chapter in chapters:
        reference = readings[(arms[0]["arm"], chapter)]
        source, source_sha = reference["source"], reference["source_sha256"]
        if source_hashes is not None and source_hashes.get(chapter) != source_sha:
            raise ValueError(f"Source hash mismatch for chapter {chapter}")
        for entry in arms:
            candidate = readings[(entry["arm"], chapter)]
            if candidate["source_sha256"] != source_sha or candidate["source"] != source:
                raise ValueError(f"Source snapshots differ for chapter {chapter}")
        order = list(arms)
        rng.shuffle(order)
        case = {"chapter": chapter, "candidates": []}
        panels = [f"<section><h3>Japanese source</h3><pre>{escape(source)}</pre></section>"]
        for index, entry in enumerate(order):
            label = HUMAN_LABELS[index]
            candidate = readings[(entry["arm"], chapter)]
            case["candidates"].append(
                {
                    "label": label,
                    **{field: None for field in HUMAN_COUNT_FIELDS},
                    HUMAN_NATURALNESS: None,
                    HUMAN_FLAG: False,
                    **{field: "" for field in HUMAN_TEXT_FIELDS},
                }
            )
            panels.append(
                f"<section><h3>Candidate {label}</h3>"
                f"<pre>{escape(candidate['translation'])}</pre></section>"
            )
            key_units.append(
                {
                    "chapter": chapter,
                    "label": label,
                    "arm": entry["arm"],
                    "run_id": entry["run_id"],
                    "source_sha256": source_sha,
                    "text_sha256": reference["text_sha256"],
                    "translation_sha256": candidate["translation_sha256"],
                    "source_chars": len(source),
                }
            )
            public_units.append(
                {
                    "chapter": chapter,
                    "label": label,
                    "translation_sha256": candidate["translation_sha256"],
                    "source_chars": len(source),
                }
            )
        blocks.append((chapter, "".join(panels)))
        cases.append(case)
    bundle_id = sha256(canonical({"seed": seed, "units": key_units}))[:20]
    layers = {entry["arm"]: identities[entry["arm"]]["candidate_layer"] for entry in arms}
    layer_label = (
        next(iter(set(layers.values()))) if len(set(layers.values())) == 1 else "mixed"
    )
    scorecard = {"bundle_id": bundle_id, "seed": seed, "cases": cases}
    answer_key = {
        "bundle_id": bundle_id,
        "seed": seed,
        "candidate_layer": layer_label,
        "candidate_layers": layers,
        "chapters": list(chapters),
        "identities": identities,
        "units": key_units,
    }
    manifest = {
        "bundle_id": bundle_id,
        "seed": seed,
        "candidate_layer": layer_label,
        "chapters": list(chapters),
        "scoring_fields": {
            "counts": list(HUMAN_COUNT_FIELDS),
            "severe_semantic": list(HUMAN_SEVERE_FIELDS),
            "naturalness": f"{HUMAN_NATURALNESS} 0-5",
            "flag": HUMAN_FLAG,
            "text": list(HUMAN_TEXT_FIELDS),
        },
        "candidates": public_units,
        "note": "Arm identities are hidden in index.html; the mapping is answer-key.json.",
    }
    write_json(directory / "scorecard.json", scorecard)
    write_json(directory / "answer-key.json", answer_key)
    write_json(directory / "manifest.json", manifest)
    write_text(directory / "index.html", render_blind_human(scorecard, blocks))
    return {
        "directory": str(directory),
        "index": str(directory / "index.html"),
        "bundle_id": bundle_id,
        "chapters": list(chapters),
    }


def render_blind_human(scorecard: dict, texts: list[tuple[int, str]]) -> str:
    captions = {
        "meaning": "Meaning errors",
        "omission": "Omissions",
        "negation": "Negation errors",
        "actor_patient": "Actor/patient errors",
        "speaker_gender": "Speaker/gender errors",
        "terminology": "Major terminology errors",
        "corruption": "Corruption artifacts",
        "naturalness": "Naturalness/readability (0-5)",
        "notes": "Notes",
        "evidence": "Evidence span",
    }
    body = []
    for case, (chapter, panels) in zip(scorecard["cases"], texts, strict=True):
        forms = []
        for candidate in case["candidates"]:
            label = candidate["label"]
            counts = "".join(
                f'<label>{captions[field]}<input required type="number" min="0" step="1" '
                f'name="{chapter}:{label}:{field}"></label>'
                for field in HUMAN_COUNT_FIELDS
            )
            naturalness = (
                f'<label>{captions[HUMAN_NATURALNESS]}<input required type="number" min="0" '
                f'max="5" step="1" name="{chapter}:{label}:{HUMAN_NATURALNESS}"></label>'
            )
            flag = (
                f'<label class="flag"><input type="checkbox" name="{chapter}:{label}:{HUMAN_FLAG}"> '
                "Catastrophic failure (semantic collapse, meaning inversion, unusable)</label>"
            )
            text_fields = "".join(
                f'<label>{captions[field]}<textarea name="{chapter}:{label}:{field}" rows="2"></textarea></label>'
                for field in HUMAN_TEXT_FIELDS
            )
            forms.append(
                f'<fieldset><legend>Candidate {label}</legend>{counts}{naturalness}{flag}'
                f"{text_fields}</fieldset>"
            )
        body.append(
            f'<article><h2>Chapter {chapter}</h2><div class="panels">{panels}</div>'
            f'<div class="scores">{"".join(forms)}</div></article>'
        )
    template = json.dumps(scorecard, ensure_ascii=False).replace("<", "\\u003c")
    return (
        """<!doctype html><html lang="ru"><meta charset="utf-8"><title>Blind human scoring</title>
<style>body{font:16px/1.5 system-ui;margin:24px;background:#f4f6fa;color:#182332}article{margin:30px 0;padding:20px;background:white;border-radius:12px}.panels{display:flex;gap:20px;overflow:auto}.panels section{flex:1;min-width:320px}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}.scores{display:flex;flex-wrap:wrap;gap:16px}fieldset{flex:1;min-width:280px}label{display:block;margin:8px 0}input,textarea{display:block;padding:6px;max-width:220px;width:100%}input.flag,input[type=checkbox]{display:inline;width:auto}.flag{margin:12px 0}button{font:inherit;padding:12px 22px}h3{position:sticky;top:0;background:white}</style>
<h1>Blind human scoring of JA→RU drafts</h1><p>Candidate names, run IDs, models and retries are hidden. Score each candidate against the Japanese source. Counts are the number of separate defects; 0 means checked and none found. Naturalness 0 (unreadable) to 5 (natural). Tick catastrophic only for semantic collapse, meaning inversion, major omission or unusable corruption. One defect counts in one primary category. Notes/evidence optional. Download the JSON with the button; do not edit it by hand.</p><form id="scores">"""
        + "".join(body)
        + """<button>Download scores JSON</button></form><script>
const template="""
        + template
        + """;
document.querySelector('form').addEventListener('submit',event=>{event.preventDefault();const v=new FormData(event.target);const result=structuredClone(template);for(const c of result.cases){for(const cand of c.candidates){for(const k of cand){if(k==='label')continue;const name=`${c.chapter}:${cand.label}:${k}`;if(k==='catastrophic'){cand[k]=v.get(name)==='on';}else if(Object.prototype.hasOwnProperty.call(cand,k)&&typeof cand[k]==='boolean'){cand[k]=v.get(name)==='on';}else if(['notes','evidence'].includes(k)){cand[k]=String(v.get(name)||'');}else{cand[k]=Number(v.get(name));}}}}const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='scores-'+result.bundle_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});</script></html>"""
    )


def score_blind_human(
    directory: Path | str, scores: Path | str, *, assessment: str = "user_supplied"
) -> dict:
    """Validate a blind scorecard and summarize it per hidden arm.

    ``assessment`` labels the *type* of adjudication and must reflect the real
    source. A human scorecard stays ``user_supplied``; an external blind model
    review must pass ``assessment="external_ai_blind"`` and is never relabelled as
    human evidence.
    """
    directory, scores = Path(directory), Path(scores)
    key = json.loads((directory / "answer-key.json").read_text("utf-8"))
    result = json.loads(scores.read_text("utf-8"))
    if (
        not isinstance(result, dict)
        or result.get("bundle_id") != key["bundle_id"]
        or not isinstance(result.get("cases"), list)
    ):
        raise ValueError("Scorecard belongs to another evaluation")
    lookup = {(unit["chapter"], unit["label"]): unit for unit in key["units"]}
    expected, seen, summaries = set(lookup), set(), {}
    for case in result["cases"]:
        if (
            not isinstance(case, dict)
            or set(case) != {"chapter", "candidates"}
            or type(case["chapter"]) is not int
            or not isinstance(case["candidates"], list)
        ):
            raise ValueError("Invalid scorecard case")
        for candidate in case["candidates"]:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("label"), str):
                raise ValueError("Invalid candidate score")  # noqa: TRY004 - invalid JSON input
            pair = (case["chapter"], candidate["label"])
            if pair not in expected or pair in seen:
                raise ValueError("Duplicate or unknown evaluation candidate")
            seen.add(pair)
            for field in HUMAN_COUNT_FIELDS:
                value = candidate.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"Complete every count with a nonnegative integer: {field}")
            naturalness = candidate.get(HUMAN_NATURALNESS)
            if isinstance(naturalness, bool) or not isinstance(naturalness, int) or not (
                0 <= naturalness <= 5
            ):
                raise ValueError("Naturalness must be an integer 0-5")
            if not isinstance(candidate.get(HUMAN_FLAG), bool):
                raise ValueError("Catastrophic must be a boolean")  # noqa: TRY004
            for field in HUMAN_TEXT_FIELDS:
                value = candidate.get(field, "")
                if value is None:
                    value = ""
                if not isinstance(value, str):
                    raise ValueError(f"{field} must be a string")  # noqa: TRY004
            unit = lookup[pair]
            summary = summaries.setdefault(
                unit["arm"],
                {
                    "run_id": unit["run_id"],
                    "severity": {field: 0 for field in HUMAN_SEVERE_FIELDS},
                    "corruption": 0,
                    "total_severe": 0,
                    "catastrophic_chapters": [],
                    "naturalness": {},
                    "chapters": {},
                },
            )
            chapter_summary = summary["chapters"].setdefault(
                case["chapter"],
                {"total_severe": 0, "corruption": 0, "catastrophic": False, "naturalness": naturalness},
            )
            for field in HUMAN_SEVERE_FIELDS:
                summary["severity"][field] += candidate[field]
                summary["total_severe"] += candidate[field]
                chapter_summary["total_severe"] += candidate[field]
            summary["corruption"] += candidate["corruption"]
            chapter_summary["corruption"] += candidate["corruption"]
            if candidate[HUMAN_FLAG]:
                summary["catastrophic_chapters"].append(case["chapter"])
                chapter_summary["catastrophic"] = True
            summary["naturalness"][case["chapter"]] = naturalness
    if seen != expected:
        raise ValueError("Scorecard is incomplete")
    return {
        "bundle_id": key["bundle_id"],
        "assessment": assessment,
        "candidate_layer": key["candidate_layer"],
        "arms": summaries,
    }
