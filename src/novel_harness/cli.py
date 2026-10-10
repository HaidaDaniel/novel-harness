"""Thin CLI adapters. JSON output is stable and contains no progress chatter."""

import json
import sys
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
from filelock import Timeout
from pydantic import ValidationError
from rich.console import Console

from .cli_context import (
    has_explicit_config,
    resolve_config,
    resolve_config_for_run,
    set_root_config,
)
from .cli_selection import complete_run_ids, resolve_run_id
from .config import load_config
from .runners.base import RunnerExecutionError
from .service import TranslationService
from .source_import import parse_chapter_numbers

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Resumable Japanese → Russian translation harness",
)
library_app = typer.Typer(no_args_is_help=True, help="Novel Library admin and recovery")
app.add_typer(library_app, name="library", rich_help_panel="Library")
connections_app = typer.Typer(no_args_is_help=True, help="Connection and execution profiles")
app.add_typer(connections_app, name="connections", rich_help_panel="Configuration")
experiment_app = typer.Typer(no_args_is_help=True, help="Bounded generation-matrix control")
app.add_typer(experiment_app, name="experiment", rich_help_panel="Research and review")
console = Console()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    app()


def handled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            if "config" in kwargs:
                selector = kwargs.get("run_id") or kwargs.get("run_option")
                if selector is not None:
                    selected = (
                        resolve_config(kwargs["config"])
                        if has_explicit_config(kwargs["config"])
                        else None
                    )
                    kwargs["config"] = resolve_config_for_run(selector, selected)
                else:
                    kwargs["config"] = resolve_config(kwargs["config"])
            return function(*args, **kwargs)
        except ValidationError as error:
            # Pydantic's default error includes input_value, which could contain API keys.
            from .cli_output import sanitize_error

            payload = {
                "error": "Invalid configuration",
                "details": [
                    {"loc": list(e["loc"]), "msg": sanitize_error(e["msg"])}
                    for e in error.errors()
                ],
            }
            if kwargs.get("as_json", False):
                typer.echo(json.dumps(payload, ensure_ascii=False), err=True)
            else:
                typer.echo(f"Error: {payload['error']}. Check the configuration and try again.", err=True)
            raise typer.Exit(2) from error
        except (ValueError, OSError, RunnerExecutionError, Timeout) as error:
            from .cli_output import sanitize_error

            if kwargs.get("as_json", False):
                typer.echo(
                    json.dumps({"error": sanitize_error(error)}, ensure_ascii=False),
                    err=True,
                )
            else:
                from .cli_output import render_error

                typer.echo(render_error(error), err=True)
            raise typer.Exit(2) from error

    return wrapped


ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML config path")
]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable JSON")]


@app.callback()
def configure_cli(config: ConfigOption = None):
    """Set the optional config path placed before a command."""
    set_root_config(config)


RunIdArgument = Annotated[
    str | None, typer.Argument(shell_complete=complete_run_ids)
]
RunOption = Annotated[
    str | None, typer.Option("--run", shell_complete=complete_run_ids)
]


def output(value, as_json: bool):
    if as_json:
        typer.echo(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        from .cli_output import render

        console.print(render(value), markup=False)


@library_app.command("list")
@handled
def library_list(config: ConfigOption = None, as_json: JsonOption = False):
    """List indexed source novels and their saved-run progress."""
    output(TranslationService.from_config(config).library_novels(), as_json)


@library_app.command("remaining")
@handled
def library_remaining(config: ConfigOption = None, as_json: JsonOption = False):
    """List ready novels with untranslated source chapters."""
    output(TranslationService.from_config(config).library_remaining(), as_json)


@library_app.command("launch-remaining")
@handled
def library_launch_remaining(
    novel_ids: Annotated[
        list[str] | None, typer.Option("--novel-id", help="Stable ID; repeat for several novels")
    ] = None,
    all_remaining: Annotated[bool, typer.Option("--all")] = False,
    preset: Annotated[str | None, typer.Option("--preset")] = None,
    runner: Annotated[str | None, typer.Option()] = None,
    pipeline: Annotated[str | None, typer.Option()] = None,
    translator: Annotated[str | None, typer.Option()] = None,
    editor: Annotated[str | None, typer.Option()] = None,
    force: Annotated[bool, typer.Option()] = False,
    chunk_chars: Annotated[int | None, typer.Option()] = None,
    temperature: Annotated[float | None, typer.Option()] = None,
    prompt_version: Annotated[str | None, typer.Option()] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Launch workers for selected remaining novels without inline inference."""
    if all_remaining == bool(novel_ids):
        raise ValueError("Choose exactly one of --all or --novel-id")
    _preset_conflicts(preset, runner, pipeline, translator, editor, prompt_version)
    service = TranslationService.from_config(config)
    service.rescan_library()
    remaining = service.library_remaining()["novels"]
    available = {item["novel_id"]: item for item in remaining}
    selected = sorted(available) if all_remaining else sorted(set(novel_ids or []))
    unknown = [novel_id for novel_id in selected if novel_id not in available]
    if unknown:
        raise ValueError(f"Novel is not remaining or selectable: {', '.join(unknown)}")
    results = []
    for novel_id in selected:
        results.append(
            service.launch_remaining_novel(
                novel_id,
                config,
                preset=preset,
                runner=runner,
                pipeline=pipeline,
                translator=translator,
                editor=editor,
                force=force,
                chunk_chars=chunk_chars,
                temperature=temperature,
                prompt_version=prompt_version,
            )
        )
    output({"novels": results}, as_json)


@library_app.command("show")
@handled
def library_show(
    novel_id: str,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Show one source novel by stable ID."""
    output(TranslationService.from_config(config).library_show(novel_id), as_json)


@library_app.command("rescan")
@handled
def library_rescan(config: ConfigOption = None, as_json: JsonOption = False):
    """Explicitly reconcile the library index with configured filesystem roots."""
    output(TranslationService.from_config(config).rescan_library(), as_json)


@library_app.command("import-narou")
@handled
def library_import_narou(
    source: Annotated[Path, typer.Option("--source", help="Completed downloader output directory")],
    ncode: Annotated[str, typer.Option("--ncode", help="Narou work identifier")],
    title: Annotated[str, typer.Option("--title", help="Work title for the Library card")],
    expected_chapters: Annotated[
        str,
        typer.Option(
            "--expected-chapters",
            help="Required comma-separated chapter numbers or inclusive ranges (for example 1-24)",
        ),
    ],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Validate and copy completed local Narou files into the configured Library."""
    output(
        TranslationService.from_config(config).import_narou_source(
            source,
            ncode=ncode,
            title=title,
            expected_chapters=parse_chapter_numbers(expected_chapters),
        ),
        as_json,
    )


@library_app.command("import-epub")
@handled
def library_import_epub(
    source: Annotated[Path, typer.Option("--source", help="Local Japanese source EPUB file")],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Import a local Japanese EPUB while keeping its original archive for round-trip."""
    output(
        TranslationService.from_config(config).import_epub_source(source),
        as_json,
    )


@library_app.command("invoke-narou")
@handled
def library_invoke_narou(
    ncode: Annotated[str, typer.Option("--ncode", help="Narou work identifier")],
    source_url: Annotated[
        str, typer.Option("--url", help="HTTPS URL for an episode in this work")
    ],
    title: Annotated[str, typer.Option("--title", help="Work title for the Library card")],
    expected_chapters: Annotated[
        str,
        typer.Option(
            "--expected-chapters",
            help="Expected chapter numbers or ranges (for example 1-24)",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", help="New or empty downloader output directory")
    ],
    command_json: Annotated[
        str,
        typer.Option(
            "--command-json",
            help=(
                "JSON argv array; use {url}, {ncode} and {output_dir} tokens; no shell is used"
            ),
        ),
    ],
    timeout: Annotated[
        float, typer.Option("--timeout", help="Maximum downloader runtime in seconds")
    ] = 1800,
    cwd: Annotated[
        Path | None, typer.Option("--cwd", help="Optional downloader working directory")
    ] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Run a protocol-v1 Narou downloader and import only verified complete output."""
    try:
        command = json.loads(command_json)
    except json.JSONDecodeError as error:
        raise ValueError("--command-json must be a JSON array of argument strings") from error
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError("--command-json must be a JSON array of argument strings")
    output(
        TranslationService.from_config(config).invoke_narou_downloader(
            command,
            ncode=ncode,
            source_url=source_url,
            title=title,
            expected_chapters=parse_chapter_numbers(expected_chapters),
            output_dir=output_dir,
            timeout_seconds=timeout,
            cwd=cwd,
        ),
        as_json,
    )


@connections_app.command("list")
@handled
def connections_list(
    config: ConfigOption = None, as_json: JsonOption = False
):
    """Show local harness discovery and configured secret-free profiles."""
    output(TranslationService.from_config(config).connections(), as_json)


@connections_app.command("import-opencode")
@handled
def connections_import_opencode(
    profile_id: Annotated[str, typer.Option("--profile-id")],
    agent_id: Annotated[str, typer.Option("--agent")],
    advanced: Annotated[bool, typer.Option("--advanced")] = False,
    model_override: Annotated[
        str | None, typer.Option("--model", help="Advanced explicit provider/model override")
    ] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Preview a safe OpenCode profile import; config is not written automatically."""
    output(
        TranslationService.from_config(config).import_opencode_connection(
            profile_id, agent_id, advanced=advanced, model_override=model_override
        ),
        as_json,
    )


@connections_app.command("import-direct")
@handled
def connections_import_direct(
    profile_id: Annotated[str, typer.Option("--profile-id")],
    model: Annotated[str, typer.Option("--model")],
    base_url: Annotated[str, typer.Option("--base-url")],
    api_key_env: Annotated[str, typer.Option("--api-key-env")],
    timeout_seconds: Annotated[float, typer.Option("--timeout")] = 180,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Preview a direct API profile using an environment-variable name for its key."""
    output(
        TranslationService.from_config(config).import_direct_connection(
            profile_id,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
        ),
        as_json,
    )


@connections_app.command("import-codex")
@handled
def connections_import_codex(
    profile_id: Annotated[str, typer.Option("--profile-id")],
    codex_profile: Annotated[str | None, typer.Option("--profile")] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Preview a Codex profile reference without inspecting its credentials."""
    output(
        TranslationService.from_config(config).import_codex_connection(
            profile_id, codex_profile
        ),
        as_json,
    )


@experiment_app.command("plan")
@handled
def experiment_plan(
    matrix: Annotated[Path, typer.Option("--matrix")],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Freeze a bounded matrix and create its ordinary persisted runs."""
    from .experiment_runs import plan_matrix

    output(plan_matrix(TranslationService.from_config(config), matrix), as_json)


@experiment_app.command("preflight")
@handled
def experiment_preflight(
    plan: Annotated[Path, typer.Option("--plan")],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Prove frozen arm settings without making model calls."""
    from .experiment_runs import preflight_matrix

    output(preflight_matrix(TranslationService.from_config(config), plan), as_json)


@experiment_app.command("run")
@handled
def experiment_run(
    plan: Annotated[Path, typer.Option("--plan")],
    arms: Annotated[str | None, typer.Option("--arms")] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Start selected frozen arms in the existing independent worker."""
    from .experiment_runs import run_matrix

    selected = {item.strip() for item in arms.split(",")} if arms else None
    output(run_matrix(TranslationService.from_config(config), config, plan, arm_ids=selected), as_json)


@experiment_app.command("status")
@handled
def experiment_status(
    plan: Annotated[Path, typer.Option("--plan")],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Show exact per-arm progress from persisted run state."""
    from .experiment_runs import status_matrix

    output(status_matrix(TranslationService.from_config(config), plan), as_json)


@experiment_app.command("resume")
@handled
def experiment_resume(
    plan: Annotated[Path, typer.Option("--plan")],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Resume every unfinished arm without regenerating valid chunks."""
    from .experiment_runs import resume_matrix

    output(resume_matrix(TranslationService.from_config(config), config, plan), as_json)


@experiment_app.command("export")
@handled
def experiment_export(
    plan: Annotated[Path, typer.Option("--plan")],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Export per-arm progress, attempts, failures and provenance."""
    from .experiment_runs import export_matrix

    output(export_matrix(TranslationService.from_config(config), plan), as_json)


def parse_chapter_list(chapters_csv: str) -> list[int]:
    """Parse a comma-separated sparse chapter list; reject duplicates and negatives."""
    selected: list[int] = []
    seen: set[int] = set()
    for token in chapters_csv.split(","):
        text = token.strip()
        if not text:
            raise ValueError("Chapters must be comma-separated numbers")
        try:
            number = int(text)
        except ValueError as error:
            raise ValueError("Chapters must be comma-separated numbers") from error
        if number < 0:
            raise ValueError("Chapter numbers must be nonnegative")
        if number in seen:
            raise ValueError(f"Duplicate chapter in chapter list: {number}")
        seen.add(number)
        selected.append(number)
    if not selected:
        raise ValueError("Chapters must be comma-separated numbers")
    return selected


def selection(
    chapter: int | None,
    chapter_range: str | None,
    all_chapters: bool,
    chapters_csv: str | None = None,
):
    if sum(
        (
            chapter is not None,
            chapter_range is not None,
            all_chapters,
            chapters_csv is not None,
        )
    ) != 1:
        raise ValueError("Specify exactly one of --chapter, --range, --all, --chapters")
    if chapter is not None:
        if chapter < 0:
            raise ValueError("Chapter number must be nonnegative")
        return [chapter]
    if chapters_csv is not None:
        return parse_chapter_list(chapters_csv)
    if chapter_range is not None:
        import re

        match = re.fullmatch(r"(\d+)-(\d+)", chapter_range)
        if not match or int(match[1]) > int(match[2]):
            raise ValueError("Range must be START-END, for example 1-10")
        if int(match[2]) - int(match[1]) > 100000:
            raise ValueError("Chapter range is too large")
        return list(range(int(match[1]), int(match[2]) + 1))
    return None


def _preset_conflicts(
    preset: str | None,
    runner: str | None,
    pipeline: str | None,
    translator: str | None,
    editor: str | None,
    prompt_version: str | None,
) -> None:
    """Fail closed when --preset is combined with explicit raw selections."""
    if preset is None:
        return
    if any(
        value is not None
        for value in (runner, pipeline, translator, editor, prompt_version)
    ):
        raise ValueError(
            "--preset cannot be combined with explicit "
            "runner/pipeline/translator/editor/prompt_version"
        )


@app.command(rich_help_panel="Configuration")
@handled
def runners(config: ConfigOption = None, as_json: JsonOption = False):
    """List profiles and configuration availability (without API calls)."""
    output(TranslationService.from_config(config).runners(), as_json)


@app.command("presets", rich_help_panel="Configuration")
@handled
def presets(config: ConfigOption = None, as_json: JsonOption = False):
    """List user-facing translation modes; never calls a model endpoint."""
    service = TranslationService.from_config(config)
    output({"presets": service.presets(), "default": service.default_preset()}, as_json)


@app.command("doctor", rich_help_panel="Configuration")
@handled
def doctor(
    config: ConfigOption = None,
    as_json: JsonOption = False,
    timeout: Annotated[float, typer.Option("--timeout", help="Per-probe timeout seconds")] = 3.0,
):
    """Check runner readiness without invoking model inference."""
    from .doctor import diagnose

    output(diagnose(load_config(config), endpoint_timeout=timeout), as_json)


@app.command("translate", rich_help_panel="Translation control")
@app.command("run", rich_help_panel="Translation control")
@handled
def run(
    preset: Annotated[
        str | None, typer.Option("--preset", help="User-facing translation mode")
    ] = None,
    execution_profile: Annotated[
        str | None, typer.Option("--execution-profile", help="Configured execution profile")
    ] = None,
    runner: Annotated[str | None, typer.Option()] = None,
    pipeline: Annotated[str | None, typer.Option()] = None,
    translator: Annotated[str | None, typer.Option()] = None,
    editor: Annotated[str | None, typer.Option()] = None,
    chapter: Annotated[int | None, typer.Option()] = None,
    chapter_range: Annotated[str | None, typer.Option("--range")] = None,
    all_chapters: Annotated[bool, typer.Option("--all")] = False,
    force: Annotated[bool, typer.Option()] = False,
    chunk_chars: Annotated[int | None, typer.Option()] = None,
    temperature: Annotated[float | None, typer.Option()] = None,
    prompt_version: Annotated[str | None, typer.Option()] = None,
    novel_id: Annotated[
        str | None, typer.Option("--novel-id", help="Stable Novel Library ID")
    ] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Translate selected chapters; identical invocation resumes automatically."""
    _preset_conflicts(preset, runner, pipeline, translator, editor, prompt_version)
    if preset and execution_profile:
        raise ValueError("--preset cannot be combined with --execution-profile")
    chapters = selection(chapter, chapter_range, all_chapters)
    service = TranslationService.from_config(config)
    run_id = service.create_run(
        execution_profile=execution_profile,
        preset=preset,
        runner=runner,
        pipeline=pipeline,
        translator=translator,
        editor=editor,
        chapters=chapters,
        force=force,
        chunk_chars=chunk_chars,
        temperature=temperature,
        prompt_version=prompt_version,
        novel_id=novel_id,
    )
    if not as_json:
        console.print(f"Run: {run_id}")
    result = service.execute(run_id)
    output(result, as_json)
    if result["status"] != "completed":
        raise typer.Exit(1)


@app.command("sample", rich_help_panel="Translation control")
@handled
def sample(
    preset: Annotated[
        str | None, typer.Option("--preset", help="User-facing translation mode")
    ] = None,
    execution_profile: Annotated[
        str | None, typer.Option("--execution-profile", help="Configured execution profile")
    ] = None,
    runner: Annotated[str | None, typer.Option()] = None,
    pipeline: Annotated[str | None, typer.Option()] = None,
    chapters: Annotated[
        str | None,
        typer.Option("--chapters", help="one or two chapters, for example 1 or 1,2"),
    ] = None,
    chunk_chars: Annotated[int | None, typer.Option()] = None,
    temperature: Annotated[float | None, typer.Option()] = None,
    prompt_version: Annotated[str | None, typer.Option()] = None,
    novel_id: Annotated[
        str | None, typer.Option("--novel-id", help="Stable Novel Library ID")
    ] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Create a small frozen sample run for review; whole-book launch stays ungated."""
    _preset_conflicts(preset, runner, pipeline, None, None, prompt_version)
    if preset and execution_profile:
        raise ValueError("--preset cannot be combined with --execution-profile")
    service = TranslationService.from_config(config)
    selected = None
    if chapters:
        try:
            selected = [int(token.strip()) for token in chapters.split(",") if token.strip()]
        except ValueError as error:
            raise ValueError("Chapters must be comma-separated numbers") from error
    run_id = service.create_sample_run(
        execution_profile=execution_profile,
        preset=preset,
        runner=runner,
        pipeline=pipeline,
        chapters=selected,
        chunk_chars=chunk_chars,
        temperature=temperature,
        prompt_version=prompt_version,
        novel_id=novel_id,
    )
    output(
        {"run_id": run_id, "sample": service.repository.run(run_id)["manifest"].get("sample")},
        as_json,
    )


@app.command("review", rich_help_panel="Research and review")
@handled
def review(
    run_id: RunIdArgument = None,
    run_option: Annotated[str | None, typer.Option("--run", help="Exact direct parent run", shell_complete=complete_run_ids)] = None,
    reviewer: Annotated[str | None, typer.Option(help="Reviewer profile")] = None,
    chapters: Annotated[str | None, typer.Option(help="Comma-separated chapter numbers; default all")] = None,
    force: Annotated[bool, typer.Option()] = False,
    prompt_version: Annotated[str, typer.Option()] = "review-v1",
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Prepare an immutable derived review; use start to execute in the worker."""
    if reviewer is None:
        raise ValueError("Missing option '--reviewer'")
    selected = None
    if chapters is not None:
        try:
            selected = [int(c.strip()) for c in chapters.split(",")]
        except ValueError as error:
            raise ValueError("Chapters must be comma-separated numbers") from error
        if not selected or any(c < 0 for c in selected):
            raise ValueError("Select nonnegative chapter numbers")
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    derived = service.create_review_run(run_id, reviewer=reviewer, chapters=selected,
                                       force=force, prompt_version=prompt_version)
    output(service.status(derived), as_json)


@app.command("context-generate", rich_help_panel="Research and review")
@handled
def context_generate(
    runner: Annotated[str, typer.Option(help="Runner profile used to generate context")],
    out: Annotated[Path, typer.Option("--out", help="Frozen artifact output path")],
    chapter: Annotated[int | None, typer.Option()] = None,
    chapter_range: Annotated[str | None, typer.Option("--range")] = None,
    all_chapters: Annotated[bool, typer.Option("--all")] = False,
    chapters_csv: Annotated[
        str | None,
        typer.Option(
            "--chapters",
            help="Comma-separated sparse chapter list, for example 33,49,63,91,217; duplicates rejected",
        ),
    ] = None,
    horizon: Annotated[str, typer.Option("--horizon")] = "chapter_preflight",
    prompt_version: Annotated[str, typer.Option()] = "context-v5",
    retry_ceiling: Annotated[int, typer.Option("--retry-ceiling")] = 2,
    prior: Annotated[Path | None, typer.Option("--prior", help="Frozen artifact for prior self-context")] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Generate and freeze a source-only self-context artifact. Requires a model endpoint."""
    from .artifacts import sha256
    from .context_runs import (
        generate_artifact,
        identity_from_runner,
        load_artifact,
        write_artifact,
    )

    chapters = selection(chapter, chapter_range, all_chapters, chapters_csv)
    service = TranslationService.from_config(config)
    cfg = service.config
    if runner not in cfg.runners:
        raise ValueError(f"Unknown runner: {runner}")
    profile = cfg.runners[runner]
    if profile.structured_decoding != "json-schema":
        raise ValueError("Context generation requires structured_decoding: json-schema")
    resolved, _identity = identity_from_runner(profile.public_settings())
    prompt = (cfg.translation.prompts_dir / f"{prompt_version}.md").read_text("utf-8")
    if not prompt.strip():
        raise ValueError(f"Empty context prompt: {prompt_version}")
    prior_artifact = None
    if prior is not None:
        prior_path = Path(prior)
        prior_artifact = (load_artifact(prior_path), str(prior_path))
    artifact = generate_artifact(
        cfg.novel.source_dir,
        chapters=chapters,
        horizon_mode=horizon,
        call=service.runner_factory(runner).run,
        generator={
            "provider": profile.type,
            "configured_model": resolved,
            "resolved_model_id": resolved,
            "weights_sha256": None,
            "quantization": (profile.extra_params or {}).get("quantization"),
            "harness": profile.type,
            "profile": f"context-{horizon}-v1",
        },
        prompt=prompt,
        prompt_version=prompt_version,
        retry_ceiling=retry_ceiling,
        prior_artifact=prior_artifact,
        runner_decoding={
            "mode": profile.structured_decoding,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
        },
    )
    snapshots = [s["target_chapter"] for s in artifact["snapshots"]]
    if chapters is not None and sorted(snapshots) != sorted(chapters):
        raise ValueError(
            "Generated context artifact does not match the selected chapter set: "
            f"expected {sorted(chapters)}, got {sorted(snapshots)}"
        )
    write_artifact(out, artifact)
    output(
        {
            "artifact_path": str(out),
            "artifact_sha256": artifact["artifact_sha256"],
            "prompt_sha256": sha256(prompt),
            "horizon_mode": artifact["horizon_mode"],
            "snapshots": snapshots,
        },
        as_json,
    )


@app.command(rich_help_panel="Translation control")
@handled
def status(
    config: ConfigOption = None,
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    as_json: JsonOption = False,
):
    """Inspect persisted progress."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, required=False, repository=getattr(service, "repository", None)
    )
    if as_json:
        output(service.status(run_id), True)
    elif run_id is not None:
        from .progress import render_rich, snapshot

        console.print(render_rich(snapshot(service, run_id)))
    else:
        from .run_dashboard import DashboardState, render_page, run_rows

        console.print(render_page(service, run_rows(service), DashboardState(), console, once=True))


@app.command("dashboard", rich_help_panel="Translation control")
@handled
def dashboard_command(
    config: ConfigOption = None,
    interval: Annotated[float, typer.Option("--interval")] = 5.0,
    page: Annotated[int, typer.Option("--page")] = 1,
    once: Annotated[bool, typer.Option("--once")] = False,
):
    """Browse saved translations in a read-only, paginated terminal table."""
    from .run_dashboard import dashboard

    dashboard(TranslationService.from_config(config), interval=interval, once=once, page=page)


@app.command("tui", rich_help_panel="Translation control")
@handled
def tui_command(config: ConfigOption = None):
    """Interactively control persisted workers without running inference inline."""
    from .terminal_ui import tui

    tui(TranslationService.from_config(config), config)


@app.command("resume", rich_help_panel="Translation control")
@app.command("retry-failed", rich_help_panel="Translation control")
@handled
def resume(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    background: Annotated[bool, typer.Option("--background")] = False,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Resume a run, preserving successful chunks and successful earlier stages."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    if background:
        from .launch import launch_run

        result = launch_run(service, config, run_id)
        output(result, as_json)
        if not result.get("started") and not result.get("already_running") and not result.get(
            "completed"
        ):
            raise typer.Exit(1)
        return
    result = service.resume(run_id)
    output(result, as_json)
    if result["status"] != "completed":
        raise typer.Exit(1)


@app.command("chapter", rich_help_panel="Translation control")
@handled
def inspect_chapter(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    chapter: Annotated[int | None, typer.Option()] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Read the frozen original and current translation."""
    if chapter is None:
        raise ValueError("Missing option '--chapter'")
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    output(service.inspect_chapter(run_id, chapter), as_json)


@app.command("continue", rich_help_panel="Translation control")
@handled
def continue_run(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    runner: Annotated[str | None, typer.Option()] = None,
    chunk_chars: Annotated[int, typer.Option()] = 1500,
    editor: Annotated[str | None, typer.Option()] = None,
    pipeline: Annotated[str | None, typer.Option()] = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Create a continuation with a new runner; inherit verified completed chunks."""
    if runner is None:
        raise ValueError("Missing option '--runner'")
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    new_id = service.continue_run(
        run_id, runner=runner, chunk_chars=chunk_chars, editor=editor, pipeline=pipeline
    )
    service.activate(new_id, config)
    output(service.status(new_id), as_json)


@app.command("start", rich_help_panel="Translation control")
@handled
def start(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    monitor: Annotated[bool, typer.Option("--monitor")] = False,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Start/resume a persisted run in an independent background worker."""
    from .launch import launch_run

    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    output(launch_run(service, config, run_id, monitor=monitor), as_json)


@app.command("activate", rich_help_panel="Translation control")
@handled
def activate(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Select a saved run for its harness; do not start or resume it."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    output(service.activate(run_id, config), as_json)


@app.command("pause", rich_help_panel="Translation control")
@handled
def pause(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Pause an active worker after its current chunk; use resume to continue."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    output(service.pause(run_id), as_json)


@app.command("build", rich_help_panel="Translation control")
@handled
def build(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
    metadata_path: Annotated[
        Path | None,
        typer.Option("--metadata", help="Explicit shared book metadata for a legacy export"),
    ] = None,
):
    """Build a deterministic EPUB from a complete run."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    path = service.build_epub(run_id, metadata_path=metadata_path)
    metadata = json.loads((service.run_dir(run_id) / "book.json").read_text("utf-8"))
    output(
        {
            "path": str(path),
            "sha256": metadata["sha256"],
            "warnings": metadata["warnings"],
            "name_revision": metadata.get("name_revision"),
            "novel": metadata.get("novel"),
        },
        as_json,
    )


@app.command("revise-names", rich_help_panel="Translation control")
@handled
def revise_names(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Save an audited name revision/partial preview; preserve raw translations."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    report = service.revise_names(run_id)
    output({k: v for k, v in report.items() if k not in {"chunks", "rules", "chapters"}}, as_json)


@app.command("audit", rich_help_panel="Translation control")
@handled
def audit(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Check saved text without changing it or invoking a model."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    result = service.audit(run_id)
    output(result, as_json)
    if result["blocking_chapters"]:
        raise typer.Exit(1)


@app.command("metrics", rich_help_panel="Translation control")
@handled
def metrics(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Measured stage timing, usage coverage, effective settings and provenance."""
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    output(service.metrics(run_id), as_json)


@app.command("plan-evaluation", rich_help_panel="Research and review")
@handled
def plan_evaluation(
    matrix: Annotated[Path, typer.Option()],
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Freeze candidate runs on identical chapters; make no model calls."""
    from .evaluation import prepare_plan

    output(prepare_plan(TranslationService.from_config(config), matrix), as_json)


@app.command("evaluate", rich_help_panel="Research and review")
@handled
def evaluate(
    runs: Annotated[str, typer.Option()],
    count: Annotated[int, typer.Option()] = 5,
    seed: Annotated[int, typer.Option()] = 20260908,
    exclude: Annotated[str, typer.Option()] = "",
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Export a seeded blind comparison and human score form from saved results."""
    from .evaluation import export_evaluation

    result = export_evaluation(
        TranslationService.from_config(config),
        [r.strip() for r in runs.split(",")],
        count=count,
        seed=seed,
        exclude=[int(n) for n in exclude.split(",") if n.strip()],
    )
    output(result, as_json)


@app.command("score-evaluation", rich_help_panel="Research and review")
@handled
def score_evaluation(
    directory: Annotated[Path, typer.Option()],
    scores: Annotated[Path, typer.Option()],
    as_json: JsonOption = False,
):
    """Aggregate completed human scores; blank ratings are never treated as zero."""
    from .evaluation import score_evaluation as score

    output(score(directory, scores), as_json)


@app.command("compare", rich_help_panel="Research and review")
@handled
def compare(
    chapter: Annotated[int, typer.Option()],
    runs: Annotated[str, typer.Option()],
    config: ConfigOption = None,
    as_json: JsonOption = False,
    open_browser: Annotated[bool, typer.Option("--open")] = False,
):
    """Write a standalone HTML original/translation comparison."""
    service = TranslationService.from_config(config)
    run_ids = [run_id.strip() for run_id in runs.split(",")]
    path = service.compare_html(chapter, run_ids)
    output({"path": str(path), "chapter": chapter, "runs": run_ids}, as_json)
    if open_browser:
        import webbrowser

        webbrowser.open(path.as_uri())


@app.command("serve", rich_help_panel="Configuration")
@handled
def serve(
    config: ConfigOption = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
):
    """Serve the developer/research API (requires the `web` extra)."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(config), host=host, port=port)


@app.command("mcp", rich_help_panel="Configuration")
@handled
def mcp(config: ConfigOption = None):
    """Serve the optional MCP stdio adapter over the same service orchestration."""
    from .mcp import serve_stdio

    serve_stdio(TranslationService.from_config(config), config_path=config)


@app.command("watch", rich_help_panel="Translation control")
@handled
def watch(
    run_id: RunIdArgument = None,
    run_option: RunOption = None,
    config: ConfigOption = None,
    interval: Annotated[float, typer.Option()] = 2.0,
    once: Annotated[bool, typer.Option("--once")] = False,
    as_json: JsonOption = False,
):
    """Read-only live progress monitor; never calls a model or mutates state."""
    from .monitor import MIN_INTERVAL
    from .monitor import watch as watch_run
    from .progress import snapshot

    if interval < MIN_INTERVAL:
        raise ValueError(f"interval must be at least {MIN_INTERVAL} seconds")
    if as_json and not once:
        raise ValueError("--json requires --once; use `novel watch --once --json`")
    service = TranslationService.from_config(config)
    run_id = resolve_run_id(
        run_id, run_option, repository=getattr(service, "repository", None)
    )
    if once:
        output(snapshot(service, run_id), as_json)
        return
    watch_run(service, run_id, interval=interval)


@app.command("launch", rich_help_panel="Translation control")
@handled
def launch(
    preset: Annotated[
        str | None, typer.Option("--preset", help="User-facing translation mode")
    ] = None,
    runner: Annotated[str | None, typer.Option()] = None,
    pipeline: Annotated[str | None, typer.Option()] = None,
    translator: Annotated[str | None, typer.Option()] = None,
    editor: Annotated[str | None, typer.Option()] = None,
    chapter: Annotated[int | None, typer.Option()] = None,
    chapter_range: Annotated[str | None, typer.Option("--range")] = None,
    all_chapters: Annotated[bool, typer.Option("--all")] = False,
    force: Annotated[bool, typer.Option()] = False,
    chunk_chars: Annotated[int | None, typer.Option()] = None,
    temperature: Annotated[float | None, typer.Option()] = None,
    prompt_version: Annotated[str | None, typer.Option()] = None,
    novel_id: Annotated[
        str | None, typer.Option("--novel-id", help="Stable Novel Library ID")
    ] = None,
    monitor: Annotated[bool, typer.Option("--monitor")] = False,
    config: ConfigOption = None,
    as_json: JsonOption = False,
):
    """Create/reuse a run, start the independent worker, return immediately."""
    from .launch import launch_run

    _preset_conflicts(preset, runner, pipeline, translator, editor, prompt_version)
    chapters = selection(chapter, chapter_range, all_chapters)
    service = TranslationService.from_config(config)
    run_id = service.create_run(
        preset=preset,
        runner=runner,
        pipeline=pipeline,
        translator=translator,
        editor=editor,
        chapters=chapters,
        force=force,
        chunk_chars=chunk_chars,
        temperature=temperature,
        prompt_version=prompt_version,
        novel_id=novel_id,
    )
    output(launch_run(service, config, run_id, monitor=monitor), as_json)
