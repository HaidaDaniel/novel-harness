"""Shared CLI selection contracts for persisted run actions."""

from __future__ import annotations

import yaml
from click.shell_completion import CompletionItem
from pydantic import ValidationError

from .cli_context import resolve_config, set_root_config
from .config import AppConfig
from .persistence import completion_rows


def resolve_run_id(
    positional: str | None,
    option: str | None,
    *,
    required: bool = True,
    repository=None,
) -> str | None:
    """Resolve one run selector without mutating service state."""
    if positional is not None and option is not None and positional != option:
        raise ValueError("Positional RUN_ID and --run must identify the same run")
    target = positional or option
    if required and target is None:
        raise ValueError("Provide RUN_ID or --run RUN_ID")
    return repository.resolve_run_selector(target) if repository is not None and target else target


def complete_run_ids(ctx, _param, incomplete: str) -> list[CompletionItem]:
    """Suggest bounded IDs from the config-selected store only."""
    try:
        local = ctx.params.get("config")
        root = ctx.find_root().params.get("config")
        set_root_config(root)
        config_path = resolve_config(local)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = AppConfig.model_validate(raw)
        store = (config_path.parent / config.novel.output_dir).resolve() / "state.sqlite3"
        return [
            CompletionItem(
                row["id"],
                help=f"{row.get('title') or '-'} · {row.get('status') or '-'}",
            )
            for row in completion_rows(store, incomplete)
        ]
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError):
        return []
