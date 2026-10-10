"""CLI-only context resolution kept outside service configuration loading."""

from __future__ import annotations

import json
import os
import sqlite3
from contextvars import ContextVar
from pathlib import Path

_ROOT_CONFIG: ContextVar[Path | None] = ContextVar("novel_cli_root_config", default=None)
_MAX_PROJECT_CONFIG_CANDIDATES = 64


def set_root_config(path: Path | None) -> None:
    """Remember the global CLI option for the current invocation."""
    _ROOT_CONFIG.set(path)


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _selected(path: str | Path, source: str) -> Path:
    resolved = _absolute(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Selected {source} config does not exist: {resolved}. "
            "Pass --config PATH or set NOVEL_CONFIG to a YAML file."
        )
    return resolved


def _nearest_config() -> Path | None:
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / "config.yaml"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _project_config_candidates() -> list[Path]:
    """Return config files from the nearest project-like directory only."""
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidates = sorted(
            {
                path.resolve()
                for pattern in ("config.yaml", "config*.yaml", "config*.yml")
                for path in directory.glob(pattern)
                if path.is_file()
            },
            key=lambda path: str(path),
        )
        if candidates:
            if len(candidates) > _MAX_PROJECT_CONFIG_CANDIDATES:
                raise ValueError(
                    "Too many project config candidates; pass --config PATH explicitly."
                )
            return candidates
    return []


def _read_run(path: Path, selector: str) -> tuple[str, dict] | None:
    state = path / "state.sqlite3"
    if not state.is_file():
        return None
    try:
        uri = f"file:{state.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            exact = db.execute("SELECT id, manifest FROM runs WHERE id=?", (selector,)).fetchone()
            if exact is None:
                if not selector or len(selector) < 4:
                    return None
                escaped = selector.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                rows = db.execute(
                    "SELECT id, manifest FROM runs "
                    "WHERE id LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\' "
                    "ORDER BY id LIMIT 7",
                    (f"{escaped}%", f"%-{escaped}%"),
                ).fetchall()
                if len(rows) > 1:
                    choices = ", ".join(row[0] for row in rows[:5])
                    raise ValueError(
                        f"Ambiguous run selector '{selector}' in output store {path}: {choices}"
                    )
                if not rows:
                    return None
                exact = rows[0]
            return exact["id"], json.loads(exact["manifest"])
    except sqlite3.Error as error:
        raise ValueError(f"Cannot inspect output store {path}: {error.__class__.__name__}") from error


def _config_matches_run(
    config, config_path: Path, manifest: dict, store: Path
) -> bool:
    if config.novel.output_dir.resolve() != store.resolve():
        return False
    if Path(manifest.get("source_dir", "")).resolve() != config.novel.source_dir.resolve():
        return False
    for stage in manifest.get("stages", []):
        runner_name = stage.get("runner")
        runner = config.runners.get(runner_name)
        if runner is None or runner.public_settings() != stage.get("runner_config"):
            return False
    identity = manifest.get("config_identity")
    return not (
        isinstance(identity, dict)
        and identity.get("path")
        and Path(identity["path"]).resolve() != config_path.resolve()
    )


def _run_identity_matches(config_path: Path, manifest: dict) -> bool:
    identity = manifest.get("config_identity")
    return isinstance(identity, dict) and Path(identity.get("path", "")).resolve() == config_path.resolve()


def resolve_config_for_run(selector: str, preferred: Path | None = None) -> Path:
    """Resolve and validate the one project-local config that owns a saved run."""
    candidates = [preferred.resolve()] if preferred is not None else _project_config_candidates()
    if not candidates:
        raise FileNotFoundError(
            f"Cannot resolve config for run '{selector}' from this project. "
            "Pass --config PATH or set NOVEL_CONFIG; run-only resolution does not scan unrelated directories."
        )
    from .config import load_config

    matches: list[tuple[Path, bool]] = []
    invalid: list[Path] = []
    found_any = False
    for path in candidates:
        try:
            config = load_config(path)
        except (OSError, ValueError, TypeError):
            invalid.append(path)
            continue
        found = _read_run(config.novel.output_dir, selector)
        if found is None:
            continue
        found_any = True
        _run_id, manifest = found
        if _config_matches_run(
            config,
            path,
            manifest,
            config.novel.output_dir,
        ):
            matches.append((path, _run_identity_matches(path, manifest)))
    identity_matches = [path for path, exact in matches if exact]
    if len(identity_matches) == 1:
        return identity_matches[0]
    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        shown = ", ".join(str(path) for path, _ in matches[:5])
        raise ValueError(
            f"Run '{selector}' matches multiple project configs: {shown}. Pass --config PATH."
        )
    if preferred is not None:
        if not found_any:
            # Preserve the normal repository error for a selector that is simply
            # absent; a present-but-incompatible run must fail closed below.
            return preferred.resolve()
        raise ValueError(
            f"Config {preferred} does not contain a compatible saved run '{selector}'. "
            "Pass the config that owns the run."
        )
    if not found_any and not invalid:
        discovered = _nearest_config()
        if discovered is not None:
            return discovered
    if invalid and len(invalid) == len(candidates):
        raise ValueError(
            f"No usable project config can be inspected for run '{selector}'. "
            "Pass --config PATH to the run's config."
        )
    raise ValueError(
        f"No compatible project config found for run '{selector}'. "
        "The associated config may be missing or moved; pass --config PATH."
    )


def has_explicit_config(command_config: Path | None = None) -> bool:
    return command_config is not None or _ROOT_CONFIG.get() is not None or bool(
        os.environ.get("NOVEL_CONFIG")
    )


def resolve_config(command_config: Path | None = None) -> Path:
    """Resolve a CLI config without changing non-CLI loading defaults."""
    if command_config is not None:
        return _selected(command_config, "command-local")
    root_config = _ROOT_CONFIG.get()
    if root_config is not None:
        return _selected(root_config, "root-global")
    environment_config = os.environ.get("NOVEL_CONFIG")
    if environment_config:
        return _selected(environment_config, "NOVEL_CONFIG")
    discovered = _nearest_config()
    if discovered is not None:
        return discovered
    current = Path.cwd().resolve()
    raise FileNotFoundError(
        f"No config.yaml found at or above {current}. "
        "Pass --config PATH or set NOVEL_CONFIG to a YAML file."
    )
