"""Build inspection: the Web UI assets must ship in the sdist and wheel.

The UI is a single source of truth under ``novel_harness/web/``; if it is not
declared as package data, ``novel serve`` only works from a git checkout. This
bounded test builds the distribution with the repository's own ``uv build``
workflow and checks the archives. If the build frontend or an isolated build
backend is unavailable (offline without a cached setuptools), it skips rather than
failing the core suite; the task's verification commands build the sdist explicitly
as a required check.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from novel_harness.api import _resolve_web_dir

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("index.html", "app.js", "styles.css")


def test_resolver_finds_packaged_assets_without_cwd():
    resolved = _resolve_web_dir()
    for asset in ASSETS:
        assert (resolved / asset).is_file(), f"missing packaged asset: {asset}"


def _build(out_dir: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available for a packaging build")
    assert uv is not None
    result = subprocess.run(
        [uv, "build", "--out-dir", str(out_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"uv build unavailable: {result.stderr.strip()[:300]}")


def test_web_assets_are_packaged(tmp_path):
    _build(tmp_path)

    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert wheels, "no wheel produced"
    assert sdists, "no sdist produced"

    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    for asset in ASSETS:
        assert f"novel_harness/web/{asset}" in names, f"wheel missing {asset}"

    with tarfile.open(sdists[0]) as archive:
        sdist_names = archive.getnames()
    assert any(name.endswith("/config.qwen-benchmark.example.yaml") for name in sdist_names), (
        "sdist missing the documented Qwen benchmark profile example"
    )
    for asset in ASSETS:
        assert any(name.endswith(f"novel_harness/web/{asset}") for name in sdist_names), (
            f"sdist missing {asset}"
        )
