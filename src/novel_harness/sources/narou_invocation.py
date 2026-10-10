"""Bounded invocation protocol for the separate Narou downloader checkout.

The companion process is deliberately treated as an executable boundary. It
receives a versioned request through environment variables and must publish a
request-bound completion manifest before its output can reach the library
filesystem importer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from ..artifacts import canonical, sha256, write_text
from ..source_import import SourceImportRequest

PROTOCOL_VERSION = 1
REQUEST_ENV = "NOVEL_HARNESS_NAROU_REQUEST"
COMPLETION_ENV = "NOVEL_HARNESS_NAROU_COMPLETION"
PROTOCOL_ENV = "NOVEL_HARNESS_NAROU_PROTOCOL"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_STDERR_TAIL_BYTES = 4096
_NAROU_URL = re.compile(r"/([A-Za-z0-9]+)/([1-9][0-9]*)/?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLACEHOLDERS = {"{url}", "{ncode}", "{output_dir}"}


class NarouInvocationError(ValueError):
    """Invocation or completion-protocol failure; no Library import occurred."""


def _canonical_url(url: str, ncode: str) -> str:
    if not isinstance(url, str) or len(url) > 2048:
        raise NarouInvocationError("Narou URL must be a single episode URL")
    parsed = urlsplit(url)
    match = _NAROU_URL.fullmatch(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "ncode.syosetu.com"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or not match
        or match.group(1).lower() != ncode.lower()
    ):
        raise NarouInvocationError(
            "Narou URL must be https://ncode.syosetu.com/<matching-ncode>/<episode>/"
        )
    return f"https://ncode.syosetu.com/{ncode.lower()}/{int(match.group(2))}/"


def _command_argv(command: Sequence[str], *, url: str, ncode: str, output_dir: Path) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise NarouInvocationError("Downloader command must be a non-empty argument list")
    replacements = {"{url}": url, "{ncode}": ncode.lower(), "{output_dir}": str(output_dir)}
    argv: list[str] = []
    for item in command:
        if not isinstance(item, str) or not item or "\0" in item:
            raise NarouInvocationError("Downloader command arguments must be non-empty strings")
        if item in _PLACEHOLDERS:
            argv.append(replacements[item])
        elif "{url}" in item or "{ncode}" in item or "{output_dir}" in item:
            raise NarouInvocationError("Downloader placeholders must occupy a whole argument")
        else:
            argv.append(item)
    if argv[0] == str(output_dir):
        raise NarouInvocationError("Downloader executable cannot be the output directory")
    if "{url}" not in command or "{output_dir}" not in command:
        raise NarouInvocationError(
            "Downloader command must include {url} and {output_dir} arguments"
        )
    return argv


def _prepare_output_dir(output_dir: Path, library_roots: Sequence[Path]) -> Path:
    path = Path(os.path.abspath(output_dir))
    if path == Path(path.anchor):
        raise NarouInvocationError("Unsafe downloader output path: filesystem root")
    if path.is_symlink():
        raise NarouInvocationError("Downloader output path must not be a symlink")
    for parent in path.parents:
        if parent.exists() and parent.is_symlink():
            raise NarouInvocationError("Downloader output path cannot pass through a symlink")
    resolved = path.resolve(strict=False)
    for root in library_roots:
        library_root = Path(root).resolve(strict=False)
        if (
            resolved == library_root
            or resolved.is_relative_to(library_root)
            or library_root.is_relative_to(resolved)
        ):
            raise NarouInvocationError(
                "Downloader output must be separate from configured Library roots"
            )
    if path.exists():
        if not path.is_dir():
            raise NarouInvocationError("Downloader output path must be a directory")
        if any(path.iterdir()):
            raise NarouInvocationError(
                "Downloader output directory must be empty; use a fresh path for retries"
            )
    else:
        path.mkdir(parents=True)
    if path.is_symlink() or path.resolve(strict=True) != resolved:
        raise NarouInvocationError("Downloader output path changed while preparing it")
    return resolved


def _stderr_tail(stream, tail: bytearray) -> None:
    while True:
        chunk = stream.read(1024)
        if not chunk:
            return
        tail.extend(chunk)
        if len(tail) > _STDERR_TAIL_BYTES:
            del tail[:-_STDERR_TAIL_BYTES]


def _kill_process_group(process: subprocess.Popen, *, include_exited: bool = False) -> None:
    if process.poll() is not None and not (include_exited and os.name == "posix"):
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def _run_process(argv: list[str], *, cwd: Path | None, timeout: float, env: dict[str, str]) -> tuple[int, str]:
    process: subprocess.Popen | None = None
    tail = bytearray()
    reader: threading.Thread | None = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise NarouInvocationError(f"Could not start Narou downloader: {error}") from error
    assert process.stderr is not None
    reader = threading.Thread(target=_stderr_tail, args=(process.stderr, tail), daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _kill_process_group(process)
        reader.join(timeout=2)
        detail = tail.decode("utf-8", errors="replace").strip()
        suffix = f"; stderr tail: {detail}" if detail else ""
        raise NarouInvocationError(
            f"Narou downloader exceeded timeout of {timeout:g} seconds; no import performed{suffix}"
        ) from error
    except BaseException:
        _kill_process_group(process)
        reader.join(timeout=2)
        raise
    _kill_process_group(process, include_exited=True)
    reader.join(timeout=2)
    if reader.is_alive():
        _kill_process_group(process, include_exited=True)
        reader.join(timeout=2)
    return returncode, tail.decode("utf-8", errors="replace").strip()


def _read_manifest(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise NarouInvocationError("Downloader did not write a fresh completion manifest")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise NarouInvocationError("Downloader completion manifest exceeds the size limit")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NarouInvocationError("Downloader completion manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise NarouInvocationError("Downloader completion manifest must be a JSON object")
    return value


def _verify_completion(
    completion_path: Path,
    *,
    request_sha256: str,
    source: dict,
    output_dir: Path,
    expected_chapters: tuple[int, ...],
) -> None:
    manifest = _read_manifest(completion_path)
    required = {
        "version",
        "request_sha256",
        "status",
        "source",
        "output_dir",
        "expected_chapters",
        "chapters",
        "diagnostics",
    }
    if set(manifest) != required:
        raise NarouInvocationError("Downloader completion manifest has unknown or missing fields")
    if type(manifest["version"]) is not int or manifest["version"] != PROTOCOL_VERSION:
        raise NarouInvocationError("Unsupported Narou completion protocol version")
    if manifest["request_sha256"] != request_sha256:
        raise NarouInvocationError("Downloader completion manifest does not match this request")
    diagnostics = manifest["diagnostics"]
    if not isinstance(diagnostics, list) or any(not isinstance(item, str) for item in diagnostics):
        raise NarouInvocationError("Downloader completion diagnostics must be a string list")
    if manifest["status"] != "complete":
        detail = "; ".join(item for item in diagnostics if isinstance(item, str))
        status = manifest["status"]
        suffix = f": {detail}" if detail else ""
        raise NarouInvocationError(
            f"Downloader reported {status!r}; partial output was not imported{suffix}"
        )
    if manifest["source"] != source:
        raise NarouInvocationError("Downloader completion source identity does not match the request")
    if manifest["output_dir"] != str(output_dir):
        raise NarouInvocationError("Downloader completion output path does not match the request")
    if manifest["expected_chapters"] != list(expected_chapters):
        raise NarouInvocationError("Downloader completion chapter request does not match")
    chapters = manifest["chapters"]
    if not isinstance(chapters, list):
        raise NarouInvocationError("Downloader completion chapters must be a list")
    expected_keys = {"number", "filename", "sha256"}
    numbers: list[int] = []
    names: set[str] = set()
    for chapter in chapters:
        if not isinstance(chapter, dict) or set(chapter) != expected_keys:
            raise NarouInvocationError("Downloader completion has an invalid chapter record")
        number, filename, digest = chapter["number"], chapter["filename"], chapter["sha256"]
        if type(number) is not int or number < 1:
            raise NarouInvocationError("Downloader completion has an invalid chapter number")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".txt")
            or Path(filename).name != filename
            or filename in {".", ".."}
            or ":" in filename
            or "/" in filename
            or "\\" in filename
            or any(ord(char) < 32 for char in filename)
            or filename in names
        ):
            raise NarouInvocationError("Downloader completion has an unsafe chapter filename")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise NarouInvocationError("Downloader completion has an invalid chapter hash")
        match = re.match(r"([0-9]+)_", filename)
        if not match or int(match.group(1)) != number:
            raise NarouInvocationError("Downloader chapter filename does not match its number")
        numbers.append(number)
        names.add(filename)
    if tuple(numbers) != expected_chapters:
        raise NarouInvocationError("Downloader completion chapter inventory is partial or unordered")
    entries = list(output_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise NarouInvocationError("Downloader output must contain only regular chapter files")
    if {entry.name for entry in entries} != names:
        raise NarouInvocationError("Downloader output files differ from the completion inventory")
    for chapter in chapters:
        path = output_dir / chapter["filename"]
        try:
            with path.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            raise NarouInvocationError(f"Could not verify downloaded chapter {path.name}") from error
        if actual_hash != chapter["sha256"]:
            raise NarouInvocationError(f"Downloaded chapter hash differs from completion manifest: {path.name}")


def invoke_narou_download(
    *,
    command: Sequence[str],
    ncode: str,
    source_url: str,
    title: str,
    expected_chapters: tuple[int, ...],
    output_dir: Path,
    timeout_seconds: float,
    library_roots: Sequence[Path],
    import_completed: Callable[[Path, tuple[int, ...]], dict],
    cwd: Path | None = None,
) -> dict:
    """Run a protocol-v1 downloader and import only request-bound complete output."""
    valid = SourceImportRequest(
        provider="narou",
        source_id=ncode,
        title=title,
        expected_chapters=expected_chapters,
    ).validate()
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 86_400
    ):
        raise NarouInvocationError("Downloader timeout must be greater than 0 and at most 86400 seconds")
    url = _canonical_url(source_url, valid.source_id)
    output_path = _prepare_output_dir(Path(output_dir), library_roots)
    argv = _command_argv(
        command, url=url, ncode=valid.source_id, output_dir=output_path
    )
    source = {"provider": "narou", "id": valid.source_id.lower(), "url": url}
    request_payload = {
        "version": PROTOCOL_VERSION,
        "operation": "narou-episode-download",
        "source": source,
        "title": valid.title,
        "expected_chapters": list(valid.expected_chapters),
        "output_dir": str(output_path),
        "output_format": "flat-utf8-txt-v1",
    }
    request_digest = sha256(canonical(request_payload))

    with tempfile.TemporaryDirectory(prefix="novel-narou-request-") as control_dir:
        request_path = Path(control_dir) / "request.json"
        completion_path = Path(control_dir) / "completion.json"
        write_text(
            request_path,
            canonical({**request_payload, "request_sha256": request_digest}) + "\n",
        )
        env = os.environ.copy()
        env[PROTOCOL_ENV] = str(PROTOCOL_VERSION)
        env[REQUEST_ENV] = str(request_path)
        env[COMPLETION_ENV] = str(completion_path)
        returncode, stderr = _run_process(
            argv,
            cwd=Path(cwd).resolve() if cwd is not None else None,
            timeout=float(timeout_seconds),
            env=env,
        )
        if returncode != 0:
            suffix = f"; stderr tail: {stderr}" if stderr else ""
            raise NarouInvocationError(
                f"Narou downloader exited with status {returncode}; partial output was not imported{suffix}"
            )
        if output_path.is_symlink() or output_path.resolve(strict=True) != output_path:
            raise NarouInvocationError("Downloader output path changed during invocation")
        _verify_completion(
            completion_path,
            request_sha256=request_digest,
            source=source,
            output_dir=output_path,
            expected_chapters=valid.expected_chapters,
        )
        imported = import_completed(output_path, valid.expected_chapters)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_sha256": request_digest,
        "process_exit_code": 0,
        "status": imported.get("status"),
        "source_identity": imported.get("source_identity", source),
        "chapter_numbers": list(valid.expected_chapters),
        "import": imported,
    }
