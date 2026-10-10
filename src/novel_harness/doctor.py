"""Runner/provider readiness checks.

Readiness is configuration and reachability only: the doctor never sends a
translation prompt or any chat-completion request, and never prints credentials.
An HTTP error still proves an endpoint is reachable; only a connection failure
is reported as unavailable.
"""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request

from .config import AppConfig
from .runners import runner_info

_SUBPROCESS_TYPES = {"codex", "opencode"}


def probe_endpoint(base_url: str, timeout: float = 3.0) -> dict:
    """Probe ``GET {base_url}/models`` without sending credentials."""
    url = base_url.rstrip("/") + "/models"
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"id": "endpoint", "status": "pass", "detail": f"HTTP {response.status}"}
    except urllib.error.HTTPError as error:
        # Any HTTP status (including 401/403) proves the endpoint answered.
        return {"id": "endpoint", "status": "pass", "detail": f"HTTP {error.code}"}
    except (urllib.error.URLError, OSError, ValueError) as error:
        return {"id": "endpoint", "status": "fail", "detail": type(error).__name__}


def probe_cli_version(executable: str, timeout: float = 5.0) -> dict:
    """Ask a CLI harness for its version; never starts a model session."""
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"id": "cli-version", "status": "warn", "detail": type(error).__name__}
    output = (completed.stdout or completed.stderr).strip().splitlines()
    detail = output[0][:120] if output else f"exit {completed.returncode}"
    return {
        "id": "cli-version",
        "status": "pass" if completed.returncode == 0 else "warn",
        "detail": detail,
    }


def diagnose(
    config: AppConfig,
    *,
    endpoint_timeout: float = 3.0,
    cli_timeout: float = 5.0,
) -> dict:
    """Return a readiness report for every configured runner."""
    reports = []
    for name in sorted(config.runners):
        profile = config.runners[name]
        info = runner_info(config, name)
        checks = [
            {
                "id": "configuration",
                "status": "pass" if info["available"] else "fail",
                "detail": info["reason"] or "configuration ok",
            }
        ]
        if profile.type == "openai":
            checks.append(probe_endpoint(profile.base_url, endpoint_timeout))
        if profile.type in _SUBPROCESS_TYPES:
            executable = info.get("executable")
            checks.append(
                {
                    "id": "cli",
                    "status": "pass" if executable else "fail",
                    "detail": executable or "executable not found",
                }
            )
            if executable:
                checks.append(probe_cli_version(executable, cli_timeout))
        ready = all(check["status"] != "fail" for check in checks)
        reports.append(
            {
                "name": name,
                "type": profile.type,
                "model": profile.model,
                "ready": ready,
                "checks": checks,
            }
        )
    return {
        "runners": reports,
        "ready": sum(1 for report in reports if report["ready"]),
        "total": len(reports),
        "inference": "none",
    }
