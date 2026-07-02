"""Build-and-run smoke test, gated on a container runtime being present.

Skipped when neither docker nor podman is available, so the unit suite stays
runnable in a plain Python environment while a host with a runtime gets the
real build, start, health-check, and teardown.
"""

import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from manager import ContainerConfig, sandbox_session

CONTAINER_RUNTIME = shutil.which("container") or shutil.which("docker") or shutil.which("podman")


@pytest.mark.skipif(CONTAINER_RUNTIME is None, reason="no container runtime (docker/podman) available")
def test_image_builds_and_serves_health():
    runtime = Path(CONTAINER_RUNTIME).name
    context = Path(__file__).resolve().parent.parent / "container"
    build = subprocess.run(
        [runtime, "build", "-t", "python-sandbox:test", "-f", str(context / "Containerfile"), str(context)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    config = ContainerConfig(image="python-sandbox:test", port=8080)
    payload = None
    manifest = None
    with sandbox_session(config, runtime=runtime):
        for _ in range(30):
            try:
                with urllib.request.urlopen("http://localhost:8080/health", timeout=2) as response:
                    payload = json.loads(response.read())
                    break
            except (OSError, ValueError):
                time.sleep(1)
        if payload is not None:
            with urllib.request.urlopen("http://localhost:8080/tools", timeout=2) as response:
                manifest = json.loads(response.read())

    assert payload is not None
    assert payload.get("status") == "healthy"
    assert manifest is not None
    assert "run_python" in {entry["name"] for entry in manifest["tools"]}
