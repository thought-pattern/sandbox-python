"""Live build/start/auth/tool/egress/teardown smoke test."""

import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from manager import ContainerConfig, sandbox_session

CONTAINER_RUNTIME = shutil.which("container") or shutil.which("docker") or shutil.which("podman") or ""
NO_PAYLOAD = {}


def request_json(url, token, *, method="GET", payload=NO_PAYLOAD, timeout=2):
    headers = {"Authorization": f"Bearer {token}"}
    if payload is NO_PAYLOAD:
        request = urllib.request.Request(url, headers=headers, method=method)
    else:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = (response.status, json.loads(response.read()))
        return result


def wait_for_health(base_url, token, attempts=30):
    for _ in range(attempts):
        try:
            _, payload = request_json(f"{base_url}/health", token)
            if payload.get("status") == "healthy":
                return payload
        except (OSError, ValueError):
            time.sleep(0.25)
    return {}


@pytest.mark.skipif(not CONTAINER_RUNTIME, reason="no supported container runtime available")
def test_image_builds_and_serves_authenticated_bounded_tools():
    runtime = Path(CONTAINER_RUNTIME).name
    context = Path(__file__).resolve().parent.parent / "container"
    build = subprocess.run(
        [runtime, "build", "-t", "python-sandbox:test", "-f", str(context / "Containerfile"), str(context)],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    config = ContainerConfig(image="python-sandbox:test", port=8080)
    with sandbox_session(config, runtime=runtime):
        health = wait_for_health(config.base_url, config.auth_token)
        assert health
        assert health["egressPolicy"] == "deny"

        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            urllib.request.urlopen(f"{config.base_url}/tools", timeout=2)
        assert unauthorized.value.code == 401

        _, manifest = request_json(f"{config.base_url}/tools", config.auth_token)
        assert manifest["apiVersion"] == "1.0"
        assert "run_python" in {entry["name"] for entry in manifest["tools"]}

        _, written = request_json(
            f"{config.base_url}/tools/file_write",
            config.auth_token,
            method="POST",
            payload={"path": "smoke.txt", "content": "hello"},
        )
        assert written["result"]["bytesWritten"] == 5
        _, read = request_json(
            f"{config.base_url}/tools/file_read",
            config.auth_token,
            method="POST",
            payload={"path": "smoke.txt"},
        )
        assert read["result"] == "hello"

        network_probe = (
            "import socket\n"
            "try:\n"
            " socket.create_connection(('1.1.1.1', 80), 0.5)\n"
            " raise SystemExit(9)\n"
            "except OSError:\n"
            " raise SystemExit(0)\n"
        )
        _, result = request_json(
            f"{config.base_url}/tools/run_python",
            config.auth_token,
            method="POST",
            payload={"script": network_probe, "timeout": 2},
            timeout=4,
        )
        assert result["result"]["exitCode"] == 0

        with socket.create_connection((config.host_bind, config.host_port), timeout=2):
            pass
