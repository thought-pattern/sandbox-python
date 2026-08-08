"""Tests for the authenticated sandbox HTTP tool server."""

import json
import threading
import urllib.error
import urllib.request

import pytest

import server

TOKEN = "t" * 43


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "BROKER_URL", "")
    monkeypatch.setattr(server, "BROKER_TOKEN", "")
    monkeypatch.setattr(server, "BROKER_PACKAGE_DESTINATION", "pypi")


@pytest.fixture
def http_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "EGRESS_POLICY", "deny")
    instance = server.ToolHTTPServer(
        ("127.0.0.1", 0),
        server.ToolHandler,
        auth_token=TOKEN,
        max_request_bytes=128,
        max_concurrent_requests=2,
        request_read_timeout=1,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def request_json(url, *, token=TOKEN, method="GET", payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read())


def test_resolve_path_allows_within_workspace():
    target = server.resolve_path("sub/file.py")
    assert target.is_relative_to(server.WORKSPACE)


@pytest.mark.parametrize("path", ["../etc/passwd", "/etc/passwd"])
def test_resolve_path_rejects_escape(path):
    with pytest.raises(server.ToolError) as caught:
        server.resolve_path(path)
    assert caught.value.code == "path_escape"


def test_manifest_is_versioned_typed_and_camel_case():
    manifest = server.build_manifest()
    tools = {tool["name"]: tool for tool in manifest["tools"]}
    assert manifest["apiVersion"] == "1.0"
    assert manifest["protocol"] == "tapestry.workspace.http"
    parameters = {parameter["name"]: parameter for parameter in tools["file_list"]["parameters"]}
    assert parameters["maxResults"]["type"] == "integer"
    assert parameters["maxResults"]["required"] is False


def test_tool_round_trip_is_atomic_and_counts_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    result = server.file_write("notes/todo.txt", "héllo")
    assert result["bytesWritten"] == len("héllo".encode())
    assert server.file_read("notes/todo.txt") == "héllo"


def test_http_requires_authentication(http_server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request_json(f"{http_server}/health", token="")
    assert caught.value.code == 401


def test_authenticated_health_and_manifest(http_server):
    status, health = request_json(f"{http_server}/health")
    _, manifest = request_json(f"{http_server}/tools")
    assert status == 200
    assert health["status"] == "healthy"
    assert health["egressPolicy"] == "deny"
    assert manifest["apiVersion"] == "1.0"


def test_http_tool_accepts_camel_case_arguments(http_server):
    status, payload = request_json(
        f"{http_server}/tools/file_list",
        method="POST",
        payload={"path": ".", "maxResults": 10},
    )
    assert status == 200
    assert payload["result"] == {"files": [], "truncated": False}


def test_http_rejects_large_request(http_server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        request_json(
            f"{http_server}/tools/run_python",
            method="POST",
            payload={"script": "x" * 200},
        )
    assert caught.value.code == 413


def test_process_result_preserves_nonzero_exit():
    result = server.run_python("raise SystemExit(7)", timeout=2)
    assert result["ok"] is False
    assert result["exitCode"] == 7
    assert result["timedOut"] is False


def test_process_timeout_terminates_group():
    result = server.run_python("import time; time.sleep(10)", timeout=0.05)
    assert result["ok"] is False
    assert result["exitCode"] == -1
    assert result["timedOut"] is True


def test_process_timeout_covers_descendant_holding_output_pipe():
    script = "import subprocess, sys\n" "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
    result = server.run_python(script, timeout=0.1)
    assert result["exitCode"] == -1
    assert result["timedOut"] is True


def test_process_output_is_bounded(monkeypatch):
    monkeypatch.setattr(server, "MAX_OUTPUT_BYTES", 64)
    result = server.run_python("print('x' * 10000)", timeout=2)
    assert result["ok"] is False
    assert result["exitCode"] == -2
    assert result["outputLimited"] is True
    assert len(result["stdout"].encode()) <= 64


def test_network_tools_fail_closed_under_deny_policy(monkeypatch):
    monkeypatch.setattr(server, "EGRESS_POLICY", "deny")
    with pytest.raises(server.ToolError) as caught:
        server.pip_install(["pytest"])
    assert caught.value.code == "egress_denied"


def test_brokered_pip_uses_only_broker_index_and_redacts_token(monkeypatch):
    token = "broker-secret-" + "x" * 32
    calls = []
    monkeypatch.setattr(server, "EGRESS_POLICY", "broker")
    monkeypatch.setattr(server, "BROKER_URL", "http://192.168.64.9:8090")
    monkeypatch.setattr(server, "BROKER_TOKEN", token)

    def run(command, **kwargs):
        calls.append(command)
        return {
            "ok": True,
            "stdout": f"index={token}",
            "stderr": "",
            "exitCode": 0,
            "timedOut": False,
            "outputLimited": False,
        }

    monkeypatch.setattr(server, "_run_process", run)

    result = server.pip_install(["pytest==8.0.0"])

    command = calls[0]
    index_url = command[command.index("--index-url") + 1]
    assert index_url.startswith("http://broker-secret-")
    assert index_url.endswith(":x@192.168.64.9:8090/v1/proxy/pypi/simple/")
    assert "pypi.org" not in " ".join(command)
    assert token not in result["stdout"]


def test_broker_resolver_rejects_invalid_proxy_path(monkeypatch):
    monkeypatch.setattr(server, "EGRESS_POLICY", "broker")
    monkeypatch.setattr(server, "BROKER_URL", "http://192.168.64.9:8090")
    monkeypatch.setattr(server, "BROKER_TOKEN", "b" * 43)
    monkeypatch.setattr(server, "_broker_request", lambda *args: {"proxyPath": "https://evil.example/"})

    with pytest.raises(server.ToolError) as caught:
        server._resolve_broker_url("https://github.com/example/repo.git", "git")
    assert caught.value.code == "broker_protocol"


def test_configure_requires_broker_endpoint_and_token(tmp_path):
    args = server.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--auth-token",
            "a" * 43,
            "--egress-policy",
            "broker",
        ]
    )
    with pytest.raises(ValueError, match="broker"):
        server.configure(args)
