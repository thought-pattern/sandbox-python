"""Tests for the authenticated sandbox HTTP tool server."""

import json
import threading
import urllib.error
import urllib.request

import pytest

import server

TOKEN = "t" * 43
NO_PAYLOAD = {}


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
    monkeypatch.setattr(server.shutil, "which", lambda name: f"/usr/bin/{name}")
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


def request_json(url, *, token=TOKEN, method="GET", payload=NO_PAYLOAD):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if payload is NO_PAYLOAD:
        request = urllib.request.Request(url, headers=headers, method=method)
    else:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=2) as response:
        result = (response.status, json.loads(response.read()))
        return result


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
    assert "default" not in {parameter["name"]: parameter for parameter in tools["file_read"]["parameters"]}["path"]
    assert {"git_init", "git_log", "workspace_tree"}.issubset(tools)


def test_tool_round_trip_is_atomic_and_counts_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    result = server.file_write("notes/todo.txt", "héllo")
    assert result["bytesWritten"] == len("héllo".encode())
    assert server.file_read("notes/todo.txt") == "héllo"


def test_file_read_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "MAX_FILE_BYTES", 4)
    (tmp_path / "large.txt").write_text("12345")

    with pytest.raises(server.ToolError) as caught:
        server.file_read("large.txt")

    assert caught.value.code == "file_too_large"
    assert caught.value.status == 413


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
    assert health["checks"]["workspaceWritable"] is True
    assert all(health["checks"]["tools"].values())
    assert manifest["apiVersion"] == "1.0"


def test_readiness_reports_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())

    def find_tool(name):
        location = "" if name == "rg" else f"/usr/bin/{name}"
        return location

    monkeypatch.setattr(server.shutil, "which", find_tool)
    report = server.build_readiness_report()

    assert report["status"] == "unhealthy"
    assert report["checks"]["tools"]["rg"] is False


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


def test_http_rejects_large_response(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "MAX_FILE_BYTES", 4_096)
    monkeypatch.setattr(server.shutil, "which", lambda name: f"/usr/bin/{name}")
    (tmp_path / "large.txt").write_text("x" * 2_000)
    instance = server.ToolHTTPServer(
        ("127.0.0.1", 0),
        server.ToolHandler,
        auth_token=TOKEN,
        max_request_bytes=4_096,
        max_response_bytes=512,
        max_concurrent_requests=1,
        request_read_timeout=1,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            request_json(
                f"http://127.0.0.1:{instance.server_port}/tools/file_read",
                method="POST",
                payload={"path": "large.txt"},
            )
        error = json.loads(caught.value.read())
        assert caught.value.code == 413
        assert error["error"]["code"] == "response_too_large"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


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


def test_git_init_log_and_workspace_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    initialized = server.git_init("feature/test")
    server._require_process_success(server._run_git(["config", "user.email", "sandbox@example.test"]), "git config")
    server._require_process_success(server._run_git(["config", "user.name", "Sandbox Test"]), "git config")
    server.file_write("src/example.py", "print('ok')\n")
    server.git_commit("Add example")

    history = server.git_log(limit=1)
    tree = server.workspace_tree(depth=3, max_results=10)

    assert initialized["branch"] == "feature/test"
    assert initialized["initialized"] is True
    assert history["commits"][0]["subject"] == "Add example"
    assert history["truncated"] is False
    assert "src/example.py" in tree["files"]


def test_git_processes_disable_interactive_credentials(monkeypatch):
    captured = {}

    def run_process(command, **settings):
        captured["command"] = command
        captured["environment"] = settings["environment"]
        result = {
            "ok": True,
            "stdout": "",
            "stderr": "",
            "exitCode": 0,
            "timedOut": False,
            "outputLimited": False,
            "durationMs": 0.0,
        }
        return result

    monkeypatch.setattr(server, "_run_process", run_process)
    server._run_git(["status"])

    assert captured["command"] == ["git", "status"]
    assert captured["environment"]["GIT_TERMINAL_PROMPT"] == "0"
    assert captured["environment"]["GCM_INTERACTIVE"] == "never"


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
