"""Tests for the authenticated sandbox HTTP tool server."""

from importlib import import_module as imported_import_module
from json import dumps as json_dumps
from json import loads as json_loads
from threading import Thread as threading_Thread
from urllib import error as urllib_error
from urllib import request as urllib_request

from pytest import fixture as pytest_fixture
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

server = imported_import_module("server")

TOKEN = "t" * 43
NO_PAYLOAD = {}


@pytest_fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "BROKER_URL", "")
    monkeypatch.setattr(server, "BROKER_TOKEN", "")
    monkeypatch.setattr(server, "BROKER_PACKAGE_DESTINATION", "pypi")
    return False


@pytest_fixture
def http_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "EGRESS_POLICY", "deny")
    monkeypatch.setattr(server, "shutil_which", lambda name: f"/usr/bin/{name}")
    instance = server.ToolHTTPServer(
        ("127.0.0.1", 0),
        server.ToolHandler,
        auth_token=TOKEN,
        max_request_bytes=128,
        max_concurrent_requests=2,
        request_read_timeout=1,
    )
    thread = threading_Thread(target=instance.serve_forever, daemon=True)
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
        request = urllib_request.Request(url, headers=headers, method=method)
    else:
        body = json_dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        request = urllib_request.Request(url, data=body, headers=headers, method=method)
    with urllib_request.urlopen(request, timeout=2) as response:
        result = (response.status, json_loads(response.read()))
        return result


def test_resolve_path_allows_within_workspace():
    target = server.resolve_path("sub/file.py")
    assert target.is_relative_to(server.WORKSPACE)


@pytest_mark.parametrize("path", ["../etc/passwd", "/etc/passwd"])
def test_resolve_path_rejects_escape(path):
    with pytest_raises(server.ToolError) as caught:
        server.resolve_path(path)
    assert caught.value.code == "path_escape"


def test_manifest_is_versioned_typed_and_camel_case():
    manifest = server.build_manifest()
    tools = {tool.get("name", ""): tool for tool in manifest.get("tools", [])}
    assert manifest.get("apiVersion", "") == "1.0"
    assert manifest.get("protocol", "") == "tapestry.workspace.http"
    parameters = {parameter.get("name", ""): parameter for parameter in tools.get("file_list", {}).get("parameters", [])}
    assert parameters.get("maxResults", {}).get("type", "") == "integer"
    assert parameters.get("maxResults", {}).get("required", False) is False
    assert "default" not in {
        parameter.get("name", ""): parameter for parameter in tools.get("file_read", {}).get("parameters", [])
    }.get("path", "")
    assert {"git_init", "git_log", "workspace_tree"}.issubset(tools)


def test_tool_round_trip_is_atomic_and_counts_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    result = server.file_write("notes/todo.txt", "héllo")
    assert result.get("bytesWritten", False) == len("héllo".encode())
    assert server.file_read("notes/todo.txt") == "héllo"


def test_file_read_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "MAX_FILE_BYTES", 4)
    (tmp_path / "large.txt").write_text("12345")

    with pytest_raises(server.ToolError) as caught:
        server.file_read("large.txt")

    assert caught.value.code == "file_too_large"
    assert caught.value.status == 413


def test_http_requires_authentication(http_server):
    with pytest_raises(urllib_error.HTTPError) as caught:
        request_json(f"{http_server}/health", token="")
    assert caught.value.code == 401


def test_authenticated_health_and_manifest(http_server):
    status, health = request_json(f"{http_server}/health")
    _, manifest = request_json(f"{http_server}/tools")
    assert status == 200
    assert health.get("status", "") == "healthy"
    assert health.get("egressPolicy", "") == "deny"
    assert health.get("checks", {}).get("workspaceWritable", False) is True
    assert all(health.get("checks", {}).get("tools", {}).values())
    assert manifest.get("apiVersion", "") == "1.0"


def test_readiness_reports_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())

    def find_tool(name):
        location = "" if name == "rg" else f"/usr/bin/{name}"
        return location

    monkeypatch.setattr(server, "shutil_which", find_tool)
    report = server.build_readiness_report()

    assert report.get("status", "") == "unhealthy"
    assert report.get("checks", {}).get("tools", {}).get("rg", False) is False


def test_http_tool_accepts_camel_case_arguments(http_server):
    status, payload = request_json(
        f"{http_server}/tools/file_list",
        method="POST",
        payload={"path": ".", "maxResults": 10},
    )
    assert status == 200
    assert payload.get("result", {}) == {"files": [], "truncated": False}


def test_http_rejects_large_request(http_server):
    with pytest_raises(urllib_error.HTTPError) as caught:
        request_json(
            f"{http_server}/tools/run_python",
            method="POST",
            payload={"script": "x" * 200},
        )
    assert caught.value.code == 413


def test_http_rejects_large_response(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    monkeypatch.setattr(server, "MAX_FILE_BYTES", 4_096)
    monkeypatch.setattr(server, "shutil_which", lambda name: f"/usr/bin/{name}")
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
    thread = threading_Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest_raises(urllib_error.HTTPError) as caught:
            request_json(
                f"http://127.0.0.1:{instance.server_port}/tools/file_read",
                method="POST",
                payload={"path": "large.txt"},
            )
        error = json_loads(caught.value.read())
        assert caught.value.code == 413
        assert error.get("error", {}).get("code", "") == "response_too_large"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def test_process_result_preserves_nonzero_exit():
    result = server.run_python("raise SystemExit(7)", timeout=2)
    assert result.get("ok", False) is False
    assert result.get("exitCode", 0) == 7
    assert result.get("timedOut", False) is False


def test_process_timeout_terminates_group():
    result = server.run_python("import time; time.sleep(10)", timeout=0.05)
    assert result.get("ok", False) is False
    assert result.get("exitCode", False) == -1
    assert result.get("timedOut", False) is True


def test_process_timeout_covers_descendant_holding_output_pipe():
    script = "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
    result = server.run_python(script, timeout=0.1)
    assert result.get("exitCode", False) == -1
    assert result.get("timedOut", False) is True


def test_process_output_is_bounded(monkeypatch):
    monkeypatch.setattr(server, "MAX_OUTPUT_BYTES", 64)
    result = server.run_python("print('x' * 10000)", timeout=2)
    assert result.get("ok", False) is False
    assert result.get("exitCode", False) == -2
    assert result.get("outputLimited", False) is True
    assert len(result.get("stdout", "").encode()) <= 64


def test_git_init_log_and_workspace_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    initialized = server.git_init("feature/test")
    server.require_process_success(server.run_git(["config", "user.email", "sandbox@example.test"]), "git config")
    server.require_process_success(server.run_git(["config", "user.name", "Sandbox Test"]), "git config")
    server.file_write("src/example.py", "print('ok')\n")
    server.git_commit("Add example")

    history = server.git_log(limit=1)
    tree = server.workspace_tree(depth=3, max_results=10)

    assert initialized.get("branch", "") == "feature/test"
    assert initialized.get("initialized", False) is True
    assert history.get("commits", [])[0].get("subject", "") == "Add example"
    assert history.get("truncated", False) is False
    assert "src/example.py" in tree.get("files", [])


def test_git_processes_disable_interactive_credentials(monkeypatch):
    captured = {}

    def run_process(command, **settings):
        captured["command"] = command
        captured["environment"] = settings.get("environment", False)
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

    monkeypatch.setattr(server, "internal_run_process", run_process)
    server.run_git(["status"])

    assert captured.get("command", []) == ["git", "status"]
    assert captured.get("environment", {}).get("GIT_TERMINAL_PROMPT", "") == "0"
    assert captured.get("environment", {}).get("GCM_INTERACTIVE", "") == "never"


def test_network_tools_fail_closed_under_deny_policy(monkeypatch):
    monkeypatch.setattr(server, "EGRESS_POLICY", "deny")
    with pytest_raises(server.ToolError) as caught:
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
        computed_return_value = {
            "ok": True,
            "stdout": f"index={token}",
            "stderr": "",
            "exitCode": 0,
            "timedOut": False,
            "outputLimited": False,
        }
        return computed_return_value

    monkeypatch.setattr(server, "internal_run_process", run)

    result = server.pip_install(["pytest==8.0.0"])

    command = calls[0]
    index_url = command[command.index("--index-url") + 1]
    assert index_url.startswith("http://broker-secret-")
    assert index_url.endswith(":x@192.168.64.9:8090/v1/proxy/pypi/simple/")
    assert "pypi.org" not in " ".join(command)
    assert token not in result.get("stdout", False)


def test_broker_resolver_rejects_invalid_proxy_path(monkeypatch):
    monkeypatch.setattr(server, "EGRESS_POLICY", "broker")
    monkeypatch.setattr(server, "BROKER_URL", "http://192.168.64.9:8090")
    monkeypatch.setattr(server, "BROKER_TOKEN", "b" * 43)
    monkeypatch.setattr(server, "broker_request", lambda *args: {"proxyPath": "https://evil.example/"})

    with pytest_raises(server.ToolError) as caught:
        server.resolve_broker_url("https://github.com/example/repo.git", "git")
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
    with pytest_raises(ValueError, match="broker"):
        server.configure(args)
