# Copyright 2025-2026 Thought Pattern AI, LLC
# SPDX-License-Identifier: Apache-2.0

"""PDC-DEVELOP batch 211: actual gRPC tools and subprocesses without a container runtime."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.client import HTTPConnection
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from json import loads
from os import dup, environ, mkfifo
from pathlib import Path
from sys import executable
from threading import Thread
from uuid import UUID
from zipfile import ZipFile

from pytest import fixture, raises
from server import (
    EGRESS_POLICY,
    PROCESS_WATCHDOG_SECONDS,
    WORKSPACE,
    HealthHTTPServer,
    ToolError,
    WorkspaceService,
    configure,
    parse_args,
    read_auth_token,
    require_process_success,
    run_git,
    scoped_repository_bytes,
    shutil_which,
)
from workspace_client import WorkspaceClient, WorkspaceToolError

TOKEN = "direct-interface-test-" + "t" * 32


def configured_workspace(tmp_path: Path) -> Path:
    """The Workspace the direct_tools fixture configures for the test using `tmp_path`."""
    workspace = tmp_path / "work"
    return workspace


def task_folder(tmp_path: Path, client: WorkspaceClient) -> Path:
    """The folder the client's root Task works in, inside the configured Workspace."""
    folder = configured_workspace(tmp_path) / ".tapestry-work" / client.root_task_id
    return folder


class ArtifactServer(ThreadingHTTPServer):
    """Serve only fixture files; never forward requests to another destination."""

    def __init__(self, directory):
        self.directory = directory
        self.paths = []
        super().__init__(("127.0.0.1", 0), ArtifactHandler)


class ArtifactHandler(SimpleHTTPRequestHandler):
    server: ArtifactServer

    def __init__(self, request, address, owner):
        super().__init__(request, address, owner, directory=str(owner.directory))

    def do_GET(self):
        self.server.paths.append(self.path)
        super().do_GET()

    def log_message(self, format, *arguments):
        """Requests are retained by the fixture owner."""


@fixture
def direct_tools(tmp_path, monkeypatch):
    """Use the real server configuration, gRPC owner, client and subprocess owner."""
    monkeypatch.setenv("PATH", f"{Path(executable).parent}:{environ.get('PATH', '')}")
    # configure() rebinds these server globals; each test starts from, and is restored to, the imported defaults.
    configured_defaults = {
        "WORKSPACE": WORKSPACE,
        "EGRESS_POLICY": EGRESS_POLICY,
        "PROCESS_WATCHDOG_SECONDS": PROCESS_WATCHDOG_SECONDS,
    }
    for name, value in configured_defaults.items():
        monkeypatch.setattr(f"server.{name}", value)
    key_file = tmp_path / "workspace-key"
    key_file.write_text(TOKEN + "\n")
    with key_file.open("rb") as key_stream:
        args = parse_args(["--workspace", str(configured_workspace(tmp_path)), "--auth-token-fd", str(dup(key_stream.fileno()))])
        assert read_auth_token(args) == TOKEN
    configure(args)
    owner = WorkspaceService(TOKEN, args)
    owner.start("127.0.0.1:0")
    with HealthHTTPServer(("127.0.0.1", 0), owner, args.request_read_timeout) as endpoint:
        worker = Thread(target=endpoint.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            with WorkspaceClient(f"127.0.0.1:{owner.port}", TOKEN) as client:
                client.health_address = ("127.0.0.1", endpoint.server_port)
                yield client
        finally:
            owner.close()
            endpoint.shutdown()
            worker.join(timeout=5)
            assert not worker.is_alive()


@fixture
def artifacts(tmp_path):
    directory = tmp_path / "destinations"
    directory.mkdir()
    with ArtifactServer(directory) as endpoint:
        worker = Thread(target=endpoint.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield endpoint
        finally:
            endpoint.shutdown()
            worker.join(timeout=5)
            assert not worker.is_alive()


def test_authenticated_direct_interface_runs_real_tools_and_preserves_failures(direct_tools, tmp_path):
    health = direct_tools.health()
    assert health.get("egressPolicy", "") == "direct"
    incarnation = health.get("incarnation", "")
    assert UUID(incarnation).version == 4
    assert direct_tools.health().get("incarnation", "") == incarnation
    # Readiness answers without the key; the manifest, which names the same incarnation, and tool calls require it.
    with WorkspaceClient(direct_tools.address, "incorrect-identity") as unauthorized:
        assert unauthorized.health().get("incarnation", "") == incarnation
        with raises(WorkspaceToolError) as rejected:
            unauthorized.describe()
        assert rejected.value.status_code == 401
        with raises(WorkspaceToolError) as refused:
            unauthorized.call_tool("python_version")
        assert refused.value.status_code == 401
    manifest = direct_tools.describe()
    assert (manifest.get("protocol", ""), manifest.get("incarnation", "")) == ("tapestry.workspace.grpc", incarnation)
    assert direct_tools.call_tool("file_write", {"path": "smoke.txt", "content": "hello"}).get("bytesWritten", 0) == 5
    assert direct_tools.call_tool("file_read", {"path": "smoke.txt"}) == "hello"
    result = direct_tools.call_tool("run_python", {"script": "print(6 * 7)"})
    assert result.get("exitCode", -1) == 0 and result.get("stdout", "").strip() == "42"
    script = "#" + "x" * 300_000 + "\n"
    script += (
        "import sys\n"
        "answer = 42\n"
        "def observed_answer():\n    return answer\n"
        "assert observed_answer() == 42\n"
        "assert __name__ == '__main__' and '__file__' not in globals()\n"
        "assert sys.argv == ['-c']\n"
        "print('large source: π')\n"
    )
    result = direct_tools.call_tool("run_python", {"script": script})
    assert result.get("ok", False) and result.get("stdout", "").strip() == "large source: π"
    with raises(WorkspaceToolError) as failed:
        direct_tools.call_tool("run_python", {"script": "raise SystemExit(7)"})
    assert failed.value.result.get("exitCode", -1) == 7

    # One Workspace serves every root Task, each in its own folder; a child Task names its root and shares its folder.
    # Request threads never share cwd.
    roots = (
        "019cdc3d-a856-74f7-b720-f730ead81438",
        "019cdc3d-a857-74f7-b720-f730ead81438",
    )
    sources = ("print('decoder')\n", "print('sieve')\n")
    for root, source in zip(roots, sources, strict=True):
        envelope = direct_tools.invoke_tool(
            "file_write",
            {"path": "solution.py", "content": source},
            root_task_id=root,
        )
        assert envelope.get("status_code", 0) == 200
        assert envelope.get("workspace_context", {}) == {"root_task_id": root}
        assert envelope.get("result", {}).get("path") == "solution.py"
        assert (configured_workspace(tmp_path) / ".tapestry-work" / root / "solution.py").read_text() == source
    # A third root Task writes the same file name in its own folder.
    direct_tools.call_tool("file_write", {"path": "solution.py", "content": "print('third')\n"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        operations = [
            executor.submit(
                direct_tools.invoke_tool,
                "run_python",
                {
                    "script": "import pathlib, time\ntime.sleep(0.05)\nexec(pathlib.Path('solution.py').read_text())",
                },
                root_task_id=root,
            )
            for root in roots
        ]
        for root, name, operation in zip(roots, ("decoder", "sieve"), operations, strict=True):
            envelope = operation.result()
            assert envelope.get("status_code", 0) == 200 and envelope.get("result", {}).get("stdout") == name + "\n"
            assert envelope.get("workspace_context", {}) == {"root_task_id": root}
    child_read = direct_tools.invoke_tool("file_read", {"path": "solution.py"}, root_task_id=roots[0])
    assert child_read.get("result") == sources[0]
    assert direct_tools.call_tool("file_read", {"path": "solution.py"}) == "print('third')\n"
    assert not (configured_workspace(tmp_path) / "solution.py").exists()
    for root, path, code in (
        (roots[0], "../solution.py", "path_escape"),
        (
            "019cdc3d-a856-44f7-b720-f730ead81438",
            "solution.py",
            "invalid_working_context",
        ),
        ("", "solution.py", "invalid_working_context"),
    ):
        with raises(WorkspaceToolError) as refused:
            direct_tools.invoke_tool("file_read", {"path": path}, root_task_id=root)
        assert refused.value.code == code


def test_the_watchdog_ends_only_a_process_that_stalls(direct_tools, monkeypatch):
    monkeypatch.setattr("server.PROCESS_WATCHDOG_SECONDS", 2)
    working = "import time\nfor step in range(6):\n    print(step, flush=True)\n    time.sleep(0.75)\n"
    finished = direct_tools.call_tool("run_python", {"script": working})
    assert (finished.get("exitCode", -1), finished.get("stalled", True)) == (0, False)
    with raises(WorkspaceToolError) as ended:
        direct_tools.call_tool("run_python", {"script": "import time\ntime.sleep(60)\n"})
    assert (ended.value.code, ended.value.result.get("exitCode", 0), ended.value.result.get("stalled", False)) == (
        "stalled",
        -1,
        True,
    )


def test_health_page_and_api_answer_without_the_key_and_count_tool_calls_without_content(direct_tools, monkeypatch):
    """The page and its API need no key, and the figures move with a real tool call but never hold its content."""
    # Readiness requires ripgrep, which this test does not exercise; the host running it may not have it installed.
    monkeypatch.setattr("server.shutil_which", lambda name: "/usr/bin/rg" if name == "rg" else shutil_which(name))
    connection = HTTPConnection(*direct_tools.health_address, timeout=5)

    def fetch(path):
        connection.request("GET", path)
        response = connection.getresponse()
        reply = (response.status, response.getheader("Content-Security-Policy", ""), response.read())
        return reply

    try:
        for path, marker in (
            ("/", b"Workspace health"),
            ("/health.js", b"renderStandard"),
            ("/health-common.js", b"export function start"),
        ):
            status, policy, body = fetch(path)
            assert status == 200 and marker in body and "frame-ancestors 'none'" in policy
        assert fetch("/health/live")[0] == 200
        before = loads(fetch("/api/health")[2])
        direct_tools.call_tool("file_write", {"path": "secret.txt", "content": "private file text"})
        with raises(WorkspaceToolError):
            direct_tools.call_tool("run_python", {"script": "raise SystemExit(3)"})
        status, _, raw = fetch("/api/health")
    finally:
        connection.close()
    after = loads(raw)
    assert status == 200 and after.get("ready", False) and after.get("incarnation", "") == before.get("incarnation", "")
    assert after.get("readiness", {}).get("checks", {}).get("workspaceWritable", False)
    assert after.get("egress_policy", "") == "direct"
    by_tool = after.get("requests", {}).get("by_operation", {})
    assert by_tool.get("file_write", {}).get("outcomes", {}) == {"OK": 1}
    assert by_tool.get("run_python", {}).get("outcomes", {}) == {"PROCESS_FAILED": 1}
    assert after.get("requests", {}).get("received", 0) == before.get("requests", {}).get("received", 0) + 2
    assert [item.get("operation", "") for item in after.get("recent_outcomes", [])][:2] == ["run_python", "file_write"]
    assert b"private file text" not in raw and b"secret.txt" not in raw and TOKEN.encode() not in raw


def test_unhealthy_readiness_names_each_failed_check(direct_tools, monkeypatch):
    monkeypatch.setattr("server.shutil_which", lambda name: "" if name == "rg" else shutil_which(name))
    with raises(WorkspaceToolError) as unhealthy:
        direct_tools.health()
    assert unhealthy.value.status_code == 503
    assert unhealthy.value.code == "unhealthy"
    assert "required tools not on PATH: rg" in str(unhealthy.value)


def test_scoped_read_returns_provenance_and_rejects_escapes(direct_tools, tmp_path):
    repository = task_folder(tmp_path, direct_tools) / "repository"
    (repository / "package").mkdir(parents=True)
    source = "VALUE = 'café'\n"
    (repository / "package" / "module.py").write_text(source, encoding="utf-8")
    read = direct_tools.call_tool(
        "file_read_scoped",
        {"repository_root": "repository", "path": "package/module.py"},
    )
    assert read == {
        "repository_root": str(repository),
        "resolved_path": str(repository / "package" / "module.py"),
        "relative_path": "package/module.py",
        "content": source,
        "content_sha256": sha256(source.encode("utf-8")).hexdigest(),
        "byte_count": len(source.encode("utf-8")),
    }
    for path in (
        "../outside.py",
        "/etc/passwd",
        "package//module.py",
        "./package/module.py",
    ):
        with raises(WorkspaceToolError) as rejected:
            direct_tools.call_tool("file_read_scoped", {"repository_root": "repository", "path": path})
        assert rejected.value.code == "bad_arguments"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = 1\n", encoding="utf-8")
    (repository / "escape").symlink_to(outside)
    (repository / "secret_link.py").symlink_to(outside / "secret.py")
    for path in ("escape/secret.py", "secret_link.py"):
        with raises(WorkspaceToolError) as escaped:
            direct_tools.call_tool("file_read_scoped", {"repository_root": "repository", "path": path})
        assert escaped.value.code == "path_escape"


def test_scoped_read_returns_whole_files_and_enforces_encoding_and_regular_files(direct_tools, tmp_path):
    repository = task_folder(tmp_path, direct_tools) / "repository"
    (repository / "package").mkdir(parents=True)
    acquired = "#" + "x" * 300_000 + "\n"
    (repository / "above_old_limit.py").write_text(acquired, encoding="utf-8")
    observed = direct_tools.call_tool("file_read_scoped", {"repository_root": "repository", "path": "above_old_limit.py"})
    assert observed.get("content", "") == acquired
    assert observed.get("byte_count", 0) == len(acquired.encode("utf-8"))
    (repository / "latin1.py").write_bytes(b"caf\xe9\n")
    mkfifo(repository / "pipe.py")
    for arguments, code, status in (
        ({"path": "latin1.py"}, "invalid_source_encoding", 422),
        ({"path": "package"}, "not_file", 400),
        ({"path": "pipe.py"}, "bad_arguments", 400),
        ({"path": "missing.py"}, "not_found", 404),
        ({"path": "missing_folder/module.py"}, "not_found", 404),
        ({"path": "latin1.py/module.py"}, "not_directory", 400),
    ):
        with raises(WorkspaceToolError) as rejected:
            direct_tools.call_tool("file_read_scoped", {"repository_root": "repository", **arguments})
        assert (rejected.value.code, rejected.value.status_code) == (code, status)


def test_scoped_bytes_refuse_symlinks_swapped_in_after_resolution(tmp_path):
    """Pass unresolved targets, as if a component became a symlink after resolve() checked it."""
    repository = tmp_path.resolve() / "repository"
    repository.mkdir()
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = 1\n", encoding="utf-8")
    (repository / "package").symlink_to(outside)
    (repository / "module.py").symlink_to(outside / "secret.py")
    for target in (repository / "package" / "secret.py", repository / "module.py"):
        with raises(ToolError) as refused:
            scoped_repository_bytes(repository, target)
        assert refused.value.code == "path_escape"


def test_scoped_search_returns_relative_python_leads_only(direct_tools, tmp_path):
    repository = task_folder(tmp_path, direct_tools) / "repository"
    (repository / "package").mkdir(parents=True)
    (repository / "package" / "module.py").write_text("def target_symbol():\n    return 1\n", encoding="utf-8")
    (repository / "notes.txt").write_text("target_symbol\n", encoding="utf-8")
    found = direct_tools.call_tool(
        "file_search_scoped",
        {"repository_root": "repository", "symbol": "target_symbol"},
    )
    assert found == {
        "repository_root": str(repository),
        "symbol": "target_symbol",
        "matches": [{"relative_path": "package/module.py", "line": 1}],
        "truncated": False,
    }
    for position in range(65):
        (repository / f"module_{position:03}.py").write_text("def target_symbol():\n    return 1\n", encoding="utf-8")
    for allowance, expected, truncated in ((0, 66, False), (65, 65, True), (66, 66, False)):
        observed = direct_tools.call_tool(
            "file_search_scoped", {"repository_root": "repository", "symbol": "target_symbol", "max_results": allowance}
        )
        assert len(observed.get("matches", [])) == expected
        assert observed.get("truncated", True) is truncated
    for arguments, code in (
        ({"symbol": "target symbol"}, "bad_arguments"),
        ({"symbol": "target_symbol", "max_results": -1}, "bad_arguments"),
        ({"repository_root": "missing", "symbol": "target_symbol"}, "path_missing"),
        ({"repository_root": "..", "symbol": "target_symbol"}, "path_escape"),
    ):
        with raises(WorkspaceToolError) as rejected:
            direct_tools.call_tool("file_search_scoped", {"repository_root": "repository", **arguments})
        assert rejected.value.code == code


def test_pip_downloads_from_the_actual_index_without_forwarding(direct_tools, artifacts, tmp_path, monkeypatch):
    """Install a local fixture wheel into an owned target, never the interpreter environment."""
    package_directory = artifacts.directory / "packages"
    package_directory.mkdir()
    wheel = package_directory / "workspace_fixture-1.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("workspace_fixture.py", "VALUE = 42\n")
        archive.writestr(
            "workspace_fixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: workspace-fixture\nVersion: 1.0\n",
        )
        archive.writestr(
            "workspace_fixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("workspace_fixture-1.0.dist-info/RECORD", "")
    index = artifacts.directory / "simple" / "workspace-fixture"
    index.mkdir(parents=True)
    (index / "index.html").write_text(f'<a href="../../packages/{wheel.name}">{wheel.name}</a>', encoding="utf-8")
    target = tmp_path / "installed"
    monkeypatch.setenv("PIP_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("PIP_INDEX_URL", f"http://127.0.0.1:{artifacts.server_port}/simple/")
    monkeypatch.setenv("PIP_TARGET", str(target))
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    for name in (
        "PIP_NO_INDEX",
        "PIP_EXTRA_INDEX_URL",
        "PIP_REQUIRE_VIRTUALENV",
        "PIP_REQUIRE_VENV",
    ):
        monkeypatch.delenv(name, raising=False)
    result = direct_tools.call_tool("pip_install", {"packages": ["workspace-fixture==1.0"]}, timeout=30)
    assert result.get("ok", False) and result.get("exitCode", -1) == 0
    assert (target / "workspace_fixture.py").read_text() == "VALUE = 42\n"
    assert "/simple/workspace-fixture/" in artifacts.paths
    assert f"/packages/{wheel.name}" in artifacts.paths
    assert all(not path.startswith("/v1/") for path in artifacts.paths)


def test_git_uses_original_http_destination_and_native_remote_with_real_process_results(direct_tools, artifacts, tmp_path):
    seed = artifacts.directory / "seed"
    origin = artifacts.directory / "origin.git"
    require_process_success(run_git(["init", "--initial-branch=main", str(seed)]), "fixture init")
    (seed / "sample.txt").write_text("initial\n", encoding="utf-8")
    for arguments in (
        ["-C", str(seed), "add", "sample.txt"],
        [
            "-C",
            str(seed),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "initial",
        ],
        ["clone", "--bare", str(seed), str(origin)],
        ["--git-dir", str(origin), "update-server-info"],
    ):
        require_process_success(run_git(arguments), "fixture setup")
    destination = f"http://127.0.0.1:{artifacts.server_port}/origin.git"
    cloned = direct_tools.call_tool("git_clone", {"repoUrl": destination, "branch": "main"})
    assert cloned == {"repository": destination, "branch": "main", "cloned": True}
    assert direct_tools.call_tool("file_read", {"path": "sample.txt"}) == "initial\n"
    # Outside a request the server's own Git runs in the Workspace root, so it names the Task folder the clone went to.
    folder = str(task_folder(tmp_path, direct_tools))
    assert run_git(["-C", folder, "remote", "get-url", "origin"]).get("stdout", "").strip() == destination
    assert artifacts.paths and all(path.startswith("/origin.git/") for path in artifacts.paths)
    for arguments in (
        ["-C", folder, "config", "user.name", "Fixture"],
        ["-C", folder, "config", "user.email", "fixture@example.test"],
        ["-C", folder, "remote", "set-url", "origin", str(origin)],
    ):
        require_process_success(run_git(arguments), "fixture remote")
    direct_tools.call_tool("file_write", {"path": "sample.txt", "content": "changed\n"})
    direct_tools.call_tool("git_commit", {"message": "changed"})
    pushed = direct_tools.call_tool("git_push", {"remote": "origin", "branch": "main"})
    assert pushed == {"remote": "origin", "branch": "main", "pushed": True}
    assert run_git(["--git-dir", str(origin), "show", "main:sample.txt"]).get("stdout", "") == "changed\n"
    with raises(WorkspaceToolError) as missing:
        direct_tools.call_tool("git_clone", {"repoUrl": destination + "-missing", "branch": "main"})
    assert missing.value.code == "process_failed"
    assert direct_tools.call_tool("file_read", {"path": "sample.txt"}) == "changed\n"
