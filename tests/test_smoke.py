# Copyright 2025-2026 Jason E. Robinson.
# SPDX-License-Identifier: Apache-2.0

"""PDC-DEVELOP batch 211: actual HTTP tools and subprocesses without a container runtime."""

from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from os import environ, mkfifo
from pathlib import Path
from sys import executable
from threading import Thread
from zipfile import ZipFile

from pytest import fixture, raises

import server
from workspace_client import WorkspaceClient, WorkspaceToolError

TOKEN = "direct-interface-test-" + "t" * 32


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
    """Use the real server configuration, HTTP handlers, client and subprocess owner."""
    monkeypatch.setenv("PATH", f"{Path(executable).parent}:{environ.get('PATH', '')}")
    for name in ("WORKSPACE", "EGRESS_POLICY", "MAX_OUTPUT_BYTES", "MAX_FILE_BYTES", "MAX_TOOL_TIMEOUT"):
        monkeypatch.setattr(server, name, getattr(server, name))
    args = server.parse_args(["--workspace", str(tmp_path / "work"), "--auth-token", TOKEN])
    server.configure(args)
    with server.ToolHTTPServer(("127.0.0.1", 0), server.ToolHandler, auth_token=TOKEN) as endpoint:
        worker = Thread(target=endpoint.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            with WorkspaceClient(f"http://127.0.0.1:{endpoint.server_port}", TOKEN) as client:
                yield client
        finally:
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


def test_authenticated_direct_interface_runs_real_tools_and_preserves_failures(direct_tools):
    assert direct_tools.health().get("egressPolicy", "") == "direct"
    with WorkspaceClient(direct_tools.base_url, "incorrect-identity") as unauthorized:
        with raises(WorkspaceToolError) as rejected:
            unauthorized.fetch_manifest()
    assert rejected.value.status_code == 401
    assert direct_tools.fetch_manifest().get("protocol", "") == "tapestry.workspace.http"
    assert direct_tools.call_tool("file_write", {"path": "smoke.txt", "content": "hello"}).get("bytesWritten", 0) == 5
    assert direct_tools.call_tool("file_read", {"path": "smoke.txt"}) == "hello"
    result = direct_tools.call_tool("run_python", {"script": "print(6 * 7)", "timeout": 2})
    assert result.get("exitCode", -1) == 0 and result.get("stdout", "").strip() == "42"
    with raises(WorkspaceToolError) as failed:
        direct_tools.call_tool("run_python", {"script": "raise SystemExit(7)", "timeout": 2})
    assert failed.value.result.get("exitCode", -1) == 7


def test_unhealthy_readiness_names_each_failed_check(direct_tools, monkeypatch):
    real_which = server.shutil_which
    monkeypatch.setattr(server, "shutil_which", lambda name: "" if name == "rg" else real_which(name))
    with raises(WorkspaceToolError) as unhealthy:
        direct_tools.health()
    assert unhealthy.value.status_code == 503
    assert unhealthy.value.code == "unhealthy"
    assert "required tools not on PATH: rg" in str(unhealthy.value)


def test_scoped_read_returns_provenance_and_rejects_escapes(direct_tools, tmp_path):
    repository = server.WORKSPACE / "repository"
    (repository / "package").mkdir(parents=True)
    source = "VALUE = 'café'\n"
    (repository / "package" / "module.py").write_text(source, encoding="utf-8")
    read = direct_tools.call_tool("file_read_scoped", {"repository_root": "repository", "path": "package/module.py"})
    assert read == {
        "repository_root": str(repository),
        "resolved_path": str(repository / "package" / "module.py"),
        "relative_path": "package/module.py",
        "content": source,
        "content_sha256": sha256(source.encode("utf-8")).hexdigest(),
        "byte_count": len(source.encode("utf-8")),
    }
    for path in ("../outside.py", "/etc/passwd", "package//module.py", "./package/module.py"):
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


def test_scoped_read_enforces_byte_limit_encoding_and_regular_files(direct_tools):
    repository = server.WORKSPACE / "repository"
    (repository / "package").mkdir(parents=True)
    (repository / "large.py").write_text("x" * 65, encoding="utf-8")
    (repository / "latin1.py").write_bytes(b"caf\xe9\n")
    mkfifo(repository / "pipe.py")
    for arguments, code, status in (
        ({"path": "large.py", "max_bytes": 64}, "file_too_large", 413),
        ({"path": "latin1.py"}, "invalid_source_encoding", 422),
        ({"path": "package"}, "not_file", 400),
        ({"path": "pipe.py"}, "bad_arguments", 400),
        ({"path": "missing.py"}, "not_found", 404),
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
        with raises(server.ToolError) as refused:
            server.scoped_repository_bytes(repository, target, 1024)
        assert refused.value.code == "path_escape"


def test_scoped_search_returns_relative_python_leads_only(direct_tools):
    repository = server.WORKSPACE / "repository"
    (repository / "package").mkdir(parents=True)
    (repository / "package" / "module.py").write_text("def target_symbol():\n    return 1\n", encoding="utf-8")
    (repository / "notes.txt").write_text("target_symbol\n", encoding="utf-8")
    found = direct_tools.call_tool("file_search_scoped", {"repository_root": "repository", "symbol": "target_symbol"})
    assert found == {
        "repository_root": str(repository),
        "symbol": "target_symbol",
        "matches": [{"relative_path": "package/module.py", "line": 1}],
        "truncated": False,
    }
    for arguments, code in (
        ({"symbol": "target symbol"}, "bad_arguments"),
        ({"symbol": "target_symbol", "max_results": 65}, "bad_arguments"),
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
            "workspace_fixture-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: workspace-fixture\nVersion: 1.0\n"
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
    for name in ("PIP_NO_INDEX", "PIP_EXTRA_INDEX_URL", "PIP_REQUIRE_VIRTUALENV", "PIP_REQUIRE_VENV"):
        monkeypatch.delenv(name, raising=False)
    result = direct_tools.call_tool("pip_install", {"packages": ["workspace-fixture==1.0"]}, timeout=30)
    assert result.get("ok", False) and result.get("exitCode", -1) == 0
    assert (target / "workspace_fixture.py").read_text() == "VALUE = 42\n"
    assert "/simple/workspace-fixture/" in artifacts.paths
    assert f"/packages/{wheel.name}" in artifacts.paths
    assert all(not path.startswith("/v1/") for path in artifacts.paths)


def test_git_uses_original_http_destination_and_native_remote_with_real_process_results(direct_tools, artifacts):
    seed = artifacts.directory / "seed"
    origin = artifacts.directory / "origin.git"
    server.require_process_success(server.run_git(["init", "--initial-branch=main", str(seed)]), "fixture init")
    (seed / "sample.txt").write_text("initial\n", encoding="utf-8")
    for arguments in (
        ["-C", str(seed), "add", "sample.txt"],
        ["-C", str(seed), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "initial"],
        ["clone", "--bare", str(seed), str(origin)],
        ["--git-dir", str(origin), "update-server-info"],
    ):
        server.require_process_success(server.run_git(arguments), "fixture setup")
    destination = f"http://127.0.0.1:{artifacts.server_port}/origin.git"
    cloned = direct_tools.call_tool("git_clone", {"repoUrl": destination, "branch": "main"})
    assert cloned == {"repository": destination, "branch": "main", "cloned": True}
    assert direct_tools.call_tool("file_read", {"path": "sample.txt"}) == "initial\n"
    assert server.run_git(["remote", "get-url", "origin"]).get("stdout", "").strip() == destination
    assert artifacts.paths and all(path.startswith("/origin.git/") for path in artifacts.paths)
    for arguments in (
        ["config", "user.name", "Fixture"],
        ["config", "user.email", "fixture@example.test"],
        ["remote", "set-url", "origin", str(origin)],
    ):
        server.require_process_success(server.run_git(arguments), "fixture remote")
    direct_tools.call_tool("file_write", {"path": "sample.txt", "content": "changed\n"})
    direct_tools.call_tool("git_commit", {"message": "changed"})
    pushed = direct_tools.call_tool("git_push", {"remote": "origin", "branch": "main"})
    assert pushed == {"remote": "origin", "branch": "main", "pushed": True}
    assert server.run_git(["--git-dir", str(origin), "show", "main:sample.txt"]).get("stdout", "") == "changed\n"
    with raises(WorkspaceToolError) as missing:
        direct_tools.call_tool("git_clone", {"repoUrl": destination + "-missing", "branch": "main"})
    assert missing.value.code == "process_failed"
    assert direct_tools.call_tool("file_read", {"path": "sample.txt"}) == "changed\n"
