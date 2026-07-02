"""Tests for the sandbox HTTP tool server.

server.py imports only the standard library, so it loads directly with no
stubbing. These cover workspace path confinement, the self-describing manifest,
and a tool round-trip against a temporary workspace.
"""

import pytest

import server


def test_resolve_path_allows_within_workspace():
    target = server.resolve_path("sub/file.py")
    assert target.is_relative_to(server.WORKSPACE)


def test_resolve_path_rejects_parent_traversal():
    with pytest.raises(ValueError):
        server.resolve_path("../etc/passwd")


def test_resolve_path_rejects_absolute_escape():
    with pytest.raises(ValueError):
        server.resolve_path("/etc/passwd")


def test_manifest_self_describes_required_params():
    tools = {t["name"]: t for t in server.build_manifest()["tools"]}
    assert "file_read" in tools
    assert "run_python" in tools
    write = tools["file_write"]
    assert write["description"]
    params = {p["name"]: p for p in write["parameters"]}
    assert params["path"]["required"] is True
    assert params["content"]["required"] is True


def test_manifest_reports_optional_defaults():
    tools = {t["name"]: t for t in server.build_manifest()["tools"]}
    params = {p["name"]: p for p in tools["file_list"]["parameters"]}
    assert params["depth"]["required"] is False
    assert params["depth"]["default"] == 2


def test_tool_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path.resolve())
    server.file_write("notes/todo.txt", "hello")
    assert server.file_read("notes/todo.txt") == "hello"
