"""MCP server for Python sandbox environment."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
mcp = FastMCP("python-sandbox")

COMMAND_TIMEOUT = 60
CLONE_TIMEOUT = 300
MAX_FILE_BYTES = 10 * 1024 * 1024
# Disable git's interactive credential prompt so clone/push fail fast instead
# of blocking forever waiting on stdin.
GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def resolve_path(path):
    """Resolve path within workspace, raise if outside."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    return target


@mcp.tool()
def file_read(path: str):
    """Read contents of a file."""
    target = resolve_path(path)
    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_BYTES})")
    content = target.read_text()
    return content


@mcp.tool()
def file_write(path: str, content: str):
    """Write content to a file."""
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    message = f"Wrote {len(content)} bytes"
    return message


@mcp.tool()
def file_patch(path: str, patches: list):
    """Apply find/replace patches to a file."""
    target = resolve_path(path)
    content = target.read_text()
    for p in patches:
        old = p.get("old")
        new = p.get("new")
        if old is None or new is None:
            raise ValueError("Each patch must have 'old' and 'new'")
        if old not in content:
            raise ValueError(f"Not found: {old[:50]}...")
        content = content.replace(old, new, 1)
    target.write_text(content)
    message = f"Applied {len(patches)} patches"
    return message


@mcp.tool()
def file_delete(path: str):
    """Delete a file."""
    resolve_path(path).unlink(missing_ok=False)
    message = f"Deleted {path}"
    return message


@mcp.tool()
def file_list(path: str = ".", depth: int = 2):
    """List files in directory."""
    target = resolve_path(path)
    result = subprocess.run(
        ["find", str(target), "-maxdepth", str(depth), "-type", "f"],
        capture_output=True,
        text=True,
        cwd=WORKSPACE,
        timeout=COMMAND_TIMEOUT,
    )
    lines = [line for line in result.stdout.strip().split("\n") if line]
    files = [str(Path(f).relative_to(WORKSPACE)) for f in lines]
    return files


@mcp.tool()
def file_search(pattern: str, path: str = "."):
    """Search for pattern in files using ripgrep."""
    target = resolve_path(path)
    try:
        result = subprocess.run(
            ["rg", "--json", pattern, str(target)],
            capture_output=True,
            text=True,
            cwd=WORKSPACE,
            timeout=COMMAND_TIMEOUT,
        )
    except FileNotFoundError:
        raise RuntimeError("ripgrep (rg) is not installed")

    matches = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") != "match":
            continue
        d = data.get("data", {})
        path_text = d.get("path", {}).get("text", "")
        if not path_text:
            continue
        lines_text = d.get("lines", {}).get("text", "")
        matches.append(
            {
                "file": str(Path(path_text).relative_to(WORKSPACE)),
                "line": d.get("line_number"),
                "content": lines_text.strip(),
            }
        )
    return matches


@mcp.tool()
def pip_install(packages: list):
    """Install Python packages."""
    for pkg in packages:
        if not re.match(r"^[a-zA-Z0-9_.-]+([=<>!~\[\]][a-zA-Z0-9._,<>=!~\[\]]*)?$", pkg):
            raise ValueError(f"Invalid package: {pkg}")
    result = subprocess.run(
        ["pip", "install", "--no-cache-dir"] + packages, capture_output=True, text=True, timeout=300
    )
    output = {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    return output


@mcp.tool()
def pip_uninstall(packages: list):
    """Uninstall Python packages."""
    for pkg in packages:
        if not re.match(r"^[a-zA-Z0-9_.-]+$", pkg):
            raise ValueError(f"Invalid package: {pkg}")
    result = subprocess.run(
        ["pip", "uninstall", "-y"] + packages, capture_output=True, text=True, timeout=COMMAND_TIMEOUT
    )
    output = {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    return output


@mcp.tool()
def pip_list():
    """List installed packages."""
    result = subprocess.run(
        ["pip", "list", "--format=json"], capture_output=True, text=True, timeout=COMMAND_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    packages = json.loads(result.stdout)
    return packages


@mcp.tool()
def pip_freeze():
    """Get installed packages in requirements.txt format."""
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
    freeze = result.stdout
    return freeze


@mcp.tool()
def run_command(command: str, timeout: int = 60):
    """Execute shell command in workspace."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=WORKSPACE, timeout=timeout)
        output = {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        output = {"stdout": "", "stderr": f"Timed out after {timeout}s", "exit_code": -1}
    return output


@mcp.tool()
def run_python(script: str, timeout: int = 60):
    """Execute Python code."""
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=WORKSPACE, timeout=timeout)
        output = {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        output = {"stdout": "", "stderr": f"Timed out after {timeout}s", "exit_code": -1}
    return output


@mcp.tool()
def git_init(branch: str = "main"):
    """Initialize a new git repository in the workspace."""
    result = subprocess.run(
        ["git", "init", f"--initial-branch={branch}"],
        capture_output=True,
        text=True,
        cwd=WORKSPACE,
        env=GIT_ENV,
        timeout=COMMAND_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    message = result.stdout.strip()
    return message


@mcp.tool()
def git_clone(repo_url: str, branch: str = "main"):
    """Clone repository into workspace."""
    for item in WORKSPACE.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    result = subprocess.run(
        ["git", "clone", "--branch", branch, repo_url, "."],
        capture_output=True,
        text=True,
        cwd=WORKSPACE,
        env=GIT_ENV,
        timeout=CLONE_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    message = f"Cloned {repo_url} (branch: {branch})"
    return message


@mcp.tool()
def git_status():
    """Get current git status."""
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=COMMAND_TIMEOUT
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=COMMAND_TIMEOUT
    )
    changes = [line for line in status.stdout.strip().split("\n") if line]
    output = {"branch": branch.stdout.strip(), "changes": changes}
    return output


@mcp.tool()
def git_diff(staged: bool = False):
    """Get diff of changes."""
    cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=COMMAND_TIMEOUT)
    diff = result.stdout
    return diff


@mcp.tool()
def git_commit(message: str):
    """Stage all changes and commit."""
    add_result = subprocess.run(
        ["git", "add", "-A"], capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=COMMAND_TIMEOUT
    )
    if add_result.returncode != 0:
        raise RuntimeError(add_result.stderr)
    result = subprocess.run(
        ["git", "commit", "-m", message], capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=COMMAND_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    output = result.stdout
    return output


@mcp.tool()
def git_push(remote: str = "origin", branch=None):
    """Push commits to remote."""
    if branch is None:
        result = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=COMMAND_TIMEOUT
        )
        branch = result.stdout.strip()
    result = subprocess.run(
        ["git", "push", remote, branch], capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=CLONE_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    message = f"Pushed to {remote}/{branch}"
    return message


@mcp.tool()
def python_version():
    """Get Python version."""
    version = sys.version
    return version


@mcp.resource("workspace://tree")
def workspace_tree():
    """Current workspace file tree."""
    result = subprocess.run(
        ["find", ".", "-maxdepth", "3", "-type", "f"], capture_output=True, text=True, cwd=WORKSPACE, timeout=COMMAND_TIMEOUT
    )
    tree = "\n".join(result.stdout.strip().split("\n")[:100])
    return tree


@mcp.resource("workspace://git-log")
def git_log():
    """Recent git history."""
    result = subprocess.run(
        ["git", "log", "--oneline", "-20"], capture_output=True, text=True, cwd=WORKSPACE, env=GIT_ENV, timeout=COMMAND_TIMEOUT
    )
    log = result.stdout
    return log


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request):
    """Readiness probe: verify the workspace is writable and required tools exist.

    Returns 200 when every dependency the sandbox tools rely on is present, and
    503 otherwise so orchestrators (ECS/ALB health checks) can react. A plain
    'process is up' probe would report healthy even with /workspace read-only or
    ripgrep missing.
    """
    required_tools = ("git", "rg", "find", "pip")
    tools = {tool: shutil.which(tool) is not None for tool in required_tools}

    workspace_writable = False
    workspace_free_mb = None
    if WORKSPACE.is_dir():
        try:
            with tempfile.NamedTemporaryFile(dir=WORKSPACE):
                pass
            workspace_writable = True
        except OSError:
            workspace_writable = False
        workspace_free_mb = shutil.disk_usage(WORKSPACE).free // (1024 * 1024)

    healthy = workspace_writable and all(tools.values())
    payload = {
        "status": "healthy" if healthy else "unhealthy",
        "python": sys.version,
        "checks": {
            "workspace_path": str(WORKSPACE),
            "workspace_writable": workspace_writable,
            "workspace_free_mb": workspace_free_mb,
            "tools": tools,
        },
    }
    status_code = 200 if healthy else 503
    response = JSONResponse(payload, status_code=status_code)
    return response


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "8080"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
