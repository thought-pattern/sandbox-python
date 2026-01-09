"""MCP server for Python sandbox environment."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
mcp = FastMCP("python-sandbox")


def resolve_path(path):
    """Resolve path within workspace, raise if outside."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    return target


@mcp.tool()
def file_read(path: str):
    """Read contents of a file."""
    return resolve_path(path).read_text()


@mcp.tool()
def file_write(path: str, content: str):
    """Write content to a file."""
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes"


@mcp.tool()
def file_patch(path: str, patches: list):
    """Apply find/replace patches to a file."""
    target = resolve_path(path)
    content = target.read_text()
    for p in patches:
        if p["old"] not in content:
            raise ValueError(f"Not found: {p['old'][:50]}...")
        content = content.replace(p["old"], p["new"], 1)
    target.write_text(content)
    return f"Applied {len(patches)} patches"


@mcp.tool()
def file_delete(path: str):
    """Delete a file."""
    resolve_path(path).unlink(missing_ok=False)
    return f"Deleted {path}"


@mcp.tool()
def file_list(path: str = ".", depth: int = 2):
    """List files in directory."""
    target = resolve_path(path)
    result = subprocess.run(
        ["find", str(target), "-maxdepth", str(depth), "-type", "f"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    lines = [l for l in result.stdout.strip().split("\n") if l]
    return [str(Path(f).relative_to(WORKSPACE)) for f in lines]


@mcp.tool()
def file_search(pattern: str, path: str = "."):
    """Search for pattern in files using ripgrep."""
    target = resolve_path(path)
    result = subprocess.run(
        ["rg", "--json", pattern, str(target)],
        capture_output=True, text=True, cwd=WORKSPACE
    )

    matches = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        if data["type"] == "match":
            d = data["data"]
            matches.append({
                "file": str(Path(d["path"]["text"]).relative_to(WORKSPACE)),
                "line": d["line_number"],
                "content": d["lines"]["text"].strip()
            })
    return matches


@mcp.tool()
def pip_install(packages: list):
    """Install Python packages."""
    for pkg in packages:
        if not re.match(r'^[a-zA-Z0-9_.-]+([=<>!~\[\]][a-zA-Z0-9._,<>=!~\[\]]*)?$', pkg):
            raise ValueError(f"Invalid package: {pkg}")
    result = subprocess.run(
        ["pip", "install", "--no-cache-dir"] + packages,
        capture_output=True, text=True, timeout=300
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


@mcp.tool()
def pip_uninstall(packages: list):
    """Uninstall Python packages."""
    for pkg in packages:
        if not re.match(r'^[a-zA-Z0-9_.-]+$', pkg):
            raise ValueError(f"Invalid package: {pkg}")
    result = subprocess.run(
        ["pip", "uninstall", "-y"] + packages,
        capture_output=True, text=True, timeout=60
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


@mcp.tool()
def pip_list():
    """List installed packages."""
    result = subprocess.run(["pip", "list", "--format=json"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


@mcp.tool()
def pip_freeze():
    """Get installed packages in requirements.txt format."""
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    return result.stdout


@mcp.tool()
def run_command(command: str, timeout: int = 60):
    """Execute shell command in workspace."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=WORKSPACE, timeout=timeout
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "exit_code": -1}


@mcp.tool()
def run_python(script: str, timeout: int = 60):
    """Execute Python code."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, cwd=WORKSPACE, timeout=timeout
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "exit_code": -1}


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
        capture_output=True, text=True, cwd=WORKSPACE
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return f"Cloned {repo_url} (branch: {branch})"


@mcp.tool()
def git_status():
    """Get current git status."""
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    return {
        "branch": branch.stdout.strip(),
        "changes": [l for l in status.stdout.strip().split("\n") if l]
    }


@mcp.tool()
def git_diff(staged: bool = False):
    """Get diff of changes."""
    cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE)
    return result.stdout


@mcp.tool()
def git_commit(message: str):
    """Stage all changes and commit."""
    add_result = subprocess.run(["git", "add", "-A"], capture_output=True, text=True, cwd=WORKSPACE)
    if add_result.returncode != 0:
        raise RuntimeError(add_result.stderr)
    result = subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


@mcp.tool()
def git_push(remote: str = "origin", branch=None):
    """Push commits to remote."""
    if branch is None:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=WORKSPACE
        )
        branch = result.stdout.strip()
    result = subprocess.run(
        ["git", "push", remote, branch],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return f"Pushed to {remote}/{branch}"


@mcp.tool()
def python_version():
    """Get Python version."""
    return sys.version


@mcp.resource("workspace://tree")
def workspace_tree():
    """Current workspace file tree."""
    result = subprocess.run(
        ["find", ".", "-maxdepth", "3", "-type", "f"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    return "\n".join(result.stdout.strip().split("\n")[:100])


@mcp.resource("workspace://git-log")
def git_log():
    """Recent git history."""
    result = subprocess.run(
        ["git", "log", "--oneline", "-20"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    return result.stdout


def health(request):
    return JSONResponse({"status": "healthy", "python": sys.version})


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "8080"))
    app = Starlette(routes=[
        Route("/health", health),
        Mount("/", app=mcp.sse_app()),
    ])
    uvicorn.run(app, host="0.0.0.0", port=port)
