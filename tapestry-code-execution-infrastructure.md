# Tapestry Code Execution Infrastructure

## Design Document

**Date:** 2026-01-02  
**Status:** Design  
**Stage:** Prototype → MVP → GA

---

## 1. Problem Statement

Tapestry needs mechanisms to work with code: starting containers, downloading codebases, examining code, modifying code, executing code, and pushing changes back to repositories.

The requirement is for a **headless development environment**—containers that expose tools via a standard protocol, with no opinions about what agent is driving them. This allows Tapestry's Actor/Regulator architecture to maintain control over all operations.

### Requirements

- Self-hosted (no external service dependencies)
- Python-based internal mechanisms
- Agent logic separate from execution environment
- Regulator can intercept all tool calls
- Tool sequences logged for proof construction
- Portable across Docker (Prototype/MVP) and ECS/Fargate (GA)

---

## 2. Options Evaluated

### 2.1 OpenHands (formerly OpenDevin)

**66k GitHub stars, MIT licensed**

OpenHands provides containerized development environments with:
- Docker sandbox with full OS capabilities
- SSH-mediated interface to containers
- MCP integration (added in V1 SDK)
- Jupyter kernel, browser automation, REST API

**Verdict:** Too tightly coupled. OpenHands bundles the agent intelligence *with* the execution environment. Would require fighting their architecture to use just the container runtime while replacing their agent with Tapestry's Actor/Regulator.

### 2.2 E2B

**Cloud-hosted Firecracker microVMs (~150ms cold start)**

- Python/JS SDK for control
- File operations, shell execution, package installation
- You bring your own LLM/agent

**Verdict:** Good architecture (headless), but cloud-hosted. Infrastructure is open source for self-hosting but adds complexity. External dependency for core functionality.

### 2.3 Gru Sandbox (gbox)

**Lightweight MCP sandbox, self-hostable**

- Native MCP support
- Simpler than OpenHands
- Less feature-rich

**Verdict:** Viable for pure code execution, but lacks full development environment features.

### 2.4 Claude Code Web Architecture

**Anthropic's approach for their web-based coding agent**

Architecture:
- Isolated VM/sandbox running Claude Code CLI with `--dangerously-skip-permissions`
- Two isolation boundaries: filesystem + network
- Git operations via proxy with scoped credentials
- Open sourced `anthropic-experimental/sandbox-runtime` for OS-level sandboxing

**Key insight:** Claude Code Web bundles the agent inside the container. Tapestry needs the agent *outside* the container, calling in via MCP.

### 2.5 Custom MCP Server in Container (Selected)

**Build a purpose-built MCP server for Tapestry's container runtime**

- Use existing MCP primitives as reference
- ~200-300 lines of Python
- Full control over tool definitions
- Regulator intercepts all calls before execution
- Same image works local and cloud

**Verdict:** Selected approach. Minimal code, maximum control, portable across environments.

---

## 3. Architecture

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                        TAPESTRY                              │
│                                                              │
│  Actor ──► MCP Client (Python) ──────────────┐              │
│                                               │              │
│  Regulator ◄── intercepts calls ─────────────┤              │
│                                               │              │
└───────────────────────────────────────────────┼──────────────┘
                                                │ HTTP
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONTAINER                                 │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            MCP Server (Python + FastMCP)              │  │
│  │                                                        │  │
│  │  Tools:                                               │  │
│  │    file_read/write/patch/delete/list/search           │  │
│  │    pip_install/uninstall/list/freeze                  │  │
│  │    run_command, run_python, python_version            │  │
│  │    git_init/clone/status/diff/commit/push             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  /workspace/  ← mounted or cloned repo                      │
│  Runtimes: python, git, ripgrep                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Agent outside container | Regulator can gate every tool call; proof extraction possible |
| MCP protocol | Standardized, tools self-describe, portable |
| Streamable HTTP transport | One MCP transport local and cloud; works over the network |
| Same image everywhere | Build once, deploy anywhere |

### 3.3 Comparison to Claude Code Web

| Aspect | Claude Code Web | Tapestry MCP Container |
|--------|-----------------|------------------------|
| Agent location | Inside container | Outside container |
| Tool interface | Direct bash/file access | MCP protocol over HTTP |
| Agent logic | Claude Code CLI bundled | Tapestry Actor (separate) |
| Oversight | Permission prompts | Regulator intercepts tool calls |
| Learning | None | Proof extraction from tool sequences |

---

## 4. Environment Progression

| Stage | Container Runtime | Orchestration | MCP Transport |
|-------|-------------------|---------------|---------------|
| Prototype | Docker/Podman local | `manager.py` (CLI subprocess) | Streamable HTTP |
| MVP | Docker/Podman local/remote | `manager.py` (CLI subprocess) | Streamable HTTP |
| GA | ECS/Fargate | `manager.py` (boto3) | Streamable HTTP |

The MCP server code stays identical across all environments—only the orchestration layer changes.

---

## 5. Implementation

### 5.1 Project Structure

```
sandbox-python/                 # this repo: container image + host-side manager
├── container/
│   ├── Containerfile           # Python 3.12-slim image
│   ├── requirements.txt        # fastmcp, starlette, uvicorn
│   └── server.py               # MCP server (runs IN container)
├── manager.py                  # container lifecycle functions (host side)
├── requirements.txt            # boto3 (Fargate management)
└── README.md

Tapestry-side integration (separate codebase, not in this repo):
    agent/dev_environment.py     # MCP client wrapper
    agent/code_agent.py          # Actor/Regulator orchestration loop
```

### 5.2 MCP Server

The MCP server runs inside the container and exposes development tools.

**File:** `container/server.py`

```python
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
```

**File:** `container/requirements.txt`

```
fastmcp==2.14.2
starlette
uvicorn
```

**File:** `container/Containerfile`

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/mcp/
RUN pip install --no-cache-dir -r /opt/mcp/requirements.txt

COPY server.py /opt/mcp/

RUN mkdir /workspace
WORKDIR /workspace
ENV WORKSPACE=/workspace

EXPOSE 8080

ENTRYPOINT ["python", "/opt/mcp/server.py"]
```

### 5.3 Container Manager

Container lifecycle functions for local runtimes (Docker/Podman CLI) and AWS Fargate (boto3). Configuration is a plain dict merged over `DEFAULT_CONFIG`; there are no classes or dataclasses.

**File:** `manager.py`

```python
"""Container lifecycle management for Python sandbox."""

import subprocess

import boto3

try:
    ecs = boto3.client("ecs")
    ecs_init_error = None
except Exception as exc:  # region/profile/config resolution errors surface here
    ecs = None
    ecs_init_error = exc

task_def_cache = {}

DEFAULT_CONFIG = {
    "image": "python-sandbox:latest",
    "workspace_mount": "",
    "environment": {},
    "memory_limit": "2g",
    "cpu_limit": 1.0,
    "port": 8080,
}

CREATE_TIMEOUT = 120
RUNTIME_TIMEOUT = 60


def require_ecs():
    """Return the ECS client, or raise with the original initialization error.

    Returns:
        The boto3 ECS client.

    Raises:
        RuntimeError: If the client failed to initialize (missing region,
            credentials, or configuration).
    """
    if ecs is None:
        raise RuntimeError(f"AWS ECS client unavailable: {ecs_init_error}")
    return ecs


def parse_memory(mem):
    """Parse a memory string like '2g' or '512m' into a MB string.

    Args:
        mem: Memory size such as '2g' or '512m'.

    Returns:
        The size in megabytes as a string, e.g. '2048'.

    Raises:
        ValueError: If the format is not recognized.
    """
    mem = mem.lower()
    if mem.endswith("g"):
        mb = str(int(mem[:-1]) * 1024)
    elif mem.endswith("m"):
        mb = mem[:-1]
    else:
        raise ValueError(f"Invalid memory format: {mem}. Use '2g' or '512m'.")
    return mb


def create_container(config, runtime="docker"):
    """Create a sandbox container from a config dict.

    Args:
        config: Partial container config; missing keys fall back to
            DEFAULT_CONFIG.
        runtime: Container runtime executable ('docker' or 'podman').

    Returns:
        The created container id.

    Raises:
        RuntimeError: If the runtime fails to create the container.
    """
    config = {**DEFAULT_CONFIG, **config}
    port = config.get("port")
    cmd = [
        runtime,
        "create",
        "--publish",
        f"{port}:{port}",
        "--memory",
        config.get("memory_limit"),
        "--cpus",
        str(config.get("cpu_limit")),
        "--env",
        f"MCP_PORT={port}",
    ]
    for k, v in config.get("environment", {}).items():
        cmd.extend(["--env", f"{k}={v}"])
    workspace_mount = config.get("workspace_mount")
    if workspace_mount:
        # --mount separates source/destination by commas rather than colons,
        # so Windows source paths (e.g. C:\work) are not mis-split.
        cmd.extend(["--mount", f"type=bind,source={workspace_mount},destination=/workspace"])
    cmd.append(config.get("image"))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=CREATE_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create container: {result.stderr}")
    container_id = result.stdout.strip()
    return container_id


def start_container(container_id, runtime="docker"):
    """Start a previously created container.

    Args:
        container_id: Id returned by create_container.
        runtime: Container runtime executable.

    Raises:
        RuntimeError: If the runtime fails to start the container.
    """
    result = subprocess.run(
        [runtime, "start", container_id], capture_output=True, text=True, timeout=RUNTIME_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")


def stop_container(container_id, runtime="docker"):
    """Stop a running container.

    Args:
        container_id: Id of the container to stop.
        runtime: Container runtime executable.

    Raises:
        RuntimeError: If the runtime fails to stop the container.
    """
    result = subprocess.run(
        [runtime, "stop", "-t", "10", container_id], capture_output=True, text=True, timeout=RUNTIME_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to stop container: {result.stderr}")


def destroy_container(container_id, runtime="docker"):
    """Remove a container, force-killing it if still running.

    Args:
        container_id: Id of the container to remove.
        runtime: Container runtime executable.

    Raises:
        RuntimeError: If the runtime fails to remove the container.
    """
    result = subprocess.run(
        [runtime, "rm", "-f", container_id], capture_output=True, text=True, timeout=RUNTIME_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to destroy container: {result.stderr}")


def create_fargate_task(config, cluster, subnet_ids, security_group_ids):
    """Run the sandbox as a Fargate task, reusing a cached task definition.

    Args:
        config: Partial container config; missing keys fall back to
            DEFAULT_CONFIG.
        cluster: ECS cluster name or ARN.
        subnet_ids: Non-empty list of subnet ids for the task ENI.
        security_group_ids: Non-empty list of security group ids.

    Returns:
        The running task ARN.

    Raises:
        ValueError: If subnet_ids or security_group_ids is empty.
        RuntimeError: If ECS is unavailable or task placement fails.
    """
    client = require_ecs()
    if not subnet_ids:
        raise ValueError("subnet_ids must not be empty")
    if not security_group_ids:
        raise ValueError("security_group_ids must not be empty")

    config = {**DEFAULT_CONFIG, **config}
    # Environment is baked into the task definition, so it must be part of the
    # cache key or two configs differing only in env would share a definition.
    env_key = ",".join(f"{k}={v}" for k, v in sorted(config.get("environment", {}).items()))
    key = f"{config.get('image')}:{config.get('memory_limit')}:{config.get('cpu_limit')}:{config.get('port')}:{env_key}"

    task_def = task_def_cache.get(key)
    if task_def is None:
        task_def = register_task_def(config)
        task_def_cache[key] = task_def

    resp = client.run_task(
        cluster=cluster,
        taskDefinition=task_def,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {"subnets": subnet_ids, "securityGroups": security_group_ids, "assignPublicIp": "DISABLED"}
        },
    )

    failures = resp.get("failures")
    if failures:
        raise RuntimeError(f"Failed to run Fargate task: {failures}")
    tasks = resp.get("tasks")
    if not tasks:
        raise RuntimeError("run_task returned no tasks and no failures")
    task_arn = tasks[0].get("taskArn")
    return task_arn


def register_task_def(config):
    """Register an ECS task definition for the sandbox container.

    Args:
        config: Partial container config; missing keys fall back to
            DEFAULT_CONFIG.

    Returns:
        The registered task definition ARN.

    Raises:
        RuntimeError: If ECS is unavailable.
    """
    client = require_ecs()

    config = {**DEFAULT_CONFIG, **config}
    cpu_limit = config.get("cpu_limit")
    if not isinstance(cpu_limit, (int, float)):
        raise ValueError(f"cpu_limit must be numeric, got {cpu_limit!r}")

    env = [{"name": "MCP_PORT", "value": str(config.get("port"))}]
    env.extend({"name": k, "value": v} for k, v in config.get("environment", {}).items())

    resp = client.register_task_definition(
        family="python-sandbox",
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu=str(int(cpu_limit * 1024)),
        memory=parse_memory(config.get("memory_limit")),
        containerDefinitions=[
            {
                "name": "sandbox",
                "image": config.get("image"),
                "essential": True,
                "environment": env,
                "portMappings": [{"containerPort": config.get("port")}],
            }
        ],
    )
    task_def_arn = resp.get("taskDefinition", {}).get("taskDefinitionArn")
    return task_def_arn


def wait_for_fargate_task(task_arn, cluster):
    """Block until the given Fargate task reaches the RUNNING state.

    Args:
        task_arn: ARN of the task to wait on.
        cluster: ECS cluster name or ARN.

    Raises:
        RuntimeError: If ECS is unavailable.
    """
    client = require_ecs()
    waiter = client.get_waiter("tasks_running")
    waiter.wait(cluster=cluster, tasks=[task_arn])


def stop_fargate_task(task_arn, cluster):
    """Stop a running Fargate task.

    Args:
        task_arn: ARN of the task to stop.
        cluster: ECS cluster name or ARN.

    Raises:
        RuntimeError: If ECS is unavailable.
    """
    client = require_ecs()
    client.stop_task(cluster=cluster, task=task_arn)


def get_fargate_endpoint(task_arn, cluster, port):
    """Resolve the private HTTP endpoint of a running Fargate task.

    Args:
        task_arn: ARN of the running task.
        cluster: ECS cluster name or ARN.
        port: Container port to address.

    Returns:
        An 'http://IP:port' endpoint string.

    Raises:
        RuntimeError: If ECS is unavailable or the task IP cannot be found.
    """
    client = require_ecs()
    resp = client.describe_tasks(cluster=cluster, tasks=[task_arn])
    tasks = resp.get("tasks")
    if not tasks:
        raise RuntimeError("Task not found")
    task = tasks[0]

    for att in task.get("attachments", []):
        if att.get("type") == "ElasticNetworkInterface":
            for d in att.get("details", []):
                if d.get("name") == "privateIPv4Address":
                    ip = d.get("value")
                    if ip:
                        endpoint = f"http://{ip}:{port}"
                        return endpoint

    raise RuntimeError("Could not find task IP address")
```

### 5.4 Dev Environment Client

Wrapper connecting Tapestry to the containerized MCP server.

> Status: forward-looking Tapestry integration — the MCP client and agent loop below are not yet implemented in this repo. The code targets the flat-function `manager.py` API from §5.3 (dict config, no manager classes).

**File:** `agent/dev_environment.py`

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import manager


class DevEnvironment:
    """Manages a containerized dev environment via MCP over Streamable HTTP."""

    def __init__(self, config=None, backend="local", runtime="podman", fargate=None):
        self.config = config or {"port": 8080}
        self.backend = backend
        self.runtime = runtime
        self.fargate = fargate or {}
        self.container_id = None
        self.task_arn = None
        self.endpoint = None
        self.session = None

    async def start(self):
        """Start the container/task and record its MCP endpoint."""
        port = self.config.get("port", 8080)
        if self.backend == "local":
            self.container_id = manager.create_container(self.config, runtime=self.runtime)
            manager.start_container(self.container_id, runtime=self.runtime)
            self.endpoint = f"http://localhost:{port}/mcp"
        else:
            cluster = self.fargate.get("cluster")
            self.task_arn = manager.create_fargate_task(
                self.config,
                cluster,
                self.fargate.get("subnet_ids"),
                self.fargate.get("security_group_ids"),
            )
            manager.wait_for_fargate_task(self.task_arn, cluster)
            self.endpoint = manager.get_fargate_endpoint(self.task_arn, cluster, port) + "/mcp"
        # MCP session wiring over Streamable HTTP (streamablehttp_client) TBD

    async def stop(self):
        """Tear down the container/task."""
        if self.backend == "local":
            if self.container_id:
                manager.destroy_container(self.container_id, runtime=self.runtime)
        elif self.task_arn:
            manager.stop_fargate_task(self.task_arn, self.fargate.get("cluster"))

    async def call_tool(self, name, arguments):
        """Call a tool in the dev environment."""
        result = await self.session.call_tool(name, arguments)
        return result

    async def list_tools(self):
        """Get available tools."""
        tools = await self.session.list_tools()
        return tools

    async def read_resource(self, uri):
        """Read a resource."""
        resource = await self.session.read_resource(uri)
        return resource
```

### 5.5 Code Agent Orchestration

Integration with Tapestry's Actor/Regulator pattern.

**File:** `agent/code_agent.py`

```python
from tapestry.agent.dev_environment import DevEnvironment


class TapestryCodeAgent:
    """Orchestrates code tasks with Actor/Regulator oversight."""
    
    def __init__(self, actor, regulator, backend="local", config=None, fargate=None):
        self.actor = actor
        self.regulator = regulator
        self.backend = backend
        self.config = config or {"port": 8080}
        self.fargate = fargate or {}
        self.env = None
    
    async def execute_task(self, task: str, repo_url: str | None = None) -> str:
        """Execute a coding task with full oversight."""
        self.env = DevEnvironment(config=self.config, backend=self.backend, fargate=self.fargate)
        await self.env.start()
        
        try:
            # Clone repo if specified
            if repo_url:
                await self.env.call_tool("git_clone", {"repo_url": repo_url})
            
            # Get tools and context for Actor
            tools = await self.env.list_tools()
            workspace = await self.env.read_resource("workspace://tree")
            
            # Agent loop
            context = self.build_context(tools, workspace)
            messages = [{"role": "user", "content": task}]
            
            while True:
                response = await self.actor.complete(context, messages)
                tool_calls = self.extract_tool_calls(response)
                
                if not tool_calls:
                    return response.content
                
                results = []
                for call in tool_calls:
                    # Regulator gate - intercept before execution
                    decision = await self.regulator.evaluate(
                        call, 
                        context, 
                        workspace
                    )
                    
                    if decision.action == "BLOCK":
                        results.append({
                            "tool": call.name,
                            "blocked": True,
                            "reason": decision.reason
                        })
                        continue
                    
                    # Execute via MCP
                    result = await self.env.call_tool(call.name, call.arguments)
                    results.append({"tool": call.name, "result": result})
                    
                    # Log for proof construction
                    self.log_execution(call, result, decision)
                
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": self.format_results(results)})
        
        finally:
            await self.env.stop()
    
    def build_context(self, tools: list, workspace: str) -> str:
        """Build context string for Actor."""
        tool_docs = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        return f"""Available tools:
{tool_docs}

Workspace structure:
{workspace}
"""
    
    def extract_tool_calls(self, response) -> list:
        """Extract tool calls from Actor response."""
        # Implementation depends on response format
        pass
    
    def format_results(self, results: list[dict]) -> str:
        """Format tool results for next Actor turn."""
        formatted = []
        for r in results:
            if r.get("blocked"):
                formatted.append(f"Tool {r.get('tool')} BLOCKED: {r.get('reason')}")
            else:
                formatted.append(f"Tool {r.get('tool')} result: {r.get('result')}")
        return "\n".join(formatted)
    
    def log_execution(self, call, result, decision) -> None:
        """Log tool execution for proof construction."""
        # Store in format suitable for proof extraction
        pass
```

---

## 6. Tool Summary

| Tool | Purpose |
|------|---------|
| `file_read` | Read file contents (size-capped) |
| `file_write` | Create/overwrite files |
| `file_patch` | Apply targeted edits (find/replace) |
| `file_delete` | Delete a file |
| `file_list` | List directory contents |
| `file_search` | Search files with ripgrep |
| `pip_install` | Install Python packages |
| `pip_uninstall` | Uninstall Python packages |
| `pip_list` | List installed packages |
| `pip_freeze` | Installed packages in requirements format |
| `run_command` | Execute shell command |
| `run_python` | Execute Python code |
| `git_init` | Initialize a repository |
| `git_clone` | Clone repository |
| `git_status` | Get repo status |
| `git_diff` | View changes |
| `git_commit` | Stage and commit |
| `git_push` | Push to remote |
| `python_version` | Python version string |

Resources: `workspace://tree` (file tree), `workspace://git-log` (recent history).
HTTP route: `GET /health` (readiness probe; 200 healthy / 503 unhealthy).

---

## 7. Build and Deployment

### 7.1 Build Image

```bash
# Build the container image
cd container
podman build -t python-sandbox:latest -f Containerfile .

# Test locally
podman run -p 8080:8080 python-sandbox:latest
```

### 7.2 Push to ECR (for Fargate)

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  podman login --username AWS --password-stdin $ECR_REPO

# Tag and push
podman tag python-sandbox:latest $ECR_REPO/python-sandbox:latest
podman push $ECR_REPO/python-sandbox:latest
```

### 7.3 Usage

```python
# Prototype/MVP - Local Docker/Podman
from tapestry.agent.code_agent import TapestryCodeAgent

agent = TapestryCodeAgent(actor, regulator, backend="local", config={"port": 8080})
result = await agent.execute_task(
    "Fix the bug in auth.py",
    repo_url="https://github.com/user/project"
)

# GA - Fargate
agent = TapestryCodeAgent(
    actor,
    regulator,
    backend="fargate",
    config={"port": 8080},
    fargate={
        "cluster": "tapestry-cluster",
        "subnet_ids": ["subnet-xxx"],
        "security_group_ids": ["sg-xxx"],
    },
)
result = await agent.execute_task(...)
```

---

## 8. Integration with Tapestry Architecture

### 8.1 Regulator Interception

Every tool call passes through the Regulator before execution:

```
Actor proposes: file_write("config.py", "...")
    │
    ▼
Regulator evaluates:
  - Will hierarchy compliance?
  - Security concerns?
  - Scope appropriate?
    │
    ├── APPROVE → Execute via MCP
    ├── BLOCK → Return blocked message to Actor
    └── ASK_USER → Escalate for human decision
```

### 8.2 Proof Construction

Tool sequences are logged for proof extraction:

```python
ToolExecution {
    tool: "file_write"
    arguments: {"path": "src/auth.py", "content": "..."}
    result: "Wrote 1234 bytes to src/auth.py"
    regulator_decision: APPROVE
    timestamp: ...
    session_id: ...
}
```

Successful task completions (tests pass, PR merged) validate the tool sequence as a proof that can train the Actor.

### 8.3 Knowledge Graph Integration

Execution results feed the Knowledge Engine:

- Code facts extracted from workspace
- Test results as validation signals
- Git history as provenance

---

## 9. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Container escape | Docker isolation (Prototype/MVP), Fargate isolation (GA) |
| Network exfiltration | Fargate `awsvpc` task with `assignPublicIp: DISABLED`, a private subnet (no NAT/internet route), and a restrictive-egress security group |
| Filesystem access | All paths validated against `/workspace` |
| Resource exhaustion | CPU/memory limits set in the task definition |
| Malicious code execution | Regulator gates all tool calls |

`manager.py` sets `assignPublicIp: DISABLED`; the caller supplies the locked-down subnets and security groups. For a Fargate deployment the task definition and networking must also provide:
- Private subnets with no NAT route (or an egress-filtered path) so tools cannot reach the internet
- A security group allowing only the MCP port (8080) inbound from the Tapestry client, with minimal egress
- An execution role (ECR pull + CloudWatch Logs) and a least-privilege task role
- `awslogs` log configuration for task output; CloudTrail for the ECS/API control plane

---

## 10. Future Enhancements

- **Language server integration** — LSP for code intelligence (symbols, references, definitions)
- **Multi-file coordination** — Transactions across related files
- **Session persistence** — Pause/resume long-running development sessions
- **Custom runtime images** — Per-project container configurations
- **Parallel execution** — Multiple containers for large tasks

---

## References

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [FastMCP Library](https://github.com/jlowin/fastmcp)
- [Anthropic Claude Code Sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)
- [E2B Documentation](https://e2b.dev/docs)
- [OpenHands Architecture](https://github.com/OpenHands/OpenHands)
