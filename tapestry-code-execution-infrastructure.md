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
                                                │ stdio/SSE
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONTAINER                                 │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            MCP Server (Python + FastMCP)              │  │
│  │                                                        │  │
│  │  Tools:                                                │  │
│  │    file_read, file_write, file_patch, file_list       │  │
│  │    run_command, run_tests                             │  │
│  │    git_clone, git_status, git_commit, git_push        │  │
│  │    install_packages                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  /workspace/  ← mounted or cloned repo                      │
│  Runtimes: python, node, git, ripgrep                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Agent outside container | Regulator can gate every tool call; proof extraction possible |
| MCP protocol | Standardized, tools self-describe, portable |
| Stdio transport (local) | Simple, direct, no network overhead |
| SSE transport (Fargate) | Works over network, same protocol |
| Same image everywhere | Build once, deploy anywhere |

### 3.3 Comparison to Claude Code Web

| Aspect | Claude Code Web | Tapestry MCP Container |
|--------|-----------------|------------------------|
| Agent location | Inside container | Outside container |
| Tool interface | Direct bash/file access | MCP protocol over stdio |
| Agent logic | Claude Code CLI bundled | Tapestry Actor (separate) |
| Oversight | Permission prompts | Regulator intercepts tool calls |
| Learning | None | Proof extraction from tool sequences |

---

## 4. Environment Progression

| Stage | Container Runtime | Orchestration | MCP Transport |
|-------|-------------------|---------------|---------------|
| Prototype | Docker local | Python `docker` SDK | stdio |
| MVP | Docker local/remote | Docker SDK | stdio |
| GA | ECS/Fargate | Boto3 / AWS SDK | SSE |

The MCP server code stays identical across all environments—only the orchestration layer changes.

---

## 5. Implementation

### 5.1 Project Structure

```
tapestry/
├── containers/
│   ├── dev/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── server.py           # MCP server (runs IN container)
│   └── manager.py              # ContainerManager abstraction
├── agent/
│   ├── dev_environment.py      # MCP client wrapper
│   └── code_agent.py           # Actor/Regulator orchestration loop
└── config/
    └── environments.py         # Docker vs Fargate configuration
```

### 5.2 MCP Server

The MCP server runs inside the container and exposes development tools.

**File:** `containers/dev/server.py`

```python
import asyncio
import json
import os
import subprocess
from pathlib import Path
from fastmcp import FastMCP

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))

mcp = FastMCP("tapestry-dev")

# ═══════════════════════════════════════════════════════════
# File Operations
# ═══════════════════════════════════════════════════════════

@mcp.tool()
def file_read(path: str) -> str:
    """Read contents of a file."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    return target.read_text()


@mcp.tool()
def file_write(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


@mcp.tool()
def file_patch(path: str, patches: list[dict]) -> str:
    """Apply surgical edits to a file.
    
    Args:
        path: File path relative to workspace
        patches: List of {"old": "text to find", "new": "replacement"}
    """
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    
    content = target.read_text()
    for patch in patches:
        old, new = patch["old"], patch["new"]
        if old not in content:
            raise ValueError(f"Could not find: {old[:50]}...")
        content = content.replace(old, new, 1)
    
    target.write_text(content)
    return f"Applied {len(patches)} patches to {path}"


@mcp.tool()
def file_list(path: str = ".", depth: int = 2) -> list[str]:
    """List files in directory."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    
    result = subprocess.run(
        ["find", str(target), "-maxdepth", str(depth), "-type", "f"],
        capture_output=True, text=True, cwd=WORKSPACE
    )
    files = [str(Path(f).relative_to(WORKSPACE)) 
             for f in result.stdout.strip().split("\n") if f]
    return files


@mcp.tool()
def file_search(pattern: str, path: str = ".") -> list[dict]:
    """Search for pattern in files using ripgrep."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    
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
            matches.append({
                "file": str(Path(data["data"]["path"]["text"]).relative_to(WORKSPACE)),
                "line": data["data"]["line_number"],
                "content": data["data"]["lines"]["text"].strip()
            })
    return matches


# ═══════════════════════════════════════════════════════════
# Command Execution
# ═══════════════════════════════════════════════════════════

@mcp.tool()
def run_command(command: str, timeout: int = 60) -> dict:
    """Execute shell command in workspace."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "exit_code": -1
        }


@mcp.tool()
def run_tests(command: str = "pytest --tb=short -q") -> dict:
    """Run test suite."""
    result = run_command(command)
    return {
        "passed": result["exit_code"] == 0,
        "output": result["stdout"],
        "errors": result["stderr"],
        "exit_code": result["exit_code"]
    }


@mcp.tool()
def install_packages(packages: list[str], manager: str = "pip") -> dict:
    """Install packages."""
    if manager == "pip":
        cmd = f"pip install {' '.join(packages)}"
    elif manager == "npm":
        cmd = f"npm install {' '.join(packages)}"
    else:
        raise ValueError(f"Unknown package manager: {manager}")
    
    return run_command(cmd, timeout=300)


# ═══════════════════════════════════════════════════════════
# Git Operations
# ═══════════════════════════════════════════════════════════

@mcp.tool()
def git_clone(repo_url: str, branch: str = "main") -> str:
    """Clone repository into workspace."""
    run_command("rm -rf /workspace/*")
    result = run_command(f"git clone --branch {branch} {repo_url} .")
    if result["exit_code"] != 0:
        raise RuntimeError(result["stderr"])
    return f"Cloned {repo_url} (branch: {branch})"


@mcp.tool()
def git_status() -> dict:
    """Get current git status."""
    branch = run_command("git branch --show-current")
    status = run_command("git status --porcelain")
    return {
        "branch": branch["stdout"].strip(),
        "changes": [l for l in status["stdout"].strip().split("\n") if l]
    }


@mcp.tool()
def git_diff(staged: bool = False) -> str:
    """Get diff of changes."""
    cmd = "git diff --staged" if staged else "git diff"
    result = run_command(cmd)
    return result["stdout"]


@mcp.tool()
def git_commit(message: str) -> str:
    """Stage all changes and commit."""
    run_command("git add -A")
    result = run_command(f'git commit -m "{message}"')
    if result["exit_code"] != 0:
        raise RuntimeError(result["stderr"])
    return result["stdout"]


@mcp.tool()
def git_push(remote: str = "origin", branch: str | None = None) -> str:
    """Push commits to remote."""
    if branch is None:
        branch_result = run_command("git branch --show-current")
        branch = branch_result["stdout"].strip()
    
    result = run_command(f"git push {remote} {branch}")
    if result["exit_code"] != 0:
        raise RuntimeError(result["stderr"])
    return f"Pushed to {remote}/{branch}"


# ═══════════════════════════════════════════════════════════
# Resources (contextual info for the LLM)
# ═══════════════════════════════════════════════════════════

@mcp.resource("workspace://tree")
def workspace_tree() -> str:
    """Current workspace file tree."""
    result = run_command("find . -maxdepth 3 -type f | head -100")
    return result["stdout"]


@mcp.resource("workspace://git-log")  
def git_log() -> str:
    """Recent git history."""
    result = run_command("git log --oneline -20")
    return result["stdout"]


# ═══════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse", port=8080)
```

**File:** `containers/dev/requirements.txt`

```
fastmcp>=0.1.0
```

**File:** `containers/dev/Dockerfile`

```dockerfile
FROM python:3.11-slim

# System tools
RUN apt-get update && apt-get install -y \
    git \
    ripgrep \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Node.js (optional, for JS projects)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python MCP server
COPY requirements.txt /opt/mcp/
RUN pip install --no-cache-dir -r /opt/mcp/requirements.txt

COPY server.py /opt/mcp/

# Workspace
RUN mkdir /workspace
WORKDIR /workspace

ENV WORKSPACE=/workspace

# MCP server on stdio
ENTRYPOINT ["python", "/opt/mcp/server.py"]
```

### 5.3 Container Manager

Abstract container lifecycle management with implementations for Docker and Fargate.

**File:** `containers/manager.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ContainerConfig:
    image: str = "tapestry-dev:latest"
    workspace_mount: str | None = None
    environment: dict | None = None
    network_mode: str = "none"
    memory_limit: str = "2g"
    cpu_limit: float = 1.0


class ContainerManager(ABC):
    """Abstract container lifecycle management."""
    
    @abstractmethod
    async def create(self, config: ContainerConfig) -> str:
        """Create container, return container_id."""
        pass
    
    @abstractmethod
    async def start(self, container_id: str) -> None:
        """Start container."""
        pass
    
    @abstractmethod
    async def stop(self, container_id: str) -> None:
        """Stop container."""
        pass
    
    @abstractmethod
    async def destroy(self, container_id: str) -> None:
        """Remove container."""
        pass
    
    @abstractmethod
    async def exec(self, container_id: str, command: list[str]) -> tuple[int, str, str]:
        """Execute command, return (exit_code, stdout, stderr)."""
        pass
    
    @abstractmethod
    async def attach_stdio(self, container_id: str):
        """Get stdin/stdout streams for MCP connection."""
        pass


class DockerManager(ContainerManager):
    """Local Docker implementation."""
    
    def __init__(self):
        import docker
        self.client = docker.from_env()
    
    async def create(self, config: ContainerConfig) -> str:
        container = self.client.containers.create(
            config.image,
            stdin_open=True,
            tty=False,
            detach=True,
            network_mode=config.network_mode,
            mem_limit=config.memory_limit,
            nano_cpus=int(config.cpu_limit * 1e9),
            environment=config.environment,
        )
        return container.id
    
    async def start(self, container_id: str) -> None:
        container = self.client.containers.get(container_id)
        container.start()
    
    async def stop(self, container_id: str) -> None:
        container = self.client.containers.get(container_id)
        container.stop(timeout=10)
    
    async def destroy(self, container_id: str) -> None:
        container = self.client.containers.get(container_id)
        container.remove(force=True)
    
    async def exec(self, container_id: str, command: list[str]) -> tuple[int, str, str]:
        container = self.client.containers.get(container_id)
        result = container.exec_run(command, demux=True)
        stdout = result.output[0].decode() if result.output[0] else ""
        stderr = result.output[1].decode() if result.output[1] else ""
        return result.exit_code, stdout, stderr
    
    async def attach_stdio(self, container_id: str):
        container = self.client.containers.get(container_id)
        return container.attach_socket(params={"stdin": 1, "stdout": 1, "stream": 1})


class FargateManager(ContainerManager):
    """AWS ECS/Fargate implementation."""
    
    def __init__(self, cluster: str, subnet_ids: list[str], security_group_ids: list[str]):
        import boto3
        self.ecs = boto3.client("ecs")
        self.cluster = cluster
        self.subnet_ids = subnet_ids
        self.security_group_ids = security_group_ids
        self.task_definitions = {}
    
    async def create(self, config: ContainerConfig) -> str:
        task_def_arn = await self._ensure_task_definition(config)
        
        response = self.ecs.run_task(
            cluster=self.cluster,
            taskDefinition=task_def_arn,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": self.subnet_ids,
                    "securityGroups": self.security_group_ids,
                    "assignPublicIp": "DISABLED"
                }
            }
        )
        
        task_arn = response["tasks"][0]["taskArn"]
        return task_arn
    
    async def _ensure_task_definition(self, config: ContainerConfig) -> str:
        # Register or return cached task definition
        cache_key = f"{config.image}:{config.memory_limit}:{config.cpu_limit}"
        if cache_key in self.task_definitions:
            return self.task_definitions[cache_key]
        
        # Register new task definition
        response = self.ecs.register_task_definition(
            family="tapestry-dev",
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            cpu=str(int(config.cpu_limit * 1024)),
            memory=config.memory_limit.replace("g", "GB"),
            containerDefinitions=[{
                "name": "dev",
                "image": config.image,
                "essential": True,
                "environment": [
                    {"name": "MCP_TRANSPORT", "value": "sse"}
                ],
                "portMappings": [{"containerPort": 8080}]
            }]
        )
        
        arn = response["taskDefinition"]["taskDefinitionArn"]
        self.task_definitions[cache_key] = arn
        return arn
    
    async def start(self, container_id: str) -> None:
        waiter = self.ecs.get_waiter("tasks_running")
        waiter.wait(cluster=self.cluster, tasks=[container_id])
    
    async def stop(self, container_id: str) -> None:
        self.ecs.stop_task(cluster=self.cluster, task=container_id)
    
    async def destroy(self, container_id: str) -> None:
        await self.stop(container_id)
    
    async def exec(self, container_id: str, command: list[str]) -> tuple[int, str, str]:
        response = self.ecs.execute_command(
            cluster=self.cluster,
            task=container_id,
            container="dev",
            interactive=False,
            command=" ".join(command)
        )
        # Parse response
        return 0, "", ""
    
    async def attach_stdio(self, container_id: str):
        # For Fargate, return task IP for SSE connection
        response = self.ecs.describe_tasks(cluster=self.cluster, tasks=[container_id])
        eni_id = response["tasks"][0]["attachments"][0]["details"][1]["value"]
        
        ec2 = boto3.client("ec2")
        eni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
        return eni["NetworkInterfaces"][0]["PrivateIpAddress"]
```

### 5.4 Dev Environment Client

Wrapper connecting Tapestry to the containerized MCP server.

**File:** `agent/dev_environment.py`

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tapestry.containers.manager import ContainerManager, DockerManager, FargateManager, ContainerConfig


class DevEnvironment:
    """Manages a containerized dev environment via MCP."""
    
    def __init__(self, manager: ContainerManager, config: ContainerConfig | None = None):
        self.manager = manager
        self.config = config or ContainerConfig()
        self.container_id: str | None = None
        self.session: ClientSession | None = None
    
    async def start(self) -> None:
        """Start container and establish MCP connection."""
        self.container_id = await self.manager.create(self.config)
        await self.manager.start(self.container_id)
        
        if isinstance(self.manager, DockerManager):
            await self._connect_stdio()
        elif isinstance(self.manager, FargateManager):
            await self._connect_sse()
    
    async def _connect_stdio(self) -> None:
        """Connect via stdio for local Docker."""
        socket = await self.manager.attach_stdio(self.container_id)
        # Establish MCP session over socket
        # Implementation depends on mcp library specifics
        pass
    
    async def _connect_sse(self) -> None:
        """Connect via SSE for Fargate."""
        task_ip = await self.manager.attach_stdio(self.container_id)
        # Establish MCP session over HTTP/SSE
        # self.session = await connect_sse(f"http://{task_ip}:8080/mcp")
        pass
    
    async def stop(self) -> None:
        """Tear down container."""
        if self.session:
            # Close MCP session
            pass
        if self.container_id:
            await self.manager.destroy(self.container_id)
    
    async def call_tool(self, name: str, arguments: dict) -> any:
        """Call a tool in the dev environment."""
        result = await self.session.call_tool(name, arguments)
        return result
    
    async def list_tools(self) -> list:
        """Get available tools."""
        return await self.session.list_tools()
    
    async def read_resource(self, uri: str) -> str:
        """Read a resource."""
        return await self.session.read_resource(uri)
```

### 5.5 Code Agent Orchestration

Integration with Tapestry's Actor/Regulator pattern.

**File:** `agent/code_agent.py`

```python
from tapestry.agent.dev_environment import DevEnvironment
from tapestry.containers.manager import ContainerManager, ContainerConfig


class TapestryCodeAgent:
    """Orchestrates code tasks with Actor/Regulator oversight."""
    
    def __init__(self, actor, regulator, container_manager: ContainerManager):
        self.actor = actor
        self.regulator = regulator
        self.container_manager = container_manager
        self.env: DevEnvironment | None = None
    
    async def execute_task(self, task: str, repo_url: str | None = None) -> str:
        """Execute a coding task with full oversight."""
        self.env = DevEnvironment(self.container_manager)
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
                formatted.append(f"Tool {r['tool']} BLOCKED: {r['reason']}")
            else:
                formatted.append(f"Tool {r['tool']} result: {r['result']}")
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
| `file_read` | Read file contents |
| `file_write` | Create/overwrite files |
| `file_patch` | Apply targeted edits (find/replace) |
| `file_list` | List directory contents |
| `file_search` | Search files with ripgrep |
| `run_command` | Execute shell commands |
| `run_tests` | Run test suite |
| `install_packages` | Install pip/npm packages |
| `git_clone` | Clone repository |
| `git_status` | Get repo status |
| `git_diff` | View changes |
| `git_commit` | Stage and commit |
| `git_push` | Push to remote |

---

## 7. Build and Deployment

### 7.1 Build Image

```bash
# Build the container image
cd tapestry/containers/dev
docker build -t tapestry-dev:latest .

# Test locally
docker run -i tapestry-dev:latest
```

### 7.2 Push to ECR (for Fargate)

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_REPO

# Tag and push
docker tag tapestry-dev:latest $ECR_REPO/tapestry-dev:latest
docker push $ECR_REPO/tapestry-dev:latest
```

### 7.3 Usage

```python
# Prototype/MVP - Local Docker
from tapestry.containers.manager import DockerManager
from tapestry.agent.code_agent import TapestryCodeAgent

manager = DockerManager()
agent = TapestryCodeAgent(actor, regulator, manager)
result = await agent.execute_task(
    "Fix the bug in auth.py",
    repo_url="https://github.com/user/project"
)

# GA - Fargate
from tapestry.containers.manager import FargateManager

manager = FargateManager(
    cluster="tapestry-cluster",
    subnet_ids=["subnet-xxx"],
    security_group_ids=["sg-xxx"]
)
agent = TapestryCodeAgent(actor, regulator, manager)
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
| Network exfiltration | `network_mode: none` by default |
| Filesystem access | All paths validated against `/workspace` |
| Resource exhaustion | Memory/CPU limits on containers |
| Malicious code execution | Regulator gates all tool calls |

For GA deployment, additional considerations:
- VPC isolation for Fargate tasks
- IAM roles with minimal permissions
- CloudTrail logging for audit

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
