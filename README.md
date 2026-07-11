# sandbox-python

Python 3.12 sandbox container exposing development tools via MCP over Streamable HTTP.

## Project Structure

```
sandbox-python/
├── container/
│   ├── Containerfile     # Python 3.12-slim
│   ├── requirements.txt
│   └── server.py         # MCP server
├── manager.py            # Container lifecycle functions
└── README.md
```

## Building

```bash
cd container
podman build -t python-sandbox:latest -f Containerfile .
```

## Running

```bash
podman run -p 8080:8080 python-sandbox:latest
```

## MCP Tools

### Files

| Tool | Description |
|------|-------------|
| `file_read(path)` | Read file |
| `file_write(path, content)` | Write file |
| `file_patch(path, patches)` | Find/replace patches |
| `file_delete(path)` | Delete file |
| `file_list(path, depth)` | List directory |
| `file_search(pattern, path)` | Ripgrep search |

### Pip

| Tool | Description |
|------|-------------|
| `pip_install(packages)` | Install packages |
| `pip_uninstall(packages)` | Uninstall packages |
| `pip_list()` | List installed |
| `pip_freeze()` | Requirements format |

### Git

| Tool | Description |
|------|-------------|
| `git_init(branch)` | Initialize repo |
| `git_clone(repo_url, branch)` | Clone repo |
| `git_status()` | Status |
| `git_diff(staged)` | Diff |
| `git_commit(message)` | Commit all |
| `git_push(remote, branch)` | Push |

### Execution

| Tool | Description |
|------|-------------|
| `run_command(command, timeout)` | Shell command |
| `run_python(script, timeout)` | Python code |

### Environment

| Tool | Description |
|------|-------------|
| `python_version()` | Python version string |

### Resources

| Resource URI | Description |
|--------------|-------------|
| `workspace://tree` | Workspace file tree (up to 100 files) |
| `workspace://git-log` | Recent git history (last 20 commits) |

## Container Manager

```python
from manager import create_container, start_container, destroy_container

config = {"port": 8080}
container_id = create_container(config, runtime="podman")
start_container(container_id, runtime="podman")

# connect MCP client to http://localhost:8080/mcp

destroy_container(container_id, runtime="podman")
```

### Fargate

```python
from manager import create_fargate_task, wait_for_fargate_task, get_fargate_endpoint

config = {"port": 8080}
task_arn = create_fargate_task(config, cluster, subnet_ids, security_group_ids)
wait_for_fargate_task(task_arn, cluster)
endpoint = get_fargate_endpoint(task_arn, cluster, config.get("port"))
```

## Health Check

```bash
curl http://localhost:8080/health
```

Returns `200` when the workspace is writable and every required tool is present, `503` otherwise.

```json
{
  "status": "healthy",
  "python": "3.12.x ...",
  "checks": {
    "workspace_path": "/workspace",
    "workspace_writable": true,
    "workspace_free_mb": 12345,
    "tools": {"git": true, "rg": true, "find": true, "pip": true}
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_PORT` | 8080 | Server port |
| `WORKSPACE` | /workspace | Working directory |
