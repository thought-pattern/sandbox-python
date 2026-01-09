# sandbox-python

Python 3.12 sandbox container exposing development tools via MCP over SSE.

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

## Container Manager

```python
from manager import ContainerConfig, create_container, start_container, destroy_container

config = ContainerConfig(port=8080)
container_id = create_container(config, runtime="podman")
start_container(container_id, runtime="podman")

# connect MCP client to http://localhost:8080

destroy_container(container_id, runtime="podman")
```

### Fargate

```python
from manager import ContainerConfig, create_fargate_task, wait_for_fargate_task, get_fargate_endpoint

config = ContainerConfig(port=8080)
task_arn = create_fargate_task(config, cluster, subnet_ids, security_group_ids)
wait_for_fargate_task(task_arn, cluster)
endpoint = get_fargate_endpoint(task_arn, cluster, config.port)
```

## Health Check

```bash
curl http://localhost:8080/health
```

```json
{"status": "healthy", "python": "3.12.x ..."}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_PORT` | 8080 | Server port |
| `WORKSPACE` | /workspace | Working directory |
