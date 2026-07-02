# sandbox-python

Python 3.12 sandbox container exposing development tools over a plain HTTP tool interface.

## Project Structure

```
sandbox-python/
├── container/
│   ├── Containerfile     # Python 3.12-slim, runs non-root
│   ├── requirements.txt  # empty; the server is standard-library only
│   └── server.py         # HTTP tool server
├── manager.py            # Container lifecycle (Apple container / Docker / podman, Fargate)
├── tests/                # Unit tests + a runtime-gated build/run smoke test
└── README.md
```

## Building

```bash
cd container
container build -t python-sandbox:latest -f Containerfile .
```

`docker` or `podman` work identically in place of `container`.

## Running

```bash
container run --rm -p 8080:8080 python-sandbox:latest
```

## HTTP Tool Interface

The server exposes three endpoints from the Python standard library. The same interface serves development and production; only the base URL changes.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness: `{"status": "healthy", "python": "..."}` |
| GET | `/tools` | Self-describing manifest of every tool |
| POST | `/tools/<name>` | Invoke a tool with a JSON object of arguments |

A successful call returns `{"result": ...}`; a failure returns a non-2xx status with `{"error": "..."}`. The manifest is derived from the tool functions by introspection, so it cannot drift from the code:

```json
{
  "tools": [
    {"name": "file_read", "description": "Read contents of a file.",
     "parameters": [{"name": "path", "required": true, "default": null}]}
  ]
}
```

### Tools

- **Files:** `file_read`, `file_write`, `file_patch`, `file_delete`, `file_list`, `file_search`
- **Pip:** `pip_install`, `pip_uninstall`, `pip_list`, `pip_freeze`
- **Git:** `git_clone`, `git_status`, `git_diff`, `git_commit`, `git_push`
- **Execution:** `run_command`, `run_python`
- **Environment:** `python_version`

## Container Manager

```python
from manager import ContainerConfig, sandbox_session

config = ContainerConfig(port=8080)
with sandbox_session(config, runtime="container") as container_id:
    # connect an HTTP client to http://localhost:8080
    ...
# the container is always destroyed on exit
```

`runtime` defaults to `container` (Apple's container runtime); `docker` and `podman` are interchangeable. The lower-level `create_container` / `start_container` / `stop_container` / `destroy_container` functions are available when you need to drive the lifecycle directly.

### Fargate

```python
from manager import ContainerConfig, create_fargate_task, wait_for_fargate_task, get_fargate_endpoint

config = ContainerConfig(port=8080)
task_arn = create_fargate_task(config, cluster, subnet_ids, security_group_ids)
wait_for_fargate_task(task_arn, cluster)
endpoint = get_fargate_endpoint(task_arn, cluster, config.port)
```

## Tests

```bash
python3 -m pytest tests/ -q
```

Unit tests run anywhere; the build-and-run smoke test runs only when a container runtime (`container`, `docker`, or `podman`) is present and skips otherwise.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_PORT` | 8080 | Server port |
| `WORKSPACE` | /workspace | Working directory |
