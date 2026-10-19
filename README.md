# sandbox-python

`sandbox-python` provides an authenticated, disposable Python 3.12 execution
environment through a gRPC tool interface and a separate HTTP health listener. The subproject contains the container
image, tool server, lifecycle manager, and verification suite.

## Boundary

- The host port is bound to `127.0.0.1`: the instance Workspace's
  `sandbox.host_port`, or a dynamically allocated one for a disposable session.
- Every tool endpoint requires the instance's one Workspace key,
  `sandbox.auth_token` in Tapestry's `config.yml`, as a bearer key.
- Every tool call names its root Task in the native `root_task_id` field (a canonical UUIDv7); a call that names none is refused with
  `invalid_working_context`. The tool runs in that root Task's folder,
  `<workspace>/.tapestry-work/<root Task ID>`, created on first use, and the
  reply echoes `workspace_context: {"root_task_id": ...}`. A disposable session
  names a fresh root for each run.
- The entrypoint drops permanently to the non-root `sandbox` user before the
  gRPC server or any tool code starts.
- CPU, memory, tool concurrency, process trees, and process count
  are bounded, and a watchdog ends a tool process that stalls. The container engine (`docker` or
  `podman`, `sandbox.container_engine`) applies a capability drop and
  `no-new-privileges`.
- Tapestry-managed sessions default to `web`; the root entrypoint installs
  IPv4/IPv6 output rules that allow DNS and outbound TCP 80/443 to public
  addresses, so destinations may change IP addresses during a session. On
  Fargate, `web` opens TCP 80/443 in the security group and the DNS Firewall
  limits name resolution to the configured `egress_allowlist`.
  This standalone server defaults to `direct` when started without the manager.
  The explicit `deny` option is offline. There is no forwarding service or
  Workspace SQLite dependency.

The entrypoint consumes `--auth-token-file` and passes the opened descriptor as `--auth-token-fd` to the server. The manager mounts the configured host key read-only at `/run/secrets/workspace-key`. The root entrypoint verifies that `/run/secrets` is canonical and root-owned, restricts it to mode 0700 and passes only the opened descriptor across the privilege drop. This protects the key even when file sharing maps its owner onto the sandbox UID. The server reads and closes that descriptor before accepting tools. The physical directory-mount manifest excludes inaccessible paths without granting source access.

On Fargate the controller supplies `TAPESTRY_WORKSPACE_KEY` for bootstrap. The nonroot entrypoint removes that environment value, writes its temporary key file, opens a descriptor and removes the file and its directory before service exec. The server receives only the descriptor. Existing entrypoint checks pass in the rebuilt image, and the [native DEV boundary](../../logs/alt-plan-20261016/dev-workspace-key-adoption-20261017.yml) verifies that generated tool code cannot reopen the key while ordinary Task-folder work succeeds.

## Fargate deployment

`workspace-fargate.yml` provides the declarative production substrate. Supply
the engagement's dedicated Workspace VPC, private subnets and route tables,
an immutable service-only ECR image digest and its repository ARN. Account
placement supplies the NAT route and direct application reachability. Attach
the emitted controller policy to the application role. DNS Firewall must have
fail-open disabled. This stack's DNS allow/block policy belongs only to the
dedicated Workspace VPC.

Select the adapter explicitly:

```python
from tapestry.config import reload_config
from tapestry.workspace.execution_sandbox import ExecutionSandbox

workspace = ExecutionSandbox(
    reload_config().get("sandbox", {}),
    runtime="fargate",
    implementation_directory="",
    image="123456789012.dkr.ecr.us-east-1.amazonaws.com/workspace@sha256:" + "a" * 64,
    deployment={
        "account_id": "123456789012",
        "region": "us-east-1",
        "cluster": "arn:aws:ecs:us-east-1:123456789012:cluster/workspace",
        "task_definition": "arn:aws:ecs:us-east-1:123456789012:task-definition/workspace:1",
        "vpc_id": "vpc-...",
        "subnets": ["subnet-..."],
        "application_security_group": "sg-...",
        "endpoint_ids": ["vpce-ecr-api", "vpce-ecr-registry", "vpce-logs", "vpce-s3"],
        "dns_rule_group_id": "rslvr-frg-...",
        "startup_timeout": 90,
        "cleanup_timeout": 60,
    },
)
with workspace.session() as run_probe:
    result = run_probe("print(6 * 7)", 5)
```

The adapter verifies deployment resources before creating a session's security
group. It removes broad default egress, grants only resolved public destination
and declared bootstrap traffic, launches one private task with no guest AWS
role, and stops the task before deleting the group. Cleanup preserves primary
failures and waits for ENI release. Fargate starts the server as nonroot with
`--egress-enforcement deployment`; it does not request NET_ADMIN. Health and
manifest report the enforcement owner.

CloudFormation lint and offline native AWS SDK lifecycle validation pass.
Production acceptance requires deployment in an actual engagement account;
local container verification does not establish Fargate isolation.

## Structure

```text
sandbox-python/
├── container/
│   ├── Containerfile
│   ├── assets/             # health page, its renderer and copies of Tapestry's shared page assets
│   ├── entrypoint.py       # privilege drop, optional offline mode, server exec
│   ├── health.py           # health figures and page files
│   ├── requirements.txt    # image dependencies
│   └── server.py           # standard-library gRPC server
└── tests/                  # direct API, subprocess, and lifecycle checks
```

## Build and run

Start the instance Workspace from Tapestry's repository root; it reads the
`sandbox` block of `config.yml` and publishes at `sandbox.host_port` on loopback:

```bash
python3 scripts/workspace_start.py --config config.yml
python3 scripts/workspace_stop.py --config config.yml
```

A disposable execution session uses the same configured engine and key, with
a fresh loopback port and no Workspace mount:

```python
from tapestry.config import reload_config
from tapestry.workspace.execution_sandbox import ExecutionSandbox

with ExecutionSandbox(reload_config().get("sandbox", {}), egress_policy="deny").session() as run_probe:
    print(run_probe("print(6 * 7)", 5))
```

Build the image with the configured engine (`podman build` takes the same
arguments), from the repository root. The root `.dockerignore` admits only Workspace implementation/assets and the declared protobuf/codec library; application sources, data and configuration are excluded:

```bash
docker build -t python-sandbox:latest \
  -f container/sandbox-python/container/Containerfile .
```

## Service interfaces

`WorkspaceService` exposes `CheckHealth`, `Describe` and `InvokeTool` with protobuf envelopes carrying native MessagePack mappings. Tools use `sandbox.port` (8080), published at `sandbox.host_port`, and authenticate with `authorization: Bearer <Workspace key>` metadata. `Describe` takes `request_id`; `InvokeTool` takes exactly `name`, `arguments`, `request_id` and `root_task_id`. An observation contains `request_id`, `status_code`, `result` and `workspace_context`. Tool errors are `result.error` with `code` and `message`; gRPC transport/authentication failures remain separate. Accepted tool functions retain their slot and drain even when the requester disconnects or its deadline expires.

The HTTP health listener uses `sandbox.health_port` (8081), published independently at `sandbox.health_host_port`. DEV assigns 8104; a disposable session requests a fresh loopback port. It serves `/`, `/health.js`, `/health-common.js`, `/health.css`, `/api/health`, `/health/live` and `/health/ready` without a key. The page and API share the owner's safe snapshot and carry no tool arguments, results, file contents or key.

Shell and Python tools return an explicit process result so clients can
distinguish command failure from a transport or service failure.

## Tools

- Files: `file_read`, `file_write`, `file_patch`, `file_delete`, `file_list`,
  `file_search`, `workspace_tree`
- Repository-scoped reads: `file_read_scoped` (UTF-8 bytes beneath one
  resolved repository root, with provenance) and `file_search_scoped` (bounded
  Python-source leads for one symbol in that repository)
- Python: `run_python`, `python_version`
- Shell: `run_command`
- Packages: `pip_install`, `pip_uninstall`, `pip_list`, `pip_freeze`
- Git: `git_init`, `git_clone`, `git_status`, `git_diff`, `git_commit`,
  `git_push`, `git_log`

File reads, program output and RPC observations have no size limit of their
own; each message is bound only by protobuf's largest message, 2 GB less one
byte. A file that cannot fit in one answer fails with `file_too_large` before
its contents are allocated, and a result that cannot be one message is replaced
with a structured `response_too_large` error. Program output goes to anonymous
files on disk while the program runs. Git commands disable interactive credential
prompts so unavailable credentials fail within the normal tool contract rather
than waiting for the watchdog.

Remote package installation, clone, and push fail under explicit `deny` mode.
Under explicitly selected `direct` mode they use the actual index, repository URL or remote
without rewriting. Local operations work under every policy.

## Verification

Tests run the actual gRPC owner and its separate HTTP health listener and package/Git subprocesses against local
fixtures. They do not build images, require Docker or change installed project
packages. They need `git`, `pip` and `rg` (ripgrep) on `PATH`, the same tools
the image installs.

```bash
PYTHONPATH=.:container/sandbox-python/container:container/sandbox-python/tests python3 -m pytest container/sandbox-python/tests/test_smoke.py -q
```
