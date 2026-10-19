# Tapestry Workspace Container

**Status:** MVP implementation  
**Interface:** authenticated HTTP, API version 1.0  
**Runtime:** the configured container engine, Docker or Podman (`sandbox.container_engine`), or Fargate

## Design

Actor, Will, Regulator, Utility, Motivator, KE, Research, Engram and Channels run
as independent native services outside this image. The image contains only the
Workspace service and its declared dependencies, with no Tapestry project copy
or application credentials. It provides a disposable Python
workspace through a small, self-describing HTTP tool interface.

```text
Tapestry client
  -> loopback/private authenticated HTTP
  -> direct tool endpoint
  -> entrypoint drops to non-root sandbox user
  -> bounded tool server
  -> bounded process group inside /workspace
```

Tapestry-managed networking defaults to `web`; package and Git tools
connect directly to their actual destinations. The root entrypoint installs
IPv4/IPv6 output rules before dropping privileges: DNS to the configured
resolvers and TCP 80/443 to public addresses, with private, loopback,
link-local and multicast networks dropped. Rules match ports rather than
startup-resolved addresses, so CDN and failover address changes keep working.
The standalone
server defaults to `direct` when started without the manager; explicit `deny`
supports offline execution. Deployment owns the EC2/Fargate network boundary.
No intermediary rewrites or forwards requests, and Workspace has no SQLite
storage dependency.

## Security invariants

1. The host publish address is `127.0.0.1` in local development.
2. Every tool call carries the instance's one configured Workspace key, which the
   server reads once from a file and never from its command line.
3. The HTTP server and tool subprocesses run as the `sandbox` user.
4. Deployment owns the EC2/Fargate boundary; managed web and deny modes
   install guest output rules, while explicit direct startup adds no guest firewall.
5. Concurrency, CPU, memory, and process count are bounded. Requests,
   subprocess output, file reads and responses are bound only by protobuf's
   largest message; subprocess output goes to anonymous files on disk.
6. A watchdog ends a program once neither its CPU time nor its output has moved
   for `process_watchdog_seconds` (default 120); a program that keeps working runs
   to its end. The complete process group is ended once its leader exits or stalls.
7. File writes and patches replace their target atomically.
8. Cleanup failure is reported; it is not silently described as successful.
9. A container receives no Tapestry credential environment variables.

## Runtime lifecycle

An owned native configuration carries image, requested port, loopback bind,
resource bounds and networking mode. The instance Workspace is one long-lived
container started by `scripts/workspace_start.py` and removed by
`scripts/workspace_stop.py`; each Task works in its own folder inside it. A
disposable `SandboxSession` owns fresh identity, its own key file and the
observed endpoint: it creates and starts one container, yields its endpoint,
then force-removes it and deletes the key file. If work and cleanup both fail,
the original work failure is preserved and annotated with the cleanup failure.

The configured engine applies fractional CPU limits, `--cap-drop ALL`,
`no-new-privileges`, a PID limit and the `SETGID`/`SETUID` capabilities needed to
drop identity. The `web` and `deny` egress policies add `NET_ADMIN`, which only
the root entrypoint uses to install the firewall before privileges drop; the
`direct` policy leaves egress to the host and does not add it.
The entrypoint clears groups and changes permanently to
the non-root user.

## API contract

- `GET /tools`
- `POST /tools/<tool-name>`
- `GET /health/ready`, `GET /health/live`, `GET /api/health` and the health
  page at `GET /` with its assets

The manifest and tool calls authenticate; the health routes do not, and they
carry no tool arguments, results, file contents or token. The manifest is
generated from the registered tool functions and includes the protocol name,
the server's incarnation, egress policy, parameter names, required/default
state, and JSON types. External field names are
camelCase; request arguments are translated to internal Python names.

Every tool response carries a request ID. Server/API failures are structured.
Process tools return `ok`, `stdout`, `stderr`, `exitCode`, `stalled`, and
`durationMs`. The normal Tapestry client raises on a failed
process; execution verification explicitly consumes the nonzero result so it
can distinguish refutation from infrastructure failure.

The readiness endpoint (`/health/ready`) is a readiness probe rather than a
process liveness assertion; `/health/live` answers while the server runs. It verifies the workspace exists and is writable, reports
free workspace capacity, and checks that Git, pip, and ripgrep are available.
An unhealthy report returns 503 with an `unhealthy` error whose message names
each failed check.
Local Git commands execute with interactive credential acquisition disabled.
Dedicated `git_init`, `git_log`, and `workspace_tree` tools provide bounded,
structured startup context without requiring shell parsing.

## Verification

Tests consume the actual HTTP tool interface and run repository-controlled
subprocess fixtures without Docker. Package/Git tests use disposable local
destinations. The live developer-task harness consumes independent caller-provisioned
execution and acceptance interfaces; it does not provision containers or execute
model-authored code on the host.
