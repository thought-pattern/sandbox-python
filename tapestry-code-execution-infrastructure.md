# Tapestry Workspace Container

**Status:** MVP implementation  
**Interface:** authenticated HTTP, API version 1.0  
**Runtime:** Docker in development

## Design

The Actor and all governance remain outside this image. The image contains no
agent and receives no application credentials. It provides a disposable Python
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
2. A fresh bearer token is generated for every sandbox session.
3. The HTTP server and tool subprocesses run as the `sandbox` user.
4. Deployment owns the EC2/Fargate boundary; managed web and deny modes
   install guest output rules, while explicit direct startup adds no guest firewall.
5. Request bytes, subprocess output, file reads, serialized responses,
   concurrency, timeout, CPU, memory, and process count are bounded.
6. Timeouts and output overruns terminate the complete process group.
7. File writes and patches replace their target atomically.
8. Cleanup failure is reported; it is not silently described as successful.
9. A container receives no Tapestry credential environment variables.

## Runtime lifecycle

An owned native configuration carries image, requested port, loopback bind,
resource bounds and networking mode. The session owns fresh identity and the
observed endpoint. `sandbox_session` creates and starts one container, yields its endpoint, then force-removes
it. If work and cleanup both fail, the original work failure is preserved and
annotated with the cleanup failure.

Docker applies fractional CPU limits, `--cap-drop ALL`,
`no-new-privileges`, a PID limit and the `SETGID`/`SETUID` capabilities needed to
drop identity. Only explicit local offline mode adds `NET_ADMIN`.
The entrypoint clears groups and changes permanently to
the non-root user.

## API contract

- `GET /health`
- `GET /tools`
- `POST /tools/<tool-name>`

All endpoints authenticate. The manifest is generated from the registered tool
functions and includes the protocol name, API version, egress policy, parameter
names, required/default state, and JSON types. External field names are
camelCase; request arguments are translated to internal Python names.

Every tool response carries a request ID. Server/API failures are structured.
Process tools return `ok`, `stdout`, `stderr`, `exitCode`, `timedOut`,
`outputLimited`, and `durationMs`. The normal Tapestry client raises on a failed
process; execution verification explicitly consumes the nonzero result so it
can distinguish refutation from infrastructure failure.

The authenticated health endpoint is a readiness probe rather than a process
liveness assertion. It verifies the workspace exists and is writable, reports
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
