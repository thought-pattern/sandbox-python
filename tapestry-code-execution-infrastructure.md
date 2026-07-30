# Tapestry Workspace Container

**Status:** MVP implementation  
**Interface:** authenticated HTTP, API version 1.0  
**Runtimes:** Apple container, Docker, podman; Fargate contract present but not
production-validated

## Design

The Actor and all governance remain outside this image. The image contains no
agent and receives no application credentials. It provides a disposable Python
workspace through a small, self-describing HTTP tool interface.

```text
Tapestry client
  -> loopback/private authenticated HTTP
  -> container entrypoint establishes network policy
  -> entrypoint drops to non-root sandbox user
  -> bounded tool server
  -> bounded process group inside /workspace
```

The current MVP policy is fail-closed network isolation. Direct outbound flows
are blocked by the guest firewall; package installation and remote Git tools
return a structured `egress_denied` response. The future `broker` mode assumes
the production task network permits traffic only to the controlled, audited
broker. Selecting broker mode without that external enforcement is invalid.

## Security invariants

1. The host publish address is `127.0.0.1` in local development.
2. A fresh bearer token is generated for every sandbox session.
3. The HTTP server and tool subprocesses run as the `sandbox` user.
4. Direct egress is denied before the untrusted tool surface starts.
5. Request bytes, output bytes, concurrency, timeout, CPU, and memory are
   bounded; Docker/podman additionally bound process count.
6. Timeouts and output overruns terminate the complete process group.
7. File writes and patches replace their target atomically.
8. Cleanup failure is reported; it is not silently described as successful.
9. A container receives no Tapestry credential environment variables.

## Runtime lifecycle

`ContainerConfig` owns the image, internal port, dynamically selected host
port, loopback bind, per-session token, resource bounds, and egress policy.
`sandbox_session` creates and starts the container, yields it, then force-removes
it. If work and cleanup both fail, the original work failure is preserved and
annotated with the cleanup failure.

Apple's runtime requires whole-number CPU allocations. Docker and podman retain
fractional CPU limits and add `--cap-drop ALL`, `no-new-privileges`, a PID limit,
and only the temporary `NET_ADMIN` capability needed by the root entrypoint.
After firewall setup, the entrypoint clears groups and changes permanently to
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

## Remaining production dependency

The controlled egress broker is intentionally outside this container. It must
provide destination policy, scoped credentials, request and response bounds,
and an audit record. Fargate broker mode additionally requires private subnets
and security groups whose only egress destination is that broker. Until this is
deployed, Tapestry configuration accepts only `egress_policy: deny`.
