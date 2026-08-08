"""Authenticated, bounded HTTP tool server for the Python sandbox.

The server is intentionally headless.  A controller supplies a per-session
bearer token and reaches the three versioned endpoints over a host-loopback
published port.  Tool execution remains free inside the disposable container,
while request size, concurrency, wall time, process trees, and captured output
are bounded so one tool call cannot exhaust the service.
"""

import argparse
import hmac
import inspect
import json
import logging
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import get_args, get_origin
from uuid import uuid4

API_VERSION = "1.0"
PROTOCOL_NAME = "tapestry.workspace.http"
DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_PORT = 8080
DEFAULT_MAX_REQUEST_BYTES = 1_048_576
DEFAULT_MAX_OUTPUT_BYTES = 1_048_576
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
DEFAULT_MAX_TOOL_TIMEOUT = 300
DEFAULT_REQUEST_READ_TIMEOUT = 10.0
DEFAULT_MAX_LIST_RESULTS = 5_000
DEFAULT_MAX_SEARCH_RESULTS = 1_000

WORKSPACE = DEFAULT_WORKSPACE
EGRESS_POLICY = "deny"
MAX_OUTPUT_BYTES = DEFAULT_MAX_OUTPUT_BYTES
MAX_TOOL_TIMEOUT = DEFAULT_MAX_TOOL_TIMEOUT
BROKER_URL = ""
BROKER_TOKEN = ""
BROKER_PACKAGE_DESTINATION = "pypi"

TOOLS = {}
FILE_LOCK = threading.RLock()
LOGGER = logging.getLogger("tapestry.workspace.server")


class ToolError(RuntimeError):
    """An expected tool/API failure with a stable machine-readable code."""

    def __init__(self, message, *, code="tool_error", status=422):
        super().__init__(message)
        self.code = code
        self.status = status


def tool(fn):
    """Register a function as a callable tool surfaced in the manifest."""
    TOOLS[fn.__name__] = fn
    return fn


def resolve_path(path):
    """Resolve a path within the workspace, rejecting anything outside it."""
    target = (WORKSPACE / str(path)).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ToolError("Path must be within workspace", code="path_escape", status=400)
    return target


def _atomic_write_text(target, content):
    """Replace a text file atomically after fully writing its new content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tapestry-write-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _bounded_timeout(timeout):
    try:
        value = float(timeout)
    except (TypeError, ValueError) as err:
        raise ToolError("timeout must be numeric", code="invalid_timeout", status=400) from err
    if value <= 0:
        raise ToolError("timeout must be greater than zero", code="invalid_timeout", status=400)
    return min(value, float(MAX_TOOL_TIMEOUT))


def _terminate_process_group(process):
    """Terminate a tool process and every descendant in its process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        if process.poll() is None:
            process.wait(timeout=0.25)
        else:
            time.sleep(0.05)
    except (ProcessLookupError, subprocess.TimeoutExpired, ChildProcessError):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass


def _run_process(command, *, shell=False, timeout=60, cwd=None):
    """Run a bounded process and return a structured camelCase result."""
    timeout = _bounded_timeout(timeout)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd or WORKSPACE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    captured = 0
    timed_out = False
    output_limited = False
    deadline = started + timeout

    try:
        while selector.get_map():
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                timed_out = True
                _terminate_process_group(process)
                break
            events = selector.select(timeout=min(0.1, remaining_time))
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                available = max(0, MAX_OUTPUT_BYTES - captured)
                buffers[key.data].extend(chunk[:available])
                captured += min(len(chunk), available)
                if len(chunk) > available:
                    output_limited = True
                    _terminate_process_group(process)
                    break
            if output_limited:
                break
    finally:
        selector.close()

    if process.poll() is None:
        _terminate_process_group(process)
    return_code = process.wait()
    if timed_out:
        return_code = -1
    elif output_limited:
        return_code = -2

    return {
        "ok": return_code == 0,
        "stdout": buffers["stdout"].decode("utf-8", errors="replace"),
        "stderr": buffers["stderr"].decode("utf-8", errors="replace"),
        "exitCode": return_code,
        "timedOut": timed_out,
        "outputLimited": output_limited,
        "durationMs": round((time.monotonic() - started) * 1000, 3),
    }


def _require_process_success(result, operation):
    if result["ok"]:
        return result
    if result["timedOut"]:
        code = "timeout"
    elif result["outputLimited"]:
        code = "output_limit"
    else:
        code = "process_failed"
    detail = (result.get("stderr") or result.get("stdout") or "").strip()
    raise ToolError(
        f"{operation} failed with exit code {result['exitCode']}" + (f": {detail}" if detail else ""),
        code=code,
    )


def _require_network(operation):
    if EGRESS_POLICY == "deny":
        raise ToolError(
            f"{operation} requires controlled egress, but this sandbox is deny-by-default",
            code="egress_denied",
            status=403,
        )


def _redact_broker_token(result):
    """Remove the session broker token from process output before returning it."""
    if not BROKER_TOKEN:
        return result
    for name in ("stdout", "stderr"):
        value = result.get(name)
        if isinstance(value, str):
            result[name] = value.replace(BROKER_TOKEN, "[broker-token-redacted]")
    return result


def _broker_request(path, payload):
    """Call one authenticated broker control endpoint with bounded JSON."""
    if EGRESS_POLICY != "broker" or not BROKER_URL or not BROKER_TOKEN:
        raise ToolError("controlled egress broker is not configured", code="broker_unavailable", status=503)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{BROKER_URL.rstrip('/')}{path}",
        data=body,
        headers={
            "Authorization": f"Bearer {BROKER_TOKEN}",
            "Content-Type": "application/json",
            "X-Tapestry-Request-ID": str(uuid4()),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(float(MAX_TOOL_TIMEOUT), 30.0)) as response:
            result = json.loads(response.read(MAX_OUTPUT_BYTES + 1))
    except urllib.error.HTTPError as err:
        try:
            detail = json.loads(err.read(MAX_OUTPUT_BYTES + 1)).get("error", {})
        except Exception:
            detail = {}
        raise ToolError(
            str(detail.get("message") or "egress broker rejected the request"),
            code=str(detail.get("code") or "broker_rejected"),
            status=err.code,
        ) from err
    except Exception as err:
        raise ToolError("egress broker is unavailable", code="broker_unavailable", status=503) from err
    if not isinstance(result, dict):
        raise ToolError("egress broker returned an invalid response", code="broker_protocol", status=502)
    return result


def _broker_proxy_url(proxy_path):
    """Add the ephemeral broker token as URL basic-auth userinfo for pip/Git."""
    if not isinstance(proxy_path, str) or not proxy_path.startswith("/v1/proxy/"):
        raise ToolError("egress broker returned an invalid proxy path", code="broker_protocol", status=502)
    parsed = urllib.parse.urlsplit(BROKER_URL)
    token = urllib.parse.quote(BROKER_TOKEN, safe="")
    host = f"{parsed.hostname}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, f"{token}:x@{host}", proxy_path, "", ""))


def _resolve_broker_url(external_url, operation):
    result = _broker_request("/v1/resolve", {"url": external_url, "operation": operation})
    return _broker_proxy_url(result.get("proxyPath"))


@tool
def file_read(path: str):
    """Read UTF-8 contents of a file."""
    with FILE_LOCK:
        return resolve_path(path).read_text(encoding="utf-8")


@tool
def file_write(path: str, content: str):
    """Atomically write UTF-8 content to a file."""
    if not isinstance(content, str):
        raise ToolError("content must be a string", code="bad_arguments", status=400)
    target = resolve_path(path)
    with FILE_LOCK:
        _atomic_write_text(target, content)
    return {"path": str(target.relative_to(WORKSPACE)), "bytesWritten": len(content.encode("utf-8"))}


@tool
def file_patch(path: str, patches: list):
    """Atomically apply ordered find/replace patches to a file."""
    target = resolve_path(path)
    if not isinstance(patches, list):
        raise ToolError("patches must be a list", code="bad_arguments", status=400)
    with FILE_LOCK:
        content = target.read_text(encoding="utf-8")
        for patch in patches:
            if not isinstance(patch, dict) or "old" not in patch or "new" not in patch:
                raise ToolError("each patch requires old and new strings", code="bad_arguments", status=400)
            old = patch["old"]
            new = patch["new"]
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                raise ToolError("patch old/new values must be strings and old must not be empty", code="bad_arguments", status=400)
            if old not in content:
                raise ToolError(f"Not found: {old[:50]}...", code="patch_target_missing")
            content = content.replace(old, new, 1)
        _atomic_write_text(target, content)
    return {"path": str(target.relative_to(WORKSPACE)), "patchesApplied": len(patches)}


@tool
def file_delete(path: str):
    """Delete one file from the workspace."""
    target = resolve_path(path)
    with FILE_LOCK:
        target.unlink(missing_ok=False)
    return {"path": str(target.relative_to(WORKSPACE)), "deleted": True}


@tool
def file_list(path: str = ".", depth: int = 2, max_results: int = DEFAULT_MAX_LIST_RESULTS):
    """List a bounded number of files beneath a workspace directory."""
    target = resolve_path(path)
    depth = max(0, min(int(depth), 64))
    max_results = max(1, min(int(max_results), DEFAULT_MAX_LIST_RESULTS))
    files = []
    truncated = False
    for candidate in target.rglob("*"):
        try:
            relative_to_target = candidate.relative_to(target)
        except ValueError:
            continue
        if len(relative_to_target.parts) > depth or not candidate.is_file():
            continue
        files.append(str(candidate.relative_to(WORKSPACE)))
        if len(files) >= max_results:
            truncated = True
            break
    return {"files": sorted(files), "truncated": truncated}


@tool
def file_search(pattern: str, path: str = ".", max_results: int = DEFAULT_MAX_SEARCH_RESULTS):
    """Search files with ripgrep and return bounded structured matches."""
    target = resolve_path(path)
    max_results = max(1, min(int(max_results), DEFAULT_MAX_SEARCH_RESULTS))
    result = _run_process(["rg", "--json", pattern, str(target)], timeout=30)
    if result["exitCode"] not in (0, 1, -2):
        _require_process_success(result, "file search")
    matches = []
    for line in result["stdout"].splitlines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            if result["outputLimited"]:
                break
            raise
        if data.get("type") != "match":
            continue
        match = data.get("data", {})
        matches.append(
            {
                "file": str(Path(match["path"]["text"]).relative_to(WORKSPACE)),
                "line": match["line_number"],
                "content": match["lines"]["text"].strip(),
            }
        )
        if len(matches) >= max_results:
            break
    return {
        "matches": matches,
        "truncated": len(matches) >= max_results or result["outputLimited"],
    }


PACKAGE_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+([=<>!~\[\]][a-zA-Z0-9._,<>=!~\[\]]*)?$")


@tool
def pip_install(packages: list):
    """Install packages when a controlled egress path is enabled."""
    _require_network("pip_install")
    if not isinstance(packages, list) or not packages:
        raise ToolError("packages must be a non-empty list", code="bad_arguments", status=400)
    for package in packages:
        if not isinstance(package, str) or not PACKAGE_PATTERN.fullmatch(package):
            raise ToolError(f"Invalid package: {package}", code="bad_arguments", status=400)
    command = ["pip", "install", "--no-cache-dir"]
    if EGRESS_POLICY == "broker":
        index_path = f"/v1/proxy/{urllib.parse.quote(BROKER_PACKAGE_DESTINATION, safe='')}/simple/"
        index_url = _broker_proxy_url(index_path)
        broker_host = urllib.parse.urlsplit(BROKER_URL).hostname
        command.extend(["--index-url", index_url, "--trusted-host", broker_host, "--disable-pip-version-check"])
    result = _redact_broker_token(_run_process([*command, *packages], timeout=MAX_TOOL_TIMEOUT))
    return _require_process_success(
        result,
        "pip install",
    )


@tool
def pip_uninstall(packages: list):
    """Uninstall packages from the sandbox virtual environment."""
    if not isinstance(packages, list) or not packages:
        raise ToolError("packages must be a non-empty list", code="bad_arguments", status=400)
    if any(not isinstance(package, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", package) for package in packages):
        raise ToolError("Invalid package name", code="bad_arguments", status=400)
    return _require_process_success(
        _run_process(["pip", "uninstall", "-y", *packages], timeout=60),
        "pip uninstall",
    )


@tool
def pip_list():
    """List installed packages."""
    result = _require_process_success(
        _run_process(["pip", "list", "--format=json"], timeout=30),
        "pip list",
    )
    return json.loads(result["stdout"])


@tool
def pip_freeze():
    """Return installed packages in requirements-file form."""
    result = _require_process_success(_run_process(["pip", "freeze"], timeout=30), "pip freeze")
    return result["stdout"]


@tool
def run_command(command: str, timeout: float = 60):
    """Execute a bounded shell command in the workspace."""
    if not isinstance(command, str) or not command.strip():
        raise ToolError("command must be a non-empty string", code="bad_arguments", status=400)
    return _run_process(command, shell=True, timeout=timeout)


@tool
def run_python(script: str, timeout: float = 60):
    """Execute bounded Python code in the workspace."""
    if not isinstance(script, str) or not script.strip():
        raise ToolError("script must be a non-empty string", code="bad_arguments", status=400)
    return _run_process([sys.executable, "-c", script], timeout=timeout)


def _validate_git_atom(value, label):
    if not isinstance(value, str) or not value or value.startswith("-") or "\x00" in value:
        raise ToolError(f"Invalid Git {label}", code="bad_arguments", status=400)


@tool
def git_clone(repo_url: str, branch: str = "main"):
    """Stage a clone before replacing the workspace when egress is enabled."""
    _require_network("git_clone")
    _validate_git_atom(repo_url, "repository URL")
    _validate_git_atom(branch, "branch")
    clone_url = _resolve_broker_url(repo_url, "git") if EGRESS_POLICY == "broker" else repo_url
    staging = WORKSPACE / f".tapestry-clone-{uuid4().hex}"
    with FILE_LOCK:
        try:
            result = _run_process(
                ["git", "clone", "--branch", branch, "--", clone_url, str(staging)],
                timeout=MAX_TOOL_TIMEOUT,
            )
            result = _redact_broker_token(result)
            _require_process_success(result, "git clone")
            _require_process_success(
                _run_process(["git", "-C", str(staging), "remote", "set-url", "origin", repo_url], timeout=30),
                "git remote sanitization",
            )
            for item in list(WORKSPACE.iterdir()):
                if item == staging:
                    continue
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            for item in list(staging.iterdir()):
                shutil.move(str(item), WORKSPACE / item.name)
            staging.rmdir()
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {"repository": repo_url, "branch": branch, "cloned": True}


@tool
def git_status():
    """Return the current repository branch and changes."""
    branch = _require_process_success(
        _run_process(["git", "branch", "--show-current"], timeout=30),
        "git branch",
    )
    status = _require_process_success(
        _run_process(["git", "status", "--porcelain"], timeout=30),
        "git status",
    )
    return {
        "branch": branch["stdout"].strip(),
        "changes": [line for line in status["stdout"].splitlines() if line],
    }


@tool
def git_diff(staged: bool = False):
    """Return a bounded repository diff."""
    command = ["git", "diff", "--staged"] if staged else ["git", "diff"]
    result = _run_process(command, timeout=30)
    if not result["ok"] and not result["outputLimited"]:
        _require_process_success(result, "git diff")
    return {
        "diff": result["stdout"],
        "truncated": result["outputLimited"],
    }


@tool
def git_commit(message: str):
    """Stage all workspace changes and commit them locally."""
    if not isinstance(message, str) or not message.strip():
        raise ToolError("message must be a non-empty string", code="bad_arguments", status=400)
    _require_process_success(_run_process(["git", "add", "-A"], timeout=30), "git add")
    result = _require_process_success(
        _run_process(["git", "commit", "-m", message], timeout=60),
        "git commit",
    )
    return result["stdout"]


@tool
def git_push(remote: str = "origin", branch: str = None):
    """Push commits when a controlled egress path is enabled."""
    _require_network("git_push")
    _validate_git_atom(remote, "remote")
    if branch is None:
        current = _require_process_success(
            _run_process(["git", "branch", "--show-current"], timeout=30),
            "git branch",
        )
        branch = current["stdout"].strip()
    _validate_git_atom(branch, "branch")
    push_target = remote
    if EGRESS_POLICY == "broker":
        remote_url = _require_process_success(
            _run_process(["git", "remote", "get-url", remote], timeout=30),
            "git remote lookup",
        )["stdout"].strip()
        push_target = _resolve_broker_url(remote_url, "git")
    result = _redact_broker_token(_run_process(["git", "push", "--", push_target, branch], timeout=MAX_TOOL_TIMEOUT))
    _require_process_success(result, "git push")
    return {"remote": remote, "branch": branch, "pushed": True}


@tool
def python_version():
    """Get the Python version string."""
    return sys.version


def _json_type(annotation):
    origin = get_origin(annotation)
    if origin is list or annotation is list:
        arguments = get_args(annotation)
        schema = {"type": "array"}
        if arguments:
            schema["items"] = {"type": _json_type(arguments[0])}
        return schema
    if origin is dict or annotation is dict:
        return {"type": "object"}
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }.get(annotation, "any")


def _camel_case(name):
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


def _snake_case(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def build_manifest():
    """Describe every tool with a versioned JSON-schema-shaped signature."""
    tools = []
    for name, fn in TOOLS.items():
        parameters = []
        for param_name, param in inspect.signature(fn).parameters.items():
            required = param.default is inspect.Parameter.empty
            type_schema = _json_type(param.annotation)
            if isinstance(type_schema, str):
                type_schema = {"type": type_schema}
            parameters.append(
                {
                    "name": _camel_case(param_name),
                    "required": required,
                    "default": None if required else param.default,
                    **type_schema,
                }
            )
        tools.append(
            {
                "name": name,
                "description": (fn.__doc__ or "").strip(),
                "parameters": parameters,
            }
        )
    return {
        "apiVersion": API_VERSION,
        "protocol": PROTOCOL_NAME,
        "egressPolicy": EGRESS_POLICY,
        "tools": tools,
    }


class ToolHTTPServer(ThreadingHTTPServer):
    """Threaded server carrying immutable per-session boundary settings."""

    daemon_threads = True

    def server_bind(self):
        """Bind without HTTPServer's blocking reverse-DNS lookup."""
        TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        auth_token,
        max_request_bytes=DEFAULT_MAX_REQUEST_BYTES,
        max_concurrent_requests=DEFAULT_MAX_CONCURRENT_REQUESTS,
        request_read_timeout=DEFAULT_REQUEST_READ_TIMEOUT,
    ):
        super().__init__(server_address, handler_class)
        self.auth_token = auth_token
        self.max_request_bytes = max_request_bytes
        self.request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self.request_read_timeout = request_read_timeout


class ToolHandler(BaseHTTPRequestHandler):
    """Authenticated request handler for health, manifest, and tool calls."""

    server_version = "TapestryWorkspace/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.request_read_timeout)

    def send_json(self, status, payload, *, request_id=None):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if request_id:
            self.send_header("X-Tapestry-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, code, message, *, request_id=None):
        self.send_json(
            status,
            {"error": {"code": code, "message": str(message)}, "requestId": request_id},
            request_id=request_id,
        )

    def authorized(self):
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.auth_token}"
        return hmac.compare_digest(header, expected)

    def require_authorized(self, request_id):
        if self.authorized():
            return True
        self.send_error_json(401, "unauthorized", "A valid sandbox session token is required", request_id=request_id)
        return False

    def request_id(self):
        supplied = self.headers.get("X-Tapestry-Request-ID", "").strip()
        if supplied and len(supplied) <= 128 and re.fullmatch(r"[A-Za-z0-9._:-]+", supplied):
            return supplied
        return uuid4().hex

    def do_GET(self):
        request_id = self.request_id()
        if not self.require_authorized(request_id):
            return
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "healthy",
                    "python": sys.version,
                    "apiVersion": API_VERSION,
                    "egressPolicy": EGRESS_POLICY,
                    "requestId": request_id,
                },
                request_id=request_id,
            )
        elif self.path == "/tools":
            payload = build_manifest()
            payload["requestId"] = request_id
            self.send_json(200, payload, request_id=request_id)
        else:
            self.send_error_json(404, "not_found", f"no such path: {self.path}", request_id=request_id)

    def read_arguments(self, request_id):
        if self.headers.get("Transfer-Encoding"):
            raise ToolError("Transfer-Encoding is not supported", code="unsupported_transfer_encoding", status=400)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as err:
            raise ToolError("Content-Length must be an integer", code="invalid_content_length", status=400) from err
        if length < 0:
            raise ToolError("Content-Length must not be negative", code="invalid_content_length", status=400)
        if length > self.server.max_request_bytes:
            raise ToolError("Request body exceeds the configured limit", code="request_too_large", status=413)
        if length and self.headers.get_content_type() != "application/json":
            raise ToolError("Content-Type must be application/json", code="unsupported_media_type", status=415)
        raw = self.rfile.read(length) if length else b""
        try:
            arguments = json.loads(raw) if raw else {}
        except (UnicodeDecodeError, ValueError) as err:
            raise ToolError(f"invalid JSON body: {err}", code="invalid_json", status=400) from err
        if not isinstance(arguments, dict):
            raise ToolError("request body must be a JSON object of arguments", code="bad_arguments", status=400)
        return {_snake_case(key): value for key, value in arguments.items()}

    def do_POST(self):
        request_id = self.request_id()
        if not self.require_authorized(request_id):
            return
        if not self.path.startswith("/tools/"):
            self.send_error_json(404, "not_found", f"no such path: {self.path}", request_id=request_id)
            return
        name = self.path[len("/tools/") :]
        fn = TOOLS.get(name)
        if fn is None:
            self.send_error_json(404, "unknown_tool", f"no such tool: {name}", request_id=request_id)
            return
        if not self.server.request_slots.acquire(blocking=False):
            self.send_error_json(429, "busy", "sandbox tool concurrency limit reached", request_id=request_id)
            return

        started = time.monotonic()
        status = 200
        try:
            arguments = self.read_arguments(request_id)
            result = fn(**arguments)
            self.send_json(200, {"result": result, "requestId": request_id}, request_id=request_id)
        except ToolError as err:
            status = err.status
            self.send_error_json(err.status, err.code, err, request_id=request_id)
        except TypeError as err:
            status = 400
            self.send_error_json(400, "bad_arguments", f"bad arguments for {name}: {err}", request_id=request_id)
        except Exception:
            status = 500
            LOGGER.exception("Unhandled tool error request_id=%s tool=%s", request_id, name)
            self.send_error_json(500, "internal_error", "tool execution failed", request_id=request_id)
        finally:
            self.server.request_slots.release()
            LOGGER.info(
                "tool_request request_id=%s tool=%s status=%s duration_ms=%.3f",
                request_id,
                name,
                status,
                (time.monotonic() - started) * 1000,
            )

    def log_message(self, format, *args):
        return


def parse_args(argv=None):
    """Parse explicit server settings supplied by the container manager."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--auth-token", required=True)
    parser.add_argument("--egress-policy", choices=("deny", "broker", "unrestricted"), default="deny")
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--max-tool-timeout", type=int, default=DEFAULT_MAX_TOOL_TIMEOUT)
    parser.add_argument("--request-read-timeout", type=float, default=DEFAULT_REQUEST_READ_TIMEOUT)
    parser.add_argument("--broker-url", default="")
    parser.add_argument("--broker-token", default="")
    parser.add_argument("--broker-package-destination", default="pypi")
    return parser.parse_args(argv)


def configure(args):
    """Apply validated process-wide settings before accepting requests."""
    global WORKSPACE, EGRESS_POLICY, MAX_OUTPUT_BYTES, MAX_TOOL_TIMEOUT, BROKER_URL, BROKER_TOKEN, BROKER_PACKAGE_DESTINATION
    if len(args.auth_token) < 32:
        raise ValueError("auth token must be at least 32 characters")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    for name in ("max_request_bytes", "max_output_bytes", "max_concurrent_requests", "max_tool_timeout"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.request_read_timeout <= 0:
        raise ValueError("request_read_timeout must be positive")
    if args.egress_policy == "broker":
        parsed_broker = urllib.parse.urlsplit(args.broker_url)
        if parsed_broker.scheme != "http" or not parsed_broker.hostname or parsed_broker.port is None:
            raise ValueError("broker policy requires an explicit HTTP broker URL and port")
        if len(args.broker_token) < 32:
            raise ValueError("broker policy requires a broker token of at least 32 characters")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", args.broker_package_destination):
            raise ValueError("broker package destination is invalid")
    elif args.broker_url or args.broker_token:
        raise ValueError("broker URL and token require egress_policy=broker")
    WORKSPACE = args.workspace.resolve()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    EGRESS_POLICY = args.egress_policy
    MAX_OUTPUT_BYTES = args.max_output_bytes
    MAX_TOOL_TIMEOUT = args.max_tool_timeout
    BROKER_URL = args.broker_url
    BROKER_TOKEN = args.broker_token
    BROKER_PACKAGE_DESTINATION = args.broker_package_destination


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    configure(args)
    server = ToolHTTPServer(
        ("0.0.0.0", args.port),
        ToolHandler,
        auth_token=args.auth_token,
        max_request_bytes=args.max_request_bytes,
        max_concurrent_requests=args.max_concurrent_requests,
        request_read_timeout=args.request_read_timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
