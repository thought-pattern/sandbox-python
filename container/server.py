"""Authenticated, bounded HTTP tool server for the Python sandbox.

The server is intentionally headless.  A controller supplies a per-session
bearer token and reaches the three versioned endpoints over a host-loopback
published port.  Tool execution remains free inside the disposable container,
while request size, concurrency, wall time, process trees, and captured output
are bounded so one tool call cannot exhaust the service.
"""

from argparse import ArgumentParser as argparse_ArgumentParser
from hmac import compare_digest as hmac_compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from inspect import Parameter as inspect_Parameter
from inspect import signature as inspect_signature
from json import dumps as json_dumps
from json import loads as json_loads
from logging import INFO as logging_INFO
from logging import basicConfig as logging_basicConfig
from logging import getLogger as logging_getLogger
from os import environ as os_environ
from os import fdopen as os_fdopen
from os import fsync as os_fsync
from os import getpid as os_getpid
from os import kill as os_kill
from os import killpg as os_killpg
from os import read as os_read
from pathlib import Path
from re import compile as re_compile
from re import fullmatch as re_fullmatch
from re import sub as re_sub
from selectors import EVENT_READ as selectors_EVENT_READ
from selectors import DefaultSelector as selectors_DefaultSelector
from shutil import disk_usage as shutil_disk_usage
from shutil import move as shutil_move
from shutil import rmtree as shutil_rmtree
from shutil import which as shutil_which
from signal import SIGKILL as signal_SIGKILL
from signal import SIGTERM as signal_SIGTERM
from socketserver import TCPServer
from subprocess import PIPE as subprocess_PIPE
from subprocess import Popen as subprocess_Popen
from subprocess import TimeoutExpired as subprocess_TimeoutExpired
from sys import argv as sys_argv
from sys import executable as sys_executable
from sys import version as sys_version
from tempfile import NamedTemporaryFile as tempfile_NamedTemporaryFile
from tempfile import mkstemp as tempfile_mkstemp
from threading import BoundedSemaphore as threading_BoundedSemaphore
from threading import RLock as threading_RLock
from time import monotonic as time_monotonic
from uuid import uuid4

DEFAULT_ARGUMENT_DICT = {}
INVOCATION_MARKER_VARIABLE = "TAPESTRY_TOOL_INVOCATION"

API_VERSION = "1.0"
PROTOCOL_NAME = "tapestry.workspace.http"
DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_PORT = 8080
DEFAULT_MAX_REQUEST_BYTES = 1_048_576
DEFAULT_MAX_OUTPUT_BYTES = 1_048_576
DEFAULT_MAX_FILE_BYTES = 10_485_760
DEFAULT_MAX_RESPONSE_BYTES = 12_582_912
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
DEFAULT_MAX_TOOL_TIMEOUT = 300
DEFAULT_REQUEST_READ_TIMEOUT = 10.0
DEFAULT_MAX_LIST_RESULTS = 5_000
DEFAULT_MAX_SEARCH_RESULTS = 1_000
DEFAULT_MAX_GIT_LOG_RESULTS = 100
MIN_MAX_RESPONSE_BYTES = 512

WORKSPACE = DEFAULT_WORKSPACE
EGRESS_POLICY = "direct"
MAX_OUTPUT_BYTES = DEFAULT_MAX_OUTPUT_BYTES
MAX_FILE_BYTES = DEFAULT_MAX_FILE_BYTES
MAX_TOOL_TIMEOUT = DEFAULT_MAX_TOOL_TIMEOUT
GIT_ENV = {
    **os_environ,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "never",
}
TOOLS = {}
FILE_LOCK = threading_RLock()
LOGGER = logging_getLogger("tapestry.workspace.server")


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


def atomic_write_text(target, content):
    """Replace a text file atomically after fully writing its new content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile_mkstemp(prefix=".tapestry-write-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os_fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os_fsync(handle.fileno())
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return False


def bounded_timeout(timeout):
    try:
        value = float(timeout)
    except (TypeError, ValueError) as err:
        raise ToolError("timeout must be numeric", code="invalid_timeout", status=400) from err
    if value <= 0:
        raise ToolError("timeout must be greater than zero", code="invalid_timeout", status=400)
    computed_return_value = min(value, float(MAX_TOOL_TIMEOUT))
    return computed_return_value


def invocation_process_ids(root_pid, marker):
    """Collect a tool invocation's live processes by parent links and its inherited marker.

    A descendant that calls setsid or is reparented after its parent exits leaves the original
    process group and parent chain, but still inherits the invocation marker in its environment.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return set()
    marker_entry = f"{INVOCATION_MARKER_VARIABLE}={marker}".encode("utf-8")
    parents = {}
    marked = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            parents[pid] = int(stat[stat.rindex(")") + 2 :].split()[1])
            if marker_entry in (entry / "environ").read_bytes().split(b"\0"):
                marked.add(pid)
        except (OSError, ValueError, IndexError):
            continue
    descendants = set()
    frontier = [root_pid]
    while frontier:
        current = frontier.pop()
        for pid, parent in parents.items():
            if parent == current and pid not in descendants:
                descendants.add(pid)
                frontier.append(pid)
    return (descendants | marked | {root_pid}) - {os_getpid()}


def signal_invocation(process, marker, signal_number):
    """Signal the invocation's process group and every collected invocation process."""
    try:
        os_killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Some kernels refuse a group signal once the reaped leader leaves only exited members.
        if process.poll() is None:
            raise
    for pid in invocation_process_ids(process.pid, marker):
        try:
            os_kill(pid, signal_number)
        except ProcessLookupError:
            continue


def terminate_process_group(process, marker):
    """Terminate a tool invocation and all descendants, always escalating to SIGKILL."""
    signal_invocation(process, marker, signal_SIGTERM)
    try:
        process.wait(timeout=0.25)
    except (ChildProcessError, subprocess_TimeoutExpired):
        LOGGER.warning("tool invocation %s did not fully stop after SIGTERM; sending SIGKILL", process.pid)
    # The leader may exit on SIGTERM while a descendant ignores it, so the kill never depends on the leader.
    signal_invocation(process, marker, signal_SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=0.25)
        except ChildProcessError:
            return False
        except subprocess_TimeoutExpired as err:
            raise ToolError(
                f"tool process group {process.pid} remained alive after SIGKILL",
                code="process_termination_failed",
                status=500,
            ) from err
    return False


def internal_run_process(command, *, shell=False, timeout: float = 60.0, cwd="", environment=DEFAULT_ARGUMENT_DICT):
    """Run a bounded process and return a structured camelCase result."""
    if environment is DEFAULT_ARGUMENT_DICT:
        environment = DEFAULT_ARGUMENT_DICT.copy()
    timeout = bounded_timeout(timeout)
    started = time_monotonic()
    marker = uuid4().hex
    process_environment = {**(environment or os_environ), INVOCATION_MARKER_VARIABLE: marker}
    process = subprocess_Popen(
        command,
        shell=shell,
        cwd=cwd or WORKSPACE,
        env=process_environment,
        stdout=subprocess_PIPE,
        stderr=subprocess_PIPE,
        start_new_session=True,
    )
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    if stdout_stream is None or stderr_stream is None:
        terminate_process_group(process, marker)
        raise ToolError(
            "tool process did not expose bounded output streams",
            code="process_stream_unavailable",
            status=500,
        )
    selector = selectors_DefaultSelector()
    selector.register(stdout_stream, selectors_EVENT_READ, "stdout")
    selector.register(stderr_stream, selectors_EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    captured = 0
    timed_out = False
    output_limited = False
    deadline = started + timeout

    try:
        while selector.get_map():
            remaining_time = deadline - time_monotonic()
            if remaining_time <= 0:
                timed_out = True
                terminate_process_group(process, marker)
                break
            leader_exited = process.poll() is not None
            # After the direct child exits, drain what is already buffered and stop: a background
            # descendant holding the pipe open must not turn a finished tool into a timeout.
            events = selector.select(timeout=0 if leader_exited else min(0.1, remaining_time))
            if leader_exited and not events:
                break
            for key, _ in events:
                file_descriptor = key.fileobj
                if not isinstance(file_descriptor, int):
                    file_descriptor = file_descriptor.fileno()
                chunk = os_read(file_descriptor, 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                available = max(0, MAX_OUTPUT_BYTES - captured)
                channel = str(key.data)
                buffer = buffers.get(channel, bytearray())
                buffer.extend(chunk[:available])
                captured += min(len(chunk), available)
                if len(chunk) > available:
                    output_limited = True
                    terminate_process_group(process, marker)
                    break
            if output_limited:
                break
    finally:
        selector.close()

    terminate_process_group(process, marker)
    return_code = process.wait()
    if timed_out:
        return_code = -1
    elif output_limited:
        return_code = -2

    computed_return_value = {
        "ok": return_code == 0,
        "stdout": buffers.get("stdout", b"").decode("utf-8", errors="replace"),
        "stderr": buffers.get("stderr", b"").decode("utf-8", errors="replace"),
        "exitCode": return_code,
        "timedOut": timed_out,
        "outputLimited": output_limited,
        "durationMs": round((time_monotonic() - started) * 1000, 3),
    }
    return computed_return_value


def run_git(arguments, timeout: float = 30.0):
    """Run Git without allowing an interactive credential prompt."""
    command = ["git", *arguments]
    result = internal_run_process(command, timeout=timeout, environment=GIT_ENV)
    return result


def require_process_success(result, operation):
    if result.get("ok", False):
        return result
    if result.get("timedOut", False):
        code = "timeout"
    elif result.get("outputLimited", False):
        code = "output_limit"
    else:
        code = "process_failed"
    detail = (result.get("stderr", False) or result.get("stdout", False) or "").strip()
    raise ToolError(
        f"{operation} failed with exit code {result.get('exitCode', False)}" + (f": {detail}" if detail else ""),
        code=code,
    )


def require_network(operation):
    if EGRESS_POLICY == "deny":
        raise ToolError(
            f"{operation} requires networking, but this sandbox explicitly denies egress",
            code="egress_denied",
            status=403,
        )
    return False


@tool
def file_read(path: str):
    """Read bounded UTF-8 contents of a file."""
    target = resolve_path(path)
    with FILE_LOCK:
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(
                f"File exceeds the configured {MAX_FILE_BYTES}-byte read limit",
                code="file_too_large",
                status=413,
            )
        content = target.read_text(encoding="utf-8")
    return content


@tool
def file_write(path: str, content: str):
    """Atomically write UTF-8 content to a file."""
    if not isinstance(content, str):
        raise ToolError("content must be a string", code="bad_arguments", status=400)
    target = resolve_path(path)
    with FILE_LOCK:
        atomic_write_text(target, content)
    computed_return_value = {
        "path": str(target.relative_to(WORKSPACE)),
        "bytesWritten": len(content.encode("utf-8")),
    }
    return computed_return_value


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
                raise ToolError(
                    "each patch requires old and new strings",
                    code="bad_arguments",
                    status=400,
                )
            old = patch.get("old", False)
            new = patch.get("new", False)
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                raise ToolError(
                    "patch old/new values must be strings and old must not be empty",
                    code="bad_arguments",
                    status=400,
                )
            if old not in content:
                raise ToolError(f"Not found: {old[:50]}...", code="patch_target_missing")
            content = content.replace(old, new, 1)
        atomic_write_text(target, content)
    computed_return_value = {
        "path": str(target.relative_to(WORKSPACE)),
        "patchesApplied": len(patches),
    }
    return computed_return_value


@tool
def file_delete(path: str):
    """Delete one file from the workspace."""
    target = resolve_path(path)
    with FILE_LOCK:
        target.unlink(missing_ok=False)
    computed_return_value = {"path": str(target.relative_to(WORKSPACE)), "deleted": True}
    return computed_return_value


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
    computed_return_value = {"files": sorted(files), "truncated": truncated}
    return computed_return_value


@tool
def file_search(pattern: str, path: str = ".", max_results: int = DEFAULT_MAX_SEARCH_RESULTS):
    """Search files with ripgrep and return bounded structured matches."""
    target = resolve_path(path)
    max_results = max(1, min(int(max_results), DEFAULT_MAX_SEARCH_RESULTS))
    result = internal_run_process(["rg", "--json", "-e", pattern, "--", str(target)], timeout=30)
    if result.get("exitCode", ()) not in (0, 1, -2):
        require_process_success(result, "file search")
    matches = []
    for line in result.get("stdout", "").splitlines():
        if not line:
            continue
        try:
            data = json_loads(line)
        except ValueError:
            if result.get("outputLimited", False):
                break
            raise
        if data.get("type", "") != "match":
            continue
        match = data.get("data", {})
        matches.append(
            {
                "file": str(Path(match.get("path", {}).get("text", "")).relative_to(WORKSPACE)),
                "line": match.get("line_number", False),
                "content": match.get("lines", {}).get("text", "").strip(),
            }
        )
        if len(matches) >= max_results:
            break
    computed_return_value = {
        "matches": matches,
        "truncated": len(matches) >= max_results or result.get("outputLimited", False),
    }
    return computed_return_value


PACKAGE_PATTERN = re_compile(r"^[a-zA-Z0-9_.-]+([=<>!~\[\]][a-zA-Z0-9._,<>=!~\[\]]*)?$")


@tool
def pip_install(packages: list):
    """Install packages directly when networking is enabled."""
    require_network("pip_install")
    if not isinstance(packages, list) or not packages:
        raise ToolError("packages must be a non-empty list", code="bad_arguments", status=400)
    for package in packages:
        if not isinstance(package, str) or not PACKAGE_PATTERN.fullmatch(package):
            raise ToolError(f"Invalid package: {package}", code="bad_arguments", status=400)
    command = ["pip", "install", "--no-cache-dir"]
    result = internal_run_process([*command, *packages], timeout=MAX_TOOL_TIMEOUT)
    computed_return_value = require_process_success(
        result,
        "pip install",
    )
    return computed_return_value


@tool
def pip_uninstall(packages: list):
    """Uninstall packages from the sandbox virtual environment."""
    if not isinstance(packages, list) or not packages:
        raise ToolError("packages must be a non-empty list", code="bad_arguments", status=400)
    if any(not isinstance(package, str) or not re_fullmatch(r"[a-zA-Z0-9_.-]+", package) for package in packages):
        raise ToolError("Invalid package name", code="bad_arguments", status=400)
    computed_return_value = require_process_success(
        internal_run_process(["pip", "uninstall", "-y", *packages], timeout=60),
        "pip uninstall",
    )
    return computed_return_value


@tool
def pip_list():
    """List installed packages."""
    result = require_process_success(
        internal_run_process(["pip", "list", "--format=json"], timeout=30),
        "pip list",
    )
    computed_return_value = json_loads(result.get("stdout", False))
    return computed_return_value


@tool
def pip_freeze():
    """Return installed packages in requirements-file form."""
    result = require_process_success(internal_run_process(["pip", "freeze"], timeout=30), "pip freeze")
    computed_return_value = result.get("stdout", "")
    return computed_return_value


@tool
def run_command(command: str, timeout: float = 60.0):
    """Execute a bounded shell command in the workspace."""
    if not isinstance(command, str) or not command.strip():
        raise ToolError("command must be a non-empty string", code="bad_arguments", status=400)
    computed_return_value = internal_run_process(command, shell=True, timeout=timeout)
    return computed_return_value


@tool
def run_python(script: str, timeout: float = 60.0):
    """Execute bounded Python code in the workspace."""
    if not isinstance(script, str) or not script.strip():
        raise ToolError("script must be a non-empty string", code="bad_arguments", status=400)
    computed_return_value = internal_run_process([sys_executable, "-c", script], timeout=timeout)
    return computed_return_value


def validate_git_atom(value, label):
    if not isinstance(value, str) or not value or value.startswith("-") or "\x00" in value:
        raise ToolError(f"Invalid Git {label}", code="bad_arguments", status=400)
    return False


@tool
def git_init(branch: str = "main"):
    """Initialize a local Git repository with a validated initial branch."""
    validate_git_atom(branch, "branch")
    validation = run_git(["check-ref-format", "--branch", branch])
    require_process_success(validation, "git branch validation")
    result = run_git(["init", f"--initial-branch={branch}"])
    require_process_success(result, "git init")
    output = {
        "branch": branch,
        "initialized": True,
        "message": result.get("stdout", "").strip(),
    }
    return output


@tool
def git_clone(repo_url: str, branch: str = "main"):
    """Stage a clone before replacing the workspace when egress is enabled."""
    require_network("git_clone")
    validate_git_atom(repo_url, "repository URL")
    validate_git_atom(branch, "branch")
    staging = WORKSPACE / f".tapestry-clone-{uuid4().hex}"
    with FILE_LOCK:
        try:
            result = run_git(
                ["clone", "--branch", branch, "--", repo_url, str(staging)],
                timeout=MAX_TOOL_TIMEOUT,
            )
            require_process_success(result, "git clone")
            for item in list(WORKSPACE.iterdir()):
                if item == staging:
                    continue
                shutil_rmtree(item) if item.is_dir() else item.unlink()
            for item in list(staging.iterdir()):
                shutil_move(str(item), WORKSPACE / item.name)
            staging.rmdir()
        except Exception:
            shutil_rmtree(staging, ignore_errors=True)
            raise
    return {"repository": repo_url, "branch": branch, "cloned": True}


@tool
def git_status():
    """Return the current repository branch and changes."""
    branch = require_process_success(
        run_git(["branch", "--show-current"], timeout=30),
        "git branch",
    )
    status = require_process_success(
        run_git(["status", "--porcelain"], timeout=30),
        "git status",
    )
    computed_return_value = {
        "branch": branch.get("stdout", "").strip(),
        "changes": [line for line in status.get("stdout", "").splitlines() if line],
    }
    return computed_return_value


@tool
def git_diff(staged: bool = False):
    """Return a bounded repository diff."""
    command = ["git", "diff", "--staged"] if staged else ["git", "diff"]
    result = run_git(command[1:], timeout=30)
    if not result.get("ok", False) and not result.get("outputLimited", False):
        require_process_success(result, "git diff")
    computed_return_value = {
        "diff": result.get("stdout", False),
        "truncated": result.get("outputLimited", False),
    }
    return computed_return_value


@tool
def git_commit(message: str):
    """Stage all workspace changes and commit them locally."""
    if not isinstance(message, str) or not message.strip():
        raise ToolError("message must be a non-empty string", code="bad_arguments", status=400)
    require_process_success(run_git(["add", "-A"], timeout=30), "git add")
    result = require_process_success(
        run_git(["commit", "-m", message], timeout=60),
        "git commit",
    )
    computed_return_value = result.get("stdout", "")
    return computed_return_value


@tool
def git_push(remote: str = "origin", branch: str = ""):
    """Push commits directly to the selected remote when networking is enabled."""
    require_network("git_push")
    validate_git_atom(remote, "remote")
    if not branch:
        current = require_process_success(
            run_git(["branch", "--show-current"], timeout=30),
            "git branch",
        )
        branch = current.get("stdout", "").strip()
    validate_git_atom(branch, "branch")
    result = run_git(["push", "--", remote, branch], timeout=MAX_TOOL_TIMEOUT)
    require_process_success(result, "git push")
    return {"remote": remote, "branch": branch, "pushed": True}


@tool
def git_log(limit: int = 20):
    """Return bounded recent Git history as structured commit records."""
    try:
        requested_limit = int(limit)
    except (TypeError, ValueError) as err:
        raise ToolError("limit must be an integer", code="bad_arguments", status=400) from err
    if requested_limit < 1:
        raise ToolError("limit must be positive", code="bad_arguments", status=400)
    bounded_limit = min(requested_limit, DEFAULT_MAX_GIT_LOG_RESULTS)
    result = run_git(
        [
            "log",
            f"--max-count={bounded_limit + 1}",
            "--format=%H%x00%h%x00%aI%x00%s",
        ],
        timeout=30,
    )
    require_process_success(result, "git log")
    lines = result.get("stdout", "").splitlines()
    truncated = len(lines) > bounded_limit
    commits = []
    for line in lines[:bounded_limit]:
        commit_hash, short_hash, authored_at, subject = line.split("\x00", 3)
        commits.append(
            {
                "hash": commit_hash,
                "shortHash": short_hash,
                "authoredAt": authored_at,
                "subject": subject,
            }
        )
    output = {"commits": commits, "truncated": truncated}
    return output


@tool
def workspace_tree(depth: int = 3, max_results: int = 100):
    """Return a bounded workspace file tree for initial task context."""
    result = file_list(".", depth=depth, max_results=max_results)
    return result


@tool
def python_version():
    """Get the Python version string."""
    return sys_version


def json_type(annotation):
    origin = getattr(annotation, "__origin__", False)
    if origin is list or annotation is list:
        arguments = getattr(annotation, "__args__", ())
        schema: dict = {"type": "array"}
        if arguments:
            schema["items"] = {"type": json_type(arguments[0])}
        return schema
    if origin is dict or annotation is dict:
        return {"type": "object"}
    computed_return_value = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }.get(annotation, "any")
    return computed_return_value


def camel_case(name):
    first, *rest = name.split("_")
    computed_return_value = first + "".join(part.capitalize() for part in rest)
    return computed_return_value


def snake_case(name):
    computed_return_value = re_sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return computed_return_value


def build_manifest():
    """Describe every tool with a versioned JSON-schema-shaped signature."""
    tools = []
    for name, fn in TOOLS.items():
        parameters = []
        for param_name, param in inspect_signature(fn).parameters.items():
            required = param.default is inspect_Parameter.empty
            type_schema = json_type(param.annotation)
            if isinstance(type_schema, str):
                type_schema = {"type": type_schema}
            descriptor = {
                "name": camel_case(param_name),
                "required": required,
                **type_schema,
            }
            if not required:
                descriptor["default"] = param.default
            parameters.append(descriptor)
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


def build_readiness_report():
    """Return dependency, workspace, and capacity readiness information."""
    required_tools = ("git", "pip", "rg")
    tools = {name: bool(shutil_which(name)) for name in required_tools}
    workspace_exists = WORKSPACE.is_dir()
    workspace_writable = False
    workspace_free_bytes = 0
    if workspace_exists:
        try:
            with tempfile_NamedTemporaryFile(dir=WORKSPACE):
                workspace_writable = True
            workspace_free_bytes = shutil_disk_usage(WORKSPACE).free
        except OSError:
            workspace_writable = False
    dependencies_ready = all(tools.values())
    ready = workspace_exists and workspace_writable and dependencies_ready
    status = "healthy" if ready else "unhealthy"
    result = {
        "status": status,
        "checks": {
            "workspaceExists": workspace_exists,
            "workspaceWritable": workspace_writable,
            "workspaceFreeBytes": workspace_free_bytes,
            "tools": tools,
        },
    }
    return result


class ToolHTTPServer(ThreadingHTTPServer):
    """Threaded server carrying immutable per-session boundary settings."""

    server_name: str
    server_port: int
    daemon_threads = True

    def server_bind(self):
        """Bind without HTTPServer's blocking reverse-DNS lookup."""
        TCPServer.server_bind(self)
        bound_address = self.server_address
        if not isinstance(bound_address, tuple) or len(bound_address) != 2:
            raise RuntimeError("Workspace HTTP server did not bind an internet address")
        bound_name = bound_address[0]
        bound_port = bound_address[1]
        if not isinstance(bound_name, str) or not isinstance(bound_port, int):
            raise RuntimeError("Workspace HTTP server returned an invalid bound address")
        self.server_name = bound_name
        self.server_port = bound_port
        return False

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        auth_token,
        max_request_bytes=DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes=DEFAULT_MAX_RESPONSE_BYTES,
        max_concurrent_requests=DEFAULT_MAX_CONCURRENT_REQUESTS,
        request_read_timeout=DEFAULT_REQUEST_READ_TIMEOUT,
    ):
        super().__init__(server_address, handler_class)
        self.auth_token = auth_token
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.request_slots = threading_BoundedSemaphore(max_concurrent_requests)
        self.request_read_timeout = request_read_timeout


class ToolHandler(BaseHTTPRequestHandler):
    """Authenticated request handler for health, manifest, and tool calls."""

    server: ToolHTTPServer
    server_version = "TapestryWorkspace/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.request_read_timeout)
        return False

    def send_json(self, status, payload, *, request_id=""):
        body = json_dumps(payload, separators=(",", ":")).encode("utf-8")
        response_status = status
        if len(body) > self.server.max_response_bytes:
            response_status = 413
            payload = {
                "error": {
                    "code": "response_too_large",
                    "message": "Tool response exceeds the configured byte limit",
                },
                "requestId": request_id,
            }
            body = json_dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if request_id:
            self.send_header("X-Tapestry-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)
        return response_status

    def send_error_json(self, status, code, message, *, request_id=""):
        response_status = self.send_json(
            status,
            {"error": {"code": code, "message": str(message)}, "requestId": request_id},
            request_id=request_id,
        )
        return response_status

    def authorized(self):
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.auth_token}"
        computed_return_value = hmac_compare_digest(header.encode("utf-8"), expected.encode("utf-8"))
        return computed_return_value

    def require_authorized(self, request_id):
        if self.authorized():
            return True
        self.send_error_json(
            401,
            "unauthorized",
            "A valid sandbox session token is required",
            request_id=request_id,
        )
        return False

    def request_id(self):
        supplied = self.headers.get("X-Tapestry-Request-ID", "").strip()
        if supplied and len(supplied) <= 128 and re_fullmatch(r"[A-Za-z0-9._:-]+", supplied):
            return supplied
        return uuid4().hex

    def do_GET(self):
        request_id = self.request_id()
        if not self.require_authorized(request_id):
            return False
        if self.path == "/health":
            payload = build_readiness_report()
            payload["python"] = sys_version
            payload["apiVersion"] = API_VERSION
            payload["egressPolicy"] = EGRESS_POLICY
            payload["requestId"] = request_id
            status = 200 if payload.get("status", "") == "healthy" else 503
            self.send_json(status, payload, request_id=request_id)
        elif self.path == "/tools":
            payload = build_manifest()
            payload["requestId"] = request_id
            self.send_json(200, payload, request_id=request_id)
        else:
            self.send_error_json(404, "not_found", f"no such path: {self.path}", request_id=request_id)
        return False

    def read_arguments(self, request_id):
        if self.headers.get("Transfer-Encoding", False):
            raise ToolError(
                "Transfer-Encoding is not supported",
                code="unsupported_transfer_encoding",
                status=400,
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as err:
            raise ToolError(
                "Content-Length must be an integer",
                code="invalid_content_length",
                status=400,
            ) from err
        if length < 0:
            raise ToolError(
                "Content-Length must not be negative",
                code="invalid_content_length",
                status=400,
            )
        if length > self.server.max_request_bytes:
            raise ToolError(
                "Request body exceeds the configured limit",
                code="request_too_large",
                status=413,
            )
        if length and self.headers.get_content_type() != "application/json":
            raise ToolError(
                "Content-Type must be application/json",
                code="unsupported_media_type",
                status=415,
            )
        raw = self.rfile.read(length) if length else b""
        try:
            arguments = json_loads(raw) if raw else {}
        except (UnicodeDecodeError, ValueError) as err:
            raise ToolError(f"invalid JSON body: {err}", code="invalid_json", status=400) from err
        if not isinstance(arguments, dict):
            raise ToolError(
                "request body must be a JSON object of arguments",
                code="bad_arguments",
                status=400,
            )
        computed_return_value = {snake_case(key): value for key, value in arguments.items()}
        return computed_return_value

    def do_POST(self):
        request_id = self.request_id()
        if not self.require_authorized(request_id):
            return False
        if not self.path.startswith("/tools/"):
            self.send_error_json(404, "not_found", f"no such path: {self.path}", request_id=request_id)
            return False
        name = self.path[len("/tools/") :]
        fn = TOOLS.get(name, False)
        if fn is False:
            self.send_error_json(404, "unknown_tool", f"no such tool: {name}", request_id=request_id)
            return False
        if not self.server.request_slots.acquire(blocking=False):
            self.send_error_json(
                429,
                "busy",
                "sandbox tool concurrency limit reached",
                request_id=request_id,
            )
            return False

        started = time_monotonic()
        status = 200
        try:
            arguments = self.read_arguments(request_id)
            result = fn(**arguments)
            status = self.send_json(200, {"result": result, "requestId": request_id}, request_id=request_id)
        except ToolError as err:
            status = err.status
            self.send_error_json(err.status, err.code, err, request_id=request_id)
        except FileNotFoundError:
            status = 404
            self.send_error_json(404, "not_found", "requested file or tool was not found", request_id=request_id)
        except IsADirectoryError:
            status = 400
            self.send_error_json(400, "not_file", "expected a file", request_id=request_id)
        except NotADirectoryError:
            status = 400
            self.send_error_json(400, "not_directory", "expected a directory", request_id=request_id)
        except PermissionError:
            status = 403
            self.send_error_json(403, "permission_denied", "tool access was denied", request_id=request_id)
        except OSError:
            status = 422
            self.send_error_json(422, "tool_io_error", "tool I/O failed", request_id=request_id)
        except TypeError as err:
            status = 400
            self.send_error_json(
                400,
                "bad_arguments",
                f"bad arguments for {name}: {err}",
                request_id=request_id,
            )
        except ValueError:
            status = 400
            self.send_error_json(400, "bad_arguments", f"invalid arguments for {name}", request_id=request_id)
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
                (time_monotonic() - started) * 1000,
            )
        return False

    def log_message(self, format, *args):
        return False


def parse_args(argv: list):
    """Parse explicit server settings supplied by the container manager."""
    parser = argparse_ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--auth-token", required=True)
    parser.add_argument("--egress-policy", choices=("allowlist", "deny", "direct"), default="direct")
    parser.add_argument("--egress-allowlist", default="")
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--max-tool-timeout", type=int, default=DEFAULT_MAX_TOOL_TIMEOUT)
    parser.add_argument("--request-read-timeout", type=float, default=DEFAULT_REQUEST_READ_TIMEOUT)
    computed_return_value = parser.parse_args(argv)
    return computed_return_value


def configure(args):
    """Apply validated process-wide settings before accepting requests."""
    global WORKSPACE, EGRESS_POLICY, MAX_OUTPUT_BYTES, MAX_FILE_BYTES
    global MAX_TOOL_TIMEOUT
    if len(args.auth_token) < 32:
        raise ValueError("auth token must be at least 32 characters")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    for name in (
        "max_request_bytes",
        "max_output_bytes",
        "max_file_bytes",
        "max_response_bytes",
        "max_concurrent_requests",
        "max_tool_timeout",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_response_bytes < MIN_MAX_RESPONSE_BYTES:
        raise ValueError(f"max_response_bytes must be at least {MIN_MAX_RESPONSE_BYTES}")
    if args.request_read_timeout <= 0:
        raise ValueError("request_read_timeout must be positive")
    WORKSPACE = args.workspace.resolve()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    EGRESS_POLICY = args.egress_policy
    MAX_OUTPUT_BYTES = args.max_output_bytes
    MAX_FILE_BYTES = args.max_file_bytes
    MAX_TOOL_TIMEOUT = args.max_tool_timeout
    return False


def main(argv: list):
    logging_basicConfig(level=logging_INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    configure(args)
    server = ToolHTTPServer(
        ("0.0.0.0", args.port),
        ToolHandler,
        auth_token=args.auth_token,
        max_request_bytes=args.max_request_bytes,
        max_response_bytes=args.max_response_bytes,
        max_concurrent_requests=args.max_concurrent_requests,
        request_read_timeout=args.request_read_timeout,
    )
    server.serve_forever()
    return False


if __name__ == "__main__":
    main(sys_argv[1:])
