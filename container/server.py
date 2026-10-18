"""Authenticated, bounded gRPC tool server for the Python sandbox.

The component owns gRPC tools and a separate observation-only HTTP listener.
Accepted tool functions drain even when their caller disconnects.
"""

from argparse import ArgumentParser as argparse_ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from hashlib import sha256 as hashlib_sha256
from hmac import compare_digest as hmac_compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from inspect import Parameter as inspect_Parameter, signature as inspect_signature
from json import dumps as json_dumps, loads as json_loads
from logging import (
    INFO as logging_INFO,
    basicConfig as logging_basicConfig,
    getLogger as logging_getLogger,
)
from math import isfinite
from os import (
    O_CLOEXEC,
    O_DIRECTORY,
    O_NOFOLLOW,
    O_NONBLOCK,
    O_RDONLY,
    ST_RDONLY,
    close as os_close,
    environ as os_environ,
    fdopen as os_fdopen,
    fstat as os_fstat,
    fsync as os_fsync,
    getpid as os_getpid,
    kill as os_kill,
    killpg as os_killpg,
    open as os_open,
    read as os_read,
    scandir as os_scandir,
    statvfs as os_statvfs,
)
from pathlib import Path
from re import compile as re_compile, fullmatch as re_fullmatch, sub as re_sub
from selectors import (
    EVENT_READ as selectors_EVENT_READ,
    DefaultSelector as selectors_DefaultSelector,
)
from shutil import (
    disk_usage as shutil_disk_usage,
    move as shutil_move,
    rmtree as shutil_rmtree,
    which as shutil_which,
)
from signal import (
    SIGINT as signal_SIGINT,
    SIGKILL as signal_SIGKILL,
    SIGTERM as signal_SIGTERM,
    signal as signal_signal,
)
from socketserver import TCPServer
from stat import S_ISREG
from subprocess import (
    PIPE as subprocess_PIPE,
    Popen as subprocess_Popen,
    TimeoutExpired as subprocess_TimeoutExpired,
)
from sys import argv as sys_argv, executable as sys_executable, version as sys_version
from tempfile import (
    NamedTemporaryFile as tempfile_NamedTemporaryFile,
    TemporaryFile as tempfile_TemporaryFile,
    mkstemp as tempfile_mkstemp,
)
from threading import (
    BoundedSemaphore as threading_BoundedSemaphore,
    Event as threading_Event,
    RLock as threading_RLock,
)
from time import monotonic as time_monotonic
from uuid import UUID, uuid4

from grpc import StatusCode, server as grpc_server
from health import PAGE_HEADERS, WorkspaceHealth, page_file

from tapestry.interfaces.envelope import decode_envelope, encode_envelope
from tapestry.interfaces.workspace.constants import (
    CONTROL_WORKERS,
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    ENVELOPE_FRAME_BYTES,
)
from tapestry.interfaces.workspace.workspace_pb2 import WorkspaceEnvelope
from tapestry.interfaces.workspace.workspace_pb2_grpc import WorkspaceServiceServicer, add_WorkspaceServiceServicer_to_server

DEFAULT_ARGUMENT_DICT = {}
INVOCATION_MARKER_VARIABLE = "TAPESTRY_TOOL_INVOCATION"

PROTOCOL_NAME = "tapestry.workspace.grpc"
DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_PORT = 8080
DEFAULT_HEALTH_PORT = 8081
DEFAULT_MAX_OUTPUT_BYTES = 1_048_576
DEFAULT_MAX_FILE_BYTES = 10_485_760
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
DEFAULT_MAX_TOOL_TIMEOUT = 300
DEFAULT_REQUEST_READ_TIMEOUT = 10.0
DEFAULT_MAX_LIST_RESULTS = 5_000
DEFAULT_MAX_LIST_ENTRIES = 10_000
DEFAULT_LIST_TIMEOUT = 5.0
DEFAULT_MAX_SEARCH_RESULTS = 1_000
DEFAULT_MAX_GIT_LOG_RESULTS = 100
MIN_MAX_RESPONSE_BYTES = 512
# The bearer key file holds one line of at least this many characters, within this byte bound.
MIN_AUTH_TOKEN_LENGTH = 32
MAX_AUTH_TOKEN_FILE_BYTES = 4096

WORKSPACE = DEFAULT_WORKSPACE
# Outside a request the context is the shared empty mapping; each request sets its own resolved mapping.
WORKING_CONTEXT = ContextVar("tapestry.workspace.working_context", default=DEFAULT_ARGUMENT_DICT)
EGRESS_POLICY = "direct"
EGRESS_ENFORCEMENT = "container"
# The container image the manager started this server from, shown on the health page; empty when not supplied.
IMAGE = ""
# Routes answered without the Workspace key: the health page, its assets and the health API.
PUBLIC_PATHS = {"/", "/health.js", "/health-common.js", "/health.css", "/api/health", "/health/live", "/health/ready"}
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


def current_workspace() -> Path:
    """Use this request's directory without changing the process-wide cwd."""
    context = WORKING_CONTEXT.get()
    directory = context.get("directory", WORKSPACE)
    return directory


def request_working_context(root_task_id: str) -> dict:
    """Resolve the folder of the root Task that sent this request inside the writable mount."""
    if not root_task_id:
        raise ToolError(
            "Every tool call names its root Task",
            code="invalid_working_context",
            status=400,
        )
    try:
        identity = UUID(root_task_id)
    except ValueError as err:
        raise ToolError(
            "Working context requires a canonical native root Task",
            code="invalid_working_context",
            status=400,
        ) from err
    if identity.version != 7 or str(identity) != root_task_id:
        raise ToolError(
            "Working context requires a canonical native root Task",
            code="invalid_working_context",
            status=400,
        )
    parent = WORKSPACE / ".tapestry-work"
    directory = parent / root_task_id
    with FILE_LOCK:
        parent.mkdir(exist_ok=True)
        if parent.resolve() != parent or directory.is_symlink():
            raise ToolError(
                "Working context cannot redirect outside its assigned directory",
                code="path_escape",
                status=400,
            )
        directory.mkdir(exist_ok=True)
        if directory.resolve() != directory:
            raise ToolError(
                "Working context cannot redirect outside its assigned directory",
                code="path_escape",
                status=400,
            )
    return {"root_task_id": root_task_id, "directory": directory}


def wire_working_context(context: dict) -> dict:
    """Return the request's root Task folder as the reply names it, without the host directory."""
    root_task_id = context.get("root_task_id", "")
    wire = {"rootTaskId": root_task_id} if root_task_id else {}
    return wire


def resolve_path(path):
    """Resolve a path within the workspace, rejecting anything outside it."""
    workspace = current_workspace()
    target = (workspace / str(path)).resolve()
    if not target.is_relative_to(workspace):
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
    """Admit a finite native positive tool timeout before acquiring a process."""
    if type(timeout) not in (int, float):
        raise ToolError("timeout must be a native number", code="invalid_timeout", status=400)
    try:
        value = float(timeout)
    except OverflowError as err:
        raise ToolError("timeout must be finite", code="invalid_timeout", status=400) from err
    if not isfinite(value):
        raise ToolError("timeout must be finite", code="invalid_timeout", status=400)
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
        no_processes = set()
        return no_processes
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
    invocation_ids = (descendants | marked | {root_pid}) - {os_getpid()}
    return invocation_ids


def signal_invocation(process, marker, signal_number):
    """Signal the invocation's process group and every collected invocation process."""
    try:
        os_killpg(process.pid, signal_number)
    except ProcessLookupError:
        # The group has already exited; marked invocation processes that left it are still signalled below.
        LOGGER.debug(
            "Invocation process group %s exited before signal %s",
            process.pid,
            signal_number,
        )
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
        LOGGER.warning(
            "tool invocation %s did not fully stop after SIGTERM; sending SIGKILL",
            process.pid,
        )
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


def internal_run_process(
    command,
    *,
    shell=False,
    timeout: float = 60.0,
    cwd="",
    environment=DEFAULT_ARGUMENT_DICT,
    inherited_descriptors: tuple = (),
):
    """Run a bounded process and return a structured camelCase result."""
    if environment is DEFAULT_ARGUMENT_DICT:
        environment = DEFAULT_ARGUMENT_DICT.copy()
    timeout = bounded_timeout(timeout)
    started = time_monotonic()
    marker = uuid4().hex
    process_environment = {
        **(environment or os_environ),
        INVOCATION_MARKER_VARIABLE: marker,
    }
    process = subprocess_Popen(
        command,
        shell=shell,
        cwd=cwd or current_workspace(),
        env=process_environment,
        stdout=subprocess_PIPE,
        stderr=subprocess_PIPE,
        start_new_session=True,
        pass_fds=inherited_descriptors,
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
    detail = (result.get("stderr", "") or result.get("stdout", "")).strip()
    raise ToolError(
        f"{operation} failed with exit code {result.get('exitCode', -1)}" + (f": {detail}" if detail else ""),
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
        content = workspace_file_text(target)
    return content


def workspace_file_text(target) -> str:
    """Read bounded regular-file bytes through the canonical descriptor owner."""
    try:
        source = scoped_repository_bytes(current_workspace(), target, MAX_FILE_BYTES)
    except ToolError as err:
        if err.code == "bad_arguments":
            raise ToolError("Expected a regular file", code="not_file", status=400) from err
        raise
    if len(source) > MAX_FILE_BYTES:
        raise ToolError("File exceeds its byte allowance", code="file_too_large", status=413)
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as err:
        raise ToolError("File is not UTF-8", code="invalid_source_encoding", status=422) from err
    return text


@tool
def file_read_scoped(repository_root: str, path: str, max_bytes: int = 0):
    """Read exact UTF-8 bytes beneath one resolved repository root with provenance."""
    if type(repository_root) is not str or not repository_root or type(path) is not str or not path:
        raise ToolError(
            "Scoped read requires repository root and relative path",
            code="bad_arguments",
            status=400,
        )
    if type(max_bytes) is not int or max_bytes < 0:
        raise ToolError("Scoped read byte limit is invalid", code="bad_arguments", status=400)
    byte_budget = min(max_bytes, MAX_FILE_BYTES) if max_bytes else MAX_FILE_BYTES
    parts = path.split("/")
    if Path(path).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ToolError(
            "Scoped read path must be canonical and relative",
            code="bad_arguments",
            status=400,
        )
    root = resolve_path(repository_root)
    target = (root / path).resolve()
    if not root.is_dir() or not target.is_relative_to(root):
        raise ToolError("Scoped read escaped its repository root", code="path_escape", status=400)
    with FILE_LOCK:
        source = scoped_repository_bytes(root, target, byte_budget)
    if len(source) > byte_budget:
        raise ToolError(
            "Scoped source changed beyond its byte allowance",
            code="file_too_large",
            status=413,
        )
    try:
        content = source.decode("utf-8")
    except UnicodeDecodeError as err:
        raise ToolError("Scoped source is not UTF-8", code="invalid_source_encoding", status=422) from err
    scoped_read = {
        "repository_root": str(root),
        "resolved_path": str(target),
        "relative_path": str(target.relative_to(root)),
        "content": content,
        "content_sha256": hashlib_sha256(source).hexdigest(),
        "byte_count": len(source),
    }
    return scoped_read


def scoped_repository_bytes(root: Path, target: Path, byte_limit: int) -> bytes:
    """Open canonical path components by descriptor so later symlink swaps cannot escape."""
    directory = os_open("/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    try:
        relative = target.relative_to(root)
        if not relative.name:
            raise ToolError("Scoped source is not a regular file", code="bad_arguments", status=400)
        for component in (*root.parts[1:], *relative.parts[:-1]):
            try:
                child = os_open(
                    component,
                    O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC,
                    dir_fd=directory,
                )
            except PermissionError:
                raise
            except OSError as err:
                raise ToolError(
                    "Scoped repository directory changed or is unavailable",
                    code="path_escape",
                    status=400,
                ) from err
            os_close(directory)
            directory = child
        try:
            descriptor = os_open(
                relative.name,
                O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC,
                dir_fd=directory,
            )
        except (FileNotFoundError, PermissionError):
            raise
        except OSError as err:
            raise ToolError(
                "Scoped repository file changed or is unavailable",
                code="path_escape",
                status=400,
            ) from err
        with os_fdopen(descriptor, "rb") as opened:
            state = os_fstat(opened.fileno())
            if not S_ISREG(state.st_mode):
                raise ToolError(
                    "Scoped source is not a regular file",
                    code="bad_arguments",
                    status=400,
                )
            if state.st_size > byte_limit:
                raise ToolError(
                    "Scoped source exceeds its byte allowance",
                    code="file_too_large",
                    status=413,
                )
            source = opened.read(byte_limit + 1)
            return source
    finally:
        os_close(directory)


@tool
def file_write(path: str, content: str):
    """Atomically write UTF-8 content to a file."""
    if not isinstance(content, str):
        raise ToolError("content must be a string", code="bad_arguments", status=400)
    target = resolve_path(path)
    with FILE_LOCK:
        atomic_write_text(target, content)
    computed_return_value = {
        "path": str(target.relative_to(current_workspace())),
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
        content = workspace_file_text(target)
        for patch in patches:
            if not isinstance(patch, dict) or "old" not in patch or "new" not in patch:
                raise ToolError(
                    "each patch requires old and new strings",
                    code="bad_arguments",
                    status=400,
                )
            old = patch.get("old", "")
            new = patch.get("new", "")
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
        "path": str(target.relative_to(current_workspace())),
        "patchesApplied": len(patches),
    }
    return computed_return_value


@tool
def file_delete(path: str):
    """Delete one file from the workspace."""
    target = resolve_path(path)
    with FILE_LOCK:
        target.unlink(missing_ok=False)
    computed_return_value = {
        "path": str(target.relative_to(current_workspace())),
        "deleted": True,
    }
    return computed_return_value


@tool
def file_list(path: str = ".", depth: int = 2, max_results: int = DEFAULT_MAX_LIST_RESULTS):
    """List the inspected root, files and directories with explicit recursive coverage."""
    target = resolve_path(path)
    workspace = current_workspace()
    if not target.exists():
        raise ToolError(
            f"No such directory: {path!r} is resolved relative to the working directory, and '.' lists the working "
            "directory itself",
            code="not_found",
            status=404,
        )
    if not target.is_dir():
        raise ToolError("Path is not a directory", code="not_directory", status=400)
    depth = max(0, min(int(depth), 64))
    max_results = max(1, min(int(max_results), DEFAULT_MAX_LIST_RESULTS))
    started = time_monotonic()
    deadline = started + min(DEFAULT_LIST_TIMEOUT, MAX_TOOL_TIMEOUT)
    files = []
    directories = []
    links = []
    other = []
    depth_limited = depth == 0
    pending = [(target, 0)] if depth else []
    visited_entries = 0
    visited_directories = 0
    stop_reason = "complete"
    root = {
        "path": str(target.relative_to(workspace)),
        "kind": "directory",
        "children": [],
    }
    tree_nodes = {root.get("path", ""): root}
    while pending:
        if visited_entries >= DEFAULT_MAX_LIST_ENTRIES:
            stop_reason = "entry_limit"
            break
        if time_monotonic() >= deadline:
            stop_reason = "time_limit"
            break
        directory, directory_depth = pending.pop()
        entries = []
        with os_scandir(directory) as iterator:
            visited_directories += 1
            while True:
                if visited_entries >= DEFAULT_MAX_LIST_ENTRIES:
                    stop_reason = "entry_limit"
                    break
                if time_monotonic() >= deadline:
                    stop_reason = "time_limit"
                    break
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                visited_entries += 1
                entries.append(entry)
        # A partial directory cannot supply deterministic selection from its raw filesystem order.
        if stop_reason != "complete":
            break
        children = []
        parent_node = tree_nodes.get(str(directory.relative_to(workspace)), {})
        # A directory is queued only after its own node is recorded, so a missing parent is a broken listing.
        if not parent_node:
            raise RuntimeError("workspace listing lost the node of a queued directory")
        for entry in sorted(entries, key=lambda item: item.name):
            if time_monotonic() >= deadline:
                stop_reason = "time_limit"
                break
            candidate = Path(entry.path)
            relative_path = str(candidate.relative_to(workspace))
            if entry.is_symlink():
                links.append(relative_path)
                kind = "link"
            elif entry.is_file(follow_symlinks=False):
                files.append(relative_path)
                kind = "file"
            elif entry.is_dir(follow_symlinks=False):
                directories.append(relative_path)
                kind = "directory"
                if directory_depth + 1 < depth:
                    children.append((candidate, directory_depth + 1))
                else:
                    depth_limited = True
            else:
                other.append(relative_path)
                kind = "other"
            node: dict = {"path": relative_path, "kind": kind}
            if kind == "directory":
                node["children"] = []
                tree_nodes[relative_path] = node
            parent_node.get("children", []).append(node)
            if len(files) + len(directories) + len(links) + len(other) >= max_results:
                stop_reason = "result_limit"
                break
        if stop_reason != "complete":
            break
        pending.extend(reversed(children))
    computed_return_value = {
        "workspaceRoot": str(workspace),
        "root": root,
        "files": sorted(files),
        "directories": sorted(directories),
        "links": sorted(links),
        "other": sorted(other),
        "scan": {
            "depth": depth,
            "depthLimited": depth_limited,
            "complete": stop_reason == "complete" and not depth_limited,
        },
        "truncated": stop_reason != "complete",
        "visitedEntries": visited_entries,
        "visitedDirectories": visited_directories,
        "durationMs": round((time_monotonic() - started) * 1000, 3),
        "stopReason": stop_reason,
    }
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
                "file": str(Path(match.get("path", {}).get("text", "")).relative_to(current_workspace())),
                "line": match.get("line_number", 0),
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


@tool
def file_search_scoped(repository_root: str, symbol: str, max_results: int = 0):
    """Return distinct Python-source paths within configured retrieval and optional caller budgets."""
    if type(repository_root) is not str or not repository_root or type(symbol) is not str:
        raise ToolError(
            "Scoped Python search requires repository root and symbol",
            code="bad_arguments",
            status=400,
        )
    if re_fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", symbol) is None:
        raise ToolError("Scoped Python search symbol is invalid", code="bad_arguments", status=400)
    if type(max_results) is not int or max_results < 0:
        raise ToolError("Scoped Python search limit is invalid", code="bad_arguments", status=400)
    root = resolve_path(repository_root)
    if not root.is_dir():
        raise ToolError("Scoped Python repository root is missing", code="path_missing", status=404)
    result = internal_run_process(
        [
            "rg",
            "--json",
            "--fixed-strings",
            "--glob",
            "*.py",
            "-e",
            symbol,
            "--",
            str(root),
        ],
        timeout=MAX_TOOL_TIMEOUT,
    )
    if result.get("exitCode", -1) not in (0, 1, -2):
        require_process_success(result, "scoped Python search")
    matches_by_path = {}
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
        selected = Path(data.get("data", {}).get("path", {}).get("text", "")).resolve()
        if not selected.is_relative_to(root):
            raise ToolError(
                "Scoped Python search escaped its repository root",
                code="path_escape",
                status=400,
            )
        relative = str(selected.relative_to(root))
        if relative not in matches_by_path:
            matches_by_path[relative] = {
                "relative_path": str(selected.relative_to(root)),
                "line": data.get("data", {}).get("line_number", 0),
            }
    matches = [matches_by_path.get(path, {}) for path in sorted(matches_by_path)]
    truncated = bool(result.get("outputLimited", False) or result.get("timedOut", False) or result.get("exitCode", 0) == -2)
    if max_results and len(matches) > max_results:
        matches = matches[:max_results]
        truncated = True
    search = {
        "repository_root": str(root),
        "symbol": symbol,
        "matches": matches,
        "truncated": truncated,
    }
    return search


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
    """Uninstall packages that pip_install added at runtime."""
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
    computed_return_value = json_loads(result.get("stdout", ""))
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
    with tempfile_TemporaryFile(mode="w+b") as source:
        source.write(script.encode("utf-8"))
        source.seek(0)
        descriptor = source.fileno()
        bootstrap = (
            f"with open({descriptor}, encoding='utf-8') as source:\n"
            "    source_code = compile(source.read(), '<string>', 'exec')\n"
            "del source\n"
            "exec(globals().pop('source_code'))"
        )
        result = internal_run_process([sys_executable, "-c", bootstrap], timeout=timeout, inherited_descriptors=(descriptor,))
    return result


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
    workspace = current_workspace()
    staging = workspace / f".tapestry-clone-{uuid4().hex}"
    with FILE_LOCK:
        try:
            result = run_git(
                ["clone", "--branch", branch, "--", repo_url, str(staging)],
                timeout=MAX_TOOL_TIMEOUT,
            )
            require_process_success(result, "git clone")
            for item in list(workspace.iterdir()):
                if item == staging:
                    continue
                shutil_rmtree(item) if item.is_dir() else item.unlink()
            for item in list(staging.iterdir()):
                shutil_move(str(item), workspace / item.name)
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
        "diff": result.get("stdout", ""),
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


def observed_mounts() -> list:
    """Describe actual directory mount permissions; metadata grants no Task authority."""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError) as err:
        LOGGER.warning("Workspace mount metadata unavailable error_type=%s", type(err).__name__)
        unavailable = []
        return unavailable
    mounts = {}
    for line in lines:
        fields = line.split(" - ", 1)[0].split()
        if len(fields) < 6:
            raise ValueError("Workspace mount metadata has an invalid record")
        mount_path = fields[4]
        for encoded, decoded in (
            ("\\040", " "),
            ("\\011", "\t"),
            ("\\012", "\n"),
            ("\\134", "\\"),
        ):
            mount_path = mount_path.replace(encoded, decoded)
        target = Path(mount_path)
        try:
            if target.is_dir() and str(target.resolve()) == mount_path:
                mounts[mount_path] = {
                    "path": mount_path,
                    "readOnly": bool(os_statvfs(mount_path).f_flag & ST_RDONLY),
                }
        except PermissionError:
            LOGGER.debug("Workspace mount excluded reason=permission_denied")
    observed = [mounts.get(path, {}) for path in sorted(mounts)]
    return observed


def build_manifest():
    """Describe every tool with a JSON-schema-shaped signature."""
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
    manifest = {
        "protocol": PROTOCOL_NAME,
        "incarnation": "",
        "egressPolicy": EGRESS_POLICY,
        "egressEnforcement": EGRESS_ENFORCEMENT,
        "mounts": observed_mounts(),
        "tools": tools,
    }
    return manifest


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
    if not ready:
        problems = []
        if not workspace_exists:
            problems.append(f"workspace {WORKSPACE} does not exist")
        elif not workspace_writable:
            problems.append(f"workspace {WORKSPACE} is not writable")
        missing = [name for name, found in tools.items() if not found]
        if missing:
            problems.append(f"required tools not on PATH: {', '.join(missing)}")
        result["error"] = {
            "code": "unhealthy",
            "message": "sandbox is not ready: " + "; ".join(problems),
        }
    return result


class WorkspaceService(WorkspaceServiceServicer):
    """Own authenticated RPC admission, accepted work, tool slots and one health projection."""

    def __init__(self, auth_token: str, args):
        self.auth_token = auth_token
        self.incarnation = str(uuid4())
        self.max_request_bytes = args.max_request_bytes
        self.max_response_bytes = args.max_response_bytes
        self.max_concurrent_requests = args.max_concurrent_requests
        self.request_slots = threading_BoundedSemaphore(self.max_concurrent_requests)
        self.health = WorkspaceHealth(self.incarnation)
        self.workers = ThreadPoolExecutor(max_workers=self.max_concurrent_requests + CONTROL_WORKERS)
        self.rpc = grpc_server(
            self.workers,
            options=(
                ("grpc.max_receive_message_length", self.max_request_bytes + ENVELOPE_FRAME_BYTES),
                ("grpc.max_send_message_length", self.max_response_bytes + ENVELOPE_FRAME_BYTES),
            ),
            maximum_concurrent_rpcs=self.max_concurrent_requests + CONTROL_WORKERS,
        )
        add_WorkspaceServiceServicer_to_server(self, self.rpc)
        self.port = 0
        self.closed = False

    def start(self, address: str) -> int:
        """Bind and start the sole tool transport; a failed bind cannot become ready."""
        self.port = self.rpc.add_insecure_port(address)
        if not self.port:
            raise RuntimeError("Workspace gRPC address could not bind")
        self.rpc.start()
        return self.port

    def close(self) -> None:
        """Stop admission and drain actual tool functions even after their RPCs disconnected."""
        if not self.closed:
            self.closed = True
            self.rpc.stop(0).wait()
            self.workers.shutdown(wait=True)

    def authorize(self, context, operation: str) -> None:
        headers = [value for key, value in context.invocation_metadata() if key == "authorization"]
        expected = f"Bearer {self.auth_token}".encode("utf-8")
        if len(headers) != 1 or type(headers[0]) is not str or not hmac_compare_digest(headers[0].encode("utf-8"), expected):
            self.health.refused(operation, "UNAUTHORIZED", "unauthorized")
            context.abort(StatusCode.UNAUTHENTICATED, "A valid Workspace service key is required")

    def decode(self, request, context, fields: set) -> dict:
        try:
            payload = decode_envelope(request.payload, self.max_request_bytes, "Workspace")
            if set(payload) != fields:
                raise ValueError("Workspace RPC fields differ from its operation contract")
            if "request_id" in fields:
                identifier = payload.get("request_id", "")
                if type(identifier) is not str or not re_fullmatch(r"[A-Za-z0-9._:-]{1,128}", identifier):
                    raise ValueError("Workspace request ID is invalid")
        except (TypeError, ValueError, OverflowError) as err:
            self.health.refused("request", "INVALID_ARGUMENT", "invalid_request")
            context.abort(StatusCode.INVALID_ARGUMENT, str(err))
            raise
        return payload

    def encode(self, payload: dict):
        encoded = encode_envelope(payload, self.max_response_bytes, "Workspace")
        response = WorkspaceEnvelope(payload=encoded)
        return response

    def readiness(self) -> dict:
        payload = build_readiness_report()
        payload.update({"incarnation": self.incarnation, "egressPolicy": EGRESS_POLICY, "egressEnforcement": EGRESS_ENFORCEMENT})
        return payload

    def health_snapshot(self) -> dict:
        snapshot = self.health.snapshot(
            build_readiness_report(),
            {
                "egress_policy": EGRESS_POLICY,
                "egress_enforcement": EGRESS_ENFORCEMENT,
                "image": IMAGE,
                "tool_call_limit": self.max_concurrent_requests,
            },
        )
        return snapshot

    def CheckHealth(self, request, context):
        self.decode(request, context, set())
        response = self.encode(self.readiness())
        return response

    def Describe(self, request, context):
        self.authorize(context, "manifest")
        payload = self.decode(request, context, {"request_id"})
        identifier = payload.get("request_id", "")
        key = self.health.received("manifest", {"request_id": identifier})
        try:
            manifest = build_manifest()
            manifest.update({"incarnation": self.incarnation, "request_id": identifier})
            response = self.encode(manifest)
        except (TypeError, ValueError, OSError):
            self.health.finished(key, "MANIFEST_FAILED", "manifest_failed")
            context.abort(StatusCode.INTERNAL, "Workspace manifest is unavailable")
            raise
        self.health.finished(key, "OK", "")
        return response

    def InvokeTool(self, request, context):
        self.authorize(context, "tool")
        payload = self.decode(request, context, {"name", "arguments", "request_id", "root_task_id"})
        name = payload.get("name", "")
        arguments = payload.get("arguments", {})
        root = payload.get("root_task_id", "")
        if type(name) is not str or not re_fullmatch(r"[a-z][a-z0-9_]*", name):
            context.abort(StatusCode.INVALID_ARGUMENT, "Workspace tool name is invalid")
        if type(arguments) is not dict or any(type(key) is not str for key in arguments):
            context.abort(StatusCode.INVALID_ARGUMENT, "Workspace arguments require a string-key mapping")
        identifier = payload.get("request_id", "")
        fn = TOOLS.get(name, False)
        operation = name if fn is not False else "unknown tool"
        working = (
            {"root_task_id": root}
            if type(root) is str and re_fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", root)
            else {}
        )
        if not working:
            response = self.refuse(identifier, operation, 400, "invalid_working_context", "A canonical root Task is required", {})
            return response
        if fn is False:
            response = self.refuse(identifier, operation, 404, "unknown_tool", "Workspace tool is unknown", working)
            return response
        if not self.request_slots.acquire(blocking=False):
            response = self.refuse(identifier, operation, 429, "busy", "Workspace tool concurrency limit reached", working)
            return response
        started = time_monotonic()
        key = self.health.received(name, {"request_id": identifier, "root_task_id": root})
        context_token = False
        observation = {"request_id": identifier, "status_code": 200, "result": {}, "workspace_context": working}
        code = ""
        try:
            normalized = {snake_case(key): value for key, value in arguments.items()}
            if len(normalized) != len(arguments):
                raise ToolError("Tool arguments repeat a normalized name", code="bad_arguments", status=400)
            inspect_signature(fn).bind(**normalized)
            resolved = request_working_context(root)
            context_token = WORKING_CONTEXT.set(resolved)
            observation["result"] = fn(**normalized)
        except ToolError as err:
            code = err.code
            observation.update({"status_code": err.status, "result": {"error": {"code": code, "message": str(err)}}})
        except (OSError, TypeError, ValueError) as err:
            status, code, message = tool_failure(err)
            observation.update({"status_code": status, "result": {"error": {"code": code, "message": message}}})
        except Exception:
            code = "internal_error"
            LOGGER.exception("Unhandled tool error request_id=%s tool=%s", identifier, name)
            observation.update({"status_code": 500, "result": {"error": {"code": code, "message": "Tool execution failed"}}})
        finally:
            if context_token is not False:
                WORKING_CONTEXT.reset(context_token)
            # Encoding is part of the outcome, and a disconnected client cannot release this slot early.
            try:
                response = self.encode(observation)
            except (TypeError, ValueError, OverflowError):
                code = "response_too_large"
                observation.update(
                    {"status_code": 413, "result": {"error": {"code": code, "message": "Tool response exceeds its allowance"}}}
                )
                response = self.encode(observation)
            finally:
                self.request_slots.release()
                result = observation.get("result", {})
                outcome = (
                    code.upper() if code else "PROCESS_FAILED" if type(result) is dict and result.get("ok", True) is False else "OK"
                )
                self.health.finished(key, outcome, code)
                LOGGER.info(
                    "tool_request request_id=%s tool=%s status=%s outcome=%s duration_ms=%.3f",
                    identifier,
                    name,
                    observation.get("status_code", 0),
                    outcome,
                    (time_monotonic() - started) * 1000,
                )
        return response

    def refuse(self, identifier: str, operation: str, status: int, code: str, message: str, working: dict):
        self.health.refused(operation, code.upper(), code)
        response = self.encode(
            {
                "request_id": identifier,
                "status_code": status,
                "result": {"error": {"code": code, "message": message}},
                "workspace_context": working,
            }
        )
        return response


def tool_failure(error: Exception) -> tuple:
    """Classify the owning tool's bounded failure without leaking paths, arguments or subprocess output."""
    if isinstance(error, FileNotFoundError):
        failure = (404, "not_found", "Requested file or tool was not found")
    elif isinstance(error, IsADirectoryError):
        failure = (400, "not_file", "Expected a file")
    elif isinstance(error, NotADirectoryError):
        failure = (400, "not_directory", "Expected a directory")
    elif isinstance(error, PermissionError):
        failure = (403, "permission_denied", "Tool access was denied")
    elif isinstance(error, OSError):
        failure = (422, "tool_io_error", "Tool I/O failed")
    else:
        failure = (400, "bad_arguments", "Invalid tool arguments")
    return failure


class HealthHTTPServer(ThreadingHTTPServer):
    """Own the observation listener while the gRPC service owns its safe snapshot."""

    daemon_threads = False

    def __init__(self, address: tuple, owner: WorkspaceService, request_read_timeout: float):
        self.owner = owner
        self.request_read_timeout = request_read_timeout
        super().__init__(address, HealthHandler)

    def server_bind(self):
        TCPServer.server_bind(self)
        address = self.server_address
        if not isinstance(address, tuple) or len(address) != 2:
            raise RuntimeError("Workspace health listener requires an internet address")
        host, port = address
        if type(host) is not str or type(port) is not int:
            raise RuntimeError("Workspace health listener returned an invalid address")
        self.server_name = host
        self.server_port = port


class HealthHandler(BaseHTTPRequestHandler):
    """Public observation only; tools and manifests have no HTTP routes."""

    server: HealthHTTPServer
    server_version = "TapestryWorkspace"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.request_read_timeout)

    def do_GET(self):
        body, media_type = page_file(self.path)
        status = 200
        if not media_type:
            media_type = "application/json"
            if self.path == "/api/health":
                payload = self.server.owner.health_snapshot()
            elif self.path == "/health/live":
                payload = {"status": "running"}
            elif self.path == "/health/ready":
                payload = self.server.owner.readiness()
                status = 200 if payload.get("status", "") == "healthy" else 503
            else:
                payload = {"error": "not_found"}
                status = 404
            body = json_dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in PAGE_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return False


def parse_args(argv: list):
    """Parse explicit server settings supplied by the container manager."""
    parser = argparse_ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--health-port", type=int, default=DEFAULT_HEALTH_PORT)
    parser.add_argument("--auth-token-fd", type=int, required=True)
    parser.add_argument("--egress-policy", choices=("web", "deny", "direct"), default="direct")
    parser.add_argument("--egress-enforcement", choices=("container", "deployment"), default="container")
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--max-tool-timeout", type=int, default=DEFAULT_MAX_TOOL_TIMEOUT)
    parser.add_argument("--request-read-timeout", type=float, default=DEFAULT_REQUEST_READ_TIMEOUT)
    parser.add_argument("--image", default="")
    computed_return_value = parser.parse_args(argv)
    return computed_return_value


def read_auth_token(args):
    """Read the bearer key once from a regular file and close it, so no tool child inherits it."""
    descriptor = args.auth_token_fd
    with os_fdopen(descriptor, "rb") as stream:
        metadata = os_fstat(stream.fileno())
        if not S_ISREG(metadata.st_mode):
            raise ValueError("auth token file must be a regular file")
        content = stream.read(MAX_AUTH_TOKEN_FILE_BYTES + 1)
    if len(content) > MAX_AUTH_TOKEN_FILE_BYTES:
        raise ValueError("auth token file exceeds its bound")
    token = content.decode("utf-8").removesuffix("\n")
    if len(token) < MIN_AUTH_TOKEN_LENGTH or any(character in token for character in ("\r", "\n", "\x00")):
        raise ValueError(f"auth token must be one line of at least {MIN_AUTH_TOKEN_LENGTH} characters")
    return token


def configure(args):
    """Apply validated process-wide settings before accepting requests."""
    global WORKSPACE, EGRESS_POLICY, EGRESS_ENFORCEMENT, MAX_OUTPUT_BYTES, MAX_FILE_BYTES
    global MAX_TOOL_TIMEOUT, IMAGE
    if not 1 <= args.port <= 65535 or not 1 <= args.health_port <= 65535 or args.port == args.health_port:
        raise ValueError("gRPC and health ports must be distinct ports between 1 and 65535")
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
    EGRESS_ENFORCEMENT = args.egress_enforcement
    MAX_OUTPUT_BYTES = args.max_output_bytes
    MAX_FILE_BYTES = args.max_file_bytes
    MAX_TOOL_TIMEOUT = args.max_tool_timeout
    IMAGE = args.image
    return False


def main(argv: list):
    logging_basicConfig(level=logging_INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    auth_token = read_auth_token(args)
    configure(args)
    owner = WorkspaceService(auth_token, args)
    server = HealthHTTPServer(("0.0.0.0", args.health_port), owner, args.request_read_timeout)
    owner.start(f"0.0.0.0:{args.port}")
    stopped = threading_Event()
    shutdown = {"signal": 0}

    def request_shutdown(number, frame):
        """Ask the component owner to stop accepting and drain current work."""
        shutdown["signal"] = number
        stopped.set()

    previous_handlers = {}
    try:
        for number in (signal_SIGTERM, signal_SIGINT):
            previous_handlers[number] = signal_signal(number, request_shutdown)
        server.timeout = 0.25
        while not stopped.is_set():
            server.handle_request()
            owner.health.sample()
        LOGGER.info(
            "workspace_shutdown requested signal=%s; draining accepted requests",
            shutdown.get("signal", 0),
        )
    finally:
        try:
            owner.close()
            server.server_close()
        finally:
            for number, handler in previous_handlers.items():
                signal_signal(number, handler)
    LOGGER.info("workspace_shutdown completed accepted requests drained")
    return False


if __name__ == "__main__":
    main(sys_argv[1:])
