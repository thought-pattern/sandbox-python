"""Control of a tool invocation's processes for the Workspace server: finding every process an invocation started, how
far they have got, and ending them all.

A tool invocation is its leader process, everything the leader started, and every process that still carries the
invocation's marker in its environment after leaving the leader's group or parent chain. The server's watchdog reads
the invocation's activity (`invocation_activity`) and ends the invocation with `terminate_process_group`.
"""

from logging import getLogger as logging_getLogger
from os import fstat as os_fstat, getpid as os_getpid, kill as os_kill, killpg as os_killpg
from pathlib import Path
from signal import SIGKILL as signal_SIGKILL, SIGTERM as signal_SIGTERM
from subprocess import TimeoutExpired as subprocess_TimeoutExpired

# The environment variable each tool invocation's processes inherit, whose value names the invocation.
INVOCATION_MARKER_VARIABLE = "TAPESTRY_TOOL_INVOCATION"
LOGGER = logging_getLogger("tapestry.workspace.server")


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
    """Terminate a tool invocation and all descendants, always escalating to SIGKILL. The caller then waits, with no
    limit, until the kernel has ended the leader and it is reaped. The quarter second between SIGTERM and SIGKILL
    applies only to an invocation the watchdog or its leader's exit has already ended; it decides nothing about whether
    work ends."""
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
    return False


def invocation_activity(process, marker, outputs: tuple) -> tuple:
    """Read how far a running tool invocation has got: its processes' CPU ticks and its output sizes.

    The ticks are each live invocation process's own and its reaped children's user and system time. Without /proc only
    the output sizes are read. Any change between two readings is activity.
    """
    ticks = 0
    proc = Path("/proc")
    if proc.is_dir():
        for pid in invocation_process_ids(process.pid, marker):
            try:
                stat = (proc / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
                ticks += sum(int(value) for value in stat[stat.rindex(")") + 2 :].split()[11:15])
            except (OSError, ValueError, IndexError):
                continue
    activity = (ticks, *(os_fstat(output.fileno()).st_size for output in outputs))
    return activity
