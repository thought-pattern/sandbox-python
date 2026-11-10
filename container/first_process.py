"""The Workspace server's duties as its container's first process (D-1182, D-1183, D-649).

The server is the container's PID 1, with no separate init. pid_namespaces(7): only signals for which the first process
has established a handler can be sent to it by the other members of its namespace; SIGKILL and SIGSTOP are forced
through only from an ancestor namespace; and a signal sent from an ancestor namespace shows its sender as process 0. So
code a Task runs can neither kill nor stop the server. The server holds SIGTERM and SIGINT blocked in every thread and
reads each one with sigwaitinfo. It stops only on one sent from outside the container, which is the engine's stop or the
deployment's. A tool's signal to the server names the tool and is refused.

Every process whose parent exits is handed to the first process, so the server reaps each orphan that has ended (D-649).
Each tool leader's exit status stays with its own wait: a leader is held out of the reaper's reach from the moment it is
started until its wait has it.

A blocked signal stays blocked across fork and exec, so each tool program starts through the image's Python, run isolated
from everything a Task can write (`-I -S`: no user site, no PYTHON* variables, no site module). That step unblocks the
held signals and gives SIGPIPE and SIGXFSZ back the default actions a newly started program has, then execs the program
in its own place, with the same process, environment and descriptors.
"""

from errno import ENOENT
from logging import getLogger as logging_getLogger
from os import WNOHANG, defpath as os_defpath, getpid as os_getpid, waitpid as os_waitpid
from pathlib import Path
from shutil import which as shutil_which
from signal import SIG_BLOCK, SIGCHLD, SIGINT, SIGTERM, pthread_sigmask as signal_pthread_sigmask
from sys import executable as sys_executable, platform as sys_platform
from threading import Event as threading_Event, Lock as threading_Lock

# sigwaitinfo exists only on Linux, where the server runs as a container's first process; the in-process uses of the
# server on another host never serve its signals. Imported apart so Pyright checks it for Linux only.
if sys_platform == "linux":
    from signal import sigwaitinfo as signal_sigwaitinfo  # pylint: disable=ungrouped-imports

# The signals the server holds blocked in every thread and reads itself; a tool program gets them unblocked.
HELD_SIGNALS = (SIGTERM, SIGINT, SIGCHLD)
# A held signal that stops the server when it comes from outside the container.
STOP_SIGNALS = (SIGTERM, SIGINT)
# The process number the first process sees as the sender of a signal from outside its namespace.
OUTSIDE_SENDER = 0
FIRST_PROCESS_ID = 1
# Run as `python -I -S -c PROGRAM_START <program path> <argv...>`: argv[1] is the program, argv[2:] its own argv.
PROGRAM_START = (
    "import os, signal, sys\n"
    "signal.signal(signal.SIGPIPE, signal.SIG_DFL)\n"
    "signal.signal(signal.SIGXFSZ, signal.SIG_DFL)\n"
    "signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGTERM, signal.SIGINT, signal.SIGCHLD))\n"
    "os.execv(sys.argv[1], sys.argv[2:])\n"
)
LOGGER = logging_getLogger("tapestry.workspace.server")


def hold_signals() -> None:
    """Block the held signals in the calling thread; every thread and process it starts afterwards inherits the block."""
    signal_pthread_sigmask(SIG_BLOCK, HELD_SIGNALS)


def program_command(command, shell: bool, environment: dict) -> list:
    """The command that starts a tool's program through PROGRAM_START.

    The program is found as starting it directly finds it: a name with a slash is the path itself, any other name is
    searched for along the PATH of the environment it runs with. A program that cannot be found raises
    FileNotFoundError before anything starts, as starting it directly raises.
    """
    if shell:
        arguments = ["/bin/sh", "-c", command]
    elif type(command) is str:
        arguments = [command]
    else:
        arguments = [*command]
    program = shutil_which(arguments[0], path=environment.get("PATH", os_defpath))
    if not program:
        raise FileNotFoundError(ENOENT, "No such file or directory", arguments[0])
    started = [sys_executable, "-I", "-S", "-c", PROGRAM_START, program, *arguments]
    return started


class FirstProcess:
    """The server's signals and orphans as its container's first process.

    `start` starts a tool leader and `release` hands it back once its wait has it; `serve` reads the held signals for
    the server's life; `stopped` is set once a stop signal comes from outside the container, naming it in `stop_signal`.
    """

    def __init__(self):
        self.lock = threading_Lock()
        self.leaders = set()
        self.stopped = threading_Event()
        self.stop_signal = 0

    def start(self, open_process):
        """Start one tool leader with `open_process()` and hold it out of the reaper's reach until it is released."""
        with self.lock:
            process = open_process()
            self.leaders.add(process.pid)
        return process

    def release(self, process) -> None:
        """Hand a tool leader back once its own wait has its exit status, or once nobody will wait for it."""
        with self.lock:
            self.leaders.discard(process.pid)

    def reap_orphans(self) -> int:
        """Reap every child of the server that has ended and is not a held tool leader; return how many were reaped."""
        own = str(os_getpid())
        reaped = 0
        with self.lock:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit() or int(entry.name) in self.leaders:
                    continue
                try:
                    stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
                    fields = stat[stat.rindex(")") + 2 :].split()
                except (OSError, ValueError):
                    # The process ended and was reaped, or its status could not be read; it is no zombie child to reap.
                    continue
                if fields[:2] != ["Z", own]:
                    continue
                try:
                    ended, status = os_waitpid(int(entry.name), WNOHANG)
                except ChildProcessError:
                    continue
                if ended:
                    reaped += 1
                    LOGGER.debug("orphan %s reaped with status %s", ended, status)
        return reaped

    def accepts(self, sender: int) -> bool:
        """Whether a stop signal from `sender` stops the server: one from outside the container always does. A server
        that is not its container's first process has no protection to keep, so a stop signal from anyone stops it."""
        accepted = sender == OUTSIDE_SENDER or os_getpid() != FIRST_PROCESS_ID
        return accepted

    def serve(self) -> None:
        """Read the held signals for the server's life: reap orphans on SIGCHLD, stop on a stop signal from outside the
        container, refuse one from a process inside it."""
        if sys_platform != "linux":
            raise RuntimeError("the Workspace server reads its signals only on Linux")
        try:
            while True:
                information = signal_sigwaitinfo(HELD_SIGNALS)
                number, sender = information.si_signo, information.si_pid
                if number == SIGCHLD:
                    self.reap_orphans()
                elif self.accepts(sender):
                    self.stop_signal = number
                    self.stopped.set()
                    LOGGER.info("workspace stop signal %s accepted from outside the container", number)
                else:
                    LOGGER.warning("workspace stop signal %s from process %s inside the container refused", number, sender)
        except BaseException:
            # A server whose signals nobody reads could never be stopped, so it drains and ends instead.
            LOGGER.exception("workspace signal reader failed; the server drains and stops")
            self.stopped.set()
            raise
