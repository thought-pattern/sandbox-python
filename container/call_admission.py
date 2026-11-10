"""How the Workspace server takes its calls: every gRPC call runs on a thread of its own, and tool calls take the tool
slots in the order they arrived.

A tool call that finds every slot taken waits its turn for as long as the calls ahead of it run; it is never refused
for want of a slot. A waiting call holds only its own thread, so a health or manifest call is answered while any number
of tool calls wait, and a stop answers every call accepted before it.
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock, Thread, current_thread


class CallThreads(ThreadPoolExecutor):
    """The gRPC server's executor: each call submitted runs on a new thread, so no call ever waits for a thread.

    It is a ThreadPoolExecutor because grpc's server takes one; it replaces the pool's `submit` and `shutdown`, the only
    methods grpc and the server call, so the pool's own workers are never started.
    """

    def __init__(self):
        super().__init__(thread_name_prefix="workspace-call")
        self.lock = Lock()
        self.threads = set()
        self.closed = False

    def submit(self, fn, /, *args, **kwargs) -> Future:
        """Start `fn` on a thread of its own and return the future its result or exception settles."""
        future = Future()
        thread = Thread(target=self.run, args=(future, fn, args, kwargs), name="workspace-call")
        with self.lock:
            if self.closed:
                raise RuntimeError("Workspace call threads are shut down")
            self.threads.add(thread)
        try:
            thread.start()
        except RuntimeError as err:
            with self.lock:
                self.threads.discard(thread)
            raise RuntimeError(f"Workspace call thread could not start: {err}") from err
        return future

    def run(self, future: Future, fn, args: tuple, kwargs: dict) -> None:
        """Run one call and settle its future, then leave the set of running call threads."""
        try:
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                except BaseException as err:
                    future.set_exception(err)
                else:
                    future.set_result(result)
        finally:
            with self.lock:
                self.threads.discard(current_thread())

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Take no more calls and, with `wait`, wait for every running call's thread to end.

        A call is started as soon as it is submitted, so there is never a pending call for `cancel_futures` to cancel.
        """
        with self.lock:
            self.closed = True
            threads = list(self.threads)
        if wait:
            for thread in threads:
                thread.join()


class ToolTurns:
    """The tool slots, taken in arrival order: a call takes a free slot only when no call waits, and a slot a call
    leaves passes straight to the call that has waited longest."""

    def __init__(self, slots: int):
        self.lock = Lock()
        self.free = slots
        self.waiting = deque()

    def take(self) -> None:
        """Hold a tool slot, waiting with no limit until every call that arrived earlier has taken one."""
        with self.lock:
            if self.free and not self.waiting:
                self.free -= 1
                return
            turn = Event()
            self.waiting.append(turn)
        turn.wait()

    def leave(self) -> None:
        """Pass this call's slot to the call that has waited longest, or free it when none waits."""
        with self.lock:
            if self.waiting:
                self.waiting.popleft().set()
            else:
                self.free += 1
