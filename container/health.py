"""The Workspace server's read-only health figures and page, served on its own port without the Workspace key.

The tool server reports each tool call and manifest request here as it arrives and answers. The figures keep the calls
by tool and outcome, the calls in progress, the latest finished calls, the errors by kind and a sample of recent
activity every SAMPLE_SECONDS. Only tool names, outcomes, error codes, times and the UUIDv7 request and root Task IDs
in ID_FIELDS are kept, never arguments, results, file contents or the key. Every figure covers the running server only.

The page assets in `assets/` are copies of Tapestry's shared health page assets (`tapestry/health_assets/`) with this
server's own page and renderer, because the Workspace runs without the tapestry package. The snapshot has the same shape
as every Tapestry component's, so the shared renderer draws it unchanged. The Workspace makes no model calls.
"""

from collections import deque
from datetime import UTC, datetime
from os import getpid
from pathlib import Path
from re import fullmatch
from threading import Lock
from time import monotonic
from uuid import uuid4

RECENT_OUTCOMES = 32
SAMPLE_SECONDS = 5.0
# 120 five-second samples keep ten minutes of recent activity.
HISTORY_SAMPLES = 120
ID_FIELDS = ("request_id", "root_task_id")
UUIDV7_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
ASSETS = Path(__file__).with_name("assets")
# The page's routes, each answered from one file in ASSETS with its media type.
PAGE_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/health.js": ("health.js", "text/javascript; charset=utf-8"),
    "/health-common.js": ("health-common.js", "text/javascript; charset=utf-8"),
    "/health.css": ("health.css", "text/css; charset=utf-8"),
}
PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    ),
}
DEFAULT_IDS = {}


def health_ids(values: dict) -> dict:
    """Keep only the named UUIDv7 identifiers the page may show."""
    ids = {name: values.get(name, "") for name in ID_FIELDS if fullmatch(UUIDV7_PATTERN, str(values.get(name, "")))}
    return ids


def operation_counts() -> dict:
    counts = {"received": 0, "responses": 0, "outcomes": {}}
    return counts


class WorkspaceHealth:
    """The tool server's figures since it started; every method takes the lock only briefly, from any request thread."""

    def __init__(self, incarnation: str):
        self.incarnation = incarnation
        self.lock = Lock()
        self.began = monotonic()
        self.started_at = datetime.now(UTC).isoformat()
        self.items = {}
        self.requests = {"received": 0, "responses": 0, "last_received_at": "", "last_response_at": "", "by_operation": {}}
        self.timing = {"total_seconds": 0.0, "terminal_count": 0, "execution_seconds": 0.0, "execution_count": 0}
        self.timing.update(queue_only_seconds=0.0, queue_only_count=0)
        self.finished_count = 0
        self.errors = {"by_kind": {}, "last": {}}
        self.recent = deque(maxlen=RECENT_OUTCOMES)
        self.history = deque(maxlen=HISTORY_SAMPLES)
        self.last_sample = self.began - SAMPLE_SECONDS

    def received(self, operation: str, ids: dict = DEFAULT_IDS) -> str:
        """Count one call arriving and hold it as running until `finished`; return its key."""
        key = uuid4().hex
        observed_at = datetime.now(UTC).isoformat()
        with self.lock:
            self.items[key] = {"operation": operation, "ids": health_ids(ids), "started": monotonic(), "accepted_at": observed_at}
            self.count_received(operation, observed_at)
        return key

    def count_received(self, operation: str, observed_at: str) -> None:
        """Count one call received; the lock is held."""
        self.requests["received"] = self.requests.get("received", 0) + 1
        self.requests["last_received_at"] = observed_at
        row = self.requests.get("by_operation", {}).setdefault(operation, operation_counts())
        row["received"] = row.get("received", 0) + 1

    def count_answer(self, operation: str, outcome: str, error_kind: str, observed_at: str) -> None:
        """Count one reply sent by operation and outcome, and its error kind; the lock is held."""
        self.requests["responses"] = self.requests.get("responses", 0) + 1
        self.requests["last_response_at"] = observed_at
        row = self.requests.get("by_operation", {}).setdefault(operation, operation_counts())
        row["responses"] = row.get("responses", 0) + 1
        outcomes = row.get("outcomes", {})
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if error_kind:
            kinds = self.errors.get("by_kind", {})
            kinds[error_kind] = kinds.get(error_kind, 0) + 1
            self.errors["last"] = {"kind": error_kind, "operation": operation, "observed_at": observed_at}

    def finished(self, key: str, outcome: str, error_kind: str = "") -> bool:
        """Record a running call's reply; return whether the key was held, so a call finishes once."""
        now = monotonic()
        observed_at = datetime.now(UTC).isoformat()
        with self.lock:
            item = self.items.pop(key, {})
            if not item:
                return False
            operation = item.get("operation", "")
            elapsed = max(0.0, now - item.get("started", now))
            self.count_answer(operation, outcome, error_kind, observed_at)
            for name, value in (("total_seconds", elapsed), ("execution_seconds", elapsed)):
                self.timing[name] = self.timing.get(name, 0.0) + value
            for name in ("terminal_count", "execution_count"):
                self.timing[name] = self.timing.get(name, 0) + 1
            self.finished_count += 1
            self.recent.append(
                {
                    "operation": operation,
                    "kind": "request",
                    "outcome": outcome,
                    "error_kind": error_kind,
                    "finished_at": observed_at,
                    "queue_seconds": 0.0,
                    "executed": True,
                    "execution_seconds": elapsed,
                    "total_seconds": elapsed,
                    "ids": dict(item.get("ids", {})),
                }
            )
        return True

    def refused(self, operation: str, outcome: str, error_kind: str) -> None:
        """Count a call answered without running: no key, an unknown tool, or every tool slot in use."""
        observed_at = datetime.now(UTC).isoformat()
        with self.lock:
            self.count_received(operation, observed_at)
            self.count_answer(operation, outcome, error_kind, observed_at)
            self.timing["terminal_count"] = self.timing.get("terminal_count", 0) + 1
            self.timing["queue_only_count"] = self.timing.get("queue_only_count", 0) + 1

    def sample(self) -> bool:
        """Keep one recent-activity sample, at most one every SAMPLE_SECONDS, and return whether it was kept."""
        now = monotonic()
        with self.lock:
            if now - self.last_sample < SAMPLE_SECONDS:
                return False
            self.last_sample = now
            self.history.append(
                {
                    "observed_at": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": now - self.began,
                    "queue_depth": 0,
                    "finished": self.finished_count,
                    "responses": self.requests.get("responses", 0),
                    "provider_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )
        return True

    def snapshot(self, readiness: dict, members: dict) -> dict:
        """The standard snapshot every Tapestry page shows, then the Workspace's readiness and `members`.

        A tool call never waits: one beyond the concurrency limit is refused as busy, so the queue is always empty.
        """
        now = monotonic()
        with self.lock:
            active = [
                {
                    "operation": item.get("operation", ""),
                    "kind": "request",
                    "stage": "",
                    "elapsed_seconds": max(0.0, now - item.get("started", now)),
                    "stage_seconds": max(0.0, now - item.get("started", now)),
                    "ids": dict(item.get("ids", {})),
                }
                for item in self.items.values()
            ]
            by_operation = {
                operation: {**row, "outcomes": dict(row.get("outcomes", {}))}
                for operation, row in self.requests.get("by_operation", {}).items()
            }
            requests = {**self.requests, "by_operation": by_operation}
            errors = {"by_kind": dict(self.errors.get("by_kind", {})), "last": dict(self.errors.get("last", {}))}
            timing = dict(self.timing)
            recent = list(reversed(self.recent))
            history = list(self.history)
        for average, total, samples in (
            ("average_total_seconds", "total_seconds", "terminal_count"),
            ("average_execution_seconds", "execution_seconds", "execution_count"),
        ):
            timing[average] = timing.get(total, 0.0) / timing.get(samples, 0) if timing.get(samples, 0) else 0.0
        ready = readiness.get("status", "") == "healthy"
        snapshot = {
            "component": "Workspace",
            "observed_at": datetime.now(UTC).isoformat(),
            "incarnation": self.incarnation,
            "process_id": getpid(),
            "started_at": self.started_at,
            "uptime_seconds": max(0.0, now - self.began),
            "ready": ready,
            "stopping": False,
            "fault_type": "",
            "http": {"enabled": True, "available": True, "error_type": "", "error_code": ""},
            "refresh_seconds": SAMPLE_SECONDS,
            "requests": requests,
            "timing": timing,
            "queue_depth": 0,
            "oldest_wait_seconds": 0.0,
            "queue": [],
            "queue_display_clipped": False,
            "active_work": active,
            "work": {"by_operation": {}},
            "errors": errors,
            "recent_outcomes": recent,
            "history": history,
            "usage": {},
            "readiness": {"status": readiness.get("status", ""), "checks": readiness.get("checks", {})},
            "readiness_problem": readiness.get("error", {}).get("message", ""),
            **members,
        }
        return snapshot


def page_file(path: str) -> tuple:
    """Return the body and media type of the page file a route names, or an empty pair for another path."""
    name, media_type = PAGE_FILES.get(path, ("", ""))
    if not name:
        nothing = (b"", "")
        return nothing
    page = ((ASSETS / name).read_bytes(), media_type)
    return page
