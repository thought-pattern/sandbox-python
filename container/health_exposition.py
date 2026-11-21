"""Prometheus text exposition of the Workspace's health snapshot, served at /metrics on the health port.

It uses only the standard library and reads only the explicit fields below; content, identifiers of work, paths, recent
observations and arbitrary nested members are never metric labels. Counters cover the running server only. The metric
names are the ones a scraper already reads from this port, so they are part of the port's interface.
"""

from datetime import datetime
from math import isfinite

METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def append_metric(lines: list, name: str, kind: str, description: str, samples: list) -> None:
    """Append one metric family, omitting missing or invalid observations rather than synthesizing zero."""
    rendered = []
    for value, labels in samples:
        if type(value) not in (bool, int, float) or not isfinite(value):
            continue
        escaped = {key: str(label).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') for key, label in labels.items()}
        suffix = ",".join(key + '="' + label + '"' for key, label in sorted(escaped.items()))
        number = str(int(value)) if type(value) is bool else str(value)
        rendered.append(name + "{" + suffix + "} " + number)
    if rendered:
        lines.extend(["# HELP " + name + " " + description, "# TYPE " + name + " " + kind, *rendered])


def append_process_metrics(lines: list, snapshot: dict, labels: dict) -> None:
    """Append the server process's identity, readiness, age, queue and calls in progress."""
    incarnation = snapshot.get("incarnation", "")
    if incarnation:
        append_metric(
            lines, "tapestry_component_info", "gauge", "Current process incarnation.", [(1, {**labels, "incarnation": incarnation})]
        )
    for field, name, description in (
        ("ready", "tapestry_component_ready", "Whether the owner is ready for work."),
        ("stopping", "tapestry_component_stopping", "Whether graceful owner shutdown has begun."),
        ("process_id", "tapestry_component_process_id", "Process identifier in the owner's host namespace."),
        ("uptime_seconds", "tapestry_component_uptime_seconds", "Seconds since the owner started."),
        ("queue_depth", "tapestry_queue_depth", "Waiting items visible to the owning queue."),
        ("oldest_wait_seconds", "tapestry_queue_oldest_wait_seconds", "Age of the oldest item in the owning queue."),
    ):
        append_metric(lines, name, "gauge", description, [(snapshot.get(field), labels)])
    started_at = snapshot.get("started_at", "")
    if type(started_at) is str and started_at:
        try:
            started = datetime.fromisoformat(started_at)
        except ValueError:
            pass
        else:
            if started.tzinfo is not None:
                append_metric(
                    lines,
                    "tapestry_component_start_time_seconds",
                    "gauge",
                    "Owner start time since Unix epoch.",
                    [(started.timestamp(), labels)],
                )
    active = snapshot.get("active_work")
    if type(active) is list:
        append_metric(
            lines,
            "tapestry_active_work",
            "gauge",
            "Owner work items currently in progress, including their executing requests.",
            [(len(active), labels)],
        )


def append_request_metrics(lines: list, snapshot: dict, labels: dict) -> None:
    """Append the calls received and answered, in all and by tool, their outcomes, their elapsed time and the errors."""
    requests = snapshot.get("requests", {})
    for field in ("received", "responses"):
        append_metric(
            lines,
            "tapestry_requests_" + field + "_total",
            "counter",
            "Process request " + field + ".",
            [(requests.get(field), labels)],
        )
    operations = requests.get("by_operation", {})
    for field in ("received", "responses"):
        samples = [(row.get(field), {**labels, "operation": operation}) for operation, row in operations.items()]
        append_metric(
            lines,
            "tapestry_operation_requests_" + field + "_total",
            "counter",
            "Process request " + field + " by operation.",
            samples,
        )
    samples = [
        (count, {**labels, "operation": operation, "outcome": outcome})
        for operation, row in operations.items()
        for outcome, count in row.get("outcomes", {}).items()
    ]
    append_metric(
        lines,
        "tapestry_request_outcomes_total",
        "counter",
        "Process request outcomes, including Regulator verdict-stage ACCEPT and REJECT.",
        samples,
    )
    timing = snapshot.get("timing", {})
    for metric, suffix in (("seconds", "seconds"), ("samples", "count")):
        samples = [
            (timing.get(field + "_" + suffix), {**labels, "phase": phase})
            for phase, field in (
                ("total", "total" if metric == "seconds" else "terminal"),
                ("execution", "execution"),
                ("queue_only", "queue_only"),
            )
        ]
        append_metric(
            lines,
            "tapestry_request_elapsed_" + metric + "_total",
            "counter",
            "Request elapsed " + metric + " by phase; each phase has its own sample denominator.",
            samples,
        )
    errors = snapshot.get("errors", {}).get("by_kind", {})
    append_metric(
        lines,
        "tapestry_errors_total",
        "counter",
        "Process errors by kind.",
        [(count, {**labels, "kind": kind}) for kind, count in errors.items()],
    )


def health_exposition(snapshot: dict) -> str:
    """Format the bounded public process metrics; the caller owns and obtains the snapshot."""
    lines = []
    labels = {"component": snapshot.get("component", "")}
    append_process_metrics(lines, snapshot, labels)
    append_request_metrics(lines, snapshot, labels)
    exposition = "\n".join(lines) + "\n"
    return exposition
