// Shared health page code: polling, formatting and the standard sections every component's page shows.
// It refers to nothing outside this file and health.css, so a service outside this package can serve copies of both
// unchanged at /health-common.js and /health.css. A page script imports from it, mounts the standard sections, and
// renders its own sections from the same snapshot.

const SVG = "http://www.w3.org/2000/svg";
// Samples further apart than this many seconds leave a gap in a chart: the page was not sampling then.
const GAP_SECONDS = 15;
const SUMMARY = `
<dl class="summary-values">
  <div><dt>Service</dt><dd id="std-readiness">-</dd></div>
  <div><dt>Requests</dt><dd id="std-received">-</dd></div>
  <div><dt>Errors</dt><dd id="std-errors">-</dd></div>
</dl>
<details><summary>Service diagnostics</summary>
  <div class="summary-details">
    <section><h3>Service</h3><dl id="std-service"></dl></section>
    <section><h3>Requests</h3><dl id="std-requests"></dl></section>
    <section><h3>Errors</h3><dl id="std-error-facts"></dl></section>
  </div>
</details>`;

const DETAIL = `
<section class="panel"><h2>Recent Activity</h2><p class="subtle">Five-second samples for ten minutes; gaps stay visible; counters reset when the component restarts.</p>
  <div class="charts">
    <div><h3>Items Waiting</h3><div class="chart-peak"></div><svg id="std-queue-chart" viewBox="0 0 600 150" role="img" aria-label="Items waiting over time"></svg><div class="chart-axis"></div></div>
    <div><h3>Items Finished per Second</h3><div class="chart-peak"></div><svg id="std-finished-chart" viewBox="0 0 600 150" role="img" aria-label="Items finished per second"></svg><div class="chart-axis"></div></div>
  </div>
</section>
<section class="tables">
  <article class="panel"><h2>Queue</h2><p id="std-queue-note" class="subtle"></p><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Wait</th><th>Accepted</th><th>IDs</th></tr></thead><tbody id="std-queue-rows"></tbody></table></div></article>
  <article class="panel"><h2>Active Work</h2><div class="table-scroll"><table><thead><tr><th>Work</th><th>Stage</th><th>Elapsed</th><th>In stage</th><th>IDs</th></tr></thead><tbody id="std-active-rows"></tbody></table></div></article>
</section>
<section class="panel"><h2>Recent Outcomes</h2><p class="subtle">The latest 32 finished items, newest first.</p><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Outcome</th><th>Wait / run</th><th>Finished</th><th>IDs</th></tr></thead><tbody id="std-outcome-rows"></tbody></table></div></section>
<section class="tables operation-tables">
  <article class="panel"><h2>Requests by Operation</h2><div class="table-scroll"><table class="operation-table"><thead><tr><th>Operation</th><th>Received</th><th>Responses</th><th>Outcomes</th></tr></thead><tbody id="std-request-rows"></tbody></table></div></article>
  <article class="panel"><h2>Work Handed to Other Components</h2><div class="table-scroll"><table class="operation-table"><thead><tr><th>Operation</th><th>Started</th><th>Finished</th><th>Outcomes</th></tr></thead><tbody id="std-work-rows"></tbody></table></div></article>
</section>`;

export const byId = (id) => document.getElementById(id);

export const count = (value) => (Number.isFinite(value) ? value.toLocaleString() : "Not observed");

export const duration = (value) => {
  if (!Number.isFinite(value)) return "Not observed";
  if (value < 60) return value.toFixed(1) + " s";
  if (value < 3600) return Math.floor(value / 60) + " min " + Math.floor(value % 60) + " s";
  return (value / 3600).toFixed(1) + " h";
};

const utcDateTime = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC", year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
});
export function time(value) {
  const date = new Date(value);
  return value && Number.isFinite(date.getTime()) ? utcDateTime.format(date) + " UTC" : "Not observed";
}

export const yesNo = (value) => (value ? "Yes" : "No");

export function text(id, value) {
  const node = byId(id);
  if (node) node.textContent = value;
}

export function element(name, value = "", className = "") {
  const node = document.createElement(name);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

// "NAME: n · NAME: n" for a mapping of names to counts.
export function tally(values, empty = "None") {
  const entries = Object.entries(values || {});
  return entries.length ? entries.map(([name, total]) => name + ": " + count(total)).join(" · ") : empty;
}

// An item's identifiers, collapsed by default.
export function correlation(ids) {
  const entries = Object.entries(ids || {}).filter(([, value]) => value);
  if (!entries.length) return element("span", "None", "subtle");
  const details = document.createElement("details");
  details.append(element("summary", entries.length === 1 ? "1 ID" : entries.length + " IDs"));
  for (const [name, value] of entries) details.append(element("p", name.replace(/_/g, " ") + ": " + value, "mono"));
  return details;
}

function filled(node, value) {
  if (value instanceof Node) node.append(value);
  else node.textContent = value;
  return node;
}

// Fill a <dl> with [label, value] pairs; a value is text or a node.
export function facts(id, pairs) {
  const list = byId(id);
  if (!list) return;
  list.replaceChildren();
  for (const [label, value] of pairs) list.append(element("dt", label), filled(document.createElement("dd"), value));
}

function activeWorkLabel(item) {
  const words = item.operation.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/[_-]+/g, " ");
  const activity = words.charAt(0).toUpperCase() + words.slice(1);
  if (!item.work_name && !item.ids?.task_id) return activity;
  const label = element("div", item.work_name || "Task name unavailable", "active-work-name");
  label.append(element("small", activity, "work-activity"));
  return label;
}

// Fill the table with one row per item.
export function table(id, values, columns, empty) {
  const body = byId(id);
  if (!body) return;
  body.replaceChildren();
  if (!values.length) {
    const cell = element("td", empty, "subtle");
    cell.colSpan = columns.length;
    const row = element("tr");
    row.append(cell);
    body.append(row);
    return;
  }
  const headings = body.closest("table")?.tHead?.rows[0]?.cells;
  for (const value of values) {
    const row = element("tr");
    columns.forEach((column, index) => {
      const cell = filled(document.createElement("td"), column(value));
      if (index && headings?.[index]) cell.dataset.label = headings[index].textContent;
      row.append(cell);
    });
    body.append(row);
  }
}

// A line chart of one sample field, or of its rate per second; samples further apart than GAP_SECONDS break the line.
export function chart(id, samples, field, rate = false) {
  const svg = byId(id);
  if (!svg) return;
  svg.replaceChildren();
  const points = samples.map((sample, index) => {
    if (!rate) return { x: sample.elapsed_seconds, y: sample[field] };
    const prior = samples[index - 1];
    const interval = prior ? sample.elapsed_seconds - prior.elapsed_seconds : 0;
    return { x: sample.elapsed_seconds, y: interval > 0 && interval <= GAP_SECONDS ? (sample[field] - prior[field]) / interval : null };
  });
  const valid = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const label = svg.previousElementSibling;
  const axis = svg.nextElementSibling;
  axis.replaceChildren();
  label.textContent = valid.length ? "Peak " + Math.max(...valid.map((point) => point.y)).toFixed(1) : "No samples yet";
  const subject = svg.parentElement.querySelector("h3").textContent;
  svg.setAttribute("aria-label", subject + " · " + label.textContent);
  if (samples.length) {
    const startAt = samples[0].observed_at;
    const endAt = samples[samples.length - 1].observed_at;
    const beginning = Date.parse(startAt);
    const end = Date.parse(endAt);
    const fractions = beginning === end ? [0] : [0, 0.5, 1];
    const crossesDate = startAt.slice(0, 10) !== endAt.slice(0, 10);
    for (const fraction of fractions) {
      const iso = new Date(beginning + (end - beginning) * fraction).toISOString();
      const x = 20 + 570 * fraction;
      const tick = document.createElementNS(SVG, "path");
      tick.setAttribute("d", "M " + x + " 134 v5");
      tick.setAttribute("stroke", "currentColor");
      const stamp = element("time", "", "time-tick");
      stamp.dateTime = iso;
      stamp.textContent = (crossesDate && fraction !== 0.5 ? iso.slice(5, 10) + " " : "") +
        iso.slice(11, 19) + (fraction === fractions[fractions.length - 1] ? " UTC" : "");
      svg.append(tick);
      axis.append(stamp);
    }
    svg.setAttribute("aria-label", subject + " · " + label.textContent + " · " + time(startAt) + " to " + time(endAt));
  }
  if (!valid.length) return;
  const first = samples[0].elapsed_seconds;
  const last = samples[samples.length - 1].elapsed_seconds;
  const maximum = Math.max(1, ...valid.map((point) => point.y));
  let segment = [];
  const flush = () => {
    if (!segment.length) return;
    const line = document.createElementNS(SVG, "polyline");
    line.setAttribute("points", segment.join(" "));
    svg.append(line);
    segment = [];
  };
  for (let index = 0; index < points.length; index++) {
    const point = points[index];
    const prior = points[index - 1];
    if (!Number.isFinite(point.y) || (prior && point.x - prior.x > GAP_SECONDS)) flush();
    if (Number.isFinite(point.y)) segment.push(20 + (570 * (point.x - first)) / Math.max(1, last - first) + "," + (130 - (100 * point.y) / maximum));
  }
  flush();
}

// The local inference cache figures a component keeps: its state, reads and hit rate, writes and size.
export function cacheFacts(id, cache) {
  const figures = cache || {};
  facts(id, [
    ["State", figures.enabled ? "Enabled" : "Disabled"],
    ["Reads / hit rate", figures.reads ? count(figures.reads) + " · " + ((100 * figures.lifetime_hits) / figures.reads).toFixed(1) + "% hits" : "No reads yet"],
    ["Hits / misses", count(figures.lifetime_hits) + " / " + count(figures.lifetime_misses)],
    ["Writes / replacements", count(figures.writes) + " / " + count(figures.replacements)],
    ["Entries", count(figures.size) + " of " + count(figures.max_entries)],
    ["Payload", count(figures.payload_bytes) + " of " + count(figures.max_payload_bytes) + " bytes"],
  ]);
}

export function readiness(snapshot) {
  if (snapshot.stopping) return "Stopping";
  if (snapshot.fault_type) return "Faulted";
  return snapshot.ready ? "Ready" : "Starting";
}

function listener(http) {
  if (!http || !http.enabled) return "Disabled";
  if (http.available) return http.url || "Serving";
  return "Not serving · " + ([http.error_type, http.error_code].filter(Boolean).join(" ") || "starting");
}

function average(seconds, samples, noun) {
  return samples ? duration(seconds) + " · " + count(samples) + " " + noun : "No samples yet";
}

function operationColumns(first, second) {
  return [([operation]) => operation, ([, row]) => count(row[first]), ([, row]) => count(row[second]), ([, row]) => tally(row.outcomes)];
}

// Build the standard sections: `summary` holds the cards, `detail` the charts and tables.
export function mountStandard(summary, detail) {
  summary.innerHTML = SUMMARY;
  detail.innerHTML = DETAIL;
}

export function renderStandard(component, snapshot) {
  text("scope", readiness(snapshot) + " · Observed " + time(snapshot.observed_at));
  text("observed", "Observed " + time(snapshot.observed_at));
  text("std-readiness", readiness(snapshot));
  facts("std-service", [
    ["Started", time(snapshot.started_at)],
    ["Uptime", duration(snapshot.uptime_seconds)],
    ["Process ID", count(snapshot.process_id)],
    ["Incarnation", element("span", snapshot.incarnation || "Not observed", "mono")],
    ["Fault", snapshot.fault_type || "None"],
    ["Health page listener", listener(snapshot.http)],
    ["gRPC listener", listener(snapshot.grpc)],
  ]);
  const requests = snapshot.requests || {};
  const timing = snapshot.timing || {};
  text("std-received", count(requests.received));
  facts("std-requests", [
    ["Responses sent", count(requests.responses)],
    ["Last request received", time(requests.last_received_at)],
    ["Last response sent", time(requests.last_response_at)],
    ["Average time to answer, waiting included", average(timing.average_total_seconds, timing.terminal_count, "answered")],
    ["Average running time", average(timing.average_execution_seconds, timing.execution_count, "run")],
    ["Answered without running", count(timing.queue_only_count)],
  ]);
  const errors = snapshot.errors || {};
  const last = errors.last || {};
  text("std-errors", count(Object.values(errors.by_kind || {}).reduce((sum, value) => sum + value, 0)));
  facts("std-error-facts", [
    ["Last error", last.kind ? [last.kind, last.operation, time(last.observed_at)].filter(Boolean).join(" · ") : "None"],
    ["By kind", tally(errors.by_kind)],
  ]);
  const history = snapshot.history || [];
  chart("std-queue-chart", history, "queue_depth");
  chart("std-finished-chart", history, "finished", true);
  const queue = snapshot.queue || [];
  text(
    "std-queue-note",
    count(snapshot.queue_depth) + " waiting · oldest " + duration(snapshot.oldest_wait_seconds || 0) + (snapshot.queue_display_clipped ? " · first " + queue.length + " shown, in order" : "")
  );
  table("std-queue-rows", queue, [(item) => item.operation, (item) => duration(item.wait_seconds), (item) => time(item.accepted_at), (item) => correlation(item.ids)], "Nothing waiting.");
  table(
    "std-active-rows",
    snapshot.active_work || [],
    [activeWorkLabel, (item) => item.stage || "-", (item) => duration(item.elapsed_seconds), (item) => duration(item.stage_seconds), (item) => correlation(item.ids)],
    "Nothing in progress."
  );
  table(
    "std-outcome-rows",
    snapshot.recent_outcomes || [],
    [
      (item) => item.operation,
      (item) => item.outcome + (item.error_kind && item.error_kind !== item.outcome ? " · " + item.error_kind : ""),
      (item) => duration(item.queue_seconds) + " / " + (item.executed ? duration(item.execution_seconds) : "Not run"),
      (item) => time(item.finished_at),
      (item) => correlation(item.ids),
    ],
    "Nothing finished yet."
  );
  table("std-request-rows", Object.entries(requests.by_operation || {}), operationColumns("received", "responses"), "No requests yet.");
  table("std-work-rows", Object.entries((snapshot.work || {}).by_operation || {}), operationColumns("started", "finished"), "No work handed on.");
}

// Poll /api/health with one request outstanding, paused while the tab is hidden; a failed poll marks the page stale,
// and a new incarnation shows the restart notice.
export function start(component, render) {
  const state = { snapshot: null, incarnation: "", pending: false, timer: null };
  async function load() {
    if (state.pending || document.hidden) return;
    state.pending = true;
    try {
      const reply = await fetch("/api/health", { cache: "no-store" });
      if (!reply.ok) throw new Error("HTTP " + reply.status);
      const snapshot = await reply.json();
      if (state.incarnation && state.incarnation !== snapshot.incarnation) {
        byId("notice").hidden = false;
        text("notice", component + " restarted. Its counters now cover the new process.");
      }
      render(snapshot);
      state.incarnation = snapshot.incarnation;
      state.snapshot = snapshot;
      text("connection", "Live");
      byId("connection").className = "status live";
    } catch (error) {
      text("connection", "Stale · " + error.message);
      byId("connection").className = "status stale";
    } finally {
      state.pending = false;
      clearTimeout(state.timer);
      if (!document.hidden) state.timer = setTimeout(load, 1000 * (state.snapshot?.refresh_seconds || 5));
    }
  }
  document.addEventListener("visibilitychange", () => {
    clearTimeout(state.timer);
    if (!document.hidden) load();
  });
  load();
}
