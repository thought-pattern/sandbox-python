// Shared health page code: polling, formatting and the standard sections every component's page shows.
// It refers to nothing outside this file and health.css, so a service outside this package can serve copies of both
// unchanged at /health-common.js and /health.css. A page script imports from it, mounts the standard sections, and
// renders its own sections from the same snapshot.

const SVG = "http://www.w3.org/2000/svg";
// Samples further apart than this many seconds leave a gap in a chart: the page was not sampling then.
const GAP_SECONDS = 15;
const SCOPES = {
  own_model: "This component's own model calls",
  requested: "Model work this component requested from other components",
};

const SUMMARY = `
<section class="cards" aria-label="Standard metrics">
  <article><h2>Service</h2><p id="std-readiness" class="value">-</p><dl id="std-service"></dl></article>
  <article><h2>Requests</h2><p id="std-received" class="value">-</p><p class="subtle">Requests received</p><dl id="std-requests"></dl></article>
  <article><h2>Tokens</h2><p id="std-calls" class="value">-</p><p id="std-usage-scope" class="subtle"></p><dl id="std-tokens"></dl></article>
  <article><h2>Errors</h2><p id="std-errors" class="value">-</p><p class="subtle">Errors since start</p><dl id="std-error-facts"></dl></article>
</section>`;

const DETAIL = `
<section class="panel"><h2>Recent activity</h2><p class="subtle">Five-second samples for ten minutes; gaps stay visible; counters reset when the component restarts.</p>
  <div class="charts">
    <div><h3>Items waiting</h3><svg id="std-queue-chart" viewBox="0 0 600 150" role="img" aria-label="Items waiting over time"></svg></div>
    <div><h3>Items finished per second</h3><svg id="std-finished-chart" viewBox="0 0 600 150" role="img" aria-label="Items finished per second"></svg></div>
    <div><h3>Tokens per second</h3><svg id="std-token-chart" viewBox="0 0 600 150" role="img" aria-label="Tokens per second"></svg></div>
  </div>
</section>
<section class="tables">
  <article class="panel"><h2>Queue</h2><p id="std-queue-note" class="subtle"></p><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Wait</th><th>Accepted</th><th>IDs</th></tr></thead><tbody id="std-queue-rows"></tbody></table></div></article>
  <article class="panel"><h2>Active work</h2><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Stage</th><th>Elapsed</th><th>In stage</th><th>IDs</th></tr></thead><tbody id="std-active-rows"></tbody></table></div></article>
</section>
<section class="panel"><h2>Recent outcomes</h2><p class="subtle">The latest 32 finished items, newest first.</p><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Outcome</th><th>Wait / run</th><th>Finished</th><th>IDs</th></tr></thead><tbody id="std-outcome-rows"></tbody></table></div></section>
<section class="tables">
  <article class="panel"><h2>Requests by operation</h2><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Received</th><th>Responses</th><th>Outcomes</th></tr></thead><tbody id="std-request-rows"></tbody></table></div></article>
  <article class="panel"><h2>Work handed to other components</h2><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Started</th><th>Finished</th><th>Outcomes</th></tr></thead><tbody id="std-work-rows"></tbody></table></div></article>
</section>
<section class="tables">
  <article class="panel"><h2>Tokens by model</h2><div class="table-scroll"><table><thead><tr><th>Model</th><th>Calls</th><th>Input</th><th>Output</th></tr></thead><tbody id="std-model-rows"></tbody></table></div></article>
  <article class="panel"><h2>Tokens by operation</h2><div class="table-scroll"><table><thead><tr><th>Operation</th><th>Calls</th><th>Input</th><th>Output</th></tr></thead><tbody id="std-token-operation-rows"></tbody></table></div></article>
</section>`;

export const byId = (id) => document.getElementById(id);

export const count = (value) => (Number.isFinite(value) ? value.toLocaleString() : "Not observed");

export const duration = (value) => {
  if (!Number.isFinite(value)) return "Not observed";
  if (value < 60) return value.toFixed(1) + " s";
  if (value < 3600) return Math.floor(value / 60) + " min " + Math.floor(value % 60) + " s";
  return (value / 3600).toFixed(1) + " h";
};

export const time = (value) => (value ? new Date(value).toLocaleString() : "Not observed");

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

export function provider(cache, direction) {
  const reported = (cache || {})[direction + "_reported_calls"] || 0;
  return reported ? count(cache[direction + "_input_tokens"]) + " tokens · " + count(reported) + " reporting calls" : "Not reported";
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

// Fill a <tbody> with one row per value; each column maps a value to text or a node.
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
  for (const value of values) {
    const row = element("tr");
    for (const column of columns) row.append(filled(document.createElement("td"), column(value)));
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
    return { x: sample.elapsed_seconds, y: interval > 0 ? (sample[field] - prior[field]) / interval : null };
  });
  const valid = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const label = document.createElementNS(SVG, "text");
  label.setAttribute("x", "0");
  label.setAttribute("y", "14");
  label.textContent = valid.length ? "Peak " + Math.max(...valid.map((point) => point.y)).toFixed(1) : "No samples yet";
  svg.append(label);
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

function tokenColumns() {
  return [([name]) => name, ([, usage]) => count(usage.calls), ([, usage]) => count(usage.prompt_tokens), ([, usage]) => count(usage.completion_tokens)];
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
  text("scope", "Since " + component + " started · " + time(snapshot.started_at));
  text("observed", "Observed " + time(snapshot.observed_at));
  text("std-readiness", readiness(snapshot));
  facts("std-service", [
    ["Started", time(snapshot.started_at)],
    ["Uptime", duration(snapshot.uptime_seconds)],
    ["Process ID", count(snapshot.process_id)],
    ["Incarnation", element("span", snapshot.incarnation || "Not observed", "mono")],
    ["Fault", snapshot.fault_type || "None"],
    ["Health page listener", listener(snapshot.http)],
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
  const usage = snapshot.usage || {};
  const called = (usage.total_calls || 0) > 0 || Object.keys(usage.by_model || {}).length > 0;
  text("std-calls", called ? count(usage.total_calls) : "No model calls");
  text("std-usage-scope", SCOPES[usage.scope] || "");
  const cache = usage.provider_cache || {};
  facts("std-tokens", [
    ["Tokens in", count(usage.total_prompt_tokens)],
    ["Tokens out", count(usage.total_completion_tokens)],
    ["Provider cache read", provider(cache, "read")],
    ["Provider cache write", provider(cache, "write")],
    ["Provider failures", count(usage.total_provider_failures)],
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
  chart("std-token-chart", history.map((sample) => ({ ...sample, tokens: sample.input_tokens + sample.output_tokens })), "tokens", true);
  const queue = snapshot.queue || [];
  text(
    "std-queue-note",
    count(snapshot.queue_depth) + " waiting · oldest " + duration(snapshot.oldest_wait_seconds || 0) + (snapshot.queue_display_clipped ? " · first " + queue.length + " shown, in order" : "")
  );
  table("std-queue-rows", queue, [(item) => item.operation, (item) => duration(item.wait_seconds), (item) => time(item.accepted_at), (item) => correlation(item.ids)], "Nothing waiting.");
  table(
    "std-active-rows",
    snapshot.active_work || [],
    [(item) => item.operation, (item) => item.stage || "-", (item) => duration(item.elapsed_seconds), (item) => duration(item.stage_seconds), (item) => correlation(item.ids)],
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
  table("std-model-rows", Object.entries(usage.by_model || {}), tokenColumns(), "No model calls.");
  table("std-token-operation-rows", Object.entries(usage.by_operation || {}), tokenColumns(), "No model calls.");
}

// Poll /api/health with one request outstanding and a four-second timeout, paused while the tab is hidden; a failed
// poll marks the page stale, and a new incarnation shows the restart notice.
export function start(component, render) {
  const state = { snapshot: null, incarnation: "", pending: false, timer: null };
  async function load() {
    if (state.pending || document.hidden) return;
    state.pending = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const reply = await fetch("/api/health", { cache: "no-store", signal: controller.signal });
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
      text("connection", "Stale · " + (error.name === "AbortError" ? "timed out" : error.message));
      byId("connection").className = "status stale";
    } finally {
      clearTimeout(timeout);
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
