import {byId, count, yesNo, text, facts, mountStandard, renderStandard, start} from "/health-common.js";

mountStandard(byId("standard-summary"), byId("standard-detail"));

function tools(found) {
  const entries = Object.entries(found || {});
  return entries.length ? entries.map(([name, present]) => name + ": " + (present ? "found" : "missing")).join(" · ") : "Not observed";
}

function render(snapshot) {
  renderStandard("Workspace", snapshot);
  const readiness = snapshot.readiness || {};
  const checks = readiness.checks || {};
  facts("readiness-facts", [
    ["Status", readiness.status === "healthy" ? "Healthy" : "Unhealthy"],
    ["Workspace present", yesNo(checks.workspaceExists)],
    ["Workspace writable", yesNo(checks.workspaceWritable)],
    ["Free space", count(checks.workspaceFreeBytes) + " bytes"],
    ["Tools on the path", tools(checks.tools)],
    ["Problem", snapshot.readiness_problem || "None"],
  ]);
  facts("workspace-facts", [
    ["Egress policy", snapshot.egress_policy || "Not observed"],
    ["Egress enforcement", snapshot.egress_enforcement || "Not observed"],
    ["Image", snapshot.image || "Not supplied"],
    ["Tool calls at once", count(snapshot.tool_call_limit)],
    ["Tool calls in progress", count((snapshot.active_work || []).length)],
  ]);
  text("std-usage-scope", "The Workspace makes no model calls");
}

start("Workspace", render);
