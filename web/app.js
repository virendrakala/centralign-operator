const $ = (selector, root = document) => root.querySelector(selector);
const state = { runId: null, pollTimer: null, finished: true, lastFrame: null, approvalToken: null };

async function requestJson(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || data.error || `Request failed (${response.status})`);
  return data;
}
function node(tag, className, text) { const item = document.createElement(tag); if (className) item.className = className; if (text !== undefined) item.textContent = text; return item; }
function timestamp(value) { const date = new Date(value); return Number.isNaN(date.valueOf()) ? "" : new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(date); }
function friendlyStatus(status) { return ({ idle: "Ready", running: "Running", awaiting_approval: "Approval needed", complete: "Verified", needs_input: "Needs attention", needs_review: "Needs attention", incomplete: "Incomplete" })[status] || String(status || "Ready").replaceAll("_", " "); }
function setStatus(status, detail = "") {
  const pill = $("#runIndicator"); pill.dataset.state = status || "idle"; pill.className = "status-pill";
  if (status === "complete") pill.classList.add("status-success");
  else if (status === "awaiting_approval") pill.classList.add("status-warning");
  else if (["needs_input", "needs_review", "incomplete"].includes(status)) pill.classList.add("status-error");
  else pill.classList.add("status-neutral");
  $("#runIndicatorText").textContent = detail ? `${friendlyStatus(status)} · ${detail}` : friendlyStatus(status);
}
function setRunButton(running) { const button = $("#runButton"); button.disabled = running; button.replaceChildren(document.createTextNode(running ? "Running…" : "Run Operator ")); if (!running) { const arrow = node("span", "", "→"); arrow.setAttribute("aria-hidden", "true"); button.append(arrow); } }
function eventTitle(event) {
  const payload = event.payload || {};
  if (event.event_type === "credentials_supplied") return "Credentials supplied by host";
  const titles = { thought: "Thought", action: payload.tool || "Action", observation: payload.tool ? `Observation · ${payload.tool}` : "DOM observation", loop_guard_warning: "Repeated action warning", self_correction: "Self-correction", approval_requested: "Human approval requested", approval_decision: "Approval decision", planner: `Planner · ${payload.mode || "LLM"}`, planner_fallback: "Planner fallback", provider_turn: `Provider answer · turn ${payload.turn || "?"}`, rate_limit_recovery: "[RATE_LIMIT_RECOVERY] Recovery event", verification: "Independent ledger verification", provenance_attached: "Source provenance attached by host", tool_retry_budget: "Tool retry budget", task_finished: "Run finished", run_error: "Run error" };
  return titles[event.event_type] || String(event.event_type || "Event").replaceAll("_", " ");
}
function eventSummary(event) {
  const p = event.payload || {};
  if (event.event_type === "credentials_supplied") return p.message || "ERP login credentials are supplied by the host and never exposed to the model.";
  if (event.event_type === "provenance_attached") return p.message || `Host attached source provenance for ${p.invoice_number || "the selected invoice"}.`;
  if (event.event_type === "tool_retry_budget") return `${p.consecutive_identical_failures}/${p.limit} consecutive identical failures for ${p.tool}. Last error: ${p.error}`;
  if (event.event_type === "thought" || event.event_type === "self_correction") return p.text || "";
  if (event.event_type === "action") {
    if (p.tool === "list_files") return "List the available invoice sources and procurement policy.";
    if (p.tool === "read_document") return `Read ${p.path} as untrusted invoice data.`;
    if (p.tool === "read_policy") return "Read the procurement rules and approval threshold.";
    if (p.tool === "browser_open") return `Open the local ERP route ${p.path}.`;
    if (p.tool === "erp_login") return "Sign in using credentials supplied by the host.";
    if (p.tool === "browser_snapshot") return "Inspect the visible ERP page and its alerts.";
    if (p.tool === "browser_fill") return `Fill ${p.selector} · ${p.value === "[redacted]" ? "sensitive value redacted" : p.value || ""}`;
    if (p.tool === "browser_fill_form") return `Fill ${p.fields?.length || 0} validated form fields in one batch.`;
    if (p.tool === "browser_action_sequence") return `Run ${p.actions?.length || 0} host-validated browser actions in sequence.`;
    if (p.tool === "browser_click") return `Click ${p.selector} with Playwright locator.click.`;
    if (p.tool === "browser_check") return `Check ${p.selector}.`;
    if (p.tool === "verify_ledger_row") return "Compare the live ERP ledger row with the selected source.";
    if (p.tool === "request_human_approval") return "Pause for the human approval decision.";
    if (p.tool === "finish") return `Finish with status ${p.status || "unknown"}.`;
  }
  if (event.event_type === "observation") {
    if (p.message) return p.warning ? `${p.message} ${p.warning}` : p.message;
    if (p.alert) return `ERP alert: ${p.alert}`;
    if (p.session_required) return p.session_required;
    if (p.warning) return p.warning;
    if (p.fields) return `${p.fields.invoice_number} · ${p.fields.vendor} · ${p.fields.amount} · due ${p.fields.due_date}`;
    if (p.files) return `${p.files.length} source and policy files available.`;
    if (p.dom_tree) return p.summary || p.dom_tree.slice(0, 240);
    if (p.error) return p.error;
  }
  if (event.event_type === "loop_guard_warning") return p.warning || `Repeated ${p.tool || "action"} without a page-state change.`;
  if (event.event_type === "approval_requested") return p.reason || "A human decision is required.";
  if (event.event_type === "approval_decision") return p.token_minted ? `Approved. Runtime token ${p.token_preview} was minted and delivered to the worker.` : "Approval was declined or timed out.";
  if (event.event_type === "planner") return `${p.mode === "replay" ? "Replay" : "Live provider"} · ${p.provider} · ${p.model}`;
  if (event.event_type === "provider_turn") return `${p.provider} · ${p.model}`;
  if (event.event_type === "rate_limit_recovery") {
    if (p.action === "retry_wait") return `${p.marker || "[RATE_LIMIT_RECOVERY]"} ${p.provider} returned HTTP ${p.status_code}; waiting ${p.wait_seconds}s before retry ${p.retry}/${p.retry_limit}.`;
    if (p.action === "retries_exhausted") return `${p.marker || "[RATE_LIMIT_RECOVERY]"} ${p.provider} still returned HTTP ${p.status_code} after ${p.retry_count} retries.${p.response_body ? ` ${p.response_body}` : ""}`;
    if (p.action === "provider_fallback") return `${p.marker || "[RATE_LIMIT_RECOVERY]"} ${p.previous_provider} retries failed; switching to live provider ${p.provider}.`;
  }
  if (event.event_type === "verification") return p.status === "verified" ? `All ${Object.keys(p.matches || {}).length} fields match. Source ${p.source_file}.` : `Ledger verification status: ${p.status}.`;
  if (event.event_type === "task_finished") return p.summary || p.status || "";
  if (event.event_type === "run_error") return p.error || "The run stopped safely.";
  return "";
}
function renderTrace(run) {
  const list = $("#traceList"); list.replaceChildren();
  const events = (run.events || []).filter((event) => event.event_type !== "browser_frame");
  $("#traceEmpty").hidden = events.length > 0;
  for (const event of events) {
    const item = node("li", "trace-event"); item.dataset.type = event.event_type;
    const mark = node("span", "trace-mark", ({ thought: "T", action: "A", observation: "D", self_correction: "↻", verification: "✓", approval_requested: "!", run_error: "×" })[event.event_type] || "·"); mark.setAttribute("aria-hidden", "true");
    const content = node("div", "trace-content");
    const top = node("div", "trace-top"); top.append(node("h3", "trace-title", eventTitle(event)), node("time", "trace-time", timestamp(event.timestamp))); content.append(top);
    const summary = eventSummary(event); if (summary) content.append(node("p", "trace-summary", summary));
    if (event.payload && ["action", "observation", "loop_guard_warning", "verification", "provenance_attached", "tool_retry_budget", "planner", "planner_fallback", "provider_turn", "rate_limit_recovery", "approval_requested", "approval_decision", "run_error"].includes(event.event_type)) {
      const details = node("details", "trace-more"); const toggle = node("summary", "", "View details"); const pre = node("pre", "trace-json", JSON.stringify(event.payload, null, 2)); details.append(toggle, pre); content.append(details);
    }
    item.append(mark, content); list.append(item);
  }
}
function renderApproval(run) {
  const request = [...(run.events || [])].reverse().find((event) => event.event_type === "approval_requested");
  const visible = Boolean(run.awaiting_approval && request);
  $("#approvalBanner").hidden = !visible;
  if (!visible) return;
  const invoice = request.payload.invoice || {};
  $("#approvalTitle").textContent = `Approve ${invoice.invoice_number || "invoice"} before ERP entry`;
  const facts = $("#approvalFacts"); facts.replaceChildren();
  for (const [name, value, mono] of [["Vendor", invoice.vendor, false], ["Invoice #", invoice.invoice_number, true], ["Amount", formatAmount(invoice.currency, invoice.amount), true], ["Due date", invoice.due_date, true], ["Source PDF", request.payload.source_file, false]]) {
    const cell = node("div", "approval-fact"); cell.append(node("span", "", name), node("strong", mono ? "mono" : "", value || "—")); facts.append(cell);
  }
  $("#approvalDetail").textContent = request.payload.reason || (request.payload.policy_reasons || []).join(" ");
  $("#approveButton").disabled = false; $("#rejectButton").disabled = false;
}
function shortHash(value) { return value && value.length > 42 ? `${value.slice(0, 20)}…${value.slice(-12)}` : value || "—"; }
function formatAmount(currency, amount) {
  const numeric = Number(amount);
  const formatted = Number.isFinite(numeric) ? numeric.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : amount || "—";
  return `${currency || ""} ${formatted}`.trim();
}
function renderEvidence(run) {
  const proof = [...(run.events || [])].reverse().find((event) => event.event_type === "verification")?.payload;
  const badge = $("#evidenceBadge"); badge.className = "semantic-badge";
  if (!proof) {
    badge.textContent = run.awaiting_approval ? "Awaiting approval" : run.status === "running" ? "In progress" : "Not verified";
    badge.classList.add(run.awaiting_approval ? "badge-warning" : "badge-neutral");
    $("#evidenceSource").textContent = "—"; $("#evidenceHash").textContent = "—";
    const body = $("#evidenceBody"); body.replaceChildren(); const row = node("tr"); const cell = node("td", "empty-cell", "No ledger read-back has been captured."); cell.colSpan = 4; row.append(cell); body.append(row);
    $("#ledgerCrop").hidden = true; $("#cropPlaceholder").hidden = false; $("#cropStatus").textContent = "No capture"; return;
  }
  const verified = proof.status === "verified";
  badge.textContent = verified ? "Verified" : "Needs review"; badge.classList.add(verified ? "badge-success" : "badge-error");
  $("#evidenceSource").textContent = proof.source_file || "—"; $("#evidenceHash").textContent = shortHash(proof.source_sha256); $("#evidenceHash").title = proof.source_sha256 || "";
  const body = $("#evidenceBody"); body.replaceChildren();
  const fields = proof.expected || {}; const actual = proof.actual || {}; const matches = proof.matches || {};
  for (const key of ["vendor", "invoice_number", "invoice_date", "due_date", "amount", "currency"]) {
    const row = node("tr"); row.append(node("td", "", key.replaceAll("_", " ")));
    const sourceValue = key === "vendor" ? `${proof.source_vendor || fields.vendor} → ${proof.canonical_vendor || fields.vendor}` : fields[key];
    row.append(node("td", "", sourceValue || "—"), node("td", "", actual[key] || "—"));
    const statusCell = node("td"); statusCell.append(node("span", `proof-badge${matches[key] ? "" : " mismatch"}`, matches[key] ? "Match" : "Mismatch")); row.append(statusCell); body.append(row);
  }
  if (proof.screenshot_url) {
    const image = $("#ledgerCrop"); image.src = `${proof.screenshot_url}?v=${Date.now()}`; image.hidden = false;
    $("#cropPlaceholder").hidden = true; $("#cropStatus").textContent = "Captured";
  }
}
function renderFrame(run) {
  const frame = [...(run.events || [])].reverse().find((event) => event.event_type === "browser_frame" && event.payload?.kind !== "ledger_crop");
  if (!frame) return;
  const payload = frame.payload; state.lastFrame = payload.screenshot_index;
  const image = $("#browserFrame"); image.src = `${payload.screenshot_url}?v=${Date.now()}`; image.hidden = false; $("#browserPlaceholder").hidden = true;
  $("#browserUrl").textContent = payload.url || "/erp/login";
  $("#frameAction").textContent = `Frame ${payload.screenshot_index} · ${payload.action || "browser action"}${payload.highlight ? " · element highlighted" : ""}`;
  $("#frameDimensions").textContent = payload.viewport || "16:10 · 1280 × 800";
  $("#frameBadge").textContent = payload.kind === "ledger_crop" ? "Ledger proof captured" : `Frame ${payload.screenshot_index}`;
}
function renderHistory(runs) {
  const list = $("#runHistory"); list.replaceChildren();
  if (!runs.length) { list.append(node("p", "muted-copy", "No runs yet.")); return; }
  for (const run of runs.slice(0, 8)) {
    const button = node("button", "history-row"); button.type = "button"; button.setAttribute("aria-current", String(run.id === state.runId));
    button.append(node("span", "history-task", run.task || "Invoice task"), node("span", "history-state", friendlyStatus(run.status)), node("span", "history-time", `${run.id} · ${timestamp(run.started_at)}`));
    button.addEventListener("click", () => loadRun(run.id)); list.append(button);
  }
}
function renderRun(run) {
  state.runId = run.id; $("#traceRunId").textContent = run.id || "—";
  renderTrace(run); renderApproval(run); renderEvidence(run); renderFrame(run);
  if (run.awaiting_approval) { state.finished = false; setStatus("awaiting_approval"); }
  else if (run.status === "running") { state.finished = false; setStatus("running"); }
  else { state.finished = true; setRunButton(false); setStatus(run.status || "needs_review"); if (state.pollTimer) clearTimeout(state.pollTimer); refreshState(false); }
}
async function pollRun(runId) {
  try { const run = await requestJson(`/api/runs/${encodeURIComponent(runId)}`); if (state.runId !== runId) return; renderRun(run); if (!state.finished && state.runId === runId) state.pollTimer = setTimeout(() => pollRun(runId), 600); }
  catch (error) { state.finished = true; setRunButton(false); setStatus("needs_review", error.message); }
}
async function loadRun(runId) {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  try { const run = await requestJson(`/api/runs/${encodeURIComponent(runId)}`); state.runId = runId; state.finished = !["running", "awaiting_approval"].includes(run.status) && !run.awaiting_approval; renderRun(run); if (!state.finished) pollRun(runId); }
  catch (error) { setStatus("needs_review", error.message); }
}
async function refreshState(showError = true) {
  try {
    const data = await requestJson("/api/state"); renderHistory(data.runs || []);
    const labels = { offline: "Offline replay", "offline-replay": "FakeLLM · offline replay", unconfigured: "Live API key required", openai: `OpenAI · ${data.model}`, openai_responses: `OpenAI · ${data.model}`, openai_chat: `OpenAI · ${data.model}`, gemini: `Gemini · ${data.model}`, groq: `Groq · ${data.model}`, openrouter: `OpenRouter · ${data.model}` };
    $("#providerBadge").textContent = labels[data.provider] || "Provider unavailable";
  } catch (error) { if (showError) setStatus("needs_review", error.message); }
}
async function decideApproval(approved) {
  if (!state.runId) return;
  $("#approveButton").disabled = true; $("#rejectButton").disabled = true;
  try {
    const result = await requestJson(`/api/runs/${encodeURIComponent(state.runId)}/approval`, { method: "POST", body: JSON.stringify({ approved }) });
    state.approvalToken = result.approval_token || null;
    if (state.approvalToken) { $("#approvalTokenMessage").hidden = false; $("#approvalTokenMessage").textContent = `Cryptographic runtime token minted: ${state.approvalToken}. The worker will enter it into the ERP form.`; }
    $("#approvalBanner").hidden = true; setStatus("running"); pollRun(state.runId);
  } catch (error) { $("#approveButton").disabled = false; $("#rejectButton").disabled = false; setStatus("needs_review", error.message); }
}
$("#taskForm").addEventListener("submit", async (event) => {
  event.preventDefault(); const task = $("#taskInput").value.trim(); if (!task) return;
  if (state.pollTimer) clearTimeout(state.pollTimer); state.finished = false; state.runId = null; state.lastFrame = null; state.approvalToken = null;
  $("#traceRunId").textContent = "Starting…"; $("#traceList").replaceChildren(); $("#traceEmpty").hidden = false; $("#approvalBanner").hidden = true;
  $("#browserFrame").hidden = true; $("#browserPlaceholder").hidden = false; $("#browserUrl").textContent = "/erp/login"; $("#frameBadge").textContent = "Starting browser";
  $("#ledgerCrop").hidden = true; $("#cropPlaceholder").hidden = false; $("#cropStatus").textContent = "No capture";
  $("#evidenceBadge").textContent = "Not verified"; $("#evidenceBadge").className = "semantic-badge badge-neutral";
  $("#evidenceSource").textContent = "—"; $("#evidenceHash").textContent = "—";
  const empty = node("tr"); const cell = node("td", "empty-cell", "No ledger read-back has been captured."); cell.colSpan = 4; empty.append(cell); $("#evidenceBody").replaceChildren(empty);
  setStatus("running"); setRunButton(true);
  try { const result = await requestJson("/api/runs", { method: "POST", body: JSON.stringify({ task, approval_mode: "policy_driven" }) }); state.runId = result.run_id; pollRun(state.runId); }
  catch (error) { state.finished = true; setRunButton(false); setStatus("needs_review", error.message); }
});
const SCENARIOS = {
  acme: "Find the latest invoice from Acme, check policy compliance, enter it into the ERP portal, and verify it in the ledger",
  northstar: "Find the latest invoice from Northstar, check policy compliance, enter it into the ERP portal, and verify it in the ledger",
  vertex: "Find the latest invoice from Vertex, check policy compliance, enter it into the ERP portal, and verify it in the ledger",
};
for (const button of document.querySelectorAll(".scenario-chip")) button.addEventListener("click", () => { $("#taskInput").value = SCENARIOS[button.dataset.scenario] || SCENARIOS.acme; document.querySelectorAll(".scenario-chip").forEach((chip) => chip.setAttribute("aria-pressed", String(chip === button))); $("#taskInput").focus(); });
$("#approveButton").addEventListener("click", () => decideApproval(true)); $("#rejectButton").addEventListener("click", () => decideApproval(false));
$("#resetDemoButton").addEventListener("click", async () => {
  if (!window.confirm("Clear the demo ledger and run history? Invoice files and policy will stay in place.")) return;
  const button = $("#resetDemoButton"); button.disabled = true;
  if (state.pollTimer) clearTimeout(state.pollTimer);
  try {
    await requestJson("/api/demo/reset", { method: "POST" });
    window.location.reload();
  } catch (error) {
    button.disabled = false;
    setStatus("needs_review", error.message);
  }
});
document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") $("#taskForm").requestSubmit(); });
setStatus("idle"); refreshState();
