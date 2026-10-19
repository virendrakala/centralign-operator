const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const boundary = source.indexOf("\nconst $ =");
assert.notEqual(boundary, -1, "summary helper stays at the top of app.js");
const moduleStub = { exports: {} };
vm.runInNewContext(source.slice(0, boundary), { module: moduleStub, window: {} });
const { groupTraceEvents } = moduleStub.exports;

test("groups tool actions with observations and keeps selection and recovery meaningful", () => {
  const events = [
    { event_type: "provider_turn", sequence: 1, payload: { turn: 2 } },
    { event_type: "action", sequence: 2, timestamp: "2026-10-19T12:00:00Z", payload: { tool: "read_document", path: "inbox/INV_2026_10_Acme.pdf" } },
    { event_type: "host_run_state", sequence: 3, payload: { form_status: "empty" } },
    { event_type: "observation", sequence: 4, timestamp: "2026-10-19T12:00:03Z", payload: { tool: "read_document", invoice: { vendor: "Acme Analytics", invoice_number: "INV-2026-2116", invoice_date: "2026-10-21", currency: "INR", amount: "68500.00" } } },
    { event_type: "selection_check", sequence: 5, payload: { status: "passed", selected_invoice: "INV-2026-2116", criterion: "latest_by_invoice_date" } },
    { event_type: "rate_limit_recovery", sequence: 6, payload: { action: "retry_wait", provider: "groq", status_code: 429, wait_seconds: 4, retry: 1, retry_limit: 4 } },
    { event_type: "action", sequence: 7, payload: { tool: "browser_click", selector: "#submit_btn" } },
    { event_type: "observation", sequence: 8, payload: { tool: "browser_click", alert: "Manager approval token must be verified before submission." } },
    { event_type: "action", sequence: 9, payload: { tool: "request_human_approval" } },
  ];
  const result = groupTraceEvents(events);
  assert.equal(result.rows.length, 5);
  assert.equal(result.rows[0].title, "Read INV_2026_10_Acme.pdf");
  assert.match(result.rows[0].result, /INV-2026-2116/);
  assert.equal(result.rows[0].duration, "3s");
  assert.match(result.rows[1].title, /Chose INV-2026-2116/);
  assert.equal(result.rows[2].status, "warning");
  assert.equal(result.rows[3].title, "ERP rejected: approval required");
  assert.match(result.rows[3].result, /Agent then requested human approval/);
  assert.equal(result.rateLimitWaits, 1);
});

test("merges host approval and verification records into their tool steps", () => {
  const events = [
    { event_type: "action", sequence: 1, payload: { tool: "request_human_approval" } },
    { event_type: "approval_requested", sequence: 2, payload: { reason: "Exceeds INR 50,000 policy threshold", invoice: { invoice_number: "INV-2026-2116", amount: "68500.00", currency: "INR" } } },
    { event_type: "observation", sequence: 3, payload: { tool: "request_human_approval", message: "Waiting for the human decision." } },
    { event_type: "action", sequence: 4, payload: { tool: "verify_ledger_row" } },
    { event_type: "verification", sequence: 5, payload: { status: "verified", matches: { vendor: true, invoice_number: true, amount: true }, ledger_row_id: 1 } },
    { event_type: "observation", sequence: 6, payload: { tool: "verify_ledger_row", message: "All fields match." } },
  ];
  const result = groupTraceEvents(events);
  assert.equal(result.rows.length, 2);
  assert.match(result.rows[0].title, /Approval requested: INR 68,500/);
  assert.equal(result.rows[0].result, "Exceeds INR 50,000 policy threshold");
  assert.equal(result.rows[1].title, "Verified: all 3 fields match ledger row #1");
});

test("does not duplicate a host refusal in the trace summary", () => {
  const result = groupTraceEvents([
    { event_type: "task_intent", sequence: 1, payload: { intent: "unsupported", clarifying_question: "Would you like me to record it only?" } },
    { event_type: "final_result", sequence: 2, payload: { status: "needs_input", summary: "I can record it, but cannot pay it. Would you like me to proceed?" } },
  ]);
  assert.equal(result.rows.length, 1);
  assert.equal(result.rows[0].title, "Request needs a capability clarification");
});
