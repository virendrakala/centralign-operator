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
const { groupTraceEvents, progressSteps } = moduleStub.exports;

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
  assert.equal(result.rows[2].title, "ERP rejected: approval required");
  assert.match(result.rows[2].result, /Agent then requested human approval/);
  assert.equal(result.rateLimitWaits, 1);
  assert.equal(result.rateLimitNotes[0], "Waiting 4s: groq rate limit");
  assert.equal(result.rows.at(-1).rowType, "system_activity");
  assert.match(result.rows.at(-1).title, /Provider and system activity/);
  assert.equal(result.rows.filter((row) => /Rate limit wait/.test(row.title)).length, 0);
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

test("groups a browser form batch into one completed step and marks a pre-submit verification miss as a warning", () => {
  const result = groupTraceEvents([
    { event_type: "action", sequence: 1, payload: { tool: "browser_fill_form", fields: [{ selector: "#invoice_number_input" }] } },
    { event_type: "action", sequence: 2, payload: { tool: "browser_fill", selector: "#invoice_number_input", value: "INV-2026-2116" } },
    { event_type: "action", sequence: 3, payload: { tool: "browser_fill", selector: "#amount_input", value: "68500.00" } },
    { event_type: "observation", sequence: 4, payload: { tool: "browser_fill_form", next_step: "All required fields filled. Next: click #submit_btn.", status: "filled_form" } },
    { event_type: "action", sequence: 5, payload: { tool: "verify_ledger_row", invoice_number: "INV-2026-2116" } },
    { event_type: "observation", sequence: 6, payload: { tool: "verify_ledger_row", status: "not_found", pass: false, message: "No ledger row was found. No submit was performed." } },
  ]);
  assert.equal(result.rows.length, 2);
  assert.equal(result.rows[0].title, "Prepared invoice form");
  assert.match(result.rows[0].result, /All required fields filled/);
  assert.equal(result.rows[1].title, "Ledger row not found");
  assert.equal(result.rows[1].status, "warning");
  assert.match(result.rows[1].result, /No submit was performed/);
});

test("finish action includes the final observation and policy steps show their thresholds", () => {
  const result = groupTraceEvents([
    { event_type: "action", sequence: 1, payload: { tool: "read_policy" } },
    { event_type: "observation", sequence: 2, payload: { tool: "read_policy", policy: { thresholds: { INR: "50000.00", USD: "1000.00" } } } },
    { event_type: "action", sequence: 3, payload: { tool: "finish", status: "complete" } },
    { event_type: "final_result", sequence: 4, payload: { status: "complete" } },
    { event_type: "observation", sequence: 5, payload: { tool: "finish", summary: "Invoice posted and verified." } },
  ]);
  assert.equal(result.rows[0].result, "Approval thresholds: INR 50000.00 · USD 1000.00.");
  assert.equal(result.rows[1].result, "Invoice posted and verified.");
});

test("does not duplicate a host refusal in the trace summary", () => {
  const result = groupTraceEvents([
    { event_type: "task_intent", sequence: 1, payload: { intent: "unsupported", clarifying_question: "Would you like me to record it only?" } },
    { event_type: "final_result", sequence: 2, payload: { status: "needs_input", summary: "I can record it, but cannot pay it. Would you like me to proceed?" } },
  ]);
  assert.equal(result.rows.filter((row) => row.rowType !== "system_activity").length, 1);
  assert.equal(result.rows[0].title, "Request needs a capability clarification");
});

test("renders a four-step progress model from existing events", () => {
  const selecting = progressSteps({
    status: "running", events: [
      { event_type: "task_intent", payload: { intent: "write_invoice_entry" } },
      { event_type: "action", payload: { tool: "list_invoice_candidates" } },
    ],
  });
  assert.equal(selecting[1].state, "current", "candidate discovery is not a declared invoice selection");

  const awaitingApproval = progressSteps({
    status: "awaiting_approval", awaiting_approval: true,
    events: [
      { event_type: "task_intent", payload: { intent: "write_invoice_entry" } },
      { event_type: "action", payload: { tool: "list_invoice_candidates" } },
      { event_type: "selection_check", payload: { status: "passed" } },
      { event_type: "approval_requested", payload: {} },
    ],
  });
  assert.deepEqual(Array.from(awaitingApproval, (step) => step.title), ["Understand request", "Select invoice", "Approve", "Submit & verify"]);
  assert.deepEqual(Array.from(awaitingApproval, (step) => step.state), ["done", "done", "current", "upcoming"]);

  const refused = progressSteps({
    status: "needs_input", events: [{ event_type: "task_intent", payload: { intent: "unsupported" } }],
  });
  assert.deepEqual(Array.from(refused, (step) => step.state), ["warning", "skipped", "skipped", "skipped"]);

  const verified = progressSteps({
    status: "complete", events: [
      { event_type: "task_intent", payload: { intent: "write_invoice_entry" } },
      { event_type: "selection_check", payload: { status: "passed" } },
      { event_type: "approval_decision", payload: { decision: "approved" } },
      { event_type: "verification", payload: { status: "verified" } },
    ],
  });
  assert.equal(verified.length, 4);
  assert.equal(verified[3].state, "done");
});
