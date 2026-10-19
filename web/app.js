(function attachTraceSummary(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CentrAlignTrace = api;
})(typeof window === "undefined" ? globalThis : window, function createTraceSummary() {
  const META_EVENTS = new Set([
    "provider_turn", "host_run_state", "prompt_size", "prompt_size_summary", "planner",
    "planner_fallback", "task_intent", "intent_guard", "long_operation", "credentials_supplied",
    "provenance_attached", "approval_token_injected", "pending_question_stored", "finish_nudge",
  ]);
  const SYSTEM_ACTIVITY_EVENTS = new Set([
    ...META_EVENTS, "tool_retry_budget", "provider_turn", "rate_limit_recovery", "long_operation",
  ]);

  function payload(event) { return event && event.payload && typeof event.payload === "object" ? event.payload : {}; }
  function dateMs(event) {
    const value = Date.parse(event && event.timestamp);
    return Number.isFinite(value) ? value : null;
  }
  function duration(start, end) {
    const from = dateMs(start);
    const to = dateMs(end);
    if (from === null || to === null || to < from) return "";
    if (to - from < 1000) return "";
    const seconds = Math.round((to - from) / 1000);
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  }
  function basename(path) { return String(path || "").split(/[\\/]/).pop() || ""; }
  function displayAmount(invoice) {
    if (!invoice) return "";
    const amount = invoice.amount === undefined || invoice.amount === null ? "" : Number(invoice.amount).toLocaleString("en-IN", { maximumFractionDigits: 2 });
    return [invoice.currency, amount].filter(Boolean).join(" ");
  }
  function actionTitle(event) {
    const p = payload(event);
    const tool = p.tool || "browser action";
    if (tool === "list_invoice_candidates") return `Compared invoice candidates${p.vendor_query ? ` for ${p.vendor_query}` : ""}`;
    if (tool === "read_document") return `Read ${basename(p.path) || "invoice document"}`;
    if (tool === "read_policy") return "Checked procurement policy";
    if (tool === "read_ledger") return `Read ERP ledger${p.vendor_query ? ` for ${p.vendor_query}` : ""}`;
    if (tool === "browser_open") return `Opened ERP page ${p.path || ""}`.trim();
    if (tool === "erp_login") return "Signed in to ERP";
    if (tool === "browser_fill" || tool === "browser_fill_form" || tool === "browser_action_sequence") return "Prepared invoice form";
    if (tool === "browser_click" && /submit_btn/i.test(p.selector || "")) return "Submitted to ERP";
    if (tool === "browser_click" && /login_btn/i.test(p.selector || "")) return "Signed in to ERP";
    if (tool === "browser_click") return `Used ERP control ${p.selector || "button"}`;
    if (tool === "browser_snapshot") return "Checked the ERP page";
    if (tool === "browser_check") return "Checked an ERP field";
    if (tool === "verify_ledger_row") return "Verified the ERP ledger";
    if (tool === "request_human_approval") return "Requested human approval";
    if (tool === "finish") return "Finished the task";
    return String(tool).replaceAll("_", " ");
  }
  function actionResult(event, observation) {
    const p = payload(observation);
    const action = payload(event);
    if (!observation) return p.error || "Waiting for the tool result.";
    if (p.error) return p.error;
    if (p.warning) return p.message ? `${p.message} ${p.warning}` : p.warning;
    if (action.tool === "read_document" && p.invoice) {
      const invoice = p.invoice;
      return [invoice.vendor, invoice.invoice_number, displayAmount(invoice), invoice.invoice_date && `dated ${invoice.invoice_date}`].filter(Boolean).join(" · ");
    }
    if (action.tool === "read_policy" && p.policy) {
      const thresholds = p.policy.thresholds || p.policy.approval_thresholds || {};
      const summary = Object.entries(thresholds).map(([currency, amount]) => `${currency} ${amount}`).join(" · ");
      return summary ? `Approval thresholds: ${summary}.` : "Procurement policy was read by the host.";
    }
    if (action.tool === "list_invoice_candidates" && Array.isArray(p.candidates)) return `${p.candidates.length} invoice candidates found${action.vendor_query ? ` for ${action.vendor_query}` : ""}.`;
    if (action.tool === "read_ledger" && Array.isArray(p.bills)) return `${p.bills.length} ledger invoice${p.bills.length === 1 ? "" : "s"} found.`;
    if (p.fields && typeof p.fields === "object") return `${p.fields.invoice_number || "Invoice"} · ${p.fields.vendor || "vendor"} · ${p.fields.amount || "amount"} · due ${p.fields.due_date || "date unavailable"}`;
    if (p.alert) return `ERP alert: ${p.alert}`;
    if (p.message) return p.message;
    if (p.summary) return p.summary;
    if (p.next_step) return p.next_step;
    if (p.status === "ok" || p.status === "success" || p.status === "submitted") return "Completed successfully.";
    if (action.tool === "browser_fill" || action.tool === "browser_fill_form") return "Fields were checked by the host.";
    if (action.tool === "browser_click" && /submit_btn/i.test(action.selector || "")) return "ERP accepted the invoice.";
    if (action.tool === "erp_login") return p.success === false ? "ERP sign-in needs attention." : "ERP session is ready.";
    if (action.tool === "finish") return p.summary || p.answer?.text || "Task finished.";
    return "Done.";
  }
  function matchObservation(events, actionIndex) {
    const action = events[actionIndex];
    const tool = payload(action).tool;
    for (let index = actionIndex + 1; index < events.length; index += 1) {
      const current = events[index];
      if (current.event_type === "action") break;
      if (current.event_type === "final_result" && tool === "finish") continue;
      if (current.event_type === "final_result" || current.event_type === "run_error") break;
      if (current.event_type !== "observation") continue;
      const p = payload(current);
      if (p.tool === tool || p.tool === "tool_validation_feedback" || p.tool === "tool_use_failed" || p.error || p.alert) return index;
    }
    return -1;
  }
  function matchFillBatch(events, actionIndex) {
    const children = [];
    for (let index = actionIndex + 1; index < events.length; index += 1) {
      const current = events[index];
      if (current.event_type === "action") {
        if (payload(current).tool === "browser_fill") { children.push(index); continue; }
        break;
      }
      if (current.event_type === "final_result" || current.event_type === "run_error") break;
      if (current.event_type !== "observation") continue;
      const p = payload(current);
      if (p.tool === "browser_fill_form" || p.tool === "tool_validation_feedback" || p.tool === "tool_use_failed" || p.error) return { observationIndex: index, children };
    }
    return { observationIndex: -1, children };
  }
  function isFillBatchChild(events, actionIndex) {
    for (let index = actionIndex - 1; index >= 0; index -= 1) {
      const current = events[index];
      if (current.event_type !== "action") continue;
      const tool = payload(current).tool;
      if (tool === "browser_fill") continue;
      return tool === "browser_fill_form";
    }
    return false;
  }
  function ledgerRowLabel(value) {
    const id = String(value ?? "?");
    return id.startsWith("#") ? id : `#${id}`;
  }
  function nextActionTitle(events, startIndex) {
    for (let index = startIndex + 1; index < events.length; index += 1) {
      if (events[index].event_type === "action") return actionTitle(events[index]).toLowerCase();
      if (events[index].event_type === "final_result" || events[index].event_type === "run_error") return "";
    }
    return "";
  }
  function failedActionTitle(event, retrying) {
    const tool = payload(event).tool;
    const labels = {
      browser_fill: "Invoice field fill",
      browser_fill_form: "Invoice form fill",
      browser_action_sequence: "Invoice form fill",
      read_policy: "Procurement policy check",
      list_invoice_candidates: "Invoice candidate search",
      verify_ledger_row: "Ledger verification",
      erp_login: "ERP login",
      browser_open: "ERP navigation",
      browser_snapshot: "ERP page check",
    };
    const label = tool === "browser_click" && /submit_btn/i.test(payload(event).selector || "")
      ? "ERP submit"
      : labels[tool] || actionTitle(event);
    return `${label} failed${retrying ? ", retrying" : ""}`;
  }
  function shortError(value) {
    const message = String(value || "The tool call failed.").replace(/\s+/g, " ").trim();
    return message.length > 180 ? `${message.slice(0, 177)}…` : message;
  }
  function unsupportedActionReason(actions) {
    const reasons = [...new Set((Array.isArray(actions) ? actions : []).map((action) => {
      const value = String(action || "").trim().toLowerCase();
      if (value === "email vendor") return "Emailing the vendor is not supported.";
      if (value === "email") return "Email is not supported.";
      if (["pay", "payment", "approve payment"].includes(value)) return "Making payments is not supported.";
      if (["notify finance", "inform finance"] .includes(value)) return "Notifying finance is not supported.";
      if (["notify vendor", "inform vendor"].includes(value)) return "Notifying the vendor is not supported.";
      if (value === "transfer") return "Transferring funds is not supported.";
      if (value === "delete") return "Deleting records is not supported.";
      if (value === "refund") return "Issuing refunds is not supported.";
      if (["send", "send message"].includes(value)) return "Sending messages is not supported.";
      return value ? `${value[0].toUpperCase()}${value.slice(1)} is not supported.` : "";
    }).filter(Boolean))];
    return reasons.join(" ");
  }
  function criterionLabel(value) {
    return ({ latest_by_invoice_date: "latest by invoice date", specific_invoice_number: "requested invoice number", other: "requested selection" })[value] || value || "requested selection";
  }
  function pushRow(rows, { event, endEvent, title, result, status = "ok", rowType, detail, sequence }) {
    rows.push({
      id: `${event.sequence ?? rows.length}-${event.event_type}`,
      status,
      icon: status === "ok" ? "✓" : status === "running" ? "◌" : status === "error" ? "×" : "!",
      title,
      result: result || "",
      duration: duration(event, endEvent || event),
      timestamp: event.timestamp || "",
      rowType: rowType || event.event_type,
      detail: detail === undefined ? payload(endEvent || event) : detail,
      sequence: sequence ?? (Number.isFinite(Number(event.sequence)) ? Number(event.sequence) : rows.length),
      insertionOrder: rows.length,
    });
  }

  function groupTraceEvents(input) {
    const events = (Array.isArray(input) ? input : []).map((event, index) => ({ event, index }))
      .filter(({ event }) => event && event.event_type !== "browser_frame")
      .sort((left, right) => {
        const leftSequence = Number.isFinite(Number(left.event.sequence)) ? Number(left.event.sequence) : left.index;
        const rightSequence = Number.isFinite(Number(right.event.sequence)) ? Number(right.event.sequence) : right.index;
        return leftSequence - rightSequence || left.index - right.index;
      })
      .map(({ event }) => event);
    const rows = [];
    const consumed = new Set();
    const seenSelections = new Set();
    let rateLimitWaits = 0;
    let correctedCalls = 0;
    const rateLimitNotes = [];
    const systemEvents = events.filter((event) => SYSTEM_ACTIVITY_EVENTS.has(event.event_type));

    for (let index = 0; index < events.length; index += 1) {
      const event = events[index];
      if (consumed.has(index)) continue;
      const p = payload(event);
      const type = event.event_type;

      if (type === "action") {
        if (p.tool === "browser_fill" && isFillBatchChild(events, index)) continue;
        const batch = p.tool === "browser_fill_form" ? matchFillBatch(events, index) : null;
        const observationIndex = batch ? batch.observationIndex : matchObservation(events, index);
        for (const childIndex of batch?.children || []) consumed.add(childIndex);
        const observation = observationIndex >= 0 ? events[observationIndex] : null;
        if (observationIndex >= 0) consumed.add(observationIndex);
        const tool = p.tool;
        let associatedIndex = -1;
        const associatedType = tool === "request_human_approval" ? "approval_requested" : tool === "verify_ledger_row" ? "verification" : "";
        if (associatedType) {
          for (let candidate = index + 1; candidate < events.length; candidate += 1) {
            if (events[candidate].event_type === "action" || events[candidate].event_type === "run_error" || events[candidate].event_type === "final_result") break;
            if (events[candidate].event_type === associatedType) { associatedIndex = candidate; break; }
          }
        }
        const associated = associatedIndex >= 0 ? events[associatedIndex] : null;
        if (associatedIndex >= 0) consumed.add(associatedIndex);
        const observationPayload = payload(observation);
        const result = actionResult(event, observation);
        const semantic = payload(associated);
        const failed = Boolean(observationPayload.error || observationPayload.warning || observationPayload.alert || observationPayload.success === false || observationPayload.pass === false || ["error", "failed", "rejected", "not_found"].includes(observationPayload.status) || semantic.status === "mismatch" || semantic.pass === false);
        let summary = result;
        let title = actionTitle(event);
        if (associatedType === "approval_requested" && associated) {
          const invoice = semantic.invoice || {};
          title = `Approval requested${displayAmount(invoice) ? `: ${displayAmount(invoice)}` : ""}`;
          summary = semantic.reason || (semantic.policy_reasons || []).join(" ") || result;
        }
        if (associatedType === "verification" && associated) {
          const matches = semantic.matches || {};
          const passed = semantic.status === "verified" || (Object.keys(matches).length > 0 && Object.values(matches).every(Boolean));
          title = passed ? `Verified: all ${Object.keys(matches).length} fields match ledger row ${ledgerRowLabel(semantic.ledger_row_id)}` : "Ledger verification failed";
          summary = passed ? `The ledger row matches the selected invoice.${semantic.screenshot_path ? ` Evidence: ${semantic.screenshot_path.split(/[\\/]/).pop()}.` : ""}` : semantic.message || semantic.error || `Verification status: ${semantic.status || "not verified"}.`;
        }
        if (tool === "verify_ledger_row" && failed && observationPayload.status === "not_found") title = "Ledger row not found";
        const notFound = tool === "verify_ledger_row" && observationPayload.status === "not_found";
        if (notFound) {
          title = "Ledger row not found";
          summary = shortError(observationPayload.message || result);
        } else if (failed) {
          const next = nextActionTitle(events, observationIndex >= 0 ? observationIndex : index);
          title = failedActionTitle(event, Boolean(next));
          const error = shortError(observationPayload.error || observationPayload.alert || observationPayload.warning || result);
          if (next) summary = `${error} Agent then ${next}.`;
          else summary = error;
          if (next) correctedCalls += 1;
        }
        const running = !observation && index === events.map((item, eventIndex) => item.event_type === "action" ? eventIndex : -1).filter((value) => value >= 0).at(-1);
        pushRow(rows, { event, endEvent: observation, title, result: summary, status: failed ? "warning" : running ? "running" : "ok", detail: { action: p, observation: observationPayload, ...(associated ? { [associatedType]: semantic } : {}) } });
        continue;
      }

      if (type === "selection_check") {
        const selectionKey = `${p.selected_invoice || p.invoice_number || ""}|${p.criterion || ""}|${p.status || ""}`;
        if (seenSelections.has(selectionKey)) continue;
        seenSelections.add(selectionKey);
        const chosen = p.selected_invoice || p.invoice_number || "invoice";
        const criterion = criterionLabel(p.criterion);
        const passed = p.status === "passed";
        const title = passed ? `Chose ${chosen} (${criterion})` : `Invoice selection needs attention`;
        const result = p.reason || (passed ? "Selection matches the requested criterion." : "The requested invoice selection did not pass the host check.");
        pushRow(rows, { event, title, result, status: passed ? "ok" : "warning" });
        continue;
      }

      if (type === "approval_requested") {
        const invoice = p.invoice || {};
        const amount = displayAmount(invoice);
        pushRow(rows, { event, title: `Approval requested${amount ? `: ${amount}` : ""}`, result: p.reason || (p.policy_reasons || []).join(" ") || `${invoice.invoice_number || "Invoice"} requires a human decision.`, status: "warning" });
        continue;
      }

      if (type === "approval_decision") {
        const approved = p.decision === "approved" || p.approved === true || p.token_minted === true;
        pushRow(rows, { event, title: approved ? "Approval granted" : "Approval denied", result: approved ? "The host recorded approval for this invoice." : p.message || "The invoice was not approved.", status: approved ? "ok" : "warning" });
        continue;
      }

      if (type === "verification") {
        const count = Object.keys(p.matches || {}).length;
        const passed = p.status === "verified" || (count > 0 && Object.values(p.matches).every(Boolean));
        const title = passed ? `Verified: all ${count} fields match ledger row ${ledgerRowLabel(p.ledger_row_id)}` : "Ledger verification failed";
        const result = passed ? `The ledger row matches the selected invoice.${p.screenshot_path ? ` Evidence: ${basename(p.screenshot_path)}.` : ""}` : p.message || `Verification status: ${p.status || "not verified"}.`;
        pushRow(rows, { event, title, result, status: passed ? "ok" : "error" });
        continue;
      }

      if (type === "rate_limit_recovery") {
        const action = p.action || "recovery";
        if (action === "retry_wait") {
          rateLimitWaits += 1;
          rateLimitNotes.push(`Waiting ${p.wait_seconds ?? "?"}s: ${p.provider || "provider"} rate limit`);
        }
        continue;
      }

      if (type === "tool_retry_warning" || type === "loop_guard_warning" || type === "self_correction") {
        const message = p.warning || p.text || p.error || p.message || (type === "loop_guard_warning" ? `Repeated ${p.tool || "action"}; the agent is checking for a new path.` : "The agent corrected a tool call.");
        const next = nextActionTitle(events, index);
        if (type === "tool_retry_warning" && next) correctedCalls += 1;
        pushRow(rows, { event, title: type === "loop_guard_warning" ? "Repeated action detected" : type === "self_correction" ? "Agent corrected its approach" : "Tool call needs correction", result: next ? `${message} Next, the agent ${next}.` : message, status: "warning" });
        continue;
      }

      if (type === "run_error") {
        pushRow(rows, { event, title: "Run stopped with an error", result: p.error || "The run stopped safely.", status: "error" });
        continue;
      }

      if (type === "prompt_injection_detected") {
        pushRow(rows, { event, title: "Instruction-like text ignored", result: p.message || `Ignored untrusted text in ${basename(p.path) || "the invoice"}.`, status: "warning" });
        continue;
      }

      if (type === "pending_question_reply") {
        const confirmed = p.status === "confirmed";
        const cancelled = p.status === "cancelled";
        pushRow(rows, { event, title: confirmed ? "Previous question confirmed" : cancelled ? "Previous question cancelled" : "Follow-up clarified", result: p.message || "The follow-up was handled in this run.", status: cancelled ? "warning" : "ok" });
        continue;
      }

      if (type === "task_intent" && (p.intent === "unsupported" || p.intent === "ambiguous")) {
        const guardedActions = p.unsupported_actions || p.intent_guard?.unsupported_actions;
        const guardReason = p.intent === "unsupported" ? unsupportedActionReason(guardedActions) : "";
        pushRow(rows, { event, title: p.intent === "unsupported" ? "Request needs a capability clarification" : "Request needs clarification", result: guardReason || p.clarifying_question || p.reason || "The host stopped before taking action.", status: "warning" });
        continue;
      }

      if (type === "final_result" && p.status && p.status !== "complete") {
        const hostGateAlreadyExplains = p.status === "needs_input" && events.slice(0, index).some((prior) => prior.event_type === "task_intent" && ["unsupported", "ambiguous"].includes(payload(prior).intent));
        if (hostGateAlreadyExplains) continue;
        pushRow(rows, { event, title: p.status === "needs_input" ? "Input needed to continue" : "Run finished", result: p.summary || p.what_was_done || "The run ended without an ERP write.", status: p.status === "needs_input" ? "warning" : "ok" });
        continue;
      }

      if (META_EVENTS.has(type) || ["task_started", "task_finished", "final_result", "planner", "provider_turn", "tool_retry_budget", "observation"].includes(type)) continue;
      if (type === "approval_token_injected") continue;
    }

    const budgetEvents = events.filter((event) => event.event_type === "tool_retry_budget");
    const feedbackEvents = events.filter((event) => event.event_type === "observation" && ["tool_validation_feedback", "tool_use_failed"].includes(payload(event).tool));
    correctedCalls = Math.max(correctedCalls, budgetEvents.length, feedbackEvents.length);
    if (systemEvents.length) {
      const providers = [...new Set(systemEvents.map((event) => payload(event).provider).filter(Boolean))];
      const result = [
        `${systemEvents.length} provider and host events`,
        providers.length ? `providers: ${providers.join(", ")}` : "",
        rateLimitWaits ? `${rateLimitWaits} rate-limit ${rateLimitWaits === 1 ? "wait" : "waits"}` : "",
      ].filter(Boolean).join(" · ");
      pushRow(rows, {
        event: systemEvents[systemEvents.length - 1], title: `Provider and system activity (${systemEvents.length})`,
        result, status: rateLimitWaits ? "warning" : "ok", rowType: "system_activity", detail: systemEvents,
        sequence: Number.MAX_SAFE_INTEGER,
      });
    }
    rows.sort((left, right) => left.sequence - right.sequence || left.insertionOrder - right.insertionOrder);
    return { rows, rateLimitWaits, correctedCalls, rateLimitNotes };
  }

  function progressSteps(run) {
    const events = Array.isArray(run && run.events) ? run.events : [];
    const types = new Set(events.map((event) => event.event_type));
    const finalEvent = [...events].reverse().find((event) => event.event_type === "final_result");
    const intentEvent = events.find((event) => event.event_type === "task_intent");
    const intent = payload(intentEvent).intent || "";
    const readOnly = intent === "read_only";
    const refused = ["unsupported", "ambiguous"].includes(intent) || payload(finalEvent).status === "needs_input";
    const approved = events.some((event) => event.event_type === "approval_decision" && (payload(event).decision === "approved" || payload(event).token_minted === true));
    const requested = types.has("approval_requested");
    const selected = events.some((event) => event.event_type === "selection_check" && payload(event).status === "passed")
      || events.some((event) => event.event_type === "action" && payload(event).tool === "read_document");
    const proof = [...events].reverse().find((event) => event.event_type === "verification");
    const verified = payload(proof).status === "verified";
    const hostStates = events.filter((event) => event.event_type === "host_run_state").map(payload);
    const host = hostStates.at(-1) || {};
    const submitted = host.submission_status === "submitted" || /submitted/i.test(host.workflow_hint || "") || events.some((event) => event.event_type === "action" && payload(event).tool === "browser_click" && /submit_btn/i.test(payload(event).selector || ""));
    const writing = !readOnly && (intent === "write_invoice_entry" || selected || submitted || requested || verified);
    const completeRead = readOnly && ["complete", "verified"].includes(run.status);
    const terminal = !["running", "awaiting_approval"].includes(run.status) && !run.awaiting_approval;
    const failed = ["needs_review", "incomplete", "failed"].includes(run.status) || types.has("run_error");
    const steps = [
      { key: "understand", title: "Understand request", state: "done" },
      { key: "select", title: "Select invoice", state: selected ? "done" : "upcoming" },
      { key: "approve", title: "Approve", state: requested ? (approved ? "done" : run.awaiting_approval ? "current" : terminal ? "skipped" : "upcoming") : "skipped" },
      { key: "submit_verify", title: "Submit & verify", state: verified ? "done" : submitted ? (failed ? "error" : "current") : "upcoming" },
    ];

    if (!writing && (readOnly || refused)) {
      steps[0].state = refused ? "warning" : "done";
      if (readOnly && completeRead) steps[0].detail = "Read-only request";
      steps[1].state = "skipped";
      steps[2].state = "skipped";
      steps[3].state = "skipped";
      return steps;
    }

    if (failed) {
      if (!selected) steps[1].state = "error";
      else if (requested && !approved) steps[2].state = "error";
      else steps[3].state = "error";
    } else if (!selected && !terminal) steps[1].state = "current";
    else if (!requested && selected && ["required", "policy_not_read"].includes(host.approval_status) && !terminal) steps[2].state = "current";
    else if (selected && !requested && !submitted && !terminal) steps[3].state = host.approval_status === "not_required" ? "current" : "upcoming";
    else if (selected && approved && !submitted && !terminal) steps[3].state = "current";
    else if (submitted && !verified && !terminal) steps[3].state = "current";
    if (terminal && !verified && !failed) {
      if (!selected) steps[1].state = "skipped";
      if (!submitted) steps[3].state = "skipped";
    }
    return steps;
  }

  return { groupTraceEvents, progressSteps, actionTitle };
});

const $ = (selector, root = document) => root.querySelector(selector);
const state = { runId: null, pollTimer: null, finished: true, run: null, view: "summary", approvalSeen: false };

async function requestJson(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || data.error || `Request failed (${response.status})`);
  return data;
}
function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}
function getSessionId() {
  let id = sessionStorage.getItem("centralign-session-id");
  if (!id) { id = crypto.randomUUID(); sessionStorage.setItem("centralign-session-id", id); }
  return id;
}
function friendlyStatus(status) {
  return ({ idle: "Ready", running: "Running", awaiting_approval: "Awaiting approval", complete: "Verified", needs_input: "Needs input", needs_review: "Failed", incomplete: "Failed", failed: "Failed", stopped_by_user: "Stopped" })[status] || String(status || "Ready").replaceAll("_", " ");
}
function setStatus(status, detail = "") {
  const pill = $("#runIndicator");
  pill.dataset.state = status || "idle";
  pill.className = "status-pill";
  if (status === "complete") pill.classList.add("status-success");
  else if (status === "awaiting_approval") pill.classList.add("status-warning");
  else if (["needs_input", "needs_review", "incomplete", "failed"].includes(status)) pill.classList.add("status-error");
  else pill.classList.add("status-neutral");
  $("#runIndicatorText").textContent = detail ? `${friendlyStatus(status)} · ${detail}` : friendlyStatus(status);
}
function setRunButton(running) {
  const button = $("#runButton");
  button.disabled = running;
  button.replaceChildren(document.createTextNode(running ? "Running…" : "Run task "));
  if (!running) { const arrow = node("span", "", "→"); arrow.setAttribute("aria-hidden", "true"); button.append(arrow); }
}
function eventPayload(event) { return event && event.payload && typeof event.payload === "object" ? event.payload : {}; }
function eventTitle(event) {
  const p = eventPayload(event);
  if (event.event_type === "action") return p.tool || "Action";
  if (event.event_type === "observation") return p.tool ? `Observation · ${p.tool}` : "Observation";
  const titles = {
    task_started: "Task started", thought: "Thought", self_correction: "Self-correction", approval_requested: "Human approval requested",
    approval_decision: "Approval decision", approval_token_injected: "Host approval token", planner: `Planner · ${p.mode || "LLM"}`,
    planner_fallback: "Planner fallback", provider_turn: `Provider turn ${p.turn || ""}`, rate_limit_recovery: "Rate-limit recovery",
    long_operation: "Long operation", verification: "Ledger verification", final_result: "Final result", provenance_attached: "Source provenance",
    tool_retry_budget: "Tool retry budget", host_run_state: `Host run state · turn ${p.turn || ""}`, prompt_size: `Prompt estimate · turn ${p.turn || ""}`,
    prompt_size_summary: "Average prompt estimate", task_intent: "Task intent", intent_guard: "Write permission guard", pending_question_stored: "Follow-up saved",
    prompt_injection_detected: "Instruction-like text ignored", selection_check: "Invoice selection check", finish_nudge: "Finish reminder",
    task_finished: "Task finished", run_error: "Run error", loop_guard_warning: "Repeated action warning", tool_retry_warning: "Tool retry warning",
    credentials_supplied: "Credentials supplied by host", pending_question_reply: "Follow-up reply",
  };
  return titles[event.event_type] || String(event.event_type || "Event").replaceAll("_", " ");
}
function eventSummary(event) {
  const p = eventPayload(event);
  if (event.event_type === "action") {
    if (p.tool === "read_document") return `Read ${p.path || "invoice source"}.`;
    if (p.tool === "browser_open") return `Open local ERP route ${p.path || ""}.`;
    if (p.tool === "browser_fill_form") return `Fill ${p.fields?.length || 0} validated form fields.`;
    if (p.tool === "browser_click") return `Click ${p.selector || "ERP control"}.`;
    if (p.tool === "request_human_approval") return "Pause for human approval.";
    if (p.tool === "verify_ledger_row") return "Compare the submitted invoice with the ledger.";
    return p.tool || "Tool action";
  }
  if (event.event_type === "observation") {
    if (p.error) return p.error;
    if (p.warning) return p.message ? `${p.message} ${p.warning}` : p.warning;
    if (p.message) return p.message;
    if (p.approval_hint) return p.approval_hint;
    if (p.alert) return `ERP alert: ${p.alert}`;
    if (p.next_step) return p.next_step;
    if (p.invoice) return `${p.invoice.vendor || ""} · ${p.invoice.invoice_number || ""} · ${p.invoice.currency || ""} ${p.invoice.amount || ""}`;
    if (p.candidates) return `${p.candidates.length} parsed invoice candidates.`;
    if (p.bills) return `${p.bills.length} ledger invoice(s).`;
    if (p.content) return p.content;
    return p.summary || "Tool result returned.";
  }
  if (event.event_type === "task_started") return p.task || "The operator received a task.";
  if (event.event_type === "thought" || event.event_type === "self_correction") return p.text || p.message || "";
  if (event.event_type === "host_run_state") {
    const invoice = p.selected_invoice;
    return [p.logged_in ? "signed in" : "not signed in", invoice ? `invoice ${invoice.invoice_number} · ${invoice.amount} ${invoice.currency}` : "no invoice selected", `approval ${p.approval_status || "unknown"}`, `form ${p.form_status || "unknown"}`, p.workflow_hint || p.next_action || p.last_erp_alert].filter(Boolean).join(" · ");
  }
  if (event.event_type === "task_intent") return `${p.intent || "unknown"}${p.vendor_query ? ` · vendor ${p.vendor_query}` : ""}${p.reason ? ` · ${p.reason}` : ""}${p.clarifying_question ? ` · ${p.clarifying_question}` : ""}`;
  if (event.event_type === "intent_guard") return `Classifier ${p.classifier_intent || "unknown"}; write tools ${p.write_allowed ? "unlocked" : "locked"}${p.unsupported_actions?.length ? `; unsupported ${p.unsupported_actions.join(", ")}` : ""}`;
  if (event.event_type === "prompt_size") return `Approx. ${p.approx_tokens ?? "?"} tokens (${p.approx_characters ?? "?"} chars). ${Object.entries(p.breakdown || {}).map(([key, part]) => `${key} ${part.approx_tokens} tok`).join(" · ")}`;
  if (event.event_type === "prompt_size_summary") return `Average ${p.average_tokens_with_run_state ?? "?"} tokens per turn across ${p.turns ?? "?"} turns.`;
  if (event.event_type === "rate_limit_recovery") return p.action === "retry_wait" ? `${p.provider || "Provider"} HTTP ${p.status_code || 429}; wait ${p.wait_seconds ?? "?"} seconds before retry ${p.retry ?? "?"}/${p.retry_limit ?? "?"}.` : `${p.action || "Recovery"} · ${p.provider || p.previous_provider || "provider"}`;
  if (event.event_type === "long_operation") return `${p.operation || "Operation"} ${p.status || "completed"} after ${p.elapsed_seconds ?? "?"} seconds.`;
  if (event.event_type === "selection_check") return `${p.status || "unknown"} · ${p.criterion || "criterion unavailable"} · ${p.selected_invoice || "invoice unavailable"} · ${p.reason || ""}`;
  if (event.event_type === "approval_requested") return p.reason || (p.policy_reasons || []).join(" ") || "A human decision is required.";
  if (event.event_type === "approval_decision") return p.decision || (p.token_minted ? "approved; runtime token minted" : "denied or timed out");
  if (event.event_type === "verification") return `${p.status || "unknown"}; row ${p.ledger_row_id ?? "?"}; ${Object.keys(p.matches || {}).length} fields checked.`;
  if (event.event_type === "final_result") return p.summary || p.what_was_done || p.ledger_answer?.text || p.candidate_answer?.text || JSON.stringify(p);
  if (event.event_type === "run_error") return p.error || "The run stopped safely.";
  if (event.event_type === "tool_retry_budget" || event.event_type === "tool_retry_warning") return `${p.failure_count ?? "?"}/${p.limit ?? "?"} failures for ${p.tool || "tool"}. ${p.error || p.warning || ""}`;
  if (event.event_type === "loop_guard_warning") return p.warning || `Repeated ${p.tool || "action"}.`;
  if (event.event_type === "prompt_injection_detected") return `${p.message || "Untrusted instruction-like invoice text was ignored."} ${p.path || ""}`;
  if (event.event_type === "provenance_attached") return p.message || `${p.invoice_number || "Invoice"} source provenance attached by host.`;
  if (event.event_type === "credentials_supplied") return p.message || "Credentials supplied by host.";
  if (event.event_type === "pending_question_reply") return p.message || `Follow-up ${p.status || "handled"}.`;
  if (event.event_type === "pending_question_stored") return `Saved a follow-up that expires at ${p.expires_at || "the session timeout"}.`;
  if (event.event_type === "task_finished") return `${p.status || "finished"} · ${p.summary || ""}`;
  return p.message || p.summary || p.error || p.warning || p.reason || JSON.stringify(p);
}
function formatAmount(currency, amount) {
  const numeric = Number(amount);
  const formatted = Number.isFinite(numeric) ? numeric.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : amount || "—";
  return `${currency || ""} ${formatted}`.trim();
}
function setRunLoading(visible) { $("#traceLoading").hidden = !visible; }
function shortPath(value) {
  const text = String(value || "");
  return text.length > 42 ? `${text.slice(0, 21)}…${text.slice(-16)}` : text || "—";
}
function appendCopyValue(container, value, { path = false, label = "Copy value" } = {}) {
  const full = String(value || "—");
  const shown = node("span", `truncate-value${path ? " mono-path" : ""}`, path ? shortPath(full) : full);
  shown.title = full;
  container.append(shown);
  if (full !== "—" && full.length > 27) {
    const button = node("button", "copy-button", "Copy");
    button.type = "button";
    button.setAttribute("aria-label", label);
    button.dataset.copy = full;
    container.append(button);
  }
}
function appendEvidenceLink(container, href, label) {
  if (!href) return;
  const link = node("a", "evidence-link", label || "Open screenshot evidence ↗");
  link.href = href;
  link.target = "_blank";
  link.rel = "noreferrer";
  container.append(link);
}
function traceDurationLabel(value) { return value || ""; }

function renderSummaryTrace(run) {
  const summary = window.CentrAlignTrace.groupTraceEvents(run.events || []);
  const list = $("#traceList");
  list.replaceChildren();
  $("#traceCounts").replaceChildren();
  $("#rateLimitNotes").replaceChildren();
  const counts = $("#traceCounts");
  if (summary.rateLimitWaits) counts.append(node("span", "count-badge count-blue", `${summary.rateLimitWaits} rate-limit ${summary.rateLimitWaits === 1 ? "wait" : "waits"}`));
  if (summary.correctedCalls) counts.append(node("span", "count-badge count-amber", `${summary.correctedCalls} corrected ${summary.correctedCalls === 1 ? "tool call" : "tool calls"}`));
  for (const note of summary.rateLimitNotes || []) $("#rateLimitNotes").append(node("span", "rate-limit-note", note));
  $("#traceEmpty").hidden = summary.rows.length > 0;
  for (const row of summary.rows) {
    if (row.rowType === "system_activity") {
      const item = node("li", "system-activity-event");
      const details = node("details", "system-activity-details");
      const heading = node("summary", "system-activity-summary", row.title);
      heading.title = row.result;
      details.append(heading, node("pre", "diagnostic-payload", JSON.stringify(row.detail || [], null, 2)));
      item.append(details);
      list.append(item);
      continue;
    }
    const item = node("li", "summary-event");
    item.dataset.status = row.status;
    const mark = node("span", `summary-mark mark-${row.status}`, row.icon);
    mark.setAttribute("aria-hidden", "true");
    const content = node("div", "summary-content");
    const top = node("div", "summary-top");
    top.append(node("h3", "summary-title", row.title));
    const side = node("span", "summary-side");
    side.append(node("time", "summary-time", row.timestamp ? new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(row.timestamp)) : ""));
    if (row.duration) side.append(node("span", "summary-duration", traceDurationLabel(row.duration)));
    top.append(side);
    content.append(top);
    if (row.result) {
      const result = node("p", "summary-result", row.result);
      result.title = row.result;
      content.append(result);
    }
    item.append(mark, content);
    list.append(item);
  }
}
function renderDiagnosticsTrace(run, list = $("#diagnosticsList")) {
  list.replaceChildren();
  const events = (run.events || []).filter((event) => event.event_type !== "browser_frame");
  $("#diagnosticsEmpty").hidden = events.length > 0;
  for (const event of events) {
    const item = node("li", "diagnostic-event");
    item.dataset.type = event.event_type;
    const mark = node("span", "diagnostic-mark", event.event_type === "run_error" ? "×" : event.event_type === "verification" ? "✓" : "·");
    mark.setAttribute("aria-hidden", "true");
    const content = node("div", "diagnostic-content");
    const top = node("div", "diagnostic-top");
    top.append(node("h3", "diagnostic-title", eventTitle(event)));
    const timestamp = event.timestamp ? new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(event.timestamp)) : "";
    top.append(node("time", "diagnostic-time", timestamp));
    content.append(top);
    const summary = eventSummary(event);
    if (summary) { const line = node("p", "diagnostic-summary", summary); line.title = summary; content.append(line); }
    const details = node("details", "diagnostic-details");
    details.append(node("summary", "details-toggle", "View details"));
    details.append(node("pre", "diagnostic-payload", JSON.stringify(event.payload || {}, null, 2)));
    content.append(details);
    item.append(mark, content);
    list.append(item);
  }
}
function renderTrace(run) {
  const scroll = $("#traceScroll");
  const previousTop = scroll.scrollTop;
  const follow = $("#followLatest").checked;
  renderSummaryTrace(run);
  renderDiagnosticsTrace(run);
  if (follow) requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
  else scroll.scrollTop = previousTop;
  $("#traceSubtitle").textContent = "Actions and their results, grouped into the steps that matter.";
}
function getRunFacts(run) {
  const events = run.events || [];
  const finalEvent = [...events].reverse().find((event) => event.event_type === "final_result");
  const errorEvent = [...events].reverse().find((event) => event.event_type === "run_error");
  const proofEvent = [...events].reverse().find((event) => event.event_type === "verification");
  const hostEvent = [...events].reverse().find((event) => event.event_type === "host_run_state");
  const approvalEvent = [...events].reverse().find((event) => event.event_type === "approval_requested");
  const final = eventPayload(finalEvent);
  const proof = eventPayload(proofEvent);
  const host = eventPayload(hostEvent);
  const approval = eventPayload(approvalEvent);
  const evidence = final.evidence || {};
  const invoice = final.selected_invoice || host.selected_invoice || approval.invoice || null;
  const readOnly = events.some((event) => event.event_type === "task_intent" && eventPayload(event).intent === "read_only");
  const status = run.awaiting_approval ? "awaiting_approval" : run.status || "running";
  let summary = final.summary || final.what_was_done || final.ledger_answer?.text || final.candidate_answer?.text || "The operator is working through the request.";
  summary = summary.replace(/\s*Not done:\s*none\.?$/i, "");
  if (status === "running") summary = "The operator is working through the request.";
  else if (status === "awaiting_approval") summary = "Review the selected invoice before it is submitted to the ERP.";
  else if (["needs_review", "incomplete", "failed"].includes(status)) summary = final.summary || eventPayload(errorEvent).error || "The task stopped before a verified result was available.";
  else if (status === "needs_input") summary = final.summary || eventPayload(events.find((event) => event.event_type === "task_intent"))?.clarifying_question || "More information is needed before this task can continue.";
  else if (status === "complete" && !final.summary && !final.what_was_done && !final.ledger_answer && !final.candidate_answer) summary = proof.status === "verified" ? "The invoice was recorded and independently checked against the ERP ledger." : "The task completed successfully.";
  let elapsed = final.elapsed_seconds;
  if (elapsed === undefined || elapsed === null) {
    const start = Date.parse(run.started_at || "");
    const endEvent = [...events].reverse().find((event) => event.event_type === "task_finished") || events.at(-1);
    const end = Date.parse(endEvent?.timestamp || "");
    if (Number.isFinite(start) && Number.isFinite(end) && end >= start) elapsed = Math.round((end - start) / 1000);
    else if (status === "running" && Number.isFinite(start)) elapsed = Math.max(0, Math.round((Date.now() - start) / 1000));
  }
  const notDone = [...new Set((final.not_done_actions || []).map((item) => String(item).trim()).filter(Boolean))];
  const screenshotUrl = proof.screenshot_url || evidence.ledger_screenshot_url || evidence.screenshot_url || "";
  return {
    status, summary, invoice, notDone, elapsed, proof, evidence, readOnly,
    ledgerRowId: proof.ledger_row_id ?? evidence.ledger_row_id,
    screenshotPath: proof.screenshot_path || evidence.ledger_screenshot_path || evidence.screenshot_path || "",
    screenshotUrl,
    sourceFile: proof.source_file || evidence.source_file || invoice?.document_path || "",
    sourceHash: proof.source_sha256 || evidence.source_sha256 || "",
    approvalReason: approval.reason || (approval.policy_reasons || []).join(" "),
  };
}
function renderResult(run) {
  const facts = getRunFacts(run);
  let label = friendlyStatus(facts.status);
  let icon = "◌";
  let iconClass = "icon-neutral";
  if (facts.status === "complete") {
    label = facts.readOnly && facts.proof.status !== "verified" ? "Answered" : "Verified";
    icon = facts.readOnly && facts.proof.status !== "verified" ? "✓" : "✓";
    iconClass = "icon-success";
  } else if (facts.status === "awaiting_approval") { label = "Awaiting approval"; icon = "!"; iconClass = "icon-warning"; }
  else if (facts.status === "needs_input") { label = "Needs input"; icon = "!"; iconClass = "icon-warning"; }
  else if (["needs_review", "incomplete", "failed"].includes(facts.status)) { label = "Failed"; icon = "×"; iconClass = "icon-error"; }
  else if (facts.status === "stopped_by_user") { label = "Stopped"; icon = "—"; iconClass = "icon-neutral"; }
  $("#resultStatus").textContent = label;
  $("#resultIcon").textContent = icon;
  $("#resultIcon").className = `result-icon ${iconClass}`;
  $("#resultSummary").textContent = facts.summary;
  $("#resultElapsed").textContent = facts.elapsed === undefined ? "" : `${facts.elapsed}s elapsed`;
  const selected = $("#selectedInvoice");
  selected.replaceChildren();
  selected.hidden = !facts.invoice;
  if (facts.invoice) {
    selected.append(node("span", "selected-label", "Selected invoice"));
    const invoiceValue = node("span", "selected-value");
    const invoiceText = [facts.invoice.invoice_number, facts.invoice.vendor, facts.invoice.invoice_date].filter(Boolean).join(" · ");
    appendCopyValue(invoiceValue, invoiceText || "Invoice details unavailable");
    selected.append(invoiceValue);
    if (facts.invoice.amount !== undefined) selected.append(node("span", "selected-amount", formatAmount(facts.invoice.currency, facts.invoice.amount)));
  }
  const notDone = $("#notDone");
  notDone.replaceChildren();
  notDone.hidden = facts.notDone.length === 0;
  if (facts.notDone.length) notDone.append(node("strong", "", "Not done:"), node("span", "", ` ${facts.notDone.join(", ")}`));

  const resultEvidence = $("#resultEvidence");
  resultEvidence.replaceChildren();
  resultEvidence.hidden = !facts.ledgerRowId && !facts.screenshotPath && !facts.sourceFile;
  if (!resultEvidence.hidden) {
    resultEvidence.append(node("span", "selected-label", "Evidence"));
    const evidenceLines = node("div", "result-evidence-lines");
    if (facts.ledgerRowId !== undefined) {
      const ledgerId = String(facts.ledgerRowId);
      evidenceLines.append(node("span", "evidence-summary", `Ledger row ${ledgerId.startsWith("#") ? ledgerId : `#${ledgerId}`}`));
    }
    if (facts.sourceFile) { const source = node("span", "result-evidence-path"); appendCopyValue(source, facts.sourceFile, { path: true, label: "Copy source file path" }); evidenceLines.append(source); }
    if (facts.sourceHash) { const hash = node("span", "result-evidence-path"); appendCopyValue(hash, facts.sourceHash, { path: true, label: "Copy source SHA-256" }); evidenceLines.append(hash); }
    appendEvidenceLink(evidenceLines, facts.screenshotUrl, "Open ledger screenshot ↗");
    resultEvidence.append(evidenceLines);
  }
}
function renderProgress(run) {
  const steps = window.CentrAlignTrace.progressSteps(run);
  const list = $("#progressStepper");
  list.replaceChildren();
  for (const [index, step] of steps.entries()) {
    const item = node("li", "progress-step");
    item.dataset.state = step.state;
    const mark = node("span", "progress-mark", step.state === "done" ? "✓" : step.state === "current" ? "●" : step.state === "warning" ? "⚠" : step.state === "error" ? "×" : step.state === "skipped" ? "–" : String(index + 1));
    mark.setAttribute("aria-hidden", "true");
    item.setAttribute("aria-label", `${step.title}: ${step.state}`);
    item.append(mark, node("span", "progress-label", step.title));
    if (step.state === "skipped") item.append(node("span", "progress-skipped", "Skipped"));
    list.append(item);
  }
}
function renderApproval(run) {
  const request = [...(run.events || [])].reverse().find((event) => event.event_type === "approval_requested");
  const visible = Boolean(run.awaiting_approval && request);
  $("#approvalBanner").hidden = !visible;
  if (!visible) return;
  const payload = eventPayload(request);
  const invoice = payload.invoice || {};
  $("#approvalTitle").textContent = `Approve ${invoice.invoice_number || "invoice"} before ERP entry`;
  const facts = $("#approvalFacts");
  facts.replaceChildren();
  for (const [name, value, path] of [["Vendor", invoice.vendor, false], ["Invoice #", invoice.invoice_number, false], ["Amount", formatAmount(invoice.currency, invoice.amount), false], ["Due date", invoice.due_date, false], ["Source file", payload.source_file, true]]) {
    const cell = node("div", "approval-fact");
    cell.append(node("span", "approval-fact-label", name));
    const displayed = node("strong", path ? "mono-path truncate-value" : "truncate-value", value || "—");
    displayed.title = value || "—";
    cell.append(displayed);
    if (path && value && value.length > 27) {
      const copy = node("button", "copy-button", "Copy"); copy.type = "button"; copy.dataset.copy = value; copy.setAttribute("aria-label", "Copy source file path"); cell.append(copy);
    }
    facts.append(cell);
  }
  $("#approvalDetail").textContent = payload.reason || (payload.policy_reasons || []).join(" ") || "Policy requires a human decision before the invoice is submitted.";
  $("#approveButton").disabled = false;
  $("#rejectButton").disabled = false;
  if (!state.approvalSeen) {
    state.approvalSeen = true;
    const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    requestAnimationFrame(() => $("#approvalBanner").scrollIntoView({ block: "start", behavior }));
  }
}
function renderEvidence(run) {
  const proofEvent = [...(run.events || [])].reverse().find((event) => event.event_type === "verification");
  const proof = eventPayload(proofEvent);
  const badge = $("#evidenceBadge");
  badge.className = "semantic-badge";
  if (!proofEvent) {
    badge.textContent = run.awaiting_approval ? "Awaiting approval" : run.status === "running" ? "In progress" : "Not verified";
    badge.classList.add(run.awaiting_approval ? "badge-warning" : "badge-neutral");
    $("#evidenceSource").replaceChildren(document.createTextNode("—"));
    $("#evidenceHash").replaceChildren(document.createTextNode("—"));
    const body = $("#evidenceBody");
    body.replaceChildren();
    const row = node("tr"); const cell = node("td", "empty-cell", "No ledger read-back yet."); cell.colSpan = 4; row.append(cell); body.append(row);
    $("#cropLink").hidden = true; $("#ledgerCrop").hidden = true; $("#cropPlaceholder").hidden = false; $("#cropStatus").textContent = "No capture";
    return;
  }
  const matches = proof.matches || {};
  const verified = proof.status === "verified" || Object.keys(matches).length > 0 && Object.values(matches).every(Boolean);
  badge.textContent = verified ? "Verified" : "Needs review";
  badge.classList.add(verified ? "badge-success" : "badge-error");
  $("#evidenceSource").replaceChildren(); appendCopyValue($("#evidenceSource"), proof.source_file || "—", { path: true, label: "Copy source file path" });
  $("#evidenceHash").replaceChildren(); appendCopyValue($("#evidenceHash"), proof.source_sha256 || "—", { path: true, label: "Copy source SHA-256" });
  const body = $("#evidenceBody");
  body.replaceChildren();
  const fields = proof.expected || {};
  const actual = proof.actual || {};
  for (const key of ["vendor", "invoice_number", "invoice_date", "due_date", "amount", "currency"]) {
    if (!(key in fields) && !(key in actual) && !(key in matches)) continue;
    const row = node("tr");
    const fieldLabel = key === "invoice_number" ? "invoice #" : key.replaceAll("_", " ");
    row.append(node("td", "field-name", fieldLabel));
    for (const [value, label] of [
      [key === "vendor" ? `${proof.source_vendor || fields.vendor || "—"}${proof.canonical_vendor ? ` → ${proof.canonical_vendor}` : ""}` : fields[key] || "—", `Copy invoice source ${fieldLabel}`],
      [actual[key] || "—", `Copy ledger ${fieldLabel}`],
    ]) {
      const cell = node("td");
      cell.title = value;
      if (String(value).length > 27) {
        const copy = node("span", "copy-value");
        appendCopyValue(copy, value, { label });
        cell.append(copy);
      } else cell.textContent = value;
      row.append(cell);
    }
    const statusCell = node("td");
    statusCell.append(node("span", `proof-badge${matches[key] ? "" : " mismatch"}`, matches[key] ? "Match" : "Mismatch"));
    row.append(statusCell); body.append(row);
  }
  const screenshotUrl = proof.screenshot_url || "";
  if (screenshotUrl) {
    const link = $("#cropLink"); link.href = screenshotUrl; link.hidden = false;
    const image = $("#ledgerCrop");
    image.onload = () => { link.hidden = false; $("#cropPlaceholder").hidden = true; };
    image.onerror = () => { link.hidden = true; $("#cropPlaceholder").hidden = false; $("#cropStatus").textContent = "Screenshot unavailable"; };
    image.src = `${screenshotUrl}${screenshotUrl.startsWith("data:") ? "" : `?v=${Date.now()}`}`; image.hidden = false;
    $("#cropPlaceholder").hidden = true; $("#cropStatus").textContent = "Captured";
  } else {
    $("#cropLink").hidden = true; $("#ledgerCrop").hidden = true; $("#cropPlaceholder").hidden = false; $("#cropStatus").textContent = "Path recorded";
  }
}
function renderFrame(run) {
  const frame = [...(run.events || [])].reverse().find((event) => event.event_type === "browser_frame" && event.payload?.kind !== "ledger_crop");
  if (!frame) {
    $("#browserFrame").hidden = true;
    $("#browserPlaceholder").hidden = false;
    return;
  }
  const payload = eventPayload(frame);
  const image = $("#browserFrame");
  image.onload = () => { image.hidden = false; $("#browserPlaceholder").hidden = true; };
  image.onerror = () => { image.hidden = true; $("#browserPlaceholder").hidden = false; $("#frameBadge").textContent = "Screenshot unavailable"; };
  image.src = `${payload.screenshot_url || ""}${payload.screenshot_url?.startsWith("data:") ? "" : `${payload.screenshot_url?.includes("?") ? "&" : "?"}v=${Date.now()}`}`;
  image.hidden = !payload.screenshot_url;
  $("#browserPlaceholder").hidden = Boolean(payload.screenshot_url);
  $("#browserUrl").textContent = payload.url || "/erp/login";
  $("#frameAction").textContent = `Frame ${payload.screenshot_index ?? "—"} · ${payload.action || "browser action"}${payload.highlight ? " · element highlighted" : ""}`;
  $("#frameDimensions").textContent = payload.viewport || "16:10 · 1280 × 800";
  $("#frameBadge").textContent = `Frame ${payload.screenshot_index ?? "ready"}`;
}
function renderHistory(runs) {
  const list = $("#runHistory");
  list.replaceChildren();
  const recent = runs.slice(0, 5);
  $("#historyCount").textContent = String(recent.length);
  $("#historySubtitle").textContent = `${recent.length ? `Last ${recent.length}` : "No"} ${recent.length === 1 ? "run" : "runs"}`;
  if (!recent.length) { list.append(node("p", "muted-copy", "No runs yet. Completed and active tasks will appear here.")); return; }
  for (const run of recent) {
    const button = node("button", "history-row");
    button.type = "button";
    button.setAttribute("aria-current", String(run.id === state.runId));
    button.title = run.task || "Invoice task";
    const statusTone = run.status === "complete" ? "success" : run.status === "awaiting_approval" || run.status === "needs_input" ? "warning" : ["needs_review", "incomplete", "failed"].includes(run.status) ? "error" : "neutral";
    button.append(node("span", "history-task", run.task || "Invoice task"), node("span", `history-state history-state-${statusTone}`, friendlyStatus(run.status)));
    const time = node("span", "history-time", `${run.id || ""} · ${run.started_at ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(run.started_at)) : ""}`);
    button.append(time);
    button.addEventListener("click", () => loadRun(run.id));
    list.append(button);
  }
}
function renderRun(run) {
  state.run = run;
  state.runId = run.id;
  $("#traceRunId").textContent = run.id || "—";
  $("#traceRunId").title = run.id || "";
  renderResult(run);
  renderProgress(run);
  renderTrace(run);
  renderApproval(run);
  renderEvidence(run);
  renderFrame(run);
  setRunLoading(false);
  if (run.awaiting_approval) { state.finished = false; setStatus("awaiting_approval"); setRunButton(true); }
  else if (run.status === "running") { state.finished = false; setStatus("running"); setRunButton(true); }
  else { state.finished = true; setRunButton(false); setStatus(run.status || "needs_review"); if (state.pollTimer) clearTimeout(state.pollTimer); refreshState(false); }
}
function setResultTab(view) {
  state.view = view;
  const tabs = [
    ["summary", "summaryViewButton", "summaryPanel"],
    ["diagnostics", "diagnosticsViewButton", "diagnosticsPanel"],
  ];
  for (const [key, buttonId, panelId] of tabs) {
    const active = key === view;
    const button = $(`#${buttonId}`);
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    $(`#${panelId}`).hidden = !active;
  }
  if (view === "diagnostics" && state.run) renderDiagnosticsTrace(state.run);
}
async function pollRun(runId) {
  try {
    const run = await requestJson(`/api/runs/${encodeURIComponent(runId)}`);
    if (state.runId !== runId) return;
    renderRun(run);
    if (!state.finished && state.runId === runId) state.pollTimer = setTimeout(() => pollRun(runId), 700);
  } catch (error) {
    state.finished = true; setRunButton(false); setStatus("needs_review", error.message);
    $("#resultStatus").textContent = "Failed"; $("#resultSummary").textContent = error.message;
    $("#resultIcon").textContent = "×"; $("#resultIcon").className = "result-icon icon-error";
    setRunLoading(false);
  }
}
async function loadRun(runId) {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.runId = runId;
  setRunLoading(true);
  try {
    const run = await requestJson(`/api/runs/${encodeURIComponent(runId)}`);
    state.finished = !["running", "awaiting_approval"].includes(run.status) && !run.awaiting_approval;
    renderRun(run);
    if (!state.finished) pollRun(runId);
  } catch (error) {
    setRunLoading(false); setStatus("needs_review", error.message);
  }
}
async function refreshState(showError = true) {
  try {
    const data = await requestJson("/api/state");
    renderHistory(data.runs || []);
    const labels = { offline: "Offline replay", "offline-replay": "FakeLLM · offline replay", unconfigured: "Live API key required", openai: `OpenAI · ${data.model}`, openai_responses: `OpenAI · ${data.model}`, openai_chat: `OpenAI · ${data.model}`, gemini: `Gemini · ${data.model}`, groq: `Groq · ${data.model}`, openrouter: `OpenRouter · ${data.model}` };
    $("#providerBadge").textContent = labels[data.provider] || "Provider unavailable";
  } catch (error) { if (showError) setStatus("needs_review", error.message); }
}
async function decideApproval(approved) {
  if (!state.runId) return;
  $("#approveButton").disabled = true;
  $("#rejectButton").disabled = true;
  try {
    await requestJson(`/api/runs/${encodeURIComponent(state.runId)}/approval`, { method: "POST", body: JSON.stringify({ approved }) });
    $("#approvalBanner").hidden = true;
    setStatus("running");
    pollRun(state.runId);
  } catch (error) {
    $("#approveButton").disabled = false; $("#rejectButton").disabled = false; setStatus("needs_review", error.message);
    $("#approvalDetail").textContent = error.message;
  }
}
function clearRunPanel() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.finished = false; state.runId = null; state.run = null;
  state.approvalSeen = false;
  setResultTab("summary");
  $("#traceRunId").textContent = "Starting…";
  $("#traceRunId").title = "";
  $("#traceList").replaceChildren();
  $("#diagnosticsList").replaceChildren(); $("#diagnosticsEmpty").hidden = false;
  $("#traceEmpty").hidden = false;
  $("#traceLoading").hidden = false;
  $("#approvalBanner").hidden = true;
  $("#approvalTokenMessage").hidden = true;
  $("#browserFrame").hidden = true; $("#browserPlaceholder").hidden = false; $("#browserUrl").textContent = "/erp/login"; $("#frameBadge").textContent = "Starting browser";
  $("#frameAction").textContent = "Waiting for the first browser action";
  $("#ledgerCrop").hidden = true; $("#cropLink").hidden = true; $("#cropPlaceholder").hidden = false; $("#cropStatus").textContent = "No capture";
  $("#evidenceBadge").textContent = "Not verified"; $("#evidenceBadge").className = "semantic-badge badge-neutral";
  $("#evidenceSource").replaceChildren(document.createTextNode("—")); $("#evidenceHash").replaceChildren(document.createTextNode("—"));
  const empty = node("tr"); const cell = node("td", "empty-cell", "No ledger read-back yet."); cell.colSpan = 4; empty.append(cell); $("#evidenceBody").replaceChildren(empty);
  $("#resultStatus").textContent = "Running"; $("#resultIcon").textContent = "◌"; $("#resultIcon").className = "result-icon icon-neutral";
  $("#resultSummary").textContent = "Starting the operator and connecting to the ERP sandbox."; $("#resultElapsed").textContent = "";
  $("#selectedInvoice").replaceChildren(); $("#selectedInvoice").hidden = true; $("#notDone").replaceChildren(); $("#notDone").hidden = true; $("#resultEvidence").replaceChildren(); $("#resultEvidence").hidden = true;
  $("#traceCounts").replaceChildren(); $("#rateLimitNotes").replaceChildren(); $("#progressStepper").replaceChildren();
  setStatus("running"); setRunButton(true);
}
$("#taskForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const task = $("#taskInput").value.trim();
  if (!task) return;
  clearRunPanel();
  try {
    const result = await requestJson("/api/runs", { method: "POST", body: JSON.stringify({ task, approval_mode: "policy_driven", session_id: getSessionId() }) });
    state.runId = result.run_id;
    $("#traceRunId").textContent = result.run_id || "Starting…";
    pollRun(state.runId);
  } catch (error) {
    state.finished = true; setRunButton(false); setStatus("needs_review", error.message); setRunLoading(false);
    $("#resultStatus").textContent = "Failed"; $("#resultSummary").textContent = error.message;
    $("#resultIcon").textContent = "×"; $("#resultIcon").className = "result-icon icon-error";
  }
});
const SCENARIOS = {
  acme: "Find the latest invoice from Acme, check policy compliance, enter it into the ERP portal, and verify it in the ledger",
  northstar: "Find the latest invoice from Northstar, check policy compliance, enter it into the ERP portal, and verify it in the ledger",
  vertex: "Find the latest invoice from Vertex, check policy compliance, enter it into the ERP portal, and verify it in the ledger",
};
for (const button of document.querySelectorAll(".scenario-chip")) button.addEventListener("click", () => {
  $("#taskInput").value = SCENARIOS[button.dataset.scenario] || SCENARIOS.acme;
  document.querySelectorAll(".scenario-chip").forEach((chip) => chip.setAttribute("aria-pressed", String(chip === button)));
  $("#taskInput").focus();
});
$("#approveButton").addEventListener("click", () => decideApproval(true));
$("#rejectButton").addEventListener("click", () => decideApproval(false));
$("#summaryViewButton").addEventListener("click", () => setResultTab("summary"));
$("#diagnosticsViewButton").addEventListener("click", () => setResultTab("diagnostics"));
document.querySelector(".result-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const buttons = Array.from(document.querySelectorAll(".result-tabs [role=tab]"));
  const index = buttons.indexOf(document.activeElement);
  if (index < 0) return;
  event.preventDefault();
  const next = (index + (event.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length;
  buttons[next].focus(); buttons[next].click();
});
$("#resetDemoButton").addEventListener("click", async () => {
  if (!window.confirm("Clear the demo ledger and run history? Invoice files and policy will stay in place.")) return;
  const button = $("#resetDemoButton"); button.disabled = true;
  if (state.pollTimer) clearTimeout(state.pollTimer);
  try { await requestJson("/api/demo/reset", { method: "POST" }); window.location.reload(); }
  catch (error) { button.disabled = false; setStatus("needs_review", error.message); }
});
document.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-copy]");
  if (!button) return;
  const original = button.textContent;
  try {
    await navigator.clipboard.writeText(button.dataset.copy || "");
    button.textContent = "Copied";
  } catch {
    const input = document.createElement("textarea"); input.value = button.dataset.copy || ""; input.style.position = "fixed"; input.style.opacity = "0"; document.body.append(input); input.select(); document.execCommand("copy"); input.remove(); button.textContent = "Copied";
  }
  setTimeout(() => { if (button.isConnected) button.textContent = original; }, 1200);
});
document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") $("#taskForm").requestSubmit(); });
$("#followLatest").addEventListener("change", () => { if ($("#followLatest").checked) $("#traceScroll").scrollTop = $("#traceScroll").scrollHeight; });
setStatus("idle");
setResultTab("summary");
setRunLoading(false);
refreshState();
