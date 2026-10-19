"""Single observe/decide/execute loop shared by live and offline planners."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from browser_tools import APPROVAL_CHECK_SELECTORS, CLICK_SELECTORS, ERP_ROUTES, MODEL_FORM_SELECTORS, BrowserTools
from company_app import CompanyApp
from documents import list_invoice_candidates, list_workspace_files, read_document, remove_prompt_injection_lines
from intent_rules import HOST_ENTRY_VERBS as _HOST_ENTRY_VERBS, HOST_INVOICE_TOPIC as _HOST_INVOICE_TOPIC, deterministic_unsupported_actions as _deterministic_unsupported_actions
from llm_clients import FakeLLM, LLMClient, ProviderRequestError, SYSTEM_INSTRUCTIONS, ToolCall, make_client
from policy import approval_required


MAX_STEPS = 64
MAX_REPEATED_TOOL_ERRORS = 4
WARN_REPEATED_TOOL_ERRORS = 2
MAX_TOOL_VALIDATION_RETRIES = 3
MAX_SUBMIT_ATTEMPTS = 3
LOOP_GUARD_WARN_AFTER = 3
LOOP_GUARD_STOP_AFTER = 5
FULL_OBSERVATIONS_TO_KEEP = 3
LONG_OPERATION_TRACE_SECONDS = 30
STOP_STATUSES = {"complete", "needs_input", "needs_review", "stopped_by_user"}
MAX_FAILED_GENERATION_CHARS = 300
_ROUTE_CHOICES = "; ".join(f"{path}: {description}" for path, description in ERP_ROUTES.items())
_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12",
}
_CAPABILITIES = {"enter_invoice", "read_ledger", "list_candidates", "unsupported"}


def _explicit_task_months(task: str) -> list[str]:
    """Find month names used as invoice-date criteria, not incidental vendor words."""
    return [
        month for month in _MONTHS
        if re.search(
            rf"\b{month}\b(?:\s+[\w&.-]+){{0,2}}\s+(?:invoice|bill)\b"
            rf"|\b(?:invoice|bill)\s+(?:from|in|for|dated)\s+(?:the\s+)?{month}\b"
            rf"|\bmonth\s+of\s+{month}\b",
            task, re.IGNORECASE,
        )
    ]


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {"name": "list_files", "description": "List invoice and policy files.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "read_document", "description": "Read a listed invoice and extract its fields.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "read_policy", "description": "Read AP rules and approval thresholds.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "list_invoice_candidates", "description": "Parse every inbox PDF/TXT and list structured invoice candidates for the requested vendor. The model chooses one candidate.", "parameters": {"type": "object", "properties": {"vendor_query": {"type": "string"}}, "required": ["vendor_query"], "additionalProperties": False}},
    {"name": "read_ledger", "description": "Read-only lookup of posted ledger invoices for a vendor.", "parameters": {"type": "object", "properties": {"vendor_query": {"type": "string"}}, "required": ["vendor_query"], "additionalProperties": False}},
    {"name": "browser_open", "description": "Open an allowed ERP route; warns about unsaved form fields.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "enum": list(ERP_ROUTES), "description": f"Choose an allowed route: {_ROUTE_CHOICES}"}}, "required": ["path"]}},
    {"name": "erp_login", "description": "Host signs in using company memory and returns the landing page and alert.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "browser_snapshot", "description": "Inspect URL, title, alerts, field states, buttons, and table rows. Secrets are redacted.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "browser_fill", "description": "Fill one allowlisted invoice field with source data or an observed vendor.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(MODEL_FORM_SELECTORS), "description": "Allowlisted invoice field."}, "value": {"type": "string"}}, "required": ["selector", "value"]}},
    {"name": "browser_fill_form", "description": "Fill up to ten allowlisted invoice fields.", "parameters": {"type": "object", "properties": {"fields": {"type": "array", "minItems": 1, "maxItems": 10, "items": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(MODEL_FORM_SELECTORS), "description": "Allowlisted invoice field."}, "value": {"type": "string"}}, "required": ["selector", "value"], "additionalProperties": False}}}, "required": ["fields"], "additionalProperties": False}},
    {"name": "browser_action_sequence", "description": "Run two or three field-fill actions only: browser_fill, browser_check, or browser_fill_form. Navigation, clicks, approval requests, and ledger verification must each be separate tool calls so their results can be observed first.", "parameters": {"type": "object", "properties": {"actions": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "object", "properties": {"tool": {"type": "string", "enum": ["browser_fill", "browser_check", "browser_fill_form"]}, "arguments": {"type": "object"}}, "required": ["tool", "arguments"], "additionalProperties": False}}}, "required": ["actions"], "additionalProperties": False}},
    {"name": "browser_check", "description": "Check the approval confirmation control.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(APPROVAL_CHECK_SELECTORS)}}, "required": ["selector"]}},
    {"name": "browser_click", "description": "Submit the invoice. Declare the task's selection criterion and reason; the host checks them against the discovered candidates.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(CLICK_SELECTORS)}, "selection_criterion": {"type": "string", "enum": ["latest_by_invoice_date", "specific_invoice_number", "other"]}, "selection_reason": {"type": "string"}}, "required": ["selector", "selection_criterion", "selection_reason"], "additionalProperties": False}},
    {"name": "request_human_approval", "description": "Request policy approval for a read candidate. Declare how it matches the task; the host validates latest selections.", "parameters": {"type": "object", "properties": {"invoice_number": {"type": "string"}, "document_path": {"type": "string"}, "canonical_vendor": {"type": "string"}, "reason": {"type": "string"}, "selection_criterion": {"type": "string", "enum": ["latest_by_invoice_date", "specific_invoice_number", "other"]}, "selection_reason": {"type": "string"}}, "required": ["selection_criterion", "selection_reason"], "additionalProperties": False}},
    {"name": "verify_ledger_row", "description": "Verify the submitted source invoice in the ERP ledger. Optional invoice_number only.", "parameters": {"type": "object", "properties": {"invoice_number": {"type": "string"}}, "required": [], "additionalProperties": False}},
    {"name": "finish", "description": "End every task with finish(status, summary); plain-text answers are not allowed. Complete invoice writes only after verification.", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["complete", "needs_input", "needs_review", "stopped_by_user"]}, "summary": {"type": "string"}}, "required": ["status", "summary"], "additionalProperties": False}},
]
TOOL_BY_NAME = {tool["name"]: tool for tool in TOOL_DEFINITIONS}


def _clean_text(value: Any, limit: int = 4000) -> str:
    return str(value or "")[:limit]


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


class AgentLoop:
    """Validates each planner action, executes one tool, and returns its observation."""

    def __init__(
        self,
        data_dir: Path,
        run_id: str,
        task: str,
        base_url: str,
        emit: Callable[[str, dict[str, Any]], None],
        approval_handler: Callable[[dict[str, Any]], Any],
        *,
        approval_mode: str = "policy_driven",
        offline: bool = False,
        client: LLMClient | None = None,
        confirmed_pending_context: dict[str, Any] | None = None,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.run_id = run_id
        self.task = task
        self.confirmed_pending_context = dict(confirmed_pending_context or {})
        self.base_url = base_url
        self.emit = emit
        self.approval_handler = approval_handler
        self.approval_mode = approval_mode
        self.client = client or make_client(
            offline=offline,
            on_recovery=self._provider_recovery,
            on_provider_turn=self._provider_answered,
        )
        self.company = CompanyApp(self.data_dir)
        self.browser = BrowserTools(data_dir, run_id, base_url, self._emit_browser_event)
        self.observations: list[dict[str, Any]] = []
        self.listed_files: set[str] = set()
        self.documents: dict[str, dict[str, Any]] = {}
        self.policy: dict[str, Any] | None = None
        self.selected_invoice: dict[str, Any] | None = None
        self.selected_invoice_path: str | None = None
        self.task_intent: dict[str, Any] | None = None
        self.intent_guard_result: dict[str, Any] | None = None
        self.candidate_invoices: list[dict[str, str]] = []
        self.candidate_listed = False
        self.injection_flagged_paths: set[str] = set()
        self.selection_criterion: str | None = None
        self.selection_reason: str | None = None
        self.approval_tokens: dict[str, str] = {}
        self.denied_invoices: set[str] = set()
        self.prompt_tokens_before: list[int] = []
        self.prompt_tokens_after: list[int] = []
        self.prompt_breakdown_tokens: list[dict[str, int]] = []
        self.duplicate_checked = False
        self.duplicate_present = False
        self.verification: dict[str, Any] | None = None
        self.submit_attempts = 0
        self.submit_performed = False
        self.tool_failure_counts: dict[tuple[str, str], int] = {}
        self.finished: tuple[str, str] | None = None
        self.final_result: dict[str, Any] | None = None
        self._final_result_finalized = False
        self.read_ledger_rows: list[dict[str, Any]] | None = None
        self.read_ledger_vendor_query: str | None = None
        self.approval_record: dict[str, Any] | None = None
        self._plain_text_nudge_used = False
        self.started_monotonic = time.monotonic()
        self.provider_rate_limit_wait_seconds = 0.0
        self._provider_wait_summary_emitted = False
        self._last_page_snapshot: dict[str, Any] | None = None
        self._last_action_key: str | None = None
        self._last_action_page_signature: str | None = None
        self._same_action_no_change_count = 0

    def _emit_browser_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "verification" and self.selection_criterion:
            payload = dict(payload)
            payload["selection_check"] = {
                "status": "passed",
                "criterion": self.selection_criterion,
                "reason": self.selection_reason,
            }
        self.emit(event, payload)

    def _event_action(self, call: ToolCall) -> None:
        safe_args = self._redact_arguments(call.name, call.arguments)
        self.emit("action", {"tool": call.name, **safe_args})

    def _conversation_observations(self, run_state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Keep recent observations intact and compress earlier context to one-line summaries."""
        if getattr(self.client, "name", "") == "offline-replay":
            # FakeLLM is deterministic local replay, not a token-billed conversation model.
            recent = self.observations
        elif len(self.observations) <= FULL_OBSERVATIONS_TO_KEEP:
            recent = self.observations
        else:
            older = self.observations[:-FULL_OBSERVATIONS_TO_KEEP]
            summaries = [self._observation_one_line(item) for item in older]
            recent = [{
                "tool": "history_summary",
                "arguments": {},
                "result": {"summaries": summaries},
            }, *self.observations[-FULL_OBSERVATIONS_TO_KEEP:]]
        state = run_state or self._host_run_state()
        return [{"tool": "host_run_state", "arguments": {}, "result": state}, *recent]

    def _invoice_key(self) -> str | None:
        return self.selected_invoice_path or (str(self.selected_invoice.get("source_file")) if self.selected_invoice else None)

    def _invoice_is_approved(self) -> bool:
        key = self._invoice_key()
        return bool(key and key in self.approval_tokens)

    def _host_run_state(self) -> dict[str, Any]:
        snapshot = self._last_page_snapshot or {}
        page_path = str(snapshot.get("url_path") or urlparse(str(snapshot.get("url", ""))).path)
        invoice = self.selected_invoice or {}
        selected = None
        if invoice:
            selected = {
                "invoice_number": invoice.get("invoice_number"),
                "document_path": self.selected_invoice_path,
                "amount": invoice.get("amount"),
                "currency": invoice.get("currency"),
                "invoice_date": invoice.get("invoice_date"),
                "due_date": invoice.get("due_date"),
            }
        legal_vendor = invoice.get("canonical_vendor")
        observed_vendors = {row[2] for row in getattr(self.browser, "vendor_rows", []) if len(row) >= 3}
        if legal_vendor not in observed_vendors:
            legal_vendor = None
        if not invoice:
            approval_status = "not_selected"
        elif self._invoice_is_approved():
            approval_status = "approved"
        elif self._invoice_key() in self.denied_invoices:
            approval_status = "denied"
        elif not self.policy:
            approval_status = "policy_not_read"
        else:
            approval_status = "required" if self._needs_approval() else "not_required"
        fields = snapshot.get("form_fields", [])
        required_fields = [field for field in fields if isinstance(field, dict) and field.get("required")]
        required_filled = [field for field in required_fields if field.get("state") == "filled"]
        posted = bool(getattr(self.browser, "write_submitted", False))
        if posted:
            form_status = "submitted"
        elif page_path == "/erp/invoices/new" and required_fields and len(required_filled) == len(required_fields):
            form_status = "filled"
        elif page_path == "/erp/invoices/new" and required_filled:
            form_status = "partial"
        else:
            form_status = "empty"
        submission_status = (
            f"submitted: {invoice['invoice_number']}" if posted and invoice
            else "submit_attempted" if self.submit_performed else None
        )
        verification_passed = bool(self.verification and self.verification.get("status") == "verified" and self.verification.get("pass"))
        next_action = "finish" if verification_passed else "verify_ledger_row" if posted else "inspect_submit_alert" if self.submit_performed else None
        workflow_hint = f"{submission_status}; next: {next_action}" if submission_status and next_action else None
        if verification_passed:
            workflow_hint = "Ledger verification passed; next: finish."
        alerts = snapshot.get("alerts") or ([snapshot.get("alert")] if snapshot.get("alert") else [])
        approval_hint = None
        if approval_status == "required":
            approval_hint = f"Approval required: call request_human_approval with invoice_number {invoice['invoice_number']}."
        return {
            "intent": dict(self.task_intent or {}),
            "logged_in": bool(getattr(self.browser, "logged_in", False)),
            "selected_invoice": selected,
            "legal_vendor": legal_vendor,
            "approval_status": approval_status,
            "approval_hint": approval_hint,
            "form_status": form_status,
            "submission_status": submission_status,
            "next_action": next_action,
            "workflow_hint": workflow_hint,
            "last_erp_alert": str(alerts[-1])[:300] if alerts else None,
            "selection_criterion": self.selection_criterion,
            "selection_reason": self.selection_reason,
            "candidate_count": len(self.candidate_invoices),
            "verification_status": "verified" if verification_passed else str((self.verification or {}).get("status") or "pending"),
        }

    def _canonical_action_label(self, action: str) -> str:
        action = " ".join(action.casefold().split())
        if re.search(r"\bapprove\b.*\bpayments?\b", action):
            return "approve payment"
        if re.search(r"\b(?:pay|payment|paid|paying)\b", action):
            return "pay"
        if re.search(r"\b(?:email|e-mail)\b", action):
            return "email vendor" if re.search(r"\b(?:vendor|supplier)\b", action) or re.search(r"\b(?:vendor|supplier)\b", self.task, re.IGNORECASE) else "email"
        if re.search(r"\b(?:wire|transfer)\b", action):
            return "transfer"
        if re.search(r"\b(?:delete|remove)\b", action):
            return "delete"
        if re.search(r"\brefund\b", action):
            return "refund"
        if re.search(r"\b(?:contact|notify|inform|alert|tell)\b", action):
            if re.search(r"\bfinance\b", action):
                return "notify finance"
            if re.search(r"\b(?:notify|inform|alert|tell)\b", action):
                return "notify vendor" if re.search(r"\b(?:vendor|supplier)\b", action) else "notify finance"
            return "contact vendor" if re.search(r"\b(?:vendor|supplier)\b", action) else "contact"
        if re.search(r"\bsend\b", action):
            if re.search(r"\b(?:email|e-mail)\b", self.task, re.IGNORECASE):
                return "email vendor" if re.search(r"\b(?:vendor|supplier)\b", self.task, re.IGNORECASE) else "email"
            return "send"
        return action[:120]

    def _apply_host_intent_guard(self, result: dict[str, Any]) -> dict[str, Any]:
        classifier_intent = result["intent"]
        actions = [dict(item) for item in result["requested_actions"]]
        unsupported = _deterministic_unsupported_actions(self.task)
        explicit_unsupported_labels = {item["action"] for item in unsupported}
        entry_text = re.sub(r"\blog\s+in(?:to)?\b", " ", self.task, flags=re.IGNORECASE)
        explicit_entry_verb = bool(
            _HOST_ENTRY_VERBS.search(entry_text)
            and _HOST_INVOICE_TOPIC.search(self.task)
            and not re.search(
                r"\b(?:do\s+not|don't|never|cannot|can't|should\s+not|must\s+not)\s+(?:enter|record|post|log|add|submit)\b",
                entry_text,
                re.IGNORECASE,
            )
        )
        normalized_actions = []
        for item in actions:
            if item["capability"] == "unsupported":
                item["action"] = self._canonical_action_label(item["action"])
                if item["action"] in {"email", "email vendor"} and re.search(r"\b(?:vendor|supplier)\b", self.task, re.IGNORECASE):
                    item["action"] = "email vendor"
                if item["action"] == "send" and re.search(r"\bsend\s+(?:an?\s+)?e-?mail\b", self.task, re.IGNORECASE):
                    item["action"] = "email vendor" if re.search(r"\b(?:vendor|supplier)\b", self.task, re.IGNORECASE) else "email"
                # The classifier reports its interpretation, but only the host's
                # affirmative-action parser may lock writes. This drops labels
                # inferred from negated or incidental mentions.
                if item["action"] not in explicit_unsupported_labels:
                    continue
            if not any(
                prior["capability"] == item["capability"] and prior["action"].casefold() == item["action"].casefold()
                for prior in normalized_actions
            ):
                normalized_actions.append(item)
        actions = normalized_actions
        if explicit_entry_verb and not any(item["capability"] == "enter_invoice" for item in actions):
            actions.append({"action": "enter invoice in the ledger", "capability": "enter_invoice"})
        action_keys = {(item["capability"], item["action"].casefold()) for item in actions}
        for item in unsupported:
            key = (item["capability"], item["action"].casefold())
            if key not in action_keys:
                actions.append(item)
                action_keys.add(key)

        classifier_unsupported = any(item["capability"] == "unsupported" for item in actions)
        if classifier_intent == "unsupported" and not classifier_unsupported and explicit_entry_verb and not unsupported:
            # The host found an explicit entry request and no affirmative
            # unavailable action. This also covers policy-compliance language,
            # which is supported within the invoice-entry workflow.
            classifier_intent = "write_invoice_entry"
            result["intent"] = classifier_intent
            if not any(item["capability"] == "enter_invoice" for item in actions):
                actions.append({"action": "enter invoice in the ledger", "capability": "enter_invoice"})
            classifier_unsupported = False
        elif classifier_intent == "unsupported" and not classifier_unsupported and not unsupported:
            classifier_intent = "ambiguous"
            result["intent"] = classifier_intent
            result["clarifying_question"] = "Please clarify the invoice task and the invoice you want me to use."
        unsupported_actions = [item["action"] for item in actions if item["capability"] == "unsupported"]
        has_enter_capability = any(item["capability"] == "enter_invoice" for item in actions)
        confirmed_reduced_action = (
            self.confirmed_pending_context.get("proposed_reduced_action") == "enter_invoice"
            and bool(self.confirmed_pending_context.get("confirmed"))
        )
        guard_allows_write = (
            classifier_intent == "write_invoice_entry"
            and explicit_entry_verb
            and has_enter_capability
            and (not unsupported_actions or confirmed_reduced_action)
        )

        effective_intent = classifier_intent
        clarifying_question = result["clarifying_question"]
        if (classifier_unsupported or unsupported) and not confirmed_reduced_action:
            effective_intent = "unsupported"
            clarifying_question = ""
        elif classifier_intent == "write_invoice_entry" and not guard_allows_write:
            effective_intent = "ambiguous"
            clarifying_question = (
                "I can't start an invoice write because the task does not explicitly ask me to enter, record, post, or log an invoice. "
                "Would you like me to record it in the ledger?"
            )
        elif explicit_entry_verb and classifier_intent == "read_only":
            effective_intent = "ambiguous"
            clarifying_question = (
                "The task asks to enter or record an invoice, but the classifier did not authorize that write. "
                "Should I record the invoice in the ledger?"
            )

        host_guard = {
            "classifier_intent": classifier_intent,
            "explicit_entry_verb": explicit_entry_verb,
            "unsupported_actions": unsupported_actions,
            "classifier_reported_unsupported": classifier_unsupported,
            "confirmed_reduced_action": confirmed_reduced_action,
            "write_allowed": guard_allows_write and effective_intent == "write_invoice_entry",
            "effective_intent": effective_intent,
        }
        self.intent_guard_result = host_guard
        result["requested_actions"] = actions
        result["classifier_intent"] = classifier_intent
        result["intent_guard"] = host_guard
        result["intent"] = effective_intent
        result["clarifying_question"] = clarifying_question
        if effective_intent == "unsupported" and unsupported_actions:
            result["unsupported_actions"] = unsupported_actions
        return result

    async def _classify_task_intent(self) -> dict[str, Any]:
        vendors = self.company.list_vendor_directory()
        if self.confirmed_pending_context.get("confirmed"):
            pending = self.confirmed_pending_context
            vendor = str(pending.get("vendor") or "").strip()
            invoice_number = str(pending.get("invoice_number") or "").strip()
            criterion = str(pending.get("selection_criterion") or "other")
            reason = str(pending.get("selection_reason") or "The user confirmed invoice recording only.")
            task_intent = {
                "intent": "write_invoice_entry",
                "vendor_query": vendor,
                "invoice_number": invoice_number,
                "selection_criterion": criterion,
                "reason": reason,
                "clarifying_question": "",
                "requested_actions": [
                    {"action": "record invoice in the ledger", "capability": "enter_invoice"},
                    *[
                        {"action": str(action), "capability": "unsupported"}
                        for action in pending.get("unsupported_actions", [])
                    ],
                ],
            }
            self.intent_guard_result = {
                "classifier_intent": "write_invoice_entry",
                "explicit_entry_verb": True,
                "unsupported_actions": [str(action) for action in pending.get("unsupported_actions", [])],
                "classifier_reported_unsupported": bool(pending.get("unsupported_actions")),
                "confirmed_reduced_action": True,
                "write_allowed": True,
                "effective_intent": "write_invoice_entry",
            }
            task_intent["classifier_intent"] = "write_invoice_entry"
            task_intent["intent_guard"] = dict(self.intent_guard_result)
            return task_intent

        preflight = self._preflight_intent_gate(vendors)
        if preflight is not None:
            return self._apply_host_intent_guard(preflight)

        classifier = getattr(self.client, "classify_task_intent", None)
        if callable(classifier):
            result = await self._await_with_long_operation_trace(
                "intent_classification",
                asyncio.to_thread(classifier, self.task, vendors),
                details={"provider": getattr(self.client, "name", type(self.client).__name__)},
            )
        else:
            # Compatibility for injected test planners. All built-in live/offline adapters implement
            # the schema-constrained classifier above.
            result = FakeLLM().classify_task_intent(self.task, vendors)
        required = {
            "intent", "vendor_query", "invoice_number", "selection_criterion", "reason", "clarifying_question",
            "requested_actions",
        }
        if not isinstance(result, dict) or set(result) != required:
            raise ValueError("The intent classifier did not return the required strict-JSON fields.")
        if result.get("intent") not in {"read_only", "write_invoice_entry", "unsupported", "ambiguous"}:
            raise ValueError("The intent classifier returned an unsupported intent.")
        if result.get("selection_criterion") not in {"latest_by_invoice_date", "specific_invoice_number", "other"}:
            raise ValueError("The intent classifier returned an unsupported selection criterion.")
        if not all(isinstance(result[key], str) for key in required - {"requested_actions"}):
            raise ValueError("Intent classification fields must be strings.")
        if not isinstance(result["requested_actions"], list):
            raise ValueError("Intent classification requested_actions must be a list.")
        normalized_actions = []
        for item in result["requested_actions"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"action", "capability"}
                or not isinstance(item.get("action"), str)
                or not item["action"].strip()
                or item.get("capability") not in _CAPABILITIES
            ):
                raise ValueError("Intent classification contains an invalid requested action mapping.")
            normalized_actions.append({"action": item["action"].strip(), "capability": item["capability"]})
        result = {key: result[key].strip() for key in required - {"requested_actions"}}
        result["requested_actions"] = normalized_actions
        if result["intent"] == "write_invoice_entry":
            explicit_number = re.search(r"\b(?:AC[-_][A-Z0-9_-]+|INV[-_][A-Z0-9_-]+)\b", self.task, re.IGNORECASE)
            explicit_date = re.search(r"\b(20\d{2})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", self.task)
            explicit_months = _explicit_task_months(self.task)
            if explicit_number:
                result["invoice_number"] = explicit_number.group(0)
                result["selection_criterion"] = "specific_invoice_number"
                result["reason"] = f"The task explicitly names invoice {explicit_number.group(0)}."
            elif explicit_date or len(explicit_months) == 1:
                criterion_text = explicit_date.group(0) if explicit_date else explicit_months[0].capitalize()
                result["selection_criterion"] = "other"
                result["reason"] = f"The task explicitly requests the invoice dated {criterion_text}."
        intent = result["intent"]
        if intent in {"read_only", "write_invoice_entry"}:
            query = result["vendor_query"]
            if not query:
                result["intent"] = "ambiguous"
                result["clarifying_question"] = "Which vendor should I use?"
            else:
                matches = []
                query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
                for vendor in vendors:
                    aliases = (vendor.get("short_name", ""), vendor.get("legal_name", ""))
                    if any(
                        alias and (
                            " ".join(query.casefold().split()) in " ".join(alias.casefold().split())
                            or query_terms.issubset(set(re.findall(r"[a-z0-9]+", alias.casefold())))
                        )
                        for alias in aliases
                    ):
                        matches.append(vendor)
                if len(matches) != 1:
                    known = ", ".join(item["short_name"] for item in vendors)
                    result["intent"] = "ambiguous"
                    result["clarifying_question"] = (
                        f"I couldn't identify '{query}' in the vendor directory. Which vendor did you mean: {known}?"
                    )
                else:
                    result["vendor_query"] = matches[0]["short_name"]
        if result["intent"] == "ambiguous" and not result["clarifying_question"]:
            result["clarifying_question"] = "Please clarify the vendor and invoice you want me to use."
        return self._apply_host_intent_guard(result)

    def _preflight_intent_gate(self, vendors: list[dict[str, str]]) -> dict[str, Any] | None:
        """Stop obvious unsupported or underspecified requests before contacting a provider."""
        unsupported = _deterministic_unsupported_actions(self.task)
        has_invoice_topic = bool(_HOST_INVOICE_TOPIC.search(self.task))
        entry_text = re.sub(r"\blog\s+in(?:to)?\b", " ", self.task, flags=re.IGNORECASE)
        explicit_entry = bool(_HOST_ENTRY_VERBS.search(entry_text) and has_invoice_topic)

        aliases: list[tuple[int, str]] = []
        lowered = self.task.casefold()
        for item in vendors:
            for alias in (str(item.get("short_name", "")), str(item.get("legal_name", ""))):
                if alias and re.search(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])", lowered):
                    aliases.append((len(alias), str(item.get("short_name") or item.get("legal_name") or alias)))
        vendor_query = ""
        if aliases:
            max_length = max(size for size, _ in aliases)
            matches = sorted({name for size, name in aliases if size == max_length}, key=str.casefold)
            if len(matches) == 1:
                vendor_query = matches[0]
            else:
                return {
                    "intent": "ambiguous", "vendor_query": "", "invoice_number": "",
                    "selection_criterion": "other", "reason": "The vendor name matches multiple directory entries.",
                    "clarifying_question": "Which vendor did you mean: " + ", ".join(matches) + "?",
                    "requested_actions": ([{"action": "enter invoice in the ledger", "capability": "enter_invoice"}] if explicit_entry else []),
                }
        else:
            phrase = re.search(r"\b(?:from|for|vendor)\s+([A-Z][A-Za-z0-9& .'-]{1,60})", self.task)
            if phrase:
                vendor_query = re.split(
                    r"\b(?:invoice|invoices|bill|bills|latest|newest|record|enter|post|submit|in|and|email|pay)\b",
                    phrase.group(1), flags=re.IGNORECASE,
                )[0].strip(" .,-")

        explicit_number = re.search(r"\b(?:AC[-_][A-Z0-9_-]+|INV[-_][A-Z0-9_-]+|NS[-_][A-Z0-9_-]+)\b", self.task, re.IGNORECASE)
        months = _explicit_task_months(self.task)
        if explicit_number:
            criterion, reason, invoice_number = "specific_invoice_number", f"The task explicitly names invoice {explicit_number.group(0)}.", explicit_number.group(0)
        elif re.search(r"\b(latest|newest|most recent)\b", self.task, re.IGNORECASE):
            criterion, reason, invoice_number = "latest_by_invoice_date", "The task requests the latest invoice by invoice date.", ""
        elif months:
            criterion, reason, invoice_number = "other", f"The task specifies the {months[0].capitalize()} invoice.", ""
        else:
            criterion, reason, invoice_number = "other", "The task does not specify an invoice number or date criterion.", ""

        actions = list(unsupported)
        if explicit_entry:
            actions.insert(0, {"action": "record invoice in the ledger", "capability": "enter_invoice"})
        if unsupported:
            return {
                "intent": "unsupported", "vendor_query": vendor_query, "invoice_number": invoice_number,
                "selection_criterion": criterion, "reason": reason, "clarifying_question": "",
                "requested_actions": actions,
            }

        read_only_invoice_task = has_invoice_topic and bool(re.search(r"\b(?:what|show|list|which|lookup|look up|find|review|read)\b", self.task, re.IGNORECASE))
        if explicit_entry or read_only_invoice_task:
            if not vendor_query:
                if re.search(r"\b(?:from|for)\s+[A-Z][A-Za-z0-9& .'-]{1,60}", self.task):
                    question = f"I couldn't identify vendor {vendor_query or 'in that request'} in the company directory. Which listed vendor should I use?"
                else:
                    question = "Which vendor's invoice should I use?" if explicit_entry else "Which vendor's ledger entries should I look up?"
                return {
                    "intent": "ambiguous", "vendor_query": "", "invoice_number": invoice_number,
                    "selection_criterion": criterion, "reason": reason, "clarifying_question": question,
                    "requested_actions": ([{"action": "record invoice in the ledger", "capability": "enter_invoice"}] if explicit_entry else [{"action": "read ledger", "capability": "read_ledger"}]),
                }
            known = any(vendor_query.casefold() in {str(item.get("short_name", "")).casefold(), str(item.get("legal_name", "")).casefold()} for item in vendors)
            if not known:
                return {
                    "intent": "ambiguous", "vendor_query": vendor_query, "invoice_number": invoice_number,
                    "selection_criterion": criterion, "reason": reason,
                    "clarifying_question": f"I couldn't identify vendor {vendor_query} in the company directory. Which listed vendor should I use?",
                    "requested_actions": ([{"action": "record invoice in the ledger", "capability": "enter_invoice"}] if explicit_entry else [{"action": "read ledger", "capability": "read_ledger"}]),
                }
            if explicit_entry and not explicit_number and criterion == "other" and not months and not re.search(r"\b20\d{2}[-/]\d{1,2}\b", self.task):
                return {
                    "intent": "ambiguous", "vendor_query": vendor_query, "invoice_number": "",
                    "selection_criterion": "other", "reason": "The task does not identify which invoice to enter.",
                    "clarifying_question": f"Which {vendor_query} invoice should I enter: the latest, a specific invoice number, or another date range?",
                    "requested_actions": [{"action": "record invoice in the ledger", "capability": "enter_invoice"}],
                }
        elif not has_invoice_topic:
            return {
                "intent": "unsupported", "vendor_query": "", "invoice_number": "",
                "selection_criterion": "other", "reason": "This request is outside invoice entry and ledger lookup.",
                "clarifying_question": "I can record invoices in the ledger or look up existing invoice entries. What invoice task would you like help with?",
                "requested_actions": [],
            }
        return None

    def _require_write_intent(self) -> None:
        # Directly injected test planners may set a classification without going
        # through _classify_task_intent; rerun the same host guard before writes.
        if self.task_intent and self.intent_guard_result is None:
            self.task_intent = self._apply_host_intent_guard(self.task_intent)
        if (
            not self.task_intent
            or self.task_intent.get("intent") != "write_invoice_entry"
            or not (self.intent_guard_result or {}).get("write_allowed")
        ):
            raise ValueError("Invoice write tools are unavailable unless the host classified this task as write_invoice_entry.")

    def _validate_selection_declaration(self, criterion: str, reason: str) -> None:
        self._require_write_intent()
        if not self.selected_invoice or not self.selected_invoice_path:
            raise ValueError("Read and select one invoice from list_invoice_candidates before declaring selection.")
        if not any(item.get("path") == self.selected_invoice_path for item in self.candidate_invoices):
            raise ValueError("The selected invoice must be one of the candidates returned by list_invoice_candidates.")
        if not criterion or not reason.strip() or len(reason) > 240:
            raise ValueError("Provide the selection criterion and a short reason of at most 240 characters.")
        declared = str(self.task_intent.get("selection_criterion", "other"))
        if criterion != declared:
            raise ValueError(f"The task's declared selection criterion is {declared}; use that criterion for approval and submit.")
        if criterion == "specific_invoice_number":
            requested_number = str(self.task_intent.get("invoice_number", ""))
            if not requested_number or requested_number.casefold() != str(self.selected_invoice.get("invoice_number", "")).casefold():
                raise ValueError(f"The task specifies invoice {requested_number or '(no invoice number parsed)'}; the selected candidate does not match it.")
        if criterion == "other":
            explicit_date = re.search(r"\b(20\d{2})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", self.task)
            explicit_months = _explicit_task_months(self.task)
            selected_date = str(self.selected_invoice.get("invoice_date", ""))
            expected_month = explicit_date.group(2).zfill(2) if explicit_date else None
            expected_year = explicit_date.group(1) if explicit_date else None
            if not expected_month and len(explicit_months) == 1:
                expected_month = _MONTHS[explicit_months[0]]
            if expected_month and (
                len(selected_date) < 7 or selected_date[5:7] != expected_month
                or (expected_year and selected_date[:4] != expected_year)
                or (explicit_date and explicit_date.group(3) and selected_date[8:10] != explicit_date.group(3).zfill(2))
            ):
                label = explicit_date.group(0) if explicit_date else explicit_months[0].capitalize()
                raise ValueError(
                    f"The task explicitly requests the {label} invoice; selected {self.selected_invoice.get('invoice_number')} "
                    f"is dated {selected_date}. Choose a candidate that matches the requested date."
                )
        if criterion == "latest_by_invoice_date":
            latest = max(
                self.candidate_invoices,
                key=lambda item: (item.get("invoice_date", ""), item.get("invoice_number", "")),
                default=None,
            )
            if latest and latest.get("path") != self.selected_invoice_path:
                raise ValueError(
                    f"Latest-by-invoice-date selection rejected: {latest['invoice_number']} at {latest['path']} "
                    f"dated {latest['invoice_date']} is newer than selected {self.selected_invoice['invoice_number']} "
                    f"dated {self.selected_invoice['invoice_date']}."
                )
        self.selection_criterion = criterion
        self.selection_reason = reason.strip()
        self.emit("selection_check", {
            "status": "passed",
            "criterion": criterion,
            "reason": reason.strip(),
            "selected_invoice": self.selected_invoice.get("invoice_number"),
            "candidate_count": len(self.candidate_invoices),
        })

    def _form_state(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        state = self.browser.describe_form_state(snapshot)
        invoice = self.selected_invoice
        if (
            invoice and state["current_page"] == "/erp/invoices/new"
            and self.policy and self._needs_approval() and not self._invoice_is_approved()
        ):
            state["approval_required"] = True
            state["next_step"] = (
                "Approval required: call request_human_approval with invoice_number "
                f"{invoice['invoice_number']}."
            )
        else:
            state["approval_required"] = False
        return state

    def _render_read_ledger_answer(self) -> dict[str, Any]:
        """Render a read-only answer exclusively from host-held ledger rows."""
        rows = self.read_ledger_rows or []
        vendor = str(
            rows[0].get("vendor") if rows else
            self.read_ledger_vendor_query or (self.task_intent or {}).get("vendor_query") or "the requested vendor"
        )
        invoices = [
            {
                "vendor": str(row.get("vendor", "")),
                "invoice_number": str(row.get("invoice_number", "")),
                "invoice_date": str(row.get("invoice_date", "")),
                "due_date": str(row.get("due_date", "")),
                "amount": str(row.get("amount", "")),
                "currency": str(row.get("currency", "")),
            }
            for row in rows
        ]
        if invoices:
            details = "; ".join(
                f"{item['invoice_number']} dated {item['invoice_date']} for {item['currency']} {item['amount']} (due {item['due_date']})"
                for item in invoices
            )
            text = f"{vendor} has {len(invoices)} ledger invoice(s): {details}"
        else:
            text = f"No ledger invoices were found for {vendor}."
        return {"vendor": vendor, "invoice_count": len(invoices), "invoices": invoices, "text": text}

    def _render_candidate_answer(self) -> dict[str, Any]:
        vendor = self.read_ledger_vendor_query or str((self.task_intent or {}).get("vendor_query") or "the requested vendor")
        candidates = [
            {
                "path": str(row.get("path", "")),
                "vendor": str(row.get("vendor", "")),
                "invoice_number": str(row.get("invoice_number", "")),
                "invoice_date": str(row.get("invoice_date", "")),
                "due_date": str(row.get("due_date", "")),
                "amount": str(row.get("amount", "")),
                "currency": str(row.get("currency", "")),
            }
            for row in self.candidate_invoices
        ]
        if candidates:
            details = "; ".join(
                f"{row['invoice_number']} dated {row['invoice_date']} for {row['currency']} {row['amount']} (due {row['due_date']})"
                for row in candidates
            )
            text = f"Found {len(candidates)} invoice candidate(s) for {vendor}: {details}"
        else:
            text = f"No invoice candidates were found for {vendor}."
        return {"vendor": vendor, "candidate_count": len(candidates), "candidates": candidates, "text": text}

    def _not_done_action_labels(self) -> list[str]:
        if self.task_intent is None:
            return ["requested actions could not be classified"]
        actions = (self.task_intent or {}).get("requested_actions", [])
        not_done: list[str] = []
        verified = bool(self.verification and self.verification.get("status") == "verified" and self.verification.get("pass"))
        for item in actions:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "requested action").strip()
            capability = item.get("capability")
            verification_is_part_of_entry = bool(
                verified
                and self.task_intent.get("intent") == "write_invoice_entry"
                and capability == "read_ledger"
                and re.search(r"\b(?:verify|verification|check|confirm|validate|read.?back)\b", action, re.IGNORECASE)
            )
            unperformed = (
                capability == "unsupported"
                or (capability == "enter_invoice" and not verified)
                or (capability == "read_ledger" and self.read_ledger_rows is None and not verification_is_part_of_entry)
                or (capability == "list_candidates" and not self.candidate_listed)
            )
            if capability == "enter_invoice" and self.confirmed_pending_context.get("confirmed"):
                # The user confirmed the reduced record-only proposal. Keep unsupported
                # actions visible, but do not repeat the confirmed proposal as not done.
                unperformed = False
            if not unperformed:
                continue
            if capability == "enter_invoice":
                label = "record invoice"
            elif capability == "read_ledger":
                label = "read ledger"
            elif capability == "list_candidates":
                label = "list candidates"
            elif capability == "unsupported":
                label = self._canonical_action_label(action)
            else:
                label = "requested action"
            if label and label not in not_done:
                not_done.append(label)
        return not_done

    def _unsupported_task_summary(self) -> str:
        unsupported = list((self.intent_guard_result or {}).get("unsupported_actions", []))
        if not unsupported:
            unsupported = [
                item.get("action", "unsupported requested action")
                for item in (self.task_intent or {}).get("requested_actions", [])
                if isinstance(item, dict) and item.get("capability") == "unsupported"
            ]
        wording: list[str] = []
        for action in unsupported:
            label = str(action).casefold()
            phrase = (
                "make payments" if label in {"pay", "payment", "approve payment"} else
                "send email" if label in {"email", "email vendor", "send email"} else
                "transfer funds" if label == "transfer" else
                "delete records" if label == "delete" else
                "issue refunds" if label == "refund" else
                "contact vendors" if label in {"contact", "contact vendor"} else
                "send messages" if label == "send" else label
            )
            if phrase not in wording:
                wording.append(phrase)
        if not wording:
            return str((self.task_intent or {}).get("clarifying_question") or "I need clarification before taking action. Would you like me to record the invoice in the ledger?")
        if len(wording) == 1:
            unsupported_text = wording[0]
        elif len(wording) == 2:
            unsupported_text = " or ".join(wording)
        else:
            unsupported_text = ", ".join(wording[:-1]) + f", or {wording[-1]}"
        return (
            f"I can record the invoice in the ledger, but cannot {unsupported_text}. "
            "Would you like me to proceed with just recording it?"
        )

    @staticmethod
    def _with_not_done_line(summary: str, not_done: list[str]) -> str:
        line = "Not done: " + (", ".join(not_done) if not_done else "none") + "."
        summary = re.sub(r"(?im)\s*Not done:\s*.*$", "", summary).rstrip()
        return f"{summary} {line}" if summary else line

    def _finalize_run_result(self, status: str, summary: str) -> tuple[str, str]:
        """Attach requested/unperformed actions to every terminal run result."""
        if self._final_result_finalized:
            return self.finished or (status, summary)
        if self.task_intent and self.task_intent.get("intent") == "unsupported":
            status, summary = "needs_input", self._unsupported_task_summary()
        elif self.task_intent and self.task_intent.get("intent") == "ambiguous":
            status = "needs_input"
            summary = str(self.task_intent.get("clarifying_question") or summary or "Please clarify the requested invoice action.")
        if status == "complete" and self.task_intent and self.task_intent.get("intent") == "write_invoice_entry":
            if not (self.verification and self.verification.get("status") == "verified" and self.verification.get("pass")):
                status = "needs_review"
                summary = "The invoice entry did not pass ledger verification."

        not_done = self._not_done_action_labels()
        # Needs-input summaries are the user-facing message; the UI renders the
        # structured not-done actions in their own box for these results.
        if status == "needs_input":
            summary = re.sub(r"(?im)\s*Not done:\s*.*$", "", summary).rstrip()
        else:
            summary = self._with_not_done_line(summary, not_done)
        result = dict(self.final_result or {})
        result.update({
            "status": status,
            "summary": summary,
            "requested_actions": list((self.task_intent or {}).get("requested_actions", [])),
            "not_done_actions": not_done,
            "not_done": "none" if not not_done else ", ".join(not_done),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 2),
        })
        if self.confirmed_pending_context.get("confirmed"):
            result["original_task"] = self.confirmed_pending_context.get("original_task")
            result["confirmed_reduced_action"] = "enter_invoice"
        if not result.get("what_was_done"):
            result["what_was_done"] = "No requested actions were completed." if not_done else "Completed the requested actions."
        self.final_result = result
        self.finished = (status, summary)
        self._final_result_finalized = True
        self.emit("final_result", self.final_result)
        return self.finished

    def _tools_for_intent(self) -> list[dict[str, Any]]:
        if not self.task_intent or self.task_intent.get("intent") in {"unsupported", "ambiguous"}:
            return [TOOL_BY_NAME["finish"]]
        if self.task_intent.get("intent") == "read_only":
            capabilities = {
                item.get("capability")
                for item in self.task_intent.get("requested_actions", [])
                if isinstance(item, dict)
            }
            allowed = {"finish"}
            if "read_ledger" in capabilities:
                allowed.add("read_ledger")
            if "list_candidates" in capabilities:
                allowed.add("list_invoice_candidates")
            return [tool for tool in TOOL_DEFINITIONS if tool["name"] in allowed]
        if self.task_intent.get("intent") != "write_invoice_entry" or not (self.intent_guard_result or {}).get("write_allowed"):
            return [TOOL_BY_NAME["finish"]]
        return TOOL_DEFINITIONS

    def _estimate_prompt_tokens(
        self, observations: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> tuple[int, int]:
        prompt = {
            "system": SYSTEM_INSTRUCTIONS,
            "task": self.task,
            "tools": tools or self._tools_for_intent(),
            "observations": observations,
        }
        chars = len(json.dumps(prompt, ensure_ascii=False, separators=(",", ":")))
        return chars, math.ceil(chars / 4)

    def _prompt_size_breakdown(
        self, observations: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, dict[str, int]]:
        state = next((item.get("result", {}) for item in observations if item.get("tool") == "host_run_state"), {})
        history = [item for item in observations if item.get("tool") != "host_run_state"]
        components = {
            "system_prompt": SYSTEM_INSTRUCTIONS,
            "tool_schemas": tools or self._tools_for_intent(),
            "run_state": state,
            "history": {"task": self.task, "observations": history},
        }
        result: dict[str, dict[str, int]] = {}
        for key, value in components.items():
            chars = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            result[key] = {"characters": chars, "approx_tokens": math.ceil(chars / 4)}
        return result

    async def _await_with_long_operation_trace(
        self,
        label: str,
        awaitable: Any,
        *,
        details: dict[str, Any] | None = None,
    ) -> Any:
        """Emit a visible marker when an async wait/request crosses the long-operation threshold."""
        task = asyncio.ensure_future(awaitable)
        started = time.monotonic()
        done, _ = await asyncio.wait({task}, timeout=LONG_OPERATION_TRACE_SECONDS)
        if done:
            return task.result()
        metadata = dict(details or {})
        self.emit("long_operation", {
            "operation": label,
            "status": "still_running",
            "elapsed_seconds": LONG_OPERATION_TRACE_SECONDS,
            **metadata,
        })
        try:
            return await task
        finally:
            elapsed = round(time.monotonic() - started, 2)
            self.emit("long_operation", {
                "operation": label,
                "status": "finished" if task.done() and not task.cancelled() and task.exception() is None else "ended_with_error",
                "elapsed_seconds": elapsed,
                **metadata,
            })

    @staticmethod
    def _observation_one_line(item: dict[str, Any]) -> str:
        tool = str(item.get("tool", "observation"))
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if tool == "read_document":
            invoice = result.get("invoice") or {}
            if invoice:
                text = "invoice " + ", ".join(
                    f"{key}={invoice.get(key, '')}" for key in
                    ("vendor", "invoice_number", "invoice_date", "due_date", "amount", "currency")
                )
                text += f"; document_path={result.get('path', args.get('path', ''))}"
            else:
                text = str(result.get("path", args.get("path", "document read")))
        elif tool == "read_policy":
            policy = result.get("policy") or {}
            text = f"thresholds={policy.get('thresholds', {})}; approved vendors={policy.get('approved_vendors', [])}"
        else:
            snapshot = result.get("snapshot") or {}
            path = snapshot.get("url_path") or urlparse(str(snapshot.get("url", ""))).path
            alerts = snapshot.get("alerts") or ([snapshot.get("alert")] if snapshot.get("alert") else [])
            text = ", ".join(part for part in (
                f"path={path}" if path else "",
                f"status={result.get('status')}" if result.get("status") else "",
                f"alert={'; '.join(str(alert) for alert in alerts)}" if alerts else "",
                f"session={snapshot.get('session_required')}" if snapshot.get("session_required") else "",
                f"warning={result.get('warning')}" if result.get("warning") else "",
                f"fields={','.join(str(field.get('selector')) + ':' + str(field.get('state')) for field in snapshot.get('form_fields', []) if isinstance(field, dict))}" if snapshot.get("form_fields") else "",
                f"message={result.get('message')}" if result.get("message") else "",
                f"args={json.dumps(args, ensure_ascii=False, sort_keys=True)}" if args else "",
            ) if part)
        return _clean_text(f"{tool}: {text or result.get('status', 'observed')}", 240).replace("\n", " ")

    @staticmethod
    def _snapshot_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
        snapshot = result.get("snapshot")
        if isinstance(snapshot, dict):
            return snapshot
        if any(key in result for key in ("url_path", "form_fields", "alerts", "tables")):
            return result
        return None

    @staticmethod
    def _page_signature(snapshot: dict[str, Any] | None) -> str | None:
        if snapshot is None:
            return None
        state = {key: snapshot.get(key) for key in (
            "url_path", "page_title", "alerts", "notice", "form_fields", "buttons", "tables", "session_required"
        )}
        if not state.get("url_path"):
            state["url_path"] = urlparse(str(snapshot.get("url", ""))).path
        if not state.get("page_title"):
            state["page_title"] = snapshot.get("title", "")
        if not state.get("alerts"):
            state["alerts"] = [snapshot.get("alert")] if snapshot.get("alert") else []
        return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _last_alert(snapshot: dict[str, Any] | None) -> str:
        if not snapshot:
            return "none"
        alerts = snapshot.get("alerts")
        if isinstance(alerts, list) and alerts:
            return _clean_text(alerts[-1], 300)
        return _clean_text(snapshot.get("alert"), 300) or "none"

    @staticmethod
    def _action_label(name: str, arguments: dict[str, Any]) -> str:
        if name == "browser_click":
            return f"{name}({arguments.get('selector', 'unknown selector')})"
        if name == "browser_open":
            return f"{name}({arguments.get('path', 'unknown route')})"
        return f"{name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':'))})"

    def _provider_recovery(self, payload: dict[str, Any]) -> None:
        if payload.get("action") == "retry_wait":
            try:
                wait_seconds = float(payload.get("wait_seconds", 0))
            except (TypeError, ValueError):
                wait_seconds = 0.0
            if math.isfinite(wait_seconds) and wait_seconds > 0:
                self.provider_rate_limit_wait_seconds += wait_seconds
        self.emit("rate_limit_recovery", payload)
        action = payload.get("action")
        if action == "provider_fallback":
            self.emit("planner", {"mode": "live", "provider": payload.get("provider"), "model": payload.get("model")})

    def _emit_provider_wait_summary(self) -> None:
        if self._provider_wait_summary_emitted or self.provider_rate_limit_wait_seconds <= 60:
            return
        self._provider_wait_summary_emitted = True
        seconds = round(self.provider_rate_limit_wait_seconds, 1)
        self.emit("provider_wait_summary", {
            "total_wait_seconds": seconds,
            "message": f"Provider rate limits added {seconds:g}s of waiting to this run.",
        })

    def _provider_answered(self, payload: dict[str, Any]) -> None:
        self.emit("provider_turn", payload)

    @staticmethod
    def _is_recoverable_provider_tool_error(error: ProviderRequestError) -> bool:
        if error.status_code != 400:
            return False
        code = (error.error_code or "").casefold().replace("-", "_")
        if code in {"tool_use_failed", "output_parse_failed", "invalid_tool_call", "tool_validation_failed", "schema_validation_error"}:
            return True
        message = (error.response_body or "").casefold().replace("-", "_")
        return "tool_use_failed" in message or "output_parse_failed" in message or (
            any(token in message for token in ("schema", "validation"))
            and any(token in message for token in ("tool", "function", "argument", "parameter"))
        ) or "failed to call a function" in message or "invalid tool arguments" in message

    @staticmethod
    def _schemas_for_feedback(tool_name: str | None, error_text: str) -> list[dict[str, Any]]:
        if tool_name and tool_name in TOOL_BY_NAME:
            return [TOOL_BY_NAME[tool_name]]
        lowered = error_text.casefold()
        matching = [tool for tool in TOOL_DEFINITIONS if tool["name"].casefold() in lowered]
        return matching or TOOL_DEFINITIONS

    def _append_tool_validation_feedback(
        self,
        *,
        retry_number: int,
        tool_name: str | None,
        error_text: str,
    ) -> None:
        schemas = self._schemas_for_feedback(tool_name, error_text)
        feedback = {
            "tool": "tool_validation_feedback",
            "arguments": {},
            "result": {
                "status": "retrying",
                "error": error_text,
                "retry": retry_number,
                "retry_limit": MAX_TOOL_VALIDATION_RETRIES,
                "tool_schemas": schemas,
            },
        }
        self.observations.append(feedback)
        self.emit("observation", {
            "tool": "tool_validation_feedback",
            "error": error_text,
            "retry": retry_number,
            "retry_limit": MAX_TOOL_VALIDATION_RETRIES,
            "tool_schemas": schemas,
        })

    @staticmethod
    def _redact_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        safe = dict(arguments)
        host_credentials = {"#email_input", "#password_input"}
        sensitive = {"#approval_token", "#source_sha256_input", "#source_file_input"}
        if name == "browser_fill" and safe.get("selector") in host_credentials:
            safe["value"] = "[ignored; use erp_login]"
        elif name == "browser_fill" and safe.get("selector") in sensitive:
            safe["value"] = "[attached by host]" if safe.get("selector") in {"#source_sha256_input", "#source_file_input"} else "[redacted]"
        elif name == "browser_fill_form":
            fields = safe.get("fields")
            if isinstance(fields, list):
                safe["fields"] = [
                    {**field, "value": "[ignored; use erp_login]"}
                    if isinstance(field, dict) and field.get("selector") in host_credentials
                    else {**field, "value": "[attached by host]"}
                    if isinstance(field, dict) and field.get("selector") in {"#source_file_input", "#source_sha256_input"}
                    else {**field, "value": "[redacted]"}
                    if isinstance(field, dict) and field.get("selector") in sensitive
                    else field
                    for field in fields
                ]
        elif name == "browser_action_sequence":
            actions = safe.get("actions")
            if isinstance(actions, list):
                safe["actions"] = [
                    {**item, "arguments": AgentLoop._redact_arguments(str(item.get("tool", "")), item.get("arguments", {}))}
                    if isinstance(item, dict) and isinstance(item.get("arguments"), dict) else item
                    for item in actions
                ]
        for key in list(safe):
            if key.casefold() in {"password", "passwd", "secret", "api_key", "access_token"}:
                safe[key] = "[redacted]"
        return safe

    def _invoice_by_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        invoice_number = str(fields.get("invoice_number", ""))
        document_path = str(fields.get("document_path", ""))
        candidates = [
            (path, document["invoice"]) for path, document in self.documents.items()
            if document.get("invoice")
            and document["invoice"]["invoice_number"] == invoice_number
            and all(
                key not in fields or str(fields[key]) == str(document["invoice"][key])
                for key in ("vendor", "invoice_date", "due_date", "amount", "currency")
            )
            and (not document_path or path == document_path)
        ]
        if len(candidates) != 1:
            raise ValueError("Select one read invoice by its content; include its read_document path if identical documents are ambiguous.")
        self.selected_invoice_path, invoice = candidates[0]
        for key in ("vendor", "invoice_date", "due_date", "amount", "currency"):
            if str(fields.get(key, "")) != str(invoice[key]):
                raise ValueError(f"{key.replace('_', ' ')} must match the selected source document.")
        self.selected_invoice = invoice
        self.browser.set_invoice(invoice)
        return invoice

    def _read_invoice_candidates(self) -> str:
        items = []
        for path, document in self.documents.items():
            invoice = document.get("invoice")
            if invoice:
                items.append(
                    f"{invoice.get('invoice_number', '?')} | {path} | "
                    f"{invoice.get('currency', '')} {invoice.get('amount', '')}".strip()
                )
        return "Invoices read so far (number | path | amount): " + ("; ".join(items) if items else "none") + "."

    def _validate_vendor_evidence(self, legal_vendor: str) -> None:
        observed = {row[2] for row in self.browser.vendor_rows if len(row) >= 3}
        if legal_vendor not in observed:
            raise ValueError("The legal vendor must come from an observed ERP vendor directory row.")
        if self.selected_invoice:
            source_terms = set(re.findall(r"[a-z0-9]+", self.selected_invoice["vendor"].casefold()))
            row = next(row for row in self.browser.vendor_rows if len(row) >= 3 and row[2] == legal_vendor)
            directory_terms = set(re.findall(r"[a-z0-9]+", " ".join(row[1:3]).casefold()))
            if not source_terms.intersection(directory_terms):
                raise ValueError("The observed vendor directory entry does not correspond to the source vendor label.")

    def _attach_directory_vendor(self, invoice: dict[str, Any]) -> None:
        """Resolve a source label to one matching legal entity from the observed directory."""
        if invoice.get("canonical_vendor"):
            return
        source_terms = set(re.findall(r"[a-z0-9]+", str(invoice.get("vendor", "")).casefold()))
        matches = {
            str(row[2]) for row in getattr(self.browser, "vendor_rows", [])
            if len(row) >= 3
            and source_terms.intersection(set(re.findall(r"[a-z0-9]+", " ".join(map(str, row[1:3])).casefold())))
        }
        if len(matches) == 1:
            candidate = matches.pop()
            self._validate_vendor_evidence(candidate)
            invoice["canonical_vendor"] = candidate

    async def _execute(
        self, name: str, arguments: dict[str, Any], *, include_snapshot: bool = True
    ) -> dict[str, Any]:
        if name in {"browser_fill", "browser_fill_form", "browser_action_sequence", "browser_check", "browser_click", "request_human_approval"}:
            self._require_write_intent()
        if name == "list_files":
            files = await asyncio.to_thread(list_workspace_files, self.data_dir)
            self.listed_files = {item["path"] for item in files}
            return {"status": "listed", "files": files}
        if name == "read_document":
            path = _clean_text(arguments.get("path"), 500)
            if path not in self.listed_files:
                raise ValueError("Read only documents returned by list_files.")
            result = await asyncio.to_thread(read_document, self.data_dir, path)
            self.documents[path] = result
            if result.get("prompt_injection_detected") and path not in self.injection_flagged_paths:
                self.injection_flagged_paths.add(path)
                self.emit("prompt_injection_detected", {
                    "path": path,
                    "message": "Instruction-like invoice text was detected and ignored; structured invoice fields remain authoritative.",
                })
            selected_row = next((item for item in self.candidate_invoices if item.get("path") == path), None)
            if selected_row and result.get("invoice"):
                self.selected_invoice_path = path
                self.selected_invoice = result["invoice"]
                self.browser.set_invoice(self.selected_invoice)
                if self.task_intent and self.task_intent.get("intent") == "write_invoice_entry":
                    self._validate_selection_declaration(
                        str(self.task_intent.get("selection_criterion", "other")),
                        str(self.task_intent.get("reason", "Selected from the discovered invoice candidates.")),
                    )
            model_result = {key: value for key, value in result.items() if key != "sha256"}
            model_result["content"] = remove_prompt_injection_lines(str(result.get("content", "")))
            model_result.pop("prompt_injection_detected", None)
            if isinstance(result.get("invoice"), dict):
                model_result["invoice"] = {
                    key: value for key, value in result["invoice"].items()
                    if key not in {"source_file", "source_sha256"}
                }
            return model_result
        if name == "read_policy":
            result = await asyncio.to_thread(self.company.read_company_policy)
            self.policy = result
            return {"status": "read", "policy": result}
        if name == "list_invoice_candidates":
            if self.task_intent and self.task_intent.get("intent") == "write_invoice_entry":
                self._require_write_intent()
            elif not any(
                isinstance(item, dict) and item.get("capability") == "list_candidates"
                for item in (self.task_intent or {}).get("requested_actions", [])
            ):
                raise ValueError("Candidate listing is available only for invoice entry or a host-classified list_candidates request.")
            vendor_query = _clean_text(arguments.get("vendor_query"), 160)
            if not vendor_query:
                raise ValueError("Provide the vendor query from the task intent.")
            if _normalized(vendor_query) != _normalized(str((self.task_intent or {}).get("vendor_query", ""))):
                raise ValueError("vendor_query must match the vendor identified by the host intent classification.")
            all_rows = await asyncio.to_thread(list_invoice_candidates, self.data_dir, vendor_query)
            for row in all_rows:
                path = row["path"]
                self.listed_files.add(path)
                document = await asyncio.to_thread(read_document, self.data_dir, path)
                if document.get("invoice"):
                    self.documents[path] = document
                    if document.get("prompt_injection_detected") and path not in self.injection_flagged_paths:
                        self.injection_flagged_paths.add(path)
                        self.emit("prompt_injection_detected", {
                            "path": path,
                            "message": "Instruction-like invoice text was detected and ignored; structured invoice fields remain authoritative.",
                        })
            self.candidate_invoices = all_rows
            self.candidate_listed = True
            return {"status": "listed", "vendor_query": vendor_query, "candidates": all_rows}
        if name == "read_ledger":
            if not self.task_intent or self.task_intent.get("intent") != "read_only":
                raise ValueError("read_ledger is available only for a host-classified read_only task.")
            if not any(
                isinstance(item, dict) and item.get("capability") == "read_ledger"
                for item in self.task_intent.get("requested_actions", [])
            ):
                raise ValueError("read_ledger is available only when the host classified that requested capability.")
            vendor_query = _normalized(_clean_text(arguments.get("vendor_query"), 160))
            classified_vendor = _normalized(str(self.task_intent.get("vendor_query", "")))
            if not vendor_query or vendor_query != classified_vendor:
                raise ValueError("vendor_query must match the vendor identified by the host intent classification.")
            bills = await asyncio.to_thread(self.company.list_bills)
            matches = [
                bill for bill in bills
                if vendor_query and (
                    vendor_query in _normalized(str(bill.get("vendor", "")))
                    or _normalized(str(bill.get("vendor", ""))) in vendor_query
                )
            ]
            self.read_ledger_rows = matches
            self.read_ledger_vendor_query = str(self.task_intent.get("vendor_query", ""))
            return {"status": "read", "vendor_query": self.read_ledger_vendor_query, "bills": matches}
        if name == "browser_open":
            path = _clean_text(arguments.get("path"), 100)
            if self.task_intent and self.task_intent.get("intent") == "read_only":
                raise ValueError("Read-only tasks use read_ledger and cannot navigate the ERP write interface.")
            if path == "/erp/invoices/new":
                self._require_write_intent()
            if path == "/erp/invoices/new" and self.selected_invoice and self._needs_approval() and not self._invoice_is_approved():
                raise ValueError(
                    "Approval required: call request_human_approval with invoice_number "
                    f"{self.selected_invoice['invoice_number']}."
                )
            if path == "/erp/invoices/new" and self.selected_invoice:
                self.browser.set_invoice(self.selected_invoice)
            result = await self.browser.browser_open(path)
            snapshot = self._snapshot_from_result(result)
            if snapshot is not None:
                self._last_page_snapshot = snapshot
            if result.get("snapshot") and path == "/erp/invoices/new":
                result["form_state"] = self._form_state(result["snapshot"])
                result["next_step"] = result["form_state"]["next_step"]
            if result.get("provenance_attached") and self.selected_invoice:
                invoice = self.selected_invoice
                document_path = next((doc_path for doc_path, doc in self.documents.items() if doc.get("invoice") is invoice), "")
                self.emit("provenance_attached", {
                    "message": "Host attached source_file and source SHA-256 from the selected read document.",
                    "document_path": document_path,
                    "source_file": invoice["source_file"],
                    "source_sha256": invoice["source_sha256"],
                    "invoice_number": invoice["invoice_number"],
                })
            return result
        if name == "erp_login":
            self._require_write_intent()
            self.emit("credentials_supplied", {
                "message": "Credentials supplied by host for ERP sign-in; the model never receives their values."
            })
            return await self.browser.erp_login()
        if name == "browser_snapshot":
            result = await self.browser.browser_snapshot()
            snapshot = self._snapshot_from_result(result)
            if snapshot is not None:
                self._last_page_snapshot = snapshot
            if self.selected_invoice and self.policy and self._needs_approval() and not self._invoice_is_approved():
                result["approval_hint"] = (
                    "Approval required: call request_human_approval with invoice_number "
                    f"{self.selected_invoice['invoice_number']}."
                )
                if result.get("snapshot"):
                    result["snapshot"]["approval_hint"] = result["approval_hint"]
            return result
        if name == "browser_fill":
            selector = _clean_text(arguments.get("selector"), 80)
            try:
                if self.selected_invoice:
                    expected_fields = {
                        "#invoice_number_input": "invoice_number", "#invoice_date_input": "invoice_date",
                        "#due_date_input": "due_date", "#amount_input": "amount", "#currency_input": "currency",
                    }
                    expected_key = expected_fields.get(selector)
                    if expected_key and _clean_text(arguments.get("value"), 600) != str(self.selected_invoice.get(expected_key, "")):
                        raise ValueError(f"{expected_key.replace('_', ' ').capitalize()} must come from the parsed structured fields of the selected invoice.")
                result = await self.browser.browser_fill(selector, _clean_text(arguments.get("value"), 600))
            except Exception as exc:
                result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            if include_snapshot:
                snapshot = await self.browser.compact_snapshot("locator.fill", selector)
                result["snapshot"] = snapshot
                self._last_page_snapshot = snapshot
                result["form_state"] = self._form_state(snapshot)
                result["next_step"] = result["form_state"]["next_step"]
            return result
        if name == "browser_fill_form":
            fields = arguments.get("fields")
            if not isinstance(fields, list) or not 1 <= len(fields) <= 10:
                raise ValueError("browser_fill_form requires one to ten field entries.")
            prepared: list[tuple[str, str]] = []
            seen = set()
            for field in fields:
                if not isinstance(field, dict) or set(field) != {"selector", "value"}:
                    allowed = ", ".join(sorted(MODEL_FORM_SELECTORS))
                    raise ValueError(
                        f"Each form field requires only selector and value. Allowed ERP fields: {allowed}."
                    )
                selector = _clean_text(field["selector"], 80)
                if selector in seen:
                    raise ValueError("A form control may appear only once in a fill batch.")
                seen.add(selector)
                prepared.append((selector, _clean_text(field["value"], 600)))

            fields_set: list[str] = []
            alert = None
            error = None
            for selector, value in prepared:
                call = ToolCall("browser_fill", {"selector": selector, "value": value})
                self._event_action(call)
                try:
                    result = await self._execute(call.name, call.arguments, include_snapshot=False)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
                if result.get("status") == "error":
                    error = str(result.get("message", "field fill failed"))
                    break
                fields_set.append(selector)
                alert = result.get("alert")
                if alert:
                    break
            result = {
                "status": "error" if error else "filled_form" if not alert else "stopped_on_alert",
                "fields_set": fields_set,
                "alert": alert,
            }
            if include_snapshot:
                snapshot = await self.browser.compact_snapshot("browser_fill_form")
                self._last_page_snapshot = snapshot
                result["alert"] = alert or snapshot.get("alert")
                result["snapshot"] = snapshot
                result["form_state"] = self._form_state(snapshot)
                result["next_step"] = result["form_state"]["next_step"]
            if error:
                result["message"] = f"The form batch stopped at the first rejected field: {error}"
            return result
        if name == "browser_action_sequence":
            actions = arguments.get("actions")
            if not isinstance(actions, list) or len(actions) not in {2, 3}:
                raise ValueError("An action sequence must contain two or three actions.")
            fill_actions = {"browser_fill", "browser_check", "browser_fill_form"}
            validated_actions: list[ToolCall] = []
            for item in actions:
                if not isinstance(item, dict) or set(item) != {"tool", "arguments"} or not isinstance(item.get("arguments"), dict):
                    raise ValueError("Each sequence action requires only a tool and object arguments.")
                child = ToolCall(str(item.get("tool", "")), item.get("arguments", {}))
                if child.name not in fill_actions:
                    raise ValueError("A fill batch may contain only browser_fill, browser_check, and browser_fill_form; navigation and clicks are single-step actions.")
                self._validate_arguments(child)
                validated_actions.append(child)
            fields_set: list[str] = []
            checks: list[str] = []
            actions_completed: list[str] = []
            alert = None
            error = None
            for child in validated_actions:
                self._event_action(child)
                try:
                    result = await self._execute(child.name, child.arguments, include_snapshot=False)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
                if result.get("status") == "error":
                    error = result.get("message", "action failed")
                    break
                actions_completed.append(child.name)
                if child.name in {"browser_fill", "browser_fill_form"}:
                    if child.name == "browser_fill":
                        fields_set.append(str(child.arguments.get("selector", "")))
                    else:
                        fields_set.extend(str(field.get("selector", "")) for field in child.arguments.get("fields", []) if isinstance(field, dict))
                elif child.name == "browser_check":
                    checks.append(str(child.arguments.get("selector", "")))
                alert = result.get("alert")
                if alert:
                    break
            snapshot = await self.browser.compact_snapshot("browser_action_sequence")
            self._last_page_snapshot = snapshot
            result = {
                "status": "error" if error else "sequence_stopped" if alert else "sequence_complete",
                "actions_completed": actions_completed,
                "fields_set": fields_set,
                "checks_completed": checks,
                "alert": alert or snapshot.get("alert"),
                "snapshot": snapshot,
            }
            if any(child.name in {"browser_fill", "browser_fill_form"} for child in validated_actions):
                result["form_state"] = self._form_state(snapshot)
                result["next_step"] = result["form_state"]["next_step"]
            if error:
                result["message"] = f"The browser sequence stopped at the first rejected action: {error}."
            elif alert:
                result["reason"] = "A visible ERP alert requires a separate observation/action call."
            return result
        if name == "browser_check":
            selector = _clean_text(arguments.get("selector"), 80)
            result = await self.browser.browser_check(selector)
            if include_snapshot:
                result["snapshot"] = await self.browser.compact_snapshot("locator.check", selector)
            return result
        if name == "browser_click":
            selector = _clean_text(arguments.get("selector"), 80)
            if selector == "#submit_btn":
                if not self.selected_invoice or not self.policy:
                    raise ValueError("Read the invoice and procurement policy before a ledger write.")
                criterion = _clean_text(arguments.get("selection_criterion"), 60)
                selection_reason = _clean_text(arguments.get("selection_reason"), 240)
                self._validate_selection_declaration(criterion, selection_reason)
                if not self.duplicate_checked:
                    raise ValueError("Read the ERP ledger for an existing invoice before a write.")
                if self.duplicate_present:
                    raise ValueError("The invoice is already present in the ERP ledger; duplicate submission is blocked.")
                if not self._vendor_is_approved():
                    raise ValueError("The selected legal vendor is not approved by procurement policy.")
                if self._needs_approval() and not self._invoice_is_approved():
                    raise ValueError(
                        "Approval required: call request_human_approval with invoice_number "
                        f"{self.selected_invoice['invoice_number']}."
                    )
                if self.submit_attempts >= MAX_SUBMIT_ATTEMPTS:
                    raise ValueError("The ERP submit retry limit has been reached.")
                fresh_snapshot_result = await self.browser.browser_snapshot()
                fresh_snapshot = self._snapshot_from_result(fresh_snapshot_result)
                if fresh_snapshot is not None:
                    self._last_page_snapshot = fresh_snapshot
                form_state = self._form_state(fresh_snapshot or {})
                if not form_state.get("all_required_filled"):
                    return {
                        "status": "error",
                        "message": "Cannot submit the invoice: required form fields are empty. " + form_state["next_step"],
                        "snapshot": fresh_snapshot or {},
                        "form_state": form_state,
                        "next_step": form_state["next_step"],
                    }
                if self._needs_approval():
                    token = self.approval_tokens.get(self._invoice_key() or "")
                    if not token:
                        raise ValueError(
                            "Approval required: call request_human_approval with invoice_number "
                            f"{self.selected_invoice['invoice_number']}."
                        )
                    await self.browser.inject_approval_token(token)
                    self.emit("approval_token_injected", {
                        "invoice_number": self.selected_invoice["invoice_number"],
                        "message": "Host injected the approved token and checked the verification control immediately before submit.",
                    })
                self.submit_attempts += 1
            result = await self.browser.browser_click(selector)
            snapshot = self._snapshot_from_result(result)
            if snapshot is not None:
                self._last_page_snapshot = snapshot
            if selector == "#submit_btn":
                self.submit_performed = True
                result["submit_performed"] = True
            if selector == "#submit_btn" and result.get("snapshot", {}).get("alert"):
                alert = result["snapshot"]["alert"]
                self.emit("observation", {"tool": "erp_submit_alert", "alert": alert})
                if "vendor" in alert.casefold():
                    self.emit("self_correction", {"text": "The ERP rejected the submitted vendor value. Inspect the vendor directory and adapt from its observed row."})
            return result
        if name == "request_human_approval":
            self._require_write_intent()
            if not self.policy:
                raise ValueError("Read procurement policy before requesting approval.")
            invoice_number = _clean_text(arguments.get("invoice_number"), 100)
            document_path = _clean_text(arguments.get("document_path"), 500)
            if not invoice_number and not document_path:
                raise ValueError("Provide invoice_number or document_path. " + self._read_invoice_candidates())
            candidates = [
                (path, document["invoice"]) for path, document in self.documents.items()
                if document.get("invoice")
                and (not invoice_number or document["invoice"].get("invoice_number") == invoice_number)
                and (not document_path or path == document_path)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Approval target is missing or ambiguous. Include invoice_number or a unique document_path. "
                    + self._read_invoice_candidates()
                )
            self.selected_invoice_path, invoice = candidates[0]
            if not any(row.get("path") == self.selected_invoice_path for row in self.candidate_invoices):
                raise ValueError("Approval can be requested only for an invoice selected from list_invoice_candidates.")
            self.selected_invoice = invoice
            criterion = _clean_text(arguments.get("selection_criterion"), 60)
            selection_reason = _clean_text(arguments.get("selection_reason"), 240)
            self._validate_selection_declaration(criterion, selection_reason)
            invoice_key = self._invoice_key() or ""
            self.browser.set_invoice(invoice)
            if invoice_key in self.approval_tokens:
                return {"status": "already_approved", "invoice_number": invoice["invoice_number"]}
            canonical = _clean_text(arguments.get("canonical_vendor"), 200)
            if not canonical and invoice.get("vendor") in {row[2] for row in self.browser.vendor_rows if len(row) >= 3}:
                canonical = invoice["vendor"]
            if canonical:
                self._validate_vendor_evidence(canonical)
                invoice["canonical_vendor"] = canonical
            else:
                raise ValueError("Choose canonical_vendor from the observed ERP vendor directory before approval.")
            required, reason = self._approval_required(invoice)
            if not required and self.approval_mode != "always_gate":
                return {"status": "not_required", "reason": "The selected invoice is below its configured policy threshold."}
            reason = reason or "Always Gate mode requires human approval for every invoice write."
            payload = {
                "reason": reason,
                "policy_reasons": [reason],
                "selection_criterion": criterion,
                "selection_reason": selection_reason,
                "invoice": {**invoice, "canonical_vendor": canonical},
                "source_file": invoice["source_file"],
                "source_sha256": invoice["source_sha256"],
            }
            token = await self._await_with_long_operation_trace(
                "human_approval_wait", self.approval_handler(payload),
                details={"invoice_number": invoice["invoice_number"]},
            )
            self.approval_record = {
                "status": "approved" if token else "denied_or_timed_out",
                "invoice_number": invoice["invoice_number"],
                "reason": reason,
                "selection_criterion": criterion,
                "selection_reason": selection_reason,
            }
            if token:
                self.approval_tokens[invoice_key] = str(token)
                self.denied_invoices.discard(invoice_key)
            else:
                self.denied_invoices.add(invoice_key)
            self.browser.set_invoice(invoice)
            return {"status": "approved" if token else "denied_or_timed_out", "invoice_number": invoice["invoice_number"]}
        if name == "verify_ledger_row":
            invoice_number = _clean_text(arguments.get("invoice_number"), 100)
            if invoice_number:
                candidates = [
                    (path, document["invoice"]) for path, document in self.documents.items()
                    if document.get("invoice") and str(document["invoice"].get("invoice_number")) == invoice_number
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Verification invoice is missing or ambiguous. " + self._read_invoice_candidates()
                    )
                if (
                    getattr(self.browser, "write_submitted", False)
                    and self.selected_invoice
                    and invoice_number != str(self.selected_invoice.get("invoice_number"))
                ):
                    raise ValueError(
                        f"This run submitted {self.selected_invoice.get('invoice_number')}; verify that invoice instead."
                    )
                self.selected_invoice_path, invoice = candidates[0]
                self.selected_invoice = invoice
            elif not self.selected_invoice:
                raise ValueError("Read an invoice source before calling verify_ledger_row.")
            else:
                invoice = self.selected_invoice
            if self.task_intent and self.task_intent.get("intent") == "write_invoice_entry":
                if not any(item.get("path") == self.selected_invoice_path for item in self.candidate_invoices):
                    raise ValueError("Verification requires an invoice selected from list_invoice_candidates.")
                if not self.selection_criterion:
                    self._validate_selection_declaration(
                        str(self.task_intent.get("selection_criterion", "other")),
                        str(self.task_intent.get("reason", "Selected from the discovered invoice candidates.")),
                    )
            self._attach_directory_vendor(invoice)
            self.browser.set_invoice(invoice)
            result = await self.browser.verify_ledger_row(invoice)
            if self.selection_criterion:
                result["selection_check"] = {
                    "status": "passed",
                    "criterion": self.selection_criterion,
                    "reason": self.selection_reason,
                }
            self.verification = result if result.get("status") in {"verified", "mismatch"} else None
            if result.get("status") == "verified" and result.get("already_present"):
                self.duplicate_checked, self.duplicate_present = True, True
            elif result.get("status") == "not_found":
                self.duplicate_checked = True
                result["submit_performed"] = self.submit_performed
                if not self.submit_performed:
                    result["message"] = (
                        "No ledger row was found. No submit was performed. "
                        "Open /erp/invoices/new, fill, click #submit_btn."
                    )
                else:
                    result["message"] = "No ledger row was found after the invoice form was submitted. Inspect the ERP alert before retrying."
            elif result.get("status") == "mismatch":
                self.duplicate_checked, self.duplicate_present = True, True
            if result.get("status") == "verified" and result.get("pass"):
                approval_record = self.approval_record or {
                    "status": "not_applicable_existing_ledger_row" if result.get("already_present") else "not_required"
                }
                what_was_done = (
                    f"Verified existing ledger row for invoice {invoice['invoice_number']}."
                    if result.get("already_present")
                    else f"Submitted invoice {invoice['invoice_number']} and verified its ledger row."
                )
                self.final_result = {
                    "status": "complete",
                    "what_was_done": what_was_done,
                    "selected_invoice": {
                        "invoice_number": invoice.get("invoice_number"),
                        "vendor": invoice.get("vendor"),
                        "invoice_date": invoice.get("invoice_date"),
                        "due_date": invoice.get("due_date"),
                        "amount": invoice.get("amount"),
                        "currency": invoice.get("currency"),
                        "path": self.selected_invoice_path,
                    },
                    "candidates_considered": self.candidate_invoices,
                    "selection_criterion": self.selection_criterion,
                    "selection_reason": self.selection_reason,
                    "evidence": {
                        "ledger_row_id": result.get("ledger_row_id"),
                        "ledger_screenshot_path": result.get("screenshot_path"),
                        "approval_record": approval_record,
                        "source_file": invoice.get("source_file"),
                        "source_sha256": invoice.get("source_sha256"),
                        "selection_check": result.get("selection_check"),
                    },
                    "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 2),
                }
            return result
        if name == "finish":
            status = _clean_text(arguments.get("status"), 40)
            summary = _clean_text(arguments.get("summary"), 1000)
            if status not in STOP_STATUSES or not summary:
                raise ValueError("finish requires a supported stop status and a concise summary.")

            if self.task_intent and self.task_intent.get("intent") == "unsupported":
                status = "needs_input"
                summary = self._unsupported_task_summary()
            elif self.task_intent and self.task_intent.get("intent") == "ambiguous":
                status = "needs_input"
                summary = str(self.task_intent.get("clarifying_question") or "Please clarify which vendor or invoice you mean.")

            if status == "complete" and self.task_intent and self.task_intent.get("intent") == "read_only":
                capabilities = {
                    item.get("capability")
                    for item in self.task_intent.get("requested_actions", [])
                    if isinstance(item, dict)
                }
                if "read_ledger" in capabilities and self.read_ledger_rows is None:
                    raise ValueError("Read-only completion requires a host ledger lookup before finish.")
                if "list_candidates" in capabilities and not self.candidate_listed:
                    raise ValueError("Candidate-list completion requires the host candidate lookup before finish.")
                if not capabilities:
                    raise ValueError("Read-only completion requires a host ledger or candidate lookup before finish.")
                answers: dict[str, Any] = {}
                if "read_ledger" in capabilities:
                    answers["ledger_answer"] = self._render_read_ledger_answer()
                if "list_candidates" in capabilities:
                    answers["candidate_answer"] = self._render_candidate_answer()
                summary = "; ".join(answer["text"] for answer in answers.values())
                primary = next(iter(answers.values()))
                self.final_result = {
                    "status": "complete",
                    "what_was_done": f"Read requested invoice data for {primary['vendor']}.",
                    **answers,
                    "evidence": {},
                    "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 2),
                }
            if (
                status == "complete"
                and not (self.task_intent and self.task_intent.get("intent") == "read_only")
                and not (self.verification and self.verification.get("status") == "verified")
            ):
                raise ValueError("A complete run requires a matching read-back from the live ERP ledger DOM.")
            if status == "complete" and self.task_intent and self.task_intent.get("intent") == "write_invoice_entry":
                if not self.verification or not self.verification.get("pass") or not self.final_result:
                    raise ValueError("A complete invoice write requires a passed ledger verification before finish.")
                summary = str(self.final_result["what_was_done"])
            status, summary = self._finalize_run_result(status, summary)
            result = self.final_result or {}
            return {
                "status": "finished", "final_status": status, "summary": summary,
                "answer": result.get("ledger_answer"), "evidence": result.get("evidence", self.verification or {}),
                "not_done_actions": result.get("not_done_actions", []),
            }
        raise ValueError(f"Tool is not in the runtime allowlist: {name}")

    def _approval_required(self, invoice: dict[str, Any]) -> tuple[bool, str | None]:
        if self.approval_mode == "always_gate":
            return True, "Always Gate mode requires human approval for every invoice write."
        return approval_required(invoice, self.policy or {})

    def _needs_approval(self) -> bool:
        return bool(self.selected_invoice and self._approval_required(self.selected_invoice)[0])

    def _vendor_is_approved(self) -> bool:
        if not self.selected_invoice or not self.policy:
            return False
        vendor = self.selected_invoice.get("canonical_vendor")
        if not vendor:
            return False
        try:
            self._validate_vendor_evidence(str(vendor))
        except ValueError:
            return False
        return _normalized(vendor) in {_normalized(item) for item in self.policy.get("approved_vendors", [])}

    def _record_tool_failure(self, name: str, error: str) -> int:
        """Count matching tool/error failures over the entire run, regardless of intervening events."""
        key = (name, " ".join(error.split()))
        count = self.tool_failure_counts.get(key, 0) + 1
        self.tool_failure_counts[key] = count
        self.emit("tool_retry_budget", {
            "tool": name,
            "failure_count": count,
            "limit": MAX_REPEATED_TOOL_ERRORS,
            "remaining": max(0, MAX_REPEATED_TOOL_ERRORS - count),
            "error": error,
        })
        return count

    def _append_repeat_failure_warning(self, name: str, error: str, count: int) -> None:
        warning = (
            f"Repeated failure {count}/{MAX_REPEATED_TOOL_ERRORS} for {name}: {error}. "
            "Change the call using this error feedback; do not repeat the same failing request."
        )
        self.observations.append({"tool": "repeat_failure_warning", "arguments": {}, "result": {"warning": warning}})
        self.emit("tool_retry_warning", {"tool": name, "failure_count": count, "limit": MAX_REPEATED_TOOL_ERRORS, "error": error, "warning": warning})

    def _tool_failure_summary(self, name: str, error: str) -> str:
        done = []
        not_done = []
        if self.selected_invoice:
            done.append(f"selected invoice {self.selected_invoice.get('invoice_number', '')}")
        else:
            not_done.append("invoice selection")
        if self._invoice_is_approved():
            done.append("human approval")
        elif self._needs_approval():
            not_done.append("human approval")
        if self.submit_performed:
            done.append("invoice form submitted")
        else:
            not_done.append("invoice submission")
        if self.verification and self.verification.get("status") == "verified":
            done.append("ledger read-back verified")
        else:
            not_done.append("ledger read-back verification")
        done_text = ", ".join(done) if done else "none"
        not_done_text = ", ".join(not_done) if not_done else "none"
        return (
            f"Stopped after {MAX_REPEATED_TOOL_ERRORS} failures for {name} with the same error. "
            f"Last error: {error}. Done: {done_text}. Not done: {not_done_text}."
        )

    def _provider_tool_error_text(self, error: ProviderRequestError) -> str:
        body = error.response_body or ""
        try:
            payload = json.loads(body)
            detail = payload.get("error", payload) if isinstance(payload, dict) else {}
            message = detail.get("message") if isinstance(detail, dict) else None
            code = detail.get("code") if isinstance(detail, dict) else None
            text = f"HTTP {error.status_code} {code or error.error_code or 'tool validation error'}: {message or str(error)}"
        except (TypeError, ValueError):
            text = f"HTTP {error.status_code} {error.error_code or 'tool validation error'}: {str(error)}"
        return self._safe_validation_error(text, body)

    def _plain_text_from_provider_error(self, error: ProviderRequestError) -> str | None:
        code = (error.error_code or "").casefold().replace("-", "_")
        if error.status_code != 400 or code != "tool_use_failed":
            return None
        generation = error.failed_generation
        if not generation:
            try:
                payload = json.loads(error.response_body or "")
            except (TypeError, ValueError):
                payload = {}
            if isinstance(payload, dict):
                detail = payload.get("error") if isinstance(payload.get("error"), dict) else {}
                for source in (detail, payload):
                    for key in ("failed_generation", "generated_text", "completion"):
                        value = source.get(key)
                        if isinstance(value, str) and value.strip():
                            generation = value.strip()
                            break
                    if generation:
                        break
        if not isinstance(generation, str) or not generation.strip():
            return None
        safe = self._safe_validation_error(generation.strip(), generation.strip())
        return safe[:MAX_FAILED_GENERATION_CHARS]

    def _stop_for_repeated_failure(self, name: str, error: str, count: int) -> tuple[str, str] | None:
        if count < MAX_REPEATED_TOOL_ERRORS:
            return None
        reason = self._tool_failure_summary(name, error)
        self.emit("run_error", {"error": reason})
        return "needs_input", reason

    async def run(self) -> tuple[str, str]:
        status, summary = await self._run_loop()
        if self._final_result_finalized:
            return self.finished or (status, summary)
        return self._finalize_run_result(status, summary)

    async def _run_loop(self) -> tuple[str, str]:
        provider_name = getattr(self.client, "name", type(self.client).__name__)
        model_name = getattr(self.client, "model", "unknown")
        mode = "replay" if provider_name == "offline-replay" else "live"
        self.emit("planner", {"mode": mode, "provider": provider_name, "model": model_name})
        validation_retries = 0
        last_validation_error = ""
        try:
            try:
                self.task_intent = await self._classify_task_intent()
            except Exception as exc:
                reason = f"Task intent classification failed safely before any ERP action: {type(exc).__name__}: {exc}"
                self.emit("run_error", {"error": reason})
                return "needs_review", reason
            self.emit("intent_guard", self.intent_guard_result or {})
            self.emit("task_intent", self.task_intent)
            if self.task_intent["intent"] in {"unsupported", "ambiguous"}:
                summary = (
                    self._unsupported_task_summary()
                    if self.task_intent["intent"] == "unsupported"
                    else str(self.task_intent.get("clarifying_question") or "Please clarify the requested invoice task.")
                )
                return self._finalize_run_result("needs_input", summary)
            if self.task_intent["intent"] == "write_invoice_entry":
                await self.browser.start()
            for step in range(1, MAX_STEPS + 1):
                try:
                    run_state = self._host_run_state()
                    turn_observations = self._conversation_observations(run_state)
                    turn_tools = self._tools_for_intent()
                    self.emit("host_run_state", {"turn": step, **run_state})
                    baseline_chars, baseline_tokens = self._estimate_prompt_tokens(turn_observations[1:], turn_tools)
                    prompt_chars, prompt_tokens = self._estimate_prompt_tokens(turn_observations, turn_tools)
                    prompt_breakdown = self._prompt_size_breakdown(turn_observations, turn_tools)
                    self.prompt_tokens_before.append(baseline_tokens)
                    self.prompt_tokens_after.append(prompt_tokens)
                    self.prompt_breakdown_tokens.append({key: value["approx_tokens"] for key, value in prompt_breakdown.items()})
                    self.emit("prompt_size", {
                        "turn": step,
                        "approx_characters": prompt_chars,
                        "approx_tokens": prompt_tokens,
                        "without_run_state_characters": baseline_chars,
                        "without_run_state_tokens": baseline_tokens,
                        "breakdown": prompt_breakdown,
                        "token_heuristic": "serialized characters divided by four; estimate only",
                    })
                    call = await self._await_with_long_operation_trace(
                        "provider_request",
                        asyncio.to_thread(self.client.next_tool_call, self.task, turn_observations, turn_tools),
                        details={"provider": provider_name, "turn": step},
                    )
                except ProviderRequestError as exc:
                    if not self._is_recoverable_provider_tool_error(exc):
                        provider_name = getattr(self.client, "name", type(self.client).__name__)
                        reason = f"{provider_name} decision failed at step {step}: {type(exc).__name__}: {exc}"
                        self.emit("run_error", {"error": reason})
                        return "needs_review", reason
                    plain_text = self._plain_text_from_provider_error(exc)
                    if plain_text is not None:
                        last_validation_error = "Provider returned plain text where a finish tool call was required."
                        failed_tool = "finish"
                        failure_count = self._record_tool_failure(failed_tool, last_validation_error)
                        stopped = self._stop_for_repeated_failure(failed_tool, last_validation_error, failure_count)
                        if stopped:
                            return stopped
                        if failure_count == WARN_REPEATED_TOOL_ERRORS:
                            self._append_repeat_failure_warning(failed_tool, last_validation_error, failure_count)
                        if validation_retries >= MAX_TOOL_VALIDATION_RETRIES:
                            reason = self._validation_retry_limit_message(last_validation_error, failed_tool)
                            self.emit("run_error", {"error": reason})
                            return "needs_review", reason
                        validation_retries += 1
                        if not self._plain_text_nudge_used:
                            self._plain_text_nudge_used = True
                            feedback = {
                                "tool": "finish_nudge",
                                "arguments": {},
                                "result": {
                                    "instruction": "Call finish with this summary.",
                                    "summary": plain_text[:MAX_FAILED_GENERATION_CHARS],
                                },
                            }
                            self.observations.append(feedback)
                            self.emit("finish_nudge", feedback["result"])
                        else:
                            self._append_tool_validation_feedback(
                                retry_number=validation_retries,
                                tool_name="finish",
                                error_text=last_validation_error,
                            )
                        continue
                    last_validation_error = self._provider_tool_error_text(exc)
                    failed_tool = exc.rejected_tool_name or "provider_tool_call"
                    failure_count = self._record_tool_failure(failed_tool, last_validation_error)
                    stopped = self._stop_for_repeated_failure(failed_tool, last_validation_error, failure_count)
                    if stopped:
                        return stopped
                    if failure_count == WARN_REPEATED_TOOL_ERRORS:
                        self._append_repeat_failure_warning(failed_tool, last_validation_error, failure_count)
                    if validation_retries >= MAX_TOOL_VALIDATION_RETRIES:
                        reason = self._validation_retry_limit_message(last_validation_error, exc.rejected_tool_name)
                        self.emit("run_error", {"error": reason})
                        return "needs_review", reason
                    validation_retries += 1
                    self._append_tool_validation_feedback(
                        retry_number=validation_retries,
                        tool_name=exc.rejected_tool_name,
                        error_text=last_validation_error,
                    )
                    continue
                except Exception as exc:
                    provider_name = getattr(self.client, "name", type(self.client).__name__)
                    reason = f"{provider_name} decision failed at step {step}: {type(exc).__name__}: {exc}"
                    self.emit("run_error", {"error": reason})
                    return "needs_review", reason
                provider_name = getattr(self.client, "name", type(self.client).__name__)
                if call is None:
                    return "needs_review", f"{provider_name} stopped without calling finish; the run did not claim completion."

                if call.name == "__malformed_tool_arguments__":
                    invalid_tool = str(call.arguments.get("tool_name") or "provider_tool_call")
                    last_validation_error = self._safe_validation_error(
                        "Malformed tool arguments: " + str(call.arguments.get("message", "invalid tool arguments")),
                        str(call.arguments.get("message", "")),
                    )
                    failure_count = self._record_tool_failure(invalid_tool, last_validation_error)
                    stopped = self._stop_for_repeated_failure(invalid_tool, last_validation_error, failure_count)
                    if stopped:
                        return stopped
                    if failure_count == WARN_REPEATED_TOOL_ERRORS:
                        self._append_repeat_failure_warning(invalid_tool, last_validation_error, failure_count)
                    if validation_retries >= MAX_TOOL_VALIDATION_RETRIES:
                        reason = self._validation_retry_limit_message(last_validation_error, str(invalid_tool) if invalid_tool else None)
                        self.emit("run_error", {"error": reason})
                        return "needs_review", reason
                    validation_retries += 1
                    self._append_tool_validation_feedback(
                        retry_number=validation_retries,
                        tool_name=str(invalid_tool) if invalid_tool else None,
                        error_text=last_validation_error,
                    )
                    continue
                self._event_action(call)
                if call.name not in TOOL_BY_NAME:
                    last_validation_error = f"Unknown tool name: {call.name}. Use one of the supplied tool schemas."
                    failure_count = self._record_tool_failure(call.name, last_validation_error)
                    stopped = self._stop_for_repeated_failure(call.name, last_validation_error, failure_count)
                    if stopped:
                        return stopped
                    if failure_count == WARN_REPEATED_TOOL_ERRORS:
                        self._append_repeat_failure_warning(call.name, last_validation_error, failure_count)
                    if validation_retries >= MAX_TOOL_VALIDATION_RETRIES:
                        reason = self._validation_retry_limit_message(last_validation_error, call.name)
                        self.emit("run_error", {"error": reason})
                        return "needs_review", reason
                    validation_retries += 1
                    self._append_tool_validation_feedback(
                        retry_number=validation_retries, tool_name=None, error_text=last_validation_error
                    )
                    continue
                else:
                    try:
                        self._validate_arguments(call)
                    except Exception as exc:
                        last_validation_error = self._safe_validation_error(
                            f"Malformed arguments for {call.name}: {type(exc).__name__}: {exc}", str(exc)
                        )
                        failure_count = self._record_tool_failure(call.name, last_validation_error)
                        stopped = self._stop_for_repeated_failure(call.name, last_validation_error, failure_count)
                        if stopped:
                            return stopped
                        if failure_count == WARN_REPEATED_TOOL_ERRORS:
                            self._append_repeat_failure_warning(call.name, last_validation_error, failure_count)
                        if validation_retries >= MAX_TOOL_VALIDATION_RETRIES:
                            reason = self._validation_retry_limit_message(last_validation_error, call.name)
                            self.emit("run_error", {"error": reason})
                            return "needs_review", reason
                        validation_retries += 1
                        self._append_tool_validation_feedback(
                            retry_number=validation_retries, tool_name=call.name, error_text=last_validation_error
                        )
                        continue
                    try:
                        result = await self._await_with_long_operation_trace(
                            f"tool:{call.name}", self._execute(call.name, call.arguments),
                            details={"tool": call.name, "turn": step},
                        )
                    except Exception as exc:
                        result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
                safe_arguments = self._redact_arguments(call.name, call.arguments)
                safe_result = self._redact_nested_result(result)
                model_result = self._model_observation_result(safe_result)
                self.observations.append({"tool": call.name, "arguments": safe_arguments, "result": model_result})
                self.emit("observation", {"tool": call.name, **safe_result})
                page_before = self._page_signature(self._last_page_snapshot)
                observed_snapshot = self._snapshot_from_result(safe_result)
                if observed_snapshot is not None:
                    self._last_page_snapshot = observed_snapshot
                page_after = self._page_signature(self._last_page_snapshot)
                action_key = json.dumps([call.name, safe_arguments], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if call.name != "finish":
                    if (
                        action_key == self._last_action_key
                        and page_before == self._last_action_page_signature
                        and page_after == page_before
                    ):
                        self._same_action_no_change_count += 1
                    elif page_after == page_before:
                        self._same_action_no_change_count = 1
                    else:
                        self._same_action_no_change_count = 0
                    self._last_action_key = action_key
                    self._last_action_page_signature = page_after
                    action_label = self._action_label(call.name, safe_arguments)
                    if self._same_action_no_change_count == LOOP_GUARD_WARN_AFTER:
                        warning = (
                            f"Loop guard: {action_label} has repeated {LOOP_GUARD_WARN_AFTER} times with no page-state change. "
                            f"Last observed alert: {self._last_alert(self._last_page_snapshot)}. Choose a different action or stop."
                        )
                        self.observations.append({"tool": "loop_guard_warning", "arguments": {}, "result": {"warning": warning}})
                        self.emit("loop_guard_warning", {"tool": call.name, "arguments": safe_arguments, "count": self._same_action_no_change_count, "alert": self._last_alert(self._last_page_snapshot), "warning": warning})
                    elif self._same_action_no_change_count >= LOOP_GUARD_STOP_AFTER:
                        last_alert = self._last_alert(self._last_page_snapshot)
                        reason = (
                            f"Stopped after repeated action {action_label} {LOOP_GUARD_STOP_AFTER} times with no page-state change. "
                            f"Last observed alert: {last_alert}."
                        )
                        self.emit("run_error", {"error": reason})
                        return "needs_review", reason
                if result.get("status") == "error":
                    last_error = str(result.get("message", "tool action failed"))
                    failure_count = self._record_tool_failure(call.name, last_error)
                    if failure_count == WARN_REPEATED_TOOL_ERRORS:
                        self._append_repeat_failure_warning(call.name, last_error, failure_count)
                    stopped = self._stop_for_repeated_failure(call.name, last_error, failure_count)
                    if stopped:
                        return stopped
                if self.finished:
                    return self.finished
            return "needs_review", f"The agent reached the {MAX_STEPS}-step limit without calling finish."
        finally:
            self._emit_provider_wait_summary()
            if self.prompt_tokens_after:
                breakdown_keys = self.prompt_breakdown_tokens[0].keys()
                self.emit("prompt_size_summary", {
                    "turns": len(self.prompt_tokens_after),
                    "average_tokens_without_run_state": round(sum(self.prompt_tokens_before) / len(self.prompt_tokens_before), 1),
                    "average_tokens_with_run_state": round(sum(self.prompt_tokens_after) / len(self.prompt_tokens_after), 1),
                    "average_breakdown_tokens": {
                        key: round(sum(item.get(key, 0) for item in self.prompt_breakdown_tokens) / len(self.prompt_breakdown_tokens), 1)
                        for key in breakdown_keys
                    },
                    "token_heuristic": "serialized characters divided by four; estimate only",
                })
            await self.browser.close()

    def _safe_validation_error(self, error: str, body: str) -> str:
        text = error or body or "Provider rejected tool arguments or schema."
        for secret in (self.browser.password, self.browser.email):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(r'(?i)("?(?:password|passwd|secret|api[_-]?key)"?\s*[:=]\s*")[^"\\]*(")', r"\1[REDACTED]\2", text)
        return _clean_text(text, 1200)

    @staticmethod
    def _validation_retry_limit_message(last_error: str, tool_name: str | None = None) -> str:
        label = f" ({tool_name})" if tool_name else ""
        return f"Tool validation retry limit reached after {MAX_TOOL_VALIDATION_RETRIES} retries. Last tool error{label}: {last_error}"

    @staticmethod
    def _validate_arguments(call: ToolCall) -> dict[str, Any]:
        definition = TOOL_BY_NAME[call.name]
        schema = definition["parameters"]
        arguments = call.arguments
        if schema.get("type") == "object" and not schema.get("properties"):
            if arguments is None or arguments == {"arguments": {}} or arguments == {"arguments": None}:
                arguments = {}
        AgentLoop._validate_schema_value(arguments, schema, "Tool arguments")
        return arguments

    @staticmethod
    def _validate_schema_value(value: Any, schema: dict[str, Any], label: str) -> None:
        expected = schema.get("type")
        if expected == "object":
            if not isinstance(value, dict):
                raise ValueError(f"{label} must be an object.")
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            if required - value.keys():
                raise ValueError(f"Missing required tool arguments: {', '.join(sorted(required - value.keys()))}")
            closed_object = schema.get("additionalProperties") is False or (
                "properties" in schema and schema.get("additionalProperties") is not True
            )
            if closed_object and value.keys() - properties.keys():
                unexpected = ", ".join(sorted(value.keys() - properties.keys()))
                allowed = ", ".join(sorted(properties)) or "none"
                message = f"Unexpected tool arguments: {unexpected}. Allowed fields: {allowed}."
                if label.startswith("Tool argument fields["):
                    selectors = ", ".join(sorted(MODEL_FORM_SELECTORS))
                    message += f" Allowed ERP field selectors: {selectors}."
                raise ValueError(message)
            for key, item in value.items():
                if key in properties:
                    AgentLoop._validate_schema_value(item, properties[key], f"Tool argument {key}")
            return
        if expected == "array":
            if not isinstance(value, list):
                raise ValueError(f"{label} must be an array.")
            if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1000):
                raise ValueError(f"{label} must contain between {schema.get('minItems', 0)} and {schema.get('maxItems', 1000)} items.")
            for index, item in enumerate(value):
                AgentLoop._validate_schema_value(item, schema.get("items", {}), f"{label}[{index}]")
            return
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"{label} must be a string.")
        if expected == "string" and schema.get("enum") and value not in schema["enum"]:
            allowed = ", ".join(str(option) for option in schema["enum"])
            raise ValueError(f"{label} must be one of the allowed values: {allowed}.")

    @staticmethod
    def _redact_nested_result(result: dict[str, Any]) -> dict[str, Any]:
        safe = dict(result)
        if isinstance(safe.get("steps"), list):
            safe["steps"] = [
                {
                    **step,
                    "arguments": AgentLoop._redact_arguments(str(step.get("tool", "")), step.get("arguments", {})),
                    "result": AgentLoop._redact_nested_result(step.get("result", {})),
                }
                if isinstance(step, dict) and isinstance(step.get("result"), dict) else step
                for step in safe["steps"]
            ]
        return safe

    @staticmethod
    def _model_observation_result(result: dict[str, Any]) -> dict[str, Any]:
        """Keep host-generated provenance in the trace, outside model conversation history."""
        safe = dict(result)
        safe.pop("source_file", None)
        safe.pop("source_sha256", None)
        return safe


async def run_computer_use_agent(
    data_dir: Path,
    run_id: str,
    task: str,
    base_url: str,
    emit: Callable[[str, dict[str, Any]], None],
    approval_handler: Callable[[dict[str, Any]], Any],
    *,
    approval_mode: str = "policy_driven",
    offline: bool = False,
    confirmed_pending_context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Run with a live provider, or FakeLLM replay when explicitly offline."""
    loop = AgentLoop(
        data_dir, run_id, task, base_url, emit, approval_handler,
        approval_mode=approval_mode, offline=offline,
        confirmed_pending_context=confirmed_pending_context,
    )
    return await loop.run()
