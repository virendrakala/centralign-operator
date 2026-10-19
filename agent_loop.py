"""Single observe/decide/execute loop shared by live and offline planners."""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from browser_tools import APPROVAL_CHECK_SELECTORS, CLICK_SELECTORS, ERP_ROUTES, MODEL_FORM_SELECTORS, BrowserTools
from company_app import CompanyApp
from documents import list_workspace_files, read_document
from llm_clients import LLMClient, ProviderRequestError, SYSTEM_INSTRUCTIONS, ToolCall, make_client
from policy import approval_required


MAX_STEPS = 64
MAX_CONSECUTIVE_IDENTICAL_TOOL_FAILURES = 3
MAX_TOOL_VALIDATION_RETRIES = 3
MAX_SUBMIT_ATTEMPTS = 3
LOOP_GUARD_WARN_AFTER = 3
LOOP_GUARD_STOP_AFTER = 5
FULL_OBSERVATIONS_TO_KEEP = 6
STOP_STATUSES = {"complete", "needs_input", "needs_review", "stopped_by_user"}
_ROUTE_CHOICES = "; ".join(f"{path}: {description}" for path, description in ERP_ROUTES.items())


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {"name": "list_files", "description": "List invoice and policy files.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "read_document", "description": "Read a listed invoice and extract its fields.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "read_policy", "description": "Read AP rules and approval thresholds.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "browser_open", "description": "Open an allowed ERP route; warns about unsaved form fields.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "enum": list(ERP_ROUTES), "description": f"Choose an allowed route: {_ROUTE_CHOICES}"}}, "required": ["path"]}},
    {"name": "erp_login", "description": "Host signs in using company memory and returns the landing page and alert.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "browser_snapshot", "description": "Inspect URL, title, alerts, field states, buttons, and table rows. Secrets are redacted.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "browser_fill", "description": "Fill one allowlisted invoice field with source data or an observed vendor.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(MODEL_FORM_SELECTORS), "description": "Allowlisted invoice field."}, "value": {"type": "string"}}, "required": ["selector", "value"]}},
    {"name": "browser_fill_form", "description": "Fill up to ten allowlisted invoice fields.", "parameters": {"type": "object", "properties": {"fields": {"type": "array", "minItems": 1, "maxItems": 10, "items": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(MODEL_FORM_SELECTORS), "description": "Allowlisted invoice field."}, "value": {"type": "string"}}, "required": ["selector", "value"], "additionalProperties": False}}}, "required": ["fields"], "additionalProperties": False}},
    {"name": "browser_action_sequence", "description": "Run two or three field-fill actions only: browser_fill, browser_check, or browser_fill_form. Navigation, clicks, approval requests, and ledger verification must each be separate tool calls so their results can be observed first.", "parameters": {"type": "object", "properties": {"actions": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "object", "properties": {"tool": {"type": "string", "enum": ["browser_fill", "browser_check", "browser_fill_form"]}, "arguments": {"type": "object"}}, "required": ["tool", "arguments"], "additionalProperties": False}}}, "required": ["actions"], "additionalProperties": False}},
    {"name": "browser_check", "description": "Check the approval confirmation control.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(APPROVAL_CHECK_SELECTORS)}}, "required": ["selector"]}},
    {"name": "browser_click", "description": "Click the invoice submit control.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "enum": sorted(CLICK_SELECTORS)}}, "required": ["selector"]}},
    {"name": "request_human_approval", "description": "Request policy approval by invoice_number or read_document path. Include canonical_vendor from the observed directory when needed. The host injects source provenance and the approved token.", "parameters": {"type": "object", "properties": {"invoice_number": {"type": "string"}, "document_path": {"type": "string"}, "canonical_vendor": {"type": "string"}, "reason": {"type": "string"}}, "required": [], "additionalProperties": False}},
    {"name": "verify_ledger_row", "description": "Read back the selected invoice from the ERP ledger and compare its fields.", "parameters": {"type": "object", "properties": {"invoice_number": {"type": "string"}, "expected_fields": {"type": "object", "properties": {"document_path": {"type": "string", "description": "Optional read_document path to disambiguate."}, "vendor": {"type": "string"}, "canonical_vendor": {"type": "string"}, "invoice_number": {"type": "string"}, "invoice_date": {"type": "string"}, "due_date": {"type": "string"}, "amount": {"type": "string"}, "currency": {"type": "string"}}, "required": ["vendor", "invoice_number", "invoice_date", "due_date", "amount", "currency"]}}, "required": ["invoice_number", "expected_fields"]}},
    {"name": "finish", "description": "Stop the run with a final status and concise evidence-based summary.", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["complete", "needs_input", "needs_review", "stopped_by_user"]}, "summary": {"type": "string"}, "evidence": {"type": "object"}}, "required": ["status", "summary"]}},
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
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.run_id = run_id
        self.task = task
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
        self.browser = BrowserTools(data_dir, run_id, base_url, emit)
        self.observations: list[dict[str, Any]] = []
        self.listed_files: set[str] = set()
        self.documents: dict[str, dict[str, Any]] = {}
        self.policy: dict[str, Any] | None = None
        self.selected_invoice: dict[str, Any] | None = None
        self.selected_invoice_path: str | None = None
        self.approval_tokens: dict[str, str] = {}
        self.denied_invoices: set[str] = set()
        self.prompt_tokens_before: list[int] = []
        self.prompt_tokens_after: list[int] = []
        self.duplicate_checked = False
        self.duplicate_present = False
        self.verification: dict[str, Any] | None = None
        self.submit_attempts = 0
        self.submit_performed = False
        self.last_tool_failure_signature: str | None = None
        self.consecutive_identical_tool_failures = 0
        self.finished: tuple[str, str] | None = None
        self._last_page_snapshot: dict[str, Any] | None = None
        self._last_action_key: str | None = None
        self._last_action_page_signature: str | None = None
        self._same_action_no_change_count = 0

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
        form_filled = any(isinstance(field, dict) and field.get("state") == "filled" for field in fields)
        if self.submit_performed:
            form_status = "submitted"
        elif page_path == "/erp/invoices/new" and form_filled:
            form_status = "filled"
        else:
            form_status = "empty"
        alerts = snapshot.get("alerts") or ([snapshot.get("alert")] if snapshot.get("alert") else [])
        approval_hint = None
        if approval_status == "required":
            approval_hint = f"Approval required: call request_human_approval with invoice_number {invoice['invoice_number']}."
        return {
            "logged_in": bool(getattr(self.browser, "logged_in", False)),
            "selected_invoice": selected,
            "legal_vendor": legal_vendor,
            "approval_status": approval_status,
            "approval_hint": approval_hint,
            "form_status": form_status,
            "last_erp_alert": str(alerts[-1])[:300] if alerts else None,
        }

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

    def _estimate_prompt_tokens(self, observations: list[dict[str, Any]]) -> tuple[int, int]:
        prompt = {
            "system": SYSTEM_INSTRUCTIONS,
            "task": self.task,
            "tools": TOOL_DEFINITIONS,
            "observations": observations,
        }
        chars = len(json.dumps(prompt, ensure_ascii=False, separators=(",", ":")))
        return chars, math.ceil(chars / 4)

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
        self.emit("rate_limit_recovery", payload)
        action = payload.get("action")
        if action == "provider_fallback":
            self.emit("planner", {"mode": "live", "provider": payload.get("provider"), "model": payload.get("model")})

    def _provider_answered(self, payload: dict[str, Any]) -> None:
        self.emit("provider_turn", payload)

    @staticmethod
    def _is_recoverable_provider_tool_error(error: ProviderRequestError) -> bool:
        if error.status_code != 400:
            return False
        code = (error.error_code or "").casefold().replace("-", "_")
        if code in {"tool_use_failed", "invalid_tool_call", "tool_validation_failed", "schema_validation_error"}:
            return True
        message = (error.response_body or "").casefold().replace("-", "_")
        return "tool_use_failed" in message or (
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

    async def _execute(
        self, name: str, arguments: dict[str, Any], *, include_snapshot: bool = True
    ) -> dict[str, Any]:
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
            model_result = {key: value for key, value in result.items() if key != "sha256"}
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
        if name == "browser_open":
            path = _clean_text(arguments.get("path"), 100)
            if path == "/erp/invoices/new" and self.selected_invoice and self._needs_approval() and not self._invoice_is_approved():
                raise ValueError(
                    "Approval required: call request_human_approval with invoice_number "
                    f"{self.selected_invoice['invoice_number']}."
                )
            if path == "/erp/invoices/new" and self.selected_invoice:
                self.browser.set_invoice(self.selected_invoice)
            result = await self.browser.browser_open(path)
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
            self.emit("credentials_supplied", {
                "message": "Credentials supplied by host for ERP sign-in; the model never receives their values."
            })
            return await self.browser.erp_login()
        if name == "browser_snapshot":
            result = await self.browser.browser_snapshot()
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
                result = await self.browser.browser_fill(selector, _clean_text(arguments.get("value"), 600))
            except Exception as exc:
                result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            if include_snapshot:
                snapshot = await self.browser.compact_snapshot("locator.fill", selector)
                result["snapshot"] = snapshot
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
            self.selected_invoice = invoice
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
                "invoice": {**invoice, "canonical_vendor": canonical},
                "source_file": invoice["source_file"],
                "source_sha256": invoice["source_sha256"],
            }
            token = await self.approval_handler(payload)
            if token:
                self.approval_tokens[invoice_key] = str(token)
                self.denied_invoices.discard(invoice_key)
            else:
                self.denied_invoices.add(invoice_key)
            self.browser.set_invoice(invoice)
            return {"status": "approved" if token else "denied_or_timed_out", "invoice_number": invoice["invoice_number"]}
        if name == "verify_ledger_row":
            fields = arguments.get("expected_fields")
            if not isinstance(fields, dict) or str(arguments.get("invoice_number", "")) != str(fields.get("invoice_number", "")):
                raise ValueError("Ledger verification must identify the selected source invoice.")
            invoice = self._invoice_by_fields(fields)
            canonical = str(fields.get("canonical_vendor", ""))
            if canonical:
                self._validate_vendor_evidence(canonical)
                invoice["canonical_vendor"] = canonical
            self.selected_invoice = invoice
            self.browser.set_invoice(invoice)
            result = await self.browser.verify_ledger_row(invoice)
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
            return result
        if name == "finish":
            status = _clean_text(arguments.get("status"), 40)
            summary = _clean_text(arguments.get("summary"), 1000)
            if status not in STOP_STATUSES or not summary:
                raise ValueError("finish requires a supported stop status and a concise summary.")
            if status == "complete" and not (self.verification and self.verification.get("status") == "verified"):
                raise ValueError("A complete run requires a matching read-back from the live ERP ledger DOM.")
            self.finished = status, summary
            return {"status": "finished", "final_status": status, "summary": summary, "evidence": self.verification or {}}
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

    def _record_tool_failure(self, name: str, arguments: dict[str, Any], error: str) -> bool:
        """Count only adjacent failures with the same tool, arguments, and error."""
        signature = json.dumps([name, arguments, error], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature == self.last_tool_failure_signature:
            self.consecutive_identical_tool_failures += 1
        else:
            self.last_tool_failure_signature = signature
            self.consecutive_identical_tool_failures = 1
        count = self.consecutive_identical_tool_failures
        self.emit("tool_retry_budget", {
            "tool": name,
            "consecutive_identical_failures": count,
            "limit": MAX_CONSECUTIVE_IDENTICAL_TOOL_FAILURES,
            "remaining": max(0, MAX_CONSECUTIVE_IDENTICAL_TOOL_FAILURES - count),
            "error": error,
        })
        return count >= MAX_CONSECUTIVE_IDENTICAL_TOOL_FAILURES

    def _clear_tool_failures(self) -> None:
        self.last_tool_failure_signature = None
        self.consecutive_identical_tool_failures = 0

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
            f"Stopped after {MAX_CONSECUTIVE_IDENTICAL_TOOL_FAILURES} consecutive identical failures "
            f"in {name}. Last error: {error}. Done: {done_text}. Not done: {not_done_text}."
        )

    async def run(self) -> tuple[str, str]:
        await self.browser.start()
        provider_name = getattr(self.client, "name", type(self.client).__name__)
        model_name = getattr(self.client, "model", "unknown")
        mode = "replay" if provider_name == "offline-replay" else "live"
        self.emit("planner", {"mode": mode, "provider": provider_name, "model": model_name})
        validation_retries = 0
        last_validation_error = ""
        try:
            for step in range(1, MAX_STEPS + 1):
                try:
                    run_state = self._host_run_state()
                    turn_observations = self._conversation_observations(run_state)
                    self.emit("host_run_state", {"turn": step, **run_state})
                    baseline_chars, baseline_tokens = self._estimate_prompt_tokens(turn_observations[1:])
                    prompt_chars, prompt_tokens = self._estimate_prompt_tokens(turn_observations)
                    self.prompt_tokens_before.append(baseline_tokens)
                    self.prompt_tokens_after.append(prompt_tokens)
                    self.emit("prompt_size", {
                        "turn": step,
                        "approx_characters": prompt_chars,
                        "approx_tokens": prompt_tokens,
                        "without_run_state_characters": baseline_chars,
                        "without_run_state_tokens": baseline_tokens,
                        "token_heuristic": "serialized characters divided by four; estimate only",
                    })
                    call = await asyncio.to_thread(self.client.next_tool_call, self.task, turn_observations, TOOL_DEFINITIONS)
                except ProviderRequestError as exc:
                    if not self._is_recoverable_provider_tool_error(exc):
                        provider_name = getattr(self.client, "name", type(self.client).__name__)
                        reason = f"{provider_name} decision failed at step {step}: {type(exc).__name__}: {exc}"
                        self.emit("run_error", {"error": reason})
                        return "needs_review", reason
                    last_validation_error = self._safe_validation_error(str(exc), exc.response_body or "")
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
                    invalid_tool = call.arguments.get("tool_name")
                    last_validation_error = self._safe_validation_error(
                        "Malformed tool arguments: " + str(call.arguments.get("message", "invalid tool arguments")),
                        str(call.arguments.get("message", "")),
                    )
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
                        result = await self._execute(call.name, call.arguments)
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
                    if self._record_tool_failure(call.name, safe_arguments, last_error):
                        reason = self._tool_failure_summary(call.name, last_error)
                        self.emit("run_error", {"error": reason})
                        return "needs_input", reason
                else:
                    self._clear_tool_failures()
                if self.finished:
                    return self.finished
            return "needs_review", f"The agent reached the {MAX_STEPS}-step limit without calling finish."
        finally:
            if self.prompt_tokens_after:
                self.emit("prompt_size_summary", {
                    "turns": len(self.prompt_tokens_after),
                    "average_tokens_without_run_state": round(sum(self.prompt_tokens_before) / len(self.prompt_tokens_before), 1),
                    "average_tokens_with_run_state": round(sum(self.prompt_tokens_after) / len(self.prompt_tokens_after), 1),
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
    def _validate_arguments(call: ToolCall) -> None:
        definition = TOOL_BY_NAME[call.name]
        schema = definition["parameters"]
        AgentLoop._validate_schema_value(call.arguments, schema, "Tool arguments")

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
) -> tuple[str, str]:
    """Run with a live provider, or FakeLLM replay when explicitly offline."""
    loop = AgentLoop(
        data_dir, run_id, task, base_url, emit, approval_handler,
        approval_mode=approval_mode, offline=offline,
    )
    return await loop.run()
