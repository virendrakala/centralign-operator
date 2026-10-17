"""Single observe/decide/execute loop shared by live and offline planners."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Callable

from browser_tools import BrowserTools
from company_app import CompanyApp
from documents import list_workspace_files, read_document
from llm_clients import LLMClient, ToolCall, make_client
from policy import approval_required


MAX_STEPS = 64
MAX_TOOL_ERRORS = 3
MAX_SUBMIT_ATTEMPTS = 3
STOP_STATUSES = {"complete", "needs_input", "needs_review", "stopped_by_user"}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {"name": "list_files", "description": "List invoice sources and the procurement policy available to this run.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "read_document", "description": "Read an allowed invoice document and inspect its extracted text and fields.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "read_policy", "description": "Read the procurement policy and parsed approval/vendor rules before deciding on a write.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "browser_open", "description": "Open one of the local ERP routes in the Playwright browser.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "browser_snapshot", "description": "Inspect the current ERP page's simplified DOM, visible alerts, form values, and tables.", "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "browser_fill", "description": "Fill a known ERP form control using a source field, observed vendor row, or sandbox credential placeholder.", "parameters": {"type": "object", "properties": {"selector": {"type": "string"}, "value": {"type": "string"}}, "required": ["selector", "value"]}},
    {"name": "browser_check", "description": "Check the manager-approval confirmation control after approval has been granted.", "parameters": {"type": "object", "properties": {"selector": {"type": "string"}}, "required": ["selector"]}},
    {"name": "browser_click", "description": "Click the ERP login or invoice-submit control after host preconditions pass.", "parameters": {"type": "object", "properties": {"selector": {"type": "string"}}, "required": ["selector"]}},
    {"name": "request_human_approval", "description": "Pause and request approval for the selected invoice when procurement policy requires it.", "parameters": {"type": "object", "properties": {"reason": {"type": "string"}, "extracted_fields": {"type": "object", "properties": {"vendor": {"type": "string"}, "canonical_vendor": {"type": "string"}, "invoice_number": {"type": "string"}, "invoice_date": {"type": "string"}, "due_date": {"type": "string"}, "amount": {"type": "string"}, "currency": {"type": "string"}, "source_file": {"type": "string"}, "source_sha256": {"type": "string"}}, "required": ["vendor", "canonical_vendor", "invoice_number", "invoice_date", "due_date", "amount", "currency", "source_file", "source_sha256"]}}, "required": ["reason", "extracted_fields"]}},
    {"name": "verify_ledger_row", "description": "Read a matching ERP ledger row from the live DOM and compare it to the selected source document.", "parameters": {"type": "object", "properties": {"invoice_number": {"type": "string"}, "expected_fields": {"type": "object", "properties": {"vendor": {"type": "string"}, "canonical_vendor": {"type": "string"}, "invoice_number": {"type": "string"}, "invoice_date": {"type": "string"}, "due_date": {"type": "string"}, "amount": {"type": "string"}, "currency": {"type": "string"}, "source_file": {"type": "string"}, "source_sha256": {"type": "string"}}, "required": ["vendor", "invoice_number", "invoice_date", "due_date", "amount", "currency", "source_file", "source_sha256"]}}, "required": ["invoice_number", "expected_fields"]}},
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
        self.client = client or make_client(offline=offline)
        self.company = CompanyApp(self.data_dir)
        self.browser = BrowserTools(data_dir, run_id, base_url, emit)
        self.observations: list[dict[str, Any]] = []
        self.listed_files: set[str] = set()
        self.documents: dict[str, dict[str, Any]] = {}
        self.policy: dict[str, Any] | None = None
        self.selected_invoice: dict[str, Any] | None = None
        self.approval_granted = False
        self.duplicate_checked = False
        self.duplicate_present = False
        self.verification: dict[str, Any] | None = None
        self.submit_attempts = 0
        self.tool_errors = 0
        self.finished: tuple[str, str] | None = None

    def _event_action(self, call: ToolCall) -> None:
        safe_args = dict(call.arguments)
        if call.name == "browser_fill" and safe_args.get("selector") in {"#password_input", "#approval_token", "#source_sha256_input"}:
            safe_args["value"] = "[redacted]"
        self.emit("action", {"tool": call.name, **safe_args})

    def _invoice_by_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        invoice_number = str(fields.get("invoice_number", ""))
        source_file = str(fields.get("source_file", ""))
        candidates = [
            document["invoice"] for document in self.documents.values()
            if document.get("invoice")
            and document["invoice"]["invoice_number"] == invoice_number
            and document["invoice"]["source_file"] == source_file
        ]
        if len(candidates) != 1:
            raise ValueError("Select one invoice that was read from the current workspace.")
        invoice = candidates[0]
        for key in ("vendor", "invoice_date", "due_date", "amount", "currency", "source_sha256"):
            if str(fields.get(key, "")) != str(invoice[key]):
                raise ValueError(f"{key.replace('_', ' ')} must match the selected source document.")
        return invoice

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

    async def _execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
            return result
        if name == "read_policy":
            result = await asyncio.to_thread(self.company.read_company_policy)
            self.policy = result
            return {"status": "read", "policy": result}
        if name == "browser_open":
            path = _clean_text(arguments.get("path"), 100)
            if path == "/erp/invoices/new" and self.selected_invoice and self._needs_approval() and not self.approval_granted:
                raise ValueError("Procurement approval is required before opening the invoice write form.")
            return await self.browser.browser_open(path)
        if name == "browser_snapshot":
            return await self.browser.browser_snapshot()
        if name == "browser_fill":
            if not self.selected_invoice and arguments.get("selector") not in {"#email_input", "#password_input"}:
                raise ValueError("Choose and verify an invoice before filling the ERP form.")
            return await self.browser.browser_fill(_clean_text(arguments.get("selector"), 80), _clean_text(arguments.get("value"), 600))
        if name == "browser_check":
            return await self.browser.browser_check(_clean_text(arguments.get("selector"), 80))
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
                if self._needs_approval() and not self.approval_granted:
                    raise ValueError("A human approval decision is required before this invoice can be submitted.")
                if self.submit_attempts >= MAX_SUBMIT_ATTEMPTS:
                    raise ValueError("The ERP submit retry limit has been reached.")
                self.submit_attempts += 1
            result = await self.browser.browser_click(selector)
            if selector == "#submit_btn" and result.get("snapshot", {}).get("alert"):
                alert = result["snapshot"]["alert"]
                self.emit("observation", {"tool": "erp_submit_alert", "alert": alert})
                if "vendor" in alert.casefold():
                    self.emit("self_correction", {"text": "The ERP rejected the submitted vendor value. Inspect the vendor directory and adapt from its observed row."})
            return result
        if name == "request_human_approval":
            if not self.policy:
                raise ValueError("Read procurement policy before requesting approval.")
            fields = arguments.get("extracted_fields")
            if not isinstance(fields, dict):
                raise ValueError("Approval requires extracted invoice fields.")
            invoice = self._invoice_by_fields(fields)
            canonical = str(fields.get("canonical_vendor", ""))
            self._validate_vendor_evidence(canonical)
            invoice["canonical_vendor"] = canonical
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
            self.approval_granted = bool(token)
            self.browser.set_invoice(invoice)
            self.browser.set_approval_token(token)
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

    async def run(self) -> tuple[str, str]:
        await self.browser.start()
        provider_name = getattr(self.client, "name", type(self.client).__name__)
        model_name = getattr(self.client, "model", "unknown")
        mode = "replay" if provider_name == "offline-replay" else "live"
        self.emit("planner", {"mode": mode, "provider": provider_name, "model": model_name})
        try:
            for step in range(1, MAX_STEPS + 1):
                try:
                    call = await asyncio.to_thread(self.client.next_tool_call, self.task, self.observations, TOOL_DEFINITIONS)
                except Exception as exc:
                    reason = f"{provider_name} decision failed at step {step}: {type(exc).__name__}: {exc}"
                    self.emit("run_error", {"error": reason})
                    return "needs_review", reason
                if call is None:
                    return "needs_review", f"{provider_name} stopped without calling finish; the run did not claim completion."

                self._event_action(call)
                if call.name not in TOOL_BY_NAME:
                    result = {"status": "error", "message": f"Tool is not allowed: {call.name}"}
                else:
                    try:
                        self._validate_arguments(call)
                        result = await self._execute(call.name, call.arguments)
                    except Exception as exc:
                        result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
                self.observations.append({"tool": call.name, "arguments": call.arguments, "result": result})
                self.emit("observation", {"tool": call.name, **result})
                if result.get("status") == "error":
                    self.tool_errors += 1
                    if self.tool_errors >= MAX_TOOL_ERRORS:
                        return "needs_review", f"The tool retry limit was reached: {result.get('message', 'repeated tool errors')}"
                if self.finished:
                    return self.finished
            return "needs_review", f"The agent reached the {MAX_STEPS}-step limit without calling finish."
        finally:
            await self.browser.close()

    @staticmethod
    def _validate_arguments(call: ToolCall) -> None:
        definition = TOOL_BY_NAME[call.name]
        schema = definition["parameters"]
        if not isinstance(call.arguments, dict):
            raise ValueError("Tool arguments must be an object.")
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        if required - call.arguments.keys():
            raise ValueError(f"Missing required tool arguments: {', '.join(sorted(required - call.arguments.keys()))}")
        if call.arguments.keys() - properties.keys():
            raise ValueError(f"Unexpected tool arguments: {', '.join(sorted(call.arguments.keys() - properties.keys()))}")
        for key, value in call.arguments.items():
            expected = properties[key].get("type")
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"Tool argument {key} must be a string.")
            if expected == "object" and not isinstance(value, dict):
                raise ValueError(f"Tool argument {key} must be an object.")


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
    """Run the common browser agent loop with a live provider or FakeLLM replay."""
    loop = AgentLoop(
        data_dir, run_id, task, base_url, emit, approval_handler,
        approval_mode=approval_mode, offline=offline,
    )
    return await loop.run()
