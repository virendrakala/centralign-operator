"""CentrAlign Operator: local FastAPI workbench and sandbox ERP."""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import secrets
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from company_app import CompanyApp
from documents import list_invoice_candidates
from worker import ensure_operator_assets, list_workspace_files, planner_config, run_computer_use_agent

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULT_DATA_DIR = ROOT / "data"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
SECRET_VALUE_RE = re.compile(r"\b(?:APR-[A-Z0-9]{8,}|gsk_[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,})\b")
SECRET_PAYLOAD_KEYS = {"password", "passwd", "secret", "api_key", "access_token", "authorization", "approval_token", "token_preview", "token", "session_token", "refresh_token"}


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _erp_shell(title: str, body: str, active: str = "") -> str:
    links = "".join(
        f'<a class="{("active" if active == path else "")}" href="{path}">{label}</a>'
        for path, label in [("/erp/vendors", "Vendors"), ("/erp/invoices/new", "Invoice Entry"), ("/erp/ledger", "General Ledger")]
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{_escape(title)} · CentrAlign ERP</title><style>
      *{{box-sizing:border-box}}body{{margin:0;background:#eef1f4;color:#1f2937;font:14px/1.5 Arial,sans-serif}}header{{background:#172b4d;color:white;padding:0 28px;display:flex;align-items:center;justify-content:space-between;min-height:64px}}.erp-brand{{font-weight:700;letter-spacing:.02em}}nav{{display:flex;gap:20px;flex-wrap:wrap}}nav a{{color:#d9e2f2;text-decoration:none;padding:20px 0}}nav a.active,nav a:hover{{color:#fff;border-bottom:3px solid #7da7e8}}main{{width:min(100% - 32px,1050px);margin:28px auto 56px}}.crumb{{color:#64748b;font-size:12px;margin-bottom:8px}}h1{{font-size:24px;margin:0 0 6px}}.lede{{color:#64748b;margin:0 0 20px}}.panel{{background:#fff;border:1px solid #cbd5e1;border-radius:4px;padding:22px;box-shadow:0 2px 4px #0f172a0c}}.alert{{padding:12px 14px;border:1px solid #fca5a5;background:#fef2f2;color:#b91c1c;margin-bottom:16px;border-radius:3px;font-weight:600}}.success{{padding:12px 14px;border:1px solid #86efac;background:#f0fdf4;color:#166534;margin-bottom:16px;border-radius:3px}}.form-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 18px}}label.field{{display:grid;gap:5px;color:#334155;font-weight:600;font-size:12px}}input,select{{width:100%;height:38px;border:1px solid #94a3b8;border-radius:3px;padding:7px 9px;background:white;color:#111827;font-size:14px}}input:focus,select:focus{{outline:2px solid #93c5fd;border-color:#2563eb}}.wide{{grid-column:1/-1}}.check-row{{display:flex;align-items:center;gap:9px;border:1px solid #dbe2ea;padding:10px;background:#f8fafc;font-weight:600}}.check-row input{{width:16px;height:16px}}button,.submit{{display:inline-flex;align-items:center;justify-content:center;border:1px solid #163b70;border-radius:3px;background:#1d4d89;color:#fff;padding:10px 16px;font-weight:700;cursor:pointer}}button:hover{{background:#163b70}}.actions{{display:flex;justify-content:flex-end;margin-top:18px}}table{{width:100%;border-collapse:collapse;background:white}}th{{text-align:left;background:#e8edf3;color:#334155;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}th,td{{border:1px solid #cbd5e1;padding:10px 11px;vertical-align:top;overflow-wrap:anywhere}}.code{{font-family:ui-monospace,monospace}}.login-card{{max-width:460px;margin:50px auto}}.erp-note{{margin-top:18px;color:#64748b;font-size:12px}}@media(max-width:600px){{header{{padding:12px 16px;align-items:flex-start;gap:8px;flex-direction:column}}nav{{gap:14px}}nav a{{padding:6px 0}}.form-grid{{grid-template-columns:1fr}}.wide{{grid-column:auto}}.panel{{padding:15px}}.table-wrap{{overflow:auto}}table{{min-width:680px}}}}
    </style></head><body><header><div class="erp-brand">CentrAlign · Legacy AP Portal</div><nav>{links}</nav></header><main>{body}</main></body></html>"""


def _login_page(error: str = "") -> str:
    alert = f'<div id="ui_error_banner" class="alert" role="alert">{_escape(error)}</div>' if error else ""
    return _erp_shell("ERP Login", f"""<section class="panel login-card"><div class="crumb">SECURE SANDBOX ACCESS</div><h1>Employee sign in</h1><p class="lede">Sign in to the CentrAlign accounts payable portal.</p>{alert}<form method="post" action="/erp/login"><div class="form-grid"><label class="field wide" for="email_input">Company email<input id="email_input" name="email" type="email" autocomplete="username" required></label><label class="field wide" for="password_input">Password<input id="password_input" name="password" type="password" autocomplete="current-password" required></label></div><div class="actions"><button id="login_btn" type="submit">Sign in</button></div></form><p class="erp-note">Sandbox account access is maintained in CentrAlign company memory.</p></section>""")


def _vendors_page(vendors: list[dict[str, str]]) -> str:
    rows = "".join(f"<tr><td class=\"code\">{_escape(vendor['code'])}</td><td>{_escape(vendor['short_name'])}</td><td>{_escape(vendor['legal_name'])}</td><td>Approved</td></tr>" for vendor in vendors)
    return _erp_shell("Vendor Directory", f"""<div class="crumb">ACCOUNTS PAYABLE / REFERENCE DATA</div><h1>Vendor Directory</h1><p class="lede">Use the exact legal entity name shown here when entering an invoice.</p><section class="panel"><table id="vendor_directory"><thead><tr><th>ERP Vendor Code</th><th>Short Name</th><th>Exact Legal Entity</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></section>""", "/erp/vendors")


def _invoice_form(values: dict[str, str] | None = None, error: str = "") -> str:
    values = values or {}
    alert = f'<div id="ui_error_banner" class="alert" role="alert">{_escape(error)}</div>' if error else ""
    fields = [
        ("Vendor legal entity", "vendor_input", "vendor", "text", "Exact legal name from Vendor Directory"),
        ("Invoice number", "invoice_number_input", "invoice_number", "text", ""),
        ("Invoice date", "invoice_date_input", "invoice_date", "date", ""),
        ("Due date", "due_date_input", "due_date", "date", ""),
        ("Amount", "amount_input", "amount", "number", ""),
        ("Currency", "currency_input", "currency", "text", "INR"),
        ("Source file", "source_file_input", "source_file", "text", ""),
        ("Source SHA-256", "source_sha256_input", "source_sha256", "text", ""),
    ]
    controls = "".join(f'<label class="field" for="{ident}">{label}<input id="{ident}" name="{name}" type="{kind}" step="{("0.01" if kind == "number" else "any")}" placeholder="{_escape(placeholder)}" value="{_escape(values.get(name, ""))}" required></label>' for label, ident, name, kind, placeholder in fields)
    approval_checked = " checked" if values.get("approval_verified") == "on" else ""
    return _erp_shell("New Invoice", f"""<div class="crumb">ACCOUNTS PAYABLE / INVOICE ENTRY</div><h1>New Invoice Entry</h1><p class="lede">Enter an invoice for review and posting to the general ledger.</p><section class="panel">{alert}<form method="post" action="/erp/invoices/new"><div class="form-grid">{controls}<div class="wide"><label class="check-row" for="approval_verified"><input id="approval_verified" name="approval_verified" type="checkbox"{approval_checked}> Manager Approval Token Verified</label></div><label class="field wide" for="approval_token">Manager approval token<input id="approval_token" name="approval_token" type="password" value="{_escape(values.get("approval_token", ""))}" autocomplete="off"></label></div><div class="actions"><button id="submit_btn" type="submit">Submit Invoice</button></div></form></section>""", "/erp/invoices/new")


def _ledger_page(bills: list[dict[str, Any]], posted: str | None = None) -> str:
    banner = f'<div id="post_success" class="success" role="status">Invoice {"already existed" if posted == "already_exists" else "posted"} successfully. Ledger read-back is available below.</div>' if posted else ""
    if bills:
        rows = []
        for bill in bills:
            rows.append(
                f'<tr data-invoice-number="{_escape(bill["invoice_number"])}">'
                f'<td class="code" data-field="bill_id">#{_escape(bill["id"])}</td>'
                f'<td data-field="vendor">{_escape(bill["vendor"])}</td>'
                f'<td class="code" data-field="invoice_number">{_escape(bill["invoice_number"])}</td>'
                f'<td class="code" data-field="invoice_date">{_escape(bill["invoice_date"])}</td>'
                f'<td class="code" data-field="due_date">{_escape(bill["due_date"])}</td>'
                f'<td class="code" data-field="amount">{_escape(bill["currency"])} {_escape(bill["amount"])}</td>'
                f'<td class="code" data-field="currency">{_escape(bill["currency"])}</td></tr>'
            )
        row_html = "".join(rows)
    else:
        row_html = '<tr><td colspan="7">No posted invoices yet.</td></tr>'
    return _erp_shell("Posted General Ledger", f"""<div class="crumb">ACCOUNTS PAYABLE / FINANCE</div><h1>Posted General Ledger</h1><p class="lede">Read-only view of invoices recorded in the sandbox ledger.</p>{banner}<section class="panel table-wrap"><table id="ledger_table"><thead><tr><th>Bill ID</th><th>Vendor</th><th>Invoice Number</th><th>Invoice Date</th><th>Due Date</th><th>Amount</th><th>Currency</th></tr></thead><tbody>{row_html}</tbody></table></section>""", "/erp/ledger")


class RunCreate(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    approval_mode: Literal["policy_driven", "always_gate"] = "policy_driven"
    session_id: str = Field(default="", max_length=128)


class ApprovalDecision(BaseModel):
    approved: bool


class RunManager:
    """Owns async tasks, audit events, approval futures, and short-lived ERP tokens."""

    PENDING_QUESTION_TTL_SECONDS = 600
    _AFFIRMATIVE_REPLIES = {
        "yes", "yes please", "yes just record it", "yes go ahead", "yes go ahead and record it",
        "yes just record it in the ledger", "go ahead", "go ahead and record it", "proceed",
        "proceed with just recording it", "just record it", "record it", "okay", "ok", "sure",
    }
    _NEGATIVE_REPLIES = {"no", "no thanks", "cancel", "cancel it", "do not proceed", "don't proceed", "stop", "not now"}

    def __init__(self, company: CompanyApp, data_dir: Path, offline: bool = False) -> None:
        self.company = company
        self.data_dir = data_dir
        self.offline = offline
        self._sequence: dict[str, int] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._pending: dict[str, asyncio.Future[str | None]] = {}
        self._approval_payload: dict[str, dict[str, Any]] = {}
        self._tokens: dict[str, dict[str, Any]] = {}
        self._pending_question_contexts: dict[str, dict[str, Any]] = {}
        self._pending_question_lock = threading.Lock()
        self.run_log_dir = self.data_dir / "runs"
        self.run_log_dir.mkdir(parents=True, exist_ok=True)
        self._run_log_lock = threading.Lock()
        self._known_secrets: set[str] = set()
        memory_path = self.data_dir / "policies" / "company_memory.md"
        if memory_path.exists():
            for line in memory_path.read_text(encoding="utf-8").splitlines():
                if line.casefold().startswith(("email:", "password:")):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        self._known_secrets.add(value)
        for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(key, "").strip()
            if value:
                self._known_secrets.add(value)

    def _redact_event_value(self, value: Any, key: str = "") -> Any:
        normalized_key = key.casefold().replace("-", "_")
        if normalized_key in SECRET_PAYLOAD_KEYS or normalized_key.endswith(("_token", "_api_key")):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {str(item_key): self._redact_event_value(item_value, str(item_key)) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [self._redact_event_value(item) for item in value]
        if isinstance(value, str):
            safe = value
            for secret in self._known_secrets:
                safe = safe.replace(secret, "[REDACTED]")
            return SECRET_VALUE_RE.sub("[REDACTED]", safe)
        return value

    def event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        sequence = self._sequence.get(run_id, 0) + 1
        self._sequence[run_id] = sequence
        safe_payload = self._redact_event_value(payload)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.company.append_event(run_id, sequence, event_type, safe_payload)
        line = {"sequence": sequence, "timestamp": timestamp, "event_type": event_type, "payload": safe_payload}
        with self._run_log_lock:
            with (self.run_log_dir / f"{run_id}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _reply_kind(text: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9' ]+", " ", text.casefold())
        normalized = " ".join(normalized.split())
        return "affirmative" if normalized in RunManager._AFFIRMATIVE_REPLIES else "negative" if normalized in RunManager._NEGATIVE_REPLIES else None

    @staticmethod
    def _is_bare_affirmative(text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9' ]+", " ", text.casefold())
        return " ".join(normalized.split()) in RunManager._AFFIRMATIVE_REPLIES

    def _pop_pending_question(self, session_id: str) -> tuple[dict[str, Any] | None, bool]:
        if not session_id:
            return None, False
        with self._pending_question_lock:
            pending = self._pending_question_contexts.pop(session_id, None)
        if not pending:
            return None, False
        expires_at = float(pending.get("expires_at_epoch", 0))
        expired = expires_at <= datetime.now(timezone.utc).timestamp()
        return (None, True) if expired else (pending, False)

    def _save_pending_question(self, session_id: str, run_id: str, original_task: str, status: str) -> None:
        if not session_id or status != "needs_input":
            return
        events = self.company.read_audit(run_id)
        intent = next((event.get("payload", {}) for event in reversed(events) if event.get("event_type") == "task_intent"), {})
        final = next((event.get("payload", {}) for event in reversed(events) if event.get("event_type") == "final_result"), {})
        summary = str(final.get("summary", ""))
        actions = intent.get("requested_actions", []) if isinstance(intent, dict) else []
        unsupported = []
        for item in actions:
            if isinstance(item, dict) and item.get("capability") == "unsupported":
                label = str(item.get("action", "")).strip()
                if label and label not in unsupported:
                    unsupported.append(label)
        vendor = str(intent.get("vendor_query", "")).strip() if isinstance(intent, dict) else ""
        vendor_known = any(
            vendor.casefold() in {str(item.get("short_name", "")).casefold(), str(item.get("legal_name", "")).casefold()}
            for item in self.company.list_vendor_directory()
        ) if vendor else False
        proposed_action = "enter_invoice" if vendor_known and unsupported and "proceed with just recording" in summary.casefold() else None
        candidate = None
        invoice_number = str(intent.get("invoice_number", "")).strip() if isinstance(intent, dict) else ""
        criterion = str(intent.get("selection_criterion", "other")) if isinstance(intent, dict) else "other"
        reason = str(intent.get("reason", "")) if isinstance(intent, dict) else ""
        candidates: list[dict[str, str]] = []
        if proposed_action and vendor:
            try:
                candidates = list_invoice_candidates(self.data_dir, vendor)
            except Exception:
                candidates = []
            if invoice_number:
                candidate = next((item for item in candidates if item.get("invoice_number", "").casefold() == invoice_number.casefold()), None)
            elif (
                len(candidates) == 1
                and criterion == "other"
                and not re.search(r"\b20\d{2}[-/]\d{1,2}", original_task)
                and not re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", original_task, re.IGNORECASE)
            ):
                candidate = candidates[0]
                invoice_number = str(candidate.get("invoice_number", ""))
                criterion = "specific_invoice_number"
                reason = f"The vendor has one parsed inbox invoice, {invoice_number}."
        now = datetime.now(timezone.utc)
        pending = {
            "original_task": original_task,
            "requested_actions": actions if isinstance(actions, list) else [],
            "vendor": vendor,
            "vendor_known": vendor_known,
            "proposed_reduced_action": proposed_action,
            "unsupported_actions": unsupported,
            "selection_criterion": criterion,
            "selection_reason": reason,
            "invoice_number": invoice_number,
            "candidate": candidate,
            "candidates": [
                {key: item.get(key, "") for key in ("path", "invoice_number", "invoice_date", "due_date", "amount", "currency")}
                for item in candidates
            ],
            "created_at": now.isoformat(timespec="seconds"),
            "expires_at": datetime.fromtimestamp(now.timestamp() + self.PENDING_QUESTION_TTL_SECONDS, timezone.utc).isoformat(timespec="seconds"),
            "expires_at_epoch": now.timestamp() + self.PENDING_QUESTION_TTL_SECONDS,
        }
        with self._pending_question_lock:
            self._pending_question_contexts[session_id] = pending
        self.event(run_id, "pending_question_stored", {
            "vendor": vendor or None, "proposed_reduced_action": proposed_action,
            "unsupported_actions": unsupported, "expires_at": pending["expires_at"],
        })

    def start(self, task: str, approval_mode: str, base_url: str, session_id: str = "") -> str:
        session_id = session_id.strip()[:128]
        pending, expired = self._pop_pending_question(session_id)
        reply = self._reply_kind(task)
        clarify = "What would you like me to confirm?"
        if pending and reply == "affirmative" and pending.get("proposed_reduced_action") == "enter_invoice":
            has_date_scope = bool(
                pending.get("invoice_number")
                or pending.get("selection_criterion") == "latest_by_invoice_date"
                or re.search(r"\b20\d{2}[-/]\d{1,2}", str(pending.get("original_task") or ""))
                or re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", str(pending.get("original_task") or ""), re.IGNORECASE)
            )
            if has_date_scope:
                run_task = self._confirmed_task(pending)
                confirmation_context = {**pending, "confirmed": True}
                resolution = "confirmed"
            else:
                run_task = task
                confirmation_context = None
                resolution = "needs_clarification"
                clarify = self._pending_clarification(pending)
        elif pending and reply == "negative":
            run_task = task
            confirmation_context = None
            resolution = "cancelled"
        elif pending and reply == "affirmative":
            run_task = task
            confirmation_context = None
            resolution = "needs_clarification"
            clarify = self._pending_clarification(pending)
        elif pending:
            run_task = task
            confirmation_context = None
            resolution = "discarded_new_task"
        elif self._is_bare_affirmative(task):
            run_task = task
            confirmation_context = None
            resolution = "needs_clarification"
        else:
            run_task = task
            confirmation_context = None
            resolution = "expired" if expired else None

        run_id = uuid.uuid4().hex[:12]
        self.company.start_run(run_id, run_task)
        self._sequence[run_id] = 0
        cancel_event = threading.Event()
        self._cancel_events[run_id] = cancel_event
        self.event(run_id, "task_started", {"task": run_task, "user_reply": task, "approval_mode": approval_mode, "runtime": "single-tool-call-agent-loop"})
        if resolution:
            self.event(run_id, "pending_question_reply", {
                "status": resolution,
                "message": "Replying to previous question" if resolution in {"confirmed", "cancelled", "needs_clarification"} and pending else "Previous question context expired; treating this as a new task." if expired else "Previous question context discarded; treating this as a new task.",
                "original_task": pending.get("original_task") if pending else None,
                "proposed_reduced_action": pending.get("proposed_reduced_action") if pending else None,
                "unsupported_actions": pending.get("unsupported_actions", []) if pending else [],
            })
        if resolution == "cancelled":
            summary = "Canceled the proposal. Nothing was recorded."
            not_done = list(pending.get("unsupported_actions", []))
            self._start_host_only_finish(run_id, "stopped_by_user", summary, pending, not_done)
            task_handle = asyncio.create_task(asyncio.sleep(0), name=f"operator-{run_id}")
        elif resolution == "needs_clarification":
            self._start_host_only_finish(run_id, "needs_input", clarify, pending or {}, list((pending or {}).get("unsupported_actions", [])))
            task_handle = asyncio.create_task(asyncio.sleep(0), name=f"operator-{run_id}")
        else:
            task_handle = asyncio.create_task(
                self._execute(run_id, run_task, approval_mode, base_url, session_id, confirmation_context, cancel_event),
                name=f"operator-{run_id}",
            )
        self._tasks[run_id] = task_handle
        return run_id

    @staticmethod
    def _confirmed_task(pending: dict[str, Any]) -> str:
        vendor = str(pending.get("vendor") or "").strip()
        invoice_number = str(pending.get("invoice_number") or "").strip()
        if invoice_number:
            return f"Record invoice {invoice_number} from {vendor} in the ledger."
        criterion = pending.get("selection_criterion")
        if criterion == "latest_by_invoice_date":
            return f"Record the latest invoice from {vendor} in the ledger."
        original_task = str(pending.get("original_task") or "")
        explicit_date = re.search(r"\b(20\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?)\b", original_task)
        if explicit_date:
            return f"Record the invoice dated {explicit_date.group(1)} from {vendor} in the ledger."
        month = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", original_task, re.IGNORECASE)
        if month and criterion == "other":
            return f"Record the {month.group(1)} invoice from {vendor} in the ledger."
        return f"Record an invoice from {vendor} in the ledger."

    @staticmethod
    def _pending_clarification(pending: dict[str, Any]) -> str:
        if pending.get("vendor") and not pending.get("vendor_known"):
            return f"I couldn't identify vendor {pending['vendor']} in the company directory. Which listed vendor should I use?"
        candidates = pending.get("candidates") or []
        if pending.get("proposed_reduced_action") == "enter_invoice" and candidates:
            options = "; ".join(
                f"{item.get('invoice_number')} dated {item.get('invoice_date')} for {item.get('currency')} {item.get('amount')}"
                for item in candidates[:8]
            )
            return f"I can proceed with invoice entry only. Which {pending.get('vendor')} invoice should I record? Candidates: {options}."
        if pending.get("proposed_reduced_action") == "enter_invoice":
            return f"I couldn't find a parsed invoice candidate for {pending.get('vendor')}. Please check the inbox or clarify the vendor."
        return "What would you like me to confirm?"

    def _start_host_only_finish(self, run_id: str, status: str, summary: str, pending: dict[str, Any], not_done: list[str]) -> None:
        unique_not_done = list(dict.fromkeys(str(item) for item in not_done if str(item).strip()))
        result = {
            "status": status,
            "summary": summary,
            "what_was_done": "No requested actions were completed.",
            "requested_actions": pending.get("requested_actions", []),
            "not_done_actions": unique_not_done,
            "not_done": "none" if not unique_not_done else ", ".join(unique_not_done),
            "evidence": {},
            "elapsed_seconds": 0.0,
            "original_task": pending.get("original_task"),
        }
        self.event(run_id, "final_result", result)
        self.event(run_id, "task_finished", {"status": status, "summary": summary})
        self.company.finish_run(run_id, status, summary)

    async def request_approval(self, run_id: str, payload: dict[str, Any]) -> str | None:
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._pending[run_id] = future
        self._approval_payload[run_id] = payload
        self.event(run_id, "approval_requested", payload)
        try:
            token = await asyncio.wait_for(future, timeout=300)
        except asyncio.TimeoutError:
            token = None
        finally:
            self._pending.pop(run_id, None)
            self._approval_payload.pop(run_id, None)
        self.event(run_id, "approval_decision", {"decision": "approved" if token else "denied_or_timed_out", "token_minted": bool(token)})
        return token

    def decide(self, run_id: str, approved: bool) -> str | None:
        future = self._pending.get(run_id)
        if not future or future.done():
            return None
        token = None
        if approved:
            payload = self._approval_payload.get(run_id, {})
            invoice = payload.get("invoice", {})
            token = f"APR-{secrets.token_hex(5).upper()}"
            self._tokens[token] = {
                "run_id": run_id,
                "invoice_number": invoice.get("invoice_number"),
                "vendor": invoice.get("canonical_vendor"),
                "amount": str(invoice.get("amount")),
                "currency": invoice.get("currency"),
                "expires_at": asyncio.get_running_loop().time() + 300,
                "used": False,
            }
        future.set_result(token)
        return token

    def consume_approval_token(self, token: str, form: dict[str, str]) -> bool:
        record = self._tokens.get(token)
        if not record or record["used"] or asyncio.get_running_loop().time() > record["expires_at"]:
            return False
        expected = (record["invoice_number"], record["vendor"], record["amount"], record["currency"])
        actual = (form.get("invoice_number"), form.get("vendor"), form.get("amount"), form.get("currency"))
        if expected != actual:
            return False
        record["used"] = True
        return True

    async def _execute(
        self, run_id: str, task: str, approval_mode: str, base_url: str,
        session_id: str = "", confirmed_pending_context: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        try:
            status, summary = await run_computer_use_agent(
                self.data_dir,
                run_id,
                task,
                base_url,
                lambda event_type, payload: self.event(run_id, event_type, payload),
                lambda payload: self.request_approval(run_id, payload),
                approval_mode=approval_mode,
                offline=self.offline,
                confirmed_pending_context=confirmed_pending_context,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            status = "stopped_by_user"
            summary = "The run was stopped before completion was confirmed."
            self.event(run_id, "final_result", {
                "status": status,
                "summary": summary,
                "what_was_done": "The run was stopped before completion was confirmed.",
                "requested_actions": [],
                "not_done_actions": [],
                "not_done": "completion was not confirmed",
                "evidence": {},
                "elapsed_seconds": 0.0,
            })
        except Exception as exc:
            final_event = next(
                (event.get("payload", {}) for event in reversed(self.company.read_audit(run_id)) if event.get("event_type") == "final_result"),
                None,
            )
            if isinstance(final_event, dict) and final_event.get("status") in {"complete", "needs_input", "needs_review", "stopped_by_user"}:
                status = str(final_event["status"])
                summary = str(final_event.get("summary", "The host already finalized this run."))
            else:
                status = "needs_review"
                summary = f"The computer-use run stopped safely: {type(exc).__name__}: {exc}"
                self.event(run_id, "run_error", {"error": summary})
        self._save_pending_question(session_id, run_id, task, status)
        self.event(run_id, "task_finished", {"status": status, "summary": summary})
        self.company.finish_run(run_id, status, summary)
        self._cancel_events.pop(run_id, None)

    async def cancel_run(self, run_id: str) -> bool:
        """Stop an active run and signal any provider retry thread to exit."""
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        cancel_event = self._cancel_events.get(run_id)
        if cancel_event is not None:
            cancel_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.company.get_run(run_id)
        if not run:
            return None
        run["events"] = self.company.read_audit(run_id)
        run["awaiting_approval"] = run_id in self._pending and not self._pending[run_id].done()
        if run["awaiting_approval"]:
            run["status"] = "awaiting_approval"
        return run

    def list_runs(self) -> list[dict[str, Any]]:
        result = []
        for run in self.company.list_runs():
            result.append({**run, "awaiting_approval": run["id"] in self._pending and not self._pending[run["id"]].done()})
        return result

    async def reset_demo_data(self) -> None:
        """Stop active work before atomically clearing the demo ledger and run history."""
        current = asyncio.current_task()
        pending = [future for future in self._pending.values() if not future.done()]
        for future in pending:
            future.cancel()
        running = [task for task in self._tasks.values() if task is not current and not task.done()]
        for run_id, event in self._cancel_events.items():
            if self._tasks.get(run_id) in running:
                event.set()
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        self._tasks.clear()
        self._pending.clear()
        self._approval_payload.clear()
        self._tokens.clear()
        self._pending_question_contexts.clear()
        self._sequence.clear()
        for path in self.run_log_dir.glob("*.jsonl"):
            path.unlink(missing_ok=True)
        self.company.reset_demo_data()


def create_app(data_dir: Path | None = None, *, offline: bool = False) -> FastAPI:
    selected_data_dir = Path(data_dir or DEFAULT_DATA_DIR).resolve()
    ensure_operator_assets(selected_data_dir)
    company = CompanyApp(selected_data_dir)
    manager = RunManager(company, selected_data_dir, offline=offline)
    application = FastAPI(title="CentrAlign Operator", version="3.0.0")
    application.state.data_dir = selected_data_dir
    application.state.company = company
    application.state.run_manager = manager
    application.state.erp_sessions = set()

    def authenticated(request: Request) -> bool:
        return request.cookies.get("erp_session") in request.app.state.erp_sessions

    def login_credentials() -> tuple[str, str]:
        memory = (selected_data_dir / "policies" / "company_memory.md").read_text(encoding="utf-8")
        email = re.search(r"^Email:\s*(.+)$", memory, re.MULTILINE)
        password = re.search(r"^Password:\s*(.+)$", memory, re.MULTILINE)
        return (email.group(1).strip() if email else "", password.group(1).strip() if password else "")

    @application.get("/", response_class=HTMLResponse)
    async def operator_home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    @application.get("/app.js")
    async def operator_script() -> FileResponse:
        return FileResponse(WEB_DIR / "app.js", media_type="text/javascript")

    @application.get("/styles.css")
    async def operator_styles() -> FileResponse:
        return FileResponse(WEB_DIR / "styles.css", media_type="text/css")

    @application.get("/api/state")
    async def api_state() -> dict[str, Any]:
        provider, model, _ = planner_config(offline=offline)
        return {
            "invoices": company.list_invoices(),
            "bills": company.list_bills(),
            "runs": manager.list_runs(),
            "files": list_workspace_files(selected_data_dir),
            "provider": provider,
            "model": model,
        }

    @application.post("/api/demo/reset")
    async def reset_demo() -> dict[str, str]:
        await manager.reset_demo_data()
        application.state.erp_sessions.clear()
        return {"status": "reset"}

    @application.post("/api/runs", status_code=202)
    async def start_run(payload: RunCreate, request: Request) -> dict[str, str]:
        host = request.url.hostname
        if host not in {"127.0.0.1", "localhost"}:
            raise HTTPException(status_code=403, detail="The Operator only starts browser actions on the local ERP host.")
        port = request.url.port or 8765
        base_url = f"http://{host}:{port}"
        run_id = manager.start(payload.task.strip(), payload.approval_mode, base_url, payload.session_id)
        return {"run_id": run_id}

    @application.get("/api/runs")
    async def list_runs() -> dict[str, list[dict[str, Any]]]:
        """Return the persisted run history used by the operator workbench."""
        return {"runs": manager.list_runs()}

    @application.get("/api/runs/{run_id}")
    async def read_run(run_id: str) -> dict[str, Any]:
        if not ID_RE.fullmatch(run_id):
            raise HTTPException(status_code=400, detail="Invalid run id.")
        run = manager.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return run

    @application.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        if not ID_RE.fullmatch(run_id):
            raise HTTPException(status_code=400, detail="Invalid run id.")
        run = manager.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        cancelled = await manager.cancel_run(run_id)
        updated = manager.get_run(run_id) or run
        return {"cancelled": cancelled, "status": updated.get("status", "unknown")}

    @application.get("/api/runs/{run_id}/screenshots/{index}")
    async def read_screenshot(run_id: str, index: int) -> FileResponse:
        if not ID_RE.fullmatch(run_id) or index < 1:
            raise HTTPException(status_code=404, detail="Screenshot not found.")
        path = selected_data_dir / "evidence" / run_id / f"frame-{index:04d}.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Screenshot not found.")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @application.post("/api/runs/{run_id}/approval")
    async def approve_run(run_id: str, payload: ApprovalDecision) -> dict[str, Any]:
        if not ID_RE.fullmatch(run_id):
            raise HTTPException(status_code=400, detail="Invalid run id.")
        token = manager.decide(run_id, payload.approved)
        if payload.approved and not token:
            raise HTTPException(status_code=409, detail="This run is not waiting for approval.")
        if not payload.approved and run_id not in manager._pending:
            raise HTTPException(status_code=409, detail="This run is not waiting for approval.")
        return {"status": "approved" if token else "denied", "approval_token": token}

    @application.get("/api/files")
    async def api_file(path: str) -> dict[str, str]:
        normalized = Path(path).as_posix()
        item = next((entry for entry in list_workspace_files(selected_data_dir) if entry["path"] == normalized), None)
        if item is None:
            raise HTTPException(status_code=404, detail="Workspace document not found.")
        full_path = selected_data_dir / normalized
        content = await asyncio.to_thread(__import__("worker").extract_document_text, selected_data_dir, normalized)
        import hashlib
        return {**item, "content": content, "sha256": hashlib.sha256(full_path.read_bytes()).hexdigest()}

    @application.get("/erp/login", response_class=HTMLResponse)
    async def erp_login() -> HTMLResponse:
        return HTMLResponse(_login_page())

    @application.post("/erp/login", response_class=HTMLResponse)
    async def erp_login_post(request: Request) -> Any:
        values = parse_qs((await request.body()).decode("utf-8", "replace"))
        email = values.get("email", [""])[0]
        password = values.get("password", [""])[0]
        expected_email, expected_password = login_credentials()
        if email != expected_email or password != expected_password:
            return HTMLResponse(_login_page("Invalid sandbox credentials."), status_code=401)
        session_id = secrets.token_urlsafe(24)
        request.app.state.erp_sessions.add(session_id)
        response = RedirectResponse("/erp/vendors", status_code=303)
        response.set_cookie("erp_session", session_id, httponly=True, samesite="strict", max_age=3600)
        return response

    @application.get("/erp/vendors", response_class=HTMLResponse)
    async def erp_vendors(request: Request) -> Any:
        if not authenticated(request):
            return RedirectResponse("/erp/login", status_code=303)
        return HTMLResponse(_vendors_page(company.list_vendor_directory()))

    @application.get("/erp/invoices/new", response_class=HTMLResponse)
    async def erp_new_invoice(request: Request) -> Any:
        if not authenticated(request):
            return RedirectResponse("/erp/login", status_code=303)
        return HTMLResponse(_invoice_form())

    @application.post("/erp/invoices/new", response_class=HTMLResponse)
    async def erp_post_invoice(request: Request) -> Any:
        if not authenticated(request):
            return RedirectResponse("/erp/login", status_code=303)
        raw = parse_qs((await request.body()).decode("utf-8", "replace"), keep_blank_values=True)
        values = {key: items[-1] for key, items in raw.items()}
        vendor = values.get("vendor", "").strip()
        legal_entities = {item["legal_name"] for item in company.list_vendor_directory()}
        if vendor not in legal_entities:
            return HTMLResponse(_invoice_form(values, "UI Validation Error: Unknown vendor entity. Look up the exact legal entity name in /erp/vendors."), status_code=422)
        try:
            amount = Decimal(values.get("amount", "").replace(",", ""))
            invoice_date = datetime.strptime(values.get("invoice_date", ""), "%Y-%m-%d").date().isoformat()
            due_date = datetime.strptime(values.get("due_date", ""), "%Y-%m-%d").date().isoformat()
        except (InvalidOperation, ValueError):
            return HTMLResponse(_invoice_form(values, "UI Validation Error: Enter a valid amount, invoice date, and due date."), status_code=422)
        if amount <= 0 or not amount.is_finite():
            return HTMLResponse(_invoice_form(values, "UI Validation Error: Amount must be a positive number."), status_code=422)
        currency = values.get("currency", "").strip().upper()
        if currency not in {"INR", "USD"}:
            return HTMLResponse(_invoice_form(values, "UI Validation Error: Currency must be INR or USD."), status_code=422)
        thresholds = company.read_company_policy().get("thresholds", {})
        threshold_text = thresholds.get(currency)
        if threshold_text is None:
            return HTMLResponse(_invoice_form(values, f"UI Validation Error: No approval threshold is configured for {currency} in procurement policy."), status_code=422)
        threshold = Decimal(threshold_text)
        invoice_number = values.get("invoice_number", "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", invoice_number):
            return HTMLResponse(_invoice_form(values, "UI Validation Error: Enter a valid invoice number."), status_code=422)
        invoice = {
            "invoice_id": invoice_number,
            "invoice_number": invoice_number,
            "vendor": vendor,
            "invoice_date": invoice_date,
            "due_date": due_date,
            "amount": f"{amount:.2f}",
            "currency": currency,
            "source_file": values.get("source_file", "manual_entry.txt").strip(),
            "source_sha256": values.get("source_sha256", ""),
            "source_fields": {"vendor": vendor, "invoice_number": invoice_number, "invoice_date": invoice_date, "due_date": due_date, "amount": f"{amount:.2f}", "currency": currency},
        }
        if amount > threshold:
            approved = values.get("approval_verified") == "on" and manager.consume_approval_token(values.get("approval_token", ""), {
                "invoice_number": invoice_number, "vendor": vendor,
                "amount": f"{amount:.2f}", "currency": currency,
            })
            if not approved:
                return HTMLResponse(_invoice_form(values, "UI Validation Error: Manager approval token must be verified before posting invoices above policy threshold."), status_code=422)
        result = company.create_bill(invoice)
        return RedirectResponse(f"/erp/ledger?posted={result['status']}", status_code=303)

    @application.get("/erp/ledger", response_class=HTMLResponse)
    async def erp_ledger(request: Request, posted: str | None = None) -> Any:
        if not authenticated(request):
            return RedirectResponse("/erp/login", status_code=303)
        safe_posted = posted if posted in {"created", "already_exists"} else None
        return HTMLResponse(_ledger_page(company.list_bills(), safe_posted))

    return application


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the CentrAlign Operator and sandbox ERP.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--reset-demo-data", action="store_true", help="clear the demo ledger and run history before starting")
    args = parser.parse_args()
    import uvicorn
    selected_app = app if args.data_dir is None and not args.offline else create_app(args.data_dir, offline=args.offline)
    if args.reset_demo_data:
        run_log_dir = selected_app.state.data_dir / "runs"
        for path in run_log_dir.glob("*.jsonl"):
            path.unlink(missing_ok=True)
        selected_app.state.company.reset_demo_data()
    uvicorn.run(selected_app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
