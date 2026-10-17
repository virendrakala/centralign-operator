"""CentrAlign Operator: local FastAPI workbench and sandbox ERP."""
from __future__ import annotations

import argparse
import asyncio
import html
import json
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
from worker import ensure_operator_assets, list_workspace_files, planner_config, run_computer_use_agent

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULT_DATA_DIR = ROOT / "data"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


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


class ApprovalDecision(BaseModel):
    approved: bool


class RunManager:
    """Owns async tasks, audit events, approval futures, and short-lived ERP tokens."""

    def __init__(self, company: CompanyApp, data_dir: Path, offline: bool = False) -> None:
        self.company = company
        self.data_dir = data_dir
        self.offline = offline
        self._sequence: dict[str, int] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._pending: dict[str, asyncio.Future[str | None]] = {}
        self._approval_payload: dict[str, dict[str, Any]] = {}
        self._tokens: dict[str, dict[str, Any]] = {}

    def event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        sequence = self._sequence.get(run_id, 0) + 1
        self._sequence[run_id] = sequence
        self.company.append_event(run_id, sequence, event_type, payload)

    def start(self, task: str, approval_mode: str, base_url: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.company.start_run(run_id, task)
        self._sequence[run_id] = 0
        self.event(run_id, "task_started", {"task": task, "approval_mode": approval_mode, "runtime": "single-tool-call-agent-loop"})
        task_handle = asyncio.create_task(self._execute(run_id, task, approval_mode, base_url), name=f"operator-{run_id}")
        self._tasks[run_id] = task_handle
        return run_id

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
        self.event(run_id, "approval_decision", {"decision": "approved" if token else "denied_or_timed_out", "token_minted": bool(token), "token_preview": (token[:8] + "…") if token else None})
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

    async def _execute(self, run_id: str, task: str, approval_mode: str, base_url: str) -> None:
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
            )
        except Exception as exc:
            status = "needs_review"
            summary = f"The computer-use run stopped safely: {type(exc).__name__}: {exc}"
            self.event(run_id, "run_error", {"error": summary})
        self.event(run_id, "task_finished", {"status": status, "summary": summary})
        self.company.finish_run(run_id, status, summary)

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

    @application.post("/api/runs", status_code=202)
    async def start_run(payload: RunCreate, request: Request) -> dict[str, str]:
        host = request.url.hostname
        if host not in {"127.0.0.1", "localhost"}:
            raise HTTPException(status_code=403, detail="The Operator only starts browser actions on the local ERP host.")
        port = request.url.port or 8765
        base_url = f"http://{host}:{port}"
        run_id = manager.start(payload.task.strip(), payload.approval_mode, base_url)
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
        threshold = Decimal("50000") if currency == "INR" else Decimal("1000")
        if amount > threshold:
            approved = values.get("approval_verified") == "on" and manager.consume_approval_token(values.get("approval_token", ""), {
                "invoice_number": values.get("invoice_number", "").strip(), "vendor": vendor,
                "amount": f"{amount:.2f}", "currency": currency,
            })
            if not approved:
                return HTMLResponse(_invoice_form(values, "UI Validation Error: Manager approval token must be verified before posting invoices above policy threshold."), status_code=422)
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
    args = parser.parse_args()
    import uvicorn
    selected_app = app if args.data_dir is None and not args.offline else create_app(args.data_dir, offline=args.offline)
    uvicorn.run(selected_app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
