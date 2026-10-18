"""Playwright controls for the local sandbox ERP."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


ERP_ROUTES = {
    "/erp/login": "Employee sign-in page.",
    "/erp/vendors": "Vendor directory with approved legal entity names.",
    "/erp/invoices/new": "New invoice form for entering an invoice.",
    "/erp/ledger": "Posted invoice ledger for duplicate checks and read-back verification.",
}
FORM_SELECTORS = {
    "#email_input", "#password_input", "#vendor_input", "#invoice_number_input",
    "#invoice_date_input", "#due_date_input", "#amount_input", "#currency_input",
    "#approval_token", "#source_file_input", "#source_sha256_input",
}
APPROVAL_CHECK_SELECTORS = {"#approval_verified"}
CLICK_SELECTORS = {"#login_btn", "#submit_btn"}
SNAPSHOT_CHARACTER_CAP = 4000


def _allowed_routes() -> str:
    return ", ".join(f"{path} ({description})" for path, description in ERP_ROUTES.items())


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


class BrowserTools:
    """A per-run browser session with route, selector, and field-value checks."""

    def __init__(
        self,
        data_dir: Path,
        run_id: str,
        base_url: str,
        emit: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.data_dir = data_dir
        self.run_id = run_id
        self.base_url = base_url.rstrip("/")
        self.emit = emit
        self.playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.page: Any = None
        self.frame_index = 0
        self.evidence_dir = data_dir / "evidence" / run_id
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.invoice: dict[str, Any] | None = None
        self.approval_token: str | None = None
        self.vendor_rows: list[list[str]] = []
        self.write_submitted = False
        memory = (data_dir / "policies" / "company_memory.md").read_text(encoding="utf-8")
        email = re.search(r"^Email:\s*(.+)$", memory, re.MULTILINE)
        password = re.search(r"^Password:\s*(.+)$", memory, re.MULTILINE)
        if not email or not password:
            raise ValueError("ERP sandbox credentials are missing from company memory.")
        self.email = email.group(1).strip()
        self.password = password.group(1).strip()

    def set_invoice(self, invoice: dict[str, Any]) -> None:
        self.invoice = invoice

    def set_approval_token(self, token: str | None) -> None:
        self.approval_token = token

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        origin = urlparse(self.base_url)
        if origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Browser runtime is restricted to the local loopback ERP host.")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        self.page = await self.context.new_page()

        async def restrict_requests(route: Any) -> None:
            if route.request.url.startswith(self.base_url + "/"):
                await route.continue_()
            else:
                await route.abort()

        await self.page.route("**/*", restrict_requests)

    async def close(self) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _snapshot_data(self) -> dict[str, Any]:
        if self.page is None:
            raise ValueError("Open an ERP page before taking a browser snapshot.")
        dom_tree = await self.page.locator("body").evaluate("""body => {
          const nodes = [...body.querySelectorAll('h1,h2,h3,label,input,select,button,[role=alert],table thead th,table tbody tr')];
          return nodes.slice(0, 50).map(el => {
            const id = el.id ? `[id=${el.id}] ` : '';
            const tag = el.tagName.toLowerCase();
            const type = el.getAttribute('type') ? ` type=${el.getAttribute('type')}` : '';
            const label = el.getAttribute('aria-label') || (el.labels?.[0]?.innerText || '').trim() || el.getAttribute('placeholder') || '';
            const sensitive = el.type === 'password' || ['email_input','approval_token','source_sha256_input'].includes(el.id);
            const text = sensitive ? '[redacted]' : (el.innerText || el.value || '').trim().replace(/\\s+/g, ' ').slice(0, 120);
            const detail = label && !text.includes(label) ? ` label=${label}` : '';
            return `${id}<${tag}${type}>${detail}${text ? ` ${text}` : ''}`;
          }).join('\\n');
        }""")
        alert_locator = self.page.locator("[role='alert'], #ui_error_banner").first
        alert = None
        if await alert_locator.count() and await alert_locator.is_visible():
            alert = (await alert_locator.inner_text()).strip()
        notice_locator = self.page.locator("#post_success").first
        notice = None
        if await notice_locator.count() and await notice_locator.is_visible():
            notice = (await notice_locator.inner_text()).strip()
        inputs = await self.page.locator("input").evaluate_all("""elements => Object.fromEntries(elements.map(el => [
          el.id || el.name,
          ['email_input','password_input'].includes(el.id) ? (el.value ? '[credentials supplied by host]' : '') :
            ['approval_token','source_sha256_input'].includes(el.id) ? '[redacted]' : el.value
        ]))""")
        tables: dict[str, list[list[str]]] = {}
        for table_id in ("vendor_directory", "ledger_table"):
            locator = self.page.locator(f"#{table_id}")
            if await locator.count():
                rows = await locator.locator("tbody tr").evaluate_all(
                    "rows => rows.map(row => [...row.cells].map(cell => cell.innerText.trim()))"
                )
                if table_id == "ledger_table" and self.invoice:
                    target = str(self.invoice.get("invoice_number", ""))
                    matching = [row for row in rows if target and target in row]
                    remainder = [row for row in rows if row not in matching]
                    rows = matching + remainder[:7]
                tables[table_id] = rows[:20]
        if "vendor_directory" in tables:
            self.vendor_rows = tables["vendor_directory"]
        snapshot = {
            "url": (urlparse(self.page.url).path + (f"?{urlparse(self.page.url).query}" if urlparse(self.page.url).query else ""))[:250],
            "title": (await self.page.title())[:160],
            "dom_tree": dom_tree,
            "body_text": "\n".join(part for part in [
                await self.page.locator("h1").first.inner_text() if await self.page.locator("h1").count() else "",
                alert or "", notice or "",
            ] if part)[:800],
            "alert": alert,
            "notice": notice,
            "inputs": inputs,
            "tables": tables,
        }
        return self._bound_snapshot(snapshot)

    @staticmethod
    def _bound_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Keep forms, tables, and alerts while enforcing a compact serialized size."""
        compact = dict(snapshot)
        compact["dom_tree"] = str(compact.get("dom_tree", ""))[:1500]
        compact["body_text"] = str(compact.get("body_text", ""))[:800]
        compact["alert"] = str(compact["alert"])[:500] if compact.get("alert") else None
        compact["notice"] = str(compact["notice"])[:300] if compact.get("notice") else None
        compact["inputs"] = {
            str(key): str(value)[:180] for key, value in (compact.get("inputs") or {}).items()
        }
        compact["tables"] = {
            str(table): [[str(cell)[:160] for cell in row] for row in rows[:20]]
            for table, rows in (compact.get("tables") or {}).items()
        }

        def encoded_size() -> int:
            import json
            return len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))

        while encoded_size() > SNAPSHOT_CHARACTER_CAP:
            if len(compact["dom_tree"]) > 500:
                compact["dom_tree"] = compact["dom_tree"][: max(500, len(compact["dom_tree"]) // 2)]
            elif len(compact["body_text"]) > 250:
                compact["body_text"] = compact["body_text"][: max(250, len(compact["body_text"]) // 2)]
            elif compact["tables"] and any(compact["tables"].values()):
                table = max(compact["tables"], key=lambda key: sum(len(row) for row in compact["tables"][key]))
                compact["tables"][table] = compact["tables"][table][:-1]
            elif compact["dom_tree"]:
                compact["dom_tree"] = compact["dom_tree"][: len(compact["dom_tree"]) // 2]
            elif compact["body_text"]:
                compact["body_text"] = compact["body_text"][: len(compact["body_text"]) // 2]
            elif any(compact["inputs"].values()):
                key = max(compact["inputs"], key=lambda item: len(compact["inputs"][item]))
                compact["inputs"][key] = compact["inputs"][key][: len(compact["inputs"][key]) // 2]
            elif compact.get("title"):
                compact["title"] = compact["title"][: len(compact["title"]) // 2]
            elif compact.get("url"):
                compact["url"] = compact["url"][: len(compact["url"]) // 2]
            else:
                break
        return compact

    async def _capture(self, action: str, selector: str | None = None) -> dict[str, Any]:
        if selector and self.page is not None:
            locator = self.page.locator(selector).first
            try:
                box = await locator.bounding_box(timeout=700) if await locator.count() else None
            except Exception:
                box = None
            if box:
                await locator.evaluate("""el => {
                  document.querySelector('[data-agent-highlight]')?.remove();
                  const r = el.getBoundingClientRect();
                  const ring = document.createElement('div'); ring.dataset.agentHighlight = 'true';
                  Object.assign(ring.style, {position:'fixed', left:`${r.left-4}px`, top:`${r.top-4}px`, width:`${r.width+8}px`, height:`${r.height+8}px`, border:'3px solid #EF4444', borderRadius:'7px', boxShadow:'0 0 0 3px rgba(239,68,68,.18)', pointerEvents:'none', zIndex:'2147483647'});
                  document.body.append(ring);
                }""")
        self.frame_index += 1
        path = self.evidence_dir / f"frame-{self.frame_index:04d}.png"
        await self.page.screenshot(path=str(path), full_page=False, animations="disabled")
        current = urlparse(self.page.url)
        frame = {
            "screenshot_index": self.frame_index,
            "screenshot_url": f"/api/runs/{self.run_id}/screenshots/{self.frame_index}",
            "url": current.path + (f"?{current.query}" if current.query else ""),
            "action": action,
            "highlight": None,
            "viewport": "1280×800",
        }
        self.emit("browser_frame", frame)
        return frame

    async def _observe(self, action: str) -> dict[str, Any]:
        snapshot = await self._snapshot_data()
        frame = await self._capture(action)
        snapshot = self._bound_snapshot({**snapshot, "screenshot_index": frame["screenshot_index"]})
        self.emit("observation", {
            "tool": "browser_snapshot",
            "action": action,
            "url": urlparse(self.page.url).path,
            "dom_tree": snapshot["dom_tree"],
            "summary": snapshot["body_text"][:420],
            "alert": snapshot["alert"],
            "notice": snapshot["notice"],
            "screenshot_index": frame["screenshot_index"],
        })
        return {"status": "observed", "snapshot": snapshot}

    async def browser_open(self, path: str) -> dict[str, Any]:
        parsed = urlparse(path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or parsed.path not in ERP_ROUTES:
            raise ValueError(f"Invalid ERP route. Allowed routes: {_allowed_routes()}.")
        await self.page.goto(self.base_url + parsed.path, wait_until="domcontentloaded")
        result = await self._observe("page.goto")
        return {**result, "path": parsed.path}

    async def browser_snapshot(self) -> dict[str, Any]:
        return await self._observe("DOM snapshot")

    async def browser_fill(self, selector: str, value: str) -> dict[str, Any]:
        if selector not in FORM_SELECTORS:
            raise ValueError(f"Invalid ERP field. Allowed fields: {', '.join(sorted(FORM_SELECTORS))}.")
        path = urlparse(self.page.url).path
        if selector in {"#email_input", "#password_input"} and path != "/erp/login":
            raise ValueError("Login fields may only be filled on /erp/login. Allowed fields there: #email_input, #password_input.")
        if selector not in {"#email_input", "#password_input"} and path != "/erp/invoices/new":
            invoice_fields = sorted(FORM_SELECTORS - {"#email_input", "#password_input"})
            raise ValueError(f"Invoice fields may only be filled on /erp/invoices/new. Allowed fields there: {', '.join(invoice_fields)}.")
        if self.invoice is None and selector not in {"#email_input", "#password_input"}:
            raise ValueError("Read an invoice document before filling the ERP invoice form.")

        actual_value = value
        if selector == "#email_input":
            actual_value = self.email
        elif selector == "#password_input":
            actual_value = self.password
        elif selector == "#approval_token":
            if not self.approval_token or value != "[approved_token]":
                raise ValueError("An approved runtime token is required for this field.")
            actual_value = self.approval_token
        elif selector == "#source_sha256_input":
            if value != self.invoice["source_sha256"]:
                raise ValueError("The source hash must match the selected invoice file.")
        elif selector == "#vendor_input":
            if value != self.invoice["vendor"]:
                legal = {row[2] for row in self.vendor_rows if len(row) >= 3}
                if value not in legal:
                    raise ValueError("A replacement vendor must be read from the observed ERP vendor directory.")
                self.invoice["canonical_vendor"] = value
        else:
            expected = {
                "#invoice_number_input": "invoice_number",
                "#invoice_date_input": "invoice_date",
                "#due_date_input": "due_date",
                "#amount_input": "amount",
                "#currency_input": "currency",
                "#source_file_input": "source_file",
            }[selector]
            if value != str(self.invoice[expected]):
                raise ValueError(f"{expected.replace('_', ' ').capitalize()} must come from the selected source document.")

        await self.page.locator(selector).fill(actual_value)
        return {"status": "filled", "field": selector, "alert": await self._visible_alert()}

    async def _visible_alert(self) -> str | None:
        locator = self.page.locator("[role='alert'], #ui_error_banner").first
        if await locator.count() and await locator.is_visible():
            return (await locator.inner_text()).strip()[:500]
        return None

    async def compact_snapshot(self, action: str, selector: str | None = None) -> dict[str, Any]:
        snapshot = await self._snapshot_data()
        frame = await self._capture(action, selector)
        return self._bound_snapshot({**snapshot, "screenshot_index": frame["screenshot_index"]})

    async def browser_check(self, selector: str) -> dict[str, Any]:
        if selector not in APPROVAL_CHECK_SELECTORS:
            raise ValueError(f"Invalid ERP check selector. Allowed selectors: {', '.join(sorted(APPROVAL_CHECK_SELECTORS))}.")
        if urlparse(self.page.url).path != "/erp/invoices/new":
            raise ValueError("The manager approval control can only be checked on /erp/invoices/new.")
        if not self.approval_token:
            raise ValueError("Manager approval is required before checking this control.")
        await self.page.locator(selector).check()
        return {"status": "checked", "field": selector, "alert": await self._visible_alert()}

    async def browser_click(self, selector: str) -> dict[str, Any]:
        if selector not in CLICK_SELECTORS:
            raise ValueError(f"Invalid ERP click selector. Allowed selectors: {', '.join(sorted(CLICK_SELECTORS))}.")
        path = urlparse(self.page.url).path
        if selector == "#login_btn" and path != "/erp/login":
            raise ValueError("The login control is not available on this page.")
        if selector == "#submit_btn" and path != "/erp/invoices/new":
            raise ValueError("The submit control is not available on this page.")
        frame = await self._capture("click target", selector)
        await self.page.locator(selector).click()
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=1800)
        except Exception:
            pass
        snapshot = await self._snapshot_data()
        frame = await self._capture("locator.click", selector)
        snapshot = self._bound_snapshot({**snapshot, "screenshot_index": frame["screenshot_index"]})
        self.emit("observation", {"tool": "browser_snapshot", "action": "locator.click", "url": urlparse(self.page.url).path, "dom_tree": snapshot["dom_tree"], "summary": snapshot["body_text"][:420], "alert": snapshot["alert"], "notice": snapshot["notice"], "screenshot_index": frame["screenshot_index"]})
        if selector == "#submit_btn" and not snapshot["alert"] and urlparse(self.page.url).path == "/erp/ledger":
            self.write_submitted = True
        return {"status": "clicked", "snapshot": snapshot}

    async def verify_ledger_row(self, invoice: dict[str, Any]) -> dict[str, Any]:
        await self.browser_open("/erp/ledger")
        invoice_number = invoice["invoice_number"]
        row = self.page.locator("#ledger_table tbody tr").filter(has_text=invoice_number).first
        if not await row.count():
            return {"status": "not_found", "invoice_number": invoice_number, "already_present": False, "matches": {}}
        actual = await row.locator("[data-field]").evaluate_all("cells => Object.fromEntries(cells.map(cell => [cell.dataset.field, cell.innerText.trim()]))")
        expected_vendor = invoice.get("canonical_vendor") or invoice["vendor"]
        expected = {
            "vendor": expected_vendor,
            "invoice_number": invoice_number,
            "invoice_date": invoice["invoice_date"],
            "due_date": invoice["due_date"],
            "amount": f"{invoice['currency']} {invoice['amount']}",
            "currency": invoice["currency"],
        }
        matches = {key: _normalize(str(expected[key])) == _normalize(str(actual.get(key, ""))) for key in expected}
        self.frame_index += 1
        screenshot = self.evidence_dir / f"frame-{self.frame_index:04d}.png"
        await row.screenshot(path=str(screenshot), animations="disabled")
        frame = {
            "screenshot_index": self.frame_index,
            "screenshot_url": f"/api/runs/{self.run_id}/screenshots/{self.frame_index}",
            "url": "/erp/ledger", "action": "verification row crop",
            "highlight": None, "viewport": "cropped ledger row", "kind": "ledger_crop",
        }
        self.emit("browser_frame", frame)
        result = {
            "status": "verified" if all(matches.values()) else "mismatch",
            "invoice_number": invoice_number,
            "expected": expected,
            "actual": actual,
            "matches": matches,
            "source_vendor": invoice["vendor"],
            "canonical_vendor": expected_vendor,
            "source_file": invoice["source_file"],
            "source_sha256": invoice["source_sha256"],
            "already_present": not self.write_submitted,
            "screenshot_index": self.frame_index,
            "screenshot_url": frame["screenshot_url"],
        }
        self.emit("verification", result)
        return result
