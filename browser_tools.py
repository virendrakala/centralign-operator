"""Playwright controls for the local sandbox ERP."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


ERP_ROUTES = {"/erp/login", "/erp/vendors", "/erp/invoices/new", "/erp/ledger"}
FORM_SELECTORS = {
    "#email_input", "#password_input", "#vendor_input", "#invoice_number_input",
    "#invoice_date_input", "#due_date_input", "#amount_input", "#currency_input",
    "#approval_token", "#source_file_input", "#source_sha256_input",
}


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
          const nodes = [...body.querySelectorAll('h1,h2,h3,label,input,select,button,a,[role=alert],table thead th,table tbody tr')];
          return nodes.slice(0, 90).map(el => {
            const id = el.id ? `[id=${el.id}] ` : '';
            const tag = el.tagName.toLowerCase();
            const type = el.getAttribute('type') ? ` type=${el.getAttribute('type')}` : '';
            const label = el.getAttribute('aria-label') || (el.labels?.[0]?.innerText || '').trim() || el.getAttribute('placeholder') || '';
            const sensitive = el.type === 'password' || ['approval_token','source_sha256_input'].includes(el.id);
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
          el.type === 'password' || ['approval_token','source_sha256_input'].includes(el.id) ? '[redacted]' : el.value
        ]))""")
        tables: dict[str, list[list[str]]] = {}
        for table_id in ("vendor_directory", "ledger_table"):
            locator = self.page.locator(f"#{table_id}")
            if await locator.count():
                tables[table_id] = await locator.locator("tbody tr").evaluate_all(
                    "rows => rows.map(row => [...row.cells].map(cell => cell.innerText.trim()))"
                )
        if "vendor_directory" in tables:
            self.vendor_rows = tables["vendor_directory"]
        return {
            "url": urlparse(self.page.url).path + (f"?{urlparse(self.page.url).query}" if urlparse(self.page.url).query else ""),
            "title": await self.page.title(),
            "dom_tree": dom_tree,
            "body_text": (await self.page.locator("body").inner_text())[:1200],
            "alert": alert,
            "notice": notice,
            "inputs": inputs,
            "tables": tables,
        }

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
            raise ValueError("Browser navigation is limited to the four local ERP routes.")
        await self.page.goto(self.base_url + parsed.path, wait_until="domcontentloaded")
        result = await self._observe("page.goto")
        return {**result, "path": parsed.path}

    async def browser_snapshot(self) -> dict[str, Any]:
        return await self._observe("DOM snapshot")

    async def browser_fill(self, selector: str, value: str) -> dict[str, Any]:
        if selector not in FORM_SELECTORS:
            raise ValueError("The agent may fill only known ERP form controls.")
        path = urlparse(self.page.url).path
        if selector in {"#email_input", "#password_input"} and path != "/erp/login":
            raise ValueError("Login fields may only be filled on the ERP login page.")
        if selector not in {"#email_input", "#password_input"} and path != "/erp/invoices/new":
            raise ValueError("Invoice fields may only be filled on the ERP invoice form.")
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
        frame = await self._capture("locator.fill", selector)
        snapshot = await self._snapshot_data()
        self.emit("observation", {
            "tool": "browser_snapshot", "action": "locator.fill",
            "url": urlparse(self.page.url).path, "dom_tree": snapshot["dom_tree"],
            "summary": snapshot["body_text"][:420], "alert": snapshot["alert"],
            "notice": snapshot["notice"], "screenshot_index": frame["screenshot_index"],
        })
        return {"status": "filled", "snapshot": snapshot}

    async def browser_check(self, selector: str) -> dict[str, Any]:
        if selector != "#approval_verified" or urlparse(self.page.url).path != "/erp/invoices/new":
            raise ValueError("The agent may check only the manager approval control on the invoice form.")
        if not self.approval_token:
            raise ValueError("Manager approval is required before checking this control.")
        await self.page.locator(selector).check()
        frame = await self._capture("locator.check", selector)
        snapshot = await self._snapshot_data()
        self.emit("observation", {"tool": "browser_snapshot", "action": "locator.check", "url": urlparse(self.page.url).path, "dom_tree": snapshot["dom_tree"], "summary": snapshot["body_text"][:420], "alert": snapshot["alert"], "notice": snapshot["notice"], "screenshot_index": frame["screenshot_index"]})
        return {"status": "checked", "snapshot": snapshot}

    async def browser_click(self, selector: str) -> dict[str, Any]:
        if selector not in {"#login_btn", "#submit_btn"}:
            raise ValueError("The agent may click only the ERP login and invoice submit controls.")
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
        snapshot["screenshot_index"] = frame["screenshot_index"]
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
