"""Phase 1 tests for the shared tool-calling browser agent."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import uvicorn

from agent_loop import AgentLoop, TOOL_DEFINITIONS
from company_app import CompanyApp
from llm_clients import FakeLLM, ToolCall
from worker import planner_config
from web_app import RunManager, create_app


ROOT = Path(__file__).resolve().parent
TERMINAL = {"complete", "needs_input", "needs_review", "stopped_by_user"}


class LocalServer:
    """Run an app on a real loopback socket so Playwright can reach the ERP."""

    def __init__(self, app) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.port = self.socket.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        asyncio.run(self.server.serve(sockets=[self.socket]))

    def start(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.025)
        if not self.server.started:
            raise RuntimeError("Uvicorn did not start for the integration test.")
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.socket.close()


def http_request(base_url: str, path: str, payload: dict | None = None) -> tuple[int, bytes, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(base_url + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


def json_request(base_url: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    status, body, _ = http_request(base_url, path, payload)
    return status, json.loads(body) if body else {}


def wait_for_run(base_url: str, run_id: str, on_approval=None) -> dict:
    deadline = time.monotonic() + 20
    approval_handled = False
    latest: dict = {}
    while time.monotonic() < deadline:
        status, latest = json_request(base_url, f"/api/runs/{run_id}")
        if status != 200:
            raise AssertionError(f"Run detail endpoint returned HTTP {status}.")
        if latest.get("awaiting_approval") and not approval_handled:
            approval_handled = True
            if on_approval:
                on_approval(latest)
        if latest.get("status") in TERMINAL:
            return latest
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not finish: {latest.get('status')}")


class AgentLoopUnitTests(unittest.TestCase):
    def test_offline_client_is_explicit_fake_replay_and_returns_one_tool(self) -> None:
        client = FakeLLM()
        call = client.next_tool_call("record Acme invoice", [], TOOL_DEFINITIONS)
        self.assertEqual(client.name, "offline-replay")
        self.assertEqual(client.model, "FakeLLM scripted replay")
        self.assertEqual(call, ToolCall("list_files", {}))

    def test_provider_configuration_selects_openai_gemini_and_replay(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(planner_config(), ("offline-replay", "FakeLLM scripted replay", None))
            self.assertEqual(planner_config(offline=True)[0], "offline-replay")
        with patch.dict(os.environ, {"CENTRALIGN_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}, clear=True):
            self.assertEqual(planner_config(), ("openai", "gpt-4o", "test-key"))
        with patch.dict(os.environ, {"CENTRALIGN_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=True):
            self.assertEqual(planner_config(), ("gemini", "gemini-2.5-flash", "test-key"))

    def test_host_rejects_unlisted_tools_routes_and_unsafe_writes(self) -> None:
        with TemporaryDirectory(prefix="centralign-guards-") as temp_dir:
            data_dir = Path(temp_dir)
            loop = AgentLoop(
                data_dir, "guard-test", "record the Acme invoice", "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, offline=True,
            )
            fixture = ROOT / "data" / "inbox" / "northstar-hosting-7002.txt"
            shutil.copyfile(fixture, data_dir / "inbox" / fixture.name)
            with self.assertRaisesRegex(ValueError, "Unexpected tool arguments"):
                loop._validate_arguments(ToolCall("list_files", {"command": "anything"}))

            async def check_guards() -> None:
                with self.assertRaisesRegex(ValueError, "returned by list_files"):
                    await loop._execute("read_document", {"path": f"inbox/{fixture.name}"})
                await loop._execute("list_files", {})
                document = await loop._execute("read_document", {"path": f"inbox/{fixture.name}"})
                self.assertEqual(document["invoice"]["invoice_number"], "NS-7002")
                with self.assertRaisesRegex(ValueError, "limited to the four local ERP routes"):
                    await loop._execute("browser_open", {"path": "/outside"})
                with self.assertRaisesRegex(ValueError, "known ERP form controls"):
                    await loop.browser.browser_fill("#arbitrary-selector", "value")
                with self.assertRaisesRegex(ValueError, "Read the invoice and procurement policy"):
                    await loop._execute("browser_click", {"selector": "#submit_btn"})

                invoice = {
                    "vendor": "Acme Analytics", "invoice_number": "INV-TEST-1",
                    "invoice_date": "2026-10-21", "due_date": "2026-11-15",
                    "amount": "68500.00", "currency": "INR", "source_file": "invoice.txt",
                    "source_sha256": "0" * 64,
                }
                loop.selected_invoice = invoice
                loop.policy = await asyncio.to_thread(loop.company.read_company_policy)
                with self.assertRaisesRegex(ValueError, "approval is required before opening"):
                    await loop._execute("browser_open", {"path": "/erp/invoices/new"})

            asyncio.run(check_guards())

    def test_approval_token_is_invoice_bound_and_single_use(self) -> None:
        with TemporaryDirectory(prefix="centralign-token-") as temp_dir:
            data_dir = Path(temp_dir)
            manager = RunManager(CompanyApp(data_dir), data_dir, offline=True)
            invoice = {
                "invoice_number": "INV-TEST-1", "canonical_vendor": "Acme Analytics Pvt Ltd",
                "amount": "68500.00", "currency": "INR",
            }

            async def check_token() -> None:
                run_id = "token-test"
                future = asyncio.get_running_loop().create_future()
                manager._pending[run_id] = future
                manager._approval_payload[run_id] = {"invoice": invoice}
                token = manager.decide(run_id, True)
                self.assertIsNotNone(token)
                self.assertFalse(manager.consume_approval_token(token, {
                    "invoice_number": "INV-OTHER", "vendor": invoice["canonical_vendor"],
                    "amount": invoice["amount"], "currency": invoice["currency"],
                }))
                form = {
                    "invoice_number": invoice["invoice_number"], "vendor": invoice["canonical_vendor"],
                    "amount": invoice["amount"], "currency": invoice["currency"],
                }
                self.assertTrue(manager.consume_approval_token(token, form))
                self.assertFalse(manager.consume_approval_token(token, form))

            asyncio.run(check_token())


class BrowserAgentIntegrationTests(unittest.TestCase):
    def test_approval_recovery_readback_duplicate_and_denial(self) -> None:
        with TemporaryDirectory(prefix="centralign-agent-e2e-") as temp_dir:
            app = create_app(Path(temp_dir), offline=True)
            server = LocalServer(app)
            base_url = server.start()
            try:
                status, state = json_request(base_url, "/api/state")
                self.assertEqual(status, 200)
                self.assertEqual(state["provider"], "offline-replay")
                self.assertIn("policies/procurement_policy.txt", {item["path"] for item in state["files"]})
                for path in ("/", "/app.js", "/styles.css"):
                    self.assertEqual(http_request(base_url, path)[0], 200)

                status, started = json_request(base_url, "/api/runs", {
                    "task": "Find the latest Acme invoice and record it in the ledger."
                })
                self.assertEqual(status, 202)
                first_id = started["run_id"]

                def approve_after_inspection(run: dict) -> None:
                    actions = [event["payload"] for event in run["events"] if event["event_type"] == "action"]
                    self.assertFalse(any(action.get("path") == "/erp/invoices/new" for action in actions))
                    request_event = next(event for event in run["events"] if event["event_type"] == "approval_requested")
                    self.assertEqual(request_event["payload"]["invoice"]["invoice_number"], "INV-2026-2116")
                    self.assertIn("INR 50000.00 policy threshold", request_event["payload"]["reason"])
                    decision_status, decision = json_request(base_url, f"/api/runs/{first_id}/approval", {"approved": True})
                    self.assertEqual(decision_status, 200)
                    self.assertTrue(decision["approval_token"].startswith("APR-"))

                first_run = wait_for_run(base_url, first_id, approve_after_inspection)
                self.assertEqual(first_run["status"], "complete", first_run.get("summary"))
                self.assertIn("replay", next(event["payload"]["mode"] for event in first_run["events"] if event["event_type"] == "planner"))
                self.assertTrue(any(event["event_type"] == "self_correction" for event in first_run["events"]))
                verification = next(event["payload"] for event in first_run["events"] if event["event_type"] == "verification")
                self.assertEqual(verification["status"], "verified")
                self.assertTrue(all(verification["matches"].values()))
                self.assertEqual(verification["actual"]["vendor"], "Acme Analytics Pvt Ltd")
                screenshot_status, screenshot, headers = http_request(base_url, verification["screenshot_url"])
                self.assertEqual(screenshot_status, 200)
                self.assertEqual(headers.get_content_type(), "image/png")
                self.assertTrue(screenshot.startswith(b"\x89PNG\r\n\x1a\n"))

                status, history = json_request(base_url, "/api/runs")
                self.assertEqual(status, 200)
                self.assertIn(first_id, {run["id"] for run in history["runs"]})

                # A repeat verifies the existing ERP row and does not post a second bill.
                _, duplicate_start = json_request(base_url, "/api/runs", {
                    "task": "Find the latest Acme invoice and record it in the ledger."
                })
                duplicate = wait_for_run(base_url, duplicate_start["run_id"])
                self.assertEqual(duplicate["status"], "complete", duplicate.get("summary"))
                duplicate_evidence = next(event["payload"] for event in duplicate["events"] if event["event_type"] == "verification")
                self.assertTrue(duplicate_evidence["already_present"])
                self.assertEqual(len(json_request(base_url, "/api/state")[1]["bills"]), 1)

                # Declining a high-value Vertex invoice stops before opening the write form.
                _, denied_start = json_request(base_url, "/api/runs", {
                    "task": "Find the latest Vertex invoice and record it in the ledger."
                })
                denied_id = denied_start["run_id"]

                def decline_after_inspection(run: dict) -> None:
                    actions = [event["payload"] for event in run["events"] if event["event_type"] == "action"]
                    self.assertFalse(any(action.get("path") == "/erp/invoices/new" for action in actions))
                    decision_status, _ = json_request(base_url, f"/api/runs/{denied_id}/approval", {"approved": False})
                    self.assertEqual(decision_status, 200)

                denied = wait_for_run(base_url, denied_id, decline_after_inspection)
                self.assertEqual(denied["status"], "needs_input")
                self.assertFalse(any(
                    event["event_type"] == "action" and event["payload"].get("selector") == "#submit_btn"
                    for event in denied["events"]
                ))
                self.assertEqual(len(json_request(base_url, "/api/state")[1]["bills"]), 1)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
