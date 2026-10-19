"""Phase 1 tests for the shared tool-calling browser agent."""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
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

from agent_loop import AgentLoop, TOOL_BY_NAME, TOOL_DEFINITIONS
from browser_tools import ERP_ROUTES, MODEL_FORM_SELECTORS, SNAPSHOT_CHARACTER_CAP, BrowserTools
from company_app import CompanyApp
from llm_clients import (
    ChatCompletionsClient,
    FakeLLM,
    GeminiClient,
    INTENT_CLASSIFICATION_TOOL,
    LiveProvidersExhausted,
    OpenAIClient,
    ProviderConfigurationError,
    ProviderRequestError,
    ResilientClient,
    ToolCall,
    _request,
    make_client,
)
from documents import ensure_operator_assets, has_prompt_injection, list_invoice_candidates
from intent_rules import deterministic_unsupported_actions
from policy import approval_required, parse_policy
from worker import planner_config
from web_app import RunManager, create_app, main as web_main


ROOT = Path(__file__).resolve().parent
TERMINAL = {"complete", "needs_input", "needs_review", "stopped_by_user"}


def _test_write_intent(_task: str, _vendors: list[dict[str, str]]) -> dict[str, Any]:
    """Explicit valid intent for injected planner test doubles that are not provider adapters."""
    return {
        "intent": "write_invoice_entry", "vendor_query": "Acme", "invoice_number": "",
        "selection_criterion": "other", "reason": "The test task selects an Acme invoice.",
        "clarifying_question": "",
        "requested_actions": [{"action": "enter invoice in the ledger", "capability": "enter_invoice"}],
    }


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
    def test_acme_invoice_fixtures_have_unique_consistent_source_fields(self) -> None:
        candidates = list_invoice_candidates(ROOT / "data", "Acme")
        by_number = {item["invoice_number"]: item for item in candidates}

        self.assertEqual(len(by_number), len(candidates))
        self.assertEqual(by_number["INV-2026-2084"]["amount"], "42000.00")
        self.assertEqual(by_number["INV-2026-2084"]["invoice_date"], "2026-08-04")
        self.assertEqual(by_number["INV-2026-2116"]["amount"], "68500.00")
        self.assertEqual(by_number["INV-2026-2116"]["invoice_date"], "2026-10-21")
        self.assertTrue(by_number["INV-2026-2084"]["path"].endswith("INV_2026_08_Acme.pdf"))
        self.assertTrue(by_number["INV-2026-2116"]["path"].endswith("INV_2026_10_Acme.pdf"))

    def test_sandbox_company_memory_is_generated_locally_and_persisted(self) -> None:
        with TemporaryDirectory(prefix="centralign-company-memory-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            memory_path = data_dir / "policies" / "company_memory.md"
            first_memory = memory_path.read_text(encoding="utf-8")
            password_match = re.search(r"^Password:\s*(.+)$", first_memory, re.MULTILINE)
            self.assertIsNotNone(password_match)
            self.assertGreaterEqual(len(password_match.group(1)), 40)
            self.assertEqual(len(CompanyApp(data_dir).list_vendor_directory()), 3)

            ensure_operator_assets(data_dir)
            self.assertEqual(memory_path.read_text(encoding="utf-8"), first_memory)

    def test_offline_client_is_explicit_fake_replay_and_returns_one_tool(self) -> None:
        client = FakeLLM()
        call = client.next_tool_call("record Acme invoice", [], TOOL_DEFINITIONS)
        self.assertEqual(client.name, "offline-replay")
        self.assertEqual(client.model, "FakeLLM scripted replay")
        self.assertEqual(call, ToolCall("list_files", {}))

    def test_intent_classifier_schema_is_closed_and_requires_every_field(self) -> None:
        schema = INTENT_CLASSIFICATION_TOOL["parameters"]
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(
            schema["properties"]["intent"]["enum"],
            ["read_only", "write_invoice_entry", "unsupported", "ambiguous"],
        )
        action_schema = schema["properties"]["requested_actions"]
        capability_description = action_schema["items"]["properties"]["capability"]["description"]
        self.assertIn("verify/check/confirm the submitted invoice", capability_description)
        self.assertIn("independent lookup", capability_description)
        self.assertEqual(
            action_schema["items"]["properties"]["capability"]["enum"],
            ["enter_invoice", "read_ledger", "list_candidates", "unsupported"],
        )
        self.assertEqual(
            set(TOOL_BY_NAME["request_human_approval"]["parameters"]["required"]),
            {"selection_criterion", "selection_reason"},
        )
        self.assertEqual(
            set(TOOL_BY_NAME["browser_click"]["parameters"]["required"]),
            {"selector", "selection_criterion", "selection_reason"},
        )
        client = OpenAIClient("test-key", "test-model")
        classification = {
            "intent": "write_invoice_entry", "vendor_query": "Acme", "invoice_number": "",
            "selection_criterion": "latest_by_invoice_date", "reason": "The task requests the latest invoice.",
            "clarifying_question": "", "requested_actions": [
                {"action": "record invoice in the ledger", "capability": "enter_invoice"},
            ],
        }
        with patch("llm_clients._request", return_value={"output": [{
            "type": "function_call", "name": "classify_task_intent", "arguments": json.dumps(classification),
        }]}) as request:
            self.assertEqual(client.classify_task_intent("record latest Acme invoice", [
                {"short_name": "Acme", "legal_name": "Acme Analytics Pvt Ltd"},
            ]), classification)
        sent_tools = request.call_args.args[1]["tools"]
        self.assertEqual(sent_tools[0]["name"], "classify_task_intent")
        self.assertFalse(sent_tools[0]["parameters"]["additionalProperties"])

    def test_latest_invoice_selection_rejects_older_candidate_and_accepts_newest(self) -> None:
        with TemporaryDirectory(prefix="centralign-candidate-selection-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            events = []
            loop = AgentLoop(
                data_dir, "selection-test", "Find the latest Acme invoice and record it in the ledger.",
                "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, offline=True,
            )
            async def exercise() -> tuple[list[dict], dict]:
                loop.task_intent = await loop._classify_task_intent()
                discovery = await loop._execute("list_invoice_candidates", {"vendor_query": "Acme"})
                self.assertEqual(loop.task_intent["intent"], "write_invoice_entry")
                self.assertGreaterEqual(len(discovery["candidates"]), 3)
                old_path = "inbox/INV_2026_08_Acme.pdf"
                with self.assertRaisesRegex(ValueError, r"INV-2026-2116.*2026-10-21"):
                    await loop._execute("read_document", {"path": old_path})
                loop.policy = await asyncio.to_thread(loop.company.read_company_policy)
                with self.assertRaisesRegex(ValueError, r"Latest-by-invoice-date selection rejected.*INV-2026-2116"):
                    await loop._execute("request_human_approval", {
                        "invoice_number": "INV-2026-2084", "canonical_vendor": "Acme Analytics Pvt Ltd",
                        "selection_criterion": "latest_by_invoice_date",
                        "selection_reason": "The task requests the latest invoice by invoice date.",
                    })
                chosen = await loop._execute("read_document", {"path": "inbox/INV_2026_10_Acme.pdf"})
                return discovery["candidates"], chosen

            candidates, chosen = asyncio.run(exercise())

        self.assertIn("inbox/INV_2026_10_Acme.pdf", {item["path"] for item in candidates})
        self.assertEqual(chosen["invoice"]["invoice_number"], "INV-2026-2116")
        self.assertEqual(loop.selection_criterion, "latest_by_invoice_date")
        check = next(payload for kind, payload in events if kind == "selection_check")
        self.assertEqual(check["status"], "passed")

    def test_explicit_month_selection_is_not_rejected_by_latest_date_guard(self) -> None:
        with TemporaryDirectory(prefix="centralign-explicit-month-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            old_copy = data_dir / "inbox" / "month-august.txt"
            old_copy.write_text(
                "Invoice Number: AUG-1\nVendor: Acme Analytics\nInvoice Date: 2026-08-12\n"
                "Due Date: 2026-09-11\nAmount Due: INR 100.00\n", encoding="utf-8"
            )
            events = []
            loop = AgentLoop(
                data_dir, "explicit-month", "Enter the August Acme invoice.",
                "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, offline=True,
            )

            async def exercise() -> dict:
                loop.task_intent = await loop._classify_task_intent()
                self.assertEqual(loop.task_intent["selection_criterion"], "other")
                await loop._execute("list_invoice_candidates", {"vendor_query": "Acme"})
                return await loop._execute("read_document", {"path": "inbox/month-august.txt"})

            selected = asyncio.run(exercise())

        self.assertEqual(selected["invoice"]["invoice_number"], "AUG-1")
        self.assertEqual(loop.selection_criterion, "other")
        self.assertTrue(any(payload.get("criterion") == "other" for kind, payload in events if kind == "selection_check"))

    def test_read_only_intent_reads_ledger_without_browser_or_write_tools(self) -> None:
        class MisleadingFakeLLM(FakeLLM):
            def next_tool_call(self, task, observations, tools):
                if any(item.get("tool") == "read_ledger" for item in observations) and not any(item.get("tool") == "finish" for item in observations):
                    return ToolCall("finish", {"status": "complete", "summary": "Acme was invoiced USD 999999.99."})
                return super().next_tool_call(task, observations, tools)

        with TemporaryDirectory(prefix="centralign-read-only-") as temp_dir:
            data_dir = Path(temp_dir)
            company = CompanyApp(data_dir)
            company.create_bill({
                "vendor": "Acme Analytics Pvt Ltd", "invoice_number": "READ-1",
                "invoice_date": "2026-01-03", "due_date": "2026-02-02",
                "amount": "100.00", "currency": "INR", "source_file": "read.txt",
            })
            loop = AgentLoop(
                data_dir, "read-only-test", "What has Acme been invoiced?", "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, client=MisleadingFakeLLM(),
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())
            bills = company.list_bills()

        self.assertEqual(status, "complete", summary)
        self.assertIn("READ-1 dated 2026-01-03 for INR 100.00", summary)
        self.assertNotIn("999999", summary)
        self.assertEqual(loop.final_result["ledger_answer"]["invoice_count"], 1)
        self.assertEqual(loop.final_result["ledger_answer"]["invoices"][0]["amount"], "100.00")
        self.assertTrue(any(item.get("tool") == "finish" for item in loop.observations))
        self.assertEqual([bill["invoice_number"] for bill in bills], ["READ-1"])
        loop.browser.start.assert_not_awaited()
        self.assertEqual({tool["name"] for tool in loop._tools_for_intent()}, {"read_ledger", "finish"})
        self.assertFalse(any(event.get("tool") in {"browser_fill", "browser_click", "request_human_approval"} for event in loop.observations))

    def test_candidate_listing_is_a_host_rendered_read_capability(self) -> None:
        with TemporaryDirectory(prefix="centralign-list-candidates-read-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            loop = AgentLoop(
                data_dir, "candidate-read", "List invoice candidates for Acme",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True,
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "complete")
        self.assertIn("invoice candidate(s) for Acme", summary)
        self.assertGreater(loop.final_result["candidate_answer"]["candidate_count"], 0)
        self.assertEqual(loop.final_result["not_done_actions"], [])
        self.assertEqual({tool["name"] for tool in loop._tools_for_intent()}, {"list_invoice_candidates", "finish"})
        loop.browser.start.assert_not_awaited()

    def test_unsupported_ambiguous_and_unknown_vendor_stop_with_question_or_refusal(self) -> None:
        requests = [
            ("Pay the Northstar invoice and email the vendor", "cannot make payments"),
            ("Enter the latest invoice", "Which vendor"),
            ("Enter the latest invoice from Mistral Systems", "couldn't identify"),
        ]
        for index, (task, expected) in enumerate(requests):
            with self.subTest(task=task), TemporaryDirectory(prefix=f"centralign-intent-{index}-") as temp_dir:
                loop = AgentLoop(
                    Path(temp_dir), f"intent-{index}", task, "http://127.0.0.1:8765",
                    lambda *_: None, lambda *_: None, offline=True,
                )
                loop.browser.start = unittest.mock.AsyncMock()
                loop.browser.close = unittest.mock.AsyncMock()
                status, summary = asyncio.run(loop.run())
                self.assertEqual(status, "needs_input")
                self.assertIn(expected.casefold(), summary.casefold())
                loop.browser.start.assert_not_awaited()

    def test_northstar_pay_and_email_request_is_blocked_and_lists_unperformed_actions(self) -> None:
        with TemporaryDirectory(prefix="centralign-multi-action-unsupported-") as temp_dir:
            events = []
            loop = AgentLoop(
                Path(temp_dir), "multi-action-unsupported", "Pay the Northstar invoice and email the vendor",
                "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, offline=True,
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())
            bills = loop.company.list_bills()

        self.assertEqual(status, "needs_input")
        self.assertIn("I can record the invoice in the ledger", summary)
        self.assertIn("cannot make payments or send email", summary)
        self.assertIn("Would you like me to proceed with just recording it?", summary)
        self.assertNotIn("Not done:", summary)
        self.assertEqual(bills, [])
        self.assertEqual(loop.final_result["not_done_actions"], ["pay", "email vendor"])
        self.assertEqual(loop.final_result["requested_actions"], [
            {"action": "pay", "capability": "unsupported"},
            {"action": "email vendor", "capability": "unsupported"},
        ])
        loop.browser.start.assert_not_awaited()
        self.assertEqual({tool["name"] for tool in loop._tools_for_intent()}, {"finish"})
        self.assertFalse(any(item.get("tool") in {"browser_fill", "browser_fill_form", "browser_click", "request_human_approval"} for item in loop.observations))
        guard = next(payload for kind, payload in events if kind == "intent_guard")
        self.assertEqual(guard["classifier_intent"], "unsupported")
        self.assertFalse(guard["write_allowed"])

    def test_explicit_enter_latest_acme_intent_passes_the_write_guard(self) -> None:
        with TemporaryDirectory(prefix="centralign-write-intent-guard-") as temp_dir:
            ensure_operator_assets(Path(temp_dir))
            loop = AgentLoop(
                Path(temp_dir), "write-intent-guard", "Enter the latest Acme invoice",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True,
            )
            intent = asyncio.run(loop._classify_task_intent())
            loop.task_intent = intent

        self.assertEqual(intent["intent"], "write_invoice_entry")
        self.assertEqual(intent["requested_actions"][0]["capability"], "enter_invoice")
        self.assertTrue(loop.intent_guard_result["explicit_entry_verb"])
        self.assertTrue(loop.intent_guard_result["write_allowed"])
        self.assertIn("browser_click", {tool["name"] for tool in loop._tools_for_intent()})

    def test_unsupported_intent_guard_is_context_and_negation_aware(self) -> None:
        allowed_phrases = [
            "Enter the Acme invoice. Do not send any email.",
            "Enter it, no need to contact the vendor.",
            "Enter the invoice; the contact at the vendor is Priya.",
            "Enter the invoice and remove nothing from the ledger.",
        ]
        for task in allowed_phrases:
            with self.subTest(task=task):
                self.assertEqual(deterministic_unsupported_actions(task), [])
                loop = AgentLoop(
                    Path("/tmp/intent-guard-unit"), "intent-guard", task, "http://127.0.0.1:8765",
                    lambda *_: None, lambda *_: None, offline=True,
                )
                # Simulate a classifier that mistakes an incidental mention for
                # an action; the independent host guard resolves the request.
                classified = {
                    "intent": "unsupported", "vendor_query": "Acme", "invoice_number": "",
                    "selection_criterion": "latest_by_invoice_date", "reason": "",
                    "clarifying_question": "", "requested_actions": [
                        {"action": "enter invoice in the ledger", "capability": "enter_invoice"},
                        {"action": "email vendor", "capability": "unsupported"},
                    ],
                }
                guarded = loop._apply_host_intent_guard(classified)
                self.assertNotEqual(guarded["intent"], "unsupported")
                if "invoice" in task.casefold():
                    self.assertEqual(guarded["intent"], "write_invoice_entry")
                    self.assertTrue(guarded["intent_guard"]["write_allowed"])
                self.assertEqual(guarded["intent_guard"]["unsupported_actions"], [])

        blocked_phrases = {
            "Enter the Acme invoice and email the vendor.": "email vendor",
            "Enter the Acme invoice and pay it.": "pay",
            "Delete the Acme invoice from the ledger.": "delete",
            "Enter the Acme invoice and let finance know.": "notify finance",
        }
        for task, expected_action in blocked_phrases.items():
            with self.subTest(task=task):
                actions = deterministic_unsupported_actions(task)
                self.assertIn(expected_action, [item["action"] for item in actions])
                loop = AgentLoop(
                    Path("/tmp/intent-guard-unit"), "intent-guard", task, "http://127.0.0.1:8765",
                    lambda *_: None, lambda *_: None, offline=True,
                )
                guarded = loop._apply_host_intent_guard({
                    "intent": "write_invoice_entry", "vendor_query": "Acme", "invoice_number": "",
                    "selection_criterion": "latest_by_invoice_date", "reason": "", "clarifying_question": "",
                    "requested_actions": [{"action": "enter invoice in the ledger", "capability": "enter_invoice"}],
                })
                self.assertNotEqual(guarded["intent"], "write_invoice_entry")
                self.assertFalse(guarded["intent_guard"]["write_allowed"])
                self.assertIn(expected_action, guarded["intent_guard"]["unsupported_actions"])

    def test_enter_and_email_is_blocked_even_when_classifier_claims_write(self) -> None:
        class WrongClassifierFake(FakeLLM):
            def classify_task_intent(self, _task, _vendors):
                return {
                    "intent": "write_invoice_entry", "vendor_query": "Acme", "invoice_number": "",
                    "selection_criterion": "latest_by_invoice_date", "reason": "Classified as a write.",
                    "clarifying_question": "", "requested_actions": [
                        {"action": "enter invoice in the ledger", "capability": "enter_invoice"},
                    ],
                }

        with TemporaryDirectory(prefix="centralign-enter-email-guard-") as temp_dir:
            loop = AgentLoop(
                Path(temp_dir), "enter-email-guard", "Enter the latest Acme invoice and email the vendor",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=WrongClassifierFake(),
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "needs_input")
        self.assertIn("cannot send email", summary)
        self.assertNotIn("Not done:", summary)
        self.assertFalse(loop.intent_guard_result["write_allowed"])
        self.assertEqual(loop.task_intent["intent"], "unsupported")
        loop.browser.start.assert_not_awaited()

    def test_host_guard_overrides_write_label_when_task_has_no_entry_verb(self) -> None:
        class WrongClassifierFake(FakeLLM):
            def classify_task_intent(self, _task, _vendors):
                return {
                    "intent": "write_invoice_entry", "vendor_query": "Northstar", "invoice_number": "",
                    "selection_criterion": "latest_by_invoice_date", "reason": "Classifier guessed write.",
                    "clarifying_question": "", "requested_actions": [
                        {"action": "enter invoice", "capability": "enter_invoice"},
                    ],
                }

        with TemporaryDirectory(prefix="centralign-host-write-guard-") as temp_dir:
            ensure_operator_assets(Path(temp_dir))
            loop = AgentLoop(
                Path(temp_dir), "host-write-guard", "Show me the latest Northstar invoice",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=WrongClassifierFake(),
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "needs_input")
        self.assertIn("does not explicitly ask me to enter, record, post, or log", summary)
        self.assertNotIn("Not done:", summary)
        self.assertEqual(loop.task_intent["intent"], "ambiguous")
        self.assertFalse(loop.intent_guard_result["write_allowed"])
        loop.browser.start.assert_not_awaited()

    def test_terminal_error_result_lists_requested_actions_that_were_not_done(self) -> None:
        class BrokenProvider(FakeLLM):
            def next_tool_call(self, _task, _observations, _tools):
                raise RuntimeError("provider unavailable")

        with TemporaryDirectory(prefix="centralign-final-result-not-done-") as temp_dir:
            ensure_operator_assets(Path(temp_dir))
            events = []
            loop = AgentLoop(
                Path(temp_dir), "final-result-not-done", "Enter the latest Acme invoice",
                "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None,
                client=BrokenProvider(),
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "needs_review")
        self.assertIn("Not done: record invoice.", summary)
        self.assertEqual(loop.final_result["not_done_actions"], ["record invoice"])
        self.assertTrue(any(kind == "final_result" for kind, _ in events))

    def test_verified_write_satisfies_misclassified_ledger_verification_action(self) -> None:
        task = "Find the latest invoice from Acme, check policy compliance, enter it into the ERP portal, and verify it in the ledger"
        with TemporaryDirectory(prefix="centralign-verify-action-not-done-") as temp_dir:
            loop = AgentLoop(
                Path(temp_dir), "verify-action-not-done", task, "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, client=FakeLLM(),
            )
            loop.task_intent = {
                "intent": "write_invoice_entry",
                "requested_actions": [
                    {"action": "enter invoice in the ledger", "capability": "enter_invoice"},
                    {"action": "verify it in the ledger", "capability": "read_ledger"},
                ],
            }
            loop.verification = {"status": "verified", "pass": True}
            self.assertEqual(loop._not_done_action_labels(), [])

    def test_prompt_injection_is_flagged_removed_and_cannot_override_parsed_amount(self) -> None:
        with TemporaryDirectory(prefix="centralign-injected-invoice-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            events = []
            loop = AgentLoop(
                data_dir, "injection-test", "Enter invoice INV-ACME-2048 from Acme.",
                "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, offline=True,
            )
            fixture_text = (data_dir / "inbox" / "acme-analytics-injection.txt").read_text(encoding="utf-8")

            async def exercise() -> dict:
                loop.task_intent = await loop._classify_task_intent()
                await loop._execute("list_invoice_candidates", {"vendor_query": "Acme"})
                result = await loop._execute("read_document", {"path": "inbox/acme-analytics-injection.txt"})
                rejected_fill = await loop._execute(
                    "browser_fill", {"selector": "#amount_input", "value": "9999999"}, include_snapshot=False
                )
                self.assertEqual(rejected_fill["status"], "error")
                self.assertIn("parsed structured fields", rejected_fill["message"])
                return result

            result = asyncio.run(exercise())

        self.assertTrue(has_prompt_injection(fixture_text))
        self.assertEqual(result["invoice"]["amount"], "900.00")
        self.assertNotIn("ignore previous instructions", result["content"].casefold())
        self.assertTrue(any(kind == "prompt_injection_detected" for kind, _ in events))

    def test_policy_source_parses_inr_and_usd_thresholds(self) -> None:
        content = (ROOT / "data" / "policies" / "procurement_policy.txt").read_text(encoding="utf-8")
        parsed = parse_policy(content, [])
        self.assertEqual(parsed["thresholds"], {"INR": "50000.00", "USD": "1000.00"})
        self.assertTrue(approval_required({"amount": "50000.01", "currency": "INR"}, parsed)[0])
        self.assertTrue(approval_required({"amount": "1000.01", "currency": "USD"}, parsed)[0])
        self.assertFalse(approval_required({"amount": "1000.00", "currency": "USD"}, parsed)[0])
        self.assertTrue((ROOT / "data" / "policies" / "procurement_policy.txt").is_file())
        self.assertFalse((ROOT / "data" / "policies" / "procurement_policy.md").exists())

    def test_provider_tool_use_failed_is_fed_back_with_schema_then_retried(self) -> None:
        class StubClient:
            name = "groq"
            model = "openai/gpt-oss-120b"

            def __init__(self):
                self.calls = []
                self.responses = [
                    ProviderRequestError(
                        "groq", self.model, 400,
                        response_body='{"error":{"code":"tool_use_failed","message":"verify_ledger_row omitted invoice_number"}}',
                        error_code="tool_use_failed", rejected_tool_name="verify_ledger_row",
                    ),
                    ToolCall("finish", {"status": "needs_review", "summary": "Stopped after test recovery."}),
                ]

            def next_tool_call(self, _task, observations, _tools):
                self.calls.append(json.loads(json.dumps(observations)))
                response = self.responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        with TemporaryDirectory(prefix="centralign-validation-retry-") as temp_dir:
            events = []
            client = StubClient()
            loop = AgentLoop(
                Path(temp_dir), "validation-retry", "record latest Acme invoice", "http://127.0.0.1:8765",
                lambda kind, payload: events.append((kind, payload)), lambda *_: None, client=client,
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, _ = asyncio.run(loop.run())

        self.assertEqual(status, "needs_review")
        self.assertEqual(len(client.calls), 2)
        feedback = next(item for item in client.calls[1] if item.get("tool") == "tool_validation_feedback")
        self.assertIn("tool_use_failed", feedback["result"]["error"])
        schema = feedback["result"]["tool_schemas"][0]
        self.assertEqual(schema["name"], "verify_ledger_row")
        self.assertEqual(set(schema["parameters"]["properties"]), {"invoice_number"})
        self.assertEqual(schema["parameters"]["required"], [])
        self.assertFalse(any(kind == "credentials_supplied" for kind, _ in events))

    def test_fake_llm_plain_text_answer_is_capped_and_nudged_to_finish(self) -> None:
        failed_text = "Acme's ledger shows invoice READ-1 for INR 100.00. " + ("detail " * 100) + "TAIL-SHOULD-NOT-BE-STORED"

        class PlainTextFakeLLM(FakeLLM):
            def __init__(self):
                self.turns = 0
                self.nudge = None

            def next_tool_call(self, task, observations, tools):
                self.turns += 1
                if self.turns == 1:
                    return ToolCall("read_ledger", {"vendor_query": "Acme"})
                if self.turns == 2:
                    raise ProviderRequestError(
                        self.name, self.model, 400,
                        response_body='{"error":{"code":"tool_use_failed","message":"Tool choice is required"}}',
                        error_code="tool_use_failed", failed_generation=failed_text,
                    )
                self.nudge = next(item for item in observations if item.get("tool") == "finish_nudge")
                return ToolCall("finish", {"status": "complete", "summary": "A model typed answer with INR 999999."})

        with TemporaryDirectory(prefix="centralign-plain-answer-nudge-") as temp_dir:
            data_dir = Path(temp_dir)
            CompanyApp(data_dir).create_bill({
                "vendor": "Acme Analytics Pvt Ltd", "invoice_number": "READ-1",
                "invoice_date": "2026-01-03", "due_date": "2026-02-02",
                "amount": "100.00", "currency": "INR", "source_file": "read.txt",
            })
            client = PlainTextFakeLLM()
            loop = AgentLoop(
                data_dir, "plain-answer-nudge", "Show me what Acme has been invoiced", "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, client=client,
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "complete")
        self.assertIn("READ-1 dated 2026-01-03 for INR 100.00", summary)
        self.assertEqual(client.turns, 3)
        self.assertEqual(client.nudge["result"]["instruction"], "Call finish with this summary.")
        self.assertLessEqual(len(client.nudge["result"]["summary"]), 300)
        self.assertNotIn("TAIL-SHOULD-NOT-BE-STORED", json.dumps(loop.observations))

    def test_repeated_provider_tool_use_failed_warns_at_two_and_stops_at_four(self) -> None:
        failure = ProviderRequestError(
            "groq", "openai/gpt-oss-120b", 400,
            response_body='{"error":{"code":"tool_use_failed","message":"verify_ledger_row arguments must be empty"}}',
            error_code="tool_use_failed", rejected_tool_name="verify_ledger_row",
        )

        class StubClient:
            name = "groq"
            model = "openai/gpt-oss-120b"

            def __init__(self):
                self.calls = []

            def next_tool_call(self, _task, observations, _tools):
                self.calls.append(json.loads(json.dumps(observations)))
                raise failure

        with TemporaryDirectory(prefix="centralign-400-retry-guard-") as temp_dir:
            client = StubClient()
            events = []
            loop = AgentLoop(
                Path(temp_dir), "400-retry-guard", "record latest Acme invoice", "http://127.0.0.1:8765",
                lambda kind, payload: events.append((kind, payload)), lambda *_: None, client=client,
            )
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "needs_input")
        self.assertIn("4 failures for verify_ledger_row", summary)
        self.assertIn("tool_use_failed", summary)
        self.assertEqual(len(client.calls), 4)
        self.assertTrue(any(item.get("tool") == "repeat_failure_warning" for item in client.calls[2]))
        counts = [payload["failure_count"] for kind, payload in events if kind == "tool_retry_budget"]
        self.assertEqual(counts, [1, 2, 3, 4])
        self.assertEqual([payload["failure_count"] for kind, payload in events if kind == "tool_retry_warning"], [2])

    def test_verify_tool_needs_no_model_fields_and_host_finalizes_with_evidence(self) -> None:
        schema = TOOL_BY_NAME["verify_ledger_row"]["parameters"]
        self.assertEqual(set(schema["properties"]), {"invoice_number"})
        self.assertEqual(schema["required"], [])
        self.assertFalse(schema["additionalProperties"])
        invoice = {
            "vendor": "Acme Analytics", "canonical_vendor": "Acme Analytics Pvt Ltd",
            "invoice_number": "INV-HOST-VERIFY", "invoice_date": "2026-10-01",
            "due_date": "2026-11-01", "amount": "60000.00", "currency": "INR",
            "source_file": "inbox/invoice.txt", "source_sha256": "d" * 64,
        }

        class StubBrowser:
            write_submitted = True
            logged_in = True
            vendor_rows = [["ACM", "Acme Analytics", "Acme Analytics Pvt Ltd", "Approved"]]

            def set_invoice(self, _invoice):
                pass

            async def verify_ledger_row(self, selected):
                self.selected = selected
                return {
                    "status": "verified", "pass": True, "invoice_number": selected["invoice_number"],
                    "already_present": False, "ledger_row_id": "#42", "screenshot_path": "/tmp/ledger.png",
                    "matches": {"vendor": True, "amount": True}, "field_matches": [],
                }

        with TemporaryDirectory(prefix="centralign-host-verify-") as temp_dir:
            events = []
            loop = AgentLoop(Path(temp_dir), "host-verify", "record invoice", "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, client=FakeLLM())
            browser = StubBrowser()
            loop.browser = browser
            loop.task_intent = _test_write_intent("", [])
            loop.documents["inbox/invoice.txt"] = {"path": "inbox/invoice.txt", "invoice": invoice}
            loop.selected_invoice = invoice
            loop.selected_invoice_path = "inbox/invoice.txt"
            loop.candidate_invoices = [{
                "path": "inbox/invoice.txt", "vendor": invoice["vendor"],
                "invoice_number": invoice["invoice_number"], "invoice_date": invoice["invoice_date"],
            }]
            loop.selection_criterion = "other"
            loop.selection_reason = "The test task selects this invoice."
            loop.approval_record = {"status": "approved", "invoice_number": invoice["invoice_number"]}
            state = loop._host_run_state()
            result = asyncio.run(loop._execute("verify_ledger_row", {}))
            self.assertIsNone(loop.finished)
            self.assertEqual(loop._host_run_state()["workflow_hint"], "Ledger verification passed; next: finish.")
            finish_result = asyncio.run(loop._execute("finish", {"status": "complete", "summary": "Invoice was verified."}))

        self.assertEqual(state["workflow_hint"], "submitted: INV-HOST-VERIFY; next: verify_ledger_row")
        self.assertIs(browser.selected, invoice)
        self.assertTrue(result["pass"])
        self.assertEqual(result["selection_check"]["status"], "passed")
        self.assertEqual(loop.finished[0], "complete")
        self.assertEqual(finish_result["status"], "finished")
        final = next(payload for kind, payload in events if kind == "final_result")
        self.assertEqual(final["evidence"]["ledger_row_id"], "#42")
        self.assertEqual(final["evidence"]["ledger_screenshot_path"], "/tmp/ledger.png")
        self.assertEqual(final["evidence"]["approval_record"]["status"], "approved")
        self.assertEqual(final["evidence"]["source_file"], "inbox/invoice.txt")
        self.assertEqual(final["evidence"]["source_sha256"], "d" * 64)
        self.assertGreaterEqual(final["elapsed_seconds"], 0)

    def test_long_provider_request_emits_pending_and_finished_trace_events(self) -> None:
        class SlowClient:
            name = "groq"
            model = "slow-test"

            @staticmethod
            def next_tool_call(*_args):
                time.sleep(0.03)
                return ToolCall("finish", {"status": "needs_review", "summary": "done"})

        with TemporaryDirectory(prefix="centralign-long-request-") as temp_dir:
            events = []
            loop = AgentLoop(Path(temp_dir), "long-request", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, client=SlowClient())
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            with patch("agent_loop.LONG_OPERATION_TRACE_SECONDS", 0.005):
                asyncio.run(loop.run())
        operations = [payload for kind, payload in events if kind == "long_operation"]
        self.assertEqual([item["status"] for item in operations], ["still_running", "finished"])
        self.assertEqual(operations[0]["operation"], "provider_request")
        self.assertGreaterEqual(operations[1]["elapsed_seconds"], 0.02)

    def test_run_jsonl_persists_every_event_with_secrets_redacted(self) -> None:
        with TemporaryDirectory(prefix="centralign-jsonl-") as temp_dir:
            data_dir = Path(temp_dir)
            company = CompanyApp(data_dir)
            company.start_run("jsonl-run", "persist this run")
            manager = RunManager(company, data_dir, offline=True)
            manager._sequence["jsonl-run"] = 0
            manager.event("jsonl-run", "sensitive_probe", {
                "password": "demo-password", "approval_token": "APR-ABCDEF123456",
                "message": "provider key gsk_12345678901234567890",
            })
            path = data_dir / "runs" / "jsonl-run.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            record = json.loads(lines[0])

        serialized = json.dumps(record)
        self.assertEqual(record["event_type"], "sensitive_probe")
        self.assertIn("[REDACTED]", serialized)
        self.assertNotIn("demo-password", serialized)
        self.assertNotIn("APR-ABCDEF123456", serialized)
        self.assertNotIn("gsk_12345678901234567890", serialized)

    def test_intent_gate_finishes_without_any_provider_calls(self) -> None:
        class CountingFakeLLM(FakeLLM):
            def __init__(self):
                self.classify_calls = 0
                self.planner_calls = 0

            def classify_task_intent(self, task, vendors):
                self.classify_calls += 1
                return super().classify_task_intent(task, vendors)

            def next_tool_call(self, task, observations, tools):
                self.planner_calls += 1
                return super().next_tool_call(task, observations, tools)

        tasks = [
            "Pay the Northstar invoice and email the vendor",
            "Enter the latest invoice",
            "Enter an invoice from Northstar",
            "Enter the latest invoice from Mistral Systems",
        ]
        for index, task in enumerate(tasks):
            with self.subTest(task=task), TemporaryDirectory(prefix="centralign-gate-no-provider-") as temp_dir:
                client = CountingFakeLLM()
                loop = AgentLoop(Path(temp_dir), f"gate-{index}", task, "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=client)
                loop.browser.start = unittest.mock.AsyncMock()
                loop.browser.close = unittest.mock.AsyncMock()
                status, _ = asyncio.run(loop.run())
                self.assertEqual(status, "needs_input")
                self.assertEqual(client.classify_calls, 0)
                self.assertEqual(client.planner_calls, 0)
                loop.browser.start.assert_not_awaited()

    def test_output_parse_failed_is_fed_back_then_retried(self) -> None:
        class ParseFailureFakeLLM(FakeLLM):
            def __init__(self):
                self.calls = []

            def next_tool_call(self, task, observations, tools):
                self.calls.append(json.loads(json.dumps(observations)))
                if len(self.calls) == 1:
                    return ToolCall("read_ledger", {"vendor_query": "Acme"})
                if len(self.calls) == 2:
                    raise ProviderRequestError(
                        self.name, self.model, 400,
                        response_body='{"error":{"code":"output_parse_failed","message":"invalid tool output"}}',
                        error_code="output_parse_failed",
                    )
                return ToolCall("finish", {"status": "complete", "summary": "Done."})

        with TemporaryDirectory(prefix="centralign-output-parse-retry-") as temp_dir:
            client = ParseFailureFakeLLM()
            loop = AgentLoop(Path(temp_dir), "output-parse-retry", "Show me what Acme has been invoiced", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=client)
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, _ = asyncio.run(loop.run())

        self.assertEqual(status, "complete")
        self.assertEqual(len(client.calls), 3)
        feedback = next(item for item in client.calls[2] if item.get("tool") == "tool_validation_feedback")
        self.assertIn("output_parse_failed", feedback["result"]["error"])

        class RepeatingParseFailure(FakeLLM):
            def __init__(self):
                self.calls = 0

            def next_tool_call(self, _task, _observations, _tools):
                self.calls += 1
                raise ProviderRequestError(
                    self.name, self.model, 400,
                    response_body='{"error":{"code":"output_parse_failed","message":"invalid tool output"}}',
                    error_code="output_parse_failed",
                )

        with TemporaryDirectory(prefix="centralign-output-parse-cap-") as temp_dir:
            client = RepeatingParseFailure()
            loop = AgentLoop(Path(temp_dir), "output-parse-cap", "Show me what Acme has been invoiced", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=client)
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())
        self.assertEqual(status, "needs_input")
        self.assertEqual(client.calls, 4)
        self.assertIn("output_parse_failed", summary)

    def test_provider_exception_after_host_final_result_does_not_fail_the_run(self) -> None:
        with TemporaryDirectory(prefix="centralign-final-result-preserved-") as temp_dir:
            data_dir = Path(temp_dir)
            manager = RunManager(CompanyApp(data_dir), data_dir, offline=True)

            async def runner(_data_dir, _run_id, _task, _base_url, emit, _approval_handler, **_kwargs):
                emit("final_result", {"status": "needs_input", "summary": "Host asked for a vendor."})
                raise ProviderRequestError("groq", "test-model", 400, response_body="output_parse_failed", error_code="output_parse_failed")

            async def check() -> None:
                with patch("web_app.run_computer_use_agent", new=runner):
                    run_id = manager.start("Enter the latest invoice", "policy_driven", "http://127.0.0.1:8765", "session")
                    await manager._tasks[run_id]
                run = manager.get_run(run_id)
                self.assertEqual(run["status"], "needs_input")
                self.assertEqual(run["summary"], "Host asked for a vendor.")
                self.assertFalse(any(event["event_type"] == "run_error" for event in run["events"]))

            asyncio.run(check())

    def test_pending_question_confirmation_is_scoped_and_policy_gate_remains(self) -> None:
        with TemporaryDirectory(prefix="centralign-pending-confirmation-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            manager = RunManager(CompanyApp(data_dir), data_dir, offline=True)
            manager.company.start_run("prior-question", "Pay the Northstar invoice and email the vendor")
            manager._sequence["prior-question"] = 0
            manager.event("prior-question", "task_intent", {
                "intent": "unsupported", "vendor_query": "Northstar", "invoice_number": "",
                "selection_criterion": "other", "reason": "The request names a Northstar invoice.",
                "requested_actions": [
                    {"action": "pay", "capability": "unsupported"},
                    {"action": "email vendor", "capability": "unsupported"},
                ],
            })
            manager.event("prior-question", "final_result", {
                "status": "needs_input",
                "summary": "I can record the invoice in the ledger, but cannot make payments or send email. Would you like me to proceed with just recording it?",
            })
            manager._save_pending_question("ui-session", "prior-question", "Pay the Northstar invoice and email the vendor", "needs_input")
            pending = manager._pending_question_contexts["ui-session"]
            self.assertEqual(pending["proposed_reduced_action"], "enter_invoice")
            self.assertEqual(pending["invoice_number"], "NS-7002")
            self.assertIn("created_at", pending)
            self.assertIn("expires_at", pending)

            async def check_confirmation() -> None:
                with patch("web_app.run_computer_use_agent", new=unittest.mock.AsyncMock(return_value=("needs_input", "selection remains pending"))) as runner:
                    run_id = manager.start("yes, just record it", "policy_driven", "http://127.0.0.1:8765", "ui-session")
                    await manager._tasks[run_id]
                    kwargs = runner.await_args.kwargs
                    self.assertIn("NS-7002", runner.await_args.args[2])
                    self.assertEqual(kwargs["confirmed_pending_context"]["unsupported_actions"], ["pay", "email vendor"])
                    trace = manager.get_run(run_id)["events"]
                    self.assertTrue(any(event["event_type"] == "pending_question_reply" and "Replying to previous question" in event["payload"]["message"] for event in trace))

            asyncio.run(check_confirmation())

            context = {**pending, "confirmed": True}
            loop = AgentLoop(
                data_dir, "confirmed-policy", manager._confirmed_task(pending), "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, confirmed_pending_context=context, client=FakeLLM(),
            )
            intent = asyncio.run(loop._classify_task_intent())
            loop.task_intent = intent
            loop.selected_invoice = {
                "vendor": "Northstar Hosting", "canonical_vendor": "Northstar Cloud Hosting",
                "invoice_number": "NS-7002", "invoice_date": "2026-10-02", "due_date": "2026-11-01",
                "amount": "68500.00", "currency": "INR", "source_file": "ns.txt", "source_sha256": "a" * 64,
            }
            loop.policy = loop.company.read_company_policy()
            loop.browser.start = unittest.mock.AsyncMock()
            self.assertEqual(intent["intent"], "write_invoice_entry")
            self.assertTrue(loop.intent_guard_result["write_allowed"])
            self.assertEqual(loop._not_done_action_labels(), ["pay", "email vendor"])
            with self.assertRaisesRegex(ValueError, "Approval required: call request_human_approval"):
                asyncio.run(loop._execute("browser_open", {"path": "/erp/invoices/new"}))

    def test_not_done_actions_are_short_deduplicated_and_omit_confirmed_entry(self) -> None:
        with TemporaryDirectory(prefix="centralign-not-done-normalization-") as temp_dir:
            loop = AgentLoop(
                Path(temp_dir), "not-done-normalization", "Pay the Acme invoice and email the vendor",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None,
                confirmed_pending_context={"confirmed": True, "proposed_reduced_action": "enter_invoice"},
                client=FakeLLM(),
            )
            loop.task_intent = {"requested_actions": [
                {"action": "record invoice in the ledger", "capability": "enter_invoice"},
                {"action": "payment", "capability": "unsupported"},
                {"action": "pay", "capability": "unsupported"},
                {"action": "email", "capability": "unsupported"},
                {"action": "email vendor", "capability": "unsupported"},
            ]}
            self.assertEqual(loop._not_done_action_labels(), ["pay", "email vendor"])

    def test_pending_question_no_cancels_and_unrelated_task_discards_context(self) -> None:
        with TemporaryDirectory(prefix="centralign-pending-replies-") as temp_dir:
            data_dir = Path(temp_dir)
            manager = RunManager(CompanyApp(data_dir), data_dir, offline=True)
            manager._pending_question_contexts["session"] = {
                "original_task": "Pay the Northstar invoice and email the vendor", "vendor": "Northstar",
                "proposed_reduced_action": "enter_invoice", "unsupported_actions": ["pay", "email vendor"],
                "requested_actions": [], "expires_at_epoch": time.time() + 600,
            }

            async def check() -> None:
                cancel_id = manager.start("no", "policy_driven", "http://127.0.0.1:8765", "session")
                await manager._tasks[cancel_id]
                canceled = manager.get_run(cancel_id)
                self.assertEqual(canceled["status"], "stopped_by_user")
                self.assertIn("Canceled the proposal", canceled["summary"])
                self.assertNotIn("Not done:", canceled["summary"])
                canceled_result = next(event["payload"] for event in canceled["events"] if event["event_type"] == "final_result")
                self.assertEqual(canceled_result["not_done_actions"], ["pay", "email vendor"])

                manager._pending_question_contexts["session"] = {
                    "original_task": "Prior request", "vendor": "Northstar", "proposed_reduced_action": "enter_invoice",
                    "unsupported_actions": ["pay"], "expires_at_epoch": time.time() + 600,
                }
                with patch("web_app.run_computer_use_agent", new=unittest.mock.AsyncMock(return_value=("complete", "new task done"))) as runner:
                    new_id = manager.start("Show me what Acme has been invoiced", "policy_driven", "http://127.0.0.1:8765", "session")
                    await manager._tasks[new_id]
                    self.assertNotIn("session", manager._pending_question_contexts)
                    self.assertIsNone(runner.await_args.kwargs["confirmed_pending_context"])
                    self.assertEqual(runner.await_args.args[2], "Show me what Acme has been invoiced")

            asyncio.run(check())

    def test_expired_context_is_ignored_and_bare_yes_without_context_clarifies(self) -> None:
        with TemporaryDirectory(prefix="centralign-expired-pending-") as temp_dir:
            manager = RunManager(CompanyApp(Path(temp_dir)), Path(temp_dir), offline=True)
            manager._pending_question_contexts["session"] = {
                "original_task": "Prior request", "vendor": "Northstar", "proposed_reduced_action": "enter_invoice",
                "unsupported_actions": ["pay"], "expires_at_epoch": time.time() - 1,
            }

            async def check() -> None:
                run_id = manager.start("yes", "policy_driven", "http://127.0.0.1:8765", "session")
                await manager._tasks[run_id]
                run = manager.get_run(run_id)
                self.assertEqual(run["status"], "needs_input")
                self.assertIn("What would you like me to confirm?", run["summary"])
                reply = next(event["payload"] for event in run["events"] if event["event_type"] == "pending_question_reply")
                self.assertEqual(reply["status"], "needs_clarification")
                self.assertNotIn("session", manager._pending_question_contexts)

            asyncio.run(check())

    def test_malformed_arguments_are_fed_back_and_retry_limit_includes_last_error(self) -> None:
        class StubClient:
            name = "openai"
            model = "test-model"

            def __init__(self, responses):
                self.responses = list(responses)
                self.calls = []

            def next_tool_call(self, _task, observations, _tools):
                self.calls.append(json.loads(json.dumps(observations)))
                response = self.responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        malformed = ToolCall("__malformed_tool_arguments__", {
            "tool_name": "verify_ledger_row", "message": "arguments were not valid JSON"
        })
        finish = ToolCall("finish", {"status": "needs_review", "summary": "Recovered."})
        with TemporaryDirectory(prefix="centralign-malformed-arguments-") as temp_dir:
            client = StubClient([malformed, finish])
            loop = AgentLoop(Path(temp_dir), "malformed", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=client)
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            asyncio.run(loop.run())
            self.assertEqual(len(client.calls), 2)
            feedback = next(item for item in client.calls[1] if item.get("tool") == "tool_validation_feedback")
            self.assertEqual(feedback["result"]["tool_schemas"][0]["name"], "verify_ledger_row")

        with TemporaryDirectory(prefix="centralign-validation-cap-") as temp_dir:
            error = ProviderRequestError(
                "groq", "test-model", 400, response_body='{"error":{"code":"tool_use_failed","message":"still missing invoice_number"}}',
                error_code="tool_use_failed", rejected_tool_name="verify_ledger_row",
            )
            client = StubClient([error, error, error, error])
            loop = AgentLoop(Path(temp_dir), "validation-cap", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=client)
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())
            self.assertEqual(status, "needs_input")
            self.assertIn("4 failures for verify_ledger_row", summary)
            self.assertIn("still missing invoice_number", summary)
            self.assertEqual(len(client.calls), 4)  # initial attempt and three retries
            self.assertTrue(any(item.get("tool") == "repeat_failure_warning" for item in client.calls[2]))
            # The repeated-error guard stops on the fourth identical 400, before
            # it can send one more retry feedback turn to the provider.
            self.assertEqual(sum(item.get("tool") == "tool_validation_feedback" for item in client.calls[-1]), 2)
            self.assertTrue(any(item.get("tool") == "repeat_failure_warning" for item in client.calls[-1]))

    def test_all_live_adapters_surface_malformed_tool_arguments(self) -> None:
        openai = OpenAIClient("test-key", "test-model")
        with patch("llm_clients._request", return_value={"output": [{
            "type": "function_call", "name": "verify_ledger_row", "arguments": "{bad json"
        }]}):
            openai_call = openai.next_tool_call("task", [], TOOL_DEFINITIONS)
        self.assertEqual(openai_call.name, "__malformed_tool_arguments__")
        self.assertEqual(openai_call.arguments["tool_name"], "verify_ledger_row")

        gemini = GeminiClient("test-key", "test-model")
        with patch("llm_clients._request", return_value={"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "verify_ledger_row", "args": ["not", "an", "object"]}}
        ]}}]}):
            gemini_call = gemini.next_tool_call("task", [], TOOL_DEFINITIONS)
        self.assertEqual(gemini_call.name, "__malformed_tool_arguments__")
        self.assertEqual(gemini_call.arguments["tool_name"], "verify_ledger_row")

        chat = ChatCompletionsClient("groq", "https://api.groq.com/openai/v1/chat/completions", "test-key", "test-model")
        with patch("llm_clients._request", return_value={"choices": [{"message": {"tool_calls": [{
            "function": {"name": "verify_ledger_row", "arguments": "not-json"}
        }]}}]}):
            chat_call = chat.next_tool_call("task", [], TOOL_DEFINITIONS)
        self.assertEqual(chat_call.name, "__malformed_tool_arguments__")
        self.assertEqual(chat_call.arguments["tool_name"], "verify_ledger_row")

    def test_browser_fill_batch_returns_one_bounded_snapshot_and_masks_login_values(self) -> None:
        snapshot = {
            "url": "/erp/ledger", "title": "Ledger", "dom_tree": "control " * 2000,
            "body_text": "row " * 500, "alert": "Check invoice field", "notice": None,
            "inputs": {"password_input": "[credentials supplied by host]"},
            "alerts": ["Check invoice field", "Second visible warning"],
            "url_path": "/erp/ledger", "page_title": "Ledger",
            "form_fields": [{"selector": "#amount_input", "label": "Amount", "type": "text", "state": "filled", "value": "100.00"}],
            "buttons": [{"selector": "#submit_btn", "label": "Submit", "enabled": True}],
            "tables": {"ledger_table": [["invoice", "x" * 1000] for _ in range(30)]},
        }
        compact = BrowserTools._bound_snapshot(snapshot)
        self.assertLessEqual(len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))), SNAPSHOT_CHARACTER_CAP)
        self.assertEqual(compact["url"], "/erp/ledger")
        self.assertEqual(compact["alert"], "Check invoice field")
        self.assertEqual(compact["alerts"], snapshot["alerts"])
        self.assertEqual(compact["form_fields"][0]["state"], "filled")
        self.assertEqual(compact["buttons"][0]["label"], "Submit")
        self.assertIn("ledger_table", compact["tables"])
        self.assertLessEqual(len(compact["tables"]["ledger_table"]), 20)
        self.assertEqual(compact["inputs"]["password_input"], "[credentials supplied by host]")
        self.assertLessEqual(len(json.dumps(BrowserTools._bound_snapshot({**snapshot, "screenshot_index": 123}), ensure_ascii=False, separators=(",", ":"))), SNAPSHOT_CHARACTER_CAP)

        class StubBrowser:
            def __init__(self):
                self.snapshots = 0
                self.fills = []

            async def browser_fill(self, selector, value):
                self.fills.append((selector, value))
                return {"status": "filled", "field": selector, "alert": None}

            async def compact_snapshot(self, *_args):
                self.snapshots += 1
                return {
                    "url": "/erp/invoices/new", "url_path": "/erp/invoices/new", "inputs": {}, "tables": {}, "alert": None,
                    "form_fields": [{"selector": "#invoice_number_input", "state": "filled", "required": True}],
                }

            @staticmethod
            def describe_form_state(snapshot):
                return BrowserTools.describe_form_state(snapshot)

        with TemporaryDirectory(prefix="centralign-fill-result-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "fill-result", "Enter invoice to test the fill", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True)
            browser = StubBrowser()
            loop.browser = browser
            loop.task_intent = _test_write_intent("", [])
            loop.selected_invoice = {"invoice_number": "INV-1", "amount": "100.00"}
            result = asyncio.run(loop._execute("browser_fill_form", {"fields": [
                {"selector": "#invoice_number_input", "value": "INV-1"},
                {"selector": "#amount_input", "value": "100.00"},
            ]}))
            self.assertEqual(browser.snapshots, 1)
            self.assertEqual(result["fields_set"], ["#invoice_number_input", "#amount_input"])
            self.assertEqual(result["form_state"]["filled_fields"], ["#invoice_number_input"])
            self.assertEqual(result["next_step"], "All required fields filled. Next: click #submit_btn.")
            self.assertNotIn("steps", result)
            browser.snapshots = 0
            sequence = asyncio.run(loop._execute("browser_action_sequence", {"actions": [
                {"tool": "browser_fill", "arguments": {"selector": "#invoice_number_input", "value": "INV-1"}},
                {"tool": "browser_fill", "arguments": {"selector": "#amount_input", "value": "100.00"}},
            ]}))
            self.assertEqual(browser.snapshots, 1)
            self.assertEqual(sequence["fields_set"], ["#invoice_number_input", "#amount_input"])
            safe = AgentLoop._redact_arguments("browser_fill_form", {"fields": [
                {"selector": "#email_input", "value": "model-email@example.test"},
                {"selector": "#password_input", "value": "model-entered-secret"},
            ]})
            self.assertNotIn("model-entered-secret", json.dumps(safe))
            self.assertTrue(all(field["value"] == "[ignored; use erp_login]" for field in safe["fields"]))

    def test_host_form_state_refreshes_after_navigation_and_failed_fill(self) -> None:
        class StubBrowser:
            def __init__(self):
                self.path = "/erp/invoices/new"
                self.fields = [
                    {"selector": "#invoice_number_input", "state": "filled", "required": True},
                    {"selector": "#amount_input", "state": "filled", "required": True},
                ]

            def snapshot(self):
                return {"url_path": self.path, "url": self.path, "form_fields": list(self.fields), "alerts": [], "alert": None}

            async def browser_open(self, path):
                self.path = path
                self.fields = []
                return {"status": "observed", "snapshot": self.snapshot()}

            async def browser_fill(self, _selector, _value):
                raise ValueError("fill rejected")

            async def compact_snapshot(self, *_args):
                return self.snapshot()

            @staticmethod
            def describe_form_state(snapshot):
                return BrowserTools.describe_form_state(snapshot)

        with TemporaryDirectory(prefix="centralign-form-state-refresh-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "form-state-refresh", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True)
            loop.browser = StubBrowser()
            loop.task_intent = _test_write_intent("", [])
            loop._last_page_snapshot = loop.browser.snapshot()
            opened = asyncio.run(loop._execute("browser_open", {"path": "/erp/ledger"}))
            after_navigation = loop._host_run_state()
            loop.browser.path = "/erp/invoices/new"
            loop.browser.fields = [
                {"selector": "#invoice_number_input", "state": "filled", "required": True},
                {"selector": "#amount_input", "state": "empty", "required": True},
            ]
            failed_fill = asyncio.run(loop._execute("browser_fill", {"selector": "#amount_input", "value": "100.00"}))
            after_failed_fill = loop._host_run_state()

        self.assertEqual(opened["snapshot"]["url_path"], "/erp/ledger")
        self.assertEqual(after_navigation["form_status"], "empty")
        self.assertEqual(failed_fill["status"], "error")
        self.assertEqual(failed_fill["snapshot"]["url_path"], "/erp/invoices/new")
        self.assertEqual(after_failed_fill["form_status"], "partial")

    def test_submit_refuses_when_required_form_fields_are_empty(self) -> None:
        invoice = {
            "vendor": "Acme Analytics", "canonical_vendor": "Acme Analytics Pvt Ltd",
            "invoice_number": "INV-EMPTY-FORM", "invoice_date": "2026-10-01",
            "due_date": "2026-11-01", "amount": "100.00", "currency": "INR",
            "source_file": "invoice.txt", "source_sha256": "a" * 64,
        }

        class StubBrowser:
            write_submitted = False
            logged_in = True
            vendor_rows = [["ACM", "Acme Analytics", "Acme Analytics Pvt Ltd", "Approved"]]

            def __init__(self):
                self.clicks = 0

            def set_invoice(self, _invoice):
                return None

            async def browser_snapshot(self):
                return {"status": "observed", "snapshot": {
                    "url_path": "/erp/invoices/new", "url": "/erp/invoices/new", "alert": None, "alerts": [],
                    "form_fields": [
                        {"selector": "#invoice_number_input", "state": "empty", "required": True},
                        {"selector": "#amount_input", "state": "empty", "required": True},
                    ],
                }}

            async def browser_click(self, _selector):
                self.clicks += 1
                return {"status": "clicked"}

            @staticmethod
            def describe_form_state(snapshot):
                return BrowserTools.describe_form_state(snapshot)

        with TemporaryDirectory(prefix="centralign-empty-submit-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "empty-submit", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True)
            browser = StubBrowser()
            loop.browser = browser
            loop.task_intent = _test_write_intent("", [])
            loop.selected_invoice = invoice
            loop.selected_invoice_path = "inbox/invoice.txt"
            loop.candidate_invoices = [{"path": "inbox/invoice.txt", "invoice_number": invoice["invoice_number"], "vendor": invoice["vendor"], "invoice_date": invoice["invoice_date"]}]
            loop.selection_criterion = "other"
            loop.selection_reason = "The test task selects this invoice."
            loop.policy = {"approved_vendors": ["Acme Analytics Pvt Ltd"], "thresholds": {"INR": "50000.00"}}
            loop.duplicate_checked = True
            result = asyncio.run(loop._execute("browser_click", {
                "selector": "#submit_btn", "selection_criterion": "other", "selection_reason": "The test task selects this invoice.",
            }))

        self.assertEqual(result["status"], "error")
        self.assertIn("required form fields are empty", result["message"])
        self.assertIn("#invoice_number_input", result["message"])
        self.assertEqual(browser.clicks, 0)

    def test_fake_llm_fill_then_verify_without_submit_returns_actionable_message(self) -> None:
        invoice = {
            "vendor": "Acme Analytics", "invoice_number": "INV-NO-SUBMIT",
            "invoice_date": "2026-10-01", "due_date": "2026-11-01",
            "amount": "60000.00", "currency": "INR", "source_file": "inbox/invoice.txt",
            "source_sha256": "a" * 64,
        }

        class FakeSequence:
            name = "offline-replay"
            model = "FakeLLM test sequence"

            def __init__(self):
                self.step = 0
                self.verify_message = None
                self.run_states = []

            def next_tool_call(self, _task, observations, _tools):
                self.step += 1
                state = next(item["result"] for item in observations if item.get("tool") == "host_run_state")
                self.run_states.append(state)
                if self.step == 1:
                    return ToolCall("browser_fill_form", {"fields": [
                        {"selector": "#invoice_number_input", "value": invoice["invoice_number"]},
                    ]})
                if self.step == 2:
                    return ToolCall("verify_ledger_row", {"invoice_number": invoice["invoice_number"]})
                verified = next(item for item in observations if item.get("tool") == "verify_ledger_row")
                self.verify_message = verified["result"].get("message")
                return ToolCall("finish", {"status": "needs_input", "summary": self.verify_message})

        class StubBrowser:
            async def start(self):
                return None

            async def close(self):
                return None

            def set_invoice(self, _invoice):
                return None

            async def browser_fill(self, selector, value):
                return {"status": "filled", "field": selector, "value": value, "alert": None}

            async def compact_snapshot(self, *_args):
                return {
                    "url_path": "/erp/invoices/new", "url": "/erp/invoices/new", "form_fields": [
                        {"selector": "#invoice_number_input", "state": "filled", "required": True},
                    ], "alert": None,
                }

            async def verify_ledger_row(self, _invoice):
                return {
                    "status": "not_found", "invoice_number": invoice["invoice_number"],
                    "warning": "Navigating away will discard unsaved form fields: #invoice_number_input.",
                }

            describe_form_state = staticmethod(BrowserTools.describe_form_state)

        with TemporaryDirectory(prefix="centralign-no-submit-verify-") as temp_dir:
            client = FakeSequence()
            events = []
            loop = AgentLoop(Path(temp_dir), "no-submit", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, client=client)
            loop.browser = StubBrowser()
            loop.documents["inbox/invoice.txt"] = {"path": "inbox/invoice.txt", "invoice": invoice}
            loop.selected_invoice = invoice
            loop.selected_invoice_path = "inbox/invoice.txt"
            loop.selection_criterion = "latest_by_invoice_date"
            loop.selection_reason = "The latest Acme invoice by invoice date."
            loop.candidate_invoices = [{
                "path": "inbox/invoice.txt", "vendor": invoice["vendor"],
                "invoice_number": invoice["invoice_number"], "invoice_date": invoice["invoice_date"],
            }]
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "needs_input", summary)
        expected = "No ledger row was found. No submit was performed. Open /erp/invoices/new, fill, click #submit_btn."
        self.assertEqual(client.verify_message, expected)
        self.assertIn(expected, summary)
        self.assertIn("Navigating away will discard unsaved form fields", json.dumps(loop.observations))
        self.assertFalse(loop.submit_performed)
        self.assertEqual(len(client.run_states), client.step)
        self.assertTrue(all({"logged_in", "selected_invoice", "legal_vendor", "approval_status", "form_status", "last_erp_alert"} <= state.keys() for state in client.run_states))
        self.assertEqual(len([1 for kind, _ in events if kind == "prompt_size"]), client.step)
        prompt_summary = next(payload for kind, payload in events if kind == "prompt_size_summary")
        self.assertGreater(prompt_summary["average_tokens_with_run_state"], prompt_summary["average_tokens_without_run_state"])

    def test_fake_llm_wrong_page_fill_returns_navigation_hint_and_retry_budget_is_consecutive(self) -> None:
        exact_error = "Current page: /erp/ledger. Call browser_open with /erp/invoices/new first."

        class StubBrowser:
            async def browser_fill(self, _selector, _value):
                raise ValueError(exact_error)

            async def compact_snapshot(self, *_args):
                return {"url_path": "/erp/ledger", "url": "/erp/ledger", "form_fields": [], "alert": None}

            describe_form_state = staticmethod(BrowserTools.describe_form_state)

        class RepeatingFakeLLM:
            name = "offline-replay"
            model = "FakeLLM repeated wrong-page fill"

            @staticmethod
            def next_tool_call(*_args):
                return ToolCall("browser_fill", {"selector": "#invoice_number_input", "value": "INV-1"})

        with TemporaryDirectory(prefix="centralign-wrong-page-fill-") as temp_dir:
            events = []
            loop = AgentLoop(Path(temp_dir), "wrong-page-fill", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, client=RepeatingFakeLLM())
            loop.browser = StubBrowser()
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())

        self.assertEqual(status, "needs_input", summary)
        self.assertIn(exact_error, summary)
        self.assertNotIn("Not done:", summary)
        self.assertEqual(loop.final_result["not_done_actions"], ["record invoice"])
        retry_events = [payload for kind, payload in events if kind == "tool_retry_budget"]
        self.assertEqual([event["failure_count"] for event in retry_events], [1, 2, 3, 4])
        self.assertEqual(retry_events[-1]["limit"], 4)
        warnings = [payload for kind, payload in events if kind == "tool_retry_warning"]
        self.assertEqual([warning["failure_count"] for warning in warnings], [2])
        self.assertTrue(any(kind == "run_error" for kind, _ in events))

        with TemporaryDirectory(prefix="centralign-consecutive-counter-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "counter", "task", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=FakeLLM())
            self.assertEqual(loop._record_tool_failure("browser_fill", "first error"), 1)
            self.assertEqual(loop._record_tool_failure("browser_fill", "different error"), 1)
            self.assertEqual(loop._record_tool_failure("browser_fill", "first error"), 2)
            loop._provider_recovery({"action": "retry_wait", "wait_seconds": 4})
            self.assertEqual(loop._record_tool_failure("browser_fill", "first error"), 3)
            self.assertEqual(loop._record_tool_failure("browser_snapshot", "first error"), 1)

    def test_approval_token_is_host_injected_and_survives_a_rejected_submit(self) -> None:
        invoice = {
            "vendor": "Acme Analytics", "canonical_vendor": "Acme Analytics Pvt Ltd",
            "invoice_number": "INV-APPROVAL-1", "invoice_date": "2026-10-01",
            "due_date": "2026-11-01", "amount": "60000.00", "currency": "INR",
            "source_file": "invoice.txt", "source_sha256": "b" * 64,
        }

        class StubBrowser:
            vendor_rows = [["ACM", "Acme Analytics", "Acme Analytics Pvt Ltd", "Approved"]]

            def __init__(self):
                self.injected = []
                self.clicks = 0

            async def inject_approval_token(self, token):
                self.injected.append(token)

            async def browser_snapshot(self):
                required = [
                    "#vendor_input", "#invoice_number_input", "#invoice_date_input",
                    "#due_date_input", "#amount_input", "#currency_input",
                ]
                return {"status": "observed", "snapshot": {
                    "url_path": "/erp/invoices/new", "url": "/erp/invoices/new", "alerts": [], "alert": None,
                    "form_fields": [{"selector": selector, "state": "filled", "required": True} for selector in required],
                }}

            async def browser_click(self, _selector):
                self.clicks += 1
                alert = "UI Validation Error: Manager approval token must be verified before posting invoices above policy threshold." if self.clicks == 1 else None
                return {"status": "clicked", "snapshot": {"alert": alert}}

            @staticmethod
            def describe_form_state(snapshot):
                return BrowserTools.describe_form_state(snapshot)

        with TemporaryDirectory(prefix="centralign-host-approval-token-") as temp_dir:
            events = []
            loop = AgentLoop(Path(temp_dir), "approval-token", "submit approved invoice", "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, approval_mode="always_gate", offline=True)
            browser = StubBrowser()
            loop.browser = browser
            loop.task_intent = _test_write_intent("", [])
            loop.selected_invoice = invoice
            loop.selected_invoice_path = "inbox/invoice.txt"
            loop.candidate_invoices = [{
                "path": "inbox/invoice.txt", "vendor": invoice["vendor"],
                "invoice_number": invoice["invoice_number"], "invoice_date": invoice["invoice_date"],
            }]
            loop.approval_tokens["inbox/invoice.txt"] = "APR-SECRET-TOKEN"
            loop.policy = {"approved_vendors": ["Acme Analytics Pvt Ltd"], "thresholds": {"INR": "50000.00"}}
            loop.duplicate_checked = True
            declaration = {"selector": "#submit_btn", "selection_criterion": "other", "selection_reason": "The test task selects this invoice."}
            first = asyncio.run(loop._execute("browser_click", declaration))
            self.assertIn("Manager approval token must be verified", first["snapshot"]["alert"])
            self.assertEqual(loop.approval_tokens["inbox/invoice.txt"], "APR-SECRET-TOKEN")
            second = asyncio.run(loop._execute("browser_click", declaration))

        self.assertEqual(second["status"], "clicked")
        self.assertEqual(browser.injected, ["APR-SECRET-TOKEN", "APR-SECRET-TOKEN"])
        injection_events = [payload for kind, payload in events if kind == "approval_token_injected"]
        self.assertEqual(len(injection_events), 2)
        self.assertNotIn("APR-SECRET-TOKEN", json.dumps(injection_events))
        token_schema = TOOL_BY_NAME["browser_fill"]["parameters"]["properties"]["selector"]["enum"]
        self.assertNotIn("#approval_token", token_schema)
        self.assertNotIn("#approval_token", TOOL_BY_NAME["browser_fill_form"]["parameters"]["properties"]["fields"]["items"]["properties"]["selector"]["enum"])

    def test_approval_ambiguity_lists_read_invoices_and_repeat_returns_already_approved(self) -> None:
        invoices = {
            path: {"path": path, "invoice": {
                "vendor": "Acme Analytics", "invoice_number": "INV-DUP",
                "invoice_date": "2026-10-01", "due_date": "2026-11-01",
                "amount": amount, "currency": "INR", "source_file": path,
                "source_sha256": "c" * 64,
            }}
            for path, amount in (("inbox/one.txt", "60000.00"), ("inbox/two.txt", "61000.00"))
        }

        class StubBrowser:
            vendor_rows = [["ACM", "Acme Analytics", "Acme Analytics Pvt Ltd", "Approved"]]

            def set_invoice(self, _invoice):
                pass

        with TemporaryDirectory(prefix="centralign-approval-ambiguity-") as temp_dir:
            handler = unittest.mock.AsyncMock(return_value="APR-SHOULD-NOT-BE-MINTED")
            loop = AgentLoop(Path(temp_dir), "approval-target", "Enter invoice and request approval", "http://127.0.0.1:8765", lambda *_: None, handler, approval_mode="always_gate", offline=True)
            loop.browser = StubBrowser()
            loop.task_intent = _test_write_intent("", [])
            loop.policy = {"approved_vendors": ["Acme Analytics Pvt Ltd"], "thresholds": {"INR": "50000.00"}}
            loop.documents = invoices
            loop.candidate_invoices = [
                {"path": path, "vendor": "Acme Analytics", "invoice_number": "INV-DUP", "invoice_date": "2026-10-01"}
                for path in invoices
            ]
            with self.assertRaises(ValueError) as ambiguous:
                asyncio.run(loop._execute("request_human_approval", {
                    "invoice_number": "INV-DUP", "canonical_vendor": "Acme Analytics Pvt Ltd",
                    "selection_criterion": "other", "selection_reason": "The test task selects this invoice.",
                }))
            message = str(ambiguous.exception)
            self.assertIn("INV-DUP | inbox/one.txt | INR 60000.00", message)
            self.assertIn("INV-DUP | inbox/two.txt | INR 61000.00", message)
            handler.assert_not_awaited()

            loop.documents = {"inbox/one.txt": invoices["inbox/one.txt"]}
            loop.approval_tokens["inbox/one.txt"] = "APR-ALREADY"
            loop.candidate_invoices = [loop.candidate_invoices[0]]
            result = asyncio.run(loop._execute("request_human_approval", {
                "invoice_number": "INV-DUP", "selection_criterion": "other",
                "selection_reason": "The test task selects this invoice.",
            }))

        self.assertEqual(result["status"], "already_approved")
        handler.assert_not_awaited()

    def test_approval_form_hint_names_invoice_number(self) -> None:
        with TemporaryDirectory(prefix="centralign-approval-hint-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "approval-hint", "task", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, approval_mode="always_gate", offline=True)
            loop.selected_invoice = {"invoice_number": "INV-HINT", "amount": "70000.00", "currency": "INR"}
            loop.selected_invoice_path = "inbox/hint.txt"
            loop.policy = {"thresholds": {"INR": "50000.00"}}
            snapshot = {"url_path": "/erp/invoices/new", "form_fields": [{"selector": "#amount_input", "state": "filled", "required": True}]}
            state = loop._form_state(snapshot)
        self.assertEqual(state["next_step"], "Approval required: call request_human_approval with invoice_number INV-HINT.")

    def test_login_credentials_are_absent_from_model_fill_schemas(self) -> None:
        for tool_name in ("browser_fill",):
            schema = TOOL_BY_NAME[tool_name]["parameters"]["properties"]["selector"]
            self.assertNotIn("#email_input", schema["enum"])
            self.assertNotIn("#password_input", schema["enum"])
        batch_schema = TOOL_BY_NAME["browser_fill_form"]["parameters"]["properties"]["fields"]["items"]["properties"]["selector"]
        self.assertNotIn("#email_input", batch_schema["enum"])
        self.assertNotIn("#password_input", batch_schema["enum"])
        self.assertNotIn("#approval_token", schema["enum"])
        self.assertNotIn("#approval_token", batch_schema["enum"])
        self.assertNotIn("#login_btn", TOOL_BY_NAME["browser_click"]["parameters"]["properties"]["selector"]["enum"])
        self.assertIn("erp_login", TOOL_BY_NAME)
        self.assertIn("host", TOOL_BY_NAME["erp_login"]["description"].casefold())
        with self.assertRaisesRegex(ValueError, "allowed values"):
            AgentLoop._validate_arguments(ToolCall("browser_fill", {"selector": "#password_input", "value": "secret"}))
        ignored = AgentLoop._redact_arguments("browser_fill", {"selector": "#password_input", "value": "model-typed-secret"})
        self.assertEqual(ignored["value"], "[ignored; use erp_login]")
        self.assertNotIn("model-typed-secret", json.dumps(ignored))

    def test_live_conversation_compresses_older_observations_to_one_line_summaries(self) -> None:
        class LiveClient:
            name = "groq"
            model = "test"

        with TemporaryDirectory(prefix="centralign-context-compact-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "compact", "task", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=LiveClient())
            loop.observations = [
                {"tool": "read_document", "arguments": {"path": f"inbox/{index}.txt"}, "result": {"invoice": {
                    "vendor": "Acme Analytics", "invoice_number": f"INV-{index}", "invoice_date": "2026-01-01",
                    "due_date": "2026-02-01", "amount": "100.00", "currency": "INR", "source_file": f"inbox/{index}.txt",
                    "text": "long invoice body " * 100,
                }}}
                for index in range(10)
            ]
            compact = loop._conversation_observations()
            self.assertEqual(len(compact), 2 + 3)
            self.assertEqual(compact[0]["tool"], "host_run_state")
            summaries = compact[1]["result"]["summaries"]
            self.assertEqual(len(summaries), 7)
            self.assertTrue(all("\\n" not in line and len(line) <= 240 for line in summaries))
            self.assertNotIn("long invoice body", json.dumps(summaries))
            self.assertEqual(compact[-1], loop.observations[-1])
            breakdown = loop._prompt_size_breakdown(compact)
            self.assertEqual(set(breakdown), {"system_prompt", "tool_schemas", "run_state", "history"})
            self.assertGreater(breakdown["history"]["characters"], breakdown["system_prompt"]["characters"])
            self.assertGreater(breakdown["history"]["characters"], breakdown["tool_schemas"]["characters"])

    def test_repeated_same_action_warns_at_three_and_stops_at_five(self) -> None:
        repeated_action = ToolCall("browser_snapshot", {})

        class RepeatingFakeLLM:
            name = "groq"
            model = "fake-loop-reproducer"

            def __init__(self):
                self.calls = []

            def next_tool_call(self, _task, observations, _tools):
                self.calls.append(json.loads(json.dumps(observations)))
                return repeated_action

        with TemporaryDirectory(prefix="centralign-loop-guard-") as temp_dir:
            events = []
            client = RepeatingFakeLLM()
            loop = AgentLoop(Path(temp_dir), "login-loop", "record latest Acme invoice", "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None, client=client)
            login_snapshot = {
                "url_path": "/erp/login", "page_title": "Employee sign in",
                "alerts": [], "alert": None, "form_fields": [], "buttons": [{"selector": "#login_btn", "label": "Sign in", "enabled": True}],
                "tables": {}, "session_required": "Session required: call erp_login",
            }
            loop._last_page_snapshot = login_snapshot
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            loop.browser.browser_snapshot = unittest.mock.AsyncMock(return_value={"status": "observed", "snapshot": login_snapshot})
            status, summary = asyncio.run(loop.run())
        self.assertEqual(status, "needs_review")
        self.assertEqual(len(client.calls), 5)
        self.assertIn("browser_snapshot({}) 5 times", summary)
        self.assertIn("Last observed alert: none", summary)
        warning_events = [event for event in events if event[0] == "loop_guard_warning"]
        self.assertEqual(len(warning_events), 1)
        self.assertIn("repeated 3 times", warning_events[0][1]["warning"])
        self.assertTrue(any(item.get("tool") == "loop_guard_warning" for item in client.calls[3]))

    def test_demo_data_reset_clears_ledger_runs_and_cli_flag(self) -> None:
        with TemporaryDirectory(prefix="centralign-reset-data-") as temp_dir:
            data_dir = Path(temp_dir)
            company = CompanyApp(data_dir)
            company.create_bill({
                "vendor": "Acme Analytics Pvt Ltd", "invoice_number": "RESET-1",
                "invoice_date": "2026-10-01", "due_date": "2026-11-01",
                "amount": "100.00", "currency": "INR", "source_file": "invoice.txt",
            })
            company.start_run("reset-run", "test run")
            company.append_event("reset-run", 1, "task_started", {"task": "test"})
            run_log_dir = data_dir / "runs"
            run_log_dir.mkdir(parents=True, exist_ok=True)
            (run_log_dir / "reset-run.jsonl").write_text("{}\n", encoding="utf-8")
            company.reset_demo_data()
            self.assertEqual(company.list_bills(), [])
            self.assertEqual(company.list_runs(), [])
            self.assertEqual(company.read_audit("reset-run"), [])

            company.create_bill({
                "vendor": "Acme Analytics Pvt Ltd", "invoice_number": "RESET-2",
                "invoice_date": "2026-10-01", "due_date": "2026-11-01",
                "amount": "100.00", "currency": "INR", "source_file": "invoice.txt",
            })
            company.start_run("reset-run-2", "test run")
            (run_log_dir / "reset-run-2.jsonl").write_text("{}\n", encoding="utf-8")
            with patch("sys.argv", ["web_app.py", "--data-dir", str(data_dir), "--reset-demo-data"]), patch("uvicorn.run") as serve:
                web_main()
            serve.assert_called_once()
            self.assertEqual(CompanyApp(data_dir).list_bills(), [])
            self.assertEqual(CompanyApp(data_dir).list_runs(), [])
            self.assertEqual(list(run_log_dir.glob("*.jsonl")), [])

    def test_provider_configuration_selects_openai_gemini_and_replay(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(planner_config(), ("unconfigured", "No live provider API key configured", None))
            self.assertEqual(planner_config(offline=True)[0], "offline-replay")
            with self.assertRaisesRegex(ProviderConfigurationError, "start with --offline"):
                make_client()
        with patch.dict(os.environ, {"CENTRALIGN_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}, clear=True):
            self.assertEqual(planner_config(), ("openai", "gpt-4o", "test-key"))
        with patch.dict(os.environ, {"CENTRALIGN_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=True):
            self.assertEqual(planner_config(), ("gemini", "gemini-2.5-flash-lite", "test-key"))

    def test_provider_configuration_uses_one_key_and_explicit_live_fallback(self) -> None:
        with patch.dict(os.environ, {
            "CENTRALIGN_PROVIDER": "groq",
            "GROQ_API_KEY": "groq-key",
            "CENTRALIGN_FALLBACK_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "router-key",
        }, clear=True):
            self.assertEqual(planner_config(), ("groq", "llama-3.3-70b-versatile", "groq-key"))
            client = make_client()
            self.assertIsInstance(client, ResilientClient)
            self.assertEqual([provider for provider, _ in client.providers], ["groq", "openrouter"])
            with patch.dict(os.environ, {"CENTRALIGN_PROVIDER": "groq", "GROQ_API_KEY": "groq-key"}, clear=True):
                only_primary = make_client()
            self.assertEqual([provider for provider, _ in only_primary.providers], ["groq"])

    def test_provider_router_waits_then_retries_same_provider_and_logs_answer(self) -> None:
        class StubClient:
            def __init__(self, name, model, responses):
                self.name, self.model, self.responses = name, model, list(responses)
                self.calls = 0

            def next_tool_call(self, *_args):
                self.calls += 1
                value = self.responses.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value

        events, answered, waits = [], [], []
        provider = StubClient("gemini", "gemini-2.5-flash-lite", [
            ProviderRequestError("gemini", "gemini-2.5-flash-lite", 429, 0.4),
            ProviderRequestError("gemini", "gemini-2.5-flash-lite", 503),
            ToolCall("list_files", {}),
        ])
        client = ResilientClient(
            [("gemini", provider)], on_recovery=events.append, on_provider_turn=answered.append,
            sleeper=waits.append, jitter=lambda _low, _high: 0.25,
        )
        self.assertEqual(client.next_tool_call("task", [], TOOL_DEFINITIONS), ToolCall("list_files", {}))
        self.assertEqual(client.name, "gemini")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(waits, [0.4, 2.25])
        self.assertEqual([event["wait_seconds"] for event in events if event["action"] == "retry_wait"], waits)
        self.assertEqual(answered, [{"provider": "gemini", "model": "gemini-2.5-flash-lite", "turn": 1}])
        self.assertEqual(client._retry_wait(ProviderRequestError("gemini", "flash", 429, 90), 1), 60.0)

    def test_live_429_exhaustion_never_uses_replay_and_allows_only_explicit_live_fallback(self) -> None:
        class StubClient:
            name = "groq"
            model = "llama-3.3-70b-versatile"

            def __init__(self, statuses):
                self.statuses = list(statuses)
                self.calls = 0

            def next_tool_call(self, *_args):
                self.calls += 1
                status = self.statuses.pop(0)
                if isinstance(status, int):
                    raise ProviderRequestError(self.name, self.model, status)
                return status

        sleeps, events = [], []
        primary = StubClient([429] * 5)
        live = ResilientClient([("groq", primary)], on_recovery=events.append, sleeper=sleeps.append, jitter=lambda _low, _high: 0)
        with self.assertRaisesRegex(LiveProvidersExhausted, "no live fallback provider is configured"):
            live.next_tool_call("record Acme invoice", [], TOOL_DEFINITIONS)
        self.assertEqual(primary.calls, 5)  # first attempt plus four retries
        self.assertEqual(sleeps, [1.0, 2.0, 4.0, 8.0])
        self.assertFalse(any(event.get("provider") == "offline-replay" for event in events))

        primary = StubClient([503] * 5)
        fallback = StubClient([ToolCall("list_files", {})])
        events.clear()
        live = ResilientClient(
            [("groq", primary), ("openrouter", fallback)], on_recovery=events.append,
            sleeper=lambda _seconds: None, jitter=lambda _low, _high: 0,
        )
        self.assertEqual(live.next_tool_call("task", [], TOOL_DEFINITIONS), ToolCall("list_files", {}))
        self.assertEqual(live.name, "openrouter")
        self.assertTrue(any(event["action"] == "provider_fallback" for event in events))

        primary = StubClient([429] * 5)
        fallback = StubClient([503] * 5)
        live = ResilientClient(
            [("groq", primary), ("openrouter", fallback)],
            sleeper=lambda _seconds: None, jitter=lambda _low, _high: 0,
        )
        with self.assertRaisesRegex(LiveProvidersExhausted, "Live providers failed"):
            live.next_tool_call("task", [], TOOL_DEFINITIONS)

    def test_openai_compatible_tool_payload_and_429_retry_after(self) -> None:
        tool_call = {"choices": [{"message": {"tool_calls": [{"function": {"name": "finish", "arguments": '{"status":"needs_review","summary":"done"}'}}]}}]}
        client = ChatCompletionsClient("groq", "https://api.groq.com/openai/v1/chat/completions", "secret", "llama-3.3-70b-versatile")
        with patch("llm_clients._request", return_value=tool_call) as request:
            result = client.next_tool_call("task", [], TOOL_DEFINITIONS)
        self.assertEqual(result, ToolCall("finish", {"status": "needs_review", "summary": "done"}))
        body = request.call_args.args[1]
        self.assertEqual(body["model"], "llama-3.3-70b-versatile")
        self.assertEqual(body["tools"][0]["function"]["name"], "list_files")
        self.assertFalse(body["parallel_tool_calls"])

        error = urllib.error.HTTPError("https://example.test", 429, "busy", {"Retry-After": "4"}, io.BytesIO(b"Please retry in 7s"))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderRequestError) as caught:
                _request("https://example.test", {}, {}, provider="gemini", model="flash")
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.retry_after_seconds, 4.0)
        error = urllib.error.HTTPError("https://example.test", 503, "busy", {}, io.BytesIO(b"Please retry in 7s"))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderRequestError) as caught:
                _request("https://example.test", {}, {}, provider="gemini", model="flash")
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.retry_after_seconds, 7.0)
        self.assertEqual(caught.exception.response_body, "Please retry in 7s")

        provider_body = json.dumps({"error": {
            "code": "tool_use_failed", "message": "verify_ledger_row omitted invoice_number"
        }})
        schema_request = {"tools": [{"type": "function", "function": {
            "name": "verify_ledger_row", "parameters": {"type": "object"}
        }}]}
        provider_error = urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions", 400, "Bad request", {},
            io.BytesIO(provider_body.encode("utf-8")),
        )
        with patch("urllib.request.urlopen", side_effect=provider_error):
            with self.assertRaises(ProviderRequestError) as caught:
                _request("https://api.groq.com/openai/v1/chat/completions", schema_request, {}, provider="groq", model="test")
        self.assertEqual(caught.exception.error_code, "tool_use_failed")
        self.assertEqual(caught.exception.rejected_tool_name, "verify_ledger_row")

    def test_provider_requests_set_user_agent_and_safely_include_error_excerpt(self) -> None:
        api_key = "groq-secret-api-key"
        error_payload = f"Access denied for {api_key}. " + ("diagnostic " * 80)
        requests = []

        def reject(request, timeout):
            requests.append(request)
            raise urllib.error.HTTPError(
                request.full_url, 403, "Forbidden", {}, io.BytesIO(error_payload.encode("utf-8"))
            )

        with patch("urllib.request.urlopen", side_effect=reject):
            with self.assertRaises(ProviderRequestError) as caught:
                _request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    {},
                    {"Authorization": f"Bearer {api_key}"},
                    provider="groq",
                    model="llama-3.3-70b-versatile",
                )

        self.assertEqual(requests[0].get_header("User-agent"), "CentrAlignOperator/0.1")
        excerpt = caught.exception.response_body
        self.assertIsNotNone(excerpt)
        self.assertLessEqual(len(excerpt), 300)
        self.assertNotIn(api_key, excerpt)
        self.assertNotIn(api_key, str(caught.exception))
        self.assertIn("Access denied for [REDACTED]", excerpt)

        gemini_key = "gemini-secret-api-key"
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "https://generativelanguage.googleapis.com/v1beta?key=gemini-secret-api-key",
                403,
                "Forbidden",
                {},
                io.BytesIO(f"Rejected {gemini_key}".encode("utf-8")),
            ),
        ):
            with self.assertRaises(ProviderRequestError) as gemini_error:
                _request(
                    "https://generativelanguage.googleapis.com/v1beta?key=gemini-secret-api-key",
                    {},
                    {},
                    provider="gemini",
                    model="gemini-2.5-flash-lite",
                )
        self.assertNotIn(gemini_key, gemini_error.exception.response_body)
        self.assertIn("Rejected [REDACTED]", gemini_error.exception.response_body)

    def test_batched_browser_tool_schemas_are_bounded_and_closed(self) -> None:
        AgentLoop._validate_arguments(ToolCall("browser_fill_form", {
            "fields": [{"selector": "#vendor_input", "value": "Acme Analytics"}]
        }))
        AgentLoop._validate_arguments(ToolCall("browser_action_sequence", {
            "actions": [
                {"tool": "browser_fill", "arguments": {"selector": "#vendor_input", "value": "Acme Analytics"}},
                {"tool": "browser_check", "arguments": {"selector": "#approval_verified"}},
            ]
        }))
        with self.assertRaisesRegex(ValueError, "Allowed fields: selector, value") as invalid_field:
            AgentLoop._validate_arguments(ToolCall("browser_fill_form", {
                "fields": [{"selector": "#vendor_input", "value": "x", "javascript": "alert(1)"}]
            }))
        for selector in MODEL_FORM_SELECTORS:
            self.assertIn(selector, str(invalid_field.exception))
        self.assertNotIn("#password_input", str(invalid_field.exception))
        with self.assertRaisesRegex(ValueError, "between 2 and 3"):
            AgentLoop._validate_arguments(ToolCall("browser_action_sequence", {
                "actions": [{"tool": "browser_open", "arguments": {"path": "/erp/login"}}]
            }))
        with self.assertRaisesRegex(ValueError, "allowed values"):
            click_batch = ToolCall("browser_action_sequence", {
                "actions": [
                    {"tool": "browser_fill", "arguments": {"selector": "#vendor_input", "value": "Acme"}},
                    {"tool": "browser_click", "arguments": {"selector": "#submit_btn"}},
                ]
            })
            AgentLoop._validate_arguments(click_batch)
        with TemporaryDirectory(prefix="centralign-batch-guard-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "batch-guard", "Record invoice for the test", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True)
            loop.task_intent = _test_write_intent("", [])
            with self.assertRaisesRegex(ValueError, "navigation and clicks are single-step"):
                asyncio.run(loop._execute("browser_action_sequence", click_batch.arguments))

    def test_browser_open_schema_matches_route_allowlist_and_rejections_list_options(self) -> None:
        path_schema = TOOL_BY_NAME["browser_open"]["parameters"]["properties"]["path"]
        self.assertEqual(path_schema["enum"], list(ERP_ROUTES))
        for path, description in ERP_ROUTES.items():
            self.assertIn(path, path_schema["description"])
            self.assertIn(description, path_schema["description"])

        with TemporaryDirectory(prefix="centralign-route-options-") as temp_dir:
            loop = AgentLoop(
                Path(temp_dir), "route-options", "test", "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, offline=True,
            )
            with self.assertRaises(ValueError) as rejected:
                asyncio.run(loop._execute("browser_open", {"path": "/erp/invoice_entry"}))
            message = str(rejected.exception)
            self.assertIn("Allowed routes:", message)
            for path in ERP_ROUTES:
                self.assertIn(path, message)

            with self.assertRaises(ValueError) as schema_rejected:
                AgentLoop._validate_arguments(ToolCall("browser_open", {"path": "/erp/invoice"}))
            for path in ERP_ROUTES:
                self.assertIn(path, str(schema_rejected.exception))

    def test_tool_retry_limit_summary_includes_last_tool_error(self) -> None:
        class BadRouteClient:
            name = "test-provider"
            model = "test-model"

            @staticmethod
            def next_tool_call(*_args):
                return ToolCall("browser_open", {"path": "/erp/invoice"})

        with TemporaryDirectory(prefix="centralign-last-tool-error-") as temp_dir:
            loop = AgentLoop(
                Path(temp_dir), "last-tool-error", "record latest Acme invoice", "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, client=BadRouteClient(),
            )

            async def skip_browser() -> None:
                return None

            loop.browser.start = skip_browser
            loop.browser.close = skip_browser
            status, summary = asyncio.run(loop.run())
        self.assertEqual(status, "needs_input")
        self.assertIn("4 failures for browser_open", summary)
        self.assertIn("Last error:", summary)
        self.assertIn("one of the allowed values:", summary)

    def test_host_rejects_unlisted_tools_routes_and_unsafe_writes(self) -> None:
        with TemporaryDirectory(prefix="centralign-guards-") as temp_dir:
            data_dir = Path(temp_dir)
            loop = AgentLoop(
                data_dir, "guard-test", "record the Acme invoice", "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, offline=True,
            )
            loop.task_intent = _test_write_intent("", [])
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
                with self.assertRaisesRegex(ValueError, "Allowed routes:") as rejected_route:
                    await loop._execute("browser_open", {"path": "/outside"})
                for path in ERP_ROUTES:
                    self.assertIn(path, str(rejected_route.exception))
                with self.assertRaisesRegex(ValueError, "Allowed fields:") as rejected_field:
                    await loop.browser.browser_fill("#arbitrary-selector", "value")
                for selector in MODEL_FORM_SELECTORS:
                    self.assertIn(selector, str(rejected_field.exception))
                self.assertNotIn("#password_input", str(rejected_field.exception))
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
                with self.assertRaisesRegex(ValueError, "Approval required: call request_human_approval"):
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

    def test_verify_before_select_is_rejected_and_loop_continues(self) -> None:
        class VerifyBeforeSelectClient:
            name = "offline-replay"
            model = "test-verify-before-select"

            def __init__(self):
                self.turn = 0
                self.verify_rejected = False
                self.verify_succeeded = False

            def next_tool_call(self, _task, observations, _tools):
                self.turn += 1
                if self.turn == 1:
                    return ToolCall("list_invoice_candidates", {"vendor_query": "Acme"})
                if self.turn == 2:
                    return ToolCall("verify_ledger_row", {"invoice_number": "INV-2026-2116"})
                if self.turn == 3:
                    last_obs = observations[-1]
                    if last_obs.get("tool") == "verify_ledger_row" and "error" in last_obs.get("result", {}).get("status", ""):
                        self.verify_rejected = True
                    return ToolCall("read_document", {"path": "inbox/INV_2026_10_Acme.pdf"})
                if self.turn == 4:
                    return ToolCall("verify_ledger_row", {"invoice_number": "INV-2026-2116"})
                if self.turn == 5:
                    last_obs = observations[-1]
                    if last_obs.get("tool") == "verify_ledger_row" and last_obs.get("result", {}).get("status") == "verified":
                        self.verify_succeeded = True
                    return ToolCall("finish", {"status": "complete", "summary": "Done after selection."})
                return None

        class StubBrowser:
            async def start(self):
                return None
            async def close(self):
                return None
            def set_invoice(self, _invoice):
                return None
            async def verify_ledger_row(self, _invoice):
                return {
                    "status": "verified", "pass": True, "invoice_number": "INV-2026-2116",
                    "already_present": True, "ledger_row_id": "#1", "matches": {"vendor": True},
                }

        with TemporaryDirectory(prefix="centralign-verify-before-select-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            client = VerifyBeforeSelectClient()
            events = []
            loop = AgentLoop(
                data_dir, "verify-before-select", "Record latest Acme invoice in ledger",
                "http://127.0.0.1:8765", lambda *event: events.append(event), lambda *_: None,
                client=client,
            )
            loop.browser = StubBrowser()
            status, summary = asyncio.run(loop.run())

            self.assertTrue(client.verify_rejected)
            self.assertTrue(client.verify_succeeded)
            self.assertEqual(status, "complete")
            rejected_obs = next(
                obs for obs in loop.observations
                if obs.get("tool") == "verify_ledger_row" and obs.get("result", {}).get("status") == "error"
            )
            self.assertIn("Select and read an invoice candidate from list_invoice_candidates before verifying ledger rows", rejected_obs["result"]["message"])
            selection_event = next(payload for kind, payload in events if kind == "selection_check")
            self.assertEqual(selection_event["status"], "passed")

    def test_normal_order_selection_then_verification_succeeds(self) -> None:
        class StubBrowser:
            async def start(self):
                return None
            async def close(self):
                return None
            def set_invoice(self, _invoice):
                return None
            async def verify_ledger_row(self, _invoice):
                return {"status": "verified", "pass": True, "invoice_number": "INV-2026-2116"}

        with TemporaryDirectory(prefix="centralign-normal-order-") as temp_dir:
            data_dir = Path(temp_dir)
            ensure_operator_assets(data_dir)
            loop = AgentLoop(
                data_dir, "normal-order", "Record latest Acme invoice in ledger",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None,
                offline=True,
            )
            loop.browser = StubBrowser()
            loop.task_intent = _test_write_intent("", [])
            loop.task_intent["selection_criterion"] = "latest_by_invoice_date"
            loop.task_intent["reason"] = "The task requests the latest invoice by invoice date."
            candidates = asyncio.run(loop._execute("list_invoice_candidates", {"vendor_query": "Acme"}))
            self.assertGreaterEqual(len(candidates["candidates"]), 1)
            # Before reading, verify is rejected
            with self.assertRaisesRegex(ValueError, "Select and read an invoice candidate"):
                asyncio.run(loop._execute("verify_ledger_row", {}))
            # After read_document on the latest candidate, selection is established
            read_res = asyncio.run(loop._execute("read_document", {"path": "inbox/INV_2026_10_Acme.pdf"}))
            self.assertEqual(read_res["invoice"]["invoice_number"], "INV-2026-2116")
            self.assertTrue(loop._has_passing_selection())
            self.assertEqual(loop.selection_criterion, "latest_by_invoice_date")
            # Now verify_ledger_row succeeds without error
            verify_res = asyncio.run(loop._execute("verify_ledger_row", {}))
            self.assertIn("status", verify_res)

    def test_read_only_task_unaffected_by_selection_requirement(self) -> None:
        with TemporaryDirectory(prefix="centralign-read-only-") as temp_dir:
            loop = AgentLoop(
                Path(temp_dir), "read-only-test", "What invoices are posted for Acme?",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None,
                offline=True,
            )
            loop.task_intent = {
                "intent": "read_only", "vendor_query": "Acme",
                "requested_actions": [{"action": "read ledger", "capability": "read_ledger"}],
            }
            result = asyncio.run(loop._execute("read_ledger", {"vendor_query": "Acme"}))
            self.assertEqual(result["status"], "read")
            self.assertIn("bills", result)

    def test_duplicate_invoice_final_result_summary_and_evidence(self) -> None:
        with TemporaryDirectory(prefix="centralign-duplicate-unit-") as temp_dir:
            data_dir = Path(temp_dir)
            loop = AgentLoop(
                data_dir, "test-dup", "Record Acme invoice",
                "http://127.0.0.1:8765", lambda *_: None, lambda *_: None,
                offline=True,
            )
            loop.task_intent = {
                "intent": "write_invoice_entry",
                "vendor_query": "Acme",
                "requested_actions": [{"action": "enter invoice", "capability": "write_invoice_entry"}],
            }
            loop.selected_invoice = {
                "invoice_number": "INV-2026-2116",
                "vendor": "Acme Analytics Pvt Ltd",
                "canonical_vendor": "Acme Analytics Pvt Ltd",
                "amount": "68500.00",
                "currency": "INR",
                "invoice_date": "2026-10-21",
                "due_date": "2026-11-15",
            }
            loop.selected_invoice_path = "inbox/INV_2026_10_Acme.pdf"
            loop.candidate_invoices = [{"path": "inbox/INV_2026_10_Acme.pdf", "invoice_number": "INV-2026-2116", "invoice_date": "2026-10-21"}]
            loop.selection_criterion = "latest_by_invoice_date"
            loop.selection_reason = "Chose latest Acme invoice"

            class StubBrowser:
                def set_invoice(self, _invoice):
                    pass

                async def verify_ledger_row(self, invoice: dict) -> dict:
                    return {
                        "status": "verified",
                        "pass": True,
                        "already_present": True,
                        "ledger_row_id": "#1",
                        "screenshot_path": "/tmp/ledger.png",
                        "matches": {"vendor": True, "invoice_number": True, "amount": True},
                    }

            loop.browser = StubBrowser()
            verify_res = asyncio.run(loop._execute("verify_ledger_row", {}))
            self.assertEqual(verify_res["status"], "verified")
            self.assertTrue(verify_res["already_present"])
            self.assertTrue(loop.final_result["already_present"])
            self.assertIn("The invoice was already in the ledger (row #1)", loop.final_result["what_was_done"])
            self.assertIn("no new entry was written", loop.final_result["what_was_done"])

            finish_res = asyncio.run(loop._execute("finish", {"status": "complete", "summary": "Finished the task"}))
            self.assertEqual(finish_res["status"], "finished")
            self.assertEqual(finish_res["final_status"], "complete")
            self.assertTrue(finish_res["already_present"])
            self.assertIn("The invoice was already in the ledger (row #1)", finish_res["summary"])
            self.assertIn("no new entry was written", finish_res["summary"])
            self.assertIn("Not done: none.", finish_res["summary"])


class BrowserAgentIntegrationTests(unittest.TestCase):
    def test_erp_thresholds_are_loaded_from_the_single_policy_file(self) -> None:
        with TemporaryDirectory(prefix="centralign-policy-source-") as temp_dir:
            data_dir = Path(temp_dir)
            app = create_app(data_dir, offline=True)
            policy_path = data_dir / "policies" / "procurement_policy.txt"
            policy_path.write_text(
                "CentrAlign AP Policy\nManager approval threshold: greater than INR 100 or USD 10.\n"
                "Approved vendors: Acme Analytics Pvt Ltd.\n", encoding="utf-8"
            )
            server = LocalServer(app)
            base_url = server.start()
            browser = BrowserTools(data_dir, "policy-source", base_url, lambda *_: None)

            async def exercise() -> dict:
                await browser.start()
                try:
                    signed_in = await browser.erp_login()
                    self.assertEqual(signed_in["status"], "success")
                    await browser.page.goto(base_url + "/erp/invoices/new", wait_until="domcontentloaded")
                    values = {
                        "#vendor_input": "Acme Analytics Pvt Ltd",
                        "#invoice_number_input": "POLICY-SOURCE-1",
                        "#invoice_date_input": "2026-10-01",
                        "#due_date_input": "2026-11-01",
                        "#amount_input": "500.00",
                        "#currency_input": "INR",
                        "#source_file_input": "policy-source.txt",
                        "#source_sha256_input": "a" * 64,
                    }
                    for selector, value in values.items():
                        await browser.page.locator(selector).fill(value)
                    await browser.page.locator("#submit_btn").click()
                    snapshot = await browser.browser_snapshot()
                    return {"url": browser.page.url, "body": await browser.page.locator("body").inner_text(), "snapshot": snapshot}
                finally:
                    await browser.close()

            try:
                page = asyncio.run(exercise())
                self.assertIn("Manager approval token must be verified", json.dumps(page))
                self.assertEqual(CompanyApp(data_dir).list_bills(), [])
            finally:
                server.stop()

    def test_host_login_snapshots_and_unsaved_navigation_warning(self) -> None:
        with TemporaryDirectory(prefix="centralign-login-snapshot-") as temp_dir:
            data_dir = Path(temp_dir)
            app = create_app(data_dir, offline=True)
            server = LocalServer(app)
            base_url = server.start()
            browser = BrowserTools(data_dir, "login-snapshot", base_url, lambda *_: None)

            async def exercise() -> tuple[dict, dict, dict]:
                await browser.start()
                try:
                    await browser.page.goto(base_url + "/erp/vendors", wait_until="domcontentloaded")
                    required = await browser.browser_snapshot()
                    login_snapshot = required["snapshot"]
                    self.assertEqual(login_snapshot["url_path"], "/erp/login")
                    self.assertTrue(login_snapshot["page_title"])
                    self.assertEqual(login_snapshot["session_required"], "Session required: call erp_login")
                    field_states = {field["selector"]: field for field in login_snapshot["form_fields"]}
                    self.assertEqual(field_states["#email_input"]["state"], "empty")
                    self.assertEqual(field_states["#password_input"]["state"], "empty")
                    self.assertNotIn("value", field_states["#password_input"])
                    self.assertTrue(any(button["selector"] == "#login_btn" for button in login_snapshot["buttons"]))
                    self.assertEqual(login_snapshot["alerts"], [])

                    original_password = browser.password
                    browser.password = "incorrect-password"
                    failure = await browser.erp_login()
                    browser.password = original_password
                    self.assertEqual(failure["status"], "failure")
                    self.assertEqual(failure["landing_page"]["url_path"], "/erp/login")
                    self.assertIn("Invalid sandbox credentials", failure["alert"])
                    self.assertNotIn(original_password, json.dumps(failure))

                    success = await browser.erp_login()
                    self.assertEqual(success["status"], "success")
                    self.assertEqual(success["landing_page"]["url_path"], "/erp/vendors")
                    self.assertIsNone(success["alert"])
                    self.assertNotIn(original_password, json.dumps(success))
                    self.assertIn("vendor_directory", success["snapshot"]["tables"])

                    browser.set_invoice({
                        "vendor": "Acme Analytics", "invoice_number": "INV-UNSAVED",
                        "invoice_date": "2026-10-21", "due_date": "2026-11-15",
                        "amount": "100.00", "currency": "INR", "source_file": "invoice.txt",
                        "source_sha256": "0" * 64,
                    })
                    await browser.page.goto(base_url + "/erp/invoices/new", wait_until="domcontentloaded")
                    await browser.browser_fill("#invoice_number_input", "INV-UNSAVED")
                    populated = (await browser.browser_snapshot())["snapshot"]
                    populated_field = next(field for field in populated["form_fields"] if field["selector"] == "#invoice_number_input")
                    self.assertEqual(populated_field["state"], "filled")
                    self.assertNotIn(original_password, json.dumps(populated))
                    away = await browser.browser_open("/erp/vendors")
                    self.assertIn("#invoice_number_input", away["warning"])
                    return login_snapshot, failure, success
                finally:
                    await browser.close()

            try:
                login_snapshot, failure, success = asyncio.run(exercise())
                self.assertIn("Session required", login_snapshot["session_required"])
                self.assertEqual(failure["status"], "failure")
                self.assertEqual(success["status"], "success")
            finally:
                server.stop()

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
                self.assertIn(b'id="resetDemoButton"', http_request(base_url, "/")[1])
                for path in ("/", "/app.js", "/styles.css"):
                    self.assertEqual(http_request(base_url, path)[0], 200)

                flagship_task = (
                    "Find the latest invoice from Acme, check policy compliance, enter it into the ERP portal, "
                    "and verify it in the ledger"
                )
                status, started = json_request(base_url, "/api/runs", {"task": flagship_task})
                self.assertEqual(status, 202)
                first_id = started["run_id"]

                def approve_after_inspection(run: dict) -> None:
                    actions = [event["payload"] for event in run["events"] if event["event_type"] == "action"]
                    self.assertFalse(any(action.get("path") == "/erp/invoices/new" for action in actions))
                    request_event = next(event for event in run["events"] if event["event_type"] == "approval_requested")
                    self.assertEqual(request_event["payload"]["invoice"]["invoice_number"], "INV-2026-2116")
                    self.assertEqual(request_event["payload"]["invoice"]["amount"], "68500.00")
                    self.assertEqual(request_event["payload"]["invoice"]["currency"], "INR")
                    self.assertIn("INR 50000.00 policy threshold", request_event["payload"]["reason"])
                    decision_status, decision = json_request(base_url, f"/api/runs/{first_id}/approval", {"approved": True})
                    self.assertEqual(decision_status, 200)
                    self.assertTrue(decision["approval_token"].startswith("APR-"))

                first_run = wait_for_run(base_url, first_id, approve_after_inspection)
                self.assertEqual(first_run["status"], "complete", first_run.get("summary"))
                intent_event = next(event["payload"] for event in first_run["events"] if event["event_type"] == "task_intent")
                self.assertEqual(intent_event["intent"], "write_invoice_entry")
                self.assertIn({"action": "check policy compliance", "capability": "enter_invoice"}, intent_event["requested_actions"])
                self.assertFalse(any(action["capability"] == "read_ledger" for action in intent_event["requested_actions"]))
                self.assertTrue(any(event["event_type"] == "verification" and event["payload"].get("status") == "verified" for event in first_run["events"]))
                self.assertTrue(any(event["event_type"] == "observation" and event["payload"].get("tool") == "read_policy" for event in first_run["events"]))
                self.assertIn("replay", next(event["payload"]["mode"] for event in first_run["events"] if event["event_type"] == "planner"))
                self.assertTrue(any(event["event_type"] == "self_correction" for event in first_run["events"]))
                actions = [event["payload"] for event in first_run["events"] if event["event_type"] == "action"]
                self.assertTrue(any(action.get("tool") == "erp_login" for action in actions))
                self.assertFalse(any(action.get("selector") in {"#email_input", "#password_input"} for action in actions))
                self.assertTrue(any(event["event_type"] == "credentials_supplied" for event in first_run["events"]))
                provenance = next(event["payload"] for event in first_run["events"] if event["event_type"] == "provenance_attached")
                self.assertIn("Host attached source_file and source SHA-256", provenance["message"])
                self.assertEqual(len(provenance["source_sha256"]), 64)
                read_observation = next(event["payload"] for event in first_run["events"] if event["event_type"] == "observation" and event["payload"].get("tool") == "read_document")
                self.assertNotIn("source_sha256", json.dumps(read_observation))
                self.assertTrue(any(action.get("tool") == "browser_fill_form" for action in actions))
                self.assertFalse(any(selector in {"#source_file_input", "#source_sha256_input"} for action in actions if action.get("tool") == "browser_fill_form" for selector in [field.get("selector") for field in action.get("fields", [])]))
                self.assertFalse(any(action.get("selector") == "#approval_token" for action in actions))
                token_injection = next(event["payload"] for event in first_run["events"] if event["event_type"] == "approval_token_injected")
                self.assertIn("Host injected the approved token", token_injection["message"])
                self.assertNotIn("APR-", json.dumps(token_injection))
                run_states = [event["payload"] for event in first_run["events"] if event["event_type"] == "host_run_state"]
                self.assertTrue(run_states)
                self.assertTrue(all({"logged_in", "selected_invoice", "legal_vendor", "approval_status", "form_status", "last_erp_alert"} <= state.keys() for state in run_states))
                self.assertTrue(any(state.get("workflow_hint") == "submitted: INV-2026-2116; next: verify_ledger_row" for state in run_states))
                prompt_summary = next(event["payload"] for event in first_run["events"] if event["event_type"] == "prompt_size_summary")
                self.assertGreater(prompt_summary["average_tokens_with_run_state"], prompt_summary["average_tokens_without_run_state"])
                self.assertEqual(set(prompt_summary["average_breakdown_tokens"]), {"system_prompt", "tool_schemas", "run_state", "history"})
                prompt_turn = next(event["payload"] for event in first_run["events"] if event["event_type"] == "prompt_size")
                self.assertEqual(set(prompt_turn["breakdown"]), {"system_prompt", "tool_schemas", "run_state", "history"})
                verification = next(event["payload"] for event in first_run["events"] if event["event_type"] == "verification")
                self.assertEqual(verification["status"], "verified")
                self.assertTrue(all(verification["matches"].values()))
                self.assertTrue(verification["pass"])
                self.assertEqual(verification["ledger_row_id"], "#1")
                self.assertTrue(all(row["match"] for row in verification["field_matches"]))
                self.assertTrue(Path(verification["screenshot_path"]).exists())
                final_result = next(event["payload"] for event in first_run["events"] if event["event_type"] == "final_result")
                self.assertEqual(final_result["status"], "complete")
                self.assertIn("Not done: none.", final_result["summary"])
                self.assertEqual(final_result["not_done_actions"], [])
                self.assertEqual(final_result["evidence"]["ledger_row_id"], verification["ledger_row_id"])
                self.assertEqual(final_result["evidence"]["approval_record"]["status"], "approved")
                self.assertEqual(final_result["evidence"]["source_file"], verification["source_file"])
                self.assertEqual(final_result["evidence"]["source_sha256"], verification["source_sha256"])
                self.assertEqual(final_result["selected_invoice"]["invoice_number"], "INV-2026-2116")
                self.assertEqual(final_result["selected_invoice"]["invoice_date"], "2026-10-21")
                self.assertEqual(final_result["selection_criterion"], "latest_by_invoice_date")
                self.assertTrue(any(
                    candidate["path"] == "inbox/INV_2026_10_Acme.pdf" and candidate["invoice_date"] == "2026-10-21"
                    for candidate in final_result["candidates_considered"]
                ))
                self.assertEqual(verification["selection_check"]["status"], "passed")
                self.assertGreaterEqual(final_result["elapsed_seconds"], 0)
                verify_observation = next(event["payload"] for event in first_run["events"] if event["event_type"] == "observation" and event["payload"].get("tool") == "verify_ledger_row")
                self.assertNotIn("source_sha256", json.dumps(verify_observation))
                self.assertEqual(verification["source_sha256"], provenance["source_sha256"])
                self.assertEqual(verification["actual"]["vendor"], "Acme Analytics Pvt Ltd")
                screenshot_status, screenshot, headers = http_request(base_url, verification["screenshot_url"])
                self.assertEqual(screenshot_status, 200)
                self.assertEqual(headers.get_content_type(), "image/png")
                self.assertTrue(screenshot.startswith(b"\x89PNG\r\n\x1a\n"))

                jsonl_path = Path(temp_dir) / "runs" / f"{first_id}.jsonl"
                self.assertTrue(jsonl_path.exists())
                jsonl_records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([record["event_type"] for record in jsonl_records], [event["event_type"] for event in first_run["events"]])
                self.assertNotIn("APR-", jsonl_path.read_text(encoding="utf-8"))

                status, history = json_request(base_url, "/api/runs")
                self.assertEqual(status, 200)
                self.assertIn(first_id, {run["id"] for run in history["runs"]})

                # A repeat verifies the existing ERP row and does not post a second bill.
                _, duplicate_start = json_request(base_url, "/api/runs", {"task": flagship_task})
                duplicate = wait_for_run(base_url, duplicate_start["run_id"])
                self.assertEqual(duplicate["status"], "complete", duplicate.get("summary"))
                duplicate_evidence = next(event["payload"] for event in duplicate["events"] if event["event_type"] == "verification")
                self.assertTrue(duplicate_evidence["already_present"])
                duplicate_result = next(event["payload"] for event in duplicate["events"] if event["event_type"] == "final_result")
                self.assertIn("The invoice was already in the ledger", duplicate_result["summary"])
                self.assertIn("no new entry was written", duplicate_result["summary"])
                self.assertIn("row #1", duplicate_result["summary"])
                self.assertTrue(duplicate_result.get("already_present"))
                self.assertIn("Not done: none.", duplicate_result["summary"])
                self.assertEqual(duplicate_result["not_done_actions"], [])
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

                reset_status, reset_result = json_request(base_url, "/api/demo/reset", {})
                self.assertEqual(reset_status, 200)
                self.assertEqual(reset_result["status"], "reset")
                reset_state = json_request(base_url, "/api/state")[1]
                self.assertEqual(reset_state["bills"], [])
                self.assertEqual(reset_state["runs"], [])
                self.assertTrue(reset_state["files"])
                self.assertEqual(list((Path(temp_dir) / "runs").glob("*.jsonl")), [])
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
