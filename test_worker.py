"""Phase 1 tests for the shared tool-calling browser agent."""

from __future__ import annotations

import asyncio
import io
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

from agent_loop import AgentLoop, TOOL_BY_NAME, TOOL_DEFINITIONS
from browser_tools import ERP_ROUTES, FORM_SELECTORS, SNAPSHOT_CHARACTER_CAP, BrowserTools
from company_app import CompanyApp
from llm_clients import (
    ChatCompletionsClient,
    FakeLLM,
    GeminiClient,
    LiveProvidersExhausted,
    OpenAIClient,
    ProviderConfigurationError,
    ProviderRequestError,
    ResilientClient,
    ToolCall,
    _request,
    make_client,
)
from worker import planner_config
from web_app import RunManager, create_app, main as web_main


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
                Path(temp_dir), "validation-retry", "verify invoice", "http://127.0.0.1:8765",
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
        self.assertIn("invoice_number", schema["parameters"]["properties"]["expected_fields"]["required"])
        self.assertTrue(any(kind == "credentials_supplied" for kind, _ in events))

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
            loop = AgentLoop(Path(temp_dir), "malformed", "verify invoice", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=client)
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
            loop = AgentLoop(Path(temp_dir), "validation-cap", "verify invoice", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, client=client)
            loop.browser.start = unittest.mock.AsyncMock()
            loop.browser.close = unittest.mock.AsyncMock()
            status, summary = asyncio.run(loop.run())
            self.assertEqual(status, "needs_review")
            self.assertIn("retry limit reached after 3 retries", summary)
            self.assertIn("still missing invoice_number", summary)
            self.assertEqual(len(client.calls), 4)  # initial attempt and three retries
            self.assertEqual(sum(item.get("tool") == "tool_validation_feedback" for item in client.calls[-1]), 3)

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
            "tables": {"ledger_table": [["invoice", "x" * 1000] for _ in range(30)]},
        }
        compact = BrowserTools._bound_snapshot(snapshot)
        self.assertLessEqual(len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))), SNAPSHOT_CHARACTER_CAP)
        self.assertEqual(compact["url"], "/erp/ledger")
        self.assertEqual(compact["alert"], "Check invoice field")
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
                return {"url": "/erp/invoices/new", "inputs": {}, "tables": {}, "alert": None}

        with TemporaryDirectory(prefix="centralign-fill-result-") as temp_dir:
            loop = AgentLoop(Path(temp_dir), "fill-result", "fill invoice", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True)
            browser = StubBrowser()
            loop.browser = browser
            loop.selected_invoice = {"invoice_number": "INV-1"}
            result = asyncio.run(loop._execute("browser_fill_form", {"fields": [
                {"selector": "#invoice_number_input", "value": "INV-1"},
                {"selector": "#amount_input", "value": "100.00"},
            ]}))
            self.assertEqual(browser.snapshots, 1)
            self.assertEqual(result["fields_set"], ["#invoice_number_input", "#amount_input"])
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
            self.assertTrue(all(field["value"] == "[credentials supplied by host]" for field in safe["fields"]))

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
            with patch("sys.argv", ["web_app.py", "--data-dir", str(data_dir), "--reset-demo-data"]), patch("uvicorn.run") as serve:
                web_main()
            serve.assert_called_once()
            self.assertEqual(CompanyApp(data_dir).list_bills(), [])
            self.assertEqual(CompanyApp(data_dir).list_runs(), [])

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
            "fields": [{"selector": "#email_input", "value": "[sandbox_email]"}]
        }))
        AgentLoop._validate_arguments(ToolCall("browser_action_sequence", {
            "actions": [
                {"tool": "browser_fill", "arguments": {"selector": "#email_input", "value": "[sandbox_email]"}},
                {"tool": "browser_check", "arguments": {"selector": "#approval_verified"}},
            ]
        }))
        with self.assertRaisesRegex(ValueError, "Allowed fields: selector, value") as invalid_field:
            AgentLoop._validate_arguments(ToolCall("browser_fill_form", {
                "fields": [{"selector": "#email_input", "value": "x", "javascript": "alert(1)"}]
            }))
        for selector in FORM_SELECTORS:
            self.assertIn(selector, str(invalid_field.exception))
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
            loop = AgentLoop(Path(temp_dir), "batch-guard", "test", "http://127.0.0.1:8765", lambda *_: None, lambda *_: None, offline=True)
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
                Path(temp_dir), "last-tool-error", "test", "http://127.0.0.1:8765",
                lambda *_: None, lambda *_: None, client=BadRouteClient(),
            )

            async def skip_browser() -> None:
                return None

            loop.browser.start = skip_browser
            loop.browser.close = skip_browser
            status, summary = asyncio.run(loop.run())
        self.assertEqual(status, "needs_review")
        self.assertIn("Last tool error (browser_open)", summary)
        self.assertIn("one of the allowed values:", summary)

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
                with self.assertRaisesRegex(ValueError, "Allowed routes:") as rejected_route:
                    await loop._execute("browser_open", {"path": "/outside"})
                for path in ERP_ROUTES:
                    self.assertIn(path, str(rejected_route.exception))
                with self.assertRaisesRegex(ValueError, "Allowed fields:") as rejected_field:
                    await loop.browser.browser_fill("#arbitrary-selector", "value")
                for selector in FORM_SELECTORS:
                    self.assertIn(selector, str(rejected_field.exception))
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
                self.assertIn(b'id="resetDemoButton"', http_request(base_url, "/")[1])
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
                actions = [event["payload"] for event in first_run["events"] if event["event_type"] == "action"]
                self.assertTrue(any(action.get("tool") == "browser_fill_form" for action in actions))
                fill_sequence = next(action for action in actions if action.get("tool") == "browser_action_sequence")
                self.assertTrue(all(item.get("tool") in {"browser_fill", "browser_check", "browser_fill_form"} for item in fill_sequence["actions"]))
                self.assertFalse(any(item.get("tool") == "browser_click" for item in fill_sequence["actions"]))
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

                reset_status, reset_result = json_request(base_url, "/api/demo/reset", {})
                self.assertEqual(reset_status, 200)
                self.assertEqual(reset_result["status"], "reset")
                reset_state = json_request(base_url, "/api/state")[1]
                self.assertEqual(reset_state["bills"], [])
                self.assertEqual(reset_state["runs"], [])
                self.assertTrue(reset_state["files"])
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
