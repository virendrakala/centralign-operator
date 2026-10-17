# CentrAlign Operator

CentrAlign Operator is a local invoice-processing prototype. One bounded agent loop reads invoice sources and procurement policy, operates a sandbox ERP through Playwright, pauses for human approval when policy requires it, and verifies the resulting ledger row from the ERP page.

The ERP, invoices, credentials, approval broker, and SQLite ledger are synthetic demo data. The agent writes through the ERP form; it does not write directly to SQLite. The default setup uses an explicitly labelled FakeLLM replay and does not need an LLM API key.

## Run locally

Requirements: Python 3.10+, the packages in `requirements.txt`, and Playwright Chromium.

```bash
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
python3 web_app.py --offline
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The Operator and the sandbox ERP run in the same FastAPI process. To use a fresh data directory:

```bash
python3 web_app.py --data-dir /tmp/centralign-operator --offline
```

The first startup copies the bundled text fixtures and policy into a new data directory and creates the sample Acme PDFs if they are missing. The local ERP login details are in `data/policies/company_memory.md`; they are synthetic sandbox credentials.

## Agent loop

The provider returns one tool call at a time. Python validates the tool name and arguments, checks policy and run state, executes the call, and gives the observation back to the provider.

```text
Task
  -> AgentLoop asks one provider for one tool call
  -> host validates schema, route, selector, source, duplicate, and approval rules
  -> document tool or Playwright ERP action
  -> observation returned to the same loop
  -> finish only after live ledger read-back verification
```

The tool set is `list_files`, `read_document`, `read_policy`, `browser_open`, `browser_snapshot`, `browser_fill`, `browser_check`, `browser_click`, `request_human_approval`, `verify_ledger_row`, and `finish`.

The Acme demo follows the actual ERP validation path. The source says `Acme Analytics`, while the ERP requires the legal name displayed in its vendor directory. The agent first enters the source label. After the ERP displays its unknown-vendor alert, the agent inspects the directory, retries with the observed legal entity, and reads the posted row back from the ledger.

## Providers

Without API keys, the app selects `offline-replay` and logs that the planner is `FakeLLM scripted replay`. This mode uses the same host loop and browser tools as live providers; it is deterministic and does not represent a live model decision.

OpenAI Responses API:

```bash
export CENTRALIGN_PROVIDER=openai
export OPENAI_API_KEY="..."
export CENTRALIGN_MODEL="gpt-4o" # optional; this is the default
python3 web_app.py
```

Gemini API:

```bash
export CENTRALIGN_PROVIDER=gemini
export GEMINI_API_KEY="..."
export CENTRALIGN_MODEL="gemini-2.5-flash" # optional; this is the default
python3 web_app.py
```

When a live provider is enabled, the task and observations—including extracted source text—are sent to that provider to choose the next tool call. Use only synthetic or otherwise approved data. ERP credentials are injected by the local host and are not supplied to the model.

## Safety and verification

- Browser navigation is limited to the four local ERP routes. Form selectors are allowlisted, and source-backed values must match the selected document.
- The host checks invoice source and policy before a write, checks for a duplicate in the ERP ledger, blocks unapproved vendors, and enforces the approval gate.
- Approval tokens expire after five minutes, are bound to the invoice number, legal vendor, amount, and currency, and can be consumed once.
- Invoice and policy contents are treated as data. The model is told not to follow instructions embedded in source documents.
- A run reports `complete` only after the ERP ledger DOM matches the source vendor/legal entity, invoice number, dates, amount, and currency. A row screenshot and source SHA-256 are retained as evidence under `data/evidence/<run-id>/`.
- SQLite in `company_app.py` persists ERP bills, run history, and audit events. The agent's posting path is the Playwright-operated ERP form.

## Code map

| File | Responsibility |
|---|---|
| `agent_loop.py` | Shared bounded observe/decide/execute loop, tool schemas, host checks, and stop conditions. |
| `llm_clients.py` | Provider interface plus OpenAI, Gemini, and offline FakeLLM implementations. |
| `documents.py` | Confined invoice listing, PDF/TXT extraction, field parsing, and source hashes. |
| `policy.py` | Procurement policy parsing and threshold checks. |
| `browser_tools.py` | Playwright session, route/control allowlists, screenshots, and live ERP read-back. |
| `company_app.py` | SQLite persistence behind the ERP plus policy and vendor-directory data access. |
| `web_app.py` | FastAPI Operator APIs, approval token broker, and sandbox ERP pages. |
| `worker.py` | Small public entry point and CLI delegation. |
| `web/` | Operator workbench HTML, CSS, and JavaScript. |
| `data/inbox/`, `data/policies/` | Synthetic invoice sources, procurement rules, and sandbox company memory. |

## Checks

The browser integration test starts an isolated local server and uses FakeLLM with Playwright. It covers approval before invoice-form access, ERP-originated vendor recovery, live read-back, duplicate handling, denied approval, token binding, and tool/route guards.

```bash
python3 -m compileall -q .
python3 -m unittest -v test_worker.py
node --check web/app.js
```

## Current scope and known limitations

This submission currently handles invoice-entry tasks through one shared tool loop. Task-intent classification and read-only invoice queries are not implemented yet, so use the demo only for explicit invoice-entry requests. The ERP form still has built-in INR/USD threshold checks, and both `.txt` and `.md` procurement policy files remain; consolidating those checks and files is a Phase 2 item. Unsupported-task handling, fault injection and recovery scenarios, a broader eval suite, and interview/demo documents are planned for later phases. The approval waiter is in-memory and is intended for a single-user local demo. Live LLM availability and behavior depend on the selected provider and network access.
