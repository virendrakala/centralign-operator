# CentrAlign Operator

CentrAlign Operator is a local invoice-processing prototype. One bounded agent loop reads invoice sources and procurement policy, operates a sandbox ERP through Playwright, pauses for human approval when policy requires it, and verifies the resulting ledger row from the ERP page.

The ERP, invoices, credentials, approval broker, and SQLite ledger are synthetic demo data. The agent writes through the ERP form; it does not write directly to SQLite. The quickstart below explicitly uses FakeLLM replay and does not need an LLM API key.

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

Use `python3 web_app.py --offline --reset-demo-data` to clear the persisted ledger and run history before the server starts. The workbench also has a **Reset demo data** button; it cancels active runs, clears ledger, SQLite run records, and `runs/*.jsonl` logs, and preserves invoice files and policy.

The first startup copies the bundled text fixtures and policy into a new data directory and creates the sample Acme PDFs if they are missing. The local ERP login details are in `data/policies/company_memory.md`; they are synthetic sandbox credentials.

## Agent loop

Before browser startup, the active provider returns a schema-validated JSON task classification: `read_only`, `write_invoice_entry`, `unsupported`, or `ambiguous`. The host enforces that result and exposes only `read_ledger` and `finish` for read-only questions. Unsupported or unclear requests stop with a refusal or one clarifying question. For invoice entry, the provider returns one tool call per decision.

```text
Task
  -> AgentLoop asks the active provider for one tool call or browser action sequence
  -> host validates schema, route, selector, source, duplicate, and approval rules
  -> document tool or Playwright ERP action
  -> observation returned to the same loop
  -> finish only after live ledger read-back verification
```

The tool set includes `browser_fill_form` for a bounded batch of validated fields and `browser_action_sequence` for two or three sequential browser primitives. Both call the same route, selector, invoice-source, duplicate, vendor, and approval checks used by standalone browser actions. A visible ERP alert or failed action stops the sequence so the model can inspect and recover. Fill batches return a short list of fields set and one compact snapshot capped at 4,000 characters. Snapshots retain the URL path, page title, every visible alert, empty/filled form-field states, buttons, and capped table rows; passwords and other sensitive field values are never included.

Invoice entry starts with `list_invoice_candidates(vendor_query)`. The host parses every inbox PDF/TXT using the same structured extractor and returns each path, written vendor, invoice number, invoice date, due date, amount, and currency. The model selects from that list. For `latest_by_invoice_date`, the host checks that selection against every discovered candidate for the vendor and names a newer invoice if the selection is stale. Explicit invoice-number and other user criteria are preserved and are not replaced with a latest-date rule. The final result records the selected invoice, all candidates and dates, the criterion, and a passed selection check alongside ledger verification evidence.

Sign-in uses a dedicated `erp_login` tool. The host fills and submits credentials from `policies/company_memory.md`; the model cannot access the login selectors or password. Redirects to the login page state `Session required: call erp_login`. Navigating away from a populated form returns a warning listing the unsaved fields. If a tool with the same arguments repeats without changing page state, the agent warns after three repetitions and stops after five, including the last alert in its final message. Live providers receive the last six observations in full and one-line summaries of older observations; offline replay keeps its complete local state.

The Acme demo follows the actual ERP validation path. The source says `Acme Analytics`, while the ERP requires the legal name displayed in its vendor directory. The agent first enters the source label. After the ERP displays its unknown-vendor alert, the agent inspects the directory, retries with the observed legal entity, and reads the posted row back from the ledger.

## Providers

The app uses `FakeLLM` replay only when started explicitly with `--offline`. A live run without a configured API key, or one whose live providers fail, stops with a clear error; it never switches into replay.

For HTTP 429 and 503, the active provider is retried up to four times after the initial attempt. The agent honors `Retry-After` or a response message such as `retry in 8s`, caps the wait at 60 seconds, and otherwise uses exponential backoff with jitter. Each wait appears in the UI trace as `[RATE_LIMIT_RECOVERY]`. After those retries fail, it uses one other live provider only when `CENTRALIGN_FALLBACK_PROVIDER` is set.

Provider HTTP 400 tool-use or schema-validation errors, malformed JSON arguments, and host-side argument validation failures are returned to the same provider with the error and relevant tool schema. The model gets three retries per run; the run then stops and reports the last error. These validation retries never switch to offline replay. When `erp_login` runs, the trace states that credentials were supplied by the host. The credentials are not part of the model's tool schema or observations.

Invoice write tools report required form fields as filled or empty and give the next action when the form is ready. The model selects a source from the candidate table and reads it; the host attaches the selected document's filename and SHA-256 to the ERP form and trace. Amounts and other source-backed form values must match the parsed structured fields. A ledger check with no matching row reports whether a submit occurred and includes a recovery instruction. Repeated failures are counted by tool and error across the full run; the trace warns at two and stops with `needs_input` at four. The final summary states which parts of the invoice task were completed.

The host keeps approval tokens bound to the selected invoice and injects the token plus the verification checkbox immediately before submit. `#approval_token` is not available to model fill tools. If the policy gate is still pending, the run state and form feedback tell the model to call `request_human_approval` with the invoice number. Approval requests identify a single invoice by number or read-document path; ambiguous requests list the invoice candidates. An already-approved invoice returns `already_approved` without opening another human approval request. After a successful submit, host state says `submitted: <invoice_number>; next: verify_ledger_row`. Verification takes no required arguments: the host uses the submitted source document and returns per-field matches, the ledger row ID, and a row screenshot path. Passing verification ends the run with a structured result containing the action, ledger and approval evidence, source file/hash, and elapsed time.

Live runs keep four recent observations in full and summarize older history. The trace reports approximate per-turn prompt size broken down into system prompt, tool schemas, host run state, and task/history. Estimates use serialized characters divided by four. Requests and tool waits exceeding 30 seconds produce start and completion trace events. Every run event is also appended to `data/runs/<run_id>.jsonl` with credentials, approval tokens, and common provider-key formats redacted.

Groq (default model `llama-3.3-70b-versatile`):

```bash
export CENTRALIGN_PROVIDER=groq
export GROQ_API_KEY="..."
export CENTRALIGN_FALLBACK_PROVIDER=gemini # optional live fallback
export CENTRALIGN_GROQ_MODEL="llama-3.3-70b-versatile" # optional
python3 web_app.py
```

OpenRouter (default model `openrouter/free`):

```bash
export CENTRALIGN_PROVIDER=openrouter
export OPENROUTER_API_KEY="..."
export CENTRALIGN_OPENROUTER_MODEL="openrouter/free" # optional
python3 web_app.py
```

Gemini API (default model `gemini-2.5-flash-lite`):

```bash
export CENTRALIGN_PROVIDER=gemini
export GEMINI_API_KEY="..."
export CENTRALIGN_GEMINI_MODEL="gemini-2.5-flash-lite" # optional
python3 web_app.py
```

OpenAI Responses API:

```bash
export CENTRALIGN_PROVIDER=openai
export OPENAI_API_KEY="..."
export CENTRALIGN_MODEL="gpt-4o" # optional; this is the default
python3 web_app.py
```

`CENTRALIGN_PROVIDER` chooses the primary provider. If it is unset, the first configured provider is selected in this order: Groq, OpenRouter, Gemini, OpenAI. Set `CENTRALIGN_FALLBACK_PROVIDER` to at most one other configured provider to enable live failover. `CENTRALIGN_MODEL` overrides the primary provider's default model; a provider-specific model variable takes precedence when both are set. Gemini 2.5 Flash-Lite may not be enabled for every AI Studio account, so set `CENTRALIGN_GEMINI_MODEL` to a model available to your account if needed.

When a live provider is enabled, the task and observations—including extracted invoice text with detected instruction-like lines removed—are sent to that provider to choose the next tool call. Use only synthetic or otherwise approved data. ERP credentials and approval tokens stay on the local host and are never supplied to the model.

## Safety and verification

- Browser navigation is limited to the four local ERP routes. Form selectors are allowlisted, and source-backed values must match the selected document.
- The host checks invoice source and policy before a write, checks for a duplicate in the ERP ledger, blocks unapproved vendors, and enforces the approval gate.
- Approval tokens expire after five minutes, are bound to the invoice number, legal vendor, amount, and currency, and can be consumed once.
- Invoice and policy contents are treated as data. The model is told not to follow instructions embedded in source documents.
- One canonical policy file, `data/policies/procurement_policy.txt`, supplies the INR and USD approval thresholds and approved-vendor rules. Instruction-like invoice content is flagged in the trace, stripped from the model-facing text, and cannot override parsed amount fields.
- A run reports `complete` only after the ERP ledger DOM matches the source vendor/legal entity, invoice number, dates, amount, and currency. A row screenshot and source SHA-256 are retained as evidence under `data/evidence/<run-id>/`.
- SQLite in `company_app.py` persists ERP bills, run history, and audit events. The agent's posting path is the Playwright-operated ERP form.

## Code map

| File | Responsibility |
|---|---|
| `agent_loop.py` | Shared bounded observe/decide/execute loop, tool schemas, host checks, and stop conditions. |
| `llm_clients.py` | OpenAI, Gemini, Groq, OpenRouter, single-key provider routing, retry recovery, and offline-only FakeLLM. |
| `documents.py` | Confined invoice listing, PDF/TXT extraction, field parsing, and source hashes. |
| `policy.py` | Procurement policy parsing and threshold checks. |
| `browser_tools.py` | Playwright session, route/control allowlists, screenshots, and live ERP read-back. |
| `company_app.py` | SQLite persistence behind the ERP plus policy and vendor-directory data access. |
| `web_app.py` | FastAPI Operator APIs, approval token broker, and sandbox ERP pages. |
| `worker.py` | Small public entry point and CLI delegation. |
| `web/` | Operator workbench HTML, CSS, and JavaScript. |
| `data/inbox/`, `data/policies/` | Synthetic invoice sources, procurement rules, and sandbox company memory. |

## Checks

The tests cover strict task classification and write gating, read-only ledger lookups, unsupported and ambiguous tasks, candidate discovery and latest-date rejection, prompt injection, policy thresholds, host-managed sign-in, approval, ERP read-back, duplicate handling, token binding, loop detection, and tool/route guards.

```bash
python3 -m compileall -q .
python3 -m unittest -v test_worker.py
node --check web/app.js
```

## Current scope and known limitations

The approval waiter is in-memory and is intended for a single-user local demo. Live LLM availability and behavior depend on the selected provider and network access.
