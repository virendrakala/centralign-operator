# Architecture interview questions

### 1. What does CentrAlign Operator do?

It is a local invoice-entry prototype that reads synthetic PDF/TXT invoices, uses a bounded tool-calling loop, submits through a sandbox ERP, and verifies the saved ledger row. The loop is in `agent_loop.py`; the ERP and API are in `web_app.py`.

### 2. How does a task move through the system?

The host classifies intent, applies the deterministic write guard, then runs an observe/decide/execute loop. Tool results become observations for the next model turn. `agent_loop.py` implements that flow.

### 3. Why use both an LLM classifier and a deterministic guard?

The classifier can map natural language actions to capabilities, while the guard independently requires a recognized entry verb and blocks recognized unsupported verbs. Either layer can prevent write access. See `agent_loop.py` and `llm_clients.py`.

### 4. What can the agent write?

Only invoice entries in the local ERP flow. It does not pay invoices, send email, or operate on a real finance system. The allowed tools and finalization checks are in `agent_loop.py`.

### 5. What does the model control, and what does the host control?

The model chooses the next permitted tool call and recovery step. The host owns credentials, source parsing and hashes, approval tokens, tool validation, ERP interaction, verification, and final evidence. Host tools are in `agent_loop.py`, `documents.py`, and `browser_tools.py`.

### 6. How does ERP login work?

The model calls `erp_login`; the host reads the sandbox credential from company memory and fills the login form. Login selectors and the password are not exposed as model-fillable fields. See `web_app.py` and `browser_tools.py`.

### 7. How are approvals enforced?

`policy.py` parses currency thresholds. `web_app.py` waits for a human decision and issues a short-lived token bound to the invoice. The host supplies it at submit time; the ERP consumes it once. See `web_app.py` and `agent_loop.py`.

### 8. Why is the approval token not a model argument?

The token is an authorization secret. The host binds and injects it only after approval, so a model cannot guess, copy, or redirect the approval to another invoice. See the approval flow in `agent_loop.py` and `web_app.py`.

### 9. How is invoice provenance established?

`documents.py` confines reads to the inbox, parses labeled fields, and computes SHA-256 from source bytes. `agent_loop.py` attaches the source path/hash to the host-side ERP entry and final evidence.

### 10. How does the system handle prompt injection in invoices?

`documents.py` detects a small set of instruction-like patterns, flags them, and removes matching lines from model-facing text. Structured invoice fields are parsed separately. This is a heuristic defense, not a complete prompt-injection solution.

### 11. Why use independent ledger verification?

A form submit response is not proof that the intended values were saved. `browser_tools.py` reads the ERP ledger DOM and compares the row fields with the parsed invoice; `agent_loop.py` only permits a successful write finish after a passing result.

### 12. How does browser safety work?

`browser_tools.py` limits navigation to local ERP routes and selectors. `agent_loop.py` applies task-specific tool availability and form/source/policy checks before actions execute.

### 13. How are provider errors handled?

`llm_clients.py` retries a provider on 429/503 using the server retry hint or capped exponential backoff, then uses a configured live fallback if present. `agent_loop.py` feeds recoverable tool-schema errors back to the model and stops after its retry budget.

### 14. What is offline replay?

It is a scripted `FakeLLM` test double selected only with `--offline`; it is not a live provider and does not take over a failed live run. See `llm_clients.py`, `worker.py`, and `README.md`.

### 15. What are the main limitations and next steps?

The current workflow supports invoice entry into one fixed-selector sandbox ERP, and extraction is limited to labeled PDF/TXT files. Next steps could include generic page understanding, vision-based browser control, persistent company memory, broader evaluations, and a hosted demo. See `README.md`.
