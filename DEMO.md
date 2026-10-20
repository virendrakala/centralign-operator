# Three-minute demo script

Use a fresh local database so the approval and read-only examples are easy to follow:

```bash
python3 web_app.py --offline --reset-demo-data
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Confirm the header says **FakeLLM · offline replay**. Replay is a scripted test double; this demo does not call a live model.

## 0:00–1:00 — latest invoice and approval

Submit:

> Find the latest Acme invoice and record it in the ledger.

When the approval card appears, point out that the selected source is `INV-2026-2116`, dated 2026-10-21, for INR 68,500. The policy threshold is INR 50,000. Read the policy reason and then click **Approve**. The run should submit through the ERP and finish with a verified ledger row.

Point out the source file/hash, approval evidence, per-field ledger match, and selected invoice in the result card. The host performs login and injects the approval token; those values are not model-provided.

## 1:00–1:35 — refusal

Submit:

> Pay the Northstar invoice and email the vendor.

The run should finish `needs_input` without a browser write. Point out the unsupported requested actions and the host’s offer to record an invoice in the ledger if that is all the user wants. The demo does not make payments or send email.

## 1:35–2:00 — read-only question

Submit:

> Show me what Acme has been invoiced.

The result should be rendered from the ledger rows returned by the host. Point out that the task ends with a structured answer and does not open or submit the ERP form.

## 2:00–2:35 — August invoice

Submit:

> Enter the August Acme invoice.

The offline replay should select `INV-2026-2084`, dated 2026-08-04, for INR 42,000. This is below the INR 50,000 threshold, so no policy approval is needed. Point out the selection criterion and the independent ledger verification.

## 2:35–3:00 — what to point out

- The trace summary groups tool actions with their observations; Diagnostics exposes the raw events.
- The stepper shows intent, selection, approval, submission, and verification state.
- The result evidence comes from host-side parsing and ERP read-back.
- The UI labels this run as offline replay. For a live-provider demo, configure a key as described in `README.md` and repeat with synthetic data.

## Live demo

To demonstrate CentrAlign with a live language model (e.g. Gemini, Groq, OpenAI, or OpenRouter):

### 1. Configure provider credentials

Export the provider environment variables in your terminal:

```bash
# Example for Gemini:
export CENTRALIGN_PROVIDER=gemini
export GEMINI_API_KEY="your-gemini-api-key"
export CENTRALIGN_GEMINI_MODEL="gemini-2.5-flash-lite"

# Or example for Groq:
# export CENTRALIGN_PROVIDER=groq
# export GROQ_API_KEY="your-groq-api-key"
```

### 2. Reset demo data and start the app

Start the server with a pristine demo database:

```bash
python3 web_app.py --reset-demo-data
```

You can also reset the database at any point during your demo using either:
- **UI button:** Click **Reset demo data** in the top navigation bar.
- **REST endpoint:** `curl -X POST http://127.0.0.1:8765/api/demo/reset -H "Content-Type: application/json" -d '{}'`

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Confirm the top banner indicates your active live model (e.g. **Gemini · gemini-2.5-flash-lite**).

### 3. Step-by-step walkthrough

1. **High-value invoice requiring approval:**
   - Submit: `Find the latest Acme invoice and record it in the ledger.`
   - Observe the live model parse candidate invoices, inspect procurement policy, identify that INR 68,500 exceeds the INR 50,000 threshold, and pause for human approval.
   - Click **Approve**. The host injects the single-use approval token (`APR-...`), submits the ERP form, and independently verifies the newly created ledger row.

2. **Policy & verb guard refusal:**
   - Submit: `Pay the Northstar invoice and email the vendor.`
   - Point out that the capability classifier and deterministic verb guard immediately refuse the unsupported actions before any write tools can run.

3. **Read-only query:**
   - Submit: `Show me what Acme has been invoiced.`
   - The agent reads host ledger records without touching browser write tools, rendering the answer directly in the timeline.

4. **Duplicate invoice handling:**
   - Submit: `Find the latest Acme invoice and record it in the ledger.`
   - Because invoice `INV-2026-2116` was already posted in step 1, the agent checks the ledger, recognizes the existing row, verifies it against the source file, and finishes with: `"Already in ledger – verified, no new write"`. No duplicate ledger entry is created.

5. **Rate-limit handling (if triggered):**
   - For temporary HTTP 429 limits, point out the live status bar displaying the backoff countdown; the agent retries with jitter. A hard daily token-quota error skips the long retry waits and tries the explicitly configured live fallback, if one is set. Otherwise the run stops with a clear provider error.
