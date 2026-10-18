"""Local accounts-payable system, policy store, and auditable SQLite ledger."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from policy import parse_policy


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
POLICY_RELATIVE_PATH = Path("policies/procurement_policy.txt")
INVOICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
AMOUNT_RE = re.compile(r"^([A-Z]{3})\s+([\d,]+(?:\.\d{1,2})?)$")
FIELDS = ("vendor", "invoice_number", "invoice_date", "due_date", "amount", "currency")


class InvoiceError(ValueError):
    """Raised when an invoice document cannot be read safely."""


class TransientWriteError(RuntimeError):
    """A recoverable database write fault, used by the local failure injector."""


class CompanyApp:
    """File-backed invoice inbox and SQLite-backed mock finance application."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.inbox_dir = self.data_dir / "inbox"
        self.policy_path = self.data_dir / POLICY_RELATIVE_PATH
        self.db_path = self.data_dir / "company.sqlite3"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.policy_path.exists():
            bundled_policy = ROOT / "data" / POLICY_RELATIVE_PATH
            if bundled_policy.exists() and bundled_policy.resolve() != self.policy_path.resolve():
                shutil.copyfile(bundled_policy, self.policy_path)
        memory_path = self.data_dir / "policies" / "company_memory.md"
        bundled_memory = ROOT / "data" / "policies" / "company_memory.md"
        if not memory_path.exists() and bundled_memory.exists() and bundled_memory.resolve() != memory_path.resolve():
            shutil.copyfile(bundled_memory, memory_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vendor TEXT NOT NULL,
                    invoice_number TEXT NOT NULL,
                    invoice_date TEXT NOT NULL,
                    due_date TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(vendor, invoice_number)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    summary TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    sequence INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                """
            )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    def _invoice_files(self) -> list[Path]:
        return sorted(self.inbox_dir.glob("*.txt"))

    @staticmethod
    def _normalize_date(value: str) -> tuple[str, bool]:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
            if parsed.strftime("%Y-%m-%d") == value:
                return value, False
        except ValueError:
            pass
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(value, fmt).strftime("%Y-%m-%d"), True
            except ValueError:
                continue
        raise InvoiceError("invoice and due dates must use YYYY-MM-DD (or a recognizable day/month/year date)")

    def _parse_invoice(self, path: Path, *, allow_normalization: bool = False) -> dict[str, Any]:
        values: dict[str, str] = {}
        try:
            raw_bytes = path.read_bytes()
            raw_text = raw_bytes.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise InvoiceError(f"source file could not be read: {exc}") from exc
        for line in raw_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip().casefold()] = value.strip()

        required = ("invoice number", "vendor", "invoice date", "due date", "amount due")
        missing = [key for key in required if not values.get(key)]
        if missing:
            raise InvoiceError(f"missing field(s): {', '.join(missing)}")

        normalizations: list[dict[str, str]] = []
        dates: dict[str, str] = {}
        for label, key in (("invoice date", "invoice date"), ("due date", "due date")):
            normalized_date, changed = self._normalize_date(values[key])
            if changed:
                if not allow_normalization:
                    raise InvoiceError(f"{label} needs format normalization to YYYY-MM-DD")
                normalizations.append({"field": key.replace(" ", "_"), "from": values[key], "to": normalized_date})
            dates[key] = normalized_date

        vendor = values["vendor"]

        amount_match = AMOUNT_RE.fullmatch(values["amount due"])
        if not amount_match:
            raise InvoiceError("amount due must look like 'INR 1,234.50'")
        currency, raw_amount = amount_match.groups()
        try:
            amount = Decimal(raw_amount.replace(",", ""))
        except InvalidOperation as exc:
            raise InvoiceError("amount due is not a valid number") from exc
        if not amount.is_finite() or amount <= 0:
            raise InvoiceError("amount due must be a positive amount")

        invoice_number = values["invoice number"]
        if len(invoice_number) > 100 or len(vendor) > 160:
            raise InvoiceError("invoice number or vendor name is too long")
        return {
            "invoice_id": path.stem,
            "invoice_number": invoice_number,
            "vendor": vendor,
            "invoice_date": dates["invoice date"],
            "due_date": dates["due date"],
            "amount": f"{amount:.2f}",
            "currency": currency,
            "source_file": path.name,
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "source_fields": {
                "vendor": values["vendor"],
                "invoice_number": invoice_number,
                "invoice_date": values["invoice date"],
                "due_date": values["due date"],
                "amount": raw_amount,
                "currency": currency,
            },
            "normalizations": normalizations,
        }

    def _all_invoices(self) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        invoices: list[dict[str, Any]] = []
        unreadable: list[dict[str, str]] = []
        for path in self._invoice_files():
            try:
                invoices.append(self._parse_invoice(path, allow_normalization=True))
            except InvoiceError as exc:
                unreadable.append({"file": path.name, "reason": str(exc)})
        return invoices, unreadable

    def list_vendors(self) -> list[str]:
        return sorted({item["legal_name"] for item in self.list_vendor_directory()}, key=str.casefold)

    def list_vendor_directory(self) -> list[dict[str, str]]:
        """Read the sandbox vendor directory from company memory, not code constants."""
        path = self.data_dir / "policies" / "company_memory.md"
        if not path.exists():
            return []
        vendors = []
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"\s*([A-Za-z0-9_-]+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*", line)
            if match:
                code, short_name, legal_name = match.groups()
                vendors.append({"code": code, "short_name": short_name.strip(), "legal_name": legal_name.strip()})
        return vendors

    def list_invoices(self) -> dict[str, Any]:
        invoices, unreadable = self._all_invoices()
        invoices.sort(key=lambda item: (item["invoice_date"], item["invoice_number"]), reverse=True)
        return {"invoices": invoices, "unreadable_files": unreadable}

    def suggest_vendors(self, query: str) -> list[str]:
        vendors = self.list_vendors()
        normalized_query = self._normalize(query)
        direct = [vendor for vendor in vendors if normalized_query in self._normalize(vendor)]
        if direct:
            return direct[:5]
        query_words = re.findall(r"[a-z0-9]+", normalized_query)
        ranked: list[tuple[float, str]] = []
        for vendor in vendors:
            vendor_words = re.findall(r"[a-z0-9]+", self._normalize(vendor))
            score = max(
                (difflib.SequenceMatcher(None, query_word, vendor_word).ratio() for query_word in query_words for vendor_word in vendor_words),
                default=0.0,
            )
            if score >= 0.68:
                ranked.append((score, vendor))
        ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
        if ranked and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.08):
            return [ranked[0][1]]
        return [vendor for _, vendor in ranked[:5]]

    def search_invoices(self, vendor: str) -> dict[str, Any]:
        if not vendor.strip() or len(vendor) > 160:
            return {"status": "invalid_query", "message": "Provide a vendor name.", "candidate_vendors": []}

        invoices, unreadable = self._all_invoices()
        query = self._normalize(vendor)
        matches = [
            invoice for invoice in invoices
            if query in self._normalize(invoice["vendor"]) or self._normalize(invoice["vendor"]) in query
        ]
        matches.sort(key=lambda item: (item["invoice_date"], item["invoice_number"]), reverse=True)

        latest_id: str | None = None
        status = "no_match"
        matching_vendors = {self._normalize(item["vendor"]) for item in matches}
        if len(matching_vendors) > 1:
            status = "ambiguous_vendor"
        elif unreadable:
            status = "needs_review"
        elif matches:
            latest_date = matches[0]["invoice_date"]
            latest = [item for item in matches if item["invoice_date"] == latest_date]
            if len(latest) == 1:
                latest_id = latest[0]["invoice_id"]
                status = "found"
            else:
                status = "ambiguous_latest"

        return {
            "status": status,
            "query": vendor,
            "matching_vendors": sorted({item["vendor"] for item in matches}, key=str.casefold),
            "candidate_vendors": self.suggest_vendors(vendor) if status in {"no_match", "ambiguous_vendor"} else [],
            "match_count": len(matches),
            "latest_invoice_id": latest_id,
            "matches": matches,
            "unreadable_files": unreadable,
        }

    def read_invoice(self, invoice_id: str) -> dict[str, Any]:
        if not INVOICE_ID_RE.fullmatch(invoice_id):
            raise InvoiceError("invalid invoice id")
        path = next((item for item in self._invoice_files() if item.stem == invoice_id), None)
        if path is None:
            raise InvoiceError("invoice is no longer available in the inbox")
        return self._parse_invoice(path)

    def normalize_and_validate_invoice(self, invoice_id: str) -> dict[str, Any]:
        if not INVOICE_ID_RE.fullmatch(invoice_id):
            raise InvoiceError("invalid invoice id")
        path = next((item for item in self._invoice_files() if item.stem == invoice_id), None)
        if path is None:
            raise InvoiceError("invoice is no longer available in the inbox")
        return self._parse_invoice(path, allow_normalization=True)

    def read_company_policy(self) -> dict[str, Any]:
        try:
            content = self.policy_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InvoiceError(f"procurement policy is unavailable: {exc}") from exc
        parsed = parse_policy(content, self.list_vendor_directory())
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return {
            "path": POLICY_RELATIVE_PATH.as_posix(),
            "content": content,
            "source_sha256": digest,
            "approved_vendors": parsed["approved_vendors"],
            "thresholds": parsed["thresholds"],
            "approval_thresholds": parsed["thresholds"],
            "duplicate_key": ["vendor", "invoice_number"],
        }

    def evaluate_invoice_policy(self, invoice: dict[str, Any]) -> dict[str, Any]:
        value = Decimal(invoice["amount"])
        policy = self.read_company_policy()
        threshold_value = policy["thresholds"].get(invoice["currency"])
        threshold = Decimal(threshold_value) if threshold_value is not None else None
        high_value = threshold is not None and value > threshold
        vendor_approved = self._normalize(invoice["vendor"]) in {self._normalize(name) for name in policy["approved_vendors"]}
        currency_supported = threshold is not None
        reasons = []
        if not vendor_approved:
            reasons.append("Vendor is not on the approved-vendor list.")
        if not currency_supported:
            reasons.append(f"No approval threshold is configured for {invoice['currency']}; human review is required.")
        if high_value:
            reasons.append(f"{invoice['currency']} {invoice['amount']} exceeds the {invoice['currency']} {threshold:.2f} manager-approval threshold.")
        return {
            "vendor_approved": vendor_approved,
            "currency_supported": currency_supported,
            "requires_manager_approval": high_value,
            "approval_threshold": f"{invoice['currency']} {threshold:.2f}" if threshold is not None else None,
            "reasons": reasons,
            "compliant_to_record": vendor_approved and currency_supported,
        }

    def list_inspectable_files(self) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []
        if self.policy_path.exists():
            paths = [self.policy_path, *self._invoice_files()]
        else:
            paths = self._invoice_files()
        for path in paths:
            try:
                raw = path.read_bytes()
                files.append({
                    "name": path.name,
                    "path": POLICY_RELATIVE_PATH.as_posix() if path == self.policy_path else f"inbox/{path.name}",
                    "content": raw.decode("utf-8"),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
            except (OSError, UnicodeError):
                continue
        return files

    def find_bill(self, invoice: dict[str, Any]) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM bills WHERE vendor = ? AND invoice_number = ?",
                (invoice["vendor"], invoice["invoice_number"]),
            ).fetchone()
        return dict(row) if row else None

    def create_bill(self, invoice: dict[str, Any], *, inject_transient_fault: bool = False) -> dict[str, Any]:
        """Insert once; an explicitly injected fault affects only the current write attempt."""
        if inject_transient_fault:
            raise TransientWriteError("Simulated transient database lock; retry this write once after backoff.")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO bills (
                    vendor, invoice_number, invoice_date, due_date,
                    amount, currency, source_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vendor, invoice_number) DO NOTHING
                """,
                (invoice["vendor"], invoice["invoice_number"], invoice["invoice_date"],
                 invoice["due_date"], invoice["amount"], invoice["currency"], invoice["source_file"], created_at),
            )
            created = cursor.rowcount == 1
            row = conn.execute(
                "SELECT * FROM bills WHERE vendor = ? AND invoice_number = ?",
                (invoice["vendor"], invoice["invoice_number"]),
            ).fetchone()
        return {"status": "created" if created else "already_exists", "bill": dict(row)}

    def verify_bill(self, invoice: dict[str, Any]) -> dict[str, Any]:
        bill = self.find_bill(invoice)
        verified_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        if bill is None:
            return {
                "status": "not_found",
                "message": "No matching bill was found in the internal system.",
                "field_diffs": [],
                "source_sha256": invoice.get("source_sha256"),
                "verified_at": verified_at,
            }
        field_diffs = [
            {
                "field": field,
                "source_value": invoice.get("source_fields", {}).get(field, invoice[field]),
                "normalized_value": invoice[field],
                "ledger_value": bill[field],
                "match_status": "VERIFIED" if str(invoice[field]) == str(bill[field]) else "MISMATCH",
            }
            for field in FIELDS
        ]
        mismatches = {
            item["field"]: {"expected": item["source_value"], "actual": item["ledger_value"]}
            for item in field_diffs if item["match_status"] == "MISMATCH"
        }
        return {
            "status": "verified" if not mismatches else "mismatch",
            "bill": bill,
            "mismatches": mismatches,
            "field_diffs": field_diffs,
            "source_sha256": invoice.get("source_sha256"),
            "verified_at": verified_at,
        }

    def start_run(self, run_id: str, task: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (id, task, status, started_at) VALUES (?, ?, ?, ?)",
                (run_id, task, "running", datetime.now(timezone.utc).isoformat()),
            )

    def append_event(self, run_id: str, sequence: int, event_type: str, payload: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_events (run_id, sequence, timestamp, event_type, payload) VALUES (?, ?, ?, ?, ?)",
                (run_id, sequence, datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 event_type, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )

    def finish_run(self, run_id: str, status: str, summary: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, summary = ? WHERE id = ?",
                (status, datetime.now(timezone.utc).isoformat(), summary, run_id),
            )

    def reset_demo_data(self) -> None:
        """Clear persisted ledger, run history, and audit events, keeping source fixtures."""
        with self._connect() as conn:
            conn.execute("DELETE FROM audit_events")
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM bills")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('audit_events', 'bills')")

    def list_bills(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM bills ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def read_audit(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sequence, timestamp, event_type, payload FROM audit_events WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result
