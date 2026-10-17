"""Confined invoice and policy file access for the local agent runtime."""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
POLICY_PATH = "policies/procurement_policy.txt"
_INVOICE_LABELS = {
    "invoice_number": r"(?:invoice\s*(?:number|no\.?|#))\s*[:#]\s*(.+)",
    "vendor": r"vendor\s*:\s*(.+)",
    "invoice_date": r"invoice\s*date\s*:\s*(.+)",
    "due_date": r"due\s*date\s*:\s*(.+)",
    "amount_raw": r"amount\s*(?:due)?\s*:\s*(.+)",
}


def ensure_operator_assets(data_dir: Path) -> None:
    """Create the synthetic demo documents and policy files when a data dir is empty."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    inbox = data_dir / "inbox"
    policies = data_dir / "policies"
    inbox.mkdir(parents=True, exist_ok=True)
    policies.mkdir(parents=True, exist_ok=True)

    for source in (ROOT / "data" / "inbox").glob("*.txt"):
        target = inbox / source.name
        if not target.exists():
            shutil.copyfile(source, target)
    for name in ("procurement_policy.txt", "company_memory.md"):
        source = ROOT / "data" / "policies" / name
        target = policies / name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)

    fixtures = [
        ("INV_2026_08_Acme.pdf", "INV-2026-2084", "2026-08-04", "2026-09-03", "42,000.00"),
        ("INV_2026_10_Acme.pdf", "INV-2026-2116", "2026-10-21", "2026-11-15", "68,500.00"),
    ]
    for filename, number, invoice_date, due_date, amount in fixtures:
        target = inbox / filename
        if target.exists():
            try:
                from pypdf import PdfReader
                if PdfReader(str(target)).metadata.subject == "CentrAlign Operator Fixture v3":
                    continue
            except Exception:
                pass
        pdf = canvas.Canvas(str(target), pagesize=letter)
        pdf.setTitle(f"CentrAlign synthetic invoice {filename}")
        pdf.setSubject("CentrAlign Operator Fixture v3")
        pdf.setAuthor("CentrAlign Sandbox")
        pdf.setFillColorRGB(.06, .12, .22)
        pdf.rect(0, 708, 612, 84, stroke=0, fill=1)
        pdf.setFillColorRGB(1, 1, 1)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(52, 753, "CENTRALIGN")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(53, 736, "SANDBOX ACCOUNTS PAYABLE")
        pdf.setFillColorRGB(.06, .12, .22)
        pdf.setFont("Helvetica-Bold", 22)
        pdf.drawString(52, 665, "INVOICE")
        pdf.setFillColorRGB(.36, .42, .50)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(52, 644, "SYNTHETIC TRAINING DOCUMENT")
        pdf.setStrokeColorRGB(.83, .87, .91)
        pdf.line(52, 628, 560, 628)
        pdf.setFillColorRGB(.36, .42, .50)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(52, 602, "BILL FROM")
        pdf.drawString(315, 602, "BILL TO")
        pdf.setFillColorRGB(.08, .13, .21)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(52, 582, "CentrAlign Sandbox Services")
        pdf.drawString(315, 582, "CentrAlign Finance Operations")
        pdf.setFillColorRGB(.36, .42, .50)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(52, 566, "Synthetic vendor billing environment")
        pdf.drawString(315, 566, "Accounts Payable Department")
        pdf.setStrokeColorRGB(.90, .92, .95)
        pdf.line(52, 544, 560, 544)
        pdf.setFillColorRGB(.36, .42, .50)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(52, 524, "INVOICE DETAILS")
        pdf.setFillColorRGB(.10, .16, .25)
        pdf.setFont("Helvetica", 10)
        for y, line in zip((501, 482, 463, 444, 425), (
            f"Invoice Number: {number}", "Vendor: Acme Analytics",
            f"Invoice Date: {invoice_date}", f"Due Date: {due_date}", f"Amount Due: INR {amount}",
        )):
            pdf.drawString(58, y, line)
        pdf.setFillColorRGB(.95, .96, .98)
        pdf.roundRect(52, 337, 508, 37, 5, stroke=0, fill=1)
        pdf.setFillColorRGB(.36, .42, .50)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(63, 359, "DESCRIPTION")
        pdf.drawString(366, 359, "QTY")
        pdf.drawString(415, 359, "UNIT PRICE")
        pdf.drawString(506, 359, "AMOUNT")
        pdf.setFillColorRGB(.10, .16, .25)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(63, 346, "Cloud analytics subscription")
        pdf.drawRightString(382, 346, "1")
        pdf.drawRightString(477, 346, f"INR {amount}")
        pdf.drawRightString(554, 346, f"INR {amount}")
        pdf.setStrokeColorRGB(.83, .87, .91)
        pdf.line(52, 319, 560, 319)
        pdf.setFillColorRGB(.36, .42, .50)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(380, 290, "TOTAL DUE")
        pdf.setFillColorRGB(.06, .12, .22)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawRightString(554, 287, f"INR {amount}")
        pdf.setFillColorRGB(.36, .42, .50)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(52, 242, "Payment terms: Net 30")
        pdf.drawString(52, 225, "Please reference the invoice number with any payment inquiry.")
        pdf.setFont("Helvetica-Oblique", 9)
        pdf.setFillColorRGB(.38, .43, .50)
        pdf.drawString(52, 52, "Synthetic training data. Not a real payable or financial record.")
        pdf.save()


def list_workspace_files(data_dir: Path) -> list[dict[str, str]]:
    """List invoice files and the one policy file exposed to the planner."""
    files: list[dict[str, str]] = []
    inbox = data_dir / "inbox"
    policies = data_dir / "policies"
    for path in sorted(inbox.iterdir(), key=lambda item: item.name.casefold()) if inbox.exists() else []:
        if path.is_file() and path.suffix.casefold() in {".pdf", ".txt"}:
            files.append({"path": path.relative_to(data_dir).as_posix(), "name": path.name, "kind": path.suffix[1:].casefold()})
    policy = policies / "procurement_policy.txt"
    if policy.is_file():
        files.append({"path": "policies/procurement_policy.txt", "name": policy.name, "kind": "txt"})
    return files


def extract_document_text(data_dir: Path, file_path: str) -> str:
    """Extract plain text from an allowed inbox document without leaving the workspace."""
    candidate = (data_dir / file_path).resolve()
    allowed_roots = [(data_dir / "inbox").resolve(), (data_dir / "policies").resolve()]
    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise ValueError("Document path must stay inside the invoice inbox or policies folder.")
    if candidate.relative_to(data_dir.resolve()).as_posix() == "policies/company_memory.md":
        raise ValueError("Company memory contains sandbox credentials and is not available to the model.")
    if not candidate.is_file() or candidate.suffix.casefold() not in {".pdf", ".txt", ".md"}:
        raise ValueError("Only existing PDF, TXT, or Markdown documents can be extracted.")
    if candidate.suffix.casefold() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(candidate)).pages).strip()
    return candidate.read_text(encoding="utf-8")


def parse_invoice_text(text: str, filename: str, source_sha256: str) -> dict[str, Any] | None:
    """Extract the standard fields from text extracted from PDF or TXT invoices."""
    values = {
        key: match.group(1).strip()
        for key, pattern in _INVOICE_LABELS.items()
        if (match := re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
    }
    if len(values) != len(_INVOICE_LABELS):
        return None
    amount_match = re.fullmatch(r"([A-Z]{3})\s+([\d,]+(?:\.\d{1,2})?)", values["amount_raw"].upper())
    if not amount_match:
        return None

    def normalize_date(value: str) -> str:
        for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(value.strip(), pattern).date().isoformat()
            except ValueError:
                continue
        raise ValueError(f"Unsupported invoice date format in {filename}: {value}")

    currency, raw_amount = amount_match.groups()
    amount = Decimal(raw_amount.replace(",", ""))
    if not amount.is_finite() or amount <= 0:
        return None
    invoice_date = normalize_date(values["invoice_date"])
    due_date = normalize_date(values["due_date"])
    return {
        "invoice_number": values["invoice_number"],
        "vendor": values["vendor"],
        "invoice_date": invoice_date,
        "due_date": due_date,
        "amount": f"{amount:.2f}",
        "currency": currency,
        "source_file": filename,
        "source_sha256": source_sha256,
        "source_fields": {
            "vendor": values["vendor"], "invoice_number": values["invoice_number"],
            "invoice_date": values["invoice_date"], "due_date": values["due_date"],
            "amount": raw_amount, "currency": currency,
        },
    }


def read_document(data_dir: Path, path: str) -> dict[str, Any]:
    """Return extracted content plus structured invoice data when the file is an invoice."""
    text = extract_document_text(data_dir, path)
    digest = hashlib.sha256((data_dir / path).read_bytes()).hexdigest()
    return {
        "status": "read",
        "path": path,
        "content": text,
        "sha256": digest,
        "invoice": parse_invoice_text(text, Path(path).name, digest),
    }
