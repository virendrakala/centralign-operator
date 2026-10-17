"""Policy text parsing shared by the host guardrails and agent observations."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any


def parse_policy(content: str, vendor_directory: list[dict[str, str]]) -> dict[str, Any]:
    """Extract only policy facts that appear in the selected procurement policy."""
    thresholds: dict[str, str] = {}
    for currency, amount in re.findall(
        r"\b(INR|USD)\s+([\d,]+(?:\.\d{1,2})?)", content, re.IGNORECASE
    ):
        thresholds[currency.upper()] = f"{Decimal(amount.replace(',', '')):.2f}"

    approved_vendors = [
        vendor["legal_name"]
        for vendor in vendor_directory
        if vendor["legal_name"].casefold() in content.casefold()
    ]
    return {
        "thresholds": thresholds,
        "approved_vendors": approved_vendors,
        "duplicate_key": ["vendor", "invoice_number"],
        "content": content,
    }


def approval_required(invoice: dict[str, Any], policy: dict[str, Any]) -> tuple[bool, str | None]:
    """Check the amount against the threshold read from the policy text."""
    threshold_text = policy.get("thresholds", {}).get(str(invoice.get("currency", "")).upper())
    if threshold_text is None:
        return True, f"No approval threshold was found for {invoice.get('currency', 'this currency')} in procurement policy."
    amount = Decimal(str(invoice["amount"]))
    threshold = Decimal(threshold_text)
    if amount > threshold:
        return True, f"{invoice['currency']} {amount:.2f} exceeds the {invoice['currency']} {threshold:.2f} policy threshold."
    return False, None
