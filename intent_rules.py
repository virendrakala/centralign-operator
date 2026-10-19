"""Deterministic phrase checks for unsupported user-requested actions.

These checks are deliberately conservative: only affirmative action verbs with
an actionable object block invoice entry. Incidental mentions and negated
actions do not lock the workflow.
"""

from __future__ import annotations

import re


HOST_ENTRY_VERBS = re.compile(r"\b(?:enter|record|post|log|add|submit)\b", re.IGNORECASE)
HOST_INVOICE_TOPIC = re.compile(r"\b(?:invoices?|invoiced|bills?|billed|billing)\b", re.IGNORECASE)

_UNSUPPORTED_ACTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("approve payment", re.compile(r"\bapprove\s+(?:the\s+)?payments?\b", re.IGNORECASE)),
    ("pay", re.compile(r"\bpay(?:ing)?\b|\b(?:make|initiate|process)\s+(?:a\s+)?payments?\b", re.IGNORECASE)),
    ("email vendor", re.compile(r"\b(?:email|e-mail)\s+(?:(?:the|a|an|our|your)\s+)?(?:vendor|supplier)\b", re.IGNORECASE)),
    ("email", re.compile(r"\b(?:email|e-mail)\s+(?:(?:the|a|an|our|your)\s+)?(?:invoice|bill|recipient|customer|[\w.+-]+@[\w.-]+)\b", re.IGNORECASE)),
    ("send", re.compile(r"\bsend\s+(?:(?:any|the|a|an|our|your)\s+)?(?:e-?mail|message|invoice|bill|payment|funds?|money)\b|\bsend\s+(?:(?:the|a|an|our|your)\s+)?(?:to\s+)?(?:vendor|supplier|finance|accounting)\b", re.IGNORECASE)),
    ("transfer", re.compile(r"\b(?:transfer|wire)\s+(?:(?:the|a|an|our|your)\s+)?(?:payment|funds?|money|invoice|amount)\b|\bwire\s+(?:the\s+)?vendor\b", re.IGNORECASE)),
    ("delete", re.compile(r"\b(?:delete|remove)\s+(?!(?:nothing|anything|no\s+(?:entries|rows|invoices?|bills?))\b)(?:(?:the|this|that|an?|our|your)\s+)?(?:[\w#-]+\s+){0,4}(?:invoice|bill|ledger|entry|row|record)\b", re.IGNORECASE)),
    ("refund", re.compile(r"\brefund\s+(?:(?:the|a|an|our|your)\s+)?(?:payment|invoice|amount|vendor|supplier|customer)\b", re.IGNORECASE)),
    ("contact vendor", re.compile(r"\bcontact\s+(?:(?:the|a|an|our|your)\s+)?(?:vendor|supplier)\b", re.IGNORECASE)),
    ("notify vendor", re.compile(r"\b(?:notify|inform|alert|tell)\s+(?:(?:the|a|an|our|your)\s+)?(?:vendor|supplier)\b", re.IGNORECASE)),
    ("notify finance", re.compile(r"\b(?:let\s+(?:the\s+)?finance\s+know|(?:notify|inform|alert|tell)\s+(?:(?:the|a|an|our|your)\s+)?(?:finance|accounting))\b", re.IGNORECASE)),
)

_NEGATED_PREFIX = re.compile(
    r"(?:\b(?:do\s+not|don't|dont|never|cannot|can't|should\s+not|shouldn't|must\s+not|mustn't|need\s+not|no\s+need\s+to|without)\b)(?:\W+\w+){0,4}\W*$",
    re.IGNORECASE,
)


def _negated_request(text: str, start: int) -> bool:
    prefix = text[max(0, start - 100):start]
    return bool(_NEGATED_PREFIX.search(prefix))


def deterministic_unsupported_actions(task: str) -> list[dict[str, str]]:
    """Return affirmative unsupported actions, in the order requested."""
    found: list[tuple[int, str]] = []
    for label, pattern in _UNSUPPORTED_ACTIONS:
        for match in pattern.finditer(task):
            if not _negated_request(task, match.start()):
                found.append((match.start(), label))

    # "email" or "send" may both match the same send-an-email phrase. Keep one
    # concise action, and normalize any email-to-vendor wording consistently.
    if re.search(r"\bsend\s+(?:(?:any|the|a|an|our|your)\s+)?e-?mail\b", task, re.IGNORECASE):
        found = [(position, label) for position, label in found if label != "email"]
        found = [(position, "email vendor" if re.search(r"\b(?:vendor|supplier)\b", task, re.IGNORECASE) else "email") if label == "send" else (position, label) for position, label in found]

    actions: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, label in sorted(found):
        if label not in seen:
            actions.append({"action": label, "capability": "unsupported"})
            seen.add(label)
    return actions

