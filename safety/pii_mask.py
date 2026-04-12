from __future__ import annotations

import re


PATTERNS = [
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    re.compile(r"\b(?:\+44|0)\d{10}\b"),
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    re.compile(r"\b\d{2}-\d{2}-\d{2}\b"),
]


def mask_text(text: str) -> str:
    masked = text
    for pattern in PATTERNS:
        masked = pattern.sub("[REDACTED]", masked)
    return masked
