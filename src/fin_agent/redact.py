"""
Remove personal information (PII) from text before it leaves this module:
before an LLM call, a log line, a trace, or a commit.

Two layers, like structured output:
  1. redact()       replaces anything that looks like PII with a placeholder.
  2. contains_pii() a final yes/no check you can run right before text goes
                    somewhere it shouldn't carry PII (a "last gate").

Regex can't catch everything (a name written unusually, a new format), which is
why sensitive work also stays on the local model. Redaction is one layer of
defence, not the only one.
"""
import re

from fin_agent import config

# Order matters: the most specific patterns run first, so a card number isn't
# half-eaten by the phone-number pattern.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    # Card numbers already masked by the bank: ******1234 or 4500 XXXX XXXX 1234
    ("CARD", re.compile(r"[*X]{4,}[ -]?\d{4}\b|\b\d{4}(?:[ -]?[*X]{4}){2}[ -]?\d{4}\b")),
    # Full card numbers: 13-19 digits, optionally split by spaces or dashes
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    # Canadian Social Insurance Number: 123-456-789 / 123 456 789
    ("SIN", re.compile(r"\b\d{3}[ -]\d{3}[ -]\d{3}\b")),
    # North American phone numbers: 416-555-1234, (416) 555-1234, 1 800 555 1234
    ("PHONE", re.compile(r"(?:\+?1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]\d{3}[ .-]\d{4}\b")),
    # Canadian postal codes: M5V 2T6
    ("POSTAL", re.compile(r"\b[A-Z]\d[A-Z] ?\d[A-Z]\d\b")),
    # Any other long run of digits (account / reference numbers). Store numbers
    # like "#4521" are short and survive, since they help identify the merchant.
    ("ACCOUNT", re.compile(r"\b\d{7,}\b")),
]


def _name_patterns() -> list[tuple[str, re.Pattern]]:
    return [
        ("NAME", re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE))
        for name in config.REDACT_NAMES
    ]


def redact(text: str) -> str:
    """'CALL 416-555-1234 CARD ******9876' -> 'CALL [PHONE] CARD [CARD]'"""
    for label, pattern in PATTERNS + _name_patterns():
        text = pattern.sub(f"[{label}]", text)
    return text


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in PATTERNS + _name_patterns())
