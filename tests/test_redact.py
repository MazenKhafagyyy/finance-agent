"""Redaction tests. All values are fake."""
import pytest

from fin_agent import config
from fin_agent.redact import contains_pii, redact


@pytest.fixture(autouse=True)
def fake_names(monkeypatch):
    monkeypatch.setattr(config, "REDACT_NAMES", ["Jane", "Doe"])


@pytest.mark.parametrize("text, expected", [
    ("NETFLIX.COM 866-579-7172 ON", "NETFLIX.COM [PHONE] ON"),
    ("CALL (416) 555-1234 NOW", "CALL [PHONE] NOW"),
    ("PAYPAL ******4821 ON", "PAYPAL [CARD] ON"),
    ("Card number 4500 XXXX XXXX 1234", "Card number [CARD]"),
    ("CARD 4500 1234 5678 9012", "CARD [CARD]"),
    ("SIN 046-454-286", "SIN [SIN]"),
    ("SHIP TO M5V 2T6", "SHIP TO [POSTAL]"),
    ("REF 12345678", "REF [ACCOUNT]"),
    ("MAIL jane.doe@example.com", "MAIL [EMAIL]"),
    ("E-TRANSFER TO JANE DOE", "E-TRANSFER TO [NAME] [NAME]"),
])
def test_redacts_pii(text, expected):
    assert redact(text) == expected


@pytest.mark.parametrize("text", [
    "TIM HORTONS #4521 TORONTO ON",   # store numbers are short: kept
    "UBER* EATS 12.50",               # amounts are kept
    "AMAZON.CA AMAZON.CA ON",
])
def test_keeps_normal_merchant_text(text):
    assert redact(text) == text
    assert not contains_pii(text)


def test_contains_pii_is_false_after_redaction():
    text = "PAYPAL ******4821 CALL 416-555-1234 JANE"
    assert contains_pii(text)
    assert not contains_pii(redact(text))
