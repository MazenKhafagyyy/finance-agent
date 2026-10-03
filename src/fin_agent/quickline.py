"""
The "quick line" parser and merchant memory, ported from EXO.

    "285 mcdonalds"  ->  amount 285, EGP, description "mcdonalds"
    "mcdonalds"      ->  McDonald's, Fast Food (found in the Merchants tab)

Ported from EXO's src/core/parser.ts and src/core/categorizer.ts, and tested
with the same examples as EXO's tests. No AI: plain rules, instant and
predictable. The agent uses this as a FAST PATH: a simple line like
"285 mcdonalds" doesn't need a 2-minute trip through the language model.
"""
import re
from dataclasses import dataclass

from fin_agent.sheet_format import Merchant, normalize

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EGP_WORDS = {"egp", "le", "l.e", "l.e.", "pound", "pounds", "جنيه", "ج"}
CAD_WORDS = {"cad", "c$"}

# One token that is an amount: 285 | 1,250 | 12.5 | 1.5k | 285egp | 285le | 20cad | c$20
# [0-9] instead of \d, because in Python \d also matches digits from other scripts.
AMOUNT_RE = re.compile(r"^(c\$)?([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.([0-9]+))?(k)?(egp|le|cad)?$", re.IGNORECASE)


@dataclass
class ParsedLine:
    amount: float | None   # None = no amount found
    currency: str          # "EGP" unless the line says CAD
    description: str       # everything except the amount and currency words


def parse_amount(token: str) -> tuple[float, str | None] | None:
    """One word -> (amount, currency or None), or None if it isn't an amount."""
    match = AMOUNT_RE.match(token)
    if not match:
        return None
    prefix, whole, decimals, thousands, suffix = match.groups()
    amount = float(whole.replace(",", "") + (f".{decimals}" if decimals else ""))
    if thousands:
        amount *= 1000          # "1.5k" -> 1500
    amount = round(amount, 2)
    if amount <= 0:
        return None
    unit = (suffix or "").lower()
    currency = "CAD" if prefix or unit == "cad" else ("EGP" if unit else None)
    return amount, currency


def parse_line(text: str) -> ParsedLine | None:
    """Split a quick line into amount, currency and description. None only for empty input."""
    # Arabic digits -> Western digits ("٢٨٥" -> "285"), then split into words.
    text = "".join(str(ARABIC_DIGITS.index(ch)) if ch in ARABIC_DIGITS else ch for ch in text)
    tokens = text.split()
    if not tokens:
        return None

    # The amount usually comes first ("285 mcdonalds"), sometimes last ("mcdonalds 285").
    # Checking those positions first keeps "cafe 88 150" from reading 88 as the amount.
    last = len(tokens) - 1
    order = list(dict.fromkeys([0, last, *range(len(tokens))]))  # unique, order kept
    for i in order:
        found = parse_amount(tokens[i])
        if found:
            amount, currency = found
            return _describe(amount, currency, [t for j, t in enumerate(tokens) if j != i])
    return _describe(None, None, tokens)


def _describe(amount: float | None, currency: str | None, tokens: list[str]) -> ParsedLine:
    words = []
    for token in tokens:
        lower = token.lower()
        if lower in CAD_WORDS:
            currency = currency or "CAD"
        elif lower in EGP_WORDS:
            currency = currency or "EGP"
        else:
            words.append(token)
    description = re.sub(r"^[\s\-:,]+|[\s\-:,]+$", "", " ".join(words))  # trim "- " etc.
    return ParsedLine(amount=amount, currency=currency or "EGP", description=description)


# ---------------------------------------------------------------------------
# Merchant memory
# ---------------------------------------------------------------------------

@dataclass
class Categorization:
    merchant: str   # display name, or the description as typed when unknown
    category: str   # "" when unknown
    note: str       # leftover words, e.g. "to work" from "uber to work"
    known: bool


def find_merchant(description: str, merchants: list[Merchant]) -> tuple[Merchant, str] | None:
    """Find the LONGEST known merchant key in the text, as whole words.
    Returns (merchant, leftover words) or None.
    Longest wins so "buffalo burger" beats "burger"."""
    text = normalize(description)
    best: tuple[Merchant, int] | None = None
    for merchant in merchants:
        if not merchant.key:
            continue
        # (?<!\w) and (?!\w): not directly preceded/followed by a letter or digit,
        # i.e. whole words only ("uber" must not match "uberx").
        match = re.search(rf"(?<!\w){re.escape(merchant.key)}(?!\w)", text)
        if match and (best is None or len(merchant.key) > len(best[0].key)):
            best = (merchant, match.start())
    if best is None:
        return None
    merchant, start = best
    leftover = f"{text[:start]} {text[start + len(merchant.key):]}"
    return merchant, " ".join(leftover.split())


def categorize(description: str, merchants: list[Merchant]) -> Categorization:
    found = find_merchant(description, merchants)
    if found:
        merchant, note = found
        return Categorization(merchant.name, merchant.category, note, known=True)
    return Categorization(description.strip(), "", "", known=False)


def learn(existing: list[Merchant], original: str, new_name: str, category: str) -> list[Merchant]:
    """The Merchants rows to save so this merchant gets `category` automatically next time.
    Saves the text as typed AND the (new) display name as keys, and updates every
    existing key of the same merchant."""
    name = new_name.strip() or original.strip()
    keys: list[str] = []
    for m in existing:
        if m.name in (original, name) and m.key not in keys:
            keys.append(m.key)
    for text in (original, name):
        key = normalize(text)
        if key and key not in keys:
            keys.append(key)
    return [Merchant(key, name, category) for key in keys]
