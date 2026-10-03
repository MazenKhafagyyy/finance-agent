"""
Building a new expense Entry, with EXO's rules (ported from src/core/entries.ts).

Two ways in, like EXO:
    from a quick line:   build_from_text("285 mcdonalds", ...)
    from fields:         build_from_fields(amount=200, description="mcdonalds", ...)
The agent will use build_from_fields: the model extracts the fields from what
you said ("I paid 200 in food today (mcdonalds)") and this code checks them.

EXO's rules:
  - Everything is saved; missing parts (no amount, no category) make the entry
    "unresolved" instead of rejecting it.
  - A category that isn't in the Categories tab is refused.
  - Choosing a category for an UNKNOWN merchant teaches it (Merchants tab).
    Choosing a different category for a KNOWN merchant is a one-off.
  - Essential is copied from the category.

Nothing here writes to the Sheet. It returns the entry (and any merchants to
learn), so the agent can SHOW it to you and only save it after you confirm.
"""
import re
from dataclasses import dataclass, field

from fin_agent.dates import is_iso_date
from fin_agent.quickline import categorize, learn, parse_line
from fin_agent.sheet_format import CURRENCIES, Entry, Merchant, new_id
from fin_agent.sheets import Snapshot


class InputError(ValueError):
    """Bad input, with the name of the field at fault (so the model can fix it and retry)."""

    def __init__(self, field_name: str, message: str):
        super().__init__(f"{field_name}: {message}")
        self.field = field_name


@dataclass
class BuiltEntry:
    entry: Entry
    learn: list[Merchant] = field(default_factory=list)  # Merchants rows to save


def canonical_category(name: str | None, snap: Snapshot) -> str | None:
    """Turn what the model wrote into the exact category name from the Sheet.

    Accepts harmless variants: different capitalisation, and a trailing label such
    as "Fast Food (non-essential)" (a model once copied that label from the
    prompt). Anything else raises InputError listing the valid names.

    Why not just "no match = no results"? Because a question about an unknown
    category would then answer "you spent 0", a wrong number that looks right.
    Unknown input must fail LOUDLY so the model can correct itself.
    """
    if name is None or not name.strip():
        return None
    cleaned = re.sub(r"\s*\((?:non-)?essential\)\s*$", "", name.strip(), flags=re.IGNORECASE)
    for category in snap.categories:
        if category.name.lower() == cleaned.lower():
            return category.name
    names = ", ".join(c.name for c in snap.categories)
    raise InputError("category", f"Unknown category {name!r}. Choose one of: {names}")


def build_from_text(text: str, snap: Snapshot, now: tuple[str, str]) -> BuiltEntry:
    """A quick line -> an entry dated now. Unknown merchant -> no category (unresolved)."""
    parsed = parse_line(text.strip())
    if parsed is None:
        raise InputError("text", "Type something first")
    found = categorize(parsed.description, snap.merchants) if parsed.description else None
    category = found.category if found and _category_exists(found.category, snap) else ""
    return BuiltEntry(_make(snap, now, date=now[0], amount=parsed.amount, currency=parsed.currency,
                            merchant=found.merchant if found else "", category=category,
                            note=found.note if found else "", message=text.strip()))


def build_from_fields(snap: Snapshot, now: tuple[str, str], *, amount: float | None = None,
                      currency: str = "EGP", description: str = "", category: str = "",
                      date: str = "", message: str = "") -> BuiltEntry:
    """Structured fields -> an entry, with every field validated."""
    amount = check_amount(amount)
    if currency not in CURRENCIES:
        raise InputError("currency", "Currency must be EGP or CAD")
    description = description.strip()
    chosen = canonical_category(category, snap) or ""   # exact Sheet name, or InputError
    date = date or now[0]
    if not is_iso_date(date):
        raise InputError("date", "Date must look like 2026-09-25")
    if amount is None and not description and not chosen:
        raise InputError("form", "Type something first")

    found = categorize(description, snap.merchants) if description else None
    final_category = chosen or (found.category if found and _category_exists(found.category, snap) else "")
    # A category chosen for a NEW merchant is remembered; for a known merchant it's a one-off.
    teach = learn(snap.merchants, found.merchant, found.merchant, chosen) if chosen and found and not found.known else []
    entry = _make(snap, now, date=date, amount=amount, currency=currency,
                  merchant=found.merchant if found else "", category=final_category,
                  note=found.note if found else "", message=message or description)
    return BuiltEntry(entry, teach)


def check_amount(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise InputError("amount", "Amount must be a number more than 0")
    return round(float(value), 2)


def _category_exists(name: str, snap: Snapshot) -> bool:
    return any(c.name == name for c in snap.categories)


def _make(snap: Snapshot, now: tuple[str, str], **fields) -> Entry:
    essential = next((c.essential for c in snap.categories if c.name == fields["category"]), False)
    return Entry(id=new_id("tx"), time=now[1], essential=essential, **fields)
