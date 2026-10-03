"""
The assistant's TOOLS: plain Python functions the agent will call (Phase 3).

The model never does arithmetic. It picks a tool and fills in arguments like
period="this_week"; these functions do the exact work and return plain
dictionaries that the model turns into a sentence.

Two groups:
  READ tools   answer questions. They take a Snapshot (everything loaded from
               the Sheet) and never change anything.
  WRITE tools  prepare_expense() builds a new row WITHOUT saving it, so the
               agent can show it to you first; save_expense() writes it only
               after you confirm.

Design rules:
  - Money is summed in whole CENTS (integers) and converted back at the end,
    so float rounding errors can't creep in (0.1 + 0.2 != 0.3 in floats).
  - Every result says which period and currency it covers, and how many rows
    were left out because they have no amount (unresolved), so the answer can
    be honest about gaps instead of silently undercounting.
  - One currency at a time: EGP and CAD are never added together.
"""
from collections import defaultdict

from fin_agent.dates import describe_day
from fin_agent.entries import BuiltEntry, build_from_fields, canonical_category
from fin_agent.sheet_format import Entry
from fin_agent.sheets import Snapshot

UNRESOLVED = "Unresolved"  # the group name for rows with an amount but no category (like EXO's grey slice)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cents(amount: float) -> int:
    return round(amount * 100)


def _money(cents: int) -> float:
    return cents / 100


def _select(snap: Snapshot, start: str, end: str, currency: str, *, category: str | None = None,
            essential: bool | None = None, merchant: str | None = None) -> list[Entry]:
    """Entries in [start, end] (inclusive) in one currency, with optional filters.
    ISO date strings compare correctly as text: '2026-09-05' < '2026-09-30'."""
    chosen = []
    for e in snap.entries:
        if e.currency != currency or not (start <= e.date <= end):
            continue
        if category is not None and e.category != category:
            continue
        # essential=None means "don't filter". Rows without a category have no
        # known essential status, so they're excluded when filtering on it.
        if essential is not None and (e.category == "" or e.essential != essential):
            continue
        if merchant is not None and merchant.lower() not in e.merchant.lower():
            continue
        chosen.append(e)
    return chosen


def _period(start: str, end: str, currency: str) -> dict:
    return {"start": start, "end": end, "currency": currency}


# ---------------------------------------------------------------------------
# READ tools
# ---------------------------------------------------------------------------

def spending_total(snap: Snapshot, start: str, end: str, currency: str = "EGP", *,
                   category: str | None = None, essential: bool | None = None) -> dict:
    """Total spent in a period. Example answer to "how much did I spend this week?"."""
    # An unknown category raises here instead of quietly matching nothing (= "you spent 0").
    category = canonical_category(category, snap)
    rows = _select(snap, start, end, currency, category=category, essential=essential)
    with_amount = [e for e in rows if e.amount is not None]
    return {
        **_period(start, end, currency),
        "filters": {"category": category, "essential": essential},
        "total": _money(sum(_cents(e.amount) for e in with_amount)),
        "count": len(with_amount),
        "missing_amounts": len(rows) - len(with_amount),  # rows not counted because the amount is unknown
    }


def spending_breakdown(snap: Snapshot, start: str, end: str, currency: str = "EGP", *,
                       by: str = "category", essential: bool | None = None, top: int | None = None) -> dict:
    """Spending grouped by category or merchant, biggest first, with each group's
    share of the total. Example: "what's taking most of my money this month?"."""
    if by not in ("category", "merchant"):
        raise ValueError("by must be 'category' or 'merchant'")
    groups: dict[str, list[Entry]] = defaultdict(list)
    for e in _select(snap, start, end, currency, essential=essential):
        if e.amount is None:
            continue
        name = (e.category or UNRESOLVED) if by == "category" else (e.merchant or e.category or UNRESOLVED)
        groups[name].append(e)

    total = sum(_cents(e.amount) for rows in groups.values() for e in rows)
    items = []
    for name, rows in groups.items():
        cents = sum(_cents(e.amount) for e in rows)
        items.append({
            "name": name,
            "total": _money(cents),
            "count": len(rows),
            "share_pct": round(100 * cents / total, 1) if total else 0.0,
            # Only meaningful for categories: is this group essential?
            **({"essential": rows[0].essential if name != UNRESOLVED else None} if by == "category" else {}),
        })
    items.sort(key=lambda item: item["total"], reverse=True)
    return {**_period(start, end, currency), "by": by, "essential": essential,
            "total": _money(total), "groups": items[:top] if top else items}


def compare_periods(snap: Snapshot, a_start: str, a_end: str, b_start: str, b_end: str,
                    currency: str = "EGP", *, category: str | None = None,
                    essential: bool | None = None) -> dict:
    """Period A vs period B, e.g. this month vs last month."""
    a = spending_total(snap, a_start, a_end, currency, category=category, essential=essential)
    b = spending_total(snap, b_start, b_end, currency, category=category, essential=essential)
    difference = _money(_cents(a["total"]) - _cents(b["total"]))
    return {
        "currency": currency,
        "period_a": a,
        "period_b": b,
        "difference": difference,  # positive = spent more in A
        # None when B is zero: a percentage change from nothing is meaningless.
        "change_pct": round(100 * difference / b["total"], 1) if b["total"] else None,
    }


def list_transactions(snap: Snapshot, start: str, end: str, currency: str | None = None, *,
                      merchant: str | None = None, category: str | None = None,
                      limit: int = 20) -> dict:
    """The individual rows, newest first. For "what did I buy at Carrefour last week?"."""
    category = canonical_category(category, snap)   # fail loudly on an unknown category
    currencies = [currency] if currency else ["EGP", "CAD"]
    rows = [e for cur in currencies for e in _select(snap, start, end, cur, merchant=merchant, category=category)]
    rows.sort(key=lambda e: (e.date, e.time), reverse=True)
    return {
        "start": start, "end": end, "matches": len(rows),
        "transactions": [
            {"id": e.id, "date": e.date, "amount": e.amount, "currency": e.currency,
             "merchant": e.merchant, "category": e.category or None, "note": e.note}
            for e in rows[:limit]
        ],
    }


def list_categories(snap: Snapshot) -> list[dict]:
    """The categories with their Essential flag, so the model knows valid names."""
    return [{"name": c.name, "essential": c.essential, "hint": c.hint} for c in snap.categories]


# ---------------------------------------------------------------------------
# WRITE tools: prepare first, save only after confirmation
# ---------------------------------------------------------------------------

def prepare_expense(snap: Snapshot, now: tuple[str, str], *, amount: float | None, description: str,
                    currency: str = "EGP", category: str = "", date: str = "", message: str = "") -> BuiltEntry:
    """Build (but DON'T save) a new expense. Raises InputError on bad input."""
    return build_from_fields(snap, now, amount=amount, currency=currency, description=description,
                             category=category, date=date, message=message)


def describe_entry(built: BuiltEntry, today: str | None = None) -> str:
    """A one-line summary to show before saving, e.g.
    '200 EGP · McDonald's → Fast Food (non-essential) · Wed 2026-09-30 (today)'.

    With `today`, the date is shown with its weekday and how long ago it was, so
    a wrongly dated entry is obvious BEFORE you confirm it."""
    e = built.entry
    amount = f"{e.amount:g} {e.currency}" if e.amount is not None else f"(no amount) {e.currency}"
    who = e.merchant or "(no merchant)"
    what = f"{e.category} ({'essential' if e.essential else 'non-essential'})" if e.category else "(no category: unresolved)"
    learned = f" · will remember {who} → {e.category}" if built.learn else ""
    when = describe_day(e.date, today) if today else e.date
    return f"{amount} · {who} → {what} · {when}{learned}"


def save_expense(store, built: BuiltEntry) -> None:
    """Write the entry (and any learned merchants) to the Sheet.
    Merchants are saved FIRST, like EXO: if the entry write then fails and is
    retried, the merchant isn't learned twice and the entry isn't duplicated."""
    if built.learn:
        store.save_merchants(built.learn)
    store.add_entry(built.entry)
