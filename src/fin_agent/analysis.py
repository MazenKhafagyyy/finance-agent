"""
ANALYSIS: the "what does it mean?" layer on top of the Phase 2 tools.

Same principle as everywhere else: CODE computes every number, the MODEL only
explains. Each function takes a Snapshot (everything read from the Sheet) and
returns a plain dict the model turns into a few sentences.

    unessential_report   non-essential spending, compared with a FAIR previous period
    budget_status        spending vs the Categories tab's "Monthly budget", with a projection
    unusual_spending     unusually big charges, possible double charges, new repeating charges
    weekly_review        this week vs the same days last week, everything in one summary

Honesty rules shared by all of them: every result states its period and
currency, and counts the rows it couldn't use (no amount or no category), so an
answer can say "3 entries have no amount yet" instead of silently undercounting.
"""
from collections import defaultdict
from datetime import date
from statistics import median

from fin_agent.dates import days_in_month, previous_period, resolve_period
from fin_agent.sheet_format import Entry
from fin_agent.sheets import Snapshot
from fin_agent.tools import UNRESOLVED, _cents, _money, _select

WARN_AT = 0.80          # budget warning at 80% used
BIG_MULTIPLE = 3        # "unusually big" = more than 3x the merchant's typical amount
MIN_HISTORY = 3         # ...and only if we've seen that merchant at least 3 times before
REPEAT_TOLERANCE = 0.05  # "same amount" for repeating charges: within 5%


def _by(entries: list[Entry], key) -> dict[str, int]:
    """Sum amounts (in cents) per group. Entries without an amount are skipped."""
    totals: dict[str, int] = defaultdict(int)
    for e in entries:
        if e.amount is not None:
            totals[key(e)] += _cents(e.amount)
    return totals


def _change(now: int, before: int) -> dict:
    return {"now": _money(now), "before": _money(before), "change": _money(now - before),
            # None when there was nothing before: "+inf %" means nothing useful
            "change_pct": round(100 * (now - before) / before, 1) if before else None}


def _missing(entries: list[Entry]) -> dict:
    return {"no_amount": sum(e.amount is None for e in entries),
            "no_category": sum(e.amount is not None and not e.category for e in entries)}


# ---------------------------------------------------------------------------
# 1. "What have I been spending so much on that's unessential?"
# ---------------------------------------------------------------------------

def unessential_report(snap: Snapshot, period: str, today: str, currency: str = "EGP") -> dict:
    start, end = resolve_period(period, today)
    prev_start, prev_end = previous_period(period, today)
    now_all = _select(snap, start, end, currency)
    now_ne = _select(snap, start, end, currency, essential=False)
    before_ne = _select(snap, prev_start, prev_end, currency, essential=False)

    now_cat, before_cat = _by(now_ne, lambda e: e.category), _by(before_ne, lambda e: e.category)
    categories = sorted(
        ({"category": c, **_change(now_cat.get(c, 0), before_cat.get(c, 0))} for c in set(now_cat) | set(before_cat)),
        key=lambda row: row["now"], reverse=True)
    merchants = sorted(_by(now_ne, lambda e: e.merchant or e.category).items(), key=lambda kv: kv[1], reverse=True)

    total_now, total_all = sum(now_cat.values()), sum(_by(now_all, lambda e: "all").values())
    return {
        "period": {"start": start, "end": end}, "compared_with": {"start": prev_start, "end": prev_end},
        "currency": currency,
        "non_essential_total": _change(total_now, sum(before_cat.values())),
        "share_of_all_spending_pct": round(100 * total_now / total_all, 1) if total_all else None,
        "by_category": categories,
        "top_merchants": [{"merchant": m, "total": _money(c)} for m, c in merchants[:5]],
        # The categories that grew the most, i.e. where to look first.
        "grew_most": [row["category"] for row in categories if row["change"] > 0][:3],
        "not_counted": _missing(now_all),
    }


# ---------------------------------------------------------------------------
# 2. Budgets (the "Monthly budget" column of the Categories tab)
# ---------------------------------------------------------------------------

def budget_status(snap: Snapshot, today: str, currency: str = "EGP") -> dict:
    """This month's spending per budgeted category, with a month-end projection.
    Budgets are EGP amounts (EXO's column is "Monthly budget (EGP)")."""
    d = date.fromisoformat(today)
    start, end = d.replace(day=1).isoformat(), today
    month_days = days_in_month(d.year, d.month)
    spent = _by(_select(snap, start, end, currency), lambda e: e.category)

    rows = []
    for c in snap.categories:
        if not c.budget:
            continue
        used, budget = spent.get(c.name, 0), _cents(c.budget)
        # Straight-line projection: at this pace, where does the month end up?
        projected = round(used / d.day * month_days)
        ratio = used / budget
        rows.append({
            "category": c.name, "spent": _money(used), "budget": _money(budget),
            "used_pct": round(100 * ratio, 1),
            "status": "over" if ratio >= 1 else "warning" if ratio >= WARN_AT else "ok",
            "projected_month_end": _money(projected),
            "projected_over": projected > budget,
        })
    rows.sort(key=lambda row: row["used_pct"], reverse=True)
    return {
        "month": today[:7], "day": d.day, "days_in_month": month_days, "currency": currency,
        "budgets": rows,
        "note": None if rows else "No budgets set. Add amounts in the 'Monthly budget' column of the Categories tab.",
    }


# ---------------------------------------------------------------------------
# 3. Unusual spending
# ---------------------------------------------------------------------------

def unusual_spending(snap: Snapshot, today: str, currency: str = "EGP", recent_days: int = 30) -> dict:
    """Three simple, explainable rules (no AI):
        big        a recent charge > BIG_MULTIPLE x the merchant's median, with >= MIN_HISTORY earlier charges
        duplicate  same merchant, same amount, same day, more than once
        repeating  a merchant charged about the same amount once a month in 2+
                   months, first seen recently (a possible new subscription)
    """
    recent_start = date.fromordinal(date.fromisoformat(today).toordinal() - recent_days + 1).isoformat()
    entries = sorted((e for e in snap.entries if e.currency == currency and e.amount is not None and e.merchant
                      and e.date <= today), key=lambda e: (e.date, e.time))

    history: dict[str, list[Entry]] = defaultdict(list)
    for e in entries:
        history[e.merchant].append(e)

    big = []
    for merchant, items in history.items():
        for i, e in enumerate(items):
            earlier = [x.amount for x in items[:i]]
            if e.date >= recent_start and len(earlier) >= MIN_HISTORY and e.amount > BIG_MULTIPLE * median(earlier):
                big.append({"date": e.date, "merchant": merchant, "amount": e.amount,
                            "usual": median(earlier), "times_usual": round(e.amount / median(earlier), 1)})

    same_day: dict[tuple, list[Entry]] = defaultdict(list)
    for e in entries:
        if e.date >= recent_start:
            same_day[(e.merchant, e.date, _cents(e.amount))].append(e)
    duplicates = [{"date": d, "merchant": m, "amount": _money(c), "times": len(items)}
                  for (m, d, c), items in same_day.items() if len(items) > 1]

    repeating = []
    for merchant, items in history.items():
        months = {e.date[:7] for e in items}
        amounts = [e.amount for e in items]
        similar = max(amounts) <= min(amounts) * (1 + REPEAT_TOLERANCE)
        # One charge per month is the subscription signature. A coffee bought at the
        # same price several times a month has the same amount but many charges per month.
        once_a_month = len(items) == len(months)
        if (len(months) >= 2 and similar and once_a_month
                and items[0].date >= _months_ago(today, 3) and items[-1].date >= recent_start):
            repeating.append({"merchant": merchant, "amount": items[-1].amount, "months": sorted(months),
                              "category": items[-1].category or None})

    return {"window": {"start": recent_start, "end": today}, "currency": currency,
            "unusually_big": big, "possible_duplicates": duplicates, "new_repeating_charges": repeating,
            "rules": f"big = over {BIG_MULTIPLE}x the merchant's usual amount (needs {MIN_HISTORY}+ earlier "
                     f"charges); duplicate = same merchant, amount and day; repeating = same amount in 2+ months, "
                     f"first seen in the last 3 months"}


def _months_ago(today: str, n: int) -> str:
    d = date.fromisoformat(today)
    year, month = d.year, d.month - n
    while month < 1:
        year, month = year - 1, month + 12
    return date(year, month, 1).isoformat()


# ---------------------------------------------------------------------------
# 4. Weekly review
# ---------------------------------------------------------------------------

def weekly_review(snap: Snapshot, today: str, currency: str = "EGP") -> dict:
    """Everything worth knowing about this week, in one call (one tool call = one
    model round trip, which matters on a CPU)."""
    start, end = resolve_period("this_week", today)
    prev_start, prev_end = previous_period("this_week", today)
    now, before = _select(snap, start, end, currency), _select(snap, prev_start, prev_end, currency)

    def split(entries):
        return {"essential": _money(sum(_by([e for e in entries if e.category and e.essential], lambda e: "x").values())),
                "non_essential": _money(sum(_by([e for e in entries if e.category and not e.essential], lambda e: "x").values())),
                "uncategorized": _money(sum(_by([e for e in entries if not e.category], lambda e: "x").values()))}

    by_cat = sorted(_by(now, lambda e: e.category or UNRESOLVED).items(), key=lambda kv: kv[1], reverse=True)
    budgets = budget_status(snap, today, currency)["budgets"]
    unusual = unusual_spending(snap, today, currency, recent_days=7)
    return {
        "week": {"start": start, "end": end}, "compared_with": {"start": prev_start, "end": prev_end},
        "currency": currency,
        "total": _change(sum(_by(now, lambda e: "x").values()), sum(_by(before, lambda e: "x").values())),
        "split": split(now),
        "top_categories": [{"category": c, "total": _money(v)} for c, v in by_cat[:3]],
        "budget_alerts": [b for b in budgets if b["status"] != "ok" or b["projected_over"]],
        "unusual": {k: unusual[k] for k in ("unusually_big", "possible_duplicates", "new_repeating_charges")},
        "not_counted": _missing(now),
    }
