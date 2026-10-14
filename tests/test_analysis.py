"""Analysis tests on the fake Sheet. Today is Wed 2026-09-30; this week = Sat 26 .. Wed 30,
compared with Sat 19 .. Wed 23."""
import pytest

from fin_agent.analysis import budget_status, unessential_report, unusual_spending, weekly_review
from fin_agent.entries import build_from_text
from fin_agent.sheets import MemoryStore
from fin_agent.tools import save_expense

TODAY = "2026-09-30"


def add(store, text, day):
    save_expense(store, build_from_text(text, store.load(), (day, "12:00:00")))


def set_budget(store, category, amount):
    for row in store.tabs["Categories"][1:]:
        if row[0] == category:
            row[2] = amount


@pytest.fixture
def store():
    s = MemoryStore()
    s.setup()
    # September (this month)
    add(s, "200 mcdonalds", "2026-09-30")    # Fast Food (non-essential), this week
    add(s, "150 starbucks", "2026-09-29")    # Coffee (non-essential), this week
    add(s, "900 carrefour", "2026-09-27")    # Groceries (essential), this week
    add(s, "350 talabat", "2026-09-22")      # Fast Food, last week (Tue: inside Sat 19-Wed 23)
    add(s, "120 uber", "2026-09-20")         # Transport (essential), last week
    add(s, "lunch with ali", "2026-09-28")   # no amount
    # August (the fair comparison for this_month on the 30th is Aug 1-30)
    add(s, "300 kfc", "2026-08-15")          # Fast Food
    add(s, "100 costa", "2026-08-31")        # Coffee on Aug 31: OUTSIDE Aug 1-30, must not count
    return s


# --- unessential ------------------------------------------------------------------------

def test_unessential_report_compares_fairly(store):
    r = unessential_report(store.load(), "this_month", TODAY)
    assert r["compared_with"] == {"start": "2026-08-01", "end": "2026-08-30"}
    assert r["non_essential_total"] == {"now": 700.0, "before": 300.0, "change": 400.0, "change_pct": 133.3}
    rows = {row["category"]: row for row in r["by_category"]}
    assert rows["Fast Food"]["now"] == 550 and rows["Fast Food"]["before"] == 300
    assert rows["Coffee"]["before"] == 0            # the Aug 31 Costa is outside the fair window
    assert r["grew_most"] == ["Fast Food", "Coffee"]
    assert r["share_of_all_spending_pct"] == round(100 * 700 / 1720, 1)   # 700 of 1,720 EGP spent
    assert r["not_counted"]["no_amount"] == 1


def test_unessential_top_merchants(store):
    r = unessential_report(store.load(), "this_month", TODAY)
    assert [m["merchant"] for m in r["top_merchants"]] == ["Talabat", "McDonald's", "Starbucks"]


# --- budgets -----------------------------------------------------------------------------

def test_budget_status_levels_and_projection(store):
    set_budget(store, "Fast Food", 600)     # 550 of 600 = 91.7% -> warning
    set_budget(store, "Coffee", 100)        # 150 of 100 -> over
    set_budget(store, "Groceries", 3000)    # 900 of 3000 -> ok
    r = budget_status(store.load(), TODAY)
    rows = {row["category"]: row for row in r["budgets"]}
    assert rows["Fast Food"]["status"] == "warning" and rows["Fast Food"]["used_pct"] == 91.7
    assert rows["Coffee"]["status"] == "over"
    assert rows["Groceries"]["status"] == "ok"
    assert [row["category"] for row in r["budgets"]] == ["Coffee", "Fast Food", "Groceries"]  # worst first


def test_budget_projection_mid_month(store):
    set_budget(store, "Transport", 200)
    r = budget_status(store.load(), "2026-09-20")    # 120 spent by the 20th of a 30-day month
    row = r["budgets"][0]
    assert row["projected_month_end"] == 180.0 and not row["projected_over"]


def test_no_budgets_explains_how_to_set_them(store):
    r = budget_status(store.load(), TODAY)
    assert r["budgets"] == [] and "Monthly budget" in r["note"]


# --- unusual spending ----------------------------------------------------------------------

def test_unusually_big_charge():
    s = MemoryStore()
    for day in ("2026-06-10", "2026-07-10", "2026-08-10"):
        add(s, "40 uber", day)
    add(s, "250 uber", "2026-09-25")                 # 6.25x the usual 40
    r = unusual_spending(s.load(), TODAY)
    assert [(b["merchant"], b["times_usual"]) for b in r["unusually_big"]] == [("Uber", 6.2)]


def test_big_needs_enough_history():
    s = MemoryStore()
    add(s, "40 uber", "2026-09-01")
    add(s, "250 uber", "2026-09-25")                 # only 1 earlier charge: not enough to judge
    assert unusual_spending(s.load(), TODAY)["unusually_big"] == []


def test_possible_double_charge():
    s = MemoryStore()
    add(s, "85 netflix", "2026-09-15")
    add(s, "85 netflix", "2026-09-15")
    r = unusual_spending(s.load(), TODAY)
    assert r["possible_duplicates"] == [{"date": "2026-09-15", "merchant": "Netflix", "amount": 85.0, "times": 2}]


def test_new_repeating_charge_but_not_coffee():
    s = MemoryStore()
    add(s, "99 spotify", "2026-08-05")
    add(s, "99 spotify", "2026-09-05")               # once a month, same amount: subscription-like
    for day in ("2026-08-02", "2026-08-20", "2026-09-03", "2026-09-21"):
        add(s, "45 starbucks", day)                  # same price, but several times a month: just coffee
    r = unusual_spending(s.load(), TODAY)
    assert [x["merchant"] for x in r["new_repeating_charges"]] == ["Spotify"]


# --- weekly review ---------------------------------------------------------------------------

def test_weekly_review(store):
    set_budget(store, "Coffee", 100)
    r = weekly_review(store.load(), TODAY)
    assert r["week"] == {"start": "2026-09-26", "end": "2026-09-30"}
    assert r["compared_with"] == {"start": "2026-09-19", "end": "2026-09-23"}
    assert r["total"]["now"] == 1250 and r["total"]["before"] == 470          # 350 Talabat + 120 Uber
    assert r["split"] == {"essential": 900.0, "non_essential": 350.0, "uncategorized": 0.0}
    assert r["top_categories"][0] == {"category": "Groceries", "total": 900.0}
    assert [b["category"] for b in r["budget_alerts"]] == ["Coffee"]
    assert r["not_counted"]["no_amount"] == 1
