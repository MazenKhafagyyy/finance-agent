"""The tools, tested on the fake Sheet with EXO's seed categories and merchants.
Each test mirrors something you'd say to the assistant."""
import pytest

from fin_agent.dates import resolve_period
from fin_agent.entries import InputError, build_from_text
from fin_agent.sheets import MemoryStore
from fin_agent.tools import (
    UNRESOLVED, compare_periods, describe_entry, list_transactions, prepare_expense,
    save_expense, spending_breakdown, spending_total,
)

NOW = ("2026-09-30", "14:00:00")  # Wednesday; this week = Sat 26 .. Wed 30


def add(store, text, date):
    """Log a quick line on a given date (a helper to build test data)."""
    snap = store.load()
    built = build_from_text(text, snap, (date, "12:00:00"))
    save_expense(store, built)


@pytest.fixture
def store():
    s = MemoryStore()
    add(s, "200 mcdonalds", "2026-09-30")      # Fast Food, non-essential, this week
    add(s, "150 starbucks", "2026-09-29")      # Coffee, non-essential, this week
    add(s, "900 carrefour", "2026-09-27")      # Groceries, essential, this week
    add(s, "120 uber", "2026-09-20")           # Transport, essential, last week
    add(s, "300 kfc", "2026-08-15")            # Fast Food, last month
    add(s, "20 cad tims", "2026-09-30")        # CAD: never mixed into EGP totals
    add(s, "lunch with ali", "2026-09-28")     # no amount: unresolved
    add(s, "80 some new kiosk", "2026-09-28")  # unknown merchant: no category
    return s


# --- "I paid 200 in food today (mcdonalds)" ----------------------------------

def test_prepare_then_save_an_expense(store):
    snap = store.load()
    built = prepare_expense(snap, NOW, amount=200, description="mcdonalds", message="I paid 200 in food today (mcdonalds)")
    assert describe_entry(built) == "200 EGP · McDonald's → Fast Food (non-essential) · 2026-09-30"
    before = len(store.load().entries)
    save_expense(store, built)
    assert len(store.load().entries) == before + 1


def test_preparing_does_not_save(store):
    before = len(store.load().entries)
    prepare_expense(store.load(), NOW, amount=50, description="kfc")
    assert len(store.load().entries) == before


def test_unknown_merchant_with_chosen_category_is_learned(store):
    snap = store.load()
    built = prepare_expense(snap, NOW, amount=45, description="abu tarek", category="Fast Food")
    assert [m.key for m in built.learn] == ["abu tarek"]
    save_expense(store, built)
    assert any(m.key == "abu tarek" for m in store.load().merchants)


def test_unknown_category_is_refused_with_valid_names(store):
    with pytest.raises(InputError, match="Choose one of: Fast Food"):
        prepare_expense(store.load(), NOW, amount=45, description="kfc", category="Food")


def test_bad_amount_is_refused(store):
    with pytest.raises(InputError, match="amount"):
        prepare_expense(store.load(), NOW, amount=-5, description="kfc")


# --- "Tell me how much I spent this week" -------------------------------------

def test_spending_this_week(store):
    start, end = resolve_period("this_week", NOW[0])
    result = spending_total(store.load(), start, end)
    assert result["total"] == 200 + 150 + 900 + 80   # EGP only; the CAD coffee isn't added in
    assert result["count"] == 4
    assert result["missing_amounts"] == 1            # "lunch with ali" has no amount


def test_cad_is_separate(store):
    start, end = resolve_period("this_week", NOW[0])
    assert spending_total(store.load(), start, end, "CAD")["total"] == 20


def test_totals_are_exact_to_the_cent():
    s = MemoryStore()
    for _ in range(3):
        add(s, "0.1 kfc", "2026-09-30")
    assert spending_total(s.load(), "2026-09-30", "2026-09-30")["total"] == 0.3  # not 0.30000000000000004


# --- "What have I been spending on that's unessential?" -------------------------

def test_non_essential_breakdown(store):
    result = spending_breakdown(store.load(), "2026-09-01", "2026-09-30", essential=False)
    assert [(g["name"], g["total"]) for g in result["groups"]] == [("Fast Food", 200), ("Coffee", 150)]
    assert result["groups"][0]["share_pct"] == 57.1


def test_breakdown_shows_unresolved_group(store):
    names = [g["name"] for g in spending_breakdown(store.load(), "2026-09-01", "2026-09-30")["groups"]]
    assert names == ["Groceries", "Fast Food", "Coffee", "Transport", UNRESOLVED]


def test_breakdown_by_merchant_top(store):
    result = spending_breakdown(store.load(), "2026-09-01", "2026-09-30", by="merchant", top=2)
    assert [g["name"] for g in result["groups"]] == ["Carrefour", "McDonald's"]


# --- comparisons and listings ---------------------------------------------------

def test_compare_this_month_with_last(store):
    result = compare_periods(store.load(), "2026-09-01", "2026-09-30", "2026-08-01", "2026-08-31", category="Fast Food")
    assert (result["period_a"]["total"], result["period_b"]["total"]) == (200, 300)
    assert result["difference"] == -100
    assert result["change_pct"] == -33.3


def test_compare_against_empty_period_has_no_percentage(store):
    result = compare_periods(store.load(), "2026-09-01", "2026-09-30", "2025-01-01", "2025-01-31")
    assert result["change_pct"] is None


def test_list_transactions_newest_first(store):
    result = list_transactions(store.load(), "2026-09-01", "2026-09-30", "EGP", limit=2)
    assert result["matches"] == 6  # Sept EGP rows (KFC is August, Tims is CAD)
    assert [t["merchant"] for t in result["transactions"]] == ["McDonald's", "Starbucks"]


# --- category names: accept harmless variants, reject everything else loudly -------------

def test_category_label_and_case_variants_resolve_to_the_real_name(store):
    # gemma4 once sent "Fast Food (non-essential)" (copied from the prompt) and got "you spent 0".
    for name in ["Fast Food (non-essential)", "fast food", "  Fast Food  "]:
        assert spending_total(store.load(), "2026-09-01", "2026-09-30", category=name)["total"] == 200


def test_unknown_category_raises_instead_of_answering_zero(store):
    with pytest.raises(InputError, match="Unknown category 'Food'. Choose one of: Fast Food"):
        spending_total(store.load(), "2026-09-01", "2026-09-30", category="Food")
    with pytest.raises(InputError):
        compare_periods(store.load(), "2026-09-01", "2026-09-30", "2026-08-01", "2026-08-31", category="Takeaway")
    with pytest.raises(InputError):
        list_transactions(store.load(), "2026-09-01", "2026-09-30", category="Takeaway")


def test_expense_with_a_labelled_category_is_accepted_first_time(store):
    built = prepare_expense(store.load(), NOW, amount=45, description="abu tarek", category="Fast Food (non-essential)")
    assert built.entry.category == "Fast Food"
