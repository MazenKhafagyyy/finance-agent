"""The Sheet format must match EXO exactly. These tests pin down EXO's rules."""
import re

from fin_agent.sheet_format import (
    DEFAULT_CATEGORIES, HEADERS, SEED_MERCHANTS, Entry, date_to_serial, entry_to_row,
    format_requests, header_problem, is_unresolved, new_id, normalize, row_to_category, row_to_entry,
    row_to_merchant, seed_rows, serial_to_date, to_time,
)


def entry(**overrides):
    fields = dict(id="tx0123456789ab", date="2026-09-30", time="14:05:09", amount=200.0, currency="EGP",
                  merchant="McDonald's", category="Fast Food", essential=False, note="", message="200 mcdonalds")
    return Entry(**{**fields, **overrides})


# --- dates -------------------------------------------------------------------

def test_dates_are_sheets_serial_numbers():
    assert date_to_serial("1900-01-01") == 2           # Google Sheets' own convention
    assert serial_to_date(date_to_serial("2026-09-30")) == "2026-09-30"
    assert serial_to_date(46295.75) == serial_to_date(46295)  # a time fraction is ignored


def test_time_from_fraction_or_text():
    assert to_time(0.5) == "12:00:00"
    assert to_time("9:05") == "09:05:00"
    assert to_time("garbage") == "00:00:00"


# --- rows <-> entries ----------------------------------------------------------

def test_entry_row_matches_exo_layout():
    row = entry_to_row(entry())
    assert len(row) == len(HEADERS["Transactions"]) == 10
    assert row[0] == "tx0123456789ab"
    assert isinstance(row[1], int)          # the date is a NUMBER, like EXO writes it
    assert row[7] == "No"                   # Essential as Yes/No


def test_round_trip():
    original = entry()
    assert row_to_entry(entry_to_row(original)) == original


def test_unknown_amount_is_an_empty_cell_and_unresolved():
    e = entry(amount=None)
    assert entry_to_row(e)[3] == ""
    back = row_to_entry(entry_to_row(e))
    assert back.amount is None and is_unresolved(back)


def test_essential_is_empty_without_a_category():
    assert entry_to_row(entry(category="", essential=False))[7] == ""


def test_invalid_rows_are_skipped():
    good = entry_to_row(entry())
    assert row_to_entry([]) is None                                   # blank row
    assert row_to_entry(["", *good[1:]]) is None                      # no ID
    assert row_to_entry([good[0], "not a date", *good[2:]]) is None   # bad date
    assert row_to_entry([*good[:3], "abc", *good[4:]]) is None        # text amount
    assert row_to_entry([*good[:3], -5, *good[4:]]) is None           # negative amount
    assert row_to_entry([*good[:4], "USD", *good[5:]]) is None        # unsupported currency


def test_short_rows_and_text_amounts_with_commas():
    e = row_to_entry(["tx1", "2026-09-30", "10:00", "1,250"])
    assert e.amount == 1250.0 and e.currency == "EGP" and e.category == ""


# --- categories, merchants, seeds ---------------------------------------------

def test_seeds_match_exo():
    assert len(DEFAULT_CATEGORIES) == 16
    assert len(SEED_MERCHANTS) == 38
    assert seed_rows("Categories")[0] == ["Fast Food", "No", "", "Restaurants, takeaway, food delivery (Talabat, Elmenus)"]
    assert seed_rows("Transactions") == []


def test_category_and_merchant_rows():
    assert row_to_category(["Rent", "Yes", "8000", "Housing"]).budget == 8000
    assert row_to_category(["Rent", "Yes", "", ""]).budget is None
    assert row_to_merchant(["McDonald's", "McDonald's", "Fast Food"]).key == "mcdonalds"
    assert row_to_merchant(["kfc", "KFC", ""]) is None  # no category -> skipped


def test_normalize_like_exo():
    assert normalize("McDonald's!") == "mcdonalds"
    assert normalize("  Buffalo   BURGER ") == "buffalo burger"
    assert normalize("Mo’men") == "momen"  # curly apostrophe


def test_new_id_format():
    assert re.fullmatch(r"tx[0-9a-f]{12}", new_id())
    assert new_id() != new_id()


def test_header_check_catches_the_old_transactions_format():
    old_exo = ["ID", "Date", "Time", "Amount (EGP)", "Merchant", "Category", "Essential", "Note", "Message", ""]
    assert "expected" in header_problem("Transactions", old_exo)
    assert header_problem("Transactions", HEADERS["Transactions"]) is None
    assert header_problem("Transactions", [*HEADERS["Transactions"], "", ""]) is None  # empty trailing cells are fine


def test_old_category_header_names_are_accepted():
    assert header_problem("Categories", ["Category", "Essential", "Monthly budget (EGP)", "Hint for EXO"]) is None


def test_format_requests():
    kinds = [next(iter(r)) for r in format_requests("Transactions", 7)]
    assert kinds == ["repeatCell", "repeatCell", "addConditionalFormatRule"]
    assert len(format_requests("Merchants", 7)) == 1  # bold header only
