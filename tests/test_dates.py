from datetime import datetime, timezone

import pytest

from fin_agent.dates import is_iso_date, now_local, resolve_period

TODAY = "2026-09-30"  # a Wednesday


def test_now_is_cairo_time_not_the_laptops():
    # 22:30 UTC on Sep 30 is already 01:30 on Oct 1 in Cairo (UTC+3 in summer time).
    moment = datetime(2026, 9, 30, 22, 30, tzinfo=timezone.utc)
    assert now_local(moment) == ("2026-10-01", "01:30:00")


@pytest.mark.parametrize("name, expected", [
    ("today", ("2026-09-30", "2026-09-30")),
    ("yesterday", ("2026-09-29", "2026-09-29")),
    ("this_week", ("2026-09-26", "2026-09-30")),   # weeks start on Saturday
    ("last_week", ("2026-09-19", "2026-09-25")),
    ("this_month", ("2026-09-01", "2026-09-30")),
    ("last_month", ("2026-08-01", "2026-08-31")),
    ("last_7_days", ("2026-09-24", "2026-09-30")),
    ("this_year", ("2026-01-01", "2026-09-30")),
])
def test_periods(name, expected):
    assert resolve_period(name, TODAY) == expected


def test_last_month_across_new_year():
    assert resolve_period("last_month", "2027-01-15") == ("2026-12-01", "2026-12-31")


def test_week_starting_today():
    assert resolve_period("this_week", "2026-09-26") == ("2026-09-26", "2026-09-26")  # a Saturday


def test_unknown_period_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="this_week"):
        resolve_period("fortnight", TODAY)


def test_iso_date_check():
    assert is_iso_date("2026-02-28")
    assert not is_iso_date("2026-02-30")
    assert not is_iso_date("30/09/2026")


# --- fair comparisons ------------------------------------------------------------
from fin_agent.dates import previous_period


@pytest.mark.parametrize("name, today, expected", [
    ("this_month", "2026-09-10", ("2026-08-01", "2026-08-10")),   # same days of last month
    ("this_month", "2026-03-31", ("2026-02-01", "2026-02-28")),   # capped at February's length
    ("this_month", "2026-01-15", ("2025-12-01", "2025-12-15")),   # across the new year
    ("this_week", "2026-09-30", ("2026-09-19", "2026-09-23")),    # Sat-Wed vs the previous Sat-Wed
    ("last_week", "2026-09-30", ("2026-09-12", "2026-09-18")),
    ("last_month", "2026-09-30", ("2026-07-01", "2026-07-31")),
    ("last_7_days", "2026-09-30", ("2026-09-17", "2026-09-23")),
    ("today", "2026-09-30", ("2026-09-29", "2026-09-29")),
    ("this_year", "2026-09-30", ("2025-01-01", "2025-09-30")),
])
def test_previous_period_is_fair(name, today, expected):
    assert previous_period(name, today) == expected


# --- which DAY the user means (code decides, not the model) ------------------------
from fin_agent.dates import day_phrases, describe_day, resolve_day  # noqa: E402

FRIDAY = "2026-10-02"


@pytest.mark.parametrize("words, expected", [
    ("yesterday", "2026-10-01"),              # the real bug: the model logged 2026-09-29 for this
    ("today", "2026-10-02"),
    ("", "2026-10-02"),                       # nothing said = today
    ("last night", "2026-10-01"),
    ("day before yesterday", "2026-09-30"),
    ("the day before yesterday", "2026-09-30"),
    ("3 days ago", "2026-09-29"),
    ("1 day ago", "2026-10-01"),
    ("a week ago", "2026-09-25"),
    ("monday", "2026-09-28"),                 # the most recent Monday
    ("on Monday", "2026-09-28"),
    ("friday", "2026-10-02"),                 # today is Friday: a bare weekday can mean today
    ("last friday", "2026-09-25"),            # "last" always means before today
    ("2026-09-15", "2026-09-15"),             # an exact date is kept
])
def test_resolve_day(words, expected):
    assert resolve_day(words, FRIDAY) == expected


def test_yesterday_across_a_month_boundary():
    assert resolve_day("yesterday", "2026-10-01") == "2026-09-30"


@pytest.mark.parametrize("bad", ["2026-10-03", "tomorrowish", "32nd"])
def test_future_and_nonsense_dates_are_refused(bad):
    with pytest.raises(ValueError):
        resolve_day(bad, FRIDAY)


def test_day_phrases_found_in_a_message():
    assert day_phrases("I paid 260 for food yesterday, ordered Crepe box.") == ["yesterday"]
    assert day_phrases("add 20 mcdonalds that I ate 2 days ago") == ["2 days ago"]
    assert day_phrases("lunch the day before yesterday") == ["the day before yesterday"]   # not just "yesterday"
    assert day_phrases("260 for crepes") == []
    assert len(day_phrases("50 yesterday and 30 today")) == 2


def test_describe_day_shows_weekday_and_distance():
    assert describe_day("2026-10-01", FRIDAY) == "Thu 2026-10-01 (yesterday)"
    assert describe_day("2026-10-02", FRIDAY) == "Fri 2026-10-02 (today)"
    assert describe_day("2026-09-29", FRIDAY) == "Tue 2026-09-29 (3 days ago)"
