"""
Dates and periods, always in EXO's time zone (Africa/Cairo by default).

Why a module just for dates? "Today" depends on WHERE you are: at 1 a.m. in
Cairo it's still yesterday in Toronto. The Sheet stores Cairo dates (EXO's
rule), so every "today", "this week" or "last month" must be computed in Cairo
time, not in whatever time zone this laptop is set to.

Dates are passed around as plain "YYYY-MM-DD" strings, like in the Sheet.
Every period is INCLUSIVE on both ends: ("2026-09-01", "2026-09-30") includes
both the 1st and the 30th.
"""
import calendar
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from fin_agent import config

# Which day a week starts on (Python: Monday=0 ... Saturday=5, Sunday=6).
# Saturday matches the usual Egyptian calendar. Change to 0 for Monday weeks.
WEEK_STARTS_ON = 5

# The period names the assistant understands. The model will pick one of these
# instead of calculating dates itself (it's bad at date arithmetic too).
PERIODS = (
    "today", "yesterday", "this_week", "last_week", "this_month", "last_month",
    "last_7_days", "last_30_days", "this_year",
)


def now_local(now: datetime | None = None) -> tuple[str, str]:
    """The current date and time in the configured time zone, as ('YYYY-MM-DD', 'HH:MM:SS').
    `now` can be passed in by tests so results don't depend on the real clock."""
    moment = (now or datetime.now(tz=ZoneInfo("UTC"))).astimezone(ZoneInfo(config.TIMEZONE))
    return moment.strftime("%Y-%m-%d"), moment.strftime("%H:%M:%S")


def today_local(now: datetime | None = None) -> str:
    return now_local(now)[0]


def is_iso_date(value: str) -> bool:
    """True only for real dates like '2026-02-28' (not '2026-02-30')."""
    try:
        return len(value) == 10 and date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def resolve_period(name: str, today: str) -> tuple[str, str]:
    """Turn a period name into (start, end) dates, inclusive.

    >>> resolve_period("this_month", "2026-09-30")
    ('2026-09-01', '2026-09-30')

    "This week/month/year" end TODAY, not at the end of the period, because
    future days can't have spending yet.
    """
    d = date.fromisoformat(today)
    week_start = d - timedelta(days=(d.weekday() - WEEK_STARTS_ON) % 7)
    month_start = d.replace(day=1)
    last_month_end = month_start - timedelta(days=1)

    periods = {
        "today": (d, d),
        "yesterday": (d - timedelta(days=1), d - timedelta(days=1)),
        "this_week": (week_start, d),
        "last_week": (week_start - timedelta(days=7), week_start - timedelta(days=1)),
        "this_month": (month_start, d),
        "last_month": (last_month_end.replace(day=1), last_month_end),
        "last_7_days": (d - timedelta(days=6), d),     # 7 days including today
        "last_30_days": (d - timedelta(days=29), d),
        "this_year": (d.replace(month=1, day=1), d),
    }
    if name not in periods:
        raise ValueError(f"Unknown period {name!r}. Use one of: {', '.join(PERIODS)}")
    start, end = periods[name]
    return start.isoformat(), end.isoformat()


# ---------------------------------------------------------------------------
# "yesterday", "3 days ago", "on Monday": which DAY the user means
# ---------------------------------------------------------------------------
# The model must not do this arithmetic. Real case: the user said "yesterday" on
# Friday Oct 2 and the model logged Sep 29, anchored on older messages in the
# conversation where "yesterday" really was Sep 29. Python resolves the words.

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# Longest alternatives first, so "day before yesterday" isn't read as "yesterday".
DAY_PHRASE = re.compile(
    r"\b(the day before yesterday|day before yesterday|yesterday|last night|today|tonight|this morning"
    r"|\d+ days? ago|a week ago"
    rf"|(?:last |on )?(?:{'|'.join(WEEKDAYS)}))\b",
    re.IGNORECASE,
)


def day_phrases(text: str) -> list[str]:
    """The relative-day expressions in a message: 'paid 260 yesterday' -> ['yesterday']."""
    return [match.group(0).lower() for match in DAY_PHRASE.finditer(text)]


def resolve_day(text: str, today: str) -> str:
    """Turn 'yesterday', '3 days ago', 'monday' or an exact 'YYYY-MM-DD' into a date.

    >>> resolve_day("yesterday", "2026-10-02")
    '2026-10-01'

    A bare weekday means its most recent occurrence (today counts); "last
    monday" always means a Monday before today. Future dates are refused: an
    expense can't have happened tomorrow.
    """
    d = date.fromisoformat(today)
    phrase = " ".join(text.lower().split())
    phrase = phrase.removeprefix("on ").removeprefix("the ")
    if phrase in ("", "today", "tonight", "this morning"):
        result = d
    elif phrase in ("yesterday", "last night"):
        result = d - timedelta(days=1)
    elif phrase == "day before yesterday":
        result = d - timedelta(days=2)
    elif phrase == "a week ago":
        result = d - timedelta(days=7)
    elif match := re.fullmatch(r"(\d+) days? ago", phrase):
        result = d - timedelta(days=int(match.group(1)))
    elif phrase.removeprefix("last ") in WEEKDAYS:
        back = (d.weekday() - WEEKDAYS.index(phrase.removeprefix("last "))) % 7
        if phrase.startswith("last ") and back == 0:
            back = 7
        result = d - timedelta(days=back)
    elif is_iso_date(phrase):
        result = date.fromisoformat(phrase)
    else:
        raise ValueError(f"Can't understand the date {text!r}. Use 'today', 'yesterday', 'N days ago', "
                         f"a weekday name, or YYYY-MM-DD.")
    if result > d:
        raise ValueError(f"{result.isoformat()} is in the future (today is {today}). Expenses can't be dated ahead.")
    return result.isoformat()


def describe_day(day: str, today: str) -> str:
    """'2026-10-01' -> 'Thu 2026-10-01 (yesterday)'. Shown in the confirmation, so a
    wrong date is obvious before it's saved."""
    d, t = date.fromisoformat(day), date.fromisoformat(today)
    ago = (t - d).days
    when = "today" if ago == 0 else "yesterday" if ago == 1 else f"{ago} days ago"
    return f"{d.strftime('%a')} {day} ({when})"


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def previous_period(name: str, today: str) -> tuple[str, str]:
    """The FAIR period to compare `name` with: same length, the step before.

    Why "fair": on Sep 10, comparing this month (10 days) with ALL of August
    (31 days) always looks like you spend less. So:
        this_month  (Sep 1-10)   ->  Aug 1-10   (same days; capped at the month's length)
        this_week   (Sat-Wed)    ->  the previous Sat-Wed
        last_month  (all Aug)    ->  all of July
        last_7_days              ->  the 7 days before that
        this_year   (Jan 1-Sep 10) -> Jan 1-Sep 10 last year
    The same rule as EXO's Insights screen, so both apps agree.
    """
    start, end = (date.fromisoformat(d) for d in resolve_period(name, today))
    if name == "this_month":
        prev_end_month = start - timedelta(days=1)
        day = min(end.day, days_in_month(prev_end_month.year, prev_end_month.month))
        return prev_end_month.replace(day=1).isoformat(), prev_end_month.replace(day=day).isoformat()
    if name == "last_month":
        prev_end = start - timedelta(days=1)
        return prev_end.replace(day=1).isoformat(), prev_end.isoformat()
    if name == "this_week":
        # The SAME weekdays of last week (Sat-Wed vs Sat-Wed). Stepping back by the
        # period's length (5 days) would compare Sat-Wed with Mon-Fri.
        return (start - timedelta(days=7)).isoformat(), (end - timedelta(days=7)).isoformat()
    if name == "this_year":
        prev_start = start.replace(year=start.year - 1)
        day = min(end.day, days_in_month(end.year - 1, end.month))
        return prev_start.isoformat(), end.replace(year=end.year - 1, day=day).isoformat()
    # Every other period is a fixed number of days: step back by exactly that many.
    length = (end - start).days + 1
    return (start - timedelta(days=length)).isoformat(), (start - timedelta(days=1)).isoformat()
