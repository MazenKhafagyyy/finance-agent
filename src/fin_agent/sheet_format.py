"""
The Google Sheet format, ported from EXO (a TypeScript app that shares this Sheet).

This file is the CONTRACT between the two programs. EXO and the assistant both
read and write the same spreadsheet, so a row written here must look exactly
like a row EXO would write, or EXO will skip it (and vice versa).

Ported from EXO's:
    src/core/models.ts    -> the dataclasses below
    src/core/defaults.ts  -> DEFAULT_CATEGORIES, SEED_MERCHANTS
    src/server/rows.ts    -> HEADERS, *_to_row / row_to_*, dates, formatting
    src/core/parser.ts    -> normalize()

There is NO Google code in this file on purpose: it only converts between
Python objects and plain lists of cell values. That makes it easy to test
without a network connection (see tests/test_sheet_format.py).
"""
import re
import secrets
from dataclasses import dataclass
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# The data types (EXO: models.ts)
# ---------------------------------------------------------------------------

# EXO supports exactly two currencies. Anything else in the Currency column
# makes the row invalid (it gets skipped when reading).
CURRENCIES = ("EGP", "CAD")


@dataclass
class Category:
    name: str               # e.g. "Fast Food"
    essential: bool         # the Yes/No that powers "what's unessential?" questions
    hint: str               # a short description of what belongs here
    budget: float | None = None  # monthly budget; None = not set


@dataclass
class Merchant:
    key: str                # normalized text used for matching, e.g. "buffalo burger"
    name: str               # display name, e.g. "Buffalo Burger"
    category: str           # the category this merchant belongs to


@dataclass
class Entry:
    """One expense: one row of the Transactions tab."""
    id: str                 # "tx" + 12 hex characters, e.g. "tx3f9a0c12be45"
    date: str               # "YYYY-MM-DD" in Cairo time
    time: str               # "HH:MM:SS" in Cairo time
    amount: float | None    # None = amount not known yet (the row is "unresolved")
    currency: str           # "EGP" or "CAD"
    merchant: str           # e.g. "McDonald's"
    category: str           # "" = not chosen yet (the row is "unresolved")
    essential: bool         # copied from the category when the row is written
    note: str               # leftover words, e.g. "to work" from "uber to work"
    message: str            # what the user originally typed or said


def is_unresolved(e: Entry) -> bool:
    """EXO's definition: a row with no amount or no category still needs finishing."""
    return e.amount is None or e.category == ""


# ---------------------------------------------------------------------------
# Starting data (EXO: defaults.ts). Written to the Sheet on first setup.
# After that, the Sheet itself is the source of truth (you can edit categories there).
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES = [
    Category("Fast Food", False, "Restaurants, takeaway, food delivery (Talabat, Elmenus)"),
    Category("Coffee", False, "Coffee shops and cafés"),
    Category("Snacks", False, "Kiosk snacks, drinks, sweets"),
    Category("Groceries", True, "Supermarkets, food for home"),
    Category("Going-out", False, "Bars, clubs, cinema, outings, events"),
    Category("Nicotine", False, "Cigarettes, vapes, shisha"),
    Category("Subscriptions", False, "Recurring digital services (Netflix, Spotify, PS Plus)"),
    Category("Utilities", True, "Electricity, water, gas, internet, mobile credit, laundry"),
    Category("Rent", True, "Rent and housing payments"),
    Category("Transport", True, "Uber, Careem, taxis, fuel, metro, parking"),
    Category("Health", True, "Pharmacy, doctors, gym"),
    Category("Installments", True, "Installment and loan payments"),
    Category("Clothes", False, "Clothing and shoes"),
    Category("Games", False, "Video games and in-game purchases"),
    Category("Furniture", False, "Furniture and home items"),
    Category("Other", False, "Anything that fits nowhere else"),
]

# category -> {normalized key: display name}. Same list and order as EXO.
_SEED = {
    "Fast Food": {
        "mcdonalds": "McDonald's", "kfc": "KFC", "burger king": "Burger King", "hardees": "Hardee's",
        "pizza hut": "Pizza Hut", "papa johns": "Papa John's", "dominos": "Domino's",
        "buffalo burger": "Buffalo Burger", "cook door": "Cook Door", "momen": "Mo'men",
        "talabat": "Talabat", "elmenus": "Elmenus",
    },
    "Coffee": {"starbucks": "Starbucks", "costa": "Costa", "cilantro": "Cilantro", "tbs": "TBS"},
    "Groceries": {
        "carrefour": "Carrefour", "spinneys": "Spinneys", "seoudi": "Seoudi", "kazyon": "Kazyon",
        "hyperone": "HyperOne", "breadfast": "Breadfast",
    },
    "Transport": {"uber": "Uber", "careem": "Careem", "indrive": "inDrive", "didi": "DiDi", "swvl": "Swvl"},
    "Subscriptions": {
        "netflix": "Netflix", "spotify": "Spotify", "shahid": "Shahid", "anghami": "Anghami",
        "ps plus": "PS Plus", "youtube premium": "YouTube Premium", "icloud": "iCloud",
    },
    # Keys are matched as whole words, so EXO skips ones that are everyday words ("we", "orange").
    "Utilities": {"vodafone": "Vodafone", "etisalat": "Etisalat"},
    "Health": {"el ezaby": "El Ezaby", "seif": "Seif Pharmacy"},
}
SEED_MERCHANTS = [
    Merchant(key, name, category) for category, merchants in _SEED.items() for key, name in merchants.items()
]

# ---------------------------------------------------------------------------
# The tabs and their columns (EXO: rows.ts)
# ---------------------------------------------------------------------------

TABS = ("Transactions", "Categories", "Merchants", "Wallet")

HEADERS = {
    "Transactions": ["ID", "Date", "Time", "Amount", "Currency", "Merchant", "Category", "Essential", "Note", "Message"],
    "Categories": ["Category", "Essential", "Monthly budget", "Hint"],
    "Merchants": ["Key", "Merchant", "Category"],
    "Wallet": ["ID", "Date", "Time", "Type", "Amount (EGP)", "Note"],
}


# Older EXO versions used slightly different header NAMES for the same columns in
# the same order. Reading works by position, so these are accepted as equivalent.
LEGACY_HEADERS = {
    "Categories": [["Category", "Essential", "Monthly budget (EGP)", "Hint for EXO"]],
}


def header_problem(tab: str, header: list[str]) -> str | None:
    """None if an existing tab's header row matches the format we read and write,
    otherwise a description of the mismatch. Writing rows under the wrong header
    would put values in the wrong columns, so a mismatch must stop all writes."""
    found = [str(h).strip() for h in header]
    while found and found[-1] == "":        # ignore empty trailing cells
        found.pop()
    if found == HEADERS[tab] or found in LEGACY_HEADERS.get(tab, []):
        return None
    return f"{tab} tab header is {found}, expected {HEADERS[tab]}"


def seed_rows(tab: str) -> list[list]:
    """The rows written under the header when a tab is first created."""
    if tab == "Categories":
        return [[c.name, yes_no(c.essential), "", c.hint] for c in DEFAULT_CATEGORIES]
    if tab == "Merchants":
        return [merchant_to_row(m) for m in SEED_MERCHANTS]
    return []  # Transactions and Wallet start empty


def format_requests(tab: str, sheet_id: int) -> list[dict]:
    """Google Sheets formatting that EXO applies to a new tab:
    a bold header row, a date format on column B, and (Transactions only) a
    yellow highlight for unresolved rows. These are raw Sheets API "requests"."""
    requests = [{
        "repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
            "fields": "userEnteredFormat.textFormat.bold",
        }
    }]
    if tab in ("Transactions", "Wallet"):
        requests.append({
            "repeatCell": {
                "range": {"sheetId": sheet_id, "startRowIndex": 1, "startColumnIndex": 1, "endColumnIndex": 2},
                "cell": {"userEnteredFormat": {"numberFormat": {"type": "DATE", "pattern": "yyyy-mm-dd"}}},
                "fields": "userEnteredFormat.numberFormat",
            }
        })
    if tab == "Transactions":
        requests.append({
            "addConditionalFormatRule": {
                "index": 0,
                "rule": {
                    "ranges": [{"sheetId": sheet_id, "startRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": 10}],
                    "booleanRule": {
                        # Yellow when the row has an ID but Amount (D) or Category (G) is empty.
                        "condition": {"type": "CUSTOM_FORMULA",
                                      "values": [{"userEnteredValue": '=AND($A2<>"",OR($D2="",$G2=""))'}]},
                        "format": {"backgroundColor": {"red": 1, "green": 0.95, "blue": 0.6}},
                    },
                },
            }
        })
    return requests


# ---------------------------------------------------------------------------
# Dates: Google Sheets stores dates as numbers ("serial numbers")
# ---------------------------------------------------------------------------

# Day 0 of Google Sheets' calendar. 2026-09-30 is stored as the number of days
# since this date (46295). That's how Sheets can do date maths and formatting.
SHEETS_EPOCH = date(1899, 12, 30)


def date_to_serial(iso: str) -> int:
    """'2026-09-30' -> 46295"""
    return (date.fromisoformat(iso) - SHEETS_EPOCH).days


def serial_to_date(serial: float) -> str:
    """46295 -> '2026-09-30'. Any fraction (a time of day) is dropped."""
    return (SHEETS_EPOCH + timedelta(days=int(serial // 1))).isoformat()


# ---------------------------------------------------------------------------
# Rows <-> objects
# ---------------------------------------------------------------------------

def new_id(prefix: str = "tx") -> str:
    """EXO's ID format: a prefix + 6 random bytes as 12 hex characters, e.g. 'tx3f9a0c12be45'.
    `secrets` gives cryptographically random bytes, so two IDs practically never collide."""
    return prefix + secrets.token_hex(6)


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def is_yes(value: str) -> bool:
    return value.lower() in ("yes", "true", "1", "y")


def entry_to_row(e: Entry) -> list:
    """An Entry as the 10 cells of a Transactions row, exactly as EXO writes them."""
    return [
        e.id,
        date_to_serial(e.date),                     # a date NUMBER, not text
        e.time,
        "" if e.amount is None else e.amount,       # empty cell = unknown amount
        e.currency,
        e.merchant,
        e.category,
        yes_no(e.essential) if e.category else "",  # Essential is empty when there's no category
        e.note,
        e.message,
    ]


def row_to_entry(row: list) -> Entry | None:
    """A Transactions row as an Entry, or None for rows that aren't valid expenses
    (blank rows, notes typed by hand, text in the Amount column...). Invalid rows
    are skipped rather than crashing the program."""
    row_id = cell(row, 0)
    day = to_date(row[1] if len(row) > 1 else None)
    amount = to_amount(row[3] if len(row) > 3 else None)
    currency = cell(row, 4).upper() or "EGP"
    # `amount is _INVALID` means the cell had something that isn't a valid amount.
    if not row_id or not day or amount is _INVALID or currency not in CURRENCIES:
        return None
    category = cell(row, 6)
    return Entry(
        id=row_id, date=day, time=to_time(row[2] if len(row) > 2 else None), amount=amount,
        currency=currency, merchant=cell(row, 5), category=category,
        essential=category != "" and is_yes(cell(row, 7)), note=cell(row, 8), message=cell(row, 9),
    )


def row_to_category(row: list) -> Category | None:
    name = cell(row, 0)
    if not name:
        return None
    try:
        budget = float(cell(row, 2)) if cell(row, 2) else None
    except ValueError:
        budget = None
    return Category(name=name, essential=is_yes(cell(row, 1)), hint=cell(row, 3), budget=budget)


def merchant_to_row(m: Merchant) -> list:
    return [m.key, m.name, m.category]


def row_to_merchant(row: list) -> Merchant | None:
    key = normalize(cell(row, 0))
    category = cell(row, 2)
    if not key or not category:
        return None
    return Merchant(key=key, name=cell(row, 1) or key, category=category)


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

_INVALID = object()  # a unique marker meaning "this cell has a bad amount"


def cell(row: list, i: int) -> str:
    """The cell at position i as trimmed text; '' if the row is shorter or the cell is empty."""
    value = row[i] if i < len(row) else None
    return "" if value is None else str(value).strip()


def to_date(value) -> str | None:
    """Sheets may give us a serial number (46295) or text ('2026-09-30')."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return serial_to_date(value)
    text = value.strip() if isinstance(value, str) else ""
    return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else None


def to_time(value) -> str:
    """A time can come back as a fraction of a day (0.5 = noon) or as 'HH:MM[:SS]' text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = round((value % 1) * 86_400) % 86_400
        return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", value.strip() if isinstance(value, str) else "")
    return f"{int(match[1]):02d}:{match[2]}:{match[3] or '00'}" if match else "00:00:00"


def to_amount(value):
    """None = empty cell (amount not given yet). _INVALID = not a positive number (skip the row)."""
    if value is None or value == "":
        return None
    try:
        n = float(value) if isinstance(value, (int, float)) else float(str(value).replace(",", ""))
    except ValueError:
        return _INVALID
    return round(n, 2) if n > 0 else _INVALID


def normalize(text: str) -> str:
    """Lowercase merchant text for matching, exactly like EXO: "McDonald's!" -> "mcdonalds".
    Apostrophes are removed; any other non-letter/digit becomes a space."""
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^\w\s]", " ", text)  # \w in Python = letters, digits, underscore (incl. Arabic)
    return " ".join(text.split())
