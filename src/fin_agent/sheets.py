"""
Reading and writing the Google Sheet (the "adapter" to Google).

Two classes with the SAME methods:
    SheetStore   the real thing: talks to Google Sheets through a service account
    MemoryStore  a fake that keeps everything in Python lists, for tests

Everything else in the assistant (tools, the agent) only uses these methods:
    setup()           create any missing tabs (safe to call every time)
    load()            read everything into a Snapshot of Python objects
    add_entry(e)      append one expense row
    save_merchants()  remember merchant -> category (EXO benefits too)
So swapping the real Sheet for the fake one in tests changes nothing else.

HOW THE LOGIN WORKS
A "service account" is a robot Google account with its own key file (JSON).
It can only open spreadsheets that were SHARED with its email address. The key
file is effectively its password: we only pass its PATH to gspread, and never
read or print its contents ourselves.
"""
import re
from dataclasses import dataclass, field

import gspread

from fin_agent.sheet_format import (
    DEFAULT_CATEGORIES, HEADERS, TABS, Category, Entry, Merchant,
    entry_to_row, format_requests, header_problem, merchant_to_row, normalize,
    row_to_category, row_to_entry, row_to_merchant, seed_rows,
)


class SheetFormatError(RuntimeError):
    """The Sheet's tabs don't have the columns we expect; writing would corrupt it."""


@dataclass
class Snapshot:
    """Everything the assistant needs, read from the Sheet in one go."""
    entries: list[Entry] = field(default_factory=list)
    categories: list[Category] = field(default_factory=list)
    merchants: list[Merchant] = field(default_factory=list)


def extract_sheet_id(id_or_url: str) -> str:
    """Accept either the bare ID or the full link you copy from the browser:
    https://docs.google.com/spreadsheets/d/<THIS PART>/edit#gid=0"""
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", id_or_url)
    return match.group(1) if match else id_or_url.strip()


# ---------------------------------------------------------------------------
# The real Sheet
# ---------------------------------------------------------------------------

class SheetStore:
    def __init__(self, sheet_id: str, key_file: str):
        # gspread reads the key file and signs in as the service account.
        client = gspread.service_account(filename=key_file)
        # "book" = the whole spreadsheet file; each tab inside it is a "worksheet".
        self.book = client.open_by_key(extract_sheet_id(sheet_id))

    def setup(self) -> list[str]:
        """Check existing tabs have the expected columns (raise SheetFormatError if
        not), then create any of EXO's 4 tabs that are missing, with headers, seed
        rows and formatting, exactly as EXO does on its first run. Returns the tabs created.

        Idempotent: once the tabs exist, this does nothing. So it's safe to call
        on every start, and it never touches tabs that already exist (or their data).
        """
        worksheets = {ws.title: ws for ws in self.book.worksheets()}

        # Tabs that already exist must use the format we expect. Otherwise every
        # row we write would land in the wrong columns (this happened once: an
        # older EXO had set up Transactions without the Currency column).
        problems = [p for tab in TABS if tab in worksheets
                    if (p := header_problem(tab, worksheets[tab].row_values(1)))]
        if problems:
            raise SheetFormatError("The Sheet's format doesn't match EXO's current format, so nothing "
                                   "will be written:\n  - " + "\n  - ".join(problems))

        missing = [tab for tab in TABS if tab not in worksheets]
        if not missing:
            return []

        created = {tab: self.book.add_worksheet(title=tab, rows=1000, cols=26) for tab in missing}

        # Header row + starting rows (categories, merchants). RAW = store values
        # exactly as given; text is never interpreted as a formula.
        for tab, worksheet in created.items():
            worksheet.update([HEADERS[tab], *seed_rows(tab)], range_name="A1", value_input_option="RAW")

        # Formatting, sent as one batch of Sheets API requests:
        # freeze the header row (EXO's frozenRowCount: 1) + EXO's formatting rules.
        requests = []
        for tab, worksheet in created.items():
            requests.append({
                "updateSheetProperties": {
                    "properties": {"sheetId": worksheet.id, "gridProperties": {"frozenRowCount": 1}},
                    "fields": "gridProperties.frozenRowCount",
                }
            })
            requests += format_requests(tab, worksheet.id)
        self.book.batch_update({"requests": requests})
        return missing

    def load(self) -> Snapshot:
        """Read Transactions, Categories and Merchants in ONE request (faster than three)."""
        ranges = [f"{tab}!A2:J" for tab in ("Transactions", "Categories", "Merchants")]  # A2 = skip headers
        result = self.book.values_batch_get(ranges, params={
            # Raw values, not what's displayed: amounts as numbers, dates as serial numbers.
            "valueRenderOption": "UNFORMATTED_VALUE",
            "dateTimeRenderOption": "SERIAL_NUMBER",
        })
        transactions, categories, merchants = (r.get("values", []) for r in result["valueRanges"])
        parsed_categories = [c for c in map(row_to_category, categories) if c]
        return Snapshot(
            # `if e` drops the None results, i.e. rows that weren't valid expenses.
            entries=[e for e in map(row_to_entry, transactions) if e],
            # Same fallback as EXO: if the Categories tab is empty, use the defaults.
            categories=parsed_categories or [Category(c.name, c.essential, c.hint) for c in DEFAULT_CATEGORIES],
            merchants=[m for m in map(row_to_merchant, merchants) if m],
        )

    def add_entry(self, entry: Entry) -> None:
        """Append one row at the bottom of the Transactions tab."""
        self.book.worksheet("Transactions").append_rows(
            [entry_to_row(entry)],
            value_input_option="RAW",
            insert_data_option="OVERWRITE",  # use empty rows below the data instead of inserting new ones
            table_range="A1",                # the table starts at A1; append after its last row
        )

    def save_merchants(self, merchants: list[Merchant]) -> None:
        """Remember merchants: update the row if the key already exists, else append."""
        worksheet = self.book.worksheet("Merchants")
        keys = [normalize(str(k)) for k in worksheet.col_values(1)]  # column A, including the header
        for merchant in merchants:
            if merchant.key in keys[1:]:
                row_number = keys.index(merchant.key, 1) + 1  # +1: Sheets rows start at 1
                worksheet.update([merchant_to_row(merchant)], range_name=f"A{row_number}:C{row_number}",
                                 value_input_option="RAW")
            else:
                worksheet.append_rows([merchant_to_row(merchant)], value_input_option="RAW",
                                      insert_data_option="OVERWRITE", table_range="A1")
                keys.append(merchant.key)


# ---------------------------------------------------------------------------
# The fake Sheet, for tests
# ---------------------------------------------------------------------------

class MemoryStore:
    """Same methods as SheetStore, but everything lives in Python lists.
    Tests use this so they're fast and never touch the real Sheet.

    It stores ROWS (lists of cells), not objects, and converts with the same
    functions as SheetStore. So tests also exercise the real row format."""

    def __init__(self):
        self.tabs: dict[str, list[list]] = {}

    def setup(self) -> list[str]:
        missing = [tab for tab in TABS if tab not in self.tabs]
        for tab in missing:
            self.tabs[tab] = [HEADERS[tab], *seed_rows(tab)]
        return missing

    def load(self) -> Snapshot:
        self.setup()
        body = {tab: rows[1:] for tab, rows in self.tabs.items()}  # [1:] = skip the header
        parsed_categories = [c for c in map(row_to_category, body["Categories"]) if c]
        return Snapshot(
            entries=[e for e in map(row_to_entry, body["Transactions"]) if e],
            categories=parsed_categories or [Category(c.name, c.essential, c.hint) for c in DEFAULT_CATEGORIES],
            merchants=[m for m in map(row_to_merchant, body["Merchants"]) if m],
        )

    def add_entry(self, entry: Entry) -> None:
        self.setup()
        self.tabs["Transactions"].append(entry_to_row(entry))

    def save_merchants(self, merchants: list[Merchant]) -> None:
        self.setup()
        rows = self.tabs["Merchants"]
        for merchant in merchants:
            for i, row in enumerate(rows[1:], start=1):
                if normalize(str(row[0])) == merchant.key:
                    rows[i] = merchant_to_row(merchant)
                    break
            else:  # no break: the key wasn't found
                rows.append(merchant_to_row(merchant))
