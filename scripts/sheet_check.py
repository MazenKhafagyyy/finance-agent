"""
Connect to the real Google Sheet, set it up if needed, and print a summary.

Prints counts and dates only, never transaction details, so it's safe to run
while an AI assistant is watching the output.

Run:  uv run python scripts/sheet_check.py
"""
import sys

from fin_agent import config
from fin_agent.sheet_format import is_unresolved
from fin_agent.sheets import SheetStore

# Fail early with a clear message instead of a confusing Google error.
missing = [name for name in ("GOOGLE_SHEET_ID", "GOOGLE_SERVICE_ACCOUNT_FILE") if not getattr(config, name)]
if missing:
    sys.exit(f"Missing in .env: {', '.join(missing)}")

store = SheetStore(config.GOOGLE_SHEET_ID, config.GOOGLE_SERVICE_ACCOUNT_FILE)
print(f"Connected to spreadsheet: {store.book.title!r}")

created = store.setup()
print(f"Tabs created now: {', '.join(created) if created else 'none (already set up)'}")

snap = store.load()
dates = sorted(e.date for e in snap.entries)
print(f"Transactions: {len(snap.entries)} "
      f"({sum(is_unresolved(e) for e in snap.entries)} unresolved)"
      + (f", from {dates[0]} to {dates[-1]}" if dates else ""))
print(f"Categories:   {len(snap.categories)} ({sum(c.essential for c in snap.categories)} essential)")
print(f"Merchants:    {len(snap.merchants)}")
