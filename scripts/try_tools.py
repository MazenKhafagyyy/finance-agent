"""
Try the Phase 2 tools by hand on the REAL Sheet (no AI involved).

    uv run python scripts/try_tools.py add "200 mcdonalds"
    uv run python scripts/try_tools.py total this_week
    uv run python scripts/try_tools.py breakdown this_month
    uv run python scripts/try_tools.py breakdown this_month --nonessential
    uv run python scripts/try_tools.py list last_7_days

`add` shows the row and asks y/n before writing anything.
Periods: today, yesterday, this_week, last_week, this_month, last_month,
         last_7_days, last_30_days, this_year

PRIVACY: this prints your real spending. Read it yourself; don't paste it
into a cloud AI chat.
"""
import argparse
import json

from fin_agent import config
from fin_agent.dates import PERIODS, now_local, resolve_period
from fin_agent.entries import build_from_text
from fin_agent.sheets import SheetStore
from fin_agent.tools import describe_entry, list_transactions, save_expense, spending_breakdown, spending_total

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("add").add_argument("text", help='a quick line, e.g. "200 mcdonalds"')
for name in ("total", "breakdown", "list"):
    p = sub.add_parser(name)
    p.add_argument("period", choices=PERIODS)
    p.add_argument("--currency", default=config.DEFAULT_CURRENCY, choices=["EGP", "CAD"])
    p.add_argument("--nonessential", action="store_true", help="only non-essential spending")
args = parser.parse_args()

store = SheetStore(config.GOOGLE_SHEET_ID, config.GOOGLE_SERVICE_ACCOUNT_FILE)
store.setup()   # stops with SheetFormatError if a tab's columns don't match
snap = store.load()
now = now_local()

if args.command == "add":
    built = build_from_text(args.text, snap, now)
    print("About to add:", describe_entry(built))
    if input("Save it? [y/n] ").strip().lower() == "y":
        save_expense(store, built)
        print("Saved. It will show in EXO too.")
    else:
        print("Not saved.")
else:
    start, end = resolve_period(args.period, now[0])
    essential = False if args.nonessential else None
    if args.command == "total":
        result = spending_total(snap, start, end, args.currency, essential=essential)
    elif args.command == "breakdown":
        result = spending_breakdown(snap, start, end, args.currency, essential=essential)
    else:
        result = list_transactions(snap, start, end, args.currency)
    # This JSON is exactly what the agent's model will receive in Phase 3.
    print(json.dumps(result, indent=2, ensure_ascii=False))
