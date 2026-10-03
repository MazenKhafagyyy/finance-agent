"""The store interface, tested on the in-memory fake (never the real Sheet)."""
from fin_agent.sheet_format import HEADERS, Entry, Merchant
from fin_agent.sheets import MemoryStore, extract_sheet_id


def make_entry():
    return Entry(id="tx0123456789ab", date="2026-09-30", time="14:05:09", amount=200.0, currency="EGP",
                 merchant="McDonald's", category="Fast Food", essential=False, note="", message="200 mcdonalds")


def test_setup_creates_exo_tabs_once():
    store = MemoryStore()
    assert store.setup() == ["Transactions", "Categories", "Merchants", "Wallet"]
    assert store.tabs["Transactions"] == [HEADERS["Transactions"]]
    assert store.setup() == []  # idempotent: second call creates nothing


def test_load_after_setup_has_seeds_and_no_entries():
    snap = MemoryStore().load()
    assert snap.entries == []
    assert len(snap.categories) == 16
    assert any(m.key == "mcdonalds" for m in snap.merchants)


def test_add_entry_then_load():
    store = MemoryStore()
    store.add_entry(make_entry())
    assert store.load().entries == [make_entry()]


def test_save_merchants_updates_or_appends():
    store = MemoryStore()
    store.save_merchants([Merchant("mcdonalds", "McDonald's", "Snacks"), Merchant("abu tarek", "Abu Tarek", "Fast Food")])
    merchants = {m.key: m for m in store.load().merchants}
    assert merchants["mcdonalds"].category == "Snacks"   # updated in place
    assert merchants["abu tarek"].name == "Abu Tarek"     # appended
    assert sum(1 for m in store.load().merchants if m.key == "mcdonalds") == 1


def test_extract_sheet_id_from_link_or_id():
    link = "https://docs.google.com/spreadsheets/d/1AbC_d-EF/edit#gid=0"
    assert extract_sheet_id(link) == "1AbC_d-EF"
    assert extract_sheet_id(" 1AbC_d-EF ") == "1AbC_d-EF"
