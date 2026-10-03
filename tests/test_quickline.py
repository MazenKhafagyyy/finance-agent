"""The same cases as EXO's tests/core/parser.test.ts and categorizer.test.ts."""
import pytest

from fin_agent.quickline import ParsedLine, categorize, find_merchant, learn, parse_amount, parse_line
from fin_agent.sheet_format import DEFAULT_CATEGORIES, SEED_MERCHANTS, Merchant, normalize


@pytest.mark.parametrize("text, amount, currency, description", [
    ("285 mcdonalds", 285, "EGP", "mcdonalds"),
    ("mcdonalds 285", 285, "EGP", "mcdonalds"),
    ("285 uber to work", 285, "EGP", "uber to work"),
    ("1,250 carrefour", 1250, "EGP", "carrefour"),
    ("1.5k rent", 1500, "EGP", "rent"),
    ("12.5 coffee", 12.5, "EGP", "coffee"),
    ("285egp kfc", 285, "EGP", "kfc"),
    ("285 EGP kfc", 285, "EGP", "kfc"),
    ("kfc 285 LE", 285, "EGP", "kfc"),
    ("٢٨٥ كشري", 285, "EGP", "كشري"),
    ("cafe 88 150", 150, "EGP", "cafe 88"),  # an amount at the end beats a number inside the name
    ("285 - mcdonalds", 285, "EGP", "mcdonalds"),
    ("20 cad tims", 20, "CAD", "tims"),
    ("20cad tims", 20, "CAD", "tims"),
    ("c$20 tims", 20, "CAD", "tims"),
    ("tims 4.5 CAD", 4.5, "CAD", "tims"),
    ("tims c$ 4.5", 4.5, "CAD", "tims"),
])
def test_parse_line(text, amount, currency, description):
    assert parse_line(text) == ParsedLine(amount, currency, description)


def test_amount_without_description():
    assert parse_line("285") == ParsedLine(285, "EGP", "")


@pytest.mark.parametrize("text", ["mcdonalds", "lunch with ali", "0 mcdonalds", "mcdonalds -5"])
def test_text_without_amount(text):
    assert parse_line(text) == ParsedLine(None, "EGP", text)


def test_cad_word_without_amount():
    assert parse_line("tims cad") == ParsedLine(None, "CAD", "tims")


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_input(text):
    assert parse_line(text) is None


def test_parse_amount():
    assert parse_amount("1,250") == (1250, None)
    assert parse_amount("285egp") == (285, "EGP")
    assert parse_amount("20cad") == (20, "CAD")
    assert parse_amount("c$20") == (20, "CAD")
    for token in ["abc", "0", "1,25", "-5", "c$"]:
        assert parse_amount(token) is None


def test_normalize_matches_exo():
    assert normalize("Café’s") == "cafés"
    assert normalize("كشري التحرير") == "كشري التحرير"


# --- merchant memory ---------------------------------------------------------

MERCHANTS = [
    Merchant("uber", "Uber", "Transport"),
    Merchant("ubr", "Uber", "Transport"),
    Merchant("buffalo burger", "Buffalo Burger", "Fast Food"),
    Merchant("burger", "Burger Place", "Fast Food"),
]


def test_find_merchant_returns_leftover_words():
    assert find_merchant("Uber to work", MERCHANTS) == (MERCHANTS[0], "to work")


def test_longest_key_wins():
    assert find_merchant("buffalo burger", MERCHANTS)[0].name == "Buffalo Burger"


def test_whole_words_only():
    assert find_merchant("uberx", MERCHANTS) is None
    assert find_merchant("مطعم", MERCHANTS) is None


def test_categorize_known_and_unknown():
    assert categorize("uber to work", MERCHANTS).category == "Transport"
    unknown = categorize(" some new cafe ", MERCHANTS)
    assert (unknown.merchant, unknown.category, unknown.known) == ("some new cafe", "", False)


def test_learn():
    assert learn(MERCHANTS, "some new cafe", "Bean Bar", "Coffee") == [
        Merchant("some new cafe", "Bean Bar", "Coffee"), Merchant("bean bar", "Bean Bar", "Coffee")]
    assert learn(MERCHANTS, "Uber", "Uber", "Health") == [
        Merchant("uber", "Uber", "Health"), Merchant("ubr", "Uber", "Health")]
    assert learn(MERCHANTS, "", "", "Coffee") == []


def test_seed_merchants_use_real_categories_and_normalized_keys():
    names = {c.name for c in DEFAULT_CATEGORIES}
    assert all(m.category in names and m.key == normalize(m.key) for m in SEED_MERCHANTS)
