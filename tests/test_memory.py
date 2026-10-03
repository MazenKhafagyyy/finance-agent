"""Long-term memory tests. A fake embedding (word counts hashed into 64 slots)
stands in for Ollama: texts sharing most words get similar vectors.

Several tests re-create failures seen in live runs with a real local model."""
import threading
import zlib

import pytest

from fin_agent import memory as memory_module
from fin_agent.memory import MemoryBank, MemoryRuleError, PossibleDuplicate, normalize_topic


def fake_embed(texts):
    vectors = []
    for text in texts:
        v = [0.0] * 64
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % 64] += 1.0
        vectors.append(v)
    return vectors


@pytest.fixture
def bank():
    return MemoryBank(":memory:", embed=fake_embed)


def texts(bank):
    return [(m.topic, m.text) for m in bank.all()]


# --- basics ---------------------------------------------------------------------

def test_remember_and_list(bank):
    saved, replaced = bank.remember("payday", "Gets paid on the 25th of each month", "fact")
    assert replaced is None and saved.topic == "payday"
    assert texts(bank) == [("payday", "Gets paid on the 25th of each month")]


def test_topics_are_normalized():
    assert normalize_topic("Pay Day!") == "pay_day"
    assert normalize_topic("  answer-style ") == "answer_style"


def test_forget_by_number_or_topic(bank):
    bank.remember("home_city", "Lives in Cairo", "fact")
    bank.remember("job", "Works as a data analyst", "fact")
    assert bank.forget(1).text == "Lives in Cairo"
    assert bank.forget_topic("Job").text == "Works as a data analyst"
    assert bank.all() == []
    assert bank.forget(99) is None and bank.forget_topic("nope") is None


def test_memories_survive_a_restart(tmp_path):
    path = tmp_path / "memories.db"
    MemoryBank(path, embed=fake_embed).remember("home_city", "Lives in Cairo", "fact")
    assert texts(MemoryBank(path, embed=fake_embed)) == [("home_city", "Lives in Cairo")]


# --- updates: the part that failed in live runs --------------------------------------

def test_same_topic_replaces_even_when_worded_very_differently(bank):
    # Live run 4: "Payday moved to the 1st" was saved NEXT TO "I get paid on the 25th"
    # because the embeddings didn't match. With topics, it's a plain key match.
    bank.remember("payday", "I get paid on the 25th of every month", "fact")
    saved, replaced = bank.remember("payday", "Payday moved to the 1st of the month", "fact")
    assert replaced.text == "I get paid on the 25th of every month"
    assert texts(bank) == [("payday", "Payday moved to the 1st of the month")]


def test_updating_one_topic_never_touches_another(bank):
    # Live run 3: an update aimed at the payday overwrote the "short answers" preference.
    bank.remember("answer_style", "Prefers really short answers", "preference")
    bank.remember("payday", "Gets paid on the 25th", "fact")
    bank.remember("payday", "Gets paid on the 1st", "fact")
    assert texts(bank) == [("answer_style", "Prefers really short answers"), ("payday", "Gets paid on the 1st")]


def test_the_database_allows_only_one_memory_per_topic(bank):
    import sqlite3
    bank.remember("payday", "Gets paid on the 25th", "fact")
    with pytest.raises(sqlite3.IntegrityError):
        bank.db.execute("INSERT INTO memories (topic, text, kind, embedding, created, updated) "
                        "VALUES ('payday', 'x', 'fact', '[]', '', '')")


def test_parallel_saves_to_one_topic_leave_one_memory(bank):
    # Live runs: the model fired several memory tool calls at the same moment.
    threads = [threading.Thread(target=bank.remember, args=("payday", f"Gets paid on day {i}", "fact"))
               for i in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(bank.all()) == 1


# --- safety net for NEW topics (same fact under a new name) ---------------------------

LONG = "Gets paid by bank transfer on the 25th of each month from the company"


def test_new_topic_with_identical_content_updates_the_existing_memory(bank):
    bank.remember("payday", LONG, "fact")
    saved, replaced = bank.remember("pay_date", LONG.replace("25th", "1st"), "fact")  # ~0.92 similar
    assert saved.topic == "payday" and replaced is not None
    assert len(bank.all()) == 1


def test_new_topic_with_similar_content_asks_the_model(bank):
    bank.remember("answer_style", "Prefers short answers always", "preference")
    with pytest.raises(PossibleDuplicate, match=r"\[answer_style\]"):
        bank.remember("reply_length", "Prefers short replies always", "preference")      # 0.75 similar
    assert len(bank.all()) == 1


def test_model_can_confirm_a_similar_memory_is_distinct(bank):
    bank.remember("answer_style", "Prefers short answers always", "preference")
    bank.remember("reply_length", "Prefers short replies always", "preference", distinct=True)
    assert len(bank.all()) == 2


def test_different_facts_are_both_kept(bank):
    bank.remember("payday", "Gets paid on the 25th of each month", "fact")
    bank.remember("answer_style", "Prefers short answers", "preference")
    bank.remember("laptop_goal", "Saving for a laptop by March", "fact")
    assert len(bank.all()) == 3


# --- rules enforced in code -------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Spent 500 on food yesterday",
    "Paid 200 for groceries",
    "Bought a phone for 9,000 EGP",
    "Spent EGP 1,250 at Carrefour",
])
def test_transactions_are_refused(bank, text):
    with pytest.raises(MemoryRuleError, match="transaction"):
        bank.remember("food", text, "fact")


@pytest.mark.parametrize("text", [
    "Gets paid on the 25th of each month",    # income schedule, 25th is a date
    "Pays 8000 EGP rent every month",         # present tense: a recurring fact
    "Salary is 30,000 EGP",
    "Saving for a laptop, budget 2000, by March 2027",
])
def test_lasting_facts_with_numbers_are_allowed(bank, text):
    bank.remember("fact", text, "fact")


def test_sensitive_numbers_are_refused(bank):
    with pytest.raises(MemoryRuleError, match="card"):
        bank.remember("card", "My card is 4500 1234 5678 9012", "fact")


@pytest.mark.parametrize("topic, text, kind", [("", "Lives in Cairo", "fact"), ("city", "", "fact"),
                                               ("city", "Lives in Cairo", "secret")])
def test_invalid_input_is_refused(bank, topic, text, kind):
    with pytest.raises(MemoryRuleError):
        bank.remember(topic, text, kind)


# --- reading ----------------------------------------------------------------------------

def test_prompt_gets_all_memories_when_few(bank):
    for i, text in enumerate(["Lives in Cairo", "Prefers short answers", "Saving for a laptop"]):
        bank.remember(f"t{i}", text, "fact")
    assert len(bank.for_prompt("anything")) == 3


def test_prompt_gets_only_relevant_memories_when_many(bank, monkeypatch):
    monkeypatch.setattr(memory_module, "MAX_IN_PROMPT", 3)
    monkeypatch.setattr(memory_module, "TOP_K", 2)
    for i, text in enumerate(["Lives in Cairo", "Works as a data analyst", "Saving for a laptop by March",
                              "Laptop budget is two thousand", "Has a cat named Simba"]):
        bank.remember(f"t{i}", text, "fact", distinct=True)
    chosen = [m.text for m in bank.for_prompt("how is my laptop saving going")]
    assert len(chosen) == 2
    assert all("aptop" in text for text in chosen)


def test_old_format_file_is_migrated_not_crashed_on(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, text TEXT NOT NULL, kind TEXT NOT NULL, "
                "embedding TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL)")
    old.execute("INSERT INTO memories (text, kind, embedding, created, updated) "
                "VALUES ('Lives in Cairo', 'fact', '[1.0]', '2026-09-30', '2026-09-30')")
    old.commit()
    old.close()
    bank = MemoryBank(path, embed=fake_embed)
    assert texts(bank) == [("memory_1", "Lives in Cairo")]
    bank.remember("payday", "Gets paid on the 25th", "fact")        # new-style saves work after migrating
    assert len(bank.all()) == 2
