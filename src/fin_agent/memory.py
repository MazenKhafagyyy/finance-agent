"""
LONG-TERM MEMORY: what the assistant knows about you across conversations.

SHORT-TERM vs LONG-TERM
    The checkpointer keeps ONE conversation's messages. That's
    short-term memory: it grows every turn and is gone after /new.
    This module keeps a small, curated PROFILE of facts and preferences about
    you. It survives /new and restarts, and it goes into the system prompt so
    the model always "knows" it. (The model itself remembers nothing between
    calls: everything it knows must be in the prompt.)

WHY TOPIC KEYS (the design that finally worked)
    Every memory has a short topic, chosen by the model:
        [payday]        Gets paid on the 25th of each month   (fact)
        [answer_style]  Prefers really short answers          (preference)
    Saving under an EXISTING topic REPLACES that memory. So "my payday moved to
    the 1st" -> remember(topic="payday", ...) -> one payday, never two.

    The first design used numeric ids ("replaces=2") plus embedding similarity
    to spot updates. Live tests with a small local model failed four times in
    four different ways: duplicates, contradictions, the WRONG id replaced (a
    payday overwrote the "short answers" preference), and a forget/replace race
    that lost the fact entirely. Small models are bad at tracking numbers but
    good at reusing a word they can see ("payday"), and embeddings can't
    reliably tell "same fact, new wording" from "different fact" (the scores
    overlap). Topics make the common case deterministic.

    Embeddings remain a SAFETY NET for NEW topics only: if the model invents
    "pay_date" while "payday" says something similar, it's asked to decide.

RULES ENFORCED IN CODE (not left to the model)
    - No spending records: "spent 500 on food" belongs in the Sheet.
    - No card / account / ID numbers (the patterns in redact.py).
    - You can list and delete everything (/memories, /forget in the chat).
    - Stored only on the machine that runs it, in DATA_DIR/memories.db (gitignored).
"""
import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from fin_agent.redact import PATTERNS

# Similarity thresholds for a NEW topic vs existing memories (qwen3-embedding, CPU):
#     same fact or an update:  0.61 .. 0.97
#     different facts:         0.35 .. 0.82
# The ranges overlap, so:
#     >= SAME_FACT  it's clearly the same fact: update that memory (keep its topic)
#     >= SIMILAR    not sure: don't save yet, ask the model (reuse that topic, or distinct=true)
#     below         a genuinely new memory
SAME_FACT = 0.90
SIMILAR = 0.60

# Up to this many memories, all of them go in the prompt (simple, and the prompt
# stays identical between turns, which keeps Ollama's cache working). Beyond it,
# only the most relevant ones for the current message are included (RAG over memories).
MAX_IN_PROMPT = 30
TOP_K = 10

KINDS = ("fact", "preference")

# A spending RECORD = a past-tense spend verb near an amount: "spent 500 on food",
# "paid 200 for groceries", "bought a phone for 9000". NOT a lasting fact like
# "gets paid on the 25th" (an income schedule; 25th is a date, not an amount) or
# "pays 8000 rent monthly" (present tense: a recurring fact, useful for planning).
_SPEND = r"\b(?:spent|bought|purchased|(?<!get )(?<!gets )(?<!got )paid)\b"
_AMOUNT = r"\b\d[\d,.]*(?!\d)(?!\s?(?:st|nd|rd|th)\b)"   # a number that isn't "25th"
LOOKS_LIKE_A_TRANSACTION = re.compile(rf"{_SPEND}[^.!?]{{0,40}}{_AMOUNT}|{_AMOUNT}[^.!?]{{0,40}}{_SPEND}",
                                      re.IGNORECASE)


@dataclass
class Memory:
    id: int
    topic: str          # short key, e.g. "payday"; unique
    text: str
    kind: str           # "fact" (about you) or "preference" (how you want answers)
    created: str
    updated: str


class MemoryRuleError(ValueError):
    """A memory that breaks the rules; the message is shown to the model so it can adjust."""


class PossibleDuplicate(MemoryRuleError):
    """A new topic whose content resembles an existing memory; the model must decide."""


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def normalize_topic(topic: str) -> str:
    """'Pay Day!' -> 'pay_day'. Consistent keys make "same topic" a simple string match."""
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:40]


def default_embed(texts: list[str]) -> list[list[float]]:
    """Embeddings from the local embedding model (config.EMBED_MODEL, via Ollama)."""
    from langchain_ollama import OllamaEmbeddings  # imported here so tests don't need Ollama

    from fin_agent import config
    # num_gpu=0 (CPU only), like the chat model: on the Intel integrated GPU this
    # embedding model returned unreliable vectors (the same pair of sentences scored
    # 0.378 on one run and 0.837 on the next).
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL,
                            num_gpu=config.OLLAMA_NUM_GPU).embed_documents(texts)


COLUMNS = "id, topic, text, kind, created, updated"


class MemoryBank:
    """Memories in a small SQLite table. `embed` is injectable so tests can use a
    fake embedding function instead of Ollama."""

    def __init__(self, path: str | Path = ":memory:",
                 embed: Callable[[list[str]], list[list[float]]] = default_embed):
        self.embed = embed
        # The agent can run several tool calls IN PARALLEL (e.g. two `remember`
        # calls from one message), so every database access takes this lock.
        # RLock = the same thread may take it again (methods call each other).
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS memories (
            id        INTEGER PRIMARY KEY,
            topic     TEXT NOT NULL UNIQUE,   -- UNIQUE: the database itself forbids two memories per topic
            text      TEXT NOT NULL,
            kind      TEXT NOT NULL,
            embedding TEXT NOT NULL,          -- JSON list of floats
            created   TEXT NOT NULL,
            updated   TEXT NOT NULL)""")
        self._migrate()
        self.db.commit()

    def _migrate(self) -> None:
        """A MIGRATION: upgrade a memories.db made by the first (numbered, no-topic)
        design instead of crashing on it. Old memories get topics memory_1,
        memory_2, ... which you can rename by re-saving them under a real topic."""
        columns = [row[1] for row in self.db.execute("PRAGMA table_info(memories)")]
        if "topic" not in columns:
            self.db.execute("ALTER TABLE memories ADD COLUMN topic TEXT")
            self.db.execute("UPDATE memories SET topic = 'memory_' || id")
        # ALTER TABLE can't add UNIQUE to a column, so a unique index enforces it instead.
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS memories_topic ON memories (topic)")

    # --- read -----------------------------------------------------------------

    def all(self) -> list[Memory]:
        with self.lock:
            rows = self.db.execute(f"SELECT {COLUMNS} FROM memories ORDER BY id").fetchall()
        return [Memory(*row) for row in rows]

    def get(self, memory_id: int) -> Memory | None:
        with self.lock:
            row = self.db.execute(f"SELECT {COLUMNS} FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return Memory(*row) if row else None

    def by_topic(self, topic: str) -> Memory | None:
        with self.lock:
            row = self.db.execute(f"SELECT {COLUMNS} FROM memories WHERE topic = ?",
                                  (normalize_topic(topic),)).fetchone()
        return Memory(*row) if row else None

    def for_prompt(self, message: str) -> list[Memory]:
        """The memories to show the model for this message."""
        memories = self.all()
        if len(memories) <= MAX_IN_PROMPT:
            return memories                    # few enough: include everything
        # Many memories: RAG over memories, i.e. the TOP_K most similar to the message.
        [query] = self.embed([message])
        scored = [(cosine(query, self._vector(m.id)), m) for m in memories]
        best = sorted(scored, key=lambda pair: pair[0], reverse=True)[:TOP_K]
        return sorted((m for _, m in best), key=lambda m: m.id)   # stable order

    # --- write ----------------------------------------------------------------

    def remember(self, topic: str, text: str, kind: str = "fact",
                 distinct: bool = False) -> tuple[Memory, Memory | None]:
        """Save a memory under `topic`. Returns (the saved memory, the one it replaced or None).

        Same topic as an existing memory -> REPLACE it (that's how updates work).
        distinct=True -> the model confirmed a new topic is NOT the same as a
        similar existing memory (after a PossibleDuplicate).
        Raises MemoryRuleError for memories that break the rules."""
        topic, text = normalize_topic(topic), " ".join(text.split())
        if not topic:
            raise MemoryRuleError("Give a short topic, e.g. 'payday' or 'answer_style'.")
        if not text:
            raise MemoryRuleError("Empty memory.")
        if kind not in KINDS:
            raise MemoryRuleError(f"kind must be one of {KINDS}.")
        if LOOKS_LIKE_A_TRANSACTION.search(text):
            raise MemoryRuleError("Not saved: that looks like a transaction. Expenses belong in the Sheet "
                                  "(use add_expense), not in memory.")
        if any(pattern.search(text) for _, pattern in PATTERNS):
            raise MemoryRuleError("Not saved: it contains a card, account, ID or contact number. "
                                  "Those are never stored as memories.")

        [vector] = self.embed([text])   # slow-ish (Ollama), so done before taking the lock
        with self.lock:
            old = self.by_topic(topic)
            if old is None:
                # A NEW topic: the safety net against the same fact under a new name.
                similar = self._similar(vector)
                if similar and similar[0][0] >= SAME_FACT:
                    old = similar[0][1]
                    topic = old.topic               # clearly the same fact: keep its topic
                elif similar and not distinct:
                    listed = "; ".join(f"[{m.topic}] '{m.text}'" for _, m in similar[:3])
                    raise PossibleDuplicate(
                        f"Not saved yet: similar to {listed}. Make exactly ONE more call: if it's the same "
                        f"subject, call remember with that topic (it replaces the old one); if it's "
                        f"genuinely different, call remember again with distinct=true.")

            now = datetime.now().isoformat(timespec="seconds")
            if old:   # an update: keep the id and creation date, replace the content
                self.db.execute("UPDATE memories SET text = ?, kind = ?, embedding = ?, updated = ? WHERE id = ?",
                                (text, kind, json.dumps(vector), now, old.id))
                memory_id = old.id
            else:
                memory_id = self.db.execute(
                    "INSERT INTO memories (topic, text, kind, embedding, created, updated) VALUES (?, ?, ?, ?, ?, ?)",
                    (topic, text, kind, json.dumps(vector), now, now)).lastrowid
            self.db.commit()
            return self.get(memory_id), old

    def forget(self, memory_id: int) -> Memory | None:
        """Delete by number (the /forget command). Returns what was deleted, or None."""
        with self.lock:
            memory = self.get(memory_id)
            if memory:
                self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                self.db.commit()
            return memory

    def forget_topic(self, topic: str) -> Memory | None:
        """Delete by topic (the model's forget_memory tool)."""
        with self.lock:
            memory = self.by_topic(topic)
            return self.forget(memory.id) if memory else None

    # --- helpers ----------------------------------------------------------------

    def _vector(self, memory_id: int) -> list[float]:
        with self.lock:
            (raw,) = self.db.execute("SELECT embedding FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return json.loads(raw)

    def _similar(self, vector: list[float]) -> list[tuple[float, Memory]]:
        """Existing memories with similarity >= SIMILAR, most similar first."""
        scored = [(cosine(vector, self._vector(m.id)), m) for m in self.all()]
        return sorted([pair for pair in scored if pair[0] >= SIMILAR], key=lambda pair: pair[0], reverse=True)
