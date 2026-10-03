"""
SESSIONS: which conversation does a new message belong to?

In the terminal a conversation lasts until you type /new. A chat app has no
such moment: you just send a message, maybe hours after the last one. So the
rule here is about SILENCE:

    a message within 10 minutes of the bot's last reply   -> same conversation
    a message after more than 10 minutes of silence       -> a new conversation

"Conversation" means a LangGraph THREAD: the `thread_id` the checkpointer saves
messages under. A new session is simply a new thread_id, so the model starts
with an empty history (long-term memories are separate and carry over).

Why bother? The whole history is re-sent to the model on every call. Short
sessions keep that small, and an hour-old question rarely matters to the next one.

TWO DESIGN CHOICES
  1. The clock starts when the BOT finishes replying (`touch`), not when you
     send. A slow reply must not eat into your 10 minutes.
  2. There is NO timer running in the background. The check happens only when
     a message arrives ("is the last reply more than 10 minutes old?"). That is
     called LAZY expiry: nothing to schedule, nothing to cancel, and because
     the row is on disk it still works after the program restarts.
"""
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

IDLE_MINUTES = 10


class SessionBook:
    """One row per user: their current thread_id and when the bot last replied.

    `now` is always PASSED IN rather than read from the clock in here, so tests
    can say "11 minutes later" without waiting (the same idea as the agent's
    `now_fn`). Pass timezone-aware UTC times: the Cairo clock changing can then
    never stretch or shrink the 10 minutes.
    """

    def __init__(self, path: str | Path = ":memory:", idle_minutes: int = IDLE_MINUTES):
        self.idle = timedelta(minutes=idle_minutes)
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS sessions (
            user_id       INTEGER PRIMARY KEY,
            thread_id     TEXT NOT NULL,
            last_activity TEXT NOT NULL)""")       # ISO text, e.g. 2026-10-02T12:00:00+00:00
        self.db.commit()

    def current_thread(self, user_id: int, now: datetime) -> tuple[str, bool]:
        """(thread_id, is_new) for a message arriving at `now`."""
        row = self.db.execute("SELECT thread_id, last_activity FROM sessions WHERE user_id = ?",
                              (user_id,)).fetchone()
        if row and now - datetime.fromisoformat(row[1]) <= self.idle:
            return row[0], False
        # No session yet, or too much silence: start a fresh thread. The old
        # one stays in the checkpointer's database; it's just never used again.
        thread_id = uuid.uuid4().hex
        self.db.execute("INSERT OR REPLACE INTO sessions (user_id, thread_id, last_activity) VALUES (?, ?, ?)",
                        (user_id, thread_id, now.isoformat()))
        self.db.commit()
        return thread_id, True

    def touch(self, user_id: int, now: datetime) -> None:
        """The bot just replied: the 10 minutes start again from `now`."""
        self.db.execute("UPDATE sessions SET last_activity = ? WHERE user_id = ?", (now.isoformat(), user_id))
        self.db.commit()
