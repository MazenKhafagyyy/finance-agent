"""The 10-minute session rule, tested with a made-up clock (no waiting, no Discord)."""
from datetime import datetime, timedelta, timezone

from fin_agent.sessions import SessionBook

ME = 111
T0 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)


def after(**delta) -> datetime:
    return T0 + timedelta(**delta)


def test_first_message_starts_a_session():
    thread, is_new = SessionBook().current_thread(ME, T0)
    assert is_new and len(thread) == 32


def test_message_within_ten_minutes_of_the_reply_continues_the_session():
    book = SessionBook()
    thread, _ = book.current_thread(ME, T0)
    book.touch(ME, T0)                                       # the bot replied at 12:00
    assert book.current_thread(ME, after(minutes=10)) == (thread, False)   # exactly 10 min: still the same


def test_message_after_ten_minutes_of_silence_starts_a_new_session():
    book = SessionBook()
    thread, _ = book.current_thread(ME, T0)
    book.touch(ME, T0)
    later, is_new = book.current_thread(ME, after(minutes=10, seconds=1))
    assert is_new and later != thread


def test_the_clock_counts_from_the_bots_reply_not_from_your_message():
    book = SessionBook()
    thread, _ = book.current_thread(ME, T0)
    book.touch(ME, T0)
    book.current_thread(ME, after(minutes=8))                # you write at 12:08...
    book.touch(ME, after(minutes=9))                         # ...the bot answers at 12:09
    # 12:18:30 is 10.5 min after YOUR message but only 9.5 min after the REPLY.
    assert book.current_thread(ME, after(minutes=18, seconds=30)) == (thread, False)


def test_a_new_session_becomes_the_current_one():
    book = SessionBook()
    book.current_thread(ME, T0)
    book.touch(ME, T0)
    second, _ = book.current_thread(ME, after(minutes=30))
    book.touch(ME, after(minutes=31))
    assert book.current_thread(ME, after(minutes=35)) == (second, False)


def test_session_survives_a_restart(tmp_path):
    path = tmp_path / "sessions.db"
    first = SessionBook(path)
    thread, _ = first.current_thread(ME, T0)
    first.touch(ME, T0)
    reopened = SessionBook(path)                             # the bot was restarted
    assert reopened.current_thread(ME, after(minutes=5)) == (thread, False)
