"""The Discord bot's plain functions: who may talk to it, and how a result
becomes message text. Nothing here connects to Discord."""
from fin_agent.app import forget_text, memories_text
from fin_agent.discord_bot import allowed, confirm_text, render
from fin_agent.memory import MemoryBank
from fin_agent.turns import TurnResult
from test_memory import fake_embed

OWNER = 111
COFFEE = "150 EGP · Starbucks → Coffee (non-essential) · Wed 2026-09-30 (today)"


# --- who may talk to it -----------------------------------------------------------

def test_only_the_owners_direct_messages_are_handled():
    assert allowed(author_id=OWNER, is_bot=False, is_dm=True, owner_id=OWNER)


def test_a_stranger_who_dms_the_bot_is_ignored():
    assert not allowed(author_id=222, is_bot=False, is_dm=True, owner_id=OWNER)


def test_the_owner_in_a_server_channel_is_ignored():
    # Finance answers must never be posted where other people could read them.
    assert not allowed(author_id=OWNER, is_bot=False, is_dm=False, owner_id=OWNER)


def test_bots_are_ignored_even_with_the_owners_id():
    assert not allowed(author_id=OWNER, is_bot=True, is_dm=True, owner_id=OWNER)


def test_nobody_is_allowed_when_no_owner_is_configured():
    assert not allowed(author_id=0, is_bot=False, is_dm=True, owner_id=0)


# --- result -> message text ---------------------------------------------------------

def test_reply_gets_one_small_footer_line():
    result = TurnResult(reply="You spent 1,240 EGP this week.", tools=["get_spending_breakdown"],
                        seconds=3.21, tokens=2672)
    assert render(result) == ["You spent 1,240 EGP this week.\n\n-# get_spending_breakdown · 3.2s · 2,672 tokens"]


def test_footer_lists_every_tool_and_omits_the_list_when_none_ran():
    two = TurnResult(reply="Done.", tools=["get_spending_total", "check_budgets"], seconds=5, tokens=4000)
    assert render(two) == ["Done.\n\n-# get_spending_total, check_budgets · 5.0s · 4,000 tokens"]
    none = TurnResult(reply="Saved: 285 EGP · McDonald's", seconds=0.4, tokens=0)
    assert render(none) == ["Saved: 285 EGP · McDonald's\n\n-# 0.4s · 0 tokens"]


def test_a_cancelled_save_is_stated_before_the_reply():
    result = TurnResult(reply="You spent 1,100 EGP.", declined=[COFFEE], tools=["get_spending_total"],
                        seconds=2, tokens=900)
    assert render(result) == [f"🚫 Not saved: {COFFEE}\n\nYou spent 1,100 EGP.\n\n"
                              f"-# get_spending_total · 2.0s · 900 tokens"]


def test_an_empty_reply_is_not_sent_as_an_empty_message():
    [text] = render(TurnResult(reply="", seconds=1, tokens=10))
    assert text.startswith("(The model returned an empty reply")      # Discord rejects empty messages


def test_long_reply_is_split_at_line_breaks_under_discords_limit():
    a, b, c = "a" * 1500, "b" * 1500, "c" * 1500
    parts = render(TurnResult(reply=f"{a}\n{b}\n{c}", seconds=1, tokens=5))
    assert parts == [a, b, f"{c}\n\n-# 1.0s · 5 tokens"]


def test_one_huge_line_is_cut_so_nothing_exceeds_the_limit():
    parts = render(TurnResult(reply="x" * 4100, seconds=1, tokens=5))
    assert [len(p) for p in parts[:2]] == [2000, 2000]
    assert parts[2] == "x" * 100 + "\n\n-# 1.0s · 5 tokens"


def test_confirmation_question_shows_the_summary_and_any_cancelled_save():
    assert confirm_text(TurnResult(confirm=COFFEE, confirm_id="c1")) == f"❓ **Save this?**\n{COFFEE}"
    both = confirm_text(TurnResult(confirm=COFFEE, confirm_id="c1", declined=["200 EGP · KFC"]))
    assert both == f"🚫 Not saved: 200 EGP · KFC\n\n❓ **Save this?**\n{COFFEE}"


# --- /memories and /forget ----------------------------------------------------------

def bank():
    memory = MemoryBank(":memory:", embed=fake_embed)
    memory.remember("payday", "Gets paid on the 25th", "fact")
    memory.remember("answer_style", "Prefers short answers", "preference")
    return memory


def test_memories_are_listed_with_number_topic_and_kind():
    lines = memories_text(bank()).splitlines()
    assert lines[0].startswith("#1  [payday]  Gets paid on the 25th  (fact, updated ")
    assert lines[1].startswith("#2  [answer_style]  Prefers short answers  (preference, updated ")


def test_no_memories_says_so():
    assert memories_text(MemoryBank(":memory:", embed=fake_embed)).startswith("(no memories yet")


def test_forget_by_number_or_topic_deletes_that_memory():
    memory = bank()
    assert forget_text(memory, "#1") == "Forgot [payday]: Gets paid on the 25th"
    assert forget_text(memory, "answer_style") == "Forgot [answer_style]: Prefers short answers"
    assert memory.all() == []


def test_forgetting_something_unknown_changes_nothing():
    memory = bank()
    assert forget_text(memory, "salary") == "No memory 'salary'."
    assert len(memory.all()) == 2


# --- the Discord glue, driven with stand-ins for Discord's objects ---------------------
# A real agent (scripted model, fake Sheet) behind the real bot code. Only the
# network edge is replaced: a channel that records what was sent instead of sending.

import asyncio  # noqa: E402
import re  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import discord  # noqa: E402
import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from fin_agent import discord_bot  # noqa: E402
from fin_agent.discord_bot import ConfirmButton, ExoBot  # noqa: E402
from fin_agent.sessions import SessionBook  # noqa: E402
from test_agent import call, make  # noqa: E402

T0 = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)


class FakeDM(discord.DMChannel):
    """A direct-message channel that records messages instead of sending them."""

    def __init__(self):
        self.sent = []          # (text, view) pairs, in order

    @asynccontextmanager
    async def typing(self):
        yield

    async def send(self, content=None, *, view=None):
        self.sent.append((content, view))

    def texts(self):
        return [text for text, _ in self.sent]


class FakeResponse:
    def __init__(self):
        self.edits, self.messages = [], []

    async def edit_message(self, *, content, view):
        self.edits.append((content, view))

    async def send_message(self, content, ephemeral=False):
        self.messages.append(content)


@pytest.fixture
def clock(monkeypatch):
    """The bot's clock, movable by the test: clock["now"] += timedelta(minutes=9)."""
    state = {"now": T0}
    monkeypatch.setattr(discord_bot, "utc_now", lambda: state["now"])
    return state


def bot_with(script):
    agent, store, model = make(script)
    bot = ExoBot(agent, MemoryBank(":memory:", embed=fake_embed), SessionBook(), OWNER)
    return bot, store, model, FakeDM()


def dm(bot, channel, text, author_id=OWNER):
    message = SimpleNamespace(author=SimpleNamespace(id=author_id, bot=False), channel=channel, content=text)
    asyncio.run(bot.on_message(message))


def click(bot, channel, button, user_id=OWNER):
    question = channel.sent[-1][0]
    interaction = SimpleNamespace(client=bot, user=SimpleNamespace(id=user_id), response=FakeResponse(),
                                  message=SimpleNamespace(content=question), channel=channel)
    asyncio.run(button.callback(interaction))
    return interaction


def test_a_dm_from_the_owner_is_answered_with_a_footer(clock):
    bot, _, _, channel = bot_with([call("get_spending_total", period="this_week"), AIMessage("You spent 1,100 EGP.")])
    dm(bot, channel, "How much did I spend this week?")
    [text] = channel.texts()
    assert text.startswith("You spent 1,100 EGP.\n\n-# get_spending_total · ")


def test_a_stranger_gets_no_reply_and_the_agent_never_runs(clock):
    bot, _, model, channel = bot_with([AIMessage("You spent 1,100 EGP.")])
    dm(bot, channel, "How much did he spend this week?", author_id=222)
    assert channel.sent == [] and model.calls == 0


def test_an_expense_shows_buttons_and_yes_saves_it(clock):
    bot, store, _, channel = bot_with([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    before = len(store.load().entries)
    dm(bot, channel, "I had a 150 coffee at starbucks")
    text, view = channel.sent[-1]
    assert text == f"❓ **Save this?**\n{COFFEE}"
    yes, no = view.children
    assert (yes.answer, no.answer) == ("yes", "no") and len(store.load().entries) == before

    interaction = click(bot, channel, yes)
    assert len(store.load().entries) == before + 1
    assert interaction.response.edits == [(f"❓ **Save this?**\n{COFFEE}\n→ ✅ Yes", None)]   # buttons removed
    assert channel.texts()[-1].startswith("Saved it.\n\n-# add_expense · ")


def test_no_button_saves_nothing(clock):
    bot, store, _, channel = bot_with([call("add_expense", amount=150, description="starbucks"), AIMessage("OK, not saved.")])
    before = len(store.load().entries)
    dm(bot, channel, "I had a 150 coffee at starbucks")
    click(bot, channel, channel.sent[-1][1].children[1])
    assert len(store.load().entries) == before
    assert channel.texts()[-1].startswith("OK, not saved.")


def test_someone_elses_click_saves_nothing(clock):
    bot, store, _, channel = bot_with([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    before = len(store.load().entries)
    dm(bot, channel, "I had a 150 coffee at starbucks")
    interaction = click(bot, channel, channel.sent[-1][1].children[0], user_id=222)
    assert len(store.load().entries) == before
    assert interaction.response.messages == ["This isn't your assistant."] and interaction.response.edits == []


def test_a_button_still_works_after_a_restart(clock):
    # After a restart the bot has no button objects: Discord sends back only the
    # button's id text, which must match the template and rebuild the same button.
    bot, store, _, channel = bot_with([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    before = len(store.load().entries)
    dm(bot, channel, "I had a 150 coffee at starbucks")
    custom_id = channel.sent[-1][1].children[0].custom_id
    assert len(custom_id) <= 100                               # Discord's limit for a button id
    match =ConfirmButton.__discord_ui_compiled_template__.fullmatch(custom_id)
    rebuilt = asyncio.run(ConfirmButton.from_custom_id(None, None, match))
    click(bot, channel, rebuilt)
    assert len(store.load().entries) == before + 1


def test_a_stale_button_changes_nothing_and_says_so(clock):
    bot, store, _, channel = bot_with([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    dm(bot, channel, "I had a 150 coffee at starbucks")
    yes = channel.sent[-1][1].children[0]
    click(bot, channel, yes)
    saved = len(store.load().entries)
    click(bot, channel, yes)                                   # the same button again
    assert len(store.load().entries) == saved
    assert channel.texts()[-1] == "That question was already answered or cancelled, so nothing changed."


def history_shown_to_model(model, call_number):
    """How many chat messages (after the system prompt) the model saw on that call."""
    return len(model.seen[call_number]) - 1


def test_messages_within_ten_minutes_of_each_reply_share_a_conversation(clock):
    bot, _, model, channel = bot_with([AIMessage("One."), AIMessage("Two."), AIMessage("Three."), AIMessage("Four.")])
    dm(bot, channel, "first")
    clock["now"] += timedelta(minutes=9)
    dm(bot, channel, "second")
    clock["now"] += timedelta(minutes=9)                       # 18 min after the start, 9 after the last reply
    dm(bot, channel, "third")
    assert history_shown_to_model(model, 2) == 5               # first, One., second, Two., third
    clock["now"] += timedelta(minutes=11)
    dm(bot, channel, "fourth")
    assert history_shown_to_model(model, 3) == 1               # a new session: only "fourth"


def test_a_failed_turn_reports_the_error_and_the_bot_keeps_working(clock):
    bot, _, _, channel = bot_with([AIMessage("Hi.")])           # the script runs out on the 2nd message
    dm(bot, channel, "hello")
    dm(bot, channel, "this one makes the fake model fail")
    assert channel.texts()[-1].startswith("⚠️ Something went wrong (IndexError).")
    dm(bot, channel, "and the bot is not stuck")                # would hang forever if the lock were still held
    assert len(channel.sent) == 3
