"""
Agent tests with a SCRIPTED fake model: it returns pre-written messages instead
of thinking. This tests the GRAPH (routing, tools, pausing for confirmation,
saving) quickly and deterministically, without Ollama. How well a REAL model
chooses tools is a separate question, answered by live runs with the real model.
"""
import json
import time

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fin_agent.agent import FALSE_CLAIM_WARNING, build_agent, check_claims, is_quick_line
from fin_agent.entries import build_from_text
from fin_agent.llm import strip_thinking
from fin_agent.sheets import MemoryStore
from fin_agent.tools import prepare_expense, save_expense

NOW = ("2026-09-30", "14:00:00")  # Wednesday


class ScriptedModel(BaseChatModel):
    """Replies with the next message from `script`, in order."""
    script: list[AIMessage]
    calls: int = 0
    seen: list = []          # the messages it was shown on each call (to inspect the prompt)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        reply = self.script[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self  # the script already knows which tools to "call"


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def make(script, memory=None):
    store = MemoryStore()
    for text, day in [("200 mcdonalds", "2026-09-30"), ("900 carrefour", "2026-09-27"), ("300 kfc", "2026-08-15")]:
        save_expense(store, build_from_text(text, store.load(), (day, "12:00:00")))
    model = ScriptedModel(script=script)
    agent = build_agent(store, llm=model, checkpointer=InMemorySaver(), memory=memory, now_fn=lambda: NOW)
    return agent, store, model


CONFIG = {"configurable": {"thread_id": "test"}}


def ask(agent, text):
    return agent.invoke({"messages": [HumanMessage(text)]}, CONFIG)


def tool_results(result):
    return [m.content for m in result["messages"] if isinstance(m, ToolMessage)]


def test_question_uses_tool_then_answers():
    agent, _, model = make([call("get_spending_total", period="this_week"), AIMessage("You spent 1,100 EGP.")])
    result = ask(agent, "How much did I spend this week?")
    assert json.loads(tool_results(result)[0])["total"] == 1100   # 200 + 900 (KFC was August)
    assert result["messages"][-1].content == "You spent 1,100 EGP."
    assert model.calls == 2   # model -> tool -> model: the agent loop went round once


def test_add_expense_pauses_for_confirmation_then_saves():
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    before = len(store.load().entries)
    paused = ask(agent, "I had a 150 coffee at starbucks")
    assert paused["__interrupt__"][0].value["summary"].startswith("150 EGP · Starbucks → Coffee")
    assert len(store.load().entries) == before            # nothing saved while paused
    agent.invoke(Command(resume="yes"), CONFIG)            # the user says yes
    saved = store.load().entries[-1]
    assert (saved.merchant, saved.amount, saved.message) == ("Starbucks", 150, "I had a 150 coffee at starbucks")


def test_declining_saves_nothing():
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("OK.")])
    before = len(store.load().entries)
    ask(agent, "150 coffee at starbucks please")
    result = agent.invoke(Command(resume="no"), CONFIG)
    assert len(store.load().entries) == before
    assert "declined" in tool_results(result)[-1]


def two_expenses():
    """ONE model message carrying TWO add_expense calls ("150 starbucks and 200 mcdonalds")."""
    return AIMessage(content="", tool_calls=[
        {"name": "add_expense", "args": {"amount": 150, "description": "starbucks"}, "id": "a"},
        {"name": "add_expense", "args": {"amount": 200, "description": "mcdonalds"}, "id": "b"}])


@pytest.mark.parametrize("answers, saved", [
    (("yes", "yes"), ["Starbucks", "McDonald's"]),   # was 3 rows: Starbucks written twice
    (("yes", "no"), ["Starbucks"]),                  # was 2 rows: Starbucks written twice
    (("no", "yes"), ["McDonald's"]),
    (("no", "no"), []),
])
def test_two_expenses_in_one_message_each_get_their_own_answer(answers, saved):
    agent, store, model = make([two_expenses(), AIMessage("Done.")])
    before = len(store.load().entries)
    paused = ask(agent, "150 starbucks and 200 mcdonalds")
    assert paused["__interrupt__"][0].value["summary"].startswith("150 EGP · Starbucks")
    paused = agent.invoke(Command(resume=answers[0]), CONFIG)
    assert paused["__interrupt__"][0].value["summary"].startswith("200 EGP · McDonald's")
    result = agent.invoke(Command(resume=answers[1]), CONFIG)
    assert [e.merchant for e in store.load().entries[before:]] == saved
    # The model sees one result per call, in the order it asked, and is called once more.
    assert [r.startswith("Saved:") for r in tool_results(result)] == [a == "yes" for a in answers]
    assert model.calls == 2


def test_an_answer_goes_to_the_expense_that_was_shown_even_if_it_is_slower(monkeypatch):
    # Tool calls in one message used to run in PARALLEL threads, and the answer
    # went to whichever reached interrupt() first. Make the first one slow (as a
    # slow Sheet read would): "yes" to Starbucks must not save McDonald's.
    from fin_agent import agent as agent_module

    def slow_starbucks(snap, now, **fields):
        if fields["description"] == "starbucks":
            time.sleep(0.2)
        return prepare_expense(snap, now, **fields)

    monkeypatch.setattr(agent_module, "prepare_expense", slow_starbucks)
    agent, store, _ = make([two_expenses(), AIMessage("Done.")])
    before = len(store.load().entries)
    paused = ask(agent, "150 starbucks and 200 mcdonalds")
    assert paused["__interrupt__"][0].value["summary"].startswith("150 EGP · Starbucks")
    agent.invoke(Command(resume="yes"), CONFIG)
    agent.invoke(Command(resume="no"), CONFIG)
    assert [e.merchant for e in store.load().entries[before:]] == ["Starbucks"]


def test_unknown_merchant_asks_for_category_instead_of_saving():
    agent, store, _ = make([call("add_expense", amount=85, description="abu tarek"),
                            AIMessage("Which category is Abu Tarek?")])
    result = ask(agent, "spent 85 at abu tarek")
    assert "__interrupt__" not in result                   # no confirmation: nothing to save yet
    assert "new merchant" in tool_results(result)[0]


def test_bad_category_goes_back_to_the_model_instead_of_crashing():
    agent, _, _ = make([call("add_expense", amount=85, description="abu tarek", category="Food"),
                        AIMessage("Sorry, which category?")])
    result = ask(agent, "85 abu tarek food")
    assert "Unknown category" in tool_results(result)[0]


def test_wrong_argument_type_goes_back_to_the_model():
    # qwen3.5 once sent essential="" (text instead of true/false). That must not crash.
    agent, _, model = make([call("get_spending_breakdown", period="this_month", essential=""),
                            call("get_spending_breakdown", period="this_month", essential=False),
                            AIMessage("Fast Food is your biggest non-essential expense.")])
    result = ask(agent, "what unessential stuff do I buy?")
    assert "essential" in tool_results(result)[0].lower()      # the error text, sent to the model
    assert json.loads(tool_results(result)[1])["groups"][0]["name"] == "Fast Food"  # its retry worked
    assert model.calls == 3


def test_quick_line_skips_the_model():
    agent, store, model = make([])                         # an empty script: any model call would fail
    paused = ask(agent, "285 mcdonalds")
    assert paused["__interrupt__"][0].value["summary"].startswith("285 EGP · McDonald's → Fast Food")
    agent.invoke(Command(resume="y"), CONFIG)
    assert store.load().entries[-1].amount == 285
    assert model.calls == 0


def test_conversation_memory_across_turns():
    agent, _, _ = make([AIMessage("Hi!"), AIMessage("You said hello earlier.")])
    ask(agent, "hello")
    result = ask(agent, "what did I say?")
    assert [m.content for m in result["messages"] if isinstance(m, HumanMessage)] == ["hello", "what did I say?"]


def test_false_save_claim_gets_a_note_saying_nothing_was_saved():
    # The real lfm2.5 failure: it claimed to record an expense without calling any tool.
    claim = "I've recorded your lunch expense of 85 EGP at Abu Tarek."
    agent, store, _ = make([AIMessage(claim)])
    before = len(store.load().entries)
    result = ask(agent, "Spent 85 on lunch at abu tarek")
    # The reply is kept and the truth is added under it (a note, not a replacement).
    assert result["messages"][-1].content == f"{claim}\n\n{FALSE_CLAIM_WARNING}"
    assert len(store.load().entries) == before
    # It replaces the message in the history: there is one reply, not two.
    assert sum("I've recorded" in str(m.content) for m in result["messages"]) == 1


def test_true_save_claim_is_kept():
    agent, _, _ = make([call("add_expense", amount=150, description="starbucks"),
                        AIMessage("Your 150 EGP at Starbucks has been recorded.")])
    ask(agent, "150 at starbucks for coffee")
    result = agent.invoke(Command(resume="yes"), CONFIG)
    assert result["messages"][-1].content == "Your 150 EGP at Starbucks has been recorded."


def test_declined_and_ordinary_replies_are_left_alone():
    assert check_claims([], "OK, nothing was saved.") == "OK, nothing was saved."
    assert check_claims([], "The expense wasn't recorded.") == "The expense wasn't recorded."
    assert check_claims([], "It was not saved because you declined.") == "It was not saved because you declined."
    assert check_claims([], "You spent 70 EGP this week.") == "You spent 70 EGP this week."
    assert check_claims([], "You haven't logged any coffee this month.") == "You haven't logged any coffee this month."


@pytest.mark.parametrize("claim", [
    "The expense has been saved.",
    "I've recorded 260 EGP for Crepe box.",
    "Your 200 EGP at McDonald's has been successfully recorded.",
    "I just logged it.",
    "I added 20 EGP for road tolls.",
])
def test_fresh_save_claims_without_a_save_get_the_note(claim):
    assert check_claims([], claim) == f"{claim}\n\n{FALSE_CLAIM_WARNING}"


@pytest.mark.parametrize("honest", [
    # The real false alarm: an honest sentence about the PAST erased a legitimate answer.
    "I apologize, master. Since today is Friday, October 2nd, yesterday was Thursday, October 1st, "
    "and I incorrectly recorded the date as September 29th.",
    "The entry was recorded on the wrong date.",
    "Expenses you previously logged are in the Sheet.",
    "I cannot edit entries that were already saved.",
])
def test_talking_about_past_records_is_not_a_claim(honest):
    assert check_claims([], honest) == honest


def test_system_prompt_has_no_time_of_day_so_the_cache_works():
    from fin_agent.agent import system_prompt
    prompt = system_prompt(MemoryStore().load().categories, "2026-09-30")
    assert "Wednesday 2026-09-30" in prompt
    assert ":" not in prompt.split("\n")[1]   # no HH:MM:SS on the date line


def test_quick_line_detection():
    merchants = MemoryStore().load().merchants
    assert is_quick_line("285 mcdonalds", merchants)
    assert not is_quick_line("I paid 200 in food today (mcdonalds)", merchants)  # extra words -> model
    assert not is_quick_line("85 abu tarek", merchants)                          # unknown merchant
    assert not is_quick_line("mcdonalds", merchants)                              # no amount


def test_strip_thinking():
    assert strip_thinking("<think>hmm, let me see</think>You spent 50 EGP.") == "You spent 50 EGP."
    assert strip_thinking("Answer<think>cut off mid-thou") == "Answer"


# --- Long-term memory ------------------------------------------------------------

def memory_bank():
    from test_memory import fake_embed   # pytest puts tests/ on the import path
    from fin_agent.memory import MemoryBank
    return MemoryBank(":memory:", embed=fake_embed)


def system_text(model, call_index):
    return model.seen[call_index][0].content   # the first message of a call is the system prompt


def test_remembered_fact_is_in_the_prompt_of_a_new_conversation():
    bank = memory_bank()
    agent, _, model = make([call("remember", topic="payday", text="Gets paid on the 25th", kind="fact"),
                            AIMessage("Got it, I'll remember that."),
                            AIMessage("You get paid on the 25th.")], memory=bank)
    agent.invoke({"messages": [HumanMessage("I get paid on the 25th")]}, {"configurable": {"thread_id": "A"}})
    assert [(m.topic, m.text) for m in bank.all()] == [("payday", "Gets paid on the 25th")]
    # A DIFFERENT thread (like /new): no shared messages, but the memory is in the system prompt.
    agent.invoke({"messages": [HumanMessage("when do I get paid?")]}, {"configurable": {"thread_id": "B"}})
    assert "[payday] Gets paid on the 25th (fact)" in system_text(model, 2)


def test_forget_memory_tool():
    bank = memory_bank()
    bank.remember("home_city", "Lives in Cairo", "fact")
    agent, _, _ = make([call("forget_memory", topic="home_city"), AIMessage("Forgotten.")], memory=bank)
    result = ask(agent, "forget where I live")
    assert bank.all() == []
    assert "Forgot [home_city]" in tool_results(result)[0]


def test_remembering_a_transaction_is_refused_and_explained_to_the_model():
    bank = memory_bank()
    agent, _, _ = make([call("remember", topic="food", text="Spent 500 on food yesterday", kind="fact"),
                        AIMessage("That's an expense, not something to remember.")], memory=bank)
    result = ask(agent, "remember I spent 500 on food")
    assert bank.all() == []
    assert "transaction" in tool_results(result)[0]


def test_without_memory_there_are_no_memory_tools_or_section():
    agent, _, model = make([AIMessage("Hi!")])
    ask(agent, "hello")
    assert "What you know about the user" not in system_text(model, 0)


def test_forget_is_refused_when_the_user_did_not_ask_to_forget():
    # Live run: "my payday moved to the 1st" -> the model DELETED [payday] and never saved the new value.
    bank = memory_bank()
    bank.remember("payday", "Gets paid on the 25th", "fact")
    agent, _, _ = make([call("forget_memory", topic="payday"),
                        call("remember", topic="payday", text="Gets paid on the 1st", kind="fact"),
                        AIMessage("Your payday is now the 1st.")], memory=bank)
    result = ask(agent, "Actually my payday moved to the 1st of the month.")
    assert "Not deleted" in tool_results(result)[0]
    assert [(m.topic, m.text) for m in bank.all()] == [("payday", "Gets paid on the 1st")]


def test_false_memory_claim_gets_a_note():
    # Live run: after only deleting, the model said "Your payday has been updated to the 1st".
    from fin_agent.agent import FALSE_MEMORY_WARNING
    bank = memory_bank()
    claim = "Your payday has been updated to the 1st of each month."
    agent, _, _ = make([AIMessage(claim)], memory=bank)
    result = ask(agent, "Actually my payday moved to the 1st of the month.")
    assert result["messages"][-1].content == f"{claim}\n\n{FALSE_MEMORY_WARNING}"


def test_true_memory_claim_is_kept():
    bank = memory_bank()
    agent, _, _ = make([call("remember", topic="payday", text="Gets paid on the 1st", kind="fact"),
                        AIMessage("Noted, your payday is the 1st.")], memory=bank)
    result = ask(agent, "my payday is the 1st")
    assert result["messages"][-1].content == "Noted, your payday is the 1st."


# --- Analysis tools (fake Sheet from make(): 200 McDonald's 30/9, 900 Carrefour 27/9, 300 KFC 15/8) ---

def test_analyze_unessential_tool():
    agent, _, _ = make([call("analyze_unessential", period="this_month"), AIMessage("Fast Food.")])
    r = json.loads(tool_results(ask(agent, "what unessential stuff do I buy?"))[0])
    assert r["compared_with"] == {"start": "2026-08-01", "end": "2026-08-30"}
    assert r["non_essential_total"]["now"] == 200 and r["non_essential_total"]["before"] == 300


def test_check_budgets_tool():
    agent, store, _ = make([call("check_budgets"), AIMessage("Fine.")])
    for row in store.tabs["Categories"][1:]:
        if row[0] == "Fast Food":
            row[2] = 250
    r = json.loads(tool_results(ask(agent, "am I over budget?"))[0])
    assert r["budgets"][0]["category"] == "Fast Food" and r["budgets"][0]["status"] == "warning"   # 200/250 = 80%


def test_unusual_and_weekly_review_tools():
    agent, _, _ = make([call("find_unusual_spending", days=30), call("get_weekly_review"), AIMessage("Quiet week.")])
    results = tool_results(ask(agent, "anything unusual, and how was my week?"))
    assert json.loads(results[0])["possible_duplicates"] == []
    assert json.loads(results[1])["total"]["now"] == 1100     # 200 + 900 this week


def test_prompt_lists_category_names_exactly_as_tools_expect_them():
    from fin_agent.agent import system_prompt
    prompt = system_prompt(MemoryStore().load().categories, "2026-09-30")
    assert "(non-essential)" not in prompt                       # no label a model could copy into a tool call
    assert "Non-essential categories: Fast Food, Coffee" in prompt
    assert "Essential categories: Groceries" in prompt


def test_unknown_category_in_a_question_goes_back_to_the_model_not_zero():
    agent, _, _ = make([call("compare_spending", period_a="this_month", period_b="last_month", category="Takeaway"),
                        call("compare_spending", period_a="this_month", period_b="last_month", category="Fast Food"),
                        AIMessage("You spent 200 EGP this month vs 300 EGP last month.")])
    results = tool_results(ask(agent, "more fast food this month than last?"))
    assert "Unknown category" in results[0]
    assert json.loads(results[1])["period_b"]["total"] == 300


# --- the day of an expense is decided by code, not by the model (NOW is Wed 2026-09-30) ---

def test_users_yesterday_overrides_a_wrong_date_from_the_model():
    # The real bug: the user said "yesterday" and the model sent a date three days back.
    agent, store, _ = make([call("add_expense", amount=260, description="mcdonalds", date="2026-09-27"),
                            AIMessage("Done.")])
    paused = ask(agent, "I paid 260 for food yesterday, ordered mcdonalds.")
    assert "Tue 2026-09-29 (yesterday)" in paused["__interrupt__"][0].value["summary"]
    agent.invoke(Command(resume="yes"), CONFIG)
    assert store.load().entries[-1].date == "2026-09-29"


def test_model_passing_the_word_yesterday_is_resolved_by_code():
    # A follow-up turn: the user's last message has no day word, so the model's `date` is used.
    agent, store, _ = make([call("add_expense", amount=85, description="abu tarek", category="Fast Food", date="yesterday"),
                            AIMessage("Done.")])
    ask(agent, "Fast Food")
    agent.invoke(Command(resume="yes"), CONFIG)
    assert store.load().entries[-1].date == "2026-09-29"


def test_exact_date_from_the_user_is_kept_and_shown_with_its_weekday():
    agent, _, _ = make([call("add_expense", amount=50, description="kfc", date="2026-09-20"), AIMessage("Done.")])
    paused = ask(agent, "I paid 50 at kfc on 2026-09-20")
    assert "Sun 2026-09-20 (10 days ago)" in paused["__interrupt__"][0].value["summary"]


def test_two_day_words_fall_back_to_the_models_date():
    agent, _, _ = make([call("add_expense", amount=50, description="kfc", date="today"), AIMessage("Done.")])
    paused = ask(agent, "yesterday I forgot to say: 50 at kfc today")
    assert "Wed 2026-09-30 (today)" in paused["__interrupt__"][0].value["summary"]


def test_future_date_is_refused_and_explained_to_the_model():
    agent, store, _ = make([call("add_expense", amount=50, description="kfc", date="2026-10-05"),
                            AIMessage("That date is in the future.")])
    before = len(store.load().entries)
    result = ask(agent, "log 50 at kfc")
    assert "in the future" in tool_results(result)[0]
    assert len(store.load().entries) == before
