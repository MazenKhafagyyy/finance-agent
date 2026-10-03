"""One message in, one plain result out: the part of a chat UI that has no UI.
Uses the scripted fake model and the fake Sheet from test_agent.py."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fin_agent.turns import answer_confirmation, run_turn
from test_agent import call, make

T = "thread-1"
STARBUCKS = "150 EGP · Starbucks → Coffee (non-essential) · Wed 2026-09-30 (today)"


def usage(read, written):
    return {"input_tokens": read, "output_tokens": written, "total_tokens": read + written}


def history(agent, thread=T):
    return agent.get_state({"configurable": {"thread_id": thread}}).values["messages"]


def test_a_question_returns_the_reply_with_its_tools_and_tokens():
    first = call("get_spending_total", period="this_week")
    first.usage_metadata = usage(1000, 20)
    agent, _, _ = make([first, AIMessage("You spent 1,100 EGP.", usage_metadata=usage(1100, 9))])
    result = run_turn(agent, T, "How much did I spend this week?")
    assert result.reply == "You spent 1,100 EGP."
    assert result.tools == ["get_spending_total"]
    assert result.tokens == 2129
    assert result.confirm == "" and result.declined == []


def test_tools_and_tokens_cover_only_the_latest_message():
    agent, _, _ = make([call("get_spending_total", period="this_week"), AIMessage("1,100 EGP."),
                        AIMessage("You're welcome.", usage_metadata=usage(50, 4))])
    run_turn(agent, T, "How much this week?")
    result = run_turn(agent, T, "thanks")
    assert result.tools == [] and result.tokens == 54


def test_an_expense_waits_for_confirmation_and_saves_nothing():
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    before = len(store.load().entries)
    result = run_turn(agent, T, "I had a 150 coffee at starbucks")
    assert result.confirm == STARBUCKS and len(result.confirm_id) == 32
    assert result.reply == ""
    assert len(store.load().entries) == before


def test_answering_yes_saves_and_returns_the_reply():
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    before = len(store.load().entries)
    waiting = run_turn(agent, T, "I had a 150 coffee at starbucks")
    result = answer_confirmation(agent, T, waiting.confirm_id, "yes")
    assert len(store.load().entries) == before + 1
    assert result.reply == "Saved it." and result.confirm == ""
    assert result.tools == ["add_expense"]                 # the whole turn, including the part before the pause


def test_answering_no_saves_nothing():
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("OK, not saved.")])
    before = len(store.load().entries)
    waiting = run_turn(agent, T, "I had a 150 coffee at starbucks")
    result = answer_confirmation(agent, T, waiting.confirm_id, "no")
    assert len(store.load().entries) == before
    assert result.reply == "OK, not saved."


def test_a_second_click_on_an_answered_question_does_nothing():
    agent, store, model = make([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    waiting = run_turn(agent, T, "I had a 150 coffee at starbucks")
    answer_confirmation(agent, T, waiting.confirm_id, "yes")
    saved, calls = len(store.load().entries), model.calls
    assert answer_confirmation(agent, T, waiting.confirm_id, "yes") is None
    assert (len(store.load().entries), model.calls) == (saved, calls)


def test_an_old_button_cannot_answer_a_newer_question():
    # Real risk: the Yes button under expense A is still on screen when expense B
    # is waiting in the same conversation. Clicking it must not save B.
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("OK, not saved."),
                            call("add_expense", amount=200, description="mcdonalds"), AIMessage("Saved it.")])
    first = run_turn(agent, T, "150 coffee at starbucks")
    answer_confirmation(agent, T, first.confirm_id, "no")
    second = run_turn(agent, T, "200 at mcdonalds")
    before = len(store.load().entries)
    assert answer_confirmation(agent, T, first.confirm_id, "yes") is None
    assert len(store.load().entries) == before
    assert answer_confirmation(agent, T, second.confirm_id, "yes").reply == "Saved it."


def test_typing_yes_answers_the_waiting_question():
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("Saved it.")])
    before = len(store.load().entries)
    run_turn(agent, T, "I had a 150 coffee at starbucks")
    result = run_turn(agent, T, " Yes ")
    assert len(store.load().entries) == before + 1
    assert result.reply == "Saved it."


def test_typing_no_declines_without_starting_a_new_turn():
    agent, store, model = make([call("add_expense", amount=150, description="starbucks"), AIMessage("OK, not saved.")])
    before = len(store.load().entries)
    run_turn(agent, T, "I had a 150 coffee at starbucks")
    result = run_turn(agent, T, "no")
    assert len(store.load().entries) == before
    assert result.reply == "OK, not saved." and result.declined == []
    assert model.calls == 2                                # "no" was the answer, not a new question


def test_a_different_message_cancels_the_waiting_save_then_gets_answered():
    agent, store, _ = make([call("add_expense", amount=150, description="starbucks"), AIMessage("OK, not saved."),
                            call("get_spending_total", period="this_week"), AIMessage("You spent 1,100 EGP.")])
    before = len(store.load().entries)
    run_turn(agent, T, "I had a 150 coffee at starbucks")
    result = run_turn(agent, T, "How much did I spend this week?")
    assert len(store.load().entries) == before             # the coffee was NOT saved
    assert result.declined == [STARBUCKS]                  # ...and the result says so
    assert result.reply == "You spent 1,100 EGP."
    assert result.tools == ["get_spending_total"]
    # Every tool call in the history got its result: no question is left dangling
    # in front of the new message (a model can choke on an unanswered tool call).
    messages = history(agent)
    asked = [c["id"] for m in messages for c in getattr(m, "tool_calls", None) or []]
    answered = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert asked == answered
    new_question = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    assert isinstance(messages[new_question - 1], AIMessage) and messages[new_question - 1].content == "OK, not saved."


def test_the_quick_path_also_waits_then_saves_without_the_model():
    agent, store, model = make([])
    before = len(store.load().entries)
    waiting = run_turn(agent, T, "285 mcdonalds")
    assert waiting.confirm.startswith("285 EGP · McDonald's") and len(store.load().entries) == before
    result = answer_confirmation(agent, T, waiting.confirm_id, "yes")
    assert result.reply.startswith("Saved: 285 EGP")
    assert len(store.load().entries) == before + 1
    assert (model.calls, result.tokens, result.tools) == (0, 0, [])


def test_separate_threads_do_not_share_history():
    agent, _, model = make([AIMessage("Hi."), AIMessage("Hello.")])
    run_turn(agent, "thread-a", "hello")
    run_turn(agent, "thread-b", "hello again")
    assert [type(m) for m in model.seen[1][1:]] == [HumanMessage]   # after the system prompt: only the new message
