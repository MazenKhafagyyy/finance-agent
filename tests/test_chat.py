"""Exercise live chunks through the real graph without Ollama or Google Sheets."""
from io import StringIO

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from fin_agent.agent import FALSE_CLAIM_WARNING, build_agent
from fin_agent.chat import run_turn
from fin_agent.llm import chat_model
from fin_agent.sheets import MemoryStore
from fin_agent.stream_output import GRAY, RESET, TokenPrinter

CONFIG = {"configurable": {"thread_id": "stream-test"}}
NOW = ("2026-09-30", "14:00:00")


class StreamingModel(BaseChatModel):
    script: list[list[AIMessageChunk]]
    calls: int = 0
    fail: bool = False

    @property
    def _llm_type(self):
        return "streaming-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("The graph should request streaming from the model")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        chunks = self.script[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield ChatGenerationChunk(message=chunk)
        if self.fail:
            raise RuntimeError("connection interrupted")


def graph(model):
    store = MemoryStore()
    agent = build_agent(store, llm=model, checkpointer=InMemorySaver(), now_fn=lambda: NOW)
    return agent, store


def test_reasoning_and_answer_are_flushed_and_colors_reset():
    out = StringIO()
    printer = TokenPrinter(out, color=True)
    printer.feed(AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Let me check."}))
    assert out.getvalue() == "\n" + GRAY + "  thinking> Let me check."
    printer.feed(AIMessageChunk(content="Hello"))
    assert RESET + "\n\nassistant> Hello" in out.getvalue()
    printer.feed(AIMessageChunk(content="!"))
    assert out.getvalue().endswith("Hello!")  # visible BEFORE finish()
    assert printer.finish() == "Hello!"


def test_inline_thinking_tags_split_at_every_character():
    out = StringIO()
    printer = TokenPrinter(out, color=True)
    for char in "<think>Check the numbers.</think>You spent 50 EGP.":
        printer.feed(AIMessageChunk(content=char))
    assert printer.finish() == "You spent 50 EGP."
    assert "<think>" not in out.getvalue() and "</think>" not in out.getvalue()
    assert GRAY + "  thinking> Check the numbers." + RESET in out.getvalue()
    assert "assistant> You spent 50 EGP." in out.getvalue()


def test_unclosed_thinking_resets_color_without_becoming_an_answer():
    out = StringIO()
    printer = TokenPrinter(out, color=True)
    printer.feed(AIMessageChunk(content="<think>Still thinking"))
    assert printer.finish() == ""
    assert out.getvalue().endswith(RESET + "\n")
    printer.finish()
    assert out.getvalue().count(RESET) == 1


def test_redirected_output_is_plain_and_preserves_non_tag_text():
    out = StringIO()
    printer = TokenPrinter(out)
    printer.feed(AIMessageChunk(content="<think>Check</think>Amount < threshold"))
    assert printer.finish() == "Amount < threshold"
    assert "\033[" not in out.getvalue()


def test_real_graph_streams_answer_once_and_does_not_save_reasoning(capsys):
    model = StreamingModel(script=[[
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Consider greeting."}),
        AIMessageChunk(content="Hello"), AIMessageChunk(content=" there!"),
    ]])
    agent, _ = graph(model)
    run_turn(agent, {"messages": [HumanMessage("hello")]}, CONFIG)
    output = capsys.readouterr().out
    assert "thinking> Consider greeting." in output
    assert output.count("Hello there!") == 1
    assert "model 1x" in output
    final = agent.get_state(CONFIG).values["messages"][-1]
    assert final.content == "Hello there!"
    assert "reasoning_content" not in final.additional_kwargs


def test_guardrail_visibly_corrects_a_streamed_false_claim(capsys):
    agent, _ = graph(StreamingModel(script=[[
        AIMessageChunk(content="I've recorded "), AIMessageChunk(content="your lunch."),
    ]]))
    run_turn(agent, {"messages": [HumanMessage("I spent 85 at abu tarek")]}, CONFIG)
    output = capsys.readouterr().out
    # The streamed claim stays on screen once; only the added note is printed after it.
    assert output.count("I've recorded your lunch.") == 1
    assert f"\n{FALSE_CLAIM_WARNING}" in output
    assert agent.get_state(CONFIG).values["messages"][-1].content == f"I've recorded your lunch.\n\n{FALSE_CLAIM_WARNING}"


@pytest.mark.parametrize("answer, expected_count", [("yes", 1), ("no", 0)])
def test_tool_confirmation_and_streamed_reply(monkeypatch, capsys, answer, expected_count):
    call = AIMessageChunk(content="", tool_call_chunks=[{
        "name": "add_expense", "args": '{"amount":150,"description":"starbucks"}',
        "id": "expense", "index": 0,
    }])
    agent, store = graph(StreamingModel(script=[[call], [AIMessageChunk(content="All done.")]]))
    prompts = []

    def confirm(prompt):
        assert len(store.load().entries) == 0
        prompts.append(prompt)
        return answer

    monkeypatch.setattr("builtins.input", confirm)
    run_turn(agent, {"messages": [HumanMessage("I had coffee for 150 at starbucks")]}, CONFIG)
    output = capsys.readouterr().out
    assert len(prompts) == 1 and "150 EGP" in prompts[0]
    assert len(store.load().entries) == expected_count
    assert "add_expense(" in output and "model 2x" in output
    assert output.count("All done.") == 1


def test_quick_add_still_prints_without_model_tokens(monkeypatch, capsys):
    model = StreamingModel(script=[])
    agent, store = graph(model)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    run_turn(agent, {"messages": [HumanMessage("285 mcdonalds")]}, CONFIG)
    assert "assistant> Saved:" in capsys.readouterr().out
    assert model.calls == 0 and len(store.load().entries) == 1


def test_error_during_thinking_restores_terminal_color(monkeypatch, capsys):
    monkeypatch.setattr("fin_agent.stream_output.terminal_color", lambda _: True)
    agent, _ = graph(StreamingModel(script=[[
        AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Checking..."}),
    ]], fail=True))
    with pytest.raises(RuntimeError, match="connection interrupted"):
        run_turn(agent, {"messages": [HumanMessage("hello")]}, CONFIG)
    output = capsys.readouterr().out
    # After an error the terminal must be left clean in BOTH ways: the grey
    # "thinking" colour is reset, and then the status line gives its row back.
    release_status_row = SAVE_CURSOR + RESET_SCROLL_REGION
    assert "Checking..." + RESET + "\n" in output
    assert output.index("Checking..." + RESET) < output.rindex(release_status_row)
    assert output.endswith(RESTORE_CURSOR)


def test_terminal_can_enable_reasoning_without_changing_other_callers():
    assert chat_model().reasoning is False
    assert chat_model(reasoning=True).reasoning is True


# --- token counter -----------------------------------------------------------------

def usage(read, written):
    return {"input_tokens": read, "output_tokens": written, "total_tokens": read + written}


def test_turn_and_chat_token_counts(capsys):
    from fin_agent.chat import chat_token_usage
    model = StreamingModel(script=[[AIMessageChunk(content="Hi!", usage_metadata=usage(100, 5))],
                                   [AIMessageChunk(content="Again!", usage_metadata=usage(120, 7))]])
    agent, _ = graph(model)
    run_turn(agent, {"messages": [HumanMessage("hello")]}, CONFIG)
    assert "tokens 100 in / 5 out · chat total 105" in capsys.readouterr().out
    run_turn(agent, {"messages": [HumanMessage("hello again")]}, CONFIG)
    assert "tokens 120 in / 7 out · chat total 232" in capsys.readouterr().out      # the chat total accumulates
    assert chat_token_usage(agent, CONFIG) == (220, 12)
    assert chat_token_usage(agent, {"configurable": {"thread_id": "a-new-chat"}}) == (0, 0)   # /new starts at zero


def test_a_turn_with_a_tool_call_counts_both_model_calls(monkeypatch, capsys):
    call = AIMessageChunk(content="", usage_metadata=usage(1000, 20), tool_call_chunks=[{
        "name": "get_spending_total", "args": '{"period":"this_week"}', "id": "t1", "index": 0}])
    agent, _ = graph(StreamingModel(script=[[call], [AIMessageChunk(content="You spent 0 EGP.", usage_metadata=usage(1100, 9))]]))
    run_turn(agent, {"messages": [HumanMessage("how much this week?")]}, CONFIG)
    assert "tokens 2,100 in / 29 out" in capsys.readouterr().out


def test_quick_path_uses_no_tokens(monkeypatch, capsys):
    agent, _ = graph(StreamingModel(script=[]))
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    run_turn(agent, {"messages": [HumanMessage("285 mcdonalds")]}, CONFIG)
    assert "tokens 0 in / 0 out · chat total 0" in capsys.readouterr().out


def test_guardrail_replacement_keeps_the_tokens_that_were_spent(capsys):
    from fin_agent.chat import chat_token_usage
    agent, _ = graph(StreamingModel(script=[[AIMessageChunk(content="I've recorded your lunch.", usage_metadata=usage(900, 8))]]))
    run_turn(agent, {"messages": [HumanMessage("I spent 85 at abu tarek")]}, CONFIG)
    assert chat_token_usage(agent, CONFIG) == (900, 8)


# --- live status line ----------------------------------------------------------------

from fin_agent.stream_output import (  # noqa: E402
    CLEAR_LINE, RESET_SCROLL_REGION, RESTORE_CURSOR, SAVE_CURSOR, StatusBar, move_to, scroll_region,
)


def bar(**kwargs):
    """A status bar writing to a string, as if on an 80x24 terminal, without the ticker thread."""
    out = StringIO()
    return StatusBar(out, enabled=True, animate=False, size=(80, 24), **kwargs), out


def test_status_counts_live_then_exact():
    status, _ = bar()
    status.begin_turn(chat_total=1000)
    assert "↑ … in · ↓ 0 out · chat 1,000 tokens" in status.text()      # nothing known yet
    for word in ["You", " spent", " 350"]:
        status.add_chunk(AIMessageChunk(content=word))
    assert "answering" in status.text() and "↓ 3 out" in status.text()  # live estimate: 1 chunk ~ 1 token
    status.call_done(read=1277, written=5)                              # the model's exact numbers
    assert "↑ 1,277 in · ↓ 5 out · chat 2,282 tokens" in status.text()
    status.add_chunk(AIMessageChunk(content="more"))                    # a second call adds on top
    assert "↓ 6 out" in status.text()


def test_status_shows_what_the_model_is_doing():
    status, _ = bar()
    status.begin_turn(0)
    assert "waiting" in status.text()
    status.add_chunk(AIMessageChunk(content="", additional_kwargs={"reasoning_content": "hmm"}))
    assert "thinking" in status.text()
    status.add_chunk(AIMessageChunk(content="", tool_call_chunks=[{"name": "x", "args": "{}", "id": "1", "index": 0}]))
    assert "calling a tool" in status.text()
    status.add_chunk(AIMessageChunk(content=""))                        # an empty chunk is not a token
    assert "↓ 2 out" in status.text()


def test_status_reserves_the_bottom_row_and_gives_it_back():
    status, out = bar()
    status.begin_turn(0)
    status.start()
    assert scroll_region(1, 23) in out.getvalue()                       # rows 1-23 scroll; row 24 is ours
    assert SAVE_CURSOR + move_to(24) + CLEAR_LINE in out.getvalue()     # drawn on the last row...
    assert out.getvalue().endswith(RESTORE_CURSOR)                      # ...and the cursor put back
    status.stop()
    assert out.getvalue().endswith(SAVE_CURSOR + RESET_SCROLL_REGION + move_to(24) + CLEAR_LINE + RESTORE_CURSOR)
    written = len(out.getvalue())
    status.stop()                                                       # stopping twice is harmless
    status.draw()                                                       # and nothing is drawn once stopped
    assert len(out.getvalue()) == written


def test_status_line_never_wraps():
    out = StringIO()
    status = StatusBar(out, enabled=True, animate=False, size=(30, 24))
    status.begin_turn(123456789)
    status.start()
    drawn = out.getvalue().split(CLEAR_LINE)[-1]
    visible = drawn.replace(GRAY, "").replace(RESET, "").replace(RESTORE_CURSOR, "")
    assert len(visible) <= 29


def test_status_is_silent_when_output_is_not_a_terminal(capsys):
    status = StatusBar()                                                # capsys output is not a terminal
    status.begin_turn(0)
    status.start()
    status.draw()
    status.stop()
    assert capsys.readouterr().out == ""


def test_tiny_terminal_keeps_all_its_rows():
    out = StringIO()
    status = StatusBar(out, enabled=True, animate=False, size=(80, 4))
    status.start()
    assert out.getvalue() == ""
