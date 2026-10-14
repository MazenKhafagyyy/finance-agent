"""
TURNS: run one message through the agent and return a plain result.

This is the part of a chat interface that has NO interface in it: no printing,
no Discord, no buttons. A front-end (the Discord bot today, a web page later)
calls `run_turn` with the text you sent and gets back a `TurnResult` it can
display however it likes. Keeping it separate means the rules that protect
your data live where tests can reach them.

(The terminal chat has its own loop in chat.py because it does things a chat
app can't: live token-by-token output and a status line.)

THE ONE HARD PART: A QUESTION THAT IS STILL WAITING
---------------------------------------------------
add_expense PAUSES the graph (`interrupt`) until you answer "Save this?". In
the terminal you can't do anything else until you answer. In a chat app you
can: ignore the buttons and type something new, click twice, or click an old
button an hour later. So this module decides, in code, what each of those means:

    you click Yes/No            -> answer_confirmation: resume the graph with it
    you type "yes" / "no"       -> the same, typed instead of clicked
    you type anything else      -> the waiting save is DECLINED first, then your
                                   new message is handled. Saying nothing about
                                   a save never saves it.
    a click on an old question  -> ignored (None), never applied to a newer one

Why decline first instead of just sending the new message? The history would
contain a tool call with no result, followed by your message. Models expect
every tool call to be answered; an unanswered one is at best confusing and at
worst rejected by the model provider.
"""
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from fin_agent.llm import token_usage

YES, NO = ("y", "yes"), ("n", "no")
# After a decline the model answers once more. If it keeps asking to save
# again (qwen3.5 once did), stop cancelling after this many rounds.
MAX_DECLINES = 5


@dataclass
class TurnResult:
    reply: str = ""           # the assistant's answer ("" while it waits for a yes/no)
    confirm: str = ""         # what is waiting to be saved, e.g. "150 EGP · Starbucks → Coffee ..."
    confirm_id: str = ""      # identifies THAT question, so an old button can't answer a new one
    declined: list[str] = field(default_factory=list)   # waiting saves this message cancelled
    tools: list[str] = field(default_factory=list)      # tools called for the latest message
    seconds: float = 0.0      # time spent producing THIS reply (not your thinking time)
    tokens: int = 0           # tokens read + written for the latest message


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _waiting(agent, thread_id: str):
    """The paused question on this thread (a LangGraph Interrupt), or None."""
    interrupts = agent.get_state(_config(thread_id)).interrupts
    return interrupts[0] if interrupts else None


def _result(agent, thread_id: str, started: float, declined: list[str] | None = None) -> TurnResult:
    """Read the outcome from the SAVED state (the checkpointer), not from what
    invoke() returned: one source of truth, whether the graph finished or paused."""
    messages = agent.get_state(_config(thread_id)).values.get("messages", [])
    # "This turn" = everything after your latest message (same rule as check_reply).
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    turn = messages[last_human + 1:]
    called = [call["name"] for m in turn for call in getattr(m, "tool_calls", None) or []]
    result = TurnResult(declined=declined or [], tools=list(dict.fromkeys(called)),   # each name once, in order
                        seconds=time.time() - started, tokens=sum(token_usage(turn)))
    waiting = _waiting(agent, thread_id)
    if waiting:
        result.confirm, result.confirm_id = waiting.value["summary"], waiting.id
    elif turn and isinstance(turn[-1], AIMessage):
        result.reply = str(turn[-1].content).strip()
    return result


def run_turn(agent, thread_id: str, text: str) -> TurnResult:
    """Handle a message you typed. See the table in the module docstring."""
    started = time.time()
    declined: list[str] = []
    if _waiting(agent, thread_id):
        word = text.strip().lower()
        if word in YES + NO:
            agent.invoke(Command(resume=word), _config(thread_id))
            return _result(agent, thread_id, started)
        for _ in range(MAX_DECLINES):
            waiting = _waiting(agent, thread_id)
            if not waiting:
                break
            declined.append(waiting.value["summary"])
            agent.invoke(Command(resume="no"), _config(thread_id))
    agent.invoke({"messages": [HumanMessage(text)]}, _config(thread_id))
    return _result(agent, thread_id, started, declined)


def answer_confirmation(agent, thread_id: str, confirm_id: str, answer: str) -> TurnResult | None:
    """Handle a click on Yes/No. Returns None if THAT question is no longer the
    one waiting (already answered, or cancelled by a later message)."""
    started = time.time()
    waiting = _waiting(agent, thread_id)
    if not waiting or waiting.id != confirm_id:
        return None
    agent.invoke(Command(resume=answer), _config(thread_id))
    return _result(agent, thread_id, started)
