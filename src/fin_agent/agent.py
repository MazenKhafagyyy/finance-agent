"""
THE AGENT: the part that decides what to do.

WHAT IS AN "AGENT", ACTUALLY?
-----------------------------
Strip away the hype and an agent is three things:
    1. a language model   (the "brain": reads, decides, writes)
    2. tools              (Python functions it may ask us to run; tools.py)
    3. a LOOP             model -> tools -> model -> tools -> ... -> answer

The loop is what makes it an agent rather than a chain. A classic RAG pipeline
is a CHAIN: retrieve, then answer, always exactly once, in a fixed order. An agent
decides at run time what to do next, and can go round as many times as it needs:
"compare fast food this month vs last month" -> call a tool for this month,
look at it, call a tool for last month, look at that, then answer.

The model never runs anything itself. Each turn it either:
    - writes a reply for you (the loop ends), or
    - writes a TOOL CALL, e.g. get_spending_total(period="this_week"),
      which OUR code runs, and whose result goes back to the model.
So the agent can only ever do what our tools allow. That's the safety boundary.

THIS AGENT AS A GRAPH (LangGraph)
---------------------------------
                    ┌──────────────┐   known quick line   ┌─────────────┐
     your message ─►│ route_start  │ ───────────────────► │  quick_add  │──► END
                    └──────────────┘   ("285 mcdonalds")  └─────────────┘
                           │ anything else
                           ▼
                    ┌──────────────┐   no tool call   ┌─────────────┐
                    │    agent     │ ───────────────► │ check_reply │──► END (reply to you)
                    │ (call model) │                  │ (guardrail) │
                    └──────────────┘                  └─────────────┘
                        ▲      │ tool call(s)
                        │      ▼
                    ┌──────────────┐
                    │    tools     │  runs the Python functions
                    └──────────────┘

  - STATE: what flows through the graph. Here: the list of messages
    (yours, the model's, tool results). MessagesState keeps and appends them.
  - NODES: the boxes (Python functions that take the state and return updates).
  - EDGES: the arrows. Conditional edges decide at run time (e.g. "did the
    model ask for a tool?").
  - CHECKPOINTER: saves the state after every step, per conversation "thread".
    That gives the chat its memory, and makes pausing possible.
  - INTERRUPT: add_expense PAUSES the graph and asks you to confirm. The graph
    resumes with your answer. Nothing is written to the Sheet without a yes.
  - GUARDRAIL: check_reply adds a note to any final reply that claims an expense
    was saved when nothing was saved in that turn (plain code, not the model).
"""
import json
import re
from datetime import date as Date
from typing import Annotated, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import interrupt

from fin_agent import config
from fin_agent.analysis import budget_status, unessential_report, unusual_spending, weekly_review
from fin_agent.dates import day_phrases, is_iso_date, now_local, resolve_day, resolve_period
from fin_agent.entries import build_from_text
from fin_agent.llm import chat_model, strip_thinking
from fin_agent.memory import Memory, MemoryBank, MemoryRuleError
from fin_agent.quickline import categorize, parse_line
from fin_agent.tools import (
    compare_periods, describe_entry, list_transactions, prepare_expense,
    save_expense, spending_breakdown, spending_total,
)

# The period names, as a type. LangChain turns a Literal into an "enum" in the
# tool's schema, so the model sees the exact allowed values and can't invent
# "this_fortnight". Picking a name is also far more reliable than asking a small
# model to compute dates.
Period = Literal["today", "yesterday", "this_week", "last_week", "this_month",
                 "last_month", "last_7_days", "last_30_days", "this_year"]
Currency = Literal["EGP", "CAD"]


# ---------------------------------------------------------------------------
# The system prompt: standing instructions, rebuilt every model call.
#
# PROMPT CACHING: Ollama remembers its internal state after reading a prompt
# (the "KV cache"). If the next prompt STARTS WITH exactly the same text, it
# skips re-reading that part. So everything that comes first must stay
# identical between calls: only the DATE goes in here, never the time of day.
# (With the time in it, the first line changed every few seconds, and the
# model re-read all ~1,250 tokens on every call.) The Sheet's Time column is
# filled in by code, so the model never needs the time.
# ---------------------------------------------------------------------------

def system_prompt(categories, today: str, memories: list[Memory] | None = None) -> str:
    """`memories` is None when long-term memory is off (tests, model comparison);
    a list (possibly empty) when it's on."""
    weekday = Date.fromisoformat(today).strftime("%A")
    # Two plain lists, so the model understands "unessential" questions. An earlier
    # version wrote "Fast Food (non-essential)"; gemma4 copied that label into a
    # tool call as the category NAME and got "you spent 0". Names must appear
    # in the prompt exactly as the tools expect them.
    essential = ", ".join(c.name for c in categories if c.essential)
    non_essential = ", ".join(c.name for c in categories if not c.essential)
    prompt = f"""You are a personal finance assistant for one user. Their expenses are in a Google Sheet you access only through tools.
Today is {weekday} {today} (Cairo). Weeks start on Saturday. Currency is {config.DEFAULT_CURRENCY} unless the user says CAD.

Rules:
- Use a tool for every number. Never add up or calculate amounts yourself.
- For time ranges, pass a period name (this_week, last_month, ...) instead of computing dates.
- When the user tells you about a purchase, call add_expense. The user will be asked to confirm before it is saved.
- For add_expense's date, pass the user's own words ("yesterday", "3 days ago", "monday"). Never work out a calendar date yourself; older messages in this chat may be from earlier days.
- Only call the tools needed for what the user asked. After add_expense, just confirm the result; don't call other tools.
- Answer in 1 to 3 short sentences and always state the currency and the period.
- If a result says some entries weren't counted (no amount or no category), mention it.
- Give observations from the numbers, not financial advice or predictions beyond the tool's projection.

Essential categories: {essential}.
Non-essential categories: {non_essential}.
Use category names exactly as written above."""
    if memories is None:
        return prompt
    # LONG-TERM MEMORY goes at the END of the system prompt: everything above
    # stays identical between turns, so Ollama's cache still covers it even
    # when a memory changes.
    known = "\n".join(f"[{m.topic}] {m.text} ({m.kind})" for m in memories) or "(nothing yet)"
    return f"""{prompt}

Memory rules:
- Each memory has a short topic in brackets, e.g. [payday], [answer_style], [home_city].
- When the user shares a lasting fact about themselves or a preference for how you should answer, call remember with a short topic. Don't remember one-off events or expenses.
- To CHANGE a memory, call remember with the SAME topic; it replaces the old one. Never use a new topic for something already listed.
- Don't save things that are already listed unchanged. To answer a question, just use the list.
- If the user asks you to forget something, call forget_memory with its topic.
- Follow the user's preferences below.

What you know about the user from earlier conversations:
{known}"""


# ---------------------------------------------------------------------------
# The tools the model can call. Built inside a function so they can use the
# `store` (real Sheet or fake) and `now_fn` (real clock or a fixed test time).
# ---------------------------------------------------------------------------

def as_json(result: dict) -> str:
    """Tool results go back to the model as text. JSON is compact and unambiguous;
    ensure_ascii=False keeps Arabic merchant names readable."""
    return json.dumps(result, ensure_ascii=False)


def make_tools(store, now_fn: Callable[[], tuple[str, str]]) -> list:

    def dates(period: str | None, start: str | None, end: str | None) -> tuple[str, str]:
        """Turn a period name OR an explicit start/end into (start, end)."""
        if period:
            return resolve_period(period, now_fn()[0])
        if start and end and is_iso_date(start) and is_iso_date(end):
            return start, end
        raise ValueError("Give a period name (e.g. this_week), or both start and end as YYYY-MM-DD.")

    # The @tool decorator turns each function into a LangChain tool. Its name,
    # argument types and DOCSTRING become the description the model reads, so
    # the docstrings are written for the model: short and precise (every token
    # costs reading time on a CPU).

    @tool
    def get_spending_total(period: Period | None = None, start: str | None = None, end: str | None = None,
                           currency: Currency = "EGP", category: str | None = None,
                           essential: bool | None = None) -> str:
        """Total spent in a period. Optional filters: one category, or essential=false for non-essential spending only."""
        s, e = dates(period, start, end)
        return as_json(spending_total(store.load(), s, e, currency, category=category, essential=essential))

    @tool
    def get_spending_breakdown(period: Period | None = None, start: str | None = None, end: str | None = None,
                               by: Literal["category", "merchant"] = "category", currency: Currency = "EGP",
                               essential: bool | None = None) -> str:
        """Spending grouped by category or merchant, largest first, with each group's share. Use essential=false to see only non-essential spending."""
        s, e = dates(period, start, end)
        return as_json(spending_breakdown(store.load(), s, e, currency, by=by, essential=essential, top=10))

    @tool
    def compare_spending(period_a: Period, period_b: Period, currency: Currency = "EGP",
                         category: str | None = None, essential: bool | None = None) -> str:
        """Compare total spending in period_a with period_b (e.g. this_month vs last_month)."""
        a, b = resolve_period(period_a, now_fn()[0]), resolve_period(period_b, now_fn()[0])
        return as_json(compare_periods(store.load(), *a, *b, currency, category=category, essential=essential))

    @tool
    def find_transactions(period: Period | None = None, start: str | None = None, end: str | None = None,
                          merchant: str | None = None, category: str | None = None) -> str:
        """List individual expenses (newest first), optionally for one merchant or category."""
        s, e = dates(period, start, end)
        return as_json(list_transactions(store.load(), s, e, merchant=merchant, category=category, limit=15))

    @tool
    def add_expense(amount: float, description: str, state: Annotated[dict, InjectedState],
                    category: str = "", currency: Currency = "EGP", date: str = "") -> str:
        """Record a purchase the user tells you about. description: the merchant or item, e.g. "mcdonalds". category: only if the user named one, or the merchant is new. date: the user's own words for the day ("today", "yesterday", "3 days ago", "monday"), or YYYY-MM-DD only if they gave an exact date; never calculate a date yourself. The user confirms before saving."""
        # `state` is INJECTED by LangGraph, not written by the model (it's hidden
        # from the tool's schema). We use it to store what the user actually typed
        # in the Sheet's Message column, like EXO does.
        said = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        today = now_fn()[0]

        # WHICH DAY? Decided by code, not by the model's arithmetic.
        # If the user's message names exactly one day ("yesterday", "2 days ago",
        # "on monday"), that wins over whatever the model put in `date`. Real case:
        # the user said "yesterday" on Oct 2 and the model sent date='2026-09-29'.
        # With no such word (or several), we resolve the model's `date` instead,
        # which may itself be a word like "yesterday" or an exact date.
        mentioned = day_phrases(said)
        day = resolve_day(mentioned[0] if len(mentioned) == 1 else date, today)

        snap = store.load()
        built = prepare_expense(snap, now_fn(), amount=amount, description=description,
                                category=category, currency=currency, date=day, message=said)

        if not built.entry.category:
            # Unknown merchant and no category: don't guess. Tell the MODEL to ask
            # the user. (The model reads this return value as the tool's result.)
            names = ", ".join(c.name for c in snap.categories)
            return (f"Not saved: '{built.entry.merchant}' is a new merchant with no category. "
                    f"Ask the user which category it belongs to ({names}), then call add_expense again with it.")

        # PAUSE the whole graph here and hand this summary to the chat program,
        # which asks the user. When the user answers, the graph resumes and
        # interrupt() returns their answer. (On resume LangGraph re-runs this tool
        # from the top; that's fine because nothing was saved yet.)
        # The summary shows the weekday and "(yesterday)" etc., so a wrong date is visible.
        summary = describe_entry(built, today)
        answer = interrupt({"action": "add_expense", "summary": summary})
        if str(answer).strip().lower() in ("y", "yes"):
            save_expense(store, built)
            return f"Saved: {summary}"
        # Final wording matters: after "The user declined", qwen3.5 retried the
        # same add_expense twice more and asked again. Declining ends it.
        return "The user declined, so nothing was saved. Do not call add_expense again for this; just acknowledge it."

    # --- Phase 5: analysis tools. Each returns one complete, precomputed picture,
    # so a question needs ONE tool call (one model round trip) instead of several.

    @tool
    def analyze_unessential(period: Period = "this_month", currency: Currency = "EGP") -> str:
        """Non-essential spending by category and merchant, compared with a fair previous period (same number of days), and what grew most. Use for questions about unessential or wasteful spending."""
        return as_json(unessential_report(store.load(), period, now_fn()[0], currency))

    @tool
    def check_budgets(currency: Currency = "EGP") -> str:
        """This month's spending vs each category's monthly budget: percent used, warning/over status, and projected month-end total."""
        return as_json(budget_status(store.load(), now_fn()[0], currency))

    @tool
    def find_unusual_spending(days: int = 30, currency: Currency = "EGP") -> str:
        """Unusual charges: much bigger than the merchant's usual, possible double charges, new monthly repeating charges. days: how far back to look; keep the default 30 unless the user names a period."""
        return as_json(unusual_spending(store.load(), now_fn()[0], currency, recent_days=days))

    @tool
    def get_weekly_review(currency: Currency = "EGP") -> str:
        """Summary of this week vs the same days last week: totals, essential split, top categories, budget alerts, unusual charges."""
        return as_json(weekly_review(store.load(), now_fn()[0], currency))

    return [get_spending_total, get_spending_breakdown, compare_spending, find_transactions, add_expense,
            analyze_unessential, check_budgets, find_unusual_spending, get_weekly_review]


ASKS_TO_FORGET = re.compile(r"\b(forget|delete|remove|erase|don'?t remember|stop remembering)\b", re.IGNORECASE)


def make_memory_tools(memory: MemoryBank) -> list:
    """The two tools for LONG-TERM memory. No confirmation step (unlike add_expense):
    memories are low-risk and fully visible. The chat shows every remember/forget
    call, and /memories and /forget let you review and delete them."""

    @tool
    def remember(topic: str, text: str, kind: Literal["fact", "preference"] = "fact", distinct: bool = False) -> str:
        """Save a lasting fact about the user or a preference for how to answer. topic: a short key like "payday" or "answer_style"; an EXISTING topic replaces that memory. text: one short third-person sentence, e.g. "Gets paid on the 25th". distinct: true only after being told it's similar to another memory, if it's really different. Never use for expenses."""
        try:
            saved, old = memory.remember(topic, text, kind, distinct)
        except MemoryRuleError as err:
            return str(err)          # the model reads why, and can tell the user
        if old:
            return f"Updated [{saved.topic}]: {saved.text} (was: {old.text})"
        return f"Remembered [{saved.topic}]: {saved.text}"

    @tool
    def forget_memory(topic: str, state: Annotated[dict, InjectedState]) -> str:
        """Delete a saved memory by its [topic], ONLY when the user explicitly asks you to forget it. To change a memory, use remember with the same topic instead."""
        # Deleting needs EXPLICIT user intent, checked in code. Real case: the user
        # said "my payday moved to the 1st"; the model deleted [payday], never saved
        # the new value, and replied "your payday has been updated".
        said = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        if not ASKS_TO_FORGET.search(said):
            return ("Not deleted: the user didn't ask to forget anything. To CHANGE a memory, "
                    "call remember with the same topic; it replaces the old value.")
        gone = memory.forget_topic(topic)
        return f"Forgot [{gone.topic}]: {gone.text}" if gone else f"There is no memory with topic [{topic}]."

    return [remember, forget_memory]


# ---------------------------------------------------------------------------
# Building the graph
# ---------------------------------------------------------------------------

# Phrases that CLAIM a save JUST HAPPENED: "I've recorded", "has been saved",
# "successfully added", "I logged". Deliberately narrow: the first version matched
# any "I ... recorded" and fired on an honest sentence about the PAST ("I
# incorrectly recorded the date as September 29th").
CLAIMS_A_SAVE = re.compile(
    r"\b(?:i'?ve|i have|has been|have been|is now|are now|successfully)\s+(?:\w+\s+){0,2}?(?:recorded|saved|added|logged)\b"
    r"|\bi\s+(?:just\s+)?(?:recorded|saved|added|logged)\b",
    re.IGNORECASE,
)
FALSE_CLAIM_WARNING = ("⚠️ Note: no expense was actually saved in this turn. If you meant to log one, "
                       "tell me the amount and merchant and I'll ask you to confirm.")


# Phrases claiming a MEMORY was written: "I've noted", "has been updated", "I'll remember".
CLAIMS_A_MEMORY = re.compile(r"\b(noted|remembered|updated|i'?ll remember|i will remember)\b", re.IGNORECASE)
FALSE_MEMORY_WARNING = "⚠️ Note: nothing was actually saved to memory in this turn."


def check_claims(turn_messages: list, reply: str) -> str:
    """A GUARDRAIL written in plain code: if the reply claims an expense (or a
    memory) was saved but no tool call actually saved one in this turn, ADD a
    note saying so.

    Why this exists: in testing, lfm2.5 once answered "I've recorded your lunch
    expense of 85 EGP" WITHOUT calling any tool, and later "your payday has been
    updated" after only DELETING the old payday. Tools protect the DATA; this
    protects the TRUTH of the reply.

    Why a note and not a replacement: a text pattern can be wrong. The first
    version REPLACED the reply and once erased a legitimate answer. Appending
    means a real false claim is still corrected in plain sight, while a
    misfire costs nothing: the original answer is still there.
    """
    results = [str(m.content) for m in turn_messages if isinstance(m, ToolMessage)]
    expense_saved = any(r.startswith("Saved:") for r in results)
    memory_saved = any(r.startswith(("Remembered [", "Updated [")) for r in results)
    if claims_a_save(reply) and not expense_saved and not memory_saved:
        return f"{reply}\n\n{FALSE_CLAIM_WARNING}"
    if claims_a_memory(reply) and not memory_saved and not expense_saved:
        return f"{reply}\n\n{FALSE_MEMORY_WARNING}"
    return reply


def claims_a_memory(reply: str) -> bool:
    """Like claims_a_save, for memory words; negated phrases don't count."""
    for match in CLAIMS_A_MEMORY.finditer(reply):
        window = reply[max(0, match.start() - 12):match.end()]
        if not NEGATION.search(window):
            return True
    return False


NEGATION = re.compile(r"\b(not|nothing|no|never)\b|n't\b", re.IGNORECASE)


def claims_a_save(reply: str) -> bool:
    """True if any claim phrase is NOT negated. "has been recorded" is a claim;
    "nothing was saved" / "wasn't recorded" / "was not saved" are not. We look at
    the matched phrase plus the few characters before it for a negation."""
    for match in CLAIMS_A_SAVE.finditer(reply):
        window = reply[max(0, match.start() - 12):match.end()]
        if not NEGATION.search(window):
            return True
    return False


def is_quick_line(text: str, merchants) -> bool:
    """True for messages like "285 mcdonalds": an amount plus a KNOWN merchant and
    nothing else. Those are handled by plain code, skipping the slow model.
    "I paid 200 in food today (mcdonalds)" has extra words, so it goes to the model."""
    parsed = parse_line(text)
    if not parsed or parsed.amount is None or not parsed.description:
        return False
    found = categorize(parsed.description, merchants)
    return found.known and found.note == ""


def build_agent(store, *, llm=None, checkpointer=None, memory: MemoryBank | None = None,
                now_fn: Callable[[], tuple[str, str]] = now_local):
    """Assemble and compile the graph.

    store:        SheetStore (real) or MemoryStore (tests / model comparison)
    llm:          a chat model; defaults to config.MODEL. Tests pass a scripted fake.
    checkpointer: SHORT-term memory: this conversation's messages (SQLite for the chat).
    memory:       LONG-term memory: facts and preferences across conversations.
                  None = off (the model then doesn't get the memory tools at all).
    now_fn:       the clock; tests pass a fixed time.
    """
    tools = make_tools(store, now_fn) + (make_memory_tools(memory) if memory else [])
    # bind_tools attaches the tool schemas to every request, so the model knows
    # what it may call.
    model = (llm or chat_model()).bind_tools(tools)
    categories = store.load().categories  # rarely change; loaded once per agent

    def last_user_text(state: MessagesState) -> str:
        return next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    # --- Node: decide the route for a new message ---------------------------
    def route_start(state: MessagesState) -> str:
        return "quick_add" if is_quick_line(last_user_text(state), store.load().merchants) else "agent"

    # --- Node: the fast path, no model involved -------------------------------
    def quick_add(state: MessagesState) -> dict:
        text = last_user_text(state)
        built = build_from_text(text, store.load(), now_fn())
        summary = describe_entry(built, now_fn()[0])
        answer = interrupt({"action": "add_expense", "summary": summary})
        if str(answer).strip().lower() in ("y", "yes"):
            save_expense(store, built)
            reply = f"Saved: {summary}"
        else:
            reply = "OK, not saved."
        return {"messages": [AIMessage(content=reply)]}

    # --- Node: call the model ---------------------------------------------------
    def agent(state: MessagesState) -> dict:
        # The system prompt is added fresh on every call (not stored in the
        # history), so it always has the current date.
        # READ path of long-term memory: re-read every call, so a memory saved a
        # moment ago (or deleted with /forget) is reflected immediately.
        memories = memory.for_prompt(last_user_text(state)) if memory else None
        messages = [SystemMessage(system_prompt(categories, now_fn()[0], memories)), *state["messages"]]
        response = model.invoke(messages)
        # Streaming displays reasoning live, but it needn't enlarge the saved
        # history (or be sent back to Ollama on every subsequent model call).
        cleaned = response.model_copy(update={
            "content": strip_thinking(response.content or ""),
            "additional_kwargs": {k: v for k, v in response.additional_kwargs.items()
                                  if k != "reasoning_content"},
        })
        return {"messages": [cleaned]}   # MessagesState APPENDS this to the history

    # --- Node: check the final reply before you see it --------------------------
    def check_reply(state: MessagesState) -> dict:
        messages = state["messages"]
        # "This turn" = everything after your latest message.
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        reply = messages[-1]
        checked = check_claims(messages[last_human + 1:], reply.content)
        if checked == reply.content:
            return {}                    # nothing to change
        # Same message id = REPLACE that message in the history, instead of appending.
        # Keep its token usage: those tokens were spent even though the text is replaced.
        return {"messages": [AIMessage(content=checked, id=reply.id,
                                       usage_metadata=getattr(reply, "usage_metadata", None))]}

    def after_agent(state: MessagesState) -> str:
        # The model asked for tools -> run them; otherwise its reply is final -> check it.
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "check_reply"

    graph = StateGraph(MessagesState)
    graph.add_node("quick_add", quick_add)
    graph.add_node("agent", agent)
    graph.add_node("check_reply", check_reply)
    # ToolNode runs whatever tool calls the last AI message contains. If a tool
    # fails, for ANY reason (a bad argument type like essential="", an unknown
    # category...), the error text is sent back to the model as the tool's
    # result, so it can correct itself on the next turn instead of crashing the
    # chat. (Pausing via interrupt() is not an error; ToolNode lets it through.)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    graph.add_conditional_edges(START, route_start, ["quick_add", "agent"])
    graph.add_edge("quick_add", END)
    graph.add_conditional_edges("agent", after_agent, ["tools", "check_reply"])
    graph.add_edge("tools", "agent")   # after tools run, the model sees the results: the LOOP
    graph.add_edge("check_reply", END)
    return graph.compile(checkpointer=checkpointer)
