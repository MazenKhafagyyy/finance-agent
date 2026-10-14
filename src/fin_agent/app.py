"""
What every front-end shares: the start-up (settings, Ollama, the Sheet, the
files under DATA_DIR, the agent), and the /memories and /forget commands.

The Discord bot and the terminal chat both call `start()`, so the steps that
protect your data (checking the Sheet's columns before anything can be
written, for one) can't drift apart between the two.
"""
import sqlite3
from dataclasses import dataclass

from langgraph.checkpoint.sqlite import SqliteSaver

from fin_agent import config
from fin_agent.agent import build_agent
from fin_agent.checks import check_ollama
from fin_agent.llm import chat_model
from fin_agent.memory import MemoryBank
from fin_agent.sheets import SheetFormatError, SheetStore


@dataclass
class Assistant:
    agent: object         # the compiled LangGraph agent
    memory: MemoryBank    # long-term memories (for /memories and /forget)


def start(discord: bool = False) -> Assistant:
    """Check the settings and connect everything, or stop with a message saying
    what to fix. `discord=True` also requires the bot's two settings."""
    problems = config.settings_problems() + (config.discord_problems() if discord else [])
    if problems:
        raise SystemExit("Fix these in .env, then try again (`uv run fin-agent check` tests everything):\n"
                         + "\n".join(f"  - {p}" for p in problems))
    check_ollama()
    store = SheetStore(config.GOOGLE_SHEET_ID, config.GOOGLE_SERVICE_ACCOUNT_FILE)
    try:
        store.setup()   # checks every tab's columns BEFORE anything can be written
    except SheetFormatError as err:
        raise SystemExit(f"Stopped: {err}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the bot runs each turn in a worker thread, and
    # LangGraph may use the connection from another thread too.
    saver = SqliteSaver(sqlite3.connect(config.DATA_DIR / "assistant.db", check_same_thread=False))
    memory = MemoryBank(config.DATA_DIR / "memories.db")
    agent = build_agent(store, llm=chat_model(reasoning=False), checkpointer=saver, memory=memory)
    return Assistant(agent, memory)


def memories_text(memory: MemoryBank) -> str:
    """/memories: what the assistant remembers, one line each."""
    memories = memory.all()
    if not memories:
        return "(no memories yet: tell me lasting things like \"I get paid on the 25th\")"
    return "\n".join(f"#{m.id}  [{m.topic}]  {m.text}  ({m.kind}, updated {m.updated[:10]})" for m in memories)


def forget_text(memory: MemoryBank, what: str) -> str:
    """/forget 3  or  /forget payday. Plain code: deleting never depends on the model."""
    arg = what.strip().lstrip("#").strip("[]")
    if not arg:
        return "Give a memory's number or topic (see /memories)."
    gone = memory.forget(int(arg)) if arg.isdigit() else memory.forget_topic(arg)
    return f"Forgot [{gone.topic}]: {gone.text}" if gone else f"No memory {arg!r}."
