"""
Live test of long-term memory with the REAL model (from .env), on a fake Sheet
and a throwaway memory file. Safe to share: no real data is used.

    uv run python scripts/memory_trial.py

The scenario that exposed every memory bug so far:
  thread A: save two things (a fact and a preference) from one message
  thread B: a NEW conversation asks about the fact (should come from memory, no re-save)
  thread B: the fact changes (should REPLACE it: one payday, not two, and nothing lost)
Takes a few minutes on a CPU.
"""
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from fin_agent import config
from fin_agent.agent import build_agent
from fin_agent.chat import run_turn
from fin_agent.llm import chat_model
from fin_agent.memory import MemoryBank
from fin_agent.sheets import MemoryStore

STEPS = [("A", "Remember that I get paid on the 25th of every month, and I prefer really short answers."),
         ("B", "When do I get paid?"),
         ("B", "Actually my payday moved to the 1st of the month.")]

print("MODEL:", config.MODEL)
with tempfile.TemporaryDirectory() as folder:
    bank = MemoryBank(Path(folder) / "trial_memories.db")
    agent = build_agent(MemoryStore(), llm=chat_model(reasoning=False), checkpointer=InMemorySaver(), memory=bank)
    for thread, text in STEPS:
        print(f"\n[thread {thread}] you> {text}")
        run_turn(agent, {"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread}})
        print("   memories:", [(m.topic, m.text) for m in bank.all()])
    paydays = [m for m in bank.all() if "paid" in m.text.lower() or "payday" in m.text.lower()]
    ok = len(paydays) == 1 and "1st" in paydays[0].text and len(bank.all()) == 2
    print("\nRESULT:", "PASS (one payday, the 1st; preference kept)" if ok else "FAIL (see memories above)")
    bank.db.close()   # release the file so the temporary folder can be deleted on Windows
