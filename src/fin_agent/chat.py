"""
Chat with your finance assistant in the terminal.

    uv run python -m fin_agent.chat

Commands:  /new         start a fresh conversation (a new thread)
           /memories    list what the assistant remembers about you
           /forget <n>  delete a memory by number or topic (e.g. /forget payday)
           /tokens      tokens used by this chat so far
           /quit        exit

What you'll see for each message:
    thinking in dim gray, then answer tokens as the model generates them
    🔧 the tools the agent called ("show work"), with timings
    ❓ a confirmation question before anything is written to your Sheet
    the answer, then (1.8s · model 2x 1.6s · tools 0.2s · tokens 2,672 in / 45 out · chat total 12,340)

Conversations are saved in data/<mode>/assistant.db (gitignored), so closing
and reopening continues the same conversation. That's the checkpointer
(short-term memory). Facts and preferences about you live in
data/<mode>/memories.db (long-term memory) and survive /new.

PRIVACY: this shows your real spending. Don't paste the output into a cloud AI chat.
"""
import sqlite3
import time
import uuid

import httpx
import ollama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from fin_agent import config
from fin_agent.agent import build_agent
from fin_agent.llm import chat_model
from fin_agent.memory import MemoryBank
from fin_agent.sheets import SheetFormatError, SheetStore
from fin_agent.stream_output import StatusBar, TokenPrinter


def token_usage(messages) -> tuple[int, int]:
    """(tokens read, tokens written) as REPORTED BY THE MODEL for these messages.

    Every model reply carries `usage_metadata`: how many tokens it read (the
    instructions, tool descriptions and history, re-sent on every call) and how
    many it wrote. Cloud models bill on exactly these numbers. Messages without
    it (yours, tool results, the no-model quick path) count as zero.
    """
    read = written = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        read += usage.get("input_tokens", 0)
        written += usage.get("output_tokens", 0)
    return read, written


def chat_token_usage(agent, thread: dict) -> tuple[int, int]:
    """Totals for the whole conversation, computed from the saved messages, so
    they survive restarts and start again from zero after /new."""
    return token_usage(agent.get_state(thread).values.get("messages", []))


def run_turn(agent, payload, thread: dict) -> None:
    """Run the graph until it finishes or pauses, printing progress as it goes.

    "messages" events deliver live model chunks, even though the graph's node
    uses invoke(). "updates" events still report completed tools and pauses.
    Only completed nodes advance the timing clock, not each token.
    """
    started = last = time.time()
    model_calls, model_time, tool_time = 0, 0.0, 0.0
    turn_read = turn_written = 0   # tokens used by this message's model calls
    waited = 0.0                   # seconds spent waiting for YOU at a [y/n] prompt
    streamed_answer = ""
    # The live status line on the bottom row (does nothing unless this is a real terminal).
    status = StatusBar()
    status.begin_turn(sum(chat_token_usage(agent, thread)))
    while True:
        paused = None
        printer = TokenPrinter()
        status.start()
        try:
            for mode, event in agent.stream(payload, thread, stream_mode=["messages", "updates"]):
                if mode == "messages":
                    chunk, metadata = event
                    if metadata.get("langgraph_node") == "agent" and isinstance(chunk, AIMessage):
                        printer.feed(chunk)
                        status.add_chunk(chunk)      # the live "↓ out" estimate
                    continue
                now = time.time()
                took, last = now - last, now
                for node, update in event.items():
                    if node == "__interrupt__":
                        paused = update[0].value
                        continue
                    if node == "agent":
                        streamed_answer = printer.finish()
                        printer = TokenPrinter()
                        model_calls += 1
                        model_time += took
                    elif node == "tools":
                        tool_time += took
                    if node == "agent":
                        read, written = token_usage((update or {}).get("messages", []))
                        turn_read, turn_written = turn_read + read, turn_written + written
                        status.call_done(read, written)   # swap the estimate for the exact counts
                    for msg in (update or {}).get("messages", []):
                        if node == "agent":
                            for call in getattr(msg, "tool_calls", []) or []:
                                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                                print(f"  🔧 {call['name']}({args})   [model {took:.1f}s]")
                        elif node == "tools" and isinstance(msg, ToolMessage):
                            print(f"     ↳ {msg.content[:150]}{'…' if len(msg.content) > 150 else ''}")
        finally:
            printer.finish()  # errors / Ctrl+C must not leave the terminal gray
            status.stop()     # ...or with a reserved bottom row; also clears it before any [y/n] prompt
        if not paused:
            # Don't repeat an already streamed answer. The guardrail still runs and
            # may have ADDED a note to it (e.g. "nothing was actually saved").
            final = agent.get_state(thread).values["messages"][-1]
            text = final.content.strip() if isinstance(final, AIMessage) and final.content else ""
            if text and not streamed_answer:
                print(f"\nassistant> {text}")          # nothing was streamed (the no-model quick path)
            elif text.startswith(streamed_answer) and text != streamed_answer:
                print(f"\n{text[len(streamed_answer):].strip()}")   # only the added note
            elif text and text != streamed_answer:
                print(f"\nassistant (correction)> {text}")
            break
        # The graph is paused (e.g. before saving an expense). Ask, then resume it
        # with the answer; interrupt() inside the graph returns this answer.
        asked = time.time()
        answer = input(f"\n  ❓ Save this?  {paused['summary']}\n     [y/n] > ").strip()
        payload = Command(resume=answer)
        last = time.time()                           # don't count your thinking time...
        waited += last - asked                       # ...in the per-step timers or in the total
    chat_read, chat_written = chat_token_usage(agent, thread)
    print(f"  ({time.time() - started - waited:.1f}s · model {model_calls}x {model_time:.1f}s · tools {tool_time:.1f}s"
          f" · tokens {turn_read:,} in / {turn_written:,} out · chat total {chat_read + chat_written:,})\n")


def show_memories(memory: MemoryBank) -> None:
    memories = memory.all()
    if not memories:
        print("  (no memories yet: tell me lasting things like \"I get paid on the 25th\")\n")
        return
    for m in memories:
        print(f"  #{m.id}  [{m.topic}]  {m.text}   ({m.kind}, updated {m.updated[:10]})")
    print()


def forget_command(memory: MemoryBank, text: str) -> None:
    """/forget 3  or  /forget payday. Plain code: deleting never depends on the model."""
    arg = text.removeprefix("/forget").strip().lstrip("#").strip("[]")
    if not arg:
        print("  usage: /forget <number or topic>   (see /memories)\n")
        return
    gone = memory.forget(int(arg)) if arg.isdigit() else memory.forget_topic(arg)
    print(f"  Forgot [{gone.topic}]: {gone.text}\n" if gone else f"  No memory {arg!r}.\n")


OLLAMA_DOWN = (f"Can't reach Ollama at {config.OLLAMA_URL}. Start it (open the Ollama app, "
               f"or run `ollama serve` in another terminal), then try again.")


def check_ollama() -> None:
    """Fail early with a clear message instead of a long traceback on the first question."""
    try:
        tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()
    except httpx.HTTPError:
        raise SystemExit(OLLAMA_DOWN)
    # CLOUD models (names ending in "cloud") run on Ollama's servers and are never
    # downloaded, so they never appear in the local list. Checking for them there
    # produced a false warning on every start.
    if config.MODEL.endswith("cloud"):
        return
    names = {m["name"] for m in tags.get("models", [])}
    # "qwen3.5:4b" and "lfm2.5" (= "lfm2.5:latest") are both valid ways to name a model.
    if config.MODEL not in names and f"{config.MODEL}:latest" not in names:
        print(f"Warning: model {config.MODEL!r} isn't in `ollama list`. Run `ollama pull {config.MODEL}`.")


def main() -> None:
    check_ollama()
    store = SheetStore(config.GOOGLE_SHEET_ID, config.GOOGLE_SERVICE_ACCOUNT_FILE)
    try:
        store.setup()   # checks every tab's columns BEFORE anything can be written
    except SheetFormatError as err:
        raise SystemExit(f"Stopped: {err}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph may use the connection from another thread.
    saver = SqliteSaver(sqlite3.connect(config.DATA_DIR / "assistant.db", check_same_thread=False))
    memory = MemoryBank(config.DATA_DIR / "memories.db")
    agent = build_agent(store, llm=chat_model(reasoning=False), checkpointer=saver, memory=memory)

    # A "thread" is one conversation. Same thread_id = the agent sees the history.
    thread = {"configurable": {"thread_id": "main"}}
    print(f"Finance assistant ({config.MODEL}, {config.DEFAULT_CURRENCY}, {config.TIMEZONE}). "
          f"Remembers {len(memory.all())} things about you. /new, /memories, /forget <n>, /tokens, /quit\n")
    while True:
        text = input("you> ").strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
            print("(new conversation; long-term memories are kept)\n")
            continue
        if text == "/memories":
            show_memories(memory)
            continue
        if text.startswith("/forget"):
            forget_command(memory, text)
            continue
        if text == "/tokens":
            read, written = chat_token_usage(agent, thread)
            print(f"  This chat so far: {read:,} tokens read + {written:,} written = {read + written:,} total.\n"
                  f"  (Read is high because the instructions, tool list and history are re-sent on every model call.)\n")
            continue
        try:
            run_turn(agent, {"messages": [HumanMessage(text)]}, thread)
        except httpx.ConnectError:
            print(f"\n  ⚠️ {OLLAMA_DOWN}\n")        # Ollama stopped mid-conversation
        except ollama.ResponseError as err:
            print(f"\n  ⚠️ The model returned an error: {err}. Try again or rephrase.\n")


if __name__ == "__main__":
    main()
