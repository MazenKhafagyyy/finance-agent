"""
Chat with your finance assistant in the terminal.

    uv run fin-agent chat

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

Conversations are saved in DATA_DIR/assistant.db (data/ by default, gitignored),
so closing and reopening continues the same conversation. That's the
checkpointer (short-term memory). Facts and preferences about you live in
DATA_DIR/memories.db (long-term memory) and survive /new.

PRIVACY: this shows your real spending. Don't paste the output into a cloud AI chat.
"""
import textwrap
import time
import uuid

import httpx
import ollama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from fin_agent import config
from fin_agent.app import forget_text, memories_text, start
from fin_agent.checks import OLLAMA_DOWN
from fin_agent.llm import token_usage
from fin_agent.stream_output import StatusBar, TokenPrinter


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


def main() -> None:
    assistant = start()
    agent, memory = assistant.agent, assistant.memory

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
            print(textwrap.indent(memories_text(memory), "  ") + "\n")
            continue
        if text.startswith("/forget"):
            print(f"  {forget_text(memory, text.removeprefix('/forget'))}\n")
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
