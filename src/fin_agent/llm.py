"""
Talking to the model: the chat model the agent uses, a cleanup helper for
models that write their reasoning into the answer, and token counting.
"""
import re

from langchain_ollama import ChatOllama

from fin_agent import config


def chat_model(model: str | None = None, **overrides) -> ChatOllama:
    """The LangChain chat model used by the agent, with this machine's settings.

    - CPU only (config.OLLAMA_NUM_GPU), see the note in config.py.
    - The model's own recommended sampling, unless config.MODEL_SETTINGS overrides it.
    - reasoning=False by default; the terminal opts in to show thinking live.
    - num_ctx=8192: room for the system prompt, tool descriptions, history and results.
    Swapping models is just a different `model` name; nothing else changes.
    """
    name = model or config.MODEL
    settings = {"reasoning": False, "num_gpu": config.OLLAMA_NUM_GPU, "num_ctx": 8192,
                **config.MODEL_SETTINGS.get(name, {}), **overrides}
    return ChatOllama(model=name, base_url=config.OLLAMA_URL, **settings)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning that some models (e.g. lfm2.5) put in the
    answer even when asked not to think. An unclosed <think> (the model was cut off
    mid-thought) is removed to the end."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()


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
