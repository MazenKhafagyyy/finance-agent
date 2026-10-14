"""All settings in one place, read from the .env file (see .env.example).

Nothing here stops the program on a bad setting: `settings_problems()` lists
what's wrong in plain words, and the start-up (app.py) and `fin-agent check`
(checks.py) show that list instead of a traceback.
"""
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from fin_agent.sheet_format import CURRENCIES

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the assistant keeps its own files: conversation history, memories and
# sessions (SQLite). A relative path counts from the project folder. The whole
# data/ folder is gitignored.
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

# The chat model, by its Ollama name. The default is a CLOUD model (the name ends
# in "cloud"): it runs on Ollama's servers, so replies take seconds, but your
# messages and tool results are sent there. For a fully local setup use
# MODEL=qwen3.5:4b (slow on a CPU: a minute or more per reply).
MODEL = os.getenv("MODEL", "gemma4:31b-cloud")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# How many model layers Ollama may put on the GPU; 0 = CPU only (the default).
# Ollama 0.34 split models between the CPU and an Intel integrated GPU, and the
# GPU part produced garbage ("WWWWWWWW...") and unreliable embeddings. On a
# machine with a proper GPU, set OLLAMA_NUM_GPU=-1 to let Ollama decide.
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "0"))

# Sampling settings per model. A model NOT listed here uses its maker's own
# recommended settings (baked into its Ollama Modelfile; see `ollama show <model>`).
#   qwen3.5:4b ships with temperature 1 (tuned for creative/thinking use), too
#     random for an agent; temperature 0 worked well in our tests.
#   lfm2.5:8b ships with temperature 0.2, top_k 80, repeat_penalty 1.05, which we keep.
#   gpt-oss (Ollama cloud) can't switch thinking off; it takes a LEVEL instead:
#     "low" keeps answers fast for a tool-calling assistant.
MODEL_SETTINGS = {
    "qwen3.5:4b": {"temperature": 0},
    "gpt-oss:20b-cloud": {"reasoning": "low"},
    "gpt-oss:120b-cloud": {"reasoning": "low"},
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# --- Google Sheet (shared with the EXO app) ---
# The Sheet's ID or full link, and the PATH to the service-account key file.
# The key file itself is never read or printed by our code; gspread reads it.
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
# Must match EXO's settings, or "today" and "this week" would differ between the two apps.
TIMEZONE = os.getenv("TIMEZONE", "Africa/Cairo")
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "EGP")

# --- Discord bot (fin_agent.discord_bot) ---
# The bot's token is a PASSWORD for the bot account: it lives only in .env.
# DISCORD_OWNER_ID is YOUR Discord user ID (a long number). The bot answers
# nobody else, and with no owner set it refuses to start.
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
_owner = os.getenv("DISCORD_OWNER_ID", "").strip()
DISCORD_OWNER_ID = int(_owner) if _owner.isdigit() else 0

# Names that redact.redact() replaces with [NAME] (comma-separated in .env).
REDACT_NAMES = [n.strip() for n in os.getenv("REDACT_NAMES", "").split(",") if n.strip()]


def settings_problems() -> list[str]:
    """What's missing or wrong in .env, one plain sentence each ([] = all fine).
    The two settings only the Discord bot needs are in discord_problems()."""
    problems = []
    if os.getenv("MODE") and not os.getenv("DATA_DIR"):
        # MODE=demo/real was replaced by DATA_DIR. Starting with the new default
        # would silently open an EMPTY history and memory, so say it instead.
        problems.append(f"MODE was replaced by DATA_DIR: change MODE={os.getenv('MODE')} "
                        f"to DATA_DIR=data/{os.getenv('MODE')} to keep using the same files")
    if not GOOGLE_SHEET_ID:
        problems.append("GOOGLE_SHEET_ID is empty: paste the Sheet's link")
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        problems.append("GOOGLE_SERVICE_ACCOUNT_FILE is empty: give the path to the service account's JSON key")
    elif not Path(GOOGLE_SERVICE_ACCOUNT_FILE).is_file():   # only whether it EXISTS; never opened here
        problems.append(f"GOOGLE_SERVICE_ACCOUNT_FILE: no file at {GOOGLE_SERVICE_ACCOUNT_FILE}")
    try:
        ZoneInfo(TIMEZONE)
    except (ZoneInfoNotFoundError, ValueError):
        problems.append(f"TIMEZONE {TIMEZONE!r} isn't a time zone name (e.g. Africa/Cairo, America/Toronto)")
    if DEFAULT_CURRENCY not in CURRENCIES:
        problems.append(f"DEFAULT_CURRENCY must be one of {', '.join(CURRENCIES)}, got {DEFAULT_CURRENCY!r}")
    return problems


def discord_problems() -> list[str]:
    """Like settings_problems(), for the Discord bot's two settings."""
    problems = []
    if not DISCORD_BOT_TOKEN:
        problems.append("DISCORD_BOT_TOKEN is empty: copy the bot's token from the Discord developer portal")
    if not _owner:
        problems.append("DISCORD_OWNER_ID is empty: put your own Discord user ID there")
    elif not DISCORD_OWNER_ID:
        problems.append("DISCORD_OWNER_ID must be a number (your Discord user ID), not a user name")
    return problems
