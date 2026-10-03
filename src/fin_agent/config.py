"""All settings in one place, read from the .env file (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Which folder under data/ holds the assistant's own files (conversation
# history, memories, sessions). data/real/ is never committed to git.
MODE = os.getenv("MODE", "demo")
if MODE not in ("demo", "real"):
    raise ValueError(f"MODE must be 'demo' or 'real', got {MODE!r}")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / MODE

# The chat model, by its Ollama name. The default is a CLOUD model (the name ends
# in "cloud"): it runs on Ollama's servers, so replies take seconds, but your
# messages and tool results are sent there. For a fully local setup use
# MODEL=qwen3.5:4b (slow on a CPU: a minute or more per reply).
MODEL = os.getenv("MODEL", "gemma4:31b-cloud")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# Run models on the CPU only. Ollama 0.34 splits models between the CPU and this
# laptop's Intel integrated GPU, and the integrated-GPU part produced garbage
# ("WWWWWWWW..."). num_gpu = number of model layers put on the GPU; 0 = none.
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
DISCORD_OWNER_ID = int(os.getenv("DISCORD_OWNER_ID") or 0)

# Names that redact.redact() replaces with [NAME] (comma-separated in .env).
REDACT_NAMES = [n.strip() for n in os.getenv("REDACT_NAMES", "").split(",") if n.strip()]
