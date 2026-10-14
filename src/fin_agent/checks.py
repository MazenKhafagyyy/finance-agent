"""
IS EVERYTHING SET UP?  Two levels of checking:

    check_ollama()   the quick check every start-up does: is Ollama answering?
    run()            `uv run fin-agent check`: tries every piece the assistant
                     needs, in order, and says how to fix whatever fails

`fin-agent check` is the finish line of an install: fill in .env, run it, fix
what it reports, and run it again until every line is ✓. It prints counts
only, never a transaction, so it's safe to run while someone watches.

Each check is a small function that returns a line to show when it passes,
raises Problem (with the fix) when it fails, or raises Skip when it can't run
yet because something it depends on isn't ready.
"""
import asyncio
import textwrap
import time

import httpx

from fin_agent import config

OLLAMA_DOWN = (f"Can't reach Ollama at {config.OLLAMA_URL}. Start it (open the Ollama app, "
               f"or run `ollama serve` in another terminal), then try again.")


def pulled_models() -> set[str]:
    """Names of the models downloaded into Ollama. Raises httpx.HTTPError when
    Ollama isn't answering."""
    tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()
    return {m["name"] for m in tags.get("models", [])}


def is_pulled(model: str, pulled: set[str]) -> bool:
    # CLOUD models (names ending in "cloud") run on Ollama's servers and are never
    # downloaded, so they never appear in the local list. Checking for them there
    # produced a false warning on every start.
    # "qwen3.5:4b" and "lfm2.5" (= "lfm2.5:latest") are both valid ways to name a model.
    return model.endswith("cloud") or model in pulled or f"{model}:latest" in pulled


def check_ollama() -> None:
    """Fail early with a clear message instead of a long traceback on the first question."""
    try:
        pulled = pulled_models()
    except httpx.HTTPError:
        raise SystemExit(OLLAMA_DOWN)
    for model in (config.MODEL, config.EMBED_MODEL):
        if not is_pulled(model, pulled):
            print(f"Warning: model {model!r} isn't in `ollama list`. Run `ollama pull {model}`.")


# ---------------------------------------------------------------------------
# `fin-agent check`
# ---------------------------------------------------------------------------

class Problem(Exception):
    """This check failed. The message says how to fix it."""


class Skip(Exception):
    """This check can't run yet. The message says why."""


def _settings() -> str:
    problems = config.settings_problems()
    if problems:
        raise Problem("Fix these in .env:\n" + "\n".join(f"- {p}" for p in problems))
    return f"Settings in .env (Sheet, {config.TIMEZONE}, {config.DEFAULT_CURRENCY})"


def _ollama() -> str:
    try:
        pulled_models()
    except httpx.HTTPError:
        raise Problem(OLLAMA_DOWN)
    return f"Ollama is running at {config.OLLAMA_URL}"


def _model_ready(model: str) -> None:
    """Skip if Ollama is down (already reported); Problem if the model isn't downloaded."""
    try:
        pulled = pulled_models()
    except httpx.HTTPError:
        raise Skip("needs Ollama")
    if not is_pulled(model, pulled):
        raise Problem(f"Download it: ollama pull {model}")


def _chat_model() -> str:
    import ollama
    from fin_agent.llm import chat_model

    _model_ready(config.MODEL)
    started = time.time()
    try:
        chat_model().invoke("Reply with the single word OK.")
    except ollama.ResponseError as err:
        hint = "\nCloud models need you signed in to Ollama: run `ollama signin`." if config.MODEL.endswith("cloud") else ""
        raise Problem(f"Ollama refused the model: {err}{hint}")
    return f"Chat model {config.MODEL} answers ({time.time() - started:.1f}s)"


def _embedding_model() -> str:
    from fin_agent.memory import default_embed

    _model_ready(config.EMBED_MODEL)
    default_embed(["test"])
    return f"Embedding model {config.EMBED_MODEL} works (long-term memory)"


def _sheet() -> str:
    import gspread
    from google.auth.exceptions import GoogleAuthError

    from fin_agent.sheet_format import is_unresolved
    from fin_agent.sheets import SheetFormatError, SheetStore

    if not config.GOOGLE_SHEET_ID or not config.GOOGLE_SERVICE_ACCOUNT_FILE:
        raise Skip("needs GOOGLE_SHEET_ID and GOOGLE_SERVICE_ACCOUNT_FILE")
    try:
        store = SheetStore(config.GOOGLE_SHEET_ID, config.GOOGLE_SERVICE_ACCOUNT_FILE)
        created = store.setup()     # creates any missing tabs; checks the columns of the others
        snap = store.load()
    except FileNotFoundError:
        raise Skip("needs the key file (see Settings)")
    except (ValueError, KeyError):
        raise Problem("GOOGLE_SERVICE_ACCOUNT_FILE isn't a service-account key. In the Google Cloud "
                      "console, open the service account, Keys, Add key, JSON.")
    except GoogleAuthError:
        raise Problem("Google rejected the service-account key (deleted or disabled?). Create a new JSON key.")
    except gspread.exceptions.SpreadsheetNotFound:
        raise Problem("Google can't open the Sheet. Check GOOGLE_SHEET_ID, and share the Sheet as Editor "
                      "with the service account (the client_email inside the key file).")
    except gspread.exceptions.APIError as err:
        raise Problem(f"Google refused ({err.response.status_code}). Is the Google Sheets API enabled "
                      f"in the service account's Cloud project?")
    except SheetFormatError as err:
        raise Problem(str(err))
    line = (f"Google Sheet {store.book.title!r}: {len(snap.entries)} transactions "
            f"({sum(is_unresolved(e) for e in snap.entries)} unresolved), "
            f"{len(snap.categories)} categories, {len(snap.merchants)} merchants")
    return line + (f"; created tabs {', '.join(created)}" if created else "")


def _discord() -> str:
    from fin_agent.discord_bot import OwnerUnreachable, contact_owner

    if not config.DISCORD_BOT_TOKEN and not config.DISCORD_OWNER_ID:
        raise Skip("not set up (only `fin-agent bot` needs it)")
    problems = config.discord_problems()
    if problems:
        raise Problem("Fix these in .env:\n" + "\n".join(f"- {p}" for p in problems))
    try:
        found = asyncio.run(contact_owner(greet=False))
    except OwnerUnreachable as err:
        raise Problem(str(err))
    return f"Discord: {found}"


def _data_dir() -> str:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    probe = config.DATA_DIR / ".write-test"
    probe.touch()
    probe.unlink()
    return f"Data folder {config.DATA_DIR}"


CHECKS = [("Settings", _settings), ("Ollama", _ollama), ("Chat model", _chat_model),
          ("Embedding model", _embedding_model), ("Google Sheet", _sheet),
          ("Discord", _discord), ("Data folder", _data_dir)]


def run() -> int:
    """Run every check and print one line each. Returns the exit code (0 = all fine)."""
    print(f"Checking the setup (settings from {config.PROJECT_ROOT / '.env'})\n")
    failed = 0
    for name, check in CHECKS:
        try:
            print(f"  ✓ {check()}")
        except Skip as why:
            print(f"  - {name}: skipped, {why}")
        except Problem as fix:
            failed += 1
            print(f"  ✗ {name}\n{textwrap.indent(str(fix), '      ')}")
        except Exception as err:    # an unexpected failure is still a line, not a traceback
            failed += 1
            print(f"  ✗ {name}\n      {type(err).__name__}: {err}")
    if failed:
        print(f"\n{failed} problem{'s' if failed > 1 else ''}. Fix {'them' if failed > 1 else 'it'}, "
              f"then run `uv run fin-agent check` again.")
        return 1
    print("\nAll set. Start the bot with `uv run fin-agent bot` (or `uv run fin-agent chat` in the terminal).")
    return 0
