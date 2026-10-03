"""
Chat with your finance assistant in a Discord direct message.

    uv run python -m fin_agent.discord_bot

ONE-TIME SETUP (about 5 minutes, all on discord.com/developers/applications)
  1. New Application -> name it (e.g. EXO).
  2. "Bot" tab -> Reset Token -> copy it into .env as DISCORD_BOT_TOKEN.
     No "Privileged Gateway Intents" are needed: leave all three off.
  3. "Installation" tab -> tick "User Install" -> open the install link ->
     "Add to My Apps". That adds the bot to YOUR ACCOUNT, so no server is
     needed. (The older way also works: Guild Install with the `bot` scope
     into a server only you are in. The bot never talks there.)
  4. In Discord: Settings -> Advanced -> Developer Mode on, then right-click
     your own name -> Copy User ID -> put it in .env as DISCORD_OWNER_ID.
  5. Start this program, then open the bot's profile and send it a message.

WHAT YOU GET
    any message     -> the agent answers, with one small footer line:
                       -# get_spending_breakdown · 3.2s · 2,672 tokens
    a purchase      -> "Save this?" with Yes / No buttons (typing yes/no works too)
    /memories       -> what it remembers about you
    /forget <what>  -> delete a memory by number or topic

HOW THE PIECES FIT
    Discord DM ──► this file ──► sessions.py   which conversation is this? (10-minute rule)
                       └───────► turns.py      run it through the agent, get a plain result
This file only knows Discord: receiving, buttons, commands, formatting. Every
rule about your data lives in the other two, where the tests are.

TWO IDEAS WORTH KNOWING
  ASYNC vs BLOCKING. discord.py is asynchronous: one loop handles everything,
  including the "heartbeat" that tells Discord the bot is alive. The agent is
  ordinary blocking code that takes seconds. Called directly, it would freeze
  that loop and Discord would drop the connection. So each turn runs in a
  WORKER THREAD (`asyncio.to_thread`) while the loop stays free.

  ONE TURN AT A TIME. Two quick messages must not run through the same
  conversation at once, so every turn waits its turn behind a lock.

PRIVACY: your messages and the answers are stored by Discord. The Sheet,
memories and conversation history stay on this machine.
"""
import asyncio
import logging
import sqlite3
from datetime import datetime, timezone

import discord
import httpx
import ollama
from discord import app_commands
from langgraph.checkpoint.sqlite import SqliteSaver

from fin_agent import config
from fin_agent.agent import build_agent
from fin_agent.chat import OLLAMA_DOWN, check_ollama
from fin_agent.llm import chat_model
from fin_agent.memory import MemoryBank
from fin_agent.sessions import SessionBook
from fin_agent.sheets import SheetFormatError, SheetStore
from fin_agent.turns import TurnResult, answer_confirmation, run_turn

log = logging.getLogger("fin_agent.discord_bot")

LIMIT = 2000        # Discord rejects a message longer than this
FAILED = object()   # returned by ExoBot.work when the turn raised (you were already told why)


# ---------------------------------------------------------------------------
# Plain functions (tested in tests/test_discord_bot.py; no Discord connection)
# ---------------------------------------------------------------------------

def allowed(author_id: int, is_bot: bool, is_dm: bool, owner_id: int) -> bool:
    """Only the owner, only in a direct message. Anyone can find a bot and message
    it, and this one can read your spending, so everything else is ignored."""
    return bool(owner_id) and author_id == owner_id and is_dm and not is_bot


def split(text: str, limit: int = LIMIT) -> list[str]:
    """Cut text into pieces Discord accepts, at a line break where there is one."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit + 1)        # the last line break that still fits
        if cut <= 0:
            cut = limit                             # one huge line: cut it mid-line
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    return [part for part in parts + [text] if part]


def declined_note(result: TurnResult) -> str:
    """A save that was waiting and got cancelled by a new message is said out loud."""
    return "".join(f"🚫 Not saved: {summary}\n\n" for summary in result.declined)


def render(result: TurnResult) -> list[str]:
    """A finished reply as one or more Discord messages, with the footer on the last.
    "-# " at the start of a line is Discord's small gray "subtext"."""
    reply = result.reply or "(The model returned an empty reply. Try rephrasing.)"
    stats = [", ".join(result.tools)] if result.tools else []
    stats += [f"{result.seconds:.1f}s", f"{result.tokens:,} tokens"]
    return split(f"{declined_note(result)}{reply}\n\n-# {' · '.join(stats)}")


def confirm_text(result: TurnResult) -> str:
    return f"{declined_note(result)}❓ **Save this?**\n{result.confirm}"


def memories_text(memory: MemoryBank) -> str:
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# The Yes / No buttons
#
# A button is identified by its `custom_id`, a short text Discord sends back
# when it's clicked. Ours CONTAINS everything needed to act on the click:
#
#     exo:yes:<thread_id>:<confirm_id>
#
# so the bot keeps nothing in memory about open questions. The paused graph
# is on disk (the checkpointer), the button names it, and a click therefore
# still works after the bot was restarted or the 10-minute session ended.
# `DynamicItem` is discord.py's tool for this: it matches clicked ids against
# `template` and rebuilds the button object from the match.
# ---------------------------------------------------------------------------

class ConfirmButton(discord.ui.DynamicItem[discord.ui.Button],
                    template=r"exo:(?P<answer>yes|no):(?P<thread>[0-9a-f]+):(?P<confirm>[0-9a-f]+)"):

    def __init__(self, answer: str, thread_id: str, confirm_id: str):
        yes = answer == "yes"
        super().__init__(discord.ui.Button(
            label="Yes, save" if yes else "No",
            style=discord.ButtonStyle.success if yes else discord.ButtonStyle.secondary,
            custom_id=f"exo:{answer}:{thread_id}:{confirm_id}"))
        self.answer, self.thread_id, self.confirm_id = answer, thread_id, confirm_id

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item, match):
        return cls(match["answer"], match["thread"], match["confirm"])

    async def callback(self, interaction: discord.Interaction) -> None:
        bot: ExoBot = interaction.client
        if interaction.user.id != bot.owner_id:
            await interaction.response.send_message("This isn't your assistant.", ephemeral=True)
            return
        # Discord wants an answer to a click within 3 seconds, and the agent may
        # take longer. So first EDIT the question: that acknowledges the click and
        # removes the buttons (no double click). The real work follows.
        chosen = "✅ Yes" if self.answer == "yes" else "🚫 No"
        await interaction.response.edit_message(content=f"{interaction.message.content}\n→ {chosen}", view=None)
        async with bot.busy:
            result = await bot.work(interaction.channel, answer_confirmation,
                                    bot.agent, self.thread_id, self.confirm_id, self.answer)
            if result is None:      # a newer message cancelled it, or it was answered by typing
                await interaction.channel.send("That question was already answered or cancelled, so nothing changed.")
            elif result is not FAILED:
                await bot.show(interaction.channel, self.thread_id, result)
            bot.sessions.touch(bot.owner_id, utc_now())


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------

class ExoBot(discord.Client):

    def __init__(self, agent, memory: MemoryBank, sessions: SessionBook, owner_id: int):
        # INTENTS = which events Discord sends us. The defaults include direct
        # messages. Reading message text in SERVERS needs the privileged
        # "message content" intent, but text in a DM with the bot is always
        # delivered, and DMs are all we handle.
        super().__init__(intents=discord.Intents.default())
        self.agent, self.memory, self.sessions, self.owner_id = agent, memory, sessions, owner_id
        self.busy = asyncio.Lock()
        self.tree = app_commands.CommandTree(self)     # the slash commands
        self.add_commands()

    async def setup_hook(self) -> None:
        """Runs once after login, before any event arrives."""
        self.add_dynamic_items(ConfirmButton)          # recognise our buttons, even ones sent before a restart
        await self.tree.sync()                         # tell Discord which slash commands exist

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user}. Model {config.MODEL}. DM me from the owner account. Ctrl+C to stop.")

    async def on_message(self, message: discord.Message) -> None:
        is_dm = isinstance(message.channel, discord.DMChannel)
        if not allowed(message.author.id, message.author.bot, is_dm, self.owner_id):
            return
        text = message.content.strip()
        if not text:                                   # e.g. a picture with no caption
            return
        async with self.busy:
            # Decided INSIDE the lock: by now the previous turn has finished and
            # set the session clock, so a quick second message joins its session.
            thread_id, _ = self.sessions.current_thread(self.owner_id, utc_now())
            result = await self.work(message.channel, run_turn, self.agent, thread_id, text)
            if result is not FAILED:
                await self.show(message.channel, thread_id, result)
            self.sessions.touch(self.owner_id, utc_now())   # the 10 minutes count from the reply

    async def work(self, channel, blocking_function, *args):
        """Run blocking agent code in a worker thread while Discord shows "typing…".
        Returns its result, or FAILED after telling you what went wrong."""
        try:
            async with channel.typing():
                return await asyncio.to_thread(blocking_function, *args)
        except httpx.ConnectError:
            await channel.send(f"⚠️ {OLLAMA_DOWN}")
        except ollama.ResponseError as err:
            await channel.send(f"⚠️ The model returned an error: {err}. Try again or rephrase.")
        except Exception as err:    # anything else: say so and keep the bot running
            log.exception("turn failed")                # full details in the console, not in the chat
            await channel.send(f"⚠️ Something went wrong ({type(err).__name__}). "
                               f"Nothing was saved unless you saw \"Saved\". Details are in the bot's console.")
        return FAILED

    async def show(self, channel, thread_id: str, result: TurnResult) -> None:
        if result.confirm:
            view = discord.ui.View(timeout=None)        # None = the buttons never stop working
            view.add_item(ConfirmButton("yes", thread_id, result.confirm_id))
            view.add_item(ConfirmButton("no", thread_id, result.confirm_id))
            await channel.send(confirm_text(result), view=view)
            return
        for part in render(result):
            await channel.send(part)

    def add_commands(self) -> None:
        async def reply(interaction: discord.Interaction, make_text) -> None:
            # `make_text` is a FUNCTION, called only for the owner: a stranger's
            # /forget must not delete anything, not merely get a different reply.
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("This isn't your assistant.", ephemeral=True)
                return
            first, *rest = split(make_text())
            await interaction.response.send_message(first)
            for part in rest:
                await interaction.followup.send(part)

        # Where the slash commands exist: whichever way the bot was installed
        # (to your account or to a server), but ONLY in the DM with the bot.
        dm_only = app_commands.allowed_contexts(guilds=False, dms=True, private_channels=False)
        any_install = app_commands.allowed_installs(guilds=True, users=True)

        @self.tree.command(name="memories", description="List what the assistant remembers about you")
        @any_install
        @dm_only
        async def memories(interaction: discord.Interaction):
            await reply(interaction, lambda: memories_text(self.memory))

        @self.tree.command(name="forget", description="Delete a memory by number or topic")
        @app_commands.describe(what="The memory's number or topic, as shown by /memories")
        @any_install
        @dm_only
        async def forget(interaction: discord.Interaction, what: str):
            await reply(interaction, lambda: forget_text(self.memory, what))


def main() -> None:
    if not config.DISCORD_BOT_TOKEN or not config.DISCORD_OWNER_ID:
        raise SystemExit("Set DISCORD_BOT_TOKEN and DISCORD_OWNER_ID in .env (steps at the top of discord_bot.py).")
    # From here on, the same start-up as the terminal chat (chat.py).
    check_ollama()
    store = SheetStore(config.GOOGLE_SHEET_ID, config.GOOGLE_SERVICE_ACCOUNT_FILE)
    try:
        store.setup()   # checks every tab's columns BEFORE anything can be written
    except SheetFormatError as err:
        raise SystemExit(f"Stopped: {err}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: each turn runs in a worker thread (see `work`).
    saver = SqliteSaver(sqlite3.connect(config.DATA_DIR / "assistant.db", check_same_thread=False))
    memory = MemoryBank(config.DATA_DIR / "memories.db")
    agent = build_agent(store, llm=chat_model(reasoning=False), checkpointer=saver, memory=memory)
    sessions = SessionBook(config.DATA_DIR / "sessions.db")
    # run() connects and blocks until Ctrl+C. The token is passed along, never printed.
    ExoBot(agent, memory, sessions, config.DISCORD_OWNER_ID).run(config.DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
