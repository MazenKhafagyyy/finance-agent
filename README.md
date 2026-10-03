# Finance agent

A personal finance assistant you text on Discord. Send it a direct message
about what you spent, or ask where your money went, and it answers from a
Google Sheet of your transactions. It asks before it writes anything, and it
remembers lasting facts about you between conversations.

```
You   285 mcdonalds
Bot   ❓ Save this?
      285 EGP · McDonald's → Fast Food (non-essential) · Fri 2026-10-02 (today)
      [ Yes, save ]  [ No ]

You   how much did I spend this week?
Bot   You spent 1,240 EGP this week (Sat–Fri). Fast Food was the biggest part: 480 EGP.
      get_spending_breakdown · 3.2s · 2,672 tokens
```

(An illustration of the format, not real data.)

It is built with [LangGraph](https://langchain-ai.github.io/langgraph/),
[discord.py](https://discordpy.readthedocs.io) and a model served by
[Ollama](https://ollama.com). It was written as a learning project, so the code
is small and heavily commented: each module explains the idea it implements
and the real failures that shaped it.

## What you can say

| You send | The bot does |
|---|---|
| `285 mcdonalds` | Recognises a known merchant and offers to save the row, without calling the model |
| "I paid 200 for lunch at mcdonalds yesterday" | Builds the row, shows it with **Yes / No** buttons, and saves it only after a yes |
| "How much did I spend this week?" | Sums the Sheet with code and answers in a sentence |
| "What unessential stuff am I spending on?" | Ranks non-essential spending and compares it with a fair previous period |
| "Am I over budget?" | Compares this month with each category's budget |
| "Remember I get paid on the 25th" | Saves a long-term memory it uses in later conversations |
| `/memories`, `/forget payday` | Lists what it remembers; deletes a memory |

## How it works

```
 Discord DM ─► discord_bot.py ─► sessions.py   which conversation is this?
                    │
                    └──────────► turns.py ───► agent (LangGraph) ─► tools (plain Python) ─► Google Sheet
                                                    │
                                                    └─► memory (SQLite, on your machine)
```

- **Only you, only in a DM.** The bot answers one Discord account and ignores
  everyone and everywhere else.
- **Code does the math, the model does the language.** Every number comes from
  a Python tool; the model picks the tool and words the answer.
- **Nothing is written without a yes.** Adding an expense pauses the agent and
  shows buttons. Typing something else instead cancels the save, and an old
  button can never answer a newer question.
- **Sessions by silence.** A message more than 10 minutes after the bot's last
  reply starts a new conversation, which keeps the history sent to the model
  short. Long-term memories carry over.
- **Guardrails in code.** A reply that claims something was saved when nothing
  was gets a visible correction.

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you)
- [Ollama](https://ollama.com), installed and running, with an Ollama account
  for the default cloud model
- A Google account, for the Sheet and a service account
- A Discord account

## Setup

### 1. Install

```bash
git clone https://github.com/MazenKhafagyyy/finance-agent.git
cd finance-agent
uv sync
```

### 2. Models

The default chat model is `gemma4:31b-cloud`, which runs on Ollama's servers:
replies take seconds and there is nothing to download. Sign in to Ollama first
(open the Ollama app, or run `ollama signin`).

Long-term memory uses a small embedding model that runs on your machine:

```bash
ollama pull qwen3-embedding:0.6b
```

Prefer to keep everything local? Run `ollama pull qwen3.5:4b` and set
`MODEL=qwen3.5:4b` in `.env`. It works on a CPU, slowly: expect a minute or
more per reply.

### 3. Google Sheet

1. In the [Google Cloud console](https://console.cloud.google.com), create a
   project and enable the **Google Sheets API**.
2. Create a **service account** and download its JSON key. Keep that file
   outside this folder; it is a password.
3. Create an empty Google Sheet and **share it with the service account's
   email** as Editor.

The assistant creates the tabs it needs (Transactions, Categories, Merchants,
Wallet) the first time it connects.

### 4. Discord bot

At [discord.com/developers/applications](https://discord.com/developers/applications):

1. **New Application**, then open the **Bot** tab and click **Reset Token**.
   Copy the token for the next step. Leave the privileged intents off.
2. Open the **Installation** tab, tick **User Install**, open the install link
   and choose **Add to My Apps**. This adds the bot to your account, so no
   server is needed.
3. In Discord, turn on Developer Mode (Settings, Advanced), right-click your
   own name and choose **Copy User ID**.

### 5. Settings

```bash
cp .env.example .env
```

Then edit `.env`:

| Setting | Meaning |
|---|---|
| `DISCORD_BOT_TOKEN` | The bot token from step 4. It is a password: never share or commit it |
| `DISCORD_OWNER_ID` | Your own Discord user ID. The bot answers only this account |
| `GOOGLE_SHEET_ID` | The Sheet's link or ID |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | Path to the JSON key file |
| `MODEL` | The Ollama chat model (default `gemma4:31b-cloud`) |
| `MODE` | `demo` or `real`: which folder under `data/` holds the assistant's own files. `data/real/` is never committed |
| `TIMEZONE`, `DEFAULT_CURRENCY` | Used for "today", "this week" and amounts |

Check the Sheet connection (prints counts only, never your transactions):

```bash
uv run python scripts/sheet_check.py
```

## Run the bot

```bash
uv run python -m fin_agent.discord_bot
```

When it prints `Logged in as ...`, open your direct message with the bot and
type. If you can't find the bot in Discord, make it message you first:

```bash
uv run python scripts/discord_hello.py
```

Each reply ends with one small line showing the tools it used, the time taken
and the tokens spent.

### Keeping it online

The bot answers only while this program is running, so it needs a machine that
stays on: a spare laptop, a small home server or a rented one. It connects
outward to Discord, so it needs no open ports and no public address.

## Terminal chat

The same assistant, without Discord. Useful for development, and it shows more
detail (the model's thinking, each tool call and its result):

```bash
uv run python -m fin_agent.chat
```

Commands: `/new` (fresh conversation), `/memories`, `/forget <number or topic>`,
`/tokens`, `/quit`.

## Tests

```bash
uv run pytest
```

The tests use a fake Sheet, a scripted fake model and a stand-in Discord
channel, so they need no network, no Ollama and no accounts.

## Project layout

| Path | What it is |
|---|---|
| `src/fin_agent/discord_bot.py` | The Discord bot: DMs, buttons, slash commands, formatting |
| `src/fin_agent/turns.py` | One message in, one result out; what a waiting "Save this?" means |
| `src/fin_agent/sessions.py` | The 10-minute session rule |
| `src/fin_agent/agent.py` | The LangGraph agent: prompt, tools, confirmation pause, guardrail |
| `src/fin_agent/tools.py`, `analysis.py`, `dates.py` | The calculations the model can call |
| `src/fin_agent/sheets.py`, `sheet_format.py` | Reading and writing the Sheet, and a fake for tests |
| `src/fin_agent/memory.py` | Long-term memory |
| `src/fin_agent/chat.py` | Terminal chat |
| `src/fin_agent/quickline.py`, `redact.py`, `llm.py`, `stream_output.py` | Quick-line parser, sensitive-number patterns, model setup, terminal output |
| `scripts/sheet_check.py` | Check the Sheet connection (counts only) |
| `scripts/discord_hello.py` | Make the bot message you first |
| `scripts/try_tools.py` | Run the tools by hand on your Sheet, without the model |
| `scripts/memory_trial.py` | Live test of long-term memory with the real model, on fake data |
| `tests/` | The test suite |

## Privacy

- **The default model is a cloud model.** Your messages and the tool results
  (which include your transactions) are sent to Ollama's cloud. Set
  `MODEL=qwen3.5:4b` to keep the model on your machine.
- **Discord stores the messages** you exchange with the bot.
- Conversation history and memories are SQLite files on the machine that runs
  the bot.
- `.env`, key files, `data/real/` and every `*.db` file are ignored by git.
- Memories are refused if they look like a transaction or contain a card,
  account or ID number, and you can list and delete them at any time.

## License

[MIT](LICENSE)
