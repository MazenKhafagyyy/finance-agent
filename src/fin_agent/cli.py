"""
The `fin-agent` command. pyproject.toml declares it under [project.scripts],
so `uv sync` installs it into the project's virtual environment:

    uv run fin-agent check    is everything set up? tries each piece, says how to fix it
    uv run fin-agent bot      run the Discord bot
    uv run fin-agent chat     chat in the terminal (shows thinking and tool calls)
    uv run fin-agent hello    make the bot message you first, if you can't find it in Discord
"""
import argparse
import sys

COMMANDS = {
    "check": "test the setup and say how to fix what's missing",
    "bot": "run the Discord bot",
    "chat": "chat with the assistant in this terminal",
    "hello": "make the bot send you a first message on Discord",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fin-agent", description="A personal finance assistant on a Google Sheet.")
    commands = parser.add_subparsers(dest="command", metavar="command")
    for name, help_text in COMMANDS.items():
        commands.add_parser(name, help=help_text, description=help_text)
    command = parser.parse_args(argv).command
    if command is None:             # plain `fin-agent`: show the commands, not just an error
        parser.print_help(sys.stderr)
        sys.exit(2)

    # Emoji and ✓ must not crash output that goes to a file or a service log
    # (Windows writes those in an old code page that has no such characters).
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    # Imported here, not at the top: each command loads only what it needs,
    # and `fin-agent --help` stays instant.
    if command == "check":
        from fin_agent.checks import run
        sys.exit(run())
    elif command == "bot":
        from fin_agent.discord_bot import main as run_bot
        run_bot()
    elif command == "chat":
        from fin_agent.chat import main as run_chat
        run_chat()
    elif command == "hello":
        from fin_agent.discord_bot import hello
        hello()
