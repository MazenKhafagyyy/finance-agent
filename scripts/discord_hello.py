"""
Make the bot message YOU first, so the conversation appears in your DM list.

    uv run python scripts/discord_hello.py

Use it once, when you can't find the bot in Discord. It logs in, sends one
message to DISCORD_OWNER_ID and exits (the bot itself can keep running).
If Discord refuses, it prints why and what to fix.

How: Discord has two channels to a bot. The GATEWAY is the live connection the
running bot listens on. The REST API is plain web requests ("send this message").
This script only uses REST, so it never competes with the running bot.
"""
import asyncio

import discord

from fin_agent import config


async def main() -> None:
    if not config.DISCORD_BOT_TOKEN or not config.DISCORD_OWNER_ID:
        raise SystemExit("Set DISCORD_BOT_TOKEN and DISCORD_OWNER_ID in .env first.")
    client = discord.Client(intents=discord.Intents.none())
    async with client:
        await client.login(config.DISCORD_BOT_TOKEN)      # REST only: no gateway connection
        print(f"Logged in as the bot: {client.user}")
        try:
            owner = await client.fetch_user(config.DISCORD_OWNER_ID)
        except discord.NotFound:
            raise SystemExit("DISCORD_OWNER_ID is not a Discord user. Copy YOUR user ID again "
                             "(Developer Mode on, right-click your own name, Copy User ID).")
        print(f"Owner account: {owner}   <- this must be you")
        try:
            await owner.send("👋 I'm online. Message me here: try `how much did I spend this week?`")
        except discord.Forbidden:
            raise SystemExit(
                "Discord refused the message: the bot may not DM this account yet.\n"
                "That means it is neither installed to your account nor in a server with you.\n"
                "Fix ONE of these, then run this script again:\n"
                "  a) Developer portal -> Installation -> tick User Install -> open the install link\n"
                "     in a browser while logged in as the owner account -> Add to My Apps.\n"
                "  b) Add the bot to a server you are in (OAuth2 -> URL Generator -> scope `bot`).\n"
                "Also check Discord Settings -> Content & Social -> allow DMs.")
        print("Sent. Open Discord: the conversation with the bot is now at the top of your Direct Messages.")


if __name__ == "__main__":
    asyncio.run(main())
