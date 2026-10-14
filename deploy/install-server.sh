#!/usr/bin/env bash
# Sets up the Discord bot on a Linux server (Ubuntu or Debian) and keeps it
# running with systemd. Run it from the project folder, as the user the bot
# should run as (not root; it asks for sudo when it needs it):
#
#   git clone https://github.com/MazenKhafagyyy/finance-agent.git
#   cd finance-agent
#   bash deploy/install-server.sh
#
# Safe to run again: every step skips what's already done. After `git pull`,
# running it again is also how you update.
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
RUN_AS="$(id -un)"

if [ "$RUN_AS" = "root" ]; then
    echo "Run this as the user the bot will run as, not root (see 'adduser' and 'usermod -aG sudo')."
    exit 1
fi

step() { printf '\n== %s\n' "$1"; }

step "1/6 uv (installs Python and the project's packages)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "2/6 Ollama (serves the models)"
if ! command -v ollama >/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
ollama --version

step "3/6 Python packages"
uv sync --no-dev

step "4/6 Settings"
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "Created .env from .env.example. Fill it in (nano .env), copy the Google key file"
    echo "to this server (outside this folder), then run this script again."
    exit 0
fi
chmod 600 .env     # it holds the bot token: readable by this user only

step "5/6 Models"
MODEL="$(uv run --no-dev python -c 'from fin_agent import config; print(config.MODEL)')"
EMBED_MODEL="$(uv run --no-dev python -c 'from fin_agent import config; print(config.EMBED_MODEL)')"
ollama pull "$EMBED_MODEL"
case "$MODEL" in
    *cloud) echo "$MODEL runs on Ollama's servers: nothing to download. If the check below says"
            echo "Ollama refused it, run 'ollama signin', open the link it prints, then run this again." ;;
    *)      ollama pull "$MODEL" ;;
esac

uv run --no-dev fin-agent check || {
    echo
    echo "Fix the problems above, then run this script again."
    exit 1
}

step "6/6 Service (starts the bot now, at every boot, and after a crash)"
sed -e "s|__USER__|$RUN_AS|g" -e "s|__PROJECT__|$PROJECT|g" deploy/fin-agent.service \
    | sudo tee /etc/systemd/system/fin-agent.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable fin-agent
sudo systemctl restart fin-agent     # restart, not start: picks up an update too
sleep 5
sudo systemctl --no-pager status fin-agent | head -n 5

echo
echo "Done. The bot is running. Watch its log with: journalctl -u fin-agent -f"
