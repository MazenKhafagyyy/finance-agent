@echo off
rem Starts the Discord bot. The "Finance Agent" desktop shortcut points at this file.
rem Close the window (or press Ctrl+C) to stop the bot.
title Finance Agent

rem %~dp0 is the folder this file is in (deploy\), so ..\ is the project root.
rem That keeps the launcher working if the project folder is ever moved.
cd /d "%~dp0.."

rem The bot refuses to start when Ollama isn't answering (check_ollama in checks.py),
rem so open the Ollama app first if it isn't already running, then wait for it.
set OLLAMA_TAGS=http://127.0.0.1:11434/api/tags
curl -s -o nul %OLLAMA_TAGS%
if not errorlevel 1 goto run

echo Starting Ollama...
start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
set tries=0
:wait
timeout /t 1 /nobreak >nul
curl -s -o nul %OLLAMA_TAGS%
if not errorlevel 1 goto run
set /a tries+=1
if %tries% lss 30 goto wait
echo Ollama did not answer after 30 seconds. Open the Ollama app, then run this again.
goto end

:run
echo Starting the bot. Wait for "Logged in as ...", then message it on Discord.
echo.
uv run fin-agent bot

:end
rem Keep the window open so a start-up error can be read before it disappears.
echo.
echo The bot has stopped.
pause
