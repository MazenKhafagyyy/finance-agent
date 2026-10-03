"""Render model chunks immediately, with thinking in dim gray on terminals, plus
a live status line (spinner, elapsed time, token counts) pinned to the bottom row."""
import os
import shutil
import sys
import threading
import time

GRAY = "\033[2;90m"
RESET = "\033[0m"

# --- ANSI escape sequences used by the status line -----------------------------
SAVE_CURSOR = "\0337"          # remember where the cursor is (and the current colour)
RESTORE_CURSOR = "\0338"       # jump back to it
CLEAR_LINE = "\033[2K"
RESET_SCROLL_REGION = "\033[r"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def scroll_region(top: int, bottom: int) -> str:
    """Tell the terminal that only rows top..bottom scroll. Rows outside stay put."""
    return f"\033[{top};{bottom}r"


def move_to(row: int, column: int = 1) -> str:
    return f"\033[{row};{column}H"


def terminal_color(output) -> bool:
    """Keep redirected logs plain; enable ANSI colors on Windows consoles."""
    if not output.isatty() or "NO_COLOR" in os.environ or os.getenv("TERM") == "dumb":
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
    return True


class TokenPrinter:
    """One model response. Tool-call JSON is handled separately by chat.py.

    Some models put reasoning in a separate field; others emit literal think
    tags. Retain only a possible tag prefix between chunks, so normal text is
    flushed immediately even when the model splits a tag across tokens.
    """

    def __init__(self, output=None, *, color=None):
        self.output = output if output is not None else sys.stdout
        self.color = terminal_color(self.output) if color is None else color
        self.section = None
        self.thinking = False
        self.pending = ""
        self.answer = ""

    def _write(self, text: str, *, thinking: bool) -> None:
        if not text:
            return
        section = "thinking" if thinking else "answer"
        if self.section != section:
            self._end_section()
            label = "  thinking> " if thinking else "assistant> "
            self.output.write("\n" + (GRAY if thinking and self.color else "") + label)
            self.section = section
        self.output.write(text)
        self.output.flush()
        if not thinking:
            self.answer += text

    def _end_section(self) -> None:
        if self.section is not None:
            if self.section == "thinking" and self.color:
                self.output.write(RESET)
            self.output.write("\n")
            self.output.flush()
            self.section = None

    def feed(self, chunk) -> None:
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            self._write(reasoning, thinking=True)
        self.pending += chunk.text
        while self.pending:
            tag = "</think>" if self.thinking else "<think>"
            position = self.pending.find(tag)
            if position >= 0:
                self._write(self.pending[:position], thinking=self.thinking)
                self.pending = self.pending[position + len(tag):]
                self.thinking = not self.thinking
                continue
            # Keep the longest suffix that could become a tag next time.
            keep = next((n for n in range(min(len(tag) - 1, len(self.pending)), 0, -1)
                         if self.pending.endswith(tag[:n])), 0)
            ready = self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            self._write(ready, thinking=self.thinking)
            break

    def finish(self) -> str:
        """Finish a response (also on errors), always restoring normal color."""
        self._write(self.pending, thinking=self.thinking)
        self.pending = ""
        self._end_section()
        return self.answer.strip()


class StatusBar:
    """A live status line pinned to the BOTTOM ROW of the terminal while the
    assistant works, like the token counter in Claude Code:

        ⠹ 1.2s · answering · ↑ 2,720 in · ↓ 31 out · chat 4,041 tokens

    HOW THE PINNING WORKS
    Terminals have a "scroll region": we tell the terminal that only rows
    1..N-1 scroll, which leaves row N free. The answer keeps streaming and
    scrolling in the region above; to update row N we save the cursor, jump to
    the last row, rewrite it, and jump back. A background thread redraws about
    10 times a second, so the spinner and the clock move even while nothing is
    arriving from the model.

    THE NUMBERS
    ↑ in   tokens the model has READ this turn. Exact, but only known when a
           model call finishes (shown as "…" before that).
    ↓ out  tokens the model has WRITTEN this turn. Live ESTIMATE while
           streaming (one streamed chunk is about one token), replaced by the
           model's exact count when the call finishes.
    chat   the conversation total (the saved total + this turn so far).

    It is only active on a real terminal. When output is redirected (tests,
    scripts, logs) every method does nothing, so nothing else has to care.
    """

    def __init__(self, output=None, *, enabled=None, animate=True, size=None):
        self.output = output if output is not None else sys.stdout
        if enabled is None:
            # terminal_color() also switches on ANSI escape handling on Windows.
            # LIVE_STATUS=0 in .env turns the status line off.
            enabled = os.getenv("LIVE_STATUS", "1") != "0" and terminal_color(self.output)
        self.enabled = enabled
        self.animate = animate              # False in tests: no background thread
        self.size = size                    # (columns, rows); None = ask the terminal
        self.lock = threading.Lock()        # the ticker thread and the main thread both write
        self.active = False
        self.rows = self.columns = 0
        self.started = time.time()
        self.phase = "waiting"
        self.read = 0                       # exact, from finished model calls
        self.written = 0                    # exact, from finished model calls
        self.live_written = 0               # estimate for the call in progress
        self.calls_done = 0
        self.chat_before = 0                # the chat's total before this turn
        self._halt = threading.Event()
        self._thread = None

    # --- the numbers -------------------------------------------------------------

    def begin_turn(self, chat_total: int) -> None:
        self.started = time.time()
        self.chat_before = chat_total
        self.read = self.written = self.live_written = self.calls_done = 0
        self.phase = "waiting"

    def add_chunk(self, chunk) -> None:
        """A piece of the model's reply arrived. Count it and note what kind it is."""
        if chunk.additional_kwargs.get("reasoning_content"):
            self.phase = "thinking"
        elif getattr(chunk, "tool_call_chunks", None):
            self.phase = "calling a tool"
        elif chunk.text:
            self.phase = "answering"
        else:
            return                          # an empty bookkeeping chunk: not a token
        self.live_written += 1

    def call_done(self, read: int, written: int) -> None:
        """A model call finished: swap the estimate for the exact counts it reported."""
        self.read += read
        self.written += written
        self.live_written = 0
        self.calls_done += 1
        self.phase = "running tools"

    def text(self) -> str:
        elapsed = time.time() - self.started
        spinner = SPINNER[int(elapsed * 10) % len(SPINNER)]
        written = self.written + self.live_written
        read = f"{self.read:,}" if self.calls_done else "…"
        chat = self.chat_before + self.read + written
        return (f"{spinner} {elapsed:.1f}s · {self.phase} · ↑ {read} in · ↓ {written:,} out"
                f" · chat {chat:,} tokens")

    # --- the terminal --------------------------------------------------------------

    def start(self) -> None:
        """Reserve the bottom row and start redrawing it."""
        if not self.enabled or self.active:
            return
        self.columns, self.rows = self.size or shutil.get_terminal_size()
        if self.rows < 5:
            return                          # too small to give a row away
        with self.lock:
            # "\n" then "cursor up": makes sure there is a free row below the cursor
            # (if we were on the last row, the screen scrolls up by one).
            # Setting a scroll region moves the cursor to the top, so we save and
            # restore the position around it.
            self.output.write("\n\033[1A" + SAVE_CURSOR + scroll_region(1, self.rows - 1) + RESTORE_CURSOR)
            self.output.flush()
            self.active = True
        self.draw()
        if self.animate:
            self._halt.clear()
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

    def _tick(self) -> None:
        while not self._halt.wait(0.1):     # every 0.1s until stop() sets the event
            self.draw()

    def draw(self) -> None:
        line = self.text()[: max(self.columns - 1, 0)]  # never wrap onto a second row
        with self.lock:
            if not self.active:
                return
            # One write() call for the whole sequence, so it can't be interleaved
            # with the answer being streamed by the main thread.
            self.output.write(SAVE_CURSOR + move_to(self.rows) + CLEAR_LINE + GRAY + line + RESET + RESTORE_CURSOR)
            self.output.flush()

    def stop(self) -> None:
        """Give the bottom row back and erase the status line. Safe to call twice,
        and always called on errors / Ctrl+C so the terminal is never left broken."""
        if self._thread is not None:
            self._halt.set()
            self._thread.join(timeout=1)
            self._thread = None
        with self.lock:
            if not self.active:
                return
            self.active = False
            self.output.write(SAVE_CURSOR + RESET_SCROLL_REGION + move_to(self.rows) + CLEAR_LINE + RESTORE_CURSOR)
            self.output.flush()
