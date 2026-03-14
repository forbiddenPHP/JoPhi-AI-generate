"""
progress — Unified progress/output streaming for worker subprocesses.

Replaces subprocess.run(capture_output=True) with streaming Popen that
reads stderr in real-time, parses progress events, and collects stdout
for JSON results.

Usage in revoicer.py:
    from progress import run_worker

    result = run_worker(cmd, timeout=600, on_event=print_event)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr_tail}", file=sys.stderr)
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Events ────────────────────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """Structured event emitted by a worker parser."""
    type: str               # "progress" | "stage" | "log" | "error" | "warning"
    message: str            # raw line or parsed message
    percent: float | None = None
    stage: str | None = None
    ts: float = 0.0

    def __post_init__(self):
        if self.ts == 0.0:
            self.ts = time.time()

    def to_json(self) -> str:
        d = {"type": self.type, "message": self.message, "ts": self.ts}
        if self.percent is not None:
            d["percent"] = round(self.percent, 1)
        if self.stage is not None:
            d["stage"] = self.stage
        return json.dumps(d, ensure_ascii=False)


# ── Parsers ───────────────────────────────────────────────────────────────────

# Regex for tqdm-style progress: "  5%|█         | 5/100 [00:02<00:38, 2.50it/s]"
_TQDM_RE = re.compile(
    r"(\d+)%\|"           # percent
    r"[^|]*\|\s*"         # bar
    r"(\d+)/(\d+)"        # current/total
)

# Simpler tqdm: "100%|██████████| 5/5"
_TQDM_SIMPLE_RE = re.compile(r"(\d+)%\|")

# ffmpeg time progress: "time=00:01:23.45"
_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")

# Worker counter format: "[1/5] enhance: file.wav"
_COUNTER_RE = re.compile(r"\[(\d+)/(\d+)\]")


def parse_stderr_line(line: str, duration_s: float = 0.0) -> ProgressEvent:
    """Parse a single stderr line into a ProgressEvent.

    This is a universal parser that detects the format automatically:
    - tqdm progress bars → progress event with percent
    - [i/n] counter lines → progress event with percent
    - ffmpeg time= lines → progress event with percent (needs duration_s)
    - Everything else → log event
    """
    stripped = line.strip()
    if not stripped:
        return ProgressEvent(type="log", message="")

    # Check for tqdm-style progress
    m = _TQDM_RE.search(stripped)
    if m:
        pct = float(m.group(1))
        return ProgressEvent(type="progress", message=stripped, percent=pct)

    m = _TQDM_SIMPLE_RE.search(stripped)
    if m:
        pct = float(m.group(1))
        return ProgressEvent(type="progress", message=stripped, percent=pct)

    # Check for [i/n] counter
    m = _COUNTER_RE.search(stripped)
    if m:
        current = int(m.group(1))
        total = int(m.group(2))
        pct = (current / total) * 100 if total > 0 else 0
        return ProgressEvent(type="progress", message=stripped, percent=pct)

    # Check for ffmpeg time=
    if duration_s > 0:
        m = _FFMPEG_TIME_RE.search(stripped)
        if m:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            pct = min(100.0, (t / duration_s) * 100)
            return ProgressEvent(type="progress", message=stripped, percent=pct)

    # Check for explicit error markers (only real errors, not warnings containing "exception")
    if stripped.startswith("ERROR:") or stripped.startswith("ERROR ") or stripped.startswith("Traceback"):
        return ProgressEvent(type="error", message=stripped)

    # Check for warnings
    if stripped.startswith("WARNING:") or stripped.startswith("WARNING "):
        return ProgressEvent(type="warning", message=stripped)

    # Stage-like messages (e.g., "Loading model ...", "Separating stems ...")
    if stripped.endswith("...") or stripped.endswith("…"):
        return ProgressEvent(type="stage", message=stripped, stage=stripped.rstrip(".… "))

    # Default: log — everything else is informational
    return ProgressEvent(type="log", message=stripped)


# Known noise patterns to skip (warnings from conda envs)
_NOISE_PATTERNS = (
    "FutureWarning",
    "UserWarning",
    "pynvml",
    "import pynvml",
    "ds_accelerator",
    "bitsandbytes",
)


def is_noise(line: str) -> bool:
    """Check if a stderr line is a known noisy warning."""
    return any(p in line for p in _NOISE_PATTERNS)


# ── Stderr reader ────────────────────────────────────────────────────────────

def _iter_stderr_lines(proc: subprocess.Popen, poll_interval: float = 1.0):
    """Yield complete lines from process stderr, handling \\r (tqdm) correctly.

    tqdm uses \\r to overwrite the current line. We accumulate characters
    and yield the line content whenever we see \\n or \\r.

    Uses select() to avoid blocking forever — polls every poll_interval seconds
    so the caller can check if the process is still alive.
    """
    fd = proc.stderr.fileno()
    buf = ""

    while True:
        # Use select to avoid blocking forever on os.read
        ready, _, _ = select.select([fd], [], [], poll_interval)
        if not ready:
            # No data available — check if process exited
            if proc.poll() is not None:
                # Process exited, drain remaining data
                try:
                    chunk = os.read(fd, 65536)
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        for ch in text:
                            if ch == "\n":
                                yield buf
                                buf = ""
                            elif ch == "\r":
                                if buf:
                                    yield buf
                                buf = ""
                            else:
                                buf += ch
                except OSError:
                    pass
                break
            continue

        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break

        if not chunk:
            break

        text = chunk.decode("utf-8", errors="replace")

        for ch in text:
            if ch == "\n":
                yield buf
                buf = ""
            elif ch == "\r":
                if buf:
                    yield buf
                buf = ""
            else:
                buf += ch

    # Flush remaining buffer
    if buf:
        yield buf


# ── Run worker ────────────────────────────────────────────────────────────────

@dataclass
class WorkerResult:
    """Result of a worker subprocess execution."""
    returncode: int
    stdout: str                              # collected stdout (JSON result)
    events: list[ProgressEvent] = field(default_factory=list)
    stderr_tail: str = ""                    # last ~500 chars of stderr for error messages


def run_worker(
    cmd: list[str],
    on_event: Callable[[ProgressEvent], None] | None = None,
    duration_s: float = 0.0,
    timeout: int | None = None,
) -> WorkerResult:
    """Run a worker subprocess with real-time stderr streaming.

    Waits until the process finishes — no artificial timeout. Workers that
    run for hours (music generation, enhancement) are expected.

    Args:
        cmd: Command to execute.
        on_event: Callback for each parsed event. If None, events are still collected.
        duration_s: Total expected duration (for ffmpeg progress calculation).
        timeout: Optional safety timeout in seconds. None = wait forever.

    Returns:
        WorkerResult with returncode, stdout, events list, and stderr tail.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # binary mode for stderr reading
    )

    events: list[ProgressEvent] = []
    stderr_lines: list[str] = []
    stdout_chunks: list[bytes] = []

    # Read stdout in a thread to avoid deadlock
    import threading

    def _read_stdout():
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            stdout_chunks.append(chunk)

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stdout_thread.start()

    # Stream stderr — runs until process exits and stderr is drained
    try:
        for line in _iter_stderr_lines(proc):
            stderr_lines.append(line)

            # Skip noise but still collect it in stderr_lines
            if is_noise(line):
                continue

            event = parse_stderr_line(line, duration_s=duration_s)
            events.append(event)

            if on_event is not None:
                on_event(event)

    except Exception:
        pass

    # Wait for process to finish (should already be done after stderr EOF)
    proc.wait()
    stdout_thread.join(timeout=10)

    stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")

    # Build stderr tail for error reporting
    tail_lines = stderr_lines[-20:] if stderr_lines else []
    stderr_tail = "\n".join(tail_lines)[-500:]

    return WorkerResult(
        returncode=proc.returncode or 0,
        stdout=stdout_text,
        events=events,
        stderr_tail=stderr_tail,
    )


# ── CLI display ───────────────────────────────────────────────────────────────

_last_was_progress = False
_CLEAR_LINE = "\033[2K"  # ANSI: erase entire line


def print_event_tui(event: ProgressEvent):
    """Print a progress event to stderr in TUI format (human-readable).

    While a progress bar is active, log lines are suppressed to keep
    the bar overwriting on a single line.  Stage/error/warning always
    break through (they mark a new phase).
    """
    global _last_was_progress

    if event.type == "progress" and event.percent is not None:
        # Counter lines like [1/10] — show step + bar on two lines, update in-place
        if _COUNTER_RE.search(event.message or ""):
            if _last_was_progress:
                # Move up one line (bar) to overwrite previous step+bar
                print(f"\033[A{_CLEAR_LINE}\r{_CLEAR_LINE}", end="", file=sys.stderr)
            # Step info line
            print(f"\r{_CLEAR_LINE}  {event.message}", file=sys.stderr, flush=True)
        bar_width = 30
        filled = int(bar_width * event.percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r{_CLEAR_LINE}  {bar} {event.percent:5.1f}%",
              end="", file=sys.stderr, flush=True)
        _last_was_progress = True

    elif event.type == "stage":
        if _last_was_progress:
            print("", file=sys.stderr)
        print(f"\n  {event.message}", file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "error":
        if _last_was_progress:
            print("", file=sys.stderr)
        print(f"\n  ERROR: {event.message}", file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "warning":
        # suppress warnings while progress bar is active
        if not _last_was_progress:
            print(f"  {event.message}", file=sys.stderr, flush=True)

    else:
        # log — suppress while progress bar is active
        if not _last_was_progress and event.message:
            print(f"  {event.message}", file=sys.stderr, flush=True)


def print_event_json(event: ProgressEvent):
    """Print a progress event as JSON line to stderr (machine-readable)."""
    print(event.to_json(), file=sys.stderr, flush=True)


# Default handler alias (backwards compat)
print_event = print_event_tui


def finish_progress():
    """Print newline after progress bar output to clean up terminal."""
    print("", file=sys.stderr)
