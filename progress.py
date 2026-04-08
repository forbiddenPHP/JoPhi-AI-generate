"""
progress — Unified progress/output streaming for worker subprocesses.

Replaces subprocess.run(capture_output=True) with streaming Popen that
reads stderr in real-time, parses progress events, and collects stdout
for JSON results.

Event types:
    stage    — Phase label ("Loading model ...", "Generating ...")
    progress — Progress bar / counter with percent
    info     — Relevant status info ("Device: mps", "Saved: output.wav")
    warning  — Warnings (library or worker)
    error    — Errors and tracebacks
    noise    — Library chatter (tokenizer info, torch compat notices)
    log      — Everything else

Usage:
    from progress import run_worker, print_event_tui
    result = run_worker(cmd, on_event=print_event_tui)
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
    type: str               # "progress" | "stage" | "info" | "warning" | "error" | "noise" | "log"
                            # | "env_update" | "env_update_done"
                            # | "inference_gotcha" | "inference_mode" | "inference_token" | "inference_result"
    message: str            # raw line or parsed message
    percent: float | None = None
    current: float | None = None    # counter: current step (or bytes in display unit)
    total: float | None = None      # counter: total steps (or bytes in display unit)
    stage: str | None = None
    chunk: int | None = None      # chunk index (1-based), set when progress resets
    counters: list | None = None  # parsed desc counters: [{"label":"Step","current":1,"total":3}, ...]
    data: dict | None = None      # structured payload for @inference: events
    ts: float = 0.0

    def __post_init__(self):
        if self.ts == 0.0:
            self.ts = time.time()

    def to_json(self) -> str:
        # Inference events: merge data fields up, drop raw message + redundant event key
        if self.data is not None:
            d = {"type": self.type, "ts": self.ts}
            for k, v in self.data.items():
                if k != "event":  # type already carries this
                    d[k] = v
            return json.dumps(d, ensure_ascii=False)
        d = {"type": self.type, "message": self.message, "ts": self.ts}
        if self.percent is not None:
            d["percent"] = round(self.percent, 1)
        if self.current is not None:
            d["current"] = self.current
        if self.total is not None:
            d["total"] = self.total
        if self.stage is not None:
            d["stage"] = self.stage
        if self.chunk is not None:
            d["chunk"] = self.chunk
        if self.counters:
            d["counters"] = self.counters
        return json.dumps(d, ensure_ascii=False)


# ── Parsers ───────────────────────────────────────────────────────────────────

# Regex for parsing desc counters: "Denoising  3/20 Chunk 1/3" → label + counter pairs
_DESC_COUNTER_RE = re.compile(r"(\d+)/(\d+)")
# Match "Label N" (word followed by a standalone number, not part of x/y)
_DESC_SINGLE_RE = re.compile(r"([A-Za-z]\w*)\s+(\d+)(?!\s*/)")


def _parse_desc_counters(desc: str) -> tuple[str, list[dict] | None]:
    """Parse a tqdm description into a clean label and counter list.

    "Denoising  3/20 Chunk 1/3"     → ("Denoising", [{current:3,total:20}, {label:"Chunk",current:1,total:3}])
    "Denoising  2/4 Block 7 Chunk 2/3" → ("Denoising", [{current:2,total:4}, {label:"Block",current:7}, {label:"Chunk",current:2,total:3}])
    "Denoising" → ("Denoising", None)
    """
    if not desc:
        return desc, None

    # Collect all counter tokens with their positions
    tokens = []

    # x/y counters
    for m in _DESC_COUNTER_RE.finditer(desc):
        between = desc[:m.start()].split()
        label = ""
        # Find the word directly before this counter
        text_before = desc[:m.start()].rstrip()
        if text_before:
            last_word = text_before.split()[-1]
            # Don't use the stage label as a counter label for the first match
            if not tokens:
                label = ""
            else:
                label = last_word
        tokens.append({
            "pos": m.start(),
            "label": label,
            "current": int(m.group(1)),
            "total": int(m.group(2)),
        })

    # Single-number counters (e.g. "Block 7") — only if not already part of x/y
    for m in _DESC_SINGLE_RE.finditer(desc):
        num_pos = m.start(2)
        # Skip if this number is part of an x/y counter
        already_covered = any(t["pos"] <= num_pos < t["pos"] + 10 for t in tokens)
        if not already_covered:
            tokens.append({
                "pos": m.start(),
                "label": m.group(1),
                "current": int(m.group(2)),
            })

    if not tokens:
        return desc, None

    # Sort by position in the string
    tokens.sort(key=lambda t: t["pos"])

    # Build counters list (without pos)
    counters = [{k: v for k, v in t.items() if k != "pos"} for t in tokens]

    # Stage label: text before the first token
    first_pos = tokens[0]["pos"]
    stage_label = desc[:first_pos].strip()
    if not stage_label:
        stage_label = desc

    return stage_label, counters


# Regex for tqdm-style progress: "  5%|█         | 5/100 [00:02<00:38, 2.50it/s]"
_TQDM_RE = re.compile(
    r"(\d+)%\|"           # percent
    r"[^|]*\|\s*"         # bar
    r"([\d.]+\s*[kMGTBi]*)"   # current (with optional unit like M, G, GiB)
    r"/([\d.]+\s*[kMGTBi]*)"  # total
)

# Simpler tqdm: "100%|██████████| 5/5"
_TQDM_SIMPLE_RE = re.compile(r"(\d+)%\|")

# ffmpeg time progress: "time=00:01:23.45"
_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")

# Worker counter format: "[1/5] enhance: file.wav"
_COUNTER_RE = re.compile(r"\[(\d+)/(\d+)\]")


# ── Noise patterns ───────────────────────────────────────────────────────────

_NOISE_PATTERNS = (
    # Conda/torch environment noise
    "FutureWarning",
    "UserWarning",
    "pynvml",
    "import pynvml",
    "ds_accelerator",
    "bitsandbytes",
    # Torch/MPS compat
    "Skipping import of cpp extensions",
    "NOTE: Redirects are currently not supported",
    # HuggingFace cache checks
    "Fetching",
    # AI-TTS / mlx-audio internals
    "Initialized encoder codebooks",
    "Loaded speech tokenizer from",
    "You are using a model of type",
    "The tokenizer you are loading",
    "fix_mistral_regex",
    # ACE-Step logger noise
    "| INFO     |",
    "| DEBUG    |",
    # torch autocast
    "MPS Autocast only supports",
    "with torch.autocast(",
)

_NOISE_REGEXES = (
    # W0315 20:35:03... torch distributed warnings
    re.compile(r"^W\d{4}\s"),
)


def is_noise(line: str) -> bool:
    """Check if a line is known library noise."""
    if any(p in line for p in _NOISE_PATTERNS):
        return True
    if any(r.search(line) for r in _NOISE_REGEXES):
        return True
    return False


# ── Warning patterns ─────────────────────────────────────────────────────────

_WARNING_PATTERNS = (
    "WARNING:",
    "WARNING ",
    "[WARN]",
    "Calibration failed",
    "Could not determine target F0",
)


def _is_warning(line: str) -> bool:
    """Check if a line is a warning (regardless of position)."""
    return any(p in line for p in _WARNING_PATTERNS)


# ── Info patterns ─────────────────────────────────────────────────────────────

_INFO_PATTERNS = (
    "Device:",
    "Saved:",
    "Generated in",
    "Transcribed in",
    "Done —",
    "Done -",
    "Seed:",
    "Dtype:",
)

_INFO_PREFIX_RE = re.compile(r"^\s*(Voice|Language|Output|Model|Caption|Duration|Tags|Task|Input|Format):\s")


def _is_info(line: str) -> bool:
    """Check if a line is an info/status line."""
    stripped = line.strip()
    if any(p in stripped for p in _INFO_PATTERNS):
        return True
    if _INFO_PREFIX_RE.match(stripped):
        return True
    if stripped.startswith("✓ ") or "✓ " in stripped:
        return True
    return False


_BYTE_UNITS = {"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}


def _parse_byte_value(s: str) -> float | None:
    """Parse '469M' or '46.1G' or '0.00' into bytes. None if no unit found."""
    m = re.match(r"([\d.]+)\s*([kKmMgGtT])?[iI]?[bB]?$", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit:
        return val * _BYTE_UNITS.get(unit.lower(), 1)
    return None  # plain number, no unit


def _parse_tqdm_counter(raw_cur: str, raw_tot: str) -> tuple[int | None, int | None]:
    """Parse tqdm current/total into display values.

    Plain integers (5/8) → pass through.
    Byte values (469M/46.1G) → convert to same unit, return as int in that unit.
    """
    cur_bytes = _parse_byte_value(raw_cur)
    tot_bytes = _parse_byte_value(raw_tot)
    # If total has a unit, treat unitless current as 0 bytes
    if cur_bytes is None and tot_bytes is not None:
        cur_bytes = float(re.sub(r"[^\d.]", "", raw_cur) or "0")
    if cur_bytes is not None and tot_bytes is not None:
        # Both have units → pick the best display unit from total
        for label, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
            if tot_bytes >= scale:
                return round(cur_bytes / scale, 1), round(tot_bytes / scale, 1)
        return round(cur_bytes, 1), round(tot_bytes, 1)
    # Plain integer counters (5/8, 100/100)
    cur_s = re.sub(r"[^\d.]", "", raw_cur)
    tot_s = re.sub(r"[^\d.]", "", raw_tot)
    cur = int(float(cur_s)) if cur_s else 0
    tot = int(float(tot_s)) if tot_s else 0
    return cur, tot


def parse_stderr_line(line: str, duration_s: float = 0.0) -> ProgressEvent:
    """Parse a single stderr line into a ProgressEvent.

    Classification priority:
    1. tqdm progress bars → progress
    2. [i/n] counter lines → progress
    3. ffmpeg time= lines → progress
    4. Noise (library chatter) → noise
    5. Errors/tracebacks → error
    6. Warnings → warning
    7. Stage labels ("Loading model ...") → stage
    8. Info lines (Device:, Saved:, etc.) → info
    9. Everything else → log
    """
    stripped = line.strip()
    if not stripped:
        return ProgressEvent(type="log", message="")

    # 0. Check for @inference: structured events (from text worker)
    if stripped.startswith("@inference:"):
        try:
            payload = json.loads(stripped[11:])
            return ProgressEvent(
                type=payload.get("event", "log"),
                message=stripped,
                data=payload,
            )
        except json.JSONDecodeError:
            pass  # fall through to normal parsing

    # 1. Check for tqdm-style progress
    m = _TQDM_RE.search(stripped)
    if m:
        pct = float(m.group(1))
        raw_cur, raw_tot = m.group(2).strip(), m.group(3).strip()
        cur, tot = _parse_tqdm_counter(raw_cur, raw_tot)
        # Extract tqdm description (text before "XX%|")
        desc = stripped[:m.start()].strip().rstrip(":").strip() or None
        # Parse structured counters from desc (e.g. "Denoising  3/20 Chunk 1/3")
        stage_label, counters = _parse_desc_counters(desc) if desc else (desc, None)
        return ProgressEvent(type="progress", message=stripped, percent=pct,
                             current=cur, total=tot, stage=stage_label,
                             counters=counters)

    m = _TQDM_SIMPLE_RE.search(stripped)
    if m:
        pct = float(m.group(1))
        desc = stripped[:m.start()].strip().rstrip(":").strip() or None
        stage_label, counters = _parse_desc_counters(desc) if desc else (desc, None)
        return ProgressEvent(type="progress", message=stripped, percent=pct,
                             stage=stage_label, counters=counters)

    # 2. Check for [i/n] counter — e.g. "[7/48] Denoise 2/8 – Pass 1/2"
    m = _COUNTER_RE.search(stripped)
    if m:
        current = int(m.group(1))
        total = int(m.group(2))
        pct = (current / total) * 100 if total > 0 else 0
        # Parse text after [i/n] for additional counters (Denoise x/y, Pass a/b, etc.)
        after = stripped[m.end():].strip().lstrip("–—-").strip()
        stage_label, counters = _parse_desc_counters(after) if after else (None, None)
        # Build counters: the [i/n] block counter first, then any parsed from the text
        all_counters = [{"label": "Block", "current": current, "total": total}]
        if counters:
            all_counters.extend(counters)
        return ProgressEvent(type="progress", message=stripped, percent=pct,
                             current=current, total=total, stage=stage_label,
                             counters=all_counters)

    # 3. Check for ffmpeg time=
    if duration_s > 0:
        m = _FFMPEG_TIME_RE.search(stripped)
        if m:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            pct = min(100.0, (t / duration_s) * 100)
            return ProgressEvent(type="progress", message=stripped, percent=pct)

    # 4. Noise — library chatter
    if is_noise(stripped):
        return ProgressEvent(type="noise", message=stripped)

    # 5. Errors
    if stripped.startswith("ERROR:") or stripped.startswith("ERROR ") or stripped.startswith("Traceback"):
        return ProgressEvent(type="error", message=stripped)

    # 6. Warnings
    if _is_warning(stripped):
        return ProgressEvent(type="warning", message=stripped)

    # 7. Stage labels — only Unicode ellipsis (…), never ASCII "..."
    if stripped.endswith("…"):
        return ProgressEvent(type="stage", message=stripped, stage=stripped.rstrip("… "))

    # 8. Info lines
    if _is_info(stripped):
        return ProgressEvent(type="info", message=stripped)

    # 9. Default: log
    return ProgressEvent(type="log", message=stripped)


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
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # binary mode for stderr reading
        env=env,
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
            # Filter CUDA-only triton noise (irrelevant on macOS/Apple Silicon)
            if "triton" in line.lower():
                continue

            stderr_lines.append(line)

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

    # Filter triton noise from stdout (CUDA-only, irrelevant on macOS)
    stdout_text = "\n".join(
        l for l in stdout_text.splitlines() if "triton" not in l.lower()
    )

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
_current_stage = ""          # last stage label, used as progress bar prefix
_CLEAR_LINE = "\033[2K"      # ANSI: erase entire line

# Visual markers
_MARKER_STAGE   = "●"
_MARKER_INFO    = "·"
_MARKER_WARNING = "⚠"
_MARKER_ERROR   = "✗"
_MARKER_DONE    = "✓"


def _format_bar(percent: float, current: float | None = None,
                total: float | None = None, label: str = "",
                counters: list | None = None) -> str:
    """Build a unified progress bar line: Label  Step 1/3  Chunk 1/2  x/n  ██████  %"""
    bar_width = 30
    filled = int(bar_width * percent / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    parts = []
    if label:
        parts.append(label)
    # Show structured counters from desc (Step 1/3, Chunk 1/2, etc.)
    if counters:
        for c in counters:
            lbl = c.get("label", "")
            cur, tot = c["current"], c["total"]
            parts.append(f"{lbl} {cur}/{tot}" if lbl else f"{cur}/{tot}")
    if current is not None and total is not None:
        if isinstance(current, float) or isinstance(total, float):
            parts.append(f"{current:.1f}/{total:.1f}")
        else:
            parts.append(f"{current}/{total}")
    parts.append(bar)
    parts.append(f"{percent:5.1f}%")

    return "  " + "  ".join(parts)


def print_event_tui(event: ProgressEvent):
    """Print a progress event to stderr in TUI format (human-readable).

    Progress bars use unified format: Label  x/n  ██████  %
    Stage/error/warning always break through the progress bar.
    Noise is suppressed.
    """
    global _last_was_progress, _current_stage

    if event.type == "progress" and event.percent is not None:
        # Counter lines like [1/10] — extract info text above the bar
        msg = event.message or ""
        counter_match = _COUNTER_RE.search(msg)
        label = _current_stage
        if counter_match:
            # Extract the description after [i/n] and use as bar label
            after_counter = msg[counter_match.end():].strip()
            if after_counter:
                label = after_counter
            else:
                label = ""

        # Use tqdm description as label if available
        if event.stage:
            label = event.stage

        bar_line = _format_bar(
            event.percent,
            current=event.current,
            total=event.total,
            label=label,
            counters=event.counters,
        )
        print(f"\r{_CLEAR_LINE}{bar_line}",
              end="", file=sys.stderr, flush=True)
        _last_was_progress = True

    elif event.type == "stage":
        if _last_was_progress:
            print("", file=sys.stderr)
        _current_stage = event.stage or ""
        print(f"\n  {_MARKER_STAGE} {event.message}", file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "error":
        if _last_was_progress:
            print("", file=sys.stderr)
        print(f"  {_MARKER_ERROR} {event.message}", file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "warning":
        if _last_was_progress:
            print("", file=sys.stderr)
            _last_was_progress = False
        print(f"  {_MARKER_WARNING} {event.message}", file=sys.stderr, flush=True)

    elif event.type == "info":
        # Show info lines only when no progress bar is active
        if not _last_was_progress and event.message:
            print(f"  {_MARKER_INFO} {event.message}", file=sys.stderr, flush=True)

    elif event.type == "env_update":
        msg = event.data.get("message", "Updating environment …") if event.data else "Updating environment …"
        if _last_was_progress:
            print("", file=sys.stderr)
        print(f"  {_MARKER_INFO} {msg}", file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "env_update_done":
        msg = event.data.get("message", "Done.") if event.data else "Done."
        print(f"  {_MARKER_INFO} {msg}", file=sys.stderr, flush=True)

    elif event.type == "inference_gotcha":
        if _last_was_progress:
            print("", file=sys.stderr)
        print(f"\n  {_MARKER_STAGE} Inference …", file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "inference_mode":
        mode = event.data.get("mode", "?") if event.data else "?"
        print_event_tui._inference_mode = mode
        print(f"  {_MARKER_INFO} Mode: {mode}", file=sys.stderr, flush=True)

    elif event.type == "inference_token":
        text = event.data.get("text", "") if event.data else ""
        if text:
            print(text, end="", file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "inference_result":
        # After streaming: just newline. After sync: print full text.
        mode = getattr(print_event_tui, "_inference_mode", "sync")
        text = event.data.get("text", "") if event.data else ""
        if mode == "stream":
            print("", file=sys.stderr, flush=True)
        elif text:
            print(text, file=sys.stderr, flush=True)
        _last_was_progress = False

    elif event.type == "noise":
        # Noise is suppressed in TUI
        pass

    else:
        # log — suppress while progress bar is active
        if not _last_was_progress and event.message:
            print(f"    {event.message}", file=sys.stderr, flush=True)


def print_event_json(event: ProgressEvent):
    """Print a progress event as JSON line to stderr (machine-readable)."""
    print(event.to_json(), file=sys.stderr, flush=True)


# Default handler alias (backwards compat)
print_event = print_event_tui


def finish_progress():
    """Print newline after progress bar output to clean up terminal."""
    global _last_was_progress, _current_stage
    if _last_was_progress:
        print("", file=sys.stderr)
    _last_was_progress = False
    _current_stage = ""
