"""Human console output; machine-readable artifacts remain separate and unchanged."""
from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import time

_TQDM = None


def duration(seconds):
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def interactive(file=None):
    file = sys.stderr if file is None else file
    return os.environ.get("_RUNPOD_PROGRESS_TTY") == "1" or bool(getattr(file, "isatty", lambda: False)())


def message(text):
    if _TQDM is not None:
        _TQDM.write(str(text), file=sys.stdout)
    else:
        print(text, flush=True)


class PlainProgress:
    """Throttled readable lines for redirected/non-TTY output; no tqdm import."""
    def __init__(self, total, initial=0, desc="", unit="it", file=None, disabled=False):
        self.total, self.n, self.initial = total, initial, initial
        self.desc, self.unit = desc, unit
        self.file = sys.stderr if file is None else file
        self.disabled, self.closed, self.postfix = disabled, False, {}
        self.started = self.last = time.monotonic()
        self.emit()

    def emit(self):
        if self.disabled:
            return
        elapsed = time.monotonic() - self.started
        advanced = self.n - self.initial
        rate = advanced / elapsed if elapsed > 0 and advanced else 0
        eta = (self.total - self.n) / rate if rate > 0 else None
        details = " | ".join(f"{key}={value}" for key, value in self.postfix.items())
        line = (f"{self.desc}: {self.n:,}/{self.total:,} {self.unit} "
                f"({100 * self.n / max(1, self.total):.1f}%) | elapsed {duration(elapsed)} "
                f"| ETA {duration(eta)}")
        if rate:
            line += f" | {rate:.2f} {self.unit}/s"
        if details:
            line += " | " + details
        self.file.write(line + "\n")
        self.file.flush()
        self.last = time.monotonic()

    def update(self, count=1):
        self.n += count
        if time.monotonic() - self.last >= 30:
            self.emit()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        self.postfix = {**(ordered_dict or {}), **kwargs}

    def close(self):
        if not self.closed:
            self.emit()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def progress(total, *, initial=0, desc="", unit="it", file=None):
    global _TQDM
    disabled = os.environ.get("TQDM_DISABLE", "").lower() in ("1", "true", "yes")
    if not disabled and interactive(file):
        try:
            from tqdm import tqdm
        except ImportError:
            pass
        else:
            _TQDM = tqdm
            return tqdm(total=total, initial=initial, desc=desc, unit=unit,
                        file=sys.stderr if file is None else file, dynamic_ncols=True,
                        mininterval=1.0, maxinterval=10.0, smoothing=.1,
                        unit_scale=unit == "B", unit_divisor=1024)
    return PlainProgress(total, initial, desc, unit, file, disabled)


def value_text(value, key=""):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and (key.endswith("_seconds") or key == "seconds"):
        return duration(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return f"{len(value)} fields"
    if isinstance(value, list):
        return ", ".join(map(str, value)) if not any(isinstance(x, dict) for x in value) else f"{len(value)} records"
    return str(value)


def table(rows):
    if not rows:
        return
    first = rows[0]
    if "estimated_training_seconds" in first:
        keys = ["cell", "points", "updates", "estimated_training_seconds",
                "estimated_evaluation_seconds", "estimated_total_seconds"]
        headers = ["Cell", "Points", "Updates", "Train", "Eval", "Total"]
    elif "rouge2" in first:
        keys = ["cell", "status", "rouge2", "training_seconds", "evaluation_seconds", "microbatch"]
        headers = ["Cell", "Status", "ROUGE-2", "Train", "Eval", "Microbatch"]
    elif "budget_points" in first:
        keys = ["id", "budget_points", "on_policy_percent", "points", "examples", "updates"]
        headers = ["Cell", "Budget", "Online %", "Spent", "Examples", "Updates"]
    else:
        keys = list(first)[:7]
        headers = [key.replace("_", " ").title() for key in keys]
    content = [[value_text(row.get(key), key) for key in keys] for row in rows]
    widths = [max(len(label), *(len(row[i]) for row in content)) for i, label in enumerate(headers)]
    for row in [headers, ["-" * width for width in widths], *content]:
        message("  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip())


def output(value, title="", json_mode=False):
    if json_mode:
        print(json.dumps(value, indent=2), flush=True)
        return
    if title:
        message(title)
        message("=" * len(title))
    def render(mapping, indent=0):
        for key, item in mapping.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(item, dict):
                if key == "code":
                    message(" " * indent + f"{label}: {len(item)} source files recorded")
                else:
                    message(" " * indent + label + ":")
                    render(item, indent + 2)
            elif isinstance(item, list) and item and isinstance(item[0], dict):
                message(" " * indent + f"{label} ({len(item)}):")
                table(item)
            else:
                message(" " * indent + f"{label}: {value_text(item, key)}")
    render(value)


def run_logged(argv, log, stop, *, cwd=None, env=None):
    """Relay child output live, including tqdm carriage returns; retain a text log."""
    child_env = {**os.environ, **(env or {}),
                 "_RUNPOD_PROGRESS_TTY": "1" if interactive() else "0"}
    with log.open("x", encoding="utf-8") as stream:
        stop.child = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      bufsize=0, start_new_session=os.name != "nt", env=child_env)
        if stop.requested:
            import signal
            stop.child.send_signal(signal.SIGTERM)
        pipe = getattr(stop.child, "stdout", None)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            if pipe is not None:
                while chunk := pipe.read(4096):
                    text = decoder.decode(chunk)
                    sys.stderr.write(text)
                    sys.stderr.flush()
                    stream.write(text.replace("\r\n", "\n").replace("\r", "\n"))
                    stream.flush()
                tail = decoder.decode(b"", final=True)
                if tail:
                    sys.stderr.write(tail)
                    sys.stderr.flush()
                    stream.write(tail.replace("\r\n", "\n").replace("\r", "\n"))
            code = stop.child.wait()
            stop.child = None
            return code
        finally:
            if pipe is not None:
                pipe.close()
