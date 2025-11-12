"""Clock utilities for host and monotonic timekeeping."""

from __future__ import annotations

import time


def now_ns() -> int:
    """Return a UNIX epoch timestamp in nanoseconds."""
    return time.time_ns()


def now_mono() -> float:
    """Return the current monotonic time in seconds."""
    return time.monotonic()


def now_mono_ns() -> int:
    """Return the current monotonic time in nanoseconds."""
    return time.perf_counter_ns()
