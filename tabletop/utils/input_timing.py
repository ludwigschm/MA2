"""Helpers for debouncing high-frequency UI events."""

from __future__ import annotations

import threading
import time
from typing import Dict

__all__ = ["Debouncer"]


class Debouncer:
    """Return False when events fire faster than the configured interval."""

    def __init__(self, interval_ms: float = 50.0) -> None:
        self._interval = max(0.0, float(interval_ms)) / 1000.0
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return True only if the previous event is outside the interval."""

        now = time.perf_counter()
        with self._lock:
            last = self._last.get(key)
            if last is None or now - last >= self._interval:
                self._last[key] = now
                return True
            return False
