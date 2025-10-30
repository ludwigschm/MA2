"""Helpers to query environment-driven runtime toggles used in hot paths.

This module centralises checks for environment-controlled behaviour so callers
can cheaply determine whether the low-latency pipeline or additional
performance logging should be active. The helpers are kept simple on purpose –
they are imported in hot paths.
"""

from __future__ import annotations

import os


_LOW_LATENCY_ENV = "LOW_LATENCY_OFF"
_PERF_ENV = "TABLETOP_PERF"


def is_low_latency_disabled() -> bool:
    """Return ``True`` when the low-latency pipeline is disabled.

    The environment variable :envvar:`LOW_LATENCY_OFF` set to ``"1"`` turns the
    optimised queues back into their synchronous legacy behaviour. This allows
    quick diagnostics without code changes.
    """

    return os.environ.get(_LOW_LATENCY_ENV, "").strip() == "1"


def is_perf_logging_enabled() -> bool:
    """Return whether verbose performance logging is requested."""

    if is_low_latency_disabled():
        return False
    return os.environ.get(_PERF_ENV, "").strip() == "1"


__all__ = [
    "is_low_latency_disabled",
    "is_perf_logging_enabled",
]

