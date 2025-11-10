"""Adapter for game engine event logging."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from tabletop.engine import EventLogger, Phase as EnginePhase
from tabletop.logging import events_bridge

__all__ = ["Events", "EnginePhase"]


_log = logging.getLogger(__name__)
_PUPYLABS_CLIENT_INITIALISED = False
_PUPYLABS_INCOMPLETE_WARNED = False


def _coerce_float(value: Optional[str], default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        _log.warning("Invalid Pupylabs timeout value: %r", value)
        return default


def _coerce_int(value: Optional[str], default: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        _log.warning("Invalid Pupylabs retry value: %r", value)
        return default
    return max(0, parsed)


def _maybe_init_pupylabs() -> None:
    global _PUPYLABS_CLIENT_INITIALISED, _PUPYLABS_INCOMPLETE_WARNED

    if _PUPYLABS_CLIENT_INITIALISED:
        return

    base_url = os.environ.get("PUPYLABS_CLOUD_BASE_URL") or os.environ.get(
        "PUPYLABS_BASE_URL"
    )
    api_key = os.environ.get("PUPYLABS_CLOUD_API_KEY") or os.environ.get(
        "PUPYLABS_API_KEY"
    )

    if not base_url or not api_key:
        if (base_url or api_key) and not _PUPYLABS_INCOMPLETE_WARNED:
            _log.warning("Incomplete Pupylabs configuration; cloud logging disabled")
            _PUPYLABS_INCOMPLETE_WARNED = True
        return

    timeout = _coerce_float(
        os.environ.get("PUPYLABS_TIMEOUT_S")
        or os.environ.get("PUPYLABS_CLOUD_TIMEOUT_S"),
        2.0,
    )
    retries = _coerce_int(
        os.environ.get("PUPYLABS_MAX_RETRIES")
        or os.environ.get("PUPYLABS_CLOUD_MAX_RETRIES"),
        3,
    )

    events_bridge.init_client(base_url, api_key, timeout_s=timeout, max_retries=retries)
    if getattr(events_bridge, "_client", None) is None:
        _PUPYLABS_CLIENT_INITIALISED = True
        _log.warning("Pupylabs cloud logging unavailable; requests module missing")
        return
    _PUPYLABS_CLIENT_INITIALISED = True
    _log.info("Pupylabs cloud logging enabled")


class Events:
    """Thin wrapper around :class:`tabletop.engine.EventLogger`."""

    def __init__(self, session_id: str, db_path: str, csv_path: Optional[str] = None):
        _maybe_init_pupylabs()
        self._session_id = session_id
        self._logger = EventLogger(db_path, csv_path)

    def log(
        self,
        round_idx: int,
        phase: EnginePhase,
        actor: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Forward events to the underlying logger while fixing defaults."""

        return self._logger.log(
            self._session_id,
            round_idx,
            phase,
            actor,
            action,
            payload or {},
        )

    def close(self) -> None:
        """Close the underlying logger."""

        self._logger.close()
