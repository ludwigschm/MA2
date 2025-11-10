from __future__ import annotations

import logging
from typing import Optional

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

from tabletop.logging.async_bridge import enqueue
from tabletop.logging.pupylabs_cloud import PupylabsCloudLogger

_log = logging.getLogger(__name__)

_session: Optional["requests.Session"]
if requests is not None:
    _session = requests.Session()
else:  # pragma: no cover - optional dependency
    _session = None
_client: Optional[PupylabsCloudLogger] = None


def init_client(
    base_url: str,
    api_key: str,
    timeout_s: float = 2.0,
    max_retries: int = 3,
) -> None:
    """Initialise the shared Pupylabs cloud client."""

    global _client
    if requests is None or _session is None:
        _log.warning(
            "Requests not available; cannot initialise Pupylabs cloud client"
        )
        _client = None
        return
    _client = PupylabsCloudLogger(_session, base_url, api_key, timeout_s, max_retries)


def push_async(event: dict) -> None:
    """Schedule *event* for asynchronous delivery to Pupylabs."""

    if _client is None:
        _log.debug("Pupylabs client not initialized; dropping event")
        return

    def _dispatch() -> None:
        try:
            _client.send(event)
        except Exception as exc:  # pragma: no cover - defensive logging
            _log.exception("Failed to push event: %r", exc)

    enqueue(_dispatch)
