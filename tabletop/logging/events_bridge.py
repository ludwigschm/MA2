from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

from core.config import CLOUD_API_SETTINGS, CLOUD_BASE_URL
from tabletop.logging.async_bridge import enqueue
from tabletop.logging.pupylabs_cloud import PupylabsCloudLogger
from tabletop.utils.http_client import HttpClient

_log = logging.getLogger(__name__)

_client: Optional[PupylabsCloudLogger] = None


def init_client(
    base_url: str,
    api_key: str,
    timeout_s: float = 2.0,
    max_retries: int = 3,
) -> None:
    """Initialise the shared Pupylabs cloud client."""

    global _client
    if requests is None:
        _log.warning(
            "Requests not available; cannot initialise Pupylabs cloud client"
        )
        _client = None
        return

    target_base = base_url or CLOUD_BASE_URL or ""
    if not target_base:
        _log.info("No cloud base URL configured; cloud ingest disabled")
        _client = None
        return

    settings = CLOUD_API_SETTINGS
    if timeout_s:
        settings = replace(
            settings,
            timeout_s=timeout_s,
            retry_max=max_retries,
        )
    try:
        http_client = HttpClient(
            target_base,
            settings=settings,
            session_factory=requests.Session,
            name="cloud",
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("Unable to create cloud HTTP client: %s", exc)
        _client = None
        return

    _client = PupylabsCloudLogger(http_client, api_key)


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
