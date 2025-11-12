from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, Iterable, Optional

from core.config import (
    API_HEALTH_PATHS,
    DEFAULT_OFFLINE_QUEUE,
    CLOUD_BASE_URL as _CLOUD_BASE_URL,
)
from tabletop.utils.http_client import (
    ApiDnsError,
    ApiError,
    ApiNotFound,
    ApiTimeout,
    HttpClient,
)

log = logging.getLogger(__name__)

__all__ = ["PupylabsCloudLogger", "CLOUD_BASE_URL"]

CLOUD_BASE_URL = _CLOUD_BASE_URL


class _OfflineQueue:
    """Very small NDJSON-backed queue used when the cloud is unavailable."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = Lock()
        self._buffer: Deque[Dict[str, Any]] = deque()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:  # pragma: no cover - defensive
                log.debug("Failed to create offline queue parent", exc_info=True)
            self._load()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._buffer.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except Exception:  # pragma: no cover - defensive
            log.debug("Failed to load offline queue", exc_info=True)

    def _persist(self) -> None:
        if self._path is None:
            return
        try:
            data = "\n".join(json.dumps(item) for item in self._buffer)
            self._path.write_text(data + ("\n" if data else ""), encoding="utf-8")
        except Exception:  # pragma: no cover - defensive
            log.debug("Failed to persist offline queue", exc_info=True)

    def append(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(dict(event))
            self._persist()

    def extend_left(self, events: Iterable[Dict[str, Any]]) -> None:
        with self._lock:
            for event in reversed(list(events)):
                self._buffer.appendleft(dict(event))
            self._persist()

    def pop_left(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._buffer:
                return None
            event = self._buffer.popleft()
            self._persist()
            return event

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)


class PupylabsCloudLogger:
    """Robust HTTP client for forwarding events to the Pupylabs cloud."""

    def __init__(
        self,
        http_client: HttpClient,
        api_key: str,
        *,
        queue_path: Path | None = DEFAULT_OFFLINE_QUEUE,
    ) -> None:
        self._client = http_client
        self._api_key = api_key
        self._queue = _OfflineQueue(queue_path)
        self._health_path: Optional[str] = None
        self._disabled = False

    @property
    def base_url(self) -> str:
        return self._client.base_url

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def ensure_health(self) -> bool:
        if self._disabled:
            return False
        if self._health_path:
            return True
        try:
            path = self._client.health_check(API_HEALTH_PATHS, headers=self._headers())
        except ApiNotFound:
            path = None
        except ApiDnsError as exc:
            self._handle_dns_failure(exc)
            return False
        except ApiError as exc:
            log.debug("Cloud health check failed", exc_info=True)
            return False
        if path:
            self._health_path = path
            return True
        return False

    def _handle_dns_failure(self, exc: ApiDnsError) -> None:
        log.warning("Cloud DNS failure; enabling offline queue")
        if self._client.base_url.endswith(".example"):
            log.error("Disabling cloud ingest due to invalid host %s", self._client.base_url)
            self._disabled = True

    def send(self, event: Dict[str, Any]) -> None:
        """Attempt to deliver *event*, queueing it on failure."""

        self._queue.append(event)
        self.flush()

    def flush(self) -> None:
        if self._disabled:
            return
        if not self.ensure_health():
            return
        pending: list[Dict[str, Any]] = []
        while True:
            item = self._queue.pop_left()
            if item is None:
                break
            pending.append(item)
        if not pending:
            return
        headers = self._headers()
        for idx, event in enumerate(pending):
            try:
                self._client.post(
                    "/v1/events/ingest",
                    json=event,
                    headers=headers,
                    idempotent=True,
                    allow_statuses={200, 201, 202, 204},
                )
            except ApiDnsError as exc:
                self._handle_dns_failure(exc)
                self._queue.extend_left(pending[idx:])
                return
            except ApiTimeout:
                self._queue.extend_left(pending[idx:])
                return
            except ApiError as exc:
                log.warning("Cloud ingest failed: %s", exc)
                self._queue.extend_left(pending[idx:])
                return

