"""HTTP client for sending annotations to a Pupil Labs Neon headset."""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import requests
from requests import RequestException

from .config import NeonConfig

log = logging.getLogger(__name__)

_API_ROOT = "/api"
_RECORDING_START = f"{_API_ROOT}/recordings/start"
_RECORDING_STOP = f"{_API_ROOT}/recordings/stop"
_ANNOTATIONS = f"{_API_ROOT}/annotations"


class NeonClient:
    """Small helper to talk to the Neon HTTP API without hard dependency."""

    def __init__(
        self,
        *,
        config: NeonConfig,
        timeout: float = 0.5,
    ) -> None:
        self._config = config
        self._timeout = timeout
        self._base_url = config.base_url.rstrip("/")

    @classmethod
    def for_vp(
        cls,
        player: str,
        *,
        timeout: float = 0.5,
        path: Optional[Path] = None,
    ) -> Optional["NeonClient"]:
        config = NeonConfig.for_player(player, path=path)
        if not config.enabled or not config.base_url:
            log.info("Neon disabled for %s (missing configuration)", player)
            return None
        return cls(config=config, timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    def start_recording(self, *, session: Any, block: Any) -> None:
        payload = {
            "session": session,
            "block": block,
        }
        if self._config.device_id:
            payload["device_id"] = self._config.device_id
        self._post(_RECORDING_START, json=payload)

    def stop_recording(self) -> None:
        payload: dict[str, Any] = {}
        if self._config.device_id:
            payload["device_id"] = self._config.device_id
        self._post(_RECORDING_STOP, json=payload)

    def annotate(self, label: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        data: dict[str, Any] = {
            "label": label,
            "ts": _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        }
        if payload is not None:
            data["data"] = dict(payload)
        self._post(_ANNOTATIONS, json=data)

    # ------------------------------------------------------------------
    # Internal helpers
    def _post(self, path: str, *, json: Optional[Mapping[str, Any]] = None) -> None:
        if not self._base_url:
            return
        url = f"{self._base_url}{path}"
        try:
            response = requests.post(url, json=json, timeout=self._timeout)
            response.raise_for_status()
        except RequestException as exc:
            log.warning("Neon request to %s failed: %s", path, exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("Unexpected Neon error for %s: %s", path, exc)
