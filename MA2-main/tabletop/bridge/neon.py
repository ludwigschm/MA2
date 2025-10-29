"""Bridge utilities for communicating with Pupil Labs Neon eye trackers."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

from tabletop.data.config import ROOT

log = logging.getLogger(__name__)

NEON_CONFIG_FILENAME = "neon_devices.txt"


@dataclass(slots=True)
class NeonDevice:
    """Runtime configuration for a single Neon headset."""

    identifier: str
    ip: str
    port: int = 8080

    @property
    def base_url(self) -> str:
        if not self.ip:
            return ""
        port_segment = f":{self.port}" if self.port else ""
        return f"http://{self.ip}{port_segment}"


def _parse_config_lines(lines: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_neon_configuration(path: Optional[Path] = None) -> Dict[str, NeonDevice]:
    """Load all configured Neon headsets from ``neon_devices.txt``.

    The configuration file stores key-value pairs per participant, e.g.::

        VP1_ID=abc123
        VP1_IP=192.168.0.5
        VP1_PORT=8080

    The ``*_ID`` entry is optional.  If it is omitted the player label will be
    used as identifier when communicating with the REST API.
    """

    if path is None:
        path = ROOT / NEON_CONFIG_FILENAME

    try:
        contents = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        log.info("Neon configuration file %s not found", path)
        return {}
    except OSError as exc:  # pragma: no cover - unexpected filesystem error
        log.warning("Failed to read Neon configuration %s: %s", path, exc)
        return {}

    parsed = _parse_config_lines(contents)
    devices: Dict[str, NeonDevice] = {}

    for label in ("VP1", "VP2"):
        ip = parsed.get(f"{label}_IP", "")
        if not ip:
            continue
        identifier = parsed.get(f"{label}_ID", "") or label
        port_value = parsed.get(f"{label}_PORT", "")
        try:
            port = int(port_value) if port_value else 8080
        except ValueError:
            log.warning(
                "Ignoring invalid port %r for %s in %s", port_value, label, path
            )
            port = 8080
        devices[label] = NeonDevice(identifier=identifier, ip=ip, port=port)

    return devices


class NeonEyeTrackerBridge:
    """Small helper that sends events to a Pupil Labs Neon headset."""

    def __init__(
        self,
        device: NeonDevice,
        *,
        player_label: str = "VP1",
        player_index: int = 1,
        timeout: float = 4.0,
    ) -> None:
        self._device = device
        self._player_label = player_label
        self._player_index = player_index
        self._timeout = timeout

        self._session_label: Optional[str] = None
        self._session_number: Optional[int] = None
        self._block_index: Optional[int] = None
        self._recording_requested = False
        self._recording_id: Optional[str] = None
        self._last_error: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        *,
        player: str = "VP1",
        path: Optional[Path] = None,
        timeout: float = 4.0,
    ) -> Optional["NeonEyeTrackerBridge"]:
        devices = load_neon_configuration(path=path)
        device = devices.get(player)
        if device is None or not device.base_url:
            return None

        player_index = 1
        try:
            suffix = "".join(ch for ch in player if ch.isdigit())
            player_index = int(suffix) if suffix else 1
        except ValueError:
            player_index = 1

        return cls(device, player_label=player, player_index=player_index, timeout=timeout)

    # ------------------------------------------------------------------
    # Public helpers
    @property
    def enabled(self) -> bool:
        return bool(self._device and self._device.base_url)

    def update_context(
        self,
        *,
        session_label: Optional[str] = None,
        session_number: Optional[int] = None,
        block: Optional[int] = None,
    ) -> None:
        if session_label:
            self._session_label = str(session_label)
        if session_number is not None:
            self._session_number = session_number
            if not self._session_label:
                self._session_label = str(session_number)
        if block is not None:
            self._block_index = block

    def ensure_recording(
        self,
        *,
        session_label: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        if self._recording_requested:
            return

        label = session_label or self._session_label or time.strftime("session-%Y%m%d-%H%M%S")
        payload: Dict[str, Any] = {"name": label, "tags": [self._player_label]}
        if self._device.identifier:
            payload["device_id"] = self._device.identifier

        if metadata:
            payload["properties"] = self._normalise_payload(metadata)

        response = self._request("POST", "/recordings", payload)
        if isinstance(response, Mapping):
            recording_id = response.get("id")
            if isinstance(recording_id, str):
                self._recording_id = recording_id
        self._recording_requested = True

    def handle_event(self, name: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        if not self.enabled:
            return

        payload = payload or {}

        if name == "action.session_start":
            session_payload = payload.get("payload") if isinstance(payload, Mapping) else None
            if isinstance(session_payload, Mapping):
                session_label = session_payload.get("session_id")
                session_number = session_payload.get("session_number")
                block = session_payload.get("start_block")
                self.update_context(
                    session_label=str(session_label) if session_label else None,
                    session_number=self._safe_int(session_number),
                    block=self._safe_int(block),
                )
                meta_payload = dict(session_payload)
            else:
                meta_payload = dict(payload)
            self.ensure_recording(metadata=meta_payload)
            self._record_annotation("session_start", payload)
            return

        if name.startswith("button."):
            player = payload.get("game_player") if isinstance(payload, Mapping) else None
            if self._safe_int(player) != self._player_index:
                return
            self.ensure_recording()
            self._record_annotation(name, payload)
            return

        if name == "fixation.start":
            self.ensure_recording()
            self._record_annotation("fixation_cross", payload)
            return

        if name == "fixation.tone":
            self.ensure_recording()
            self._record_annotation("fixation_tone", payload)
            return

    # ------------------------------------------------------------------
    # Internal helpers
    def _make_url(self, path: str) -> str:
        base = self._device.base_url.rstrip("/")
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    def _request(self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None) -> Any:
        url = self._make_url(path)
        data: Optional[bytes] = None
        headers: Dict[str, str] = {}
        if payload is not None:
            data = json.dumps(self._normalise_payload(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=data, headers=headers, method=method.upper())

        try:
            with request.urlopen(req, timeout=self._timeout) as response:
                status = getattr(response, "status", response.getcode())
                if status < 200 or status >= 300:
                    raise error.HTTPError(url, status, "Unexpected response", response.headers, None)
                body = response.read()
        except error.HTTPError as exc:
            self._handle_failure(f"Neon request {method} {path} failed", exc)
            return None
        except error.URLError as exc:
            self._handle_failure(f"Neon request {method} {path} unreachable", exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._handle_failure(f"Neon request {method} {path} error", exc)
            return None

        self._last_error = None
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _handle_failure(self, message: str, exc: BaseException) -> None:
        text = f"{message}: {exc}"
        if text != self._last_error:
            log.warning(text)
            self._last_error = text
        else:
            log.debug(text)

    def _record_annotation(self, label: str, payload: Mapping[str, Any]) -> None:
        annotation: Dict[str, Any] = {
            "timestamp": time.time(),
            "duration": 0.0,
            "label": label,
            "properties": self._event_payload(payload),
        }
        if self._recording_id:
            annotation["recording_id"] = self._recording_id
        annotation["source"] = "tabletop"
        self._request("POST", "/annotations", annotation)

    def _event_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        combined = self._normalise_payload(payload)
        if self._session_label and "session_id" not in combined:
            combined["session_id"] = self._session_label
        if self._session_number is not None and "session_number" not in combined:
            combined["session_number"] = self._session_number
        if self._block_index is not None and "block_index" not in combined:
            combined["block_index"] = self._block_index
        combined.setdefault("player_label", self._player_label)
        combined.setdefault("player_index", self._player_index)
        return combined

    def _normalise_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {str(key): self._normalise_value(value) for key, value in payload.items()}

    def _normalise_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(k): self._normalise_value(v) for k, v in value.items()}
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [self._normalise_value(v) for v in value]
        return str(value)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None


__all__ = ["NeonEyeTrackerBridge", "load_neon_configuration", "NeonDevice"]
