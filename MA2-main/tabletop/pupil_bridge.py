"""Integration helpers for communicating with Pupil Labs devices."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from pupil_labs.realtime_api.simple import discover_devices
except Exception:  # pragma: no cover - optional dependency
    discover_devices = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class PupilBridge:
    """Facade around the Pupil Labs realtime API with graceful fallbacks."""

    DEFAULT_MAPPING: Dict[str, str] = {"146118": "VP1"}
    _PLAYER_INDICES: Dict[str, int] = {"VP1": 1, "VP2": 2}

    def __init__(
        self,
        serial_to_player: Optional[Dict[str, str]] = None,
        connect_timeout: float = 10.0,
    ) -> None:
        self._serial_to_player: Dict[str, str] = (
            dict(serial_to_player) if serial_to_player is not None else dict(self.DEFAULT_MAPPING)
        )
        self._connect_timeout = float(connect_timeout)
        self._devices: Dict[str, Any] = {
            player: None for player in {player for player in self._serial_to_player.values()}
        }
        self._active_recordings: set[str] = set()
        self._recording_metadata: Dict[str, Dict[str, Any]] = {}
        if "VP1" not in self._devices:
            self._devices.setdefault("VP1", None)
        if "VP2" not in self._devices:
            self._devices.setdefault("VP2", None)

    # ---------------------------------------------------------------------
    # Lifecycle management
    def connect(self) -> None:
        """Discover devices and map them to configured players."""

        if discover_devices is None:
            log.warning(
                "Pupil Labs realtime API not available. Running without device integration."
            )
            return

        try:
            found_devices = discover_devices(timeout_seconds=self._connect_timeout)
        except Exception as exc:  # pragma: no cover - network/hardware dependent
            log.exception("Failed to discover Pupil devices: %s", exc)
            return

        if not found_devices:
            log.warning("No Pupil devices discovered within %.1fs", self._connect_timeout)
            return

        for device in found_devices:
            serial = getattr(device, "serial_number", None) or getattr(device, "serial", None)
            if not serial:
                log.debug("Skipping device without serial: %r", device)
                continue
            player = self._serial_to_player.get(str(serial))
            if not player:
                log.info("Ignoring unmapped device with serial %s", serial)
                continue
            self._devices[player] = device
            log.info("Mapped Pupil device %s to %s", serial, player)

        missing_players = [player for player, device in self._devices.items() if device is None]
        if missing_players:
            log.warning(
                "No device found for players: %s", ", ".join(sorted(missing_players))
            )

    def close(self) -> None:
        """Close all connected devices if necessary."""

        for player, device in list(self._devices.items()):
            if device is None:
                continue
            try:
                close_fn = getattr(device, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as exc:  # pragma: no cover - hardware dependent
                log.exception("Failed to close device for %s: %s", player, exc)
            finally:
                self._devices[player] = None
        self._active_recordings.clear()
        self._recording_metadata.clear()

    # ------------------------------------------------------------------
    # Recording helpers
    def start_recording(self, session: int, block: int, player: str) -> None:
        """Start a recording for the given player using the agreed label schema."""

        device = self._devices.get(player)
        if device is None:
            log.warning("Cannot start recording for %s: no device connected", player)
            return

        if player in self._active_recordings:
            log.debug("Recording already active for %s", player)
            return

        vp_index = self._PLAYER_INDICES.get(player, 0)
        recording_label = f"{session}.{block}.{vp_index}"

        try:
            try:
                device.recording_start(label=recording_label)
            except TypeError:
                device.recording_start(recording_label)
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.exception("Failed to start recording for %s: %s", player, exc)
            return

        payload = {
            "session": session,
            "block": block,
            "player": player,
            "recording_label": recording_label,
        }
        self.send_event("session.recording_started", player, payload)
        self._active_recordings.add(player)
        self._recording_metadata[player] = payload

    def stop_recording(self, player: str) -> None:
        """Stop the active recording for the player if possible."""

        device = self._devices.get(player)
        if device is None:
            log.warning("Cannot stop recording for %s: no device connected", player)
            return

        if player not in self._active_recordings:
            log.debug("No active recording to stop for %s", player)
            return

        stop_payload = dict(self._recording_metadata.get(player, {"player": player}))
        stop_payload["player"] = player
        stop_payload["event"] = "stop"
        self.send_event(
            "session.recording_stopped",
            player,
            stop_payload,
        )
        try:
            stop_fn = getattr(device, "recording_stop_and_save", None)
            if callable(stop_fn):
                stop_fn()
            else:
                log.warning("Device for %s lacks recording_stop_and_save", player)
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.exception("Failed to stop recording for %s: %s", player, exc)
        finally:
            self._active_recordings.discard(player)
            self._recording_metadata.pop(player, None)

    def connected_players(self) -> set[str]:
        """Return the set of players that currently have a connected device."""

        return {player for player, device in self._devices.items() if device is not None}

    # ------------------------------------------------------------------
    # Event helpers
    def send_event(
        self,
        name: str,
        player: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an event to the player's device, encoding payload as JSON suffix."""

        device = self._devices.get(player)
        if device is None:
            return

        event_label = name
        if payload:
            try:
                payload_json = json.dumps(payload, separators=(",", ":"), default=str)
            except TypeError:
                safe_payload = self._stringify_payload(payload)
                payload_json = json.dumps(safe_payload, separators=(",", ":"))
            event_label = f"{name}|{payload_json}"

        try:
            device.send_event(event_label)
        except TypeError:
            try:
                device.send_event(name, payload)
            except Exception as exc:  # pragma: no cover - hardware dependent
                log.exception("Failed to send event %s for %s: %s", name, player, exc)
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.exception("Failed to send event %s for %s: %s", name, player, exc)

    # ------------------------------------------------------------------
    def estimate_time_offset(self, player: str) -> Optional[float]:
        """Return the estimated time offset between host and device if available."""

        device = self._devices.get(player)
        if device is None:
            return None
        estimator = getattr(device, "estimate_time_offset", None)
        if not callable(estimator):
            return None
        try:
            return float(estimator())
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.exception("Failed to estimate time offset for %s: %s", player, exc)
            return None

    def is_connected(self, player: str) -> bool:
        """Return whether the given player has an associated device."""

        return self._devices.get(player) is not None

    # ------------------------------------------------------------------
    @staticmethod
    def _stringify_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert non-serialisable payload entries to strings."""

        result: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[key] = value
            elif isinstance(value, dict):
                result[key] = PupilBridge._stringify_payload(value)  # type: ignore[arg-type]
            elif isinstance(value, (list, tuple)):
                result[key] = [PupilBridge._coerce_item(item) for item in value]
            else:
                result[key] = str(value)
        return result

    @staticmethod
    def _coerce_item(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return PupilBridge._stringify_payload(value)  # type: ignore[arg-type]
        if isinstance(value, (list, tuple)):
            return [PupilBridge._coerce_item(item) for item in value]
        return str(value)


__all__ = ["PupilBridge"]
