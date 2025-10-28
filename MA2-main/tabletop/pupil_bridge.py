"""Integration helpers for communicating with Pupil Labs devices."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency
    from pupil_labs.realtime_api.simple import Device, discover_devices
except Exception:  # pragma: no cover - optional dependency
    Device = None  # type: ignore[assignment]
    discover_devices = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


CONFIG_TEMPLATE = """# Neon Geräte-Konfiguration
# Leerlassen = ignorieren. Nur Gerät 1 ist Pflicht, Gerät 2 optional.

VP1_SERIAL=
VP1_IP=192.168.137.121
VP1_PORT=8080

VP2_SERIAL=
VP2_IP=
VP2_PORT=8080
"""

CONFIG_PATH = Path(__file__).resolve().parent.parent / "neon_devices.txt"


def _ensure_config_file(path: Path) -> None:
    if path.exists():
        return
    try:
        path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    except Exception:  # pragma: no cover - defensive fallback
        log.exception("Konfigurationsdatei %s konnte nicht erstellt werden", path)


def _parse_port(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        log.warning("Ungültiger Portwert in neon_devices.txt: %r", value)
        return None


@dataclass
class NeonDeviceConfig:
    player: str
    serial: str = ""
    ip: str = ""
    port: Optional[int] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.ip or self.serial)

    @property
    def address(self) -> Optional[str]:
        if not self.ip:
            return None
        if self.port:
            return f"{self.ip}:{self.port}"
        return self.ip

    def summary(self) -> str:
        if not self.is_configured:
            return "deaktiviert"
        port_display = str(self.port) if self.port is not None else "-"
        ip_display = self.ip or "-"
        serial_display = self.serial or "-"
        return f"ip={ip_display}, port={port_display}, serial={serial_display}"


def _load_device_config(path: Path) -> Dict[str, NeonDeviceConfig]:
    configs: Dict[str, NeonDeviceConfig] = {
        "VP1": NeonDeviceConfig("VP1"),
        "VP2": NeonDeviceConfig("VP2"),
    }
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return configs
    except Exception:  # pragma: no cover - defensive fallback
        log.exception("Konfiguration %s konnte nicht gelesen werden", path)
        return configs

    parsed: Dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed[key.strip().upper()] = value.strip()

    vp1 = configs["VP1"]
    vp1.serial = parsed.get("VP1_SERIAL", vp1.serial)
    vp1.ip = parsed.get("VP1_IP", vp1.ip)
    vp1.port = _parse_port(parsed.get("VP1_PORT", "")) or vp1.port

    vp2 = configs["VP2"]
    vp2.serial = parsed.get("VP2_SERIAL", vp2.serial)
    vp2.ip = parsed.get("VP2_IP", vp2.ip)
    vp2.port = _parse_port(parsed.get("VP2_PORT", "")) or vp2.port

    log.info(
        "Konfig geladen: VP1(%s), VP2(%s)",
        vp1.summary(),
        vp2.summary(),
    )

    if not vp2.is_configured:
        log.info("VP2 deaktiviert (keine IP/Serial)")

    return configs


_ensure_config_file(CONFIG_PATH)


class PupilBridge:
    """Facade around the Pupil Labs realtime API with graceful fallbacks."""

    DEFAULT_MAPPING: Dict[str, str] = {"146118": "VP1"}
    _PLAYER_INDICES: Dict[str, int] = {"VP1": 1, "VP2": 2}

    def __init__(
        self,
        serial_to_player: Optional[Dict[str, str]] = None,
        connect_timeout: float = 10.0,
        *,
        config_path: Optional[Path] = None,
    ) -> None:
        config_file = config_path or CONFIG_PATH
        _ensure_config_file(config_file)
        self._device_config = _load_device_config(config_file)
        self._serial_to_player: Dict[str, str] = (
            dict(serial_to_player) if serial_to_player is not None else dict(self.DEFAULT_MAPPING)
        )
        self._connect_timeout = float(connect_timeout)
        self._device_by_player: Dict[str, Any] = {"VP1": None, "VP2": None}
        self._active_recordings: set[str] = set()
        self._recording_metadata: Dict[str, Dict[str, Any]] = {}
        self._auto_session: Optional[int] = None
        self._auto_block: Optional[int] = None
        self._auto_players: set[str] = set()

    # ---------------------------------------------------------------------
    # Lifecycle management
    def connect(self) -> bool:
        """Discover or configure devices and map them to configured players."""

        configured_players = {
            player for player, cfg in self._device_config.items() if cfg.is_configured
        }
        if configured_players:
            return self._connect_from_config(configured_players)
        return self._connect_via_discovery()

    def _connect_from_config(self, configured_players: Iterable[str]) -> bool:
        if Device is None:
            raise RuntimeError(
                "Pupil Labs realtime API not available – direkte Verbindung nicht möglich."
            )

        success = True
        for player in ("VP1", "VP2"):
            cfg = self._device_config.get(player)
            if not cfg or not cfg.is_configured:
                continue
            try:
                device = self._connect_device_with_retries(player, cfg)
            except Exception as exc:  # pragma: no cover - hardware dependent
                if player == "VP1":
                    raise RuntimeError(f"VP1 konnte nicht verbunden werden: {exc}") from exc
                log.warning("Verbindung zu VP2 fehlgeschlagen: %s", exc)
                success = False
                continue
            self._device_by_player[player] = device
            log.info(
                "Verbunden mit %s (ip=%s, serial=%s)",
                player,
                cfg.ip or "-",
                cfg.serial or "-",
            )
        if "VP1" in configured_players and self._device_by_player.get("VP1") is None:
            raise RuntimeError("VP1 ist konfiguriert, konnte aber nicht verbunden werden.")
        return success and (self._device_by_player.get("VP1") is not None)

    def _connect_device_with_retries(
        self, player: str, cfg: NeonDeviceConfig
    ) -> Any:  # pragma: no cover - hardware dependent
        delays = [1.0, 2.0, 4.0]
        last_error: Optional[BaseException] = None
        for attempt, delay in enumerate(delays, start=1):
            try:
                device = self._connect_device_once(cfg)
                self._assert_device_ready(device, cfg)
                return device
            except Exception as exc:
                last_error = exc
                log.warning(
                    "Verbindungsversuch %s/3 für %s fehlgeschlagen: %s",
                    attempt,
                    player,
                    exc,
                )
                if attempt < len(delays):
                    time.sleep(delay)
        raise last_error if last_error else RuntimeError("Unbekannter Verbindungsfehler")

    def _connect_device_once(self, cfg: NeonDeviceConfig) -> Any:
        assert Device is not None  # guarded by caller
        device: Any

        # Try factory helpers first
        if cfg.address:
            factory = getattr(Device, "from_address", None)
            if callable(factory):
                try:
                    return factory(cfg.address)
                except Exception:
                    log.debug("Device.from_address(%s) fehlgeschlagen", cfg.address, exc_info=True)

        if cfg.serial:
            from_serial = getattr(Device, "from_serial", None)
            if callable(from_serial):
                try:
                    return from_serial(cfg.serial)
                except Exception:
                    log.debug("Device.from_serial(%s) fehlgeschlagen", cfg.serial, exc_info=True)

        # fall back to direct instantiation with different signatures
        attempts: list[Dict[str, Any]] = []
        if cfg.ip:
            if cfg.address:
                attempts.append({"address": cfg.address})
            attempts.append({"host": cfg.ip, "port": cfg.port})
            attempts.append({"ip": cfg.ip, "port": cfg.port})
        if cfg.serial:
            attempts.append({"serial": cfg.serial})
            attempts.append({"serial_number": cfg.serial})

        device = None
        for entry in attempts:
            cleaned = {k: v for k, v in entry.items() if v is not None}
            if not cleaned:
                continue
            try:
                device = Device(**cleaned)
                break
            except TypeError:
                continue
        if device is None:
            kwargs: Dict[str, Any] = {}
            if cfg.address:
                kwargs["address"] = cfg.address
            if cfg.serial:
                kwargs["serial_number"] = cfg.serial
            device = Device(**{k: v for k, v in kwargs.items() if v})

        connect_fn = getattr(device, "connect", None)
        if callable(connect_fn):
            try:
                connect_fn()
            except TypeError:
                connect_fn(device)
        return device

    def _assert_device_ready(self, device: Any, cfg: NeonDeviceConfig) -> None:
        status_checked = False
        for attr in ("api_status", "status", "get_status"):
            status_fn = getattr(device, attr, None)
            if callable(status_fn):
                try:
                    status = status_fn()
                    if status is not None:
                        status_checked = True
                        break
                except Exception:
                    log.debug("Statusabfrage über %s fehlgeschlagen", attr, exc_info=True)
        if not status_checked and requests is not None and cfg.address:
            url = f"http://{cfg.address}/api/status"
            try:
                resp = requests.get(url, timeout=self._connect_timeout)
                resp.raise_for_status()
                status_checked = True
            except Exception:
                log.debug("HTTP-Statusabfrage %s fehlgeschlagen", url, exc_info=True)
        if not status_checked:
            log.warning("/api/status konnte nicht bestätigt werden (ip=%s)", cfg.ip or "-")

    def _connect_via_discovery(self) -> bool:
        if discover_devices is None:
            log.warning(
                "Pupil Labs realtime API not available. Running without device integration."
            )
            return False

        try:
            try:
                found_devices = discover_devices(timeout_seconds=self._connect_timeout)
            except TypeError:
                try:
                    found_devices = discover_devices(timeout=self._connect_timeout)
                except TypeError:
                    found_devices = discover_devices(self._connect_timeout)
        except Exception as exc:  # pragma: no cover - network/hardware dependent
            log.exception("Failed to discover Pupil devices: %s", exc)
            return False

        if not found_devices:
            log.warning("No Pupil devices discovered within %.1fs", self._connect_timeout)
            return False

        for device in found_devices:
            serial = getattr(device, "serial_number", None) or getattr(device, "serial", None)
            if not serial:
                log.debug("Skipping device without serial: %r", device)
                continue
            player = self._serial_to_player.get(str(serial))
            if not player:
                log.info("Ignoring unmapped device with serial %s", serial)
                continue
            self._device_by_player[player] = device
            log.info("Mapped Pupil device %s to %s", serial, player)

        missing_players = [player for player, device in self._device_by_player.items() if device is None]
        if missing_players:
            log.warning(
                "No device found for players: %s", ", ".join(sorted(missing_players))
            )
        return self._device_by_player.get("VP1") is not None

    def close(self) -> None:
        """Close all connected devices if necessary."""

        for player, device in list(self._device_by_player.items()):
            if device is None:
                continue
            try:
                close_fn = getattr(device, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as exc:  # pragma: no cover - hardware dependent
                log.exception("Failed to close device for %s: %s", player, exc)
            finally:
                self._device_by_player[player] = None
        self._active_recordings.clear()
        self._recording_metadata.clear()

    # ------------------------------------------------------------------
    # Recording helpers
    def ensure_recordings(
        self,
        *,
        session: Optional[int] = None,
        block: Optional[int] = None,
        players: Optional[Iterable[str]] = None,
    ) -> set[str]:
        if session is not None:
            self._auto_session = session
        if block is not None:
            self._auto_block = block
        if players is not None:
            self._auto_players = {p for p in players if p}

        if self._auto_players:
            target_players = self._auto_players
        else:
            target_players = {p for p, dev in self._device_by_player.items() if dev is not None}

        if self._auto_session is None or self._auto_block is None:
            return set()

        started: set[str] = set()
        for player in target_players:
            self.start_recording(self._auto_session, self._auto_block, player)
            if player in self._active_recordings:
                started.add(player)
        return started

    def start_recording(self, session: int, block: int, player: str) -> None:
        """Start a recording for the given player using the agreed label schema."""

        device = self._device_by_player.get(player)
        if device is None:
            log.info("recording.start übersprungen (%s nicht verbunden)", player)
            return

        if player in self._active_recordings:
            log.debug("Recording already active for %s", player)
            return

        vp_index = self._PLAYER_INDICES.get(player, 0)
        recording_label = f"{session}.{block}.{vp_index}"

        log.info("recording.start gesendet (%s, label=%s)", player, recording_label)
        begin_info = self._send_recording_start(device, recording_label)
        if begin_info is None:
            log.warning("Timeout, retry/abort (%s)", player)
            return

        recording_id = self._extract_recording_id(begin_info)
        log.info("recording.begin (%s, id=%s)", player, recording_id or "?")

        payload = {
            "session": session,
            "block": block,
            "player": player,
            "recording_label": recording_label,
        }
        self.send_event("session.recording_started", player, payload)
        self._active_recordings.add(player)
        self._recording_metadata[player] = payload

    def _send_recording_start(self, device: Any, label: str) -> Optional[Any]:
        try:
            try:
                result = device.recording_start(label=label)
            except TypeError:
                result = device.recording_start(label)
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.exception("Failed to start recording: %s", exc)
            return None

        return self._wait_for_notification(device, "recording.begin") or result

    def _wait_for_notification(
        self, device: Any, event: str, timeout: float = 5.0
    ) -> Optional[Any]:
        waiters = ["wait_for_notification", "wait_for_event", "await_notification"]
        for attr in waiters:
            wait_fn = getattr(device, attr, None)
            if callable(wait_fn):
                try:
                    return wait_fn(event, timeout=timeout)
                except TypeError:
                    return wait_fn(event, timeout)
                except TimeoutError:
                    return None
                except Exception:
                    log.debug("Warten auf %s via %s fehlgeschlagen", event, attr, exc_info=True)
        return None

    def _extract_recording_id(self, info: Any) -> Optional[str]:
        if isinstance(info, dict):
            for key in ("recording_id", "id", "uuid"):
                value = info.get(key)
                if value:
                    return str(value)
        return None

    def stop_recording(self, player: str) -> None:
        """Stop the active recording for the player if possible."""

        device = self._device_by_player.get(player)
        if device is None:
            log.info("recording.stop übersprungen (%s: nicht konfiguriert/verbunden)", player)
            return

        if player not in self._active_recordings:
            log.debug("No active recording to stop for %s", player)
            return

        log.info("recording.stop (%s)", player)

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
            return

        end_info = self._wait_for_notification(device, "recording.end")
        if end_info is not None:
            recording_id = self._extract_recording_id(end_info)
            log.info("recording.end empfangen (%s, id=%s)", player, recording_id or "?")
        else:
            log.info("recording.end nicht bestätigt (%s)", player)

        self._active_recordings.discard(player)
        self._recording_metadata.pop(player, None)

    def connected_players(self) -> set[str]:
        """Return the set of players that currently have a connected device."""

        return {player for player, device in self._device_by_player.items() if device is not None}

    # ------------------------------------------------------------------
    # Event helpers
    def send_event(
        self,
        name: str,
        player: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an event to the player's device, encoding payload as JSON suffix."""

        device = self._device_by_player.get(player)
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

        device = self._device_by_player.get(player)
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

        return self._device_by_player.get(player) is not None

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
