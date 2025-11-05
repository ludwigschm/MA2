"""Tests for WebSocket keepalive helpers."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

from tabletop.pupil_bridge import (
    NeonDeviceConfig,
    PupilBridge,
    WS_PING_INTERVAL_SEC,
    WS_READ_TIMEOUT_SEC,
)


class _DummyNotifications:
    def __init__(self) -> None:
        self.connect_calls: List[Dict[str, Any]] = []
        self.subscriptions: List[Tuple[str, Callable[..., None]]] = []

    def connect(self, **kwargs: Any) -> None:
        self.connect_calls.append(kwargs)

    def subscribe(self, topic: str, callback: Callable[..., None]) -> None:
        self.subscriptions.append((topic, callback))


class _DummyDevice:
    def __init__(self) -> None:
        self.notifications = _DummyNotifications()


def _build_bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PupilBridge:
    monkeypatch.setenv("LOW_LATENCY_DISABLED", "1")
    monkeypatch.setattr("tabletop.pupil_bridge.requests", None, raising=False)
    config_path = tmp_path / "devices.txt"
    config_path.write_text("VP1_IP=127.0.0.1\nVP1_PORT=8080\n", encoding="utf-8")
    return PupilBridge(device_mapping={}, config_path=config_path)


def test_subscribe_notifications_triggers_keepalive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bridge = _build_bridge(monkeypatch, tmp_path)
    device = _DummyDevice()
    cfg = NeonDeviceConfig(player="VP1", ip="127.0.0.1", port=8080)
    bridge._device_by_player["VP1"] = device  # type: ignore[attr-defined]
    bridge._on_device_connected("VP1", device, cfg, "dev-1")  # type: ignore[attr-defined]

    callback_called = []

    def _callback(*_args: Any, **_kwargs: Any) -> None:
        callback_called.append(True)

    assert bridge.subscribe_notifications("VP1", "demo.topic", _callback)  # type: ignore[attr-defined]
    assert device.notifications.connect_calls
    first_call = device.notifications.connect_calls[0]
    assert first_call.get("ping_interval") == WS_PING_INTERVAL_SEC
    assert first_call.get("read_timeout") == WS_READ_TIMEOUT_SEC
    assert device.notifications.subscriptions
    topic, cb = device.notifications.subscriptions[0]
    assert topic == "demo.topic"
    cb({})
    assert callback_called

    bridge.close()
