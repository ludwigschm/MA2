"""Placeholder tests around already-recording recovery flows."""

from pathlib import Path

import pytest

from tabletop.pupil_bridge import PupilBridge


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PupilBridge:
    monkeypatch.setenv("LOW_LATENCY_DISABLED", "1")
    monkeypatch.setattr("tabletop.pupil_bridge.requests", None, raising=False)
    config_path = tmp_path / "devices.txt"
    config_path.write_text("VP1_IP=127.0.0.1\nVP1_PORT=8080\n", encoding="utf-8")
    instance = PupilBridge(device_mapping={}, config_path=config_path)
    yield instance
    instance.close()


def test_warn_device_id_once_logs_single_entry(bridge: PupilBridge, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        bridge._warn_device_id_once("demo", "warn %s", "value")  # type: ignore[attr-defined]
        bridge._warn_device_id_once("demo", "warn %s", "value")  # type: ignore[attr-defined]
    warnings = [record for record in caplog.records if "warn value" in record.getMessage()]
    assert len(warnings) == 1
