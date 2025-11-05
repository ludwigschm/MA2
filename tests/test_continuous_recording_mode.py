"""Smoke tests for continuous recording toggles."""

from tabletop import pupil_bridge


def test_continuous_recording_mode_flag_is_bool() -> None:
    assert isinstance(pupil_bridge.CONTINUOUS_RECORDING_MODE, bool)
