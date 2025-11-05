"""Smoke tests for event label configuration flags."""

from tabletop import pupil_bridge


def test_force_event_labels_flag_is_bool() -> None:
    assert isinstance(pupil_bridge.FORCE_EVENT_LABELS, bool)
