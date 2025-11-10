import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("KIVY_WINDOW", "mock")
os.environ.setdefault("KIVY_GRAPHICS", "mock")
os.environ.setdefault("KIVY_AUDIO", "mock")
os.environ.setdefault("KIVY_TEXT", "mock")

pytest.importorskip("kivy")

from tabletop.tabletop_view import TabletopRoot


@pytest.fixture
def tabletop_root():
    root = TabletopRoot.__new__(TabletopRoot)
    root._input_debouncer = SimpleNamespace(allow=lambda *args, **kwargs: True)
    root._emit_button_bridge_event = lambda *args, **kwargs: None
    root._record_handler_duration = lambda *args, **kwargs: None
    root.record_action = lambda *args, **kwargs: None
    root.update_user_displays = lambda *args, **kwargs: None
    root.goto = lambda *args, **kwargs: None
    root.signal_buttons = {}
    root.ids = {}
    root.wid_safe = lambda name: None
    root.controller = SimpleNamespace()
    return root


def test_tap_card_logs_before_flip(mocker, tabletop_root):
    order = []

    result = SimpleNamespace(
        allowed=True,
        log_action="card_flip",
        log_payload={"card_slot": "inner"},
        record_text=None,
        next_phase=None,
    )

    tabletop_root.controller = SimpleNamespace(
        tap_card=lambda who, which: result,
    )

    widget = SimpleNamespace()
    tabletop_root.card_widget_for_player = lambda who, which: widget

    log_mock = mocker.patch.object(
        tabletop_root,
        "log_event",
        side_effect=lambda *args, **kwargs: order.append("log"),
    )
    flip_mock = mocker.patch.object(
        widget,
        "flip",
        side_effect=lambda *args, **kwargs: order.append("flip"),
    )

    tabletop_root.tap_card(1, "inner")

    assert log_mock.call_count == 1
    assert flip_mock.call_count == 1
    assert order == ["log", "flip"]


def test_pick_signal_logs_before_pressed_state(mocker, tabletop_root):
    order = []

    result = SimpleNamespace(
        accepted=True,
        log_payload={"signal_level": "mid"},
        record_text=None,
        next_phase=None,
    )

    tabletop_root.controller = SimpleNamespace(
        pick_signal=lambda player, level: result,
    )

    class DummyButton:
        def __init__(self, name):
            self.name = name
            self.disabled = False

        def set_pressed_state(self):
            raise AssertionError("set_pressed_state should be patched in tests")

        def set_live(self, value):
            return None

    selected_button = DummyButton("selected")
    other_button = DummyButton("other")
    button_map = {"btn_selected": selected_button, "btn_other": other_button}

    tabletop_root.signal_buttons = {1: {"mid": "btn_selected", "low": "btn_other"}}
    tabletop_root.wid_safe = lambda name: button_map.get(name)

    log_mock = mocker.patch.object(
        tabletop_root,
        "log_event",
        side_effect=lambda *args, **kwargs: order.append("log"),
    )
    pressed_mock = mocker.patch.object(
        selected_button,
        "set_pressed_state",
        side_effect=lambda *args, **kwargs: order.append("pressed"),
    )

    tabletop_root.pick_signal(1, "mid")

    assert log_mock.call_count == 1
    assert pressed_mock.call_count == 1
    assert order[0] == "log"
    assert "pressed" in order
    assert order.index("log") < order.index("pressed")
