from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics import (
    Color, RoundedRectangle, PushMatrix, PopMatrix, Rotate
)
from kivy.properties import NumericProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# Optional Vollbild
# Config.set('graphics', 'fullscreen', '1')

from game_engine import (
    GameEngine, GameEngineConfig,
    Player, VP, SignalLevel, Call, hand_category
)

# --- Wahrheitsregel (anpassbar) ---
def signal_truth_mapping(p1_cards: Tuple[int, int], level: SignalLevel) -> bool:
    return hand_category(*p1_cards) == level


class AutoLabel(Label):
    def __init__(self, **kwargs):
        kwargs.setdefault("halign", "center")
        kwargs.setdefault("valign", "middle")
        super().__init__(**kwargs)
        self.bind(size=self._update_text_size)

    def _update_text_size(self, *_):
        self.text_size = self.size


class InfoBox(BoxLayout):
    def __init__(self, **kwargs):
        padding = kwargs.pop("padding", (24, 18, 24, 18))
        spacing = kwargs.pop("spacing", 12)
        super().__init__(orientation="vertical", padding=padding, spacing=spacing, **kwargs)
        with self.canvas.before:
            self._bg_color = Color(1, 1, 1, 1)
            self._bg_rect = RoundedRectangle(radius=[18], pos=self.pos, size=self.size)
        self.bind(pos=self._update_rect, size=self._update_rect)

    def _update_rect(self, *_):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size


class RotatableBoxLayout(BoxLayout):
    angle = NumericProperty(0)

    def __init__(self, angle: float = 0, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            PushMatrix()
            self._rot = Rotate(angle=angle, origin=self.center)
        with self.canvas.after:
            PopMatrix()
        self.angle = angle
        self.bind(pos=self._update_transform, size=self._update_transform, angle=self._update_transform)

    def _update_transform(self, *_):
        if hasattr(self, "_rot"):
            self._rot.origin = self.center
            self._rot.angle = self.angle


class IconButton(Button):
    def __init__(self, stop_source: str, live_source: str, **kwargs):
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("background_color", (1, 1, 1, 0))
        kwargs.setdefault("border", (0, 0, 0, 0))
        super().__init__(**kwargs)
        self.stop_source = stop_source
        self.live_source = live_source
        self.set_state(False, highlighted=False)

    def set_state(self, enabled: bool, highlighted: bool = False):
        show_live = highlighted or enabled
        self.disabled = not enabled
        normal = self.live_source if show_live else self.stop_source
        self.background_normal = normal
        self.background_down = self.live_source
        self.background_disabled_normal = self.live_source if highlighted else self.stop_source
        self.background_disabled_down = self.background_disabled_normal


class StatusBadge(BoxLayout):
    def __init__(self, text: str, **kwargs):
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("width", 170)
        kwargs.setdefault("height", 90)
        kwargs.setdefault("padding", (10, 8, 10, 8))
        super().__init__(orientation="vertical", **kwargs)
        with self.canvas.before:
            self._color = Color(0.75, 0.75, 0.75, 1)
            self._rect = RoundedRectangle(radius=[16], pos=self.pos, size=self.size)
        self.bind(pos=self._update_rect, size=self._update_rect)
        self.label = AutoLabel(text=text, font_size=22, color=(0, 0, 0, 1))
        self.add_widget(self.label)

    def _update_rect(self, *_):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def set_active(self, active: bool):
        self._color.rgba = (0.25, 0.25, 0.25, 1) if active else (0.75, 0.75, 0.75, 1)


class PlayerPanel(RotatableBoxLayout):
    def __init__(self, ux_dir: Path, **kwargs):
        angle = kwargs.pop("angle", 0)
        super().__init__(angle=angle, orientation="vertical", spacing=18, size_hint=(None, None), **kwargs)
        self.bind(minimum_height=self.setter("height"))
        self.width = 220

        self.play_button = IconButton(str(ux_dir / "play_stop.png"), str(ux_dir / "play_live.png"), size=(130, 130))
        self.add_widget(self.play_button)

        self.add_widget(Widget(size_hint=(1, None), height=10))

        # Call buttons
        call_box = BoxLayout(orientation="vertical", spacing=12, size_hint=(1, None))
        call_box.bind(minimum_height=call_box.setter("height"))
        self.call_buttons = {
            Call.WAHRHEIT: IconButton(str(ux_dir / "wahr_stop.png"), str(ux_dir / "wahr_live.png"), size=(130, 130)),
            Call.BLUFF: IconButton(str(ux_dir / "bluff_stop.png"), str(ux_dir / "bluff_live.png"), size=(130, 130)),
        }
        call_box.add_widget(self.call_buttons[Call.WAHRHEIT])
        call_box.add_widget(self.call_buttons[Call.BLUFF])
        self.add_widget(call_box)

        # Signal buttons
        signal_box = BoxLayout(orientation="vertical", spacing=12, size_hint=(1, None))
        signal_box.bind(minimum_height=signal_box.setter("height"))
        self.signal_buttons = {}
        for level in (SignalLevel.HOCH, SignalLevel.MITTEL, SignalLevel.TIEF):
            btn = IconButton(str(ux_dir / f"{level.value}_stop.png"), str(ux_dir / f"{level.value}_live.png"), size=(130, 130))
            self.signal_buttons[level] = btn
            signal_box.add_widget(btn)
        self.add_widget(signal_box)

        # Status badges
        badge_box = BoxLayout(orientation="vertical", spacing=10, size_hint=(1, None))
        badge_box.bind(minimum_height=badge_box.setter("height"))
        self.status_badges = {
            SignalLevel.HOCH: StatusBadge("HOCH\n19"),
            SignalLevel.MITTEL: StatusBadge("MITTEL\n18/17/16"),
            SignalLevel.TIEF: StatusBadge("TIEF\n15/14"),
        }
        for badge in self.status_badges.values():
            badge_box.add_widget(badge)
        self.add_widget(badge_box)

        self.score_label = AutoLabel(text="", font_size=20, color=(0, 0, 0, 1), size_hint=(1, None), height=50)
        self.add_widget(self.score_label)

    def bind_play(self, callback):
        self.play_button.bind(on_release=callback)

    def bind_call(self, callback):
        for call, btn in self.call_buttons.items():
            btn.bind(on_release=partial(callback, call))

    def bind_signal(self, callback):
        for level, btn in self.signal_buttons.items():
            btn.bind(on_release=partial(callback, level))

    def set_play_state(self, enabled: bool):
        self.play_button.set_state(enabled, highlighted=enabled)

    def set_call_state(self, enabled: bool, selected: Optional[Call]):
        for call, btn in self.call_buttons.items():
            highlight = (selected == call)
            btn.set_state(enabled if not highlight else False, highlighted=highlight)

    def set_signal_state(self, enabled: bool, selected: Optional[SignalLevel]):
        for level, btn in self.signal_buttons.items():
            highlight = (selected == level)
            btn.set_state(enabled if not highlight else False, highlighted=highlight)

    def set_category(self, category: Optional[SignalLevel]):
        for level, badge in self.status_badges.items():
            badge.set_active(category == level)

    def set_score(self, text: str):
        self.score_label.text = text


class CardWidget(ButtonBehavior, Image):
    angle = NumericProperty(0)

    def __init__(self, ui: "TwoPlayerUI", owner: VP, index: int, angle: float = 0, **kwargs):
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("allow_stretch", True)
        kwargs.setdefault("keep_ratio", True)
        super().__init__(**kwargs)
        self.ui = ui
        self.owner = owner
        self.index = index
        self.angle = angle
        self.face_up = False
        self.expected = False
        self.back_source = ui.img_back
        self.front_source = ui.img_back
        with self.canvas.before:
            PushMatrix()
            self._rot = Rotate(angle=angle, origin=self.center)
        with self.canvas.after:
            PopMatrix()
        self.source = self.back_source
        self.opacity = 0.7
        self.bind(pos=self._update_transform, size=self._update_transform)

    def _update_transform(self, *_):
        self._rot.origin = self.center

    def on_release(self):
        if not self.disabled:
            self.ui._reveal(self.owner, self.index)

    def set_card(self, value: Optional[int], face_up: bool, front_source: Optional[str] = None):
        if front_source:
            self.front_source = front_source
        else:
            self.front_source = self.ui._img_for_value(value)
        self.face_up = face_up
        self.source = self.front_source if face_up else self.back_source
        self.opacity = 1.0 if face_up or not self.disabled else 0.7
        self.reload()

    def set_interactive(self, enabled: bool):
        self.disabled = not enabled
        if not enabled and not self.face_up:
            self.opacity = 0.7
        elif enabled:
            self.opacity = 1.0

class TwoPlayerUI(BoxLayout):
    def __init__(self, **kwargs):
        Window.clearcolor = (191 / 255, 191 / 255, 191 / 255, 1)
        super().__init__(orientation="vertical", padding=0, spacing=0, **kwargs)

        base = Path(__file__).resolve().parent
        self.base = base
        self.card_dir = base / "Karten"
        self.img_back = str(self.card_dir / "back.png") if (self.card_dir / "back.png").exists() else ""

        self.engine: Optional[GameEngine] = None
        self.session_popup: Optional[Popup] = None
        self._session_inputs = {}
        self.session_message = ""

        # Container, der zwischen Spielbrett und Übergangsseite wechselt
        self.content_container = FloatLayout()
        self.add_widget(self.content_container)

        ux_dir = base / "UX"

        # Hauptebene des Spielbretts
        self.board = FloatLayout(size_hint=(1, 1))
        self.content_container.add_widget(self.board)

        # Informationsboxen
        self.top_info_wrapper = RotatableBoxLayout(
            angle=180,
            size_hint=(0.62, None),
            height=190,
            pos_hint={"center_x": 0.5, "top": 0.97},
        )
        self.top_info_box = InfoBox(size_hint=(1, 1))
        self.top_info_label = AutoLabel(text="", font_size=32, color=(0, 0, 0, 1))
        self.top_info_box.add_widget(self.top_info_label)
        self.top_info_wrapper.add_widget(self.top_info_box)
        self.board.add_widget(self.top_info_wrapper)

        self.bottom_info_box = InfoBox(
            size_hint=(0.62, None),
            height=190,
            pos_hint={"center_x": 0.5, "y": 0.03},
        )
        self.bottom_info_label = AutoLabel(text="", font_size=32, color=(0, 0, 0, 1))
        self.bottom_detail_label = AutoLabel(text="", font_size=24, color=(0.2, 0.2, 0.2, 1))
        self.bottom_info_box.add_widget(self.bottom_info_label)
        self.bottom_info_box.add_widget(self.bottom_detail_label)
        self.board.add_widget(self.bottom_info_box)

        # Kartenraster
        card_width, card_height = 220, 320
        self.card_grid = GridLayout(cols=2, rows=2, spacing=30, size_hint=(None, None))
        self.card_grid.size = (card_width * 2 + 30, card_height * 2 + 30)
        self.card_grid.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        self.board.add_widget(self.card_grid)

        self.card_widgets: Dict[Tuple[VP, int], CardWidget] = {}
        for vp, idx, angle in [
            (VP.VP2, 0, 180),
            (VP.VP2, 1, 180),
            (VP.VP1, 0, 0),
            (VP.VP1, 1, 0),
        ]:
            widget = CardWidget(self, vp, idx, angle=angle, size=(card_width, card_height))
            self.card_widgets[(vp, idx)] = widget
            self.card_grid.add_widget(widget)

        # Steuerpanels
        self.vp1_panel = PlayerPanel(ux_dir, angle=0)
        self.vp1_panel.pos_hint = {"x": 0.035, "center_y": 0.5}
        self.vp1_panel.bind_play(lambda *_: self._start_or_next_for_vp(VP.VP1))
        self.vp1_panel.bind_signal(lambda level, *_: self._signal_from_vp(VP.VP1, level))
        self.vp1_panel.bind_call(lambda call, *_: self._call_from_vp(VP.VP1, call))
        self.board.add_widget(self.vp1_panel)

        self.vp2_panel = PlayerPanel(ux_dir, angle=180)
        self.vp2_panel.pos_hint = {"right": 0.965, "center_y": 0.5}
        self.vp2_panel.bind_play(lambda *_: self._start_or_next_for_vp(VP.VP2))
        self.vp2_panel.bind_signal(lambda level, *_: self._signal_from_vp(VP.VP2, level))
        self.vp2_panel.bind_call(lambda call, *_: self._call_from_vp(VP.VP2, call))
        self.board.add_widget(self.vp2_panel)

        # Übergangsüberlagerung
        self.transition_box = FloatLayout()
        self.transition_panel = InfoBox(size_hint=(0.7, None), height=420, pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.lbl_transition = AutoLabel(text="", font_size=32, color=(0, 0, 0, 1))
        self.transition_panel.add_widget(self.lbl_transition)
        self.transition_panel.add_widget(Widget(size_hint=(1, None), height=30))
        btn_row = BoxLayout(size_hint=(1, None), height=180, spacing=60)

        self.transition_btn_vp1 = BoxLayout(orientation="vertical", spacing=12, size_hint=(0.5, 1))
        self.btn_transition_vp1 = IconButton(str(ux_dir / "play_stop.png"), str(ux_dir / "play_live.png"), size=(150, 150))
        self.transition_label_vp1 = AutoLabel(text="", font_size=26, color=(0, 0, 0, 1), size_hint=(1, None), height=40)
        self.btn_transition_vp1.bind(on_release=lambda *_: self._continue_after_block(VP.VP1))
        self.transition_btn_vp1.add_widget(self.btn_transition_vp1)
        self.transition_btn_vp1.add_widget(self.transition_label_vp1)

        self.transition_btn_vp2 = BoxLayout(orientation="vertical", spacing=12, size_hint=(0.5, 1))
        self.btn_transition_vp2 = IconButton(str(ux_dir / "play_stop.png"), str(ux_dir / "play_live.png"), size=(150, 150))
        self.transition_label_vp2 = AutoLabel(text="", font_size=26, color=(0, 0, 0, 1), size_hint=(1, None), height=40)
        self.btn_transition_vp2.bind(on_release=lambda *_: self._continue_after_block(VP.VP2))
        self.transition_btn_vp2.add_widget(self.btn_transition_vp2)
        self.transition_btn_vp2.add_widget(self.transition_label_vp2)

        btn_row.add_widget(self.transition_btn_vp1)
        btn_row.add_widget(self.transition_btn_vp2)
        self.transition_panel.add_widget(btn_row)
        self.transition_box.add_widget(self.transition_panel)

        self.in_transition = False
        self.transition_message = ""
        self.transition_final = False
        self.transition_ready_vp1 = False
        self.transition_ready_vp2 = False

        self.block_sequence = [
            {"block": 1, "csv": "Paare1.csv", "condition": "no_payout", "payout": False},
            {"block": 2, "csv": "Paare3.csv", "condition": "payout", "payout": True},
            {"block": 3, "csv": "Paare2.csv", "condition": "no_payout", "payout": False},
            {"block": 4, "csv": "Paare4.csv", "condition": "payout", "payout": True},
        ]
        self.current_block_idx: Optional[int] = None
        self.next_block_idx: int = 0
        self.session_identifier = ""
        self.session_number_value: Optional[int] = None

        self.log_dir = self.base / "logs"

        Clock.schedule_interval(lambda dt: self.refresh(), 0.1)
        Clock.schedule_once(lambda dt: self._open_session_dialog(), 0.1)
        self.refresh()

    def _open_session_dialog(self):
        if self.session_popup:
            return

        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        form = GridLayout(cols=2, spacing=6, size_hint_y=None)
        form.bind(minimum_height=form.setter("height"))

        def add_field(label_text: str, key: str, default: str, **kwargs):
            form.add_widget(Label(text=label_text, size_hint_y=None, height=32))
            ti = TextInput(text=default, multiline=False, **kwargs)
            form.add_widget(ti)
            self._session_inputs[key] = ti

        add_field("Session (Zahl)", "session", "", input_filter="int")
        add_field("Startblock", "block", "1", input_filter="int")

        content.add_widget(form)
        self._session_error = Label(text="", color=(1, 0, 0, 1), size_hint_y=None, height=24)
        content.add_widget(self._session_error)

        btn_box = BoxLayout(size_hint_y=None, height=44, spacing=8)
        btn_ok = Button(text="Start", on_release=lambda *_: self._confirm_session())
        btn_box.add_widget(btn_ok)
        content.add_widget(btn_box)

        popup = Popup(title="Sessiondaten", content=content, size_hint=(0.6, 0.5), auto_dismiss=False)
        self.session_popup = popup
        popup.open()

    def _confirm_session(self):
        session_text = self._session_inputs["session"].text.strip()
        block_text = self._session_inputs["block"].text.strip()

        try:
            session_num = int(session_text)
            if session_num <= 0:
                raise ValueError
        except ValueError:
            self._session_error.text = "Bitte eine positive Session-Zahl eingeben."
            return

        try:
            block_num = int(block_text) if block_text else 1
            if not (1 <= block_num <= len(self.block_sequence)):
                raise ValueError
        except ValueError:
            self._session_error.text = f"Block muss zwischen 1 und {len(self.block_sequence)} liegen."
            return

        self.session_identifier = f"S{session_num:03d}"
        self.session_number_value = session_num
        self.next_block_idx = block_num - 1
        self.current_block_idx = None

        if self.engine:
            self.engine.close()
            self.engine = None

        if self.session_popup:
            self.session_popup.dismiss()
            self.session_popup = None

        self.session_message = f"Session {session_num}"
        self._start_block()

    def _show_board(self):
        if self.transition_box.parent is self.content_container:
            self.content_container.remove_widget(self.transition_box)
        if self.board.parent is None:
            self.content_container.add_widget(self.board)
        self.in_transition = False
        self.transition_message = ""

    def _show_transition(self, message: str, button_text: str):
        self.transition_message = message
        self.transition_label_vp1.text = f"{button_text} (VP1)"
        self.transition_label_vp2.text = f"{button_text} (VP2)"
        self.btn_transition_vp1.set_state(True, highlighted=True)
        self.btn_transition_vp2.set_state(True, highlighted=True)
        self.transition_ready_vp1 = False
        self.transition_ready_vp2 = False
        self.lbl_transition.text = message
        if self.board.parent is self.content_container:
            self.content_container.remove_widget(self.board)
        if self.transition_box.parent is None:
            self.content_container.add_widget(self.transition_box)
        self.in_transition = True
        self.bottom_info_label.text = message
        self.bottom_detail_label.text = ""
        self.top_info_label.text = message
        for panel in (self.vp1_panel, self.vp2_panel):
            panel.set_play_state(False)
            panel.set_signal_state(False, None)
            panel.set_call_state(False, None)
            panel.set_score("")
            panel.set_category(None)
        for widget in self.card_widgets.values():
            widget.set_card(None, False)
            widget.set_interactive(False)

    def _start_block(self):
        if not self.session_identifier:
            return
        if self.next_block_idx >= len(self.block_sequence):
            message = (
                "Alle Blöcke sind abgeschlossen. Vielen Dank!"
            )
            self.transition_final = True
            self._show_transition(message, "Experiment beenden")
            return

        block_info = self.block_sequence[self.next_block_idx]
        csv_file = self.base / block_info["csv"]
        if not csv_file.exists():
            self.bottom_info_label.text = "Fehlende CSV"
            self.bottom_detail_label.text = f"CSV {block_info['csv']} nicht gefunden."
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        cfg = GameEngineConfig(
            session_id=self.session_identifier,
            session_number=self.session_number_value,
            block=block_info["block"],
            condition=block_info["condition"],
            csv_path=str(csv_file),
            db_path=str(self.log_dir / f"events_{self.session_identifier}.sqlite3"),
            csv_log_path=None,
            log_dir=str(self.log_dir),
            payout=block_info["payout"],
            payout_start_points=16 if block_info["payout"] else 0,
        )

        if self.engine:
            self.engine.close()

        self.engine = GameEngine(cfg)
        self.current_block_idx = self.next_block_idx
        self.next_block_idx = self.current_block_idx + 1
        condition_label = "Auszahlung" if block_info["payout"] else "ohne Auszahlung"
        self.session_message = (
            f"Session {self.session_number_value}, Block {block_info['block']} ({condition_label})"
        )
        self.transition_final = False
        self._show_board()
        self.refresh()

    def _handle_block_finished(self):
        if self.in_transition:
            return
        finished_info = None
        if self.current_block_idx is not None and self.current_block_idx < len(self.block_sequence):
            finished_info = self.block_sequence[self.current_block_idx]
        if self.engine:
            self.engine.close()
            self.engine = None
        block_label = finished_info["block"] if finished_info else ""
        if self.next_block_idx < len(self.block_sequence):
            message = (
                f"Block {block_label} ist fertig. Atmen Sie einen Moment durch und klicken Sie auf Weiter, "
                "wenn Sie bereit für den nächsten Block sind."
            )
            button_text = "Weiter"
            self.transition_final = False
        else:
            message = (
                f"Block {block_label} ist fertig. Atmen Sie einen Moment durch und klicken Sie auf Weiter, "
                "wenn Sie bereit sind, das Experiment zu beenden."
            )
            button_text = "Experiment beenden"
            self.transition_final = True
        self.current_block_idx = None
        self._show_transition(message, button_text)

    def _continue_after_block(self, vp: VP):
        if vp == VP.VP1:
            self.transition_ready_vp1 = True
            self.btn_transition_vp1.set_state(False, highlighted=True)
        elif vp == VP.VP2:
            self.transition_ready_vp2 = True
            self.btn_transition_vp2.set_state(False, highlighted=True)

        if not (self.transition_ready_vp1 and self.transition_ready_vp2):
            return

        if self.transition_final and self.next_block_idx >= len(self.block_sequence):
            self.bottom_info_label.text = "Experiment abgeschlossen."
            self.top_info_label.text = "Experiment abgeschlossen."
            self.bottom_detail_label.text = ""
            return
        self._start_block()

    # ===== Helper =====
    def _img_for_value(self, val: Optional[int]) -> str:
        if val is None:
            return self.img_back
        p = self.card_dir / f"{val}.png"
        return str(p) if p.exists() else self.img_back

    def _player_for_vp(self, vp: VP) -> Player:
        if not self.engine:
            raise RuntimeError("Engine nicht initialisiert")
        r = self.engine.current.roles
        return Player.P1 if r.p1_is == vp else Player.P2

    def _revealed(self, vp: VP, idx: int) -> bool:
        """Direkt aus dem Engine-Inneren lesen (robuster)."""
        if not self.engine:
            return False
        vis = self.engine.current.vis
        roles = self.engine.current.roles
        if roles.p1_is == vp:
            return vis.p1_revealed[idx]
        else:
            return vis.p2_revealed[idx]

    def _expected_reveal(self):
        """Welcher VP ist als nächstes dran (rollenrichtig) und welche Karte (0/1)?"""
        if not self.engine:
            return None
        st = self.engine.get_public_state()
        if st["phase"] != "DEALING":
            return None
        vp_p1 = VP[st["roles"]["P1"]]   # 'VP1'/'VP2' → Enum
        vp_p2 = VP[st["roles"]["P2"]]
        p1r0, p1r1 = st["p1_revealed"]
        p2r0, p2r1 = st["p2_revealed"]
        if not p1r0:  # P1-K1
            return (vp_p1, 0)
        if not p2r0:  # P2-K1
            return (vp_p2, 0)
        if not p1r1:  # P1-K2
            return (vp_p1, 1)
        if not p2r1:  # P2-K2
            return (vp_p2, 1)
        return None

    def _cards_for_vp(self, vp: VP) -> Tuple[int, int]:
        if not self.engine:
            return (0, 0)
        plan = self.engine.current.plan
        return plan.vp1_cards if vp == VP.VP1 else plan.vp2_cards

    def _category_for_cards(self, cards: Tuple[int, int]) -> Optional[SignalLevel]:
        total = sum(cards)
        if total == 19:
            return SignalLevel.HOCH
        if total in (16, 17, 18):
            return SignalLevel.MITTEL
        if total in (14, 15):
            return SignalLevel.TIEF
        return None

    def _info_text_for_vp(self, vp: VP, st: Dict[str, Any], rs) -> str:
        lines = []
        role_label = "Spieler 1" if st["roles"]["P1"] == vp.value else "Spieler 2"
        lines.append(f"Du bist {role_label}")

        if rs.p1_signal:
            sig_text = rs.p1_signal.value.capitalize()
            if rs.roles.p1_is == vp:
                lines.append(f"DEINE WAHL: {sig_text}")
            else:
                lines.append(f"ANDERER SPIELER: {sig_text}")

        if rs.p2_call:
            call_text = rs.p2_call.value.capitalize()
            if rs.roles.p2_is == vp:
                lines.append(f"DEINE WAHL: {call_text}")
            else:
                lines.append(f"ANDERER SPIELER: {call_text}")

        phase = st.get("phase")
        if phase in ("REVEAL_SCORE", "ROUND_DONE"):
            winner = st.get("winner")
            if winner:
                winner_vp = rs.roles.p1_is if winner == Player.P1.value else rs.roles.p2_is
                lines.append("GEWONNEN" if winner_vp == vp else "VERLOREN")
            else:
                reason = st.get("outcome_reason", "")
                if "Unentschieden" in reason:
                    lines.append("UNENTSCHIEDEN")

        return "\n".join(lines)

    # ===== Actions =====
    def _start_or_next_for_vp(self, vp: VP):
        if not self.engine:
            return
        st = self.engine.get_public_state()
        player = self._player_for_vp(vp)
        try:
            if st["phase"] == "WAITING_START":
                self.engine.click_start(player)
            elif st["phase"] == "ROUND_DONE":
                self.engine.click_next_round(player)
        except Exception as e:
            self.bottom_detail_label.text = f"Start/Next-Fehler ({vp.value}): {e}"
        self.refresh()

    def _reveal(self, vp: VP, idx: int):
        if not self.engine:
            return
        try:
            p = self._player_for_vp(vp)
            self.engine.click_reveal_card(p, idx)
        except Exception as e:
            self.bottom_detail_label.text = f"Reveal-Fehler ({vp.value} K{idx+1}): {e}"
        self.refresh()

    def _signal_from_vp(self, vp: VP, level: SignalLevel):
        # Nur die VP, die gerade Spieler 1 ist, darf signalen
        if not self.engine:
            return
        if self.engine.current.roles.p1_is != vp:
            return
        try:
            self.engine.p1_signal(level)
        except Exception as e:
            self.bottom_detail_label.text = f"Signal-Fehler ({vp.value}): {e}"
        self.refresh()

    def _call_from_vp(self, vp: VP, call: Call):
        # Nur die VP, die gerade Spieler 2 ist, darf callen
        if not self.engine:
            return
        if self.engine.current.roles.p2_is != vp:
            return
        rs = self.engine.current
        if rs.roles.p1_is == VP.VP1:
            p1_cards = rs.plan.vp1_cards
        else:
            p1_cards = rs.plan.vp2_cards
        truth = signal_truth_mapping(p1_cards, rs.p1_signal) if rs.p1_signal else None
        try:
            self.engine.p2_call(call, p1_hat_wahrheit_gesagt=truth)
        except Exception as e:
            self.bottom_detail_label.text = f"Call-Fehler ({vp.value}): {e}"
        self.refresh()

    # ===== Refresh =====
    def refresh(self):
        if self.in_transition:
            return

        if not self.engine:
            self.top_info_label.text = "Session vorbereiten"
            self.bottom_info_label.text = "Bitte Sessiondaten bestätigen."
            self.bottom_detail_label.text = ""
            for panel in (self.vp1_panel, self.vp2_panel):
                panel.set_play_state(False)
                panel.set_signal_state(False, None)
                panel.set_call_state(False, None)
                panel.set_score("")
                panel.set_category(None)
            for widget in self.card_widgets.values():
                widget.set_card(None, False)
                widget.set_interactive(False)
            return

        st = self.engine.get_public_state()
        if st.get("phase") == "FINISHED":
            self._handle_block_finished()
            return

        rs = self.engine.current
        ph = st["phase"]

        self.top_info_label.text = self._info_text_for_vp(VP.VP2, st, rs)
        self.bottom_info_label.text = self._info_text_for_vp(VP.VP1, st, rs)

        outcome = st.get("outcome_reason")
        self.bottom_detail_label.text = outcome or self.session_message

        scores = st.get("scores")
        if scores:
            self.vp1_panel.set_score(f"{scores.get('VP1', '')} Punkte")
            self.vp2_panel.set_score(f"{scores.get('VP2', '')} Punkte")
        else:
            self.vp1_panel.set_score("")
            self.vp2_panel.set_score("")

        is_vp1_p1 = (st["roles"]["P1"] == "VP1")
        is_vp2_p1 = (st["roles"]["P1"] == "VP2")

        vp1_ready_start = st["p1_ready"] if is_vp1_p1 else st["p2_ready"]
        vp1_ready_next = st["next_ready_p1"] if is_vp1_p1 else st["next_ready_p2"]
        enable_play_vp1 = (ph == "WAITING_START" and not vp1_ready_start) or (
            ph == "ROUND_DONE" and not vp1_ready_next
        )
        self.vp1_panel.set_play_state(enable_play_vp1)

        vp2_ready_start = st["p1_ready"] if is_vp2_p1 else st["p2_ready"]
        vp2_ready_next = st["next_ready_p1"] if is_vp2_p1 else st["next_ready_p2"]
        enable_play_vp2 = (ph == "WAITING_START" and not vp2_ready_start) or (
            ph == "ROUND_DONE" and not vp2_ready_next
        )
        self.vp2_panel.set_play_state(enable_play_vp2)

        enable_sig_vp1 = (ph == "SIGNAL_WAIT" and is_vp1_p1 and rs.p1_signal is None)
        selected_sig_vp1 = rs.p1_signal if rs.roles.p1_is == VP.VP1 else None
        self.vp1_panel.set_signal_state(enable_sig_vp1, selected_sig_vp1)

        enable_sig_vp2 = (ph == "SIGNAL_WAIT" and is_vp2_p1 and rs.p1_signal is None)
        selected_sig_vp2 = rs.p1_signal if rs.roles.p1_is == VP.VP2 else None
        self.vp2_panel.set_signal_state(enable_sig_vp2, selected_sig_vp2)

        enable_call_vp1 = (ph == "CALL_WAIT" and rs.roles.p2_is == VP.VP1 and rs.p2_call is None)
        selected_call_vp1 = rs.p2_call if rs.roles.p2_is == VP.VP1 else None
        self.vp1_panel.set_call_state(enable_call_vp1, selected_call_vp1)

        enable_call_vp2 = (ph == "CALL_WAIT" and rs.roles.p2_is == VP.VP2 and rs.p2_call is None)
        selected_call_vp2 = rs.p2_call if rs.roles.p2_is == VP.VP2 else None
        self.vp2_panel.set_call_state(enable_call_vp2, selected_call_vp2)

        self.vp1_panel.set_category(self._category_for_cards(self._cards_for_vp(VP.VP1)))
        self.vp2_panel.set_category(self._category_for_cards(self._cards_for_vp(VP.VP2)))

        show_mid = ph in ("REVEAL_SCORE", "ROUND_DONE")
        expected = self._expected_reveal()

        for (vp, idx), widget in self.card_widgets.items():
            cards = self._cards_for_vp(vp)
            value = cards[idx] if cards else None
            face_up = show_mid or self._revealed(vp, idx)
            widget.set_card(value, face_up)
            enable = ph == "DEALING" and expected is not None and expected == (vp, idx)
            widget.set_interactive(enable and not show_mid)
            if not face_up and not enable:
                widget.opacity = 0.7


class TouchGameApp(App):
    def build(self):
        return TwoPlayerUI()

    def on_stop(self):
        root = self.root
        if root and getattr(root, "engine", None):
            root.engine.close()


if __name__ == "__main__":
    TouchGameApp().run()
