from __future__ import annotations

import csv
import itertools
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import DictProperty, NumericProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.switch import Switch
from kivy.uix.textinput import TextInput

from tabletop.data.blocks import load_blocks, load_csv_rounds, value_to_card_path
from tabletop.data.config import ARUCO_OVERLAY_PATH, ROOT
from tabletop.logging.events import Events
from tabletop.logging.round_csv import (
    close_round_log,
    init_round_log,
    round_log_action_label,
    write_round_log,
)
from tabletop.overlay.fixation import (
    generate_fixation_tone,
    play_fixation_tone as overlay_play_fixation_tone,
    run_fixation_sequence as overlay_run_fixation_sequence,
)
from tabletop.overlay.process import start_overlay_process, stop_overlay_process
from tabletop.state.controller import TabletopController, TabletopState
from tabletop.state.phases import UXPhase, to_engine_phase
from tabletop.ui import widgets as ui_widgets
from tabletop.engine import POINTS_PER_WIN
from tabletop.ui.assets import (
    ASSETS,
    FIX_LIVE_IMAGE,
    FIX_STOP_IMAGE,
    resolve_background_texture,
)
from tabletop.ui.widgets import CardWidget, IconButton, RotatableLabel


ui_widgets.ASSETS = ASSETS

STATE_FIELD_NAMES = set(TabletopState.__dataclass_fields__)


class TabletopRoot(FloatLayout):
    _STATE_FIELDS = STATE_FIELD_NAMES

    SCALE_FACTOR = NumericProperty(0.7)

    bg_texture = ObjectProperty(None, rebind=True)
    base_width = NumericProperty(3840.0)
    base_height = NumericProperty(2160.0)
    button_scale = NumericProperty(0.8)
    scale = NumericProperty(1.0)
    horizontal_offset = NumericProperty(0.08)
    # Responsive Seitenränder (Prozent + physische Untergrenze)
    side_margin_frac = NumericProperty(0.14)
    side_margin_min_px = NumericProperty(280.0)
    side_margin_max_frac = NumericProperty(0.22)
    side_margin_target_cm = NumericProperty(3.2)

    # Ergebniswert (in Pixeln)
    horizontal_margin_px = NumericProperty(0.0)

    btn_start_p1 = ObjectProperty(None)
    btn_start_p2 = ObjectProperty(None)
    p1_outer = ObjectProperty(None)
    p1_inner = ObjectProperty(None)
    p2_outer = ObjectProperty(None)
    p2_inner = ObjectProperty(None)
    intro_overlay = ObjectProperty(None)
    pause_cover = ObjectProperty(None)
    fixation_overlay = ObjectProperty(None)
    fixation_image = ObjectProperty(None)
    round_badge = ObjectProperty(None)

    signal_buttons = DictProperty({})
    decision_buttons = DictProperty({})
    center_cards = DictProperty({})
    user_displays = DictProperty({})
    intro_labels = DictProperty({})
    pause_labels = DictProperty({})

    def wid(self, name: str):
        # Liefert das Widget-Objekt oder None, ohne Truthiness auf WeakProxy auszulösen
        return self.ids.get(name, None)

    def wid_safe(self, name: str):
        # Wie wid(), aber tolerant gegen bereits freigegebene WeakProxy-Objekte
        w = self.ids.get(name, None)
        if w is None:
            return None
        try:
            # sanfter Deref-Test, löst ReferenceError aus, falls freigegeben
            _ = w.opacity
        except ReferenceError:
            return None
        return w

    def __init__(
        self,
        *,
        controller: Optional[TabletopController] = None,
        state: Optional[TabletopState] = None,
        events_factory: Callable[[str, str], Events] = Events,
        start_overlay: Callable[..., Optional[Any]] = start_overlay_process,
        stop_overlay: Callable[[Optional[Any]], Optional[Any]] = stop_overlay_process,
        fixation_runner: Callable[..., Any] = overlay_run_fixation_sequence,
        fixation_player: Callable[[Any], None] = overlay_play_fixation_tone,
        fixation_tone_factory: Callable[[int], Any] = generate_fixation_tone,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.events_factory = events_factory
        self.start_overlay = start_overlay
        self.stop_overlay = stop_overlay
        self.fixation_runner = fixation_runner
        self.fixation_player = fixation_player
        self.fixation_tone_factory = fixation_tone_factory
        self.bg_texture = resolve_background_texture()
        Window.bind(on_resize=self._on_window_resize)
        self.bind(size=self._update_scale)

        if state is None:
            state = TabletopState(blocks=load_blocks())
        elif not state.blocks:
            state.blocks = load_blocks()
        self.controller = controller or TabletopController(state)
        self._blocks = state.blocks if state.blocks else load_blocks()
        self.aruco_enabled = False
        self._aruco_proc = None
        self.start_block = 1
        # Versuchsperson 1 sitzt immer unten (Spieler 1), Versuchsperson 2 oben (Spieler 2)
        self._fixed_role_mapping = {1: 1, 2: 2}
        self.role_by_physical = self._fixed_role_mapping.copy()
        self.physical_by_role = {role: player for player, role in self.role_by_physical.items()}
        self.update_turn_order()
        self.phase = UXPhase.WAIT_BOTH_START
        self.session_number = None
        self.session_id = None
        self.session_storage_id = None
        self.logger = None
        self.log_dir = Path(ROOT) / 'logs'
        self.session_popup = None
        self.session_configured = False
        self.round_log_path = None
        self.round_log_fp = None
        self.round_log_writer = None
        self.round_log_buffer = []

        # --- UI Elemente initialisieren
        self._configure_widgets()
        self.setup_round()
        self.apply_phase()
        Clock.schedule_once(lambda *_: self.prompt_session_number(), 0.1)

    def __setattr__(self, key, value):
        if key in self._STATE_FIELDS and 'controller' in self.__dict__:
            setattr(self.controller.state, key, value)
            return
        if key == 'overlay_process':
            super().__setattr__(key, value)
            object.__setattr__(self, '_aruco_proc', value)
            return
        super().__setattr__(key, value)

    def __getattr__(self, item):
        if item in self._STATE_FIELDS and 'controller' in self.__dict__:
            return getattr(self.controller.state, item)
        raise AttributeError(item)

    # --- Layout & Elemente
    def _configure_widgets(self):
        btn_start_p1 = self.wid_safe('btn_start_p1')
        if btn_start_p1 is not None:
            btn_start_p1.bind(on_release=lambda *_: self.start_pressed(1))
            btn_start_p1.set_rotation(0)
        btn_start_p2 = self.wid_safe('btn_start_p2')
        if btn_start_p2 is not None:
            btn_start_p2.bind(on_release=lambda *_: self.start_pressed(2))
            btn_start_p2.set_rotation(180)

        p1_outer = self.wid_safe('p1_outer')
        if p1_outer is not None:
            p1_outer.bind(on_release=lambda *_: self.tap_card(1, 'outer'))
        p1_inner = self.wid_safe('p1_inner')
        if p1_inner is not None:
            p1_inner.bind(on_release=lambda *_: self.tap_card(1, 'inner'))
        p2_outer = self.wid_safe('p2_outer')
        if p2_outer is not None:
            p2_outer.bind(on_release=lambda *_: self.tap_card(2, 'outer'))
        p2_inner = self.wid_safe('p2_inner')
        if p2_inner is not None:
            p2_inner.bind(on_release=lambda *_: self.tap_card(2, 'inner'))

        self.signal_buttons = {
            1: {
                'low': 'signal_p1_low',
                'mid': 'signal_p1_mid',
                'high': 'signal_p1_high',
            },
            2: {
                'low': 'signal_p2_low',
                'mid': 'signal_p2_mid',
                'high': 'signal_p2_high',
            },
        }
        for level, btn_id in self.signal_buttons.get(1, {}).items():
            btn = self.wid_safe(btn_id)
            if btn is not None:
                btn.bind(on_release=lambda _, lvl=level: self.pick_signal(1, lvl))
                btn.set_rotation(0)
        for level, btn_id in self.signal_buttons.get(2, {}).items():
            btn = self.wid_safe(btn_id)
            if btn is not None:
                btn.bind(on_release=lambda _, lvl=level: self.pick_signal(2, lvl))
                btn.set_rotation(180)

        self.decision_buttons = {
            1: {
                'bluff': 'decision_p1_bluff',
                'wahr': 'decision_p1_wahr',
            },
            2: {
                'bluff': 'decision_p2_bluff',
                'wahr': 'decision_p2_wahr',
            },
        }
        for choice, btn_id in self.decision_buttons.get(1, {}).items():
            btn = self.wid_safe(btn_id)
            if btn is not None:
                btn.bind(on_release=lambda _, ch=choice: self.pick_decision(1, ch))
                btn.set_rotation(0)
        for choice, btn_id in self.decision_buttons.get(2, {}).items():
            btn = self.wid_safe(btn_id)
            if btn is not None:
                btn.bind(on_release=lambda _, ch=choice: self.pick_decision(2, ch))
                btn.set_rotation(180)

        self.center_cards = {
            1: ['center_p1_card_right', 'center_p1_card_left'],
            2: ['center_p2_card_left', 'center_p2_card_right'],
        }

        self.user_displays = {
            1: 'user_display_p1',
            2: 'user_display_p2',
        }
        for player, display_id in self.user_displays.items():
            display = self.wid_safe(display_id)
            if display is not None:
                display.set_rotation(0 if player == 1 else 180)
                display.text = ''
                display.opacity = 1

        self.intro_labels = {
            1: 'intro_label_p1',
            2: 'intro_label_p2',
        }
        for player, label_id in self.intro_labels.items():
            label = self.wid_safe(label_id)
            if label is not None:
                label.set_rotation(0 if player == 1 else 180)

        self.pause_labels = {
            1: 'pause_label_p1',
            2: 'pause_label_p2',
        }
        for player, label_id in self.pause_labels.items():
            label = self.wid_safe(label_id)
            if label is not None:
                label.set_rotation(0 if player == 1 else 180)
                label.bind(texture_size=lambda *_: None)

        fixation_overlay = self.wid_safe('fixation_overlay')
        if fixation_overlay is not None:
            fixation_overlay.opacity = 0
            fixation_overlay.disabled = True
        fixation_image = self.wid_safe('fixation_image')
        if fixation_image is not None:
            fixation_image.opacity = 1

        self.bring_start_buttons_to_front()

        self._update_scale()

        # interne States
        self.p1_pressed = False
        self.p2_pressed = False
        self.player_signals = {1: None, 2: None}
        self.player_decisions = {1: None, 2: None}
        self.status_lines = {1: [], 2: []}
        self.status_labels = {1: None, 2: None}
        self.last_outcome = {
            'winner': None,
            'truthful': None,
            'actual_level': None,
            'signal_choice': None,
            'judge_choice': None
        }
        self.card_cycle = itertools.cycle(['7.png', '8.png', '9.png', '10.png', '11.png'])

        self.total_rounds_planned = sum(len(block['rounds']) for block in self.blocks)
        self.overlay_process = self._aruco_proc
        self.fixation_running = False
        self.fixation_required = False
        self.pending_fixation_callback = None
        self.intro_active = True
        self.next_block_preview = None
        self.fixation_tone_fs = 44100
        self.fixation_tone = self.fixation_tone_factory(self.fixation_tone_fs)

        self._update_scale()
        self.update_user_displays()
        self.update_intro_overlay()

    def bring_start_buttons_to_front(self):
        btn_start_p1 = self.wid_safe('btn_start_p1')
        btn_start_p2 = self.wid_safe('btn_start_p2')
        if btn_start_p1 is not None and btn_start_p1.parent is self:
            self.remove_widget(btn_start_p1)
            self.add_widget(btn_start_p1)
        if btn_start_p2 is not None and btn_start_p2.parent is self:
            self.remove_widget(btn_start_p2)
            self.add_widget(btn_start_p2)

    def update_intro_overlay(self):
        intro_overlay = self.wid_safe('intro_overlay')
        if intro_overlay is None:
            return
        active = bool(self.intro_active)
        if active:
            if intro_overlay.parent is None:
                self.add_widget(intro_overlay)
            intro_overlay.opacity = 1
            intro_overlay.disabled = False
            self.bring_start_buttons_to_front()
        else:
            intro_overlay.opacity = 0
            intro_overlay.disabled = True
            if intro_overlay.parent is not None:
                self.remove_widget(intro_overlay)
                self.bring_start_buttons_to_front()

    def _on_window_resize(self, *_):
        self.size = Window.size
        self._update_scale()

    def _update_scale(self, *_):
        base_w = self.base_width or 3840.0
        base_h = self.base_height or 2160.0
        width = self.width or Window.width
        height = self.height or Window.height

        base_scale = min(width / base_w, height / base_h)
        self.scale = self.SCALE_FACTOR * base_scale
        self.horizontal_offset = 0.05 if width < 2500 else 0.08

        # --- Responsive Margin berechnen ---
        frac_px = float(width) * float(self.side_margin_frac)

        # physikalischer Zielwert → px
        dpi = getattr(Window, "dpi", 96.0) or 96.0
        px_per_cm = dpi / 2.54
        cm_px = float(self.side_margin_target_cm) * px_per_cm

        # harte Klammern: mindestens min_px bzw. cm, maximal Anteil der Breite
        min_px = max(float(self.side_margin_min_px), cm_px)
        max_px = float(width) * float(self.side_margin_max_frac)

        # finale Margin
        self.horizontal_margin_px = max(min(frac_px, max_px), min_px)

    @staticmethod
    def _parse_value(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip().replace(',', '.')
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    def _cards_for_role(self, role: int):
        if role not in (1, 2):
            return None
        plan_info = self.get_current_plan()
        if plan_info:
            _, plan = plan_info
            cards = plan.get(f'vp{role}') if plan else None
            if cards and len(cards) == 2 and not any(card is None for card in cards):
                return tuple(cards)
        # Fallback über die sichtbaren Karten
        player = self.physical_by_role.get(role)
        if player == 1:
            inner_widget = self.wid_safe('p1_inner')
            outer_widget = self.wid_safe('p1_outer')
        elif player == 2:
            inner_widget = self.wid_safe('p2_inner')
            outer_widget = self.wid_safe('p2_outer')
        else:
            return None
        if inner_widget is None or outer_widget is None:
            return None
        inner_val = self.card_value_from_path(inner_widget.front_image)
        outer_val = self.card_value_from_path(outer_widget.front_image)
        if inner_val is None or outer_val is None:
            return None
        return (inner_val, outer_val)

    def get_hand_total_for_role(self, role: int):
        cards = self._cards_for_role(role)
        if not cards:
            return None
        return sum(cards)

    def get_hand_value_for_role(self, role: int):
        total = self.get_hand_total_for_role(role)
        if total is None:
            return None
        return 0 if total in (20, 21, 22) else total

    def get_hand_value_for_player(self, player: int):
        role = self.role_by_physical.get(player)
        value = self.get_hand_value_for_role(role)
        if value is not None:
            return value
        total = self.get_hand_total_for_player(player)
        if total is None:
            return None
        return 0 if total in (20, 21, 22) else total

    def get_hand_total_for_player(self, player: int):
        role = self.role_by_physical.get(player)
        return self.get_hand_total_for_role(role) if role in (1, 2) else None

    def signal_level_from_value(self, value):
        parsed = self._parse_value(value)
        if parsed is None:
            return None
        if parsed <= 0:
            return None
        if parsed in (20, 21, 22):
            return None
        if parsed == 19:
            return 'high'
        if parsed in (16, 17, 18):
            return 'mid'
        if parsed in (14, 15):
            return 'low'
        if parsed > 22:
            return None
        if parsed >= 16:
            return 'mid'
        return 'low'

    def set_cards_from_plan(self, plan):
        if plan:
            vp1_cards = plan['vp1']
            vp2_cards = plan['vp2']
            first_vp1, second_vp1 = vp1_cards[0], vp1_cards[1]
            first_vp2, second_vp2 = vp2_cards[0], vp2_cards[1]
            p1_inner = self.wid_safe('p1_inner')
            if p1_inner is not None:
                p1_inner.set_front(value_to_card_path(first_vp1))
            p1_outer = self.wid_safe('p1_outer')
            if p1_outer is not None:
                p1_outer.set_front(value_to_card_path(second_vp1))
            p2_inner = self.wid_safe('p2_inner')
            if p2_inner is not None:
                p2_inner.set_front(value_to_card_path(first_vp2))
            p2_outer = self.wid_safe('p2_outer')
            if p2_outer is not None:
                p2_outer.set_front(value_to_card_path(second_vp2))
        else:
            default = ASSETS['cards']['back']
            for card_id in ('p1_inner', 'p1_outer', 'p2_inner', 'p2_outer'):
                widget = self.wid_safe(card_id)
                if widget is not None:
                    widget.set_front(default)

    def compute_global_round(self):
        return self.controller.compute_global_round()

    def score_line_text(self):
        if self.score_state:
            return f'Spielstand – VP1: {self.score_state[1]} | VP2: {self.score_state[2]}'
        return 'Spielstand – VP1: - | VP2: -'

    def get_current_plan(self):
        return self.controller.get_current_plan()

    def peek_next_round_info(self):
        """Ermittelt Metadaten zur nächsten Runde ohne den Status zu verändern."""
        return self.controller.peek_next_round_info()

    def advance_round_pointer(self):
        self.controller.advance_round_pointer()

    # --- Logik
    def apply_phase(self):
        phase_state = self.controller.apply_phase()
        for card_id in ('p1_outer', 'p1_inner', 'p2_outer', 'p2_inner'):
            widget = self.wid_safe(card_id)
            if widget is not None:
                widget.set_live(False)
        for buttons in self.signal_buttons.values():
            for btn_id in buttons.values():
                btn = self.wid_safe(btn_id)
                if btn is not None:
                    btn.set_live(False)
                    btn.disabled = True
        for buttons in self.decision_buttons.values():
            for btn_id in buttons.values():
                btn = self.wid_safe(btn_id)
                if btn is not None:
                    btn.set_live(False)
                    btn.disabled = True

        if not phase_state.show_showdown:
            self.refresh_center_cards(reveal=False)

        start_active = phase_state.start_active
        if self.fixation_running:
            start_active = False
        ready = phase_state.ready
        btn_start_p1 = self.wid_safe('btn_start_p1')
        btn_start_p2 = self.wid_safe('btn_start_p2')
        if btn_start_p1 is not None:
            btn_start_p1.set_live(start_active and ready)
        if btn_start_p2 is not None:
            btn_start_p2.set_live(start_active and ready)

        if ready:
            for player, cards in phase_state.active_cards.items():
                for which in cards:
                    widget = self.card_widget_for_player(player, which)
                    if widget is not None:
                        widget.set_live(True)
            for player, levels in phase_state.active_signal_buttons.items():
                for level in levels:
                    btn_id = self.signal_buttons.get(player, {}).get(level)
                    btn = self.wid_safe(btn_id) if btn_id else None
                    if btn is not None:
                        btn.set_live(True)
                        btn.disabled = False
            for player, decisions in phase_state.active_decision_buttons.items():
                for decision in decisions:
                    btn_id = self.decision_buttons.get(player, {}).get(decision)
                    btn = self.wid_safe(btn_id) if btn_id else None
                    if btn is not None:
                        btn.set_live(True)
                        btn.disabled = False

        if phase_state.show_showdown:
            if btn_start_p1 is not None:
                btn_start_p1.set_live(True)
            if btn_start_p2 is not None:
                btn_start_p2.set_live(True)
            self.update_showdown()

        round_badge = self.wid_safe('round_badge')
        if round_badge is not None:
            round_badge.text = ''
        self.update_user_displays()
        self.update_pause_overlay()

    def continue_after_start_press(self):
        result = self.controller.continue_after_start_press()
        if result.blocked:
            return
        if result.intro_deactivated:
            self.update_user_displays()
            self.update_intro_overlay()

        def proceed():
            self.log_round_start_if_pending()
            self.apply_phase()

        if result.requires_fixation and not self.fixation_running:
            self.run_fixation_sequence(proceed)
        else:
            proceed()

    def start_pressed(self, who:int):
        if self.session_finished:
            return
        if self.phase not in (UXPhase.WAIT_BOTH_START, UXPhase.SHOWDOWN):
            return
        if who == 1:
            self.p1_pressed = True
        else:
            self.p2_pressed = True
        self.record_action(who, 'Play gedrückt')
        if self.session_configured:
            action = 'start_click' if self.phase == UXPhase.WAIT_BOTH_START else 'next_round_click'
            self.log_event(who, action)
        if self.p1_pressed and self.p2_pressed:
            # in nächste Phase
            self.p1_pressed = False
            self.p2_pressed = False
            if self.in_block_pause:
                self.in_block_pause = False
                self.pause_message = ''
                self.setup_round()
                if self.session_finished:
                    self.apply_phase()
                    return
                self.phase = UXPhase.WAIT_BOTH_START
                self.apply_phase()
                self.continue_after_start_press()
            elif self.phase == UXPhase.SHOWDOWN:
                self.prepare_next_round(start_immediately=True)
            else:
                self.continue_after_start_press()

    def run_fixation_sequence(self, on_complete=None):
        self.fixation_runner(
            self,
            schedule_once=Clock.schedule_once,
            stop_image=FIX_STOP_IMAGE,
            live_image=FIX_LIVE_IMAGE,
            on_complete=on_complete,
        )

    def play_fixation_tone(self):
        self.fixation_player(self)

    def tap_card(self, who:int, which:str):
        result = self.controller.tap_card(who, which)
        if not result.allowed:
            return
        widget = self.card_widget_for_player(who, which)
        if widget is None:
            return
        widget.flip()
        if result.record_text:
            self.record_action(who, result.record_text)
        if result.log_action:
            self.log_event(who, result.log_action, result.log_payload or {})
        if result.next_phase:
            Clock.schedule_once(lambda *_: self.goto(result.next_phase), 0.2)

    def pick_signal(self, player:int, level:str):
        result = self.controller.pick_signal(player, level)
        if not result.accepted:
            return
        for lvl, btn_id in self.signal_buttons.get(player, {}).items():
            btn = self.wid_safe(btn_id)
            if btn is None:
                continue
            if lvl == level:
                btn.set_pressed_state()
            else:
                btn.set_live(False)
                btn.disabled = True
        self.record_action(player, f'Signal gewählt: {self.describe_level(level)}')
        if result.log_payload:
            self.log_event(player, 'signal_choice', result.log_payload)
        self.update_user_displays()
        if result.next_phase:
            Clock.schedule_once(lambda *_: self.goto(result.next_phase), 0.2)
            self.update_user_displays()

    def pick_decision(self, player:int, decision:str):
        result = self.controller.pick_decision(player, decision)
        if not result.accepted:
            return
        for choice, btn_id in self.decision_buttons.get(player, {}).items():
            btn = self.wid_safe(btn_id)
            if btn is None:
                continue
            if choice == decision:
                btn.set_pressed_state()
            else:
                btn.set_live(False)
                btn.disabled = True
        self.record_action(player, f'Entscheidung: {decision.upper()}')
        if result.log_payload:
            self.log_event(player, 'call_choice', result.log_payload)
        self.update_user_displays()
        if result.next_phase:
            Clock.schedule_once(lambda *_: self.goto(result.next_phase), 0.2)
            self.update_user_displays()

    def goto(self, phase):
        self.phase = phase
        self.apply_phase()

    def prepare_next_round(self, start_immediately: bool = False):
        result = self.controller.prepare_next_round(start_immediately=start_immediately)
        self.update_role_assignments()
        self._apply_round_setup(result.setup)
        self.apply_phase()
        if result.session_finished:
            self.update_user_displays()
            return
        if result.in_block_pause:
            return

        def proceed():
            if result.start_phase:
                self.phase = result.start_phase
            self.log_round_start_if_pending()
            self.apply_phase()

        if result.requires_fixation and not self.fixation_running:
            self.run_fixation_sequence(proceed)
        elif start_immediately:
            proceed()

    def setup_round(self):
        result = self.controller.setup_round()
        self._apply_round_setup(result)

    def _apply_round_setup(self, result):
        plan = result.plan if result else None
        self.set_cards_from_plan(plan)
        for card_id in ('p1_inner', 'p1_outer', 'p2_inner', 'p2_outer'):
            widget = self.wid_safe(card_id)
            if widget is not None:
                widget.reset()
        for buttons in self.signal_buttons.values():
            for btn_id in buttons.values():
                btn = self.wid_safe(btn_id)
                if btn is not None:
                    btn.reset()
        for buttons in self.decision_buttons.values():
            for btn_id in buttons.values():
                btn = self.wid_safe(btn_id)
                if btn is not None:
                    btn.reset()
        self.status_lines = {1: [], 2: []}
        self.update_status_label(1)
        self.update_status_label(2)
        self.refresh_center_cards(reveal=False)
        self.update_user_displays()

    def refresh_center_cards(self, reveal: bool):
        if reveal:
            p1_inner = self.wid_safe('p1_inner')
            p1_outer = self.wid_safe('p1_outer')
            p2_inner = self.wid_safe('p2_inner')
            p2_outer = self.wid_safe('p2_outer')
            sources = {
                1: [
                    p1_inner.front_image if p1_inner is not None else None,
                    p1_outer.front_image if p1_outer is not None else None,
                ],
                2: [
                    p2_inner.front_image if p2_inner is not None else None,
                    p2_outer.front_image if p2_outer is not None else None,
                ],
            }
        else:
            back = ASSETS['cards']['back']
            sources = {1: [back, back], 2: [back, back]}

        for player, imgs in self.center_cards.items():
            for idx, img_id in enumerate(imgs):
                img_widget = self.wid_safe(img_id)
                if img_widget is not None:
                    img_widget.source = sources[player][idx]
                    img_widget.opacity = 1

    def update_showdown(self):
        # Karten in der Mitte anzeigen
        self.refresh_center_cards(reveal=True)
        outcome = self.compute_outcome()
        if (
            self.current_round_has_stake
            and self.score_state
            and not self.outcome_score_applied
        ):
            winner = outcome.get('winner') if outcome else None
            if winner in (1, 2):
                winner_role = self.role_by_physical.get(winner)
                if winner_role in (1, 2):
                    self.score_state[winner_role] += POINTS_PER_WIN
                    self.outcome_score_applied = True
        if self.session_configured:
            self.log_event(None, 'showdown', outcome or {})
        self.update_user_displays()

    def card_value_from_path(self, path: str):
        if not path:
            return None
        name = os.path.basename(path)
        digits = ''.join(ch for ch in name if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    def determine_signal_level(self, player: int):
        value = self.get_hand_value_for_player(player)
        return self.signal_level_from_value(value)

    def compute_outcome(self):
        outcome = self.controller.compute_outcome(
            signaler_total=self.get_hand_total_for_player(self.signaler),
            judge_total=self.get_hand_total_for_player(self.judge),
            signaler_value=self.get_hand_value_for_player(self.signaler),
            judge_value=self.get_hand_value_for_player(self.judge),
            level_from_value=self.signal_level_from_value,
        )
        return outcome

    def player_descriptor(self, player: int) -> str:
        role = self.role_by_physical.get(player)
        if role in (1, 2):
            return f'VP {role} – Spieler {player}'
        return f'Spieler {player}'

    def format_signal_choice(self, level: str):
        mapping = {
            'low': 'Tief',
            'mid': 'Mittel',
            'high': 'Hoch',
        }
        return mapping.get(level)

    def format_decision_choice(self, decision: str):
        mapping = {
            'wahr': 'Wahrheit',
            'bluff': 'Bluff',
        }
        return mapping.get(decision)
    def _result_signal_text(self, truthful: bool | None) -> str:
        if truthful is None:
            return 'Signal: -'
        return 'Signal: Wahr' if truthful else 'Signal: Bluff'

    def _result_judge_text(self, judge_ok: bool | None) -> str:
        if judge_ok is None:
            return 'Urteil: -'
        return 'Urteil: korrekt' if judge_ok else 'Urteil: inkorrekt'

    def _outcome_statement(self, truthful: bool | None, judge_choice: str | None) -> str:
        if truthful is None or judge_choice not in ('wahr', 'bluff'):
            return ''
        key = ('wahr' if truthful else 'bluff', judge_choice)
        mapping = {
            ('bluff', 'wahr'): 'Sp2 wurde getäuscht:',
            ('bluff', 'bluff'): 'Sp2 erkennt den Bluff:',
            ('wahr', 'wahr'): 'Showdown:',
            ('wahr', 'bluff'): 'SP1 ist ehrlich:',
        }
        return mapping.get(key, '')

    def _signal_label_german(self, level: str | None):
        return self.format_signal_choice(level) or '-'

    def _urteil_label_german(self, decision: str | None):
        return self.format_decision_choice(decision) or '-'

    def _judge_correct(self, truthful: bool | None, judge_choice: str | None):
        if truthful is None or judge_choice is None:
            return None
        expected = 'wahr' if truthful else 'bluff'
        return (judge_choice == expected)

    def _vp_for_player(self, player:int):
        vp = self.role_by_physical.get(player)
        return vp if vp in (1,2) else None

    def _result_for_vp(self, vp:int):
        """Gewonnen/Verloren/Unentschieden relativ zu VP1/VP2."""
        if not isinstance(self.last_outcome, dict):
            return ''
        winner_player = self.last_outcome.get('winner') if self.last_outcome else None
        if winner_player not in (1,2):
            if self.last_outcome.get('draw'):
                return 'Unentschieden'
            return ''
        winner_vp = self.role_by_physical.get(winner_player)
        if winner_vp == vp:
            return 'Gewonnen'
        return 'Verloren'

    def _result_with_score_for_vp(self, vp:int):
        base = self._result_for_vp(vp)
        if not base:
            return ''
        if base == 'Unentschieden':
            return 'Unentschieden 0'
        if base == 'Gewonnen':
            return f'Gewonnen +{POINTS_PER_WIN}'
        return 'Verloren 0'

    def _points_for_vp(self, vp:int):
        if not self.score_state:
            return None
        return self.score_state.get(vp)

    def format_user_display_text(self, vp:int):
        """Erzeugt den Text fürs Display gemäß Block (1/3 vs. 2/4)."""
        if self.intro_active:
            return ''
        # Runde im Block / total (Blockgröße variabel, Übung ohne Logging)
        total_rounds = max(1, self.current_block_total_rounds or 16)
        rnd_in_block = self.round_in_block or 1
        rnd_display = min(max(1, rnd_in_block), total_rounds)
        block_suffix = ' (Übung)' if self.is_practice_block_active() else ''
        header_round = f'Runde {rnd_display}/{total_rounds}{block_suffix}'

        # Zuordnung VP -> Spieler
        player = self.physical_by_role.get(vp)
        role_number = self.player_roles.get(player) if player in (1, 2) else None
        if role_number in (1, 2):
            header_role = f'VP {vp}: Spieler {role_number}'
        elif player in (1, 2):
            header_role = f'VP {vp}: Spieler {player}'
        else:
            header_role = f'VP {vp}'

        # Block-Logik
        block_idx = self.current_block_info['index'] if self.current_block_info else None
        with_points = bool(self.current_round_has_stake) and block_idx in (2,4)

        # Signal & Urteil (global – beziehen sich auf aktuelle Runde)
        signal_choice = self.last_outcome.get('signal_choice') if self.last_outcome else self.player_signals.get(self.signaler)
        judge_choice = self.last_outcome.get('judge_choice') if self.last_outcome else self.player_decisions.get(self.judge)

        truthful = self.last_outcome.get('truthful') if self.last_outcome else None
        judge_ok = self._judge_correct(truthful, judge_choice)

        signal_line = f"Signal: {self._signal_label_german(signal_choice)}"
        urteil_line = f"Urteil: {self._urteil_label_german(judge_choice)}"
        ergebnis_signal = self._result_signal_text(truthful)
        ergebnis_urteil = self._result_judge_text(judge_ok)
        outcome_statement = self._outcome_statement(truthful, judge_choice)

        if with_points:
            points = self._points_for_vp(vp)
            punkte = f' | Punkte: {points}' if points is not None else ''
            header = f'{header_round} | {header_role}{punkte}'
            result_line = self._result_with_score_for_vp(vp)
        else:
            header = f'{header_round} | {header_role}'
            result_line = self._result_for_vp(vp)

        column_width = 34

        def pad_column(text: str) -> str:
            padded = f"{text:<{column_width}}"
            return padded.replace(' ', '\u00A0')

        header_row = f"[b]{pad_column('Züge')}[/b][b]Ergebnis[/b]"
        move_rows = [
            f"{pad_column(signal_line)}{ergebnis_signal}",
            f"{pad_column(urteil_line)}{ergebnis_urteil}",
        ]

        lines = [
            f"[b]{header}[/b]",
            '',
            header_row,
            *move_rows,
        ]
        if outcome_statement:
            lines.extend(['', outcome_statement])
        if result_line and result_line.strip():
            lines.append(f"[b]{result_line}[/b]")

        # Mehrzeilig – leichte Abstände über \n
        return "\n".join(lines)

    def update_user_displays(self):
        """Setzt die Texte in den beiden Displays (unten=VP1, oben=VP2)."""
        for vp, display_id in self.user_displays.items():
            display = self.wid_safe(display_id)
            if display is not None:
                display.text = self.format_user_display_text(vp)

    def update_pause_overlay(self):
        pause_cover = self.wid_safe('pause_cover')
        if pause_cover is None:
            return
        active = (self.in_block_pause or self.session_finished) and bool(self.pause_message)
        if active:
            if pause_cover.parent is None:
                self.add_widget(pause_cover)
                # Start-Buttons über das Overlay legen
                self.bring_start_buttons_to_front()
            pause_cover.opacity = 1
            pause_cover.disabled = False
            for label_id in self.pause_labels.values():
                lbl = self.wid_safe(label_id)
                if lbl is not None:
                    lbl.text = self.pause_message
        else:
            pause_cover.opacity = 0
            pause_cover.disabled = True
            for label_id in self.pause_labels.values():
                lbl = self.wid_safe(label_id)
                if lbl is not None:
                    lbl.text = ''
            if pause_cover.parent is not None:
                self.remove_widget(pause_cover)
                # Reihenfolge der Buttons erhalten
                self.bring_start_buttons_to_front()



    def describe_level(self, level:str) -> str:
        return self.format_signal_choice(level) or (level or '-')

    def is_practice_block_active(self) -> bool:
        return bool(self.current_block_info and self.current_block_info.get('practice'))

    def choice_labels_for_vp(self, vp: int):
        physical = self.physical_by_role.get(vp)
        if physical not in (1, 2):
            return (None, None)
        if physical == self.signaler:
            return 'Eigenes Signal', 'Anderes Urteil'
        if physical == self.judge:
            return 'Eigenes Urteil', 'Anderes Signal'
        return (None, None)

    def update_role_assignments(self):
        """Stelle sicher, dass die Versuchspersonen fest ihren Sitzplätzen zugeordnet bleiben."""
        # Die Sitzordnung ist fix: Spieler 1 unten = VP1, Spieler 2 oben = VP2.
        # Rollenwechsel (Signaler/Judge) wird separat über self.signaler/self.judge abgebildet.
        self.role_by_physical = self._fixed_role_mapping.copy()
        self.physical_by_role = {role: player for player, role in self.role_by_physical.items()}

    def update_turn_order(self):
        self.controller.update_turn_order()

    def phase_for_player(self, player: int, which: str):
        return self.controller.phase_for_player(player, which)

    def card_widget_for_player(self, player: int, which: str):
        if player == 1:
            if which == 'inner':
                return self.wid_safe('p1_inner')
            if which == 'outer':
                return self.wid_safe('p1_outer')
        elif player == 2:
            if which == 'inner':
                return self.wid_safe('p2_inner')
            if which == 'outer':
                return self.wid_safe('p2_outer')
        return None

    def current_engine_phase(self):
        return to_engine_phase(self.phase)

    def log_event(self, player: int, action: str, payload=None):
        if self.is_practice_block_active() and action not in ('session_start',):
            return
        if not self.logger or not self.session_configured:
            return
        payload = payload or {}
        if player is None:
            actor = 'SYS'
        else:
            role = self.player_roles.get(player)
            if role == 1:
                actor = 'P1'
            elif role == 2:
                actor = 'P2'
            else:
                actor = 'P1' if player == 1 else 'P2'
        round_idx = max(0, self.round - 1)
        self.logger.log(
            round_idx,
            self.current_engine_phase(),
            actor,
            action,
            payload
        )
        write_round_log(self, actor, action, payload, player)

    def prompt_session_number(self):
        if self.session_popup:
            return

        content = BoxLayout(orientation='vertical', spacing=12, padding=12)

        header = Label(text='Bitte Session ID eingeben:', size_hint_y=None, height='32dp')
        session_input = TextInput(
            hint_text='Session ID',
            multiline=False,
            size_hint_y=None,
            height='40dp'
        )

        row1 = BoxLayout(size_hint_y=None, height='40dp', spacing=8)
        row1.add_widget(Label(text='Aruco-Overlay aktivieren'))
        overlay_switch = Switch(active=self.aruco_enabled)
        row1.add_widget(overlay_switch)

        row2 = BoxLayout(size_hint_y=None, height='40dp', spacing=8)
        row2.add_widget(Label(text='Startblock (1=Übung, 2–5=Experimental)'))
        block_spinner = Spinner(
            text=str(self.start_block),
            values=[str(i) for i in range(1, 6)],
            size_hint=(None, None),
            size=('120dp', '40dp')
        )
        row2.add_widget(block_spinner)

        error_label = Label(text='', color=(1, 0, 0, 1), size_hint_y=None, height='24dp')

        buttons = BoxLayout(size_hint_y=None, height='44dp', spacing=8)
        ok_button = Button(text='OK')
        cancel_button = Button(text='Abbrechen')
        buttons.add_widget(ok_button)
        buttons.add_widget(cancel_button)

        content.add_widget(header)
        content.add_widget(session_input)
        content.add_widget(row1)
        content.add_widget(row2)
        content.add_widget(error_label)
        content.add_widget(buttons)

        popup = Popup(
            title='Session starten',
            content=content,
            size_hint=(0.6, 0.6),
            auto_dismiss=False
        )
        self.session_popup = popup

        def _on_ok(_btn):
            session_text = session_input.text.strip()
            if not session_text:
                error_label.text = 'Bitte Session ID eingeben.'
                return

            self.session_id = session_text
            digits = ''.join(ch for ch in session_text if ch.isdigit())
            self.session_number = int(digits) if digits else None
            self.aruco_enabled = bool(overlay_switch.active)
            try:
                self.start_block = int(block_spinner.text)
            except Exception:
                self.start_block = 1

            safe_session_id = ''.join(
                ch if ch.isalnum() or ch in ('-', '_') else '_'
                for ch in self.session_id
            ) or 'session'

            self.session_configured = True
            self.log_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.log_dir / f'events_{safe_session_id}.sqlite3'
            self.session_storage_id = safe_session_id
            self.logger = self.events_factory(self.session_id, str(db_path))
            init_round_log(self)
            self.update_role_assignments()

            popup.dismiss()
            self.session_popup = None

            self.log_event(
                None,
                'session_start',
                {
                    'session_number': self.session_number,
                    'session_id': self.session_id,
                    'aruco_enabled': self.aruco_enabled,
                    'start_block': self.start_block,
                },
            )
            self._apply_session_options_and_start()

        def _on_cancel(_btn):
            popup.dismiss()
            self.session_popup = None

        ok_button.bind(on_release=_on_ok)
        cancel_button.bind(on_release=_on_cancel)
        popup.open()

    def _start_overlay_with_path(self, process: Optional[Any]) -> Optional[Any]:
        """Start the ArUco overlay process with the relocated script path."""

        try:
            return self.start_overlay(process, overlay_path=ARUCO_OVERLAY_PATH)
        except TypeError:
            return self.start_overlay(process)

    def _apply_session_options_and_start(self):
        if self._aruco_proc is None and getattr(self, 'overlay_process', None):
            self._aruco_proc = self.overlay_process

        if self.aruco_enabled:
            self._aruco_proc = self._start_overlay_with_path(self._aruco_proc)
        else:
            self._aruco_proc = self.stop_overlay(self._aruco_proc)
        self.overlay_process = self._aruco_proc

        if not self._blocks:
            self._blocks = load_blocks()

        available_blocks = list(self._blocks) if self._blocks else []
        self._blocks = available_blocks
        if not available_blocks:
            self.blocks = []
            self.apply_phase()
            return

        start_index = max(0, min(len(available_blocks) - 1, self.start_block - 1))
        selected_blocks = available_blocks[start_index:]
        if not selected_blocks:
            selected_blocks = available_blocks[-1:]

        self.blocks = selected_blocks
        self.current_block_idx = 0
        self.current_round_idx = 0
        self.current_block_info = None
        self.round_in_block = 0
        self.current_block_total_rounds = 0
        self.session_finished = False
        self.in_block_pause = False
        self.pause_message = ''
        self.next_block_preview = None
        self.fixation_required = False
        self.pending_round_start_log = False
        self.round = 1
        self.outcome_score_applied = False
        self.score_state = None
        self.score_state_block = None
        self.score_state_round_start = None
        self.phase = UXPhase.WAIT_BOTH_START
        self.intro_active = True
        self.p1_pressed = False
        self.p2_pressed = False

        self.total_rounds_planned = sum(
            len(block.get('rounds') or []) for block in self.blocks
        )

        self.reset_ui_for_new_block()

    def reset_ui_for_new_block(self):
        self.setup_round()
        self.apply_phase()
        self.update_user_displays()
        self.update_intro_overlay()

    def log_round_start(self):
        if not self.session_configured:
            return
        self.log_event(None, 'round_start', {
            'round': self.round,
            'block': self.current_block_info['index'] if self.current_block_info else None,
            'round_in_block': self.round_in_block if self.current_block_info else None,
            'payout': bool(self.current_round_has_stake),
            'signaler': self.signaler,
            'judge': self.judge,
            'vp_roles': self.role_by_physical.copy(),
            'player_roles': self.player_roles.copy(),
        })
        self.pending_round_start_log = False

    def log_round_start_if_pending(self):
        if self.pending_round_start_log:
            self.log_round_start()

    def record_action(self, player:int, text:str):
        self.status_lines[player].append(text)
        self.update_status_label(player)

    def update_status_label(self, player:int):
        label = self.status_labels.get(player)
        if label is None:
            return
        role = 'Signal' if self.signaler == player else 'Judge'
        header = [f"Du bist Spieler {player}", f"Rolle: {role}"]
        body = self.status_lines[player]
        self.status_labels[player].text = "\n".join(header + body)


__all__ = ["TabletopRoot"]
