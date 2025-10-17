# tabletop_ux_kivy.py
# -------------------------------------------------------------
# Fertiges UX-Skript für deine Masterarbeit
# - 43" Tisch-Display, 3840x2160 (4K UHD)
# - Vollbild, Hintergrundfarbe #BFBFBF
# - Ordnerstruktur exakt wie in deinen Screenshots:
#     ./UX/  -> play_*.png, hoch_*.png, mittel_*.png, tief_*.png, bluff_*.png, wahr_*.png
#     ./Karten/ -> back.png, back_stop.png  (Kartenwerte optional)
# - Ablauf exakt nach Beschreibung inkl. Rollenwechsel pro Runde
# - Beide Karten pro Spieler werden im Ablauf aufgedeckt (erst innere, dann äußere)
# - Buttons sind nur „live“, wenn *_live.png verwendet wird; sonst *_stop.png
# - Keine Anpassungen nötig, einfach `python tabletop_ux_kivy.py` starten.
# -------------------------------------------------------------

from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
import os
import csv
import itertools
from pathlib import Path
from datetime import datetime

from game_engine import EventLogger, Phase as EnginePhase

# --- Display fest auf 3840x2160, Vollbild aktivierbar (kommentiere die nächste Zeile, falls du Fenster willst)
Config.set('graphics', 'fullscreen', 'auto')

# --- Konstanten & Assets
ROOT = os.path.dirname(os.path.abspath(__file__))
UX_DIR = os.path.join(ROOT, 'UX')
CARD_DIR = os.path.join(ROOT, 'Karten')

ASSETS = {
    'play': {
        'live':  os.path.join(UX_DIR, 'play_live.png'),
        'stop':  os.path.join(UX_DIR, 'play_stop.png'),
    },
    'signal': {
        'low':   {'live': os.path.join(UX_DIR, 'tief_live.png'),   'stop': os.path.join(UX_DIR, 'tief_stop.png')},
        'mid':   {'live': os.path.join(UX_DIR, 'mittel_live.png'), 'stop': os.path.join(UX_DIR, 'mittel_stop.png')},
        'high':  {'live': os.path.join(UX_DIR, 'hoch_live.png'),   'stop': os.path.join(UX_DIR, 'hoch_stop.png')},
    },
    'decide': {
        'bluff': {'live': os.path.join(UX_DIR, 'bluff_live.png'),  'stop': os.path.join(UX_DIR, 'bluff_stop.png')},
        'wahr':  {'live': os.path.join(UX_DIR, 'wahr_live.png'),   'stop': os.path.join(UX_DIR, 'wahr_stop.png')},
    },
    'cards': {
        'back':      os.path.join(CARD_DIR, 'back.png'),
        'back_stop': os.path.join(CARD_DIR, 'back_stop.png'),
    }
}

# --- Phasen der Runde
PH_WAIT_BOTH_START = 'WAIT_BOTH_START'
PH_P1_INNER = 'P1_INNER'
PH_P2_INNER = 'P2_INNER'
PH_P1_OUTER = 'P1_OUTER'
PH_P2_OUTER = 'P2_OUTER'
PH_SIGNALER = 'SIGNALER'
PH_JUDGE = 'JUDGE'
PH_SHOWDOWN = 'SHOWDOWN'

class CardWidget(Button):
    """Karten-Slot: zeigt back_stop bis aktiv und/oder aufgedeckt."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.live = False
        self.face_up = False
        self.front_image = ASSETS['cards']['back']
        self.border = (0, 0, 0, 0)
        self.background_normal = ASSETS['cards']['back_stop']
        self.background_down = ASSETS['cards']['back_stop']
        self.background_disabled_normal = ASSETS['cards']['back_stop']
        self.background_disabled_down = ASSETS['cards']['back_stop']
        self.disabled_color = (1,1,1,1)
        self.update_visual()

    def set_live(self, v: bool):
        self.live = v
        self.disabled = not v
        self.update_visual()

    def flip(self):
        if not self.live:
            return
        self.face_up = True
        self.set_live(False)

    def reset(self):
        self.live = False
        self.face_up = False
        self.disabled = True
        self.update_visual()

    def set_front(self, img_path: str):
        self.front_image = img_path
        if not os.path.exists(img_path):
            self.front_image = ASSETS['cards']['back']
        self.update_visual()

    def update_visual(self):
        if self.face_up:
            img = self.front_image
        elif self.live:
            img = ASSETS['cards']['back']
        else:
            img = ASSETS['cards']['back_stop']
        self.background_normal = img
        self.background_down = img
        self.background_disabled_normal = img
        self.background_disabled_down = img
        self.opacity = 1.0 if (self.live or self.face_up) else 0.55

class IconButton(Button):
    """Button, der automatisch live/stop-Grafiken nutzt."""
    def __init__(self, asset_pair: dict, label_text: str = '', **kw):
        super().__init__(**kw)
        self.asset_pair = asset_pair
        self.live = False
        self.selected = False
        self.border = (0, 0, 0, 0)
        self.background_normal = asset_pair['stop']
        self.background_down = asset_pair['stop']
        self.background_disabled_normal = asset_pair['stop']
        self.disabled_color = (1,1,1,1)
        self.text = ''  # wir nutzen die Grafik
        self.rotation_angle = 0
        with self.canvas.before:
            self._push_matrix = PushMatrix()
            self._rotation = Rotate(angle=0, origin=self.center)
        with self.canvas.after:
            self._pop_matrix = PopMatrix()
        self.bind(pos=self._update_transform, size=self._update_transform)
        self.update_visual()

    def set_live(self, v: bool):
        self.live = v
        self.disabled = not v
        self.update_visual()

    def set_pressed_state(self):
        # nach Auswahl bleibt die live-Grafik sichtbar, ohne dass der Button live bleibt
        self.selected = True
        self.live = False
        self.disabled = True
        self.update_visual()

    def reset(self):
        self.selected = False
        self.live = False
        self.disabled = True
        self.update_visual()

    def set_rotation(self, angle: float):
        self.rotation_angle = angle
        self._update_transform()

    def _update_transform(self, *args):
        if hasattr(self, '_rotation'):
            self._rotation.origin = self.center
            self._rotation.angle = self.rotation_angle

    def update_visual(self):
        img = self.asset_pair['live'] if (self.live or self.selected) else self.asset_pair['stop']
        self.background_normal = img
        self.background_down = img
        self.background_disabled_normal = img
        self.background_disabled_down = img
        self.opacity = 1.0 if (self.live or self.selected) else 0.6


class RotatedLabel(Label):
    """Label mit Rotationsunterstützung (für gespiegelte Anzeige)."""
    def __init__(self, angle: float = 0, **kwargs):
        self.rotation_angle = angle
        super().__init__(**kwargs)
        with self.canvas.before:
            self._push_matrix = PushMatrix()
            self._rotation = Rotate(angle=self.rotation_angle, origin=self.center)
        with self.canvas.after:
            self._pop_matrix = PopMatrix()
        self.bind(pos=self._update_transform, size=self._update_transform)

    def set_rotation(self, angle: float):
        self.rotation_angle = angle
        self._update_transform()

    def _update_transform(self, *args):
        if hasattr(self, '_rotation'):
            self._rotation.origin = self.center
            self._rotation.angle = self.rotation_angle

class TabletopRoot(FloatLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        with self.canvas.before:
            Color(0.75, 0.75, 0.75, 1)  # #BFBFBF
            self.bg = Rectangle(pos=(0,0), size=Window.size)
        Window.bind(on_resize=self.on_resize)

        self.round = 1
        self.signaler = 1
        self.judge = 2
        self.phase = PH_WAIT_BOTH_START
        self.role_by_physical = {1: 1, 2: 2}
        self.physical_by_role = {1: 1, 2: 2}
        self.session_number = None
        self.session_id = None
        self.logger = None
        self.log_dir = Path(ROOT) / 'logs'
        self.session_popup = None
        self.session_configured = False
        self.round_log_path = None
        self.round_log_fp = None
        self.round_log_writer = None

        # --- UI Elemente platzieren
        self.make_ui()
        self.setup_round()
        self.apply_phase()
        Clock.schedule_once(lambda *_: self.prompt_session_number(), 0.1)

    # --- Layout & Elemente
    def on_resize(self, *_):
        self.bg.size = Window.size
        self.update_layout()

    def make_ui(self):
        # Start-Buttons links/rechts (für beide Spieler)
        self.btn_start_p1 = IconButton(
            ASSETS['play'],
            size_hint=(None, None),
        )
        self.btn_start_p1.bind(on_release=lambda *_: self.start_pressed(1))
        self.add_widget(self.btn_start_p1)

        self.btn_start_p2 = IconButton(
            ASSETS['play'],
            size_hint=(None, None),
        )
        self.btn_start_p2.bind(on_release=lambda *_: self.start_pressed(2))
        self.add_widget(self.btn_start_p2)

        # Ergebnis-Labels oben/unten (oben gespiegelt)
        self.info_labels = {
            'bottom': RotatedLabel(
                color=(1, 1, 1, 1),
                size_hint=(None, None),
                halign='center',
                valign='middle'
            ),
            'top': RotatedLabel(
                angle=180,
                color=(1, 1, 1, 1),
                size_hint=(None, None),
                halign='center',
                valign='middle'
            )
        }
        for lbl in self.info_labels.values():
            self.add_widget(lbl)

        self.outcome_labels = {
            1: RotatedLabel(
                color=(1, 1, 1, 1),
                size_hint=(None, None),
                halign='center',
                valign='middle'
            ),
            2: RotatedLabel(
                angle=180,
                color=(1, 1, 1, 1),
                size_hint=(None, None),
                halign='center',
                valign='middle'
            )
        }
        for lbl in self.outcome_labels.values():
            self.add_widget(lbl)

        # Spielerzonen (je 2 Karten in den Ecken)
        self.p1_outer = CardWidget(size_hint=(None, None))
        self.p1_outer.bind(on_release=lambda *_: self.tap_card(1, 'outer'))
        self.add_widget(self.p1_outer)

        self.p1_inner = CardWidget(size_hint=(None, None))
        self.p1_inner.bind(on_release=lambda *_: self.tap_card(1, 'inner'))
        self.add_widget(self.p1_inner)

        self.p2_outer = CardWidget(size_hint=(None, None))
        self.p2_outer.bind(on_release=lambda *_: self.tap_card(2, 'outer'))
        self.add_widget(self.p2_outer)

        self.p2_inner = CardWidget(size_hint=(None, None))
        self.p2_inner.bind(on_release=lambda *_: self.tap_card(2, 'inner'))
        self.add_widget(self.p2_inner)

        # Button-Cluster für Signale & Entscheidungen pro Spieler
        self.signal_buttons = {1: {}, 2: {}}
        self.decision_buttons = {1: {}, 2: {}}

        for level in ['low', 'mid', 'high']:
            btn = IconButton(ASSETS['signal'][level], size_hint=(None, None))
            btn.bind(on_release=lambda _, lvl=level: self.pick_signal(1, lvl))
            self.signal_buttons[1][level] = btn
            self.add_widget(btn)

        for choice in ['bluff', 'wahr']:
            btn = IconButton(ASSETS['decide'][choice], size_hint=(None, None))
            btn.bind(on_release=lambda _, ch=choice: self.pick_decision(1, ch))
            self.decision_buttons[1][choice] = btn
            self.add_widget(btn)

        for level in ['low', 'mid', 'high']:
            btn = IconButton(ASSETS['signal'][level], size_hint=(None, None))
            btn.bind(on_release=lambda _, lvl=level: self.pick_signal(2, lvl))
            self.signal_buttons[2][level] = btn
            self.add_widget(btn)

        for choice in ['bluff', 'wahr']:
            btn = IconButton(ASSETS['decide'][choice], size_hint=(None, None))
            btn.bind(on_release=lambda _, ch=choice: self.pick_decision(2, ch))
            self.decision_buttons[2][choice] = btn
            self.add_widget(btn)

        # Showdown-Karten in der Mitte (immer sichtbar, zuerst verdeckt)
        self.center_cards = {
            1: [Image(size_hint=(None, None)), Image(size_hint=(None, None))],
            2: [Image(size_hint=(None, None)), Image(size_hint=(None, None))],
        }
        for imgs in self.center_cards.values():
            for img in imgs:
                img.fit_mode = "contain"
                self.add_widget(img)

        # Rundenbadge unten Mitte
        self.round_badge = Label(
            text='',
            color=(1, 1, 1, 1),
            size_hint=(None, None),
            halign='center',
            valign='middle'
        )
        self.round_badge.opacity = 0
        self.add_widget(self.round_badge)

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

        self.blocks = self.load_blocks()
        self.total_rounds_planned = sum(len(block['rounds']) for block in self.blocks)
        self.current_block_idx = 0
        self.current_round_idx = 0
        self.current_block_info = None
        self.round_in_block = 1
        self.in_block_pause = False
        self.pause_message = ''
        self.session_finished = False
        self.current_round_has_stake = False
        self.score_state = None
        self.score_state_block = None
        self.score_state_round_start = None
        self.outcome_score_applied = False

        self.update_layout()
        self._update_outcome_labels()

    def update_layout(self):
        W, H = Window.size
        base_w, base_h = 3840.0, 2160.0
        scale = min(W / base_w if base_w else 1, H / base_h if base_h else 1)

        self.bg.pos = (0, 0)
        self.bg.size = (W, H)

        corner_margin = 120 * scale
        card_width, card_height = 420 * scale, 640 * scale
        card_gap = 70 * scale
        start_size = (360 * scale, 360 * scale)

        # Start buttons
        self.btn_start_p1.size = start_size
        start_margin = 60 * scale
        self.btn_start_p1.pos = (start_margin, H - start_margin - start_size[1])
        self.btn_start_p1.set_rotation(180)

        self.btn_start_p2.size = start_size
        self.btn_start_p2.pos = (W - start_margin - start_size[0], start_margin)
        self.btn_start_p2.set_rotation(0)

        # Cards positions
        p1_outer_pos = (corner_margin, corner_margin)
        p1_inner_pos = (corner_margin + card_width + card_gap, corner_margin)
        self.p1_outer.size = (card_width, card_height)
        self.p1_outer.pos = p1_outer_pos
        self.p1_inner.size = (card_width, card_height)
        self.p1_inner.pos = p1_inner_pos

        p2_outer_pos = (W - corner_margin - card_width, H - corner_margin - card_height)
        p2_inner_pos = (p2_outer_pos[0] - card_width - card_gap, p2_outer_pos[1])
        self.p2_outer.size = (card_width, card_height)
        self.p2_outer.pos = p2_outer_pos
        self.p2_inner.size = (card_width, card_height)
        self.p2_inner.pos = p2_inner_pos

        # Button stacks
        btn_width, btn_height = 260 * scale, 260 * scale
        vertical_gap = 40 * scale
        horizontal_gap = 60 * scale
        cluster_shift = 620 * scale
        vertical_offset = 140 * scale

        # Player 1 (bottom right)
        signal_x = W - corner_margin - btn_width - cluster_shift
        base_y = corner_margin + vertical_offset
        for idx, level in enumerate(['low', 'mid', 'high']):
            btn = self.signal_buttons[1][level]
            btn.size = (btn_width, btn_height)
            btn.pos = (signal_x, base_y + idx * (btn_height + vertical_gap))
            btn.set_rotation(0)

        decision_x = signal_x - horizontal_gap - btn_width
        for idx, choice in enumerate(['bluff', 'wahr']):
            btn = self.decision_buttons[1][choice]
            btn.size = (btn_width, btn_height)
            btn.pos = (decision_x, base_y + idx * (btn_height + vertical_gap))
            btn.set_rotation(0)

        # Player 2 (top left)
        signal2_x = corner_margin + cluster_shift
        top_y = H - corner_margin - vertical_offset
        for idx, level in enumerate(['low', 'mid', 'high']):
            btn = self.signal_buttons[2][level]
            btn.size = (btn_width, btn_height)
            btn.pos = (signal2_x, top_y - btn_height - idx * (btn_height + vertical_gap))
            btn.set_rotation(180)

        decision2_x = signal2_x + btn_width + horizontal_gap
        for idx, choice in enumerate(['bluff', 'wahr']):
            btn = self.decision_buttons[2][choice]
            btn.size = (btn_width, btn_height)
            btn.pos = (decision2_x, top_y - btn_height - idx * (btn_height + vertical_gap))
            btn.set_rotation(180)

        # Center cards
        center_card_width, center_card_height = 380 * scale, 560 * scale
        center_gap_x = 90 * scale
        center_gap_y = 60 * scale
        left_x = W / 2 - center_card_width - center_gap_x / 2
        right_x = W / 2 + center_gap_x / 2
        bottom_y = H / 2 - center_card_height - center_gap_y / 2
        top_y_center = H / 2 + center_gap_y / 2

        for idx, img in enumerate(self.center_cards[1]):
            img.size = (center_card_width, center_card_height)
        self.center_cards[1][0].pos = (right_x, bottom_y)
        self.center_cards[1][1].pos = (left_x, bottom_y)

        for idx, img in enumerate(self.center_cards[2]):
            img.size = (center_card_width, center_card_height)
        self.center_cards[2][0].pos = (left_x, top_y_center)
        self.center_cards[2][1].pos = (right_x, top_y_center)

        # Info labels
        info_width, info_height = 2000 * scale, 160 * scale
        info_margin = 60 * scale

        bottom_label = self.info_labels['bottom']
        bottom_label.size = (info_width, info_height)
        bottom_label.font_size = 56 * scale if scale else 56
        bottom_label.pos = (W / 2 - info_width / 2, bottom_y - info_height - info_margin)
        bottom_label.text_size = (info_width, info_height)
        bottom_label.set_rotation(0)

        top_label = self.info_labels['top']
        top_label.size = (info_width, info_height)
        top_label.font_size = 56 * scale if scale else 56
        top_label.pos = (W / 2 - info_width / 2, top_y_center + center_card_height + info_margin)
        top_label.text_size = (info_width, info_height)
        top_label.set_rotation(180)

        # Outcome labels per player (above clusters)
        outcome_width = btn_width * 2 + horizontal_gap
        outcome_height = 120 * scale

        bottom_outcome = self.outcome_labels[1]
        bottom_outcome.size = (outcome_width, outcome_height)
        bottom_outcome.font_size = 64 * scale if scale else 64
        bottom_outcome.pos = (decision_x, base_y + 2 * (btn_height + vertical_gap))
        bottom_outcome.text_size = (outcome_width, outcome_height)
        bottom_outcome.set_rotation(0)

        top_outcome = self.outcome_labels[2]
        top_outcome.size = (outcome_width, outcome_height)
        top_outcome.font_size = 64 * scale if scale else 64
        top_outcome.pos = (decision2_x, top_y - 2 * (btn_height + vertical_gap) - outcome_height)
        top_outcome.text_size = (outcome_width, outcome_height)
        top_outcome.set_rotation(180)

        # Round badge
        badge_width, badge_height = 1400 * scale, 70 * scale
        self.round_badge.size = (badge_width, badge_height)
        self.round_badge.font_size = 40 * scale if scale else 40
        self.round_badge.pos = (W / 2 - badge_width / 2, corner_margin / 2)
        self.round_badge.text_size = (badge_width, badge_height)

        # Refresh transforms after layout changes
        for buttons in self.signal_buttons.values():
            for btn in buttons.values():
                btn._update_transform()
        for buttons in self.decision_buttons.values():
            for btn in buttons.values():
                btn._update_transform()
        self.btn_start_p1._update_transform()
        self.btn_start_p2._update_transform()
        for lbl in self.info_labels.values():
            lbl._update_transform()
        for lbl in self.outcome_labels.values():
            lbl._update_transform()

    # --- Datenquellen & Hilfsfunktionen ---
    def load_blocks(self):
        order = [
            (1, 'Paare1.csv', False),
            (2, 'Paare3.csv', True),
            (3, 'Paare2.csv', False),
            (4, 'Paare4.csv', True),
        ]
        blocks = []
        for index, filename, payout in order:
            path = Path(ROOT) / filename
            rounds = self.load_csv_rounds(path)
            blocks.append({
                'index': index,
                'csv': filename,
                'path': path,
                'rounds': rounds,
                'payout': payout,
            })
        return blocks

    def load_csv_rounds(self, path: Path):
        rounds = []
        try:
            with open(path, newline='', encoding='utf-8') as fp:
                rows = list(csv.reader(fp))
        except FileNotFoundError:
            return rounds
        except Exception:
            return rounds

        def parse_cards(row, start, end):
            values = []
            for idx in range(start, min(end, len(row))):
                cell = (row[idx] or '').strip()
                if not cell:
                    continue
                try:
                    # Einige CSVs enthalten Ganzzahlen ohne Dezimalstellen, andere mit.
                    values.append(int(float(cell)))
                except ValueError:
                    continue
                if len(values) == 2:
                    break
            if len(values) < 2:
                raise ValueError('Zu wenige Karten')
            return tuple(values[:2])

        start_idx = 0
        if rows:
            try:
                parse_cards(rows[0], 2, 4)
                parse_cards(rows[0], 7, 9)
            except Exception:
                start_idx = 1

        for row in rows[start_idx:]:
            if not row or all((cell or '').strip() == '' for cell in row):
                continue
            try:
                vp1_cards = parse_cards(row, 2, 4)
                vp2_cards = parse_cards(row, 7, 9)
            except Exception:
                continue
            rounds.append({'vp1': vp1_cards, 'vp2': vp2_cards})
        return rounds

    def value_to_card_path(self, value):
        try:
            number = int(value)
        except (TypeError, ValueError):
            return ASSETS['cards']['back']
        filename = f'{number}.png'
        path = os.path.join(CARD_DIR, filename)
        return path if os.path.exists(path) else ASSETS['cards']['back']

    def set_cards_from_plan(self, plan):
        if plan:
            vp1_cards = plan['vp1']
            vp2_cards = plan['vp2']
            self.p1_inner.set_front(self.value_to_card_path(vp1_cards[0]))
            self.p1_outer.set_front(self.value_to_card_path(vp1_cards[1]))
            self.p2_inner.set_front(self.value_to_card_path(vp2_cards[0]))
            self.p2_outer.set_front(self.value_to_card_path(vp2_cards[1]))
        else:
            default = ASSETS['cards']['back']
            for widget in (self.p1_inner, self.p1_outer, self.p2_inner, self.p2_outer):
                widget.set_front(default)

    def compute_global_round(self):
        if not self.blocks:
            return self.round
        total = 0
        for idx, block in enumerate(self.blocks):
            if idx < self.current_block_idx:
                total += len(block['rounds'])
        if self.current_block_idx >= len(self.blocks):
            return max(1, total)
        return total + self.current_round_idx + 1

    def score_line_text(self):
        if self.score_state:
            return f'Spielstand – VP1: {self.score_state[1]} | VP2: {self.score_state[2]}'
        return 'Spielstand – VP1: - | VP2: -'

    def get_current_plan(self):
        if not self.blocks or self.session_finished or self.in_block_pause:
            return None
        if self.current_block_idx >= len(self.blocks):
            return None
        block = self.blocks[self.current_block_idx]
        rounds = block['rounds']
        if not rounds:
            return None
        if self.current_round_idx >= len(rounds):
            return None
        return block, rounds[self.current_round_idx]

    def peek_next_round_info(self):
        """Ermittelt Metadaten zur nächsten Runde ohne den Status zu verändern."""
        if not self.blocks:
            return None
        if self.current_block_idx >= len(self.blocks):
            return None
        block_idx = self.current_block_idx
        round_idx = self.current_round_idx + 1
        while block_idx < len(self.blocks):
            block = self.blocks[block_idx]
            rounds = block.get('rounds') or []
            if round_idx < len(rounds):
                return {
                    'block': block,
                    'round_index': round_idx,
                    'round_in_block': round_idx + 1,
                }
            block_idx += 1
            round_idx = 0
        return None

    def advance_round_pointer(self):
        if not self.blocks or self.session_finished:
            self.round += 1
            return
        if self.current_block_idx >= len(self.blocks):
            self.session_finished = True
            return
        block = self.blocks[self.current_block_idx]
        self.current_round_idx += 1
        if self.current_round_idx >= len(block['rounds']):
            completed_block = block
            self.current_block_idx += 1
            self.current_round_idx = 0
            if self.current_block_idx >= len(self.blocks):
                self.session_finished = True
                self.in_block_pause = False
                self.pause_message = 'Alle Blöcke abgeschlossen. Vielen Dank!'
            else:
                self.in_block_pause = True
                next_block = self.blocks[self.current_block_idx]
                condition = 'Stake' if next_block['payout'] else 'ohne Stake'
                self.pause_message = (
                    f'Block {completed_block["index"]} beendet.\n'
                    'Pause.\n'
                    f'Weiter mit Block {next_block["index"]} ({condition}).'
                )
        self.round = self.compute_global_round()

    # --- Logik
    def apply_phase(self):
        # Alles zunächst deaktivieren
        for c in (self.p1_outer, self.p1_inner, self.p2_outer, self.p2_inner):
            c.set_live(False)
        for buttons in self.signal_buttons.values():
            for b in buttons.values():
                b.set_live(False)
        for buttons in self.decision_buttons.values():
            for b in buttons.values():
                b.set_live(False)

        # Showdown zurücksetzen
        if self.phase != PH_SHOWDOWN:
            self.refresh_center_cards(reveal=False)

        # Startbuttons
        start_active = (self.phase in (PH_WAIT_BOTH_START, PH_SHOWDOWN))
        ready = self.session_configured and not self.session_finished
        self.btn_start_p1.set_live(start_active and ready)
        self.btn_start_p2.set_live(start_active and ready)

        # Phasen-spezifisch
        if not ready:
            pass
        elif self.phase == PH_P1_INNER:
            self.p1_inner.set_live(True)
        elif self.phase == PH_P2_INNER:
            self.p2_inner.set_live(True)
        elif self.phase == PH_P1_OUTER:
            self.p1_outer.set_live(True)
        elif self.phase == PH_P2_OUTER:
            self.p2_outer.set_live(True)
        elif self.phase == PH_SIGNALER:
            signaler = self.signaler
            for btn in self.signal_buttons[signaler].values():
                btn.set_live(True)
        elif self.phase == PH_JUDGE:
            judge = self.judge
            for btn in self.decision_buttons[judge].values():
                btn.set_live(True)
        elif self.phase == PH_SHOWDOWN:
            self.btn_start_p1.set_live(True)
            self.btn_start_p2.set_live(True)
            self.update_showdown()

        # Badge unten ist deaktiviert
        self.round_badge.text = ''
        self.update_info_labels()

    def start_pressed(self, who:int):
        if self.session_finished:
            return
        if self.phase not in (PH_WAIT_BOTH_START, PH_SHOWDOWN):
            return
        if who == 1:
            self.p1_pressed = True
        else:
            self.p2_pressed = True
        self.record_action(who, 'Play gedrückt')
        if self.session_configured:
            action = 'start_click' if self.phase == PH_WAIT_BOTH_START else 'next_round_click'
            self.log_event(who, action)
        if self.p1_pressed and self.p2_pressed:
            # in nächste Phase
            self.p1_pressed = False
            self.p2_pressed = False
            if self.in_block_pause:
                self.in_block_pause = False
                self.pause_message = ''
                self.setup_round()
                if not self.session_finished:
                    self.phase = PH_P1_INNER
                    self.apply_phase()
            elif self.phase == PH_SHOWDOWN:
                self.prepare_next_round(start_immediately=True)
            else:
                self.phase = PH_P1_INNER
                self.apply_phase()

    def tap_card(self, who:int, which:str):
        # which in {'inner','outer'}
        if who == 1 and which == 'inner' and self.phase == PH_P1_INNER:
            self.p1_inner.flip()
            self.record_action(1, 'Karte innen aufgedeckt')
            self.log_event(1, 'reveal_inner', {'card': 1})
            Clock.schedule_once(lambda *_: self.goto(PH_P2_INNER), 0.2)
        elif who == 2 and which == 'inner' and self.phase == PH_P2_INNER:
            self.p2_inner.flip()
            self.record_action(2, 'Karte innen aufgedeckt')
            self.log_event(2, 'reveal_inner', {'card': 1})
            Clock.schedule_once(lambda *_: self.goto(PH_P1_OUTER), 0.2)
        elif who == 1 and which == 'outer' and self.phase == PH_P1_OUTER:
            self.p1_outer.flip()
            self.record_action(1, 'Karte außen aufgedeckt')
            self.log_event(1, 'reveal_outer', {'card': 2})
            Clock.schedule_once(lambda *_: self.goto(PH_P2_OUTER), 0.2)
        elif who == 2 and which == 'outer' and self.phase == PH_P2_OUTER:
            self.p2_outer.flip()
            self.record_action(2, 'Karte außen aufgedeckt')
            self.log_event(2, 'reveal_outer', {'card': 2})
            Clock.schedule_once(lambda *_: self.goto(PH_SIGNALER), 0.2)

    def pick_signal(self, player:int, level:str):
        if self.phase != PH_SIGNALER or player != self.signaler:
            return
        self.player_signals[player] = level
        # fixiere Auswahl optisch (Button bleibt live)
        for lvl, btn in self.signal_buttons[player].items():
            if lvl == level:
                btn.set_pressed_state()
            else:
                btn.set_live(False)
                btn.disabled = True
        self.record_action(player, f'Signal gewählt: {self.describe_level(level)}')
        self.log_event(player, 'signal_choice', {'level': level})
        self.update_info_labels()
        Clock.schedule_once(lambda *_: self.goto(PH_JUDGE), 0.2)

    def pick_decision(self, player:int, decision:str):
        if self.phase != PH_JUDGE or player != self.judge:
            return
        self.player_decisions[player] = decision
        for choice, btn in self.decision_buttons[player].items():
            if choice == decision:
                btn.set_pressed_state()
            else:
                btn.set_live(False)
                btn.disabled = True
        self.record_action(player, f'Entscheidung: {decision.upper()}')
        self.log_event(player, 'call_choice', {'decision': decision})
        self.update_info_labels()
        Clock.schedule_once(lambda *_: self.goto(PH_SHOWDOWN), 0.2)

    def goto(self, phase):
        self.phase = phase
        self.apply_phase()

    def prepare_next_round(self, start_immediately: bool = False):
        # Rollen tauschen
        self.signaler, self.judge = self.judge, self.signaler
        self.update_role_assignments()
        self.advance_round_pointer()
        self.phase = PH_WAIT_BOTH_START
        self.setup_round()
        if start_immediately and not self.in_block_pause and not self.session_finished:
            self.phase = PH_P1_INNER
        else:
            self.phase = PH_WAIT_BOTH_START
        self.apply_phase()

    def setup_round(self):
        self.outcome_score_applied = False
        # ggf. leere Blöcke überspringen
        if self.blocks and not self.session_finished and not self.in_block_pause:
            while (
                self.current_block_idx < len(self.blocks)
                and not self.blocks[self.current_block_idx]['rounds']
            ):
                self.current_block_idx += 1
        plan_info = self.get_current_plan()
        if plan_info:
            block, plan = plan_info
            self.current_block_info = block
            self.round_in_block = self.current_round_idx + 1
            self.current_round_has_stake = block['payout']
            if block['payout'] and self.score_state_block != block['index']:
                self.score_state = {1: 16, 2: 16}
                self.score_state_block = block['index']
            if not block['payout']:
                self.score_state = None
                self.score_state_block = None
            self.score_state_round_start = (
                self.score_state.copy() if self.score_state else None
            )
            self.set_cards_from_plan(plan)
            self.round = self.compute_global_round()
        else:
            if self.current_block_idx >= len(self.blocks):
                self.session_finished = True
            self.current_block_info = None
            self.round_in_block = 0
            self.current_round_has_stake = False
            self.set_cards_from_plan(None)
            self.round = self.compute_global_round()
            self.score_state_round_start = (
                self.score_state.copy() if self.score_state else None
            )

        for c in (self.p1_inner, self.p1_outer, self.p2_inner, self.p2_outer):
            c.reset()
        # Reset Buttons
        for buttons in self.signal_buttons.values():
            for btn in buttons.values():
                btn.reset()
        for buttons in self.decision_buttons.values():
            for btn in buttons.values():
                btn.reset()
        # Reset Status
        self.player_signals = {1: None, 2: None}
        self.player_decisions = {1: None, 2: None}
        self.status_lines = {1: [], 2: []}
        self.update_status_label(1)
        self.update_status_label(2)
        # Showdown Elements
        self.last_outcome = {
            'winner': None,
            'truthful': None,
            'actual_level': None,
            'signal_choice': None,
            'judge_choice': None,
            'payout': self.current_round_has_stake,
        }
        self.refresh_center_cards(reveal=False)
        self.update_info_labels()
        if plan_info:
            self.log_round_start()

    def refresh_center_cards(self, reveal: bool):
        if reveal:
            sources = {
                1: [self.p1_inner.front_image, self.p1_outer.front_image],
                2: [self.p2_inner.front_image, self.p2_outer.front_image],
            }
        else:
            back = ASSETS['cards']['back']
            sources = {1: [back, back], 2: [back, back]}

        for player, imgs in self.center_cards.items():
            for idx, img in enumerate(imgs):
                img.source = sources[player][idx]
                img.opacity = 1

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
                    loser_role = 1 if winner_role == 2 else 2
                    self.score_state[winner_role] += 1
                    self.score_state[loser_role] -= 1
                    self.outcome_score_applied = True
        self.update_info_labels()
        if self.session_configured:
            self.log_event(None, 'showdown', outcome or {})

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
        if player == 1:
            inner_widget, outer_widget = self.p1_inner, self.p1_outer
        else:
            inner_widget, outer_widget = self.p2_inner, self.p2_outer
        inner_val = self.card_value_from_path(inner_widget.front_image)
        outer_val = self.card_value_from_path(outer_widget.front_image)
        if inner_val is None or outer_val is None:
            return None

        total = inner_val + outer_val
        if total == 19:
            return 'high'
        if total in (16, 17, 18):
            return 'mid'
        if total in (14, 15):
            return 'low'
        if total in (20, 21, 22):       # Wert 0 → eigene „Level“-Marke
            return 'zero'
        return None


    def compute_outcome(self):
        signaler = self.signaler
        judge = self.judge
        signal_choice = self.player_signals.get(signaler)
        judge_choice = self.player_decisions.get(judge)
        actual_level = self.determine_signal_level(signaler)

        truthful = None
        if (signal_choice is not None) and (actual_level is not None):
            truthful = (signal_choice == actual_level)

        winner = None
        if judge_choice and truthful is not None:
            if judge_choice == 'wahr':
                winner = judge if truthful else signaler
            elif judge_choice == 'bluff':
                winner = judge if not truthful else signaler

        self.last_outcome = {
            'winner': winner,
            'truthful': truthful,
            'actual_level': actual_level,
            'signal_choice': signal_choice,
            'judge_choice': judge_choice,
            'payout': self.current_round_has_stake,
        }
        return self.last_outcome

    def update_info_labels(self):
        for lbl in self.info_labels.values():
            lbl.text = ''

        if not self.session_configured:
            msg = 'Bitte Sessionnummer eingeben, um zu starten.'
            self.info_labels['bottom'].text = msg
            self.info_labels['top'].text = msg
            for lbl in self.outcome_labels.values():
                lbl.text = ''
            return

        if self.session_finished:
            message = self.pause_message or 'Experiment beendet. Vielen Dank!'
            self.info_labels['bottom'].text = message
            self.info_labels['top'].text = message
            self._update_outcome_labels()
            return

        if self.in_block_pause:
            message = self.pause_message or 'Pause. Drückt Play, um fortzufahren.'
            self.info_labels['bottom'].text = message
            self.info_labels['top'].text = message
            self._update_outcome_labels()
            return

        self.compute_outcome()
        total_rounds = self.total_rounds_planned or 0

        for vp in (1, 2):
            physical = self.physical_by_role.get(vp)
            if physical not in (1, 2):
                continue

            label_key = 'bottom' if physical == 1 else 'top'
            label = self.info_labels[label_key]

            lines = []
            header_line = self.format_round_header(vp, physical, total_rounds)
            if header_line:
                lines.append(header_line)

            score_line = self.format_score_line(vp)
            if score_line:
                lines.append(score_line)

            own_label, other_label = self.choice_labels_for_vp(vp)
            own_choice, other_choice = self.choice_texts_for_vp(vp)
            if own_choice and own_label:
                lines.append(f'{own_label}: {own_choice}')
            if other_choice and other_label:
                lines.append(f'{other_label}: {other_choice}')

            summary_text = self.outcome_summary_text()
            if summary_text:
                lines.append(summary_text)

            player_result = self.result_line_for_vp(vp)
            if player_result:
                lines.append(player_result)

            label.text = "\n".join(lines)

        self._update_outcome_labels()

    def player_descriptor(self, player: int) -> str:
        role = self.role_by_physical.get(player)
        if role in (1, 2):
            return f'Versuchsperson {role} – Spieler {player}'
        return f'Spieler {player}'

    def _update_outcome_labels(self):
        for label in self.outcome_labels.values():
            label.text = ''

    def format_round_header(self, vp: int, physical: int, total_rounds: int) -> str:
        if total_rounds:
            round_part = f'Runde {self.round}/{total_rounds}'
        else:
            round_part = f'Runde {self.round}'
        return f'{round_part} | Versuchsperson {vp}: Spieler {physical}'

    def format_score_line(self, vp: int) -> str:
        if not (self.current_round_has_stake and self.score_state_round_start):
            return ''
        start_score = self.score_state_round_start.get(vp)
        if start_score is None:
            return ''
        if self.outcome_score_applied and self.score_state:
            end_score = self.score_state.get(vp, start_score)
            delta = end_score - start_score
            if delta > 0:
                return f'Punkte: {start_score} +{delta}'
            if delta < 0:
                return f'Punkte: {start_score} - {abs(delta)}'
        return f'Punkte: {start_score}'

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

    def choice_texts_for_vp(self, vp: int):
        physical = self.physical_by_role.get(vp)
        if physical not in (1, 2):
            return (None, None)
        other_vp = 2 if vp == 1 else 1
        other_physical = self.physical_by_role.get(other_vp)
        own_choice = None
        other_choice = None
        if physical == self.signaler:
            own_choice = self.format_signal_choice(self.player_signals.get(physical))
            if other_physical:
                other_choice = self.format_decision_choice(
                    self.player_decisions.get(other_physical)
                )
        else:
            own_choice = self.format_decision_choice(self.player_decisions.get(physical))
            if other_physical:
                other_choice = self.format_signal_choice(
                    self.player_signals.get(other_physical)
                )
        return own_choice, other_choice

    def outcome_summary_text(self) -> str:
        if not self.last_outcome:
            return ''
        truthful = self.last_outcome.get('truthful')
        judge_choice = self.last_outcome.get('judge_choice')
        if truthful is None or not judge_choice:
            return ''

        summary_map = {
            (True, 'wahr'): 'Korrektes Signal - korrektes Urteil',
            (True, 'bluff'): 'Korrektes Signal - falsches Urteil',
            (False, 'bluff'): 'Falsches Signal - korrektes Urteil',
            (False, 'wahr'): 'Falsches Signal - falsches Urteil',
        }
        return summary_map.get((truthful, judge_choice), '')

    def result_line_for_vp(self, vp: int) -> str:
        if not self.last_outcome:
            return ''
        winner_physical = self.last_outcome.get('winner')
        if winner_physical not in (1, 2):
            return ''
        winner_vp = self.role_by_physical.get(winner_physical)
        if winner_vp not in (1, 2):
            return ''
        payout = self.last_outcome.get('payout')
        base = 'Gewonnen' if winner_vp == vp else 'Verloren'
        if not payout:
            return base
        if not (self.score_state_round_start and self.score_state):
            return base
        start_score = self.score_state_round_start.get(vp)
        end_score = self.score_state.get(vp)
        if start_score is None or end_score is None:
            return base
        delta = end_score - start_score
        if delta > 0:
            return f'{base} +{delta}'
        if delta < 0:
            return f'{base} - {abs(delta)}'
        return base

    def describe_level(self, level:str) -> str:
        return self.format_signal_choice(level) or (level or '-')

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
        if self.signaler == 1:
            self.role_by_physical = {1: 1, 2: 2}
        else:
            self.role_by_physical = {1: 2, 2: 1}
        self.physical_by_role = {role: player for player, role in self.role_by_physical.items()}

    def current_engine_phase(self):
        mapping = {
            PH_WAIT_BOTH_START: EnginePhase.WAITING_START,
            PH_P1_INNER: EnginePhase.DEALING,
            PH_P2_INNER: EnginePhase.DEALING,
            PH_P1_OUTER: EnginePhase.DEALING,
            PH_P2_OUTER: EnginePhase.DEALING,
            PH_SIGNALER: EnginePhase.SIGNAL_WAIT,
            PH_JUDGE: EnginePhase.CALL_WAIT,
            PH_SHOWDOWN: EnginePhase.REVEAL_SCORE,
        }
        return mapping.get(self.phase, EnginePhase.DEALING)

    def log_event(self, player: int, action: str, payload=None):
        if not self.logger or not self.session_configured:
            return
        payload = payload or {}
        if player is None:
            actor = 'SYS'
        else:
            role = self.role_by_physical.get(player)
            actor = 'P1' if role == 1 else 'P2'
        round_idx = max(0, self.round - 1)
        self.logger.log(
            self.session_id,
            round_idx,
            self.current_engine_phase(),
            actor,
            action,
            payload
        )
        self.write_round_log(actor, action, payload, player)

    def init_round_log(self):
        if not self.session_id:
            return
        if self.round_log_fp:
            self.close_round_log()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f'round_log_{self.session_id}.csv'
        new_file = not path.exists()
        self.round_log_path = path
        self.round_log_fp = open(path, 'a', encoding='utf-8', newline='')
        self.round_log_writer = csv.writer(self.round_log_fp)
        if new_file:
            header = [
                'Session',
                'Bedingung',
                'Block',
                'Runde im Block',
                'Spieler 1',
                'VP',
                'Karte1 VP1',
                'Karte2 VP1',
                'Karte1 VP2',
                'Karte2 VP2',
                'Aktion',
                'Zeit',
                'Gewinner',
                'Punktestand VP1',
                'Punktestand VP2',
            ]
            self.round_log_writer.writerow(header)
            self.round_log_fp.flush()

    def round_log_action_label(self, action: str, payload: dict) -> str:
        if action in ('start_click', 'round_start'):
            return 'Start'
        if action == 'next_round_click':
            return 'Nächste Runde'
        if action == 'reveal_inner':
            return 'Karte 1'
        if action == 'reveal_outer':
            return 'Karte 2'
        if action == 'signal_choice':
            return self.format_signal_choice(payload.get('level')) or 'Signal'
        if action == 'call_choice':
            return self.format_decision_choice(payload.get('decision')) or 'Entscheidung'
        if action == 'showdown':
            return 'Showdown'
        if action == 'session_start':
            return 'Session'
        return action

    def write_round_log(self, actor: str, action: str, payload: dict, player: int):
        if not self.round_log_writer:
            return
        if player not in (1, 2):
            return
        if action == 'showdown':
            return
        block_condition = ''
        block_number = ''
        round_in_block = ''
        next_round_info = None
        if action == 'next_round_click':
            next_round_info = self.peek_next_round_info()
        if next_round_info:
            block = next_round_info['block']
            block_condition = 'pay' if block.get('payout') else 'no_pay'
            block_number = block.get('index', '')
            round_in_block = next_round_info['round_in_block']
        elif self.current_block_info:
            block_condition = 'pay' if self.current_round_has_stake else 'no_pay'
            block_number = self.current_block_info['index']
            round_in_block = self.round_in_block
        plan = None
        plan_info = self.get_current_plan()
        if plan_info:
            plan = plan_info[1]
        vp1_cards = plan['vp1'] if plan else (None, None)
        vp2_cards = plan['vp2'] if plan else (None, None)
        if not vp1_cards:
            vp1_cards = (None, None)
        if not vp2_cards:
            vp2_cards = (None, None)
        actor_vp = ''
        if player in (1, 2):
            vp_num = self.role_by_physical.get(player)
            if vp_num in (1, 2):
                actor_vp = f'VP{vp_num}'
        spieler1_vp = ''
        vp_player1 = self.role_by_physical.get(1)
        if vp_player1 in (1, 2):
            spieler1_vp = f'VP{vp_player1}'
        action_label = self.round_log_action_label(action, payload)
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        winner_label = ''
        if self.last_outcome and self.last_outcome.get('winner') in (1, 2):
            winner_vp = self.role_by_physical.get(self.last_outcome.get('winner'))
            if winner_vp in (1, 2):
                winner_label = f'VP{winner_vp}'
        if self.score_state:
            score_vp1 = self.score_state.get(1, '')
            score_vp2 = self.score_state.get(2, '')
        elif self.score_state_round_start:
            score_vp1 = self.score_state_round_start.get(1, '')
            score_vp2 = self.score_state_round_start.get(2, '')
        else:
            score_vp1 = ''
            score_vp2 = ''
        def _card_value(val):
            return '' if val is None else val

        row = [
            self.session_id or '',
            block_condition,
            block_number,
            round_in_block,
            spieler1_vp,
            actor_vp,
            _card_value(vp1_cards[0]) if vp1_cards else '',
            _card_value(vp1_cards[1]) if vp1_cards else '',
            _card_value(vp2_cards[0]) if vp2_cards else '',
            _card_value(vp2_cards[1]) if vp2_cards else '',
            action_label,
            timestamp,
            winner_label,
            score_vp1,
            score_vp2,
        ]
        self.round_log_writer.writerow(row)
        self.round_log_fp.flush()

    def close_round_log(self):
        if self.round_log_fp:
            self.round_log_fp.close()
            self.round_log_fp = None
            self.round_log_writer = None


    def prompt_session_number(self):
        if self.session_popup:
            return

        layout = FloatLayout()
        popup_width = 800
        popup_height = 500

        lbl = Label(
            text='Bitte Sessionnummer eingeben:',
            size_hint=(0.8, 0.2),
            pos_hint={'center_x': 0.5, 'top': 0.95}
        )
        layout.add_widget(lbl)

        self.session_input = TextInput(
            multiline=False,
            input_filter='int',
            size_hint=(0.6, 0.2),
            pos_hint={'center_x': 0.5, 'center_y': 0.6}
        )
        layout.add_widget(self.session_input)

        self.session_error = Label(
            text='',
            color=(1, 0, 0, 1),
            size_hint=(0.8, 0.15),
            pos_hint={'center_x': 0.5, 'center_y': 0.4}
        )
        layout.add_widget(self.session_error)

        btn = Button(
            text='Start',
            size_hint=(0.3, 0.18),
            pos_hint={'center_x': 0.5, 'y': 0.05}
        )
        btn.bind(on_release=self.confirm_session_number)
        layout.add_widget(btn)

        popup = Popup(
            title='Sessionnummer',
            content=layout,
            size_hint=(None, None),
            size=(popup_width, popup_height),
            auto_dismiss=False
        )
        self.session_popup = popup
        popup.open()

    def confirm_session_number(self, *_):
        text = self.session_input.text.strip() if hasattr(self, 'session_input') else ''
        try:
            number = int(text)
            if number <= 0:
                raise ValueError
        except ValueError:
            if hasattr(self, 'session_error'):
                self.session_error.text = 'Bitte eine positive Zahl eingeben.'
            return

        self.session_number = number
        self.session_id = f'S{number:03d}'
        self.session_configured = True
        self.log_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.log_dir / f'events_{self.session_id}.sqlite3'
        self.logger = EventLogger(str(db_path))
        self.init_round_log()
        self.update_role_assignments()
        if self.session_popup:
            self.session_popup.dismiss()
            self.session_popup = None
        self.log_event(None, 'session_start', {'session_number': number})
        self.log_round_start()
        self.apply_phase()
        self.update_info_labels()

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
        })

    def record_action(self, player:int, text:str):
        self.status_lines[player].append(text)
        self.update_status_label(player)

    def update_status_label(self, player:int):
        label = self.status_labels.get(player)
        if not label:
            return
        role = 'Signal' if self.signaler == player else 'Judge'
        header = [f"Du bist Spieler {player}", f"Rolle: {role}"]
        body = self.status_lines[player]
        self.status_labels[player].text = "\n".join(header + body)

class TabletopApp(App):
    def build(self):
        self.title = 'Masterarbeit – Tabletop UX'
        root = TabletopRoot()
        return root

    def on_stop(self):
        root = self.root
        if root and root.logger:
            root.logger.close()
        if root:
            root.close_round_log()

if __name__ == '__main__':
    TabletopApp().run()
 
