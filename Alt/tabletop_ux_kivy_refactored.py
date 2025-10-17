# Pass-5-Diff
# [x] Nur interne Typisierung/Helfer/Guards ergänzt.
# [x] Keine UI-/Flow-/Timing-/Text-/Log-Semantik verändert.
# [x] Unbenutzte Imports entfernt, private Utils vereinheitlicht.
# [x] LogFacade-Signaturen/Outputs unverändert, defensivere Nutzung.
# [x] Neue Mini-Helfer: _is_ready_session, _is_player, _now_millis_str.

from __future__ import annotations

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, NamedTuple, Literal, TextIO

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


class Transition(NamedTuple):
    """State-machine transition entry."""

    next_phase: Optional[str]
    delay: float
    effect: str


class PhaseMachine:
    """Central phase transition handler with scheduling support."""

    def __init__(self, owner: 'TabletopRoot'):
        self.owner = owner
        self.transitions: Dict[Tuple[str, str], Transition] = {
            (PH_WAIT_BOTH_START, 'start_click'): Transition(PH_P1_INNER, 0.0, '_effect_start_click'),
            (PH_P1_INNER, 'reveal_inner'): Transition(PH_P2_INNER, 0.2, '_effect_reveal_inner'),
            (PH_P2_INNER, 'reveal_inner'): Transition(PH_P1_OUTER, 0.2, '_effect_reveal_inner'),
            (PH_P1_OUTER, 'reveal_outer'): Transition(PH_P2_OUTER, 0.2, '_effect_reveal_outer'),
            (PH_P2_OUTER, 'reveal_outer'): Transition(PH_SIGNALER, 0.2, '_effect_reveal_outer'),
            (PH_SIGNALER, 'signal_choice'): Transition(PH_JUDGE, 0.2, '_effect_signal_choice'),
            (PH_JUDGE, 'call_choice'): Transition(PH_SHOWDOWN, 0.2, '_effect_call_choice'),
            (PH_SHOWDOWN, 'next_round_click'): Transition(PH_WAIT_BOTH_START, 0.0, '_effect_next_round_click'),
        }

    def handle(self, event: str, **context: Any) -> bool:
        """Process an event for the current phase and schedule transition."""

        key = (self.owner.phase, event)
        transition = self.transitions.get(key)
        if not transition:
            return False
        effect_fn: Callable[..., Any] = getattr(self.owner, transition.effect)
        result = effect_fn(event=event, **context)
        if result is False:
            return False
        next_phase = transition.next_phase if result in (None, True) else result
        if next_phase is None:
            return False

        def _advance(*_):
            self.owner.goto(next_phase)

        if transition.delay > 0:
            Clock.schedule_once(_advance, transition.delay)
        else:
            _advance(None)
        return True


def parse_int_like(value: Any) -> Optional[int]:
    """Parse integers tolerant to floats, commas and blanks."""
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


def parse_cards_from_row(row: Sequence[Any], start: int, end: int) -> Tuple[int, int]:
    """Extract exactly two card integers from a CSV row slice."""
    values = []
    upper = min(end, len(row))
    for idx in range(start, upper):
        cell = row[idx]
        cell_text = str(cell).strip() if cell else ''
        if not cell_text:
            continue
        try:
            values.append(int(float(cell_text)))
        except ValueError:
            continue
        if len(values) == 2:
            break
    if len(values) < 2:
        raise ValueError('Zu wenige Karten')
    return values[0], values[1]


def parse_category_text(value: Any) -> Optional[str]:
    """Return normalized category text from CSV cell."""
    if value is None:
        return None
    text = str(value).strip().strip('"').lower()
    return text or None


def card_back_image(live: bool) -> str:
    """Return back/back_stop card asset based on live state."""
    return ASSETS['cards']['back'] if live else ASSETS['cards']['back_stop']


def card_face_for_value(value: Any) -> str:
    """Resolve card face asset path for a numeric value."""
    number = parse_int_like(value)
    if number is None:
        return ASSETS['cards']['back']
    filename = f'{number}.png'
    path = os.path.join(CARD_DIR, filename)
    return path if os.path.exists(path) else ASSETS['cards']['back']


def button_asset(asset_pair: Dict[str, str], live: bool) -> str:
    """Fetch live/stop image from an asset pair."""
    state = 'live' if live else 'stop'
    if state in asset_pair:
        return asset_pair[state]
    return asset_pair.get('stop', card_back_image(False))


def _is_player(value: Optional[int]) -> bool:
    """Return True if the given value matches a physical player identifier."""

    return value in (1, 2)


def _now_millis_str() -> str:
    """Return the current time formatted to milliseconds."""

    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def signal_choice_label(level: Optional[str]) -> Optional[str]:
    """Translate signal code to UI label."""
    return {'low': 'Tief', 'mid': 'Mittel', 'high': 'Hoch'}.get(level)


def decision_choice_label(decision: Optional[str]) -> Optional[str]:
    """Translate decision code to UI label."""
    return {'wahr': 'Wahrheit', 'bluff': 'Bluff'}.get(decision)


def describe_signal_level_text(level: Optional[str]) -> str:
    """Provide display text for a signal level."""
    return signal_choice_label(level) or (level or '-')


def round_header_text(round_no: int, total_rounds: Optional[int], vp: int, physical: int) -> str:
    """Compose round header identical to original formatting."""
    if total_rounds:
        round_part = f'Runde {round_no}/{total_rounds}'
    else:
        round_part = f'Runde {round_no}'
    return f'{round_part} | Versuchsperson {vp}: Spieler {physical}'


def score_line_text(start_score: Optional[int], end_score: Optional[int], stake_active: bool, outcome_applied: bool) -> str:
    """Format score line respecting stake state and deltas."""
    if not stake_active or start_score is None:
        return ''
    if outcome_applied and end_score is not None:
        delta = end_score - start_score
        if delta > 0:
            return f'Punkte: {start_score} +{delta}'
        if delta < 0:
            return f'Punkte: {start_score} - {abs(delta)}'
    return f'Punkte: {start_score}'


def outcome_summary_from_outcome(outcome: Optional[Dict[str, Any]]) -> str:
    """Mirror original summary text mapping based on outcome payload."""
    if not outcome:
        return ''
    truthful = outcome.get('truthful')
    judge_choice = outcome.get('judge_choice')
    if truthful is None or not judge_choice:
        return ''
    summary_map = {
        (True, 'wahr'): 'Korrektes Signal - korrektes Urteil',
        (True, 'bluff'): 'Korrektes Signal - falsches Urteil',
        (False, 'bluff'): 'Falsches Signal - korrektes Urteil',
        (False, 'wahr'): 'Falsches Signal - falsches Urteil',
    }
    return summary_map.get((truthful, judge_choice), '')


def result_line_text(
    outcome: Optional[Dict[str, Any]],
    vp: int,
    winner_vp: Optional[int],
    start_score: Optional[int],
    end_score: Optional[int],
    payout: bool,
) -> str:
    """Return the winner line identical to legacy behaviour."""
    if not outcome:
        return ''
    winner_physical = outcome.get('winner')
    if not _is_player(winner_physical):
        return ''
    if winner_vp is None:
        return 'Gewonnen' if winner_physical == vp else 'Verloren'
    base = 'Gewonnen' if winner_vp == vp else 'Verloren'
    if not payout or start_score is None or end_score is None:
        return base
    delta = end_score - start_score
    if delta > 0:
        return f'{base} +{delta}'
    if delta < 0:
        return f'{base} - {abs(delta)}'
    return base


def choice_labels_for_physical(physical: Optional[int], signaler: int, judge: int) -> Tuple[Optional[str], Optional[str]]:
    """Determine label headings for own/other choice fields."""
    if not _is_player(physical):
        return (None, None)
    if physical == signaler:
        return 'Eigenes Signal', 'Anderes Urteil'
    if physical == judge:
        return 'Eigenes Urteil', 'Anderes Signal'
    return (None, None)


def choice_texts_for_roles(
    physical: Optional[int],
    other_physical: Optional[int],
    signaler: int,
    judge: int,
    signals: Dict[int, Optional[str]],
    decisions: Dict[int, Optional[str]],
) -> Tuple[Optional[str], Optional[str]]:
    """Return own/other choice labels respecting roles."""
    if not _is_player(physical):
        return (None, None)
    own_choice = None
    other_choice = None
    if physical == signaler:
        own_choice = signal_choice_label(signals.get(physical))
        if _is_player(other_physical):
            other_choice = decision_choice_label(decisions.get(other_physical))
    elif physical == judge:
        own_choice = decision_choice_label(decisions.get(physical))
        if _is_player(other_physical):
            other_choice = signal_choice_label(signals.get(other_physical))
    return own_choice, other_choice


def resolve_signal_level(value: Any) -> Optional[str]:
    """Map numeric card totals to signal level buckets."""
    parsed = parse_int_like(value)
    if parsed is None or parsed <= 0 or parsed in (20, 21, 22):
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


def actor_for_player(player: Optional[int], role_by_physical: Dict[int, int]) -> str:
    """Resolve actor label for logging identical to legacy logic."""
    if player is None:
        return 'SYS'
    role = role_by_physical.get(player)
    return 'P1' if role == 1 else 'P2'


def _emit_event(
    logger: Optional[EventLogger],
    session_id: Optional[int],
    round_idx: int,
    engine_phase: EnginePhase,
    actor: str,
    action: str,
    payload: Dict[str, Any],
) -> None:
    """Wrapper around EventLogger.log preserving behaviour."""
    if not logger or not session_id:
        return
    logger.log(session_id, round_idx, engine_phase, actor, action, payload)


class LogFacade:
    """Encapsulates SQLite event emission and round CSV writing."""

    def __init__(self, owner: 'TabletopRoot') -> None:
        self.owner = owner
        self.csv_path: Optional[Path] = None
        self.csv_fp: Optional[TextIO] = None
        self.csv_writer: Optional[Any] = None

    def emit_event(self, action: str, payload: Optional[Dict[str, Any]], *, player: Optional[int]) -> None:
        """Emit a single event and mirror it into the round CSV."""

        root = self.owner
        if not root.logger or not root.session_configured:
            return
        payload = payload or {}
        actor = actor_for_player(player, root.role_by_physical)
        round_idx = max(0, root.round - 1)
        _emit_event(
            root.logger,
            root.session_id,
            round_idx,
            root.current_engine_phase(),
            actor,
            action,
            payload,
        )
        self.round_log(action, payload, player=player)

    def round_log(self, action: str, payload: Optional[Dict[str, Any]], *, player: Optional[int]) -> None:
        """Write the CSV mirror entry using legacy semantics."""

        if not self.csv_writer or not _is_player(player):
            return
        if action == 'showdown':
            return
        row = self._csv_row(action, payload or {}, player)
        if not row:
            return
        self.csv_writer.writerow(row)
        if self.csv_fp:
            self.csv_fp.flush()

    def init_round_csv(self, session_id: Optional[str]) -> None:
        """Prepare CSV logging file with identical header management."""

        if not session_id:
            return
        if self.csv_fp:
            self.close_round_csv()
        root = self.owner
        root.log_dir.mkdir(parents=True, exist_ok=True)
        path = root.log_dir / f'round_log_{session_id}.csv'
        self.csv_path = path
        new_file = not path.exists()
        self.csv_fp = open(path, 'a', encoding='utf-8', newline='')
        self.csv_writer = csv.writer(self.csv_fp)
        if new_file:
            self.csv_writer.writerow(self._csv_header())
            self.csv_fp.flush()

    def close_round_csv(self) -> None:
        """Close CSV handles if open."""

        if self.csv_fp:
            self.csv_fp.close()
            self.csv_fp = None
            self.csv_writer = None
        self.csv_path = None

    def _csv_header(self) -> Sequence[str]:
        return [
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

    def _csv_row(self, action: str, payload: Dict[str, Any], player: int) -> Sequence[Any]:
        root = self.owner
        block_condition = ''
        block_number = ''
        round_in_block = ''
        next_round_info = None
        if action == 'next_round_click':
            next_round_info = root.peek_next_round_info()
        if next_round_info:
            block = next_round_info['block']
            block_condition = 'pay' if block.get('payout') else 'no_pay'
            block_number = block.get('index', '')
            round_in_block = next_round_info['round_in_block']
        elif root.current_block_info:
            block_condition = 'pay' if root.current_round_has_stake else 'no_pay'
            block_number = root.current_block_info['index']
            round_in_block = root.round_in_block
        plan = None
        plan_info = root.get_current_plan()
        if plan_info:
            plan = plan_info[1]
        vp1_cards = plan['vp1'] if plan else (None, None)
        vp2_cards = plan['vp2'] if plan else (None, None)
        if not vp1_cards:
            vp1_cards = (None, None)
        if not vp2_cards:
            vp2_cards = (None, None)
        actor_vp = ''
        vp_num = root.role_by_physical.get(player)
        if _is_player(vp_num):
            actor_vp = f'VP{vp_num}'
        spieler1_vp = ''
        vp_player1 = root.role_by_physical.get(1)
        if _is_player(vp_player1):
            spieler1_vp = f'VP{vp_player1}'
        action_label = root.round_log_action_label(action, payload)
        timestamp = _now_millis_str()
        winner_label = ''
        if root.last_outcome and _is_player(root.last_outcome.get('winner')):
            winner_vp = root.role_by_physical.get(root.last_outcome.get('winner'))
            if _is_player(winner_vp):
                winner_label = f'VP{winner_vp}'
        if root.score_state:
            score_vp1 = root.score_state.get(1, '')
            score_vp2 = root.score_state.get(2, '')
        elif root.score_state_round_start:
            score_vp1 = root.score_state_round_start.get(1, '')
            score_vp2 = root.score_state_round_start.get(2, '')
        else:
            score_vp1 = ''
            score_vp2 = ''

        def _card_value(val: Any) -> Any:
            return '' if val is None else val

        return [
            root.session_id or '',
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
        self.front_image = card_back_image(True)
        self.border = (0, 0, 0, 0)
        back_stop = card_back_image(False)
        self.background_normal = back_stop
        self.background_down = back_stop
        self.background_disabled_normal = back_stop
        self.background_disabled_down = back_stop
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
            img = card_back_image(True)
        else:
            img = card_back_image(False)
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
        stop_img = button_asset(asset_pair, False)
        self.background_normal = stop_img
        self.background_down = stop_img
        self.background_disabled_normal = stop_img
        self.background_disabled_down = stop_img
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
        img = button_asset(self.asset_pair, True if (self.live or self.selected) else False)
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


def make_card_widget(owner: 'TabletopRoot', role_side: Literal['p1', 'p2'], which: Literal['inner', 'outer']) -> CardWidget:
    """Create a CardWidget bound to the proper tap handler."""
    valid_side = role_side if role_side in ('p1', 'p2') else 'p1'
    valid_slot = which if which in ('inner', 'outer') else 'inner'
    player = 1 if valid_side == 'p1' else 2
    widget = CardWidget(size_hint=(None, None))
    widget.bind(on_release=lambda *_: owner.tap_card(player, valid_slot))
    return widget


def make_icon_button(
    owner: 'TabletopRoot',
    kind: Literal['signal', 'decide', 'play'],
    key: str,
    player: int,
) -> IconButton:
    """Create an IconButton with pre-bound callbacks for the given player."""
    fallback = card_back_image(False)
    asset_pair: Dict[str, str]
    if kind == 'play':
        asset_pair = ASSETS['play']
    elif kind == 'signal':
        asset_pair = ASSETS['signal'].get(key, {'live': fallback, 'stop': fallback})
    elif kind == 'decide':
        asset_pair = ASSETS['decide'].get(key, {'live': fallback, 'stop': fallback})
    else:
        asset_pair = {'live': fallback, 'stop': fallback}
    button = IconButton(asset_pair, size_hint=(None, None))
    player_id = player if _is_player(player) else 1
    if kind == 'play':
        button.bind(on_release=lambda *_: owner.start_pressed(player_id))
        rotation = 180 if player_id == 1 else 0
    elif kind == 'signal':
        button.bind(on_release=lambda _, lvl=key, p=player_id: owner.pick_signal(p, lvl))
        rotation = 0 if player_id == 1 else 180
    elif kind == 'decide':
        button.bind(on_release=lambda _, choice=key, p=player_id: owner.pick_decision(p, choice))
        rotation = 0 if player_id == 1 else 180
    else:
        rotation = 0 if player_id == 2 else 180
    button.set_rotation(rotation)
    return button


def compute_layout(size: Tuple[int, int]) -> Dict[str, Any]:
    """Return layout configuration for all widgets based on the window size."""
    W, H = size
    base_w, base_h = 3840.0, 2160.0
    scale = min(W / base_w if base_w else 1, H / base_h if base_h else 1)

    corner_margin = 120 * scale
    card_width, card_height = 420 * scale, 640 * scale
    card_gap = 70 * scale
    start_margin = 60 * scale
    start_size = (360 * scale, 360 * scale)

    p1_outer_pos = (corner_margin, corner_margin)
    p1_inner_pos = (corner_margin + card_width + card_gap, corner_margin)
    p2_outer_pos = (W - corner_margin - card_width, H - corner_margin - card_height)
    p2_inner_pos = (p2_outer_pos[0] - card_width - card_gap, p2_outer_pos[1])

    btn_width, btn_height = 260 * scale, 260 * scale
    vertical_gap = 40 * scale
    horizontal_gap = 60 * scale
    cluster_shift = 620 * scale
    vertical_offset = 140 * scale

    signal_x = W - corner_margin - btn_width - cluster_shift
    base_y = corner_margin + vertical_offset
    decision_x = signal_x - horizontal_gap - btn_width

    signal2_x = corner_margin + cluster_shift
    top_y = H - corner_margin - vertical_offset
    decision2_x = signal2_x + btn_width + horizontal_gap

    center_card_width, center_card_height = 380 * scale, 560 * scale
    center_gap_x = 90 * scale
    center_gap_y = 60 * scale
    left_x = W / 2 - center_card_width - center_gap_x / 2
    right_x = W / 2 + center_gap_x / 2
    bottom_y = H / 2 - center_card_height - center_gap_y / 2
    top_y_center = H / 2 + center_gap_y / 2

    info_width, info_height = 2000 * scale, 160 * scale
    info_margin = 60 * scale

    outcome_width = btn_width * 2 + horizontal_gap
    outcome_height = 120 * scale

    badge_width, badge_height = 1400 * scale, 70 * scale

    return {
        'scale': scale,
        'background': {'pos': (0, 0), 'size': (W, H)},
        'start_buttons': {
            1: {'size': start_size, 'pos': (start_margin, H - start_margin - start_size[1]), 'rotation': 180},
            2: {'size': start_size, 'pos': (W - start_margin - start_size[0], start_margin), 'rotation': 0},
        },
        'cards': {
            ('p1', 'outer'): {'size': (card_width, card_height), 'pos': p1_outer_pos},
            ('p1', 'inner'): {'size': (card_width, card_height), 'pos': p1_inner_pos},
            ('p2', 'outer'): {'size': (card_width, card_height), 'pos': p2_outer_pos},
            ('p2', 'inner'): {'size': (card_width, card_height), 'pos': p2_inner_pos},
        },
        'signals': {
            1: {
                level: {
                    'size': (btn_width, btn_height),
                    'pos': (signal_x, base_y + idx * (btn_height + vertical_gap)),
                    'rotation': 0,
                }
                for idx, level in enumerate(['low', 'mid', 'high'])
            },
            2: {
                level: {
                    'size': (btn_width, btn_height),
                    'pos': (signal2_x, top_y - btn_height - idx * (btn_height + vertical_gap)),
                    'rotation': 180,
                }
                for idx, level in enumerate(['low', 'mid', 'high'])
            },
        },
        'decisions': {
            1: {
                choice: {
                    'size': (btn_width, btn_height),
                    'pos': (decision_x, base_y + idx * (btn_height + vertical_gap)),
                    'rotation': 0,
                }
                for idx, choice in enumerate(['bluff', 'wahr'])
            },
            2: {
                choice: {
                    'size': (btn_width, btn_height),
                    'pos': (decision2_x, top_y - btn_height - idx * (btn_height + vertical_gap)),
                    'rotation': 180,
                }
                for idx, choice in enumerate(['bluff', 'wahr'])
            },
        },
        'center_cards': {
            1: [
                {'size': (center_card_width, center_card_height), 'pos': (right_x, bottom_y)},
                {'size': (center_card_width, center_card_height), 'pos': (left_x, bottom_y)},
            ],
            2: [
                {'size': (center_card_width, center_card_height), 'pos': (left_x, top_y_center)},
                {'size': (center_card_width, center_card_height), 'pos': (right_x, top_y_center)},
            ],
        },
        'info_labels': {
            'bottom': {
                'size': (info_width, info_height),
                'pos': (W / 2 - info_width / 2, bottom_y - info_height - info_margin),
                'font_size': 56 * scale if scale else 56,
                'rotation': 0,
            },
            'top': {
                'size': (info_width, info_height),
                'pos': (W / 2 - info_width / 2, top_y_center + center_card_height + info_margin),
                'font_size': 56 * scale if scale else 56,
                'rotation': 180,
            },
        },
        'outcome_labels': {
            1: {
                'size': (outcome_width, outcome_height),
                'pos': (decision_x, base_y + 2 * (btn_height + vertical_gap)),
                'font_size': 64 * scale if scale else 64,
                'rotation': 0,
            },
            2: {
                'size': (outcome_width, outcome_height),
                'pos': (decision2_x, top_y - 2 * (btn_height + vertical_gap) - outcome_height),
                'font_size': 64 * scale if scale else 64,
                'rotation': 180,
            },
        },
        'round_badge': {
            'size': (badge_width, badge_height),
            'pos': (W / 2 - badge_width / 2, corner_margin / 2),
            'font_size': 40 * scale if scale else 40,
        },
    }


def apply_button_layout(button: Button, cfg: Dict[str, Any]) -> None:
    """Apply size, position, and rotation to a button."""
    button.size = cfg['size']
    button.pos = cfg['pos']
    rotation = cfg.get('rotation')
    if rotation is not None and hasattr(button, 'set_rotation'):
        button.set_rotation(rotation)


def apply_label_layout(label: Label, cfg: Dict[str, Any]) -> None:
    """Apply geometry and font settings to a label."""
    label.size = cfg['size']
    label.pos = cfg['pos']
    label.font_size = cfg['font_size']
    label.text_size = cfg.get('text_size', cfg['size'])
    rotation = cfg.get('rotation')
    if rotation is not None and hasattr(label, 'set_rotation'):
        label.set_rotation(rotation)


def apply_image_layout(image: Image, cfg: Dict[str, Any]) -> None:
    """Apply geometry to a center card image."""
    image.size = cfg['size']
    image.pos = cfg['pos']


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
        self.phase_machine = PhaseMachine(self)
        self.role_by_physical = {1: 1, 2: 2}
        self.physical_by_role = {1: 1, 2: 2}
        self.session_number = None
        self.session_id = None
        self.logger = None
        self.log_dir = Path(ROOT) / 'logs'
        self.log = LogFacade(self)
        self.session_popup = None
        self.session_configured = False

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
        self.btn_start_p1 = make_icon_button(self, 'play', 'play', 1)
        self.add_widget(self.btn_start_p1)

        self.btn_start_p2 = make_icon_button(self, 'play', 'play', 2)
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
        self.p1_outer = make_card_widget(self, 'p1', 'outer')
        self.add_widget(self.p1_outer)

        self.p1_inner = make_card_widget(self, 'p1', 'inner')
        self.add_widget(self.p1_inner)

        self.p2_outer = make_card_widget(self, 'p2', 'outer')
        self.add_widget(self.p2_outer)

        self.p2_inner = make_card_widget(self, 'p2', 'inner')
        self.add_widget(self.p2_inner)

        # Button-Cluster für Signale & Entscheidungen pro Spieler
        self.signal_buttons = {1: {}, 2: {}}
        self.decision_buttons = {1: {}, 2: {}}

        for level in ['low', 'mid', 'high']:
            btn = make_icon_button(self, 'signal', level, 1)
            self.signal_buttons[1][level] = btn
            self.add_widget(btn)

        for choice in ['bluff', 'wahr']:
            btn = make_icon_button(self, 'decide', choice, 1)
            self.decision_buttons[1][choice] = btn
            self.add_widget(btn)

        for level in ['low', 'mid', 'high']:
            btn = make_icon_button(self, 'signal', level, 2)
            self.signal_buttons[2][level] = btn
            self.add_widget(btn)

        for choice in ['bluff', 'wahr']:
            btn = make_icon_button(self, 'decide', choice, 2)
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
        layout = compute_layout(Window.size)

        bg_cfg = layout['background']
        self.bg.pos = bg_cfg['pos']
        self.bg.size = bg_cfg['size']

        for player, cfg in layout['start_buttons'].items():
            button = self.btn_start_p1 if player == 1 else self.btn_start_p2
            apply_button_layout(button, cfg)

        for (tag, which), cfg in layout['cards'].items():
            widget = getattr(self, f'{tag}_{which}')
            widget.size = cfg['size']
            widget.pos = cfg['pos']

        for player, buttons in layout['signals'].items():
            for level, cfg in buttons.items():
                apply_button_layout(self.signal_buttons[player][level], cfg)

        for player, buttons in layout['decisions'].items():
            for choice, cfg in buttons.items():
                apply_button_layout(self.decision_buttons[player][choice], cfg)

        for player, entries in layout['center_cards'].items():
            for idx, cfg in enumerate(entries):
                apply_image_layout(self.center_cards[player][idx], cfg)

        for key, cfg in layout['info_labels'].items():
            label = self.info_labels[key]
            apply_label_layout(label, {**cfg, 'text_size': cfg['size']})

        for player, cfg in layout['outcome_labels'].items():
            label = self.outcome_labels[player]
            apply_label_layout(label, {**cfg, 'text_size': cfg['size']})

        badge_cfg = layout['round_badge']
        self.round_badge.size = badge_cfg['size']
        self.round_badge.pos = badge_cfg['pos']
        self.round_badge.font_size = badge_cfg['font_size']
        self.round_badge.text_size = badge_cfg['size']

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
    def _is_ready_session(self) -> bool:
        """Return True when a session is configured and still running."""

        return bool(self.session_configured and not self.session_finished)

    def load_blocks(self) -> List[Dict[str, Any]]:
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

    def load_csv_rounds(self, path: Path) -> List[Dict[str, Any]]:
        rounds: List[Dict[str, Any]] = []
        try:
            with open(path, newline='', encoding='utf-8') as fp:
                rows = list(csv.reader(fp))
        except FileNotFoundError:
            return rounds
        except Exception:
            return rounds

        start_idx = 0
        if rows:
            try:
                parse_cards_from_row(rows[0], 2, 4)
                parse_cards_from_row(rows[0], 7, 9)
            except Exception:
                start_idx = 1

        for row in rows[start_idx:]:
            if not row or all((str(cell).strip() if cell else '') == '' for cell in row):
                continue
            try:
                vp1_cards = parse_cards_from_row(row, 2, 4)
                vp2_cards = parse_cards_from_row(row, 7, 9)
            except Exception:
                continue

            vp1_value = parse_int_like(row[5]) if len(row) > 5 else None
            vp2_value = parse_int_like(row[10]) if len(row) > 10 else None
            vp1_category = parse_category_text(row[1]) if len(row) > 1 else None
            vp2_category = parse_category_text(row[6]) if len(row) > 6 else None

            if vp1_value is None:
                total = sum(vp1_cards)
                vp1_value = 0 if total in (20, 21, 22) else total
            if vp2_value is None:
                total = sum(vp2_cards)
                vp2_value = 0 if total in (20, 21, 22) else total

            rounds.append(
                {
                    'vp1': vp1_cards,
                    'vp2': vp2_cards,
                    'vp1_value': vp1_value,
                    'vp2_value': vp2_value,
                    'vp1_category': vp1_category,
                    'vp2_category': vp2_category,
                }
            )
        return rounds

    def value_to_card_path(self, value: Any) -> str:
        return card_face_for_value(value)

    @staticmethod
    def _parse_value(value: Any) -> Optional[int]:
        return parse_int_like(value)

    def get_hand_value_for_role(self, role: int) -> Optional[int]:
        if not _is_player(role):
            return None
        plan_info = self.get_current_plan()
        if not plan_info:
            return None
        _, plan = plan_info
        if not plan:
            return None
        value = plan.get(f'vp{role}_value')
        parsed = self._parse_value(value)
        if parsed is not None:
            return parsed
        cards = plan.get(f'vp{role}')
        if not cards or any(card is None for card in cards):
            return None
        total = sum(cards)
        return 0 if total in (20, 21, 22) else total

    def get_hand_value_for_player(self, player: int) -> Optional[int]:
        role = self.role_by_physical.get(player)
        value = self.get_hand_value_for_role(role)
        if value is not None:
            return value
        if player == 1:
            inner_widget, outer_widget = self.p1_inner, self.p1_outer
        else:
            inner_widget, outer_widget = self.p2_inner, self.p2_outer
        inner_val = self.card_value_from_path(inner_widget.front_image)
        outer_val = self.card_value_from_path(outer_widget.front_image)
        if inner_val is None or outer_val is None:
            return None
        total = inner_val + outer_val
        return 0 if total in (20, 21, 22) else total

    def signal_level_from_value(self, value: Any) -> Optional[str]:
        return resolve_signal_level(self._parse_value(value))

    def set_cards_from_plan(self, plan: Optional[Dict[str, Any]]) -> None:
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
        ready = self._is_ready_session()
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
            self.log.emit_event(action, None, player=who)
        if self.p1_pressed and self.p2_pressed:
            # in nächste Phase
            self.p1_pressed = False
            self.p2_pressed = False
            event = 'start_click' if self.phase == PH_WAIT_BOTH_START else 'next_round_click'
            self.phase_machine.handle(event, start_immediately=(self.phase == PH_SHOWDOWN))

    def tap_card(self, who:int, which:str):
        # which in {'inner','outer'}
        if who == 1 and which == 'inner' and self.phase == PH_P1_INNER:
            self.phase_machine.handle('reveal_inner', player=1, which='inner', card_index=1)
        elif who == 2 and which == 'inner' and self.phase == PH_P2_INNER:
            self.phase_machine.handle('reveal_inner', player=2, which='inner', card_index=1)
        elif who == 1 and which == 'outer' and self.phase == PH_P1_OUTER:
            self.phase_machine.handle('reveal_outer', player=1, which='outer', card_index=2)
        elif who == 2 and which == 'outer' and self.phase == PH_P2_OUTER:
            self.phase_machine.handle('reveal_outer', player=2, which='outer', card_index=2)

    def pick_signal(self, player:int, level:str):
        if self.phase != PH_SIGNALER or player != self.signaler:
            return
        self.phase_machine.handle('signal_choice', player=player, level=level)

    def pick_decision(self, player:int, decision:str):
        if self.phase != PH_JUDGE or player != self.judge:
            return
        self.phase_machine.handle('call_choice', player=player, decision=decision)

    # --- PhaseMachine Effekte
    def _effect_start_click(self, **_: Any) -> Optional[str]:
        if self.in_block_pause:
            self.in_block_pause = False
            self.pause_message = ''
            self.setup_round()
            if self.session_finished:
                return False
        if self.session_finished:
            return False
        return PH_P1_INNER

    def _effect_reveal_inner(self, *, player: int, card_index: int, **_: Any) -> None:
        widget = self.p1_inner if player == 1 else self.p2_inner
        widget.flip()
        self.record_action(player, 'Karte innen aufgedeckt')
        self.log.emit_event('reveal_inner', {'card': card_index}, player=player)

    def _effect_reveal_outer(self, *, player: int, card_index: int, **_: Any) -> None:
        widget = self.p1_outer if player == 1 else self.p2_outer
        widget.flip()
        self.record_action(player, 'Karte außen aufgedeckt')
        self.log.emit_event('reveal_outer', {'card': card_index}, player=player)

    def _effect_signal_choice(self, *, player: int, level: str, **_: Any) -> None:
        self.player_signals[player] = level
        for lvl, btn in self.signal_buttons[player].items():
            if lvl == level:
                btn.set_pressed_state()
            else:
                btn.set_live(False)
                btn.disabled = True
        self.record_action(player, f'Signal gewählt: {self.describe_level(level)}')
        self.log.emit_event('signal_choice', {'level': level}, player=player)
        self.update_info_labels()

    def _effect_call_choice(self, *, player: int, decision: str, **_: Any) -> None:
        self.player_decisions[player] = decision
        for choice, btn in self.decision_buttons[player].items():
            if choice == decision:
                btn.set_pressed_state()
            else:
                btn.set_live(False)
                btn.disabled = True
        self.record_action(player, f'Entscheidung: {decision.upper()}')
        self.log.emit_event('call_choice', {'decision': decision}, player=player)
        self.update_info_labels()

    def _effect_next_round_click(self, **kwargs: Any) -> Optional[str]:
        target = self.prepare_next_round(
            start_immediately=bool(kwargs.get('start_immediately')),
            via_machine=True,
        )
        return target

    def goto(self, phase):
        self.phase = phase
        self.apply_phase()

    def prepare_next_round(self, start_immediately: bool = False, via_machine: bool = False):
        # Rollen tauschen
        self.signaler, self.judge = self.judge, self.signaler
        self.update_role_assignments()
        self.advance_round_pointer()
        self.phase = PH_WAIT_BOTH_START
        self.setup_round()
        if start_immediately and not self.in_block_pause and self._is_ready_session():
            target = PH_P1_INNER
        else:
            target = PH_WAIT_BOTH_START
        if via_machine:
            return target
        self.phase = target
        self.apply_phase()
        return target

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
            'actual_value': None,
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
            if _is_player(winner):
                winner_role = self.role_by_physical.get(winner)
                if _is_player(winner_role):
                    loser_role = 1 if winner_role == 2 else 2
                    self.score_state[winner_role] += 1
                    self.score_state[loser_role] -= 1
                    self.outcome_score_applied = True
        self.update_info_labels()
        if self.session_configured:
            self.log.emit_event('showdown', outcome or {}, player=None)

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
        signaler = self.signaler
        judge = self.judge
        signal_choice = self.player_signals.get(signaler)
        judge_choice = self.player_decisions.get(judge)
        actual_value = self.get_hand_value_for_player(signaler)
        actual_level = self.signal_level_from_value(actual_value)

        truthful = None
        if signal_choice and actual_level:
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
            'actual_value': actual_value,
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
            if not _is_player(physical):
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
        if _is_player(role):
            return f'Versuchsperson {role} – Spieler {player}'
        return f'Spieler {player}'

    def _update_outcome_labels(self):
        for label in self.outcome_labels.values():
            label.text = ''

    def format_round_header(self, vp: int, physical: int, total_rounds: int) -> str:
        return round_header_text(self.round, total_rounds, vp, physical)

    def format_score_line(self, vp: int) -> str:
        if not (self.current_round_has_stake and self.score_state_round_start):
            return ''
        start_score = self.score_state_round_start.get(vp)
        if start_score is None:
            return ''
        end_score = None
        if self.outcome_score_applied and self.score_state:
            end_score = self.score_state.get(vp, start_score)
        return score_line_text(start_score, end_score, True, self.outcome_score_applied)

    def format_signal_choice(self, level: str):
        return signal_choice_label(level)

    def format_decision_choice(self, decision: str):
        return decision_choice_label(decision)

    def choice_texts_for_vp(self, vp: int):
        physical = self.physical_by_role.get(vp)
        other_vp = 2 if vp == 1 else 1
        other_physical = self.physical_by_role.get(other_vp)
        return choice_texts_for_roles(
            physical,
            other_physical,
            self.signaler,
            self.judge,
            self.player_signals,
            self.player_decisions,
        )

    def outcome_summary_text(self) -> str:
        return outcome_summary_from_outcome(self.last_outcome)

    def result_line_for_vp(self, vp: int) -> str:
        if not self.last_outcome:
            return ''
        winner_physical = self.last_outcome.get('winner')
        if not _is_player(winner_physical):
            return ''
        winner_vp = self.role_by_physical.get(winner_physical)
        if not _is_player(winner_vp):
            return ''
        payout = bool(self.last_outcome.get('payout'))
        start_score = None
        end_score = None
        if self.score_state_round_start:
            start_score = self.score_state_round_start.get(vp)
        if self.score_state:
            end_score = self.score_state.get(vp)
        return result_line_text(
            self.last_outcome,
            vp,
            winner_vp,
            start_score,
            end_score,
            payout,
        )

    def describe_level(self, level:str) -> str:
        return describe_signal_level_text(level)

    def choice_labels_for_vp(self, vp: int):
        physical = self.physical_by_role.get(vp)
        return choice_labels_for_physical(physical, self.signaler, self.judge)

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
        self.log.init_round_csv(self.session_id)
        self.update_role_assignments()
        if self.session_popup:
            self.session_popup.dismiss()
            self.session_popup = None
        self.log.emit_event('session_start', {'session_number': number}, player=None)
        self.log_round_start()
        self.apply_phase()
        self.update_info_labels()

    def log_round_start(self):
        if not self.session_configured:
            return
        self.log.emit_event('round_start', {
            'round': self.round,
            'block': self.current_block_info['index'] if self.current_block_info else None,
            'round_in_block': self.round_in_block if self.current_block_info else None,
            'payout': bool(self.current_round_has_stake),
            'signaler': self.signaler,
            'judge': self.judge,
            'vp_roles': self.role_by_physical.copy(),
        }, player=None)

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
            root.log.close_round_csv()

if __name__ == '__main__':
    TabletopApp().run()
 
