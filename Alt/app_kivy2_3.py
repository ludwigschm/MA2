from __future__ import annotations
from pathlib import Path
from typing import Optional

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.config import Config

# Optional Vollbild
# Config.set('graphics', 'fullscreen', '1')

from game_engine import (
    GameEngine, GameEngineConfig,
    Player, VP, SignalLevel, Call, hand_value
)

# --- Wahrheitsregel (anpassbar) ---
def signal_truth_mapping(p1_value: int, level: SignalLevel) -> bool:
    if level == SignalLevel.HOCH:
        return p1_value >= 17
    if level == SignalLevel.MITTEL:
        return 13 <= p1_value <= 16
    if level == SignalLevel.TIEF:
        return p1_value <= 12
    return False


class TwoPlayerUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=8, padding=8, **kwargs)

        base = Path(__file__).resolve().parent
        self.card_dir = base / "Karten"
        self.img_back = str(self.card_dir / "back.png") if (self.card_dir / "back.png").exists() else ""

        self.base = base
        self.engine: Optional[GameEngine] = None
        self.session_popup: Optional[Popup] = None
        self._session_inputs = {}
        self.session_message = ""

        # Kopfzeile
        self.header = BoxLayout(orientation="horizontal", size_hint_y=None, height=46, spacing=8)
        self.lbl_status = Label(text="", font_size=22)
        self.lbl_points = Label(text="", font_size=20, size_hint_x=0.5)
        self.header.add_widget(self.lbl_status)
        self.header.add_widget(self.lbl_points)
        self.add_widget(self.header)

        # Container, der zwischen Spielbrett und Übergangsseite wechselt
        self.content_container = BoxLayout(orientation="vertical")
        self.add_widget(self.content_container)

        # Hauptbereich: VP1 links – Mitte – VP2 rechts
        self.board = GridLayout(cols=3, spacing=10, size_hint_y=0.82)

        # ===== VP1 (immer links) =====
        self.col_vp1 = BoxLayout(orientation="vertical", spacing=6)
        self.lbl_vp1 = Label(text="VP1", font_size=18, size_hint_y=None, height=26)
        self.col_vp1.add_widget(self.lbl_vp1)

        self.vp1_img1 = Image(size_hint_y=0.42)
        self.vp1_img2 = Image(size_hint_y=0.42)
        self.col_vp1.add_widget(self.vp1_img1)
        self.col_vp1.add_widget(self.vp1_img2)

        # Start/Weiter für VP1 (rollenrichtig)
        self.btn_start_vp1 = Button(text="Start (VP1)", size_hint_y=None, height=48,
                                    on_release=lambda *_: self._start_or_next_for_vp(VP.VP1))
        self.col_vp1.add_widget(self.btn_start_vp1)

        # Signal & Call unter VP1
        row_vp1_actions = GridLayout(cols=2, spacing=6, size_hint_y=None, height=96)
        # Signal (S1)
        box_sig1 = BoxLayout(orientation="vertical", spacing=2)
        box_sig1.add_widget(Label(text="Signal", size_hint_y=None, height=20))
        row_sig1 = BoxLayout(spacing=4)
        self.btn_sig1_h = Button(text="hoch", on_release=lambda *_: self._signal_from_vp(VP.VP1, SignalLevel.HOCH))
        self.btn_sig1_m = Button(text="mittel", on_release=lambda *_: self._signal_from_vp(VP.VP1, SignalLevel.MITTEL))
        self.btn_sig1_t = Button(text="tief", on_release=lambda *_: self._signal_from_vp(VP.VP1, SignalLevel.TIEF))
        row_sig1.add_widget(self.btn_sig1_h); row_sig1.add_widget(self.btn_sig1_m); row_sig1.add_widget(self.btn_sig1_t)
        box_sig1.add_widget(row_sig1)
        # Call (S2)
        box_call1 = BoxLayout(orientation="vertical", spacing=2)
        box_call1.add_widget(Label(text="Call", size_hint_y=None, height=20))
        row_call1 = BoxLayout(spacing=4)
        self.btn_call1_truth = Button(text="Wahrheit", on_release=lambda *_: self._call_from_vp(VP.VP1, Call.WAHRHEIT))
        self.btn_call1_bluff = Button(text="Bluff", on_release=lambda *_: self._call_from_vp(VP.VP1, Call.BLUFF))
        row_call1.add_widget(self.btn_call1_truth); row_call1.add_widget(self.btn_call1_bluff)
        box_call1.add_widget(row_call1)
        row_vp1_actions.add_widget(box_sig1)
        row_vp1_actions.add_widget(box_call1)
        self.col_vp1.add_widget(row_vp1_actions)

        # Aufdeck-Buttons für VP1
        self.btn_vp1_c1 = Button(text="VP1: Karte 1", size_hint_y=None, height=48,
                                 on_release=lambda *_: self._reveal(VP.VP1, 0))
        self.btn_vp1_c2 = Button(text="VP1: Karte 2", size_hint_y=None, height=48,
                                 on_release=lambda *_: self._reveal(VP.VP1, 1))
        self.col_vp1.add_widget(self.btn_vp1_c1)
        self.col_vp1.add_widget(self.btn_vp1_c2)

        # ===== Mitte (beide Hände nach Call) =====
        self.center_box = BoxLayout(orientation="vertical", spacing=6)
        self.lbl_roles = Label(text="", font_size=16, size_hint_y=None, height=24)
        self.center_box.add_widget(self.lbl_roles)
        self.lbl_next = Label(text="", font_size=16, size_hint_y=None, height=22)
        self.center_box.add_widget(self.lbl_next)

        grid_mid = GridLayout(cols=2, spacing=6, size_hint_y=0.9)
        # VP1 Hand
        self.mid_vp1_1, self.mid_vp1_2 = Image(), Image()
        box_mid1 = BoxLayout(orientation="vertical")
        box_mid1.add_widget(Label(text="VP1-Hand", size_hint_y=None, height=20))
        box_mid1.add_widget(self.mid_vp1_1)
        box_mid1.add_widget(self.mid_vp1_2)
        # VP2 Hand
        self.mid_vp2_1, self.mid_vp2_2 = Image(), Image()
        box_mid2 = BoxLayout(orientation="vertical")
        box_mid2.add_widget(Label(text="VP2-Hand", size_hint_y=None, height=20))
        box_mid2.add_widget(self.mid_vp2_1)
        box_mid2.add_widget(self.mid_vp2_2)
        grid_mid.add_widget(box_mid1); grid_mid.add_widget(box_mid2)
        self.center_box.add_widget(grid_mid)

        self.lbl_info = Label(text="", font_size=18, size_hint_y=None, height=28)
        self.center_box.add_widget(self.lbl_info)

        # ===== VP2 (immer rechts) =====
        self.col_vp2 = BoxLayout(orientation="vertical", spacing=6)
        self.lbl_vp2 = Label(text="VP2", font_size=18, size_hint_y=None, height=26)
        self.col_vp2.add_widget(self.lbl_vp2)

        self.vp2_img1 = Image(size_hint_y=0.42)
        self.vp2_img2 = Image(size_hint_y=0.42)
        self.col_vp2.add_widget(self.vp2_img1)
        self.col_vp2.add_widget(self.vp2_img2)

        # Start/Weiter für VP2
        self.btn_start_vp2 = Button(text="Start (VP2)", size_hint_y=None, height=48,
                                    on_release=lambda *_: self._start_or_next_for_vp(VP.VP2))
        self.col_vp2.add_widget(self.btn_start_vp2)

        # Signal & Call unter VP2
        row_vp2_actions = GridLayout(cols=2, spacing=6, size_hint_y=None, height=96)
        # Signal (S1)
        box_sig2 = BoxLayout(orientation="vertical", spacing=2)
        box_sig2.add_widget(Label(text="Signal", size_hint_y=None, height=20))
        row_sig2 = BoxLayout(spacing=4)
        self.btn_sig2_h = Button(text="hoch", on_release=lambda *_: self._signal_from_vp(VP.VP2, SignalLevel.HOCH))
        self.btn_sig2_m = Button(text="mittel", on_release=lambda *_: self._signal_from_vp(VP.VP2, SignalLevel.MITTEL))
        self.btn_sig2_t = Button(text="tief", on_release=lambda *_: self._signal_from_vp(VP.VP2, SignalLevel.TIEF))
        row_sig2.add_widget(self.btn_sig2_h); row_sig2.add_widget(self.btn_sig2_m); row_sig2.add_widget(self.btn_sig2_t)
        box_sig2.add_widget(row_sig2)
        # Call (S2)
        box_call2 = BoxLayout(orientation="vertical", spacing=2)
        box_call2.add_widget(Label(text="Call", size_hint_y=None, height=20))
        row_call2 = BoxLayout(spacing=4)
        self.btn_call2_truth = Button(text="Wahrheit", on_release=lambda *_: self._call_from_vp(VP.VP2, Call.WAHRHEIT))
        self.btn_call2_bluff = Button(text="Bluff", on_release=lambda *_: self._call_from_vp(VP.VP2, Call.BLUFF))
        row_call2.add_widget(self.btn_call2_truth); row_call2.add_widget(self.btn_call2_bluff)
        box_call2.add_widget(row_call2)
        row_vp2_actions.add_widget(box_sig2)
        row_vp2_actions.add_widget(box_call2)
        self.col_vp2.add_widget(row_vp2_actions)

        # Aufdeck-Buttons für VP2
        self.btn_vp2_c1 = Button(text="VP2: Karte 1", size_hint_y=None, height=48,
                                 on_release=lambda *_: self._reveal(VP.VP2, 0))
        self.btn_vp2_c2 = Button(text="VP2: Karte 2", size_hint_y=None, height=48,
                                 on_release=lambda *_: self._reveal(VP.VP2, 1))
        self.col_vp2.add_widget(self.btn_vp2_c1)
        self.col_vp2.add_widget(self.btn_vp2_c2)

        # Board zusammenbauen
        self.board.add_widget(self.col_vp1)
        self.board.add_widget(self.center_box)
        self.board.add_widget(self.col_vp2)
        self.content_container.add_widget(self.board)

        # Übergangsseite (Blank Page) vorbereiten
        self.transition_box = BoxLayout(orientation="vertical", spacing=20, padding=40)
        self.lbl_transition = Label(text="", font_size=22)
        self.transition_box.add_widget(self.lbl_transition)
        btn_row = BoxLayout(size_hint_y=None, height=60, spacing=40)
        self.btn_transition_vp1 = Button(size_hint=(0.45, 1))
        self.btn_transition_vp2 = Button(size_hint=(0.45, 1))
        self.btn_transition_vp1.bind(on_release=lambda *_: self._continue_after_block(VP.VP1))
        self.btn_transition_vp2.bind(on_release=lambda *_: self._continue_after_block(VP.VP2))
        btn_row.add_widget(self.btn_transition_vp1)
        btn_row.add_widget(self.btn_transition_vp2)
        self.transition_box.add_widget(btn_row)

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
        self.btn_transition_vp1.text = f"{button_text} (VP1)"
        self.btn_transition_vp2.text = f"{button_text} (VP2)"
        self.btn_transition_vp1.disabled = False
        self.btn_transition_vp2.disabled = False
        self.transition_ready_vp1 = False
        self.transition_ready_vp2 = False
        self.lbl_transition.text = message
        if self.board.parent is self.content_container:
            self.content_container.remove_widget(self.board)
        if self.transition_box.parent is None:
            self.content_container.add_widget(self.transition_box)
        self.in_transition = True
        self.lbl_points.text = ""

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
            self.lbl_info.text = f"CSV {block_info['csv']} nicht gefunden."
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
            self.btn_transition_vp1.disabled = True
        elif vp == VP.VP2:
            self.transition_ready_vp2 = True
            self.btn_transition_vp2.disabled = True

        if not (self.transition_ready_vp1 and self.transition_ready_vp2):
            return

        if self.transition_final and self.next_block_idx >= len(self.block_sequence):
            self.lbl_info.text = "Experiment abgeschlossen."
            self.lbl_status.text = "Experiment abgeschlossen."
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
            self.lbl_info.text = f"Start/Next-Fehler ({vp.value}): {e}"
        self.refresh()

    def _reveal(self, vp: VP, idx: int):
        if not self.engine:
            return
        try:
            p = self._player_for_vp(vp)
            self.engine.click_reveal_card(p, idx)
        except Exception as e:
            self.lbl_info.text = f"Reveal-Fehler ({vp.value} K{idx+1}): {e}"
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
            self.lbl_info.text = f"Signal-Fehler ({vp.value}): {e}"
        self.refresh()

    def _call_from_vp(self, vp: VP, call: Call):
        # Nur die VP, die gerade Spieler 2 ist, darf callen
        if not self.engine:
            return
        if self.engine.current.roles.p2_is != vp:
            return
        rs = self.engine.current
        p1_val = hand_value(*rs.plan.vp1_cards if rs.roles.p1_is == VP.VP1 else rs.plan.vp2_cards)
        truth = signal_truth_mapping(p1_val, rs.p1_signal) if rs.p1_signal else None
        try:
            self.engine.p2_call(call, p1_hat_wahrheit_gesagt=truth)
        except Exception as e:
            self.lbl_info.text = f"Call-Fehler ({vp.value}): {e}"
        self.refresh()

    # ===== Refresh =====
    def refresh(self):
        if self.in_transition:
            self.lbl_status.text = self.transition_message or "Blockpause"
            self.lbl_roles.text = ""
            self.lbl_next.text = "Nächster Zug: –"
            self.lbl_info.text = self.transition_message
            self.lbl_points.text = ""
            for btn in [
                self.btn_start_vp1, self.btn_start_vp2,
                self.btn_vp1_c1, self.btn_vp1_c2,
                self.btn_vp2_c1, self.btn_vp2_c2,
                self.btn_sig1_h, self.btn_sig1_m, self.btn_sig1_t,
                self.btn_sig2_h, self.btn_sig2_m, self.btn_sig2_t,
                self.btn_call1_truth, self.btn_call1_bluff,
                self.btn_call2_truth, self.btn_call2_bluff,
            ]:
                btn.disabled = True
            for img in [self.vp1_img1, self.vp1_img2, self.vp2_img1, self.vp2_img2]:
                img.source = self.img_back
                img.reload()
            for img in [self.mid_vp1_1, self.mid_vp1_2, self.mid_vp2_1, self.mid_vp2_2]:
                img.source = ""
                img.reload()
            return

        if not self.engine:
            self.lbl_status.text = "Session wählen"
            self.lbl_roles.text = ""
            self.lbl_next.text = "Nächster Zug: –"
            self.lbl_info.text = "Bitte Sessiondaten bestätigen."
            self.lbl_points.text = ""
            self.btn_start_vp1.disabled = True
            self.btn_start_vp2.disabled = True
            self.btn_vp1_c1.disabled = True
            self.btn_vp1_c2.disabled = True
            self.btn_vp2_c1.disabled = True
            self.btn_vp2_c2.disabled = True
            for btn in [
                self.btn_sig1_h, self.btn_sig1_m, self.btn_sig1_t,
                self.btn_sig2_h, self.btn_sig2_m, self.btn_sig2_t,
                self.btn_call1_truth, self.btn_call1_bluff,
                self.btn_call2_truth, self.btn_call2_bluff,
            ]:
                btn.disabled = True
            for img in [self.vp1_img1, self.vp1_img2, self.vp2_img1, self.vp2_img2]:
                img.source = self.img_back
                img.reload()
            for img in [self.mid_vp1_1, self.mid_vp1_2, self.mid_vp2_1, self.mid_vp2_2]:
                img.source = ""
                img.reload()
            return

        st = self.engine.get_public_state()
        if st.get("phase") == "FINISHED":
            self._handle_block_finished()
            return

        rs = self.engine.current

        self.lbl_status.text = f"Runde {st['round_index']+1} – Phase: {st['phase']}"
        self.lbl_roles.text = f"Rollen: S1={st['roles']['P1']} | S2={st['roles']['P2']}"
        outcome = st.get("outcome_reason")
        if outcome:
            self.lbl_info.text = outcome
        else:
            self.lbl_info.text = self.session_message

        scores = st.get("scores")
        if scores:
            self.lbl_points.text = (
                f"Punkte – VP1: {scores.get('VP1', '')} | VP2: {scores.get('VP2', '')}"
            )
        else:
            self.lbl_points.text = ""

        # S1/S2-Beschriftung korrekt je Seite
        is_vp1_p1 = (st["roles"]["P1"] == "VP1")
        is_vp2_p1 = (st["roles"]["P1"] == "VP2")
        self.lbl_vp1.text = f"VP1 ({'S1' if is_vp1_p1 else 'S2'})"
        self.lbl_vp2.text = f"VP2 ({'S1' if is_vp2_p1 else 'S2'})"

        ph = st["phase"]

        # Start/Next-Buttons (aktiv, bis jeweilige Seite geklickt hat)
        vp1_ready_start = st["p1_ready"] if is_vp1_p1 else st["p2_ready"]
        vp1_ready_next  = st["next_ready_p1"] if is_vp1_p1 else st["next_ready_p2"]
        self.btn_start_vp1.text = "Start (VP1)" if ph == "WAITING_START" else "Weiter (VP1)"
        self.btn_start_vp1.disabled = not ((ph == "WAITING_START" and not vp1_ready_start) or
                                           (ph == "ROUND_DONE" and not vp1_ready_next))

        vp2_ready_start = st["p1_ready"] if is_vp2_p1 else st["p2_ready"]
        vp2_ready_next  = st["next_ready_p1"] if is_vp2_p1 else st["next_ready_p2"]
        self.btn_start_vp2.text = "Start (VP2)" if ph == "WAITING_START" else "Weiter (VP2)"
        self.btn_start_vp2.disabled = not ((ph == "WAITING_START" and not vp2_ready_start) or
                                           (ph == "ROUND_DONE" and not vp2_ready_next))

        # --- Reveal-Buttons: nur der ERWARTETE Schritt ist aktiv ---
        exp = self._expected_reveal()
        if exp is None:
            self.lbl_next.text = "Nächster Zug: –"
            self.btn_vp1_c1.disabled = True
            self.btn_vp1_c2.disabled = True
            self.btn_vp2_c1.disabled = True
            self.btn_vp2_c2.disabled = True
        else:
            exp_vp, exp_idx = exp
            self.lbl_next.text = f"Nächster Zug: {exp_vp.value} • Karte {exp_idx+1}"
            self.btn_vp1_c1.disabled = not (ph == "DEALING" and exp_vp == VP.VP1 and exp_idx == 0)
            self.btn_vp1_c2.disabled = not (ph == "DEALING" and exp_vp == VP.VP1 and exp_idx == 1)
            self.btn_vp2_c1.disabled = not (ph == "DEALING" and exp_vp == VP.VP2 and exp_idx == 0)
            self.btn_vp2_c2.disabled = not (ph == "DEALING" and exp_vp == VP.VP2 and exp_idx == 1)

        # Signal/Call je Seite abhängig von der Rolle
        enable_sig_vp1 = (ph == "SIGNAL_WAIT" and is_vp1_p1)
        enable_call_vp1 = (ph == "CALL_WAIT" and not is_vp1_p1)
        self.btn_sig1_h.disabled = self.btn_sig1_m.disabled = self.btn_sig1_t.disabled = not enable_sig_vp1
        self.btn_call1_truth.disabled = self.btn_call1_bluff.disabled = not enable_call_vp1

        enable_sig_vp2 = (ph == "SIGNAL_WAIT" and is_vp2_p1)
        enable_call_vp2 = (ph == "CALL_WAIT" and not is_vp2_p1)
        self.btn_sig2_h.disabled = self.btn_sig2_m.disabled = self.btn_sig2_t.disabled = not enable_sig_vp2
        self.btn_call2_truth.disabled = self.btn_call2_bluff.disabled = not enable_call_vp2

        # Karten (eigene) links/rechts anzeigen
        v1_c1, v1_c2 = rs.plan.vp1_cards
        v2_c1, v2_c2 = rs.plan.vp2_cards

        show_mid = ph in ("REVEAL_SCORE", "ROUND_DONE")  # ab Reveal beide Hände sicher zeigen

        self.vp1_img1.source = self._img_for_value(v1_c1) if (self._revealed(VP.VP1, 0) or show_mid) else self.img_back
        self.vp1_img2.source = self._img_for_value(v1_c2) if (self._revealed(VP.VP1, 1) or show_mid) else self.img_back
        self.vp2_img1.source = self._img_for_value(v2_c1) if (self._revealed(VP.VP2, 0) or show_mid) else self.img_back
        self.vp2_img2.source = self._img_for_value(v2_c2) if (self._revealed(VP.VP2, 1) or show_mid) else self.img_back

        # WICHTIG: Bilder neu laden (Cache umgehen)
        self.vp1_img1.reload(); self.vp1_img2.reload(); self.vp2_img1.reload(); self.vp2_img2.reload()

        # Mitte: beide Hände nach Call
        self.mid_vp1_1.source = self._img_for_value(v1_c1) if show_mid else ""
        self.mid_vp1_2.source = self._img_for_value(v1_c2) if show_mid else ""
        self.mid_vp2_1.source = self._img_for_value(v2_c1) if show_mid else ""
        self.mid_vp2_2.source = self._img_for_value(v2_c2) if show_mid else ""


class TouchGameApp(App):
    def build(self):
        return TwoPlayerUI()

    def on_stop(self):
        root = self.root
        if root and getattr(root, "engine", None):
            root.engine.close()


if __name__ == "__main__":
    TouchGameApp().run()
