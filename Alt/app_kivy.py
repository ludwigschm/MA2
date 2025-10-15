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

        cfg = GameEngineConfig(
            session_id="S001",
            csv_path=str(base / "Paare1.csv"),
            db_path=str(base / "logs" / "events.sqlite3"),
            csv_log_path=str(base / "logs" / "events.csv"),
        )
        self.engine = GameEngine(cfg)

        # Kopfzeile
        header = BoxLayout(orientation="horizontal", size_hint_y=None, height=46, spacing=8)
        self.lbl_status = Label(text="", font_size=22)
        header.add_widget(self.lbl_status)
        self.add_widget(header)

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
        self.add_widget(self.board)

        Clock.schedule_interval(lambda dt: self.refresh(), 0.1)
        self.refresh()

    # ===== Helper =====
    def _img_for_value(self, val: Optional[int]) -> str:
        if val is None:
            return self.img_back
        p = self.card_dir / f"{val}.png"
        return str(p) if p.exists() else self.img_back

    def _player_for_vp(self, vp: VP) -> Player:
        r = self.engine.current.roles
        return Player.P1 if r.p1_is == vp else Player.P2

    def _revealed(self, vp: VP, idx: int) -> bool:
        st = self.engine.get_public_state()
        if st["roles"]["P1"] == vp.value:
            return st["p1_revealed"][idx]
        else:
            return st["p2_revealed"][idx]

    def _expected_reveal(self) -> Optional[tuple[VP, int]]:
        """Welcher VP ist als NÄCHSTES dran (rollenrichtig) und welche Karte (0/1)?
        Nutzt die Engine-Flags p1_revealed/p2_revealed (rollenbezogen)."""
        st = self.engine.get_public_state()
        if st["phase"] != "DEALING":
            return None
        # Rollen -> welcher VP ist gerade Spieler 1 / 2?
        vp_p1 = VP[st["roles"]["P1"]]  # 'VP1'/'VP2' -> VP.VP1/VP.VP2
        vp_p2 = VP[st["roles"]["P2"]]
        p1r0, p1r1 = st["p1_revealed"]
        p2r0, p2r1 = st["p2_revealed"]

        if not p1r0:
            return (vp_p1, 0)  # P1-K1
        if not p2r0:
            return (vp_p2, 0)  # P2-K1
        if not p1r1:
            return (vp_p1, 1)  # P1-K2
        if not p2r1:
            return (vp_p2, 1)  # P2-K2
        return None  # fertig


    # ===== Actions =====
    def _start_or_next_for_vp(self, vp: VP):
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
        try:
            p = self._player_for_vp(vp)
            self.engine.click_reveal_card(p, idx)
        except Exception as e:
            self.lbl_info.text = f"Reveal-Fehler ({vp.value}): {e}"
        self.refresh()

    def _signal_from_vp(self, vp: VP, level: SignalLevel):
        # Nur die VP, die gerade Spieler 1 ist, darf signalen
        if self.engine.current.roles.p1_is != vp:
            return
        try:
            self.engine.p1_signal(level)
        except Exception as e:
            self.lbl_info.text = f"Signal-Fehler ({vp.value}): {e}"
        self.refresh()

    def _call_from_vp(self, vp: VP, call: Call):
        # Nur die VP, die gerade Spieler 2 ist, darf callen
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
        st = self.engine.get_public_state()
        rs = self.engine.current

        self.lbl_status.text = f"Runde {st['round_index']+1} – Phase: {st['phase']}"
        self.lbl_roles.text = f"Rollen: S1={st['roles']['P1']} | S2={st['roles']['P2']}"
        self.lbl_info.text = st.get("outcome_reason") or ""

        # Labels unter den VPs
        self.lbl_vp1.text = f"VP1 ({'S1' if st['roles']['P1']=='VP1' else 'S2'})"
        self.lbl_vp2.text = f"VP2 ({'S1' if st['roles']['P2']=='VP2' else 'S2'})"

        ph = st["phase"]

        # Start/Next-Buttons je VP (aktiv nur, wenn in entsprechender Phase und dieser VP noch nicht gedrückt hat)
        # Welche Flags gelten für diesen VP?
        # VP1:
        is_vp1_p1 = (st["roles"]["P1"] == "VP1")
        vp1_ready_start = st["p1_ready"] if is_vp1_p1 else st["p2_ready"]
        vp1_ready_next  = st["next_ready_p1"] if is_vp1_p1 else st["next_ready_p2"]
        self.btn_start_vp1.text = "Start (VP1)" if ph == "WAITING_START" else "Weiter (VP1)"
        self.btn_start_vp1.disabled = not ((ph == "WAITING_START" and not vp1_ready_start) or (ph == "ROUND_DONE" and not vp1_ready_next))

        # VP2:
        is_vp2_p1 = (st["roles"]["P1"] == "VP2")
        vp2_ready_start = st["p1_ready"] if is_vp2_p1 else st["p2_ready"]
        vp2_ready_next  = st["next_ready_p1"] if is_vp2_p1 else st["next_ready_p2"]
        self.btn_start_vp2.text = "Start (VP2)" if ph == "WAITING_START" else "Weiter (VP2)"
        self.btn_start_vp2.disabled = not ((ph == "WAITING_START" and not vp2_ready_start) or (ph == "ROUND_DONE" and not vp2_ready_next))

        # Reveal-Buttons (Sequenz wird zusätzlich in der Engine gesichert)
        exp = self._expected_reveal()  # z. B. (VP.VP2, 0)
        if exp is None:
            # keine Reveal-Phase
            self.btn_vp1_c1.disabled = True
            self.btn_vp1_c2.disabled = True
            self.btn_vp2_c1.disabled = True
            self.btn_vp2_c2.disabled = True
        else:
            exp_vp, exp_idx = exp
            self.btn_vp1_c1.disabled = not (st["phase"] == "DEALING" and exp_vp == VP.VP1 and exp_idx == 0)
            self.btn_vp1_c2.disabled = not (st["phase"] == "DEALING" and exp_vp == VP.VP1 and exp_idx == 1)
            self.btn_vp2_c1.disabled = not (st["phase"] == "DEALING" and exp_vp == VP.VP2 and exp_idx == 0)
            self.btn_vp2_c2.disabled = not (st["phase"] == "DEALING" and exp_vp == VP.VP2 and exp_idx == 1)

        # Signal/Call je Seite aktivieren abhängig von der Rolle
        enable_sig_vp1 = (ph == "SIGNAL_WAIT" and is_vp1_p1)
        enable_call_vp1 = (ph == "CALL_WAIT" and not is_vp1_p1)
        self.btn_sig1_h.disabled = self.btn_sig1_m.disabled = self.btn_sig1_t.disabled = not enable_sig_vp1
        self.btn_call1_truth.disabled = self.btn_call1_bluff.disabled = not enable_call_vp1

        enable_sig_vp2 = (ph == "SIGNAL_WAIT" and is_vp2_p1)
        enable_call_vp2 = (ph == "CALL_WAIT" and not is_vp2_p1)
        self.btn_sig2_h.disabled = self.btn_sig2_m.disabled = self.btn_sig2_t.disabled = not enable_sig_vp2
        self.btn_call2_truth.disabled = self.btn_call2_bluff.disabled = not enable_call_vp2

        # Karten (eigene, aufgedeckte) links/rechts anzeigen
        v1_c1, v1_c2 = rs.plan.vp1_cards
        v2_c1, v2_c2 = rs.plan.vp2_cards
        self.vp1_img1.source = self._img_for_value(v1_c1) if self._revealed(VP.VP1, 0) else self.img_back
        self.vp1_img2.source = self._img_for_value(v1_c2) if self._revealed(VP.VP1, 1) else self.img_back
        self.vp2_img1.source = self._img_for_value(v2_c1) if self._revealed(VP.VP2, 0) else self.img_back
        self.vp2_img2.source = self._img_for_value(v2_c2) if self._revealed(VP.VP2, 1) else self.img_back

        # Mitte: beide Hände nach Call
        show_mid = ph in ("REVEAL_SCORE", "ROUND_DONE")
        self.mid_vp1_1.source = self._img_for_value(v1_c1) if show_mid else ""
        self.mid_vp1_2.source = self._img_for_value(v1_c2) if show_mid else ""
        self.mid_vp2_1.source = self._img_for_value(v2_c1) if show_mid else ""
        self.mid_vp2_2.source = self._img_for_value(v2_c2) if show_mid else ""


class TouchGameApp(App):
    def build(self):
        return TwoPlayerUI()


if __name__ == "__main__":
    TouchGameApp().run()
