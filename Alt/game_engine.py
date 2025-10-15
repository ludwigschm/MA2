# game_engine.py  (rollen-tausch + vollständiges Logging)
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple
import csv, time, json, sqlite3, pathlib
from datetime import datetime, timezone

# ---------------- Enums ----------------

class Phase(Enum):
    WAITING_START = auto()   # (nur Runde 1) beide drücken "Runde beginnen"
    DEALING = auto()         # abwechselnd: S1-K1 -> S2-K1 -> S1-K2 -> S2-K2
    SIGNAL_WAIT = auto()     # Spieler 1 (aktuell) signalisiert hoch/mittel/tief
    CALL_WAIT = auto()       # Spieler 2 (aktuell) sagt Wahrheit/Bluff
    REVEAL_SCORE = auto()    # beide Hände offen + Scoring
    ROUND_DONE = auto()      # Anzeige Ergebnis; warten auf "Nächste Runde"
    FINISHED = auto()        # kein Eintrag mehr in CSV

class Player(Enum):
    P1 = "P1"
    P2 = "P2"

class VP(Enum):
    VP1 = "VP1"
    VP2 = "VP2"

class SignalLevel(Enum):
    HOCH = "hoch"
    MITTEL = "mittel"
    TIEF = "tief"

class Call(Enum):
    WAHRHEIT = "wahrheit"
    BLUFF = "bluff"

# -------------- Strukturen --------------

@dataclass
class RoundPlan:
    # Karten sind VP-bezogen (nicht rollenbezogen!)
    vp1_cards: Tuple[int, int]
    vp2_cards: Tuple[int, int]

@dataclass
class RoleMap:
    # Welche VP spielt in dieser Runde Spieler-1/Spieler-2?
    p1_is: VP
    p2_is: VP

@dataclass
class VisibleCardState:
    p1_revealed: Tuple[bool, bool] = (False, False)  # rollenbezogen
    p2_revealed: Tuple[bool, bool] = (False, False)

@dataclass
class RoundState:
    index: int
    plan: RoundPlan                # VP-bezogene Karten
    roles: RoleMap                 # Rollenzuordnung für diese Runde
    phase: Phase = Phase.WAITING_START
    p1_ready: bool = False         # für "Runde beginnen" (nur Runde 1)
    p2_ready: bool = False
    next_ready_p1: bool = False    # für "Nächste Runde" (ab Runde 1)
    next_ready_p2: bool = False
    vis: VisibleCardState = field(default_factory=VisibleCardState)
    p1_signal: Optional[SignalLevel] = None
    p2_call: Optional[Call] = None
    winner: Optional[Player] = None
    outcome_reason: Optional[str] = None

# -------------- CSV-Lader --------------

class RoundSchedule:
    """
    CSV: pro Zeile eine Runde. Spalten 2–5 (Index 1–4) -> VP1; Spalten 6–10 (Index 5–9) -> VP2.
    Jeweils die ersten zwei nicht-leeren Einträge als Karten (int).
    """
    def __init__(self, csv_path: str):
        self.rounds: List[RoundPlan] = self._load(csv_path)

    def _parse_two(self, row: List[str], start: int, end: int) -> Tuple[int,int]:
        vals = []
        for i in range(start, min(end, len(row))):
            cell = (row[i] or "").strip()
            if not cell: continue
            try:
                vals.append(int(cell))
            except ValueError:
                continue
            if len(vals) == 2: break
        if len(vals) < 2:
            raise ValueError(f"Zu wenige Karten in Spalten {start+1}–{end}.")
        return vals[0], vals[1]

    def _load(self, path: str) -> List[RoundPlan]:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        start = 0
        try:
            self._parse_two(rows[0], 1, 5); self._parse_two(rows[0], 7, 11)
        except Exception:
            start = 1
        out = []
        for r in rows[start:]:
            if not r or all((c or "").strip()=="" for c in r): continue
            vp1 = self._parse_two(r, 1, 5)
            vp2 = self._parse_two(r, 7, 11)
            out.append(RoundPlan(vp1_cards=vp1, vp2_cards=vp2))
        if not out: raise ValueError("Keine Runden in CSV gefunden.")
        return out

# -------------- Logger --------------

class EventLogger:
    def __init__(self, db_path: str, csv_path: Optional[str] = None):
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS events(
          session_id TEXT, round_idx INT, phase TEXT, actor TEXT, action TEXT,
          payload TEXT, t_mono_ns INTEGER, t_utc_iso TEXT
        )""")
        self.conn.commit()
        self.csv_fp = None
        if csv_path:
            pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            self.csv_fp = open(csv_path, "a", encoding="utf-8", newline="")

    def log(self, session_id: str, round_idx: int, phase: Phase,
            actor: str, action: str, payload: Dict[str, Any]):
        t_mono_ns = time.perf_counter_ns()
        t_utc_iso = datetime.now(timezone.utc).isoformat()
        row = (session_id, round_idx, phase.name, actor, action,
               json.dumps(payload, ensure_ascii=False), t_mono_ns, t_utc_iso)
        cur = self.conn.cursor()
        cur.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", row)
        self.conn.commit()
        if self.csv_fp:
            csv.writer(self.csv_fp).writerow(row); self.csv_fp.flush()

    def close(self):
        if self.csv_fp: self.csv_fp.close()
        self.conn.close()

# -------------- Engine --------------

def hand_value(a: int, b: int) -> int:
    s = a + b
    return 0 if s in (20, 21, 22) else s

@dataclass
class GameEngineConfig:
    session_id: str
    csv_path: str
    db_path: str = "logs/events.sqlite3"
    csv_log_path: Optional[str] = "logs/events.csv"

class GameEngine:
    """
    - Runde 1: beide drücken "Runde beginnen" (WAITING_START -> DEALING).
    - Dealing: S1-K1 → S2-K1 → S1-K2 → S2-K2 (rollenbezogen).
    - P1 signalisiert; P2 callt; Scoring + Reveal.
    - Beide drücken "Nächste Runde" -> Rollen werden getauscht, nächste Runde startet in DEALING.
    - CSV ist VP-bezogen; Karten pro Runde werden via aktueller Rollen-zu-VP-Mapping gezogen.
    """
    def __init__(self, cfg: GameEngineConfig):
        self.cfg = cfg
        self.schedule = RoundSchedule(cfg.csv_path)
        self.logger = EventLogger(cfg.db_path, cfg.csv_log_path)
        # Runde 1: VP1 ist Spieler 1, VP2 ist Spieler 2
        roles = RoleMap(p1_is=VP.VP1, p2_is=VP.VP2)
        self.round_idx = 0
        self.current = RoundState(index=0, plan=self.schedule.rounds[0], roles=roles)

    # --- Hilfen ---
    def _ensure(self, allowed: List[Phase]):
        if self.current.phase not in allowed:
            raise RuntimeError(f"Falsche Phase: {self.current.phase.name}")

    def _log(self, actor: str, action: str, payload: Dict[str, Any]):
        self.logger.log(self.cfg.session_id, self.current.index, self.current.phase, actor, action, payload)

    def _cards_of(self, player: Player) -> Tuple[int,int]:
        # Hole Karten der VP, die aktuell diese Spielerrolle hat
        vp = self.current.roles.p1_is if player == Player.P1 else self.current.roles.p2_is
        return (self.current.plan.vp1_cards if vp == VP.VP1 else self.current.plan.vp2_cards)

    # --- Öffentliche API (UI) ---

    def click_start(self, player: Player):
        """ Beide drücken 'Runde beginnen' (nur in Runde 1 relevant). """
        self._ensure([Phase.WAITING_START])
        if player == Player.P1 and not self.current.p1_ready:
            self.current.p1_ready = True; self._log("P1", "start_click", {})
        if player == Player.P2 and not self.current.p2_ready:
            self.current.p2_ready = True; self._log("P2", "start_click", {})
        if self.current.p1_ready and self.current.p2_ready:
            self.current.phase = Phase.DEALING
            self._log("SYS", "phase_change", {"to": "DEALING"})

    def click_reveal_card(self, player: Player, card_idx: int):
        """ Rollenbezogenes Aufdecken in fixer Reihenfolge. """
        self._ensure([Phase.DEALING])
        if card_idx not in (0,1): raise ValueError("card_idx ∈ {0,1}")
        v = self.current.vis

        # Reihenfolge erzwingen:
        if not v.p1_revealed[0]:
            if player != Player.P1 or card_idx != 0:
                raise RuntimeError("Zuerst: Spieler 1, Karte 1.")
            v.p1_revealed = (True, v.p1_revealed[1])
        elif not v.p2_revealed[0]:
            if player != Player.P2 or card_idx != 0:
                raise RuntimeError("Zweitens: Spieler 2, Karte 1.")
            v.p2_revealed = (True, v.p2_revealed[1])
        elif not v.p1_revealed[1]:
            if player != Player.P1 or card_idx != 1:
                raise RuntimeError("Drittens: Spieler 1, Karte 2.")
            v.p1_revealed = (v.p1_revealed[0], True)
        elif not v.p2_revealed[1]:
            if player != Player.P2 or card_idx != 1:
                raise RuntimeError("Viertens: Spieler 2, Karte 2.")
            v.p2_revealed = (v.p2_revealed[0], True)
            self.current.phase = Phase.SIGNAL_WAIT
            self._log("SYS", "phase_change", {"to": "SIGNAL_WAIT"})
        else:
            raise RuntimeError("Alle Karten bereits aufgedeckt.")

        # Wert loggen (rollenrichtig, aber VP-Karte)
        c1, c2 = self._cards_of(player)
        val = c1 if card_idx == 0 else c2
        self._log(player.value, "reveal_card", {
            "card_idx": card_idx, "value": val,
            "role_vp": (self.current.roles.p1_is.value if player==Player.P1 else self.current.roles.p2_is.value)
        })

    def p1_signal(self, level: SignalLevel):
        self._ensure([Phase.SIGNAL_WAIT])
        if self.current.p1_signal is not None:
            raise RuntimeError("Signal bereits gesetzt.")
        self.current.p1_signal = level
        self._log("P1", "signal", {"level": level.value})
        self.current.phase = Phase.CALL_WAIT
        self._log("SYS", "phase_change", {"to": "CALL_WAIT"})

    def p2_call(self, call: Call, p1_hat_wahrheit_gesagt: Optional[bool]):
        self._ensure([Phase.CALL_WAIT])
        if self.current.p2_call is not None:
            raise RuntimeError("Call bereits gesetzt.")
        self.current.p2_call = call
        self._log("P2", "call", {"call": call.value})

        winner, reason = self._compute_winner(call, p1_hat_wahrheit_gesagt)
        self.current.winner = winner
        self.current.outcome_reason = reason

        # Für Reveal/Score: echte Kartenwerte beider VPs
        vp1 = self.current.plan.vp1_cards
        vp2 = self.current.plan.vp2_cards
        self.current.phase = Phase.REVEAL_SCORE
        self._log("SYS", "reveal_and_score", {
            "winner": None if winner is None else winner.value,
            "reason": reason,
            "vp1_cards": vp1, "vp2_cards": vp2,
            "vp1_value": hand_value(*vp1), "vp2_value": hand_value(*vp2),
            "roles": {"P1": self.current.roles.p1_is.value, "P2": self.current.roles.p2_is.value}
        })

        self.current.phase = Phase.ROUND_DONE
        self._log("SYS", "phase_change", {"to": "ROUND_DONE"})

    def click_next_round(self, player: Player):
        """ Beide drücken 'Nächste Runde'. Danach: Rollen tauschen, nächste Runde → DEALING. """
        self._ensure([Phase.ROUND_DONE])
        if player == Player.P1 and not self.current.next_ready_p1:
            self.current.next_ready_p1 = True; self._log("P1", "next_round_click", {})
        if player == Player.P2 and not self.current.next_ready_p2:
            self.current.next_ready_p2 = True; self._log("P2", "next_round_click", {})

        if self.current.next_ready_p1 and self.current.next_ready_p2:
            self._advance_and_swap_roles()

    # --- State-Exposure ---

    def get_public_state(self) -> Dict[str, Any]:
        rs = self.current
        return {
            "round_index": rs.index,
            "phase": rs.phase.name,
            "roles": {"P1": rs.roles.p1_is.value, "P2": rs.roles.p2_is.value},  # welche VP spielt welche Rolle
            "p1_ready": rs.p1_ready, "p2_ready": rs.p2_ready,
            "next_ready_p1": rs.next_ready_p1, "next_ready_p2": rs.next_ready_p2,
            "p1_revealed": rs.vis.p1_revealed, "p2_revealed": rs.vis.p2_revealed,
            "p1_signal": None if rs.p1_signal is None else rs.p1_signal.value,
            "p2_call": None if rs.p2_call is None else rs.p2_call.value,
            "winner": None if rs.winner is None else rs.winner.value,
            "outcome_reason": rs.outcome_reason,
        }

    # --- Interna ---

    def _compute_winner(self, call: Call, p1_truth: Optional[bool]) -> Tuple[Optional[Player], str]:
        if p1_truth is None:
            return (None, "Unbestimmt: p1_hat_wahrheit_gesagt fehlt (Signal→Wahrheit-Mapping notwendig).")

        # „P1“/„P2“ sind hier rollenspezifisch (aktueller Spieler 1/2)
        # Handwerte: über VPs ermitteln
        vp1_val = hand_value(*self.current.plan.vp1_cards)
        vp2_val = hand_value(*self.current.plan.vp2_cards)

        # Wer ist aktuell Spieler 1? (kann VP1 oder VP2 sein)
        p1_vp = self.current.roles.p1_is
        p2_vp = self.current.roles.p2_is
        p1_val = vp1_val if p1_vp == VP.VP1 else vp2_val
        p2_val = vp2_val if p2_vp == VP.VP2 else vp1_val

        if p1_truth and call == Call.BLUFF:
            return (Player.P1, "P1 sagte Wahrheit, P2 sagte Bluff → P1 gewinnt.")
        if (not p1_truth) and call == Call.BLUFF:
            return (Player.P2, "P1 bluffte, P2 erkannte Bluff → P2 gewinnt.")

        # call == Wahrheit (P2 glaubt)
        if call == Call.WAHRHEIT:
            if p1_truth:
                if p1_val > p2_val:
                    return (Player.P1, f"P2 glaubte, Vergleich: {p1_val} vs {p2_val} → P1 gewinnt.")
                elif p2_val > p1_val:
                    return (Player.P2, f"P2 glaubte, Vergleich: {p1_val} vs {p2_val} → P2 gewinnt.")
                else:
                    return (None, f"P2 glaubte, Vergleich: {p1_val} vs {p2_val} → Unentschieden.")
            else:
                return (Player.P1, "P1 bluffte, P2 glaubte → P1 gewinnt.")

        return (None, "Unerwartete Konstellation.")

    def _advance_and_swap_roles(self):
        self.round_idx += 1
        if self.round_idx >= len(self.schedule.rounds):
            self.current.phase = Phase.FINISHED
            self._log("SYS", "phase_change", {"to": "FINISHED"})
            return

        # Rollen tauschen:
        new_roles = RoleMap(p1_is=self.current.roles.p2_is, p2_is=self.current.roles.p1_is)
        self.current = RoundState(
            index=self.round_idx,
            plan=self.schedule.rounds[self.round_idx],
            roles=new_roles,
            phase=Phase.DEALING  # nächste Runde beginnt direkt mit Aufdecken
        )
        self._log("SYS", "phase_change", {
            "to": "DEALING",
            "roles": {"P1": new_roles.p1_is.value, "P2": new_roles.p2_is.value}
        })

    # --- Cleanup ---

    def close(self):
        self.logger.close()



# --- Datei-Ende: Demo/Startkonfiguration robust machen ---
from pathlib import Path
import os

def main():
    BASE = Path(__file__).resolve().parent
    CSV_PATH = BASE / "Paare1.csv"
    LOG_DIR = BASE / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    print("Arbeitsverzeichnis (cwd):", os.getcwd())
    print("Skriptordner (__file__):", BASE)
    print("Erwarte CSV unter:", CSV_PATH)
    print("Existiert CSV?", CSV_PATH.exists())

    cfg = GameEngineConfig(
        session_id="S001",
        csv_path=str(CSV_PATH),                          # <- absoluter Pfad
        db_path=str(LOG_DIR / "events.sqlite3"),
        csv_log_path=str(LOG_DIR / "events.csv"),
    )

    eng = GameEngine(cfg)
    try:
        print("Init:", eng.get_public_state())

        # Runde 1: beide starten
        eng.click_start(Player.P1); eng.click_start(Player.P2)

        # Aufdecken 1
        eng.click_reveal_card(Player.P1, 0)
        eng.click_reveal_card(Player.P2, 0)
        eng.click_reveal_card(Player.P1, 1)
        eng.click_reveal_card(Player.P2, 1)

        # Signal & Call (Beispiel)
        eng.p1_signal(SignalLevel.MITTEL)
        eng.p2_call(Call.WAHRHEIT, p1_hat_wahrheit_gesagt=True)
        print("Round done:", eng.get_public_state())

        # Nächste Runde (beide drücken)
        eng.click_next_round(Player.P1); eng.click_next_round(Player.P2)
        print("Nach Rollentausch:", eng.get_public_state())
    finally:
        eng.close()

if __name__ == "__main__":
    main()
