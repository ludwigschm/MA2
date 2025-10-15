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
    UEBERSPIEL = "überspiel"

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
    CSV: pro Zeile eine Runde. Für VP1 werden die Spalten 2–5 (Index 1–4) ausgewertet,
    für VP2 die Spalten 8–11 (Index 7–10). Aus dem jeweiligen Bereich werden die ersten
    beiden nicht-leeren Integer-Werte als Karten interpretiert.
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
        return {
            "session_id": session_id,
            "round_idx": round_idx,
            "phase": phase.name,
            "actor": actor,
            "action": action,
            "payload": payload,
            "t_utc_iso": t_utc_iso,
        }

    def close(self):
        if self.csv_fp: self.csv_fp.close()
        self.conn.close()

# -------------- Engine --------------

def hand_value(a: int, b: int) -> int:
    s = a + b
    return 0 if s in (20, 21, 22) else s


def hand_category(a: int, b: int) -> SignalLevel:
    total = a + b
    if total in (20, 21, 22):
        return SignalLevel.UEBERSPIEL
    if total == 19:
        return SignalLevel.HOCH
    if total in (17, 18):
        return SignalLevel.MITTEL
    if total in (14, 15, 16):
        return SignalLevel.TIEF
    # Falls Werte außerhalb des erwarteten Bereichs auftauchen, ordnen wir sie dem
    # nächsten sinnvollen Bereich zu, statt einen Laufzeitfehler zu riskieren.
    if total > 22:
        return SignalLevel.UEBERSPIEL
    if total >= 17:
        return SignalLevel.MITTEL
    return SignalLevel.TIEF

@dataclass
class SessionCsvLogger:
    HEADER = [
        "Spiel", "Block", "Bedingung", "Runde", "Spieler", "VP",
        "Karte1 VP1", "Karte2 VP1", "Karte1 VP2", "Karte2 VP2",
        "Taste", "Time", "Gewinner", "Punkte VP1", "Punkte VP2",
    ]

    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        self._fp = open(path, "a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fp)
        if new_file:
            self._writer.writerow(self.HEADER)
            self._fp.flush()

    def _action_label(self, actor: str, action: str, payload: Dict[str, Any]) -> str:
        if action == "start_click":
            return "Start"
        if action == "next_round_click":
            return "Weiter"
        if action == "signal":
            return payload.get("level", "")
        if action == "call":
            return payload.get("call", "")
        if action == "reveal_card":
            idx = payload.get("card_idx")
            if idx is not None:
                return f"Karte {idx + 1}"
        if action == "phase_change":
            return f"Phase → {payload.get('to', '')}"
        if action == "reveal_and_score":
            return "Reveal/Score"
        return action

    def log(self, cfg: "GameEngineConfig", rs: RoundState,
            actor: str, action: str, payload: Dict[str, Any], timestamp_iso: str,
            round_index_override: Optional[int] = None,
            scores: Optional[Dict[VP, int]] = None):
        if actor == "SYS":
            return
        if cfg.session_number is None:
            session_value = cfg.session_id
        else:
            session_value = cfg.session_number

        if actor == "P1":
            vp_actor = rs.roles.p1_is.value
        elif actor == "P2":
            vp_actor = rs.roles.p2_is.value
        else:
            vp_actor = ""

        vp1_cards = rs.plan.vp1_cards
        vp2_cards = rs.plan.vp2_cards

        winner = payload.get("winner") or ""
        if not winner and rs.winner is not None:
            winner = rs.winner.value

        round_idx = rs.index if round_index_override is None else round_index_override
        score_vp1 = ""
        score_vp2 = ""
        if scores:
            score_vp1 = scores.get(VP.VP1, "")
            score_vp2 = scores.get(VP.VP2, "")

        row = [
            session_value,
            cfg.block,
            cfg.condition,
            round_idx + 1,
            actor if actor != "SYS" else "",
            vp_actor,
            vp1_cards[0],
            vp1_cards[1],
            vp2_cards[0],
            vp2_cards[1],
            self._action_label(actor, action, payload),
            timestamp_iso,
            winner,
            score_vp1,
            score_vp2,
        ]
        self._writer.writerow(row)
        self._fp.flush()

    def close(self):
        self._fp.close()


@dataclass
class GameEngineConfig:
    session_id: str
    csv_path: str
    db_path: str = "logs/events.sqlite3"
    csv_log_path: Optional[str] = None
    session_number: Optional[int] = None
    block: int = 1
    condition: str = "no_payout"
    log_dir: str = "logs"
    payout: bool = False
    payout_start_points: int = 0

    def __post_init__(self):
        if self.session_number is None:
            digits = "".join(ch for ch in self.session_id if ch.isdigit())
            self.session_number = int(digits) if digits else None

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
        session_identifier = (
            cfg.session_number if cfg.session_number is not None else cfg.session_id
        )
        condition_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_"
                                   for ch in cfg.condition.lower())
        session_csv_path = pathlib.Path(cfg.log_dir) / (
            f"session_{session_identifier}_{condition_slug}.csv"
        )
        self.session_csv = SessionCsvLogger(session_csv_path)
        self.scores: Optional[Dict[VP, int]] = None
        if cfg.payout:
            start_points = cfg.payout_start_points
            self.scores = {VP.VP1: start_points, VP.VP2: start_points}
        # Runde 1: VP1 ist Spieler 1, VP2 ist Spieler 2
        roles = RoleMap(p1_is=VP.VP1, p2_is=VP.VP2)
        self.round_idx = 0
        self.current = RoundState(index=0, plan=self.schedule.rounds[0], roles=roles)

    # --- Hilfen ---
    def _ensure(self, allowed: List[Phase]):
        if self.current.phase not in allowed:
            raise RuntimeError(f"Falsche Phase: {self.current.phase.name}")

    def _score_snapshot(self) -> Optional[Dict[VP, int]]:
        if self.scores is None:
            return None
        return {VP.VP1: self.scores[VP.VP1], VP.VP2: self.scores[VP.VP2]}

    def _log(self, actor: str, action: str, payload: Dict[str, Any],
             round_index_override: Optional[int] = None):
        round_idx = self.current.index if round_index_override is None else round_index_override
        data = self.logger.log(
            self.cfg.session_id, round_idx, self.current.phase, actor, action, payload
        )
        self.session_csv.log(
            self.cfg, self.current, actor, action, payload, data["t_utc_iso"],
            round_index_override=round_idx, scores=self._score_snapshot()
        )

    def _cards_of(self, player: Player) -> Tuple[int,int]:
        # Hole Karten der VP, die aktuell diese Spielerrolle hat
        vp = self.current.roles.p1_is if player == Player.P1 else self.current.roles.p2_is
        return (self.current.plan.vp1_cards if vp == VP.VP1 else self.current.plan.vp2_cards)

    def _update_scores(self, winner: Optional[Player]):
        if self.scores is None or winner is None:
            return
        if winner == Player.P1:
            winner_vp = self.current.roles.p1_is
            loser_vp = self.current.roles.p2_is
        else:
            winner_vp = self.current.roles.p2_is
            loser_vp = self.current.roles.p1_is
        self.scores[winner_vp] += 1
        self.scores[loser_vp] -= 1

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
        if level == SignalLevel.UEBERSPIEL:
            raise ValueError("Das Signal 'überspiel' kann nicht aktiv gewählt werden.")
        self.current.p1_signal = level
        self._log("P1", "signal", {"level": level.value})
        self.current.phase = Phase.CALL_WAIT
        self._log("SYS", "phase_change", {"to": "CALL_WAIT"})

    def p2_call(self, call: Call, p1_hat_wahrheit_gesagt: Optional[bool]):
        self._ensure([Phase.CALL_WAIT])
        if self.current.p2_call is not None:
            raise RuntimeError("Call bereits gesetzt.")
        self.current.p2_call = call
        outcome = self._resolve_outcome(call)
        winner, reason, actual_truth = outcome
        self.current.winner = winner
        self.current.outcome_reason = reason
        self._update_scores(winner)

        payload_call: Dict[str, Any] = {"call": call.value}
        if actual_truth is not None:
            payload_call["p1_truth"] = actual_truth
        if (
            p1_hat_wahrheit_gesagt is not None
            and actual_truth is not None
            and bool(p1_hat_wahrheit_gesagt) != actual_truth
        ):
            payload_call["p1_truth_ui"] = bool(p1_hat_wahrheit_gesagt)
        if winner is not None:
            payload_call["winner"] = winner.value
        if self.scores is not None:
            payload_call["scores"] = {
                VP.VP1.value: self.scores[VP.VP1],
                VP.VP2.value: self.scores[VP.VP2],
            }
        self._log("P2", "call", payload_call)

        # Für Reveal/Score: echte Kartenwerte beider VPs
        vp1 = self.current.plan.vp1_cards
        vp2 = self.current.plan.vp2_cards
        self.current.phase = Phase.REVEAL_SCORE
        self._log("SYS", "reveal_and_score", {
            "winner": None if winner is None else winner.value,
            "reason": reason,
            "vp1_cards": vp1, "vp2_cards": vp2,
            "vp1_value": hand_value(*vp1), "vp2_value": hand_value(*vp2),
            "vp1_category": hand_category(*vp1).value,
            "vp2_category": hand_category(*vp2).value,
            "roles": {"P1": self.current.roles.p1_is.value, "P2": self.current.roles.p2_is.value}
        })

        self.current.phase = Phase.ROUND_DONE
        self._log("SYS", "phase_change", {"to": "ROUND_DONE"})

    def click_next_round(self, player: Player):
        """ Beide drücken 'Nächste Runde'. Danach: Rollen tauschen, nächste Runde → DEALING. """
        self._ensure([Phase.ROUND_DONE])
        next_round_idx = self.current.index
        if self.current.index + 1 < len(self.schedule.rounds):
            next_round_idx = self.current.index + 1
        if player == Player.P1 and not self.current.next_ready_p1:
            self.current.next_ready_p1 = True
            self._log("P1", "next_round_click", {}, round_index_override=next_round_idx)
        if player == Player.P2 and not self.current.next_ready_p2:
            self.current.next_ready_p2 = True
            self._log("P2", "next_round_click", {}, round_index_override=next_round_idx)

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
            "scores": None if self.scores is None else {
                VP.VP1.value: self.scores[VP.VP1],
                VP.VP2.value: self.scores[VP.VP2],
            },
        }

    # --- Interna ---

    def _determine_truth(self) -> Tuple[Optional[bool], Optional[SignalLevel], Optional[SignalLevel]]:
        signal = self.current.p1_signal
        if signal is None:
            return (None, None, None)
        p1_cards = self._cards_of(Player.P1)
        p2_cards = self._cards_of(Player.P2)
        p1_category = hand_category(*p1_cards)
        p2_category = hand_category(*p2_cards)
        return (signal == p1_category, p1_category, p2_category)

    def _resolve_outcome(self, call: Call) -> Tuple[Optional[Player], str, Optional[bool]]:
        actual_truth, p1_category, _ = self._determine_truth()

        if actual_truth is None:
            return (
                None,
                "Unbestimmt: Kein Signal gesetzt, Ergebnis kann nicht berechnet werden.",
                None,
            )

        # Werte für Vergleich (20/21/22 werden als 0 behandelt)
        p1_cards = self._cards_of(Player.P1)
        p2_cards = self._cards_of(Player.P2)
        p1_val = hand_value(*p1_cards)
        p2_val = hand_value(*p2_cards)

        if call == Call.BLUFF:
            if actual_truth:
                return (
                    Player.P1,
                    "P1 sagte die richtige Kategorie, P2 erwartete Bluff → P1 gewinnt.",
                    actual_truth,
                )
            return (
                Player.P2,
                "P1 bluffte über die Kategorie, P2 erkannte den Bluff → P2 gewinnt.",
                actual_truth,
            )

        # call == Wahrheit
        if actual_truth:
            if p1_val > p2_val:
                return (
                    Player.P1,
                    (
                        f"P1 sagte {p1_category.value} (korrekt), P2 glaubte → "
                        f"{p1_val} vs {p2_val} → P1 gewinnt."
                    ),
                    actual_truth,
                )
            if p2_val > p1_val:
                return (
                    Player.P2,
                    (
                        f"P1 sagte {p1_category.value} (korrekt), P2 glaubte → "
                        f"{p1_val} vs {p2_val} → P2 gewinnt."
                    ),
                    actual_truth,
                )
            return (
                None,
                (
                    f"P1 sagte {p1_category.value} (korrekt), P2 glaubte → "
                    f"{p1_val} vs {p2_val} → Unentschieden."
                ),
                actual_truth,
            )

        return (
            Player.P1,
            "P1 bluffte über die Kategorie, P2 glaubte → P1 gewinnt.",
            actual_truth,
        )

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
        self.session_csv.close()



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
