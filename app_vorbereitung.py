# app.py
# -----------------------------------------------------------------------------
# Merged script combining game_engine_w.py, aruco_overlay.py, and
# tabletop_ux_kivy_base_w.py so the entire application can be imported from a
# single module.
# -----------------------------------------------------------------------------
from __future__ import annotations


# ==== Begin original game_engine_w.py ====
# game_engine.py  (rollen-tausch + vollständiges Logging)
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


FORCED_BLUFF_LABEL = "erzwungener_bluff"


def hand_category(a: int, b: int) -> Optional[SignalLevel]:
    total = a + b
    if total == 19:
        return SignalLevel.HOCH
    if total in (16, 17, 18):
        return SignalLevel.MITTEL
    if total in (14, 15):
        return SignalLevel.TIEF
    if total in (20, 21, 22):
        return None
    # Falls Werte außerhalb des erwarteten Bereichs auftauchen, ordnen wir sie dem
    # nächsten sinnvollen Bereich zu, statt einen Laufzeitfehler zu riskieren.
    if total > 22:
        return None
    if total >= 16:
        return SignalLevel.MITTEL
    return SignalLevel.TIEF


def hand_category_label(a: int, b: int) -> str:
    level = hand_category(a, b)
    return FORCED_BLUFF_LABEL if level is None else level.value


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
        is_reveal = (actor == "SYS" and action == "reveal_and_score")
        if actor == "SYS" and not is_reveal:
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
            start_points = 0
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
        else:
            winner_vp = self.current.roles.p2_is
        self.scores[winner_vp] += 1

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
            "vp1_category": hand_category_label(*vp1),
            "vp2_category": hand_category_label(*vp2),
            "roles": {"P1": self.current.roles.p1_is.value, "P2": self.current.roles.p2_is.value}
        })

        self.current.phase = Phase.ROUND_DONE
        self._log("SYS", "phase_change", {"to": "ROUND_DONE"})

    def click_next_round(self, player: Player):
        """ Beide drücken 'Nächste Runde'. Danach: Rollen tauschen, nächste Runde → DEALING. """
        self._ensure([Phase.ROUND_DONE])
        if player == Player.P1 and not self.current.next_ready_p1:
            self.current.next_ready_p1 = True
            self._log("P1", "next_round_click", {})
        if player == Player.P2 and not self.current.next_ready_p2:
            self.current.next_ready_p2 = True
            self._log("P2", "next_round_click", {})

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
        forced_bluff = (p1_category is None)

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
            if forced_bluff:
                return (
                    Player.P2,
                    "P1 musste bluffen (20–22), P2 erwartete den Bluff → P2 gewinnt.",
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
            (
                "P1 musste bluffen (20–22) und P2 glaubte → P1 gewinnt."
                if forced_bluff
                else "P1 bluffte über die Kategorie, P2 glaubte → P1 gewinnt."
            ),
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
# ==== End original {name} ====

# ==== Begin original aruco_overlay.py ====
# requirements:
#   pip install pyqt6 opencv-contrib-python
#
# Nutzung (kompatibel zu deinem Game):
#   # Alte Art (unbedingt 8 IDs übergeben, sonst werden nur die ersten Positionen belegt):
#   MarkerOverlay(geo, marker_ids=[1,7,23,37,55,71,89,101])
#   # Empfohlen (feste Zuordnung):
#   MarkerOverlay(geo, layout=MARKER_LAYOUT)
#
# Tasten im Overlay:
#   M   -> Marker ein/ausblenden
#   +   -> Marker um +5% größer (nur wenn USE_FIXED_SIZE=False)
#   -   -> Marker um -5% kleiner (nur wenn USE_FIXED_SIZE=False)
#   Esc -> Programm beenden

import sys, os, json
from typing import List, Dict, Tuple, Optional
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent
from PyQt6.QtCore import Qt, QRect
import cv2
import numpy as np

# -------------------- EMPFOHLENE IDs & Positionen ----------------------------
# Robuste, weit auseinanderliegende AprilTag-IDs (tag36h11)
MARKER_LAYOUT: Dict[str, int] = {
    "top_left":     1,
    "top_right":    7,
    "bottom_left":  23,
    "bottom_right": 37,
    "top_mid":      55,
    "bottom_mid":   71,
    "left_mid":     89,
    "right_mid":    101,
}
# Reihenfolge der Platzierung (und Mapping-Reihenfolge für marker_ids)
POSITION_ORDER: List[str] = [
    "top_left", "top_right", "bottom_left", "bottom_right",
    "top_mid", "bottom_mid", "left_mid", "right_mid",
]

# -------------------- RENDER-PARAMETER ---------------------------------------
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
QUIET_ZONE_RATIO = 0.08                          # Weißer Rand (schmaler gemacht)
BG_WHITE_CSS = "background: white;"
# WAR: LABEL_CSS = "background: white; color: black; font: 12pt 'Segoe UI';"
# NEU: transparent, damit kein weißes Feld unterhalb sichtbar ist
LABEL_CSS   = "background: transparent; color: black; font: 12pt 'Segoe UI';"

# Markergröße: entweder FIX (deterministisch) ODER prozentual
USE_FIXED_SIZE = True
FIXED_SIZE_PX  = 280                              # z. B. 280 px inkl. Quiet-Zone
SIZE_PERCENT   = 0.16                              # falls USE_FIXED_SIZE=False
MIN_SIZE_PX    = 160
MAX_SIZE_PX    = 560

# -------------------- TAG-RENDERING ------------------------------------------
def generate_apriltag_qpixmap(tag_id: int, size: int, quiet_zone_ratio: float = QUIET_ZONE_RATIO) -> QPixmap:
    """Render AprilTag in weißem Quadrat (size x size) mit Quiet-Zone."""
    size = int(size)
    q = max(0.05, min(quiet_zone_ratio, 0.40))      # clamp 5..40%
    inner = int(round(size * (1.0 - 2.0 * q)))
    inner = max(32, inner)

    canvas = np.full((size, size), 255, dtype=np.uint8)  # weiß
    aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    tag_img = np.zeros((inner, inner), dtype=np.uint8)   # schwarz
    cv2.aruco.generateImageMarker(aruco_dict, tag_id, inner, tag_img, 1)

    y0 = (size - inner) // 2
    x0 = (size - inner) // 2
    canvas[y0:y0 + inner, x0:x0 + inner] = tag_img

    qimg = QImage(canvas.data, size, size, size, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg)

# -------------------- OVERLAY-FENSTER ----------------------------------------
class MarkerOverlay(QMainWindow):
    def __init__(
        self,
        screen_geometry: QRect,
        layout: Optional[Dict[str, int]] = None,
        marker_ids: Optional[List[int]] = None,   # Abwärtskompatibel
    ):
        """
        Entweder 'layout' übergeben (empfohlen) ODER 'marker_ids' (werden in POSITION_ORDER gemappt).
        """
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(BG_WHITE_CSS)
        self.setGeometry(screen_geometry)

        # --- Eingabe normalisieren ---
        if layout is not None:
            self.layout: Dict[str, int] = {name: layout[name] for name in POSITION_ORDER if name in layout}
        elif marker_ids is not None:
            n = min(len(marker_ids), len(POSITION_ORDER))
            self.layout = {POSITION_ORDER[i]: int(marker_ids[i]) for i in range(n)}
        else:
            # Default: alle 8 empfohlenen Marker
            self.layout = {name: MARKER_LAYOUT[name] for name in POSITION_ORDER}

        self.pos_order: List[str] = [name for name in POSITION_ORDER if name in self.layout]

        self.marker_labels: List[QLabel] = []
        self.text_labels: List[QLabel] = []
        self.markers_visible = True

        # Größen-Parameter
        self.size_percent = SIZE_PERCENT
        self.min_size = MIN_SIZE_PX
        self.max_size = MAX_SIZE_PX
        self.use_fixed = USE_FIXED_SIZE
        self.fixed_size = FIXED_SIZE_PX

        # UI-Objekte
        for _ in self.pos_order:
            lab = QLabel(self)
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lab.setStyleSheet(BG_WHITE_CSS)    # weißes Label = sichere Quiet-Zone
            lab.setScaledContents(False)
            self.marker_labels.append(lab)

            txt = QLabel(self)
            txt.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            txt.setStyleSheet(LABEL_CSS)       # transparent -> keine weiße Leiste
            txt.hide()                         # direkt verstecken
            self.text_labels.append(txt)

        # Zuordnung ausgeben & speichern
        print("Feste Marker-Zuordnung (Position → ID):")
        for name in self.pos_order:
            print(f"  {name:12s} -> {self.layout[name]}")
        try:
            mapping_path = os.path.join(os.getcwd(), "marker_layout.json")
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump({name: int(self.layout[name]) for name in self.pos_order}, f, ensure_ascii=False, indent=2)
            print(f"(Gespeichert als {mapping_path})")
        except Exception as e:
            print(f"Warnung: Konnte marker_layout.json nicht schreiben: {e}")

        self._layout_and_render_markers()

    @staticmethod
    def _positions_full(w: int, h: int, msize: int, margin: int) -> Dict[str, Tuple[int, int]]:
        # Ecken + Kantenmitten
        return {
            "top_left":     (margin, margin),
            "top_right":    (w - margin - msize, margin),
            "bottom_left":  (margin, h - margin - msize),
            "bottom_right": (w - margin - msize, h - margin - msize),
            "top_mid":      (w // 2 - msize // 2, margin),
            "bottom_mid":   (w // 2 - msize // 2, h - margin - msize),
            "left_mid":     (margin, h // 2 - msize // 2),
            "right_mid":    (w - margin - msize, h // 2 - msize // 2),
        }

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_and_render_markers()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_M:
            self.toggle_markers()
        elif event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            if not self.use_fixed:
                self.size_percent *= 1.05
                self._layout_and_render_markers()
        elif event.key() == Qt.Key.Key_Minus:
            if not self.use_fixed:
                self.size_percent /= 1.05
                self._layout_and_render_markers()
        elif event.key() == Qt.Key.Key_Escape:
            QApplication.instance().quit()

    def toggle_markers(self):
        self.markers_visible = not self.markers_visible
        self._layout_and_render_markers()

    def _layout_and_render_markers(self):
        w = max(1, self.width())
        h = max(1, self.height())

        # Markergröße
        if self.use_fixed:
            msize = int(self.fixed_size)
        else:
            base = int(min(w, h) * self.size_percent)
            msize = max(self.min_size, min(base, self.max_size))

        margin = max(6, int(msize * 0.08))  # Abstand zum Rand
        pos_map = self._positions_full(w, h, msize, margin)

        # Alle Labels erst verstecken, dann neu zeichnen
        for lab, txt in zip(self.marker_labels, self.text_labels):
            lab.setVisible(False)
            txt.setVisible(False)

        for (name, tag_id), lab in zip(
            [(n, self.layout[n]) for n in self.pos_order],
            self.marker_labels,
        ):
            x, y = pos_map[name]
            lab.resize(msize, msize)
            lab.move(x, y)
            lab.setPixmap(generate_apriltag_qpixmap(tag_id, msize, QUIET_ZONE_RATIO))
            lab.setVisible(self.markers_visible)

        # WICHTIG: keine Textlabels setzen/anzeigen -> keine weiße Fläche unterhalb

# -------------------- STANDALONE-TEST ----------------------------------------
def main():
    app = QApplication(sys.argv)

    # Standard: 8 Marker aus MARKER_LAYOUT
    layout = MARKER_LAYOUT

    overlays: List[MarkerOverlay] = []
    screens = app.screens()
    if not screens:
        geom = QRect(100, 100, 1280, 720)
        win = MarkerOverlay(geom, layout=layout)
        win.show()
        overlays.append(win)
    else:
        for s in screens:
            geom = s.geometry()
            # Alternativ kompatibel:
            # win = MarkerOverlay(geom, marker_ids=[1,7,23,37,55,71,89,101])
            win = MarkerOverlay(geom, layout=layout)
            win.showFullScreen()
            overlays.append(win)

    sys.exit(app.exec())
# ==== End original {name} ====

# ==== Begin original tabletop_ux_kivy_base_w.py ====
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
from kivy.graphics import Color, Rectangle
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.core.image import Image as CoreImage
import os
import csv
import itertools
import subprocess
import sys
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd



from game_engine_w import EventLogger, Phase as EnginePhase
from tabletop.ui import widgets as ui_widgets
from tabletop.ui.widgets import RotatableLabel, CardWidget, IconButton

# --- Display fest auf 3840x2160, Vollbild aktivierbar (kommentiere die nächste Zeile, falls du Fenster willst)
Config.set('graphics', 'fullscreen', 'auto')

# --- Konstanten & Assets
ROOT = os.path.dirname(os.path.abspath(__file__))
UX_DIR = os.path.join(ROOT, 'UX')
CARD_DIR = os.path.join(ROOT, 'Karten')

BACKGROUND_IMAGE = os.path.join(UX_DIR, 'Hintergrund.png')
FIX_STOP_IMAGE = os.path.join(UX_DIR, 'fix_stop.png')
FIX_LIVE_IMAGE = os.path.join(UX_DIR, 'fix_live.png')


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

ui_widgets.ASSETS = ASSETS


def resolve_background_texture():
    if not os.path.exists(BACKGROUND_IMAGE):
        return None
    try:
        return CoreImage(BACKGROUND_IMAGE).texture
    except Exception:
        return None


# --- Phasen der Runde
PH_WAIT_BOTH_START = 'WAIT_BOTH_START'
PH_P1_INNER = 'P1_INNER'
PH_P2_INNER = 'P2_INNER'
PH_P1_OUTER = 'P1_OUTER'
PH_P2_OUTER = 'P2_OUTER'
PH_SIGNALER = 'SIGNALER'
PH_JUDGE = 'JUDGE'
PH_SHOWDOWN = 'SHOWDOWN'


class TabletopRoot(FloatLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.bg_texture = resolve_background_texture()
        with self.canvas.before:
            if self.bg_texture:
                Color(1, 1, 1, 1)
                self.bg = Rectangle(pos=(0, 0), size=Window.size, texture=self.bg_texture)
            else:
                Color(0.75, 0.75, 0.75, 1)  # #BFBFBF fallback
                self.bg = Rectangle(pos=(0, 0), size=Window.size)
        Window.bind(on_resize=self.on_resize)

        self.round = 1
        self.signaler = 1
        self.judge = 2
        self.first_player = None
        self.second_player = None
        self.player_roles = {}
        self.update_turn_order()
        self.phase = PH_WAIT_BOTH_START
        # Versuchsperson 1 sitzt immer unten (Spieler 1), Versuchsperson 2 oben (Spieler 2)
        self._fixed_role_mapping = {1: 1, 2: 2}
        self.role_by_physical = self._fixed_role_mapping.copy()
        self.physical_by_role = {role: player for player, role in self.role_by_physical.items()}
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
            1: [Image(size_hint=(None, None), allow_stretch=True, keep_ratio=True),
                Image(size_hint=(None, None), allow_stretch=True, keep_ratio=True)],
            2: [Image(size_hint=(None, None), allow_stretch=True, keep_ratio=True),
                Image(size_hint=(None, None), allow_stretch=True, keep_ratio=True)],
        }
        for imgs in self.center_cards.values():
            for img in imgs:
                self.add_widget(img)

         # --- User-Displays (unter/über den vier Karten in der Mitte)
        # --- User-Displays: je Seite eines (unten VP1, oben VP2 – oben rotiert)
        self.user_displays = {
            1: RotatableLabel(
                size_hint=(None, None),
                halign='left',
                valign='top',
                color=(1, 1, 1, 1),
                markup=True,
            ),  # unten
            2: RotatableLabel(
                size_hint=(None, None),
                halign='left',
                valign='top',
                color=(1, 1, 1, 1),
                markup=True,
            ),  # oben (180°)
        }
        for lbl in self.user_displays.values():
            lbl.text = ''
            lbl.opacity = 1
            self.add_widget(lbl)

        # Intro-Overlay für den Startbildschirm
        intro_text = "[b]Willkommen![/b]\nUm zu Beginnen drücken Sie bitte auf den Play-Button"
        self.intro_overlay = FloatLayout(size_hint=(1, 1))
        with self.intro_overlay.canvas.before:
            Color(0.75, 0.75, 0.75, 1)
            self.intro_bg = Rectangle(pos=(0, 0), size=Window.size)
        self.intro_labels = {
            1: RotatableLabel(
                text=intro_text,
                halign='center',
                valign='middle',
                color=(0, 0, 0, 1),
                markup=True,
                size_hint=(None, None),
            ),
            2: RotatableLabel(
                text=intro_text,
                halign='center',
                valign='middle',
                color=(0, 0, 0, 1),
                markup=True,
                size_hint=(None, None),
            ),
        }
        self.intro_labels[1].set_rotation(0)
        self.intro_labels[2].set_rotation(180)
        for lbl in self.intro_labels.values():
            self.intro_overlay.add_widget(lbl)
        self.add_widget(self.intro_overlay)

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

        # Pause-Overlay (für Blockpausen)
        self.pause_cover = FloatLayout(size_hint=(1, 1))
        with self.pause_cover.canvas.before:
            Color(0.75, 0.75, 0.75, 1)
            self.pause_bg = Rectangle(pos=(0, 0), size=Window.size)
        self.pause_labels = {
            1: RotatableLabel(
                text='',
                color=(0, 0, 0, 1),
                halign='center',
                valign='middle',
                size_hint=(None, None),
            ),
            2: RotatableLabel(
                text='',
                color=(0, 0, 0, 1),
                halign='center',
                valign='middle',
                size_hint=(None, None),
            ),
        }
        self.pause_labels[1].set_rotation(0)
        self.pause_labels[2].set_rotation(180)
        for lbl in self.pause_labels.values():
            lbl.bind(texture_size=lambda *_: None)
            self.pause_cover.add_widget(lbl)
        self.pause_cover.opacity = 0
        self.pause_cover.disabled = True
        self.add_widget(self.pause_cover)

        # Start-Buttons nach vorn holen (über dem Overlay)
        self.bring_start_buttons_to_front()

        # Fixations-Overlay vorbereiten (wird bei Bedarf eingeblendet)
        self.fixation_overlay = FloatLayout(size_hint=(1, 1))
        self.fixation_overlay.opacity = 0
        self.fixation_overlay.disabled = True
        self.fixation_image = Image(size_hint=(None, None), allow_stretch=True, keep_ratio=True)
        self.fixation_overlay.add_widget(self.fixation_image)

        # Fixations-Overlay vorbereiten (wird bei Bedarf eingeblendet)
        self.fixation_overlay = FloatLayout(size_hint=(1, 1))
        self.fixation_overlay.opacity = 0
        self.fixation_overlay.disabled = True
        self.fixation_image = Image(size_hint=(None, None), allow_stretch=True, keep_ratio=True)
        self.fixation_overlay.add_widget(self.fixation_image)

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
        self.pending_round_start_log = False
        self.current_block_total_rounds = 0
        self.overlay_process = None
        self.fixation_running = False
        self.fixation_required = False
        self.pending_fixation_callback = None
        self.intro_active = True
        self.next_block_preview = None
        self.fixation_tone_fs = 44100
        fixation_duration = 0.2
        t = np.linspace(0, fixation_duration, int(self.fixation_tone_fs * fixation_duration), endpoint=False)
        self.fixation_tone = 0.9 * np.sin(2 * np.pi * 1000 * t)

        self.update_layout()
        self.update_user_displays()
        self.update_intro_overlay()
        self.start_overlay()

    def bring_start_buttons_to_front(self):
        self.remove_widget(self.btn_start_p1)
        self.remove_widget(self.btn_start_p2)
        self.add_widget(self.btn_start_p1)
        self.add_widget(self.btn_start_p2)

    def update_intro_overlay(self):
        if not hasattr(self, 'intro_overlay'):
            return
        active = bool(self.intro_active)
        if active:
            if self.intro_overlay.parent is None:
                self.add_widget(self.intro_overlay)
            self.intro_overlay.opacity = 1
            self.intro_overlay.disabled = False
            self.bring_start_buttons_to_front()
        else:
            self.intro_overlay.opacity = 0
            self.intro_overlay.disabled = True
            if self.intro_overlay.parent is not None:
                self.remove_widget(self.intro_overlay)
                self.bring_start_buttons_to_front()

    def update_layout(self):
        W, H = Window.size
        base_w, base_h = 3840.0, 2160.0
        scale = min(W / base_w if base_w else 1, H / base_h if base_h else 1)
        button_scale = 0.8

        self.bg.pos = (0, 0)
        self.bg.size = (W, H)

        corner_margin = 180 * scale
        card_width, card_height = 420 * scale, 640 * scale
        card_gap = 70 * scale
        start_size = (360 * button_scale * scale, 360 * button_scale * scale)

        # Start buttons
        self.btn_start_p1.size = start_size
        start_margin = 180 * scale
        self.btn_start_p1.pos = (W - start_margin - start_size[0], start_margin)
        self.btn_start_p1.set_rotation(0)

        self.btn_start_p2.size = start_size
        self.btn_start_p2.pos = (start_margin, H - start_margin - start_size[1])
        self.btn_start_p2.set_rotation(180)

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
        btn_width, btn_height = 260 * button_scale * scale, 260 * button_scale * scale
        vertical_gap = 40 * button_scale * scale
        horizontal_gap = 60 * button_scale * scale
        cluster_shift = 780 * scale
        vertical_offset = 220 * scale

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

        # --- User-Displays positionieren & drehen (2 Labels: unten/oben)
        display_gap = 40 * scale
        span_width = 2 * center_card_width + center_gap_x   # über beide Karten spannen
        display_width = span_width
        display_height = 660 * scale
        padding_x = 60 * scale
        padding_y = 40 * scale

        # Unteres Display (VP1): unter beiden unteren Karten, keine Drehung
        bottom_span_x = left_x
        bottom_span_y = bottom_y
        self.user_displays[1].size = (display_width, display_height)
        self.user_displays[1].pos = (
            bottom_span_x,
            bottom_span_y - display_gap - display_height
        )
        self.user_displays[1].text_size = (
            display_width - 2 * padding_x,
            display_height - 2 * padding_y,
        )
        self.user_displays[1].padding = (padding_x, padding_y)
        self.user_displays[1].font_size = 32 * scale if scale else 32
        self.user_displays[1].set_rotation(0)

        # Oberes Display (VP2): über beiden oberen Karten, 180° gedreht
        top_cards_top = top_y_center + center_card_height
        self.user_displays[2].size = (display_width, display_height)
        self.user_displays[2].pos = (
            left_x,
            top_cards_top + display_gap
        )
        self.user_displays[2].text_size = (
            display_width - 2 * padding_x,
            display_height - 2 * padding_y,
        )
        self.user_displays[2].padding = (padding_x, padding_y)
        self.user_displays[2].font_size = 32 * scale if scale else 32
        self.user_displays[2].set_rotation(180)

        # Round badge
        badge_width, badge_height = 1400 * scale, 70 * scale
        self.round_badge.size = (badge_width, badge_height)
        self.round_badge.font_size = 40 * scale if scale else 40
        self.round_badge.pos = (W / 2 - badge_width / 2, corner_margin / 2)
        self.round_badge.text_size = (badge_width, badge_height)

        # Pause-Overlay
        if hasattr(self, 'pause_bg'):
            self.pause_bg.size = (W, H)
            self.pause_bg.pos = (0, 0)
        if hasattr(self, 'pause_cover'):
            self.pause_cover.size = (W, H)
            self.pause_cover.pos = (0, 0)
        if hasattr(self, 'pause_labels'):
            label_width = W * 0.8
            label_height = H * 0.25
            gap = 40 * scale

            bottom_label = self.pause_labels[1]
            bottom_label.size = (label_width, label_height)
            bottom_label.pos = (
                W / 2 - label_width / 2,
                H / 2 - gap / 2 - label_height,
            )
            bottom_label.text_size = (label_width, label_height)
            bottom_label.font_size = 56 * scale if scale else 56
            bottom_label.set_rotation(0)

            top_label = self.pause_labels[2]
            top_label.size = (label_width, label_height)
            top_label.pos = (
                W / 2 - label_width / 2,
                H / 2 + gap / 2,
            )
            top_label.text_size = (label_width, label_height)
            top_label.font_size = 56 * scale if scale else 56
            top_label.set_rotation(180)

        if self.fixation_overlay:
            self.fixation_overlay.size = (W, H)
            self.fixation_overlay.pos = (0, 0)
        if self.fixation_image:
            self.fixation_image.size = (W, H)
            self.fixation_image.pos = (0, 0)
        if hasattr(self, 'intro_overlay'):
            self.intro_overlay.size = (W, H)
            self.intro_overlay.pos = (0, 0)
            if hasattr(self, 'intro_bg'):
                self.intro_bg.size = (W, H)
                self.intro_bg.pos = (0, 0)
            label_width = W * 0.6
            label_height = H * 0.25
            gap = 120 * scale
            padding = (40 * scale, 40 * scale)
            bottom_label = self.intro_labels[1]
            bottom_label.size = (label_width, label_height)
            bottom_label.pos = (
                W / 2 - label_width / 2,
                H / 2 - gap / 2 - label_height,
            )
            bottom_label.text_size = (label_width - 2 * padding[0], label_height - 2 * padding[1])
            bottom_label.padding = padding
            bottom_label.font_size = 64 * scale if scale else 64
            top_label = self.intro_labels[2]
            top_label.size = (label_width, label_height)
            top_label.pos = (
                W / 2 - label_width / 2,
                H / 2 + gap / 2,
            )
            top_label.text_size = (label_width - 2 * padding[0], label_height - 2 * padding[1])
            top_label.padding = padding
            top_label.font_size = 64 * scale if scale else 64

        # Refresh transforms after layout changes
        for buttons in self.signal_buttons.values():
            for btn in buttons.values():
                btn._update_transform()
        for buttons in self.decision_buttons.values():
            for btn in buttons.values():
                btn._update_transform()
        self.btn_start_p1._update_transform()
        self.btn_start_p2._update_transform()

    # --- Datenquellen & Hilfsfunktionen ---
    def load_blocks(self):
        blocks = []
        practice_path = Path(ROOT) / 'Paaretest.csv'
        practice_rounds = self.load_csv_rounds(practice_path)
        if practice_rounds:
            blocks.append({
                'index': 0,
                'label': 'Übung',
                'csv': 'Paaretest.csv',
                'path': practice_path,
                'rounds': practice_rounds,
                'payout': False,
                'practice': True,
            })
        order = [
            (1, 'Paare1.csv', False),
            (2, 'Paare2.csv', True),
            (3, 'Paare3.csv', False),
            (4, 'Paare4.csv', True),
        ]
        for index, filename, payout in order:
            path = Path(ROOT) / filename
            rounds = self.load_csv_rounds(path)
            blocks.append({
                'index': index,
                'label': f'Block {index}',
                'csv': filename,
                'path': path,
                'rounds': rounds,
                'payout': payout,
                'practice': False,
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

        def parse_numeric(cell):
            if cell is None:
                return None
            if isinstance(cell, (int, float)):
                return int(cell)
            text = str(cell).strip().replace(',', '.')
            if not text:
                return None
            try:
                return int(float(text))
            except ValueError:
                return None

        def parse_category(cell):
            text = (cell or '').strip().strip('"').lower()
            return text or None

        start_idx = 0
        if rows:
            try:
                parse_cards(rows[0], 2, 6)
                parse_cards(rows[0], 7, 11)
            except Exception:
                start_idx = 1

        for row in rows[start_idx:]:
            if not row or all((cell or '').strip() == '' for cell in row):
                continue
            try:
                vp1_cards = parse_cards(row, 2, 6)
                vp2_cards = parse_cards(row, 7, 11)
            except Exception:
                continue

            vp1_value = parse_numeric(row[5]) if len(row) > 5 else None
            vp2_value = parse_numeric(row[10]) if len(row) > 10 else None
            vp1_category = parse_category(row[1]) if len(row) > 1 else None
            vp2_category = parse_category(row[6]) if len(row) > 6 else None

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

    def value_to_card_path(self, value):
        try:
            number = int(value)
        except (TypeError, ValueError):
            return ASSETS['cards']['back']
        filename = f'{number}.png'
        path = os.path.join(CARD_DIR, filename)
        return path if os.path.exists(path) else ASSETS['cards']['back']

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
            inner_widget, outer_widget = self.p1_inner, self.p1_outer
        elif player == 2:
            inner_widget, outer_widget = self.p2_inner, self.p2_outer
        else:
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
            self.p1_inner.set_front(self.value_to_card_path(first_vp1))
            self.p1_outer.set_front(self.value_to_card_path(second_vp1))
            self.p2_inner.set_front(self.value_to_card_path(first_vp2))
            self.p2_outer.set_front(self.value_to_card_path(second_vp2))
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
                self.next_block_preview = None
            else:
                self.in_block_pause = True
                next_block = self.blocks[self.current_block_idx]
                condition = 'Stake' if next_block['payout'] else 'ohne Stake'
                self.pause_message = (
                    'Dieser Block ist vorbei. Nehmen Sie sich einen Moment zum durchatmen.\n'
                    'Wenn Sie bereit sind klicken Sie auf weiter.\n'
                    f'Als nächstes folgt Block {next_block["index"]} ({condition}).'
                )
                self.next_block_preview = {
                    'block': next_block,
                    'round_index': 0,
                    'round_in_block': 1,
                }
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
        if self.fixation_running:
            start_active = False
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
        self.update_user_displays()
        self.update_pause_overlay()

    def continue_after_start_press(self):
        if self.session_finished:
            return
        if self.intro_active:
            self.intro_active = False
            self.update_user_displays()
            self.update_intro_overlay()

        def proceed():
            start_phase = self.phase_for_player(self.first_player, 'inner') or PH_P1_INNER
            self.phase = start_phase
            self.log_round_start_if_pending()
            self.apply_phase()

        def proceed_with_fixation():
            if self.fixation_required and not self.fixation_running:
                self.run_fixation_sequence(proceed)
            else:
                proceed()

        proceed_with_fixation()

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
                if self.session_finished:
                    self.apply_phase()
                    return
                self.phase = PH_WAIT_BOTH_START
                self.apply_phase()
                self.continue_after_start_press()
            elif self.phase == PH_SHOWDOWN:
                self.prepare_next_round(start_immediately=True)
            else:
                self.continue_after_start_press()

    def run_fixation_sequence(self, on_complete=None):
        if self.fixation_running:
            return
        if not self.fixation_overlay or not self.fixation_image:
            self.fixation_required = False
            if on_complete:
                on_complete()
            return

        self.fixation_running = True
        self.pending_fixation_callback = on_complete
        self.fixation_overlay.opacity = 1
        self.fixation_overlay.disabled = False
        if self.fixation_overlay.parent is not None:
            self.remove_widget(self.fixation_overlay)
        self.add_widget(self.fixation_overlay)

        self.btn_start_p1.set_live(False)
        self.btn_start_p2.set_live(False)

        self.fixation_image.opacity = 1
        self.fixation_image.source = FIX_STOP_IMAGE if os.path.exists(FIX_STOP_IMAGE) else ''

        def finish(_dt):
            if self.fixation_overlay.parent is not None:
                self.remove_widget(self.fixation_overlay)
            self.fixation_overlay.opacity = 0
            self.fixation_overlay.disabled = True
            self.fixation_running = False
            self.fixation_required = False
            callback = self.pending_fixation_callback
            self.pending_fixation_callback = None
            if callback:
                callback()

        def show_stop_again(_dt):
            self.fixation_image.source = FIX_STOP_IMAGE if os.path.exists(FIX_STOP_IMAGE) else ''
            Clock.schedule_once(finish, 5)

        def show_live(_dt):
            self.fixation_image.source = FIX_LIVE_IMAGE if os.path.exists(FIX_LIVE_IMAGE) else ''
            self.play_fixation_tone()
            Clock.schedule_once(show_stop_again, 0.2)

        Clock.schedule_once(show_live, 5)

    def play_fixation_tone(self):
        if self.fixation_tone is None:
            return

        tone_data = self.fixation_tone.copy()
        sample_rate = self.fixation_tone_fs

        def _play():
            try:
                sd.play(tone_data, sample_rate)
                sd.wait()
            except Exception as exc:
                print(f'Warnung: Ton konnte nicht abgespielt werden: {exc}')

        threading.Thread(target=_play, daemon=True).start()

    def tap_card(self, who:int, which:str):
        # which in {'inner','outer'}
        if which not in {'inner', 'outer'}:
            return

        expected_phase = self.phase_for_player(who, which)
        if expected_phase is None or self.phase != expected_phase:
            return

        widget = self.card_widget_for_player(who, which)
        if not widget:
            return

        widget.flip()

        if which == 'inner':
            self.record_action(who, 'Karte innen aufgedeckt')
            self.log_event(who, 'reveal_inner', {'card': 1})
        else:
            self.record_action(who, 'Karte außen aufgedeckt')
            self.log_event(who, 'reveal_outer', {'card': 2})

        first = self.first_player
        second = self.second_player

        if which == 'inner':
            if who == first:
                next_phase = self.phase_for_player(second, 'inner')
            else:
                next_phase = self.phase_for_player(first, 'outer')
        else:
            if who == first:
                next_phase = self.phase_for_player(second, 'outer')
            else:
                next_phase = PH_SIGNALER

        if next_phase:
            Clock.schedule_once(lambda *_: self.goto(next_phase), 0.2)

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
        self.update_user_displays()
        Clock.schedule_once(lambda *_: self.goto(PH_JUDGE), 0.2)
        self.update_user_displays()

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
        self.update_user_displays()
        Clock.schedule_once(lambda *_: self.goto(PH_SHOWDOWN), 0.2)
        self.update_user_displays()

    def goto(self, phase):
        self.phase = phase
        self.apply_phase()

    def prepare_next_round(self, start_immediately: bool = False):
        # Rollen tauschen
        self.signaler, self.judge = self.judge, self.signaler
        self.update_turn_order()
        self.update_role_assignments()
        self.advance_round_pointer()
        self.phase = PH_WAIT_BOTH_START
        self.setup_round()
        if self.session_finished:
            self.apply_phase()
            self.update_user_displays()
            return

        self.apply_phase()

        if self.in_block_pause:
            return

        def proceed():
            start_phase = self.phase_for_player(self.first_player, 'inner') or PH_P1_INNER
            self.phase = start_phase
            self.log_round_start_if_pending()
            self.apply_phase()

        if self.fixation_required and not self.fixation_running:
            self.run_fixation_sequence(proceed)
        elif start_immediately:
            proceed()

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
            self.next_block_preview = None
            self.round_in_block = self.current_round_idx + 1
            self.current_round_has_stake = block['payout']
            self.current_block_total_rounds = len(block.get('rounds') or [])
            if block['payout'] and self.score_state_block != block['index']:
                self.score_state = {1: 0, 2: 0}
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
            self.current_block_total_rounds = 0
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
            'judge_value': None,
            'signal_choice': None,
            'judge_choice': None,
            'payout': self.current_round_has_stake,
        }
        self.refresh_center_cards(reveal=False)
        if plan_info and (self.round_in_block == 1):
            self.fixation_required = True
        elif plan_info:
            self.fixation_required = False
        else:
            self.fixation_required = False
        self.pending_round_start_log = bool(plan_info)
        self.update_user_displays()

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
                    self.score_state[winner_role] += 1
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
        signaler = self.signaler
        judge = self.judge
        signal_choice = self.player_signals.get(signaler)
        judge_choice = self.player_decisions.get(judge)
        actual_total = self.get_hand_total_for_player(signaler)
        judge_total = self.get_hand_total_for_player(judge)
        actual_value = self.get_hand_value_for_player(signaler)
        judge_value = self.get_hand_value_for_player(judge)
        actual_level = self.signal_level_from_value(actual_value)

        truthful = None
        if signal_choice:
            if actual_level:
                truthful = (signal_choice == actual_level)
            elif actual_total in (20, 21, 22):
                # Werte über 19 können nicht wahrheitsgemäß signalisiert werden
                truthful = False

        winner = None
        if judge_choice and truthful is not None:
            if judge_choice == 'wahr':
                if truthful:
                    if (
                        actual_value is not None
                        and judge_value is not None
                    ):
                        if actual_value > judge_value:
                            winner = signaler
                        elif judge_value > actual_value:
                            winner = judge
                        else:
                            winner = None
                    else:
                        winner = judge
                else:
                    winner = signaler
            elif judge_choice == 'bluff':
                winner = judge if not truthful else signaler

        draw = False
        if (
            judge_choice == 'wahr'
            and truthful is True
            and winner is None
            and actual_value is not None
            and judge_value is not None
            and actual_value == judge_value
        ):
            draw = True

        self.last_outcome = {
            'winner': winner,
            'truthful': truthful,
            'actual_level': actual_level,
            'actual_value': actual_value,
            'actual_total': actual_total,
            'judge_total': judge_total,
            'judge_value': judge_value,
            'signal_choice': signal_choice,
            'judge_choice': judge_choice,
            'payout': self.current_round_has_stake,
            'draw': draw,
        }
        return self.last_outcome

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
        if not self.last_outcome:
            return 'Unentschieden'
        winner_player = self.last_outcome.get('winner')
        if winner_player not in (1,2):
            return ' '
        winner_vp = self.role_by_physical.get(winner_player)
        if winner_vp == vp:
            return 'Gewonnen'
        return 'Verloren'

    def _result_with_score_for_vp(self, vp:int):
        base = self._result_for_vp(vp)
        if base == 'Unentschieden':
            return 'Unentschieden 0'
        if base == 'Gewonnen':
            return 'Gewonnen +1'
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
        lines.extend([ f"[b]{result_line}[/b]"])

        # Mehrzeilig – leichte Abstände über \n
        return "\n".join(lines)

    def update_user_displays(self):
        """Setzt die Texte in den beiden Displays (unten=VP1, oben=VP2)."""
        self.user_displays[1].text = self.format_user_display_text(1)  # unten
        self.user_displays[2].text = self.format_user_display_text(2)  # oben (rotiert)

    def update_pause_overlay(self):
        if not hasattr(self, 'pause_cover'):
            return
        active = (self.in_block_pause or self.session_finished) and bool(self.pause_message)
        if active:
            if self.pause_cover.parent is None:
                self.add_widget(self.pause_cover)
                # Start-Buttons über das Overlay legen
                self.bring_start_buttons_to_front()
            self.pause_cover.opacity = 1
            self.pause_cover.disabled = False
            for lbl in self.pause_labels.values():
                lbl.text = self.pause_message
        else:
            self.pause_cover.opacity = 0
            self.pause_cover.disabled = True
            for lbl in self.pause_labels.values():
                lbl.text = ''
            if self.pause_cover.parent is not None:
                self.remove_widget(self.pause_cover)
                # Reihenfolge der Buttons erhalten
                self.bring_start_buttons_to_front()



    def start_overlay(self):
        if self.overlay_process and self.overlay_process.poll() is None:
            return
        overlay_path = os.path.join(ROOT, 'aruco_overlay.py')
        if not os.path.exists(overlay_path):
            return
        try:
            self.overlay_process = subprocess.Popen([sys.executable, overlay_path])
        except Exception as exc:
            print(f'Warnung: Overlay konnte nicht gestartet werden: {exc}')
            self.overlay_process = None

    def stop_overlay(self):
        if not self.overlay_process:
            return
        if self.overlay_process.poll() is None:
            try:
                self.overlay_process.terminate()
                self.overlay_process.wait(timeout=5)
            except Exception:
                try:
                    self.overlay_process.kill()
                except Exception:
                    pass
        self.overlay_process = None


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
        first = self.signaler if self.signaler in (1, 2) else 1
        if self.judge in (1, 2) and self.judge != first:
            second = self.judge
        else:
            second = 2 if first == 1 else 1

        self.first_player = first
        self.second_player = second
        self.player_roles = {
            first: 1,
            second: 2,
        }

    def phase_for_player(self, player: int, which: str):
        if player not in (1, 2):
            return None
        if which == 'inner':
            return PH_P1_INNER if player == 1 else PH_P2_INNER
        if which == 'outer':
            return PH_P1_OUTER if player == 1 else PH_P2_OUTER
        return None

    def card_widget_for_player(self, player: int, which: str):
        if player == 1:
            if which == 'inner':
                return self.p1_inner
            if which == 'outer':
                return self.p1_outer
        elif player == 2:
            if which == 'inner':
                return self.p2_inner
            if which == 'outer':
                return self.p2_outer
        return None

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
        is_showdown = (action == 'showdown')
        if not is_showdown and player not in (1, 2):
            return

        block_condition = ''
        block_number = ''
        round_in_block = ''
        if self.current_block_info:
            block_condition = 'pay' if self.current_round_has_stake else 'no_pay'
            block_number = self.current_block_info['index']
            round_in_block = self.round_in_block
        elif self.next_block_preview:
            block = self.next_block_preview.get('block')
            if block:
                block_condition = 'pay' if block.get('payout') else 'no_pay'
                block_number = block.get('index', '')
                round_in_block = self.next_block_preview.get('round_in_block', '')

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
        if not is_showdown and player in (1, 2):
            vp_num = self.role_by_physical.get(player)
            if vp_num in (1, 2):
                actor_vp = f'VP{vp_num}'

        spieler1_vp = ''
        first_player = self.first_player if self.first_player in (1, 2) else None
        if first_player is not None:
            vp_player1 = self.role_by_physical.get(first_player)
            if vp_player1 in (1, 2):
                spieler1_vp = f'VP{vp_player1}'

        action_label = self.round_log_action_label(action, payload)
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        winner_label = ''
        if is_showdown:
            winner_player = payload.get('winner')
            if winner_player in (1, 2):
                winner_vp = self.role_by_physical.get(winner_player)
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
        self.apply_phase()

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

    def on_start(self):
        root = self.root
        if root:
            Clock.schedule_once(lambda *_: root.start_overlay(), 0)

    def on_stop(self):
        root = self.root
        if root and root.logger:
            root.logger.close()
        if root:
            root.close_round_log()
            root.stop_overlay()

if __name__ == '__main__':
    TabletopApp().run()
 
# ==== End original {name} ====
