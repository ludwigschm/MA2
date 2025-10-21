"""Round CSV logging utilities."""

from __future__ import annotations

import csv
from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # Optional dependency used when available.
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional at runtime
    _pd = None


ROUND_LOG_HEADER: List[str] = [
    "Session",
    "Bedingung",
    "Block",
    "Runde im Block",
    "Spieler 1",
    "VP",
    "Karte1 VP1",
    "Karte2 VP1",
    "Karte1 VP2",
    "Karte2 VP2",
    "Aktion",
    "Zeit",
    "Gewinner",
    "Punktestand VP1",
    "Punktestand VP2",
]


def init_round_log(app: Any) -> None:
    if not getattr(app, "session_id", None):
        return
    if getattr(app, "round_log_path", None):
        close_round_log(app)
    app.log_dir.mkdir(parents=True, exist_ok=True)
    session_fs_id = getattr(app, "session_storage_id", None) or app.session_id
    path = app.log_dir / f"round_log_{session_fs_id}.csv"
    app.round_log_path = path
    app.round_log_fp = None
    app.round_log_writer = None
    buffer: Optional[List[List[Any]]] = getattr(app, "round_log_buffer", None)
    if buffer is None:
        app.round_log_buffer = []
    else:
        buffer.clear()


def round_log_action_label(app: Any, action: str, payload: Dict[str, Any]) -> str:
    if action in ("start_click", "round_start"):
        return "Start"
    if action == "next_round_click":
        return "Nächste Runde"
    if action == "reveal_inner":
        return "Karte 1"
    if action == "reveal_outer":
        return "Karte 2"
    if action == "signal_choice":
        return app.format_signal_choice(payload.get("level")) or "Signal"
    if action == "call_choice":
        return app.format_decision_choice(payload.get("decision")) or "Entscheidung"
    if action == "showdown":
        return "Showdown"
    if action == "session_start":
        return "Session"
    return action


def write_round_log(app: Any, actor: str, action: str, payload: Dict[str, Any], player: int) -> None:
    if not getattr(app, "round_log_path", None):
        return
    is_showdown = action == "showdown"
    if not is_showdown and player not in (1, 2):
        return

    block_condition = ""
    block_number = ""
    round_in_block = ""
    if getattr(app, "current_block_info", None):
        block_condition = "pay" if app.current_round_has_stake else "no_pay"
        block_number = app.current_block_info["index"]
        round_in_block = app.round_in_block
    elif getattr(app, "next_block_preview", None):
        block = app.next_block_preview.get("block")
        if block:
            block_condition = "pay" if block.get("payout") else "no_pay"
            block_number = block.get("index", "")
            round_in_block = app.next_block_preview.get("round_in_block", "")

    plan = None
    plan_info = app.get_current_plan()
    if plan_info:
        plan = plan_info[1]
    vp1_cards = plan["vp1"] if plan else (None, None)
    vp2_cards = plan["vp2"] if plan else (None, None)
    if not vp1_cards:
        vp1_cards = (None, None)
    if not vp2_cards:
        vp2_cards = (None, None)

    actor_vp = ""
    if not is_showdown and player in (1, 2):
        vp_num = app.role_by_physical.get(player)
        if vp_num in (1, 2):
            actor_vp = f"VP{vp_num}"

    spieler1_vp = ""
    first_player = app.first_player if app.first_player in (1, 2) else None
    if first_player is not None:
        vp_player1 = app.role_by_physical.get(first_player)
        if vp_player1 in (1, 2):
            spieler1_vp = f"VP{vp_player1}"

    action_label = round_log_action_label(app, action, payload)
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    winner_label = ""
    if is_showdown:
        winner_player = payload.get("winner")
        if winner_player in (1, 2):
            winner_vp = app.role_by_physical.get(winner_player)
            if winner_vp in (1, 2):
                winner_label = f"VP{winner_vp}"

    if getattr(app, "score_state", None):
        score_vp1 = app.score_state.get(1, "")
        score_vp2 = app.score_state.get(2, "")
    elif getattr(app, "score_state_round_start", None):
        score_vp1 = app.score_state_round_start.get(1, "")
        score_vp2 = app.score_state_round_start.get(2, "")
    else:
        score_vp1 = ""
        score_vp2 = ""

    def _card_value(val: Any) -> Any:
        return "" if val is None else val

    row = [
        app.session_id or "",
        block_condition,
        block_number,
        round_in_block,
        spieler1_vp,
        actor_vp,
        _card_value(vp1_cards[0]) if vp1_cards else "",
        _card_value(vp1_cards[1]) if vp1_cards else "",
        _card_value(vp2_cards[0]) if vp2_cards else "",
        _card_value(vp2_cards[1]) if vp2_cards else "",
        action_label,
        timestamp,
        winner_label,
        score_vp1,
        score_vp2,
    ]
    buffer = getattr(app, "round_log_buffer", None)
    if buffer is None:
        buffer = []
        app.round_log_buffer = buffer
    buffer.append(row)


def flush_round_log(app: Any, pandas_module: Any | None = None) -> None:
    if not getattr(app, "round_log_path", None):
        return
    buffer: Optional[List[List[Any]]] = getattr(app, "round_log_buffer", None)
    if not buffer:
        return
    path = app.round_log_path
    app.log_dir.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    rows = list(buffer)
    pd = pandas_module if pandas_module is not None else _pd
    if pd is not None:
        df = pd.DataFrame(rows, columns=ROUND_LOG_HEADER)
        df.to_csv(path, mode="a", header=not file_exists, index=False)
    else:
        write_header = not file_exists
        with open(path, "a", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if write_header:
                writer.writerow(ROUND_LOG_HEADER)
            writer.writerows(rows)
    buffer.clear()


def close_round_log(app: Any) -> None:
    flush_round_log(app)
    if getattr(app, "round_log_fp", None):
        app.round_log_fp.close()
    app.round_log_fp = None
    app.round_log_writer = None
    if getattr(app, "round_log_path", None):
        app.round_log_path = None
