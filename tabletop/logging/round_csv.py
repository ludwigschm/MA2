"""Round CSV logging utilities."""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional dependency used when available.
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional at runtime
    _pd = None

from tabletop.utils.runtime import (
    is_low_latency_disabled,
    is_perf_logging_enabled,
)

log = logging.getLogger(__name__)

_LOW_LATENCY_DISABLED = is_low_latency_disabled()
_PERF_LOGGING = is_perf_logging_enabled()
_ROUND_QUEUE_MAXSIZE = 8
_ROUND_BUFFER_MAX = 500
_ROUND_FLUSH_INTERVAL = 1.0
_ROUND_QUEUE: Optional[queue.Queue[Tuple[Path, List[List[Any]], bool]]] = None
_ROUND_QUEUE_LOCK = threading.Lock()
_ROUND_WRITER: Optional[threading.Thread] = None


def _write_round_rows(path: Path, rows: List[List[Any]], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(ROUND_LOG_HEADER)
        writer.writerows(rows)


def _round_writer_loop(queue_obj: queue.Queue[Tuple[Path, List[List[Any]], bool]]) -> None:
    while True:
        path, rows, write_header = queue_obj.get()
        start = time.perf_counter()
        try:
            _write_round_rows(path, rows, write_header)
            if _PERF_LOGGING:
                duration = (time.perf_counter() - start) * 1000.0
                log.debug(
                    "Round CSV flush wrote %d rows in %.2f ms", len(rows), duration
                )
        except Exception:  # pragma: no cover - defensive logging
            log.exception("Failed to flush %d round log rows", len(rows))
        finally:
            queue_obj.task_done()


def _ensure_round_writer() -> queue.Queue[Tuple[Path, List[List[Any]], bool]]:
    global _ROUND_QUEUE, _ROUND_WRITER
    if _ROUND_QUEUE is not None:
        return _ROUND_QUEUE
    with _ROUND_QUEUE_LOCK:
        if _ROUND_QUEUE is not None:
            return _ROUND_QUEUE
        queue_obj: queue.Queue[Tuple[Path, List[List[Any]], bool]] = queue.Queue(
            maxsize=_ROUND_QUEUE_MAXSIZE
        )
        writer_thread = threading.Thread(
            target=_round_writer_loop,
            args=(queue_obj,),
            name="RoundCsvWriter",
            daemon=True,
        )
        writer_thread.start()
        _ROUND_QUEUE = queue_obj
        _ROUND_WRITER = writer_thread
        return queue_obj


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
    app.round_log_last_flush = time.monotonic()


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
    if not hasattr(app, "round_log_last_flush"):
        app.round_log_last_flush = time.monotonic()
    flush_round_log(app)


def flush_round_log(
    app: Any,
    pandas_module: Any | None = None,
    *,
    force: bool = False,
    wait: bool = False,
) -> None:
    if not getattr(app, "round_log_path", None):
        return
    buffer: Optional[List[List[Any]]] = getattr(app, "round_log_buffer", None)
    if not buffer:
        return

    now = time.monotonic()
    last_flush = getattr(app, "round_log_last_flush", 0.0)
    if (
        not force
        and not _LOW_LATENCY_DISABLED
        and len(buffer) < _ROUND_BUFFER_MAX
        and now - last_flush < _ROUND_FLUSH_INTERVAL
    ):
        return

    path = Path(app.round_log_path)
    app.log_dir.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    rows = list(buffer)
    buffer.clear()
    app.round_log_last_flush = now

    if _LOW_LATENCY_DISABLED:
        pd = pandas_module if pandas_module is not None else _pd
        if pd is not None:
            df = pd.DataFrame(rows, columns=ROUND_LOG_HEADER)
            df.to_csv(path, mode="a", header=not file_exists, index=False)
        else:
            _write_round_rows(path, rows, not file_exists)
        return

    queue_obj = _ensure_round_writer()
    write_header = not file_exists
    try:
        queue_obj.put_nowait((path, rows, write_header))
    except queue.Full:
        log.warning(
            "Round log queue saturated – falling back to synchronous flush (%d rows)",
            len(rows),
        )
        _write_round_rows(path, rows, write_header)
    else:
        if _PERF_LOGGING and queue_obj.maxsize:
            load = queue_obj.qsize() / queue_obj.maxsize
            if load >= 0.8:
                log.warning("Round log queue at %.0f%% capacity", load * 100.0)
        if wait:
            queue_obj.join()


def close_round_log(app: Any) -> None:
    flush_round_log(app, force=True, wait=not _LOW_LATENCY_DISABLED)
    if getattr(app, "round_log_fp", None):
        app.round_log_fp.close()
    app.round_log_fp = None
    app.round_log_writer = None
    if getattr(app, "round_log_path", None):
        app.round_log_path = None
