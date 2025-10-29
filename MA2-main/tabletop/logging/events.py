"""Adapter for game engine event logging."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from tabletop.engine import EventLogger, Phase as EnginePhase

__all__ = ["Events", "EnginePhase"]


class Events:
    """Thin wrapper around :class:`tabletop.engine.EventLogger`."""

    def __init__(
        self,
        session_id: str,
        db_path: str,
        csv_path: Optional[str] = None,
        annotator: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self._session_id = session_id
        self._logger = EventLogger(db_path, csv_path)
        self._annotate = annotator

    def log(
        self,
        round_idx: int,
        phase: EnginePhase,
        actor: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Forward events to the underlying logger while fixing defaults."""

        row = self._logger.log(
            self._session_id,
            round_idx,
            phase,
            actor,
            action,
            payload or {},
        )
        if self._annotate is not None:
            try:
                self._annotate(
                    "EVENT",
                    {
                        "actor": actor,
                        "action": action,
                        "payload": payload or {},
                        "phase": getattr(phase, "name", str(phase)),
                        "round": round_idx,
                        "session": self._session_id,
                    },
                )
            except Exception:
                pass
        return row

    def close(self) -> None:
        """Close the underlying logger."""

        self._logger.close()
