"""Background reconciliation of UI event timestamps with device timelines."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

from tabletop.engine import EventLogger
from tabletop.pupil_bridge import PupilBridge

log = logging.getLogger(__name__)


@dataclass
class _PlayerState:
    """Calibration state for a single player/device."""

    player: str
    samples: Deque[Tuple[int, int]] = field(default_factory=deque)
    intercept_ns: float = 0.0
    slope: float = 1.0
    rms_ns: float = 0.0
    confidence: float = 0.0
    mapping_version: int = 0
    last_update: float = field(default_factory=time.monotonic)


class TimeReconciler:
    """Continuously estimates clock offsets and refines provisional events."""

    def __init__(
        self,
        bridge: PupilBridge,
        logger: EventLogger,
        window_size: int = 10,
    ) -> None:
        self._bridge = bridge
        self._logger = logger
        self._window_size = max(3, int(window_size))
        self._state_lock = threading.Lock()
        self._player_states: Dict[str, _PlayerState] = {}
        self._mapping_version = 0
        self._global_model: Optional[Tuple[float, float, float, float, int]] = None
        self._task_queue: queue.Queue[Tuple[str, Tuple[Any, ...]]] = queue.Queue(
            maxsize=2000
        )
        self._queue_drop = 0
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._known_events: Dict[str, Tuple[int, int]] = {}
        self._event_order: Deque[str] = deque()
        self._event_retention = max(2000, self._window_size * 200)
        self._heartbeat_count = 0
        self._intercept_epsilon_ns = 5_000.0
        self._slope_epsilon = 1e-9
        self._huber_delta_ns = 5_000_000.0

    # ------------------------------------------------------------------
    @property
    def current_mapping_version(self) -> int:
        with self._state_lock:
            return self._mapping_version

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run,
            name="TimeReconciler",
            daemon=True,
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._enqueue("stop")
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

    # ------------------------------------------------------------------
    def submit_marker(self, label: str, t_local_ns: int) -> None:
        self._enqueue("marker", label, int(t_local_ns))

    def on_event(self, event_id: str, t_local_ns: int) -> None:
        self._enqueue("event", event_id, int(t_local_ns))

    # ------------------------------------------------------------------
    def _enqueue(self, kind: str, *args: Any) -> None:
        try:
            self._task_queue.put_nowait((kind, args))
        except queue.Full:
            self._queue_drop += 1
            log.warning(
                "TimeReconciler queue full – dropping %s (%d drops)",
                kind,
                self._queue_drop,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                kind, args = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == "stop":
                break
            try:
                if kind == "marker":
                    self._process_marker(str(args[0]), int(args[1]))
                elif kind == "event":
                    self._process_event(str(args[0]), int(args[1]))
            except Exception:
                log.exception("Error processing %s task", kind)
            finally:
                self._task_queue.task_done()
        # Drain remaining items gracefully
        while True:
            try:
                kind, args = self._task_queue.get_nowait()
            except queue.Empty:
                break
            if kind in {"marker", "event"}:
                try:
                    if kind == "marker":
                        self._process_marker(str(args[0]), int(args[1]))
                    else:
                        self._process_event(str(args[0]), int(args[1]))
                except Exception:
                    log.exception("Error processing %s during shutdown", kind)
            self._task_queue.task_done()

    # ------------------------------------------------------------------
    def _process_marker(self, label: str, t_local_ns: int) -> None:
        players = self._connected_players_snapshot()
        if not players:
            return
        offsets_ns: Dict[str, int] = {}
        for player in players:
            try:
                offset_seconds = self._bridge.estimate_time_offset(player)
            except Exception:
                log.debug("estimate_time_offset failed for %s", player, exc_info=True)
                continue
            if offset_seconds is None:
                continue
            offsets_ns[player] = int(offset_seconds * 1_000_000_000)
        if not offsets_ns:
            return

        self._heartbeat_count += 1

        for player, offset_ns in offsets_ns.items():
            device_ns = t_local_ns + offset_ns
            samples = self._append_sample(player, t_local_ns, device_ns)
            if samples is None:
                continue
            changed = self._recompute_model_from_samples(player, samples)
            if changed:
                model = self._global_model
                if model is not None:
                    intercept, slope, confidence, rms_ns, version = model
                    log.info(
                        "Mapping %s updated: a=%.0fns b=%.9f conf=%.3f rms=%.3fms (v%s)",
                        player,
                        intercept,
                        slope,
                        confidence,
                        rms_ns / 1_000_000.0,
                        version,
                    )
        model = self._global_model
        if model is not None:
            _, _, confidence, rms_ns, version = model
            log.debug(
                "Sync marker %s processed (hb=%d, conf=%.3f, rms=%.3fms, v%s)",
                label,
                self._heartbeat_count,
                confidence,
                rms_ns / 1_000_000.0,
                version,
            )

    def _append_sample(
        self, player: str, t_local_ns: int, t_device_ns: int
    ) -> Optional[list[Tuple[int, int]]]:
        with self._state_lock:
            state = self._player_states.get(player)
            if state is None:
                state = _PlayerState(
                    player=player,
                    samples=deque(maxlen=self._window_size),
                )
                self._player_states[player] = state
            elif state.samples.maxlen != self._window_size:
                state.samples = deque(state.samples, maxlen=self._window_size)
            state.samples.append((t_local_ns, t_device_ns))
            return list(state.samples)

    def _recompute_model_from_samples(
        self, player: str, samples: list[Tuple[int, int]]
    ) -> bool:
        if len(samples) < 2:
            return False
        intercept, slope, rms_ns = self._huber_fit(samples)
        confidence = self._confidence_from_rms(rms_ns)
        with self._state_lock:
            state = self._player_states[player]
            delta_intercept = abs(state.intercept_ns - intercept)
            delta_slope = abs(state.slope - slope)
            changed = (
                delta_intercept > self._intercept_epsilon_ns
                or delta_slope > self._slope_epsilon
            )
            state.intercept_ns = intercept
            state.slope = slope
            state.rms_ns = rms_ns
            state.confidence = confidence
            state.last_update = time.monotonic()
            if changed:
                state.mapping_version = self._bump_mapping_version_locked()
            else:
                state.mapping_version = max(state.mapping_version, self._mapping_version)
            self._update_global_model_locked()
            version = self._mapping_version
        if changed:
            self._refine_all_pending(version)
        return changed

    def _refine_all_pending(self, version: int) -> None:
        event_ids: list[str]
        with self._state_lock:
            event_ids = list(self._known_events.keys())
        for event_id in event_ids:
            self._refine_single_event(event_id, version_hint=version)

    def _process_event(self, event_id: str, t_local_ns: int) -> None:
        with self._state_lock:
            previous = self._known_events.get(event_id)
            last_version = previous[1] if previous else 0
            self._known_events[event_id] = (t_local_ns, last_version)
            self._event_order.append(event_id)
            while len(self._event_order) > self._event_retention:
                stale = self._event_order.popleft()
                if stale == event_id:
                    continue
                self._known_events.pop(stale, None)
        self._refine_single_event(event_id)

    def _refine_single_event(
        self, event_id: str, *, version_hint: Optional[int] = None
    ) -> None:
        with self._state_lock:
            entry = self._known_events.get(event_id)
            model = self._global_model
        if entry is None or model is None:
            return
        t_local_ns, last_version = entry
        intercept, slope, confidence, rms_ns, version = model
        effective_version = version_hint or version
        if effective_version <= last_version:
            return
        t_ref_ns = int(intercept + slope * t_local_ns)
        extra = {
            "rms_error_ns": int(rms_ns),
            "heartbeat_count": self._heartbeat_count,
        }
        queue_size, queue_capacity = self._bridge.event_queue_load()
        extra["queue"] = {"size": queue_size, "capacity": queue_capacity}
        players = self._connected_players_snapshot()
        for player in players:
            try:
                self._bridge.refine_event(
                    player,
                    event_id,
                    t_ref_ns,
                    confidence=confidence,
                    mapping_version=effective_version,
                    extra=dict(extra),
                )
            except Exception:
                log.exception("Refinement dispatch failed for %s (%s)", event_id, player)
        try:
            self._logger.record_refinement(event_id, t_ref_ns, effective_version, confidence)
        except Exception:
            log.exception("Persisting refinement failed for %s", event_id)
        log.info(
            "event %s provisional -> refined (t_ref=%d, v%s, conf=%.3f, queue=%s/%s)",
            event_id,
            t_ref_ns,
            effective_version,
            confidence,
            queue_size,
            queue_capacity,
        )
        with self._state_lock:
            self._known_events[event_id] = (t_local_ns, effective_version)

    # ------------------------------------------------------------------
    def _connected_players_snapshot(self) -> list[str]:
        try:
            players = self._bridge.connected_players()
        except Exception:
            log.debug("connected_players lookup failed", exc_info=True)
            players = []
        if not players:
            with self._state_lock:
                players = list(self._player_states.keys())
        return players

    def _huber_fit(self, samples: list[Tuple[int, int]]) -> Tuple[float, float, float]:
        xs = [float(a) for a, _ in samples]
        ys = [float(b) for _, b in samples]
        weights = [1.0] * len(samples)
        intercept = ys[0]
        slope = 1.0
        for _ in range(5):
            sum_w = sum(weights)
            if sum_w <= 0:
                break
            mean_x = sum(w * x for w, x in zip(weights, xs)) / sum_w
            mean_y = sum(w * y for w, y in zip(weights, ys)) / sum_w
            var = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
            if var <= 0:
                slope = 1.0
            else:
                cov = sum(
                    w * (x - mean_x) * (y - mean_y)
                    for w, x, y in zip(weights, xs, ys)
                )
                slope = cov / var if var else 1.0
            intercept = mean_y - slope * mean_x
            residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
            weights = [
                1.0
                if abs(r) <= self._huber_delta_ns
                else self._huber_delta_ns / max(abs(r), 1e-9)
                for r in residuals
            ]
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        rms_ns = 0.0
        if residuals:
            rms_ns = math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))
        return intercept, slope, rms_ns

    def _confidence_from_rms(self, rms_ns: float) -> float:
        if rms_ns <= 0:
            return 1.0
        rms_ms = rms_ns / 1_000_000.0
        # Map 0ms -> 1.0, 5ms -> ~0.37, >=20ms -> ~0.018
        return max(0.0, min(1.0, math.exp(-rms_ms / 5.0)))

    def _bump_mapping_version_locked(self) -> int:
        self._mapping_version += 1
        return self._mapping_version

    def _update_global_model_locked(self) -> None:
        states = [state for state in self._player_states.values() if state.samples]
        if not states:
            self._global_model = None
            return
        weights = [max(0.1, state.confidence or 0.1) for state in states]
        total = sum(weights)
        intercept = sum(w * state.intercept_ns for w, state in zip(weights, states)) / total
        slope = sum(w * state.slope for w, state in zip(weights, states)) / total
        rms_ns = math.sqrt(
            sum((state.rms_ns or 0.0) ** 2 for state in states) / len(states)
        )
        confidence = max(state.confidence for state in states)
        self._global_model = (intercept, slope, confidence, rms_ns, self._mapping_version)


__all__ = ["TimeReconciler"]
