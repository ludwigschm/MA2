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
    raw_offsets: Deque[Tuple[int, int]] = field(default_factory=deque)
    intercept_ns: float = 0.0
    slope: float = 1.0
    rms_ns: float = 0.0
    confidence: float = 0.0
    mapping_version: int = 0
    sample_count: int = 0
    offset_sign: int = 1
    last_update: float = field(default_factory=time.monotonic)


class TimeReconciler:
    """Continuously estimates clock offsets and refines provisional events."""

    CONF_MIN = 0.8

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
        self._conf_min = float(self.CONF_MIN)
        self._task_queue: queue.Queue[Tuple[str, Tuple[Any, ...]]] = queue.Queue(
            maxsize=2000
        )
        self._queue_drop = 0
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._known_events: Dict[str, Tuple[int, Dict[str, int]]] = {}
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
            if not self._player_states:
                return 0
            return max(state.mapping_version for state in self._player_states.values())

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
            self._ingest_marker(player, t_local_ns, offset_ns)

        log.debug(
            "Sync marker %s processed (hb=%d, players=%d)",
            label,
            self._heartbeat_count,
            len(offsets_ns),
        )

    def _ingest_marker(self, player: str, t_local_ns: int, offset_ns: int) -> None:
        base_samples: list[Tuple[int, int]]
        with self._state_lock:
            state = self._player_states.get(player)
            if state is None:
                state = _PlayerState(
                    player=player,
                    samples=deque(maxlen=self._window_size),
                    raw_offsets=deque(maxlen=self._window_size),
                )
                self._player_states[player] = state
            elif state.samples.maxlen != self._window_size:
                state.samples = deque(state.samples, maxlen=self._window_size)
                state.raw_offsets = deque(state.raw_offsets, maxlen=self._window_size)
            raw_samples = list(state.raw_offsets)
            current_sign = state.offset_sign or 1
            current_intercept = state.intercept_ns
            current_slope = state.slope
            current_count = state.sample_count

        raw_candidates = self._trimmed_raw_samples(raw_samples, (t_local_ns, offset_ns))

        candidate_pos = self._raw_to_samples(raw_candidates, 1)
        candidate_neg = self._raw_to_samples(raw_candidates, -1)

        pos_metrics = self._evaluate_candidate(candidate_pos)
        neg_metrics = self._evaluate_candidate(candidate_neg)

        chosen_sign = current_sign
        chosen_samples = candidate_pos
        chosen_metrics = pos_metrics

        offset_residual_pos = float("inf")
        offset_residual_neg = float("inf")
        if pos_metrics is not None:
            offset_residual_pos = self._offset_residual(
                raw_candidates, pos_metrics[0], pos_metrics[1], 1
            )
        if neg_metrics is not None:
            offset_residual_neg = self._offset_residual(
                raw_candidates, neg_metrics[0], neg_metrics[1], -1
            )
        align_pos = (
            self._offset_alignment(raw_candidates, pos_metrics[0], pos_metrics[1], 1)
            if pos_metrics is not None
            else 0.0
        )
        align_neg = (
            self._offset_alignment(raw_candidates, neg_metrics[0], neg_metrics[1], -1)
            if neg_metrics is not None
            else 0.0
        )
        if offset_residual_neg + 1e-3 < offset_residual_pos * 0.8:
            chosen_sign = -1
            chosen_samples = candidate_neg
            chosen_metrics = neg_metrics
        elif offset_residual_pos + 1e-3 < offset_residual_neg * 0.8:
            chosen_sign = 1
            chosen_samples = candidate_pos
            chosen_metrics = pos_metrics
        elif align_neg > 0 and align_pos < 0:
            chosen_sign = -1
            chosen_samples = candidate_neg
            chosen_metrics = neg_metrics
        elif align_pos > 0 and align_neg < 0:
            chosen_sign = 1
            chosen_samples = candidate_pos
            chosen_metrics = pos_metrics

        if current_count >= 2:
            predicted_ns = current_intercept + current_slope * t_local_ns
            pos_residual = abs(predicted_ns - candidate_pos[-1][1])
            neg_residual = abs(predicted_ns - candidate_neg[-1][1])
            if neg_residual + 1e-6 < pos_residual * 0.8:
                chosen_sign = -1
                chosen_samples = candidate_neg
                chosen_metrics = neg_metrics
            elif pos_residual + 1e-6 < neg_residual * 0.8:
                chosen_sign = 1
                chosen_samples = candidate_pos
                chosen_metrics = pos_metrics
            elif current_sign < 0 and neg_residual + 1e-6 < pos_residual * 1.05:
                chosen_sign = -1
                chosen_samples = candidate_neg
                chosen_metrics = neg_metrics
            elif current_sign > 0 and pos_residual + 1e-6 < neg_residual * 1.05:
                chosen_sign = 1
                chosen_samples = candidate_pos
                chosen_metrics = pos_metrics

        if pos_metrics is None and neg_metrics is not None:
            chosen_sign = -1
            chosen_samples = candidate_neg
            chosen_metrics = neg_metrics
        elif pos_metrics is not None and neg_metrics is not None:
            pos_rms = pos_metrics[2]
            neg_rms = neg_metrics[2]
            if neg_rms + 1e-9 < pos_rms * 0.8:
                chosen_sign = -1
                chosen_samples = candidate_neg
                chosen_metrics = neg_metrics
            elif pos_rms + 1e-9 < neg_rms * 0.8:
                chosen_sign = 1
                chosen_samples = candidate_pos
                chosen_metrics = pos_metrics
            else:
                chosen_sign = current_sign
                chosen_samples = candidate_pos if current_sign >= 0 else candidate_neg
                chosen_metrics = pos_metrics if current_sign >= 0 else neg_metrics
        elif pos_metrics is None and neg_metrics is None:
            chosen_sign = current_sign
            chosen_samples = candidate_pos if current_sign >= 0 else candidate_neg
            chosen_metrics = None

        with self._state_lock:
            state = self._player_states[player]
            if chosen_sign != state.offset_sign:
                state.offset_sign = chosen_sign
                log.warning(
                    "Offset semantics inverted for %s (sign=%+d)",
                    player,
                    chosen_sign,
                )
            state.raw_offsets = deque(raw_candidates, maxlen=self._window_size)
            state.samples = deque(chosen_samples, maxlen=self._window_size)
            samples_list = list(state.samples)

        self._recompute_model_from_samples(player, samples_list)

    def _trimmed_raw_samples(
        self, base: list[Tuple[int, int]], sample: Tuple[int, int]
    ) -> list[Tuple[int, int]]:
        samples = list(base)
        samples.append(sample)
        if len(samples) > self._window_size:
            samples = samples[-self._window_size :]
        return samples

    def _raw_to_samples(
        self, raw: list[Tuple[int, int]], sign: int
    ) -> list[Tuple[int, int]]:
        result: list[Tuple[int, int]] = []
        for t_local_ns, offset_ns in raw:
            device_ns = t_local_ns + sign * offset_ns
            result.append((t_local_ns, device_ns))
        return result

    def _offset_residual(
        self,
        raw: list[Tuple[int, int]],
        intercept: float,
        slope: float,
        sign: int,
    ) -> float:
        if not raw:
            return float("inf")
        residuals = []
        for t_local_ns, measured_offset in raw:
            predicted_device = intercept + slope * t_local_ns
            predicted_raw = sign * (predicted_device - t_local_ns)
            residuals.append(predicted_raw - measured_offset)
        mean_square = sum(value * value for value in residuals) / len(residuals)
        return math.sqrt(mean_square)

    def _offset_alignment(
        self, raw: list[Tuple[int, int]], intercept: float, slope: float, sign: int
    ) -> float:
        if not raw:
            return 0.0
        alignment = 0.0
        for t_local_ns, measured_offset in raw:
            predicted_device = intercept + slope * t_local_ns
            predicted_raw = sign * (predicted_device - t_local_ns)
            alignment += predicted_raw * measured_offset
        return alignment / len(raw)

    def _evaluate_candidate(
        self, samples: list[Tuple[int, int]]
    ) -> Optional[Tuple[float, float, float]]:
        if len(samples) < 2:
            return None
        return self._huber_fit(samples)

    def _recompute_model_from_samples(
        self, player: str, samples: list[Tuple[int, int]]
    ) -> bool:
        changed = False
        intercept: float
        slope: float
        rms_ns: float
        confidence: float
        mapping_version: int
        sample_count = len(samples)
        if sample_count < 2:
            with self._state_lock:
                state = self._player_states[player]
                state.sample_count = sample_count
                state.last_update = time.monotonic()
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
                or state.sample_count != sample_count
            )
            state.intercept_ns = intercept
            state.slope = slope
            state.rms_ns = rms_ns
            state.confidence = confidence
            state.sample_count = sample_count
            state.last_update = time.monotonic()
            if changed or state.mapping_version == 0:
                state.mapping_version += 1
            mapping_version = state.mapping_version
            offset_sign = state.offset_sign

        log_fn = log.info if changed else log.debug
        log_fn(
            "Mapping update %s: a=%.0fns b=%.9f rms=%.3fms conf=%.3f samples=%d v%s sign=%+d",
            player,
            intercept,
            slope,
            rms_ns / 1_000_000.0,
            confidence,
            sample_count,
            mapping_version,
            offset_sign,
        )

        if changed:
            self._refine_all_pending_for_player(player, mapping_version)
        return changed

    def _refine_all_pending_for_player(self, player: str, version: int) -> None:
        pending: list[Tuple[str, int]] = []
        with self._state_lock:
            for event_id, (t_local_ns, versions) in self._known_events.items():
                last_version = versions.get(player, 0)
                if version > last_version:
                    pending.append((event_id, t_local_ns))
        for event_id, t_local_ns in pending:
            self._refine_event_for_player(
                player, event_id, t_local_ns, version_hint=version
            )

    def _process_event(self, event_id: str, t_local_ns: int) -> None:
        with self._state_lock:
            previous = self._known_events.get(event_id)
            versions: Dict[str, int]
            if previous is None:
                versions = {}
            else:
                _, versions = previous
            self._known_events[event_id] = (t_local_ns, dict(versions))
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
        if entry is None:
            return
        t_local_ns, _ = entry
        players = set(self._connected_players_snapshot())
        with self._state_lock:
            players.update(self._player_states.keys())
        for player in sorted(players):
            self._refine_event_for_player(
                player, event_id, t_local_ns, version_hint=version_hint
            )

    def _refine_event_for_player(
        self,
        player: str,
        event_id: str,
        t_local_ns: int,
        version_hint: Optional[int] = None,
    ) -> None:
        with self._state_lock:
            state = self._player_states.get(player)
            entry = self._known_events.get(event_id)
            if state is None or entry is None:
                return
            _, version_by_player = entry
            last_version = version_by_player.get(player, 0)
            mapping_version = state.mapping_version
            intercept = state.intercept_ns
            slope = state.slope
            confidence = state.confidence
            rms_ns = state.rms_ns
            sample_count = state.sample_count
            offset_sign = state.offset_sign
            effective_version = version_hint or mapping_version
        if effective_version <= last_version:
            return
        if sample_count < 2:
            log.debug(
                "Skipping refinement for %s (event %s): insufficient samples %d",
                player,
                event_id,
                sample_count,
            )
            return
        if confidence < self._conf_min:
            log.debug(
                "Skipping refinement for %s (event %s): low confidence %.3f < %.3f",
                player,
                event_id,
                confidence,
                self._conf_min,
            )
            return
        t_ref_ns = int(intercept + slope * t_local_ns)
        extra = {
            "rms_error_ns": int(rms_ns),
            "heartbeat_count": self._heartbeat_count,
            "samples": sample_count,
            "offset_sign": offset_sign,
        }
        queue_size, queue_capacity = self._bridge.event_queue_load()
        extra["queue"] = {"size": queue_size, "capacity": queue_capacity}
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
            return
        try:
            self._logger.upsert_refinement(
                event_id,
                player,
                t_ref_ns,
                effective_version,
                confidence,
            )
        except Exception:
            log.exception("Persisting refinement failed for %s (%s)", event_id, player)
        else:
            log.info(
                "event %s refined for %s (t_local=%d, t_ref=%d, v%s, conf=%.3f)",
                event_id,
                player,
                t_local_ns,
                t_ref_ns,
                effective_version,
                confidence,
            )
            with self._state_lock:
                entry = self._known_events.get(event_id)
                if entry is not None:
                    t_ns, version_by_player = entry
                    version_by_player = dict(version_by_player)
                    version_by_player[player] = effective_version
                    self._known_events[event_id] = (t_ns, version_by_player)

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


__all__ = ["TimeReconciler"]
