"""Asynchronous device/host time synchronisation utilities."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional, Sequence

from .logging import get_logger

__all__ = [
    "TimeSyncManager",
    "TimeSyncSampleError",
    "TimeSyncMeasurement",
    "TimeSyncMetrics",
    "TimeSyncConfig",
]


class TimeSyncSampleError(RuntimeError):
    """Raised when a time synchronisation measurement fails."""


@dataclass(slots=True)
class TimeSyncMeasurement:
    """Single RTT measurement between host and device.

    The timestamps are defined relative to :func:`time.monotonic_ns` on the host
    and a device monotonic clock.  The measurement assumes a symmetrical delay
    model, using ``RTT / 2`` for the one-way propagation delay.
    """

    host_send_ns: int
    host_recv_ns: int
    device_time_ns: int

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if self.host_recv_ns < self.host_send_ns:
            raise ValueError("host_recv_ns must be >= host_send_ns")

    @property
    def rtt_ns(self) -> int:
        return self.host_recv_ns - self.host_send_ns

    @property
    def midpoint_ns(self) -> float:
        return self.host_send_ns + self.rtt_ns / 2.0

    @property
    def offset_ns(self) -> float:
        return self.device_time_ns - self.midpoint_ns

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        host_send_ns: int,
        host_recv_ns: int,
    ) -> "TimeSyncMeasurement":
        """Coerce a raw payload returned by the device into a measurement."""

        if isinstance(payload, TimeSyncMeasurement):
            return payload

        if isinstance(payload, dict):
            try:
                device_ns = int(payload["device_time_ns"])
            except Exception as exc:  # pragma: no cover - defensive
                raise TimeSyncSampleError("invalid measurement payload") from exc
            host_send = int(payload.get("host_send_ns", host_send_ns))
            host_recv = int(payload.get("host_recv_ns", host_recv_ns))
            return cls(host_send, host_recv, device_ns)

        if isinstance(payload, Sequence) and len(payload) >= 3:
            try:
                host_send = int(payload[0])
                device_ns = int(payload[1])
                host_recv = int(payload[2])
            except Exception as exc:  # pragma: no cover - defensive
                raise TimeSyncSampleError("invalid measurement sequence") from exc
            return cls(host_send, host_recv, device_ns)

        # Legacy float offset payloads.  We approximate the device timestamp by
        # applying the offset to the midpoint of the host timestamps.
        if isinstance(payload, (int, float)):
            midpoint = host_send_ns + (host_recv_ns - host_send_ns) / 2.0
            device_ns = int(midpoint + float(payload) * 1_000_000_000.0)
            return cls(host_send_ns, host_recv_ns, device_ns)

        raise TimeSyncSampleError(f"Unsupported measurement payload: {payload!r}")


@dataclass(slots=True)
class TimeSyncMetrics:
    """Diagnostic information emitted after each synchronisation window."""

    offset_ns: float
    drift_rate: float
    rtt_ns_median: float
    confidence: float
    outlier_ratio: float
    innovation_ns: float
    sample_count: int
    mode: str
    reason: str

    @property
    def offset_ms(self) -> float:
        return self.offset_ns / 1_000_000.0

    @property
    def drift_ppm(self) -> float:
        return self.drift_rate * 1_000_000.0

    @property
    def rtt_ms_median(self) -> float:
        return self.rtt_ns_median / 1_000_000.0

    @property
    def innovation_ms(self) -> float:
        return self.innovation_ns / 1_000_000.0


@dataclass(slots=True)
class TimeSyncConfig:
    """Runtime configuration for :class:`TimeSyncManager`."""

    window_size: int = 32
    outlier_threshold: float = 2.5
    mad_epsilon_ns: float = 1_000.0
    mad_confidence_scale_ns: float = 500_000.0
    innovation_confidence_scale_ns: float = 1_000_000.0
    rtt_confidence_scale_ns: float = 2_000_000.0
    min_confidence: float = 0.6
    stable_confidence: float = 0.85
    stable_window_count: int = 3
    resync_cooldown_s: float = 10.0
    drift_resync_threshold_ppm: float = 20.0
    innovation_resync_sigma: float = 3.0
    heartbeat_half_life_s: float = 45.0
    extrapolation_limit_s: float = 5.0
    q_offset_ns2_per_s: float = 2.0e10
    q_drift_per_s: float = 1.0e-10
    measurement_floor_ns: float = 200_000.0
    drift_smoothing: float = 0.25
    measurement_variance_scale: float = 0.05

    @classmethod
    def from_env(cls) -> "TimeSyncConfig":
        prefix = "MA2_TIMESYNC_"
        kwargs: dict[str, object] = {}
        for field in dataclasses.fields(cls):
            env_key = prefix + field.name.upper()
            raw = os.getenv(env_key)
            if raw is None:
                continue
            try:
                if field.type is int:
                    kwargs[field.name] = int(raw)
                else:
                    kwargs[field.name] = float(raw)
            except Exception:  # pragma: no cover - configuration error
                continue
        config = cls(**kwargs)
        config.window_size = max(3, int(config.window_size))
        config.outlier_threshold = max(1.0, float(config.outlier_threshold))
        return config


@dataclass(slots=True)
class _FilterState:
    offset_ns: float = 0.0
    drift_rate: float = 0.0
    intercept_ns: float = 0.0
    slope: float = 1.0
    last_sync_ts: float = 0.0
    last_update_ns: int = 0
    confidence: float = 0.0
    mode: str = "cold"
    mapping_version: int = 0
    stable_windows: int = 0
    innovation_ns: float = 0.0
    covariance_00: float = 1.0e12
    covariance_01: float = 0.0
    covariance_11: float = 1.0e-6


class TimeSyncManager:
    """Maintain a robust, drift-aware estimate of the device clock mapping."""

    def __init__(
        self,
        device_id: str,
        measure_fn: Callable[[int, float], Awaitable[Iterable[object]]],
        *,
        window_size: Optional[int] = None,
        sample_timeout: float = 0.25,
        resync_interval_s: float = 120.0,
        config: Optional[TimeSyncConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.device_id = device_id
        self._measure_fn = measure_fn
        self.sample_timeout = float(sample_timeout)
        self.resync_interval_s = float(resync_interval_s)
        self._config = config or TimeSyncConfig.from_env()
        if window_size is not None:
            self._config.window_size = max(3, int(window_size))
        self._state = _FilterState()
        self._lock = asyncio.Lock()
        self._log = logger or get_logger(f"core.time_sync.{device_id}")
        self._last_metrics: Optional[TimeSyncMetrics] = None
        self._last_measurement_ts: float = 0.0
        self._outlier_ratio: float = 0.0

    # ------------------------------------------------------------------
    def get_offset_s(self) -> float:
        """Return the current offset estimate in seconds."""

        return self._state.offset_ns / 1_000_000_000.0

    def get_drift_ppm(self) -> float:
        """Return the current drift estimate in parts-per-million."""

        return self._state.drift_rate * 1_000_000.0

    def confidence(self) -> float:
        return self._state.confidence

    def mode(self) -> str:
        return self._state.mode

    def mapping_version(self) -> int:
        return self._state.mapping_version

    def last_metrics(self) -> Optional[TimeSyncMetrics]:
        return self._last_metrics

    def now_monotonic_mapped(self) -> int:
        return self.map_monotonic_ns(time.monotonic_ns())

    def map_monotonic_ns(self, host_ns: int) -> int:
        state = self._state
        return int(state.intercept_ns + state.slope * float(host_ns))

    # ------------------------------------------------------------------
    async def initial_sync(self) -> float:
        async with self._lock:
            return await self._sync_locked(reason="initial")

    async def heartbeat(self) -> None:
        """Allow the confidence to decay in the absence of new measurements."""

        cfg = self._config
        now = time.monotonic()
        elapsed = max(0.0, now - self._state.last_sync_ts)
        if elapsed <= 0.0:
            return
        half_life = max(cfg.heartbeat_half_life_s, 1.0)
        decay = 0.5 ** (elapsed / half_life)
        self._state.confidence *= decay
        self._state.confidence = max(0.0, min(1.0, self._state.confidence))

    async def maybe_resync(self) -> float:
        now = time.monotonic()
        state = self._state
        cfg = self._config

        reason = None
        if state.last_sync_ts:
            interval = now - state.last_sync_ts
            if interval < cfg.resync_cooldown_s:
                return state.offset_ns / 1_000_000_000.0
            if interval < self.resync_interval_s:
                if state.confidence >= cfg.min_confidence and abs(self.get_drift_ppm()) <= cfg.drift_resync_threshold_ppm:
                    return state.offset_ns / 1_000_000_000.0
                if state.confidence >= cfg.min_confidence and abs(state.innovation_ns) < cfg.innovation_resync_sigma * cfg.innovation_confidence_scale_ns:
                    return state.offset_ns / 1_000_000_000.0
        if state.confidence < cfg.min_confidence:
            reason = "confidence"
        elif abs(self.get_drift_ppm()) > cfg.drift_resync_threshold_ppm:
            reason = "drift"
        elif abs(state.innovation_ns) >= cfg.innovation_resync_sigma * cfg.innovation_confidence_scale_ns:
            reason = "innovation"
        else:
            reason = "interval"

        async with self._lock:
            return await self._sync_locked(reason=reason)

    # ------------------------------------------------------------------
    async def _sync_locked(self, *, reason: str) -> float:
        cfg = self._config
        window = max(cfg.window_size, 3)
        try:
            raw_samples = await self._measure_fn(window, self.sample_timeout)
        except asyncio.CancelledError:  # pragma: no cover - defensive
            raise
        except Exception as exc:  # pragma: no cover - network dependent
            self._log.warning(
                "time_sync device=%s reason=%s status=failed error=%s",
                self.device_id,
                reason,
                exc,
            )
            await self.heartbeat()
            return self._state.offset_ns / 1_000_000_000.0

        measurements: list[TimeSyncMeasurement] = []
        host_now_ns = time.monotonic_ns()
        for sample in raw_samples:
            send_ns = host_now_ns
            recv_ns = host_now_ns
            if isinstance(sample, TimeSyncMeasurement):
                measurements.append(sample)
                continue
            if isinstance(sample, dict):
                send_ns = int(sample.get("host_send_ns", host_now_ns))
                recv_ns = int(sample.get("host_recv_ns", host_now_ns))
            elif isinstance(sample, Sequence) and len(sample) >= 3:
                send_ns = int(sample[0])
                recv_ns = int(sample[2])
            try:
                measurements.append(
                    TimeSyncMeasurement.from_payload(
                        sample, host_send_ns=send_ns, host_recv_ns=recv_ns
                    )
                )
            except TimeSyncSampleError as exc:
                self._log.debug("Discarding invalid measurement for %s: %s", self.device_id, exc)
                continue
        if not measurements:
            self._log.warning(
                "time_sync device=%s reason=%s status=empty",
                self.device_id,
                reason,
            )
            await self.heartbeat()
            return self._state.offset_ns / 1_000_000_000.0

        offsets = [sample.offset_ns for sample in measurements]
        rtts = [sample.rtt_ns for sample in measurements]
        median_offset = statistics_median(offsets)
        mad = median_absolute_deviation(offsets, median_offset)
        threshold = max(cfg.mad_epsilon_ns, mad * cfg.outlier_threshold)
        filtered: list[TimeSyncMeasurement] = [
            sample for sample in measurements if abs(sample.offset_ns - median_offset) <= threshold
        ]
        if len(filtered) < max(3, int(0.3 * len(measurements))):
            filtered = measurements
        outlier_ratio = 1.0 - (len(filtered) / float(len(measurements)))
        self._outlier_ratio = outlier_ratio

        offsets = [sample.offset_ns for sample in filtered]
        rtts = [sample.rtt_ns for sample in filtered]
        midpoints = [sample.midpoint_ns for sample in filtered]
        median_offset = statistics_median(offsets)
        median_rtt = statistics_median(rtts)
        median_midpoint = statistics_median(midpoints)
        mad = max(cfg.mad_epsilon_ns, median_absolute_deviation(offsets, median_offset))

        dt_ns = 0
        now_ns = int(median_midpoint)
        if self._state.last_update_ns:
            dt_ns = max(0, now_ns - self._state.last_update_ns)
        predicted_offset, _ = self._kalman_predict(dt_ns)

        measurement = median_offset
        innovation = measurement - predicted_offset
        scaled_mad = mad * cfg.measurement_variance_scale
        measurement_variance = max(scaled_mad**2, cfg.measurement_floor_ns**2)
        offset, _ = self._kalman_update(measurement, measurement_variance)
        residual = measurement - offset

        slope_estimate = regression_slope(filtered)
        drift_measure = slope_estimate - 1.0
        smoothing = max(0.0, min(1.0, cfg.drift_smoothing))
        previous_drift = self._state.drift_rate
        drift = (1.0 - smoothing) * previous_drift + smoothing * drift_measure
        slope = 1.0 + drift
        intercept = offset - drift * now_ns
        mode = self._state.mode
        stable_windows = self._state.stable_windows
        if offset != offset or math.isinf(offset):  # pragma: no cover - defensive
            offset = self._state.offset_ns
        if drift != drift or math.isinf(drift):  # pragma: no cover - defensive
            drift = self._state.drift_rate

        confidence = self._compute_confidence(
            sample_count=len(filtered),
            mad_ns=mad,
            rtt_ns=median_rtt,
            innovation_ns=residual,
        )

        if confidence >= cfg.stable_confidence:
            stable_windows += 1
        else:
            stable_windows = 0
        if stable_windows >= cfg.stable_window_count:
            mode = "stable"
        elif confidence >= cfg.min_confidence:
            mode = "tracking"
        else:
            mode = "cold"

        self._state.offset_ns = offset
        self._state.drift_rate = drift
        self._state.intercept_ns = intercept
        self._state.slope = slope
        self._state.last_sync_ts = time.monotonic()
        self._state.last_update_ns = now_ns
        self._state.confidence = confidence
        self._state.mode = mode
        self._state.stable_windows = stable_windows
        self._state.innovation_ns = residual
        self._state.mapping_version += 1

        if outlier_ratio > 0.3:
            self._log.warning(
                "time_sync device=%s warning=outliers ratio=%.2f", self.device_id, outlier_ratio
            )

        metrics = TimeSyncMetrics(
            offset_ns=offset,
            drift_rate=drift,
            rtt_ns_median=median_rtt,
            confidence=confidence,
            outlier_ratio=outlier_ratio,
            innovation_ns=residual,
            sample_count=len(filtered),
            mode=mode,
            reason=reason,
        )
        self._last_metrics = metrics
        self._last_measurement_ts = self._state.last_sync_ts

        self._log.info(
            (
                "time_sync device=%s reason=%s status=ok mode=%s "
                "offset_ms=%.3f drift_ppm=%.3f rtt_ms_median=%.3f "
                "confidence=%.3f outlier_ratio=%.2f innovation_ms=%.3f"
            ),
            self.device_id,
            reason,
            mode,
            metrics.offset_ms,
            metrics.drift_ppm,
            metrics.rtt_ms_median,
            confidence,
            outlier_ratio,
            metrics.innovation_ms,
        )

        return offset / 1_000_000_000.0

    # ------------------------------------------------------------------
    def _kalman_predict(self, dt_ns: int) -> tuple[float, float]:
        state = self._state
        if dt_ns <= 0:
            return state.offset_ns, state.drift_rate

        dt = dt_ns
        offset = state.offset_ns + state.drift_rate * dt
        drift = state.drift_rate
        p00 = state.covariance_00
        p01 = state.covariance_01
        p11 = state.covariance_11
        cfg = self._config
        q_offset = cfg.q_offset_ns2_per_s * (dt / 1_000_000_000.0)
        q_drift = cfg.q_drift_per_s * (dt / 1_000_000_000.0)

        p00 = p00 + dt * (p01 + p01) + dt * dt * p11 + q_offset
        p01 = p01 + dt * p11
        p11 = p11 + q_drift

        state.covariance_00 = p00
        state.covariance_01 = p01
        state.covariance_11 = p11
        state.offset_ns = offset
        state.drift_rate = drift
        return offset, drift

    def _kalman_update(self, measurement: float, variance: float) -> tuple[float, float]:
        state = self._state
        p00 = state.covariance_00
        p01 = state.covariance_01
        p11 = state.covariance_11
        innovation_cov = p00 + variance
        if innovation_cov <= 0:
            innovation_cov = variance if variance > 0 else 1.0
        kalman_gain_0 = p00 / innovation_cov
        kalman_gain_1 = p01 / innovation_cov
        innovation = measurement - state.offset_ns
        state.offset_ns = state.offset_ns + kalman_gain_0 * innovation
        drift_temp = state.drift_rate + kalman_gain_1 * innovation
        state.covariance_00 = (1 - kalman_gain_0) * p00
        state.covariance_01 = (1 - kalman_gain_0) * p01
        state.covariance_11 = p11 - kalman_gain_1 * p01
        return state.offset_ns, drift_temp

    def _compute_confidence(
        self,
        *,
        sample_count: int,
        mad_ns: float,
        rtt_ns: float,
        innovation_ns: float,
    ) -> float:
        cfg = self._config
        sample_term = min(1.0, sample_count / max(cfg.window_size, 1))
        mad_term = math.exp(-((mad_ns / cfg.mad_confidence_scale_ns) ** 2))
        innovation_term = math.exp(-((abs(innovation_ns) / cfg.innovation_confidence_scale_ns) ** 2))
        rtt_term = math.exp(-((rtt_ns / cfg.rtt_confidence_scale_ns) ** 2))
        confidence = sample_term * mad_term * innovation_term * rtt_term
        return max(0.0, min(1.0, confidence))


def statistics_median(values: Sequence[float]) -> float:
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def median_absolute_deviation(values: Sequence[float], median: float) -> float:
    if not values:
        return 0.0
    deviations = [abs(float(v) - median) for v in values]
    return statistics_median(deviations)


def regression_slope(samples: Sequence[TimeSyncMeasurement]) -> float:
    if len(samples) < 2:
        return 1.0
    hosts = [sample.midpoint_ns for sample in samples]
    devices = [sample.device_time_ns for sample in samples]
    mean_host = sum(hosts) / len(hosts)
    mean_device = sum(devices) / len(devices)
    numerator = 0.0
    denominator = 0.0
    for host, device in zip(hosts, devices):
        centered_host = host - mean_host
        numerator += centered_host * (device - mean_device)
        denominator += centered_host * centered_host
    if abs(denominator) < 1e-12:
        return 1.0
    return numerator / denominator
