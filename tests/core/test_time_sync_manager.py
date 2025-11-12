import asyncio
import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List

import pytest

from core.time_sync import TimeSyncConfig, TimeSyncManager, TimeSyncMeasurement


@dataclass
class _Scenario:
    intercept_ns: float
    slope: float
    rtt_ns: float
    jitter_ns: float
    noise_ns: float

    def generate_window(self, *, base_host_ns: int, count: int) -> List[TimeSyncMeasurement]:
        measurements: List[TimeSyncMeasurement] = []
        for index in range(count):
            host_send_ns = base_host_ns + index * int(self.rtt_ns)
            jitter = ((-self.jitter_ns) + 2 * self.jitter_ns * (index % 3) / 2.0)
            rtt_ns = max(10_000, int(self.rtt_ns + jitter))
            host_recv_ns = host_send_ns + rtt_ns
            midpoint_ns = host_send_ns + rtt_ns / 2.0
            device_time = self.intercept_ns + self.slope * midpoint_ns
            noise = self.noise_ns * math.sin(index)
            measurement = TimeSyncMeasurement(
                host_send_ns=host_send_ns,
                host_recv_ns=host_recv_ns,
                device_time_ns=int(device_time + noise),
            )
            measurements.append(measurement)
        return measurements


def _manager_for_windows(windows: Iterable[Iterable[TimeSyncMeasurement]], *, config: TimeSyncConfig) -> TimeSyncManager:
    windows_iter = iter(windows)
    config.resync_cooldown_s = 0.0
    config.min_confidence = 0.0
    config.drift_resync_threshold_ppm = 1_000_000.0
    config.q_offset_ns2_per_s = 5.0e9
    config.q_drift_per_s = 1.0e-11

    async def measure(samples: int, timeout: float) -> Iterable[TimeSyncMeasurement]:
        try:
            window = next(windows_iter)
        except StopIteration:  # pragma: no cover - defensive
            return []
        return list(window)

    manager = TimeSyncManager(
        "device",
        measure,
        window_size=config.window_size,
        sample_timeout=0.05,
        resync_interval_s=0.0,
        config=config,
    )
    return manager


def test_outlier_rejection_and_confidence() -> None:
    config = TimeSyncConfig(window_size=36)
    scenario = _Scenario(
        intercept_ns=8_000_000.0,
        slope=1.00001,
        rtt_ns=800_000.0,
        jitter_ns=50_000.0,
        noise_ns=30_000.0,
    )
    base_ns = 5_000_000_000
    window = scenario.generate_window(base_host_ns=base_ns, count=config.window_size + 6)
    # Inject 10% outliers with heavy delay
    for index in range(0, len(window), 10):
        measurement = window[index]
        window[index] = TimeSyncMeasurement(
            host_send_ns=measurement.host_send_ns,
            host_recv_ns=measurement.host_recv_ns + 20_000_000,
            device_time_ns=measurement.device_time_ns + 15_000_000,
        )

    manager = _manager_for_windows([window], config=config)
    offset_seconds = asyncio.run(manager.initial_sync())
    metrics = manager.last_metrics()
    assert metrics is not None
    assert metrics.sample_count >= config.window_size
    assert metrics.outlier_ratio >= 0.1
    assert metrics.confidence >= 0.75
    expected_offset_ns = scenario.intercept_ns + scenario.slope * (base_ns + scenario.rtt_ns / 2.0) - (base_ns + scenario.rtt_ns / 2.0)
    assert offset_seconds * 1_000_000_000 == pytest.approx(expected_offset_ns, abs=500_000.0)


def test_kalman_tracks_positive_drift() -> None:
    config = TimeSyncConfig(window_size=40)
    scenario = _Scenario(
        intercept_ns=12_000_000.0,
        slope=1.0 + 15e-6,
        rtt_ns=600_000.0,
        jitter_ns=0.0,
        noise_ns=0.0,
    )
    windows: List[List[TimeSyncMeasurement]] = []
    for idx in range(5):
        base_host = 1_000_000_000 + idx * 50_000_000
        window = scenario.generate_window(base_host_ns=base_host, count=config.window_size)
        windows.append(window)

    manager = _manager_for_windows(windows, config=config)
    asyncio.run(manager.initial_sync())
    for _ in range(4):
        asyncio.run(manager.maybe_resync())

    drift_ppm = manager.get_drift_ppm()
    assert drift_ppm == pytest.approx(15.0, abs=4.0)
    assert manager.confidence() >= 0.7
    metrics = manager.last_metrics()
    assert metrics is not None
    assert metrics.mode in {"tracking", "stable"}


def test_confidence_decays_on_heartbeat_and_recovers() -> None:
    config = TimeSyncConfig(window_size=32)
    scenario = _Scenario(
        intercept_ns=5_000_000.0,
        slope=0.99999,
        rtt_ns=900_000.0,
        jitter_ns=30_000.0,
        noise_ns=10_000.0,
    )
    window = scenario.generate_window(base_host_ns=3_000_000_000, count=config.window_size)
    manager = _manager_for_windows([window, window], config=config)
    asyncio.run(manager.initial_sync())
    initial_conf = manager.confidence()
    assert initial_conf > 0.6
    asyncio.run(manager.heartbeat())
    asyncio.run(manager.heartbeat())
    assert manager.confidence() < initial_conf
    asyncio.run(manager.maybe_resync())
    assert manager.confidence() >= initial_conf


def test_randomised_property_alignment() -> None:
    rng = random.Random(42)
    config = TimeSyncConfig(window_size=34)
    scenarios: List[_Scenario] = []
    for _ in range(3):
        slope = 1.0 + rng.uniform(-18e-6, 18e-6)
        intercept = rng.uniform(-10_000_000.0, 10_000_000.0)
        rtt = rng.uniform(400_000.0, 1_000_000.0)
        jitter = rng.uniform(10_000.0, 60_000.0)
        noise = rng.uniform(5_000.0, 40_000.0)
        scenarios.append(
            _Scenario(
                intercept_ns=intercept,
                slope=slope,
                rtt_ns=rtt,
                jitter_ns=jitter,
                noise_ns=noise,
            )
        )

    for scenario in scenarios:
        windows: List[List[TimeSyncMeasurement]] = []
        base_ns = rng.randrange(1_000_000_000, 3_000_000_000)
        for idx in range(4):
            window = scenario.generate_window(
                base_host_ns=base_ns + idx * int(30_000_000 + scenario.rtt_ns),
                count=config.window_size,
            )
            windows.append(window)
        manager = _manager_for_windows(windows, config=config)
        asyncio.run(manager.initial_sync())
        for _ in range(3):
            asyncio.run(manager.maybe_resync())
        probe_host = base_ns + 250_000_000
        mapped = manager.map_monotonic_ns(probe_host)
        expected = scenario.intercept_ns + scenario.slope * probe_host
        assert mapped == pytest.approx(expected, abs=600_000.0)
        assert manager.confidence() >= 0.65

