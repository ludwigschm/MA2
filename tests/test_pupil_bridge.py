import math
import sys
import time
from pathlib import Path
from typing import Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabletop.pupil_bridge import PupilBridge


class _FakeDevice:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Tuple[object, ...]]] = []

    def send_event(self, *args):  # type: ignore[no-untyped-def]
        timestamp = time.perf_counter_ns()
        self.calls.append((timestamp, args))


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOW_LATENCY_DISABLED", "0")
    monkeypatch.delenv("LOW_LATENCY_OFF", raising=False)
    monkeypatch.delenv("EVENT_BATCH_WINDOW_MS", raising=False)
    monkeypatch.delenv("EVENT_BATCH_SIZE", raising=False)
    bridge = PupilBridge(device_mapping={})
    device = _FakeDevice()
    bridge._device_by_player["VP1"] = device  # type: ignore[attr-defined]
    yield bridge, device
    bridge._sender_stop.set()  # type: ignore[attr-defined]
    if bridge._event_queue is not None:  # type: ignore[attr-defined]
        try:
            bridge._event_queue.put_nowait(bridge._queue_sentinel)  # type: ignore[attr-defined]
        except Exception:
            pass
    if bridge._sender_thread is not None:  # type: ignore[attr-defined]
        bridge._sender_thread.join(timeout=1.0)


def test_sync_events_bypass_batching(bridge):
    pupil_bridge, device = bridge
    start_ns = time.perf_counter_ns()
    pupil_bridge.send_event("sync.test", "VP1", {"event_id": "evt"})
    assert device.calls, "sync event should dispatch immediately"
    latency_ns = device.calls[0][0] - start_ns
    assert latency_ns < 20_000_000  # < 20 ms
    if pupil_bridge._event_queue is not None:  # type: ignore[attr-defined]
        assert pupil_bridge._event_queue.qsize() == 0


def test_sync_dispatch_latency_95p_below_threshold(bridge):
    pupil_bridge, device = bridge
    latencies_ms: list[float] = []
    for index in range(100):
        start_ns = time.perf_counter_ns()
        pupil_bridge.send_event("sync.flash_start", "VP1", {"event_id": f"evt{index}"})
        dispatch_ns = device.calls[-1][0]
        latencies_ms.append((dispatch_ns - start_ns) / 1_000_000.0)
        time.sleep(0.01)

    latencies_ms.sort()
    percentile_index = max(0, math.ceil(0.95 * len(latencies_ms)) - 1)
    assert latencies_ms[percentile_index] < 20.0
