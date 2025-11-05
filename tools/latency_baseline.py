#!/usr/bin/env python3
"""Compute latency baseline metrics from sqlite logs or streamed events."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

WINDOW_SECONDS_DEFAULT = 60.0


@dataclass
class Sample:
    """Represents a single host/device offset observation."""

    host_ns: int
    device_ns: int
    offset_ns: int
    mono_ns: Optional[int] = None

    @property
    def time_seconds(self) -> Optional[float]:
        if self.mono_ns is None:
            return None
        return self.mono_ns / 1_000_000_000.0


class BaselineComputationError(RuntimeError):
    """Raised when the baseline cannot be computed."""


def _find_latest_db(default_dir: Path) -> Optional[Path]:
    candidates = sorted(default_dir.glob("*.sqlite3"))
    if candidates:
        return candidates[-1]
    return None


def _load_samples_from_db(db_path: Path, window_seconds: float) -> List[Sample]:
    if not db_path.exists():
        raise BaselineComputationError(f"Log database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT MAX(t_mono_ns) AS max_ns FROM events")
        row = cur.fetchone()
        if not row or row["max_ns"] is None:
            raise BaselineComputationError("No events available in the log database")
        max_ns = int(row["max_ns"])
        window_ns = int(window_seconds * 1_000_000_000)
        cutoff = max(0, max_ns - window_ns)
        rows = conn.execute(
            """
            SELECT action, payload, t_mono_ns
            FROM events
            WHERE t_mono_ns >= ?
            ORDER BY t_mono_ns ASC
            """,
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    host_pending: Dict[str, Tuple[int, int]] = {}
    device_pending: Dict[str, Tuple[int, int]] = {}
    samples: List[Sample] = []

    for entry in rows:
        action = entry["action"]
        try:
            payload = json.loads(entry["payload"] or "{}")
        except json.JSONDecodeError:
            continue
        mono_ns = int(entry["t_mono_ns"]) if entry["t_mono_ns"] is not None else None

        sample = _sample_from_payload(payload, mono_ns)
        if sample is not None:
            samples.append(sample)
            continue

        event_id = _extract_event_id(payload)
        if not event_id:
            continue

        host_ns = _extract_int(payload, ("t_host_ns", "host_ns", "t_local_ns"))
        device_ns = _extract_int(payload, ("t_device_ns", "device_ns", "t_remote_ns"))

        if action == "sync.host_ns" and host_ns is not None:
            if event_id in device_pending:
                device_ns_stored, mono_ns_stored = device_pending.pop(event_id)
                samples.append(
                    Sample(
                        host_ns=host_ns,
                        device_ns=device_ns_stored,
                        offset_ns=int(device_ns_stored - host_ns),
                        mono_ns=mono_ns_stored,
                    )
                )
            else:
                host_pending[event_id] = (host_ns, mono_ns or 0)
        elif action.startswith("sync.") or action.startswith("fix."):
            if device_ns is None:
                continue
            if event_id in host_pending:
                host_ns_stored, mono_ns_stored = host_pending.pop(event_id)
                samples.append(
                    Sample(
                        host_ns=host_ns_stored,
                        device_ns=device_ns,
                        offset_ns=int(device_ns - host_ns_stored),
                        mono_ns=mono_ns_stored,
                    )
                )
            else:
                device_pending[event_id] = (device_ns, mono_ns or 0)

    return samples


def _sample_from_payload(payload: Dict[str, object], mono_ns: Optional[int]) -> Optional[Sample]:
    host_ns = _extract_int(payload, ("t_host_ns", "host_ns", "t_local_ns"))
    device_ns = _extract_int(payload, ("t_device_ns", "device_ns", "t_remote_ns"))
    offset_ns = _extract_int(payload, ("offset_ns", "delta_ns"))

    if host_ns is not None and device_ns is not None:
        return Sample(
            host_ns=host_ns,
            device_ns=device_ns,
            offset_ns=int(device_ns - host_ns),
            mono_ns=mono_ns,
        )
    if offset_ns is not None and host_ns is not None:
        return Sample(
            host_ns=host_ns,
            device_ns=int(host_ns + offset_ns),
            offset_ns=offset_ns,
            mono_ns=mono_ns,
        )
    if offset_ns is not None and device_ns is not None:
        return Sample(
            host_ns=int(device_ns - offset_ns),
            device_ns=device_ns,
            offset_ns=offset_ns,
            mono_ns=mono_ns,
        )
    return None


def _extract_event_id(payload: Dict[str, object]) -> Optional[str]:
    for key in ("event_id", "id", "uid"):
        raw = payload.get(key)
        if raw is not None:
            return str(raw)
    return None


def _extract_int(payload: Dict[str, object], keys: Iterable[str]) -> Optional[int]:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _compute_metrics(samples: List[Sample], window_seconds: float) -> Dict[str, object]:
    if not samples:
        raise BaselineComputationError("No sync samples were found in the requested window")

    offsets = [sample.offset_ns for sample in samples]
    mean_offset = statistics.fmean(offsets)
    median_offset = statistics.median(offsets)
    rms = math.sqrt(statistics.fmean((offset - mean_offset) ** 2 for offset in offsets))

    time_values = [sample.time_seconds for sample in samples if sample.time_seconds is not None]
    drift_ns_per_s = None
    if len(time_values) >= 2:
        t0 = time_values[0]
        shifted_times = [t - t0 for t in time_values]
        offsets_for_times = [
            sample.offset_ns
            for sample in samples
            if sample.time_seconds is not None
        ]
        drift_ns_per_s = _linear_drift(shifted_times, offsets_for_times)

    metrics = {
        "samples": len(samples),
        "window_seconds": window_seconds,
        "offset_mean_ns": mean_offset,
        "offset_median_ns": median_offset,
        "offset_rms_ns": rms,
        "drift_ns_per_s": drift_ns_per_s,
    }
    return metrics


def _linear_drift(times_s: List[float], offsets_ns: List[float]) -> float:
    if len(times_s) != len(offsets_ns) or len(times_s) < 2:
        raise BaselineComputationError("Need at least two samples with timestamps to compute drift")
    mean_t = statistics.fmean(times_s)
    mean_offset = statistics.fmean(offsets_ns)
    numerator = 0.0
    denominator = 0.0
    for t, offset in zip(times_s, offsets_ns):
        dt = t - mean_t
        numerator += dt * (offset - mean_offset)
        denominator += dt * dt
    if denominator == 0.0:
        return 0.0
    slope_ns_per_s = numerator / denominator
    return slope_ns_per_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        help="Path to the sqlite event log. Defaults to the newest file in ./logs",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=WINDOW_SECONDS_DEFAULT,
        help="Time window in seconds to include in the baseline calculation (default: 60)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write the JSON metrics. If omitted, metrics are printed to stdout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_seconds = max(1.0, float(args.window))

    db_path: Optional[Path] = args.db
    if db_path is None:
        logs_dir = Path("logs")
        db_path = _find_latest_db(logs_dir)
        if db_path is None:
            raise BaselineComputationError(
                "No sqlite logs were found. Specify --db or provide live events."
            )

    samples = _load_samples_from_db(db_path, window_seconds)
    metrics = _compute_metrics(samples, window_seconds)

    if not args.quiet:
        print("Latency baseline")
        print(f"  Samples: {metrics['samples']}")
        print(f"  Window: {metrics['window_seconds']:.1f} s")
        print(f"  Mean offset: {metrics['offset_mean_ns']:.1f} ns")
        print(f"  Median offset: {metrics['offset_median_ns']:.1f} ns")
        print(f"  RMS jitter: {metrics['offset_rms_ns']:.1f} ns")
        drift_value = metrics["drift_ns_per_s"]
        if drift_value is not None:
            print(f"  Drift: {drift_value:.3f} ns/s")
        else:
            print("  Drift: n/a")

    json_payload = json.dumps(metrics, indent=2)
    if args.json:
        args.json.write_text(json_payload, encoding="utf-8")
    else:
        print(json_payload)


if __name__ == "__main__":
    try:
        main()
    except BaselineComputationError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
