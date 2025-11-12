#!/usr/bin/env python3
"""End-to-end smoke test covering time sync, edge APIs, and cloud queue."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse, urlunparse

from core.config import (
    API_HEALTH_PATHS,
    CLOUD_API_SETTINGS,
    EDGE_API_SETTINGS,
    EDGE_BASE_URLS,
    EDGE_REFINE_PATHS,
)
from core.time_sync import (
    TimeSyncConfig,
    TimeSyncManager,
    TimeSyncMeasurement,
    TimeSyncMetrics,
    TimeSyncSampleError,
)
from tabletop.logging.pupylabs_cloud import PupylabsCloudLogger
from tabletop.utils.http_client import (
    ApiDnsError,
    ApiError,
    ApiNotFound,
    HttpClient,
)

TIME_SYNC_PATH_CANDIDATES: Sequence[str] = (
    "/api/time/sync",
    "/time/sync",
    "/v1/time/sync",
    "/api/time_sync",
)
DEVICE_TIME_KEYS: Sequence[str] = (
    "device_time_ns",
    "device_ns",
    "deviceTimestampNs",
    "device_time",
    "timestamp_ns",
)
DEFAULT_QUEUE_PATH = Path("logs/e2e_cloud_queue.ndjson")


class SmokeTestError(RuntimeError):
    """Base error for e2e smoke test failures."""


@dataclass
class EdgeCheckResult:
    base_url: str
    health_path: Optional[str]
    refine_statuses: list[int]
    failures: list[str]


@dataclass
class CloudCheckResult:
    base_url: str
    queue_path: Optional[Path]
    queued_events: int
    delivered: bool
    failure: Optional[str] = None


class EdgeProbe:
    """Helper for probing a single edge device via HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        time_sync_config: TimeSyncConfig,
        samples: int,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._config = dataclasses.replace(time_sync_config)
        self._samples = max(3, samples)
        self._timeout = max(0.05, timeout)
        self._client = HttpClient(self.base_url, settings=EDGE_API_SETTINGS)
        self._time_sync_path: Optional[str] = None
        self._health_path: Optional[str] = None
        self._refine_path: Optional[str] = None
        self._metrics: Optional[TimeSyncMetrics] = None
        self._manager = TimeSyncManager(
            device_id=self.base_url,
            measure_fn=self._collect_measurements,
            window_size=self._config.window_size,
            sample_timeout=self._timeout,
            config=self._config,
        )

    @property
    def metrics(self) -> Optional[TimeSyncMetrics]:
        return self._metrics

    @property
    def manager(self) -> TimeSyncManager:
        return self._manager

    @property
    def client(self) -> HttpClient:
        return self._client

    @property
    def health_path(self) -> Optional[str]:
        return self._health_path

    async def warmup(self) -> TimeSyncMeasurement:
        await self._manager.initial_sync()
        metrics = self._manager.last_metrics()
        if metrics is None:
            raise SmokeTestError(f"No timesync metrics emitted by {self.base_url}")
        self._metrics = metrics
        return metrics

    async def _collect_measurements(
        self, samples: int, timeout: float
    ) -> Iterable[TimeSyncMeasurement]:
        path = await self._ensure_time_sync_path(timeout)
        if not path:
            return []
        collected: list[TimeSyncMeasurement] = []
        desired = min(max(3, samples), self._samples)
        for _ in range(desired):
            host_send_ns = time.monotonic_ns()
            try:
                response = await asyncio.to_thread(
                    self._client.get,
                    path,
                    timeout=timeout,
                    idempotent=True,
                    allow_statuses={200},
                )
            except ApiError:
                break
            host_recv_ns = time.monotonic_ns()
            try:
                payload = response.json()
            except ValueError:
                break
            device_ns = _extract_device_time(payload)
            if device_ns is None:
                break
            try:
                measurement = TimeSyncMeasurement.from_payload(
                    {"device_time_ns": device_ns},
                    host_send_ns=host_send_ns,
                    host_recv_ns=host_recv_ns,
                )
            except TimeSyncSampleError:
                continue
            collected.append(measurement)
            await asyncio.sleep(0.01)
        return collected

    async def _ensure_time_sync_path(self, timeout: float) -> Optional[str]:
        if self._time_sync_path:
            return self._time_sync_path
        for candidate in TIME_SYNC_PATH_CANDIDATES:
            try:
                response = await asyncio.to_thread(
                    self._client.get,
                    candidate,
                    timeout=timeout,
                    idempotent=True,
                    allow_statuses={200},
                )
            except ApiNotFound:
                continue
            except ApiError:
                continue
            try:
                payload = response.json()
            except ValueError:
                continue
            if _extract_device_time(payload) is None:
                continue
            self._time_sync_path = candidate
            return candidate
        return None

    def check_health(self) -> Optional[str]:
        path = self._client.health_check(API_HEALTH_PATHS)
        self._health_path = path
        return path

    def _ensure_refine_path(self) -> str:
        if self._refine_path:
            return self._refine_path
        path = self._client.discover_path(
            "POST",
            EDGE_REFINE_PATHS,
            json={"probe": True},
            allow_statuses={200, 204, 400},
        )
        self._refine_path = path or EDGE_REFINE_PATHS[0]
        return self._refine_path

    def refine_events(self, count: int) -> list[int]:
        path = self._ensure_refine_path()
        statuses: list[int] = []
        mapping_version = max(0, int(self._manager.mapping_version()))
        for _ in range(count):
            payload = {
                "event_id": f"e2e-{uuid.uuid4().hex}",
                "t_ref_ns": int(self._manager.map_monotonic_ns(time.monotonic_ns())),
                "confidence": float(self._manager.confidence()),
                "mapping_version": mapping_version,
                "refined": True,
                "provisional": False,
                "origin_device": self.base_url,
            }
            try:
                response = self._client.post(
                    path,
                    json=payload,
                    allow_statuses={200, 204},
                )
            except ApiError as exc:
                raise SmokeTestError(f"Refine failed for {self.base_url}: {exc}") from exc
            status = getattr(response, "status_code", None) or 0
            if status == 200:
                _validate_refine_response(response)
            statuses.append(status)
            time.sleep(0.05)
        return statuses

    def close(self) -> None:
        self._client.close()


def _extract_device_time(payload: object) -> Optional[int]:
    if isinstance(payload, dict):
        for key in DEVICE_TIME_KEYS:
            if key in payload:
                try:
                    return int(payload[key])
                except (TypeError, ValueError):
                    return None
        nested = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(nested, dict):
            return _extract_device_time(nested)
    return None


def _validate_refine_response(response: object) -> None:
    try:
        body = response.json()
    except Exception as exc:  # pragma: no cover - defensive
        raise SmokeTestError("Invalid JSON body from refine endpoint") from exc
    if not isinstance(body, dict):
        raise SmokeTestError("Unexpected refine response schema (not an object)")
    result = body.get("result")
    if result is None:
        return
    if not isinstance(result, dict):
        raise SmokeTestError("Unexpected refine response schema (result)")
    accepted = result.get("accepted")
    if accepted is not None and not isinstance(accepted, bool):
        raise SmokeTestError("Unexpected refine response schema (accepted)")


def _override_host(base_url: str, host: str) -> str:
    parsed = urlparse(base_url or "")
    scheme = parsed.scheme or "https"
    path = parsed.path or ""
    params = parsed.params or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = host + port
    return urlunparse((scheme, netloc, path, params, query, fragment))


def _queue_count(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def _resolve_cloud_config(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    base_url = args.cloud_url or os.environ.get("PUPYLABS_CLOUD_BASE_URL")
    if not base_url:
        base_url = os.environ.get("PUPYLABS_BASE_URL")
    if not base_url:
        base_url = os.environ.get("PUPYLABS_CLOUD_URL")
    api_key = args.cloud_api_key or os.environ.get("PUPYLABS_CLOUD_API_KEY")
    if not api_key:
        api_key = os.environ.get("PUPYLABS_API_KEY")
    return base_url, api_key


async def _run_timesync(args: argparse.Namespace, config: TimeSyncConfig) -> list[EdgeProbe]:
    probes: list[EdgeProbe] = []
    failures: list[str] = []
    for url in args.edge_urls:
        probe = EdgeProbe(
            url,
            time_sync_config=config,
            samples=args.timesync_samples,
            timeout=args.timesync_timeout,
        )
        try:
            metrics = await probe.warmup()
        except SmokeTestError as exc:
            print(f"[timesync] {url}: ERROR {exc}")
            failures.append(url)
            probe.close()
            continue
        print(
            f"[timesync] {url}: offset_ms={metrics.offset_ms:.3f} "
            f"drift_ppm={metrics.drift_ppm:.3f} confidence={metrics.confidence:.3f}"
        )
        probes.append(probe)
    if failures:
        raise SmokeTestError(f"Timesync warmup failed for: {', '.join(failures)}")
    unstable = [
        probe for probe in probes if probe.metrics and probe.metrics.confidence < config.stable_confidence
    ]
    if unstable:
        for probe in unstable:
            metrics = probe.metrics
            assert metrics is not None
            print(
                f"[timesync] {probe.base_url}: WARN confidence {metrics.confidence:.3f} < {config.stable_confidence:.2f}"
            )
        raise SmokeTestError("TimeSync confidence below stable threshold")
    return probes


def _run_edge_checks(args: argparse.Namespace, probes: Sequence[EdgeProbe]) -> list[EdgeCheckResult]:
    results: list[EdgeCheckResult] = []
    for probe in probes:
        failures: list[str] = []
        health_path: Optional[str] = None
        try:
            health_path = probe.check_health()
        except ApiError as exc:
            failures.append(f"health check failed: {exc}")
        else:
            if health_path:
                print(f"[edge] {probe.base_url}: health ok via {health_path}")
            else:
                failures.append("health endpoint not discovered")
        refine_statuses: list[int] = []
        if not failures:
            try:
                refine_statuses = probe.refine_events(args.refine_count)
            except SmokeTestError as exc:
                failures.append(str(exc))
            else:
                status_summary = ", ".join(str(code) for code in refine_statuses)
                print(f"[edge] {probe.base_url}: refine statuses [{status_summary}]")
        results.append(
            EdgeCheckResult(
                base_url=probe.base_url,
                health_path=health_path,
                refine_statuses=refine_statuses,
                failures=failures,
            )
        )
    return results


def _run_cloud_check(args: argparse.Namespace) -> Optional[CloudCheckResult]:
    base_url, api_key = _resolve_cloud_config(args)
    if not base_url or not api_key:
        print("[cloud] Skipping cloud ingest (missing base URL or API key)")
        return None
    if args.simulate_cloud_dns_fail:
        override = args.simulate_cloud_dns_fail
        if override is True:  # pragma: no cover - argparse quirk
            override = "cloud.invalid"
        base_url = _override_host(base_url, str(override))
        print(f"[cloud] Simulating DNS failure using host {override}")
    queue_path = Path(args.cloud_queue or DEFAULT_QUEUE_PATH)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        queue_path.unlink()
    except FileNotFoundError:
        pass
    client = HttpClient(base_url, settings=CLOUD_API_SETTINGS)
    logger = PupylabsCloudLogger(client, api_key, queue_path=queue_path)
    event = {
        "event_id": f"e2e-{uuid.uuid4().hex}",
        "actor": "smoke-test",
        "action": "ping",
        "t_local_ns": time.monotonic_ns(),
    }
    failure: Optional[str] = None
    delivered = False
    try:
        logger.send(event)
        logger.flush()
    except ApiDnsError as exc:
        failure = f"dns error: {exc}"
    except ApiError as exc:
        failure = f"cloud error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        failure = f"unexpected error: {exc}"
    finally:
        client.close()
    queued = _queue_count(queue_path)
    if failure:
        print(f"[cloud] Failure encountered: {failure}")
    if queued:
        print(f"[cloud] Offline queue has {queued} event(s) at {queue_path}")
    else:
        delivered = failure is None
        if delivered:
            print("[cloud] Event delivered successfully")
    return CloudCheckResult(
        base_url=base_url,
        queue_path=queue_path,
        queued_events=queued,
        delivered=delivered,
        failure=failure,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edge-url",
        dest="edge_urls",
        action="append",
        default=None,
        help="Edge base URL (may be passed multiple times). Defaults to EDGE_BASE_URLS",
    )
    parser.add_argument(
        "--timesync-samples",
        type=int,
        default=32,
        help="Number of time sync samples to request per device (default: 32)",
    )
    parser.add_argument(
        "--timesync-timeout",
        type=float,
        default=0.3,
        help="Timeout (seconds) for individual time sync samples",
    )
    parser.add_argument(
        "--refine-count",
        type=int,
        default=3,
        help="Number of refine events to emit per device",
    )
    parser.add_argument(
        "--cloud-url",
        default=None,
        help="Override cloud ingest base URL",
    )
    parser.add_argument(
        "--cloud-api-key",
        default=None,
        help="Override cloud API key",
    )
    parser.add_argument(
        "--cloud-queue",
        default=str(DEFAULT_QUEUE_PATH),
        help="Path for the offline cloud queue",
    )
    parser.add_argument(
        "--simulate-cloud-dns-fail",
        nargs="?",
        const="cloud.invalid",
        help="Override the cloud host to force DNS failures (default host: cloud.invalid)",
    )
    args = parser.parse_args(argv)
    if not args.edge_urls:
        args.edge_urls = list(EDGE_BASE_URLS)
    return args


async def _async_main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    config = TimeSyncConfig.from_env()
    try:
        probes = await _run_timesync(args, config)
    except SmokeTestError as exc:
        print(f"[fatal] {exc}")
        for probe in probes if 'probes' in locals() else []:
            probe.close()
        return 1
    edge_results = _run_edge_checks(args, probes)
    edge_failures = [result for result in edge_results if result.failures]
    if edge_failures:
        for result in edge_failures:
            for failure in result.failures:
                print(f"[fatal] {result.base_url}: {failure}")
        for probe in probes:
            probe.close()
        return 2
    cloud_result = _run_cloud_check(args)
    for probe in probes:
        probe.close()
    if cloud_result is None:
        print("[summary] Edge checks passed; cloud step skipped")
        return 0
    if cloud_result.failure and cloud_result.queued_events == 0:
        print("[fatal] Cloud failure without queued events")
        return 3
    if cloud_result.failure:
        print("[summary] Cloud failure tolerated via offline queue")
        return 0
    print("[summary] Edge + cloud checks passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    try:
        return asyncio.run(_async_main(argv))
    except KeyboardInterrupt:
        print("[fatal] Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
