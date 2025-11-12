"""Central configuration helpers for API clients and discovery."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Sequence

__all__ = [
    "ApiClientSettings",
    "EDGE_BASE_URLS",
    "CLOUD_BASE_URL",
    "EDGE_API_SETTINGS",
    "CLOUD_API_SETTINGS",
    "API_HEALTH_PATHS",
    "EDGE_REFINE_PATHS",
    "DEFAULT_OFFLINE_QUEUE",
]


def _coerce_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_base_url(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned.rstrip("/")


@dataclass(frozen=True)
class ApiClientSettings:
    """Tunable parameters shared by API clients."""

    timeout_s: float = 2.5
    retry_max: int = 4
    backoff_factor: float = 0.25
    circuit_threshold: int = 5
    circuit_reset_s: float = 30.0
    name: str | None = None


EDGE_BASE_URLS: List[str] = [
    "http://192.168.137.83:8080",
    "http://192.168.137.92:8080",
]
"""Default base URLs for the two Neon edge devices."""


CLOUD_BASE_URL: str | None = _normalise_base_url(
    os.environ.get("PUPYLABS_CLOUD_URL")
    or os.environ.get("PUPYLABS_CLOUD_BASE_URL")
    or os.environ.get("PUPYLABS_BASE_URL")
)
"""Configured Pupylabs cloud endpoint (if any)."""


EDGE_API_SETTINGS = ApiClientSettings(
    timeout_s=_coerce_float(os.environ.get("EDGE_API_TIMEOUT_S"), 1.5),
    retry_max=_coerce_int(os.environ.get("EDGE_API_RETRY_MAX"), 3),
    backoff_factor=_coerce_float(os.environ.get("EDGE_API_BACKOFF_FACTOR"), 0.2),
    circuit_threshold=_coerce_int(os.environ.get("EDGE_API_CIRCUIT_THRESHOLD"), 4),
    circuit_reset_s=_coerce_float(os.environ.get("EDGE_API_CIRCUIT_RESET_S"), 15.0),
    name="edge",
)


CLOUD_API_SETTINGS = ApiClientSettings(
    timeout_s=_coerce_float(os.environ.get("CLOUD_API_TIMEOUT_S"), 2.5),
    retry_max=_coerce_int(os.environ.get("CLOUD_API_RETRY_MAX"), 4),
    backoff_factor=_coerce_float(os.environ.get("CLOUD_API_BACKOFF_FACTOR"), 0.3),
    circuit_threshold=_coerce_int(os.environ.get("CLOUD_API_CIRCUIT_THRESHOLD"), 5),
    circuit_reset_s=_coerce_float(os.environ.get("CLOUD_API_CIRCUIT_RESET_S"), 30.0),
    name="cloud",
)


API_HEALTH_PATHS: Sequence[str] = ("/health", "/api/health", "/v1/health")
"""Candidate endpoints for discovering service health URLs."""


EDGE_REFINE_PATHS: Sequence[str] = (
    "/api/annotations/refine",
    "/annotations/refine",
    "/v1/annotations/refine",
)
"""Candidate endpoints for refinement calls on edge devices."""


DEFAULT_OFFLINE_QUEUE = Path(
    os.environ.get("PUPYLABS_OFFLINE_QUEUE", "") or Path.home() / ".cache" / "ma2" / "cloud_queue.ndjson"
)
"""Location where offline cloud events should be queued."""

