"""Support utilities for optional eye tracker integrations."""

from .config import NeonConfig

try:  # pragma: no cover - optional dependency
    from .neon import NeonClient
except Exception:  # pragma: no cover - fallback when requests is missing
    NeonClient = None  # type: ignore[assignment]

__all__ = ["NeonClient", "NeonConfig"]
