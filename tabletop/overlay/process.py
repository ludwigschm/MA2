"""Utilities for managing the external ArUco overlay process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from tabletop.data.config import ARUCO_OVERLAY_PATH

PathLike = Union[str, Path]
OverlayProcess = Optional[subprocess.Popen]


def _resolve_overlay_path(overlay_path: Optional[PathLike]) -> Path:
    if overlay_path is None:
        return ARUCO_OVERLAY_PATH
    return Path(overlay_path)


def start_overlay(
    process: OverlayProcess = None,
    overlay_path: Optional[PathLike] = None,
) -> OverlayProcess:
    """Ensure the overlay process is running and return its handle.

    Args:
        process: Existing overlay process handle, if any.
        overlay_path: Optional path to the overlay script. Defaults to
            :data:`tabletop.data.config.ARUCO_OVERLAY_PATH` when not provided.

    Returns:
        A running overlay process handle or ``None`` when the overlay could not
        be started.
    """

    if process and process.poll() is None:
        return process

    path = _resolve_overlay_path(overlay_path)
    if not path.exists():
        return None

    try:
        return subprocess.Popen([sys.executable, str(path)])
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Warnung: Overlay konnte nicht gestartet werden: {exc}")
        return None


def stop_overlay(process: OverlayProcess) -> OverlayProcess:
    """Stop a previously started overlay process.

    Args:
        process: Process handle to stop.

    Returns:
        ``None``. The return type mirrors :func:`start_overlay` so callers can
        assign the result back to their stored handle without extra conditionals.
    """

    if not process:
        return None

    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:  # pragma: no cover - defensive fallback
            try:
                process.kill()
            except Exception:
                pass

    return None
