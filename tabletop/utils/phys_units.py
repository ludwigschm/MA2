"""Helpers for translating between physical units and pixels."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional

log = logging.getLogger(__name__)

_MM_PER_INCH = 25.4


def _clamp_index(index: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(index, total - 1))


def _iter_qt_app_getters() -> Iterator[Any]:  # type: ignore[misc]
    modules = (
        "PyQt6.QtGui",
        "PySide6.QtGui",
        "PyQt5.QtGui",
        "PySide2.QtGui",
    )
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=["QGuiApplication"])
            yield module.QGuiApplication  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - optional dependency
            continue


def _from_qt(screen_index: int) -> Optional[Dict[str, Any]]:
    for qapp_cls in _iter_qt_app_getters():
        if qapp_cls is None:
            continue
        owns_app = False
        app = qapp_cls.instance()
        if app is None:
            try:
                app = qapp_cls(sys.argv)
                owns_app = True
            except Exception:  # pragma: no cover - optional dependency
                continue
        try:
            screens = list(app.screens())
            if not screens:
                continue
            idx = _clamp_index(screen_index, len(screens))
            screen = screens[idx]
            geometry = screen.geometry()
            physical = screen.physicalSize()
            width_px = geometry.width()
            height_px = geometry.height()
            width_mm = getattr(physical, "width", lambda: 0)()
            height_mm = getattr(physical, "height", lambda: 0)()
            # PyQt5 returns ints, PyQt6 floats via methods
            if width_mm == 0 and hasattr(physical, "width") and not callable(physical.width):
                width_mm = getattr(physical, "width", 0)
            if height_mm == 0 and hasattr(physical, "height") and not callable(physical.height):
                height_mm = getattr(physical, "height", 0)
            width_mm = float(width_mm)
            height_mm = float(height_mm)
            if width_mm > 0 and height_mm > 0:
                px_per_mm = width_px / width_mm
                return {
                    "backend": "qt",
                    "idx": idx,
                    "width_px": int(width_px),
                    "height_px": int(height_px),
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "px_per_mm": float(px_per_mm),
                }
        finally:
            if owns_app:
                try:
                    app.quit()
                except Exception:
                    pass
    return None


def _from_winapi(screen_index: int) -> Optional[Dict[str, Any]]:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # pragma: no cover - optional dependency
        return None

    user32 = getattr(ctypes, "windll", None)
    if user32 is None:  # pragma: no cover - non-Windows
        return None
    user32 = user32.user32
    shcore = getattr(ctypes.windll, "shcore", None)
    if shcore is None:  # pragma: no cover - old Windows
        return None

    monitors: list[Dict[str, Any]] = []

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        ctypes.c_long,
    )

    def _cb(handle, _hdc, rect_ptr, _lparam):
        rect = rect_ptr.contents
        monitors.append(
            {
                "handle": handle,
                "width_px": rect.right - rect.left,
                "height_px": rect.bottom - rect.top,
            }
        )
        return 1

    if not user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_cb), 0):
        return None

    if not monitors:
        return None

    idx = _clamp_index(screen_index, len(monitors))
    monitor = monitors[idx]
    dpi_x = wintypes.UINT()
    dpi_y = wintypes.UINT()
    MDT_EFFECTIVE_DPI = 0
    if shcore.GetDpiForMonitor(monitor["handle"], MDT_EFFECTIVE_DPI, ctypes.byref(dpi_x), ctypes.byref(dpi_y)) != 0:
        return None
    dpi = float(dpi_x.value)
    px_per_mm = dpi / _MM_PER_INCH
    return {
        "backend": "winapi",
        "idx": idx,
        "width_px": int(monitor.get("width_px", 0)),
        "height_px": int(monitor.get("height_px", 0)),
        "px_per_mm": float(px_per_mm),
    }


def _from_pyglet(screen_index: int) -> Optional[Dict[str, Any]]:
    try:
        import pyglet  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        display = pyglet.canvas.get_display()
        screens = display.get_screens()
    except Exception:  # pragma: no cover - runtime failure
        return None

    if not screens:
        return None

    idx = _clamp_index(screen_index, len(screens))
    screen = screens[idx]
    width_px = getattr(screen, "width", 0)
    width_mm = getattr(screen, "width_mm", None)
    if width_mm and width_mm > 0:
        px_per_mm = float(width_px) / float(width_mm)
        return {
            "backend": "pyglet",
            "idx": idx,
            "width_px": int(width_px),
            "height_px": int(getattr(screen, "height", 0)),
            "width_mm": float(width_mm),
            "px_per_mm": float(px_per_mm),
        }
    return None


@lru_cache(maxsize=8)
def detect_px_per_mm(screen_index: int, fallback_dpi: Optional[float] = None) -> Dict[str, Any]:
    """Detect pixel density information for the target screen."""

    for getter in (_from_qt, _from_winapi, _from_pyglet):
        info = getter(screen_index)
        if info and info.get("px_per_mm"):
            return info

    dpi = None
    backend = "fallback"
    try:
        if fallback_dpi is not None:
            dpi = float(fallback_dpi)
        else:
            from kivy.core.window import Window  # type: ignore

            dpi = float(getattr(Window, "dpi", 96.0))
            backend = "kivy"
    except Exception:  # pragma: no cover - Kivy optional
        if fallback_dpi is not None:
            dpi = float(fallback_dpi)
    if dpi is None:
        dpi = 96.0
    px_per_mm = float(dpi) / _MM_PER_INCH
    return {
        "backend": backend,
        "idx": screen_index,
        "px_per_mm": float(px_per_mm),
    }


def mm_to_px(mm: float, screen_index: int, fallback_dpi: Optional[float] = None) -> int:
    """Convert a physical millimetre value to pixels for the given screen."""

    info = detect_px_per_mm(screen_index, fallback_dpi=fallback_dpi).copy()
    px_per_mm = info.get("px_per_mm", 0.0)
    pixels = int(round(mm * px_per_mm))
    backend = info.get("backend", "unknown")
    idx = info.get("idx", screen_index)

    if backend == "fallback":
        log.warning(
            "Using fallback pixel density %.3f px/mm for display %d. Assuming %.1f mm -> %d px.",
            px_per_mm,
            idx,
            mm,
            pixels,
        )
    else:
        log.info(
            "Pixel density backend=%s display=%d: %.3f px/mm -> %d px for %.1f mm.",
            backend,
            idx,
            px_per_mm,
            pixels,
            mm,
        )

    width_px = info.get("width_px")
    height_px = info.get("height_px")
    if width_px == 3840 and height_px == 2160:
        log.info(
            "43\"-4K reference: expected ≈242 px for %.1f mm, calculated %d px.",
            mm,
            pixels,
        )

    return pixels


__all__ = ["detect_px_per_mm", "mm_to_px"]
