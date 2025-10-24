"""Kivy application bootstrap for the tabletop UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import logging
import os
import sys

from kivy.app import App
from kivy.config import Config

Config.set("kivy", "exit_on_escape", "0")
Config.write()

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder

from tabletop.data.config import ARUCO_OVERLAY_PATH
from tabletop.logging.round_csv import close_round_log, flush_round_log
from tabletop.overlay.process import (
    OverlayProcess,
    start_overlay,
    stop_overlay,
)
from tabletop.tabletop_view import TabletopRoot

log = logging.getLogger(__name__)

_KV_LOADED = False


def _screens_via_qt() -> Optional[Tuple[str, List[Tuple[int, int, int, int]]]]:
    """Return screen geometries using a Qt backend when available."""

    gui_mod = None
    try:
        from PyQt5.QtGui import QGuiApplication as _QGuiApplication  # type: ignore

        gui_mod = _QGuiApplication
    except Exception:
        try:
            from PySide6.QtGui import QGuiApplication as _QGuiApplication  # type: ignore

            gui_mod = _QGuiApplication
        except Exception:
            try:
                from PyQt6.QtGui import QGuiApplication as _QGuiApplication  # type: ignore

                gui_mod = _QGuiApplication
            except Exception:
                return None

    if gui_mod is None:
        return None

    owns = gui_mod.instance() is None
    app = gui_mod.instance() or gui_mod([])
    try:
        geoms: List[Tuple[int, int, int, int]] = []
        for screen in gui_mod.screens():
            geom = screen.geometry()
            geoms.append((geom.x(), geom.y(), geom.width(), geom.height()))
        return ("qt", geoms)
    finally:
        if owns:
            app.quit()


def _screens_via_winapi() -> Optional[Tuple[str, List[Tuple[int, int, int, int]]]]:
    """Return screen geometries using the Windows API."""

    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    try:
        user32 = ctypes.windll.user32
    except Exception:
        return None

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(wintypes.RECT),
        ctypes.c_double,
    )

    rects: List[Tuple[int, int, int, int]] = []

    def _cb(_hmon, _hdc, rect_ptr, _lparam):
        rect = rect_ptr.contents
        rects.append((rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
        return 1

    if not user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_cb), 0):
        return None

    return ("winapi", rects)


def _screens_via_pyglet() -> Optional[Tuple[str, List[Tuple[int, int, int, int]]]]:
    """Return screen geometries using pyglet."""

    try:
        import pyglet

        display = pyglet.canvas.get_display()
        geoms: List[Tuple[int, int, int, int]] = []
        for screen in display.get_screens():
            geoms.append((screen.x, screen.y, screen.width, screen.height))
        return ("pyglet", geoms)
    except Exception:
        return None


def detect_screens() -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """Detect available screen geometries using supported backends."""

    for getter in (_screens_via_qt, _screens_via_winapi, _screens_via_pyglet):
        result = getter()
        if result and result[1]:
            return result

    try:
        width, height = Window.system_size
    except Exception:
        width, height = (1280, 720)

    return ("fallback", [(0, 0, width, height)])


def _parse_display_override() -> Optional[int]:
    """Parse overrides from environment variable or CLI arguments."""

    env_value = os.environ.get("TABLETOP_DISPLAY_INDEX")
    if env_value is not None:
        try:
            return int(env_value.strip())
        except (ValueError, AttributeError):
            log.warning(
                "Invalid TABLETOP_DISPLAY_INDEX value %r ignored.", env_value
            )

    for argument in sys.argv[1:]:
        if argument.startswith("--display="):
            _, value = argument.split("=", 1)
            try:
                return int(value.strip())
            except ValueError:
                log.warning("Invalid --display value %r ignored.", argument)
                break

    return None


def pick_preferred_geometry(
    default_index: int = 1,
) -> Tuple[int, int, int, int, str, int]:
    """Pick the geometry for the preferred display.

    Returns a tuple of (x, y, width, height, backend, chosen_index).
    """

    backend, screens = detect_screens()
    override = _parse_display_override()

    if override is not None:
        index = override
    elif len(screens) > default_index:
        index = default_index
    else:
        index = 0

    index = max(0, min(index, len(screens) - 1))
    x, y, width, height = screens[index]
    return x, y, width, height, backend, index


class TabletopApp(App):
    """Main Kivy application that wires the UI with infrastructure services."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._overlay_process: Optional[OverlayProcess] = None
        self._esc_handler: Optional[Any] = None
        self._preferred_geometry: Optional[Tuple[int, int, int, int]] = None
        self._preferred_backend: str = "fallback"
        self._preferred_index: int = 0

    def _ensure_preferred_geometry(self) -> None:
        if self._preferred_geometry is None:
            x, y, width, height, backend, index = pick_preferred_geometry()
            self._preferred_geometry = (x, y, width, height)
            self._preferred_backend = backend
            self._preferred_index = index
            log.info(
                "Using display %d via %s backend: x=%d y=%d w=%d h=%d",
                index,
                backend,
                x,
                y,
                width,
                height,
            )

    def build(self) -> TabletopRoot:
        """Create the root widget for the Kivy application."""
        global _KV_LOADED
        if not _KV_LOADED:
            kv_path = Path(__file__).parent / "ui" / "layout.kv"
            if kv_path.exists():
                Builder.load_file(str(kv_path))
            _KV_LOADED = True

        self._ensure_preferred_geometry()
        self._apply_preferred_geometry()
        try:
            Window.borderless = True
            Window.fullscreen = False
        except Exception as exc:  # pragma: no cover - defensive logging
            log.exception("Failed to configure initial window state: %s", exc)

        root = TabletopRoot()

        # ESC binding is scheduled in ``on_start`` once the window exists.
        return root

    def _bind_esc(self) -> None:
        """Ensure ESC toggles fullscreen without closing the app."""

        if self._esc_handler is not None:
            return

        def _on_key_down(
            _window: Window,
            key: int,
            scancode: int,
            codepoint: str,
            modifiers: list[str],
        ) -> bool:
            if key == 27:  # ESC
                try:
                    if Window.fullscreen:
                        Window.fullscreen = False
                        Window.borderless = False
                    else:
                        Window.fullscreen = "auto"
                        Window.borderless = True
                    log.info("ESC toggled fullscreen. Now fullscreen=%s", Window.fullscreen)
                except Exception as exc:  # pragma: no cover - safety net
                    log.exception("Error toggling fullscreen: %s", exc)
                return True
            return False

        self._esc_handler = _on_key_down
        Window.bind(on_key_down=self._esc_handler)

    def on_start(self) -> None:  # pragma: no cover - framework callback
        super().on_start()
        root = cast(Optional[TabletopRoot], self.root)

        self._ensure_preferred_geometry()

        def _start_overlay_late(_dt: float) -> None:
            process_handle: Optional[OverlayProcess]
            if root and getattr(root, "overlay_process", None):
                process_handle = cast(Optional[OverlayProcess], root.overlay_process)
            else:
                process_handle = self._overlay_process

            try:
                process_handle = start_overlay(
                    process_handle, overlay_path=ARUCO_OVERLAY_PATH
                )
            except Exception as exc:  # pragma: no cover - safety net
                log.exception("Overlay start failed: %s", exc)
                return

            self._overlay_process = process_handle
            if root is not None:
                root.overlay_process = process_handle
            log.info("Overlay started after window preparation.")

        def _prepare_window(_dt: float) -> None:
            try:
                self._apply_preferred_geometry()
                Window.borderless = True
                Window.fullscreen = False
            except Exception as exc:  # pragma: no cover - safety net
                log.exception("Failed to prepare window: %s", exc)

            self._bind_esc()
            Clock.schedule_once(_start_overlay_late, 0.25)

        self._apply_preferred_geometry()
        Clock.schedule_once(_prepare_window, 0.0)

    def on_stop(self) -> None:  # pragma: no cover - framework callback
        root = cast(Optional[TabletopRoot], self.root)

        process_handle: Optional[OverlayProcess]
        if root and getattr(root, "overlay_process", None):
            process_handle = cast(Optional[OverlayProcess], root.overlay_process)
        else:
            process_handle = self._overlay_process

        process_handle = stop_overlay(process_handle)
        self._overlay_process = process_handle
        if root is not None:
            root.overlay_process = process_handle

        if root is not None:
            logger = getattr(root, "logger", None)
            if logger is not None:
                close_fn = getattr(logger, "close", None)
                if callable(close_fn):
                    close_fn()
                root.logger = None
            flush_round_log(root)
            close_round_log(root)

        super().on_stop()


    def _apply_preferred_geometry(self) -> None:
        """Position the Kivy window on the preferred display when available."""

        self._ensure_preferred_geometry()
        geometry = self._preferred_geometry
        if geometry is None:
            return

        x, y, width, height = geometry
        try:
            Window.left = x
            Window.top = y
            Window.size = (width, height)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.exception("Failed to position window on preferred screen: %s", exc)

def main() -> None:
    """Run the tabletop Kivy application."""

    TabletopApp().run()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
