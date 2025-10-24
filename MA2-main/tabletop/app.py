"""Kivy application bootstrap for the tabletop UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

import logging

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


class TabletopApp(App):
    """Main Kivy application that wires the UI with infrastructure services."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._overlay_process: Optional[OverlayProcess] = None
        self._esc_handler: Optional[Any] = None

    def build(self) -> TabletopRoot:
        """Create the root widget for the Kivy application."""
        global _KV_LOADED
        if not _KV_LOADED:
            kv_path = Path(__file__).parent / "ui" / "layout.kv"
            if kv_path.exists():
                Builder.load_file(str(kv_path))
            _KV_LOADED = True

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
            log.info("Overlay started after fullscreen.")

        def _enter_fullscreen(_dt: float) -> None:
            try:
                Window.borderless = True
                Window.fullscreen = "auto"
                log.info("Fullscreen engaged (auto).")
            except Exception as exc:  # pragma: no cover - safety net
                log.exception("Failed to enter fullscreen: %s", exc)

            self._bind_esc()
            Clock.schedule_once(_start_overlay_late, 0.25)

        Clock.schedule_once(_enter_fullscreen, 0.0)

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


def main() -> None:
    """Run the tabletop Kivy application."""

    TabletopApp().run()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
