"""Kivy application bootstrap for the tabletop UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

from kivy.app import App
from kivy.config import Config

# Configure graphics before the window is created to ensure true fullscreen.
Config.set("graphics", "fullscreen", "auto")
Config.set("graphics", "borderless", "1")
Config.set("kivy", "exit_on_escape", "0")
Config.write()

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

_KV_LOADED = False


class TabletopApp(App):
    """Main Kivy application that wires the UI with infrastructure services."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._overlay_process: Optional[OverlayProcess] = None

    def build(self) -> TabletopRoot:
        """Create the root widget for the Kivy application."""
        global _KV_LOADED
        if not _KV_LOADED:
            kv_path = Path(__file__).parent / "ui" / "layout.kv"
            if kv_path.exists():
                Builder.load_file(str(kv_path))
            _KV_LOADED = True

        root = TabletopRoot()

        # Force fullscreen at runtime for providers that ignore config values.
        Window.fullscreen = True
        Window.borderless = True

        # Allow ESC to toggle out of fullscreen without closing the app.
        def _on_key_down(
            _window: Window, key: int, scancode: int, codepoint: str, modifiers: list[str]
        ) -> bool:
            if key == 27:  # ESC
                if Window.fullscreen:
                    Window.fullscreen = False
                    Window.borderless = False
                else:
                    Window.fullscreen = "auto"
                    Window.borderless = True
                return True
            return False

        Window.bind(on_key_down=_on_key_down)
        return root

    def on_start(self) -> None:  # pragma: no cover - framework callback
        super().on_start()
        root = cast(Optional[TabletopRoot], self.root)

        process_handle: Optional[OverlayProcess]
        if root and getattr(root, "overlay_process", None):
            process_handle = cast(Optional[OverlayProcess], root.overlay_process)
        else:
            process_handle = self._overlay_process

        process_handle = start_overlay(
            process_handle, overlay_path=ARUCO_OVERLAY_PATH
        )
        self._overlay_process = process_handle
        if root is not None:
            root.overlay_process = process_handle

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
