"""Kivy application bootstrap for the tabletop UI."""

from __future__ import annotations

import pathlib
from typing import Any, Optional, cast

from kivy.app import App
from kivy.config import Config
from kivy.lang import Builder

from tabletop.logging.round_csv import close_round_log
from tabletop.overlay.process import (
    OverlayProcess,
    start_overlay,
    stop_overlay,
)
from tabletop.tabletop_view import TabletopRoot

# Ensure the application starts in fullscreen mode like the legacy scripts.
Config.set("graphics", "fullscreen", "auto")


class TabletopApp(App):
    """Main Kivy application that wires the UI with infrastructure services."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._overlay_process: Optional[OverlayProcess] = None

    def build(self) -> TabletopRoot:
        """Create the root widget for the Kivy application."""
        try:
            Builder.load_file(
                str(pathlib.Path(__file__).parent / "ui" / "layout.kv")
            )
        except Exception:
            pass

        return TabletopRoot()

    def on_start(self) -> None:  # pragma: no cover - framework callback
        super().on_start()
        root = cast(Optional[TabletopRoot], self.root)

        process_handle: Optional[OverlayProcess]
        if root and getattr(root, "overlay_process", None):
            process_handle = cast(Optional[OverlayProcess], root.overlay_process)
        else:
            process_handle = self._overlay_process

        process_handle = start_overlay(process_handle)
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
            close_round_log(root)

        super().on_stop()


def main() -> None:
    """Run the tabletop Kivy application."""

    TabletopApp().run()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
