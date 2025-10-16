"""Kivy application bootstrap for the tabletop UI."""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

from kivy.app import App
from kivy.config import Config

from tabletop.logging.round_csv import close_round_log
from tabletop.overlay import process as overlay_process
from tabletop.tabletop_view import TabletopRoot

# Ensure the application starts in fullscreen mode like the legacy scripts.
Config.set("graphics", "fullscreen", "auto")

OverlayProcess = overlay_process.OverlayProcess
StartOverlayFn = Callable[[Optional[OverlayProcess]], Optional[OverlayProcess]]
StopOverlayFn = Callable[[Optional[OverlayProcess]], Optional[OverlayProcess]]


class TabletopApp(App):
    """Main Kivy application that wires the UI with infrastructure services."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._overlay_process: Optional[OverlayProcess] = None

    def build(self) -> TabletopRoot:
        """Create the root widget for the Kivy application."""

        root = TabletopRoot()
        return root

    def on_start(self) -> None:  # pragma: no cover - framework callback
        super().on_start()
        root = self.root
        if root and hasattr(root, "start_overlay"):
            start_overlay = cast(StartOverlayFn, getattr(root, "start_overlay"))
        else:
            start_overlay = overlay_process.start_overlay

        process_handle: Optional[OverlayProcess]
        if root and hasattr(root, "overlay_process"):
            process_handle = getattr(root, "overlay_process")
        else:
            process_handle = self._overlay_process

        process_handle = start_overlay(process_handle)
        self._overlay_process = process_handle
        if root and hasattr(root, "overlay_process"):
            setattr(root, "overlay_process", process_handle)

    def on_stop(self) -> None:  # pragma: no cover - framework callback
        root = self.root

        if root and hasattr(root, "stop_overlay"):
            stop_overlay = cast(StopOverlayFn, getattr(root, "stop_overlay"))
        else:
            stop_overlay = overlay_process.stop_overlay

        process_handle: Optional[OverlayProcess]
        if root and hasattr(root, "overlay_process"):
            process_handle = getattr(root, "overlay_process")
        else:
            process_handle = self._overlay_process

        process_handle = stop_overlay(process_handle)
        self._overlay_process = process_handle
        if root and hasattr(root, "overlay_process"):
            setattr(root, "overlay_process", process_handle)

        if root:
            logger = getattr(root, "logger", None)
            if logger is not None:
                close_fn = getattr(logger, "close", None)
                if callable(close_fn):
                    close_fn()
                setattr(root, "logger", None)
            close_round_log(root)

        super().on_stop()


def main() -> None:
    """Run the tabletop Kivy application."""

    TabletopApp().run()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
