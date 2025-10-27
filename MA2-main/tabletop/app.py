"""Kivy application bootstrap for the tabletop UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, cast

import os

import logging
from contextlib import suppress

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
from tabletop.pupil_bridge import PupilBridge

log = logging.getLogger(__name__)

_KV_LOADED = False


class TabletopApp(App):
    """Main Kivy application that wires the UI with infrastructure services."""

    def __init__(
        self,
        *,
        session: Optional[int] = None,
        block: Optional[int] = None,
        player: str = "VP1",
        bridge: Optional[PupilBridge] = None,
        **kwargs: Any,
    ) -> None:
        self._overlay_process: Optional[OverlayProcess] = None
        self._esc_handler: Optional[Any] = None
        self._key_up_handler: Optional[Any] = None
        self._bootstrap_screens: list[dict[str, int]] = self._probe_screens_pyqt()
        self._target_display_index: int = self._determine_display_index(
            screens=self._bootstrap_screens
        )

        self._configure_startup_display(self._target_display_index)
        self._bridge: Optional[PupilBridge] = bridge
        self._session: Optional[int] = session
        self._block: Optional[int] = block
        self._player: str = player
        self._recording_started: bool = False
        super().__init__(**kwargs)

    @staticmethod
    def _describe_window_screens() -> list[dict[str, int]]:
        """Return available screen geometries from the active Kivy window."""

        screens = getattr(Window, "screens", None)
        described: list[dict[str, int]] = []
        if not screens:
            return described

        for screen in screens:
            entry = {"left": 0, "top": 0, "width": 0, "height": 0}

            pos = getattr(screen, "pos", None)
            if pos is not None:
                with suppress(Exception):
                    entry["left"], entry["top"] = (int(pos[0]), int(pos[1]))
            else:
                entry["left"] = int(getattr(screen, "x", 0))
                entry["top"] = int(getattr(screen, "y", 0))

            size = getattr(screen, "size", None)
            if size is not None:
                with suppress(Exception):
                    entry["width"], entry["height"] = (
                        int(size[0]),
                        int(size[1]),
                    )
            else:
                entry["width"] = int(getattr(screen, "width", Window.width))
                entry["height"] = int(getattr(screen, "height", Window.height))

            described.append(entry)

        return described

    @staticmethod
    def _probe_screens_pyqt() -> list[dict[str, int]]:
        """Probe system displays via PyQt as a fallback during bootstrap."""

        try:
            from PyQt6.QtGui import QGuiApplication
        except Exception:  # pragma: no cover - optional dependency
            return []

        app = QGuiApplication.instance()
        owns_app = False
        if app is None:
            try:
                app = QGuiApplication([])
                owns_app = True
            except Exception:  # pragma: no cover - optional dependency
                return []

        screens: list[dict[str, int]] = []
        try:
            for screen in app.screens():
                try:
                    geometry = screen.geometry()
                except Exception:  # pragma: no cover - defensive fallback
                    continue
                screens.append(
                    {
                        "left": int(geometry.x()),
                        "top": int(geometry.y()),
                        "width": int(geometry.width()),
                        "height": int(geometry.height()),
                    }
                )
        finally:
            if owns_app:
                app.quit()

        return screens

    @staticmethod
    def _clamp_display_index(
        display_index: int, *, screens: Optional[Sequence[dict[str, int]]] = None
    ) -> int:
        """Clamp the desired display index to the available displays."""

        if display_index < 0:
            return 0

        if screens is None:
            screens = TabletopApp._describe_window_screens()
            if not screens:
                screens = None

        if screens:
            return min(display_index, len(screens) - 1)

        return display_index

    def _determine_display_index(
        self, *, screens: Optional[Sequence[dict[str, int]]] = None
    ) -> int:
        """Choose the preferred display for the experiment window."""

        env_value = os.environ.get("TABLETOP_DISPLAY_INDEX")
        desired_index: Optional[int] = None

        if env_value is not None:
            try:
                desired_index = int(env_value)
            except ValueError:
                log.warning(
                    "Ignoring invalid TABLETOP_DISPLAY_INDEX=%r", env_value
                )

        if desired_index is None:
            if screens is None:
                screens = TabletopApp._describe_window_screens()
                if not screens:
                    screens = self._bootstrap_screens
            count = len(screens) if screens is not None else 0
            desired_index = 1 if count >= 2 else 0

        return self._clamp_display_index(desired_index, screens=screens)

    def _apply_display_environment(self, display_index: int) -> None:
        """Persist the chosen display index for child processes."""

        os.environ["TABLETOP_DISPLAY_INDEX"] = str(display_index)
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(display_index)

    def _configure_startup_display(self, display_index: int) -> None:
        """Prepare environment and Kivy configuration for the selected monitor."""

        self._apply_display_environment(display_index)

        with suppress(Exception):
            Config.set("graphics", "display", str(display_index))

        target_screen: Optional[dict[str, int]] = None
        if 0 <= display_index < len(self._bootstrap_screens):
            target_screen = self._bootstrap_screens[display_index]

        if target_screen:
            with suppress(Exception):
                Config.set("graphics", "position", "custom")
                Config.set("graphics", "left", str(target_screen["left"]))
                Config.set("graphics", "top", str(target_screen["top"]))
                Config.set("graphics", "width", str(target_screen["width"]))
                Config.set("graphics", "height", str(target_screen["height"]))
            log.info(
                "Bootstrap configured for display %s at (%s, %s) size (%s x %s)",
                display_index,
                target_screen["left"],
                target_screen["top"],
                target_screen["width"],
                target_screen["height"],
            )

        with suppress(Exception):
            Config.write()

    def _move_window_to_display(self, display_index: int) -> int:
        """Attempt to position the window on the requested display."""

        screens = TabletopApp._describe_window_screens()
        if screens:
            self._bootstrap_screens = list(screens)
        else:
            screens = self._bootstrap_screens

        if not screens:
            return display_index

        clamped = self._clamp_display_index(display_index, screens=screens)
        try:
            target = screens[clamped]
        except Exception:  # pragma: no cover - defensive fallback
            log.exception("Failed to access display information for index %s", clamped)
            return clamped

        try:
            left = int(target.get("left", getattr(Window, "left", 0)))
            top = int(target.get("top", getattr(Window, "top", 0)))
            width = int(target.get("width", Window.width))
            height = int(target.get("height", Window.height))

            with suppress(Exception):
                Window.position = "custom"
            Window.left = left
            Window.top = top
            Window.size = (width, height)
            log.info(
                "Window moved to display %s at (%s, %s) size (%s x %s)",
                clamped,
                left,
                top,
                width,
                height,
            )
        except Exception:  # pragma: no cover - defensive fallback
            log.exception("Failed to reposition window for display %s", clamped)

        return clamped

    def build(self) -> TabletopRoot:
        """Create the root widget for the Kivy application."""
        global _KV_LOADED
        if not _KV_LOADED:
            kv_path = Path(__file__).parent / "ui" / "layout.kv"
            if kv_path.exists():
                Builder.load_file(str(kv_path))
            _KV_LOADED = True

        root = TabletopRoot(
            bridge=self._bridge,
            bridge_player=self._player,
            bridge_session=self._session,
            bridge_block=self._block,
        )

        # ESC binding is scheduled in ``on_start`` once the window exists.
        return root

    # ------------------------------------------------------------------
    # Bridge helpers
    def _bridge_payload_base(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self._session is not None:
            payload["session"] = self._session
        if self._block is not None:
            payload["block"] = self._block
        return payload

    def _format_key_name(self, key: int, codepoint: str) -> str:
        if codepoint:
            if codepoint == " ":
                return "space"
            return codepoint
        return f"code_{key}"

    def _emit_bridge_key_event(
        self,
        action: str,
        *,
        key: int,
        scancode: int,
        codepoint: str,
        modifiers: list[str],
    ) -> None:
        if not self._bridge or not self._player or not self._bridge.is_connected(self._player):
            return
        key_name = self._format_key_name(key, codepoint)
        payload = self._bridge_payload_base()
        payload.update(
            {
                "key": key_name,
                "keycode": key,
                "scancode": scancode,
                "codepoint": codepoint,
                "modifiers": modifiers,
            }
        )
        event_name = f"key.{key_name}.{action}"
        self._bridge.send_event(event_name, self._player, payload)

    def _start_bridge_recording(self) -> None:
        if (
            self._recording_started
            or self._bridge is None
            or self._player is None
            or not self._bridge.is_connected(self._player)
            or self._session is None
            or self._block is None
        ):
            return
        self._bridge.start_recording(self._session, self._block, self._player)
        self._recording_started = True

    def _stop_bridge_recording(self) -> None:
        if not self._bridge or not self._player:
            return
        try:
            self._bridge.stop_recording(self._player)
        finally:
            self._recording_started = False

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
            try:
                self._emit_bridge_key_event(
                    "down",
                    key=key,
                    scancode=scancode,
                    codepoint=codepoint,
                    modifiers=modifiers,
                )
            except Exception:  # pragma: no cover - defensive fallback
                log.exception("Failed to emit bridge key down event")
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

        if self._key_up_handler is not None:
            return

        def _on_key_up(
            _window: Window,
            key: int,
            scancode: int,
            *args: Any,
        ) -> bool:
            try:
                self._emit_bridge_key_event(
                    "up",
                    key=key,
                    scancode=scancode,
                    codepoint="",
                    modifiers=list(args[0]) if args and isinstance(args[0], (list, tuple)) else [],
                )
            except Exception:  # pragma: no cover - defensive fallback
                log.exception("Failed to emit bridge key up event")
            return False

        self._key_up_handler = _on_key_up
        Window.bind(on_key_up=self._key_up_handler)

    def on_start(self) -> None:  # pragma: no cover - framework callback
        super().on_start()
        root = cast(Optional[TabletopRoot], self.root)

        try:
            self._start_bridge_recording()
        except Exception:  # pragma: no cover - defensive fallback
            log.exception("Failed to start Pupil recording")
        if root is not None:
            try:
                root.update_bridge_context(
                    bridge=self._bridge,
                    player=self._player,
                    session=self._session,
                    block=self._block,
                )
            except AttributeError:
                pass

        self._target_display_index = self._determine_display_index()
        self._apply_display_environment(self._target_display_index)
        if root is not None:
            try:
                root.overlay_display_index = self._target_display_index
            except AttributeError:
                pass

        def _start_overlay_late(_dt: float) -> None:
            process_handle: Optional[OverlayProcess]
            if root and getattr(root, "overlay_process", None):
                process_handle = cast(Optional[OverlayProcess], root.overlay_process)
            else:
                process_handle = self._overlay_process

            try:
                process_handle = start_overlay(
                    process_handle,
                    overlay_path=ARUCO_OVERLAY_PATH,
                    display_index=self._target_display_index,
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
                self._target_display_index = self._move_window_to_display(
                    self._target_display_index
                )
                if root is not None:
                    try:
                        root.overlay_display_index = self._target_display_index
                    except AttributeError:
                        pass
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

        try:
            self._stop_bridge_recording()
        except Exception:  # pragma: no cover - defensive fallback
            log.exception("Failed to stop Pupil recording")

        super().on_stop()


def main(
    *,
    session: Optional[int] = None,
    block: Optional[int] = None,
    player: str = "VP1",
) -> None:
    """Run the tabletop Kivy application with optional Pupil bridge integration."""

    bridge = PupilBridge()
    try:
        bridge.connect()
    except Exception:  # pragma: no cover - defensive fallback
        log.exception("Failed to connect to Pupil devices")

    app = TabletopApp(
        session=session,
        block=block,
        player=player,
        bridge=bridge,
    )
    try:
        app.run()
    finally:
        try:
            bridge.stop_recording(player)
        except Exception:  # pragma: no cover - defensive fallback
            log.exception("Failed to stop recording during shutdown")
        try:
            bridge.close()
        except Exception:  # pragma: no cover - defensive fallback
            log.exception("Failed to close Pupil bridge")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
