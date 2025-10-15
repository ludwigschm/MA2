"""Utilities for managing the fixation overlay sequence and tone playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import sounddevice as sd
import threading


def generate_fixation_tone(
    sample_rate: int = 44100,
    duration: float = 0.2,
    frequency: float = 1000.0,
    amplitude: float = 0.9,
):
    """Create the sine-wave tone that is played during the fixation sequence."""

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def play_fixation_tone(controller: Any) -> None:
    """Play the fixation tone asynchronously using sounddevice."""

    tone = getattr(controller, "fixation_tone", None)
    if tone is None:
        return

    sample_rate = getattr(controller, "fixation_tone_fs", 44100)
    tone_data = tone.copy()

    def _play():
        try:
            sd.play(tone_data, sample_rate)
            sd.wait()
        except Exception as exc:  # pragma: no cover - audio hardware dependent
            print(f"Warnung: Ton konnte nicht abgespielt werden: {exc}")

    threading.Thread(target=_play, daemon=True).start()


def run_fixation_sequence(
    controller: Any,
    *,
    schedule_once: Callable[[Callable[[float], None], float], Any],
    stop_image: Optional[Path | str],
    live_image: Optional[Path | str],
    on_complete: Optional[Callable[[], None]] = None,
) -> None:
    """Execute the fixation sequence using the provided controller state."""

    if getattr(controller, "fixation_running", False):
        return

    overlay = getattr(controller, "fixation_overlay", None)
    image = getattr(controller, "fixation_image", None)
    if overlay is None or image is None:
        if hasattr(controller, "fixation_required"):
            controller.fixation_required = False
        if on_complete:
            on_complete()
        return

    controller.fixation_running = True
    controller.pending_fixation_callback = on_complete
    overlay.opacity = 1
    overlay.disabled = False

    if getattr(overlay, "parent", None) is not None:
        controller.remove_widget(overlay)
    controller.add_widget(overlay)

    for attr in ("btn_start_p1", "btn_start_p2"):
        btn = getattr(controller, attr, None)
        if btn is not None and hasattr(btn, "set_live"):
            btn.set_live(False)

    image.opacity = 1
    image.source = _path_to_source(stop_image)

    def finish(_dt: float) -> None:
        if getattr(overlay, "parent", None) is not None:
            controller.remove_widget(overlay)
        overlay.opacity = 0
        overlay.disabled = True
        controller.fixation_running = False
        if hasattr(controller, "fixation_required"):
            controller.fixation_required = False
        callback = getattr(controller, "pending_fixation_callback", None)
        controller.pending_fixation_callback = None
        if callback:
            callback()

    def show_stop_again(_dt: float) -> None:
        image.source = _path_to_source(stop_image)
        schedule_once(finish, 5)

    def show_live(_dt: float) -> None:
        image.source = _path_to_source(live_image)
        play_fixation_tone(controller)
        schedule_once(show_stop_again, 0.2)

    schedule_once(show_live, 5)


def _path_to_source(image_path: Optional[Path | str]) -> str:
    if image_path is None:
        return ""
    if isinstance(image_path, Path):
        return str(image_path) if image_path.exists() else ""
    return str(image_path)


__all__ = [
    "generate_fixation_tone",
    "play_fixation_tone",
    "run_fixation_sequence",
]
