"""Screen detection helpers shared between the game UI and the overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from PyQt6.QtGui import QGuiApplication


@dataclass(frozen=True)
class ScreenInfo:
    """Lightweight description of a display output."""

    index: int
    x: int
    y: int
    width: int
    height: int
    physical_width_mm: Optional[int]
    physical_height_mm: Optional[int]

    @property
    def horizontal_ppi(self) -> Optional[float]:
        """Return the horizontal pixel density in PPI when available."""

        if self.physical_width_mm and self.physical_width_mm > 0:
            return self.width / (self.physical_width_mm / 25.4)
        return None


def capture_screen_info(app: Optional["QGuiApplication"] = None) -> List[ScreenInfo]:
    """Collect geometry information for all connected displays.

    The function attempts to obtain screen metrics via :mod:`PyQt6`.  When the
    toolkit is unavailable the function simply returns an empty list so callers
    can gracefully fall back to default behaviour.
    """

    try:
        from PyQt6.QtGui import QGuiApplication
    except Exception:  # pragma: no cover - optional dependency
        return []

    owns_app = False

    if app is None:
        app = QGuiApplication.instance()
        if app is None:
            app = QGuiApplication([])
            owns_app = True

    infos: List[ScreenInfo] = []
    for idx, screen in enumerate(app.screens()):
        geometry = screen.geometry()
        physical = screen.physicalSize()
        infos.append(
            ScreenInfo(
                index=idx,
                x=geometry.x(),
                y=geometry.y(),
                width=geometry.width(),
                height=geometry.height(),
                physical_width_mm=physical.width(),
                physical_height_mm=physical.height(),
            )
        )

    if owns_app:
        app.quit()

    return infos


def preferred_screen_info(
    infos: Sequence[ScreenInfo], preferred_index: int = 1
) -> Optional[ScreenInfo]:
    """Return the preferred screen, falling back to the first when necessary."""

    if len(infos) > preferred_index:
        return infos[preferred_index]
    if infos:
        return infos[0]
    return None


__all__ = ["ScreenInfo", "capture_screen_info", "preferred_screen_info"]

