# requirements:
#   pip install pyqt6 opencv-contrib-python
#
# Nutzung (kompatibel zu deinem Game):
#   # Alte Art (unbedingt 8 IDs übergeben, sonst werden nur die ersten Positionen belegt):
#   MarkerOverlay(geo, marker_ids=[1,7,23,37,55,71,89,101])
#   # Empfohlen (feste Zuordnung):
#   MarkerOverlay(geo, layout=MARKER_LAYOUT)
#
# Tasten im Overlay:
#   M   -> Marker ein/ausblenden
#   +   -> Marker um +5% größer (nur wenn USE_FIXED_SIZE=False)
#   -   -> Marker um -5% kleiner (nur wenn USE_FIXED_SIZE=False)
#   Esc -> Programm beenden

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow

from .utils.phys_units import mm_to_px

# -------------------- EMPFOHLENE IDs & Positionen ----------------------------
# Robuste, weit auseinanderliegende AprilTag-IDs (tag36h11)
MARKER_LAYOUT: Dict[str, int] = {
    "top_left":     1,
    "top_right":    7,
    "bottom_left":  23,
    "bottom_right": 37,
    "top_mid":      55,
    "bottom_mid":   71,
    "left_mid":     89,
    "right_mid":    101,
}
# Reihenfolge der Platzierung (und Mapping-Reihenfolge für marker_ids)
POSITION_ORDER: List[str] = [
    "top_left", "top_right", "bottom_left", "bottom_right",
    "top_mid", "bottom_mid", "left_mid", "right_mid",
]

# -------------------- RENDER-PARAMETER ---------------------------------------
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
QUIET_ZONE_RATIO = 0.08                          # Weißer Rand (schmaler gemacht)
BG_WHITE_CSS = "background: white;"
# WAR: LABEL_CSS = "background: white; color: black; font: 12pt 'Segoe UI';"
# NEU: transparent, damit kein weißes Feld unterhalb sichtbar ist
LABEL_CSS   = "background: transparent; color: black; font: 12pt 'Segoe UI';"

# Markergröße: entweder FIX (deterministisch) ODER prozentual
USE_FIXED_SIZE = True
MARKER_MM      = 60.0
SIZE_PERCENT   = 0.16                              # falls USE_FIXED_SIZE=False
MIN_SIZE_PX    = 160
MAX_SIZE_PX    = 560

log = logging.getLogger(__name__)


def _parse_display_override() -> Optional[int]:
    value = os.environ.get("TABLETOP_DISPLAY_INDEX")
    if value is not None:
        try:
            return int(value.strip())
        except (TypeError, ValueError):
            log.warning("Invalid TABLETOP_DISPLAY_INDEX value %r ignored.", value)

    for argument in sys.argv[1:]:
        if argument.startswith("--display="):
            _, raw = argument.split("=", 1)
            try:
                return int(raw.strip())
            except ValueError:
                log.warning("Invalid --display argument %r ignored.", argument)
                break
    return None


def _effective_display_index(total_screens: int) -> int:
    override = _parse_display_override()
    if override is not None:
        return max(0, min(override, max(0, total_screens - 1)))

    inherited = os.environ.get("TABLETOP_EFFECTIVE_DISPLAY_INDEX")
    if inherited is not None:
        try:
            idx = int(inherited.strip())
            return max(0, min(idx, max(0, total_screens - 1)))
        except ValueError:
            log.warning("Invalid TABLETOP_EFFECTIVE_DISPLAY_INDEX value %r ignored.", inherited)

    if total_screens > 1:
        return 1
    return 0

# -------------------- TAG-RENDERING ------------------------------------------
def generate_apriltag_qpixmap(tag_id: int, size: int, quiet_zone_ratio: float = QUIET_ZONE_RATIO) -> QPixmap:
    """Render AprilTag in weißem Quadrat (size x size) mit Quiet-Zone."""
    size = int(size)
    q = max(0.05, min(quiet_zone_ratio, 0.40))      # clamp 5..40%
    inner = int(round(size * (1.0 - 2.0 * q)))
    inner = max(32, inner)

    canvas = np.full((size, size), 255, dtype=np.uint8)  # weiß
    aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    tag_img = np.zeros((inner, inner), dtype=np.uint8)   # schwarz
    cv2.aruco.generateImageMarker(aruco_dict, tag_id, inner, tag_img, 1)

    y0 = (size - inner) // 2
    x0 = (size - inner) // 2
    canvas[y0:y0 + inner, x0:x0 + inner] = tag_img

    qimg = QImage(canvas.data, size, size, size, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg)

# -------------------- OVERLAY-FENSTER ----------------------------------------
class MarkerOverlay(QMainWindow):
    def __init__(
        self,
        screen_geometry: QRect,
        layout: Optional[Dict[str, int]] = None,
        marker_ids: Optional[List[int]] = None,   # Abwärtskompatibel
        screen_index: int = 0,
    ):
        """
        Entweder 'layout' übergeben (empfohlen) ODER 'marker_ids' (werden in POSITION_ORDER gemappt).
        """
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(BG_WHITE_CSS)
        self.setGeometry(screen_geometry)

        self.screen_index = screen_index

        # --- Eingabe normalisieren ---
        if layout is not None:
            self.layout: Dict[str, int] = {name: layout[name] for name in POSITION_ORDER if name in layout}
        elif marker_ids is not None:
            n = min(len(marker_ids), len(POSITION_ORDER))
            self.layout = {POSITION_ORDER[i]: int(marker_ids[i]) for i in range(n)}
        else:
            # Default: alle 8 empfohlenen Marker
            self.layout = {name: MARKER_LAYOUT[name] for name in POSITION_ORDER}

        self.pos_order: List[str] = [name for name in POSITION_ORDER if name in self.layout]

        self.marker_labels: List[QLabel] = []
        self.text_labels: List[QLabel] = []
        self.markers_visible = True

        # Größen-Parameter
        self.size_percent = SIZE_PERCENT
        self.min_size = MIN_SIZE_PX
        self.max_size = MAX_SIZE_PX
        self.use_fixed = USE_FIXED_SIZE
        if self.use_fixed:
            self.fixed_size = mm_to_px(MARKER_MM, self.screen_index)
        else:
            self.fixed_size = 0

        # UI-Objekte
        for _ in self.pos_order:
            lab = QLabel(self)
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lab.setStyleSheet(BG_WHITE_CSS)    # weißes Label = sichere Quiet-Zone
            lab.setScaledContents(False)
            self.marker_labels.append(lab)

            txt = QLabel(self)
            txt.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            txt.setStyleSheet(LABEL_CSS)       # transparent -> keine weiße Leiste
            txt.hide()                         # direkt verstecken
            self.text_labels.append(txt)

        # Zuordnung ausgeben & speichern
        log.info("Feste Marker-Zuordnung (Position → ID):")
        for name in self.pos_order:
            log.info("  %-12s -> %s", name, self.layout[name])
        try:
            mapping_path = os.path.join(os.getcwd(), "marker_layout.json")
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump({name: int(self.layout[name]) for name in self.pos_order}, f, ensure_ascii=False, indent=2)
            log.info("Marker-Zuordnung gespeichert als %s", mapping_path)
        except Exception as e:
            log.warning("Konnte marker_layout.json nicht schreiben: %s", e)

        self._layout_and_render_markers()

    @staticmethod
    def _positions_full(w: int, h: int, msize: int, margin: int) -> Dict[str, Tuple[int, int]]:
        # Ecken + Kantenmitten
        return {
            "top_left":     (margin, margin),
            "top_right":    (w - margin - msize, margin),
            "bottom_left":  (margin, h - margin - msize),
            "bottom_right": (w - margin - msize, h - margin - msize),
            "top_mid":      (w // 2 - msize // 2, margin),
            "bottom_mid":   (w // 2 - msize // 2, h - margin - msize),
            "left_mid":     (margin, h // 2 - msize // 2),
            "right_mid":    (w - margin - msize, h // 2 - msize // 2),
        }

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_and_render_markers()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_M:
            self.toggle_markers()
        elif event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            if self.use_fixed:
                event.accept()
                return
            self.size_percent *= 1.05
            self._layout_and_render_markers()
        elif event.key() == Qt.Key.Key_Minus:
            if self.use_fixed:
                event.accept()
                return
            self.size_percent /= 1.05
            self._layout_and_render_markers()
        elif event.key() == Qt.Key.Key_Escape:
            QApplication.instance().quit()

    def toggle_markers(self):
        self.markers_visible = not self.markers_visible
        self._layout_and_render_markers()

    def _layout_and_render_markers(self):
        w = max(1, self.width())
        h = max(1, self.height())

        # Markergröße
        if self.use_fixed:
            msize = int(self.fixed_size)
            if msize <= 0:
                self.fixed_size = mm_to_px(MARKER_MM, self.screen_index)
                msize = int(self.fixed_size)
        else:
            base = int(min(w, h) * self.size_percent)
            msize = max(self.min_size, min(base, self.max_size))

        margin = max(6, int(msize * 0.08))  # Abstand zum Rand
        pos_map = self._positions_full(w, h, msize, margin)

        # Alle Labels erst verstecken, dann neu zeichnen
        for lab, txt in zip(self.marker_labels, self.text_labels):
            lab.setVisible(False)
            txt.setVisible(False)

        for (name, tag_id), lab in zip(
            [(n, self.layout[n]) for n in self.pos_order],
            self.marker_labels,
        ):
            x, y = pos_map[name]
            lab.resize(msize, msize)
            lab.move(x, y)
            lab.setPixmap(generate_apriltag_qpixmap(tag_id, msize, QUIET_ZONE_RATIO))
            lab.setVisible(self.markers_visible)

        # WICHTIG: keine Textlabels setzen/anzeigen -> keine weiße Fläche unterhalb

# -------------------- STANDALONE-TEST ----------------------------------------
def main():
    app = QApplication(sys.argv)

    # Standard: 8 Marker aus MARKER_LAYOUT
    layout = MARKER_LAYOUT

    overlays: List[MarkerOverlay] = []
    geometry_info = _preferred_screen_geometry(app)
    if geometry_info is None:
        geometry = QRect(100, 100, 1280, 720)
        log.warning("Keine Bildschirminformation gefunden, verwende Fallback-Geometrie.")
        win = MarkerOverlay(geometry, layout=layout, screen_index=0)
        win.show()
    else:
        geometry, screen_index = geometry_info
        log.info(
            "Starte Overlay auf Bildschirm %d bei (%d, %d) mit Größe %dx%d",
            screen_index,
            geometry.x(),
            geometry.y(),
            geometry.width(),
            geometry.height(),
        )
        win = MarkerOverlay(geometry, layout=layout, screen_index=screen_index)
        win.showFullScreen()
    overlays.append(win)

    sys.exit(app.exec())


def _preferred_screen_geometry(app: QApplication) -> Optional[Tuple[QRect, int]]:
    """Select the screen index based on overrides or defaults."""

    screens = list(app.screens())
    if not screens:
        return None

    index = _effective_display_index(len(screens))
    index = max(0, min(index, len(screens) - 1))
    return screens[index].geometry(), index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
