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

import sys, os, json
from typing import List, Dict, Tuple, Optional
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent, QGuiApplication
from PyQt6.QtCore import Qt, QRect
import cv2
import numpy as np

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
TARGET_CM      = 6.0
FIXED_SIZE_PX  = 240                              # wird dynamisch angepasst (~6 cm @ ~100 PPI)
SIZE_PERCENT   = 0.16                              # falls USE_FIXED_SIZE=False
MIN_SIZE_PX    = 160
MAX_SIZE_PX    = 560

_FIXED_SIZE_INITIALIZED = False


def _initialize_fixed_size() -> None:
    """Compute the fixed marker size in pixels based on primary screen metrics."""
    global FIXED_SIZE_PX, _FIXED_SIZE_INITIALIZED

    if _FIXED_SIZE_INITIALIZED:
        return

    screen = QGuiApplication.primaryScreen()
    fallback_reason = "no primary screen"
    ppi = None

    if screen is not None:
        p_width_px = screen.geometry().width()
        p_width_mm = screen.physicalSize().width()
        fallback_reason = "physical size unavailable"

        if p_width_mm > 0:
            ppi = p_width_px / (p_width_mm / 25.4)
            FIXED_SIZE_PX = int(round(ppi * (TARGET_CM / 2.54)))

    if ppi is not None:
        print(
            f"ArUco overlay: primary screen ≈ {ppi:.1f} PPI, target size {FIXED_SIZE_PX}px for {TARGET_CM:.1f} cm"
        )
    else:
        print(
            "ArUco overlay: PPI unavailable"
            f" ({fallback_reason}), using fallback size {FIXED_SIZE_PX}px (~{TARGET_CM:.1f} cm)"
        )

    _FIXED_SIZE_INITIALIZED = True

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

        _initialize_fixed_size()

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
        self.fixed_size = FIXED_SIZE_PX

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
        print("Feste Marker-Zuordnung (Position → ID):")
        for name in self.pos_order:
            print(f"  {name:12s} -> {self.layout[name]}")
        try:
            mapping_path = os.path.join(os.getcwd(), "marker_layout.json")
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump({name: int(self.layout[name]) for name in self.pos_order}, f, ensure_ascii=False, indent=2)
            print(f"(Gespeichert als {mapping_path})")
        except Exception as e:
            print(f"Warnung: Konnte marker_layout.json nicht schreiben: {e}")

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
    geometry = _preferred_screen_geometry(app)
    if geometry is None:
        geometry = QRect(100, 100, 1280, 720)
        print("Keine Bildschirminformation gefunden, verwende Fallback-Geometrie.")
        win = MarkerOverlay(geometry, layout=layout)
        win.show()
    else:
        print(
            "Starte Overlay auf Bildschirm bei"
            f" ({geometry.x()}, {geometry.y()}) mit Größe {geometry.width()}x{geometry.height()}"
        )
        win = MarkerOverlay(geometry, layout=layout)
        win.showFullScreen()
    overlays.append(win)

    sys.exit(app.exec())


def _preferred_screen_geometry(app: QApplication) -> Optional[QRect]:
    """Select the secondary screen when present, else fall back to primary."""

    screens = list(app.screens())
    if not screens:
        return None

    primary = app.primaryScreen()
    target = None
    if len(screens) > 1:
        for screen in screens:
            if primary is None or screen != primary:
                target = screen
                break

    if target is None:
        target = primary or screens[0]

    return target.geometry()


if __name__ == "__main__":
    main()
