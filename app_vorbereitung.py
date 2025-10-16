# app.py
# -----------------------------------------------------------------------------
# Merged script combining game_engine_w.py, aruco_overlay.py, and
# tabletop_ux_kivy_base_w.py so the entire application can be imported from a
# single module.
# -----------------------------------------------------------------------------
from __future__ import annotations

from tabletop.engine import (
    Phase,
    Player,
    VP,
    SignalLevel,
    Call,
    RoundPlan,
    RoleMap,
    VisibleCardState,
    RoundState,
    RoundSchedule,
    EventLogger,
    hand_value,
    hand_category,
    hand_category_label,
    FORCED_BLUFF_LABEL,
    SessionCsvLogger,
    GameEngineConfig,
    GameEngine,
)
from tabletop.logging.events import Events


# --- Datei-Ende: Demo/Startkonfiguration robust machen ---
from pathlib import Path
import os

def main():
    BASE = Path(__file__).resolve().parent
    CSV_PATH = BASE / "Paare1.csv"
    LOG_DIR = BASE / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    print("Arbeitsverzeichnis (cwd):", os.getcwd())
    print("Skriptordner (__file__):", BASE)
    print("Erwarte CSV unter:", CSV_PATH)
    print("Existiert CSV?", CSV_PATH.exists())

    cfg = GameEngineConfig(
        session_id="S001",
        csv_path=str(CSV_PATH),                          # <- absoluter Pfad
        db_path=str(LOG_DIR / "events.sqlite3"),
        csv_log_path=str(LOG_DIR / "events.csv"),
    )

    eng = GameEngine(cfg)
    try:
        print("Init:", eng.get_public_state())

        # Runde 1: beide starten
        eng.click_start(Player.P1); eng.click_start(Player.P2)

        # Aufdecken 1
        eng.click_reveal_card(Player.P1, 0)
        eng.click_reveal_card(Player.P2, 0)
        eng.click_reveal_card(Player.P1, 1)
        eng.click_reveal_card(Player.P2, 1)

        # Signal & Call (Beispiel)
        eng.p1_signal(SignalLevel.MITTEL)
        eng.p2_call(Call.WAHRHEIT, p1_hat_wahrheit_gesagt=True)
        print("Round done:", eng.get_public_state())

        # Nächste Runde (beide drücken)
        eng.click_next_round(Player.P1); eng.click_next_round(Player.P2)
        print("Nach Rollentausch:", eng.get_public_state())
    finally:
        eng.close()
# ==== End original {name} ====

# ==== Begin original aruco_overlay.py ====
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
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent
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
FIXED_SIZE_PX  = 280                              # z. B. 280 px inkl. Quiet-Zone
SIZE_PERCENT   = 0.16                              # falls USE_FIXED_SIZE=False
MIN_SIZE_PX    = 160
MAX_SIZE_PX    = 560

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
            if not self.use_fixed:
                self.size_percent *= 1.05
                self._layout_and_render_markers()
        elif event.key() == Qt.Key.Key_Minus:
            if not self.use_fixed:
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
    screens = app.screens()
    if not screens:
        geom = QRect(100, 100, 1280, 720)
        win = MarkerOverlay(geom, layout=layout)
        win.show()
        overlays.append(win)
    else:
        for s in screens:
            geom = s.geometry()
            # Alternativ kompatibel:
            # win = MarkerOverlay(geom, marker_ids=[1,7,23,37,55,71,89,101])
            win = MarkerOverlay(geom, layout=layout)
            win.showFullScreen()
            overlays.append(win)

    sys.exit(app.exec())
# ==== End original {name} ====

# ==== Begin original tabletop_ux_kivy_base_w.py ====
# tabletop_ux_kivy.py
# -------------------------------------------------------------
# Fertiges UX-Skript für deine Masterarbeit
# - 43" Tisch-Display, 3840x2160 (4K UHD)
# - Vollbild, Hintergrundfarbe #BFBFBF
# - Ordnerstruktur exakt wie in deinen Screenshots:
#     ./UX/  -> play_*.png, hoch_*.png, mittel_*.png, tief_*.png, bluff_*.png, wahr_*.png
#     ./Karten/ -> back.png, back_stop.png  (Kartenwerte optional)
# - Ablauf exakt nach Beschreibung inkl. Rollenwechsel pro Runde
# - Beide Karten pro Spieler werden im Ablauf aufgedeckt (erst innere, dann äußere)
# - Buttons sind nur „live“, wenn *_live.png verwendet wird; sonst *_stop.png
# - Keine Anpassungen nötig, einfach `python tabletop_ux_kivy.py` starten.
# -------------------------------------------------------------

from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
import os
import csv
import itertools
from pathlib import Path
from datetime import datetime
import numpy as np



# --- Phasenlogik
from tabletop.state.phases import UXPhase, to_engine_phase
from tabletop.logging.round_csv import (
    init_round_log,
    round_log_action_label,
    write_round_log,
    close_round_log,
)
from tabletop.ui import widgets as ui_widgets
from tabletop.ui.widgets import RotatableLabel, CardWidget, IconButton

# --- Display fest auf 3840x2160, Vollbild aktivierbar (kommentiere die nächste Zeile, falls du Fenster willst)
Config.set('graphics', 'fullscreen', 'auto')

# --- Konstanten & Assets
from tabletop.data.blocks import load_blocks, load_csv_rounds, value_to_card_path
from tabletop.data.config import ROOT
from tabletop.ui.assets import (
    ASSETS,
    FIX_LIVE_IMAGE,
    FIX_STOP_IMAGE,
    resolve_background_texture,
)
from tabletop.overlay.process import (
    start_overlay as start_overlay_process,
    stop_overlay as stop_overlay_process,
)
from tabletop.overlay.fixation import (
    generate_fixation_tone,
    play_fixation_tone as overlay_play_fixation_tone,
    run_fixation_sequence as overlay_run_fixation_sequence,
)
from tabletop.state.controller import TabletopController, TabletopState
from tabletop.tabletop_view import TabletopRoot

ui_widgets.ASSETS = ASSETS

STATE_FIELD_NAMES = set(TabletopState.__dataclass_fields__)



class TabletopApp(App):
    def build(self):
        self.title = 'Masterarbeit – Tabletop UX'
        root = TabletopRoot()
        return root

    def on_start(self):
        root = self.root
        if root:
            Clock.schedule_once(
                lambda *_: setattr(
                    root,
                    "overlay_process",
                    root.start_overlay(root.overlay_process),
                ),
                0,
            )

    def on_stop(self):
        root = self.root
        if root and root.logger:
            root.logger.close()
        if root:
            close_round_log(root)
            root.overlay_process = root.stop_overlay(root.overlay_process)

if __name__ == '__main__':
    TabletopApp().run()
 
# ==== End original {name} ====
