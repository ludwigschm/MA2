"""Reusable Kivy widget classes for the tabletop application."""

import os

from kivy.graphics import PopMatrix, PushMatrix, Rotate
from kivy.uix.button import Button
from kivy.uix.label import Label


class RotatableLabel(Label):
    """Label, das rotiert werden kann (z.B. 180° für die obere Tisch-Seite)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rotation_angle = 0
        with self.canvas.before:
            self._push_matrix = PushMatrix()
            self._rotation = Rotate(angle=0, origin=self.center)
        with self.canvas.after:
            self._pop_matrix = PopMatrix()
        self.bind(pos=self._update_transform, size=self._update_transform)

    def set_rotation(self, angle: float):
        self.rotation_angle = angle
        self._update_transform()

    def _update_transform(self, *args):
        if hasattr(self, "_rotation"):
            self._rotation.origin = self.center
            self._rotation.angle = self.rotation_angle


class CardWidget(Button):
    """Karten-Slot: zeigt back_stop bis aktiv und/oder aufgedeckt."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.live = False
        self.face_up = False
        self.front_image = ASSETS['cards']['back']
        self.border = (0, 0, 0, 0)
        self.background_normal = ASSETS['cards']['back_stop']
        self.background_down = ASSETS['cards']['back_stop']
        self.background_disabled_normal = ASSETS['cards']['back_stop']
        self.background_disabled_down = ASSETS['cards']['back_stop']
        self.disabled_color = (1, 1, 1, 1)
        self.update_visual()

    def set_live(self, v: bool):
        self.live = v
        self.disabled = not v
        self.update_visual()

    def flip(self):
        if not self.live:
            return
        self.face_up = True
        self.set_live(False)

    def reset(self):
        self.live = False
        self.face_up = False
        self.disabled = True
        self.update_visual()

    def set_front(self, img_path: str):
        self.front_image = img_path
        if not os.path.exists(img_path):
            self.front_image = ASSETS['cards']['back']
        self.update_visual()

    def update_visual(self):
        if self.face_up:
            img = self.front_image
        elif self.live:
            img = ASSETS['cards']['back']
        else:
            img = ASSETS['cards']['back_stop']
        self.background_normal = img
        self.background_down = img
        self.background_disabled_normal = img
        self.background_disabled_down = img
        self.opacity = 1.0 if (self.live or self.face_up) else 0.55


class IconButton(Button):
    """Button, der automatisch live/stop-Grafiken nutzt."""

    def __init__(self, asset_pair: dict, label_text: str = '', **kw):
        super().__init__(**kw)
        self.asset_pair = asset_pair
        self.live = False
        self.selected = False
        self.border = (0, 0, 0, 0)
        self.background_normal = asset_pair['stop']
        self.background_down = asset_pair['stop']
        self.background_disabled_normal = asset_pair['stop']
        self.disabled_color = (1, 1, 1, 1)
        self.text = ''  # wir nutzen die Grafik
        self.rotation_angle = 0
        with self.canvas.before:
            self._push_matrix = PushMatrix()
            self._rotation = Rotate(angle=0, origin=self.center)
        with self.canvas.after:
            self._pop_matrix = PopMatrix()
        self.bind(pos=self._update_transform, size=self._update_transform)
        self.update_visual()

    def set_live(self, v: bool):
        self.live = v
        self.disabled = not v
        self.update_visual()

    def set_pressed_state(self):
        # nach Auswahl bleibt die live-Grafik sichtbar, ohne dass der Button live bleibt
        self.selected = True
        self.live = False
        self.disabled = True
        self.update_visual()

    def reset(self):
        self.selected = False
        self.live = False
        self.disabled = True
        self.update_visual()

    def set_rotation(self, angle: float):
        self.rotation_angle = angle
        self._update_transform()

    def _update_transform(self, *args):
        if hasattr(self, "_rotation"):
            self._rotation.origin = self.center
            self._rotation.angle = self.rotation_angle

    def update_visual(self):
        img = self.asset_pair['live'] if (self.live or self.selected) else self.asset_pair['stop']
        self.background_normal = img
        self.background_down = img
        self.background_disabled_normal = img
        self.background_disabled_down = img
        self.opacity = 1.0 if (self.live or self.selected) else 0.6
