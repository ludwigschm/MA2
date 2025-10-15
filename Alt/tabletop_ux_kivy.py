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
import os

# --- Display fest auf 3840x2160, Vollbild aktivierbar (kommentiere die nächste Zeile, falls du Fenster willst)
Config.set('graphics', 'fullscreen', 'auto')
Window.size = (3840, 2160)

# --- Konstanten & Assets
ROOT = os.path.dirname(os.path.abspath(__file__))
UX_DIR = os.path.join(ROOT, 'UX')
CARD_DIR = os.path.join(ROOT, 'Karten')

ASSETS = {
    'play': {
        'live':  os.path.join(UX_DIR, 'play_live.png'),
        'stop':  os.path.join(UX_DIR, 'play_stop.png'),
    },
    'signal': {
        'low':   {'live': os.path.join(UX_DIR, 'tief_live.png'),   'stop': os.path.join(UX_DIR, 'tief_stop.png')},
        'mid':   {'live': os.path.join(UX_DIR, 'mittel_live.png'), 'stop': os.path.join(UX_DIR, 'mittel_stop.png')},
        'high':  {'live': os.path.join(UX_DIR, 'hoch_live.png'),   'stop': os.path.join(UX_DIR, 'hoch_stop.png')},
    },
    'decide': {
        'bluff': {'live': os.path.join(UX_DIR, 'bluff_live.png'),  'stop': os.path.join(UX_DIR, 'bluff_stop.png')},
        'wahr':  {'live': os.path.join(UX_DIR, 'wahr_live.png'),   'stop': os.path.join(UX_DIR, 'wahr_stop.png')},
    },
    'cards': {
        'back':      os.path.join(CARD_DIR, 'back.png'),
        'back_stop': os.path.join(CARD_DIR, 'back_stop.png'),
    }
}

# --- Phasen der Runde
PH_WAIT_BOTH_START = 'WAIT_BOTH_START'
PH_P1_INNER = 'P1_INNER'
PH_P2_INNER = 'P2_INNER'
PH_P1_OUTER = 'P1_OUTER'
PH_P2_OUTER = 'P2_OUTER'
PH_P1_SIGNAL = 'P1_SIGNAL'
PH_P2_DECIDE = 'P2_DECIDE'
PH_SHOWDOWN = 'SHOWDOWN'
PH_INTER_ROUND = 'INTER_ROUND'

class CardWidget(Button):
    """Karten-Slot: zeigt back_stop bis aktiv und/oder aufgedeckt."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.live = False
        self.face_up = False
        self.border = (0, 0, 0, 0)
        self.background_normal = ASSETS['cards']['back_stop']
        self.background_down = ASSETS['cards']['back_stop']
        self.background_disabled_normal = ASSETS['cards']['back_stop']
        self.disabled_color = (1,1,1,1)
        self.update_visual()

    def set_live(self, v: bool):
        self.live = v
        self.disabled = not v
        self.update_visual()

    def flip(self):
        if not self.live:
            return
        self.face_up = True
        self.update_visual()

    def reset(self):
        self.live = False
        self.face_up = False
        self.disabled = True
        self.update_visual()

    def update_visual(self):
        if self.face_up:
            self.background_normal = ASSETS['cards']['back']
            self.background_down = ASSETS['cards']['back']
        else:
            self.background_normal = ASSETS['cards']['back_stop']
            self.background_down = ASSETS['cards']['back_stop']
        self.opacity = 1.0 if self.live else 0.55

class IconButton(Button):
    """Button, der automatisch live/stop-Grafiken nutzt."""
    def __init__(self, asset_pair: dict, label_text: str = '', **kw):
        super().__init__(**kw)
        self.asset_pair = asset_pair
        self.live = False
        self.border = (0, 0, 0, 0)
        self.background_normal = asset_pair['stop']
        self.background_down = asset_pair['stop']
        self.background_disabled_normal = asset_pair['stop']
        self.disabled_color = (1,1,1,1)
        self.text = ''  # wir nutzen die Grafik
        self.update_visual()

    def set_live(self, v: bool):
        self.live = v
        self.disabled = not v
        self.update_visual()

    def set_pressed_state(self):
        # nach Auswahl bleibt die live-Grafik sichtbar
        self.set_live(True)

    def update_visual(self):
        img = self.asset_pair['live'] if self.live else self.asset_pair['stop']
        self.background_normal = img
        self.background_down = img
        self.opacity = 1.0 if self.live else 0.6

class TabletopRoot(FloatLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        with self.canvas.before:
            Color(0.75, 0.75, 0.75, 1)  # #BFBFBF
            self.bg = Rectangle(pos=(0,0), size=Window.size)
        Window.bind(on_resize=self.on_resize)

        self.round = 1
        self.p1_role = 'signaler'  # P1 startet als Signaler
        self.phase = PH_WAIT_BOTH_START

        # --- UI Elemente platzieren
        self.make_ui()
        self.apply_phase()

    # --- Layout & Elemente
    def on_resize(self, *_):
        self.bg.size = Window.size

    def make_ui(self):
        W, H = Window.size

        # Start-Buttons links/rechts (für beide Spieler)
        self.btn_start_p1 = IconButton(ASSETS['play'], size_hint=(None,None), size=(300,300), pos=(60, H/2-150))
        self.btn_start_p1.bind(on_release=lambda *_: self.start_pressed(1))
        self.add_widget(self.btn_start_p1)

        self.btn_start_p2 = IconButton(ASSETS['play'], size_hint=(None,None), size=(300,300), pos=(W-360, H/2-150))
        self.btn_start_p2.bind(on_release=lambda *_: self.start_pressed(2))
        self.add_widget(self.btn_start_p2)

        # Spielerzonen (je 2 Karten, links ausgerichtet)
        # Spieler 1 unten
        self.p1_outer = CardWidget(size_hint=(None,None), size=(320,480), pos=(220, 140))
        self.p1_outer.bind(on_release=lambda *_: self.tap_card(1, 'outer'))
        self.add_widget(self.p1_outer)

        self.p1_inner = CardWidget(size_hint=(None,None), size=(320,480), pos=(560, 140))
        self.p1_inner.bind(on_release=lambda *_: self.tap_card(1, 'inner'))
        self.add_widget(self.p1_inner)

        # Spieler 2 oben (gespiegelt in Position, aber gleiche Grafiken)
        self.p2_outer = CardWidget(size_hint=(None,None), size=(320,480), pos=(Window.size[0]-220-320-340, H-140-480))
        self.p2_outer.bind(on_release=lambda *_: self.tap_card(2, 'outer'))
        self.add_widget(self.p2_outer)

        self.p2_inner = CardWidget(size_hint=(None,None), size=(320,480), pos=(Window.size[0]-220-320, H-140-480))
        self.p2_inner.bind(on_release=lambda *_: self.tap_card(2, 'inner'))
        self.add_widget(self.p2_inner)

        # Signale P1 (unten Mitte)
        self.sig_low  = IconButton(ASSETS['signal']['low'], size_hint=(None,None), size=(260,260), pos=(W/2-420, 60))
        self.sig_mid  = IconButton(ASSETS['signal']['mid'], size_hint=(None,None), size=(260,260), pos=(W/2-130, 60))
        self.sig_high = IconButton(ASSETS['signal']['high'], size_hint=(None,None), size=(260,260), pos=(W/2+160, 60))
        self.sig_low.bind(on_release=lambda *_: self.pick_signal('low'))
        self.sig_mid.bind(on_release=lambda *_: self.pick_signal('mid'))
        self.sig_high.bind(on_release=lambda *_: self.pick_signal('high'))
        for w in (self.sig_low, self.sig_mid, self.sig_high):
            self.add_widget(w)

        # Entscheidungen P2 (oben Mitte)
        self.dec_bluff = IconButton(ASSETS['decide']['bluff'], size_hint=(None,None), size=(260,260), pos=(W/2-160-260, H-60-260))
        self.dec_wahr  = IconButton(ASSETS['decide']['wahr'],  size_hint=(None,None), size=(260,260), pos=(W/2+160, H-60-260))
        self.dec_bluff.bind(on_release=lambda *_: self.pick_decision('bluff'))
        self.dec_wahr.bind(on_release=lambda *_: self.pick_decision('wahr'))
        self.add_widget(self.dec_bluff)
        self.add_widget(self.dec_wahr)

        # Showdown-Label (Mitte) + Weiter-Button
        self.showdown_label = Label(text='', font_size=56, color=(1,1,1,1), size_hint=(None,None), size=(1200,80), pos=(W/2-600, H/2-40))
        self.add_widget(self.showdown_label)

        self.btn_next_round = Button(text='Karten gezeigt – nächste Runde', font_size=32, size_hint=(None,None), size=(760,90), pos=(W/2-380, H/2-180))
        self.btn_next_round.bind(on_release=lambda *_: self.prepare_next_round())
        self.add_widget(self.btn_next_round)

        # Rundenbadge unten Mitte
        self.round_badge = Label(text='', font_size=36, color=(1,1,1,1), size_hint=(None,None), size=(1200,60), pos=(W/2-600, 10))
        self.add_widget(self.round_badge)

        # interne States
        self.p1_pressed = False
        self.p2_pressed = False
        self.p1_signal = None
        self.p2_decision = None

    # --- Logik
    def apply_phase(self):
        # Alles zunächst deaktivieren
        for c in (self.p1_outer, self.p1_inner, self.p2_outer, self.p2_inner):
            c.set_live(False)
        for b in (self.sig_low, self.sig_mid, self.sig_high, self.dec_bluff, self.dec_wahr):
            b.set_live(False)
        self.btn_next_round.disabled = True
        self.btn_next_round.opacity = 0
        self.showdown_label.text = ''

        # Startbuttons
        both_start = (self.phase in (PH_WAIT_BOTH_START, PH_INTER_ROUND))
        self.btn_start_p1.set_live(both_start)
        self.btn_start_p2.set_live(both_start)

        # Phasen-spezifisch
        if self.phase == PH_P1_INNER:
            self.p1_inner.set_live(True)
        elif self.phase == PH_P2_INNER:
            self.p2_inner.set_live(True)
        elif self.phase == PH_P1_OUTER:
            self.p1_outer.set_live(True)
        elif self.phase == PH_P2_OUTER:
            self.p2_outer.set_live(True)
        elif self.phase == PH_P1_SIGNAL:
            # nur P1-Signale
            self.sig_low.set_live(True)
            self.sig_mid.set_live(True)
            self.sig_high.set_live(True)
        elif self.phase == PH_P2_DECIDE:
            self.dec_bluff.set_live(True)
            self.dec_wahr.set_live(True)
        elif self.phase == PH_SHOWDOWN:
            self.showdown_label.text = 'SHOWDOWN'
            self.btn_next_round.disabled = False
            self.btn_next_round.opacity = 1

        # Badge unten
        role_txt = f"P1: {'Signal' if self.p1_role=='signaler' else 'Judge'} · P2: {'Signal' if self.p1_role!='signaler' else 'Judge'}"
        self.round_badge.text = f"Runde {self.round} · {role_txt}"

    def start_pressed(self, who:int):
        if self.phase not in (PH_WAIT_BOTH_START, PH_INTER_ROUND):
            return
        if who == 1:
            self.p1_pressed = True
        else:
            self.p2_pressed = True
        if self.p1_pressed and self.p2_pressed:
            # in nächste Phase
            self.p1_pressed = False
            self.p2_pressed = False
            self.phase = PH_P1_INNER
            self.apply_phase()

    def tap_card(self, who:int, which:str):
        # which in {'inner','outer'}
        if who == 1 and which == 'inner' and self.phase == PH_P1_INNER:
            self.p1_inner.flip()
            Clock.schedule_once(lambda *_: self.goto(PH_P2_INNER), 0.2)
        elif who == 2 and which == 'inner' and self.phase == PH_P2_INNER:
            self.p2_inner.flip()
            Clock.schedule_once(lambda *_: self.goto(PH_P1_OUTER), 0.2)
        elif who == 1 and which == 'outer' and self.phase == PH_P1_OUTER:
            self.p1_outer.flip()
            Clock.schedule_once(lambda *_: self.goto(PH_P2_OUTER), 0.2)
        elif who == 2 and which == 'outer' and self.phase == PH_P2_OUTER:
            self.p2_outer.flip()
            Clock.schedule_once(lambda *_: self.goto(PH_P1_SIGNAL), 0.2)

    def pick_signal(self, level:str):
        if self.phase != PH_P1_SIGNAL:
            return
        self.p1_signal = level
        # fixiere Auswahl optisch (Button bleibt live)
        if level == 'low':
            self.sig_low.set_pressed_state()
        elif level == 'mid':
            self.sig_mid.set_pressed_state()
        else:
            self.sig_high.set_pressed_state()
        Clock.schedule_once(lambda *_: self.goto(PH_P2_DECIDE), 0.2)

    def pick_decision(self, decision:str):
        if self.phase != PH_P2_DECIDE:
            return
        self.p2_decision = decision
        if decision == 'bluff':
            self.dec_bluff.set_pressed_state()
        else:
            self.dec_wahr.set_pressed_state()
        Clock.schedule_once(lambda *_: self.goto(PH_SHOWDOWN), 0.2)

    def goto(self, phase):
        self.phase = phase
        self.apply_phase()

    def prepare_next_round(self):
        # Rollen tauschen
        self.p1_role = 'judge' if self.p1_role == 'signaler' else 'signaler'
        self.round += 1
        # Reset Karten & Auswahlen
        for c in (self.p1_inner, self.p1_outer, self.p2_inner, self.p2_outer):
            c.reset()
        self.p1_signal = None
        self.p2_decision = None
        # Nächste Runde – beide Play aktiv
        self.phase = PH_INTER_ROUND
        self.apply_phase()

class TabletopApp(App):
    def build(self):
        self.title = 'Masterarbeit – Tabletop UX'
        root = TabletopRoot()
        return root

if __name__ == '__main__':
    TabletopApp().run()
 