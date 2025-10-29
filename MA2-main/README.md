Folgende Pakete müssen installiert werden:

kivy
pandas
numpy
opencv-contrib-python
sounddevice

Für die optionale Neon-Anbindung zusätzlich:

requests


pip install kivy pandas numpy opencv-contrib-python sounddevice

### Neon-Integration (optional)

Die Anbindung an Pupil Labs Neon ist standardmäßig deaktiviert. Konfiguriere die
Geräte in `neon_devices.txt` und installiere `requests`, um die Integration zu
nutzen.

Beispiel:

```
pip install requests
python bluffing_eyes.py --eye-tracker neon --player VP1
```

Alternativ kann die Umgebungsvariable `EYE_TRACKER=neon` gesetzt werden. Beim
Start und Ende der Fixation werden `FIXATION_START`- bzw.
`FIXATION_END`-Annotations verschickt. Tastatureingaben erzeugen zusätzliche
`KEY_DOWN`-Annotations.
