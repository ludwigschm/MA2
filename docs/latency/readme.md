# Latenz-Baseline

Das Skript `tools/latency_baseline.py` wertet die vorhandenen Event-Logs aus und berechnet eine Baseline über 60 Sekunden. Die Ausgabe enthält statistische Kennzahlen zum Offset sowie eine Drift-Schätzung.

## Ausführung

```bash
python tools/latency_baseline.py --db logs/events_12.sqlite3
```

Ohne Angabe von `--db` wählt das Skript automatisch die neueste `*.sqlite3`-Datei im Verzeichnis `logs/`.

## Ausgabe

Die Kennzahlen werden sowohl als menschenlesbare Zusammenfassung auf der Konsole als auch als JSON ausgegeben. Der JSON-Block umfasst:

- `samples`: Anzahl der ausgewerteten Sync-Paare
- `window_seconds`: Länge des ausgewerteten Fensters
- `offset_mean_ns`: Mittelwert des Offsets in Nanosekunden
- `offset_median_ns`: Median des Offsets in Nanosekunden
- `offset_rms_ns`: RMS-Abweichung (Jitter) in Nanosekunden
- `drift_ns_per_s`: Geschätzte Drift in Nanosekunden pro Sekunde (falls berechenbar)

Über die Option `--json <pfad>` lässt sich die JSON-Ausgabe zusätzlich in eine Datei schreiben.
