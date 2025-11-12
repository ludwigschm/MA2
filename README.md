Folgende Pakete müssen installiert werden: 

kivy	
pandas
numpy	
opencv-contrib-python
sounddevice


pip install kivy pandas numpy opencv-contrib-python sounddevice

### Optional env vars (Windows/PowerShell)

```
setx FORCE_EVENT_LABELS "1"
setx RECORDING_BEGIN_TIMEOUT_MS "3000"
setx WS_PING_INTERVAL_SEC "12"
setx AUTO_START_EYETRACKER "1"
setx CONTINUOUS_RECORDING_MODE "1"
```

## Event-Zeitmodell & Refinement

Die Tabletop-App vergibt jetzt für jedes UI-Ereignis sofort eine eindeutige
`event_id` sowie einen hochauflösenden Zeitstempel (`t_local_ns`) auf Basis von
`time.perf_counter_ns()`. Diese Metadaten werden zusammen mit
`mapping_version`, `origin_device` und `provisional=True` an beide Pupil-Brillen
geschickt. Der `PupilBridge` stellt sicher, dass alle Events – auch Fallbacks –
dieses Format einhalten und bietet mit `refine_event(...)` eine API zum späteren
Verfeinern.

Der neue `TimeReconciler` läuft im Hintergrund, verarbeitet regelmäßige
`sync.heartbeat`-Marker und schätzt daraus Offset und Drift je Gerät (robuste
lineare Regression mit Huber-Loss). Sobald verlässliche Parameter vorliegen,
werden alle `provisional`-Events mit einem gemeinsamen Referenzzeitpunkt
(`t_ref_ns`) aktualisiert. Die Ergebnisse landen sowohl über die Bridge (Cloud)
als auch lokal in den Session-Datenbanken (`logs/events_<session>.sqlite3`)
innerhalb der Tabelle `event_refinements`.

Einen schnellen Smoke-Test liefert:

```bash
python -m tabletop.app --demo
```

Der Demo-Lauf simuliert Button-Events sowie Heartbeats, zeigt den Übergang von
„provisional“ → „refined“ mit `event_id`, aktueller `mapping_version`,
Konfidenz und Queue-Last direkt in der Konsole und erzeugt eine Demo-Datenbank
(`logs/demo_refinement.sqlite3`).

## E2E-Smoketest (Edge & Cloud)

Für einen integrierten Happy-Path-Check steht `tools/e2e_smoke.py` bereit. Das
Script wärmt die TimeSync-Schätzung auf, ruft die Edge-Health-API ab, sendet
einige Refine-Events und prüft optional den Cloud-Ingest inklusive Offline-Queue.

```bash
make e2e
```

Über `ARGS` lassen sich zusätzliche Parameter übergeben, z. B. alternative Edge-
URLs oder die Simulation eines Cloud-DNS-Fehlers:

```bash
make e2e ARGS='--edge-url http://192.168.137.83:8080 --simulate-cloud-dns-fail'
```

### Konfiguration

* Edge-Basis-URLs werden standardmäßig aus `core/config.py` übernommen. Mit
  `--edge-url` können beliebige Endpunkte angegeben werden (mehrfach möglich).
* Für den Cloud-Schritt liest das Script `PUPYLABS_CLOUD_BASE_URL` bzw.
  `PUPYLABS_BASE_URL` sowie `PUPYLABS_CLOUD_API_KEY`/`PUPYLABS_API_KEY`. Die
  Option `--cloud-url`/`--cloud-api-key` überschreibt diese Werte.
* Die Offline-Queue liegt per Default unter `logs/e2e_cloud_queue.ndjson` und
  kann mit `--cloud-queue` umgelenkt werden.

### Exit-Codes & typische Fehlerbilder

| Exit-Code | Bedeutung | Hinweise |
|-----------|-----------|----------|
| `0` | Alle Checks erfolgreich oder Cloud-DNS-Fehler durch Queue abgefedert | Konsolenausgabe zeigt Offset/Drift, HTTP-Status und ggf. Queue-Pfad. |
| `1` | TimeSync instabil | Prüfen, ob `/api/time/sync` erreichbar ist und ob genügend Messwerte zurückkommen. Optional `--timesync-samples` erhöhen. |
| `2` | Edge-Health oder Refine fehlgeschlagen | Sicherstellen, dass die Edge-Instanz läuft und `EDGE_REFINE_PATHS` unterstützt. Log-Ausgabe nennt den fehlerhaften Schritt. |
| `3` | Cloud-Fehler ohne Offline-Queue | Pfad aus `--cloud-queue` kontrollieren und Schreibrechte prüfen. Bei DNS-Tests sicherstellen, dass `--simulate-cloud-dns-fail` gesetzt ist. |

### Troubleshooting & Queue-Flush

* Bei `--simulate-cloud-dns-fail` wird der Cloud-Host gezielt auf einen
  ungültigen Namen umgebogen. Das Script bestätigt anschließend, dass Ereignisse
  in der Offline-Queue landen. Sobald DNS wieder funktioniert, `make e2e`
  ohne diese Option erneut ausführen oder die Tabletop-App starten – die Queue
  wird automatisch leergezogen.
* Bleibt die TimeSync-Konfidenz niedrig, hilft oft ein Neustart der Brille bzw.
  ein Abgleich der lokalen Uhrzeit. Der Parameter `--timesync-timeout` erlaubt
  längere Antwortzeiten.
