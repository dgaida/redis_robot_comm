# Fehlerbehebung

Hier finden Sie Lösungen für häufig auftretende Probleme.

## Verbindungsprobleme zu Redis

### Fehlermeldung: `Failed to connect to Redis`

**Mögliche Ursachen:**
1. Redis-Server läuft nicht.
2. Falsche Host- oder Port-Konfiguration.
3. Firewall blockiert den Zugriff.

**Lösungen:**
* Überprüfen Sie, ob Redis läuft: `redis-cli ping` sollte mit `PONG` antworten.
* Stellen Sie sicher, dass der Docker-Container läuft (falls verwendet): `docker ps`.
* Überprüfen Sie Ihre `REDIS_HOST` und `REDIS_PORT` Einstellungen.

## Performance-Probleme

### Hohe Latenz beim Bild-Streaming

**Mögliche Ursachen:**
1. Große Bilder ohne Kompression.
2. Netzwerk-Bandbreite ist ausgelastet.
3. Hohe CPU-Last durch JPEG-Kodierung.

**Lösungen:**
* Aktivieren Sie die JPEG-Kompression: `compress_jpeg=True`.
* Reduzieren Sie die JPEG-Qualität: `quality=70`.
* Verringern Sie die Bildauflösung vor dem Senden.

## Datenverlust

### Alte Nachrichten verschwinden aus dem Stream

**Ursache:**
* Die Parameter `maxlen` (standardmäßig 500 für Objekte, 5 für Bilder) begrenzen die Anzahl der gespeicherten Einträge.

**Lösung:**
* Erhöhen Sie den `maxlen` Wert beim Veröffentlichen, falls Sie einen größeren Puffer benötigen.

## Debugging

Aktivieren Sie den `verbose`-Modus, um mehr Informationen über interne Vorgänge zu erhalten:

```python
streamer = RedisImageStreamer()
streamer.verbose = True
```
