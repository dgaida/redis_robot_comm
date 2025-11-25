# API-Referenz

## RedisMessageBroker

```python
broker = RedisMessageBroker(host="localhost", port=6379, db=0)
```

**Methoden:**

| Methode | Beschreibung | Rückgabe |
|---------|--------------|----------|
| `publish_objects(objects, camera_pose)` | Objektliste publizieren | Stream-ID oder None |
| `get_latest_objects(max_age_seconds)` | Neueste Objekte abrufen | Liste von Objekten |
| `get_objects_in_timerange(start, end)` | Objekte im Zeitbereich | Liste von Objekten |
| `subscribe_objects(callback)` | Kontinuierliches Abonnement | - (blocking) |
| `clear_stream()` | Stream löschen | Anzahl gelöschter Einträge |
| `get_stream_info()` | Stream-Informationen | Dict mit Statistiken |
| `test_connection()` | Verbindungstest | True/False |

**Objektformat:**
```python
{
    "id": str,              # Eindeutige Objekt-ID
    "class_name": str,      # Objektklasse
    "confidence": float,    # Konfidenz (0-1)
    "position": {           # Position im Raum
        "x": float,
        "y": float,
        "z": float
    },
    "timestamp": float      # Unix-Zeitstempel
}
```

---

## RedisImageStreamer

```python
streamer = RedisImageStreamer(
    host="localhost",
    port=6379,
    stream_name="robot_camera"
)
```

**Methoden:**

| Methode | Beschreibung | Rückgabe |
|---------|--------------|----------|
| `publish_image(image, metadata, compress_jpeg, quality, maxlen)` | Bild publizieren | Stream-ID |
| `get_latest_image(timeout_ms)` | Neuestes Bild abrufen | (image, metadata) oder None |
| `subscribe_variable_images(callback, block_ms)` | Kontinuierliches Streaming | - (blocking) |
| `get_stream_stats()` | Stream-Statistiken | Dict mit Statistiken |

**Parameter:**

- `image`: NumPy-Array (OpenCV-Format)
- `metadata`: Dict mit beliebigen Metadaten
- `compress_jpeg`: Bool - JPEG-Kompression aktivieren
- `quality`: Int (1-100) - JPEG-Qualität
- `maxlen`: Int - Maximale Anzahl Frames im Stream
