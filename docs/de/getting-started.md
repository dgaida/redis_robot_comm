# Erste Schritte

Diese Anleitung hilft Ihnen beim schnellen Einstieg in `redis_robot_comm`.

## Voraussetzungen

* **Python** ≥ 3.8
* **Redis-Server** ≥ 5.0 (für Streams-Unterstützung)

## Schnellstart-Beispiele

### 1. Objekterkennung

```python
from redis_robot_comm import RedisMessageBroker
import time

broker = RedisMessageBroker()

# Verbindung testen
if broker.test_connection():
    print("✓ Verbunden mit Redis")

# Beispielobjekte publizieren
objects = [
    {
        "id": "obj_1",
        "class_name": "cube",
        "confidence": 0.95,
        "position": {"x": 0.1, "y": 0.2, "z": 0.05},
        "timestamp": time.time()
    }
]

broker.publish_objects(objects)

# Neueste Objekte abrufen
latest = broker.get_latest_objects(max_age_seconds=2.0)
print(f"Gefundene Objekte: {len(latest)}")
```

### 2. Bild-Streaming

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer()

# Beispielbild laden
image = cv2.imread("example.jpg")

# Bild veröffentlichen
streamer.publish_image(image, compress_jpeg=True, quality=85)

# Neuestes Bild abrufen
result = streamer.get_latest_image()
if result:
    img, metadata = result
    cv2.imshow("Empfangenes Bild", img)
    cv2.waitKey(0)
```

## Weitere Informationen

Für detaillierte Informationen zu den einzelnen Modulen besuchen Sie bitte die Abschnitte [Benutzung](usage/detection.md) und [API-Referenz](api/broker.md).
