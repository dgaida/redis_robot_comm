# redis_robot_comm

Redis-basiertes Kommunikations- und Streaming-Package für Roboteranwendungen.

## Badges

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/redis_robot_comm/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/redis_robot_comm)
[![Tests](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml)
[![Code Quality](https://github.com/dgaida/redis_robot_comm/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/lint.yml)
[![CodeQL](https://github.com/dgaida/redis_robot_comm/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Übersicht

Das `redis_robot_comm` Package bietet eine effiziente Redis-basierte Kommunikationsinfrastruktur für Roboteranwendungen. Es ermöglicht den Austausch von Kamerabildern und Objektdetektionen zwischen verschiedenen Prozessen oder Systemen in Echtzeit.

### Hauptfunktionen

* 📦 **Objekterkennung** - Streaming von Detektionsergebnissen über `RedisMessageBroker`
* 📷 **Bild-Streaming** - Variable Bildgrößen mit JPEG-Kompression über `RedisImageStreamer`
* ⚡ **Echtzeitfähig** - Sub-Millisekunden-Latenz für lokale Redis-Server
* 🔄 **Asynchron** - Entkoppelte Producer-Consumer-Architektur
* 📊 **Metadaten** - Automatische Zeitstempel, Roboterposen, Workspace-Informationen
* 🎯 **Robotik-optimiert** - Speziell für Pick-and-Place und Vision-Anwendungen

---

## Anwendungsfälle

Das Package wird in zwei größeren Robotik-Frameworks als Kommunikations-Backbone eingesetzt:

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Objekterkennung mit OwlV2, YOLO-World, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robotersteuerung mit visueller Objekterkennung

Für detaillierte Workflow-Dokumentation siehe: **[docs/README.md](docs/README.md)**

---

## Installation

```bash
git clone https://github.com/dgaida/redis_robot_comm.git
cd redis_robot_comm
pip install -e .
```

### Voraussetzungen

* **Python** ≥ 3.8
* **Redis-Server** ≥ 5.0 (für Streams-Unterstützung)
* **Abhängigkeiten**: `redis`, `opencv-python`, `numpy`

### Redis-Server starten

```bash
# Mit Docker (empfohlen)
docker run -p 6379:6379 redis:alpine

# Oder lokal installieren
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis
```

---

## Schnellstart

### 1. Objekterkennung mit `RedisMessageBroker`

![Objekterkennungs-Workflow](docs/workflow_detector.png)

```python
from redis_robot_comm import RedisMessageBroker
import time

broker = RedisMessageBroker()

# Verbindung testen
if broker.test_connection():
    print("✓ Verbunden mit Redis")

# Stream für Tests leeren
broker.clear_stream()

# Beispielobjekte publizieren
objects = [
    {
        "id": "obj_1",
        "class_name": "cube",
        "confidence": 0.95,
        "position": {"x": 0.1, "y": 0.2, "z": 0.05},
        "timestamp": time.time()
    },
    {
        "id": "obj_2",
        "class_name": "cylinder",
        "confidence": 0.87,
        "position": {"x": 0.3, "y": 0.1, "z": 0.05},
        "timestamp": time.time()
    }
]

camera_pose = {
    "x": 0.0, "y": 0.0, "z": 0.5,
    "roll": 0.0, "pitch": 1.57, "yaw": 0.0
}

broker.publish_objects(objects, camera_pose)

# Neueste Objekte abrufen
latest = broker.get_latest_objects(max_age_seconds=2.0)
print(f"Gefundene Objekte: {len(latest)}")
for obj in latest:
    print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")
```

**Funktionen:**
- `publish_objects()` - Objekte mit Metadaten publizieren
- `get_latest_objects()` - Neueste Objekte mit Altersfilter abrufen
- `get_objects_in_timerange()` - Objekte in Zeitbereich abfragen
- `subscribe_objects()` - Kontinuierliches Abonnement (blocking)
- `clear_stream()` - Stream zurücksetzen
- `get_stream_info()` - Stream-Statistiken abrufen

---

### 2. Bild-Streaming mit `RedisImageStreamer`

![Bild-Streaming-Workflow](docs/workflow_streamer.png)

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")

# Beispielbild laden
image = cv2.imread("example.jpg")

# Bild mit Metadaten veröffentlichen
stream_id = streamer.publish_image(
    image,
    metadata={
        "robot": "arm1",
        "workspace": "A",
        "frame_id": 42
    },
    compress_jpeg=True,
    quality=85,
    maxlen=5  # Nur letzten 5 Frames behalten
)

print(f"Bild publiziert: {stream_id}")

# Neuestes Bild abrufen
result = streamer.get_latest_image()
if result:
    img, metadata = result
    print(f"Metadaten: {metadata}")
    cv2.imshow("Empfangenes Bild", img)
    cv2.waitKey(0)
```

**Features:**
- ✅ **Variable Bildgrößen** - Automatische Anpassung an verschiedene Auflösungen
- ✅ **JPEG-Kompression** - Konfigurierbare Qualität (1-100)
- ✅ **Raw-Modus** - Verlustfreie Übertragung möglich
- ✅ **Metadaten** - Roboterposen, Workspace-IDs, Zeitstempel
- ✅ **Stream-Verwaltung** - Automatisches Entfernen alter Frames

**Funktionen:**
- `publish_image()` - Bild mit optionaler Kompression publizieren
- `get_latest_image()` - Neuestes Bild abrufen
- `subscribe_variable_images()` - Kontinuierliches Streaming (blocking)
- `get_stream_stats()` - Stream-Statistiken abrufen

---

### 3. Kontinuierliches Streaming

```python
import cv2
import threading
from redis_robot_comm import RedisImageStreamer

streamer = RedisImageStreamer()

# Callback-Funktion für empfangene Bilder
def on_frame(image, metadata, image_info):
    print(f"Frame {image_info['width']}×{image_info['height']} empfangen")
    cv2.imshow("Live Stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

# Subscriber in eigenem Thread starten
stop_flag = threading.Event()

def subscriber_loop():
    def callback(img, meta, info):
        if not on_frame(img, meta, info):
            stop_flag.set()
    
    streamer.subscribe_variable_images(
        callback=callback,
        block_ms=500
    )

thread = threading.Thread(target=subscriber_loop, daemon=True)
thread.start()

# Publisher-Loop
cap = cv2.VideoCapture(0)
try:
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if ret:
            streamer.publish_image(frame, metadata={"source": "webcam"})
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
```

---

## API-Referenz

### RedisMessageBroker

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

### RedisImageStreamer

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

---

## Projektstruktur

```
redis_robot_comm/
│
├── redis_robot_comm/
│   ├── __init__.py
│   ├── redis_client.py           # RedisMessageBroker
│   ├── redis_image_streamer.py   # RedisImageStreamer
│
├── docs/
│   ├── README.md                  # Workflow-Dokumentation
│   ├── workflow_detector.png      # Objekterkennungs-Workflow
│   └── workflow_streamer.png      # Bild-Streaming-Workflow
│
├── tests/
│   ├── __init__.py
│   └── test_redis_robot_comm.py
│
├── main.py                        # Beispiel-Skript
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Performance

### Latenzzeiten (lokaler Redis-Server)

| Operation | Typische Latenz | Anmerkungen |
|-----------|----------------|-------------|
| Bild publizieren (640×480) | 5-20 ms | Abhängig von JPEG-Qualität |
| Bild abrufen | <1 ms | In-Memory-Operation |
| Objekt publizieren | <1 ms | JSON-Serialisierung |
| Objekt abrufen | <1 ms | JSON-Deserialisierung |

### Durchsatz

- **Bild-Streaming**: 30-60 FPS (JPEG, quality=85)
- **Objekt-Publishing**: 1000+ Objekte/Sekunde
- **Multi-Consumer**: Keine signifikante Performance-Beeinträchtigung

### Optimierungsoptionen

```python
# Hohe Kompression (schneller, kleiner)
streamer.publish_image(image, compress_jpeg=True, quality=70)

# Niedrige Kompression (langsamer, bessere Qualität)
streamer.publish_image(image, compress_jpeg=True, quality=95)

# Keine Kompression (am langsamsten, verlustfrei)
streamer.publish_image(image, compress_jpeg=False)

# Stream-Größe begrenzen
streamer.publish_image(image, maxlen=5)  # Nur letzte 5 Frames
```

---

## Erweiterte Verwendung

### Objekte in Zeitbereichen abfragen

```python
import time

start_time = time.time() - 10  # Letzte 10 Sekunden
end_time = time.time()

objects = broker.get_objects_in_timerange(start_time, end_time)
print(f"Objekte in den letzten 10 Sekunden: {len(objects)}")
```

### Asynchrones Objekt-Abonnement

```python
def on_detection(data):
    objects = data["objects"]
    camera_pose = data["camera_pose"]
    timestamp = data["timestamp"]
    
    print(f"Empfangen: {len(objects)} Objekte")
    for obj in objects:
        print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")

try:
    broker.subscribe_objects(on_detection)  # Blocking
except KeyboardInterrupt:
    print("Abonnement beendet")
```

### Stream-Statistiken

```python
# RedisMessageBroker
info = broker.get_stream_info()
print(f"Stream-Länge: {info['length']}")
print(f"Erster Eintrag: {info['first-entry']}")
print(f"Letzter Eintrag: {info['last-entry']}")

# RedisImageStreamer
stats = streamer.get_stream_stats()
print(f"Gesamt-Nachrichten: {stats['total_messages']}")
print(f"Erste Frame-ID: {stats['first_entry_id']}")
print(f"Letzte Frame-ID: {stats['last_entry_id']}")
```

---

## Beispiele

Das Repository enthält ein vollständiges Beispielskript, das alle Funktionen demonstriert:

```bash
# Redis starten
docker run -p 6379:6379 redis:alpine

# Beispiel ausführen
python main.py
```

Das Skript zeigt:
- ✅ Verbindungstest
- ✅ Objekt-Publishing und -Retrieval
- ✅ Bild-Publishing und -Retrieval
- ✅ Asynchrones Bild-Streaming mit Callback
- ✅ Visualisierung der Ergebnisse

---

## Integration in eigene Projekte

### Objekterkennung integrieren

```python
from redis_robot_comm import RedisMessageBroker
from your_detector import YourDetector

broker = RedisMessageBroker()
detector = YourDetector()

# Objekte erkennen und publizieren
def detect_and_publish(image):
    objects = detector.detect(image)
    broker.publish_objects(
        objects,
        camera_pose={"x": 0.0, "y": 0.0, "z": 0.5}
    )
    print(f"Publiziert: {len(objects)} Objekte")
```

### Robotersteuerung mit Objektdaten

```python
from redis_robot_comm import RedisMessageBroker
from your_robot import YourRobot

broker = RedisMessageBroker()
robot = YourRobot()

# Objekte abrufen und greifen
def pick_object(label):
    objects = broker.get_latest_objects()
    for obj in objects:
        if obj["class_name"] == label:
            position = obj["position"]
            robot.pick(position["x"], position["y"], position["z"])
            return True
    return False

success = pick_object("cube")
print(f"Objekt gegriffen: {success}")
```

---

## Fehlerbehandlung

```python
from redis.exceptions import ConnectionError

try:
    broker = RedisMessageBroker()
    if not broker.test_connection():
        print("❌ Keine Verbindung zu Redis")
except ConnectionError as e:
    print(f"❌ Redis-Verbindungsfehler: {e}")
    print("Stelle sicher, dass Redis läuft:")
    print("  docker run -p 6379:6379 redis:alpine")
```

---

## Tests

```bash
# Entwicklungsabhängigkeiten installieren
pip install -r requirements-dev.txt

# Tests ausführen
pytest tests/ -v

# Mit Coverage
pytest tests/ --cov=redis_robot_comm --cov-report=html
```

---

## Entwicklung

### Code-Qualität

Das Projekt verwendet moderne Python-Tools für Code-Qualität:

```bash
# Linting mit Ruff
ruff check .

# Formatierung mit Black
black .

# Type-Checking mit mypy
mypy redis_robot_comm --ignore-missing-imports

# Sicherheitscheck mit Bandit
bandit -r redis_robot_comm/
```

### Pre-Commit-Hooks

```bash
pip install pre-commit
pre-commit install
```

---

## Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**. Siehe [LICENSE](LICENSE) für Details.

---

## Verwandte Projekte

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Objekterkennung mit OwlV2, YOLO-World, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robotersteuerung mit visueller Objekterkennung

---

## Autor

**Daniel Gaida**  
E-Mail: daniel.gaida@th-koeln.de  
GitHub: [@dgaida](https://github.com/dgaida)

Project Link: https://github.com/dgaida/redis_robot_comm

---

## Acknowledgments

- [Redis](https://redis.io/) - Für die leistungsstarke In-Memory-Datenbank
- [OpenCV](https://opencv.org/) - Für Bildverarbeitung
- [Python Redis Client](https://github.com/redis/redis-py) - Für die Python-Integration
