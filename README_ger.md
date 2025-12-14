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

Das `redis_robot_comm` Package bietet eine effiziente Redis-basierte Kommunikationsinfrastruktur für Roboteranwendungen. Es ermöglicht den Austausch von Kamerabildern, Objektdetektionen, Metadaten und Text-Overlays zwischen verschiedenen Prozessen oder Systemen in Echtzeit.

### Hauptfunktionen

* 📦 **Objekterkennung** - Streaming von Detektionsergebnissen über `RedisMessageBroker`
* 📷 **Bild-Streaming** - Variable Bildgrößen mit JPEG-Kompression über `RedisImageStreamer`
* 🏷️ **Label-Verwaltung** - Dynamische Objektlabels mit `RedisLabelManager`
* 📝 **Text-Overlays** - Video-Aufnahme-Integration mit `RedisTextOverlayManager` (NEU!)
* ⚡ **Echtzeitfähig** - Sub-Millisekunden-Latenz für lokale Redis-Server
* 🔄 **Asynchron** - Entkoppelte Producer-Consumer-Architektur
* 📊 **Metadaten** - Automatische Zeitstempel, Roboterposen, Workspace-Informationen
* 🎯 **Robotik-optimiert** - Speziell für Pick-and-Place und Vision-Anwendungen

---

## Anwendungsfälle

Das Package wird in Robotik-Frameworks als Kommunikations-Backbone eingesetzt:

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Objekterkennung mit OwlV2, YOLO-World, YOLOE, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robotersteuerung mit visueller Objekterkennung
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - LLM-basierte Robotersteuerung mit MCP

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
    metadata={"robot": "arm1", "workspace": "A"},
    compress_jpeg=True,
    quality=85,
    maxlen=5  # Nur letzten 5 Frames behalten
)

# Neuestes Bild abrufen
result = streamer.get_latest_image()
if result:
    img, metadata = result
    print(f"Metadaten: {metadata}")
    cv2.imshow("Empfangenes Bild", img)
    cv2.waitKey(0)
```

---

### 3. Label-Verwaltung mit `RedisLabelManager`

```python
from redis_robot_comm import RedisLabelManager

label_mgr = RedisLabelManager()

# Labels publizieren
labels = ["cube", "sphere", "cylinder"]
label_mgr.publish_labels(labels, metadata={"model_id": "yolo-v8"})

# Aktuelle Labels abrufen
current_labels = label_mgr.get_latest_labels(timeout_seconds=5.0)
print(f"Erkennbare Objekte: {current_labels}")

# Neues Label hinzufügen
label_mgr.add_label("prism")
```

---

### 4. Text-Overlays mit `RedisTextOverlayManager` (NEU!)

Der neue `RedisTextOverlayManager` ermöglicht die Integration von Text-Overlays für Videoaufnahmen:

```python
from redis_robot_comm import RedisTextOverlayManager

text_mgr = RedisTextOverlayManager()

# Benutzer-Aufgabe publizieren (persistent)
text_mgr.publish_user_task(
    task="Nimm den Stift und lege ihn neben den Würfel"
)

# Roboter-Aussage publizieren (zeitlich begrenzt, 4 Sekunden)
text_mgr.publish_robot_speech(
    speech="🤖 Ich nehme jetzt den Stift auf",
    duration_seconds=4.0
)

# System-Nachricht publizieren
text_mgr.publish_system_message(
    message="🎥 Aufnahme gestartet",
    duration_seconds=3.0
)

# Text-Updates abonnieren
def on_text_update(text_data):
    print(f"{text_data['type']}: {text_data['text']}")

text_mgr.subscribe_to_texts(on_text_update)
```

**Anwendungsfälle:**
- Videoaufnahmen mit Aufgaben-Overlays
- Roboter-Aktions-Kommentare
- System-Status-Meldungen
- Bildungsvideos
- Dokumentationsvideos

Siehe **[docs/text_overlay_readme.md](docs/text_overlay_readme.md)** für detaillierte Dokumentation.

---

## Utility-Skripte

### Annotierte Frames visualisieren

```bash
python scripts/visualize_annotated_frames.py --stream-name annotated_camera
```

**Steuerung:**
- `q/ESC` - Beenden
- `s` - Screenshot speichern
- `p` - Pause/Fortsetzen
- `f` - FPS-Anzeige umschalten

### Kamera mit Text-Overlays aufnehmen

```bash
python scripts/record_camera_with_overlays.py \
  --camera 0 \
  --stream annotated_camera \
  --layout side-by-side
```

**Features:**
- Benutzer-Aufgaben-Anzeige (persistent)
- Roboter-Sprach-Overlays (zeitlich begrenzt)
- TH Köln Branding
- Side-by-Side oder Overlay-Layouts
- Unicode/Emoji-Unterstützung

**Steuerung:**
- `q/ESC` - Aufnahme stoppen
- `p` - Pause/Fortsetzen
- `s` - Screenshot

---

## Performance

### Latenzzeiten (lokaler Redis-Server)

| Operation | Typische Latenz | Anmerkungen |
|-----------|----------------|-------------|
| Bild publizieren (640×480) | 5-20 ms | Abhängig von JPEG-Qualität |
| Bild abrufen | <1 ms | In-Memory-Operation |
| Objekt publizieren | <1 ms | JSON-Serialisierung |
| Objekt abrufen | <1 ms | JSON-Deserialisierung |
| Text-Overlay publizieren | <1 ms | Leichtgewichtige Operation |

### Durchsatz

- **Bild-Streaming**: 30-60 FPS (JPEG, quality=85)
- **Objekt-Publishing**: 1000+ Objekte/Sekunde
- **Text-Overlays**: 10000+ Operationen/Sekunde
- **Multi-Consumer**: Keine signifikante Performance-Beeinträchtigung

---

## Erweiterte Verwendung

### Kontinuierliches Bild-Streaming

```python
import cv2
import threading
from redis_robot_comm import RedisImageStreamer

streamer = RedisImageStreamer()
stop_flag = threading.Event()

def on_frame(image, metadata, image_info):
    print(f"Frame {image_info['width']}×{image_info['height']} empfangen")
    cv2.imshow("Live Stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_flag.set()
        return False
    return True

# Subscriber in eigenem Thread starten
def subscriber_loop():
    streamer.subscribe_variable_images(
        lambda img, meta, info: on_frame(img, meta, info)
    )

thread = threading.Thread(target=subscriber_loop, daemon=True)
thread.start()

# Publisher-Loop
cap = cv2.VideoCapture(0)
try:
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if ret:
            streamer.publish_image(frame)
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
```

---

## Projektstruktur

```
redis_robot_comm/
│
├── redis_robot_comm/
│   ├── __init__.py
│   ├── redis_client.py           # RedisMessageBroker
│   ├── redis_image_streamer.py   # RedisImageStreamer
│   ├── redis_label_manager.py    # RedisLabelManager
│   └── redis_text_overlay.py     # RedisTextOverlayManager (NEU!)
│
├── scripts/
│   ├── visualize_annotated_frames.py
│   ├── record_camera_with_overlays.py  # Erweitert mit Text-Overlays
│   └── camera_recorder_audio.py
│
├── docs/
│   ├── README.md                  # Workflow-Dokumentation
│   ├── api.md                     # API-Referenz
│   ├── text_overlay_readme.md     # Text-Overlay-Anleitung (NEU!)
│   ├── TESTING.md                 # Test-Anleitung
│   └── *.png
│
├── tests/
│   ├── test_redis_robot_comm.py
│   ├── test_redis_robot_comm_extended.py
│   ├── test_redis_label_manager.py
│   └── test_redis_text_overlay.py  # NEU!
│
├── examples/
│   └── main.py
│
└── README.md
```

---

## API-Referenz

Für detaillierte API-Dokumentation siehe: **[docs/api.md](docs/api.md)**

### RedisMessageBroker

| Methode | Beschreibung |
|---------|--------------|
| `publish_objects(objects, camera_pose)` | Objekte publizieren |
| `get_latest_objects(max_age_seconds)` | Neueste Objekte abrufen |
| `subscribe_objects(callback)` | Kontinuierliches Abonnement |

### RedisImageStreamer

| Methode | Beschreibung |
|---------|--------------|
| `publish_image(image, metadata, compress_jpeg, quality)` | Bild publizieren |
| `get_latest_image()` | Neuestes Bild abrufen |
| `subscribe_variable_images(callback)` | Kontinuierliches Streaming |

### RedisLabelManager

| Methode | Beschreibung |
|---------|--------------|
| `publish_labels(labels, metadata)` | Label-Liste publizieren |
| `get_latest_labels(timeout_seconds)` | Aktuelle Labels abrufen |
| `add_label(new_label)` | Neues Label hinzufügen |

### RedisTextOverlayManager (NEU!)

| Methode | Beschreibung |
|---------|--------------|
| `publish_user_task(task)` | Persistente Benutzer-Aufgabe publizieren |
| `publish_robot_speech(speech, duration)` | Zeitlich begrenzte Roboter-Nachricht |
| `publish_system_message(message, duration)` | System-Nachricht publizieren |
| `get_latest_texts(max_age_seconds)` | Letzte Texte abrufen |
| `subscribe_to_texts(callback)` | Text-Updates überwachen |

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

**Test-Abdeckung:** >90% über alle Module

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

## Integration in eigene Projekte

### Objekterkennung integrieren

```python
from redis_robot_comm import RedisMessageBroker

broker = RedisMessageBroker()

def detect_and_publish(image):
    objects = detector.detect(image)
    broker.publish_objects(
        objects,
        camera_pose={"x": 0.0, "y": 0.0, "z": 0.5}
    )
```

### Videoaufnahme mit Text-Overlays

```python
from redis_robot_comm import RedisTextOverlayManager

text_mgr = RedisTextOverlayManager()

# MCP Server publiziert Benutzer-Aufgaben
def handle_user_command(command: str):
    text_mgr.publish_user_task(command)
    # Befehl ausführen...

# Roboter publiziert Aktions-Kommentare
def robot_action(action: str):
    text_mgr.publish_robot_speech(
        speech=f"🤖 {action}",
        duration_seconds=4.0
    )
```

---

## Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**. Siehe [LICENSE](LICENSE) für Details.

---

## Verwandte Projekte

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Objekterkennung mit OwlV2, YOLO-World, YOLOE, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robotersteuerung mit visueller Objekterkennung
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - LLM-basierte Robotersteuerung mit MCP

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
