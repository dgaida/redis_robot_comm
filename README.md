# redis_robot_comm

Redis-basiertes Kommunikations- und Streaming-Package für Roboteranwendungen.
Es bietet zwei Hauptfunktionen:

* **Objekterkennung** über `RedisMessageBroker`
* **Bild-Streaming variabler Größe** über `RedisImageStreamer`

Das Package eignet sich z. B. für Robotik-Szenarien, in denen Detektionen und Kameradaten effizient zwischen Prozessen oder Maschinen ausgetauscht werden müssen.

---

## Badges

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml)
[![Code Quality](https://github.com/dgaida/redis_robot_comm/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/lint.yml)
[![CodeQL](https://github.com/dgaida/redis_robot_comm/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Installation

```bash
git clone https://github.com/dgaida/redis_robot_comm.git
cd redis_robot_comm
pip install -e .
```

Voraussetzungen:

* Python **>=3.8**
* Laufender **Redis-Server**
* Abhängigkeiten: `redis`, `opencv-python`, `numpy`

---

## Nutzung

### 1. Objekterkennung mit `RedisMessageBroker`

```bash
docker run -p 6379:6379 redis:alpine
```

```python
from redis_robot_comm import RedisMessageBroker
import time

broker = RedisMessageBroker()

# Verbindung testen
if broker.test_connection():
    print("Verbunden mit Redis")

# Stream löschen (nur zum Testen)
broker.clear_stream()

# Beispielobjekte publizieren
objects = [
    {"id": "obj_1", "class_name": "cube", "confidence": 0.95, "position": {"x": 0.1, "y": 0.2, "z": 0.05}, "timestamp": time.time()},
    {"id": "obj_2", "class_name": "cylinder", "confidence": 0.87, "position": {"x": 0.3, "y": 0.1, "z": 0.05}, "timestamp": time.time()},
]
camera_pose = {"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 1.57, "yaw": 0.0}

broker.publish_objects(objects, camera_pose)

# Neueste Objekte abrufen
latest = broker.get_latest_objects()
print("Neueste Objekte:", latest)
```

---

### 2. Bild-Streaming mit `RedisImageStreamer`

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")

# Beispielbild laden
image = cv2.imread("example.jpg")

# Bild veröffentlichen
streamer.publish_image(image, metadata={"robot": "arm1", "workspace": "A"})

# Neuestes Bild abrufen
result = streamer.get_latest_image()
if result:
    img, metadata = result
    print("Metadaten:", metadata)
    cv2.imshow("Received Image", img)
    cv2.waitKey(0)
```

---

### 3. Beispielskript

Im Repository befindet sich eine **main.py**, die zeigt, wie der `RedisMessageBroker` verwendet wird:

```bash
python redis_robot_comm/main.py
```

---

## Projektstruktur

```
redis_robot_comm/
│
├── redis_robot_comm/
│   ├── __init__.py
│   ├── redis_client.py
│   ├── redis_image_streamer.py
│
├── tests/
├── main.py         # Beispiel zur Nutzung
├── pyproject.toml
└── README.md
```

---

## Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**.
