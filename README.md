# redis_robot_comm

**Redis-based communication and streaming package for robotics applications**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/redis_robot_comm/branch/main/graph/badge.svg)](https://codecov.io/gh/dgaida/redis_robot_comm)
[![Tests](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml)
[![Code Quality](https://github.com/dgaida/redis_robot_comm/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/lint.yml)
[![CodeQL](https://github.com/dgaida/redis_robot_comm/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Overview

`redis_robot_comm` provides a high-performance Redis-based communication infrastructure for robotics applications. It enables real-time exchange of camera images, object detections, metadata, and text overlays between distributed processes with sub-millisecond latency.

### Key Features

* 🎯 **Object Detection Streaming** - Publish and consume detection results with `RedisMessageBroker`
* 📷 **Variable-Size Image Streaming** - JPEG-compressed or raw image transfer with `RedisImageStreamer`
* 🏷️ **Label Management** - Dynamic object label configuration with `RedisLabelManager`
* 📝 **Text Overlay Support** - Video recording integration with `RedisTextOverlayManager`
* ⚡ **Real-Time Performance** - Sub-millisecond latency for local Redis servers
* 🔄 **Asynchronous Architecture** - Decoupled producer-consumer patterns
* 📊 **Rich Metadata Support** - Timestamps, camera poses, workspace information
* 🎯 **Robotics-Optimized** - Designed for pick-and-place and computer vision tasks

---

## Use Cases

This package serves as the communication backbone for robotics frameworks:

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Object detection with OwlV2, YOLO-World, YOLOE, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robot control with visual object recognition
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - LLM-based robot control using Model Context Protocol

For detailed workflow documentation, see **[docs/README.md](docs/README.md)**

### Data Flow via Redis Streams

![Data Flow via Redis Streams](https://github.com/dgaida/robot_mcp/blob/master/docs/redis_applications_architecture_diagram.png)

---

## Installation

### Prerequisites

* **Python** ≥ 3.8
* **Redis Server** ≥ 5.0 (for Streams support)

### Install Package

```bash
git clone https://github.com/dgaida/redis_robot_comm.git
cd redis_robot_comm
pip install -e .
```

### Start Redis Server

```bash
# Using Docker (recommended)
docker run -p 6379:6379 redis:alpine

# Or install locally
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis
```

---

## Quick Start

### 1. Object Detection with RedisMessageBroker

![Object Detection Workflow](docs/workflow_detector.png)

```python
from redis_robot_comm import RedisMessageBroker
import time

broker = RedisMessageBroker()

# Test connection
if broker.test_connection():
    print("✓ Connected to Redis")

# Publish detected objects
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

# Retrieve latest objects
latest = broker.get_latest_objects(max_age_seconds=2.0)
print(f"Found objects: {len(latest)}")
for obj in latest:
    print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")
```

---

### 2. Image Streaming with RedisImageStreamer

![Image Streaming Workflow](docs/workflow_streamer.png)

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")

# Publish image
image = cv2.imread("example.jpg")
stream_id = streamer.publish_image(
    image,
    metadata={"robot": "arm1", "workspace": "A"},
    compress_jpeg=True,
    quality=85,
    maxlen=5  # Keep only last 5 frames
)

# Retrieve latest image
result = streamer.get_latest_image()
if result:
    img, metadata = result
    print(f"Metadata: {metadata}")
    cv2.imshow("Received Image", img)
    cv2.waitKey(0)
```

**Features:**
- ✅ **Variable Image Sizes** - Automatic adaptation to different resolutions
- ✅ **JPEG Compression** - Configurable quality (1-100)
- ✅ **Raw Mode** - Lossless transfer option
- ✅ **Metadata Support** - Robot poses, workspace IDs, timestamps
- ✅ **Stream Management** - Automatic removal of old frames

**Core Methods:**
- `publish_image()` - Publish image with optional compression
- `get_latest_image()` - Retrieve newest image
- `subscribe_variable_images()` - Continuous streaming (blocking)
- `get_stream_stats()` - Stream statistics

---

### 3. Label Management with RedisLabelManager

```python
from redis_robot_comm import RedisLabelManager

label_mgr = RedisLabelManager()

# Publish available labels
labels = ["cube", "sphere", "cylinder"]
label_mgr.publish_labels(labels, metadata={"model_id": "yolo-v8"})

# Retrieve current labels
current_labels = label_mgr.get_latest_labels(timeout_seconds=5.0)
print(f"Detectable objects: {current_labels}")

# Add new label dynamically
label_mgr.add_label("prism")
```

---

### 4. Text Overlays with RedisTextOverlayManager (NEW!)

The new `RedisTextOverlayManager` enables integration of text overlays for video recording:

```python
from redis_robot_comm import RedisTextOverlayManager

text_mgr = RedisTextOverlayManager()

# Publish user task (persistent)
text_mgr.publish_user_task(
    task="Pick up the pencil and place it next to the cube"
)

# Publish robot speech (timed, 4 seconds)
text_mgr.publish_robot_speech(
    speech="🤖 I'm picking up the pencil now",
    duration_seconds=4.0
)

# Publish system message
text_mgr.publish_system_message(
    message="🎥 Recording started",
    duration_seconds=3.0
)

# Subscribe to text updates
def on_text_update(text_data):
    print(f"{text_data['type']}: {text_data['text']}")

text_mgr.subscribe_to_texts(on_text_update)
```

**Use Cases:**
- Video recording with task overlays
- Robot action commentary
- System status messages
- Educational demonstrations
- Documentation videos

See **[docs/text_overlay_readme.md](docs/text_overlay_readme.md)** for detailed documentation.

---

## Utility Scripts

### Visualize Annotated Frames

```bash
python scripts/visualize_annotated_frames.py --stream-name annotated_camera
```

**Controls:**
- `q/ESC` - Quit
- `s` - Save screenshot
- `p` - Pause/unpause
- `f` - Toggle FPS display

### Record Camera with Text Overlays

```bash
python scripts/record_camera_with_overlays.py \
  --camera 0 \
  --stream annotated_camera \
  --layout side-by-side
```

**Features:**
- User task display (persistent)
- Robot speech overlays (timed)
- TH Köln branding
- Side-by-side or overlay layouts
- Unicode/emoji support

**Controls:**
- `q/ESC` - Stop recording
- `p` - Pause/unpause
- `s` - Screenshot

---

## Performance

### Latency (Local Redis Server)

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Image publish (640×480) | 5-20 ms | Depends on JPEG quality |
| Image retrieve | <1 ms | In-memory operation |
| Object publish | <1 ms | JSON serialization |
| Object retrieve | <1 ms | JSON deserialization |
| Text overlay publish | <1 ms | Lightweight operation |

### Throughput

- **Image Streaming**: 30-60 FPS (JPEG, quality=85)
- **Object Publishing**: 1000+ objects/second
- **Text Overlays**: 10000+ operations/second
- **Multi-Consumer**: No significant performance impact

---

## Advanced Usage

### Continuous Image Streaming

```python
import cv2
import threading
from redis_robot_comm import RedisImageStreamer

streamer = RedisImageStreamer()
stop_flag = threading.Event()

def on_frame(image, metadata, image_info):
    print(f"Frame {image_info['width']}×{image_info['height']} received")
    cv2.imshow("Live Stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_flag.set()
        return False
    return True

# Subscriber thread
def subscriber_loop():
    streamer.subscribe_variable_images(lambda img, meta, info: on_frame(img, meta, info))

thread = threading.Thread(target=subscriber_loop, daemon=True)
thread.start()

# Publisher loop
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

### Query Objects in Time Ranges

```python
import time

start_time = time.time() - 10  # Last 10 seconds
end_time = time.time()

objects = broker.get_objects_in_timerange(start_time, end_time)
print(f"Objects in last 10 seconds: {len(objects)}")
```

---

## Project Structure

```
redis_robot_comm/
│
├── redis_robot_comm/
│   ├── __init__.py
│   ├── redis_client.py           # RedisMessageBroker
│   ├── redis_image_streamer.py   # RedisImageStreamer
│   ├── redis_label_manager.py    # RedisLabelManager
│   └── redis_text_overlay.py     # RedisTextOverlayManager (NEW!)
│
├── scripts/
│   ├── visualize_annotated_frames.py
│   ├── record_camera_with_overlays.py  # Enhanced with text overlays
│   └── camera_recorder_audio.py
│
├── docs/
│   ├── README.md                  # Workflow documentation
│   ├── api.md                     # API reference
│   ├── text_overlay_readme.md     # Text overlay guide (NEW!)
│   ├── TESTING.md                 # Testing guide
│   ├── workflow_detector.png
│   └── workflow_streamer.png
│
├── tests/                         # Comprehensive test suite
│   ├── test_redis_robot_comm.py
│   ├── test_redis_robot_comm_extended.py
│   ├── test_redis_label_manager.py
│   └── test_redis_text_overlay.py  # NEW!
│
├── examples/
│   └── main.py
│
├── .github/workflows/             # CI/CD pipelines
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Examples

Run the comprehensive example script:

```bash
# Start Redis
docker run -p 6379:6379 redis:alpine

# Run examples
python examples/main.py
```

The script demonstrates:
- ✅ Connection testing
- ✅ Object publishing and retrieval
- ✅ Image publishing and retrieval
- ✅ Asynchronous image streaming with callbacks
- ✅ Result visualization

---

## Integration Examples

### Object Detection Integration

```python
from redis_robot_comm import RedisMessageBroker
from your_detector import YourDetector

broker = RedisMessageBroker()
detector = YourDetector()

def detect_and_publish(image):
    objects = detector.detect(image)
    broker.publish_objects(
        objects,
        camera_pose={"x": 0.0, "y": 0.0, "z": 0.5}
    )
```

### Video Recording with Text Overlays

```python
from redis_robot_comm import RedisTextOverlayManager

text_mgr = RedisTextOverlayManager()

# MCP Server publishes user tasks
def handle_user_command(command: str):
    text_mgr.publish_user_task(command)
    # Execute command...

# Robot publishes action commentary
def robot_action(action: str):
    text_mgr.publish_robot_speech(
        speech=f"🤖 {action}",
        duration_seconds=4.0
    )
```

---

## API Reference

For detailed API documentation, see **[docs/api.md](docs/api.md)**

### RedisMessageBroker

| Method | Description |
|--------|-------------|
| `publish_objects(objects, camera_pose)` | Publish object list |
| `get_latest_objects(max_age_seconds)` | Retrieve latest objects |
| `subscribe_objects(callback)` | Continuous subscription |

### RedisImageStreamer

| Method | Description |
|--------|-------------|
| `publish_image(image, metadata, compress_jpeg, quality)` | Publish image |
| `get_latest_image()` | Retrieve latest image |
| `subscribe_variable_images(callback)` | Continuous streaming |

### RedisLabelManager

| Method | Description |
|--------|-------------|
| `publish_labels(labels, metadata)` | Publish label list |
| `get_latest_labels(timeout_seconds)` | Retrieve current labels |
| `add_label(new_label)` | Add new label |

### RedisTextOverlayManager (NEW!)

| Method | Description |
|--------|-------------|
| `publish_user_task(task)` | Publish persistent user task |
| `publish_robot_speech(speech, duration)` | Publish timed robot message |
| `publish_system_message(message, duration)` | Publish system message |
| `get_latest_texts(max_age_seconds)` | Retrieve recent texts |
| `subscribe_to_texts(callback)` | Monitor text updates |

---

## Testing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=redis_robot_comm --cov-report=html
```

**Test Coverage:** >90% across all modules

---

## Development

### Code Quality Tools

```bash
# Linting with Ruff
ruff check .

# Formatting with Black
black .

# Type checking with mypy
mypy redis_robot_comm --ignore-missing-imports

# Security scanning with Bandit
bandit -r redis_robot_comm/
```

### Pre-Commit Hooks

```bash
pip install pre-commit
pre-commit install
```

The pre-commit hooks automatically run:
- Ruff (linting)
- Black (formatting)
- Bandit (security checks)
- Trailing whitespace removal
- YAML validation
- Large file checks

---

## CI/CD

Comprehensive GitHub Actions workflows:

- **Tests** - Multi-platform (Ubuntu, Windows, macOS), Python 3.8-3.11
- **Code Quality** - Ruff, Black, mypy, Bandit
- **CodeQL** - Security vulnerability scanning
- **Dependency Review** - Security audit
- **Release** - Automated package building

---

## Error Handling

```python
from redis.exceptions import ConnectionError

try:
    broker = RedisMessageBroker()
    if not broker.test_connection():
        print("❌ No connection to Redis")
except ConnectionError as e:
    print(f"❌ Redis connection error: {e}")
    print("Make sure Redis is running:")
    print("  docker run -p 6379:6379 redis:alpine")
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Related Projects

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Object detection with OwlV2, YOLO-World, YOLOE, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robot control with visual object recognition
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - LLM-based robot control using MCP

---

## Author

**Daniel Gaida**  
Email: daniel.gaida@th-koeln.de  
GitHub: [@dgaida](https://github.com/dgaida)

Project Link: https://github.com/dgaida/redis_robot_comm

---

## Acknowledgments

- [Redis](https://redis.io/) - High-performance in-memory database
- [OpenCV](https://opencv.org/) - Computer vision library
- [Python Redis Client](https://github.com/redis/redis-py) - Python Redis integration
