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

`redis_robot_comm` provides a high-performance Redis-based communication infrastructure for robotics applications. It enables real-time exchange of camera images, object detections, and metadata between distributed processes with sub-millisecond latency.

### Key Features

* 🎯 **Object Detection Streaming** - Publish and consume detection results with `RedisMessageBroker`
* 📷 **Variable-Size Image Streaming** - JPEG-compressed or raw image transfer with `RedisImageStreamer`
* 🏷️ **Label Management** - Dynamic object label configuration with `RedisLabelManager`
* ⚡ **Real-Time Performance** - Sub-millisecond latency for local Redis servers
* 🔄 **Asynchronous Architecture** - Decoupled producer-consumer patterns
* 📊 **Rich Metadata Support** - Timestamps, camera poses, workspace information
* 🎯 **Robotics-Optimized** - Designed for pick-and-place and computer vision tasks

---

## Use Cases

This package serves as the communication backbone for two major robotics frameworks:

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Object detection with OwlV2, YOLO-World, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robot control with visual object recognition

For detailed workflow documentation, see **[docs/README.md](docs/README.md)**

### Data Flow via Redis Streams in the mentioned repositories

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

# Clear stream for testing
broker.clear_stream()

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

**Core Methods:**
- `publish_objects()` - Publish objects with metadata
- `get_latest_objects()` - Retrieve latest objects with age filter
- `get_objects_in_timerange()` - Query objects by time range
- `subscribe_objects()` - Continuous subscription (blocking)
- `clear_stream()` - Reset stream
- `get_stream_info()` - Stream statistics

---

### 2. Image Streaming with RedisImageStreamer

![Image Streaming Workflow](docs/workflow_streamer.png)

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")

# Load example image
image = cv2.imread("example.jpg")

# Publish with metadata
stream_id = streamer.publish_image(
    image,
    metadata={
        "robot": "arm1",
        "workspace": "A",
        "frame_id": 42
    },
    compress_jpeg=True,
    quality=85,
    maxlen=5  # Keep only last 5 frames
)

print(f"Image published: {stream_id}")

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

# Add new label
label_mgr.add_label("prism")

# Subscribe to label updates
def on_label_update(labels, metadata):
    print(f"Labels updated: {labels}")

label_mgr.subscribe_to_label_updates(on_label_update)
```

**Core Methods:**
- `publish_labels()` - Publish list of detectable labels
- `get_latest_labels()` - Retrieve current labels
- `add_label()` - Add new label to existing list
- `subscribe_to_label_updates()` - Monitor label changes
- `clear_stream()` - Reset label stream

---

### 4. Continuous Streaming

```python
import cv2
import threading
from redis_robot_comm import RedisImageStreamer

streamer = RedisImageStreamer()
stop_flag = threading.Event()

# Callback for received images
def on_frame(image, metadata, image_info):
    print(f"Frame {image_info['width']}×{image_info['height']} received")
    cv2.imshow("Live Stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_flag.set()
        return False
    return True

# Subscriber thread
def subscriber_loop():
    def callback(img, meta, info):
        if not on_frame(img, meta, info):
            stop_flag.set()

    streamer.subscribe_variable_images(callback, block_ms=500)

thread = threading.Thread(target=subscriber_loop, daemon=True)
thread.start()

# Publisher loop
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

## Utility Scripts

The package includes utility scripts for visualization and recording:

### Visualize Annotated Frames

```bash
python scripts/visualize_annotated_frames.py --stream-name annotated_camera
```

**Controls:**
- `q/ESC` - Quit
- `s` - Save screenshot
- `p` - Pause/unpause
- `f` - Toggle FPS display

### Record Camera with Annotations

```bash
python scripts/record_camera_script.py --camera 0 --stream annotated_camera
```

**Controls:**
- `q/ESC` - Stop recording
- `p` - Pause/unpause recording
- `s` - Take screenshot

---

## Performance

### Latency (Local Redis Server)

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Image publish (640×480) | 5-20 ms | Depends on JPEG quality |
| Image retrieve | <1 ms | In-memory operation |
| Object publish | <1 ms | JSON serialization |
| Object retrieve | <1 ms | JSON deserialization |

### Throughput

- **Image Streaming**: 30-60 FPS (JPEG, quality=85)
- **Object Publishing**: 1000+ objects/second
- **Multi-Consumer**: No significant performance impact

### Optimization Options

```python
# High compression (faster, smaller)
streamer.publish_image(image, compress_jpeg=True, quality=70)

# Low compression (slower, better quality)
streamer.publish_image(image, compress_jpeg=True, quality=95)

# No compression (slowest, lossless)
streamer.publish_image(image, compress_jpeg=False)

# Limit stream size
streamer.publish_image(image, maxlen=5)  # Keep only last 5 frames
```

---

## Advanced Usage

### Query Objects in Time Ranges

```python
import time

start_time = time.time() - 10  # Last 10 seconds
end_time = time.time()

objects = broker.get_objects_in_timerange(start_time, end_time)
print(f"Objects in last 10 seconds: {len(objects)}")
```

### Asynchronous Object Subscription

```python
def on_detection(data):
    objects = data["objects"]
    camera_pose = data["camera_pose"]
    timestamp = data["timestamp"]

    print(f"Received: {len(objects)} objects")
    for obj in objects:
        print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")

try:
    broker.subscribe_objects(on_detection)  # Blocking
except KeyboardInterrupt:
    print("Subscription ended")
```

### Stream Statistics

```python
# RedisMessageBroker
info = broker.get_stream_info()
print(f"Stream length: {info['length']}")
print(f"First entry: {info['first-entry']}")
print(f"Last entry: {info['last-entry']}")

# RedisImageStreamer
stats = streamer.get_stream_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"First frame ID: {stats['first_entry_id']}")
print(f"Last frame ID: {stats['last_entry_id']}")
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
│   └── redis_label_manager.py    # RedisLabelManager
│
├── scripts/
│   ├── visualize_annotated_frames.py
│   └── record_camera_script.py
│
├── docs/
│   ├── README.md                  # Workflow documentation
│   ├── api.md                     # API reference
│   ├── workflow_detector.png
│   └── workflow_streamer.png
│
├── tests/
│   ├── test_redis_robot_comm.py
│   ├── test_redis_robot_comm_extended.py
│   └── test_redis_label_manager.py
│
├── examples/
│   └── main.py                    # Example script
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
    print(f"Published: {len(objects)} objects")
```

### Robot Control Integration

```python
from redis_robot_comm import RedisMessageBroker
from your_robot import YourRobot

broker = RedisMessageBroker()
robot = YourRobot()

def pick_object(label):
    objects = broker.get_latest_objects()
    for obj in objects:
        if obj["class_name"] == label:
            position = obj["position"]
            robot.pick(position["x"], position["y"], position["z"])
            return True
    return False

success = pick_object("cube")
print(f"Object picked: {success}")
```

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

## Testing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=redis_robot_comm --cov-report=html

# View coverage report
open htmlcov/index.html
```

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

The project includes comprehensive GitHub Actions workflows:

- **Tests** - Multi-platform testing (Ubuntu, Windows, macOS) across Python 3.8-3.11
- **Code Quality** - Ruff, Black, mypy, Bandit
- **CodeQL** - Security vulnerability scanning
- **Dependency Review** - Security audit for dependencies
- **Release** - Automated package building and GitHub releases

---

## API Reference

For detailed API documentation, see **[docs/api.md](docs/api.md)**

### RedisMessageBroker

| Method | Description | Returns |
|--------|-------------|---------|
| `publish_objects(objects, camera_pose)` | Publish object list | Stream ID or None |
| `get_latest_objects(max_age_seconds)` | Retrieve latest objects | List of objects |
| `get_objects_in_timerange(start, end)` | Query objects by time | List of objects |
| `subscribe_objects(callback)` | Continuous subscription | - (blocking) |
| `clear_stream()` | Clear stream | Number deleted |
| `get_stream_info()` | Stream statistics | Dict |
| `test_connection()` | Connection test | True/False |

### RedisImageStreamer

| Method | Description | Returns |
|--------|-------------|---------|
| `publish_image(image, metadata, compress_jpeg, quality, maxlen)` | Publish image | Stream ID |
| `get_latest_image(timeout_ms)` | Retrieve latest image | (image, metadata) or None |
| `subscribe_variable_images(callback, block_ms)` | Continuous streaming | - (blocking) |
| `get_stream_stats()` | Stream statistics | Dict |

### RedisLabelManager

| Method | Description | Returns |
|--------|-------------|---------|
| `publish_labels(labels, metadata)` | Publish label list | Stream ID |
| `get_latest_labels(timeout_seconds)` | Retrieve current labels | List of labels |
| `add_label(new_label)` | Add new label | True/False |
| `subscribe_to_label_updates(callback)` | Monitor label changes | - (blocking) |
| `clear_stream()` | Clear label stream | True/False |

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Related Projects

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Object detection with OwlV2, YOLO-World, Grounding-DINO
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robot control with visual object recognition
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - LLM-based robot control using Model Context Protocol (MCP)

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
