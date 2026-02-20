# Getting Started

This guide will help you get started with `redis_robot_comm` quickly.

## Prerequisites

* **Python** ≥ 3.8
* **Redis Server** ≥ 5.0 (for Streams support)

## Quick Start Examples

### 1. Object Detection

```python
from redis_robot_comm import RedisMessageBroker
import time

broker = RedisMessageBroker()

# Test connection
if broker.test_connection():
    print("✓ Connected to Redis")

# Publish example objects
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

# Retrieve latest objects
latest = broker.get_latest_objects(max_age_seconds=2.0)
print(f"Found objects: {len(latest)}")
```

### 2. Image Streaming

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer()

# Load example image
image = cv2.imread("example.jpg")

# Publish image
streamer.publish_image(image, compress_jpeg=True, quality=85)

# Retrieve latest image
result = streamer.get_latest_image()
if result:
    img, metadata = result
    cv2.imshow("Received Image", img)
    cv2.waitKey(0)
```

## More Information

For detailed information about individual modules, please visit the [Usage](usage/detection.md) and [API Reference](api/broker.md) sections.
