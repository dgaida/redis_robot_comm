# Redis Robot Communication Documentation

## Overview

This directory contains documentation for the `redis_robot_comm` package, which provides Redis-based communication infrastructure for robotics applications.

## What is Redis?

Redis (Remote Dictionary Server) is an open-source, in-memory data structure store that can be used as a database, cache, message broker, and streaming engine. Redis Streams, used in this package, provide a powerful log-like data structure that enables:

- **Persistent message queues** - Messages are stored and can be read by multiple consumers
- **Pub/Sub patterns** - Efficient publisher-subscriber communication
- **Time-series data** - Automatic timestamping and ordering of entries
- **Low latency** - In-memory operations with sub-millisecond response times
- **Scalability** - Horizontal scaling through clustering

Redis is particularly well-suited for robotics applications where multiple processes need to exchange data in real-time, such as camera images, sensor readings, and object detection results.

## Package Integration

The `redis_robot_comm` package is used as the communication backbone in two major robotics frameworks:

### 1. vision_detect_segment

The vision detection package uses `redis_robot_comm` for streaming camera images and publishing object detection results. The workflow involves:

- **RedisImageStreamer** - Publishes variable-size camera frames with metadata (robot pose, workspace ID, timestamps)
- **RedisMessageBroker** - Publishes detected objects with bounding boxes, labels, confidence scores, and optional segmentation masks

For detailed workflow documentation, see: [vision_detect_segment documentation](https://github.com/dgaida/vision_detect_segment/tree/master/docs)

### 2. robot_environment

The robot environment package uses `redis_robot_comm` to integrate vision-based object detection with robotic manipulation:

- Camera frames are streamed to Redis for processing
- Object detections are consumed from Redis for pick-and-place operations
- Enables decoupled vision processing and robot control

For architecture details, see: [robot_environment documentation](https://github.com/dgaida/robot_environment/tree/master/docs)

## Communication Workflows

### Object Detection Workflow

The following diagram illustrates how object detection results flow through Redis:

![Object Detection Workflow](workflow_detector.png)

**Process:**
1. **Object Detector** detects objects in images (bounding boxes, labels, confidence scores)
2. **Publish Objects** - Detection results are published to Redis using `publish_objects()`
3. **Redis Server** stores the detection stream (`RedisMessageBroker`)
4. **Consumer** retrieves objects using `get_latest_objects()`
5. **Metadata** flows alongside detections (labels, bboxes, confidence, etc.)

**Use Cases:**
- Robot pick-and-place applications
- Real-time object tracking
- Multi-consumer detection pipelines
- Object detection logging and analysis

---

### Image Streaming Workflow

The following diagram shows how camera images are streamed through Redis:

![Image Streaming Workflow](workflow_streamer.png)

**Process:**
1. **Image Source** (camera or file) captures frames
2. **Publish Image** - Images are published to Redis using `publish_image()`
3. **Redis Server** stores the image stream (`RedisImageStreamer`)
4. **Consumer** retrieves latest frame using `get_latest_image()`
5. **Metadata** includes robot pose, workspace information, timestamps

**Features:**
- **Variable-size images** - Automatically handles different resolutions
- **JPEG compression** - Configurable quality for bandwidth optimization
- **Metadata support** - Robot pose, workspace ID, frame numbers
- **Stream management** - Automatic old frame removal (configurable maxlen)

**Use Cases:**
- Real-time camera monitoring
- Vision-based robot control
- Multi-camera systems
- Image processing pipelines

---

## Key Components

### RedisMessageBroker

Handles object detection data streaming.

**Features:**
- Publish detected objects with metadata
- Retrieve latest detections with age filtering
- Query objects in time ranges
- Subscribe to detection updates
- Stream management (clear, info)

**Typical Usage:**
```python
from redis_robot_comm import RedisMessageBroker

broker = RedisMessageBroker()

# Publish detections
objects = [
    {"label": "cube", "confidence": 0.95, "position": {"x": 0.1, "y": 0.2}},
    {"label": "sphere", "confidence": 0.87, "position": {"x": 0.3, "y": 0.1}}
]
broker.publish_objects(objects, camera_pose={"x": 0.0, "y": 0.0, "z": 0.5})

# Retrieve latest
latest_objects = broker.get_latest_objects(max_age_seconds=2.0)
```

---

### RedisImageStreamer

Handles variable-size image streaming with compression.

**Features:**
- JPEG compression with adjustable quality
- Raw image mode (lossless)
- Automatic base64 encoding
- Metadata attachment
- Stream size management

**Typical Usage:**
```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")

# Publish image
image = cv2.imread("frame.jpg")
streamer.publish_image(
    image,
    metadata={"robot": "arm1", "workspace": "A"},
    compress_jpeg=True,
    quality=85
)

# Retrieve latest
result = streamer.get_latest_image()
if result:
    image, metadata = result
    cv2.imshow("Latest Frame", image)
```

---

## Architecture Benefits

Using Redis as a communication layer provides several advantages:

✅ **Decoupling** - Image producers and consumers operate independently  
✅ **Asynchronous** - Non-blocking processing enables real-time operation  
✅ **Persistence** - Redis stores data, enabling replay and debugging  
✅ **Multi-consumer** - Multiple processes can consume the same stream  
✅ **Scalability** - Redis handles high-throughput data streams efficiently  
✅ **Monitoring** - All data flows are visible and inspectable  
✅ **Fault tolerance** - Consumers can reconnect and resume processing

---

## Requirements

- **Redis Server** - Version 5.0+ (for Streams support)
- **Python** - 3.8+
- **Dependencies**:
  - `redis>=4.0.0` - Redis client library
  - `opencv-python>=4.5.0` - Image processing
  - `numpy>=1.20.0` - Array operations

---

## Getting Started

### 1. Start Redis Server

```bash
# Using Docker (recommended)
docker run -p 6379:6379 redis:alpine

# Or install locally
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis
```

### 2. Install Package

```bash
git clone https://github.com/dgaida/redis_robot_comm.git
cd redis_robot_comm
pip install -e .
```

### 3. Run Example

```bash
python main.py
```

---

## Performance

Redis Streams provide excellent performance for robotics applications:

| Operation | Latency | Notes |
|-----------|---------|-------|
| Image publish (640×480 JPEG) | 5-20 ms | Depends on compression quality |
| Image retrieve | <1 ms | Local Redis server |
| Object publish | <1 ms | JSON serialization overhead |
| Object retrieve | <1 ms | Automatic deserialization |

**Throughput:**
- Image streaming: 30-60 FPS (JPEG, quality=85)
- Object publishing: 1000+ detections/second
- Multiple consumers: No significant performance impact

---

## Related Documentation

- [Main Package README](../README.md) - Installation and API reference
- [vision_detect_segment](https://github.com/dgaida/vision_detect_segment) - Object detection integration
- [robot_environment](https://github.com/dgaida/robot_environment) - Robot control integration
- [Redis Streams Documentation](https://redis.io/docs/data-types/streams/)

---

## License

MIT License - see [LICENSE](../LICENSE) file for details.

## Contact

Daniel Gaida - daniel.gaida@th-koeln.de

Project Link: https://github.com/dgaida/redis_robot_comm
