# Redis Robot Communication Documentation

## Overview

This directory contains comprehensive documentation for the `redis_robot_comm` package, which provides Redis-based communication infrastructure for robotics applications.

---

## Documentation Files

- **[api.md](api.md)** - Complete API reference with detailed method documentation
- **[TESTING.md](TESTING.md)** - Testing guidelines, examples, and best practices
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Development setup and contribution guidelines
- **[../README.md](../README.md)** - Main package documentation with quick start guide

---

## What is Redis?

Redis (Remote Dictionary Server) is an open-source, in-memory data structure store that can be used as a database, cache, message broker, and streaming engine. Redis Streams, used in this package, provide a powerful log-like data structure that enables:

- **Persistent message queues** - Messages are stored and can be read by multiple consumers
- **Pub/Sub patterns** - Efficient publisher-subscriber communication
- **Time-series data** - Automatic timestamping and ordering of entries
- **Low latency** - In-memory operations with sub-millisecond response times
- **Scalability** - Horizontal scaling through clustering
- **Reliability** - Durability and atomicity guarantees for each entry

Redis is particularly well-suited for robotics applications where multiple processes need to exchange data in real-time, such as camera images, sensor readings, and object detection results.

### Why Redis Streams for Robotics?

1. **Variable-size payloads** - Perfect for images that change resolution or format
2. **Back-pressure handling** - Built-in queue management with `maxlen` parameter
3. **Low latency** - Single operations complete in <1ms on local servers
4. **Robustness** - Guaranteed durability and atomicity for critical robotics data
5. **Multi-consumer support** - Multiple processes can consume the same stream independently

---

## Package Integration

The `redis_robot_comm` package is used as the communication backbone in two major robotics frameworks:

### 1. vision_detect_segment

The vision detection package uses `redis_robot_comm` for streaming camera images and publishing object detection results. The workflow involves:

- **RedisImageStreamer** - Publishes variable-size camera frames with metadata (robot pose, workspace ID, timestamps)
- **RedisMessageBroker** - Publishes detected objects with bounding boxes, labels, confidence scores, and optional segmentation masks
- **RedisLabelManager** - Manages dynamic label configuration for different detection models

**Key Features:**
- Real-time object detection streaming
- Support for multiple detection models (OwlV2, YOLO-World, Grounding-DINO)
- Annotated frame publishing for visualization
- Dynamic label management

For detailed workflow documentation, see: [vision_detect_segment documentation](https://github.com/dgaida/vision_detect_segment/tree/master/docs)

### 2. robot_environment

The robot environment package uses `redis_robot_comm` to integrate vision-based object detection with robotic manipulation:

- Camera frames are streamed to Redis for processing
- Object detections are consumed from Redis for pick-and-place operations
- Enables decoupled vision processing and robot control
- Supports multiple workspaces and robot arms

**Key Features:**
- Vision-guided manipulation
- Multi-robot coordination
- Workspace management
- Real-time feedback loops

For architecture details, see: [robot_environment documentation](https://github.com/dgaida/robot_environment/tree/master/docs)

### 3. robot_mcp

The Model Context Protocol (MCP) server uses `redis_robot_comm` for LLM-based robot control:

- Provides high-level interface for LLMs to control robots
- Uses Redis streams for command and status communication
- Integrates with vision system for object recognition
- Enables natural language robot programming

For MCP integration, see: [robot_mcp documentation](https://github.com/dgaida/robot_mcp)

---

## Communication Workflows

### Object Detection Workflow

The following diagram illustrates how object detection results flow through Redis:

![Object Detection Workflow](workflow_detector.png)

**Process:**
1. **Object Detector** detects objects in images (bounding boxes, labels, confidence scores)
2. **Publish Objects** - Detection results are published to Redis using `publish_objects()`
3. **Redis Server** stores the detection stream (`RedisMessageBroker`)
4. **Consumer** retrieves objects using `get_latest_objects()`
5. **Metadata** flows alongside detections (labels, bboxes, confidence, camera pose)

**Use Cases:**
- Robot pick-and-place applications
- Real-time object tracking
- Multi-consumer detection pipelines
- Object detection logging and analysis
- Quality control and inspection
- Warehouse automation

**Example Implementation:**

```python
from redis_robot_comm import RedisMessageBroker
import time

broker = RedisMessageBroker()

# Publisher: Vision system
def publish_detections(image, detector):
    objects = detector.detect(image)
    camera_pose = {"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 1.57, "yaw": 0.0}
    broker.publish_objects(objects, camera_pose)

# Consumer: Robot controller
def get_target_object(target_class):
    objects = broker.get_latest_objects(max_age_seconds=2.0)
    for obj in objects:
        if obj["class_name"] == target_class:
            return obj
    return None
```

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
- **Multi-format support** - JPEG compressed or raw lossless

**Use Cases:**
- Real-time camera monitoring
- Vision-based robot control
- Multi-camera systems
- Image processing pipelines
- Remote operation interfaces
- Quality assurance recording

**Example Implementation:**

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")

# Publisher: Camera capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        streamer.publish_image(
            frame,
            metadata={"robot": "arm1", "workspace": "A"},
            compress_jpeg=True,
            quality=85
        )

# Consumer: Image processing
result = streamer.get_latest_image()
if result:
    image, metadata = result
    # Process image...
```

---

### Label Management Workflow

Dynamic label configuration enables runtime updates to detectable object classes:

**Process:**
1. **Vision System** publishes available labels when model changes
2. **RedisLabelManager** stores current label configuration
3. **Robot Controller** queries available labels for task planning
4. **Subscribers** receive updates when labels change

**Example Implementation:**

```python
from redis_robot_comm import RedisLabelManager

manager = RedisLabelManager()

# Publisher: Vision system
labels = ["cube", "sphere", "cylinder"]
manager.publish_labels(labels, metadata={"model_id": "yolo-v8"})

# Consumer: Robot controller
available_labels = manager.get_latest_labels()
print(f"Can detect: {available_labels}")

# Add new label dynamically
manager.add_label("prism")

# Subscribe to updates
def on_label_update(labels, metadata):
    print(f"Updated labels: {labels}")

manager.subscribe_to_label_updates(on_label_update)
```

---

## Key Components

### RedisMessageBroker

Handles object detection data streaming with support for:

- Publishing detected objects with confidence scores and positions
- Retrieving latest detections with age filtering
- Querying objects in specific time ranges
- Subscribing to real-time detection updates
- Stream management and statistics

**Typical Use Case:** Connecting vision systems to robot controllers

**Performance:** 1000+ objects/second, <1ms latency

**Documentation:** [API Reference - RedisMessageBroker](api.md#redismessagebroker)

---

### RedisImageStreamer

Handles variable-size image streaming with features like:

- JPEG compression with adjustable quality
- Raw image mode for lossless transfer
- Automatic base64 encoding/decoding
- Rich metadata attachment
- Stream size management with automatic cleanup

**Typical Use Case:** Camera feed distribution to multiple consumers

**Performance:** 30-60 FPS with JPEG compression, 5-20ms latency

**Documentation:** [API Reference - RedisImageStreamer](api.md#redisimagestreamer)

---

### RedisLabelManager

Manages detectable object labels with:

- Dynamic label publishing and updates
- Label retrieval with timeout handling
- Adding new labels to existing lists
- Real-time label update notifications
- Case-insensitive label handling

**Typical Use Case:** Runtime configuration of detection models

**Performance:** <1ms for label operations

**Documentation:** [API Reference - RedisLabelManager](api.md#redislabelmanager)

---

## Architecture Benefits

Using Redis as a communication layer provides several advantages:

✅ **Decoupling** - Producers and consumers operate independently  
✅ **Asynchronous** - Non-blocking processing enables real-time operation  
✅ **Persistence** - Redis stores data, enabling replay and debugging  
✅ **Multi-consumer** - Multiple processes can consume the same stream  
✅ **Scalability** - Redis handles high-throughput data streams efficiently  
✅ **Monitoring** - All data flows are visible and inspectable  
✅ **Fault tolerance** - Consumers can reconnect and resume processing  
✅ **Language agnostic** - Any Redis client can integrate with the system  
✅ **Battle-tested** - Redis is proven in production environments

### Comparison with Alternatives

| Feature | Redis Streams | ROS Topics | ZeroMQ | MQTT |
|---------|--------------|------------|--------|------|
| Persistence | ✓ | ✗ | ✗ | △ |
| Multi-consumer | ✓ | ✓ | △ | ✓ |
| Time-series | ✓ | ✗ | ✗ | ✗ |
| Replay capability | ✓ | △ | ✗ | ✗ |
| Language support | ✓✓ | △ | ✓ | ✓ |
| Learning curve | Low | High | Medium | Low |

---

## Performance

Redis Streams provide excellent performance for robotics applications:

### Latency Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Image publish (640×480 JPEG) | 5-20 ms | Depends on compression quality |
| Image retrieve | <1 ms | Local Redis server |
| Object publish | <1 ms | JSON serialization overhead |
| Object retrieve | <1 ms | Automatic deserialization |
| Label publish | <1 ms | Minimal overhead |
| Label retrieve | <1 ms | In-memory operation |

### Throughput Measurements

- **Image streaming**: 30-60 FPS (JPEG, quality=85)
- **Object publishing**: 1000+ detections/second
- **Label updates**: 10000+ operations/second
- **Multiple consumers**: No significant performance impact

### Memory Usage

- **Per image entry**: ~50KB (640×480 JPEG at quality=85)
- **Per object entry**: ~1-2KB (depends on metadata)
- **Per label entry**: <1KB
- **Stream overhead**: Minimal (<1% of data size)

### Optimization Strategies

1. **Adjust JPEG quality** for bandwidth vs. quality tradeoff
2. **Use maxlen parameter** to limit memory usage
3. **Batch operations** when possible
4. **Use local Redis** for lowest latency
5. **Enable compression** for large images

---

## Utility Scripts

The package includes utility scripts for visualization and recording:

### Visualize Annotated Frames

Real-time visualization of detection results:

```bash
python scripts/visualize_annotated_frames.py --stream-name annotated_camera
```

**Features:**
- Real-time FPS display
- Pause/resume functionality
- Screenshot capture
- Detection statistics overlay

**Controls:**
- `q/ESC` - Quit
- `s` - Save screenshot
- `p` - Pause/unpause
- `f` - Toggle FPS display
- `h` - Show help

### Record Camera with Annotations

Record camera feed alongside detection results:

```bash
python scripts/record_camera_script.py --camera 0 --stream annotated_camera
```

**Features:**
- Side-by-side or overlay layout
- Real-time recording statistics
- Screenshot capability
- Multiple codec support

**Controls:**
- `q/ESC` - Stop recording
- `p` - Pause/unpause recording
- `s` - Take screenshot

---

## Testing

The package includes comprehensive test coverage. See [TESTING.md](TESTING.md) for detailed testing guidelines.

**Run Tests:**
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=redis_robot_comm --cov-report=html

# Specific test file
pytest tests/test_redis_robot_comm.py -v
```

**Test Coverage:** >90% code coverage across all modules

---

## Troubleshooting

### Common Issues

**Cannot connect to Redis:**
```bash
# Check if Redis is running
redis-cli ping

# Expected output: PONG

# If not running, start Redis
docker run -p 6379:6379 redis:alpine
```

**Images not appearing:**
- Check stream name matches between publisher and consumer
- Verify Redis connection with `test_connection()`
- Enable verbose mode: `streamer.verbose = True`
- Check Redis memory limits: `redis-cli INFO memory`

**Old data appearing:**
- Use `clear_stream()` to reset streams
- Adjust `max_age_seconds` parameter
- Check system clock synchronization

**Performance issues:**
- Use local Redis server when possible
- Adjust JPEG quality settings
- Implement proper stream cleanup with `maxlen`
- Monitor Redis memory usage

---

## Best Practices

### Stream Naming

Use descriptive, hierarchical stream names:
```python
# Good
"camera_workspace_A"
"detection_yolo_v8"
"labels_owlv2"

# Avoid
"stream1"
"data"
"test"
```

### Error Handling

Always handle connection errors:
```python
from redis.exceptions import ConnectionError

try:
    broker = RedisMessageBroker()
    if not broker.test_connection():
        raise ConnectionError("Cannot connect to Redis")
except ConnectionError as e:
    print(f"Error: {e}")
    # Implement fallback or retry logic
```

### Resource Management

Clean up streams periodically:
```python
# Limit stream size
streamer.publish_image(image, maxlen=10)

# Clear old data
broker.clear_stream()

# Monitor stream size
stats = streamer.get_stream_stats()
if stats['total_messages'] > 1000:
    streamer.clear_stream()
```

### Thread Safety

Use threading for blocking operations:
```python
import threading

def subscriber_thread():
    broker.subscribe_objects(callback)

thread = threading.Thread(target=subscriber_thread, daemon=True)
thread.start()
```

---

## Related Documentation

- **[API Reference](api.md)** - Complete API documentation
- **[Testing Guide](TESTING.md)** - Testing guidelines and examples
- **[Contributing Guide](../CONTRIBUTING.md)** - Development setup and contribution guidelines
- **[Main README](../README.md)** - Installation and quick start

## External Resources

- **[Redis Documentation](https://redis.io/docs/)** - Official Redis documentation
- **[Redis Streams](https://redis.io/docs/data-types/streams/)** - Redis Streams guide
- **[OpenCV Documentation](https://docs.opencv.org/)** - Image processing reference
- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Object detection integration
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robot control integration
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - MCP server integration

---

## License

This project is licensed under the **MIT License**. See [../LICENSE](../LICENSE) for details.

## Contact

**Daniel Gaida**  
Email: daniel.gaida@th-koeln.de  
GitHub: [@dgaida](https://github.com/dgaida)

Project Link: https://github.com/dgaida/redis_robot_comm

---

## Acknowledgments

- [Redis](https://redis.io/) - For the high-performance in-memory database
- [OpenCV](https://opencv.org/) - For computer vision capabilities
- [Python Redis Client](https://github.com/redis/redis-py) - For Python Redis integration
