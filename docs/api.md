# API Reference

Complete API documentation for the `redis_robot_comm` package.

---

## Table of Contents

- [RedisMessageBroker](#redismessagebroker)
- [RedisImageStreamer](#redisimagestreamer)
- [RedisLabelManager](#redislabelmanager)
- [Data Formats](#data-formats)
- [Error Handling](#error-handling)

---

## RedisMessageBroker

Redis-based message broker for publishing and consuming object detection results.

### Constructor

```python
RedisMessageBroker(host="localhost", port=6379, db=0)
```

**Parameters:**
- `host` (str): Redis server hostname or IP address. Default: `"localhost"`
- `port` (int): Redis server port. Default: `6379`
- `db` (int): Redis database index. Default: `0`

**Attributes:**
- `verbose` (bool): Enable verbose logging output. Default: `False`
- `client` (redis.Redis): Redis client instance

**Example:**
```python
from redis_robot_comm import RedisMessageBroker

# Default connection
broker = RedisMessageBroker()

# Custom connection
broker = RedisMessageBroker(host="192.168.1.100", port=6380, db=1)
broker.verbose = True  # Enable verbose logging
```

---

### Methods

#### `publish_objects(objects, camera_pose=None)`

Publish a list of detected objects to the Redis stream.

**Parameters:**
- `objects` (List[Dict]): List of detected objects (see [Object Format](#object-format))
- `camera_pose` (dict, optional): Camera pose information (see [Camera Pose Format](#camera-pose-format))

**Returns:**
- `str | None`: Redis stream entry ID (e.g., `"1589456675-0"`) on success, `None` on error

**Example:**
```python
objects = [
    {
        "id": "obj_001",
        "class_name": "cube",
        "confidence": 0.95,
        "position": {"x": 0.1, "y": 0.2, "z": 0.05},
        "timestamp": time.time()
    }
]

camera_pose = {
    "x": 0.0, "y": 0.0, "z": 0.5,
    "roll": 0.0, "pitch": 1.57, "yaw": 0.0
}

stream_id = broker.publish_objects(objects, camera_pose)
print(f"Published to: {stream_id}")
```

---

#### `get_latest_objects(max_age_seconds=2.0)`

Retrieve the most recent object detections from the stream.

**Parameters:**
- `max_age_seconds` (float): Maximum age of objects in seconds. Default: `2.0`

**Returns:**
- `List[Dict]`: List of detected objects, or empty list if none found or too old

**Example:**
```python
# Get objects from last 2 seconds
objects = broker.get_latest_objects()

# Get objects from last 5 seconds
objects = broker.get_latest_objects(max_age_seconds=5.0)

for obj in objects:
    print(f"{obj['class_name']}: {obj['confidence']:.2f}")
```

---

#### `get_objects_in_timerange(start_timestamp, end_timestamp=None)`

Query objects detected within a specific time range.

**Parameters:**
- `start_timestamp` (float): Start time as Unix timestamp
- `end_timestamp` (float, optional): End time as Unix timestamp. Default: current time

**Returns:**
- `List[Dict]`: List of all objects detected in the time range

**Example:**
```python
import time

# Get objects from last 10 seconds
start = time.time() - 10
end = time.time()
objects = broker.get_objects_in_timerange(start, end)

print(f"Found {len(objects)} objects in timerange")
```

---

#### `subscribe_objects(callback)`

Subscribe to object detection updates with a callback function (blocking).

**Parameters:**
- `callback` (Callable): Function that receives detection data as a dictionary

**Callback Signature:**
```python
def callback(data: Dict[str, Any]) -> None:
    """
    Args:
        data: Dictionary containing:
            - objects (List[Dict]): List of detected objects
            - camera_pose (dict): Camera pose information
            - timestamp (float): Detection timestamp
    """
    pass
```

**Example:**
```python
def on_detection(data):
    objects = data["objects"]
    camera_pose = data["camera_pose"]
    timestamp = data["timestamp"]

    print(f"[{timestamp}] Detected {len(objects)} objects")
    for obj in objects:
        print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")

try:
    broker.subscribe_objects(on_detection)  # Blocking
except KeyboardInterrupt:
    print("Subscription stopped")
```

---

#### `clear_stream()`

Delete all entries from the detection stream.

**Returns:**
- `int | bool`: Number of deleted entries, or `False` on error

**Example:**
```python
deleted = broker.clear_stream()
print(f"Cleared {deleted} entries")
```

---

#### `get_stream_info()`

Retrieve metadata about the detection stream.

**Returns:**
- `dict | None`: Stream information dictionary, or `None` on error

**Dictionary Keys:**
- `length` (int): Number of entries in stream
- `first-entry` (tuple): Oldest entry (ID, fields)
- `last-entry` (tuple): Newest entry (ID, fields)
- Additional Redis stream metadata

**Example:**
```python
info = broker.get_stream_info()
if info:
    print(f"Stream length: {info['length']}")
    print(f"First entry: {info['first-entry']}")
    print(f"Last entry: {info['last-entry']}")
```

---

#### `test_connection()`

Test the connection to the Redis server.

**Returns:**
- `bool`: `True` if connection successful, `False` otherwise

**Example:**
```python
if broker.test_connection():
    print("✓ Connected to Redis")
else:
    print("✗ Connection failed")
```

---

## RedisImageStreamer

Redis-based image streaming with support for variable-size images and JPEG compression.

### Constructor

```python
RedisImageStreamer(host="localhost", port=6379, stream_name="robot_camera")
```

**Parameters:**
- `host` (str): Redis server hostname. Default: `"localhost"`
- `port` (int): Redis server port. Default: `6379`
- `stream_name` (str): Name of the Redis stream. Default: `"robot_camera"`

**Attributes:**
- `verbose` (bool): Enable verbose logging. Default: `False`
- `client` (redis.Redis): Redis client instance
- `stream_name` (str): Stream name for images

**Example:**
```python
from redis_robot_comm import RedisImageStreamer

# Default stream
streamer = RedisImageStreamer()

# Custom stream
streamer = RedisImageStreamer(
    host="192.168.1.100",
    port=6380,
    stream_name="camera_workspace_A"
)
```

---

### Methods

#### `publish_image(image, metadata=None, compress_jpeg=True, quality=80, maxlen=5)`

Publish an image frame to the Redis stream.

**Parameters:**
- `image` (np.ndarray): OpenCV image array (H×W×C or H×W for grayscale)
- `metadata` (dict, optional): Custom metadata to attach
- `compress_jpeg` (bool): Enable JPEG compression. Default: `True`
- `quality` (int): JPEG quality (1-100). Default: `80`
- `maxlen` (int): Maximum stream entries to keep. Default: `5`

**Returns:**
- `str`: Redis stream entry ID

**Raises:**
- `ValueError`: If image is not a valid NumPy array or is empty

**Example:**
```python
import cv2

# Load image
image = cv2.imread("frame.jpg")

# Publish with compression
stream_id = streamer.publish_image(
    image,
    metadata={"robot": "arm1", "workspace": "A", "frame_id": 42},
    compress_jpeg=True,
    quality=85,
    maxlen=10
)

# Publish without compression (lossless)
stream_id = streamer.publish_image(
    image,
    compress_jpeg=False
)
```

---

#### `get_latest_image(timeout_ms=1000)`

Retrieve the newest image from the stream.

**Parameters:**
- `timeout_ms` (int): Timeout in milliseconds (not actively used in current implementation)

**Returns:**
- `tuple | None`: `(image, metadata)` if successful, `None` if no image available
  - `image` (np.ndarray): Decoded image array
  - `metadata` (dict): Attached metadata

**Example:**
```python
result = streamer.get_latest_image()

if result:
    image, metadata = result
    print(f"Image shape: {image.shape}")
    print(f"Metadata: {metadata}")

    cv2.imshow("Latest Frame", image)
    cv2.waitKey(0)
else:
    print("No image available")
```

---

#### `subscribe_variable_images(callback, block_ms=1000, start_after="$")`

Continuously subscribe to image stream and invoke callback for each frame (blocking).

**Parameters:**
- `callback` (Callable): Function to process each image
- `block_ms` (int): Redis blocking timeout in milliseconds. Default: `1000`
- `start_after` (str): Redis stream ID to start reading after. Default: `"$"` (only new frames)

**Callback Signature:**
```python
def callback(image: np.ndarray, metadata: dict, image_info: dict) -> None:
    """
    Args:
        image: Decoded image array
        metadata: Custom metadata dictionary
        image_info: Dictionary with:
            - width (int): Image width
            - height (int): Image height
            - channels (int): Number of channels
            - timestamp (float): Publication timestamp
            - compressed_size (int): Compressed size in bytes
            - original_size (int): Original size in bytes
    """
    pass
```

**Example:**
```python
def on_frame(image, metadata, image_info):
    print(f"Frame {image_info['width']}×{image_info['height']} @ {image_info['timestamp']}")
    cv2.imshow("Live Stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

try:
    streamer.subscribe_variable_images(on_frame, block_ms=500)
except KeyboardInterrupt:
    print("Streaming stopped")
finally:
    cv2.destroyAllWindows()
```

---

#### `get_stream_stats()`

Retrieve statistics about the image stream.

**Returns:**
- `dict`: Dictionary containing:
  - `total_messages` (int): Number of entries in stream
  - `first_entry_id` (str | None): ID of oldest entry
  - `last_entry_id` (str | None): ID of newest entry
  - `error` (str): Error message if stream not found

**Example:**
```python
stats = streamer.get_stream_stats()
print(f"Total frames: {stats['total_messages']}")
print(f"First frame ID: {stats['first_entry_id']}")
print(f"Last frame ID: {stats['last_entry_id']}")
```

---

## RedisLabelManager

Manages detectable object labels via Redis streams for dynamic label configuration.

### Constructor

```python
RedisLabelManager(host="localhost", port=6379, stream_name="detectable_labels")
```

**Parameters:**
- `host` (str): Redis server hostname. Default: `"localhost"`
- `port` (int): Redis server port. Default: `6379`
- `stream_name` (str): Stream name for labels. Default: `"detectable_labels"`

**Attributes:**
- `verbose` (bool): Enable verbose logging. Default: `False`
- `client` (redis.Redis): Redis client instance
- `stream_name` (str): Stream name for labels

**Example:**
```python
from redis_robot_comm import RedisLabelManager

manager = RedisLabelManager()
manager.verbose = True
```

---

### Methods

#### `publish_labels(labels, metadata=None)`

Publish the current list of detectable object labels.

**Parameters:**
- `labels` (List[str]): List of object label strings
- `metadata` (dict, optional): Optional metadata (e.g., model ID, version)

**Returns:**
- `str | None`: Redis stream entry ID on success, `None` on error

**Example:**
```python
labels = ["cube", "sphere", "cylinder", "prism"]
metadata = {"model_id": "yolo-v8", "version": "1.0.0"}

stream_id = manager.publish_labels(labels, metadata=metadata)
print(f"Published {len(labels)} labels: {stream_id}")
```

---

#### `get_latest_labels(timeout_seconds=5.0)`

Retrieve the most recent list of detectable labels.

**Parameters:**
- `timeout_seconds` (float): Maximum age of labels to accept. Default: `5.0`

**Returns:**
- `List[str] | None`: List of label strings, or `None` if not available or too old

**Example:**
```python
labels = manager.get_latest_labels(timeout_seconds=10.0)

if labels:
    print(f"Detectable objects: {', '.join(labels)}")
else:
    print("No labels available")
```

---

#### `add_label(new_label)`

Add a new label to the current list and republish.

**Parameters:**
- `new_label` (str): Label to add (case-insensitive)

**Returns:**
- `bool`: `True` if label was added, `False` if it already exists or on error

**Example:**
```python
# Add new label
success = manager.add_label("hexagon")
if success:
    print("Label added successfully")
else:
    print("Label already exists or error occurred")
```

---

#### `subscribe_to_label_updates(callback, block_ms=1000)`

Subscribe to label updates and invoke callback when labels change (blocking).

**Parameters:**
- `callback` (Callable): Function to process label updates
- `block_ms` (int): Redis blocking timeout. Default: `1000`

**Callback Signature:**
```python
def callback(labels: List[str], metadata: dict) -> None:
    """
    Args:
        labels: Updated list of labels
        metadata: Associated metadata
    """
    pass
```

**Example:**
```python
def on_label_update(labels, metadata):
    print(f"Labels updated: {labels}")
    print(f"Metadata: {metadata}")

try:
    manager.subscribe_to_label_updates(on_label_update)
except KeyboardInterrupt:
    print("Subscription stopped")
```

---

#### `clear_stream()`

Clear the labels stream.

**Returns:**
- `bool`: `True` if successful, `False` on error

**Example:**
```python
if manager.clear_stream():
    print("Labels stream cleared")
```

---

## Data Formats

### Object Format

Standard format for detected objects:

```python
{
    "id": str,              # Unique object identifier
    "class_name": str,      # Object class/label
    "confidence": float,    # Detection confidence (0.0-1.0)
    "position": {           # 3D position in workspace
        "x": float,
        "y": float,
        "z": float
    },
    "timestamp": float,     # Unix timestamp
    "bbox": [               # Optional: 2D bounding box [x, y, w, h]
        float, float, float, float
    ],
    "mask": List[List],     # Optional: Segmentation mask
    # ... additional custom fields
}
```

### Camera Pose Format

Standard format for camera pose:

```python
{
    "x": float,      # X position (meters)
    "y": float,      # Y position (meters)
    "z": float,      # Z position (meters)
    "roll": float,   # Roll angle (radians)
    "pitch": float,  # Pitch angle (radians)
    "yaw": float     # Yaw angle (radians)
}
```

---

## Error Handling

All methods handle Redis exceptions gracefully and return appropriate error values (`None`, `False`, empty lists).

### Connection Errors

```python
from redis.exceptions import ConnectionError

try:
    broker = RedisMessageBroker()
    if not broker.test_connection():
        print("Cannot connect to Redis")
except ConnectionError as e:
    print(f"Redis connection error: {e}")
```

### Verbose Mode

Enable verbose logging to see detailed error messages:

```python
broker = RedisMessageBroker()
broker.verbose = True

streamer = RedisImageStreamer()
streamer.verbose = True

manager = RedisLabelManager()
manager.verbose = True
```

### Common Error Scenarios

**Empty Streams:**
```python
objects = broker.get_latest_objects()
if not objects:
    print("No objects available")
```

**Stale Data:**
```python
objects = broker.get_latest_objects(max_age_seconds=1.0)
if not objects:
    print("No recent detections (last 1 second)")
```

**Invalid Image:**
```python
try:
    streamer.publish_image(invalid_image)
except ValueError as e:
    print(f"Invalid image: {e}")
```

---

## Performance Notes

### Latency

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Object publish | <1 ms | JSON serialization |
| Object retrieve | <1 ms | In-memory operation |
| Image publish (640×480 JPEG) | 5-20 ms | Depends on quality |
| Image retrieve | <1 ms | Base64 decode + JPEG decompress |

### Throughput

- **Object Publishing**: 1000+ objects/second
- **Image Streaming**: 30-60 FPS (JPEG quality=85)
- **Multi-Consumer**: Minimal performance impact

### Optimization Tips

```python
# High compression (faster, smaller)
streamer.publish_image(image, compress_jpeg=True, quality=70)

# Better quality (slower, larger)
streamer.publish_image(image, compress_jpeg=True, quality=95)

# Lossless (slowest, largest)
streamer.publish_image(image, compress_jpeg=False)

# Limit stream size
streamer.publish_image(image, maxlen=5)  # Keep only last 5 frames
```

---

## Thread Safety

All classes use the Redis Python client, which is thread-safe for basic operations. However:

- **Subscribe methods are blocking** - run them in separate threads if needed
- **Multiple publishers are safe** - Redis handles concurrent writes
- **Multiple consumers are safe** - each consumer has independent read position

**Example with Threading:**

```python
import threading

def subscriber_thread():
    def callback(data):
        print(f"Received: {data}")
    broker.subscribe_objects(callback)

thread = threading.Thread(target=subscriber_thread, daemon=True)
thread.start()

# Main thread continues...
```

---

## Version Information

- **Package Version**: 0.1.0
- **Python Compatibility**: >=3.8
- **Redis Version**: >=5.0 (for Streams support)

---

## See Also

- [Package README](../README.md) - Installation and quick start guide
- [Workflow Documentation](README.md) - Integration examples and workflows
- [Testing Guide](TESTING.md) - Testing guidelines and examples
- [Contributing Guide](../CONTRIBUTING.md) - Development setup and guidelines
