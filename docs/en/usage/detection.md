# Object Detection Streaming

The `RedisMessageBroker` is the central component for streaming object detection data.

## Publishing Objects

```python
from redis_robot_comm import RedisMessageBroker

broker = RedisMessageBroker()

objects = [
    {
        "id": "obj_001",
        "class_name": "cube",
        "confidence": 0.98,
        "position": {"x": 0.5, "y": 0.0, "z": 0.1}
    }
]

# Camera pose (optional)
camera_pose = {
    "x": 0.0, "y": 0.0, "z": 1.0,
    "roll": 0.0, "pitch": 0.0, "yaw": 0.0
}

broker.publish_objects(objects, camera_pose=camera_pose)
```

## Retrieving Objects

### Getting Latest Objects

```python
# Retrieves the latest objects, no older than 2 seconds
latest = broker.get_latest_objects(max_age_seconds=2.0)
```

### Querying Objects in Time Range

```python
import time

start = time.time() - 10  # 10 seconds ago
end = time.time()
objects = broker.get_objects_in_timerange(start, end)
```

## Subscription

You can wait for new detections without manually polling the stream:

```python
def on_detection(data):
    print(f"Received: {len(data['objects'])} objects")

broker.subscribe_objects(on_detection)
```
