# Label Management

The `RedisLabelManager` is used to dynamically manage the list of detectable objects.

## Publishing Labels

```python
from redis_robot_comm import RedisLabelManager

manager = RedisLabelManager()
labels = ["cube", "sphere", "cylinder"]

manager.publish_labels(labels, metadata={"model": "yolov8-robotics"})
```

## Retrieving Latest Labels

```python
current_labels = manager.get_latest_labels()
if current_labels:
    print(f"Detectable objects: {current_labels}")
```

## Adding Labels Dynamically

```python
manager.add_label("prism")
```
