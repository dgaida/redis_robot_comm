# Objekterkennung Streaming

Der `RedisMessageBroker` ist die zentrale Komponente für das Streaming von Objekterkennungsdaten.

## Objekte veröffentlichen

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

# Kamerapose (optional)
camera_pose = {
    "x": 0.0, "y": 0.0, "z": 1.0,
    "roll": 0.0, "pitch": 0.0, "yaw": 0.0
}

broker.publish_objects(objects, camera_pose=camera_pose)
```

## Objekte abrufen

### Neueste Objekte abrufen

```python
# Ruft die neuesten Objekte ab, die nicht älter als 2 Sekunden sind
latest = broker.get_latest_objects(max_age_seconds=2.0)
```

### Objekte in einem Zeitbereich abrufen

```python
import time

start = time.time() - 10  # Vor 10 Sekunden
end = time.time()
objects = broker.get_objects_in_timerange(start, end)
```

## Abonnement (Subscription)

Sie können auf neue Erkennungen warten, ohne den Stream manuell abzufragen:

```python
def on_detection(data):
    print(f"Empfangen: {len(data['objects'])} Objekte")

broker.subscribe_objects(on_detection)
```
