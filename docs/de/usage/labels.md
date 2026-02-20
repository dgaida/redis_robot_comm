# Label-Verwaltung

Der `RedisLabelManager` wird verwendet, um die Liste der erkennbaren Objekte dynamisch zu verwalten.

## Labels veröffentlichen

```python
from redis_robot_comm import RedisLabelManager

manager = RedisLabelManager()
labels = ["cube", "sphere", "cylinder"]

manager.publish_labels(labels, metadata={"model": "yolov8-robotics"})
```

## Aktuelle Labels abrufen

```python
current_labels = manager.get_latest_labels()
if current_labels:
    print(f"Erkennbare Objekte: {current_labels}")
```

## Labels dynamisch hinzufügen

```python
manager.add_label("prism")
```
