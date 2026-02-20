# Text Overlays

The `RedisTextOverlayManager` allows adding text overlays for video recordings or user interfaces.

## Publishing Messages

### User Tasks

```python
from redis_robot_comm import RedisTextOverlayManager

text_mgr = RedisTextOverlayManager()
text_mgr.publish_user_task("Pick up the red cube.")
```

### Robot Speech

```python
text_mgr.publish_robot_speech("I am moving to the cube now.", duration_seconds=3.0)
```

### System Messages

```python
text_mgr.publish_system_message("Recording started", duration_seconds=2.0)
```

## Subscribing to Overlays

This is typically used by a recording script:

```python
def display_text(data):
    print(f"[{data['type']}] {data['text']}")

text_mgr.subscribe_to_texts(display_text)
```
