# Text-Overlays

Der `RedisTextOverlayManager` ermöglicht das Hinzufügen von Texteinblendungen für Videoaufzeichnungen oder Benutzeroberflächen.

## Nachrichten veröffentlichen

### Benutzeraufgaben

```python
from redis_robot_comm import RedisTextOverlayManager

text_mgr = RedisTextOverlayManager()
text_mgr.publish_user_task("Hebe den roten Würfel auf.")
```

### Robotersprache

```python
text_mgr.publish_robot_speech("Ich bewege mich jetzt zum Würfel.", duration_seconds=3.0)
```

### Systemnachrichten

```python
text_mgr.publish_system_message("Aufzeichnung gestartet", duration_seconds=2.0)
```

## Overlays abonnieren

Dies wird typischerweise von einem Aufzeichnungsskript verwendet:

```python
def display_text(data):
    print(f"[{data['type']}] {data['text']}")

text_mgr.subscribe_to_texts(display_text)
```
