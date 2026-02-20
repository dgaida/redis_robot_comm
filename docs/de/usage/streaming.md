# Bild-Streaming

Der `RedisImageStreamer` ermöglicht das effiziente Streaming von OpenCV-Bildern.

## Bilder senden

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")
image = cv2.imread("frame.jpg")

# Senden mit JPEG-Kompression (Standard)
streamer.publish_image(image, quality=85)

# Senden ohne Kompression (verlustfrei)
streamer.publish_image(image, compress_jpeg=False)
```

## Bilder empfangen

```python
result = streamer.get_latest_image()

if result:
    image, metadata = result
    cv2.imshow("Robot View", image)
```

## Kontinuierliches Streaming

Verwenden Sie `subscribe_variable_images` für eine flüssige Anzeige:

```python
def on_frame(image, metadata, info):
    cv2.imshow("Live", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False # Beendet die Subscription
    return True

streamer.subscribe_variable_images(on_frame)
```
