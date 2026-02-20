# Image Streaming

The `RedisImageStreamer` enables efficient streaming of OpenCV images.

## Sending Images

```python
from redis_robot_comm import RedisImageStreamer
import cv2

streamer = RedisImageStreamer(stream_name="robot_camera")
image = cv2.imread("frame.jpg")

# Send with JPEG compression (default)
streamer.publish_image(image, quality=85)

# Send without compression (lossless)
streamer.publish_image(image, compress_jpeg=False)
```

## Receiving Images

```python
result = streamer.get_latest_image()

if result:
    image, metadata = result
    cv2.imshow("Robot View", image)
```

## Continuous Streaming

Use `subscribe_variable_images` for a smooth display:

```python
def on_frame(image, metadata, info):
    cv2.imshow("Live", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False # Stops the subscription
    return True

streamer.subscribe_variable_images(on_frame)
```
