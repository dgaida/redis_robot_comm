from redis_robot_comm import RedisImageStreamer
import numpy as np
import cv2


def main():
    streamer = RedisImageStreamer(stream_name="test_camera")

    # Create a dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, "Test Frame", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Publish image
    print("Publishing image...")
    stream_id = streamer.publish_image(image, metadata={"source": "example_script"}, compress_jpeg=True, quality=80)
    print(f"Published to stream ID: {stream_id}")

    # Retrieve latest image
    result = streamer.get_latest_image()
    if result:
        img, metadata = result
        print(f"Retrieved image with metadata: {metadata}")
        # In a real script you might use cv2.imshow("Stream", img)


if __name__ == "__main__":
    main()
