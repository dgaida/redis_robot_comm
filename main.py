#

import time
import threading
import cv2
from redis_robot_comm.redis_client import RedisMessageBroker
from redis_robot_comm import RedisImageStreamer


# Utility function for easy testing
def test_redis_broker():
    """Test function for the Redis broker"""
    broker = RedisMessageBroker()

    # Test connection
    if not broker.test_connection():
        print("Cannot connect to Redis. Make sure Redis server is running.")
        return

    # Clear previous data
    broker.clear_stream()

    # Test publishing
    test_objects = [
        {
            'id': 'obj_001',
            'class_name': 'cube',
            'confidence': 0.95,
            'position': {'x': 0.1, 'y': 0.2, 'z': 0.05},
            'timestamp': time.time()
        },
        {
            'id': 'obj_002',
            'class_name': 'cylinder',
            'confidence': 0.87,
            'position': {'x': 0.3, 'y': 0.1, 'z': 0.05},
            'timestamp': time.time()
        }
    ]

    test_camera_pose = {
        'x': 0.0, 'y': 0.0, 'z': 0.5,
        'roll': 0.0, 'pitch': 1.57, 'yaw': 0.0
    }

    print("Publishing test objects...")
    broker.publish_objects(test_objects, test_camera_pose)

    # Test getting latest
    print("Getting latest objects...")
    latest = broker.get_latest_objects()
    print(f"Latest objects: {latest}")

    # Test stream info
    broker.get_stream_info()


def test_image_subscription():
    """Subscribe first, publish a frame later, and exit cleanly."""
    streamer = RedisImageStreamer(stream_name="robot_camera")

    stop_flag = threading.Event()

    def image_callback(image, metadata, image_info):
        print("Received image info:", image_info)
        print("Metadata:", metadata)
        # Display the image (press 'q' to quit)
        cv2.imshow("Subscribed Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()

    # Run the subscriber in its own thread
    subscriber_thread = threading.Thread(
        target=streamer.subscribe_variable_images,
        kwargs={"callback": image_callback, "block_ms": 500}
    )
    subscriber_thread.start()

    # Warte kurz bis der Subscriber läuft
    time.sleep(0.5)

    # Jetzt ein Bild veröffentlichen – der Callback bekommt es
    test_img = cv2.imread("example.png")
    streamer.publish_image(test_img, metadata={"robot": "arm1", "workspace": "A"})

    # Hauptthread wartet auf das Quit‑Signal
    while not stop_flag.is_set():
        time.sleep(0.1)

    subscriber_thread.join()
    cv2.destroyAllWindows()


def test_image_streamer():
    streamer = RedisImageStreamer(stream_name="robot_camera")

    # Beispielbild laden
    image = cv2.imread("example.png")

    # Bild veröffentlichen
    streamer.publish_image(image, metadata={"robot": "arm1", "workspace": "A"})

    # Neuestes Bild abrufen
    result = streamer.get_latest_image()
    if result:
        img, metadata = result
        print("Metadaten:", metadata)
        cv2.imshow("Received Image", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    test_redis_broker()

    test_image_streamer()

    test_image_subscription()
