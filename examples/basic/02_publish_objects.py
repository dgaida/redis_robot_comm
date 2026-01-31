from redis_robot_comm import RedisMessageBroker

def main():
    broker = RedisMessageBroker()

    # Test connection
    if broker.test_connection():
        print("✓ Connected to Redis")

    # Publish detected objects
    objects = [
        {
            "id": "obj_1",
            "class_name": "cube",
            "confidence": 0.95,
            "position": {"x": 0.1, "y": 0.2, "z": 0.05},
        }
    ]

    camera_pose = {
        "x": 0.0, "y": 0.0, "z": 0.5,
        "roll": 0.0, "pitch": 1.57, "yaw": 0.0
    }

    print("Publishing objects...")
    broker.publish_objects(objects, camera_pose)

    # Retrieve latest objects
    latest = broker.get_latest_objects(max_age_seconds=2.0)
    print(f"Found {len(latest)} objects")
    for obj in latest:
        print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")

if __name__ == "__main__":
    main()
