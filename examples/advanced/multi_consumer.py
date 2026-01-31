import threading
import time
from redis_robot_comm import RedisMessageBroker

def consumer(consumer_id):
    broker = RedisMessageBroker()
    print(f"Consumer {consumer_id} starting...")

    def on_objects(data):
        print(f"Consumer {consumer_id} received {len(data['objects'])} objects")

    broker.subscribe_objects(on_objects)

def producer():
    broker = RedisMessageBroker()
    print("Producer starting...")
    for i in range(5):
        objects = [{"id": f"obj_{i}", "class_name": "cube"}]
        broker.publish_objects(objects)
        time.sleep(1)

def main():
    # Start two consumers
    c1 = threading.Thread(target=consumer, args=(1,), daemon=True)
    c2 = threading.Thread(target=consumer, args=(2,), daemon=True)
    c1.start()
    c2.start()

    time.sleep(1)

    # Start producer
    producer()

    print("Example finished.")

if __name__ == "__main__":
    main()
