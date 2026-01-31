import logging
import time
from redis_robot_comm import RedisMessageBroker
from redis_robot_comm.exceptions import RedisConnectionError

# Configure logging to see retry attempts
logging.basicConfig(level=logging.INFO)

def main():
    # If Redis is not running, this will fail during initialization
    try:
        broker = RedisMessageBroker(host="non-existent-host", port=1234)
    except RedisConnectionError as e:
        print(f"Caught expected connection error during init: {e}")

    # Example with a valid init but transient failure during operation
    # (Hard to simulate without actually stopping Redis, but we can see the decorator usage)
    print("\nNote: The library uses @retry_on_connection_error for automatic retries.")
    print("Check redis_robot_comm/utils.py to see the implementation.")

if __name__ == "__main__":
    main()
