"""
Redis-based manager for detectable object labels.
Handles publishing and subscribing to the stream of recognizable object labels.
"""

import redis
import json
import time
from typing import List, Optional


class RedisLabelManager:
    """
    Manages detectable object labels via Redis streams.

    Publishers (vision_detect_segment): Publish available labels when they change
    Consumers (robot_environment): Subscribe to get current detectable labels
    """

    def __init__(self, host: str = "localhost", port: int = 6379, stream_name: str = "detectable_labels"):
        """
        Initialize the label manager.

        Args:
            host: Redis server host
            port: Redis server port
            stream_name: Name of the Redis stream for labels
        """
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.stream_name = stream_name
        self.verbose = False

    def publish_labels(self, labels: List[str], metadata: dict = None) -> str:
        """
        Publish the current list of detectable object labels.

        Args:
            labels: List of object label strings
            metadata: Optional metadata (e.g., model_id, timestamp)

        Returns:
            str: Redis stream entry ID
        """
        message = {"timestamp": str(time.time()), "labels": json.dumps(labels), "label_count": str(len(labels))}

        if metadata:
            message["metadata"] = json.dumps(metadata)

        try:
            # Keep only latest entry (maxlen=1) since we only care about current state
            stream_id = self.client.xadd(self.stream_name, message, maxlen=1)

            if self.verbose:
                print(f"Published {len(labels)} labels to Redis: {stream_id}")

            return stream_id

        except Exception as e:
            if self.verbose:
                print(f"Error publishing labels: {e}")
            return None

    def get_latest_labels(self, timeout_seconds: float = 5.0) -> Optional[List[str]]:
        """
        Get the most recent list of detectable labels.

        Args:
            timeout_seconds: Maximum age of labels to accept

        Returns:
            List of label strings, or None if not available/too old
        """
        try:
            # Get the latest message
            messages = self.client.xrevrange(self.stream_name, count=1)

            if not messages:
                if self.verbose:
                    print("No labels found in stream")
                return None

            msg_id, fields = messages[0]

            # Check if labels are fresh enough
            msg_timestamp = float(fields.get("timestamp", "0"))
            current_time = time.time()

            if current_time - msg_timestamp > timeout_seconds:
                if self.verbose:
                    print(f"Labels too old: {current_time - msg_timestamp:.1f}s")
                return None

            # Parse and return labels
            labels_json = fields.get("labels", "[]")
            labels = json.loads(labels_json)

            if self.verbose:
                print(f"Retrieved {len(labels)} labels")

            return labels

        except Exception as e:
            if self.verbose:
                print(f"Error getting labels: {e}")
            return None

    def add_label(self, new_label: str) -> bool:
        """
        Add a new label to the current list and republish.

        Args:
            new_label: Label to add

        Returns:
            bool: True if successful
        """
        try:
            # Get current labels
            current_labels = self.get_latest_labels(timeout_seconds=60.0)

            if current_labels is None:
                if self.verbose:
                    print("No existing labels found, creating new list")
                current_labels = []

            # Add new label if not already present
            if new_label.lower() not in [lbl.lower() for lbl in current_labels]:
                current_labels.append(new_label.lower())

                # Republish updated list
                self.publish_labels(current_labels, metadata={"action": "add", "added_label": new_label})

                if self.verbose:
                    print(f"Added label: {new_label}")

                return True
            else:
                if self.verbose:
                    print(f"Label already exists: {new_label}")
                return False

        except Exception as e:
            if self.verbose:
                print(f"Error adding label: {e}")
            return False

    def subscribe_to_label_updates(self, callback, block_ms: int = 1000):
        """
        Subscribe to label updates and call callback when they change.

        Args:
            callback: Function to call with new labels: callback(labels, metadata)
            block_ms: Blocking timeout in milliseconds
        """
        last_id = "$"  # Start from newest

        if self.verbose:
            print("Subscribing to label updates...")

        try:
            while True:
                messages = self.client.xread({self.stream_name: last_id}, block=block_ms, count=1)

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            labels_json = fields.get("labels", "[]")
                            labels = json.loads(labels_json)

                            metadata = {}
                            if "metadata" in fields:
                                metadata = json.loads(fields["metadata"])

                            callback(labels, metadata)
                            last_id = msg_id

                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing label update: {e}")

        except KeyboardInterrupt:
            if self.verbose:
                print("Stopped subscribing to labels")
        except Exception as e:
            if self.verbose:
                print(f"Error in label subscription: {e}")

    def clear_stream(self) -> bool:
        """
        Clear the labels stream.

        Returns:
            bool: True if successful
        """
        try:
            result = self.client.delete(self.stream_name)
            if self.verbose:
                print(f"Cleared labels stream: {result}")
            return bool(result)
        except Exception as e:
            if self.verbose:
                print(f"Error clearing stream: {e}")
            return False
