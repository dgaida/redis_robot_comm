# redis_robot_comm/redis_text_overlay_manager.py
"""
Redis-based manager for text overlays in robot videos.
Handles publishing and subscribing to user tasks and robot speech.
"""

import redis
import json
import time
from typing import Optional, Dict, List
from enum import Enum


class TextType(Enum):
    """Type of text overlay."""

    USER_TASK = "user_task"
    ROBOT_SPEECH = "robot_speech"
    SYSTEM_MESSAGE = "system_message"


class RedisTextOverlayManager:
    """
    Manages text overlays for robot video recordings via Redis streams.

    Publishers (MCP server): Publish user tasks and robot speech
    Consumers (recording script): Subscribe to display texts in video
    """

    def __init__(self, host: str = "localhost", port: int = 6379, stream_name: str = "video_text_overlays"):
        """
        Initialize the text overlay manager.

        Args:
            host: Redis server host
            port: Redis server port
            stream_name: Name of the Redis stream for text overlays
        """
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.stream_name = stream_name
        self.verbose = False

    def publish_user_task(self, task: str, metadata: dict = None) -> str:
        """
        Publish a user task/command.

        Args:
            task: The user's natural language task/command
            metadata: Optional metadata (e.g., user_id, session_id)

        Returns:
            str: Redis stream entry ID
        """
        return self._publish_text(text=task, text_type=TextType.USER_TASK, metadata=metadata)

    def publish_robot_speech(self, speech: str, duration_seconds: float = 4.0, metadata: dict = None) -> str:
        """
        Publish robot speech/explanation.

        Args:
            speech: What the robot is saying
            duration_seconds: How long to display the text (default: 4.0s)
            metadata: Optional metadata (e.g., tool_name, priority)

        Returns:
            str: Redis stream entry ID
        """
        if metadata is None:
            metadata = {}

        metadata["duration_seconds"] = duration_seconds

        return self._publish_text(text=speech, text_type=TextType.ROBOT_SPEECH, metadata=metadata)

    def publish_system_message(self, message: str, duration_seconds: float = 3.0, metadata: dict = None) -> str:
        """
        Publish system message (e.g., "Recording started").

        Args:
            message: System message text
            duration_seconds: How long to display
            metadata: Optional metadata

        Returns:
            str: Redis stream entry ID
        """
        if metadata is None:
            metadata = {}

        metadata["duration_seconds"] = duration_seconds

        return self._publish_text(text=message, text_type=TextType.SYSTEM_MESSAGE, metadata=metadata)

    def _publish_text(self, text: str, text_type: TextType, metadata: dict = None) -> str:
        """
        Internal method to publish text overlay.

        Args:
            text: Text content
            text_type: Type of text (user_task, robot_speech, system_message)
            metadata: Optional metadata

        Returns:
            str: Redis stream entry ID
        """
        message = {
            "timestamp": str(time.time()),
            "text": text,
            "type": text_type.value,
            "metadata": json.dumps(metadata or {}),
        }

        try:
            # Keep last 100 entries (enough for a recording session)
            stream_id = self.client.xadd(self.stream_name, message, maxlen=100)

            if self.verbose:
                print(f"Published {text_type.value}: {text[:50]}...")

            return stream_id

        except Exception as e:
            if self.verbose:
                print(f"Error publishing text overlay: {e}")
            return None

    def get_latest_texts(self, max_age_seconds: float = 10.0, text_type: Optional[TextType] = None) -> List[Dict]:
        """
        Get recent text overlays.

        Args:
            max_age_seconds: Maximum age of texts to retrieve
            text_type: Filter by text type (None = all types)

        Returns:
            List of text overlay dictionaries
        """
        try:
            # Get recent messages
            current_time = time.time()
            start_id = f"{int((current_time - max_age_seconds) * 1000)}-0"

            messages = self.client.xrange(self.stream_name, min=start_id, max="+")

            texts = []
            for msg_id, fields in messages:
                try:
                    text_data = {
                        "id": msg_id,
                        "timestamp": float(fields.get("timestamp", "0")),
                        "text": fields.get("text", ""),
                        "type": fields.get("type", ""),
                        "metadata": json.loads(fields.get("metadata", "{}")),
                    }

                    # Filter by type if specified
                    if text_type is None or text_data["type"] == text_type.value:
                        texts.append(text_data)

                except Exception as e:
                    if self.verbose:
                        print(f"Error parsing text overlay: {e}")
                    continue

            return texts

        except Exception as e:
            if self.verbose:
                print(f"Error getting latest texts: {e}")
            return []

    def get_current_user_task(self, max_age_seconds: float = 300.0) -> Optional[str]:
        """
        Get the most recent user task (if still relevant).

        Args:
            max_age_seconds: Maximum age to consider current (default: 5 minutes)

        Returns:
            str: Current user task or None
        """
        try:
            messages = self.client.xrevrange(self.stream_name, count=50)  # Check last 50 messages

            current_time = time.time()

            for msg_id, fields in messages:
                msg_type = fields.get("type", "")

                if msg_type == TextType.USER_TASK.value:
                    msg_timestamp = float(fields.get("timestamp", "0"))

                    # Check if still relevant
                    if current_time - msg_timestamp <= max_age_seconds:
                        return fields.get("text", "")
                    else:
                        # Too old
                        return None

            return None

        except Exception as e:
            if self.verbose:
                print(f"Error getting current user task: {e}")
            return None

    def subscribe_to_texts(self, callback, block_ms: int = 1000, text_type: Optional[TextType] = None):
        """
        Subscribe to text overlays and call callback for each one.

        Args:
            callback: Function to call with text data: callback(text_dict)
            block_ms: Blocking timeout in milliseconds
            text_type: Filter by text type (None = all types)
        """
        last_id = "$"  # Start from newest

        if self.verbose:
            print(f"Subscribing to text overlays (type: {text_type or 'all'})...")

        try:
            while True:
                messages = self.client.xread({self.stream_name: last_id}, block=block_ms, count=1)

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            text_data = {
                                "id": msg_id,
                                "timestamp": float(fields.get("timestamp", "0")),
                                "text": fields.get("text", ""),
                                "type": fields.get("type", ""),
                                "metadata": json.loads(fields.get("metadata", "{}")),
                            }

                            # Filter by type if specified
                            if text_type is None or text_data["type"] == text_type.value:
                                callback(text_data)

                            last_id = msg_id

                        except Exception as e:
                            if self.verbose:
                                print(f"Error processing text overlay: {e}")

        except KeyboardInterrupt:
            if self.verbose:
                print("Stopped subscribing to text overlays")
        except Exception as e:
            if self.verbose:
                print(f"Error in text overlay subscription: {e}")

    def clear_stream(self) -> bool:
        """
        Clear the text overlay stream.

        Returns:
            bool: True if successful
        """
        try:
            result = self.client.delete(self.stream_name)
            if self.verbose:
                print(f"Cleared text overlay stream: {result}")
            return bool(result)
        except Exception as e:
            if self.verbose:
                print(f"Error clearing stream: {e}")
            return False

    def get_stream_info(self) -> Dict:
        """
        Get stream statistics.

        Returns:
            Dict: Stream information
        """
        try:
            info = self.client.xinfo_stream(self.stream_name)
            return {
                "total_messages": info.get("length", 0),
                "first_entry_id": info.get("first-entry", [None])[0],
                "last_entry_id": info.get("last-entry", [None])[0],
            }
        except Exception as e:
            return {"error": f"Stream not found or empty: {e}"}
