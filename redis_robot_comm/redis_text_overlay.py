"""Redis-based manager for text overlays in robot videos."""

import json
import time
import logging
from typing import Optional, Dict, List, Any, Callable, cast
from enum import Enum
from redis.exceptions import RedisError

import redis

from .types import StreamID, TextOverlayDict
from .exceptions import RedisConnectionError, RedisPublishError, RedisRetrievalError
from .validators import validate_stream_name
from .utils import retry_on_connection_error
from .config import RedisConfig, get_redis_config

logger = logging.getLogger(__name__)


class TextType(Enum):
    """Type of text overlay."""

    USER_TASK = "user_task"
    ROBOT_SPEECH = "robot_speech"
    SYSTEM_MESSAGE = "system_message"


class RedisTextOverlayManager:
    """
    Manages text overlays for robot video recordings via Redis streams.

    Publishers (MCP server): Publish user tasks and robot speech.
    Consumers (recording script): Subscribe to display texts in video.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        stream_name: str = "video_text_overlays",
        config: Optional[RedisConfig] = None,
    ) -> None:
        """
        Initialize the text overlay manager.

        Args:
            host: Redis server host.
            port: Redis server port.
            stream_name: Name of the Redis stream for text overlays.
            config: Optional RedisConfig instance.

        Raises:
            RedisConnectionError: If connection to Redis fails.
        """
        if config is None:
            config = get_redis_config()

        # Override config with explicit parameters if provided
        host = host or config.host
        port = port or config.port

        validate_stream_name(stream_name)
        self.stream_name: str = stream_name
        self.verbose: bool = False
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=config.db,
                password=config.password,
                socket_timeout=config.socket_timeout,
                socket_connect_timeout=config.socket_connect_timeout,
                retry_on_timeout=config.retry_on_timeout,
                max_connections=config.max_connections,
                decode_responses=True,
            )
            self.client.ping()
        except RedisError as e:
            raise RedisConnectionError(f"Failed to connect to Redis: {e}") from e

    def publish_user_task(self, task: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[StreamID]:
        """
        Publish a user task/command.

        Args:
            task: The user's natural language task/command.
            metadata: Optional metadata (e.g., user_id, session_id).

        Returns:
            Redis stream entry ID, or None if publishing fails.
        """
        return self._publish_text(text=task, text_type=TextType.USER_TASK, metadata=metadata)

    def publish_robot_speech(
        self,
        speech: str,
        duration_seconds: float = 4.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamID]:
        """
        Publish robot speech/explanation.

        Args:
            speech: What the robot is saying.
            duration_seconds: How long to display the text.
            metadata: Optional metadata (e.g., tool_name, priority).

        Returns:
            Redis stream entry ID, or None if publishing fails.
        """
        if metadata is None:
            metadata = {}

        metadata["duration_seconds"] = duration_seconds

        return self._publish_text(text=speech, text_type=TextType.ROBOT_SPEECH, metadata=metadata)

    def publish_system_message(
        self,
        message: str,
        duration_seconds: float = 3.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamID]:
        """
        Publish system message (e.g., "Recording started").

        Args:
            message: System message text.
            duration_seconds: How long to display.
            metadata: Optional metadata.

        Returns:
            Redis stream entry ID, or None if publishing fails.
        """
        if metadata is None:
            metadata = {}

        metadata["duration_seconds"] = duration_seconds

        return self._publish_text(text=message, text_type=TextType.SYSTEM_MESSAGE, metadata=metadata)

    @retry_on_connection_error(max_attempts=3, delay=0.5)
    def _publish_text(
        self,
        text: str,
        text_type: TextType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StreamID]:
        """
        Internal method to publish text overlay.

        Args:
            text: Text content.
            text_type: Type of text (user_task, robot_speech, system_message).
            metadata: Optional metadata.

        Returns:
            Redis stream entry ID, or None if publishing fails.

        Raises:
            RedisPublishError: If publishing to Redis fails.
        """
        message = {
            "timestamp": str(time.time()),
            "text": text,
            "type": text_type.value,
            "metadata": json.dumps(metadata or {}),
        }

        try:
            # Keep last 100 entries
            stream_id = self.client.xadd(self.stream_name, message, maxlen=100)

            if self.verbose:
                logger.info(f"Published {text_type.value}: {text[:50]}...")

            return cast(Optional[StreamID], stream_id)

        except RedisError as e:
            logger.error(f"Error publishing text overlay: {e}")
            raise RedisPublishError(f"Failed to publish text overlay: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error publishing text overlay: {e}")
            return None

    def get_latest_texts(
        self,
        max_age_seconds: float = 10.0,
        text_type: Optional[TextType] = None,
    ) -> List[TextOverlayDict]:
        """
        Get recent text overlays.

        Args:
            max_age_seconds: Maximum age of texts to retrieve.
            text_type: Filter by text type (None = all types).

        Returns:
            List of text overlay dictionaries.

        Raises:
            RedisRetrievalError: If retrieval from Redis fails.
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
                    logger.error(f"Error parsing text overlay: {e}")
                    continue

            return cast(List[TextOverlayDict], texts)

        except RedisError as e:
            logger.error(f"Error getting latest texts from Redis: {e}")
            raise RedisRetrievalError(f"Failed to retrieve latest texts: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting latest texts: {e}")
            return []

    def get_current_user_task(self, max_age_seconds: float = 300.0) -> Optional[str]:
        """
        Get the most recent user task (if still relevant).

        Args:
            max_age_seconds: Maximum age to consider current (default: 5 minutes).

        Returns:
            Current user task or None.

        Raises:
            RedisRetrievalError: If retrieval from Redis fails.
        """
        try:
            messages = self.client.xrevrange(self.stream_name, count=50)

            current_time = time.time()

            for msg_id, fields in messages:
                msg_type = fields.get("type", "")

                if msg_type == TextType.USER_TASK.value:
                    msg_timestamp = float(fields.get("timestamp", "0"))

                    # Check if still relevant
                    if current_time - msg_timestamp <= max_age_seconds:
                        return str(fields.get("text", ""))
                    else:
                        # Too old
                        return None

            return None

        except RedisError as e:
            logger.error(f"Error getting current user task from Redis: {e}")
            raise RedisRetrievalError(f"Failed to retrieve user task: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting current user task: {e}")
            return None

    def subscribe_to_texts(
        self,
        callback: Callable[[TextOverlayDict], None],
        block_ms: int = 1000,
        text_type: Optional[TextType] = None,
    ) -> None:
        """
        Subscribe to text overlays and call callback for each one.

        Args:
            callback: Function receiving text data dictionary.
            block_ms: Blocking timeout in milliseconds.
            text_type: Filter by text type (None = all types).

        Raises:
            RedisRetrievalError: If subscription fails.
        """
        last_id = "$"  # Start from newest

        if self.verbose:
            logger.info(f"Subscribing to text overlays on {self.stream_name} (type: {text_type or 'all'})...")

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
                            logger.error(f"Error processing text overlay: {e}")

        except KeyboardInterrupt:
            logger.info("Stopped subscribing to text overlays")
        except RedisError as e:
            logger.error(f"Redis error in text overlay subscription: {e}")
            raise RedisRetrievalError(f"Text overlay subscription failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in text overlay subscription: {e}")

    def clear_stream(self) -> bool:
        """
        Clear the text overlay stream.

        Returns:
            True if successful, False otherwise.
        """
        try:
            result = self.client.delete(self.stream_name)
            if self.verbose:
                logger.info(f"Cleared text overlay stream: {result}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error clearing stream: {e}")
            return False

    def get_stream_info(self) -> Dict[str, Any]:
        """
        Get stream statistics.

        Returns:
            Stream information dictionary.
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
