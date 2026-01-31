"""Redis-based manager for detectable object labels."""

import json
import time
import logging
from typing import List, Optional, Dict, Any, Callable, cast
from redis.exceptions import RedisError

import redis

from .types import StreamID, LabelList
from .exceptions import RedisConnectionError, RedisPublishError, RedisRetrievalError
from .validators import validate_stream_name
from .utils import retry_on_connection_error
from .config import RedisConfig, get_redis_config

logger = logging.getLogger(__name__)


class RedisLabelManager:
    """
    Manages detectable object labels via Redis streams.

    Publishers (vision_detect_segment): Publish available labels when they change.
    Consumers (robot_environment): Subscribe to get current detectable labels.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        stream_name: str = "detectable_labels",
        config: Optional[RedisConfig] = None,
    ) -> None:
        """
        Initialize the label manager.

        Args:
            host: Redis server host.
            port: Redis server port.
            stream_name: Name of the Redis stream for labels.
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

    @retry_on_connection_error(max_attempts=3, delay=0.5)
    def publish_labels(self, labels: LabelList, metadata: Optional[Dict[str, Any]] = None) -> Optional[StreamID]:
        """
        Publish the current list of detectable object labels.

        Args:
            labels: List of object label strings.
            metadata: Optional metadata (e.g., model_id, timestamp).

        Returns:
            Redis stream entry ID, or None if publishing fails.

        Raises:
            RedisPublishError: If publishing to Redis fails.
        """
        message = {
            "timestamp": str(time.time()),
            "labels": json.dumps(labels),
            "label_count": str(len(labels)),
        }

        if metadata:
            message["metadata"] = json.dumps(metadata)

        try:
            # Keep only latest entry (maxlen=1)
            stream_id = self.client.xadd(self.stream_name, message, maxlen=1)

            if self.verbose:
                logger.info(f"Published {len(labels)} labels to Redis: {stream_id}")

            return cast(Optional[StreamID], stream_id)

        except RedisError as e:
            logger.error(f"Error publishing labels to Redis: {e}")
            raise RedisPublishError(f"Failed to publish labels: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error publishing labels: {e}")
            return None

    def get_latest_labels(self, timeout_seconds: float = 5.0) -> Optional[LabelList]:
        """
        Get the most recent list of detectable labels.

        Args:
            timeout_seconds: Maximum age of labels to accept.

        Returns:
            List of label strings, or None if not available or too old.

        Raises:
            RedisRetrievalError: If retrieval from Redis fails.
        """
        try:
            # Get the latest message
            messages = self.client.xrevrange(self.stream_name, count=1)

            if not messages:
                if self.verbose:
                    logger.debug(f"No labels found in {self.stream_name}")
                return None

            msg_id, fields = messages[0]

            # Check if labels are fresh enough
            msg_timestamp = float(fields.get("timestamp", "0"))
            current_time = time.time()

            if current_time - msg_timestamp > timeout_seconds:
                if self.verbose:
                    logger.debug(f"Labels too old: {current_time - msg_timestamp:.1f}s")
                return None

            # Parse and return labels
            labels_json = fields.get("labels", "[]")
            labels = json.loads(labels_json)

            if self.verbose:
                logger.info(f"Retrieved {len(labels)} labels")

            return cast(Optional[LabelList], labels)

        except RedisError as e:
            logger.error(f"Error getting labels from Redis: {e}")
            raise RedisRetrievalError(f"Failed to retrieve labels: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting labels: {e}")
            return None

    def add_label(self, new_label: str) -> bool:
        """
        Add a new label to the current list and republish.

        Args:
            new_label: Label to add.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get current labels
            current_labels = self.get_latest_labels(timeout_seconds=60.0)

            if current_labels is None:
                if self.verbose:
                    logger.info("No existing labels found, creating new list")
                current_labels = []

            # Add new label if not already present
            if new_label.lower() not in [lbl.lower() for lbl in current_labels]:
                current_labels.append(new_label.lower())

                # Republish updated list
                result = self.publish_labels(current_labels, metadata={"action": "add", "added_label": new_label})
                if result is None:
                    return False

                if self.verbose:
                    logger.info(f"Added label: {new_label}")

                return True
            else:
                if self.verbose:
                    logger.info(f"Label already exists: {new_label}")
                return False

        except Exception as e:
            logger.error(f"Error adding label: {e}")
            return False

    def subscribe_to_label_updates(
        self,
        callback: Callable[[LabelList, Dict[str, Any]], None],
        block_ms: int = 1000,
    ) -> None:
        """
        Subscribe to label updates and call callback when they change.

        Args:
            callback: Function receiving (labels, metadata).
            block_ms: Blocking timeout in milliseconds.

        Raises:
            RedisRetrievalError: If subscription fails.
        """
        last_id = "$"  # Start from newest

        if self.verbose:
            logger.info(f"Subscribing to label updates on {self.stream_name}...")

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
                            logger.error(f"Error processing label update: {e}")

        except KeyboardInterrupt:
            logger.info("Stopped subscribing to labels")
        except RedisError as e:
            logger.error(f"Redis error in label subscription: {e}")
            raise RedisRetrievalError(f"Label subscription failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in label subscription: {e}")

    def clear_stream(self) -> bool:
        """
        Clear the labels stream.

        Returns:
            True if successful, False otherwise.
        """
        try:
            result = self.client.delete(self.stream_name)
            if self.verbose:
                logger.info(f"Cleared labels stream: {result}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error clearing stream: {e}")
            return False
