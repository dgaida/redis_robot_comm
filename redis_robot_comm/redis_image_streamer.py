"""
RedisImageStreamer
==================

A helper for streaming OpenCV images of arbitrary size through a Redis stream.
"""

import redis
import cv2
import base64
import json
import time
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from redis.exceptions import RedisError

from .types import ImageArray, ImageMetadata, StreamID
from .exceptions import RedisConnectionError, RedisPublishError, RedisRetrievalError, InvalidImageError
from .validators import validate_image, validate_stream_name
from .utils import retry_on_connection_error
from .config import RedisConfig, get_redis_config

logger = logging.getLogger(__name__)


class RedisImageStreamer:
    """
    A Redis-backed stream that can publish and consume OpenCV images of arbitrary size.

    The class serializes an image (either as raw bytes or JPEG) and stores it in a Redis stream.
    Each entry contains metadata such as the image shape, data type, and optional custom fields.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        stream_name: str = "robot_camera",
        config: Optional[RedisConfig] = None,
    ) -> None:
        """
        Initialize the Redis image streamer.

        Args:
            host: Redis server hostname or IP address.
            port: Redis server port.
            stream_name: Name of the stream that will hold the image frames.
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
    def publish_image(
        self,
        image: ImageArray,
        metadata: Optional[ImageMetadata] = None,
        compress_jpeg: bool = True,
        quality: int = 80,
        maxlen: int = 5,
    ) -> StreamID:
        """
        Publish a single image frame to the Redis stream.

        Args:
            image: OpenCV image array (H×W×C or H×W for grayscale).
            metadata: Arbitrary metadata stored alongside the frame.
            compress_jpeg: Whether to compress the image to JPEG.
            quality: JPEG compression quality (1-100).
            maxlen: Maximum number of entries to keep in the stream.

        Returns:
            The unique Redis entry ID.

        Raises:
            InvalidImageError: If the supplied image is invalid.
            RedisPublishError: If publishing to Redis fails.
        """
        try:
            validate_image(image)
        except InvalidImageError as e:
            logger.error(f"Image validation failed: {e}")
            raise

        timestamp = time.time()

        # Handle different image sizes dynamically
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        if compress_jpeg:
            # Compress to JPEG
            success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise InvalidImageError("Failed to compress image to JPEG")
            image_data = base64.b64encode(buffer).decode("utf-8")
            format_type = "jpeg"
            compressed_size = len(buffer)
        else:
            # Raw image data
            image_data = base64.b64encode(image.tobytes()).decode("utf-8")
            format_type = "raw"
            compressed_size = image.nbytes

        # Prepare message with dynamic image info
        message = {
            "timestamp": str(timestamp),
            "image_data": image_data,
            "format": format_type,
            "width": str(width),
            "height": str(height),
            "channels": str(channels),
            "dtype": str(image.dtype),
            "compressed_size": str(compressed_size),
            "original_size": str(image.nbytes),
        }

        # Add optional metadata
        if metadata:
            message["metadata"] = json.dumps(metadata)

        # Publish to Redis stream
        try:
            stream_id = self.client.xadd(self.stream_name, message, maxlen=maxlen)
            if self.verbose:
                logger.info(f"Published {width}x{height} image ({compressed_size} bytes)")
            return str(stream_id)
        except RedisError as e:
            logger.error(f"Failed to publish image to Redis: {e}")
            raise RedisPublishError(f"Failed to publish image: {e}") from e

    def get_latest_image(self) -> Optional[Tuple[ImageArray, ImageMetadata]]:
        """
        Retrieve the newest frame from the stream.

        Returns:
            A tuple of (image_array, metadata_dict) if a frame is present, otherwise None.

        Raises:
            RedisRetrievalError: If retrieval from Redis fails.
        """
        try:
            messages = self.client.xrevrange(self.stream_name, count=1)
            if not messages:
                return None

            msg_id, fields = messages[0]
            return self._decode_variable_image(fields)

        except RedisError as e:
            logger.error(f"Error getting latest image from Redis: {e}")
            raise RedisRetrievalError(f"Failed to retrieve latest image: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting latest image: {e}")
            return None

    def subscribe_variable_images(
        self,
        callback: Callable[[ImageArray, ImageMetadata, Dict[str, Any]], None],
        block_ms: int = 1000,
        start_after: str = "$",
    ) -> None:
        """
        Continuously listen for new frames and invoke callback for each one.

        Args:
            callback: Function receiving (image, metadata, image_info).
            block_ms: Timeout for the underlying Redis xread.
            start_after: Redis stream ID after which to start reading.

        Raises:
            RedisRetrievalError: If subscription fails.
        """
        last_id = start_after
        if self.verbose:
            logger.info(f"Subscribing to image stream {self.stream_name}...")

        try:
            while True:
                messages = self.client.xread({self.stream_name: last_id}, block=block_ms, count=1)

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        result = self._decode_variable_image(fields)
                        if result:
                            image, metadata = result

                            # Prepare image info for callback
                            image_info = {
                                "width": image.shape[1],
                                "height": image.shape[0],
                                "channels": (image.shape[2] if len(image.shape) == 3 else 1),
                                "timestamp": float(fields.get("timestamp", "0")),
                                "compressed_size": int(fields.get("compressed_size", "0")),
                                "original_size": int(fields.get("original_size", "0")),
                            }

                            callback(image, metadata, image_info)
                        last_id = msg_id

        except KeyboardInterrupt:
            logger.info("Stopped subscribing to images")
        except RedisError as e:
            logger.error(f"Redis error in image subscription: {e}")
            raise RedisRetrievalError(f"Image subscription failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in image subscription: {e}")
            time.sleep(0.1)

    def _decode_variable_image(self, fields: Dict[str, Any]) -> Optional[Tuple[ImageArray, ImageMetadata]]:
        """
        Decode a Redis stream entry that contains an image.

        Args:
            fields: key/value pairs from a Redis entry.

        Returns:
            A tuple of (image_array, metadata_dict) or None if decoding fails.
        """
        try:
            # Extract image parameters
            width = int(fields["width"])
            height = int(fields["height"])
            channels = int(fields["channels"])
            format_type = fields["format"]

            # Decode image data
            image_data = base64.b64decode(fields["image_data"])

            if format_type == "jpeg":
                # Decode JPEG
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("JPEG decoding returned None")
            else:
                # Decode raw image with specified dimensions
                dtype = fields["dtype"]
                shape: Tuple[int, ...]
                if channels == 1:
                    shape = (height, width)
                else:
                    shape = (height, width, channels)
                image = np.frombuffer(image_data, dtype=dtype).reshape(shape)

            # Extract metadata if available
            metadata = {}
            if "metadata" in fields:
                metadata = json.loads(fields["metadata"])

            return image, metadata

        except Exception as e:
            if self.verbose:
                logger.error(f"Error decoding variable image: {e}")
            return None

    def get_stream_stats(self) -> Dict[str, Any]:
        """
        Retrieve bookkeeping information about the Redis stream.

        Returns:
            Dictionary with stream statistics.
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
