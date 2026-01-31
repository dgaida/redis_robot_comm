"""Protocol definitions for redis_robot_comm package."""

from typing import Protocol, Optional, List, Dict, Any

from .types import ImageArray, ImageMetadata, StreamID, ObjectDict, CameraPose


class RedisStreamPublisher(Protocol):
    """Protocol for classes that publish to Redis streams."""

    def clear_stream(self) -> bool:
        """Clear the Redis stream."""
        ...


class RedisStreamSubscriber(Protocol):
    """Protocol for classes that subscribe to Redis streams."""

    def get_stream_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the Redis stream."""
        ...


class ImagePublisher(Protocol):
    """Protocol for image publishing."""

    def publish_image(
        self,
        image: ImageArray,
        metadata: Optional[ImageMetadata] = None,
        compress_jpeg: bool = True,
        quality: int = 80,
        maxlen: int = 5,
    ) -> StreamID:
        """Publish image to stream."""
        ...


class ObjectPublisher(Protocol):
    """Protocol for object detection publishing."""

    def publish_objects(
        self,
        objects: List[ObjectDict],
        camera_pose: Optional[CameraPose] = None,
        maxlen: int = 500,
    ) -> Optional[StreamID]:
        """Publish detected objects to stream."""
        ...
