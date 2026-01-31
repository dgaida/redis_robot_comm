"""Input validation utilities."""

from typing import List, Dict, Any
import numpy as np

from .exceptions import InvalidImageError, RedisPublishError


def validate_objects(objects: List[Dict[str, Any]]) -> None:
    """
    Validate object detection data structure.

    Args:
        objects: List of object dictionaries to validate.

    Raises:
        RedisPublishError: If objects data is invalid.
    """
    if not isinstance(objects, list):
        raise RedisPublishError(f"Objects must be a list, got {type(objects).__name__}")

    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise RedisPublishError(f"Object at index {i} must be a dict, got {type(obj).__name__}")

        # Validate required fields
        if "id" not in obj:
            raise RedisPublishError(f"Object at index {i} missing required field 'id'")

        if "class_name" not in obj:
            raise RedisPublishError(f"Object at index {i} missing required field 'class_name'")


def validate_image(image: np.ndarray) -> None:
    """
    Validate image array.

    Args:
        image: NumPy array to validate as image.

    Raises:
        InvalidImageError: If image is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Image must be a NumPy array, got {type(image).__name__}")

    if image.size == 0:
        raise InvalidImageError("Image array is empty")

    if image.ndim not in (2, 3):
        raise InvalidImageError(f"Image must be 2D or 3D array, got {image.ndim}D")

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise InvalidImageError(f"Image channels must be 1, 3, or 4, got {image.shape[2]}")

    # Validate data type
    if image.dtype != np.uint8:
        raise InvalidImageError(f"Image dtype must be uint8, got {image.dtype}")

    # Validate dimensions are reasonable
    max_dimension = 10000  # 10K pixels
    if image.shape[0] > max_dimension or image.shape[1] > max_dimension:
        raise InvalidImageError(f"Image dimensions too large: {image.shape}. " f"Maximum dimension is {max_dimension} pixels")


def validate_stream_name(stream_name: str) -> None:
    """
    Validate Redis stream name.

    Args:
        stream_name: Stream name to validate.

    Raises:
        ValueError: If stream name is invalid.
    """
    if not isinstance(stream_name, str):
        raise ValueError(f"Stream name must be a string, got {type(stream_name).__name__}")

    if not stream_name:
        raise ValueError("Stream name cannot be empty")

    if len(stream_name) > 255:
        raise ValueError(f"Stream name too long ({len(stream_name)} chars). Maximum is 255")

    # Redis key naming best practices
    invalid_chars = [" ", "\n", "\r", "\t"]
    for char in invalid_chars:
        if char in stream_name:
            raise ValueError(f"Stream name contains invalid character: {repr(char)}")
