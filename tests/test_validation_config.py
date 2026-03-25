from unittest.mock import ANY

import numpy as np
import pytest

from redis_robot_comm.config import ImageStreamConfig
from redis_robot_comm.redis_image_streamer import RedisImageStreamer


def test_publish_image_quality_validation(mock_redis_client):
    streamer = RedisImageStreamer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Valid qualities
    streamer.publish_image(image, quality=1)
    streamer.publish_image(image, quality=100)

    # Invalid qualities
    with pytest.raises(ValueError, match="Quality must be between 1 and 100"):
        streamer.publish_image(image, quality=0)
    with pytest.raises(ValueError, match="Quality must be between 1 and 100"):
        streamer.publish_image(image, quality=101)


def test_image_streamer_uses_config_defaults(mock_redis_client):
    config = ImageStreamConfig(max_length=42, default_quality=75)
    streamer = RedisImageStreamer(stream_config=config)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    streamer.publish_image(image)

    # Verify maxlen was used
    mock_redis_client.xadd.assert_called_with(streamer.stream_name, ANY, maxlen=42)
