"""
Integration tests requiring actual Redis server.

Run with: pytest tests/integration/ --redis-url redis://localhost:6379
"""

import pytest
import time
import numpy as np
from redis_robot_comm import (
    RedisMessageBroker,
    RedisImageStreamer,
    RedisLabelManager,
    RedisTextOverlayManager,
)


@pytest.fixture(scope="module")
def redis_url(request):
    """Get Redis URL from pytest options."""
    url = request.config.getoption("--redis-url", default=None)
    if not url:
        pytest.skip("Integration tests require --redis-url option")
    return url


@pytest.fixture
def clean_redis(redis_url):
    """Ensure clean Redis state before each test."""
    import redis
    client = redis.from_url(redis_url, decode_responses=True)

    # Clean up test streams
    test_streams = [
        "detected_objects",
        "test_camera",
        "detectable_labels",
        "video_text_overlays",
    ]
    for stream in test_streams:
        client.delete(stream)

    yield client

    # Cleanup after test
    for stream in test_streams:
        client.delete(stream)


class TestRedisIntegration:
    """Integration tests with real Redis server."""

    def test_object_publish_retrieve_flow(self, redis_url, clean_redis):
        """Test complete object detection flow with real Redis."""
        host = redis_url.split("//")[1].split(":")[0]
        port = int(redis_url.split(":")[2].split("/")[0])
        broker = RedisMessageBroker(host=host, port=port)

        # Publish objects
        objects = [
            {
                "id": "obj_1",
                "class_name": "cube",
                "confidence": 0.95,
                "position": {"x": 0.1, "y": 0.2, "z": 0.05},
            }
        ]
        camera_pose = {"x": 0.0, "y": 0.0, "z": 0.5}

        stream_id = broker.publish_objects(objects, camera_pose)
        assert stream_id is not None

        # Retrieve objects
        retrieved = broker.get_latest_objects(max_age_seconds=5.0)
        assert len(retrieved) == 1
        assert retrieved[0]["id"] == "obj_1"
        assert retrieved[0]["class_name"] == "cube"

    def test_image_streaming_flow(self, redis_url, clean_redis):
        """Test image streaming with real Redis."""
        host = redis_url.split("//")[1].split(":")[0]
        port = int(redis_url.split(":")[2].split("/")[0])
        streamer = RedisImageStreamer(
            host=host, port=port,
            stream_name="test_camera",
        )

        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        metadata = {"robot": "arm1", "frame": 42}

        # Publish image
        stream_id = streamer.publish_image(
            test_image,
            metadata=metadata,
            compress_jpeg=True,
            quality=85,
        )
        assert stream_id is not None

        # Retrieve image
        result = streamer.get_latest_image()
        assert result is not None

        image, meta = result
        assert image.shape == (100, 100, 3)
        assert meta["robot"] == "arm1"
        assert meta["frame"] == 42
