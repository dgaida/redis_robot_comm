"""Shared test fixtures for redis_robot_comm tests."""

import pytest
from unittest.mock import MagicMock
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--redis-url",
        action="store",
        default=None,
        help="Redis server URL for integration tests",
    )


@pytest.fixture
def mock_redis_client(monkeypatch):
    """Provide a mocked Redis client."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    # Setup common mocked responses
    mock_client.ping.return_value = True
    return mock_client


@pytest.fixture
def message_broker(mock_redis_client):
    """Provide a RedisMessageBroker with mocked Redis."""
    from redis_robot_comm.redis_client import RedisMessageBroker
    broker = RedisMessageBroker()
    return broker


@pytest.fixture
def image_streamer(mock_redis_client):
    """Provide a RedisImageStreamer with mocked Redis."""
    from redis_robot_comm.redis_image_streamer import RedisImageStreamer
    return RedisImageStreamer()


@pytest.fixture
def label_manager(mock_redis_client):
    """Provide a RedisLabelManager with mocked Redis."""
    from redis_robot_comm.redis_label_manager import RedisLabelManager
    return RedisLabelManager()


@pytest.fixture
def text_overlay_manager(mock_redis_client):
    """Provide a RedisTextOverlayManager with mocked Redis."""
    from redis_robot_comm.redis_text_overlay import RedisTextOverlayManager
    return RedisTextOverlayManager()


@pytest.fixture
def sample_image():
    """Provide a sample test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_objects():
    """Provide sample object detection data."""
    return [
        {
            "id": "obj_1",
            "class_name": "cube",
            "confidence": 0.95,
            "position": {"x": 0.1, "y": 0.2, "z": 0.05},
        }
    ]
