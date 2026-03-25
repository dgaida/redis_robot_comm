import pytest
from unittest.mock import MagicMock
from redis.exceptions import RedisError
from redis_robot_comm.redis_client import RedisMessageBroker
from redis_robot_comm.redis_image_streamer import RedisImageStreamer
from redis_robot_comm.exceptions import RedisRetrievalError

def test_broker_get_latest_objects_raises_on_redis_error(mock_redis_client):
    broker = RedisMessageBroker()
    mock_redis_client.xrevrange.side_effect = RedisError("Redis is down")

    with pytest.raises(RedisRetrievalError):
        broker.get_latest_objects()

def test_image_streamer_get_latest_image_raises_on_redis_error(mock_redis_client):
    streamer = RedisImageStreamer()
    mock_redis_client.xrevrange.side_effect = RedisError("Redis is down")

    with pytest.raises(RedisRetrievalError):
        streamer.get_latest_image()
