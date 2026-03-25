from unittest.mock import MagicMock

from redis_robot_comm.redis_client import RedisMessageBroker
from redis_robot_comm.redis_image_streamer import RedisImageStreamer


def test_broker_subscription_loop_on_error(mock_redis_client):
    broker = RedisMessageBroker()

    # Mock xread to return the same message twice
    mock_redis_client.xread.side_effect = [
        [("stream", [("1-0", {"objects": "[]", "timestamp": "0"})])],
        [("stream", [("1-0", {"objects": "[]", "timestamp": "0"})])],
        KeyboardInterrupt(),  # Break the loop
    ]

    callback = MagicMock(side_effect=Exception("Callback failed"))

    try:
        broker.subscribe_objects(callback)
    except (KeyboardInterrupt, Exception):
        pass

    calls = mock_redis_client.xread.call_args_list
    assert calls[0][0][0] == {broker.stream_name: "$"}
    # Now it should have updated last_id to "1-0"
    assert calls[1][0][0] == {broker.stream_name: "1-0"}


def test_image_streamer_subscription_loop_on_error(mock_redis_client):
    streamer = RedisImageStreamer()

    # Mock decoding failure or callback failure
    mock_redis_client.xread.side_effect = [
        [("stream", [("1-0", {"image_data": "invalid"})])],
        [("stream", [("1-0", {"image_data": "invalid"})])],
        KeyboardInterrupt(),
    ]

    callback = MagicMock()

    try:
        streamer.subscribe_variable_images(callback)
    except KeyboardInterrupt:
        pass

    calls = mock_redis_client.xread.call_args_list
    assert calls[1][0][0] == {streamer.stream_name: "1-0"}
