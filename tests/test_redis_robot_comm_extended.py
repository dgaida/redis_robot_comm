"""Extended test suite for redis_robot_comm package to improve code coverage."""

import numpy as np
import cv2
import json
import logging
from unittest.mock import MagicMock, patch

# ============================================================================
# RedisMessageBroker Extended Tests
# ============================================================================


def test_publish_objects_with_camera_pose(message_broker, mock_redis_client, sample_objects):
    """Test publishing objects with camera pose metadata."""
    camera_pose = {"x": 1.0, "y": 2.0, "z": 3.0, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}
    mock_redis_client.xadd.return_value = "1-0"

    msg_id = message_broker.publish_objects(sample_objects, camera_pose=camera_pose)

    assert msg_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0]
    assert "camera_pose" in call_args[1]
    assert json.loads(call_args[1]["camera_pose"]) == camera_pose


def test_publish_objects_verbose_mode(message_broker, mock_redis_client, caplog):
    """Test publishing objects with verbose output enabled."""
    message_broker.verbose = True

    objects = [{"id": "obj_1", "class_name": "cube"}, {"id": "obj_2", "class_name": "cube"}]
    mock_redis_client.xadd.return_value = "1-0"

    with caplog.at_level(logging.INFO):
        message_broker.publish_objects(objects)

    assert "Published 2 objects" in caplog.text


def test_get_latest_objects_verbose_no_messages(message_broker, mock_redis_client, caplog):
    """Test verbose output when no messages are found."""
    message_broker.verbose = True

    mock_redis_client.xrevrange.return_value = []
    with caplog.at_level(logging.DEBUG):
        result = message_broker.get_latest_objects()

    assert "No messages found" in caplog.text
    assert result == []


def test_get_latest_objects_verbose_too_old(message_broker, mock_redis_client, caplog):
    """Test verbose output when objects are too old."""
    message_broker.verbose = True

    old_time = 1000.0
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(old_time), "objects": "[]"})]

    with patch("time.time", return_value=old_time + 10.0):
        with caplog.at_level(logging.DEBUG):
            result = message_broker.get_latest_objects(max_age_seconds=2.0)

    assert "too old" in caplog.text
    assert result == []


def test_get_latest_objects_verbose_success(message_broker, mock_redis_client, caplog):
    """Test verbose output on successful object retrieval."""
    message_broker.verbose = True

    current_time = 1000.0
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=current_time + 0.5):
        with caplog.at_level(logging.INFO):
            message_broker.get_latest_objects()

    assert "Retrieved 1 fresh objects" in caplog.text


def test_get_objects_in_timerange_verbose(message_broker, mock_redis_client, caplog):
    """Test verbose output for timerange query."""
    message_broker.verbose = True

    mock_redis_client.xrange.return_value = [
        ("1-0", {"objects": '[{"id": "obj_1"}]'}),
        ("2-0", {"objects": '[{"id": "obj_2"}]'}),
    ]

    with caplog.at_level(logging.INFO):
        message_broker.get_objects_in_timerange(1000.0, 2000.0)

    assert "Retrieved 2 objects from timerange" in caplog.text


def test_get_objects_in_timerange_no_end_timestamp(message_broker, mock_redis_client):
    """Test timerange query with automatic end timestamp."""
    mock_redis_client.xrange.return_value = [("1-0", {"objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=2000.0):
        result = message_broker.get_objects_in_timerange(1000.0)

    assert len(result) == 1
    # Verify the end timestamp was set to current time
    call_args = mock_redis_client.xrange.call_args[0]
    assert call_args[2].startswith("2000000")  # 2000.0 * 1000


def test_subscribe_objects_callback_execution(message_broker, mock_redis_client):
    """Test that subscribe_objects correctly invokes callback."""
    # Mock xread to return one message then raise KeyboardInterrupt
    mock_redis_client.xread.side_effect = [
        [
            (
                "detected_objects",
                [("1-0", {"objects": '[{"id": "obj_1"}]', "camera_pose": '{"x": 1.0}', "timestamp": "1000.0"})],
            )
        ],
        KeyboardInterrupt(),
    ]

    callback_data = []

    def callback(data):
        callback_data.append(data)

    message_broker.subscribe_objects(callback)

    assert len(callback_data) == 1
    assert callback_data[0]["objects"][0]["id"] == "obj_1"
    assert callback_data[0]["camera_pose"]["x"] == 1.0
    assert callback_data[0]["timestamp"] == 1000.0


def test_subscribe_objects_error_handling(message_broker, mock_redis_client, caplog):
    """Test error handling in subscribe_objects callback processing."""
    # Mock xread to return malformed data then KeyboardInterrupt
    mock_redis_client.xread.side_effect = [
        [("detected_objects", [("1-0", {"objects": "invalid json", "timestamp": "1000.0"})])],
        KeyboardInterrupt(),
    ]

    def callback(data):
        pass

    with caplog.at_level(logging.ERROR):
        message_broker.subscribe_objects(callback)

    assert "Error processing message" in caplog.text


def test_subscribe_objects_general_exception(message_broker, mock_redis_client, caplog):
    """Test general exception handling in subscribe_objects."""
    mock_redis_client.xread.side_effect = Exception("Connection error")

    def callback(data):
        pass

    with caplog.at_level(logging.ERROR):
        message_broker.subscribe_objects(callback)

    assert "Unexpected error in subscribe_objects" in caplog.text


def test_clear_stream_verbose(message_broker, mock_redis_client, caplog):
    """Test verbose output when clearing stream."""
    message_broker.verbose = True

    mock_redis_client.delete.return_value = 5
    with caplog.at_level(logging.INFO):
        result = message_broker.clear_stream()

    assert f"Cleared {message_broker.stream_name} stream" in caplog.text
    assert result is True


def test_get_stream_info_verbose(message_broker, mock_redis_client, caplog):
    """Test verbose output when getting stream info."""
    message_broker.verbose = True

    mock_redis_client.xinfo_stream.return_value = {"length": 10}
    with caplog.at_level(logging.INFO):
        message_broker.get_stream_info()

    assert "Stream info" in caplog.text


def test_test_connection_verbose(message_broker, mock_redis_client, caplog):
    """Test verbose output for connection test."""
    message_broker.verbose = True

    mock_redis_client.ping.return_value = True
    with caplog.at_level(logging.INFO):
        message_broker.test_connection()

    assert "Redis connection test: OK" in caplog.text


# ============================================================================
# RedisImageStreamer Extended Tests
# ============================================================================


def test_publish_image_invalid_input(image_streamer):
    """Test that publish_image raises ValueError for invalid input."""
    from redis_robot_comm.exceptions import InvalidImageError
    import pytest

    with pytest.raises(InvalidImageError, match="must be a NumPy array"):
        image_streamer.publish_image(None)

    with pytest.raises(InvalidImageError, match="array is empty"):
        image_streamer.publish_image(np.array([]))

    with pytest.raises(InvalidImageError, match="must be a NumPy array"):
        image_streamer.publish_image([1, 2, 3])


def test_publish_image_grayscale(image_streamer, mock_redis_client):
    """Test publishing a grayscale image."""
    image = np.zeros((100, 100), dtype=np.uint8)
    mock_redis_client.xadd.return_value = "1-0"

    msg_id = image_streamer.publish_image(image)

    assert msg_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert call_args["channels"] == "1"


def test_publish_image_custom_quality(image_streamer, mock_redis_client):
    """Test publishing image with custom JPEG quality."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_redis_client.xadd.return_value = "1-0"

    msg_id = image_streamer.publish_image(image, quality=95)

    assert msg_id == "1-0"


def test_publish_image_custom_maxlen(image_streamer, mock_redis_client):
    """Test publishing image with custom maxlen parameter."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_redis_client.xadd.return_value = "1-0"

    image_streamer.publish_image(image, maxlen=10)

    # Verify maxlen was passed to xadd
    call_kwargs = mock_redis_client.xadd.call_args[1]
    assert call_kwargs["maxlen"] == 10


def test_publish_image_verbose(image_streamer, mock_redis_client, caplog):
    """Test verbose output when publishing image."""
    image_streamer.verbose = True

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_redis_client.xadd.return_value = "1-0"

    with caplog.at_level(logging.INFO):
        image_streamer.publish_image(image)

    assert "Published 100x100 image" in caplog.text


def test_get_latest_image_with_error(image_streamer, mock_redis_client, caplog):
    """Test error handling in get_latest_image."""
    image_streamer.verbose = True

    mock_redis_client.xrevrange.side_effect = Exception("Redis error")
    with caplog.at_level(logging.ERROR):
        result = image_streamer.get_latest_image()

    assert "Unexpected error getting latest image" in caplog.text
    assert result is None


def test_subscribe_variable_images_callback(image_streamer, mock_redis_client):
    """Test subscribe_variable_images callback execution."""
    # Create a test image
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    import base64

    encoded = base64.b64encode(buffer).decode("utf-8")

    # Mock xread to return one message then raise KeyboardInterrupt
    mock_redis_client.xread.side_effect = [
        [
            (
                "robot_camera",
                [
                    (
                        "1-0",
                        {
                            "width": "10",
                            "height": "10",
                            "channels": "3",
                            "format": "jpeg",
                            "image_data": encoded,
                            "timestamp": "1000.0",
                            "compressed_size": "100",
                            "original_size": "300",
                            "dtype": "uint8",
                        },
                    )
                ],
            )
        ],
        KeyboardInterrupt(),
    ]

    callback_data = []

    def callback(img, meta, info):
        callback_data.append((img, meta, info))

    image_streamer.subscribe_variable_images(callback)

    assert len(callback_data) == 1
    img, meta, info = callback_data[0]
    assert isinstance(img, np.ndarray)
    assert info["width"] == 10
    assert info["height"] == 10


def test_subscribe_variable_images_decode_failure(image_streamer, mock_redis_client, caplog):
    """Test subscribe_variable_images with decode failure."""
    image_streamer.verbose = True

    # Mock xread to return invalid data then KeyboardInterrupt
    mock_redis_client.xread.side_effect = [[("robot_camera", [("1-0", {"invalid": "data"})])], KeyboardInterrupt()]

    callback_called = []

    def callback(img, meta, info):
        callback_called.append(True)

    with caplog.at_level(logging.ERROR):
        image_streamer.subscribe_variable_images(callback)

    # Callback should not be called due to decode failure
    assert len(callback_called) == 0


def test_subscribe_variable_images_general_error(image_streamer, mock_redis_client, caplog):
    """Test general error handling in subscribe_variable_images."""
    image_streamer.verbose = True

    mock_redis_client.xread.side_effect = [Exception("Connection error"), KeyboardInterrupt()]

    def callback(img, meta, info):
        pass

    with caplog.at_level(logging.ERROR):
        image_streamer.subscribe_variable_images(callback)

    assert "Unexpected error in image subscription" in caplog.text


def test_decode_variable_image_jpeg_failure(image_streamer, mock_redis_client, caplog):
    """Test JPEG decode failure in _decode_variable_image."""
    image_streamer.verbose = True

    import base64

    fields = {
        "width": "10",
        "height": "10",
        "channels": "3",
        "format": "jpeg",
        "dtype": "uint8",
        "image_data": base64.b64encode(b"not a valid jpeg").decode("utf-8"),
    }

    with caplog.at_level(logging.ERROR):
        result = image_streamer._decode_variable_image(fields)

    assert "Error decoding variable image" in caplog.text
    assert result is None


def test_decode_variable_image_with_metadata(image_streamer, mock_redis_client):
    """Test decoding image with metadata."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    import base64

    fields = {
        "width": "10",
        "height": "10",
        "channels": "3",
        "format": "raw",
        "dtype": "uint8",
        "image_data": base64.b64encode(image.tobytes()).decode("utf-8"),
        "metadata": json.dumps({"robot": "arm1", "frame": 42}),
    }

    result = image_streamer._decode_variable_image(fields)

    assert result is not None
    img, meta = result
    assert meta["robot"] == "arm1"
    assert meta["frame"] == 42


def test_decode_variable_image_missing_fields(image_streamer, mock_redis_client, caplog):
    """Test decode with missing required fields."""
    image_streamer.verbose = True

    fields = {"width": "10"}  # Missing required fields

    with caplog.at_level(logging.ERROR):
        result = image_streamer._decode_variable_image(fields)

    assert "Error decoding variable image" in caplog.text
    assert result is None


def test_get_stream_stats_success(image_streamer, mock_redis_client):
    """Test successful retrieval of stream statistics."""
    mock_redis_client.xinfo_stream.return_value = {"length": 10, "first-entry": ("1-0", {}), "last-entry": ("10-0", {})}

    stats = image_streamer.get_stream_stats()

    assert stats["total_messages"] == 10
    assert stats["first_entry_id"] == "1-0"
    assert stats["last_entry_id"] == "10-0"


def test_custom_stream_name(mock_redis_client):
    """Test creating streamer with custom stream name."""
    from redis_robot_comm.redis_image_streamer import RedisImageStreamer

    custom_name = "my_custom_camera"
    streamer = RedisImageStreamer(stream_name=custom_name)

    assert streamer.stream_name == custom_name


def test_custom_redis_connection(monkeypatch):
    """Test creating clients with custom Redis connection parameters."""
    from redis_robot_comm.redis_client import RedisMessageBroker
    from redis_robot_comm.redis_image_streamer import RedisImageStreamer
    import redis

    mock_redis = MagicMock()
    monkeypatch.setattr(redis, "Redis", mock_redis)
    mock_redis.return_value.ping.return_value = True

    RedisMessageBroker(host="custom-host", port=6380, db=2)
    # Verify relevant parameters were passed
    args, kwargs = mock_redis.call_args
    assert kwargs.get("host") == "custom-host"
    assert kwargs.get("port") == 6380
    assert kwargs.get("db") == 2
    assert kwargs.get("decode_responses") is True

    RedisImageStreamer(host="another-host", port=6381)
    # Check that it was called with custom parameters
    calls = mock_redis.call_args_list
    assert any(call[1].get("host") == "another-host" for call in calls)


# ============================================================================
# Integration-style Tests
# ============================================================================


def test_end_to_end_object_flow(message_broker, mock_redis_client, sample_objects):
    """Test complete flow: publish -> retrieve objects."""
    # Publish
    camera_pose = {"x": 0.5, "y": 1.0, "z": 2.0}
    mock_redis_client.xadd.return_value = "1-0"

    publish_result = message_broker.publish_objects(sample_objects, camera_pose)
    assert publish_result == "1-0"

    # Retrieve
    current_time = 1000.0
    mock_redis_client.xrevrange.return_value = [
        (
            "1-0",
            {"timestamp": str(current_time), "objects": json.dumps(sample_objects), "camera_pose": json.dumps(camera_pose)},
        )
    ]

    with patch("time.time", return_value=current_time + 0.5):
        retrieved = message_broker.get_latest_objects()

    assert len(retrieved) == 1
    assert retrieved[0]["id"] == "obj_1"


def test_end_to_end_image_flow(image_streamer, mock_redis_client):
    """Test complete flow: publish -> retrieve image."""
    # Create and publish image
    test_image = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
    metadata = {"frame": 100, "robot": "arm1"}
    mock_redis_client.xadd.return_value = "1-0"

    publish_result = image_streamer.publish_image(test_image, metadata=metadata)
    assert publish_result == "1-0"

    # Mock retrieval
    _, buffer = cv2.imencode(".jpg", test_image)
    import base64

    encoded = base64.b64encode(buffer).decode("utf-8")

    mock_redis_client.xrevrange.return_value = [
        (
            "1-0",
            {
                "width": "60",
                "height": "50",
                "channels": "3",
                "format": "jpeg",
                "dtype": "uint8",
                "image_data": encoded,
                "timestamp": "1000.0",
                "compressed_size": str(len(buffer)),
                "original_size": str(test_image.nbytes),
                "metadata": json.dumps(metadata),
            },
        )
    ]

    result = image_streamer.get_latest_image()
    assert result is not None

    img, meta = result
    assert img.shape[:2] == (50, 60)  # Height, width
    assert meta["frame"] == 100
    assert meta["robot"] == "arm1"
