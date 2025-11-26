# tests/test_redis_robot_comm_extended.py
"""Extended test suite for redis_robot_comm package to improve code coverage."""

import pytest
import numpy as np
import cv2
import json
from unittest.mock import MagicMock, patch
from redis_robot_comm.redis_client import RedisMessageBroker
from redis_robot_comm.redis_image_streamer import RedisImageStreamer


# ============================================================================
# RedisMessageBroker Extended Tests
# ============================================================================


def test_publish_objects_with_camera_pose(monkeypatch):
    """Test publishing objects with camera pose metadata."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    objects = [{"id": "obj_1", "class_name": "cube"}]
    camera_pose = {"x": 1.0, "y": 2.0, "z": 3.0, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}
    mock_client.xadd.return_value = "1-0"

    msg_id = broker.publish_objects(objects, camera_pose=camera_pose)

    assert msg_id == "1-0"
    call_args = mock_client.xadd.call_args[0]
    assert "camera_pose" in call_args[1]
    assert json.loads(call_args[1]["camera_pose"]) == camera_pose


def test_publish_objects_verbose_mode(monkeypatch, capsys):
    """Test publishing objects with verbose output enabled."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    objects = [{"id": "obj_1"}, {"id": "obj_2"}]
    mock_client.xadd.return_value = "1-0"

    broker.publish_objects(objects)

    captured = capsys.readouterr()
    assert "Published 2 objects" in captured.out


def test_get_latest_objects_verbose_no_messages(monkeypatch, capsys):
    """Test verbose output when no messages are found."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.xrevrange.return_value = []
    result = broker.get_latest_objects()

    captured = capsys.readouterr()
    assert "No messages found" in captured.out
    assert result == []


def test_get_latest_objects_verbose_too_old(monkeypatch, capsys):
    """Test verbose output when objects are too old."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    old_time = 1000.0
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(old_time), "objects": "[]"})]

    with patch("time.time", return_value=old_time + 10.0):
        result = broker.get_latest_objects(max_age_seconds=2.0)

    captured = capsys.readouterr()
    assert "too old" in captured.out
    assert result == []


def test_get_latest_objects_verbose_success(monkeypatch, capsys):
    """Test verbose output on successful object retrieval."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    current_time = 1000.0
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=current_time + 0.5):
        broker.get_latest_objects()

    captured = capsys.readouterr()
    assert "Retrieved 1 fresh objects" in captured.out


def test_get_objects_in_timerange_verbose(monkeypatch, capsys):
    """Test verbose output for timerange query."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.xrange.return_value = [("1-0", {"objects": '[{"id": "obj_1"}]'}), ("2-0", {"objects": '[{"id": "obj_2"}]'})]

    broker.get_objects_in_timerange(1000.0, 2000.0)

    captured = capsys.readouterr()
    assert "Retrieved 2 objects from timerange" in captured.out


def test_get_objects_in_timerange_no_end_timestamp(monkeypatch):
    """Test timerange query with automatic end timestamp."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.xrange.return_value = [("1-0", {"objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=2000.0):
        result = broker.get_objects_in_timerange(1000.0)

    assert len(result) == 1
    # Verify the end timestamp was set to current time
    call_args = mock_client.xrange.call_args[0]
    assert call_args[2].startswith("2000000")  # 2000.0 * 1000


def test_subscribe_objects_callback_execution(monkeypatch):
    """Test that subscribe_objects correctly invokes callback."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Mock xread to return one message then raise KeyboardInterrupt
    mock_client.xread.side_effect = [
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

    broker.subscribe_objects(callback)

    assert len(callback_data) == 1
    assert callback_data[0]["objects"][0]["id"] == "obj_1"
    assert callback_data[0]["camera_pose"]["x"] == 1.0
    assert callback_data[0]["timestamp"] == 1000.0


def test_subscribe_objects_error_handling(monkeypatch, capsys):
    """Test error handling in subscribe_objects callback processing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Mock xread to return malformed data then KeyboardInterrupt
    mock_client.xread.side_effect = [
        [("detected_objects", [("1-0", {"objects": "invalid json", "timestamp": "1000.0"})])],
        KeyboardInterrupt(),
    ]

    def callback(data):
        pass

    broker.subscribe_objects(callback)

    captured = capsys.readouterr()
    assert "Error processing message" in captured.out


def test_subscribe_objects_general_exception(monkeypatch, capsys):
    """Test general exception handling in subscribe_objects."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.xread.side_effect = Exception("Connection error")

    def callback(data):
        pass

    broker.subscribe_objects(callback)

    captured = capsys.readouterr()
    assert "Error in subscribe_objects" in captured.out


def test_clear_stream_verbose(monkeypatch, capsys):
    """Test verbose output when clearing stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.delete.return_value = 5
    result = broker.clear_stream()

    captured = capsys.readouterr()
    assert "Cleared detected_objects stream" in captured.out
    assert result == 5


def test_get_stream_info_verbose(monkeypatch, capsys):
    """Test verbose output when getting stream info."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.xinfo_stream.return_value = {"length": 10}
    broker.get_stream_info()

    captured = capsys.readouterr()
    assert "Stream info" in captured.out


def test_test_connection_verbose(monkeypatch, capsys):
    """Test verbose output for connection test."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.ping.return_value = True
    broker.test_connection()

    captured = capsys.readouterr()
    assert "Redis connection test: OK" in captured.out


# ============================================================================
# RedisImageStreamer Extended Tests
# ============================================================================


def test_publish_image_invalid_input():
    """Test that publish_image raises ValueError for invalid input."""
    streamer = RedisImageStreamer()

    with pytest.raises(ValueError, match="must be a non‑empty NumPy array"):
        streamer.publish_image(None)

    with pytest.raises(ValueError, match="must be a non‑empty NumPy array"):
        streamer.publish_image(np.array([]))

    with pytest.raises(ValueError, match="must be a non‑empty NumPy array"):
        streamer.publish_image([1, 2, 3])


def test_publish_image_grayscale(monkeypatch):
    """Test publishing a grayscale image."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    image = np.zeros((100, 100), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    msg_id = streamer.publish_image(image)

    assert msg_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    assert call_args["channels"] == "1"


def test_publish_image_custom_quality(monkeypatch):
    """Test publishing image with custom JPEG quality."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    msg_id = streamer.publish_image(image, quality=95)

    assert msg_id == "1-0"


def test_publish_image_custom_maxlen(monkeypatch):
    """Test publishing image with custom maxlen parameter."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    streamer.publish_image(image, maxlen=10)

    # Verify maxlen was passed to xadd
    call_kwargs = mock_client.xadd.call_args[1]
    assert call_kwargs["maxlen"] == 10


def test_publish_image_verbose(monkeypatch, capsys):
    """Test verbose output when publishing image."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()
    streamer.verbose = True

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    streamer.publish_image(image)

    captured = capsys.readouterr()
    assert "Published 100x100 image" in captured.out


def test_get_latest_image_with_error(monkeypatch, capsys):
    """Test error handling in get_latest_image."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()
    streamer.verbose = True

    mock_client.xrevrange.side_effect = Exception("Redis error")
    result = streamer.get_latest_image()

    captured = capsys.readouterr()
    assert "Error getting latest image" in captured.out
    assert result is None


def test_subscribe_variable_images_callback(monkeypatch):
    """Test subscribe_variable_images callback execution."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    # Create a test image
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    import base64

    encoded = base64.b64encode(buffer).decode("utf-8")

    # Mock xread to return one message then raise KeyboardInterrupt
    mock_client.xread.side_effect = [
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

    streamer.subscribe_variable_images(callback)

    assert len(callback_data) == 1
    img, meta, info = callback_data[0]
    assert isinstance(img, np.ndarray)
    assert info["width"] == 10
    assert info["height"] == 10


def test_subscribe_variable_images_decode_failure(monkeypatch, capsys):
    """Test subscribe_variable_images with decode failure."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()
    streamer.verbose = True

    # Mock xread to return invalid data then KeyboardInterrupt
    mock_client.xread.side_effect = [[("robot_camera", [("1-0", {"invalid": "data"})])], KeyboardInterrupt()]

    callback_called = []

    def callback(img, meta, info):
        callback_called.append(True)

    streamer.subscribe_variable_images(callback)

    # Callback should not be called due to decode failure
    assert len(callback_called) == 0


def test_subscribe_variable_images_general_error(monkeypatch, capsys):
    """Test general error handling in subscribe_variable_images."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()
    streamer.verbose = True

    mock_client.xread.side_effect = [Exception("Connection error"), KeyboardInterrupt()]

    def callback(img, meta, info):
        pass

    streamer.subscribe_variable_images(callback)

    captured = capsys.readouterr()
    assert "Error in image subscription" in captured.out


def test_decode_variable_image_jpeg_failure(monkeypatch, capsys):
    """Test JPEG decode failure in _decode_variable_image."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()
    streamer.verbose = True

    import base64

    fields = {
        "width": "10",
        "height": "10",
        "channels": "3",
        "format": "jpeg",
        "dtype": "uint8",
        "image_data": base64.b64encode(b"not a valid jpeg").decode("utf-8"),
    }

    result = streamer._decode_variable_image(fields)

    captured = capsys.readouterr()
    assert "Error decoding variable image" in captured.out
    assert result is None


def test_decode_variable_image_with_metadata(monkeypatch):
    """Test decoding image with metadata."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

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

    result = streamer._decode_variable_image(fields)

    assert result is not None
    img, meta = result
    assert meta["robot"] == "arm1"
    assert meta["frame"] == 42


def test_decode_variable_image_missing_fields(monkeypatch, capsys):
    """Test decode with missing required fields."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()
    streamer.verbose = True

    fields = {"width": "10"}  # Missing required fields

    result = streamer._decode_variable_image(fields)

    captured = capsys.readouterr()
    assert "Error decoding variable image" in captured.out
    assert result is None


def test_get_stream_stats_success(monkeypatch):
    """Test successful retrieval of stream statistics."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    mock_client.xinfo_stream.return_value = {"length": 10, "first-entry": ("1-0", {}), "last-entry": ("10-0", {})}

    stats = streamer.get_stream_stats()

    assert stats["total_messages"] == 10
    assert stats["first_entry_id"] == "1-0"
    assert stats["last_entry_id"] == "10-0"


def test_custom_stream_name(monkeypatch):
    """Test creating streamer with custom stream name."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)

    custom_name = "my_custom_camera"
    streamer = RedisImageStreamer(stream_name=custom_name)

    assert streamer.stream_name == custom_name


def test_custom_redis_connection(monkeypatch):
    """Test creating clients with custom Redis connection parameters."""
    mock_redis = MagicMock()
    monkeypatch.setattr("redis.Redis", mock_redis)

    RedisMessageBroker(host="custom-host", port=6380, db=2)
    mock_redis.assert_called_with(host="custom-host", port=6380, db=2, decode_responses=True)

    RedisImageStreamer(host="another-host", port=6381)
    # Check that it was called with custom parameters
    calls = mock_redis.call_args_list
    assert any(call[1].get("host") == "another-host" for call in calls)


# ============================================================================
# Integration-style Tests
# ============================================================================


def test_end_to_end_object_flow(monkeypatch):
    """Test complete flow: publish -> retrieve objects."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Publish
    objects = [
        {"id": "obj_1", "class_name": "cube", "confidence": 0.95},
        {"id": "obj_2", "class_name": "sphere", "confidence": 0.87},
    ]
    camera_pose = {"x": 0.5, "y": 1.0, "z": 2.0}
    mock_client.xadd.return_value = "1-0"

    publish_result = broker.publish_objects(objects, camera_pose)
    assert publish_result == "1-0"

    # Retrieve
    current_time = 1000.0
    mock_client.xrevrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "objects": json.dumps(objects), "camera_pose": json.dumps(camera_pose)})
    ]

    with patch("time.time", return_value=current_time + 0.5):
        retrieved = broker.get_latest_objects()

    assert len(retrieved) == 2
    assert retrieved[0]["id"] == "obj_1"
    assert retrieved[1]["confidence"] == 0.87


def test_end_to_end_image_flow(monkeypatch):
    """Test complete flow: publish -> retrieve image."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    # Create and publish image
    test_image = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
    metadata = {"frame": 100, "robot": "arm1"}
    mock_client.xadd.return_value = "1-0"

    publish_result = streamer.publish_image(test_image, metadata=metadata)
    assert publish_result == "1-0"

    # Mock retrieval
    _, buffer = cv2.imencode(".jpg", test_image)
    import base64

    encoded = base64.b64encode(buffer).decode("utf-8")

    mock_client.xrevrange.return_value = [
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

    result = streamer.get_latest_image()
    assert result is not None

    img, meta = result
    assert img.shape[:2] == (50, 60)  # Height, width
    assert meta["frame"] == 100
    assert meta["robot"] == "arm1"
