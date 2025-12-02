# tests/test_redis_label_manager.py
"""Comprehensive test suite for RedisLabelManager class."""

import json
from unittest.mock import MagicMock, patch
from redis_robot_comm.redis_label_manager import RedisLabelManager


# ============================================================================
# Test: Initialization
# ============================================================================


def test_init_default_parameters(monkeypatch):
    """Test initialization with default parameters."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)

    manager = RedisLabelManager()

    assert manager.stream_name == "detectable_labels"
    assert manager.verbose is False


def test_init_custom_parameters(monkeypatch):
    """Test initialization with custom parameters."""
    mock_redis = MagicMock()
    monkeypatch.setattr("redis.Redis", mock_redis)

    manager = RedisLabelManager(host="custom-host", port=6380, stream_name="custom_labels")

    assert manager.stream_name == "custom_labels"
    mock_redis.assert_called_with(host="custom-host", port=6380, decode_responses=True)


# ============================================================================
# Test: publish_labels()
# ============================================================================


def test_publish_labels_success(monkeypatch):
    """Test successful label publishing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    labels = ["cube", "sphere", "cylinder"]
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_labels(labels)

    assert stream_id == "1-0"
    mock_client.xadd.assert_called_once()
    call_args = mock_client.xadd.call_args[0]
    assert call_args[0] == "detectable_labels"
    assert json.loads(call_args[1]["labels"]) == labels
    assert call_args[1]["label_count"] == "3"


def test_publish_labels_with_metadata(monkeypatch):
    """Test publishing labels with metadata."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    labels = ["cube", "sphere"]
    metadata = {"model_id": "yolo-v8", "version": "1.0"}
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_labels(labels, metadata=metadata)

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    assert "metadata" in call_args
    assert json.loads(call_args["metadata"]) == metadata


def test_publish_labels_empty_list(monkeypatch):
    """Test publishing empty label list."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    labels = []
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_labels(labels)

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    assert call_args["label_count"] == "0"


def test_publish_labels_verbose_mode(monkeypatch, capsys):
    """Test verbose output when publishing labels."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    labels = ["cube", "sphere"]
    mock_client.xadd.return_value = "1-0"

    manager.publish_labels(labels)

    captured = capsys.readouterr()
    assert "Published 2 labels to Redis" in captured.out


def test_publish_labels_with_error(monkeypatch, capsys):
    """Test error handling when publishing labels fails."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    mock_client.xadd.side_effect = Exception("Redis connection error")
    labels = ["cube"]

    result = manager.publish_labels(labels)

    assert result is None
    captured = capsys.readouterr()
    assert "Error publishing labels" in captured.out


def test_publish_labels_maxlen_parameter(monkeypatch):
    """Test that maxlen=1 is passed to xadd."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    labels = ["cube"]
    mock_client.xadd.return_value = "1-0"

    manager.publish_labels(labels)

    call_kwargs = mock_client.xadd.call_args[1]
    assert call_kwargs["maxlen"] == 1


# ============================================================================
# Test: get_latest_labels()
# ============================================================================


def test_get_latest_labels_success(monkeypatch):
    """Test successful retrieval of latest labels."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    current_time = 1000.0
    labels = ["cube", "sphere"]
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        result = manager.get_latest_labels()

    assert result == labels


def test_get_latest_labels_no_messages(monkeypatch, capsys):
    """Test when no labels are found in stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    mock_client.xrevrange.return_value = []

    result = manager.get_latest_labels()

    assert result is None
    captured = capsys.readouterr()
    assert "No labels found in stream" in captured.out


def test_get_latest_labels_too_old(monkeypatch, capsys):
    """Test when labels are too old."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    old_time = 1000.0
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(old_time), "labels": '["cube"]'})]

    with patch("time.time", return_value=old_time + 10.0):
        result = manager.get_latest_labels(timeout_seconds=5.0)

    assert result is None
    captured = capsys.readouterr()
    assert "Labels too old" in captured.out


def test_get_latest_labels_verbose_success(monkeypatch, capsys):
    """Test verbose output on successful label retrieval."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    current_time = 1000.0
    labels = ["cube", "sphere", "cylinder"]
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        manager.get_latest_labels()

    captured = capsys.readouterr()
    assert "Retrieved 3 labels" in captured.out


def test_get_latest_labels_with_error(monkeypatch, capsys):
    """Test error handling in get_latest_labels."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    mock_client.xrevrange.side_effect = Exception("Redis error")

    result = manager.get_latest_labels()

    assert result is None
    captured = capsys.readouterr()
    assert "Error getting labels" in captured.out


# ============================================================================
# Test: add_label()
# ============================================================================


def test_add_label_to_empty_list(monkeypatch):
    """Test adding label when no existing labels found."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    # No existing labels
    mock_client.xrevrange.return_value = []
    mock_client.xadd.return_value = "1-0"

    result = manager.add_label("cube")

    assert result is True
    # Verify publish was called with the new label
    call_args = mock_client.xadd.call_args[0][1]
    labels = json.loads(call_args["labels"])
    assert "cube" in labels


def test_add_label_to_existing_list(monkeypatch):
    """Test adding label to existing list."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    # Existing labels
    current_time = 1000.0
    existing_labels = ["sphere", "cylinder"]
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(existing_labels)})]
    mock_client.xadd.return_value = "2-0"

    with patch("time.time", return_value=current_time + 1.0):
        result = manager.add_label("cube")

    assert result is True
    # Verify new label was added
    call_args = mock_client.xadd.call_args[0][1]
    labels = json.loads(call_args["labels"])
    assert "cube" in labels
    assert "sphere" in labels
    assert "cylinder" in labels


def test_add_label_already_exists(monkeypatch, capsys):
    """Test adding label that already exists."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    current_time = 1000.0
    existing_labels = ["cube", "sphere"]
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(existing_labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        result = manager.add_label("cube")

    assert result is False
    captured = capsys.readouterr()
    assert "Label already exists: cube" in captured.out


def test_add_label_case_insensitive(monkeypatch):
    """Test that add_label is case-insensitive."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    current_time = 1000.0
    existing_labels = ["Cube", "Sphere"]
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(existing_labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        result = manager.add_label("CUBE")

    assert result is False  # Should detect it already exists


def test_add_label_with_metadata(monkeypatch):
    """Test that add_label includes metadata about the action."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    mock_client.xrevrange.return_value = []
    mock_client.xadd.return_value = "1-0"

    manager.add_label("cube")

    call_args = mock_client.xadd.call_args[0][1]
    metadata = json.loads(call_args["metadata"])
    assert metadata["action"] == "add"
    assert metadata["added_label"] == "cube"


def test_add_label_with_error(monkeypatch, capsys):
    """Test error handling in add_label."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    # Simulate Redis failure for both read and write operations
    mock_client.xrevrange.side_effect = Exception("Redis error")
    mock_client.xadd.side_effect = Exception("Redis error")

    result = manager.add_label("cube")

    assert result is False
    captured = capsys.readouterr()
    assert "Error" in captured.out


# ============================================================================
# Test: subscribe_to_label_updates()
# ============================================================================


def test_subscribe_to_label_updates_callback_execution(monkeypatch):
    """Test that subscribe correctly invokes callback."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    labels = ["cube", "sphere"]
    metadata = {"action": "add"}

    # Mock xread to return one message then raise KeyboardInterrupt
    mock_client.xread.side_effect = [
        [("detectable_labels", [("1-0", {"labels": json.dumps(labels), "metadata": json.dumps(metadata)})])],
        KeyboardInterrupt(),
    ]

    callback_data = []

    def callback(lbls, meta):
        callback_data.append((lbls, meta))

    manager.subscribe_to_label_updates(callback)

    assert len(callback_data) == 1
    assert callback_data[0][0] == labels
    assert callback_data[0][1] == metadata


def test_subscribe_to_label_updates_without_metadata(monkeypatch):
    """Test subscribe when metadata field is missing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    labels = ["cube"]

    mock_client.xread.side_effect = [[("detectable_labels", [("1-0", {"labels": json.dumps(labels)})])], KeyboardInterrupt()]

    callback_data = []

    def callback(lbls, meta):
        callback_data.append((lbls, meta))

    manager.subscribe_to_label_updates(callback)

    assert len(callback_data) == 1
    assert callback_data[0][0] == labels
    assert callback_data[0][1] == {}


def test_subscribe_to_label_updates_error_handling(monkeypatch, capsys):
    """Test error handling in callback processing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    # Mock xread to return malformed data
    mock_client.xread.side_effect = [[("detectable_labels", [("1-0", {"labels": "invalid json"})])], KeyboardInterrupt()]

    def callback(lbls, meta):
        pass

    manager.subscribe_to_label_updates(callback)

    captured = capsys.readouterr()
    assert "Error processing label update" in captured.out


def test_subscribe_to_label_updates_general_exception(monkeypatch, capsys):
    """Test general exception handling in subscribe."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    mock_client.xread.side_effect = Exception("Connection error")

    def callback(lbls, meta):
        pass

    manager.subscribe_to_label_updates(callback)

    captured = capsys.readouterr()
    assert "Error in label subscription" in captured.out


# ============================================================================
# Test: clear_stream()
# ============================================================================


def test_clear_stream_success(monkeypatch):
    """Test successful stream clearing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()

    mock_client.delete.return_value = 1

    result = manager.clear_stream()

    assert result is True
    mock_client.delete.assert_called_with("detectable_labels")


def test_clear_stream_verbose(monkeypatch, capsys):
    """Test verbose output when clearing stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    mock_client.delete.return_value = 1

    manager.clear_stream()

    captured = capsys.readouterr()
    assert "Cleared labels stream" in captured.out


def test_clear_stream_with_error(monkeypatch, capsys):
    """Test error handling when clearing stream fails."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisLabelManager()
    manager.verbose = True

    mock_client.delete.side_effect = Exception("Redis error")

    result = manager.clear_stream()

    assert result is False
    captured = capsys.readouterr()
    assert "Error clearing stream" in captured.out
