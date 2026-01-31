"""Comprehensive test suite for RedisLabelManager class."""

import json
import logging
from unittest.mock import MagicMock, patch


# ============================================================================
# Test: Initialization
# ============================================================================


def test_init_default_parameters(label_manager):
    """Test initialization with default parameters."""
    assert label_manager.stream_name == "detectable_labels"
    assert label_manager.verbose is False


def test_init_custom_parameters(monkeypatch):
    """Test initialization with custom parameters."""
    from redis_robot_comm.redis_label_manager import RedisLabelManager
    import redis
    mock_redis = MagicMock()
    monkeypatch.setattr(redis, "Redis", mock_redis)
    mock_redis.return_value.ping.return_value = True

    manager = RedisLabelManager(host="custom-host", port=6380, stream_name="custom_labels")

    assert manager.stream_name == "custom_labels"
    # Verify relevant parameters were passed
    args, kwargs = mock_redis.call_args
    assert kwargs.get('host') == "custom-host"
    assert kwargs.get('port') == 6380
    assert kwargs.get('decode_responses') is True
    mock_redis.return_value.ping.assert_called_once()


# ============================================================================
# Test: publish_labels()
# ============================================================================


def test_publish_labels_success(label_manager, mock_redis_client):
    """Test successful label publishing."""
    labels = ["cube", "sphere", "cylinder"]
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = label_manager.publish_labels(labels)

    assert stream_id == "1-0"
    mock_redis_client.xadd.assert_called_once()
    call_args = mock_redis_client.xadd.call_args[0]
    assert call_args[0] == "detectable_labels"
    assert json.loads(call_args[1]["labels"]) == labels
    assert call_args[1]["label_count"] == "3"


def test_publish_labels_with_metadata(label_manager, mock_redis_client):
    """Test publishing labels with metadata."""
    labels = ["cube", "sphere"]
    metadata = {"model_id": "yolo-v8", "version": "1.0"}
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = label_manager.publish_labels(labels, metadata=metadata)

    assert stream_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert "metadata" in call_args
    assert json.loads(call_args["metadata"]) == metadata


def test_publish_labels_empty_list(label_manager, mock_redis_client):
    """Test publishing empty label list."""
    labels = []
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = label_manager.publish_labels(labels)

    assert stream_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert call_args["label_count"] == "0"


def test_publish_labels_verbose_mode(label_manager, mock_redis_client, caplog):
    """Test verbose output when publishing labels."""
    label_manager.verbose = True

    labels = ["cube", "sphere"]
    mock_redis_client.xadd.return_value = "1-0"

    with caplog.at_level(logging.INFO):
        label_manager.publish_labels(labels)

    assert "Published 2 labels to Redis" in caplog.text


def test_publish_labels_with_error(label_manager, mock_redis_client, caplog):
    """Test error handling when publishing labels fails."""
    label_manager.verbose = True

    mock_redis_client.xadd.side_effect = Exception("Redis connection error")
    labels = ["cube"]

    with caplog.at_level(logging.ERROR):
        result = label_manager.publish_labels(labels)

    assert result is None
    assert "Unexpected error publishing labels" in caplog.text


def test_publish_labels_maxlen_parameter(label_manager, mock_redis_client):
    """Test that maxlen=1 is passed to xadd."""
    labels = ["cube"]
    mock_redis_client.xadd.return_value = "1-0"

    label_manager.publish_labels(labels)

    call_kwargs = mock_redis_client.xadd.call_args[1]
    assert call_kwargs["maxlen"] == 1


# ============================================================================
# Test: get_latest_labels()
# ============================================================================


def test_get_latest_labels_success(label_manager, mock_redis_client):
    """Test successful retrieval of latest labels."""
    current_time = 1000.0
    labels = ["cube", "sphere"]
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        result = label_manager.get_latest_labels()

    assert result == labels


def test_get_latest_labels_no_messages(label_manager, mock_redis_client, caplog):
    """Test when no labels are found in stream."""
    label_manager.verbose = True

    mock_redis_client.xrevrange.return_value = []

    with caplog.at_level(logging.DEBUG):
        result = label_manager.get_latest_labels()

    assert result is None
    assert f"No labels found in {label_manager.stream_name}" in caplog.text


def test_get_latest_labels_too_old(label_manager, mock_redis_client, caplog):
    """Test when labels are too old."""
    label_manager.verbose = True

    old_time = 1000.0
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(old_time), "labels": '["cube"]'})]

    with patch("time.time", return_value=old_time + 10.0):
        with caplog.at_level(logging.DEBUG):
            result = label_manager.get_latest_labels(timeout_seconds=5.0)

    assert result is None
    assert "Labels too old" in caplog.text


def test_get_latest_labels_verbose_success(label_manager, mock_redis_client, caplog):
    """Test verbose output on successful label retrieval."""
    label_manager.verbose = True

    current_time = 1000.0
    labels = ["cube", "sphere", "cylinder"]
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        with caplog.at_level(logging.INFO):
            label_manager.get_latest_labels()

    assert "Retrieved 3 labels" in caplog.text


def test_get_latest_labels_with_error(label_manager, mock_redis_client, caplog):
    """Test error handling in get_latest_labels."""
    label_manager.verbose = True

    mock_redis_client.xrevrange.side_effect = Exception("Redis error")

    with caplog.at_level(logging.ERROR):
        result = label_manager.get_latest_labels()

    assert result is None
    assert "Unexpected error getting labels" in caplog.text


# ============================================================================
# Test: add_label()
# ============================================================================


def test_add_label_to_empty_list(label_manager, mock_redis_client):
    """Test adding label when no existing labels found."""
    # No existing labels
    mock_redis_client.xrevrange.return_value = []
    mock_redis_client.xadd.return_value = "1-0"

    result = label_manager.add_label("cube")

    assert result is True
    # Verify publish was called with the new label
    call_args = mock_redis_client.xadd.call_args[0][1]
    labels = json.loads(call_args["labels"])
    assert "cube" in labels


def test_add_label_to_existing_list(label_manager, mock_redis_client):
    """Test adding label to existing list."""
    # Existing labels
    current_time = 1000.0
    existing_labels = ["sphere", "cylinder"]
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(existing_labels)})]
    mock_redis_client.xadd.return_value = "2-0"

    with patch("time.time", return_value=current_time + 1.0):
        result = label_manager.add_label("cube")

    assert result is True
    # Verify new label was added
    call_args = mock_redis_client.xadd.call_args[0][1]
    labels = json.loads(call_args["labels"])
    assert "cube" in labels
    assert "sphere" in labels
    assert "cylinder" in labels


def test_add_label_already_exists(label_manager, mock_redis_client, caplog):
    """Test adding label that already exists."""
    label_manager.verbose = True

    current_time = 1000.0
    existing_labels = ["cube", "sphere"]
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(existing_labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        with caplog.at_level(logging.INFO):
            result = label_manager.add_label("cube")

    assert result is False
    assert "Label already exists: cube" in caplog.text


def test_add_label_case_insensitive(label_manager, mock_redis_client):
    """Test that add_label is case-insensitive."""
    current_time = 1000.0
    existing_labels = ["Cube", "Sphere"]
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "labels": json.dumps(existing_labels)})]

    with patch("time.time", return_value=current_time + 1.0):
        result = label_manager.add_label("CUBE")

    assert result is False  # Should detect it already exists


def test_add_label_with_metadata(label_manager, mock_redis_client):
    """Test that add_label includes metadata about the action."""
    mock_redis_client.xrevrange.return_value = []
    mock_redis_client.xadd.return_value = "1-0"

    label_manager.add_label("cube")

    call_args = mock_redis_client.xadd.call_args[0][1]
    metadata = json.loads(call_args["metadata"])
    assert metadata["action"] == "add"
    assert metadata["added_label"] == "cube"


def test_add_label_with_error(label_manager, mock_redis_client, caplog):
    """Test error handling in add_label."""
    label_manager.verbose = True

    # Simulate Redis failure for both read and write operations
    mock_redis_client.xrevrange.side_effect = Exception("Redis error")
    mock_redis_client.xadd.side_effect = Exception("Redis error")

    with caplog.at_level(logging.ERROR):
        result = label_manager.add_label("cube")

    assert result is False


# ============================================================================
# Test: subscribe_to_label_updates()
# ============================================================================


def test_subscribe_to_label_updates_callback_execution(label_manager, mock_redis_client):
    """Test that subscribe correctly invokes callback."""
    labels = ["cube", "sphere"]
    metadata = {"action": "add"}

    # Mock xread to return one message then raise KeyboardInterrupt
    mock_redis_client.xread.side_effect = [
        [("detectable_labels", [("1-0", {"labels": json.dumps(labels), "metadata": json.dumps(metadata)})])],
        KeyboardInterrupt(),
    ]

    callback_data = []

    def callback(lbls, meta):
        callback_data.append((lbls, meta))

    label_manager.subscribe_to_label_updates(callback)

    assert len(callback_data) == 1
    assert callback_data[0][0] == labels
    assert callback_data[0][1] == metadata


def test_subscribe_to_label_updates_without_metadata(label_manager, mock_redis_client):
    """Test subscribe when metadata field is missing."""
    labels = ["cube"]

    mock_redis_client.xread.side_effect = [[("detectable_labels", [("1-0", {"labels": json.dumps(labels)})])], KeyboardInterrupt()]

    callback_data = []

    def callback(lbls, meta):
        callback_data.append((lbls, meta))

    label_manager.subscribe_to_label_updates(callback)

    assert len(callback_data) == 1
    assert callback_data[0][0] == labels
    assert callback_data[0][1] == {}


def test_subscribe_to_label_updates_error_handling(label_manager, mock_redis_client, caplog):
    """Test error handling in callback processing."""
    label_manager.verbose = True

    # Mock xread to return malformed data
    mock_redis_client.xread.side_effect = [[("detectable_labels", [("1-0", {"labels": "invalid json"})])], KeyboardInterrupt()]

    def callback(lbls, meta):
        pass

    with caplog.at_level(logging.ERROR):
        label_manager.subscribe_to_label_updates(callback)

    assert "Error processing label update" in caplog.text


def test_subscribe_to_label_updates_general_exception(label_manager, mock_redis_client, caplog):
    """Test general exception handling in subscribe."""
    label_manager.verbose = True

    mock_redis_client.xread.side_effect = Exception("Connection error")

    def callback(lbls, meta):
        pass

    with caplog.at_level(logging.ERROR):
        label_manager.subscribe_to_label_updates(callback)

    assert "Unexpected error in label subscription" in caplog.text


# ============================================================================
# Test: clear_stream()
# ============================================================================


def test_clear_stream_success(label_manager, mock_redis_client):
    """Test successful stream clearing."""
    mock_redis_client.delete.return_value = 1

    result = label_manager.clear_stream()

    assert result is True
    mock_redis_client.delete.assert_called_with("detectable_labels")


def test_clear_stream_verbose(label_manager, mock_redis_client, caplog):
    """Test verbose output when clearing stream."""
    label_manager.verbose = True

    mock_redis_client.delete.return_value = 1

    with caplog.at_level(logging.INFO):
        label_manager.clear_stream()

    assert "Cleared labels stream" in caplog.text


def test_clear_stream_with_error(label_manager, mock_redis_client, caplog):
    """Test error handling when clearing stream fails."""
    label_manager.verbose = True

    mock_redis_client.delete.side_effect = Exception("Redis error")

    with caplog.at_level(logging.ERROR):
        result = label_manager.clear_stream()

    assert result is False
    assert "Error clearing stream" in caplog.text
