"""Comprehensive test suite for RedisTextOverlayManager class."""

import json
import logging
from unittest.mock import MagicMock, patch
from redis_robot_comm.redis_text_overlay import TextType

# ============================================================================
# Test: Initialization
# ============================================================================


def test_init_default_parameters(text_overlay_manager):
    """Test initialization with default parameters."""
    assert text_overlay_manager.stream_name == "video_text_overlays"
    assert text_overlay_manager.verbose is False


def test_init_custom_parameters(monkeypatch):
    """Test initialization with custom parameters."""
    from redis_robot_comm.redis_text_overlay import RedisTextOverlayManager
    import redis

    mock_redis = MagicMock()
    monkeypatch.setattr(redis, "Redis", mock_redis)
    mock_redis.return_value.ping.return_value = True

    manager = RedisTextOverlayManager(host="custom-host", port=6380, stream_name="custom_overlays")

    assert manager.stream_name == "custom_overlays"
    # Verify relevant parameters were passed
    args, kwargs = mock_redis.call_args
    assert kwargs.get("host") == "custom-host"
    assert kwargs.get("port") == 6380
    assert kwargs.get("decode_responses") is True
    mock_redis.return_value.ping.assert_called_once()


# ============================================================================
# Test: publish_user_task()
# ============================================================================


def test_publish_user_task_success(text_overlay_manager, mock_redis_client):
    """Test successful user task publishing."""
    task = "Pick up the cube"
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = text_overlay_manager.publish_user_task(task)

    assert stream_id == "1-0"
    mock_redis_client.xadd.assert_called_once()
    call_args = mock_redis_client.xadd.call_args[0]
    assert call_args[0] == "video_text_overlays"
    assert call_args[1]["text"] == task
    assert call_args[1]["type"] == TextType.USER_TASK.value


def test_publish_user_task_with_metadata(text_overlay_manager, mock_redis_client):
    """Test publishing user task with metadata."""
    task = "Pick up the cube"
    metadata = {"user_id": "user123", "session_id": "session456"}
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = text_overlay_manager.publish_user_task(task, metadata=metadata)

    assert stream_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert "metadata" in call_args
    assert json.loads(call_args["metadata"]) == metadata


def test_publish_user_task_verbose_mode(text_overlay_manager, mock_redis_client, caplog):
    """Test verbose output when publishing user task."""
    text_overlay_manager.verbose = True

    task = "Pick up the cube and place it next to the sphere"
    mock_redis_client.xadd.return_value = "1-0"

    with caplog.at_level(logging.INFO):
        text_overlay_manager.publish_user_task(task)

    assert "Published user_task" in caplog.text
    assert "Pick up the cube" in caplog.text


# ============================================================================
# Test: publish_robot_speech()
# ============================================================================


def test_publish_robot_speech_success(text_overlay_manager, mock_redis_client):
    """Test successful robot speech publishing."""
    speech = "🤖 I'm picking up the cube now"
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = text_overlay_manager.publish_robot_speech(speech)

    assert stream_id == "1-0"
    mock_redis_client.xadd.assert_called_once()
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert call_args["text"] == speech
    assert call_args["type"] == TextType.ROBOT_SPEECH.value


def test_publish_robot_speech_with_duration(text_overlay_manager, mock_redis_client):
    """Test publishing robot speech with custom duration."""
    speech = "🤖 I'm moving to the workspace"
    duration = 5.0
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = text_overlay_manager.publish_robot_speech(speech, duration_seconds=duration)

    assert stream_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    metadata = json.loads(call_args["metadata"])
    assert metadata["duration_seconds"] == duration


def test_publish_robot_speech_with_metadata(text_overlay_manager, mock_redis_client):
    """Test publishing robot speech with additional metadata."""
    speech = "🤖 Analyzing workspace"
    metadata = {"tool_name": "move_to_pose", "priority": "high"}
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = text_overlay_manager.publish_robot_speech(speech, metadata=metadata)

    assert stream_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    parsed_metadata = json.loads(call_args["metadata"])
    assert parsed_metadata["tool_name"] == "move_to_pose"
    assert parsed_metadata["priority"] == "high"
    assert "duration_seconds" in parsed_metadata


def test_publish_robot_speech_verbose_mode(text_overlay_manager, mock_redis_client, caplog):
    """Test verbose output when publishing robot speech."""
    text_overlay_manager.verbose = True

    speech = "🤖 I'm picking up the cube now"
    mock_redis_client.xadd.return_value = "1-0"

    with caplog.at_level(logging.INFO):
        text_overlay_manager.publish_robot_speech(speech)

    assert "Published robot_speech" in caplog.text


# ============================================================================
# Test: publish_system_message()
# ============================================================================


def test_publish_system_message_success(text_overlay_manager, mock_redis_client):
    """Test successful system message publishing."""
    message = "🎥 Recording started"
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = text_overlay_manager.publish_system_message(message)

    assert stream_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert call_args["text"] == message
    assert call_args["type"] == TextType.SYSTEM_MESSAGE.value


def test_publish_system_message_with_duration(text_overlay_manager, mock_redis_client):
    """Test publishing system message with custom duration."""
    message = "⚠️ System warning"
    duration = 5.0
    mock_redis_client.xadd.return_value = "1-0"

    stream_id = text_overlay_manager.publish_system_message(message, duration_seconds=duration)

    assert stream_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    metadata = json.loads(call_args["metadata"])
    assert metadata["duration_seconds"] == duration


def test_publish_system_message_verbose_mode(text_overlay_manager, mock_redis_client, caplog):
    """Test verbose output when publishing system message."""
    text_overlay_manager.verbose = True

    message = "🎥 Recording started"
    mock_redis_client.xadd.return_value = "1-0"

    with caplog.at_level(logging.INFO):
        text_overlay_manager.publish_system_message(message)

    assert "Published system_message" in caplog.text


# ============================================================================
# Test: _publish_text() (internal method)
# ============================================================================


def test_publish_text_with_error(text_overlay_manager, mock_redis_client, caplog):
    """Test error handling when publishing text fails."""
    text_overlay_manager.verbose = True

    mock_redis_client.xadd.side_effect = Exception("Redis connection error")

    with caplog.at_level(logging.ERROR):
        stream_id = text_overlay_manager.publish_user_task("Test task")

    assert stream_id is None
    assert "Unexpected error publishing text overlay" in caplog.text


def test_publish_text_maxlen_parameter(text_overlay_manager, mock_redis_client):
    """Test that maxlen=100 is passed to xadd."""
    mock_redis_client.xadd.return_value = "1-0"
    text_overlay_manager.publish_user_task("Test task")

    call_kwargs = mock_redis_client.xadd.call_args[1]
    assert call_kwargs["maxlen"] == 100


# ============================================================================
# Test: get_latest_texts()
# ============================================================================


def test_get_latest_texts_success(text_overlay_manager, mock_redis_client):
    """Test successful retrieval of latest texts."""
    current_time = 1000.0
    mock_redis_client.xrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "text": "Task 1", "type": "user_task", "metadata": "{}"}),
        (
            "2-0",
            {
                "timestamp": str(current_time + 1.0),
                "text": "Speech 1",
                "type": "robot_speech",
                "metadata": '{"duration_seconds": 4.0}',
            },
        ),
    ]

    with patch("time.time", return_value=current_time + 2.0):
        texts = text_overlay_manager.get_latest_texts(max_age_seconds=5.0)

    assert len(texts) == 2
    assert texts[0]["text"] == "Task 1"
    assert texts[1]["text"] == "Speech 1"
    assert texts[1]["metadata"]["duration_seconds"] == 4.0


def test_get_latest_texts_filtered_by_type(text_overlay_manager, mock_redis_client):
    """Test retrieving texts filtered by type."""
    current_time = 1000.0
    mock_redis_client.xrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "text": "Task 1", "type": "user_task", "metadata": "{}"}),
        ("2-0", {"timestamp": str(current_time + 1.0), "text": "Speech 1", "type": "robot_speech", "metadata": "{}"}),
    ]

    with patch("time.time", return_value=current_time + 2.0):
        texts = text_overlay_manager.get_latest_texts(max_age_seconds=5.0, text_type=TextType.USER_TASK)

    assert len(texts) == 1
    assert texts[0]["type"] == "user_task"


def test_get_latest_texts_empty_stream(text_overlay_manager, mock_redis_client):
    """Test retrieving texts from empty stream."""
    mock_redis_client.xrange.return_value = []

    texts = text_overlay_manager.get_latest_texts()

    assert texts == []


def test_get_latest_texts_parse_error(text_overlay_manager, mock_redis_client, caplog):
    """Test handling of parse errors in get_latest_texts."""
    text_overlay_manager.verbose = True

    mock_redis_client.xrange.return_value = [
        ("1-0", {"timestamp": "invalid", "text": "Task 1", "type": "user_task", "metadata": "{}"})
    ]

    with caplog.at_level(logging.ERROR):
        texts = text_overlay_manager.get_latest_texts()

    # Should skip the invalid entry
    assert len(texts) == 0
    assert "Error parsing text overlay" in caplog.text


def test_get_latest_texts_with_error(text_overlay_manager, mock_redis_client, caplog):
    """Test error handling in get_latest_texts."""
    text_overlay_manager.verbose = True

    mock_redis_client.xrange.side_effect = Exception("Redis error")

    with caplog.at_level(logging.ERROR):
        texts = text_overlay_manager.get_latest_texts()

    assert texts == []
    assert "Unexpected error getting latest texts" in caplog.text


# ============================================================================
# Test: get_current_user_task()
# ============================================================================


def test_get_current_user_task_success(text_overlay_manager, mock_redis_client):
    """Test successful retrieval of current user task."""
    current_time = 1000.0
    mock_redis_client.xrevrange.return_value = [
        ("2-0", {"timestamp": str(current_time), "text": "Pick up the cube", "type": "user_task", "metadata": "{}"})
    ]

    with patch("time.time", return_value=current_time + 10.0):
        task = text_overlay_manager.get_current_user_task(max_age_seconds=300.0)

    assert task == "Pick up the cube"


def test_get_current_user_task_too_old(text_overlay_manager, mock_redis_client):
    """Test that old user tasks are not returned."""
    old_time = 1000.0
    mock_redis_client.xrevrange.return_value = [
        ("1-0", {"timestamp": str(old_time), "text": "Old task", "type": "user_task", "metadata": "{}"})
    ]

    with patch("time.time", return_value=old_time + 400.0):
        task = text_overlay_manager.get_current_user_task(max_age_seconds=300.0)

    assert task is None


def test_get_current_user_task_no_user_task(text_overlay_manager, mock_redis_client):
    """Test when no user task exists in stream."""
    current_time = 1000.0
    mock_redis_client.xrevrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "text": "Robot speech", "type": "robot_speech", "metadata": "{}"})
    ]

    with patch("time.time", return_value=current_time + 10.0):
        task = text_overlay_manager.get_current_user_task()

    assert task is None


def test_get_current_user_task_empty_stream(text_overlay_manager, mock_redis_client):
    """Test when stream is empty."""
    mock_redis_client.xrevrange.return_value = []

    task = text_overlay_manager.get_current_user_task()

    assert task is None


def test_get_current_user_task_with_error(text_overlay_manager, mock_redis_client, caplog):
    """Test error handling in get_current_user_task."""
    text_overlay_manager.verbose = True

    mock_redis_client.xrevrange.side_effect = Exception("Redis error")

    with caplog.at_level(logging.ERROR):
        task = text_overlay_manager.get_current_user_task()

    assert task is None
    assert "Unexpected error getting current user task" in caplog.text


# ============================================================================
# Test: subscribe_to_texts()
# ============================================================================


def test_subscribe_to_texts_callback_execution(text_overlay_manager, mock_redis_client):
    """Test that subscribe correctly invokes callback."""
    text_data = {"timestamp": "1000.0", "text": "Test task", "type": "user_task", "metadata": "{}"}

    mock_redis_client.xread.side_effect = [[("video_text_overlays", [("1-0", text_data)])], KeyboardInterrupt()]

    callback_data = []

    def callback(data):
        callback_data.append(data)

    text_overlay_manager.subscribe_to_texts(callback)

    assert len(callback_data) == 1
    assert callback_data[0]["text"] == "Test task"
    assert callback_data[0]["type"] == "user_task"


def test_subscribe_to_texts_filtered_by_type(text_overlay_manager, mock_redis_client):
    """Test subscribing with type filter."""
    mock_redis_client.xread.side_effect = [
        [
            (
                "video_text_overlays",
                [
                    ("1-0", {"timestamp": "1000.0", "text": "Task", "type": "user_task", "metadata": "{}"}),
                    ("2-0", {"timestamp": "1001.0", "text": "Speech", "type": "robot_speech", "metadata": "{}"}),
                ],
            )
        ],
        KeyboardInterrupt(),
    ]

    callback_data = []

    def callback(data):
        callback_data.append(data)

    text_overlay_manager.subscribe_to_texts(callback, text_type=TextType.USER_TASK)

    assert len(callback_data) == 1
    assert callback_data[0]["type"] == "user_task"


def test_subscribe_to_texts_verbose_mode(text_overlay_manager, mock_redis_client, caplog):
    """Test verbose output when subscribing."""
    text_overlay_manager.verbose = True

    mock_redis_client.xread.side_effect = [KeyboardInterrupt()]

    def callback(data):
        pass

    with caplog.at_level(logging.INFO):
        text_overlay_manager.subscribe_to_texts(callback)

    assert "Subscribing to text overlays" in caplog.text
    assert "Stopped subscribing to text overlays" in caplog.text


def test_subscribe_to_texts_error_handling(text_overlay_manager, mock_redis_client, caplog):
    """Test error handling in callback processing."""
    text_overlay_manager.verbose = True

    mock_redis_client.xread.side_effect = [[("video_text_overlays", [("1-0", {"timestamp": "invalid"})])], KeyboardInterrupt()]

    def callback(data):
        pass

    with caplog.at_level(logging.ERROR):
        text_overlay_manager.subscribe_to_texts(callback)

    assert "Error processing text overlay" in caplog.text


def test_subscribe_to_texts_general_exception(text_overlay_manager, mock_redis_client, caplog):
    """Test general exception handling in subscribe."""
    text_overlay_manager.verbose = True

    mock_redis_client.xread.side_effect = Exception("Connection error")

    def callback(data):
        pass

    with caplog.at_level(logging.ERROR):
        text_overlay_manager.subscribe_to_texts(callback)

    assert "Unexpected error in text overlay subscription" in caplog.text


# ============================================================================
# Test: clear_stream()
# ============================================================================


def test_clear_stream_success(text_overlay_manager, mock_redis_client):
    """Test successful stream clearing."""
    mock_redis_client.delete.return_value = 1

    result = text_overlay_manager.clear_stream()

    assert result is True
    mock_redis_client.delete.assert_called_with("video_text_overlays")


def test_clear_stream_verbose(text_overlay_manager, mock_redis_client, caplog):
    """Test verbose output when clearing stream."""
    text_overlay_manager.verbose = True

    mock_redis_client.delete.return_value = 1

    with caplog.at_level(logging.INFO):
        text_overlay_manager.clear_stream()

    assert "Cleared text overlay stream" in caplog.text


def test_clear_stream_with_error(text_overlay_manager, mock_redis_client, caplog):
    """Test error handling when clearing stream fails."""
    text_overlay_manager.verbose = True

    mock_redis_client.delete.side_effect = Exception("Redis error")

    with caplog.at_level(logging.ERROR):
        result = text_overlay_manager.clear_stream()

    assert result is False
    assert "Error clearing stream" in caplog.text


# ============================================================================
# Test: get_stream_info()
# ============================================================================


def test_get_stream_info_success(text_overlay_manager, mock_redis_client):
    """Test successful retrieval of stream info."""
    mock_redis_client.xinfo_stream.return_value = {"length": 50, "first-entry": ("1-0", {}), "last-entry": ("50-0", {})}

    info = text_overlay_manager.get_stream_info()

    assert info["total_messages"] == 50
    assert info["first_entry_id"] == "1-0"
    assert info["last_entry_id"] == "50-0"


def test_get_stream_info_with_error(text_overlay_manager, mock_redis_client):
    """Test error handling in get_stream_info."""
    mock_redis_client.xinfo_stream.side_effect = Exception("Stream not found")

    info = text_overlay_manager.get_stream_info()

    assert "error" in info
    assert "Stream not found" in info["error"]


# ============================================================================
# Test: TextType Enum
# ============================================================================


def test_text_type_enum_values():
    """Test that TextType enum has correct values."""
    assert TextType.USER_TASK.value == "user_task"
    assert TextType.ROBOT_SPEECH.value == "robot_speech"
    assert TextType.SYSTEM_MESSAGE.value == "system_message"


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


def test_end_to_end_text_flow(text_overlay_manager, mock_redis_client):
    """Test complete flow: publish -> retrieve texts."""
    # Publish user task
    mock_redis_client.xadd.return_value = "1-0"
    task_id = text_overlay_manager.publish_user_task("Pick up the cube")
    assert task_id == "1-0"

    # Publish robot speech
    mock_redis_client.xadd.return_value = "2-0"
    speech_id = text_overlay_manager.publish_robot_speech("🤖 Moving to workspace")
    assert speech_id == "2-0"

    # Mock retrieval
    current_time = 1000.0
    mock_redis_client.xrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "text": "Pick up the cube", "type": "user_task", "metadata": "{}"}),
        (
            "2-0",
            {
                "timestamp": str(current_time + 1.0),
                "text": "🤖 Moving to workspace",
                "type": "robot_speech",
                "metadata": '{"duration_seconds": 4.0}',
            },
        ),
    ]

    with patch("time.time", return_value=current_time + 2.0):
        texts = text_overlay_manager.get_latest_texts()

    assert len(texts) == 2
    assert texts[0]["text"] == "Pick up the cube"
    assert texts[1]["text"] == "🤖 Moving to workspace"


def test_multiple_text_types_in_stream(text_overlay_manager, mock_redis_client):
    """Test handling multiple text types in stream."""
    current_time = 1000.0
    mock_redis_client.xrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "text": "Pick up the cube", "type": "user_task", "metadata": "{}"}),
        (
            "2-0",
            {
                "timestamp": str(current_time + 1.0),
                "text": "🤖 Moving to workspace",
                "type": "robot_speech",
                "metadata": '{"duration_seconds": 4.0}',
            },
        ),
        (
            "3-0",
            {
                "timestamp": str(current_time + 2.0),
                "text": "🎥 Recording started",
                "type": "system_message",
                "metadata": '{"duration_seconds": 3.0}',
            },
        ),
    ]

    with patch("time.time", return_value=current_time + 3.0):
        # Get all texts
        all_texts = text_overlay_manager.get_latest_texts()
        assert len(all_texts) == 3

        # Get only robot speech
        speech_texts = text_overlay_manager.get_latest_texts(text_type=TextType.ROBOT_SPEECH)
        assert len(speech_texts) == 1
        assert speech_texts[0]["type"] == "robot_speech"

        # Get only system messages
        system_texts = text_overlay_manager.get_latest_texts(text_type=TextType.SYSTEM_MESSAGE)
        assert len(system_texts) == 1
        assert system_texts[0]["type"] == "system_message"
