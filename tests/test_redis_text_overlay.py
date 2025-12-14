# tests/test_redis_text_overlay.py
"""Comprehensive test suite for RedisTextOverlayManager class."""

import json
from unittest.mock import MagicMock, patch
from redis_robot_comm.redis_text_overlay import RedisTextOverlayManager, TextType


# ============================================================================
# Test: Initialization
# ============================================================================


def test_init_default_parameters(monkeypatch):
    """Test initialization with default parameters."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)

    manager = RedisTextOverlayManager()

    assert manager.stream_name == "video_text_overlays"
    assert manager.verbose is False


def test_init_custom_parameters(monkeypatch):
    """Test initialization with custom parameters."""
    mock_redis = MagicMock()
    monkeypatch.setattr("redis.Redis", mock_redis)

    manager = RedisTextOverlayManager(host="custom-host", port=6380, stream_name="custom_overlays")

    assert manager.stream_name == "custom_overlays"
    mock_redis.assert_called_with(host="custom-host", port=6380, decode_responses=True)


# ============================================================================
# Test: publish_user_task()
# ============================================================================


def test_publish_user_task_success(monkeypatch):
    """Test successful user task publishing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    task = "Pick up the cube"
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_user_task(task)

    assert stream_id == "1-0"
    mock_client.xadd.assert_called_once()
    call_args = mock_client.xadd.call_args[0]
    assert call_args[0] == "video_text_overlays"
    assert call_args[1]["text"] == task
    assert call_args[1]["type"] == TextType.USER_TASK.value


def test_publish_user_task_with_metadata(monkeypatch):
    """Test publishing user task with metadata."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    task = "Pick up the cube"
    metadata = {"user_id": "user123", "session_id": "session456"}
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_user_task(task, metadata=metadata)

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    assert "metadata" in call_args
    assert json.loads(call_args["metadata"]) == metadata


def test_publish_user_task_verbose_mode(monkeypatch, capsys):
    """Test verbose output when publishing user task."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    task = "Pick up the cube and place it next to the sphere"
    mock_client.xadd.return_value = "1-0"

    manager.publish_user_task(task)

    captured = capsys.readouterr()
    assert "Published user_task" in captured.out
    assert "Pick up the cube" in captured.out


# ============================================================================
# Test: publish_robot_speech()
# ============================================================================


def test_publish_robot_speech_success(monkeypatch):
    """Test successful robot speech publishing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    speech = "🤖 I'm picking up the cube now"
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_robot_speech(speech)

    assert stream_id == "1-0"
    mock_client.xadd.assert_called_once()
    call_args = mock_client.xadd.call_args[0][1]
    assert call_args["text"] == speech
    assert call_args["type"] == TextType.ROBOT_SPEECH.value


def test_publish_robot_speech_with_duration(monkeypatch):
    """Test publishing robot speech with custom duration."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    speech = "🤖 I'm moving to the workspace"
    duration = 5.0
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_robot_speech(speech, duration_seconds=duration)

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    metadata = json.loads(call_args["metadata"])
    assert metadata["duration_seconds"] == duration


def test_publish_robot_speech_with_metadata(monkeypatch):
    """Test publishing robot speech with additional metadata."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    speech = "🤖 Analyzing workspace"
    metadata = {"tool_name": "move_to_pose", "priority": "high"}
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_robot_speech(speech, metadata=metadata)

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    parsed_metadata = json.loads(call_args["metadata"])
    assert parsed_metadata["tool_name"] == "move_to_pose"
    assert parsed_metadata["priority"] == "high"
    assert "duration_seconds" in parsed_metadata


def test_publish_robot_speech_verbose_mode(monkeypatch, capsys):
    """Test verbose output when publishing robot speech."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    speech = "🤖 I'm picking up the cube now"
    mock_client.xadd.return_value = "1-0"

    manager.publish_robot_speech(speech)

    captured = capsys.readouterr()
    assert "Published robot_speech" in captured.out


# ============================================================================
# Test: publish_system_message()
# ============================================================================


def test_publish_system_message_success(monkeypatch):
    """Test successful system message publishing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    message = "🎥 Recording started"
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_system_message(message)

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    assert call_args["text"] == message
    assert call_args["type"] == TextType.SYSTEM_MESSAGE.value


def test_publish_system_message_with_duration(monkeypatch):
    """Test publishing system message with custom duration."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    message = "⚠️ System warning"
    duration = 5.0
    mock_client.xadd.return_value = "1-0"

    stream_id = manager.publish_system_message(message, duration_seconds=duration)

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    metadata = json.loads(call_args["metadata"])
    assert metadata["duration_seconds"] == duration


def test_publish_system_message_verbose_mode(monkeypatch, capsys):
    """Test verbose output when publishing system message."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    message = "🎥 Recording started"
    mock_client.xadd.return_value = "1-0"

    manager.publish_system_message(message)

    captured = capsys.readouterr()
    assert "Published system_message" in captured.out


# ============================================================================
# Test: _publish_text() (internal method)
# ============================================================================


def test_publish_text_with_error(monkeypatch, capsys):
    """Test error handling when publishing text fails."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.xadd.side_effect = Exception("Redis connection error")

    stream_id = manager.publish_user_task("Test task")

    assert stream_id is None
    captured = capsys.readouterr()
    assert "Error publishing text overlay" in captured.out


def test_publish_text_maxlen_parameter(monkeypatch):
    """Test that maxlen=100 is passed to xadd."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    mock_client.xadd.return_value = "1-0"
    manager.publish_user_task("Test task")

    call_kwargs = mock_client.xadd.call_args[1]
    assert call_kwargs["maxlen"] == 100


# ============================================================================
# Test: get_latest_texts()
# ============================================================================


def test_get_latest_texts_success(monkeypatch):
    """Test successful retrieval of latest texts."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    current_time = 1000.0
    mock_client.xrange.return_value = [
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
        texts = manager.get_latest_texts(max_age_seconds=5.0)

    assert len(texts) == 2
    assert texts[0]["text"] == "Task 1"
    assert texts[1]["text"] == "Speech 1"
    assert texts[1]["metadata"]["duration_seconds"] == 4.0


def test_get_latest_texts_filtered_by_type(monkeypatch):
    """Test retrieving texts filtered by type."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    current_time = 1000.0
    mock_client.xrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "text": "Task 1", "type": "user_task", "metadata": "{}"}),
        ("2-0", {"timestamp": str(current_time + 1.0), "text": "Speech 1", "type": "robot_speech", "metadata": "{}"}),
    ]

    with patch("time.time", return_value=current_time + 2.0):
        texts = manager.get_latest_texts(max_age_seconds=5.0, text_type=TextType.USER_TASK)

    assert len(texts) == 1
    assert texts[0]["type"] == "user_task"


def test_get_latest_texts_empty_stream(monkeypatch):
    """Test retrieving texts from empty stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    mock_client.xrange.return_value = []

    texts = manager.get_latest_texts()

    assert texts == []


def test_get_latest_texts_parse_error(monkeypatch, capsys):
    """Test handling of parse errors in get_latest_texts."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.xrange.return_value = [
        ("1-0", {"timestamp": "invalid", "text": "Task 1", "type": "user_task", "metadata": "{}"})
    ]

    texts = manager.get_latest_texts()

    # Should skip the invalid entry
    assert len(texts) == 0
    captured = capsys.readouterr()
    assert "Error parsing text overlay" in captured.out


def test_get_latest_texts_with_error(monkeypatch, capsys):
    """Test error handling in get_latest_texts."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.xrange.side_effect = Exception("Redis error")

    texts = manager.get_latest_texts()

    assert texts == []
    captured = capsys.readouterr()
    assert "Error getting latest texts" in captured.out


# ============================================================================
# Test: get_current_user_task()
# ============================================================================


def test_get_current_user_task_success(monkeypatch):
    """Test successful retrieval of current user task."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    current_time = 1000.0
    mock_client.xrevrange.return_value = [
        ("2-0", {"timestamp": str(current_time), "text": "Pick up the cube", "type": "user_task", "metadata": "{}"})
    ]

    with patch("time.time", return_value=current_time + 10.0):
        task = manager.get_current_user_task(max_age_seconds=300.0)

    assert task == "Pick up the cube"


def test_get_current_user_task_too_old(monkeypatch):
    """Test that old user tasks are not returned."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    old_time = 1000.0
    mock_client.xrevrange.return_value = [
        ("1-0", {"timestamp": str(old_time), "text": "Old task", "type": "user_task", "metadata": "{}"})
    ]

    with patch("time.time", return_value=old_time + 400.0):
        task = manager.get_current_user_task(max_age_seconds=300.0)

    assert task is None


def test_get_current_user_task_no_user_task(monkeypatch):
    """Test when no user task exists in stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    current_time = 1000.0
    mock_client.xrevrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "text": "Robot speech", "type": "robot_speech", "metadata": "{}"})
    ]

    with patch("time.time", return_value=current_time + 10.0):
        task = manager.get_current_user_task()

    assert task is None


def test_get_current_user_task_empty_stream(monkeypatch):
    """Test when stream is empty."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    mock_client.xrevrange.return_value = []

    task = manager.get_current_user_task()

    assert task is None


def test_get_current_user_task_with_error(monkeypatch, capsys):
    """Test error handling in get_current_user_task."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.xrevrange.side_effect = Exception("Redis error")

    task = manager.get_current_user_task()

    assert task is None
    captured = capsys.readouterr()
    assert "Error getting current user task" in captured.out


# ============================================================================
# Test: subscribe_to_texts()
# ============================================================================


def test_subscribe_to_texts_callback_execution(monkeypatch):
    """Test that subscribe correctly invokes callback."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    text_data = {"timestamp": "1000.0", "text": "Test task", "type": "user_task", "metadata": "{}"}

    mock_client.xread.side_effect = [[("video_text_overlays", [("1-0", text_data)])], KeyboardInterrupt()]

    callback_data = []

    def callback(data):
        callback_data.append(data)

    manager.subscribe_to_texts(callback)

    assert len(callback_data) == 1
    assert callback_data[0]["text"] == "Test task"
    assert callback_data[0]["type"] == "user_task"


def test_subscribe_to_texts_filtered_by_type(monkeypatch):
    """Test subscribing with type filter."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    mock_client.xread.side_effect = [
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

    manager.subscribe_to_texts(callback, text_type=TextType.USER_TASK)

    assert len(callback_data) == 1
    assert callback_data[0]["type"] == "user_task"


def test_subscribe_to_texts_verbose_mode(monkeypatch, capsys):
    """Test verbose output when subscribing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.xread.side_effect = [KeyboardInterrupt()]

    def callback(data):
        pass

    manager.subscribe_to_texts(callback)

    captured = capsys.readouterr()
    assert "Subscribing to text overlays" in captured.out
    assert "Stopped subscribing to text overlays" in captured.out


def test_subscribe_to_texts_error_handling(monkeypatch, capsys):
    """Test error handling in callback processing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.xread.side_effect = [[("video_text_overlays", [("1-0", {"timestamp": "invalid"})])], KeyboardInterrupt()]

    def callback(data):
        pass

    manager.subscribe_to_texts(callback)

    captured = capsys.readouterr()
    assert "Error processing text overlay" in captured.out


def test_subscribe_to_texts_general_exception(monkeypatch, capsys):
    """Test general exception handling in subscribe."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.xread.side_effect = Exception("Connection error")

    def callback(data):
        pass

    manager.subscribe_to_texts(callback)

    captured = capsys.readouterr()
    assert "Error in text overlay subscription" in captured.out


# ============================================================================
# Test: clear_stream()
# ============================================================================


def test_clear_stream_success(monkeypatch):
    """Test successful stream clearing."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    mock_client.delete.return_value = 1

    result = manager.clear_stream()

    assert result is True
    mock_client.delete.assert_called_with("video_text_overlays")


def test_clear_stream_verbose(monkeypatch, capsys):
    """Test verbose output when clearing stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.delete.return_value = 1

    manager.clear_stream()

    captured = capsys.readouterr()
    assert "Cleared text overlay stream" in captured.out


def test_clear_stream_with_error(monkeypatch, capsys):
    """Test error handling when clearing stream fails."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()
    manager.verbose = True

    mock_client.delete.side_effect = Exception("Redis error")

    result = manager.clear_stream()

    assert result is False
    captured = capsys.readouterr()
    assert "Error clearing stream" in captured.out


# ============================================================================
# Test: get_stream_info()
# ============================================================================


def test_get_stream_info_success(monkeypatch):
    """Test successful retrieval of stream info."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    mock_client.xinfo_stream.return_value = {"length": 50, "first-entry": ("1-0", {}), "last-entry": ("50-0", {})}

    info = manager.get_stream_info()

    assert info["total_messages"] == 50
    assert info["first_entry_id"] == "1-0"
    assert info["last_entry_id"] == "50-0"


def test_get_stream_info_with_error(monkeypatch):
    """Test error handling in get_stream_info."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    mock_client.xinfo_stream.side_effect = Exception("Stream not found")

    info = manager.get_stream_info()

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


def test_end_to_end_text_flow(monkeypatch):
    """Test complete flow: publish -> retrieve texts."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    # Publish user task
    mock_client.xadd.return_value = "1-0"
    task_id = manager.publish_user_task("Pick up the cube")
    assert task_id == "1-0"

    # Publish robot speech
    mock_client.xadd.return_value = "2-0"
    speech_id = manager.publish_robot_speech("🤖 Moving to workspace")
    assert speech_id == "2-0"

    # Mock retrieval
    current_time = 1000.0
    mock_client.xrange.return_value = [
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
        texts = manager.get_latest_texts()

    assert len(texts) == 2
    assert texts[0]["text"] == "Pick up the cube"
    assert texts[1]["text"] == "🤖 Moving to workspace"


def test_multiple_text_types_in_stream(monkeypatch):
    """Test handling multiple text types in stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    manager = RedisTextOverlayManager()

    current_time = 1000.0
    mock_client.xrange.return_value = [
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
        all_texts = manager.get_latest_texts()
        assert len(all_texts) == 3

        # Get only robot speech
        speech_texts = manager.get_latest_texts(text_type=TextType.ROBOT_SPEECH)
        assert len(speech_texts) == 1
        assert speech_texts[0]["type"] == "robot_speech"

        # Get only system messages
        system_texts = manager.get_latest_texts(text_type=TextType.SYSTEM_MESSAGE)
        assert len(system_texts) == 1
        assert system_texts[0]["type"] == "system_message"
