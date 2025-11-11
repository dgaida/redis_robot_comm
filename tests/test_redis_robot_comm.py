# import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from redis_robot_comm.redis_client import RedisMessageBroker
from redis_robot_comm.redis_image_streamer import RedisImageStreamer


def test_publish_objects(monkeypatch):
    """Testet das Publizieren von Objekten im RedisMessageBroker."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    objects = [{"id": "obj_1", "class_name": "cube"}]
    mock_client.xadd.return_value = "1-0"

    msg_id = broker.publish_objects(objects)

    assert msg_id == "1-0"
    mock_client.xadd.assert_called_once()


def test_get_latest_objects(monkeypatch):
    """Testet das Abrufen der neuesten Objekte mit gültigem Zeitstempel."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    current_time = 1000.0
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=current_time + 0.5):
        result = broker.get_latest_objects(max_age_seconds=2.0)

    assert len(result) == 1
    assert result[0]["id"] == "obj_1"


def test_clear_stream(monkeypatch):
    """Testet das Löschen des Streams."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.delete.return_value = 1
    result = broker.clear_stream()

    assert result == 1
    mock_client.delete.assert_called_with("detected_objects")


def test_publish_image_jpeg(monkeypatch):
    """Testet das Publizieren eines JPEG-kodierten Bildes."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    msg_id = streamer.publish_image(image)

    assert msg_id == "1-0"
    mock_client.xadd.assert_called_once()


def test_get_latest_image(monkeypatch):
    """Testet das Abrufen des neuesten Bildes und das erfolgreiche Decodieren."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    # Beispielbild erzeugen und als Base64 speichern
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    encoded_image = buffer.tobytes()

    mock_client.xrevrange.return_value = [
        (
            "1-0",
            {
                "width": "10",
                "height": "10",
                "channels": "3",
                "format": "jpeg",
                "image_data": (encoded_image.decode("latin1") if isinstance(encoded_image, bytes) else encoded_image),
                "timestamp": "1000.0",
                "compressed_size": "100",
                "original_size": "300",
            },
        )
    ]

    with patch.object(streamer, "_decode_variable_image", return_value=(image, {})):
        result = streamer.get_latest_image()

    assert result is not None
    img, meta = result
    assert isinstance(img, np.ndarray)
    assert meta == {}


def test_get_stream_stats(monkeypatch):
    """Testet das Abrufen von Stream-Informationen."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    mock_client.xinfo_stream.return_value = {
        "length": 5,
        "first-entry": ("1-0", {}),
        "last-entry": ("2-0", {}),
    }

    stats = streamer.get_stream_stats()

    assert stats["total_messages"] == 5
    assert stats["first_entry_id"] == "1-0"
    assert stats["last_entry_id"] == "2-0"
