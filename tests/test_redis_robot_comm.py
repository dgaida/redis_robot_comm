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


def test_get_objects_in_timerange(monkeypatch):
    """Testet das Abrufen von Objekten in einem bestimmten Zeitbereich."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Mock-Daten für mehrere Nachrichten im Zeitbereich
    mock_client.xrange.return_value = [
        ("1-0", {"objects": '[{"id": "obj_1", "class_name": "cube"}]'}),
        ("2-0", {"objects": '[{"id": "obj_2", "class_name": "sphere"}]'}),
    ]

    result = broker.get_objects_in_timerange(1000.0, 2000.0)

    assert len(result) == 2
    assert result[0]["id"] == "obj_1"
    assert result[1]["id"] == "obj_2"
    mock_client.xrange.assert_called_once()


def test_test_connection_success(monkeypatch):
    """Testet eine erfolgreiche Redis-Verbindung."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.ping.return_value = True
    result = broker.test_connection()

    assert result is True
    mock_client.ping.assert_called_once()


def test_test_connection_failure(monkeypatch):
    """Testet eine fehlgeschlagene Redis-Verbindung."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.ping.side_effect = Exception("Connection failed")
    result = broker.test_connection()

    assert result is False


def test_get_stream_info(monkeypatch):
    """Testet das Abrufen von Stream-Informationen."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.xinfo_stream.return_value = {
        "length": 10,
        "first-entry": ("1-0", {}),
        "last-entry": ("10-0", {}),
    }

    info = broker.get_stream_info()

    assert info["length"] == 10
    mock_client.xinfo_stream.assert_called_with("detected_objects")


def test_publish_image_raw(monkeypatch):
    """Testet das Publizieren eines unkomprimierten Raw-Bildes."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    msg_id = streamer.publish_image(image, compress_jpeg=False)

    assert msg_id == "1-0"
    # Prüfen, dass xadd mit format="raw" aufgerufen wurde
    call_args = mock_client.xadd.call_args[0][1]
    assert call_args["format"] == "raw"


def test_publish_image_with_metadata(monkeypatch):
    """Testet das Publizieren eines Bildes mit Metadaten."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    metadata = {"robot": "arm1", "workspace": "A"}
    mock_client.xadd.return_value = "1-0"

    msg_id = streamer.publish_image(image, metadata=metadata)

    assert msg_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    assert "metadata" in call_args


def test_get_latest_objects_empty_stream(monkeypatch):
    """Testet das Abrufen von Objekten aus einem leeren Stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.xrevrange.return_value = []
    result = broker.get_latest_objects()

    assert result == []


def test_get_latest_objects_old_message(monkeypatch):
    """Testet das Abrufen von Objekten, die zu alt sind."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    old_time = 1000.0
    mock_client.xrevrange.return_value = [("1-0", {"timestamp": str(old_time), "objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=old_time + 5.0):
        result = broker.get_latest_objects(max_age_seconds=2.0)

    assert result == []


def test_get_stream_stats_error(monkeypatch):
    """Testet das Abrufen von Stream-Statistiken bei einem Fehler."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    mock_client.xinfo_stream.side_effect = Exception("Stream not found")
    stats = streamer.get_stream_stats()

    assert "error" in stats
    assert "Stream not found" in stats["error"]


def test_decode_variable_image_grayscale(monkeypatch):
    """Testet das Decodieren eines Graustufenbildes."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    # Graustufenbild erstellen
    image = np.zeros((10, 10), dtype=np.uint8)
    import base64

    fields = {
        "width": "10",
        "height": "10",
        "channels": "1",
        "format": "raw",
        "dtype": "uint8",
        "image_data": base64.b64encode(image.tobytes()).decode("utf-8"),
    }

    result = streamer._decode_variable_image(fields)

    assert result is not None
    img, meta = result
    assert img.shape == (10, 10)


def test_publish_objects_with_error(monkeypatch):
    """Testet das Publizieren von Objekten mit einem Fehler."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    mock_client.xadd.side_effect = Exception("Redis error")
    objects = [{"id": "obj_1", "class_name": "cube"}]

    msg_id = broker.publish_objects(objects)

    assert msg_id is None


def test_get_objects_in_timerange_with_error(monkeypatch):
    """Testet das Abrufen von Objekten im Zeitbereich mit einem Fehler."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.xrange.side_effect = Exception("Redis error")
    result = broker.get_objects_in_timerange(1000.0, 2000.0)

    assert result == []


def test_get_latest_image_no_messages(monkeypatch):
    """Testet das Abrufen des neuesten Bildes aus einem leeren Stream."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    mock_client.xrevrange.return_value = []
    result = streamer.get_latest_image()

    assert result is None
