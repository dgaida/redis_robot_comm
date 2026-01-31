import numpy as np
import cv2
import logging
from unittest.mock import patch


def test_publish_objects(message_broker, mock_redis_client, sample_objects):
    """Testet das Publizieren von Objekten im RedisMessageBroker."""
    mock_redis_client.xadd.return_value = "1-0"

    msg_id = message_broker.publish_objects(sample_objects)

    assert msg_id == "1-0"
    mock_redis_client.xadd.assert_called_once()


def test_get_latest_objects(message_broker, mock_redis_client):
    """Testet das Abrufen der neuesten Objekte mit gültigem Zeitstempel."""
    current_time = 1000.0
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(current_time), "objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=current_time + 0.5):
        result = message_broker.get_latest_objects(max_age_seconds=2.0)

    assert len(result) == 1
    assert result[0]["id"] == "obj_1"


def test_clear_stream(message_broker, mock_redis_client):
    """Testet das Löschen des Streams."""
    mock_redis_client.delete.return_value = 1
    result = message_broker.clear_stream()

    assert result is True
    mock_redis_client.delete.assert_called_with("detected_objects")


def test_publish_image_jpeg(image_streamer, mock_redis_client, sample_image):
    """Testet das Publizieren eines JPEG-kodierten Bildes."""
    mock_redis_client.xadd.return_value = "1-0"

    msg_id = image_streamer.publish_image(sample_image)

    assert msg_id == "1-0"
    mock_redis_client.xadd.assert_called_once()


def test_get_latest_image(image_streamer, mock_redis_client, sample_image):
    """Testet das Abrufen des neuesten Bildes und das erfolgreiche Decodieren."""
    # Beispielbild erzeugen und als Base64 speichern
    _, buffer = cv2.imencode(".jpg", sample_image)
    encoded_image = buffer.tobytes()

    mock_redis_client.xrevrange.return_value = [
        (
            "1-0",
            {
                "width": "100",
                "height": "100",
                "channels": "3",
                "format": "jpeg",
                "image_data": (encoded_image.decode("latin1") if isinstance(encoded_image, bytes) else encoded_image),
                "timestamp": "1000.0",
                "compressed_size": "100",
                "original_size": "300",
            },
        )
    ]

    with patch.object(image_streamer, "_decode_variable_image", return_value=(sample_image, {})):
        result = image_streamer.get_latest_image()

    assert result is not None
    img, meta = result
    assert isinstance(img, np.ndarray)
    assert meta == {}


def test_get_stream_stats(image_streamer, mock_redis_client):
    """Testet das Abrufen von Stream-Informationen."""
    mock_redis_client.xinfo_stream.return_value = {
        "length": 5,
        "first-entry": ("1-0", {}),
        "last-entry": ("2-0", {}),
    }

    stats = image_streamer.get_stream_stats()

    assert stats["total_messages"] == 5
    assert stats["first_entry_id"] == "1-0"
    assert stats["last_entry_id"] == "2-0"


def test_get_objects_in_timerange(message_broker, mock_redis_client):
    """Testet das Abrufen von Objekten in einem bestimmten Zeitbereich."""
    # Mock-Daten für mehrere Nachrichten im Zeitbereich
    mock_redis_client.xrange.return_value = [
        ("1-0", {"objects": '[{"id": "obj_1", "class_name": "cube"}]'}),
        ("2-0", {"objects": '[{"id": "obj_2", "class_name": "sphere"}]'}),
    ]

    result = message_broker.get_objects_in_timerange(1000.0, 2000.0)

    assert len(result) == 2
    assert result[0]["id"] == "obj_1"
    assert result[1]["id"] == "obj_2"
    mock_redis_client.xrange.assert_called_once()


def test_test_connection_success(message_broker, mock_redis_client):
    """Testet eine erfolgreiche Redis-Verbindung."""
    # message_broker already called ping() once during init
    mock_redis_client.ping.return_value = True
    result = message_broker.test_connection()

    assert result is True
    # Initial call + test call
    assert mock_redis_client.ping.call_count >= 1


def test_test_connection_failure(message_broker, mock_redis_client):
    """Testet eine fehlgeschlagene Redis-Verbindung."""
    mock_redis_client.ping.side_effect = Exception("Connection failed")
    result = message_broker.test_connection()

    assert result is False


def test_get_stream_info(message_broker, mock_redis_client):
    """Testet das Abrufen von Stream-Informationen."""
    mock_redis_client.xinfo_stream.return_value = {
        "length": 10,
        "first-entry": ("1-0", {}),
        "last-entry": ("10-0", {}),
    }

    info = message_broker.get_stream_info()

    assert info["length"] == 10
    mock_redis_client.xinfo_stream.assert_called_with("detected_objects")


def test_publish_image_raw(image_streamer, mock_redis_client):
    """Testet das Publizieren eines unkomprimierten Raw-Bildes."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_redis_client.xadd.return_value = "1-0"

    msg_id = image_streamer.publish_image(image, compress_jpeg=False)

    assert msg_id == "1-0"
    # Prüfen, dass xadd mit format="raw" aufgerufen wurde
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert call_args["format"] == "raw"


def test_publish_image_with_metadata(image_streamer, mock_redis_client):
    """Testet das Publizieren eines Bildes mit Metadaten."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    metadata = {"robot": "arm1", "workspace": "A"}
    mock_redis_client.xadd.return_value = "1-0"

    msg_id = image_streamer.publish_image(image, metadata=metadata)

    assert msg_id == "1-0"
    call_args = mock_redis_client.xadd.call_args[0][1]
    assert "metadata" in call_args


def test_get_latest_objects_empty_stream(message_broker, mock_redis_client):
    """Testet das Abrufen von Objekten aus einem leeren Stream."""
    mock_redis_client.xrevrange.return_value = []
    result = message_broker.get_latest_objects()

    assert result == []


def test_get_latest_objects_old_message(message_broker, mock_redis_client):
    """Testet das Abrufen von Objekten, die zu alt sind."""
    old_time = 1000.0
    mock_redis_client.xrevrange.return_value = [("1-0", {"timestamp": str(old_time), "objects": '[{"id": "obj_1"}]'})]

    with patch("time.time", return_value=old_time + 5.0):
        result = message_broker.get_latest_objects(max_age_seconds=2.0)

    assert result == []


def test_get_stream_stats_error(image_streamer, mock_redis_client):
    """Testet das Abrufen von Stream-Statistiken bei einem Fehler."""
    mock_redis_client.xinfo_stream.side_effect = Exception("Stream not found")
    stats = image_streamer.get_stream_stats()

    assert "error" in stats
    assert "Stream not found" in stats["error"]


def test_decode_variable_image_grayscale(image_streamer):
    """Testet das Decodieren eines Graustufenbildes."""
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

    result = image_streamer._decode_variable_image(fields)

    assert result is not None
    img, meta = result
    assert img.shape == (10, 10)


def test_publish_objects_with_error(message_broker, mock_redis_client, sample_objects, caplog):
    """Testet das Publizieren von Objekten mit einem Fehler."""
    mock_redis_client.xadd.side_effect = Exception("Redis error")

    with caplog.at_level(logging.ERROR):
        msg_id = message_broker.publish_objects(sample_objects)

    assert msg_id is None
    assert "Unexpected error publishing objects" in caplog.text


def test_get_objects_in_timerange_with_error(message_broker, mock_redis_client, caplog):
    """Testet das Abrufen von Objekten im Zeitbereich mit einem Fehler."""
    message_broker.verbose = True

    mock_redis_client.xrange.side_effect = Exception("Redis error")
    with caplog.at_level(logging.ERROR):
        result = message_broker.get_objects_in_timerange(1000.0, 2000.0)

    assert result == []
    assert "Unexpected error getting objects in timerange" in caplog.text


def test_get_latest_image_no_messages(image_streamer, mock_redis_client):
    """Testet das Abrufen des neuesten Bildes aus einem leeren Stream."""
    mock_redis_client.xrevrange.return_value = []
    result = image_streamer.get_latest_image()

    assert result is None
