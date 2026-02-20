# redis_client.py
"""Redis-basierter Message Broker für Objekterkennungsdaten. (Redis-based message broker for object detection data)."""

import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any, cast

import redis
from redis.exceptions import RedisError

from .types import ObjectDict, CameraPose, StreamID
from .exceptions import RedisConnectionError, RedisPublishError, RedisRetrievalError
from .validators import validate_objects, validate_stream_name
from .utils import retry_on_connection_error
from .config import RedisConfig, get_redis_config

logger = logging.getLogger(__name__)


class RedisMessageBroker:
    """
    Redis-basierter Message Broker für die Veröffentlichung und den Empfang von Objekterkennungsergebnissen.

    Redis-based message broker for publishing and consuming object detection results.

    Diese Klasse bietet eine High-Level-Schnittstelle für das Streaming von Objekterkennungsdaten
    über Redis Streams und unterstützt mehrere Producer und Consumer mit automatischer
    Stream-Verwaltung.

    This class provides a high-level interface for streaming object detection data
    through Redis Streams, supporting multiple producers and consumers with automatic
    stream management.

    Attributes:
        verbose (bool): Aktiviert detaillierte Protokollausgaben. (Enable verbose logging output).
        client (redis.Redis): Instanz des zugrunde liegenden Redis-Clients. (Underlying Redis client instance).
        stream_name (str): Name des Redis-Streams für Objekterkennungen. (Name of the Redis stream for object detections).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        stream_name: str = "detected_objects",
        config: Optional[RedisConfig] = None,
    ) -> None:
        """
        Initialisiert den Redis Message Broker.

        Initialize the Redis message broker.

        Args:
            host (Optional[str]): Hostname oder IP-Adresse des Redis-Servers. (Redis server hostname or IP address).
            port (Optional[int]): Port des Redis-Servers. (Redis server port).
            db (Optional[int]): Index der Redis-Datenbank. (Redis database index).
            stream_name (str): Name des zu verwendenden Redis-Streams. (Name of the Redis stream to use).
            config (Optional[RedisConfig]): Optionale RedisConfig-Instanz. (Optional RedisConfig instance).

        Raises:
            RedisConnectionError: Wenn die Verbindung zu Redis fehlschlägt. (If connection to Redis fails).
        """
        if config is None:
            config = get_redis_config()

        # Override config with explicit parameters if provided
        host = host or config.host
        port = port or config.port
        db = db if db is not None else config.db

        self.verbose: bool = False
        validate_stream_name(stream_name)
        self.stream_name: str = stream_name
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=config.password,
                socket_timeout=config.socket_timeout,
                socket_connect_timeout=config.socket_connect_timeout,
                retry_on_timeout=config.retry_on_timeout,
                max_connections=config.max_connections,
                decode_responses=True,
            )
            self.client.ping()
        except RedisError as e:
            raise RedisConnectionError(f"Failed to connect to Redis: {e}") from e

    @retry_on_connection_error(max_attempts=3, delay=0.5)
    def publish_objects(
        self,
        objects: List[ObjectDict],
        camera_pose: Optional[CameraPose] = None,
        maxlen: int = 500,
    ) -> Optional[StreamID]:
        """
        Veröffentlicht erkannte Objekte im Redis-Stream.

        Publish detected objects to the Redis stream.

        Args:
            objects (List[ObjectDict]): Liste von Dictionaries der erkannten Objekte. (List of detected object dictionaries).
            camera_pose (Optional[CameraPose]): Optionale Informationen zur Kamerapose. (Optional camera pose information).
            maxlen (int): Maximale Stream-Länge (ältere Einträge werden gekürzt). (Maximum stream length (older entries are trimmed)).

        Returns:
            Optional[StreamID]: ID des Redis-Stream-Eintrags oder None, wenn die Veröffentlichung fehlschlägt. (Redis stream entry ID, or None if publishing fails).

        Raises:
            RedisPublishError: Wenn die Veröffentlichung bei Redis fehlschlägt. (If publishing to Redis fails).
        """
        validate_objects(objects)

        message = {
            "timestamp": str(time.time()),
            "objects": json.dumps(objects),
            "camera_pose": json.dumps(camera_pose or {}),
        }

        try:
            result = self.client.xadd(
                self.stream_name,
                message,
                maxlen=maxlen,
                approximate=True,
            )

            if self.verbose:
                logger.info(f"Published {len(objects)} objects to {self.stream_name}: {result}")
            return cast(Optional[StreamID], result)
        except RedisError as e:
            logger.error(f"Error publishing objects: {e}")
            raise RedisPublishError(f"Failed to publish objects: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error publishing objects: {e}")
            return None

    def get_latest_objects(self, max_age_seconds: float = 2.0) -> List[ObjectDict]:
        """
        Ruft die neuesten Objekterkennungen aus dem Stream ab.

        Retrieve the most recent object detections from the stream.

        Args:
            max_age_seconds (float): Maximales Alter der abzurufenden Objekte. (Maximum age of objects to retrieve).

        Returns:
            List[ObjectDict]: Liste der erkannten Objekte oder leere Liste, falls keine gefunden wurden oder diese zu alt sind. (List of detected objects, or empty list if none found or too old).

        Raises:
            RedisRetrievalError: Wenn der Abruf von Redis fehlschlägt. (If retrieval from Redis fails).
        """
        try:
            # Get the latest message from the stream
            messages = self.client.xrevrange(self.stream_name, count=1)
            if not messages:
                if self.verbose:
                    logger.debug(f"No messages found in {self.stream_name}")
                return []

            # Parse the latest message
            msg_id, fields = messages[0]

            # Check if message is fresh enough
            msg_timestamp = float(fields.get("timestamp", "0"))
            current_time = time.time()

            if current_time - msg_timestamp > max_age_seconds:
                if self.verbose:
                    logger.debug(f"Latest objects too old: {current_time - msg_timestamp:.2f}s > {max_age_seconds}s")
                return []

            # Parse and return objects
            objects_json = fields.get("objects", "[]")
            objects = json.loads(objects_json)
            if self.verbose:
                logger.info(f"Retrieved {len(objects)} fresh objects")
            return cast(List[ObjectDict], objects)

        except RedisError as e:
            logger.error(f"Error getting latest objects: {e}")
            raise RedisRetrievalError(f"Failed to retrieve objects: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting latest objects: {e}")
            return []

    def get_objects_in_timerange(self, start_timestamp: float, end_timestamp: Optional[float] = None) -> List[ObjectDict]:
        """
        Ruft Objekte ab, die innerhalb eines bestimmten Zeitbereichs veröffentlicht wurden.

        Retrieve objects published within a specific time range.

        Args:
            start_timestamp (float): Startzeit als Unix-Zeitstempel. (Start time as Unix timestamp).
            end_timestamp (Optional[float]): Endzeit als Unix-Zeitstempel. Falls None, wird die aktuelle Zeit verwendet. (End time as Unix timestamp. If None, uses current time).

        Returns:
            List[ObjectDict]: Liste der im Intervall gefundenen Objekte. (List of objects found in the interval).

        Raises:
            RedisRetrievalError: Wenn der Abruf von Redis fehlschlägt. (If retrieval from Redis fails).
        """
        if end_timestamp is None:
            end_timestamp = time.time()

        try:
            # Convert timestamps to Redis stream IDs
            start_id = f"{int(start_timestamp * 1000)}-0"
            end_id = f"{int(end_timestamp * 1000)}-0"

            messages = self.client.xrange(self.stream_name, start_id, end_id)

            all_objects = []
            for msg_id, fields in messages:
                objects_json = fields.get("objects", "[]")
                objects = json.loads(objects_json)
                all_objects.extend(objects)

            if self.verbose:
                logger.info(f"Retrieved {len(all_objects)} objects from timerange")
            return cast(List[ObjectDict], all_objects)

        except RedisError as e:
            logger.error(f"Error getting objects in timerange: {e}")
            raise RedisRetrievalError(f"Failed to retrieve objects in timerange: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting objects in timerange: {e}")
            return []

    def subscribe_objects(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Startet ein blockierendes Abonnement für Objekterkennungen.

        Start a blocking subscription for object detections.

        Args:
            callback (Callable[[Dict[str, Any]], None]): Funktion, die ein Dictionary mit 'objects', 'camera_pose' und 'timestamp' erhält. (Function receiving a dictionary with 'objects', 'camera_pose', and 'timestamp').
        """
        if self.verbose:
            logger.info(f"Starting to listen for object detections on {self.stream_name}...")
        last_id = "$"  # Start from newest

        try:
            while True:
                # Block for up to 1 second waiting for new messages
                messages = self.client.xread({self.stream_name: last_id}, block=1000)

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            # Parse objects from JSON
                            objects_json = fields.get("objects", "[]")
                            objects = json.loads(objects_json)

                            # Parse camera pose if available
                            camera_pose_json = fields.get("camera_pose", "{}")
                            camera_pose = json.loads(camera_pose_json)

                            # Call callback with parsed data
                            callback(
                                {
                                    "objects": objects,
                                    "camera_pose": camera_pose,
                                    "timestamp": float(fields.get("timestamp", "0")),
                                }
                            )

                            last_id = msg_id

                        except Exception as e:
                            logger.error(f"Error processing message {msg_id}: {e}")

        except KeyboardInterrupt:
            logger.info("Stopped listening for object detections")
        except RedisError as e:
            logger.error(f"Redis error in subscribe_objects: {e}")
            raise RedisRetrievalError(f"Subscription failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in subscribe_objects: {e}")

    def clear_stream(self) -> bool:
        """
        Löscht den Objekterkennungs-Stream.

        Clear the object detection stream.

        Returns:
            bool: True bei Erfolg, False andernfalls. (True if successful, False otherwise).
        """
        try:
            result = self.client.delete(self.stream_name)
            if self.verbose:
                logger.info(f"Cleared {self.stream_name} stream: {result}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error clearing stream: {e}")
            return False

    def get_stream_info(self) -> Optional[Dict[str, Any]]:
        """
        Ruft Informationen über den Redis-Stream ab.

        Retrieve information about the Redis stream.

        Returns:
            Optional[Dict[str, Any]]: Dictionary mit Stream-Informationen oder None, falls ein Fehler auftritt. (Dictionary with stream info, or None if an error occurs).
        """
        try:
            info = self.client.xinfo_stream(self.stream_name)
            if self.verbose:
                logger.info(f"Stream info: {info}")
            return cast(Optional[Dict[str, Any]], info)
        except Exception as e:
            logger.error(f"Error getting stream info: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Testet die Verbindung zum Redis-Server.

        Test the connection to the Redis server.

        Returns:
            bool: True, wenn die Verbindung erfolgreich ist, andernfalls False. (True if connection is successful, False otherwise).
        """
        try:
            pong = self.client.ping()
            if self.verbose:
                logger.info(f"Redis connection test: {'OK' if pong else 'FAILED'}")
            return bool(pong)
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
