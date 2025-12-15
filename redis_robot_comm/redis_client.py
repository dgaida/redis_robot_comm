# redis_client.py

import redis
import json
import time
from typing import Dict, List


class RedisMessageBroker:
    """Redis-Client für Objektdetektionen.

    Diese Klasse ermöglicht das Publizieren, Abonnieren und Abrufen von
    Objektdetektionen in einem Redis-Stream. Sie eignet sich für den Einsatz
    in Robotik- oder Vision-Anwendungen.

    Args:
        host (str): Redis-Host (Standard: "localhost").
        port (int): Redis-Port (Standard: 6379).
        db (int): Datenbank-Index (Standard: 0).
    """

    def __init__(self, host="localhost", port=6379, db=0):
        self.verbose = False
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def publish_objects(
        self, objects: List[Dict], camera_pose: dict = None, maxlen: int = 500  # Limit stream size (default: keep last 500)
    ):
        """Publiziert eine Liste erkannter Objekte in den Redis-Stream.

        Args:
            objects (List[Dict]): Liste erkannter Objekte.
            camera_pose (dict, optional): Pose der Kamera.
            maxlen (int): Maximum number of entries to keep in stream (default: 100).

        Returns:
            str | None: ID der Nachricht im Stream oder None bei Fehler.
        """
        message = {
            "timestamp": str(time.time()),
            "objects": json.dumps(objects),
            "camera_pose": json.dumps(camera_pose) if camera_pose else json.dumps({}),
        }

        try:
            # ✅ FIXED: Use maxlen to prevent unbounded growth
            result = self.client.xadd(
                "detected_objects",
                message,
                maxlen=maxlen,  # Keeps only the last N messages
                approximate=True,  # Allows Redis to optimize trimming
            )

            if self.verbose:
                print(f"Published {len(objects)} objects to Redis stream: {result}")
            return result
        except Exception as e:
            print(f"Error publishing objects: {e}")
            return None

    def get_latest_objects(self, max_age_seconds: float = 2.0) -> List[Dict]:
        """Ruft die neuesten Objektdetektionen ab.

        Args:
            max_age_seconds (float): Maximales Alter der Nachricht in Sekunden.

        Returns:
            List[Dict]: Liste der Objekte oder leere Liste.
        """
        try:
            # Get the latest message from the stream
            messages = self.client.xrevrange("detected_objects", count=1)
            if not messages:
                if self.verbose:
                    print("No messages found in stream")
                return []

            # Parse the latest message
            msg_id, fields = messages[0]

            # Check if message is fresh enough
            msg_timestamp = float(fields.get("timestamp", "0"))
            current_time = time.time()

            if current_time - msg_timestamp > max_age_seconds:
                if self.verbose:
                    print(f"Latest objects too old: {current_time - msg_timestamp:.2f}s > {max_age_seconds}s")
                return []

            # Parse and return objects
            objects_json = fields.get("objects", "[]")
            objects = json.loads(objects_json)
            if self.verbose:
                print(f"Retrieved {len(objects)} fresh objects")
            return objects

        except Exception as e:
            print(f"Error getting latest objects: {e}")
            return []

    def get_objects_in_timerange(self, start_timestamp: float, end_timestamp: float = None) -> List[Dict]:
        """Ruft Objekte in einem bestimmten Zeitbereich ab.

        Args:
            start_timestamp (float): Startzeit als Unix-Timestamp.
            end_timestamp (float, optional): Endzeit als Unix-Timestamp.

        Returns:
            List[Dict]: Liste der Objekte im Zeitintervall.
        """
        if end_timestamp is None:
            end_timestamp = time.time()

        try:
            # Convert timestamps to Redis stream IDs
            start_id = f"{int(start_timestamp * 1000)}-0"
            end_id = f"{int(end_timestamp * 1000)}-0"

            messages = self.client.xrange("detected_objects", start_id, end_id)

            all_objects = []
            for msg_id, fields in messages:
                objects_json = fields.get("objects", "[]")
                objects = json.loads(objects_json)
                all_objects.extend(objects)

            if self.verbose:
                print(f"Retrieved {len(all_objects)} objects from timerange")
            return all_objects

        except Exception as e:
            if self.verbose:
                print(f"Error getting objects in timerange: {e}")
            return []

    def subscribe_objects(self, callback):
        """Startet ein Blocking-Abonnement für Objektdetektionen.

        Args:
            callback (Callable): Callback-Funktion, die ein Dict mit Objekten,
                Kamerapose und Zeitstempel erhält.
        """
        if self.verbose:
            print("Starting to listen for object detections...")
        last_id = "$"  # Start from newest

        try:
            while True:
                # Block for up to 1 second waiting for new messages
                messages = self.client.xread({"detected_objects": last_id}, block=1000)

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
                            print(f"Error processing message {msg_id}: {e}")

        except KeyboardInterrupt:
            print("Stopped listening for object detections")
        except Exception as e:
            print(f"Error in subscribe_objects: {e}")

    def clear_stream(self):
        """Löscht den Stream `detected_objects`.

        Returns:
            int | bool: Anzahl gelöschter Elemente oder False bei Fehler.
        """
        try:
            # Delete the entire stream
            result = self.client.delete("detected_objects")
            if self.verbose:
                print(f"Cleared detected_objects stream: {result}")
            return result
        except Exception as e:
            print(f"Error clearing stream: {e}")
            return False

    def get_stream_info(self):
        """Liest Informationen zum Stream aus.

        Returns:
            dict | None: Stream-Info oder None bei Fehler.
        """
        try:
            info = self.client.xinfo_stream("detected_objects")
            if self.verbose:
                print(f"Stream info: {info}")
            return info
        except Exception as e:
            print(f"Error getting stream info: {e}")
            return None

    def test_connection(self):
        """Testet die Verbindung zu Redis.

        Returns:
            bool: True, wenn Verbindung erfolgreich, sonst False.
        """
        try:
            pong = self.client.ping()
            if self.verbose:
                print(f"Redis connection test: {'OK' if pong else 'FAILED'}")
            return pong
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False
