"""Base class for video recording with Redis integration."""

import cv2
import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

from redis_robot_comm import RedisImageStreamer

logger = logging.getLogger(__name__)


class BaseVideoRecorder(ABC):
    """Base class for video recording with Redis integration."""

    def __init__(
        self,
        camera_id: int = 0,
        stream_name: str = "annotated_camera",
        host: str = "localhost",
        port: int = 6379,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        output_file: Optional[str] = None,
        codec: str = "mp4v",
    ):
        """
        Initialize base recorder.

        Args:
            camera_id: USB camera device ID.
            stream_name: Redis stream name for annotated images.
            host: Redis server host.
            port: Redis server port.
            fps: Recording frame rate.
            width: Camera frame width.
            height: Camera frame height.
            output_file: Output video file path.
            codec: Video codec.
        """
        self.camera_id = camera_id
        self.stream_name = stream_name
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec

        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"recording_{timestamp}.mp4"
        else:
            self.output_file = output_file

        self._init_camera()
        self._init_redis(host, port, stream_name)

        # State
        self.recording = True
        self.paused = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_annotated_frame = None

    def _init_camera(self) -> None:
        """Initialize camera capture."""
        logger.info(f"Opening camera {self.camera_id}...")
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info("✓ Camera opened")

    def _init_redis(self, host: str, port: int, stream_name: str) -> None:
        """Initialize Redis connection."""
        try:
            self.image_streamer = RedisImageStreamer(
                host=host, port=port, stream_name=stream_name
            )
            logger.info(f"✓ Connected to Redis: {stream_name}")
        except Exception as e:
            self.camera.release()
            raise RuntimeError(f"Failed to connect to Redis: {e}")

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if needed."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def create_placeholder_frame(self, text: str = "Waiting for annotated frames...") -> np.ndarray:
        """Create placeholder frame when annotated frame is not available."""
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            text,
            ((self.width - 400) // 2, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (128, 128, 128),
            2,
        )
        return placeholder

    @abstractmethod
    def process_frame(
        self, camera_frame: np.ndarray, annotated_frame: Optional[np.ndarray]
    ) -> np.ndarray:
        """Process and combine frames. Must be implemented by subclasses."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        if hasattr(self, 'camera'):
            self.camera.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        logger.info(f"Recording finished. Saved to {self.output_file}")
        logger.info(f"Total frames: {self.frame_count}, Duration: {elapsed:.2f}s")
