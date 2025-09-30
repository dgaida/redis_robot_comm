"""
redis_image_streamer
====================

A small, dependency‑light helper for streaming OpenCV images of **any size** through a Redis stream.

Why Redis Streams?
------------------
* **Variable‑size payloads** – Redis streams can store entries of arbitrary length, making them perfect for images that change resolution or format.
* **Back‑pressure handling** – the `maxlen` argument allows you to keep a bounded queue (e.g. the last 5 frames) without manual eviction logic.
* **Low latency** – a single `xadd` is <1 ms on a local broker; read with `xread` or `xrevrange` to get the newest frame.
* **Robustness** – Redis guarantees durability and atomicity for each entry, which is critical in robotics pipelines.

Typical Use‑Case
----------------
1. Capture a frame from a camera with OpenCV.
2. Publish it with :func:`publish_image`.  
3. In a different process, either fetch the last frame with :func:`get_latest_image` or stream frames continuously with :func:`subscribe_variable_images`.

Dependencies
------------
* `redis-py <https://pypi.org/project/redis/>`_ – the official Redis client for Python.
* `opencv‑python <https://pypi.org/project/opencv-python/>`_ – for image encoding/decoding.
* `numpy <https://pypi.org/project/numpy/>`_ – for array manipulation.
"""

import redis
import cv2
import base64
import json
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any


class RedisImageStreamer:
    """
    A Redis‑backed stream that can publish and consume OpenCV images of arbitrary size.

    The class serialises an image (either as raw bytes or JPEG) and stores it in a Redis stream.
    Each entry contains metadata such as the image shape, data type, and optional custom fields
    (e.g. robot pose, workspace id).

    Parameters
    ----------
    host : str, default ``'localhost'``
        Redis server hostname or IP address.
    port : int, default ``6379``
        Redis server port.
    stream_name : str, default ``'robot_camera'``
        Name of the stream that will hold the image frames.

    Example
    -------
    >>> streamer = RedisImageStreamer(host='redis.example.com', port=6380)
    >>> frame = cv2.imread('camera_view.png')
    >>> stream_id = streamer.publish_image(frame, metadata={'pose': [0, 0, 0]})
    >>> image, meta = streamer.get_latest_image()
    >>> print(f'Received frame {stream_id} with pose {meta["pose"]}')
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, stream_name: str = 'robot_camera'):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.stream_name = stream_name

    # --------------------------------------------------------------------
    # Publishing API
    # --------------------------------------------------------------------
    def publish_image(self, image: np.ndarray, metadata: Dict[str, Any] = None,
                      compress_jpeg: bool = True, quality: int = 80, maxlen: int = 5) -> str:
        """
        Publish a single image frame to the Redis stream.

        The image is serialised in one of two ways:
        * **JPEG** – compressed with the supplied quality (fast, smaller payload).
        * **Raw** – the raw NumPy bytes (larger payload, but lossless).

        Parameters
        ----------
        image : np.ndarray
            OpenCV image array (H×W×C or H×W for grayscale).
        metadata : dict, optional
            Arbitrary key/value pairs that will be stored alongside the frame.
            Typical values: robot pose, timestamp, workspace id, etc.
        compress_jpeg : bool, default ``True``
            When ``True`` the image is compressed to JPEG before base64 encoding.
        quality : int, 1‑100, default ``80``
            JPEG compression quality (ignored if ``compress_jpeg`` is ``False``).
        maxlen : int, default ``5``
            Maximum number of entries to keep in the stream.  Redis will automatically
            remove the oldest entries when this limit is exceeded.

        Returns
        -------
        str
            The unique Redis entry ID (e.g. ``'1589456675-0'``).

        Raises
        ------
        ValueError
            If the supplied image is not a NumPy array or is empty.

        Example
        -------
        >>> streamer.publish_image(cv2.imread('sample.png'), metadata={'id': 42})
        '1589456675-0'
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("`image` must be a non‑empty NumPy array")

        timestamp = time.time()

        # Handle different image sizes dynamically
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        if compress_jpeg:
            # Compress to JPEG (handles any size)
            _, buffer = cv2.imencode('.jpg', image,
                                     [cv2.IMWRITE_JPEG_QUALITY, quality])
            image_data = base64.b64encode(buffer).decode('utf-8')
            format_type = 'jpeg'
            compressed_size = len(buffer)
        else:
            # Raw image data
            image_data = base64.b64encode(image.tobytes()).decode('utf-8')
            format_type = 'raw'
            compressed_size = image.nbytes

        # Prepare message with dynamic image info
        message = {
            'timestamp': str(timestamp),
            'image_data': image_data,
            'format': format_type,
            'width': str(width),
            'height': str(height),
            'channels': str(channels),
            'dtype': str(image.dtype),
            'compressed_size': str(compressed_size),
            'original_size': str(image.nbytes)
        }

        # Add optional metadata
        if metadata:
            message['metadata'] = json.dumps(metadata)

        # Publish to Redis stream (automatically handles variable sizes)
        stream_id = self.client.xadd(self.stream_name, message, maxlen=maxlen)

        print(f"Published {width}x{height} image ({compressed_size} bytes)")
        return stream_id

    # --------------------------------------------------------------------
    # Retrieval API
    # --------------------------------------------------------------------
    def get_latest_image(self, timeout_ms: int = 1000) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve the newest frame from the stream.

        The method performs a reverse range query (`xrevrange`) limited to a single entry
        and then decodes the stored image and metadata.

        Returns
        -------
        tuple
            ``(image_array, metadata_dict)`` if a frame is present, otherwise ``None``.

        Raises
        ------
        RuntimeError
            If the stream does not exist or is inaccessible.

        Example
        -------
        >>> image, meta = streamer.get_latest_image()
        >>> if image is not None:
        ...     cv2.imshow('Latest', image)
        ...     cv2.waitKey(0)
        """
        try:
            messages = self.client.xrevrange(self.stream_name, count=1)
            if not messages:
                return None

            msg_id, fields = messages[0]
            return self._decode_variable_image(fields)

        except Exception as e:
            print(f"Error getting latest image: {e}")
            return None

    def subscribe_variable_images(self, callback, block_ms: int = 1000, start_after: str = "$"):
        """
        Continuously listen for new frames and invoke ``callback`` for each one.

        This method blocks the current thread and will only return on a keyboard interrupt
        or a raised exception.  The callback receives three arguments:
        ``(image, metadata, image_info)``.

        Parameters
        ----------
        callback : callable
            Function signature: ``callback(image, metadata, image_info)``.
            * ``image`` – NumPy array of the decoded frame.
            * ``metadata`` – decoded metadata dictionary (empty if none).
            * ``image_info`` – dictionary with useful image attributes
              (width, height, channels, timestamp, compressed_size, original_size).
        block_ms : int, default ``1000``
            Timeout for the underlying Redis `xread`.  If no new frames appear within
            this period the loop continues – useful for graceful shutdown.
        start_after : str, default ``'$'``
            Redis stream ID after which to start reading.  ``'$'`` means "only new frames".

        Example
        -------
        >>> def on_frame(img, meta, info):
        ...     print(f"Received {info['width']}×{info['height']} frame")
        ...     cv2.imshow('Camera', img)
        ...     cv2.waitKey(1)
        >>> streamer.subscribe_variable_images(on_frame, block_ms=500)
        """
        last_id = start_after
        print("Subscribing to variable-size image stream...")

        while True:
            try:
                messages = self.client.xread(
                    {self.stream_name: last_id},
                    block=block_ms,
                    count=1
                )

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        result = self._decode_variable_image(fields)
                        if result:
                            image, metadata = result

                            # Prepare image info for callback
                            image_info = {
                                'width': image.shape[1],
                                'height': image.shape[0],
                                'channels': image.shape[2] if len(image.shape) == 3 else 1,
                                'timestamp': float(fields.get('timestamp', '0')),
                                'compressed_size': int(fields.get('compressed_size', '0')),
                                'original_size': int(fields.get('original_size', '0'))
                            }

                            callback(image, metadata, image_info)
                        last_id = msg_id

            except KeyboardInterrupt:
                print("Stopped subscribing to images")
                break
            except Exception as e:
                print(f"Error in image subscription: {e}")
                time.sleep(0.1)

    # --------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------
    def _decode_variable_image(self, fields) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Decode a Redis stream entry that contains an image.

        This private helper is used by :meth:`get_latest_image` and
        :meth:`subscribe_variable_images`.  It performs base64 decoding, JPEG
        decompression (if needed) and NumPy reconstruction.

        Parameters
        ----------
        fields : dict
            key/value pairs from a Redis entry.  Keys are returned as strings
            (because `decode_responses=True` is set in the constructor).

        Returns
        -------
        tuple
            ``(image_array, metadata_dict)`` or ``None`` if decoding fails.
        """
        try:
            # Extract image parameters
            width = int(fields['width'])
            height = int(fields['height'])
            channels = int(fields['channels'])
            format_type = fields['format']

            # Decode image data
            image_data = base64.b64decode(fields['image_data'])

            if format_type == 'jpeg':
                # Decode JPEG (automatically handles any size)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("JPEG decoding returned None")
            else:
                # Decode raw image with specified dimensions
                dtype = fields['dtype']
                if channels == 1:
                    shape = (height, width)
                else:
                    shape = (height, width, channels)
                image = np.frombuffer(image_data, dtype=dtype).reshape(shape)

            # Extract metadata if available
            metadata = {}
            if 'metadata' in fields:
                metadata = json.loads(fields['metadata'])

            return image, metadata

        except Exception as e:
            print(f"Error decoding variable image: {e}")
            return None

    def get_stream_stats(self) -> Dict[str, Any]:
        """
        Retrieve bookkeeping information about the Redis stream.

        Returns
        -------
        dict
            Keys:
            * ``total_messages`` – total entries currently in the stream.
            * ``first_entry_id`` – ID of the oldest entry (``None`` if stream is empty).
            * ``last_entry_id``   – ID of the newest entry (``None`` if stream is empty).

        Example
        -------
        >>> stats = streamer.get_stream_stats()
        >>> print(stats["total_messages"])
        """
        try:
            info = self.client.xinfo_stream(self.stream_name)
            return {
                'total_messages': info['length'],
                'first_entry_id': info['first-entry'][0] if info['first-entry'] else None,
                'last_entry_id': info['last-entry'][0] if info['last-entry'] else None
            }
        except:
            return {'error': 'Stream not found or empty'}
