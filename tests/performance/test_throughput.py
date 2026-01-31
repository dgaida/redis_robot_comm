"""Performance benchmarks for redis_robot_comm."""

import pytest
import time
import numpy as np
from redis_robot_comm import RedisMessageBroker, RedisImageStreamer


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    def test_object_publish_throughput(self, benchmark, message_broker, mock_redis_client):
        """Benchmark object publishing throughput."""
        mock_redis_client.xadd.return_value = "1-0"

        objects = [
            {"id": f"obj_{i}", "class_name": "cube", "confidence": 0.95}
            for i in range(100)
        ]

        def publish_batch():
            for _ in range(10):
                message_broker.publish_objects(objects)

        benchmark(publish_batch)
        # Benchmark results will be displayed in the summary table

    def test_image_compression_performance(self, benchmark, image_streamer, mock_redis_client):
        """Benchmark image compression performance."""
        mock_redis_client.xadd.return_value = "1-0"

        # 640x480 RGB image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        def publish_image():
            image_streamer.publish_image(image, compress_jpeg=True, quality=85)

        benchmark(publish_image)
        # Benchmark results will be displayed in the summary table
