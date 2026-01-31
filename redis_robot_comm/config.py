"""Configuration management for redis_robot_comm."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    max_connections: int = 50

    @classmethod
    def from_url(cls, url: str) -> "RedisConfig":
        """
        Create configuration from Redis URL.

        Args:
            url: Redis URL (e.g., redis://localhost:6379/0)

        Returns:
            RedisConfig instance.
        """
        import redis

        connection_pool = redis.ConnectionPool.from_url(url)
        conn_kwargs = connection_pool.connection_kwargs

        return cls(
            host=conn_kwargs.get("host", "localhost"),
            port=conn_kwargs.get("port", 6379),
            db=conn_kwargs.get("db", 0),
            password=conn_kwargs.get("password"),
        )


@dataclass
class StreamConfig:
    """Stream-specific configuration."""

    max_length: int = 500
    approximate_trim: bool = True
    blocking_timeout_ms: int = 1000


@dataclass
class ImageStreamConfig(StreamConfig):
    """Image streaming configuration."""

    default_quality: int = 80
    max_quality: int = 100
    min_quality: int = 1
    default_compression: bool = True
    max_image_dimension: int = 10000


# Global configuration instance
_redis_config: Optional[RedisConfig] = None


def get_redis_config() -> RedisConfig:
    """Get global Redis configuration."""
    global _redis_config
    if _redis_config is None:
        _redis_config = RedisConfig()
    return _redis_config


def set_redis_config(config: RedisConfig) -> None:
    """Set global Redis configuration."""
    global _redis_config
    _redis_config = config
