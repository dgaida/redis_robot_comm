"""Custom exceptions for redis_robot_comm package."""


class RedisRobotCommError(Exception):
    """Base exception for redis_robot_comm package."""

    pass


class RedisConnectionError(RedisRobotCommError):
    """Raised when Redis connection fails."""

    pass


class RedisPublishError(RedisRobotCommError):
    """Raised when publishing to Redis fails."""

    pass


class RedisRetrievalError(RedisRobotCommError):
    """Raised when retrieving from Redis fails."""

    pass


class InvalidImageError(RedisRobotCommError):
    """Raised when image data is invalid."""

    pass
