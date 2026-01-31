"""Utility functions for redis_robot_comm package."""

import time
import functools
import logging
from typing import Callable, TypeVar, Any

from .exceptions import RedisConnectionError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_on_connection_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry Redis operations on connection errors.

    Args:
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except (RedisConnectionError, ConnectionError) as e:
                    if attempt == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {func.__name__}")
                        raise

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            # Should never reach here
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator
