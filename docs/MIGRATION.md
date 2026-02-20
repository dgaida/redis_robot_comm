# Migration Guide

## Upgrading to 0.2.0

### Breaking Changes

1. **Type Hints Added**
   - All public methods now have type hints.
   - If using mypy, you may need to update type annotations in your code.

2. **Custom Exceptions**
   - Generic exceptions replaced with specific exception types in `redis_robot_comm.exceptions`.
   - Update exception handling:

   **Before:**
   ```python
   try:
       broker.publish_objects(objects)
   except Exception as e:
       print(f"Error: {e}")
   ```

   **After:**
   ```python
   from redis_robot_comm.exceptions import RedisPublishError

   try:
       broker.publish_objects(objects)
   except RedisPublishError as e:
       logger.error(f"Failed to publish: {e}")
   ```

3. **Input Validation**
   - Public methods now validate inputs (e.g., images, object lists).
   - Passing invalid data (e.g., empty lists or non-numpy arrays where expected) will now raise specific exceptions like `RedisPublishError` or `InvalidImageError`.

4. **Return Types**
   - Some methods that previously returned integers or complex types now return booleans for success/failure or more consistent types.
   - `clear_stream()` now returns `bool`.

### New Features

1. **Formal Configuration**
   - Use `RedisConfig` for centralized management of connection parameters.
   - Environment variables (`REDIS_HOST`, etc.) are now automatically picked up.

2. **Retry Logic**
   - Core operations now automatically retry on connection failures.

3. **Enhanced Logging**
   - The package now uses standard Python `logging` instead of `print` for most status messages.
