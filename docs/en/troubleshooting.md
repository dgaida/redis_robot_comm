# Troubleshooting

Here you will find solutions to common problems.

## Redis Connection Issues

### Error Message: `Failed to connect to Redis`

**Possible Causes:**
1. Redis server is not running.
2. Incorrect host or port configuration.
3. Firewall is blocking access.

**Solutions:**
* Check if Redis is running: `redis-cli ping` should respond with `PONG`.
* Ensure the Docker container is running (if used): `docker ps`.
* Check your `REDIS_HOST` and `REDIS_PORT` settings.

## Performance Issues

### High Latency during Image Streaming

**Possible Causes:**
1. Large images without compression.
2. Network bandwidth is saturated.
3. High CPU load due to JPEG encoding.

**Solutions:**
* Enable JPEG compression: `compress_jpeg=True`.
* Reduce JPEG quality: `quality=70`.
* Decrease image resolution before sending.

## Data Loss

### Old Messages Disappear from the Stream

**Cause:**
* The `maxlen` parameters (default 500 for objects, 5 for images) limit the number of stored entries.

**Solution:**
* Increase the `maxlen` value when publishing if you need a larger buffer.

## Debugging

Enable `verbose` mode to get more information about internal operations:

```python
streamer = RedisImageStreamer()
streamer.verbose = True
```
