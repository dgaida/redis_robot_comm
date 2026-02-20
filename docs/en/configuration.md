# Configuration

`redis_robot_comm` can be configured via environment variables or directly in the code.

## Environment Variables

The following environment variables are supported:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `REDIS_HOST` | Redis server hostname | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `REDIS_DB` | Redis database index | `0` |
| `REDIS_PASSWORD` | Redis password (optional) | `None` |

## Configuration in Code

You can pass configuration parameters when initializing the manager classes:

```python
from redis_robot_comm import RedisMessageBroker

# Explicit parameters
broker = RedisMessageBroker(
    host="192.168.1.100",
    port=6380,
    db=1,
    stream_name="my_objects"
)
```

Alternatively, you can use a `RedisConfig` object:

```python
from redis_robot_comm.config import RedisConfig
from redis_robot_comm import RedisImageStreamer

config = RedisConfig(
    host="localhost",
    port=6379,
    password="secret_password"
)

streamer = RedisImageStreamer(config=config)
```

## Logging Configuration

All classes have a `verbose` attribute to enable detailed logging output:

```python
broker = RedisMessageBroker()
broker.verbose = True
```
