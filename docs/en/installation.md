# Installation

This section describes how to install and set up `redis_robot_comm`.

## Install Package

You can install the package directly from the source code:

```bash
git clone https://github.com/dgaida/redis_robot_comm.git
cd redis_robot_comm
pip install -e .
```

## Set up Redis Server

`redis_robot_comm` requires a running Redis server (version ≥ 5.0).

### Using Docker (Recommended)

This is the easiest way to start Redis quickly:

```bash
docker run -d -p 6379:6379 --name redis-robot redis:alpine
```

### Local Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

## Verify Installation

After installation, you can test the connection with a simple script:

```python
from redis_robot_comm import RedisMessageBroker

try:
    broker = RedisMessageBroker()
    if broker.test_connection():
        print("✓ Installation successful: Connection to Redis established.")
    else:
        print("✗ Connection to Redis failed.")
except Exception as e:
    print(f"✗ Verification error: {e}")
```
