# Installation Instructions

## Requirements

### System Requirements

- **Redis Server** - Version 5.0+ (for Streams support)
- **Python** - 3.8 or higher
- **Operating System** - Linux, macOS, or Windows

### Python Dependencies

- `redis>=4.0.0` - Redis client library
- `opencv-python>=4.5.0` - Image processing
- `numpy>=1.20.0` - Array operations

### Development Dependencies

See [requirements-dev.txt](../requirements-dev.txt) for complete list.

---

## Getting Started

### 1. Start Redis Server

**Using Docker (Recommended):**
```bash
docker run -p 6379:6379 redis:alpine
```

**Or install locally:**

Ubuntu/Debian:
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

macOS:
```bash
brew install redis
brew services start redis
```

Windows:
```bash
# Download from https://redis.io/download
# Or use WSL with Linux instructions
```

### 2. Install Package

```bash
git clone https://github.com/dgaida/redis_robot_comm.git
cd redis_robot_comm
pip install -e .
```

### 3. Verify Installation

```python
from redis_robot_comm import RedisMessageBroker

broker = RedisMessageBroker()
if broker.test_connection():
    print("✓ Connected to Redis")
else:
    print("✗ Connection failed")
```

### 4. Run Example

```bash
python examples/main.py
```

---

## Related Documentation

- **[API Reference](api.md)** - Complete API documentation
- **[Testing Guide](TESTING.md)** - Testing guidelines and examples
- **[Contributing Guide](../CONTRIBUTING.md)** - Development setup and contribution guidelines
- **[Main README](../README.md)** - Installation and quick start

## External Resources

- **[vision_detect_segment](https://github.com/dgaida/vision_detect_segment)** - Object detection integration
- **[robot_environment](https://github.com/dgaida/robot_environment)** - Robot control integration
- **[robot_mcp](https://github.com/dgaida/robot_mcp)** - MCP server integration

---

## License

This project is licensed under the **MIT License**. See [../LICENSE](../LICENSE) for details.

## Contact

**Daniel Gaida**  
Email: daniel.gaida@th-koeln.de  
GitHub: [@dgaida](https://github.com/dgaida)

Project Link: https://github.com/dgaida/redis_robot_comm
