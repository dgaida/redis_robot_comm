# redis_robot_comm

**Redis-based communication and streaming package for robotics applications**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dgaida.github.io/redis_robot_comm/)
[![codecov](https://codecov.io/gh/dgaida/redis_robot_comm/branch/main/graph/badge.svg)](https://codecov.io/gh/dgaida/redis_robot_comm)
[![Tests](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

`redis_robot_comm` provides a high-performance Redis-based communication infrastructure for robotics applications. It enables real-time exchange of camera images, object detections, metadata, and text overlays between distributed processes with sub-millisecond latency.

### Key Features

* 🎯 **Object Detection Streaming** - Publish and consume detection results
* 📷 **Variable-Size Image Streaming** - JPEG-compressed or raw image transfer
* 🏷️ **Label Management** - Dynamic object label configuration
* 📝 **Text Overlay Support** - Video recording integration
* ⚡ **Real-Time Performance** - Sub-millisecond latency for local Redis servers
* 🔄 **Asynchronous Architecture** - Decoupled producer-consumer patterns
* 📊 **Rich Metadata Support** - Timestamps, camera poses, workspace information
* 🌐 **Bilingual Documentation** - Full support for German and English

---

## Documentation

Full professional documentation is available at: **[https://dgaida.github.io/redis_robot_comm/](https://dgaida.github.io/redis_robot_comm/)**

Includes:
- **Getting Started** & **Installation**
- **Architecture Diagrams** (Mermaid)
- **API Reference** (Auto-generated)
- **Quality Metrics** & **Changelog**

---

## Quick Start

### 1. Object Detection

```python
from redis_robot_comm import RedisMessageBroker
broker = RedisMessageBroker()
broker.publish_objects([{"id": "obj_1", "class_name": "cube", "confidence": 0.95}])
latest = broker.get_latest_objects()
```

### 2. Image Streaming

```python
from redis_robot_comm import RedisImageStreamer
streamer = RedisImageStreamer()
streamer.publish_image(frame, compress_jpeg=True, quality=85)
img, metadata = streamer.get_latest_image()
```

---

## Installation

```bash
pip install redis_robot_comm
```

*Requires Redis Server ≥ 5.0*

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Author

**Daniel Gaida**  
Email: daniel.gaida@th-koeln.de  
GitHub: [@dgaida](https://github.com/dgaida)
