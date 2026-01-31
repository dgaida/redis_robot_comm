# Architecture Overview

## System Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Client Applications                    │
│  (Vision Detection, Robot Control, MCP Server, etc.)   │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
             ↓                            ↓
    ┌────────────────┐          ┌────────────────────┐
    │ RedisMessage   │          │ RedisImage         │
    │ Broker         │          │ Streamer           │
    └────────┬───────┘          └─────────┬──────────┘
             │                            │
             └──────────┬─────────────────┘
                        ↓
             ┌──────────────────────┐
             │   Redis Server       │
             │   (Streams)          │
             └──────────────────────┘
```

### Core Classes

#### RedisMessageBroker
- **Purpose:** Manages object detection data streams
- **Key Methods:** `publish_objects()`, `get_latest_objects()`, `subscribe_objects()`
- **Stream:** `detected_objects`
- **Data Format:** JSON-encoded object lists with camera poses

#### RedisImageStreamer
- **Purpose:** Handles variable-size image streaming
- **Key Methods:** `publish_image()`, `get_latest_image()`, `subscribe_variable_images()`
- **Stream:** Configurable (default: `robot_camera`)
- **Compression:** Optional JPEG compression

#### RedisLabelManager
- **Purpose:** Manages detectable object labels
- **Key Methods:** `publish_labels()`, `get_latest_labels()`, `add_label()`
- **Stream:** `detectable_labels`
- **Use Case:** Dynamic model configuration

#### RedisTextOverlayManager
- **Purpose:** Text overlays for video recording
- **Key Methods:** `publish_user_task()`, `publish_robot_speech()`, `subscribe_to_texts()`
- **Stream:** `video_text_overlays`
- **Text Types:** `user_task`, `robot_speech`, `system_message`

### Error Handling Strategy

The package uses custom exceptions defined in `redis_robot_comm.exceptions`:
- `RedisConnectionError`: For connection-related issues.
- `RedisPublishError`: When sending data fails.
- `RedisRetrievalError`: When receiving data fails.
- `InvalidImageError`: For malformed image data.

A retry mechanism is implemented via the `@retry_on_connection_error` decorator to handle transient network issues.

### Configuration Management

Configuration is centralized in `redis_robot_comm.config`. The `RedisConfig` dataclass supports initialization from environment variables:
- `REDIS_HOST`
- `REDIS_PORT`
- `REDIS_DB`
- `REDIS_PASSWORD`
