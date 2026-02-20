# Architecture Overview

## System Design

The system is based on a decentralized architecture where Redis acts as the central message broker for all data streams.

### Component Diagram

```mermaid
graph TD
    subgraph Client Applications
        VD[Vision Detection]
        RC[Robot Control]
        MCP[MCP Server]
    end

    subgraph redis_robot_comm
        RMB[RedisMessageBroker]
        RIS[RedisImageStreamer]
        RLM[RedisLabelManager]
        RTO[RedisTextOverlayManager]
    end

    subgraph Redis Infrastructure
        RS[(Redis Streams)]
    end

    VD --> RMB
    VD --> RIS
    RC --> RMB
    RC --> RIS
    MCP --> RTO

    RMB <--> RS
    RIS <--> RS
    RLM <--> RS
    RTO <--> RS
```

## Core Components

### RedisMessageBroker
Responsible for streaming object detection data in JSON format. Supports camera poses and timestamps.

### RedisImageStreamer
Enables streaming of OpenCV images with optional JPEG compression. Optimized for low latency.

### RedisLabelManager
Dynamically manages the list of detectable objects between different processes.

### RedisTextOverlayManager
Synchronizes text overlays for video recordings, categorized into user tasks, robot speech, and system messages.
