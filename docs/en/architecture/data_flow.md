# Data Flow

Data exchange occurs asynchronously via Redis Streams. This allows for decoupling of producers (e.g., camera drivers, AI models) and consumers (e.g., robot control, visualization).

## Image Streaming Sequence

```mermaid
sequenceDiagram
    participant P as Publisher (e.g., Camera)
    participant R as Redis Stream
    participant C as Consumer (e.g., GUI)

    P->>P: Capture Image
    P->>P: Compress to JPEG
    P->>R: XADD robot_camera (image_data, metadata)
    R-->>P: Stream ID

    C->>R: XREAD/XREVRANGE robot_camera
    R-->>C: Image Data + Metadata
    C->>C: Decode JPEG
    C->>C: Display Image
```

## Object Detection Workflow

```mermaid
flowchart LR
    Img[Image Source] --> Det[Detector]
    Det --> |Object List| Broker[RedisMessageBroker]
    Broker --> |XADD| Stream[(Redis Stream)]
    Stream --> |XREAD| Control[Robot Control]
    Control --> |Action| Robot[Robot]
```
