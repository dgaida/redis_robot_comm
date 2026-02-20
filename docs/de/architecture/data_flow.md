# Datenfluss

Der Datenaustausch erfolgt asynchron über Redis Streams. Dies ermöglicht eine Entkopplung von Produzenten (z. B. Kamera-Treiber, KI-Modelle) und Konsumenten (z. B. Robotersteuerung, Visualisierung).

## Bild-Streaming-Sequenz

```mermaid
sequenceDiagram
    participant P as Publisher (z.B. Kamera)
    participant R as Redis Stream
    participant C as Consumer (z.B. GUI)

    P->>P: Erfasse Bild
    P->>P: Komprimiere zu JPEG
    P->>R: XADD robot_camera (image_data, metadata)
    R-->>P: Stream ID

    C->>R: XREAD/XREVRANGE robot_camera
    R-->>C: Bilddaten + Metadaten
    C->>C: Dekodiere JPEG
    C->>C: Zeige Bild an
```

## Objekterkennungs-Workflow

```mermaid
flowchart LR
    Img[Bildquelle] --> Det[Detektor]
    Det --> |Objektliste| Broker[RedisMessageBroker]
    Broker --> |XADD| Stream[(Redis Stream)]
    Stream --> |XREAD| Control[Robotersteuerung]
    Control --> |Aktion| Robot[Roboter]
```
