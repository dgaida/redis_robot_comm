# Architektur-Übersicht

## Systemdesign

Das System basiert auf einer dezentralen Architektur, bei der Redis als zentraler Message Broker für alle Datenströme fungiert.

### Komponenten-Diagramm

```mermaid
graph TD
    subgraph Client-Anwendungen
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

    subgraph Redis-Infrastruktur
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

## Kernkomponenten

### RedisMessageBroker
Zuständig für das Streaming von Objekterkennungsdaten im JSON-Format. Unterstützt Kameraposen und Zeitstempel.

### RedisImageStreamer
Ermöglicht das Streaming von OpenCV-Bildern mit optionaler JPEG-Kompression. Optimiert für niedrige Latenz.

### RedisLabelManager
Verwaltet dynamisch die Liste der erkennbaren Objekte zwischen verschiedenen Prozessen.

### RedisTextOverlayManager
Synchronisiert Texteinblendungen für Videoaufzeichnungen, unterteilt in Benutzeraufgaben, Robotersprache und Systemnachrichten.
