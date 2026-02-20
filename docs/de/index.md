# redis_robot_comm

**Redis-basiertes Kommunikations- und Streaming-Package für Roboteranwendungen.**

Das `redis_robot_comm` Package bietet eine effiziente Redis-basierte Kommunikationsinfrastruktur für Roboteranwendungen. Es ermöglicht den Austausch von Kamerabildern, Objektdetektionen, Metadaten und Text-Overlays zwischen verschiedenen Prozessen oder Systemen in Echtzeit.

## Hauptfunktionen

* 📦 **Objekterkennung** - Streaming von Detektionsergebnissen über `RedisMessageBroker`
* 📷 **Bild-Streaming** - Variable Bildgrößen mit JPEG-Kompression über `RedisImageStreamer`
* 🏷️ **Label-Verwaltung** - Dynamische Objektlabels mit `RedisLabelManager`
* 📝 **Text-Overlays** - Video-Aufnahme-Integration mit `RedisTextOverlayManager`
* ⚡ **Echtzeitfähig** - Sub-Millisekunden-Latenz für lokale Redis-Server
* 🔄 **Asynchron** - Entkoppelte Producer-Consumer-Architektur
* 📊 **Metadaten** - Automatische Zeitstempel, Roboterposen, Workspace-Informationen
* 🎯 **Robotik-optimiert** - Speziell für Pick-and-Place und Vision-Anwendungen

## Anwendungsfälle

Das Package wird in Robotik-Frameworks als Kommunikations-Backbone eingesetzt:

- **vision_detect_segment** - Objekterkennung mit OwlV2, YOLO-World, YOLOE, Grounding-DINO
- **robot_environment** - Robotersteuerung mit visueller Objekterkennung
- **robot_mcp** - LLM-basierte Robotersteuerung mit MCP

## Datenfluss

![Datenfluss via Redis Streams](../assets/images/workflow_detector.png)
