# redis_robot_comm

**Redis-basiertes Kommunikations- und Streaming-Package für Roboteranwendungen.**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dokumentation](https://img.shields.io/badge/docs-aktuell-blue.svg)](https://dgaida.github.io/redis_robot_comm/)
[![codecov](https://codecov.io/gh/dgaida/redis_robot_comm/branch/main/graph/badge.svg)](https://codecov.io/gh/dgaida/redis_robot_comm)
[![Tests](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/redis_robot_comm/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Übersicht

Das `redis_robot_comm` Package bietet eine effiziente Redis-basierte Kommunikationsinfrastruktur für Roboteranwendungen. Es ermöglicht den Austausch von Kamerabildern, Objektdetektionen, Metadaten und Text-Overlays zwischen verschiedenen Prozessen oder Systemen in Echtzeit.

### Hauptfunktionen

* 🎯 **Objekterkennung** - Streaming von Detektionsergebnissen
* 📷 **Bild-Streaming** - Variable Bildgrößen mit JPEG-Kompression
* 🏷️ **Label-Verwaltung** - Dynamische Objektlabels
* 📝 **Text-Overlays** - Video-Aufnahme-Integration
* ⚡ **Echtzeitfähig** - Sub-Millisekunden-Latenz für lokale Redis-Server
* 🔄 **Asynchron** - Entkoppelte Producer-Consumer-Architektur
* 📊 **Metadaten** - Automatische Zeitstempel, Roboterposen, Workspace-Informationen
* 🌐 **Bilinguale Dokumentation** - Volle Unterstützung für Deutsch und Englisch

---

## Dokumentation

Die vollständige professionelle Dokumentation finden Sie unter: **[https://dgaida.github.io/redis_robot_comm/](https://dgaida.github.io/redis_robot_comm/)**

Beinhaltet:
- **Erste Schritte** & **Installation**
- **Architektur-Diagramme** (Mermaid)
- **API-Referenz** (Automatisch generiert)
- **Qualitäts-Metriken** & **Changelog**

---

## Schnellstart

### 1. Objekterkennung

```python
from redis_robot_comm import RedisMessageBroker
broker = RedisMessageBroker()
broker.publish_objects([{"id": "obj_1", "class_name": "cube", "confidence": 0.95}])
latest = broker.get_latest_objects()
```

### 2. Bild-Streaming

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

*Erfordert Redis-Server ≥ 5.0*

---

## Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**. Siehe [LICENSE](LICENSE) for details.

---

## Autor

**Daniel Gaida**  
E-Mail: daniel.gaida@th-koeln.de  
GitHub: [@dgaida](https://github.com/dgaida)
