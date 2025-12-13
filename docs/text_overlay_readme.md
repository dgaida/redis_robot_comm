# Text Overlay System für Robot Videos

Diese Dokumentation beschreibt das neue Text-Overlay-System für Video-Aufnahmen mit Roboter-Kommentaren und Aufgabenstellungen.

## 🎯 Überblick

Das System ermöglicht die Anzeige von:
- **Benutzer-Aufgaben** (persistent während der Aufnahme)
- **Roboter-Aussagen** (zeitlich begrenzt, 4-5 Sekunden)
- **TH Köln Branding** (Logo und Beschriftung)

### Architektur

```
┌─────────────────┐
│   MCP Server    │
│  (Unified)      │
└────────┬────────┘
         │ publishes
         ↓
┌─────────────────┐
│  Redis Stream   │
│ "text_overlays" │
└────────┬────────┘
         │ subscribes
         ↓
┌─────────────────┐
│ Recording Script│
│ (Enhanced)      │
└─────────────────┘
```

## 📦 Neue Komponenten

### 1. RedisTextOverlayManager

Verwaltet Text-Overlays über Redis Streams.

**Datei:** `redis_robot_comm/redis_text_overlay_manager.py`

**Verwendung:**

```python
from redis_robot_comm import RedisTextOverlayManager

# Initialisierung
text_mgr = RedisTextOverlayManager()

# User-Aufgabe publishen
text_mgr.publish_user_task(
    task="Pick up the pencil and place it next to the cube"
)

# Roboter-Aussage publishen
text_mgr.publish_robot_speech(
    speech="🤖 I'm picking up the pencil now",
    duration_seconds=4.0
)

# System-Nachricht
text_mgr.publish_system_message(
    message="🎥 Recording started",
    duration_seconds=3.0
)
```

### 2. Unified MCP Server

Kombiniert alle drei Server-Versionen:
- Basis-Validierung
- LLM-Erklärungen (communicative)
- Config-Management

**Datei:** `server/fastmcp_robot_server_unified.py`

**Neue Features:**
- Automatisches Publishen von User-Tasks zu Redis
- Automatisches Publishen von Robot-Speech zu Redis
- Neues Tool: `set_user_task(task: str)`

**Start:**

```bash
# Mit Explanations und Text Overlays
python server/fastmcp_robot_server_unified.py --robot niryo

# Ohne Explanations
python server/fastmcp_robot_server_unified.py --robot niryo --no-explanations

# Mit Config-File
python server/fastmcp_robot_server_unified.py --config config.yaml
```

### 3. Enhanced Recording Script

Erweitertes Recording-Script mit Text-Overlays.

**Datei:** `scripts/record_camera_with_overlays.py`

**Features:**
- Anzeige von User-Aufgaben (oben, persistent)
- Anzeige von Roboter-Aussagen (Mitte, timed)
- TH Köln Branding (unten rechts)
- Zwei Layouts: side-by-side (1280x720) und overlay (640x480)

## 🚀 Verwendung

### Kompletter Workflow

**Terminal 1: Redis starten**
```bash
docker run -p 6379:6379 redis:alpine
```

**Terminal 2: Object Detection starten**
```bash
cd ../vision_detect_segment
python scripts/detect_objects_publish_annotated_frames.py
```

**Terminal 3: MCP Server starten (unified)**
```bash
cd ../robot_mcp
python server/fastmcp_robot_server_unified.py --robot niryo --no-simulation
```

**Terminal 4: Recording starten**
```bash
cd ../redis_robot_comm
python scripts/record_camera_with_overlays.py \
  --camera 0 \
  --stream annotated_camera \
  --layout side-by-side
```

**Terminal 5: MCP Client starten**
```bash
cd ../robot_mcp
python client/fastmcp_universal_client.py
```

### Im Client

```python
# User-Aufgabe wird automatisch zu Redis gepublished
You: Pick up the pencil and place it next to the red cube

# Roboter-Erklärungen werden automatisch zu Redis gepublished
🤖 I'm moving to observe the workspace...
🔍 Let me scan for objects...
🤖 I'm picking up the pencil now...
```

## 📐 Video-Layouts

### Side-by-Side Layout (1280x720)

```
┌──────────────┬──────────────┐
│   Camera     │  Annotated   │  ← 1280x480
│   Feed       │  Frame       │
├──────────────┴──────────────┤
│  AUFGABE: [User Task]       │
│  Pick up the pencil...       │
│                              │  ← 1280x240
│  ROBOTER: [Robot Speech]    │    (Text Panel)
│  🤖 I'm picking it up...     │
│                              │
│              [TH Köln Logo]  │
└──────────────────────────────┘
```

**Start:**
```bash
python scripts/record_camera_with_overlays.py --layout side-by-side
```

### Overlay Layout (640x480)

```
┌──────────────────────────────┐
│ ┌──────────────────────────┐ │
│ │ AUFGABE: [User Task]     │ │  ← Transparent overlay
│ │ Pick up the pencil...    │ │
│ └──────────────────────────┘ │
│                              │
│       Camera Feed            │
│   with Annotated Frame       │
│        overlaid              │
│                              │
│ ┌──────────────────────────┐ │
│ │ ROBOTER: [Robot Speech]  │ │  ← Transparent overlay
│ │ 🤖 I'm picking it up...  │ │
│ └──────────────────────────┘ │
│                  [TH Logo]   │  ← Transparent overlay
└──────────────────────────────┘
```

**Start:**
```bash
python scripts/record_camera_with_overlays.py --layout overlay
```

## 🎨 Anpassungen

### Text-Anzeigedauer ändern

Im MCP Server (`ExplanationGenerator`):

```python
text_overlay_manager.publish_robot_speech(
    speech=explanation,
    duration_seconds=5.0,  # Statt 4.0
    metadata={"tool_name": tool_name}
)
```

### Logo austauschen

1. Logo-Datei erstellen: `hochschule.png` (150x50 px)
2. In `VideoOverlayRenderer._create_placeholder_logo()` laden:

```python
def _create_placeholder_logo(self) -> np.ndarray:
    """Load TH Köln logo."""
    try:
        logo = cv2.imread("hochschule.png")
        if logo is None:
            return self._create_fallback_logo()
        return cv2.resize(logo, (150, 50))
    except:
        return self._create_fallback_logo()

def _create_fallback_logo(self) -> np.ndarray:
    """Create fallback logo if file not found."""
    # ... existing placeholder code ...
```

### Farben anpassen

In `VideoOverlayRenderer.__init__()`:

```python
self.color_task = (255, 255, 255)      # Weiß
self.color_speech = (100, 255, 100)    # Hellgrün
self.color_branding = (200, 200, 200)  # Hellgrau
```

### Font-Größe anpassen

In `VideoOverlayRenderer.__init__()`:

```python
self.font_scale_task = 0.7      # Größer
self.font_scale_speech = 0.6    # Größer
```

## 🔧 Troubleshooting

### Keine Text-Overlays im Video

**Problem:** Video zeigt keine Texte an

**Lösung:**
1. Prüfen ob Redis läuft:
   ```bash
   redis-cli ping
   ```

2. Prüfen ob Texte gepublished werden:
   ```bash
   redis-cli xrange video_text_overlays - +
   ```

3. Verbose-Mode aktivieren:
   ```python
   text_mgr = RedisTextOverlayManager()
   text_mgr.verbose = True
   ```

### Texte verschwinden zu schnell

**Problem:** Robot-Speech ist zu kurz sichtbar

**Lösung:** Duration erhöhen:
```python
text_overlay_manager.publish_robot_speech(
    speech=explanation,
    duration_seconds=6.0  # Länger anzeigen
)
```

### Logo wird nicht angezeigt

**Problem:** Placeholder statt echtem Logo

**Lösung:**
1. Logo-Datei `hochschule.png` erstellen (150x50 px)
2. Im Skript-Verzeichnis ablegen
3. Code in `_create_placeholder_logo()` anpassen (siehe oben)

### Video-Auflösung passt nicht

**Problem:** Text-Panel ist zu klein/groß

**Lösung:** Panel-Höhe anpassen:
```python
# In render_text_panel_sidebyside()
text_panel = np.zeros((280, 1280, 3), dtype=np.uint8)  # Statt 240
```

## 📊 Redis Stream-Struktur

### Stream: `video_text_overlays`

**Eintragsformat:**

```json
{
  "timestamp": "1702472834.123",
  "text": "Pick up the pencil and place it next to the cube",
  "type": "user_task",
  "metadata": {
    "session_id": "abc123"
  }
}
```

**Typen:**
- `user_task` - Benutzer-Aufgabe (persistent)
- `robot_speech` - Roboter-Aussage (timed)
- `system_message` - System-Nachricht (timed)

### Stream-Befehle

```bash
# Alle Einträge anzeigen
redis-cli xrange video_text_overlays - +

# Letzten Eintrag anzeigen
redis-cli xrevrange video_text_overlays + - COUNT 1

# Stream löschen
redis-cli del video_text_overlays

# Stream-Info
redis-cli xinfo stream video_text_overlays
```

## 🎓 Beispiel-Workflow

### Demo-Video erstellen

```bash
# 1. System starten (alle Terminals)
# Redis, Detection, MCP Server, Recording

# 2. Im Client: User-Aufgabe setzen
You: Pick up the pencil and place it next to the red cube

# 3. Roboter führt Aufgabe aus und kommentiert
# - "🤖 I'm moving to observe the workspace..."
# - "🔍 Let me scan for objects..."
# - "🤖 I'm picking up the pencil now..."
# - "✓ Done! The pencil is next to the red cube."

# 4. Video wird automatisch mit allen Texten aufgezeichnet

# 5. Recording stoppen (Taste 'q')

# 6. Video anschauen
vlc recording_20231212_143022.mp4
```

## 📝 Integration in bestehende Projekte

### In MCP Server einbinden

```python
from redis_robot_comm import RedisTextOverlayManager

# Beim Server-Start
text_manager = RedisTextOverlayManager()

# Bei User-Nachricht
@mcp.tool
def handle_user_command(command: str) -> str:
    # User-Aufgabe publishen
    text_manager.publish_user_task(command)

    # ... Aufgabe ausführen ...
    return result

# Bei Tool-Calls (mit Explanation Generator)
explanation = explanation_generator.generate_explanation(...)
text_manager.publish_robot_speech(
    speech=explanation,
    duration_seconds=4.0
)
```

### In eigene Recording-Tools einbinden

```python
from redis_robot_comm import RedisTextOverlayManager

text_manager = RedisTextOverlayManager()

# Texte abrufen
texts = text_manager.get_latest_texts(max_age_seconds=10.0)

for text_data in texts:
    print(f"{text_data['type']}: {text_data['text']}")

# Oder Subscribe
def on_text_update(text_data):
    print(f"New text: {text_data['text']}")

text_manager.subscribe_to_texts(on_text_update)
```

## 🎬 Qualitätseinstellungen

### Für Demo-Videos (höchste Qualität)

```bash
python scripts/record_camera_with_overlays.py \
  --layout side-by-side \
  --fps 30 \
  --codec H264 \
  --width 640 \
  --height 480
```

### Für Dokumentation (kompakt)

```bash
python scripts/record_camera_with_overlays.py \
  --layout overlay \
  --fps 15 \
  --codec mp4v \
  --width 640 \
  --height 480
```

## 📚 Weitere Dokumentation

- **RedisTextOverlayManager API:** Siehe Docstrings in `redis_text_overlay_manager.py`
- **Unified MCP Server:** Siehe `server/fastmcp_robot_server_unified.py`
- **Recording Script:** Siehe `scripts/record_camera_with_overlays.py`

## 🤝 Support

Bei Fragen oder Problemen:
- GitHub Issues: [robot_mcp](https://github.com/dgaida/robot_mcp/issues)
- E-Mail: daniel.gaida@th-koeln.de

---

**Made with ❤️ at TH Köln, Campus Gummersbach**
