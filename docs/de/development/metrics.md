# Metriken zur Dokumentationsqualität

In diesem Abschnitt finden Sie aktuelle Statistiken zur Qualität der Dokumentation und des Codes.

## 📊 API-Dokumentationsabdeckung

Wir verwenden `interrogate`, um sicherzustellen, dass alle öffentlichen APIs dokumentiert sind.

**Aktueller Status:**
![Interrogate Badge](../../assets/images/interrogate_badge.svg)

| Metrik | Wert |
|--------|-------|
| Abdeckung | 100.0% |
| Zielwert | > 95% |
| Status | ✅ Bestanden |

## 🧪 Testabdeckung

Die Testabdeckung gibt an, wie viel Prozent des Quellcodes durch automatisierte Tests geprüft werden.

| Modul | Abdeckung |
|-------|-----------|
| `redis_client.py` | 85% |
| `redis_image_streamer.py` | 92% |
| `redis_label_manager.py` | 90% |
| `redis_text_overlay.py` | 93% |
| **Gesamt** | **87%** |

## 🛠️ Code-Qualität

| Prüfung | Tool | Status |
|---------|------|--------|
| Formatierung | Black | ✅ Bestanden |
| Linting | Ruff | ✅ Bestanden |
| Typprüfung | mypy | ✅ Bestanden |
