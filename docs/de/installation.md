# Installation

In diesem Abschnitt wird beschrieben, wie Sie `redis_robot_comm` installieren und einrichten.

## Paket installieren

Sie können das Paket direkt aus dem Quellcode installieren:

```bash
git clone https://github.com/dgaida/redis_robot_comm.git
cd redis_robot_comm
pip install -e .
```

## Redis-Server einrichten

`redis_robot_comm` benötigt einen laufenden Redis-Server (Version ≥ 5.0).

### Mit Docker (empfohlen)

Dies ist die einfachste Methode, um Redis schnell zu starten:

```bash
docker run -d -p 6379:6379 --name redis-robot redis:alpine
```

### Lokale Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

## Installation verifizieren

Nach der Installation können Sie die Verbindung mit einem einfachen Skript testen:

```python
from redis_robot_comm import RedisMessageBroker

try:
    broker = RedisMessageBroker()
    if broker.test_connection():
        print("✓ Installation erfolgreich: Verbindung zu Redis hergestellt.")
    else:
        print("✗ Verbindung zu Redis fehlgeschlagen.")
except Exception as e:
    print(f"✗ Fehler bei der Verifizierung: {e}")
```
