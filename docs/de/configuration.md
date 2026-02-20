# Konfiguration

`redis_robot_comm` kann über Umgebungsvariablen oder direkt im Code konfiguriert werden.

## Umgebungsvariablen

Die folgenden Umgebungsvariablen werden unterstützt:

| Variable | Beschreibung | Standardwert |
|----------|--------------|--------------|
| `REDIS_HOST` | Hostname des Redis-Servers | `localhost` |
| `REDIS_PORT` | Port des Redis-Servers | `6379` |
| `REDIS_DB` | Index der Redis-Datenbank | `0` |
| `REDIS_PASSWORD` | Passwort für Redis (optional) | `None` |

## Konfiguration im Code

Sie können die Konfiguration beim Initialisieren der Manager-Klassen übergeben:

```python
from redis_robot_comm import RedisMessageBroker

# Explizite Parameter
broker = RedisMessageBroker(
    host="192.168.1.100",
    port=6380,
    db=1,
    stream_name="my_objects"
)
```

Alternativ können Sie ein `RedisConfig`-Objekt verwenden:

```python
from redis_robot_comm.config import RedisConfig
from redis_robot_comm import RedisImageStreamer

config = RedisConfig(
    host="localhost",
    port=6379,
    password="secret_password"
)

streamer = RedisImageStreamer(config=config)
```

## Logging-Konfiguration

Alle Klassen verfügen über ein `verbose`-Attribut, um detaillierte Protokollausgaben zu aktivieren:

```python
broker = RedisMessageBroker()
broker.verbose = True
```
