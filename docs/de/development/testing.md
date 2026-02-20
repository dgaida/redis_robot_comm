# Tests

Das Projekt verfügt über eine umfangreiche Testsuite.

## Tests ausführen

Installieren Sie die Entwicklungsabhängigkeiten:

```bash
pip install -r requirements-dev.txt
```

Führen Sie die Tests mit pytest aus:

```bash
pytest tests/ -v
```

## Testabdeckung

So generieren Sie einen Coverage-Bericht:

```bash
pytest tests/ --cov=redis_robot_comm --cov-report=term
```
