# Testing

The project has an extensive test suite.

## Running Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests with pytest:

```bash
pytest tests/ -v
```

## Test Coverage

To generate a coverage report:

```bash
pytest tests/ --cov=redis_robot_comm --cov-report=term
```
