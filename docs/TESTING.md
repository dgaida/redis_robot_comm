# Testing Guide

Comprehensive testing guidelines for the `redis_robot_comm` package.

---

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)
- [Best Practices](#best-practices)

---

## Overview

The `redis_robot_comm` package includes a comprehensive test suite with >90% code coverage. Tests are written using `pytest` and include unit tests, integration tests, and error handling scenarios.

### Test Philosophy

- **Isolation** - Tests use mocked Redis connections to avoid external dependencies
- **Comprehensive** - Cover normal operations, edge cases, and error conditions
- **Fast** - All tests complete in seconds without requiring actual Redis server
- **Maintainable** - Clear test names and well-organized test files

---

## Test Structure

### Test Files

```
tests/
├── __init__.py
├── test_redis_robot_comm.py           # Core functionality tests
├── test_redis_robot_comm_extended.py  # Extended coverage tests
└── test_redis_label_manager.py        # Label manager tests
```

### Test Organization

Tests are organized by module and functionality:

**test_redis_robot_comm.py** - Basic tests:
- Connection and initialization
- Object publishing and retrieval
- Image streaming basics
- Stream management

**test_redis_robot_comm_extended.py** - Advanced tests:
- Verbose mode and logging
- Error handling
- Edge cases
- Integration scenarios
- Performance validation

**test_redis_label_manager.py** - Label management tests:
- Label publishing and retrieval
- Dynamic label updates
- Subscription functionality
- Error scenarios

---

## Running Tests

### Prerequisites

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Required packages:
- `pytest>=8.0.0` - Test framework
- `pytest-cov` - Coverage reporting
- `black>=24.0.0` - Code formatting
- `ruff>=0.7.0` - Linting
- `mypy>=1.9.0` - Type checking

### Basic Test Execution

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_redis_robot_comm.py -v
```

Run specific test function:
```bash
pytest tests/test_redis_robot_comm.py::test_publish_objects -v
```

Run tests matching pattern:
```bash
pytest tests/ -k "test_publish" -v
```

### Coverage Reports

Run tests with coverage:
```bash
pytest tests/ --cov=redis_robot_comm --cov-report=term
```

Generate HTML coverage report:
```bash
pytest tests/ --cov=redis_robot_comm --cov-report=html
open htmlcov/index.html
```

Generate XML coverage report (for CI):
```bash
pytest tests/ --cov=redis_robot_comm --cov-report=xml
```

### Verbose Output

Enable verbose pytest output:
```bash
pytest tests/ -v -s
```

Show print statements:
```bash
pytest tests/ -s
```

### Filtering Tests

Skip slow tests (if marked):
```bash
pytest tests/ -m "not slow"
```

Run only integration tests:
```bash
pytest tests/ -m integration
```

---

## Writing Tests

### Test Template

```python
import pytest
from unittest.mock import MagicMock, patch
from redis_robot_comm import RedisMessageBroker

def test_feature_name(monkeypatch):
    """Test description explaining what is being tested."""
    # Setup
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Configure mock
    mock_client.xadd.return_value = "1-0"

    # Execute
    result = broker.publish_objects([{"id": "obj_1"}])

    # Assert
    assert result == "1-0"
    mock_client.xadd.assert_called_once()
```

### Mocking Redis Connections

Always mock Redis connections in tests:

```python
def test_with_mocked_redis(monkeypatch):
    """Example of mocking Redis client."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)

    # Now RedisMessageBroker will use the mock
    broker = RedisMessageBroker()

    # Configure mock behavior
    mock_client.xrevrange.return_value = [
        ("1-0", {"timestamp": "1000.0", "objects": "[]"})
    ]

    # Test method that uses Redis
    result = broker.get_latest_objects()
    assert result == []
```

### Testing Error Conditions

Test error handling explicitly:

```python
def test_connection_error(monkeypatch, capsys):
    """Test handling of Redis connection errors."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    # Simulate connection error
    mock_client.xadd.side_effect = Exception("Connection failed")

    # Execute
    result = broker.publish_objects([{"id": "obj_1"}])

    # Assert
    assert result is None
    captured = capsys.readouterr()
    assert "Error publishing objects" in captured.out
```

### Testing Verbose Output

Capture and verify console output:

```python
def test_verbose_logging(monkeypatch, capsys):
    """Test verbose mode produces expected output."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.xadd.return_value = "1-0"

    broker.publish_objects([{"id": "obj_1"}])

    captured = capsys.readouterr()
    assert "Published 1 objects" in captured.out
```

### Testing Time-Dependent Code

Mock time functions for deterministic tests:

```python
from unittest.mock import patch
import time

def test_with_time_control(monkeypatch):
    """Test time-sensitive functionality."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    current_time = 1000.0
    mock_client.xrevrange.return_value = [
        ("1-0", {"timestamp": str(current_time), "objects": "[]"})
    ]

    # Simulate time passing
    with patch("time.time", return_value=current_time + 5.0):
        result = broker.get_latest_objects(max_age_seconds=2.0)

    # Objects should be too old
    assert result == []
```

### Testing Image Processing

Test image encoding/decoding:

```python
import numpy as np
import cv2

def test_image_encoding(monkeypatch):
    """Test JPEG image encoding."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    # Create test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    # Test JPEG compression
    stream_id = streamer.publish_image(
        image,
        compress_jpeg=True,
        quality=85
    )

    assert stream_id == "1-0"
    call_args = mock_client.xadd.call_args[0][1]
    assert call_args["format"] == "jpeg"
    assert int(call_args["width"]) == 100
    assert int(call_args["height"]) == 100
```

### Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("quality,expected_format", [
    (70, "jpeg"),
    (85, "jpeg"),
    (95, "jpeg"),
])
def test_jpeg_quality_levels(monkeypatch, quality, expected_format):
    """Test different JPEG quality settings."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_client.xadd.return_value = "1-0"

    streamer.publish_image(image, compress_jpeg=True, quality=quality)

    call_args = mock_client.xadd.call_args[0][1]
    assert call_args["format"] == expected_format
```

---

## Test Coverage

### Current Coverage

The package maintains >90% test coverage across all modules:

| Module | Coverage | Lines | Missing |
|--------|----------|-------|---------|
| redis_client.py | 95% | 200 | 10 |
| redis_image_streamer.py | 93% | 250 | 18 |
| redis_label_manager.py | 96% | 150 | 6 |
| **Overall** | **94%** | **600** | **34** |

### Coverage Goals

- **Minimum**: 90% overall coverage
- **Target**: 95% overall coverage
- **Critical paths**: 100% coverage for error handling

### Viewing Coverage

Generate coverage report:
```bash
pytest tests/ --cov=redis_robot_comm --cov-report=html
```

View in browser:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Report Interpretation

HTML report shows:
- **Green lines**: Covered by tests
- **Red lines**: Not covered by tests
- **Yellow lines**: Partially covered (branch not taken)

Focus on:
1. Uncovered error handling paths
2. Edge cases not tested
3. Complex logic branches

---

## Continuous Integration

### GitHub Actions Workflows

The project uses GitHub Actions for automated testing:

**tests.yml** - Main test suite:
- Runs on: push to main, pull requests, manual dispatch
- Tests on: Ubuntu, Windows, macOS
- Python versions: 3.8, 3.9, 3.10, 3.11
- Includes coverage reporting to Codecov

**Quick tests** (default):
- Ubuntu + Python 3.11 only
- Fast feedback for routine changes

**Full tests** (triggered by):
- `[full-test]` in commit message
- `[full ci]` in commit message
- Manual workflow dispatch

### Running CI Tests Locally

Simulate CI environment:
```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run linting
ruff check .
black --check .
mypy redis_robot_comm --ignore-missing-imports

# Run security checks
bandit -r redis_robot_comm/ -ll

# Run tests with coverage
pytest tests/ --cov=redis_robot_comm --cov-report=xml
```

### CI Configuration

**.github/workflows/tests.yml**:
```yaml
- name: Run tests with coverage
  run: |
    pytest -v --cov=redis_robot_comm --cov-report=xml --cov-report=term
  continue-on-error: false

- name: Upload coverage to Codecov
  if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
  uses: codecov/codecov-action@v5
  with:
    files: ./coverage.xml
    flags: unittests
```

---

## Best Practices

### 1. Test Isolation

Each test should be independent:
```python
# Good - isolated test
def test_feature_a(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    # Test feature A

# Bad - depends on previous test state
def test_feature_b():
    # Uses broker from test_feature_a - DON'T DO THIS
    pass
```

### 2. Descriptive Test Names

Use clear, descriptive names:
```python
# Good
def test_publish_objects_with_camera_pose():
    """Test publishing objects with camera pose metadata."""
    pass

# Bad
def test_1():
    """Test something."""
    pass
```

### 3. Arrange-Act-Assert Pattern

Structure tests clearly:
```python
def test_publish_and_retrieve():
    # Arrange - Setup
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    mock_client.xadd.return_value = "1-0"

    # Act - Execute
    result = broker.publish_objects([{"id": "obj_1"}])

    # Assert - Verify
    assert result == "1-0"
    mock_client.xadd.assert_called_once()
```

### 4. Test Edge Cases

Always test boundary conditions:
```python
def test_empty_object_list():
    """Test handling of empty object list."""
    # Test with empty list
    pass

def test_very_large_image():
    """Test handling of large images (4K+)."""
    # Test with 3840x2160 image
    pass

def test_maximum_stream_length():
    """Test behavior when stream reaches maxlen."""
    # Test maxlen parameter
    pass
```

### 5. Mock External Dependencies

Never rely on external services:
```python
# Good - mocked Redis
def test_with_mock(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    # Test code

# Bad - requires actual Redis server
def test_without_mock():
    broker = RedisMessageBroker()  # Connects to real Redis
    # Test code - will fail in CI if Redis not available
```

### 6. Use Fixtures for Common Setup

Create reusable fixtures:
```python
@pytest.fixture
def mock_broker(monkeypatch):
    """Fixture providing mocked RedisMessageBroker."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    return RedisMessageBroker()

def test_with_fixture(mock_broker):
    """Test using fixture."""
    result = mock_broker.test_connection()
    assert result is True
```

### 7. Test Error Messages

Verify error handling provides useful feedback:
```python
def test_error_message(monkeypatch, capsys):
    """Verify error messages are helpful."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()
    broker.verbose = True

    mock_client.xadd.side_effect = Exception("Connection timeout")
    broker.publish_objects([{"id": "obj_1"}])

    captured = capsys.readouterr()
    assert "Connection timeout" in captured.out
```

### 8. Document Test Purpose

Add docstrings explaining what is tested:
```python
def test_subscribe_with_callback():
    """
    Test that subscribe_objects correctly invokes callback function
    with parsed detection data when new messages arrive.

    Verifies:
    - Callback is called with correct data structure
    - Objects are properly deserialized
    - Camera pose is correctly parsed
    - Timestamp is converted to float
    """
    pass
```

---

## Testing Checklist

Before submitting a pull request, ensure:

- [ ] All tests pass locally
- [ ] New features have corresponding tests
- [ ] Test coverage remains >90%
- [ ] Error cases are tested
- [ ] Edge cases are covered
- [ ] Tests are well-documented
- [ ] No tests depend on external services
- [ ] Tests run quickly (<1 minute total)
- [ ] Verbose output is tested where applicable
- [ ] Integration scenarios are covered

---

## Common Testing Patterns

### Pattern 1: Testing Stream Operations

```python
def test_stream_operation(monkeypatch):
    """Test Redis stream read/write operations."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Mock stream write
    mock_client.xadd.return_value = "1-0"

    # Mock stream read
    mock_client.xrevrange.return_value = [
        ("1-0", {"timestamp": "1000.0", "objects": "[]"})
    ]

    # Test write
    write_result = broker.publish_objects([])
    assert write_result == "1-0"

    # Test read
    with patch("time.time", return_value=1000.5):
        read_result = broker.get_latest_objects()
        assert read_result == []
```

### Pattern 2: Testing Image Encoding/Decoding

```python
def test_image_roundtrip(monkeypatch):
    """Test image encoding and decoding roundtrip."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    streamer = RedisImageStreamer()

    # Create test image
    original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Encode
    _, buffer = cv2.imencode(".jpg", original)
    encoded = base64.b64encode(buffer).decode("utf-8")

    # Mock retrieval
    mock_client.xrevrange.return_value = [
        ("1-0", {
            "width": "100",
            "height": "100",
            "channels": "3",
            "format": "jpeg",
            "dtype": "uint8",
            "image_data": encoded
        })
    ]

    # Decode
    result = streamer.get_latest_image()
    assert result is not None
    decoded, _ = result
    assert decoded.shape == (100, 100, 3)
```

### Pattern 3: Testing Callbacks

```python
def test_callback_execution(monkeypatch):
    """Test that callbacks are correctly executed."""
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Mock stream response
    mock_client.xread.side_effect = [
        [("detected_objects", [
            ("1-0", {
                "objects": '[{"id": "obj_1"}]',
                "camera_pose": "{}",
                "timestamp": "1000.0"
            })
        ])],
        KeyboardInterrupt()  # End loop
    ]

    # Capture callback data
    callback_data = []
    def callback(data):
        callback_data.append(data)

    # Execute
    broker.subscribe_objects(callback)

    # Verify
    assert len(callback_data) == 1
    assert callback_data[0]["objects"][0]["id"] == "obj_1"
```

---

## Troubleshooting Tests

### Tests Fail Locally But Pass in CI

Possible causes:
- Different Python versions
- Missing dependencies
- Environment-specific issues

Solution:
```bash
# Match CI environment
python --version  # Check version
pip install -r requirements-dev.txt --force-reinstall
pytest tests/ -v
```

### Flaky Tests

Tests that pass/fail intermittently:

Common causes:
- Time-dependent logic without mocking
- Race conditions in threading
- Uninitialized state

Solution:
- Always mock `time.time()`
- Use `threading.Event()` for synchronization
- Reset state in fixtures

### Slow Tests

If tests take >10 seconds:

Solutions:
- Ensure Redis is mocked
- Reduce test data size
- Use `pytest -x` to stop on first failure
- Profile tests: `pytest --durations=10`

---

## Additional Resources

- **[pytest Documentation](https://docs.pytest.org/)** - Official pytest guide
- **[unittest.mock](https://docs.python.org/3/library/unittest.mock.html)** - Python mocking library
- **[Coverage.py](https://coverage.readthedocs.io/)** - Code coverage tool
- **[GitHub Actions](https://docs.github.com/en/actions)** - CI/CD documentation

---

## Contributing Test Improvements

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Adding new tests
- Improving coverage
- Reporting test issues
- Suggesting test improvements

---

*Last updated: 2025-12-03*
