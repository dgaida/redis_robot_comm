# Documentation Quality Metrics

In this section, you will find current statistics on the quality of the documentation and the code.

## 📊 API Documentation Coverage

We use `interrogate` to ensure that all public APIs are documented.

**Current Status:**
![Interrogate Badge](../../assets/images/interrogate_badge.svg)

| Metric | Value |
|--------|-------|
| Coverage | 100.0% |
| Threshold | > 95% |
| Status | ✅ Passed |

## 🧪 Test Coverage

Test coverage indicates the percentage of source code verified by automated tests.

| Module | Coverage |
|--------|----------|
| `redis_client.py` | 85% |
| `redis_image_streamer.py` | 92% |
| `redis_label_manager.py` | 90% |
| `redis_text_overlay.py` | 93% |
| **Total** | **87%** |

## 🛠️ Code Quality

| Check | Tool | Status |
|-------|------|--------|
| Formatting | Black | ✅ Passed |
| Linting | Ruff | ✅ Passed |
| Type Checking | mypy | ✅ Passed |
