# Contributing to redis_robot_comm

Thank you for your interest in contributing to `redis_robot_comm`! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect everyone to:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling, insulting/derogatory comments, and personal attacks
- Publishing others' private information without permission
- Other conduct that could reasonably be considered inappropriate

### Enforcement

Instances of unacceptable behavior may be reported to daniel.gaida@th-koeln.de. All complaints will be reviewed and investigated promptly and fairly.

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python** 3.8 or higher
- **Git** for version control
- **Redis** server (optional for manual testing)
- **GitHub account** for pull requests

### Ways to Contribute

We welcome contributions in many forms:

1. **Bug Reports** - Report issues you encounter
2. **Feature Requests** - Suggest new features or improvements
3. **Code Contributions** - Submit bug fixes or new features
4. **Documentation** - Improve or extend documentation
5. **Testing** - Add test cases or improve coverage
6. **Examples** - Provide usage examples or tutorials
7. **Performance** - Optimize existing code

---

## Development Setup

### 1. Fork and Clone

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/redis_robot_comm.git
cd redis_robot_comm
```

Add the upstream repository:

```bash
git remote add upstream https://github.com/dgaida/redis_robot_comm.git
```

### 2. Create Virtual Environment

Create and activate a virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Using conda
conda create -n redis_robot_comm python=3.11
conda activate redis_robot_comm
```

### 3. Install in Development Mode

Install the package with development dependencies:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Development dependencies include:
- `pytest>=8.0.0` - Testing framework
- `pytest-cov` - Coverage reporting
- `black>=24.0.0` - Code formatting
- `ruff>=0.7.0` - Fast Python linter
- `mypy>=1.9.0` - Type checking
- `pre-commit>=3.6.0` - Git hooks

### 4. Install Pre-commit Hooks

Set up pre-commit hooks for automatic code quality checks:

```bash
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

### 5. Verify Setup

Test that everything is working:

```bash
# Run tests
pytest tests/ -v

# Check formatting
black --check .

# Run linter
ruff check .

# Type checking
mypy redis_robot_comm --ignore-missing-imports
```

### 6. Optional: Start Redis

For manual testing with actual Redis:

```bash
# Using Docker (recommended)
docker run -p 6379:6379 redis:alpine

# Or install locally
# Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis

# macOS:
brew install redis
brew services start redis
```

---

## Development Workflow

### 1. Create a Branch

Create a descriptive branch for your changes:

```bash
# Feature branch
git checkout -b feature/add-new-feature

# Bug fix branch
git checkout -b fix/fix-connection-issue

# Documentation branch
git checkout -b docs/update-readme
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test improvements
- `refactor/` - Code refactoring
- `perf/` - Performance improvements

### 2. Make Changes

Make your changes following the [Coding Standards](#coding-standards) below.

**Key principles:**
- Keep changes focused and atomic
- Write clear, self-documenting code
- Add tests for new functionality
- Update documentation as needed
- Follow existing code style

### 3. Test Your Changes

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=redis_robot_comm --cov-report=html

# Specific test file
pytest tests/test_redis_robot_comm.py -v

# Fast feedback (stop on first failure)
pytest tests/ -x
```

Ensure coverage remains >90%:
```bash
pytest tests/ --cov=redis_robot_comm --cov-report=term
```

### 4. Lint and Format

Run code quality tools:

```bash
# Auto-format code
black .

# Check formatting
black --check .

# Lint with Ruff
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking
mypy redis_robot_comm --ignore-missing-imports

# Security check
bandit -r redis_robot_comm/ -ll
```

Pre-commit hooks will run these automatically on commit.

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add support for dynamic stream names

- Allow users to specify custom stream names
- Add validation for stream name format
- Update tests and documentation
- Closes #42"
```

**Commit message format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes (formatting, etc.)
- `chore:` - Maintenance tasks

### 6. Push and Create Pull Request

Push your branch and create a pull request:

```bash
git push origin feature/add-new-feature
```

Go to GitHub and create a pull request from your branch to `main`.

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications enforced by Black and Ruff.

**Key guidelines:**
- Line length: 127 characters (Black configuration)
- Use type hints where possible
- Write docstrings for all public methods
- Use descriptive variable names
- Prefer explicit over implicit

### Code Formatting

We use **Black** for consistent code formatting:

```bash
# Format all files
black .

# Check formatting
black --check .

# Format specific file
black redis_robot_comm/redis_client.py
```

Black configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 127
target-version = ['py38', 'py39', 'py310', 'py311']
```

### Linting

We use **Ruff** for fast linting:

```bash
# Check all files
ruff check .

# Auto-fix issues
ruff check . --fix

# Check specific file
ruff check redis_robot_comm/redis_client.py
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def publish_objects(
    self,
    objects: List[Dict],
    camera_pose: Optional[Dict] = None
) -> Optional[str]:
    """Publish detected objects to Redis."""
    pass
```

Run type checking:
```bash
mypy redis_robot_comm --ignore-missing-imports
```

### Docstrings

Follow Google-style docstrings:

```python
def publish_image(
    self,
    image: np.ndarray,
    metadata: Dict[str, Any] = None,
    compress_jpeg: bool = True,
    quality: int = 80
) -> str:
    """
    Publish an image frame to the Redis stream.

    Args:
        image: OpenCV image array (H×W×C or H×W for grayscale)
        metadata: Custom metadata to attach
        compress_jpeg: Enable JPEG compression
        quality: JPEG quality (1-100)

    Returns:
        Redis stream entry ID

    Raises:
        ValueError: If image is not a valid NumPy array

    Example:
        >>> streamer = RedisImageStreamer()
        >>> image = cv2.imread("frame.jpg")
        >>> stream_id = streamer.publish_image(image, quality=85)
    """
    pass
```

### Error Handling

Handle errors gracefully with informative messages:

```python
def get_latest_objects(self, max_age_seconds: float = 2.0) -> List[Dict]:
    """Retrieve latest objects with error handling."""
    try:
        messages = self.client.xrevrange("detected_objects", count=1)
        if not messages:
            if self.verbose:
                print("No messages found in stream")
            return []
        # Process messages...
    except Exception as e:
        print(f"Error getting latest objects: {e}")
        return []
```

### Performance Considerations

- Minimize Redis round-trips
- Use batch operations where possible
- Implement proper stream cleanup with `maxlen`
- Cache frequently accessed data
- Profile performance-critical code

---

## Testing Guidelines

### Test Requirements

All contributions must include appropriate tests:

- **New features** - Add tests covering functionality
- **Bug fixes** - Add regression tests
- **Refactoring** - Ensure existing tests pass
- **Coverage** - Maintain >90% coverage

### Writing Tests

Follow the patterns in [docs/TESTING.md](docs/TESTING.md):

```python
def test_new_feature(monkeypatch):
    """Test description explaining what is tested."""
    # Arrange - Setup
    mock_client = MagicMock()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_client)
    broker = RedisMessageBroker()

    # Act - Execute
    result = broker.new_feature()

    # Assert - Verify
    assert result is not None
    mock_client.some_method.assert_called_once()
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_redis_robot_comm.py::test_publish_objects -v

# With coverage
pytest tests/ --cov=redis_robot_comm --cov-report=html

# Fast feedback
pytest tests/ -x  # Stop on first failure
```

### Test Coverage

Ensure coverage remains high:

```bash
pytest tests/ --cov=redis_robot_comm --cov-report=term

# View detailed report
pytest tests/ --cov=redis_robot_comm --cov-report=html
open htmlcov/index.html
```

Coverage goals:
- Overall: >90%
- New code: 100%
- Critical paths: 100%

---

## Documentation

### Types of Documentation

1. **Code Comments** - Explain complex logic
2. **Docstrings** - Document all public APIs
3. **README** - Overview and quick start
4. **API Docs** - Detailed method documentation
5. **Examples** - Usage examples and tutorials
6. **Guides** - Testing, contributing, etc.

### Updating Documentation

When changing functionality:

1. Update relevant docstrings
2. Update API documentation in `docs/api.md`
3. Update README if needed
4. Add examples for new features
5. Update workflow diagrams if applicable

### Writing Examples

Provide clear, runnable examples:

```python
# Good example
"""
Example: Publishing detected objects

This example shows how to publish object detection results
with camera pose information.
"""
from redis_robot_comm import RedisMessageBroker

broker = RedisMessageBroker()

# Detect objects (pseudo-code)
objects = detect_objects(image)

# Publish with camera pose
camera_pose = {"x": 0.0, "y": 0.0, "z": 0.5}
broker.publish_objects(objects, camera_pose)
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Explain the "why" not just "what"
- Link to related documentation
- Keep formatting consistent

---

## Pull Request Process

### Before Submitting

Ensure your PR meets these criteria:

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Coverage remains >90%
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Title Format

Use descriptive titles:
```
feat: Add support for custom stream names
fix: Resolve connection timeout issue
docs: Update API documentation for RedisImageStreamer
test: Add tests for label manager error handling
```

### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Changes Made
- Detailed list of changes
- Explain key decisions
- Note any breaking changes

## Testing
- Describe testing performed
- List any manual testing steps

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #42
Related to #38
```

### Review Process

1. **Automated Checks** - CI must pass (tests, linting, coverage)
2. **Code Review** - At least one maintainer review required
3. **Feedback** - Address review comments
4. **Approval** - Maintainer approval required
5. **Merge** - Squash and merge into main

### After PR Merge

- Delete your branch
- Update your fork:
  ```bash
  git checkout main
  git pull upstream main
  git push origin main
  ```

---

## Release Process

Releases are managed by maintainers. The process:

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0) - Breaking changes
- **MINOR** (0.1.0) - New features, backwards compatible
- **PATCH** (0.0.1) - Bug fixes, backwards compatible

### Release Steps

1. **Update Version**
   - Update version in `pyproject.toml`
   - Update version in `__init__.py`

2. **Update CHANGELOG**
   - Document all changes since last release
   - Group by: Added, Changed, Fixed, Removed

3. **Create Release**
   - Tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
   - Push: `git push origin v0.2.0`

4. **GitHub Release**
   - Automated via `.github/workflows/release.yml`
   - Creates GitHub release with notes
   - Attaches built packages

5. **Announce**
   - Update README badges
   - Announce in related projects

---

## Getting Help

### Questions?

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and discussions
- **Email** - daniel.gaida@th-koeln.de for private inquiries

### Useful Resources

- [Main README](README.md) - Package overview
- [API Documentation](docs/api.md) - Complete API reference
- [Testing Guide](docs/TESTING.md) - Testing guidelines
- [Workflow Docs](docs/README.md) - Integration examples

### Related Projects

- [vision_detect_segment](https://github.com/dgaida/vision_detect_segment) - Object detection
- [robot_environment](https://github.com/dgaida/robot_environment) - Robot control
- [robot_mcp](https://github.com/dgaida/robot_mcp) - MCP server

---

## Recognition

Contributors are recognized in several ways:

- Listed in GitHub contributors
- Mentioned in release notes
- Added to AUTHORS file (for significant contributions)

Thank you for contributing to `redis_robot_comm`!

---

## License

By contributing to redis_robot_comm, you agree that your contributions will be licensed under the MIT License.

---

## Contact

**Project Maintainer:** Daniel Gaida  
**Email:** daniel.gaida@th-koeln.de  
**GitHub:** [@dgaida](https://github.com/dgaida)  
**Project:** https://github.com/dgaida/redis_robot_comm

---

*Last updated: 2025-12-03*
