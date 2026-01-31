# Makefile
.PHONY: help install dev-install test test-cov test-integration lint format type-check security clean docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev-install:  ## Install package with development dependencies
	pip install -e .
	pip install -r requirements-dev.txt
	pre-commit install

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=redis_robot_comm --cov-report=html --cov-report=term

test-integration:  ## Run integration tests (requires Redis)
	pytest tests/integration/ -v --redis-url redis://localhost:6379

lint:  ## Run linting
	ruff check .
	black --check .

format:  ## Format code
	ruff check . --fix
	black .
	isort .

type-check:  ## Run type checking
	mypy redis_robot_comm --ignore-missing-imports

security:  ## Run security checks
	bandit -r redis_robot_comm/ -ll
	safety check

clean:  ## Clean up generated files
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && make html

all: format lint type-check test  ## Run all checks
