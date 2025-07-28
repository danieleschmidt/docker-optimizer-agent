.PHONY: help setup install test test-unit test-integration lint typecheck security clean dev build docker-build docker-test release format coverage docs monitor

# Default target
help:
	@echo "Docker Optimizer Agent - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Set up development environment (runs setup_dev.py)"
	@echo "  install        - Install dependencies only"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint           - Run ruff linter"
	@echo "  lint-fix       - Run ruff linter with auto-fix"
	@echo "  format         - Format code with black and isort"
	@echo "  typecheck      - Run mypy type checker"
	@echo "  security       - Run bandit security scanner"
	@echo "  coverage       - Run tests with coverage report"
	@echo ""
	@echo "Build Commands:"
	@echo "  build          - Build Python package"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-test    - Run tests in Docker container"
	@echo ""
	@echo "Development Commands:"
	@echo "  dev            - Install in development mode and run initial checks"
	@echo "  clean          - Clean up build artifacts and cache"
	@echo "  docs           - Generate documentation"
	@echo "  monitor        - Start monitoring stack"
	@echo ""
	@echo "Release Commands:"
	@echo "  release        - Build and package for release"
	@echo ""

# Setup development environment
setup:
	@python3 setup_dev.py

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install -e .
	@pip install -e .[dev]
	@pip install -e .[security] || echo "Security dependencies are optional"

# Testing
test:
	@echo "Running all tests..."
	@pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	@pytest tests/ -v -m "not integration"

test-integration:
	@echo "Running integration tests..."
	@pytest tests/ -v -m "integration"

# Code quality
lint:
	@echo "Running ruff linter..."
	@ruff check src/ tests/

lint-fix:
	@echo "Running ruff linter with auto-fix..."
	@ruff check --fix src/ tests/

typecheck:
	@echo "Running mypy type checker..."
	@mypy --ignore-missing-imports src/docker_optimizer/

security:
	@echo "Running bandit security scanner..."
	@bandit -r src/ -f json || true

# Development
dev: install lint typecheck test-unit
	@echo "Development environment ready!"

# Code formatting
format:
	@echo "Formatting code with black and isort..."
	@black src/ tests/
	@isort src/ tests/

# Coverage testing
coverage:
	@echo "Running tests with coverage..."
	@pytest tests/ --cov=docker_optimizer --cov-report=html --cov-report=term-missing --cov-fail-under=85

# Build Python package
build:
	@echo "Building Python package..."
	@python -m build

# Docker build
docker-build:
	@echo "Building Docker images..."
	@docker build --target development -t docker-optimizer:dev .
	@docker build --target production -t docker-optimizer:latest .
	@docker build --target cli -t docker-optimizer:cli .
	@docker build --target testing -t docker-optimizer:test .

# Docker test
docker-test:
	@echo "Running tests in Docker..."
	@docker run --rm docker-optimizer:test

# Generate documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs/generated
	@python -m pydoc -w docker_optimizer
	@mv *.html docs/generated/ 2>/dev/null || true

# Start monitoring stack
monitor:
	@echo "Starting monitoring stack..."
	@docker-compose -f monitoring/docker-compose.yml up -d
	@echo "Monitoring available at:"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

# Release build
release: clean lint typecheck security test build
	@echo "Release package ready in dist/"

# Development
dev: install lint typecheck test-unit
	@echo "Development environment ready!"

# Clean up
clean:
	@echo "Cleaning up build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf htmlcov/
	@rm -rf docs/generated/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true