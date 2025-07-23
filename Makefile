.PHONY: help setup install test test-unit test-integration lint typecheck security clean dev

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
	@echo "  typecheck      - Run mypy type checker"
	@echo "  security       - Run bandit security scanner"
	@echo ""
	@echo "Development Commands:"
	@echo "  dev            - Install in development mode and run initial checks"
	@echo "  clean          - Clean up build artifacts and cache"
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

# Clean up
clean:
	@echo "Cleaning up build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf htmlcov/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true