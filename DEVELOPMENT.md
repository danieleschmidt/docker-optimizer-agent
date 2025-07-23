# Development Guide

This document provides comprehensive guidance for setting up and working with the Docker Optimizer Agent development environment.

## Quick Start

For a fully automated setup, run:

```bash
# Using Python script (recommended)
python3 setup_dev.py

# Using Makefile
make setup

# Using shell wrapper
./setup_dev.sh
```

## Manual Setup

If you prefer to set up the development environment manually:

### 1. Prerequisites

- **Python 3.9+** (tested with Python 3.9-3.12)
- **Docker** (for integration tests)
- **Git** (for version control)

### 2. Install Dependencies

```bash
# Install main dependencies
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Install security tools (optional)
pip install -e .[security]
```

### 3. Set Up Pre-commit Hooks

```bash
pre-commit install
```

## Development Workflow

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only (no Docker required)
pytest tests/ -m "not integration"

# Integration tests only (requires Docker)
pytest tests/ -m "integration"

# With coverage
pytest tests/ --cov=docker_optimizer --cov-report=html
```

### Code Quality Checks

```bash
# Linting
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Type checking
mypy --ignore-missing-imports src/docker_optimizer/

# Security scanning
bandit -r src/ -f json
```

### Using Makefile Commands

The project includes a Makefile with convenient commands:

```bash
make help          # Show all available commands
make setup         # Set up development environment
make test          # Run all tests
make test-unit     # Run unit tests only
make lint          # Run linter
make lint-fix      # Run linter with auto-fix
make typecheck     # Run type checker
make security      # Run security scanner
make clean         # Clean up build artifacts
```

## Project Structure

```
src/docker_optimizer/
├── __init__.py
├── cli.py                 # Command-line interface
├── config.py             # Configuration management
├── models.py             # Pydantic models
├── analyzer.py           # Dockerfile analysis
├── optimizer.py          # Optimization suggestions
├── size_estimator.py     # Size estimation
├── layer_analyzer.py     # Layer analysis
├── performance.py        # Performance optimizations
├── advanced_security.py  # Security rule engine
└── suggestions.py        # Real-time suggestions

tests/
├── test_*.py             # Unit tests
├── test_integration.py   # Integration tests
└── conftest.py           # Pytest configuration
```

## Configuration

The project uses several configuration files:

- **`pyproject.toml`** - Project metadata, dependencies, and tool configuration
- **`.pre-commit-config.yaml`** - Pre-commit hooks (if present)
- **`pytest.ini`** or `pyproject.toml` - Test configuration
- **`BACKLOG.md`** - Development backlog and priorities

## Testing Guidelines

### Unit Tests

- Located in `tests/test_*.py`
- Should not require external dependencies (Docker, network)
- Use mocking for external dependencies
- Aim for >85% code coverage

### Integration Tests

- Marked with `@pytest.mark.integration`
- May require Docker to be running
- Test real interactions with Docker daemon
- Should clean up after themselves

### Test Configuration

Tests are configured to:
- Require 85% code coverage (`--cov-fail-under=85`)
- Generate HTML coverage reports in `htmlcov/`
- Use strict marker checking
- Disable warnings in output

## Code Quality Standards

### Linting (Ruff)

The project uses Ruff for fast Python linting with these rules:
- `E`, `W` - pycodestyle errors and warnings
- `F` - pyflakes
- `I` - isort (import sorting)
- `B` - flake8-bugbear
- `C4` - flake8-comprehensions
- `S` - bandit (security)

### Type Checking (MyPy)

MyPy is configured with strict settings:
- `check_untyped_defs = true`
- `disallow_any_generics = true`
- `disallow_incomplete_defs = true`
- `disallow_untyped_defs = true`

### Security Scanning (Bandit)

Bandit scans for common security issues:
- Excludes test directories
- Skips certain checks for development code
- Reports in JSON format for CI integration

## Contributing Workflow

1. **Set up development environment**: `make setup`
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** following code quality standards
4. **Run tests**: `make test`
5. **Run quality checks**: `make lint typecheck security`
6. **Commit changes** (pre-commit hooks will run automatically)
7. **Push and create pull request**

## Troubleshooting

### Common Issues

**"Docker not found" errors:**
- Install Docker Desktop or Docker Engine
- Ensure Docker daemon is running
- Add your user to the docker group (Linux)

**"Permission denied" errors:**
- Make scripts executable: `chmod +x setup_dev.py setup_dev.sh`
- On Windows, use `python setup_dev.py` instead of `./setup_dev.py`

**Import errors:**
- Ensure you installed with `-e .` for editable install
- Check that you're in the project root directory
- Activate your virtual environment if using one

**Test failures:**
- Integration tests require Docker to be running
- Some tests may fail in restricted environments (CI)
- Check that all dependencies are installed

### Virtual Environments

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install project
pip install -e .[dev]
```

### Docker Issues

If integration tests fail:

1. **Check Docker installation**: `docker --version`
2. **Check Docker daemon**: `docker info`
3. **Test Docker access**: `docker run hello-world`
4. **Check permissions**: Add user to docker group (Linux)

## Additional Resources

- **Project README**: `README.md` - General project information
- **Development Backlog**: `BACKLOG.md` - Current priorities and tasks
- **Configuration**: `pyproject.toml` - Project and tool configuration
- **License**: `LICENSE` - Project license information