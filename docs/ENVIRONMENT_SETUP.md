# Development Environment Setup

## Prerequisites
- Python 3.9+ installed
- Git installed

## Setup Steps

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e .[dev,security]
   ```

3. **Install Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify Installation**:
   ```bash
   make test
   make lint
   make typecheck
   ```

## Troubleshooting

- If you see "externally-managed-environment" error, use a virtual environment
- On Ubuntu/Debian, you may need: `sudo apt install python3-venv python3-dev`
- On macOS with Homebrew: `brew install python@3.11`

## Tool Verification

Run these commands to verify tools are installed:
- `pytest --version`
- `ruff --version` 
- `mypy --version`
- `black --version`
