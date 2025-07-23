#!/bin/bash
# Simple wrapper script for development environment setup

echo "Setting up Docker Optimizer Agent development environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Run the Python setup script
python3 setup_dev.py "$@"