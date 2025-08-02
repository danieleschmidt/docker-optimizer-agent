"""pytest configuration and shared fixtures for Docker Optimizer Agent tests.

This module provides common test fixtures, configurations, and utilities
used across all test modules.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Any
import pytest
from unittest.mock import Mock, patch

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dockerfile() -> str:
    """Return a sample Dockerfile content for testing."""
    return """FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
"""


@pytest.fixture
def complex_dockerfile() -> str:
    """Return a complex Dockerfile for advanced testing."""
    return """FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:16-alpine AS development
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]

FROM node:16-alpine AS production
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
USER node
CMD ["npm", "start"]
"""


@pytest.fixture
def dockerfile_with_security_issues() -> str:
    """Return a Dockerfile with known security issues for testing."""
    return """FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl wget
COPY --chown=root:root . /app
WORKDIR /app
USER root
EXPOSE 22
CMD ["sh", "-c", "while true; do sleep 1; done"]
"""


@pytest.fixture
def python_dockerfile() -> str:
    """Return a Python-specific Dockerfile."""
    return """FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
"""


@pytest.fixture
def mock_trivy_output() -> dict[str, Any]:
    """Return mock Trivy security scan output."""
    return {
        "Results": [
            {
                "Target": "ubuntu:20.04",
                "Class": "os-pkgs",
                "Type": "ubuntu",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2021-3711",
                        "PkgName": "openssl",
                        "InstalledVersion": "1.1.1f-1ubuntu2.8",
                        "FixedVersion": "1.1.1f-1ubuntu2.16",
                        "Severity": "HIGH",
                        "Title": "OpenSSL vulnerability",
                        "Description": "Example vulnerability description"
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_optimization_result() -> dict[str, Any]:
    """Return mock optimization result."""
    return {
        "original_dockerfile": "FROM ubuntu:20.04\nRUN apt-get update",
        "optimized_dockerfile": "FROM ubuntu:20.04-slim\nRUN apt-get update && apt-get clean",
        "size_reduction": 45.2,
        "security_score": {"grade": "B", "score": 75},
        "explanation": "Applied size optimization and security hardening",
        "recommendations": [
            "Use slim base image",
            "Clean package cache",
            "Add non-root user"
        ]
    }


@pytest.fixture
def docker_available() -> bool:
    """Check if Docker is available for integration tests."""
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@pytest.fixture
def mock_external_security_scanner():
    """Mock external security scanner for testing."""
    with patch('docker_optimizer.external_security.ExternalSecurityScanner') as mock:
        scanner_instance = Mock()
        scanner_instance.scan_dockerfile_for_vulnerabilities.return_value = {
            "vulnerabilities": [],
            "security_grade": "A",
            "recommendations": []
        }
        mock.return_value = scanner_instance
        yield scanner_instance


@pytest.fixture
def mock_performance_analyzer():
    """Mock performance analyzer for testing."""
    with patch('docker_optimizer.performance.PerformanceAnalyzer') as mock:
        analyzer_instance = Mock()
        analyzer_instance.estimate_size.return_value = 150.5
        analyzer_instance.analyze_performance.return_value = {
            "build_time_estimate": 120,
            "cache_efficiency": 85.2,
            "layer_count": 8
        }
        mock.return_value = analyzer_instance
        yield analyzer_instance


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        "TEST_MODE": "true",
        "MOCK_SECURITY_SCAN": "true",
        "MOCK_REGISTRY_API": "true",
        "LOG_LEVEL": "WARNING",
        "CACHE_SIZE_MB": "32",
        "MAX_WORKER_THREADS": "1"
    }
    
    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def fixtures_path() -> Path:
    """Return path to test fixtures directory."""
    return FIXTURES_DIR


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring Docker"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as fast unit tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests related to security functionality"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests related to performance functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        elif "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add markers based on test file names
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)
        elif "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that typically take longer
        slow_test_patterns = ["test_large_dockerfile", "test_batch_processing", "test_docker"]
        if any(pattern in item.nodeid for pattern in slow_test_patterns):
            item.add_marker(pytest.mark.slow)