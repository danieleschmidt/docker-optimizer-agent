# Testing Documentation

## Overview

The Docker Optimizer Agent uses a comprehensive testing strategy with **85%+ code coverage** requirement and multiple testing layers to ensure reliability, security, and performance.

## Testing Architecture

### Test Categories

#### 1. Unit Tests (`tests/test_*.py`)
- **Purpose**: Test individual components in isolation
- **Coverage**: All core modules (optimizer, parser, security, etc.)
- **Framework**: pytest with extensive fixtures
- **Execution**: `make test` or `pytest tests/`

#### 2. Integration Tests (`tests/test_integration.py`)
- **Purpose**: Test component interactions and workflows
- **Coverage**: End-to-end optimization processes
- **Framework**: pytest with Docker containers
- **Execution**: `make test-integration`

#### 3. Performance Tests (`benchmarks/`)
- **Purpose**: Regression testing and performance validation
- **Coverage**: Optimization speed, memory usage, accuracy
- **Framework**: pytest-benchmark
- **Execution**: `make benchmark`

#### 4. Security Tests
- **Purpose**: Validate security scanning and vulnerability detection
- **Coverage**: Trivy integration, threat detection accuracy
- **Framework**: pytest with security fixtures
- **Execution**: `make test-security`

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── fixtures/                   # Test data and mock objects
│   ├── configs/               # Test configuration files
│   ├── dockerfiles/           # Sample Dockerfiles for testing
│   └── security/              # Security scan test data
├── contracts/                  # API contract tests
└── test_*.py                  # Individual test modules
```

## Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=docker_optimizer",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=85"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "security: marks tests as security-related",
    "benchmark: marks tests as performance benchmarks"
]
```

### Coverage Requirements

- **Minimum Coverage**: 85%
- **Coverage Report**: HTML format in `htmlcov/`
- **Branch Coverage**: Enabled for comprehensive analysis
- **Exclusions**: Test files, generated code, external dependencies

## Test Execution

### Local Development

```bash
# Run all tests with coverage
make test

# Run specific test categories
pytest -m "not slow"              # Skip slow tests
pytest -m integration            # Integration tests only
pytest -m security              # Security tests only

# Run with verbose output
pytest -v tests/test_optimizer.py

# Run specific test function
pytest tests/test_optimizer.py::test_basic_optimization
```

### Continuous Integration

```bash
# Full test suite (CI/CD)
make test-all

# Security-focused testing
make test-security

# Performance regression testing
make benchmark-ci
```

## Test Data Management

### Fixtures (`tests/fixtures/`)

#### Dockerfile Fixtures
- `python-basic.dockerfile`: Simple Python application
- `nodejs-multistage.dockerfile`: Complex multi-stage build
- `security-issues.dockerfile`: Contains known vulnerabilities

#### Configuration Fixtures
- `test-config.yml`: Standard test configuration
- Security scan results and mock data

### Fixture Usage

```python
# In test files
def test_optimization(basic_dockerfile_fixture):
    result = optimizer.optimize(basic_dockerfile_fixture)
    assert result.success

# Custom fixtures in conftest.py
@pytest.fixture
def mock_security_scanner():
    return MockTrivyScanner()
```

## Testing Best Practices

### 1. Test Organization
- **One test file per module**: `test_optimizer.py` for `optimizer.py`
- **Descriptive test names**: `test_optimization_reduces_layer_count`
- **Logical grouping**: Related tests in the same class

### 2. Test Independence
- **No test dependencies**: Each test runs independently
- **Clean state**: Use fixtures for setup/teardown
- **Isolated resources**: Unique temporary directories per test

### 3. Mock Strategy
- **External dependencies**: Mock Docker API, Trivy scanner
- **File system operations**: Use temporary directories
- **Network calls**: Mock HTTP requests and responses

### 4. Assertion Quality
- **Specific assertions**: Check exact values, not just truthiness
- **Multiple assertions**: Validate all relevant aspects
- **Error conditions**: Test both success and failure paths

## Performance Testing

### Benchmark Tests (`benchmarks/`)

```python
def test_optimization_performance(benchmark):
    result = benchmark(optimizer.optimize, dockerfile_content)
    assert result.execution_time < 5.0  # 5 second limit
```

### Performance Metrics
- **Optimization Speed**: Sub-5 second analysis requirement
- **Memory Usage**: Monitor memory consumption during tests
- **Regression Detection**: Compare against baseline performance

## Security Testing

### Vulnerability Detection
- **Known CVE Testing**: Validate detection of specific vulnerabilities
- **False Positive Checking**: Ensure clean files pass security scans
- **Scanner Integration**: Test Trivy integration reliability

### Security Test Examples

```python
def test_security_vulnerability_detection():
    dockerfile_with_vuln = load_fixture("security-issues.dockerfile")
    result = security_scanner.scan(dockerfile_with_vuln)
    assert result.has_critical_vulnerabilities
    assert len(result.vulnerabilities) > 0

def test_clean_dockerfile_security():
    clean_dockerfile = load_fixture("python-basic.dockerfile")
    result = security_scanner.scan(clean_dockerfile)
    assert not result.has_critical_vulnerabilities
```

## Integration Testing

### End-to-End Workflows
- **Complete optimization pipeline**: Parse → Analyze → Optimize → Validate
- **CLI interface testing**: Command-line argument validation
- **Configuration loading**: Test various configuration scenarios

### Integration Test Structure

```python
def test_full_optimization_workflow():
    # Test complete pipeline
    input_dockerfile = load_fixture("complex-app.dockerfile")
    
    # Parse
    parsed = parser.parse(input_dockerfile)
    assert parsed.is_valid
    
    # Optimize
    optimized = optimizer.optimize(parsed)
    assert optimized.improvements_count > 0
    
    # Validate
    result = validator.validate(optimized.dockerfile)
    assert result.is_valid
```

## Test Automation

### Pre-commit Hooks
- **Test execution**: Run relevant tests before commit
- **Coverage validation**: Ensure coverage doesn't decrease
- **Linting**: Validate test code quality

### CI/CD Integration
- **Parallel execution**: Tests run in parallel for speed
- **Multiple environments**: Test on different Python versions
- **Coverage reporting**: Results sent to coverage services

## Debugging Tests

### Common Issues

1. **Flaky Tests**
   ```bash
   # Run test multiple times to identify flakiness
   pytest --count=10 tests/test_flaky.py
   ```

2. **Slow Tests**
   ```bash
   # Profile test execution time
   pytest --durations=10 tests/
   ```

3. **Coverage Gaps**
   ```bash
   # Generate detailed coverage report
   pytest --cov-report=html
   open htmlcov/index.html
   ```

### Debug Configuration

```python
# Add to conftest.py for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use pytest debugging
pytest --pdb tests/test_failing.py
```

## Maintenance

### Test Data Updates
- **Regular refresh**: Update security fixtures with latest CVE data
- **Dockerfile modernization**: Keep test Dockerfiles current
- **Configuration sync**: Align test configs with production

### Coverage Monitoring
- **Monthly review**: Analyze coverage reports for gaps
- **New feature testing**: Ensure new code has comprehensive tests
- **Refactoring support**: Update tests when code structure changes

---

## Quick Reference

### Essential Commands

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test file
pytest tests/test_optimizer.py

# Run integration tests only
pytest -m integration

# Generate coverage report
pytest --cov-report=html
```

### Coverage Targets
- **Unit Tests**: 90%+ coverage
- **Integration Tests**: Key workflows covered
- **Overall Project**: 85%+ minimum

### Performance Targets
- **Test Execution**: Complete suite under 60 seconds
- **Individual Tests**: Most tests under 1 second
- **Integration Tests**: Under 10 seconds each

---

*For questions about testing, see our [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.*