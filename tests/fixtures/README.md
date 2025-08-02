# Test Fixtures

This directory contains test fixtures used across the test suite.

## Directory Structure

- `dockerfiles/` - Sample Dockerfiles for testing different scenarios
- `security/` - Security-related test data (CVE reports, vulnerability samples)
- `performance/` - Performance test data and benchmarks
- `configs/` - Configuration files for testing
- `responses/` - Mock API responses for external services

## Using Fixtures

Fixtures can be accessed in tests using the `fixtures_path` fixture:

```python
def test_example(fixtures_path):
    dockerfile_path = fixtures_path / "dockerfiles" / "python-app.dockerfile"
    with open(dockerfile_path) as f:
        content = f.read()
    # Use fixture content in test
```

## Adding New Fixtures

When adding new fixtures:

1. Place them in the appropriate subdirectory
2. Use descriptive names that indicate the test scenario
3. Add documentation in this README if the fixture is complex
4. Keep fixtures minimal but representative of real-world scenarios