# API Contract Testing

This directory contains API contract tests to ensure that the Docker Optimizer Agent's API interfaces remain stable and compatible.

## Overview

Contract testing validates that:
- Public API interfaces don't break unexpectedly
- Input/output data structures remain consistent
- Error handling follows documented patterns
- Performance characteristics meet expectations

## Test Structure

- `optimizer_contracts.py` - Core optimization API contracts
- `security_contracts.py` - Security scanning API contracts
- `cli_contracts.py` - Command line interface contracts
- `schemas/` - JSON schemas for API validation

## Running Contract Tests

```bash
# Run all contract tests
pytest tests/contracts/ -v

# Run specific contract tests
pytest tests/contracts/test_optimizer_contracts.py -v
```

## Adding New Contracts

When adding new API methods or changing existing ones:

1. Create or update the relevant contract test
2. Define expected input/output schemas
3. Test both success and error scenarios
4. Validate backwards compatibility

## Schema Validation

Contract tests use JSON schemas to validate API responses:

```python
import jsonschema

def test_optimization_result_schema(optimization_result):
    schema = load_schema('optimization_result.json')
    jsonschema.validate(optimization_result, schema)
```