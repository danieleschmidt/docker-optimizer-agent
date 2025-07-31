# Claude Code Configuration

## Repository Context

This is the **Docker Optimizer Agent** - an intelligent Dockerfile optimization tool with security scanning, performance analysis, and best practices enforcement.

## Key Commands

### Testing
```bash
# Run full test suite
make test

# Run with coverage
make coverage

# Run type checking
make typecheck

# Run security scanning
make security
```

### Development
```bash
# Setup development environment
make setup

# Format code
make format

# Run linting
make lint

# Start monitoring stack
make monitor
```

### Build & Release
```bash
# Build package
make build

# Build Docker images
make docker-build

# Run release checks
make release
```

## Testing Strategy

- **Unit Tests**: `tests/test_*.py` (23 test files)
- **Integration Tests**: Full workflow validation
- **Benchmark Tests**: Performance regression detection
- **Security Tests**: Vulnerability scanning validation

## Code Quality Standards

- **Coverage**: Minimum 85% required
- **Linting**: Ruff with security rules enabled
- **Type Checking**: MyPy strict mode
- **Security**: Bandit + Safety scanning
- **Formatting**: Black + isort

## Architecture Notes

- **Core Module**: `src/docker_optimizer/`
- **CLI Entry Point**: `docker_optimizer.cli:main`
- **Security Scanner**: External Trivy integration
- **Performance**: LRU caching + parallel processing
- **Monitoring**: OpenTelemetry + Prometheus metrics

## Repository Maturity

**Advanced (85%+ SDLC maturity)** with:
- Comprehensive testing infrastructure
- Advanced security scanning
- Performance benchmarking
- Monitoring and observability
- Professional documentation
- Automated dependency management