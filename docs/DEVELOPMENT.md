# Development Guide

## Prerequisites

* Python 3.9+
* Docker and Docker Compose
* Make

## Quick Setup

```bash
# Clone and setup
git clone <repository-url>
cd docker-optimizer-agent
make setup

# Run tests
make test

# Start development environment  
make dev
```

## Project Structure

* `src/docker_optimizer/` - Core optimization logic
* `tests/` - Test suite
* `docs/` - Documentation
* `monitoring/` - Grafana/Prometheus configs

## Development Workflow  

1. Create feature branch
2. Make changes with tests
3. Run `make lint test`
4. Submit pull request

## Testing

* Unit tests: `make test`
* Integration tests: `make test-integration`
* Coverage: `make coverage`

See [Python Developer Guide](https://devguide.python.org/) for best practices.