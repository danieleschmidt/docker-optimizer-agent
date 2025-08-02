# Deployment Guide

This guide covers building, packaging, and deploying the Docker Optimizer Agent across different environments.

## Quick Start

```bash
# Development build
./scripts/build.sh

# Production build with push
./scripts/build.sh -t production -p -r your-registry.com

# CLI-only build
./scripts/build.sh -t cli
```

## Build System Overview

The Docker Optimizer Agent uses a multi-stage build system that creates optimized images for different use cases:

- **Development**: Full development environment with all tools
- **Testing**: Optimized for CI/CD with testing dependencies
- **Production**: Minimal runtime image for production deployment
- **CLI**: Lightweight command-line only image
- **Security**: Specialized image for security scanning

## Build Targets

### Development Image

```bash
# Build development image
docker build --target development -t docker-optimizer:dev .

# Or use build script
./scripts/build.sh -t development
```

**Features:**
- Full development environment
- All dependencies and tools
- Pre-commit hooks setup
- External security tools (Trivy)
- Volume mounting for live development

**Use Cases:**
- Local development
- IDE integration
- Interactive debugging

### Testing Image

```bash
# Build testing image
docker build --target testing -t docker-optimizer:test .

# Run tests in container
docker run --rm docker-optimizer:test
```

**Features:**
- Optimized for CI/CD
- Testing dependencies only
- Pre-installed external tools
- Fast test execution

**Use Cases:**
- Continuous Integration
- Automated testing
- Quality gates

### Production Image  

```bash
# Build production image
docker build --target production -t docker-optimizer:latest .

# Run production container
docker run -p 8000:8000 docker-optimizer:latest
```

**Features:**
- Minimal runtime dependencies
- Non-root user security
- Health checks configured
- Multi-architecture support
- Optimized for size and security

**Use Cases:**
- Production deployments
- Container orchestration
- Cloud deployments

### CLI Image

```bash
# Build CLI image
docker build --target cli -t docker-optimizer:cli .

# Use as CLI tool
docker run --rm -v $(pwd):/workspace docker-optimizer:cli \
  --dockerfile /workspace/Dockerfile --output /workspace/Dockerfile.optimized
```

**Features:**
- Lightweight CLI-only
- Minimal dependencies
- Easy integration
- Fast startup

**Use Cases:**
- CI/CD integration
- Command-line usage
- Batch processing

## Docker Compose Deployment

### Development Environment

```bash
# Start full development stack
docker-compose up -d

# Services included:
# - docker-optimizer (main service)
# - trivy-server (security scanning)
# - registry (local Docker registry)
# - prometheus (metrics)
# - grafana (dashboards)
```

### Testing Environment

```bash
# Run testing services only
docker-compose --profile testing up test-runner

# Run security scanning
docker-compose --profile security up security-scanner
```

### Production Environment

Create a production docker-compose file:

```yaml
version: '3.8'
services:
  docker-optimizer:
    image: docker-optimizer:production
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import docker_optimizer; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Build Automation

### Using the Build Script

The `scripts/build.sh` script provides comprehensive build automation:

```bash
# Full production build with all checks
./scripts/build.sh -t production

# Skip tests for faster builds
./scripts/build.sh -t production -s

# Skip security scans
./scripts/build.sh -t production -S

# Build and push to registry
./scripts/build.sh -t production -p -r your-registry.com

# Custom version override
./scripts/build.sh -t production -v 1.2.3
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --type` | Build type (development, testing, production, cli) | development |
| `-s, --skip-tests` | Skip running tests | false |
| `-S, --skip-security` | Skip security scans | false |
| `-p, --push` | Push images to registry | false |
| `-r, --registry` | Docker registry URL | - |
| `-v, --version` | Override version | from pyproject.toml |

### Makefile Integration

```bash
# Build using Makefile
make docker-build        # Build all images
make docker-test         # Test in container
make release            # Full release build
```

## Registry Management

### Docker Hub

```bash
# Login to Docker Hub
docker login

# Build and push
./scripts/build.sh -t production -p
```

### Private Registry

```bash
# Login to private registry
docker login your-registry.com

# Build and push with registry prefix
./scripts/build.sh -t production -p -r your-registry.com/docker-optimizer
```

### GitHub Container Registry

```bash
# Login with GitHub token
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build and push
./scripts/build.sh -t production -p -r ghcr.io/username/docker-optimizer
```

## Cloud Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docker-optimizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: docker-optimizer
  template:
    metadata:
      labels:
        app: docker-optimizer
    spec:
      containers:
      - name: docker-optimizer
        image: docker-optimizer:production
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Docker Swarm

```yaml
version: '3.8'
services:
  docker-optimizer:
    image: docker-optimizer:production
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
```

### AWS ECS

```json
{
  "family": "docker-optimizer",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "docker-optimizer",
      "image": "your-registry.com/docker-optimizer:production",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/docker-optimizer",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## Security Considerations

### Image Security

```bash
# Scan images for vulnerabilities
trivy image docker-optimizer:production

# Run security analysis
./scripts/build.sh -t security
docker run --rm docker-optimizer:security
```

### Runtime Security

- Images run as non-root user
- Minimal attack surface
- Security scanning integrated
- Regular base image updates
- Secrets management via environment variables

## Performance Optimization

### Image Size Optimization

- Multi-stage builds reduce final image size
- Minimal base images (alpine, distroless)
- Dependency optimization
- Layer caching optimization

### Build Performance

```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Use build cache
docker build --cache-from docker-optimizer:latest .

# Parallel builds
./scripts/build.sh -t production &
./scripts/build.sh -t cli &
wait
```

## Monitoring and Observability

### Health Checks

All images include health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import docker_optimizer; print('OK')" || exit 1
```

### Metrics Collection

Production images expose metrics:

- Prometheus metrics endpoint
- Application performance metrics
- Custom optimization metrics

### Logging

Structured logging configuration:

```bash
# Configure logging level
docker run -e LOG_LEVEL=DEBUG docker-optimizer:production

# JSON structured logs
docker run -e LOG_FORMAT=json docker-optimizer:production
```

## Troubleshooting

### Common Build Issues

1. **Docker daemon not running**
   ```bash
   sudo systemctl start docker
   ```

2. **Permission denied**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **Build context too large**
   ```bash
   # Check .dockerignore file
   docker system df
   docker system prune
   ```

4. **Registry authentication**
   ```bash
   docker login your-registry.com
   ```

### Debug Builds

```bash
# Build with debug output
docker build --progress=plain --no-cache .

# Inspect build stages
docker build --target dependencies -t debug-deps .
docker run -it debug-deps /bin/bash
```

---

*For more deployment scenarios and advanced configurations, see the examples in `docs/examples/`*