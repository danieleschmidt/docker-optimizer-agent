# Docker Optimizer Agent - Examples

This directory contains comprehensive examples demonstrating how to use the Docker Optimizer Agent for various use cases.

## Quick Start Examples

### 1. Basic Optimization

**Original Dockerfile:**
```dockerfile
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3 python3-pip curl
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
```

**CLI Command:**
```bash
docker-optimizer --dockerfile Dockerfile --output Dockerfile.optimized
```

**Expected Optimized Result:**
```dockerfile
# Optimized Dockerfile - Size reduction: ~60MB
FROM ubuntu:22.04-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install dependencies in single layer and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3=3.10.6-1~22.04 \
        python3-pip=22.0.2+dfsg-1 \
        curl=7.81.0-1ubuntu1.4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "app.py"]
```

### 2. Multi-Stage Build Optimization

**CLI Command:**
```bash
docker-optimizer --dockerfile Dockerfile --multistage --format yaml
```

**Multi-Stage Result:**
```dockerfile
# Multi-stage build for maximum size reduction
FROM ubuntu:22.04-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        gcc \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build application
COPY requirements.txt .
RUN pip3 install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM ubuntu:22.04-slim AS runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3=3.10.6-1~22.04 \
        curl=7.81.0-1ubuntu1.4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application
COPY --chown=appuser:appuser . /app

USER appuser
WORKDIR /app

# Update PATH to include user packages
ENV PATH=/home/appuser/.local/bin:$PATH

HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "app.py"]
```

### 3. Security Scanning

**CLI Command:**
```bash
docker-optimizer --dockerfile Dockerfile --security-scan --format json
```

**Sample Security Report:**
```json
{
  "security_score": {
    "grade": "C",
    "score": 65,
    "analysis": "Several security improvements needed"
  },
  "vulnerabilities": {
    "total": 23,
    "critical": 2,
    "high": 5,
    "medium": 12,
    "low": 4
  },
  "recommendations": [
    "Use specific package versions instead of latest",
    "Run as non-root user",
    "Remove package caches and temporary files",
    "Use minimal base images (e.g., alpine, distroless)",
    "Update base image to latest security patches"
  ]
}
```

## Language-Specific Examples

### Python Application

**Dockerfile:**
```dockerfile
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**Optimized Result:**
```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r pyuser && useradd -r -g pyuser pyuser

# Set working directory
WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=pyuser:pyuser . .

# Switch to non-root user
USER pyuser

# Add Python user packages to PATH
ENV PATH=/home/pyuser/.local/bin:$PATH

CMD ["python", "app.py"]
```

### Node.js Application

**Dockerfile:**
```dockerfile
FROM node:16
COPY package.json .
RUN npm install
COPY . .
CMD ["npm", "start"]
```

**Multi-Stage Optimized:**
```dockerfile
# Build stage
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Runtime stage
FROM node:16-alpine AS runtime

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

WORKDIR /app

# Copy node_modules and application
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --chown=nextjs:nodejs . .

USER nextjs

EXPOSE 3000

CMD ["npm", "start"]
```

### Go Application

**Dockerfile:**
```dockerfile
FROM golang:1.19
COPY . .
RUN go build -o app
CMD ["./app"]
```

**Multi-Stage Optimized:**
```dockerfile
# Build stage
FROM golang:1.19-alpine AS builder

WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

# Runtime stage
FROM alpine:3.18

# Install CA certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN adduser -D -s /bin/sh appuser

WORKDIR /root/

# Copy binary from builder
COPY --from=builder /build/app .

# Switch to non-root user
USER appuser

CMD ["./app"]
```

## Advanced Use Cases

### 1. Batch Processing Multiple Dockerfiles

**CLI Command:**
```bash
docker-optimizer \
  --batch ./services/api/Dockerfile \
  --batch ./services/web/Dockerfile \
  --batch ./services/worker/Dockerfile \
  --performance \
  --performance-report \
  --format json
```

**Python Script for Batch Processing:**
```python
import asyncio
from pathlib import Path
from docker_optimizer.performance import PerformanceOptimizer

async def optimize_project_dockerfiles():
    """Optimize all Dockerfiles in a project."""
    optimizer = PerformanceOptimizer()
    
    # Find all Dockerfiles
    dockerfile_paths = list(Path(".").rglob("Dockerfile*"))
    
    # Read contents
    contents = []
    for path in dockerfile_paths:
        with open(path, 'r') as f:
            contents.append(f.read())
    
    # Optimize in parallel
    results = await optimizer.optimize_multiple_with_performance(contents)
    
    # Save optimized versions
    for path, result in zip(dockerfile_paths, results):
        optimized_path = path.parent / f"{path.name}.optimized"
        with open(optimized_path, 'w') as f:
            f.write(result.optimized_dockerfile)
        
        print(f"✅ {path}: {result.original_size} → {result.optimized_size}")
    
    # Show performance report
    report = optimizer.get_performance_report()
    print(f"\n📊 Performance Report:")
    print(f"Total processing time: {report['processing_time']:.2f}s")
    print(f"Memory usage: {report['memory_usage_mb']:.1f}MB")

if __name__ == "__main__":
    asyncio.run(optimize_project_dockerfiles())
```

### 2. Integration with CI/CD

**GitHub Actions Example:**
```yaml
name: Dockerfile Optimization

on:
  pull_request:
    paths:
      - '**/Dockerfile*'

jobs:
  optimize-dockerfiles:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Docker Optimizer
      run: pip install docker-optimizer-agent
    
    - name: Optimize Dockerfiles
      run: |
        find . -name "Dockerfile*" -type f | while read dockerfile; do
          echo "Optimizing $dockerfile"
          docker-optimizer \
            --dockerfile "$dockerfile" \
            --output "${dockerfile}.optimized" \
            --security-scan \
            --format json > "${dockerfile}.report.json"
        done
    
    - name: Upload optimization reports
      uses: actions/upload-artifact@v3
      with:
        name: dockerfile-optimization-reports
        path: "**/*.report.json"
```

### 3. Custom Optimization Rules

**Python Script with Custom Logic:**
```python
from docker_optimizer import DockerfileOptimizer
from docker_optimizer.security import SecurityAnalyzer

class CustomOptimizer:
    def __init__(self):
        self.optimizer = DockerfileOptimizer()
        self.security = SecurityAnalyzer()
    
    def optimize_with_custom_rules(self, content: str) -> str:
        """Apply custom optimization rules."""
        
        # Get base optimization
        result = self.optimizer.optimize_dockerfile(content)
        optimized = result.optimized_dockerfile
        
        # Apply custom rules
        optimized = self._enforce_company_standards(optimized)
        optimized = self._add_monitoring_labels(optimized)
        optimized = self._ensure_health_check(optimized)
        
        return optimized
    
    def _enforce_company_standards(self, dockerfile: str) -> str:
        """Enforce company-specific standards."""
        lines = dockerfile.split('\n')
        
        # Ensure all images come from company registry
        for i, line in enumerate(lines):
            if line.startswith('FROM') and 'company-registry.com' not in line:
                image = line.split()[-1]
                lines[i] = f"FROM company-registry.com/approved/{image}"
        
        # Add mandatory labels
        labels = [
            'LABEL maintainer="devops@company.com"',
            'LABEL version="1.0"',
            'LABEL security-scan="required"'
        ]
        
        # Insert labels after FROM
        for i, line in enumerate(lines):
            if line.startswith('FROM'):
                for j, label in enumerate(labels):
                    lines.insert(i + j + 1, label)
                break
        
        return '\n'.join(lines)
    
    def _add_monitoring_labels(self, dockerfile: str) -> str:
        """Add monitoring and observability labels."""
        lines = dockerfile.split('\n')
        
        monitoring_labels = [
            'LABEL prometheus.scrape="true"',
            'LABEL prometheus.port="9090"',
            'LABEL logs.format="json"'
        ]
        
        # Add before CMD
        for i, line in enumerate(lines):
            if line.startswith('CMD'):
                for j, label in enumerate(monitoring_labels):
                    lines.insert(i + j, label)
                break
        
        return '\n'.join(lines)
    
    def _ensure_health_check(self, dockerfile: str) -> str:
        """Ensure health check is present."""
        if 'HEALTHCHECK' not in dockerfile:
            lines = dockerfile.split('\n')
            
            # Add generic health check before CMD
            health_check = 'HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://localhost:8000/health || exit 1'
            
            for i, line in enumerate(lines):
                if line.startswith('CMD'):
                    lines.insert(i, health_check)
                    break
            
            dockerfile = '\n'.join(lines)
        
        return dockerfile

# Usage
optimizer = CustomOptimizer()
with open('Dockerfile', 'r') as f:
    original = f.read()

optimized = optimizer.optimize_with_custom_rules(original)
print(optimized)
```

### 4. Performance Monitoring

**Performance Analysis Script:**
```python
import time
from docker_optimizer.performance import PerformanceOptimizer, PerformanceMetrics

def benchmark_optimization():
    """Benchmark optimization performance."""
    
    # Sample Dockerfiles of different sizes
    dockerfiles = {
        'small': "FROM alpine:3.18\nRUN apk add --no-cache curl\nCMD ['curl', '--version']",
        'medium': "\n".join([
            "FROM ubuntu:22.04",
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip",
            "COPY requirements.txt .",
            "RUN pip3 install -r requirements.txt",
            "COPY . /app",
            "WORKDIR /app",
            "CMD ['python3', 'app.py']"
        ]),
        'large': "\n".join([f"RUN echo 'step {i}'" for i in range(100)])
    }
    
    optimizer = PerformanceOptimizer()
    
    for size, content in dockerfiles.items():
        metrics = PerformanceMetrics()
        
        with metrics.timer():
            result = optimizer.optimize_with_performance(content)
        
        report = metrics.get_report()
        
        print(f"\n{size.upper()} Dockerfile Performance:")
        print(f"  Processing time: {report['processing_time']:.3f}s")
        print(f"  Memory usage: {report['memory_usage_mb']:.1f}MB")
        print(f"  Cache hit: {'Yes' if 'cached' in result.explanation.lower() else 'No'}")

if __name__ == "__main__":
    benchmark_optimization()
```

## Best Practices Guide

### 1. Security Best Practices

- **Always use specific version tags:** `FROM python:3.11-slim` instead of `FROM python:latest`
- **Run as non-root user:** Create and use dedicated application users
- **Minimize attack surface:** Use minimal base images like Alpine or distroless
- **Regular security scans:** Use `--security-scan` flag in CI/CD
- **Clean up package caches:** Remove apt/apk caches after installation

### 2. Performance Best Practices

- **Leverage Docker layer caching:** Copy requirements files before source code
- **Use multi-stage builds:** Separate build and runtime environments
- **Minimize layer count:** Combine RUN commands with &&
- **Use .dockerignore:** Exclude unnecessary files from build context
- **Enable performance optimization:** Use `--performance` flag for large projects

### 3. Maintenance Best Practices

- **Regular optimization:** Run optimization on Dockerfile changes
- **Monitor image sizes:** Track size reduction over time
- **Update dependencies:** Keep base images and packages current
- **Document changes:** Use descriptive commit messages for Dockerfile changes

## Troubleshooting

### Common Issues

**1. Permission denied when running as non-root:**
```dockerfile
# Fix: Change file ownership when copying
COPY --chown=appuser:appuser . /app
```

**2. Package installation fails:**
```dockerfile
# Fix: Update package lists first
RUN apt-get update && apt-get install -y package-name
```

**3. Large image sizes:**
```bash
# Solution: Use multi-stage builds
docker-optimizer --dockerfile Dockerfile --multistage
```

**4. Security vulnerabilities:**
```bash
# Solution: Run security scan and apply recommendations
docker-optimizer --dockerfile Dockerfile --security-scan
```

### Getting Help

- Check the [API documentation](../api/README.md) for detailed method signatures
- Use `docker-optimizer --help` for CLI usage
- Enable verbose mode with `--verbose` for detailed output
- Check GitHub issues for known problems and solutions