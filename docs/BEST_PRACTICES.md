# Docker Optimization Best Practices Guide

This guide provides comprehensive best practices for creating secure, efficient, and maintainable Dockerfiles using the Docker Optimizer Agent.

## Table of Contents

1. [Security Best Practices](#security-best-practices)
2. [Performance Optimization](#performance-optimization)
3. [Multi-Stage Builds](#multi-stage-builds)
4. [Image Size Reduction](#image-size-reduction)
5. [Layer Optimization](#layer-optimization)
6. [Base Image Selection](#base-image-selection)
7. [Package Management](#package-management)
8. [Health Checks and Monitoring](#health-checks-and-monitoring)
9. [CI/CD Integration](#cicd-integration)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Security Best Practices

### 1. Use Specific Version Tags

❌ **Bad:**
```dockerfile
FROM ubuntu:latest
FROM python:latest
```

✅ **Good:**
```dockerfile
FROM ubuntu:22.04-slim
FROM python:3.11-slim
```

**Rationale:** Using `latest` tags introduces unpredictability and potential security vulnerabilities. Specific versions ensure reproducible builds and allow for controlled updates.

**Docker Optimizer Command:**
```bash
docker-optimizer --dockerfile Dockerfile --security-scan
```

### 2. Run as Non-Root User

❌ **Bad:**
```dockerfile
FROM ubuntu:22.04
COPY . /app
WORKDIR /app
CMD ["./app"]
```

✅ **Good:**
```dockerfile
FROM ubuntu:22.04-slim

# Create dedicated user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory with proper ownership
RUN mkdir -p /app && chown appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy with proper ownership
COPY --chown=appuser:appuser . .

CMD ["./app"]
```

**Benefits:**
- Reduces attack surface
- Follows principle of least privilege
- Prevents container breakout scenarios

### 3. Minimize Attack Surface

❌ **Bad:**
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    vim \
    git \
    ssh \
    sudo
```

✅ **Good:**
```dockerfile
FROM ubuntu:22.04-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl=7.81.0-1ubuntu1.4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
```

**Benefits:**
- Smaller attack surface
- Reduced image size
- Fewer potential vulnerabilities

### 4. Security Scanning Integration

**Regular Security Scans:**
```bash
# In CI/CD pipeline
docker-optimizer --dockerfile Dockerfile --security-scan --format json > security-report.json

# Check security grade
if [ "$(jq -r '.security_score.grade' security-report.json)" = "F" ]; then
    echo "Security grade F - build failed"
    exit 1
fi
```

**Weekly Security Reviews:**
```yaml
# .github/workflows/security.yml
name: Weekly Security Scan
on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Security Scan
        run: |
          docker-optimizer --dockerfile Dockerfile --security-scan
```

## Performance Optimization

### 1. Leverage Docker Layer Caching

❌ **Bad (cache invalidation on code changes):**
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

✅ **Good (dependencies cached separately):**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (changes frequently)
COPY . .

CMD ["python", "app.py"]
```

**Performance Optimization Command:**
```bash
docker-optimizer --dockerfile Dockerfile --performance --performance-report
```

### 2. Use .dockerignore

Create a `.dockerignore` file:
```
.git
.gitignore
README.md
Dockerfile
.dockerignore
node_modules
npm-debug.log
coverage
.nyc_output
.env
.vscode
.idea
__pycache__
*.pyc
*.pyo
*.pyd
.pytest_cache
.coverage
.DS_Store
```

### 3. Parallel Processing for Multiple Dockerfiles

**Python Script:**
```python
import asyncio
from docker_optimizer.performance import PerformanceOptimizer

async def optimize_multiple():
    optimizer = PerformanceOptimizer(max_workers=8)
    
    dockerfiles = [
        open('service1/Dockerfile').read(),
        open('service2/Dockerfile').read(),
        open('service3/Dockerfile').read(),
    ]
    
    results = await optimizer.optimize_multiple_with_performance(dockerfiles)
    
    for i, result in enumerate(results):
        print(f"Service {i+1}: {result.original_size} → {result.optimized_size}")

asyncio.run(optimize_multiple())
```

**CLI Batch Processing:**
```bash
docker-optimizer \
  --batch service1/Dockerfile \
  --batch service2/Dockerfile \
  --batch service3/Dockerfile \
  --performance \
  --performance-report
```

## Multi-Stage Builds

### 1. Build vs Runtime Separation

❌ **Single-stage (includes build tools in final image):**
```dockerfile
FROM node:16
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

✅ **Multi-stage (build tools excluded from final image):**
```dockerfile
# Build stage
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force
COPY . .
RUN npm run build

# Runtime stage
FROM node:16-alpine AS runtime
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
WORKDIR /app

# Copy only production files
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules

USER nextjs
EXPOSE 3000
CMD ["npm", "start"]
```

**Generate Multi-Stage Builds:**
```bash
docker-optimizer --dockerfile Dockerfile --multistage --output Dockerfile.multistage
```

### 2. Language-Specific Multi-Stage Patterns

**Python with Compiled Dependencies:**
```dockerfile
# Build stage
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim AS runtime
RUN groupadd -r pyuser && useradd -r -g pyuser pyuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/pyuser/.local

WORKDIR /app
COPY --chown=pyuser:pyuser . .

USER pyuser
ENV PATH=/home/pyuser/.local/bin:$PATH

CMD ["python", "app.py"]
```

**Go Static Binary:**
```dockerfile
# Build stage
FROM golang:1.19-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

# Runtime stage
FROM scratch AS runtime
COPY --from=builder /build/app /app
EXPOSE 8080
ENTRYPOINT ["/app"]
```

## Image Size Reduction

### 1. Choose Minimal Base Images

**Base Image Size Comparison:**
```
ubuntu:22.04        → 77MB
ubuntu:22.04-slim   → 28MB
alpine:3.18         → 7MB
scratch             → 0MB (static binaries only)
distroless/static   → ~2MB
```

**Size Optimization Strategy:**
```bash
# Check current size
docker images my-app:latest

# Optimize with multi-stage
docker-optimizer --dockerfile Dockerfile --multistage

# Build optimized image
docker build -f Dockerfile.optimized -t my-app:optimized .

# Compare sizes
docker images | grep my-app
```

### 2. Remove Package Caches

❌ **Bad (leaves caches):**
```dockerfile
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip
```

✅ **Good (single layer with cleanup):**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3=3.10.6-1~22.04 \
        python3-pip=22.0.2+dfsg-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
```

### 3. Use Multi-Architecture Builds

```dockerfile
# Automatic platform detection
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

# Platform-specific optimizations
ARG TARGETPLATFORM
RUN case "${TARGETPLATFORM}" in \
    "linux/amd64") echo "Optimizing for AMD64" ;; \
    "linux/arm64") echo "Optimizing for ARM64" ;; \
    *) echo "Generic optimization" ;; \
    esac
```

## Layer Optimization

### 1. Minimize Layer Count

❌ **Bad (multiple layers):**
```dockerfile
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
```

✅ **Good (single optimized layer):**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

### 2. Order Layers by Change Frequency

```dockerfile
FROM python:3.11-slim

# 1. System dependencies (rarely change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies (change occasionally)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Application configuration (change sometimes)
COPY config/ ./config/

# 4. Application code (change frequently)
COPY src/ ./src/

# 5. Entry point (rarely changes)
CMD ["python", "src/app.py"]
```

### 3. Use Bind Mounts for Development

**Development Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Don't copy source in dev (use bind mount)
# docker run -v $(pwd):/app my-app:dev

CMD ["python", "app.py"]
```

## Base Image Selection

### 1. Security-First Selection

**Security Rating (Best to Worst):**
1. **Distroless images** - Minimal attack surface
2. **Alpine Linux** - Small, security-focused
3. **Slim variants** - Debian/Ubuntu minimal
4. **Standard images** - Full OS features
5. **Latest tags** - Unpredictable versions

**Example Selection Process:**
```bash
# Check available variants
docker-optimizer --dockerfile Dockerfile --security-scan

# Test different bases
echo "FROM alpine:3.18" > Dockerfile.alpine
echo "FROM ubuntu:22.04-slim" > Dockerfile.slim
echo "FROM distroless/python3" > Dockerfile.distroless

# Compare security scores
for dockerfile in Dockerfile.*; do
    echo "=== $dockerfile ==="
    docker-optimizer --dockerfile $dockerfile --security-scan --format json | \
        jq '.security_score.grade'
done
```

### 2. Language-Specific Recommendations

**Python:**
```dockerfile
# Production: Distroless (most secure)
FROM gcr.io/distroless/python3

# Development: Slim (good balance)
FROM python:3.11-slim

# Full features: Standard (when you need system tools)
FROM python:3.11
```

**Node.js:**
```dockerfile
# Production: Alpine (small and secure)
FROM node:16-alpine

# Development: Slim
FROM node:16-slim

# Full features: Standard
FROM node:16
```

**Go:**
```dockerfile
# Production: Scratch (smallest possible)
FROM scratch

# With SSL certificates: Distroless
FROM gcr.io/distroless/static

# Development: Alpine
FROM golang:1.19-alpine
```

## Package Management

### 1. Pin Package Versions

❌ **Bad (unpredictable versions):**
```dockerfile
RUN apt-get install -y python3 curl wget
```

✅ **Good (pinned versions):**
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3=3.10.6-1~22.04 \
    curl=7.81.0-1ubuntu1.4 \
    wget=1.21.2-2ubuntu1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```

### 2. Use Package Manager Best Practices

**APT (Debian/Ubuntu):**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        package1 \
        package2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
```

**APK (Alpine):**
```dockerfile
RUN apk add --no-cache \
    package1 \
    package2 && \
    rm -rf /var/cache/apk/*
```

**YUM/DNF (RedHat/CentOS):**
```dockerfile
RUN yum install -y \
    package1 \
    package2 && \
    yum clean all && \
    rm -rf /var/cache/yum
```

### 3. Language Package Managers

**Python pip:**
```dockerfile
# Use requirements.txt with pinned versions
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# For production, consider pip-tools
# pip-compile --generate-hashes requirements.in
# RUN pip install --no-cache-dir --require-hashes -r requirements.txt
```

**Node.js npm:**
```dockerfile
# Use package-lock.json for reproducible builds
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force
```

**Go modules:**
```dockerfile
# Leverage module cache
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o app .
```

## Health Checks and Monitoring

### 1. Implement Health Checks

```dockerfile
# HTTP service health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# TCP port health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD nc -z localhost 5432 || exit 1

# Custom script health check
COPY healthcheck.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/healthcheck.sh
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh
```

### 2. Add Observability Labels

```dockerfile
# Standard labels
LABEL maintainer="team@company.com"
LABEL version="1.2.3"
LABEL description="Application description"

# Monitoring labels
LABEL prometheus.scrape="true"
LABEL prometheus.port="9090"
LABEL prometheus.path="/metrics"

# Logging configuration
LABEL logs.format="json"
LABEL logs.level="info"

# Security and compliance
LABEL security.scan="required"
LABEL compliance.level="high"
```

### 3. Structured Logging

```dockerfile
# Configure application for structured logging
ENV LOG_FORMAT=json
ENV LOG_LEVEL=info
ENV LOG_OUTPUT=stdout

# Ensure proper signal handling
STOPSIGNAL SIGTERM
```

## CI/CD Integration

### 1. Automated Optimization in CI

**GitHub Actions Example:**
```yaml
name: Docker Optimization

on:
  push:
    paths: ['**/Dockerfile*']

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Docker Optimizer
        run: pip install docker-optimizer-agent
      
      - name: Optimize Dockerfiles
        run: |
          find . -name "Dockerfile*" -type f | while read dockerfile; do
            echo "Optimizing $dockerfile"
            
            # Run optimization with security scan
            docker-optimizer \
              --dockerfile "$dockerfile" \
              --output "${dockerfile}.optimized" \
              --security-scan \
              --format json > "${dockerfile}.report.json"
            
            # Check security grade
            grade=$(jq -r '.security_score.grade' "${dockerfile}.report.json")
            if [ "$grade" = "F" ]; then
              echo "❌ Security grade F for $dockerfile"
              exit 1
            fi
            
            echo "✅ Security grade $grade for $dockerfile"
          done
      
      - name: Create PR with optimizations
        if: github.event_name == 'push'
        run: |
          # Create branch and commit optimized Dockerfiles
          git checkout -b "optimize-dockerfiles-$(date +%s)"
          git add "*.optimized"
          git commit -m "feat: optimize Dockerfiles for security and size"
          git push origin HEAD
          
          # Create PR (requires gh CLI)
          gh pr create \
            --title "Docker Optimization Recommendations" \
            --body "Automated Dockerfile optimizations with security improvements"
```

### 2. Quality Gates

**Security Quality Gate:**
```bash
#!/bin/bash
# security-gate.sh

dockerfile="$1"
docker-optimizer --dockerfile "$dockerfile" --security-scan --format json > report.json

grade=$(jq -r '.security_score.grade' report.json)
critical=$(jq -r '.vulnerabilities.critical' report.json)

echo "Security Grade: $grade"
echo "Critical Vulnerabilities: $critical"

if [ "$grade" = "F" ] || [ "$critical" -gt 0 ]; then
    echo "❌ Security quality gate failed"
    exit 1
fi

echo "✅ Security quality gate passed"
```

**Size Quality Gate:**
```bash
#!/bin/bash
# size-gate.sh

dockerfile="$1"
max_size_mb=500

# Build and check image size
docker build -f "$dockerfile" -t temp-image .
size_bytes=$(docker inspect temp-image --format='{{.Size}}')
size_mb=$((size_bytes / 1024 / 1024))

echo "Image size: ${size_mb}MB"

if [ "$size_mb" -gt "$max_size_mb" ]; then
    echo "❌ Image too large (${size_mb}MB > ${max_size_mb}MB)"
    echo "Consider using multi-stage builds:"
    docker-optimizer --dockerfile "$dockerfile" --multistage
    exit 1
fi

echo "✅ Size quality gate passed"
docker rmi temp-image
```

### 3. Continuous Monitoring

**Dockerfile Drift Detection:**
```yaml
name: Dockerfile Drift Detection

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Detect optimization opportunities
        run: |
          # Check all Dockerfiles for new optimization opportunities
          find . -name "Dockerfile*" -type f | while read dockerfile; do
            echo "Analyzing $dockerfile"
            
            docker-optimizer \
              --dockerfile "$dockerfile" \
              --analysis-only \
              --format json > "${dockerfile}.analysis.json"
            
            # Check for new optimization opportunities
            opportunities=$(jq -r '.optimization_opportunities | length' "${dockerfile}.analysis.json")
            
            if [ "$opportunities" -gt 0 ]; then
              echo "🔍 Found $opportunities optimization opportunities in $dockerfile"
              
              # Create issue for manual review
              gh issue create \
                --title "Dockerfile optimization opportunities: $dockerfile" \
                --body "$(jq -r '.optimization_opportunities[] | "- " + .description' "${dockerfile}.analysis.json")" \
                --label "optimization,technical-debt"
            fi
          done
```

## Troubleshooting Common Issues

### 1. Permission Issues

**Problem:** Permission denied when running as non-root user
```
docker: Error response from daemon: container_linux.go: starting container process caused: 
exec: "./app": permission denied
```

**Solution:**
```dockerfile
# Ensure executable permissions
COPY --chmod=755 app /usr/local/bin/app

# Or use RUN to set permissions
COPY app /usr/local/bin/app
RUN chmod +x /usr/local/bin/app
```

### 2. Package Installation Failures

**Problem:** Package installation fails in minimal images
```
E: Unable to locate package python3-dev
```

**Solution:**
```dockerfile
# Update package lists first
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# For Alpine, use apk equivalent
RUN apk add --no-cache python3-dev
```

### 3. Large Image Sizes

**Problem:** Images are larger than expected

**Diagnosis:**
```bash
# Analyze layer sizes
docker history my-image:latest

# Use dive tool for detailed analysis
docker run --rm -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    wagoodman/dive:latest my-image:latest
```

**Solutions:**
```bash
# Use multi-stage builds
docker-optimizer --dockerfile Dockerfile --multistage

# Optimize existing Dockerfile
docker-optimizer --dockerfile Dockerfile --output Dockerfile.optimized

# Check optimization impact
docker build -f Dockerfile -t original .
docker build -f Dockerfile.optimized -t optimized .
docker images | grep -E "(original|optimized)"
```

### 4. Security Vulnerabilities

**Problem:** High number of security vulnerabilities

**Diagnosis:**
```bash
# Run comprehensive security scan
docker-optimizer --dockerfile Dockerfile --security-scan --format json > security-report.json

# Review critical vulnerabilities
jq '.vulnerabilities.cve_details[] | select(.severity == "CRITICAL")' security-report.json
```

**Solutions:**
```bash
# Update base image
sed -i 's/ubuntu:20.04/ubuntu:22.04-slim/' Dockerfile

# Use more secure base image
sed -i 's/FROM ubuntu:22.04/FROM gcr.io\/distroless\/python3/' Dockerfile

# Remove unnecessary packages
docker-optimizer --dockerfile Dockerfile --security-scan
```

### 5. Build Performance Issues

**Problem:** Slow build times

**Diagnosis:**
```bash
# Enable BuildKit for better performance
export DOCKER_BUILDKIT=1

# Analyze build performance
docker build --progress=plain -f Dockerfile .
```

**Solutions:**
```bash
# Use performance optimization
docker-optimizer --dockerfile Dockerfile --performance

# Optimize layer caching
# Move frequently changing content to end of Dockerfile

# Use .dockerignore to reduce build context
echo -e ".git\nnode_modules\n*.log" > .dockerignore

# Consider parallel builds for multiple services
docker-optimizer \
  --batch service1/Dockerfile \
  --batch service2/Dockerfile \
  --performance
```

### 6. Runtime Issues

**Problem:** Application fails to start in optimized container

**Diagnosis:**
```bash
# Compare working vs optimized Dockerfile
docker run -it original-image /bin/sh
docker run -it optimized-image /bin/sh

# Check for missing dependencies
docker-optimizer --dockerfile Dockerfile --analysis-only --verbose
```

**Common Solutions:**
- Ensure all runtime dependencies are included
- Check file permissions and ownership
- Verify environment variables are set correctly
- Ensure working directory exists and is accessible

---

## Summary

Following these best practices will help you create:

- **Secure containers** with minimal attack surface
- **Efficient images** with optimal size and performance
- **Maintainable Dockerfiles** following industry standards
- **Reliable builds** with proper CI/CD integration

Use the Docker Optimizer Agent to automate these best practices:

```bash
# Complete optimization with all best practices
docker-optimizer \
  --dockerfile Dockerfile \
  --multistage \
  --security-scan \
  --performance \
  --performance-report \
  --format json \
  --output Dockerfile.optimized
```

For more specific examples and use cases, see the [examples documentation](examples/README.md) and [API documentation](api/README.md).