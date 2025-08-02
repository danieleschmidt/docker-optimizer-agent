# Multi-stage Dockerfile for Docker Optimizer Agent
# This Dockerfile demonstrates the optimization techniques that the tool promotes

# =============================================================================
# Base stage - Common dependencies and security hardening
# =============================================================================
FROM python:3.13-slim AS base

# Security: Create non-root user
RUN groupadd -r optimizer && useradd -r -g optimizer optimizer

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl=7.* \
    git=1:2.* \
    ca-certificates=* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# Dependencies stage - Install Python dependencies
# =============================================================================
FROM base AS dependencies

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.* \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e . && \
    pip install -e .[dev,security] && \
    pip cache purge

# =============================================================================
# Development stage - Full development environment
# =============================================================================
FROM dependencies AS development

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io=20.* \
    make=4.* \
    vim=2:8.* \
    && rm -rf /var/lib/apt/lists/*

# Install external security tools
RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install in development mode
RUN pip install -e .[dev,security]

# Set up pre-commit hooks
RUN pre-commit install || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import docker_optimizer; print('OK')" || exit 1

# Default command for development
CMD ["python", "-m", "docker_optimizer.cli", "--help"]

# =============================================================================
# Testing stage - Optimized for CI/CD testing
# =============================================================================
FROM dependencies AS testing

# Install testing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io=20.* \
    && rm -rf /var/lib/apt/lists/*

# Install external tools for testing
RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

WORKDIR /app

# Copy source code and tests
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml Makefile ./

# Install in development mode for testing
RUN pip install -e .[dev,security]

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=docker_optimizer", "--cov-report=term-missing"]

# =============================================================================
# Security stage - Security scanning and analysis
# =============================================================================
FROM dependencies AS security

# Install security scanning tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install security tools
RUN pip install bandit safety semgrep

WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Security analysis command
CMD ["bandit", "-r", "src/", "-f", "json"]

# =============================================================================
# Production stage - Minimal runtime image
# =============================================================================
FROM base AS production

# Install only runtime dependencies
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
WORKDIR /app
COPY src/ ./src/
COPY pyproject.toml ./

# Install in production mode
RUN pip install --no-deps -e .

# Security: Switch to non-root user
USER optimizer

# Expose default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import docker_optimizer; print('OK')" || exit 1

# Production command
CMD ["python", "-m", "docker_optimizer.cli"]

# =============================================================================
# CLI stage - Lightweight CLI-only image
# =============================================================================
FROM production AS cli

# Remove unnecessary files for CLI usage
RUN rm -rf /app/tests /app/docs

# Set up CLI entrypoint
ENTRYPOINT ["python", "-m", "docker_optimizer.cli"]
CMD ["--help"]

# =============================================================================
# Metadata and labels
# =============================================================================
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL org.opencontainers.image.title="Docker Optimizer Agent"
LABEL org.opencontainers.image.description="Intelligent Dockerfile optimization with security scanning"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/terragonlabs/docker-optimizer-agent"
LABEL org.opencontainers.image.documentation="https://github.com/terragonlabs/docker-optimizer-agent/blob/main/README.md"

# Security labels
LABEL security.scan.enabled="true"
LABEL security.scan.vendor="Trivy"
LABEL security.policy.compliant="true"

# Build information
ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF