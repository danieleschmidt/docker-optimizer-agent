# Multi-stage build for optimal size and security
FROM ubuntu:22.04-slim AS builder
RUN set -e && apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
    set -e && pip3 install -r requirements.txt && \
    set -e && groupadd -r appuser && useradd -r -g appuser appuser && \
    rm -rf /var/lib/apt/lists/*
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json
COPY --chown=appuser:appuser . /app  
WORKDIR /app
EXPOSE 8000

FROM ubuntu:22.04-slim AS runtime
LABEL memory="512m"
LABEL cpu="0.5"
LABEL scaling="auto"
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser
WORKDIR /app
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["python3", "app.py"]
# Load balancing readiness
EXPOSE 8080
ENV HEALTH_CHECK_PATH=/health
ENV GRACEFUL_SHUTDOWN_TIMEOUT=30
# Ready for horizontal scaling
# Monitoring and observability
ENV METRICS_ENABLED=true
ENV METRICS_PORT=9090
ENV LOG_LEVEL=INFO
ENV TRACING_ENABLED=true