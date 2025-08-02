# Monitoring and Observability Guide

Complete monitoring setup for Docker Optimizer Agent with Prometheus, Grafana, and OpenTelemetry.

## Quick Start

```bash
# Start monitoring stack
make monitor

# Available services:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Docker          │────│ OpenTelemetry   │────│ Prometheus      │
│ Optimizer       │    │ Collector       │    │ (Metrics)       │
│ Agent           │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│ Jaeger          │──────────────┘
                        │ (Tracing)       │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │ Grafana         │
                        │ (Dashboards)    │
                        └─────────────────┘
```

## Metrics Collection

### Application Metrics

- **Optimization Metrics**: Success rate, processing time, size reduction
- **Security Metrics**: Vulnerability detection rate, security score distribution
- **Performance Metrics**: Memory usage, CPU utilization, cache hit rate
- **Error Metrics**: Error rate, error types, failure reasons

### Infrastructure Metrics

- **Container Metrics**: CPU, memory, disk, network
- **Docker Metrics**: Container lifecycle, image pulls, build times
- **System Metrics**: Host resources, filesystem usage

## Dashboard Configuration

Pre-configured Grafana dashboards are available in `monitoring/grafana/dashboards/`:

- **Docker Optimizer Overview**: High-level performance metrics
- **Security Analysis**: Security scanning and vulnerability trends
- **Performance Monitoring**: Detailed performance analysis
- **Error Analysis**: Error rates and troubleshooting

## Alerting Rules

Prometheus alerting rules in `monitoring/prometheus-alerts.yml`:

- **High Error Rate**: Alert when error rate exceeds 5%
- **Performance Degradation**: Alert when processing time increases 50%
- **Memory Usage**: Alert when memory usage exceeds 80%
- **Security Issues**: Alert on critical vulnerabilities detected

## Custom Metrics

Add custom metrics to your application:

```python
from prometheus_client import Counter, Histogram, Gauge

# Custom metrics
optimization_counter = Counter('optimizations_total', 'Total optimizations performed')
processing_time = Histogram('processing_seconds', 'Time spent processing Dockerfiles')
security_score = Gauge('security_score', 'Current security score')
```

## Health Checks

Comprehensive health check endpoints:

- `/health` - Basic health check
- `/health/detailed` - Detailed health information
- `/metrics` - Prometheus metrics endpoint
- `/ready` - Readiness probe for Kubernetes

For detailed configuration and troubleshooting, see the monitoring configuration files.