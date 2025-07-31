# Observability & Monitoring

## Overview

Comprehensive observability stack with metrics, tracing, and logging for production deployment.

## Monitoring Stack

### Components
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards  
- **OpenTelemetry**: Distributed tracing
- **AlertManager**: Alert routing and notification

### Quick Start
```bash
# Start monitoring stack
make monitor

# Access dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

## Metrics

### Application Metrics
- `docker_optimizer_requests_total`: Total optimization requests
- `docker_optimizer_duration_seconds`: Processing time distribution
- `docker_optimizer_cache_hits_total`: Cache performance
- `docker_optimizer_errors_total`: Error rate tracking

### System Metrics
- CPU utilization and memory usage
- I/O operations and network traffic
- Container resource consumption
- Python GC and memory allocation

## Tracing

### OpenTelemetry Integration
```python
from opentelemetry import trace
from docker_optimizer.logging_observability import tracer

@tracer.start_as_current_span("optimize_dockerfile")
def optimize_dockerfile(content: str):
    # Traced optimization logic
```

### Trace Collection
- **Jaeger**: Distributed tracing backend
- **Span Attributes**: Operation metadata
- **Baggage**: Cross-service context

## Logging

### Structured Logging
```python
import structlog

logger = structlog.get_logger()
logger.info("dockerfile_optimized", 
           size_reduction_mb=45.2,
           security_grade="A+",
           optimization_time_ms=250)
```

### Log Aggregation
- **JSON Format**: Machine-readable logs
- **Context Enrichment**: Trace correlation
- **Log Levels**: DEBUG, INFO, WARN, ERROR

## Alerting

### Alert Rules (`monitoring/prometheus-alerts.yml`)
- High error rate (>5% over 5 minutes)
- Slow response time (>2s p95)
- Low cache hit rate (<70%)
- Memory usage spike (>80% of limit)

### Notification Channels
- **Slack**: Development team alerts
- **PagerDuty**: Production incidents
- **Email**: Management notifications

## Dashboard Configuration

### Grafana Dashboards
- **Application Overview**: Key metrics and health
- **Performance Analysis**: Latency and throughput
- **Error Tracking**: Error rates and patterns
- **Resource Utilization**: System metrics

### Dashboard as Code
- JSON definitions in `monitoring/grafana/dashboards/`
- Automated provisioning via Docker Compose
- Version controlled dashboard updates

## Production Deployment

### Environment Variables
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://your-otel-endpoint"
export PROMETHEUS_ENDPOINT="https://your-prometheus"
export LOG_LEVEL="INFO"
```

### Health Checks
- `/health`: Application health endpoint
- `/metrics`: Prometheus metrics endpoint
- `/ready`: Readiness probe for Kubernetes

## Troubleshooting

### Common Issues
1. **High Memory Usage**: Enable memory profiling
2. **Slow Queries**: Check database query performance
3. **Network Timeouts**: Verify external service connectivity
4. **Cache Misses**: Analyze cache key patterns

### Debug Commands
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Memory profiling
python -m memory_profiler docker_optimizer/cli.py

# Performance tracing
python -m cProfile -o profile.stats docker_optimizer/cli.py
```