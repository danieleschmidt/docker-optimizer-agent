"""Advanced Monitoring Integration with OpenTelemetry and Prometheus."""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .health_monitor import get_health_monitor
from .resilience_engine import ResilienceConfig, ResilienceEngine

logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """Definition for a custom metric."""
    name: str
    description: str
    unit: str
    metric_type: str  # counter, histogram, gauge
    labels: List[str]


class MonitoringIntegration:
    """Advanced monitoring integration with multiple backends."""

    def __init__(self):
        self.health_monitor = get_health_monitor()
        self.resilience_engine = ResilienceEngine(ResilienceConfig())
        self.metrics_enabled = True
        self.custom_metrics: Dict[str, MetricDefinition] = {}

        # Performance counters
        self.operation_counters = {
            "optimization_requests": 0,
            "security_scans": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        self.response_times = []
        self._lock = threading.Lock()

        self._initialize_default_metrics()

    def _initialize_default_metrics(self) -> None:
        """Initialize default system metrics."""
        default_metrics = [
            MetricDefinition(
                name="docker_optimizer_requests_total",
                description="Total number of optimization requests",
                unit="count",
                metric_type="counter",
                labels=["operation_type", "status"]
            ),
            MetricDefinition(
                name="docker_optimizer_response_time_seconds",
                description="Response time for optimization operations",
                unit="seconds",
                metric_type="histogram",
                labels=["operation_type"]
            ),
            MetricDefinition(
                name="docker_optimizer_errors_total",
                description="Total number of errors",
                unit="count",
                metric_type="counter",
                labels=["error_type", "severity"]
            ),
            MetricDefinition(
                name="docker_optimizer_health_status",
                description="Current health status of the system",
                unit="status",
                metric_type="gauge",
                labels=["component"]
            ),
            MetricDefinition(
                name="docker_optimizer_cache_operations_total",
                description="Cache hit/miss operations",
                unit="count",
                metric_type="counter",
                labels=["operation"]
            )
        ]

        for metric in default_metrics:
            self.custom_metrics[metric.name] = metric

    @contextmanager
    def track_operation(self, operation_type: str, **labels):
        """Context manager to track operation performance."""
        start_time = time.time()
        operation_status = "success"

        try:
            yield
        except Exception as e:
            operation_status = "error"
            self.record_error(str(e), operation_type)
            raise
        finally:
            duration = time.time() - start_time
            self.record_operation_time(operation_type, duration)
            self.increment_counter("optimization_requests", {
                "operation_type": operation_type,
                "status": operation_status,
                **labels
            })

    def record_operation_time(self, operation_type: str, duration: float) -> None:
        """Record operation execution time."""
        with self._lock:
            self.response_times.append({
                "operation": operation_type,
                "duration": duration,
                "timestamp": datetime.now()
            })

            # Keep only last 1000 measurements
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]

    def increment_counter(self, counter_name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a performance counter."""
        with self._lock:
            if counter_name in self.operation_counters:
                self.operation_counters[counter_name] += 1

    def record_error(self, error_message: str, context: str = "general") -> None:
        """Record error occurrence."""
        with self._lock:
            self.operation_counters["errors"] += 1
            logger.error(f"Error in {context}: {error_message}")

    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.increment_counter("cache_hits")

    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.increment_counter("cache_misses")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            recent_times = [r["duration"] for r in self.response_times[-100:]]

            metrics = {
                "counters": self.operation_counters.copy(),
                "response_times": {
                    "avg": sum(recent_times) / len(recent_times) if recent_times else 0,
                    "min": min(recent_times) if recent_times else 0,
                    "max": max(recent_times) if recent_times else 0,
                    "count": len(recent_times)
                },
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "error_rate": self._calculate_error_rate(),
                "health_status": self.health_monitor.get_health_report()
            }

            return metrics

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        hits = self.operation_counters.get("cache_hits", 0)
        misses = self.operation_counters.get("cache_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate error rate."""
        errors = self.operation_counters.get("errors", 0)
        requests = self.operation_counters.get("optimization_requests", 0)
        return (errors / requests * 100) if requests > 0 else 0.0

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.metrics_enabled:
            return ""

        metrics_output = []
        performance_data = self.get_performance_metrics()

        # Export counters
        for counter_name, value in performance_data["counters"].items():
            metric_name = f"docker_optimizer_{counter_name}_total"
            metrics_output.append(f"# HELP {metric_name} Total count of {counter_name}")
            metrics_output.append(f"# TYPE {metric_name} counter")
            metrics_output.append(f"{metric_name} {value}")

        # Export response time metrics
        response_times = performance_data["response_times"]
        metrics_output.append("# HELP docker_optimizer_response_time_seconds Response time statistics")
        metrics_output.append("# TYPE docker_optimizer_response_time_seconds summary")
        metrics_output.append(f"docker_optimizer_response_time_seconds_sum {response_times['avg'] * response_times['count']}")
        metrics_output.append(f"docker_optimizer_response_time_seconds_count {response_times['count']}")

        # Export cache hit rate
        metrics_output.append("# HELP docker_optimizer_cache_hit_rate_percent Cache hit rate percentage")
        metrics_output.append("# TYPE docker_optimizer_cache_hit_rate_percent gauge")
        metrics_output.append(f"docker_optimizer_cache_hit_rate_percent {performance_data['cache_hit_rate']}")

        # Export error rate
        metrics_output.append("# HELP docker_optimizer_error_rate_percent Error rate percentage")
        metrics_output.append("# TYPE docker_optimizer_error_rate_percent gauge")
        metrics_output.append(f"docker_optimizer_error_rate_percent {performance_data['error_rate']}")

        # Export health status
        health_status = performance_data["health_status"]
        if health_status.get("overall_status"):
            status_value = self._health_status_to_value(health_status["overall_status"])
            metrics_output.append("# HELP docker_optimizer_health_status System health status")
            metrics_output.append("# TYPE docker_optimizer_health_status gauge")
            metrics_output.append(f"docker_optimizer_health_status {status_value}")

        return "\n".join(metrics_output)

    def _health_status_to_value(self, status: str) -> int:
        """Convert health status to numeric value."""
        status_map = {
            "healthy": 1,
            "degraded": 2,
            "unhealthy": 3,
            "critical": 4
        }
        return status_map.get(status.lower(), 0)

    def generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on metrics."""
        alerts = []
        metrics = self.get_performance_metrics()

        # High error rate alert
        if metrics["error_rate"] > 5.0:
            alerts.append({
                "severity": "warning",
                "title": "High Error Rate",
                "description": f"Error rate is {metrics['error_rate']:.1f}%",
                "timestamp": datetime.now().isoformat()
            })

        # Slow response time alert
        avg_response = metrics["response_times"]["avg"]
        if avg_response > 5.0:
            alerts.append({
                "severity": "warning",
                "title": "Slow Response Times",
                "description": f"Average response time is {avg_response:.2f}s",
                "timestamp": datetime.now().isoformat()
            })

        # Low cache hit rate alert
        if metrics["cache_hit_rate"] < 50.0 and self.operation_counters.get("cache_hits", 0) + self.operation_counters.get("cache_misses", 0) > 10:
            alerts.append({
                "severity": "info",
                "title": "Low Cache Hit Rate",
                "description": f"Cache hit rate is {metrics['cache_hit_rate']:.1f}%",
                "timestamp": datetime.now().isoformat()
            })

        # Add health-based alerts
        health_status = metrics["health_status"]
        if health_status.get("overall_status") in ["unhealthy", "critical"]:
            alerts.append({
                "severity": "critical",
                "title": "System Health Critical",
                "description": f"System status: {health_status.get('overall_status')}",
                "timestamp": datetime.now().isoformat()
            })

        return alerts

    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data structure for monitoring dashboard."""
        metrics = self.get_performance_metrics()

        return {
            "overview": {
                "total_requests": metrics["counters"]["optimization_requests"],
                "error_rate": metrics["error_rate"],
                "avg_response_time": metrics["response_times"]["avg"],
                "cache_hit_rate": metrics["cache_hit_rate"],
                "health_status": metrics["health_status"].get("overall_status", "unknown")
            },
            "charts": {
                "response_times": [
                    {
                        "timestamp": r["timestamp"].isoformat(),
                        "value": r["duration"],
                        "operation": r["operation"]
                    }
                    for r in self.response_times[-50:]  # Last 50 measurements
                ],
                "counters": [
                    {"name": name, "value": value}
                    for name, value in metrics["counters"].items()
                ]
            },
            "alerts": self.generate_alerts(),
            "health_components": metrics["health_status"].get("components", {}),
            "timestamp": datetime.now().isoformat()
        }

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        with self._lock:
            self.operation_counters = dict.fromkeys(self.operation_counters, 0)
            self.response_times.clear()

    def enable_metrics(self) -> None:
        """Enable metrics collection."""
        self.metrics_enabled = True
        logger.info("Metrics collection enabled")

    def disable_metrics(self) -> None:
        """Disable metrics collection."""
        self.metrics_enabled = False
        logger.info("Metrics collection disabled")


# Global monitoring integration instance
_monitoring_integration: Optional[MonitoringIntegration] = None


def get_monitoring_integration() -> MonitoringIntegration:
    """Get global monitoring integration instance."""
    global _monitoring_integration
    if _monitoring_integration is None:
        _monitoring_integration = MonitoringIntegration()
    return _monitoring_integration


def track_operation(operation_type: str, **labels):
    """Decorator/context manager for tracking operations."""
    monitor = get_monitoring_integration()
    return monitor.track_operation(operation_type, **labels)


def record_cache_hit():
    """Record a cache hit."""
    monitor = get_monitoring_integration()
    monitor.record_cache_hit()


def record_cache_miss():
    """Record a cache miss."""
    monitor = get_monitoring_integration()
    monitor.record_cache_miss()


def get_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    monitor = get_monitoring_integration()
    return monitor.get_performance_metrics()


def export_prometheus() -> str:
    """Export metrics in Prometheus format."""
    monitor = get_monitoring_integration()
    return monitor.export_prometheus_metrics()
