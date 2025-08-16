"""AI Health Monitoring and Error Recovery System."""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import psutil


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_threads: int
    response_time_ms: float
    error_rate: float
    timestamp: float


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, float]
    timestamp: float


class AIHealthMonitor:
    """Advanced health monitoring system for AI optimization engine."""

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0,
        error_rate_threshold: float = 5.0,
        response_time_threshold: float = 5000.0
    ):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.error_rate_threshold = error_rate_threshold
        self.response_time_threshold = response_time_threshold

        self.logger = logging.getLogger(__name__)
        self.health_history: List[HealthMetrics] = []
        self.error_count = 0
        self.success_count = 0
        self.response_times: List[float] = []

        # Auto-recovery settings
        self.auto_recovery_enabled = True
        self.max_retry_attempts = 3
        self.recovery_actions: Dict[str, callable] = {}

    async def check_system_health(self) -> HealthCheck:
        """Perform comprehensive system health check."""
        start_time = time.time()

        try:
            # Get system metrics
            metrics = await self._collect_metrics()

            # Determine overall health status
            status = self._evaluate_health_status(metrics)

            # Create health check result
            health_check = HealthCheck(
                name="system_health",
                status=status,
                message=self._generate_status_message(status, metrics),
                metrics={
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "disk_usage": metrics.disk_usage,
                    "response_time": metrics.response_time_ms,
                    "error_rate": metrics.error_rate
                },
                timestamp=time.time()
            )

            # Store metrics history
            self.health_history.append(metrics)
            if len(self.health_history) > 100:  # Keep last 100 entries
                self.health_history.pop(0)

            # Trigger auto-recovery if needed
            if self.auto_recovery_enabled and status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                await self._trigger_auto_recovery(health_check)

            return health_check

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return HealthCheck(
                name="system_health",
                status=HealthStatus.FAILED,
                message=f"Health check error: {e}",
                metrics={},
                timestamp=time.time()
            )

    async def _collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive system metrics."""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Disk usage for root partition
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent

        # Active threads
        active_threads = len(psutil.pids())

        # Calculate average response time
        avg_response_time = (
            sum(self.response_times[-10:]) / len(self.response_times[-10:])
            if self.response_times else 0.0
        )

        # Calculate error rate
        total_requests = self.success_count + self.error_count
        error_rate = (
            (self.error_count / total_requests) * 100
            if total_requests > 0 else 0.0
        )

        return HealthMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_threads=active_threads,
            response_time_ms=avg_response_time,
            error_rate=error_rate,
            timestamp=time.time()
        )

    def _evaluate_health_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Evaluate overall health status based on metrics."""
        critical_conditions = [
            metrics.cpu_usage > self.cpu_threshold,
            metrics.memory_usage > self.memory_threshold,
            metrics.disk_usage > self.disk_threshold,
            metrics.error_rate > self.error_rate_threshold,
            metrics.response_time_ms > self.response_time_threshold
        ]

        warning_conditions = [
            metrics.cpu_usage > self.cpu_threshold * 0.8,
            metrics.memory_usage > self.memory_threshold * 0.8,
            metrics.disk_usage > self.disk_threshold * 0.8,
            metrics.error_rate > self.error_rate_threshold * 0.8,
            metrics.response_time_ms > self.response_time_threshold * 0.8
        ]

        if any(critical_conditions):
            return HealthStatus.CRITICAL
        elif any(warning_conditions):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _generate_status_message(self, status: HealthStatus, metrics: HealthMetrics) -> str:
        """Generate human-readable status message."""
        messages = {
            HealthStatus.HEALTHY: "System operating normally",
            HealthStatus.WARNING: f"System under moderate load - CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%",
            HealthStatus.CRITICAL: f"System under heavy load - CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%, Errors: {metrics.error_rate:.1f}%",
            HealthStatus.FAILED: "System health check failed"
        }
        return messages.get(status, "Unknown status")

    async def _trigger_auto_recovery(self, health_check: HealthCheck) -> None:
        """Trigger automatic recovery actions."""
        self.logger.warning(f"Triggering auto-recovery for: {health_check.message}")

        recovery_actions = [
            self._clear_memory_cache,
            self._restart_workers,
            self._reduce_concurrent_operations,
            self._enable_resource_limits
        ]

        for action in recovery_actions:
            try:
                await action(health_check)
                self.logger.info(f"Recovery action completed: {action.__name__}")

                # Re-check health after each action
                new_health = await self.check_system_health()
                if new_health.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]:
                    self.logger.info("System recovered successfully")
                    break

            except Exception as e:
                self.logger.error(f"Recovery action failed: {action.__name__}: {e}")
                continue

    async def _clear_memory_cache(self, health_check: HealthCheck) -> None:
        """Clear memory caches to free up resources."""
        import gc
        gc.collect()
        self.logger.info("Memory cache cleared")

    async def _restart_workers(self, health_check: HealthCheck) -> None:
        """Restart worker processes if available."""
        # This would restart actual worker processes in production
        self.logger.info("Worker restart simulated")

    async def _reduce_concurrent_operations(self, health_check: HealthCheck) -> None:
        """Reduce number of concurrent operations."""
        # This would reduce concurrency limits in production
        self.logger.info("Concurrent operations reduced")

    async def _enable_resource_limits(self, health_check: HealthCheck) -> None:
        """Enable stricter resource limits."""
        # This would apply resource limits in production
        self.logger.info("Resource limits enabled")

    def record_success(self, response_time_ms: float) -> None:
        """Record successful operation."""
        self.success_count += 1
        self.response_times.append(response_time_ms)
        if len(self.response_times) > 1000:  # Keep last 1000 measurements
            self.response_times.pop(0)

    def record_error(self) -> None:
        """Record failed operation."""
        self.error_count += 1

    def get_health_trends(self) -> Dict[str, List[float]]:
        """Get health metric trends over time."""
        if not self.health_history:
            return {}

        return {
            "cpu_usage": [m.cpu_usage for m in self.health_history],
            "memory_usage": [m.memory_usage for m in self.health_history],
            "disk_usage": [m.disk_usage for m in self.health_history],
            "response_time": [m.response_time_ms for m in self.health_history],
            "error_rate": [m.error_rate for m in self.health_history],
            "timestamps": [m.timestamp for m in self.health_history]
        }

    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary statistics."""
        total_requests = self.success_count + self.error_count

        if not self.response_times:
            avg_response_time = 0.0
            p95_response_time = 0.0
        else:
            sorted_times = sorted(self.response_times)
            avg_response_time = sum(sorted_times) / len(sorted_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_response_time = sorted_times[p95_index] if sorted_times else 0.0

        return {
            "total_requests": total_requests,
            "success_rate": (self.success_count / total_requests * 100) if total_requests > 0 else 100.0,
            "error_rate": (self.error_count / total_requests * 100) if total_requests > 0 else 0.0,
            "avg_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "uptime_checks": len(self.health_history)
        }

    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous health monitoring."""
        self.logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")

        while True:
            try:
                health_check = await self.check_system_health()

                if health_check.status != HealthStatus.HEALTHY:
                    self.logger.warning(f"Health check: {health_check.message}")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval_seconds)


# Export for use in other modules
__all__ = [
    "AIHealthMonitor",
    "HealthStatus",
    "HealthMetrics",
    "HealthCheck"
]
