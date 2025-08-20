"""Advanced Health Monitoring and Self-Healing System."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

from .resilience_engine import ResilienceConfig, ResilienceEngine

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentType(str, Enum):
    """Types of system components to monitor."""
    DOCKER_DAEMON = "docker_daemon"
    EXTERNAL_SCANNER = "external_scanner"
    AI_ENGINE = "ai_engine"
    REGISTRY_API = "registry_api"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold: float
    unit: str
    status: HealthStatus
    timestamp: datetime
    description: str = ""


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component: ComponentType
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    components: Dict[ComponentType, ComponentHealth] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """Advanced health monitoring with self-healing capabilities."""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.resilience_engine = ResilienceEngine(ResilienceConfig())
        self.health_history: List[SystemHealth] = []
        self.recovery_handlers: Dict[ComponentType, Callable] = {}

        # Health thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 5000.0,  # ms
        }

        self.register_default_recovery_handlers()

    def register_default_recovery_handlers(self) -> None:
        """Register default recovery handlers for common issues."""
        self.recovery_handlers[ComponentType.DOCKER_DAEMON] = self._recover_docker_daemon
        self.recovery_handlers[ComponentType.MEMORY] = self._recover_memory_issues
        self.recovery_handlers[ComponentType.FILE_SYSTEM] = self._recover_filesystem_issues
        self.recovery_handlers[ComponentType.NETWORK] = self._recover_network_issues

    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                health = self.check_system_health()
                self._process_health_status(health)
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)

    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check."""
        components = {}

        # Check all component types
        for component_type in ComponentType:
            try:
                component_health = self._check_component_health(component_type)
                components[component_type] = component_health
            except Exception as e:
                logger.error(f"Failed to check {component_type.value}: {e}")
                components[component_type] = ComponentHealth(
                    component=component_type,
                    status=HealthStatus.UNHEALTHY,
                    error_count=1
                )

        # Determine overall status
        overall_status = self._calculate_overall_status(components)

        # Generate alerts and recommendations
        alerts = self._generate_alerts(components)
        recommendations = self._generate_recommendations(components)

        system_health = SystemHealth(
            overall_status=overall_status,
            components=components,
            alerts=alerts,
            recommendations=recommendations
        )

        # Store in history
        self.health_history.append(system_health)
        if len(self.health_history) > 100:  # Keep last 100 records
            self.health_history.pop(0)

        return system_health

    def _check_component_health(self, component: ComponentType) -> ComponentHealth:
        """Check health of a specific component."""
        metrics = []
        status = HealthStatus.HEALTHY
        error_count = 0
        recovery_actions = []

        try:
            if component == ComponentType.CPU:
                cpu_usage = psutil.cpu_percent(interval=1)
                metrics.append(HealthMetric(
                    name="cpu_usage",
                    value=cpu_usage,
                    threshold=self.thresholds["cpu_usage"],
                    unit="%",
                    status=HealthStatus.UNHEALTHY if cpu_usage > self.thresholds["cpu_usage"] else HealthStatus.HEALTHY,
                    timestamp=datetime.now(),
                    description="Current CPU usage percentage"
                ))
                if cpu_usage > self.thresholds["cpu_usage"]:
                    status = HealthStatus.UNHEALTHY
                    recovery_actions.append("Consider scaling up or optimizing CPU-intensive operations")

            elif component == ComponentType.MEMORY:
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                metrics.append(HealthMetric(
                    name="memory_usage",
                    value=memory_usage,
                    threshold=self.thresholds["memory_usage"],
                    unit="%",
                    status=HealthStatus.UNHEALTHY if memory_usage > self.thresholds["memory_usage"] else HealthStatus.HEALTHY,
                    timestamp=datetime.now(),
                    description="Current memory usage percentage"
                ))
                if memory_usage > self.thresholds["memory_usage"]:
                    status = HealthStatus.UNHEALTHY
                    recovery_actions.append("Consider clearing caches or scaling memory")

            elif component == ComponentType.FILE_SYSTEM:
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                metrics.append(HealthMetric(
                    name="disk_usage",
                    value=disk_usage,
                    threshold=self.thresholds["disk_usage"],
                    unit="%",
                    status=HealthStatus.UNHEALTHY if disk_usage > self.thresholds["disk_usage"] else HealthStatus.HEALTHY,
                    timestamp=datetime.now(),
                    description="Current disk usage percentage"
                ))
                if disk_usage > self.thresholds["disk_usage"]:
                    status = HealthStatus.CRITICAL
                    recovery_actions.append("Clean up temporary files and logs")

            elif component == ComponentType.DOCKER_DAEMON:
                # Check if Docker daemon is accessible
                try:
                    import docker
                    client = docker.from_env()
                    client.ping()
                    metrics.append(HealthMetric(
                        name="docker_accessible",
                        value=1.0,
                        threshold=1.0,
                        unit="boolean",
                        status=HealthStatus.HEALTHY,
                        timestamp=datetime.now(),
                        description="Docker daemon accessibility"
                    ))
                except Exception:
                    status = HealthStatus.CRITICAL
                    error_count = 1
                    recovery_actions.append("Start Docker daemon")
                    metrics.append(HealthMetric(
                        name="docker_accessible",
                        value=0.0,
                        threshold=1.0,
                        unit="boolean",
                        status=HealthStatus.CRITICAL,
                        timestamp=datetime.now(),
                        description="Docker daemon accessibility"
                    ))

            elif component == ComponentType.NETWORK:
                # Basic network connectivity check
                import socket
                try:
                    socket.create_connection(("8.8.8.8", 53), timeout=3)
                    metrics.append(HealthMetric(
                        name="network_connectivity",
                        value=1.0,
                        threshold=1.0,
                        unit="boolean",
                        status=HealthStatus.HEALTHY,
                        timestamp=datetime.now(),
                        description="Internet connectivity"
                    ))
                except Exception:
                    status = HealthStatus.DEGRADED
                    error_count = 1
                    recovery_actions.append("Check network configuration")
                    metrics.append(HealthMetric(
                        name="network_connectivity",
                        value=0.0,
                        threshold=1.0,
                        unit="boolean",
                        status=HealthStatus.DEGRADED,
                        timestamp=datetime.now(),
                        description="Internet connectivity"
                    ))

        except Exception as e:
            logger.error(f"Error checking {component.value}: {e}")
            status = HealthStatus.UNHEALTHY
            error_count = 1
            recovery_actions.append(f"Investigate {component.value} error: {str(e)}")

        return ComponentHealth(
            component=component,
            status=status,
            metrics=metrics,
            error_count=error_count,
            recovery_actions=recovery_actions
        )

    def _calculate_overall_status(self, components: Dict[ComponentType, ComponentHealth]) -> HealthStatus:
        """Calculate overall system health status."""
        critical_count = sum(1 for c in components.values() if c.status == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for c in components.values() if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components.values() if c.status == HealthStatus.DEGRADED)

        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif unhealthy_count > 1:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count == 1 or degraded_count > 2:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _generate_alerts(self, components: Dict[ComponentType, ComponentHealth]) -> List[str]:
        """Generate alerts based on component health."""
        alerts = []

        for component, health in components.items():
            if health.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                alerts.append(f"{component.value.upper()}: {health.status.value}")

                # Add specific metric alerts
                for metric in health.metrics:
                    if metric.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                        alerts.append(f"{metric.name}: {metric.value}{metric.unit} (threshold: {metric.threshold}{metric.unit})")

        return alerts

    def _generate_recommendations(self, components: Dict[ComponentType, ComponentHealth]) -> List[str]:
        """Generate recommendations for improving system health."""
        recommendations = []

        for component, health in components.items():
            if health.recovery_actions:
                recommendations.extend(health.recovery_actions)

        # Add general recommendations based on patterns
        unhealthy_components = [c.component for c in components.values() if c.status != HealthStatus.HEALTHY]
        if len(unhealthy_components) > 2:
            recommendations.append("Consider scaling infrastructure or reducing load")

        return recommendations

    def _process_health_status(self, health: SystemHealth) -> None:
        """Process health status and trigger recovery if needed."""
        if health.overall_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
            logger.warning(f"System health: {health.overall_status.value}")

            # Trigger recovery for unhealthy components
            for component, component_health in health.components.items():
                if component_health.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                    self._trigger_recovery(component, component_health)

    def _trigger_recovery(self, component: ComponentType, health: ComponentHealth) -> None:
        """Trigger recovery actions for unhealthy component."""
        if component in self.recovery_handlers:
            try:
                logger.info(f"Triggering recovery for {component.value}")
                self.recovery_handlers[component](health)
            except Exception as e:
                logger.error(f"Recovery failed for {component.value}: {e}")

    def _recover_docker_daemon(self, health: ComponentHealth) -> None:
        """Attempt to recover Docker daemon issues."""
        import subprocess
        try:
            # Try to start Docker daemon (this might need sudo)
            subprocess.run(["systemctl", "start", "docker"], check=True, capture_output=True)
            logger.info("Docker daemon started successfully")
        except subprocess.CalledProcessError:
            logger.warning("Failed to start Docker daemon - may need manual intervention")

    def _recover_memory_issues(self, health: ComponentHealth) -> None:
        """Attempt to recover from memory issues."""
        import gc

        # Force garbage collection
        gc.collect()

        # Clear internal caches if available
        try:
            from .performance import PerformanceOptimizer
            optimizer = PerformanceOptimizer()
            if hasattr(optimizer, 'clear_caches'):
                optimizer.clear_caches()
        except Exception:
            pass

        logger.info("Memory recovery actions completed")

    def _recover_filesystem_issues(self, health: ComponentHealth) -> None:
        """Attempt to recover from filesystem issues."""
        import shutil
        import tempfile

        # Clean up temporary files
        temp_dir = Path(tempfile.gettempdir())
        for temp_file in temp_dir.glob("docker_optimizer_*"):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
            except Exception:
                pass

        logger.info("Filesystem cleanup completed")

    def _recover_network_issues(self, health: ComponentHealth) -> None:
        """Attempt to recover from network issues."""
        # Reset network connections
        try:

            # Clear DNS cache (platform specific)
            import subprocess
            subprocess.run(["systemctl", "flush-dns"], capture_output=True, timeout=10)
        except Exception:
            pass

        logger.info("Network recovery actions completed")

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}

        latest_health = self.health_history[-1]

        return {
            "overall_status": latest_health.overall_status.value,
            "timestamp": latest_health.timestamp.isoformat(),
            "components": {
                component.value: {
                    "status": health.status.value,
                    "error_count": health.error_count,
                    "metrics": [
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "threshold": metric.threshold,
                            "unit": metric.unit,
                            "status": metric.status.value,
                            "description": metric.description
                        }
                        for metric in health.metrics
                    ],
                    "recovery_actions": health.recovery_actions
                }
                for component, health in latest_health.components.items()
            },
            "alerts": latest_health.alerts,
            "recommendations": latest_health.recommendations,
            "history_count": len(self.health_history)
        }

    def export_health_data(self, filepath: Path) -> None:
        """Export health monitoring data to file."""
        health_data = {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_duration": len(self.health_history) * self.check_interval,
            "health_history": [
                {
                    "timestamp": health.timestamp.isoformat(),
                    "overall_status": health.overall_status.value,
                    "component_count": len(health.components),
                    "alert_count": len(health.alerts),
                    "recommendation_count": len(health.recommendations)
                }
                for health in self.health_history
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(health_data, f, indent=2)

        logger.info(f"Health data exported to {filepath}")


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def start_health_monitoring() -> None:
    """Start global health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring()


def stop_health_monitoring() -> None:
    """Stop global health monitoring."""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()


def get_current_health() -> Dict[str, Any]:
    """Get current system health status."""
    monitor = get_health_monitor()
    return monitor.get_health_report()
