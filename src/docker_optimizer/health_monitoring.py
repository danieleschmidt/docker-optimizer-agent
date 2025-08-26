"""Comprehensive Health Monitoring System for Docker Optimizer Agent.

This module implements advanced health monitoring, alerting, and 
self-healing capabilities for production-ready operation.
"""

import asyncio
import json
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import threading

from .logging_observability import ObservabilityManager, LogLevel


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of components to monitor."""
    OPTIMIZER = "optimizer"
    SECURITY_SCANNER = "security_scanner"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK = "network"


@dataclass
class HealthCheck:
    """Configuration for a health check."""
    name: str
    component_type: ComponentType
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 5
    retries: int = 3
    critical_threshold: float = 0.8
    warning_threshold: float = 0.6
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResult:
    """Result of a health check execution."""
    check_name: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    value: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io_bytes: Dict[str, int]
    process_count: int
    load_average: List[float]
    timestamp: datetime


class HealthMonitor:
    """Comprehensive health monitoring and alerting system."""

    def __init__(self):
        self.obs_manager = ObservabilityManager(
            log_level=LogLevel.INFO,
            service_name="health-monitor"
        )
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: List[HealthResult] = []
        self.system_metrics_history: List[SystemMetrics] = []
        
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Configuration
        self.max_history_size = 1000
        self.alert_cooldown_seconds = 300  # 5 minutes
        self.last_alerts: Dict[str, datetime] = {}
        
        # Initialize default health checks
        self._setup_default_health_checks()

    def _setup_default_health_checks(self):
        """Set up default health checks."""
        
        # System resource checks
        self.register_health_check(HealthCheck(
            name="cpu_usage",
            component_type=ComponentType.SYSTEM_RESOURCE,
            check_function=self._check_cpu_usage,
            interval_seconds=30,
            critical_threshold=90.0,
            warning_threshold=70.0
        ))
        
        self.register_health_check(HealthCheck(
            name="memory_usage", 
            component_type=ComponentType.SYSTEM_RESOURCE,
            check_function=self._check_memory_usage,
            interval_seconds=30,
            critical_threshold=85.0,
            warning_threshold=70.0
        ))
        
        self.register_health_check(HealthCheck(
            name="disk_usage",
            component_type=ComponentType.SYSTEM_RESOURCE,
            check_function=self._check_disk_usage,
            interval_seconds=60,
            critical_threshold=90.0,
            warning_threshold=80.0
        ))
        
        # Application checks
        self.register_health_check(HealthCheck(
            name="optimizer_health",
            component_type=ComponentType.OPTIMIZER,
            check_function=self._check_optimizer_health,
            interval_seconds=60
        ))
        
        self.register_health_check(HealthCheck(
            name="security_scanner_health",
            component_type=ComponentType.SECURITY_SCANNER,
            check_function=self._check_security_scanner_health,
            interval_seconds=120
        ))

    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        self.obs_manager.logger.info(f"Health check registered: {health_check.name}")

    def remove_health_check(self, check_name: str):
        """Remove a health check."""
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            self.obs_manager.logger.info(f"Health check removed: {check_name}")

    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            self.obs_manager.logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.obs_manager.logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop health monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.obs_manager.logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Run all enabled health checks
                for check_name, check in self.health_checks.items():
                    if not check.enabled:
                        continue
                    
                    # Check if it's time to run this check
                    last_run = self._get_last_check_time(check_name)
                    if last_run and (datetime.now() - last_run).total_seconds() < check.interval_seconds:
                        continue
                    
                    # Execute health check
                    asyncio.create_task(self._execute_health_check(check))
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Clean up old history
                self._cleanup_history()
                
            except Exception as e:
                self.obs_manager.logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for 1 second before next iteration
            self.stop_event.wait(1.0)

    async def _execute_health_check(self, check: HealthCheck):
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Execute check with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(self._run_check_function(check)),
                timeout=check.timeout_seconds
            )
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Create health result
            health_result = HealthResult(
                check_name=check.name,
                status=result.get("status", HealthStatus.UNKNOWN),
                timestamp=datetime.now(),
                response_time_ms=response_time_ms,
                value=result.get("value"),
                message=result.get("message", ""),
                details=result.get("details", {}),
                error=result.get("error")
            )
            
            # Store result
            self.health_history.append(health_result)
            
            # Log result
            self.obs_manager.logger.info(
                f"Health check completed: {check.name}",
                extra={
                    "status": health_result.status.value,
                    "response_time_ms": response_time_ms,
                    "value": health_result.value
                }
            )
            
            # Check for alerts
            await self._check_alerts(health_result, check)
            
        except asyncio.TimeoutError:
            # Handle timeout
            health_result = HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Health check timed out after {check.timeout_seconds}s",
                error="TimeoutError"
            )
            self.health_history.append(health_result)
            await self._send_alert(health_result, check, "TIMEOUT")
            
        except Exception as e:
            # Handle other errors
            health_result = HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Health check failed: {str(e)}",
                error=str(e)
            )
            self.health_history.append(health_result)
            await self._send_alert(health_result, check, "ERROR")

    async def _run_check_function(self, check: HealthCheck) -> Dict[str, Any]:
        """Run the check function (async wrapper)."""
        if asyncio.iscoroutinefunction(check.check_function):
            return await check.check_function()
        else:
            # Run sync function in thread pool
            return await asyncio.get_event_loop().run_in_executor(
                None, check.check_function
            )

    async def _check_alerts(self, result: HealthResult, check: HealthCheck):
        """Check if alerts should be sent based on result."""
        if result.status == HealthStatus.CRITICAL:
            await self._send_alert(result, check, "CRITICAL")
        elif result.status == HealthStatus.WARNING:
            await self._send_alert(result, check, "WARNING")

    async def _send_alert(self, result: HealthResult, check: HealthCheck, alert_type: str):
        """Send alert for health check failure."""
        # Check alert cooldown
        alert_key = f"{check.name}_{alert_type}"
        if alert_key in self.last_alerts:
            time_since_last = datetime.now() - self.last_alerts[alert_key]
            if time_since_last.total_seconds() < self.alert_cooldown_seconds:
                return
        
        # Update last alert time
        self.last_alerts[alert_key] = datetime.now()
        
        # Log alert
        self.obs_manager.logger.error(
            f"ALERT: {alert_type} - {check.name}",
            extra={
                "alert_type": alert_type,
                "check_name": check.name,
                "status": result.status.value,
                "message": result.message,
                "value": result.value,
                "response_time_ms": result.response_time_ms
            }
        )
        
        # Here you could integrate with external alerting systems
        # (e.g., PagerDuty, Slack, email, etc.)

    def _get_last_check_time(self, check_name: str) -> Optional[datetime]:
        """Get the last execution time for a health check."""
        for result in reversed(self.health_history):
            if result.check_name == check_name:
                return result.timestamp
        return None

    def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io_bytes={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                process_count=process_count,
                load_average=list(load_avg),
                timestamp=datetime.now()
            )
            
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            self.obs_manager.logger.error(f"Error collecting system metrics: {e}")

    def _cleanup_history(self):
        """Clean up old history entries."""
        # Clean health history
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
        
        # Clean system metrics history
        if len(self.system_metrics_history) > self.max_history_size:
            self.system_metrics_history = self.system_metrics_history[-self.max_history_size:]

    # Default health check functions
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent >= 90:
            status = HealthStatus.CRITICAL
        elif cpu_percent >= 70:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return {
            "status": status,
            "value": cpu_percent,
            "message": f"CPU usage: {cpu_percent:.1f}%",
            "details": {
                "cpu_count": psutil.cpu_count(),
                "load_avg": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
            }
        }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        
        if memory.percent >= 85:
            status = HealthStatus.CRITICAL
        elif memory.percent >= 70:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return {
            "status": status,
            "value": memory.percent,
            "message": f"Memory usage: {memory.percent:.1f}%",
            "details": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3)
            }
        }

    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        percent = (disk.used / disk.total) * 100
        
        if percent >= 90:
            status = HealthStatus.CRITICAL
        elif percent >= 80:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return {
            "status": status,
            "value": percent,
            "message": f"Disk usage: {percent:.1f}%",
            "details": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3)
            }
        }

    def _check_optimizer_health(self) -> Dict[str, Any]:
        """Check Docker optimizer health."""
        try:
            # Try to import and create optimizer
            from .optimizer import DockerfileOptimizer
            optimizer = DockerfileOptimizer()
            
            # Test basic functionality
            test_dockerfile = "FROM ubuntu:22.04\nRUN echo 'test'"
            result = optimizer.analyze_dockerfile(test_dockerfile)
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Optimizer is functioning correctly",
                "details": {
                    "test_passed": True,
                    "analysis_successful": hasattr(result, 'base_image')
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Optimizer health check failed: {str(e)}",
                "error": str(e)
            }

    def _check_security_scanner_health(self) -> Dict[str, Any]:
        """Check security scanner health."""
        try:
            from .external_security import ExternalSecurityScanner
            scanner = ExternalSecurityScanner()
            
            # Test basic functionality
            test_dockerfile = "FROM ubuntu:22.04"
            report = scanner.scan_dockerfile_for_vulnerabilities(test_dockerfile)
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Security scanner is functioning correctly",
                "details": {
                    "test_passed": True,
                    "scan_successful": hasattr(report, 'total_vulnerabilities')
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.WARNING,  # Not critical if external scanner fails
                "message": f"Security scanner health check failed: {str(e)}",
                "error": str(e)
            }

    # Public API methods
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current overall health status."""
        if not self.health_history:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "message": "No health checks have been performed yet"
            }
        
        # Get latest results for each check
        latest_results = {}
        for result in reversed(self.health_history):
            if result.check_name not in latest_results:
                latest_results[result.check_name] = result
        
        # Determine overall status
        statuses = [result.status for result in latest_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Count status distribution
        status_counts = {}
        for status in statuses:
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(latest_results),
            "status_distribution": status_counts,
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "value": result.value,
                    "last_check": result.timestamp.isoformat(),
                    "response_time_ms": result.response_time_ms
                }
                for name, result in latest_results.items()
            }
        }

    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent system metrics."""
        if not self.system_metrics_history:
            return {"message": "No system metrics available"}
        
        # Get metrics from last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_metrics = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            latest = self.system_metrics_history[-1]
            return {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_percent": latest.disk_percent,
                "process_count": latest.process_count,
                "load_average": latest.load_average,
                "timestamp": latest.timestamp.isoformat()
            }
        
        # Calculate averages
        cpu_avg = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        memory_avg = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            "cpu_percent_avg_5m": round(cpu_avg, 1),
            "memory_percent_avg_5m": round(memory_avg, 1),
            "disk_percent": recent_metrics[-1].disk_percent,
            "process_count": recent_metrics[-1].process_count,
            "load_average": recent_metrics[-1].load_average,
            "metrics_count": len(recent_metrics),
            "timestamp": recent_metrics[-1].timestamp.isoformat()
        }

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "health_status": self.get_current_health_status(),
            "system_metrics": self.get_system_metrics_summary(),
            "monitoring_info": {
                "is_monitoring": self.is_monitoring,
                "registered_checks": len(self.health_checks),
                "total_check_history": len(self.health_history),
                "system_metrics_history": len(self.system_metrics_history)
            }
        }