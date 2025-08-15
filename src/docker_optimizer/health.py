"""Health check and monitoring endpoints for Docker Optimizer Agent."""

import json
import time
from typing import Dict, Any, List
from datetime import datetime, timezone
import psutil
import platform

from .config import Config
from .error_handling import (
    trivy_circuit_breaker,
    registry_circuit_breaker, 
    external_api_circuit_breaker,
    CircuitBreakerState
)


class HealthCheck:
    """Health check and system monitoring for Docker Optimizer Agent."""
    
    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        self._last_check = {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "version": self._get_version(),
            "checks": {
                "system": self._check_system_health(),
                "dependencies": self._check_dependencies(),
                "resources": self._check_resources(),
                "external_services": self._check_external_services(),
                "circuit_breakers": self._check_circuit_breakers()
            }
        }
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Check if service is ready to accept requests."""
        ready = True
        checks = {}
        
        # Check critical dependencies
        try:
            from dockerfile_parse import DockerfileParser
            checks["dockerfile_parser"] = {"status": "ok"}
        except ImportError:
            checks["dockerfile_parser"] = {"status": "error", "message": "Module not available"}
            ready = False
        
        # Check configuration
        if not self.config:
            checks["configuration"] = {"status": "error", "message": "Configuration not loaded"}
            ready = False
        else:
            checks["configuration"] = {"status": "ok"}
        
        return {
            "ready": ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks
        }
    
    def get_liveness_status(self) -> Dict[str, Any]:
        """Check if service is alive and responsive."""
        return {
            "alive": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": psutil.Process().pid,
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus-style metrics."""
        process = psutil.Process()
        
        return {
            # Process metrics
            "process_cpu_percent": process.cpu_percent(),
            "process_memory_rss": process.memory_info().rss,
            "process_memory_vms": process.memory_info().vms,
            "process_num_threads": process.num_threads(),
            "process_num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0,
            "process_uptime_seconds": time.time() - self.start_time,
            
            # System metrics
            "system_cpu_percent": psutil.cpu_percent(interval=1),
            "system_memory_total": psutil.virtual_memory().total,
            "system_memory_available": psutil.virtual_memory().available,
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_disk_usage_percent": psutil.disk_usage('/').percent if psutil.disk_usage('/') else 0,
            
            # Application metrics (would be populated by the application)
            "docker_optimizer_requests_total": 0,
            "docker_optimizer_errors_total": 0,
            "docker_optimizer_processing_duration_seconds": 0,
            "docker_optimizer_security_scans_total": 0,
            "docker_optimizer_vulnerabilities_found_total": 0,
            "docker_optimizer_size_reduction_percent": 0,
        }
    
    def _get_version(self) -> str:
        """Get application version."""
        try:
            import importlib.metadata
            return importlib.metadata.version("docker-optimizer-agent")
        except Exception:
            return "unknown"
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system-level health."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "ok",
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": disk.percent,
                "cpu_count": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        dependencies = {
            "dockerfile_parse": False,
            "pydantic": False,
            "click": False,
            "requests": False,
            "yaml": False
        }
        
        for dep in dependencies:
            try:
                __import__(dep.replace('_', '-'))
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False
        
        all_ok = all(dependencies.values())
        
        return {
            "status": "ok" if all_ok else "error",
            "dependencies": dependencies,
            "missing": [dep for dep, available in dependencies.items() if not available]
        }
    
    def _check_resources(self) -> Dict[str, Any]:
        """Check resource usage and limits."""
        process = psutil.Process()
        memory = psutil.virtual_memory()
        
        # Define thresholds
        memory_threshold = 90  # percent
        cpu_threshold = 80     # percent
        disk_threshold = 90    # percent
        
        warnings = []
        
        if memory.percent > memory_threshold:
            warnings.append(f"High memory usage: {memory.percent:.1f}%")
        
        try:
            disk_usage = psutil.disk_usage('/').percent
            if disk_usage > disk_threshold:
                warnings.append(f"High disk usage: {disk_usage:.1f}%")
        except Exception:
            warnings.append("Could not check disk usage")
        
        return {
            "status": "warning" if warnings else "ok",
            "memory_usage_percent": memory.percent,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "warnings": warnings
        }
    
    def _check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity."""
        services = {}
        
        # Check if Trivy is available
        try:
            import subprocess
            result = subprocess.run(['trivy', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            services["trivy"] = {
                "status": "ok" if result.returncode == 0 else "error",
                "version": result.stdout.strip() if result.returncode == 0 else None
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            services["trivy"] = {
                "status": "unavailable",
                "message": str(e)
            }
        
        # Check Docker daemon connectivity
        try:
            import subprocess
            result = subprocess.run(['docker', 'version', '--format', '{{.Server.Version}}'],
                                  capture_output=True, text=True, timeout=5)
            services["docker"] = {
                "status": "ok" if result.returncode == 0 else "error",
                "version": result.stdout.strip() if result.returncode == 0 else None
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            services["docker"] = {
                "status": "unavailable",
                "message": str(e)
            }
        
        all_critical_ok = services.get("docker", {}).get("status") == "ok"
        
        return {
            "status": "ok" if all_critical_ok else "degraded",
            "services": services
        }
    
    def _check_circuit_breakers(self) -> Dict[str, Any]:
        """Check status of circuit breakers."""
        breakers = {
            "trivy_scanner": {
                "state": trivy_circuit_breaker.state.value,
                "failure_count": trivy_circuit_breaker.failure_count,
                "success_count": trivy_circuit_breaker.success_count,
                "timeout_seconds": trivy_circuit_breaker.timeout,
                "healthy": trivy_circuit_breaker.state == CircuitBreakerState.CLOSED
            },
            "registry_api": {
                "state": registry_circuit_breaker.state.value,
                "failure_count": registry_circuit_breaker.failure_count,
                "success_count": registry_circuit_breaker.success_count,
                "timeout_seconds": registry_circuit_breaker.timeout,
                "healthy": registry_circuit_breaker.state == CircuitBreakerState.CLOSED
            },
            "external_api": {
                "state": external_api_circuit_breaker.state.value,
                "failure_count": external_api_circuit_breaker.failure_count,
                "success_count": external_api_circuit_breaker.success_count,
                "timeout_seconds": external_api_circuit_breaker.timeout,
                "healthy": external_api_circuit_breaker.state == CircuitBreakerState.CLOSED
            }
        }
        
        all_healthy = all(breaker["healthy"] for breaker in breakers.values())
        
        return {
            "status": "ok" if all_healthy else "degraded",
            "breakers": breakers
        }


def format_prometheus_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics in Prometheus exposition format."""
    lines = []
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Add help text for key metrics
            if key.startswith('docker_optimizer_'):
                lines.append(f"# HELP {key} Docker Optimizer Agent metric")
                lines.append(f"# TYPE {key} gauge")
            
            lines.append(f"{key} {value}")
    
    return "\n".join(lines) + "\n"


def create_health_routes(app=None, health_check: HealthCheck = None):
    """Create health check routes for web frameworks."""
    if not health_check:
        from .config import load_config
        config = load_config()
        health_check = HealthCheck(config)
    
    routes = {
        "/health": lambda: health_check.get_health_status(),
        "/ready": lambda: health_check.get_readiness_status(),
        "/live": lambda: health_check.get_liveness_status(),
        "/metrics": lambda: format_prometheus_metrics(health_check.get_metrics())
    }
    
    return routes