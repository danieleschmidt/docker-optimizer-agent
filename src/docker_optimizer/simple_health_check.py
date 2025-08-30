"""Simple health check system for Docker Optimizer Agent."""

import logging
import psutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status."""
    component: str
    status: str
    message: str
    timestamp: float
    details: Optional[Dict] = None


class SimpleHealthChecker:
    """Simple health monitoring for the optimizer."""
    
    def __init__(self):
        self.checks = {}
        self.history: List[HealthStatus] = []
    
    def add_check(self, name: str, check_func: callable, interval: int = 30):
        """Add a health check function."""
        self.checks[name] = {
            'func': check_func,
            'interval': interval,
            'last_run': 0
        }
    
    def run_checks(self) -> List[HealthStatus]:
        """Run all health checks."""
        results = []
        current_time = time.time()
        
        for name, check_info in self.checks.items():
            if current_time - check_info['last_run'] >= check_info['interval']:
                try:
                    status = check_info['func']()
                    results.append(HealthStatus(
                        component=name,
                        status="healthy" if status else "unhealthy",
                        message=f"{name} check {'passed' if status else 'failed'}",
                        timestamp=current_time
                    ))
                    check_info['last_run'] = current_time
                except Exception as e:
                    logger.error(f"Health check {name} failed: {e}")
                    results.append(HealthStatus(
                        component=name,
                        status="error",
                        message=str(e),
                        timestamp=current_time
                    ))
        
        self.history.extend(results)
        return results
    
    def get_system_health(self) -> HealthStatus:
        """Get overall system health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_info.percent,
                'disk_percent': disk_info.percent
            }
            
            # Simple health determination
            if cpu_percent > 90 or memory_info.percent > 90 or disk_info.percent > 90:
                status = "unhealthy"
                message = "System resources are under stress"
            else:
                status = "healthy"
                message = "System resources are normal"
            
            return HealthStatus(
                component="system",
                status=status,
                message=message,
                timestamp=time.time(),
                details=details
            )
        except Exception as e:
            return HealthStatus(
                component="system",
                status="error",
                message=f"System health check failed: {e}",
                timestamp=time.time()
            )
    
    def is_healthy(self) -> bool:
        """Check if the system is overall healthy."""
        recent_checks = [s for s in self.history if time.time() - s.timestamp < 300]  # Last 5 minutes
        if not recent_checks:
            return True  # No recent failures
        
        failed_checks = [s for s in recent_checks if s.status in ["unhealthy", "error"]]
        return len(failed_checks) < len(recent_checks) * 0.5  # Less than 50% failures