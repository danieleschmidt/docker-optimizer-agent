#!/usr/bin/env python3
"""Health monitoring endpoint for Docker Optimizer Agent."""

import time
import psutil
from datetime import datetime
from typing import Dict, Any

class HealthMonitor:
    """Simple health monitoring for the Docker Optimizer Agent."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        uptime = time.time() - self.start_time
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": round(uptime, 2),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": round(memory.available / 1024 / 1024, 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2)
            }
        }
        
        # Health checks
        if cpu_percent > 90:
            health_status["status"] = "warning"
            health_status["warnings"] = ["High CPU usage"]
        
        if memory.percent > 85:
            health_status["status"] = "warning" 
            health_status.setdefault("warnings", []).append("High memory usage")
            
        if disk.percent > 90:
            health_status["status"] = "critical"
            health_status["errors"] = ["Low disk space"]
            
        return health_status

def main():
    """Simple health check endpoint."""
    monitor = HealthMonitor()
    health = monitor.get_health_status()
    
    print(f"Health Status: {health['status']}")
    print(f"Uptime: {health['uptime_seconds']}s")
    print(f"CPU: {health['system_metrics']['cpu_percent']}%")
    print(f"Memory: {health['system_metrics']['memory_percent']}%")
    print(f"Disk: {health['system_metrics']['disk_percent']}%")
    
    return 0 if health['status'] == 'healthy' else 1

if __name__ == "__main__":
    exit(main())