#!/usr/bin/env python3
"""Health check script for Docker Optimizer Agent."""

import sys
import json
from pathlib import Path
from datetime import datetime

def run_health_check():
    """Run comprehensive health check."""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {},
        "version": "0.1.0"
    }
    
    try:
        # Test core imports
        from docker_optimizer.optimizer import DockerfileOptimizer
        from docker_optimizer.models import OptimizationResult
        health_status["checks"]["core_imports"] = "✅ PASS"
        
        # Test basic optimization
        optimizer = DockerfileOptimizer()
        test_dockerfile = """FROM alpine:3.18
RUN apk add --no-cache python3
CMD ["python3", "-c", "print('Hello World')"]"""
        
        result = optimizer.optimize_dockerfile(test_dockerfile)
        health_status["checks"]["basic_optimization"] = "✅ PASS"
        
        # Test CLI availability  
        import docker_optimizer.cli
        health_status["checks"]["cli_module"] = "✅ PASS"
        
        # Test configuration loading
        from docker_optimizer.config import Config
        config = Config()
        health_status["checks"]["config_loading"] = "✅ PASS"
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        health_status["checks"]["error"] = f"❌ FAIL: {e}"
    
    return health_status

if __name__ == "__main__":
    health = run_health_check()
    print(json.dumps(health, indent=2))
    sys.exit(0 if health["status"] == "healthy" else 1)