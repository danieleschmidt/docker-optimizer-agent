#!/usr/bin/env python3
"""Demonstrate observability and monitoring features."""

import json
import time
from docker_optimizer.logging_observability import ObservabilityManager, LogLevel

def demo_observability():
    """Demonstrate comprehensive observability features."""
    print("📊 Docker Optimizer Observability Demo")
    print("=" * 45)
    
    # Initialize observability manager
    obs_manager = ObservabilityManager(
        log_level=LogLevel.DEBUG,
        service_name="docker-optimizer-demo"
    )
    
    print("✅ Observability manager initialized")
    
    # Test logging
    obs_manager.logger.info("Demo started", extra={"demo_type": "observability"})
    obs_manager.logger.debug("Debug message", extra={"component": "demo"})
    obs_manager.logger.warning("Warning message", extra={"severity": "low"})
    
    print("✅ Structured logging tested")
    
    # Test metrics
    obs_manager.record_metric("optimization_count", 1)
    obs_manager.record_metric("processing_time_ms", 150.5)
    
    # Test timing context
    with obs_manager.time_operation("dockerfile_optimization"):
        time.sleep(0.1)  # Simulate work
        from docker_optimizer.optimizer import DockerfileOptimizer
        optimizer = DockerfileOptimizer()
        result = optimizer.optimize_dockerfile("FROM alpine:3.18\nRUN echo 'test'")
    
    print("✅ Metrics and timing tested")
    
    # Test health checks
    health_status = obs_manager.get_health_status()
    print(f"✅ Health status: {health_status}")
    
    # Test system metrics
    system_metrics = obs_manager.get_system_metrics()
    print(f"✅ System metrics collected: CPU {system_metrics['cpu_percent']}%, Memory {system_metrics['memory_mb']:.1f}MB")
    
    return obs_manager

def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print(f"\n🚀 Performance Monitoring Demo")
    print("=" * 35)
    
    from docker_optimizer.optimizer import DockerfileOptimizer
    
    optimizer = DockerfileOptimizer()
    
    # Performance test with metrics
    test_cases = [
        "FROM alpine:3.18\nRUN echo 'test1'",
        "FROM ubuntu:22.04\nRUN apt-get update",
        "FROM python:3.11-slim\nWORKDIR /app",
        "FROM node:18-alpine\nCOPY package.json .",
        "FROM nginx:alpine\nEXPOSE 80"
    ]
    
    execution_times = []
    
    for i, dockerfile in enumerate(test_cases, 1):
        start_time = time.time()
        result = optimizer.optimize_dockerfile(dockerfile)
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        print(f"  Test {i}: {execution_time:.3f}s - {len(result.security_fixes)} security fixes")
    
    avg_time = sum(execution_times) / len(execution_times)
    print(f"\n📈 Performance Summary:")
    print(f"  Average execution time: {avg_time:.3f}s")
    print(f"  Total optimizations: {len(test_cases)}")
    print(f"  Throughput: {len(test_cases) / sum(execution_times):.1f} files/second")
    
    if avg_time < 0.1:
        print("  ✅ Excellent performance (<100ms average)")
    elif avg_time < 0.5:
        print("  ✅ Good performance (<500ms average)")
    else:
        print("  ⚠️ Consider performance optimization")

def main():
    """Main demo function."""
    # Observability demo
    obs_manager = demo_observability()
    
    # Performance monitoring demo  
    demo_performance_monitoring()
    
    print(f"\n🎯 Observability Summary:")
    print("  ✅ Structured logging with JSON output")
    print("  ✅ Metrics collection and timing")
    print("  ✅ Health status monitoring")
    print("  ✅ System resource tracking")
    print("  ✅ Performance benchmarking")
    print("  ✅ Production-ready observability stack")

if __name__ == "__main__":
    main()