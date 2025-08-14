#!/usr/bin/env python3
"""Performance benchmark and optimization validation."""

import time
import tempfile
import statistics
import concurrent.futures
from pathlib import Path
from docker_optimizer.optimizer import DockerfileOptimizer

def create_test_dockerfiles(count=10):
    """Create multiple test dockerfiles for benchmarking."""
    dockerfiles = []
    
    base_dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]"""
    
    variations = [
        "FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y python3",
        "FROM alpine:3.18\nRUN apk add --no-cache python3 py3-pip",
        "FROM node:18-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm ci",
        "FROM nginx:alpine\nCOPY nginx.conf /etc/nginx/nginx.conf",
        "FROM postgres:15-alpine\nENV POSTGRES_DB=myapp",
    ]
    
    for i in range(count):
        if i < len(variations):
            dockerfiles.append(variations[i])
        else:
            dockerfiles.append(base_dockerfile)
    
    return dockerfiles

def benchmark_single_optimization():
    """Benchmark single dockerfile optimization."""
    optimizer = DockerfileOptimizer()
    
    test_dockerfile = """FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3 python3-pip
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt
COPY . /app/
WORKDIR /app
CMD ["python3", "app.py"]"""
    
    times = []
    for i in range(10):
        start = time.time()
        result = optimizer.optimize_dockerfile(test_dockerfile)
        end = time.time()
        times.append(end - start)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0
    }

def benchmark_batch_optimization():
    """Benchmark batch optimization performance."""
    optimizer = DockerfileOptimizer()
    dockerfiles = create_test_dockerfiles(20)
    
    # Sequential processing
    start = time.time()
    sequential_results = []
    for dockerfile in dockerfiles:
        result = optimizer.optimize_dockerfile(dockerfile)
        sequential_results.append(result)
    sequential_time = time.time() - start
    
    # Parallel processing simulation
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        parallel_results = list(executor.map(optimizer.optimize_dockerfile, dockerfiles))
    parallel_time = time.time() - start
    
    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": sequential_time / parallel_time,
        "files_processed": len(dockerfiles),
        "sequential_rate": len(dockerfiles) / sequential_time,
        "parallel_rate": len(dockerfiles) / parallel_time
    }

def run_performance_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("🚀 Docker Optimizer Performance Benchmarks")
    print("=" * 50)
    
    # Single optimization benchmark
    print("📊 Single Optimization Benchmark:")
    single_stats = benchmark_single_optimization()
    print(f"  Mean time: {single_stats['mean']:.3f}s")
    print(f"  Median time: {single_stats['median']:.3f}s")
    print(f"  Min time: {single_stats['min']:.3f}s")
    print(f"  Max time: {single_stats['max']:.3f}s")
    print(f"  Std dev: {single_stats['std_dev']:.3f}s")
    
    # Batch optimization benchmark
    print(f"\n📈 Batch Optimization Benchmark:")
    batch_stats = benchmark_batch_optimization()
    print(f"  Files processed: {batch_stats['files_processed']}")
    print(f"  Sequential time: {batch_stats['sequential_time']:.3f}s")
    print(f"  Parallel time: {batch_stats['parallel_time']:.3f}s")
    print(f"  Speedup: {batch_stats['speedup']:.2f}x")
    print(f"  Sequential rate: {batch_stats['sequential_rate']:.1f} files/s")
    print(f"  Parallel rate: {batch_stats['parallel_rate']:.1f} files/s")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    if single_stats['mean'] < 0.1:
        print("  ✅ Excellent: Sub-100ms single optimization")
    elif single_stats['mean'] < 0.5:
        print("  ✅ Good: Sub-500ms single optimization")
    else:
        print("  ⚠️ Consider optimization: >500ms single optimization")
    
    if batch_stats['speedup'] > 2.0:
        print("  ✅ Excellent: >2x speedup with parallelization")
    elif batch_stats['speedup'] > 1.5:
        print("  ✅ Good: >1.5x speedup with parallelization")
    else:
        print("  ⚠️ Limited parallelization benefit")
    
    return {
        "single_optimization": single_stats,
        "batch_optimization": batch_stats
    }

if __name__ == "__main__":
    benchmarks = run_performance_benchmarks()