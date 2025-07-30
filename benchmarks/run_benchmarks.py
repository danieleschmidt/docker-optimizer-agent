#!/usr/bin/env python3
"""
Performance benchmark runner for Docker Optimizer Agent.
Measures optimization performance across different Dockerfile types and sizes.
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from docker_optimizer import DockerfileOptimizer


def load_benchmark_dockerfiles() -> Dict[str, str]:
    """Load all benchmark Dockerfiles."""
    benchmark_dir = Path(__file__).parent / "benchmark_dockerfiles"
    dockerfiles = {}
    
    for dockerfile_path in benchmark_dir.glob("*.dockerfile"):
        with open(dockerfile_path, 'r') as f:
            dockerfiles[dockerfile_path.stem] = f.read()
    
    return dockerfiles


def benchmark_optimization(dockerfile_content: str, name: str, iterations: int = 5) -> Dict[str, Any]:
    """Benchmark optimization performance for a single Dockerfile."""
    optimizer = DockerfileOptimizer()
    times = []
    memory_usage = []
    
    print(f"Benchmarking {name}...")
    
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Run optimization
        result = optimizer.optimize_dockerfile(dockerfile_content)
        
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        times.append(execution_time)
        
        print(f"  Iteration {i+1}: {execution_time:.4f}s")
    
    return {
        "name": name,
        "iterations": iterations,
        "min_time": min(times),
        "max_time": max(times),
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
        "dockerfile_lines": len(dockerfile_content.splitlines()),
        "dockerfile_size_bytes": len(dockerfile_content.encode('utf-8'))
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and collect results."""
    dockerfiles = load_benchmark_dockerfiles()
    
    if not dockerfiles:
        print("No benchmark Dockerfiles found!")
        return {}
    
    results = {
        "timestamp": time.isoformat(time.gmtime()),
        "benchmarks": []
    }
    
    for name, content in dockerfiles.items():
        benchmark_result = benchmark_optimization(content, name)
        results["benchmarks"].append(benchmark_result)
    
    return results


def save_results(results: Dict[str, Any]) -> None:
    """Save benchmark results to JSON file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_file = results_dir / f"benchmark-{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Update latest results
    latest_file = results_dir / "latest.json"
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print benchmark summary to console."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for benchmark in results["benchmarks"]:
        print(f"\n{benchmark['name'].upper()}")
        print(f"  Lines: {benchmark['dockerfile_lines']}")
        print(f"  Size: {benchmark['dockerfile_size_bytes']} bytes")
        print(f"  Mean time: {benchmark['mean_time']:.4f}s")
        print(f"  Median time: {benchmark['median_time']:.4f}s")
        print(f"  Min time: {benchmark['min_time']:.4f}s")
        print(f"  Max time: {benchmark['max_time']:.4f}s")
        print(f"  Std dev: {benchmark['std_dev']:.4f}s")


if __name__ == "__main__":
    print("Starting Docker Optimizer Agent benchmarks...")
    
    results = run_benchmarks()
    
    if results:
        save_results(results)
        print_summary(results)
    else:
        print("No benchmarks were run.")
        sys.exit(1)