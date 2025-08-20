#!/usr/bin/env python3
"""Load testing for Docker Optimizer Agent scalability."""

import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any
import subprocess
import threading
from datetime import datetime
import json

class LoadTestRunner:
    """Load test runner for Docker Optimizer Agent."""
    
    def __init__(self, concurrent_users: int = 10, test_duration: int = 60):
        """Initialize load test runner.
        
        Args:
            concurrent_users: Number of concurrent optimization requests
            test_duration: Test duration in seconds
        """
        self.concurrent_users = concurrent_users
        self.test_duration = test_duration
        self.results = []
        self.errors = []
        
    def simulate_optimization_request(self, user_id: int) -> Dict[str, Any]:
        """Simulate a single optimization request."""
        start_time = time.time()
        
        try:
            # Run docker-optimizer command
            result = subprocess.run([
                "python", "-m", "docker_optimizer.cli",
                "--dockerfile", "test_simple.dockerfile",
                "--analysis-only",
                "--format", "json"
            ], 
            capture_output=True, 
            text=True, 
            timeout=30,
            cwd="/root/repo",
            env={"VIRTUAL_ENV": "/root/repo/venv", "PATH": "/root/repo/venv/bin:/usr/bin:/bin"}
            )
            
            end_time = time.time()
            
            return {
                "user_id": user_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "success": result.returncode == 0,
                "stdout_length": len(result.stdout) if result.stdout else 0,
                "stderr": result.stderr if result.stderr else None
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "user_id": user_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "success": False,
                "error": str(e)
            }
    
    def run_load_test(self) -> Dict[str, Any]:
        """Run the load test with multiple concurrent users."""
        print(f"🚀 Starting load test with {self.concurrent_users} concurrent users")
        print(f"⏱️  Test duration: {self.test_duration} seconds")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = []
            user_id = 0
            
            while time.time() - start_time < self.test_duration:
                # Submit new requests
                for _ in range(self.concurrent_users):
                    if time.time() - start_time >= self.test_duration:
                        break
                    future = executor.submit(self.simulate_optimization_request, user_id)
                    futures.append(future)
                    user_id += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures, timeout=120):
                try:
                    result = future.result(timeout=10)
                    self.results.append(result)
                except Exception as e:
                    self.errors.append(str(e))
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze load test results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        successful_requests = [r for r in self.results if r["success"]]
        failed_requests = [r for r in self.results if not r["success"]]
        
        durations = [r["duration"] for r in successful_requests]
        
        analysis = {
            "test_summary": {
                "total_requests": len(self.results),
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / len(self.results) * 100,
                "test_duration": self.test_duration,
                "concurrent_users": self.concurrent_users
            },
            "performance_metrics": {
                "requests_per_second": len(self.results) / self.test_duration,
                "successful_rps": len(successful_requests) / self.test_duration,
                "avg_response_time": sum(durations) / len(durations) if durations else 0,
                "min_response_time": min(durations) if durations else 0,
                "max_response_time": max(durations) if durations else 0,
                "p95_response_time": self._percentile(durations, 95) if durations else 0,
                "p99_response_time": self._percentile(durations, 99) if durations else 0
            },
            "errors": self.errors[:10]  # First 10 errors
        }
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

def main():
    """Run load test and display results."""
    # Light load test for demonstration
    load_tester = LoadTestRunner(concurrent_users=5, test_duration=30)
    
    print("🧪 Docker Optimizer Agent Load Test")
    print("=" * 50)
    
    results = load_tester.run_load_test()
    
    print("\n📊 Load Test Results:")
    print("=" * 50)
    
    summary = results["test_summary"]
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Successful: {summary['successful_requests']}")
    print(f"Failed: {summary['failed_requests']}")
    print(f"Success Rate: {summary['success_rate']:.2f}%")
    
    metrics = results["performance_metrics"]
    print(f"\n⚡ Performance Metrics:")
    print(f"Requests/Second: {metrics['requests_per_second']:.2f}")
    print(f"Successful RPS: {metrics['successful_rps']:.2f}")
    print(f"Avg Response Time: {metrics['avg_response_time']:.3f}s")
    print(f"95th Percentile: {metrics['p95_response_time']:.3f}s")
    print(f"99th Percentile: {metrics['p99_response_time']:.3f}s")
    
    if results.get("errors"):
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results["errors"][:5]:
            print(f"  - {error}")
    
    # Save detailed results
    with open("load_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: load_test_results.json")
    
    return 0 if summary["success_rate"] >= 95 else 1

if __name__ == "__main__":
    exit(main())