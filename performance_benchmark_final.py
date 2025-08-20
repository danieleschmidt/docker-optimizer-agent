#!/usr/bin/env python3
"""Final performance benchmark validation for Docker Optimizer Agent."""

import time
import subprocess
import json
from typing import Dict, Any, List
import statistics

def run_optimization_benchmark() -> Dict[str, Any]:
    """Run comprehensive optimization benchmark."""
    
    # Test cases with different complexity levels
    test_cases = [
        ("simple", "test_simple.dockerfile"),
        ("complex", "Dockerfile"), 
        ("node", "node-example.dockerfile"),
    ]
    
    results = []
    
    print("🚀 Running Performance Benchmark")
    print("=" * 50)
    
    for test_name, dockerfile in test_cases:
        print(f"📊 Testing {test_name} optimization...")
        
        # Multiple runs for statistical significance
        run_times = []
        success_count = 0
        
        for run in range(5):
            start_time = time.time()
            
            try:
                result = subprocess.run([
                    "python", "-m", "docker_optimizer.cli",
                    "--dockerfile", dockerfile,
                    "--format", "json"
                ], 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd="/root/repo",
                env={
                    "VIRTUAL_ENV": "/root/repo/venv", 
                    "PATH": "/root/repo/venv/bin:/usr/bin:/bin"
                }
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                if result.returncode == 0:
                    success_count += 1
                    run_times.append(duration)
                    
                    # Parse JSON output to validate
                    try:
                        json.loads(result.stdout)
                    except json.JSONDecodeError:
                        print(f"  ⚠️  Run {run+1}: Invalid JSON output")
                        continue
                        
                    print(f"  ✅ Run {run+1}: {duration:.3f}s")
                else:
                    print(f"  ❌ Run {run+1}: Failed ({result.returncode})")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Run {run+1}: Timeout")
            except Exception as e:
                print(f"  💥 Run {run+1}: Error - {e}")
        
        # Calculate statistics
        if run_times:
            test_result = {
                "test_name": test_name,
                "dockerfile": dockerfile,
                "success_rate": success_count / 5 * 100,
                "avg_time": statistics.mean(run_times),
                "min_time": min(run_times),
                "max_time": max(run_times),
                "std_dev": statistics.stdev(run_times) if len(run_times) > 1 else 0,
                "total_runs": 5,
                "successful_runs": success_count
            }
            results.append(test_result)
            
            print(f"  📈 Avg: {test_result['avg_time']:.3f}s, "
                  f"Success: {test_result['success_rate']:.1f}%")
        else:
            print(f"  ❌ No successful runs for {test_name}")
    
    return {"benchmark_results": results, "timestamp": time.time()}

def validate_core_features() -> Dict[str, bool]:
    """Validate core features are working."""
    
    print("\n🔍 Validating Core Features")
    print("=" * 50)
    
    features = {}
    
    # Test 1: Basic optimization
    try:
        result = subprocess.run([
            "python", "-m", "docker_optimizer.cli",
            "--dockerfile", "test_simple.dockerfile",
            "--analysis-only"
        ], capture_output=True, text=True, timeout=15, cwd="/root/repo",
        env={"VIRTUAL_ENV": "/root/repo/venv", "PATH": "/root/repo/venv/bin:/usr/bin:/bin"})
        
        features["basic_optimization"] = result.returncode == 0 and "Security Issues Found" in result.stdout
        print(f"  ✅ Basic Optimization: {'PASS' if features['basic_optimization'] else 'FAIL'}")
        
    except Exception:
        features["basic_optimization"] = False
        print("  ❌ Basic Optimization: FAIL")
    
    # Test 2: JSON Output
    try:
        result = subprocess.run([
            "python", "-m", "docker_optimizer.cli",
            "--dockerfile", "test_simple.dockerfile",
            "--format", "json"
        ], capture_output=True, text=True, timeout=15, cwd="/root/repo",
        env={"VIRTUAL_ENV": "/root/repo/venv", "PATH": "/root/repo/venv/bin:/usr/bin:/bin"})
        
        json_valid = False
        if result.returncode == 0:
            try:
                json.loads(result.stdout)
                json_valid = True
            except json.JSONDecodeError:
                pass
                
        features["json_output"] = json_valid
        print(f"  ✅ JSON Output: {'PASS' if features['json_output'] else 'FAIL'}")
        
    except Exception:
        features["json_output"] = False
        print("  ❌ JSON Output: FAIL")
    
    # Test 3: Security Scanning
    try:
        result = subprocess.run([
            "python", "-m", "docker_optimizer.cli",
            "--dockerfile", "test_simple.dockerfile",
            "--security-scan"
        ], capture_output=True, text=True, timeout=15, cwd="/root/repo",
        env={"VIRTUAL_ENV": "/root/repo/venv", "PATH": "/root/repo/venv/bin:/usr/bin:/bin"})
        
        features["security_scanning"] = result.returncode == 0 and "Security Score" in result.stdout
        print(f"  ✅ Security Scanning: {'PASS' if features['security_scanning'] else 'FAIL'}")
        
    except Exception:
        features["security_scanning"] = False
        print("  ❌ Security Scanning: FAIL")
    
    # Test 4: Batch Processing
    try:
        result = subprocess.run([
            "python", "-m", "docker_optimizer.cli",
            "--batch", "test_simple.dockerfile",
            "--batch", "node-example.dockerfile"
        ], capture_output=True, text=True, timeout=20, cwd="/root/repo",
        env={"VIRTUAL_ENV": "/root/repo/venv", "PATH": "/root/repo/venv/bin:/usr/bin:/bin"})
        
        features["batch_processing"] = result.returncode == 0 and "Results for:" in result.stdout
        print(f"  ✅ Batch Processing: {'PASS' if features['batch_processing'] else 'FAIL'}")
        
    except Exception:
        features["batch_processing"] = False
        print("  ❌ Batch Processing: FAIL")
    
    return features

def main():
    """Run complete performance validation."""
    
    print("🏁 Docker Optimizer Agent - Final Performance Validation")
    print("=" * 65)
    
    # Run performance benchmark
    benchmark_results = run_optimization_benchmark()
    
    # Validate core features
    feature_validation = validate_core_features()
    
    # Summary
    print("\n📋 Final Summary")
    print("=" * 50)
    
    total_features = len(feature_validation)
    passing_features = sum(feature_validation.values())
    feature_success_rate = passing_features / total_features * 100
    
    print(f"Core Features: {passing_features}/{total_features} ({feature_success_rate:.1f}%)")
    
    if benchmark_results["benchmark_results"]:
        avg_success_rate = statistics.mean([r["success_rate"] for r in benchmark_results["benchmark_results"]])
        avg_performance = statistics.mean([r["avg_time"] for r in benchmark_results["benchmark_results"]])
        
        print(f"Performance Tests: {avg_success_rate:.1f}% success rate")
        print(f"Average Optimization Time: {avg_performance:.3f}s")
        
        # Save detailed results
        final_report = {
            "validation_timestamp": time.time(),
            "feature_validation": feature_validation,
            "performance_benchmark": benchmark_results,
            "summary": {
                "feature_success_rate": feature_success_rate,
                "performance_success_rate": avg_success_rate,
                "avg_optimization_time": avg_performance,
                "overall_status": "PASS" if feature_success_rate >= 75 and avg_success_rate >= 80 else "FAIL"
            }
        }
        
        with open("final_validation_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📄 Detailed report: final_validation_report.json")
        print(f"🎯 Overall Status: {final_report['summary']['overall_status']}")
        
        return 0 if final_report['summary']['overall_status'] == "PASS" else 1
    else:
        print("❌ No performance data available")
        return 1

if __name__ == "__main__":
    exit(main())