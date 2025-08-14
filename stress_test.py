#!/usr/bin/env python3
"""Stress test and resilience validation for Docker Optimizer Agent."""

import tempfile
import os
from pathlib import Path
from docker_optimizer.optimizer import DockerfileOptimizer

def run_stress_tests():
    """Run comprehensive stress tests."""
    optimizer = DockerfileOptimizer()
    results = []
    
    # Test cases
    test_cases = [
        ("empty", ""),
        ("whitespace_only", "   \n\t   \n  "),
        ("invalid_syntax", "INVALID_INSTRUCTION something"),
        ("very_long_line", "FROM alpine:3.18\n" + "RUN echo " + "x" * 10000),
        ("unicode_content", "FROM alpine:3.18\nRUN echo 'Hello 世界'"),
        ("malformed_from", "FORM ubuntu:latest"),
        ("missing_from", "RUN apt-get update"),
        ("multiple_from", "FROM ubuntu:latest\nFROM alpine:3.18"),
        ("complex_valid", """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]"""),
    ]
    
    for test_name, dockerfile_content in test_cases:
        try:
            result = optimizer.optimize_dockerfile(dockerfile_content)
            status = "✅ HANDLED"
            error = None
        except Exception as e:
            status = "⚠️ ERROR" 
            error = str(e)
        
        results.append({
            "test": test_name,
            "status": status,
            "error": error
        })
        
        print(f"{status} - {test_name}: {error or 'Processed successfully'}")
    
    return results

if __name__ == "__main__":
    print("🧪 Running Docker Optimizer Stress Tests")
    print("=" * 50)
    
    results = run_stress_tests()
    
    passed = sum(1 for r in results if "✅" in r["status"])
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests handled gracefully")
    
    if passed == total:
        print("✅ All stress tests passed - system is robust!")
    else:
        print("⚠️ Some edge cases need improvement")