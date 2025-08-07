#!/usr/bin/env python3
"""
Simple performance test for DockerfileSentimentAnalyzer
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from docker_optimizer.sentiment_analyzer import DockerfileSentimentAnalyzer

def main():
    print("⚡ Simple Performance Validation")
    print("=" * 40)
    
    analyzer = DockerfileSentimentAnalyzer(enable_caching=True, enable_metrics=True)
    
    # Test basic performance
    test_messages = [
        "Excellent Docker security implementation!",
        "Critical vulnerabilities detected in base image",
        "Standard package installation procedures used",
        "Performance optimization opportunities available",
        "Great use of multi-stage builds for efficiency"
    ] * 10  # 50 messages total
    
    start_time = time.time()
    for msg in test_messages:
        analyzer.analyze_sentiment(msg)
    processing_time = (time.time() - start_time) * 1000
    
    # Get performance report
    health = analyzer.get_health_status()
    perf_report = analyzer.get_performance_report()
    cache_stats = analyzer.cache.get_stats()
    
    print(f"📊 Results:")
    print(f"  Messages processed: {len(test_messages)}")
    print(f"  Total time: {processing_time:.2f}ms")
    print(f"  Average per message: {processing_time/len(test_messages):.2f}ms")
    print(f"  Throughput: {len(test_messages)/(processing_time/1000):.0f} msg/sec")
    print(f"  Performance grade: {perf_report['performance_grade']}")
    print(f"  Health status: {health['status']}")
    print(f"  Success rate: {health['success_rate_percent']:.1f}%")
    print(f"  Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
    
    # Quality gate validation
    quality_checks = {
        "Fast processing": processing_time/len(test_messages) < 50,  # <50ms per message
        "High success rate": health['success_rate_percent'] >= 95,
        "Good performance grade": perf_report['performance_grade'] in ['A', 'B'],
        "Healthy system": health['status'] == 'HEALTHY'
    }
    
    print(f"\n🛡️ Quality Gates:")
    all_passed = True
    for check, passed in quality_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL QUALITY GATES PASSED - PRODUCTION READY!")
    else:
        print(f"\n⚠️ Some quality gates failed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)