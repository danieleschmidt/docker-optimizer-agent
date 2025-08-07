#!/usr/bin/env python3
"""
Performance benchmark for DockerfileSentimentAnalyzer
"""

import sys
import os
import time
import random
import statistics

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from docker_optimizer.sentiment_analyzer import DockerfileSentimentAnalyzer

def generate_test_messages(count=100):
    """Generate test messages for benchmarking"""
    templates = [
        "Your Dockerfile has {} security vulnerabilities that need attention.",
        "Excellent work implementing {} in your container configuration!",
        "Consider using {} to optimize your Docker image size.",
        "Warning: {} detected in your base image selection.",
        "Great job following best practices with {} implementation.",
        "The current {} configuration may impact performance.",
        "Outstanding security measures implemented with {}!",
        "Performance issues found in {} layer optimization.",
        "Recommended approach: use {} for better efficiency.",
        "Critical vulnerability: {} requires immediate fixing."
    ]
    
    topics = [
        "multi-stage builds", "security scanning", "layer caching",
        "base image selection", "user permissions", "package management",
        "environment variables", "health checks", "signal handling",
        "dependency management", "build optimization", "runtime security"
    ]
    
    messages = []
    for i in range(count):
        template = random.choice(templates)
        topic = random.choice(topics)
        messages.append(template.format(topic))
    
    return messages

def benchmark_single_analysis(analyzer, messages, runs=3):
    """Benchmark single message analysis"""
    print("🔬 Single Analysis Benchmark")
    
    times = []
    for run in range(runs):
        start_time = time.time()
        for msg in messages[:20]:  # Test with first 20 messages
            analyzer.analyze_sentiment(msg)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = statistics.mean(times)
    throughput = (20 * runs) / (sum(times) / 1000)  # messages per second
    
    print(f"  Average time (20 messages): {avg_time:.2f}ms")
    print(f"  Throughput: {throughput:.1f} messages/second")
    
    return avg_time, throughput

def benchmark_caching_performance(analyzer, messages):
    """Benchmark caching performance"""
    print("\n💾 Caching Performance Benchmark")
    
    # Test without cache
    analyzer_no_cache = DockerfileSentimentAnalyzer(enable_caching=False)
    start_time = time.time()
    for msg in messages[:10]:
        analyzer_no_cache.analyze_sentiment(msg)
        analyzer_no_cache.analyze_sentiment(msg)  # Duplicate call
    no_cache_time = (time.time() - start_time) * 1000
    
    # Test with cache
    analyzer_with_cache = DockerfileSentimentAnalyzer(enable_caching=True)
    start_time = time.time()
    for msg in messages[:10]:
        analyzer_with_cache.analyze_sentiment(msg)
        analyzer_with_cache.analyze_sentiment(msg)  # Duplicate call (should hit cache)
    cache_time = (time.time() - start_time) * 1000
    
    cache_stats = analyzer_with_cache.cache.get_stats()
    improvement = ((no_cache_time - cache_time) / no_cache_time) * 100
    
    print(f"  Without cache: {no_cache_time:.2f}ms")
    print(f"  With cache: {cache_time:.2f}ms")
    print(f"  Cache hit rate: {cache_stats['hit_rate_percent']}%")
    print(f"  Performance improvement: {improvement:.1f}%")
    
    return improvement

def benchmark_parallel_processing(analyzer, messages):
    """Benchmark parallel vs sequential processing"""
    print("\n⚡ Parallel Processing Benchmark")
    
    test_messages = messages[:50]
    
    # Sequential processing
    start_time = time.time()
    seq_results = analyzer.batch_analyze_feedback(test_messages)
    seq_time = (time.time() - start_time) * 1000
    
    # Parallel processing
    start_time = time.time()
    par_results = analyzer.parallel_analyze_batch(test_messages, max_workers=4)
    par_time = (time.time() - start_time) * 1000
    
    speedup = seq_time / par_time if par_time > 0 else 1.0
    
    print(f"  Sequential time: {seq_time:.2f}ms")
    print(f"  Parallel time: {par_time:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Messages processed: {len(par_results)}")
    
    return speedup

def run_memory_efficiency_test(analyzer, messages):
    """Test memory efficiency"""
    print("\n🧠 Memory Efficiency Test")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process many messages
    for msg in messages:
        analyzer.analyze_sentiment(msg)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory
    
    print(f"  Initial memory: {initial_memory:.1f}MB")
    print(f"  Peak memory: {peak_memory:.1f}MB")
    print(f"  Memory increase: {memory_increase:.1f}MB")
    print(f"  Memory per message: {memory_increase / len(messages) * 1024:.2f}KB")
    
    # Test cache efficiency
    cache_stats = analyzer.cache.get_stats() if analyzer.cache else {}
    if cache_stats:
        print(f"  Cache entries: {cache_stats['cache_size']}")
        print(f"  Cache efficiency: {cache_stats['hit_rate_percent']:.1f}%")
    
    return memory_increase

def stress_test(analyzer, duration_seconds=10):
    """Perform stress testing"""
    print(f"\n🔥 Stress Test ({duration_seconds}s)")
    
    messages = generate_test_messages(20)  # Smaller set for rapid iteration
    
    start_time = time.time()
    count = 0
    errors = 0
    
    while time.time() - start_time < duration_seconds:
        try:
            msg = random.choice(messages)
            analyzer.analyze_sentiment(msg)
            count += 1
        except Exception as e:
            errors += 1
            print(f"    Error during stress test: {e}")
    
    elapsed = time.time() - start_time
    rate = count / elapsed
    
    print(f"  Messages processed: {count}")
    print(f"  Errors encountered: {errors}")
    print(f"  Processing rate: {rate:.1f} messages/second")
    print(f"  Error rate: {errors/count*100:.2f}%" if count > 0 else "  Error rate: N/A")
    
    return rate, errors

def main():
    """Run comprehensive performance benchmarks"""
    print("🚀 DockerfileSentimentAnalyzer Performance Benchmark")
    print("=" * 60)
    
    # Generate test data
    messages = generate_test_messages(100)
    print(f"Generated {len(messages)} test messages")
    
    # Initialize analyzer
    analyzer = DockerfileSentimentAnalyzer(
        enable_caching=True,
        enable_metrics=True,
        cache_size=500,
        max_workers=4
    )
    
    # Run benchmarks
    single_time, throughput = benchmark_single_analysis(analyzer, messages)
    cache_improvement = benchmark_caching_performance(analyzer, messages)
    parallel_speedup = benchmark_parallel_processing(analyzer, messages)
    memory_usage = run_memory_efficiency_test(analyzer, messages[:30])  # Limit for memory test
    stress_rate, stress_errors = stress_test(analyzer, duration_seconds=5)
    
    # Generate performance report
    perf_report = analyzer.get_performance_report()
    health_status = analyzer.get_health_status()
    
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"🎯 Performance Grade: {perf_report['performance_grade']}")
    print(f"🏥 Health Status: {health_status['status']}")
    print(f"📈 Success Rate: {health_status['success_rate_percent']:.1f}%")
    print(f"⚡ Average Processing: {perf_report['average_processing_time_ms']:.2f}ms")
    print(f"🔄 Throughput: {throughput:.1f} msg/sec")
    print(f"💾 Cache Improvement: {cache_improvement:.1f}%")
    print(f"⚡ Parallel Speedup: {parallel_speedup:.2f}x")
    print(f"🧠 Memory Usage: {memory_usage:.1f}MB")
    print(f"🔥 Stress Test Rate: {stress_rate:.1f} msg/sec")
    print(f"❌ Error Rate: {stress_errors} errors")
    
    # Performance classification
    if (perf_report['performance_grade'] in ['A', 'B'] and 
        health_status['success_rate_percent'] > 95 and
        cache_improvement > 20 and
        stress_errors == 0):
        print("\n🎉 PERFORMANCE: EXCELLENT - Production Ready!")
    elif (perf_report['performance_grade'] in ['B', 'C'] and 
          health_status['success_rate_percent'] > 90):
        print("\n✅ PERFORMANCE: GOOD - Suitable for production")
    else:
        print("\n⚠️ PERFORMANCE: NEEDS IMPROVEMENT")
    
    # Cleanup
    analyzer.clear_cache()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()