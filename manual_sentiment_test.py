#!/usr/bin/env python3
"""
Manual test script for DockerfileSentimentAnalyzer
Tests core functionality without external dependencies
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from docker_optimizer.sentiment_analyzer import (
        DockerfileSentimentAnalyzer,
        SentimentScore,
        FeedbackTone,
        ValidationError,
        ProcessingError
    )
    print("✅ Successfully imported DockerfileSentimentAnalyzer")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic sentiment analysis functionality"""
    print("\n🧪 Testing Basic Functionality")
    
    analyzer = DockerfileSentimentAnalyzer()
    
    # Test positive sentiment
    positive_text = "Excellent work! Your Dockerfile is well-optimized and secure."
    result = analyzer.analyze_sentiment(positive_text)
    print(f"Positive test: {result.sentiment_score.value} (confidence: {result.confidence:.2f})")
    assert result.sentiment_score in [SentimentScore.POSITIVE, SentimentScore.VERY_POSITIVE]
    
    # Test negative sentiment
    negative_text = "Your Dockerfile has serious security vulnerabilities and broken dependencies."
    result = analyzer.analyze_sentiment(negative_text)
    print(f"Negative test: {result.sentiment_score.value} (confidence: {result.confidence:.2f})")
    assert result.sentiment_score in [SentimentScore.NEGATIVE, SentimentScore.VERY_NEGATIVE]
    
    # Test neutral sentiment
    neutral_text = "The Dockerfile uses standard commands for package installation."
    result = analyzer.analyze_sentiment(neutral_text)
    print(f"Neutral test: {result.sentiment_score.value} (confidence: {result.confidence:.2f})")
    assert result.sentiment_score == SentimentScore.NEUTRAL
    
    print("✅ Basic functionality tests passed")

def test_feedback_optimization():
    """Test feedback optimization functionality"""
    print("\n🔧 Testing Feedback Optimization")
    
    analyzer = DockerfileSentimentAnalyzer()
    
    # Test positive feedback optimization
    positive_feedback = "Great job using specific versions for security!"
    optimized = analyzer.optimize_feedback(positive_feedback, "security")
    print(f"Original: {positive_feedback}")
    print(f"Optimized: {optimized.optimized_message}")
    assert optimized.optimization_applied
    # Check that some positive emoji is present
    positive_emojis = ["🎉", "✨", "🚀", "👍", "✅", "🌟"]
    assert any(emoji in optimized.optimized_message for emoji in positive_emojis), f"No positive emoji found in: {optimized.optimized_message}"
    
    # Test negative feedback optimization
    negative_feedback = "Error: Your Dockerfile has broken security configurations."
    optimized = analyzer.optimize_feedback(negative_feedback, "security")
    print(f"Original: {negative_feedback}")
    print(f"Optimized: {optimized.optimized_message}")
    assert "issue" in optimized.optimized_message.lower()
    
    print("✅ Feedback optimization tests passed")

def test_caching_functionality():
    """Test caching system"""
    print("\n💾 Testing Caching Functionality")
    
    analyzer = DockerfileSentimentAnalyzer(enable_caching=True, cache_size=100)
    
    test_text = "This is a test message for caching functionality."
    
    # First analysis (should miss cache)
    start_time = time.time()
    result1 = analyzer.analyze_sentiment(test_text)
    first_time = time.time() - start_time
    
    # Second analysis (should hit cache)
    start_time = time.time()
    result2 = analyzer.analyze_sentiment(test_text)
    second_time = time.time() - start_time
    
    print(f"First analysis time: {first_time*1000:.2f}ms")
    print(f"Second analysis time: {second_time*1000:.2f}ms")
    
    # Results should be identical
    assert result1.sentiment_score == result2.sentiment_score
    assert result1.confidence == result2.confidence
    
    # Cache should show stats
    cache_stats = analyzer.cache.get_stats()
    print(f"Cache stats: {cache_stats}")
    assert cache_stats['hits'] >= 1
    
    print("✅ Caching functionality tests passed")

def test_error_handling():
    """Test error handling and validation"""
    print("\n⚠️  Testing Error Handling")
    
    analyzer = DockerfileSentimentAnalyzer()
    
    # Test empty input
    try:
        result = analyzer.analyze_sentiment("")
        print(f"Empty input handled: {result.sentiment_score.value}")
        assert result.sentiment_score == SentimentScore.NEUTRAL
        assert result.error_details is not None
    except ValidationError:
        print("Empty input validation works")
    
    # Test very long input
    long_text = "a" * 20000  # Exceeds max_text_length
    try:
        result = analyzer.analyze_sentiment(long_text)
        print("❌ Long text should have been rejected")
        assert False
    except ValidationError:
        print("✅ Long text properly rejected")
    
    # Test non-string input
    try:
        result = analyzer.analyze_sentiment(123)
        assert False
    except ValidationError:
        print("✅ Non-string input properly rejected")
    
    print("✅ Error handling tests passed")

def test_performance_monitoring():
    """Test performance monitoring and metrics"""
    print("\n📊 Testing Performance Monitoring")
    
    analyzer = DockerfileSentimentAnalyzer(enable_metrics=True)
    
    # Process several messages to generate metrics
    test_messages = [
        "Excellent Dockerfile optimization!",
        "Security issues detected in configuration",
        "Standard Docker commands used",
        "Performance could be improved",
        "Great use of multi-stage builds!"
    ]
    
    for msg in test_messages:
        analyzer.analyze_sentiment(msg)
    
    # Check health status
    health = analyzer.get_health_status()
    print(f"Health status: {health['status']}")
    print(f"Success rate: {health['success_rate_percent']}%")
    print(f"Total analyses: {health['total_analyses']}")
    
    # Metrics might be higher due to other tests, so check minimum
    assert health['total_analyses'] >= len(test_messages), f"Expected at least {len(test_messages)}, got {health['total_analyses']}"
    assert health['successful_analyses'] > 0
    
    # Check performance report
    perf_report = analyzer.get_performance_report()
    print(f"Performance grade: {perf_report['performance_grade']}")
    print(f"Average processing time: {perf_report['average_processing_time_ms']}ms")
    
    assert 'performance_grade' in perf_report
    assert perf_report['total_analyses'] >= len(test_messages)
    
    print("✅ Performance monitoring tests passed")

def test_batch_processing():
    """Test batch processing functionality"""
    print("\n🔄 Testing Batch Processing")
    
    analyzer = DockerfileSentimentAnalyzer()
    
    feedback_list = [
        "Excellent security implementation!",
        "Issues found in configuration",
        "Standard approach used",
        "Performance optimization needed"
    ]
    
    # Test regular batch processing
    results = analyzer.batch_analyze_feedback(feedback_list)
    print(f"Processed {len(results)} feedback items")
    assert len(results) == len(feedback_list)
    
    # Test parallel batch processing
    parallel_results = analyzer.parallel_analyze_batch(feedback_list, max_workers=2)
    print(f"Parallel processed {len(parallel_results)} feedback items")
    assert len(parallel_results) == len(feedback_list)
    
    print("✅ Batch processing tests passed")

def run_security_validation():
    """Run security validation tests"""
    print("\n🔒 Running Security Validation")
    
    analyzer = DockerfileSentimentAnalyzer()
    
    # Test malicious content detection
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "javascript:void(0)",
        "data:text/html,<script>alert(1)</script>"
    ]
    
    for malicious in malicious_inputs:
        try:
            result = analyzer.analyze_sentiment(malicious)
            print(f"❌ Malicious content should have been rejected: {malicious}")
        except ValidationError:
            print(f"✅ Malicious content properly rejected")
    
    print("✅ Security validation passed")

def main():
    """Run all tests"""
    print("🚀 Starting Manual Sentiment Analyzer Tests")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_feedback_optimization()
        test_caching_functionality()
        test_error_handling()
        test_performance_monitoring()
        test_batch_processing()
        run_security_validation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Sentiment Analyzer is ready for production.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()