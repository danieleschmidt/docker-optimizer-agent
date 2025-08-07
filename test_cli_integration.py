#!/usr/bin/env python3
"""
Test CLI integration with sentiment analyzer
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cli_integration():
    """Test that CLI can import and use sentiment analyzer"""
    print("🔗 Testing CLI Integration with Sentiment Analyzer")
    
    try:
        from docker_optimizer.cli import main
        from docker_optimizer.sentiment_analyzer import DockerfileSentimentAnalyzer
        print("✅ CLI successfully imports sentiment analyzer")
        
        # Test sentiment analyzer initialization
        analyzer = DockerfileSentimentAnalyzer()
        result = analyzer.analyze_sentiment("Test integration message")
        print(f"✅ Sentiment analysis works: {result.sentiment_score.value}")
        
        return True
    except Exception as e:
        print(f"❌ CLI integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dockerfile_optimization():
    """Test a basic Dockerfile optimization with sentiment"""
    print("\n🐳 Testing Dockerfile Optimization Integration")
    
    try:
        from docker_optimizer.optimizer import DockerfileOptimizer
        from docker_optimizer.sentiment_analyzer import DockerfileSentimentAnalyzer
        
        # Create test Dockerfile content
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3
COPY . /app
WORKDIR /app
CMD python3 app.py
"""
        
        optimizer = DockerfileOptimizer()
        sentiment_analyzer = DockerfileSentimentAnalyzer()
        
        # Optimize Dockerfile
        result = optimizer.optimize_dockerfile(dockerfile_content)
        print(f"✅ Dockerfile optimization successful")
        
        # Test sentiment on optimization explanation
        optimized_explanation = sentiment_analyzer.optimize_feedback(
            result.explanation, "dockerfile_optimization"
        )
        print(f"✅ Sentiment-optimized explanation: {optimized_explanation.optimized_message[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Dockerfile optimization integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting CLI Integration Tests")
    print("=" * 50)
    
    success1 = test_cli_integration()
    success2 = test_dockerfile_optimization()
    
    if success1 and success2:
        print("\n" + "=" * 50)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests failed")
        sys.exit(1)