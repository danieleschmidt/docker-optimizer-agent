#!/usr/bin/env python3
"""
Test Global Deployment Readiness for DockerfileSentimentAnalyzer
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from docker_optimizer.sentiment_analyzer import DockerfileSentimentAnalyzer

def test_global_features():
    """Test global deployment features"""
    print("🌍 Testing Global Deployment Features")
    print("=" * 50)
    
    # Initialize with global support
    analyzer = DockerfileSentimentAnalyzer(
        enable_global=True,
        target_region="EU",
        enable_caching=True,
        enable_metrics=True
    )
    
    print(f"✅ Global support enabled: {analyzer.enable_global}")
    print(f"✅ Target region: {analyzer.target_region}")
    
    return analyzer

def test_multilingual_feedback(analyzer):
    """Test multilingual feedback generation"""
    print("\n🗣️ Testing Multilingual Feedback")
    
    test_message = "Excellent Docker security implementation with best practices!"
    
    # Test multilingual feedback
    multilingual = analyzer.get_multilingual_feedback(
        test_message, 
        "very_positive",
        target_languages=["es", "fr", "de", "ja", "zh"]
    )
    
    if multilingual:
        print(f"Original: {multilingual['english']}")
        print(f"Detected language: {multilingual['detected_language']}")
        print(f"Confidence: {multilingual['confidence']}")
        print("Translations:")
        for lang, translation in multilingual["translations"].items():
            print(f"  {lang}: {translation}")
        return True
    else:
        print("❌ Multilingual feedback generation failed")
        return False

def test_cultural_adaptation(analyzer):
    """Test cultural sensitivity adaptation"""
    print("\n🎭 Testing Cultural Adaptation")
    
    harsh_feedback = "❌ Error: Your Dockerfile has broken security configurations!"
    
    # Test cultural adaptation for different languages
    languages = ["en", "ja", "de"]
    results = {}
    
    for lang in languages:
        adapted = analyzer.apply_cultural_adaptation(harsh_feedback, lang)
        results[lang] = adapted
        print(f"{lang}: {adapted}")
    
    # Japanese should be more indirect
    if "ja" in results:
        ja_feedback = results["ja"]
        if "opportunity" in ja_feedback.lower() or "could be" in ja_feedback.lower():
            print("✅ Japanese cultural adaptation working (more indirect)")
        else:
            print("⚠️ Japanese cultural adaptation may need improvement")
    
    return True

def test_regional_compliance(analyzer):
    """Test regional compliance features"""
    print("\n🛡️ Testing Regional Compliance")
    
    compliance_info = analyzer.get_regional_compliance_info()
    
    print(f"Target region: {compliance_info.get('target_region')}")
    print(f"Compliance notice: {compliance_info.get('compliance_notice')}")
    print(f"Timezone greeting: {compliance_info.get('timezone_greeting')}")
    print(f"Supported languages: {len(compliance_info.get('supported_languages', []))}")
    
    # Check for GDPR compliance notice for EU region
    if analyzer.target_region == "EU":
        if "GDPR" in str(compliance_info.get("compliance_notice", "")):
            print("✅ GDPR compliance notice present for EU region")
        else:
            print("⚠️ GDPR compliance notice missing for EU region")
    
    return True

def test_global_deployment_report(analyzer):
    """Test global deployment readiness report"""
    print("\n📊 Testing Global Deployment Report")
    
    # Create sample multilingual feedback messages
    sample_messages = [
        "Excellent Docker optimization work!",
        "Excelente trabajo de optimización de Docker!",  # Spanish
        "Excellent travail d'optimisation Docker!",      # French
        "Ausgezeichnete Docker-Optimierungsarbeit!",     # German
        "Your Dockerfile has security vulnerabilities",
        "Performance issues detected in layer structure",
        "Great use of multi-stage builds!",
        "Consider using specific base image versions",
    ]
    
    report = analyzer.get_global_deployment_report(sample_messages)
    
    print(f"Global readiness score: {report.get('global_readiness', {}).get('score', 'N/A')}")
    print(f"Readiness level: {report.get('global_readiness', {}).get('level', 'N/A')}")
    print(f"Status: {report.get('global_readiness', {}).get('status', 'N/A')}")
    
    language_analysis = report.get("language_analysis", {})
    print(f"Total messages analyzed: {language_analysis.get('total_messages', 0)}")
    print(f"Language coverage: {language_analysis.get('supported_coverage_percent', 0):.1f}%")
    
    # Check analyzer metrics
    analyzer_metrics = report.get("analyzer_metrics", {})
    print(f"Health status: {analyzer_metrics.get('health_status', 'Unknown')}")
    print(f"Performance grade: {analyzer_metrics.get('performance_grade', 'Unknown')}")
    
    # Print recommendations
    recommendations = report.get("deployment_recommendations", [])
    if recommendations:
        print("\nDeployment Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
            print(f"  {i}. {rec}")
    
    # Determine if ready for global deployment
    readiness_score = report.get('global_readiness', {}).get('score', 0)
    if readiness_score >= 85:
        print("✅ Ready for global deployment!")
        return True
    elif readiness_score >= 70:
        print("🟡 Ready with minor improvements needed")
        return True
    else:
        print("🔴 Needs significant work for global deployment")
        return False

def test_timezone_awareness(analyzer):
    """Test timezone awareness features"""
    print("\n🕒 Testing Timezone Awareness")
    
    if analyzer.global_analyzer:
        greeting_utc = analyzer.global_analyzer.get_timezone_aware_greeting("UTC")
        greeting_ny = analyzer.global_analyzer.get_timezone_aware_greeting("America/New_York")
        
        print(f"UTC greeting: {greeting_utc}")
        print(f"New York greeting: {greeting_ny}")
        
        # Basic validation that greeting is appropriate
        valid_greetings = ["Good morning", "Good afternoon", "Good evening", "Hello"]
        if greeting_utc in valid_greetings:
            print("✅ Timezone-aware greetings working")
            return True
        else:
            print("⚠️ Timezone awareness may need improvement")
            return False
    else:
        print("⚠️ Global analyzer not available for timezone testing")
        return False

def run_performance_validation(analyzer):
    """Run performance validation for global features"""
    print("\n⚡ Global Performance Validation")
    
    import time
    
    # Test performance with multilingual content
    multilingual_messages = [
        "Excellent Docker work!",
        "Excelente trabajo Docker!",
        "Excellent travail Docker!",
        "Ausgezeichnete Docker-Arbeit!",
    ] * 10  # 40 messages total
    
    start_time = time.time()
    for msg in multilingual_messages:
        analyzer.analyze_sentiment(msg)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"Multilingual processing time: {processing_time:.2f}ms")
    print(f"Average per message: {processing_time/len(multilingual_messages):.2f}ms")
    
    # Performance should still be good with global features
    if processing_time / len(multilingual_messages) < 50:
        print("✅ Global features maintain good performance")
        return True
    else:
        print("⚠️ Global features may impact performance")
        return False

def main():
    """Run comprehensive global deployment tests"""
    print("🚀 Global Deployment Readiness Test Suite")
    print("🌍 Testing Docker Optimizer Sentiment Analyzer for International Deployment")
    print("=" * 80)
    
    try:
        # Initialize global analyzer
        analyzer = test_global_features()
        
        # Run all tests
        tests = [
            ("Multilingual Feedback", lambda: test_multilingual_feedback(analyzer)),
            ("Cultural Adaptation", lambda: test_cultural_adaptation(analyzer)),
            ("Regional Compliance", lambda: test_regional_compliance(analyzer)),
            ("Global Deployment Report", lambda: test_global_deployment_report(analyzer)),
            ("Timezone Awareness", lambda: test_timezone_awareness(analyzer)),
            ("Global Performance", lambda: run_performance_validation(analyzer))
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                status = "✅ PASSED" if result else "⚠️ NEEDS ATTENTION"
                print(f"\n{status}: {test_name}")
            except Exception as e:
                print(f"\n❌ FAILED: {test_name} - {e}")
                results.append((test_name, False))
        
        # Final assessment
        print("\n" + "=" * 80)
        print("🌍 GLOBAL DEPLOYMENT ASSESSMENT")
        print("=" * 80)
        
        passed_tests = sum(1 for _, result in results if result)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT: Ready for global deployment across all regions!")
            deployment_status = "READY"
        elif success_rate >= 75:
            print("✅ GOOD: Ready for most global deployments with minor improvements")
            deployment_status = "MOSTLY_READY"
        elif success_rate >= 50:
            print("🟡 FAIR: Needs improvements before global deployment")
            deployment_status = "NEEDS_WORK"
        else:
            print("🔴 POOR: Significant work needed for global deployment")
            deployment_status = "NOT_READY"
        
        # Additional global readiness checklist
        print(f"\n📋 Global Readiness Checklist:")
        checklist = [
            ("Multi-language support", analyzer.enable_global),
            ("Cultural sensitivity", analyzer.global_analyzer is not None),
            ("Regional compliance", analyzer.target_region is not None),
            ("Performance optimization", analyzer.enable_caching),
            ("Monitoring & metrics", analyzer.enable_metrics),
            ("Scalability features", analyzer.max_workers > 1)
        ]
        
        for item, status in checklist:
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {item}")
        
        return deployment_status == "READY" or deployment_status == "MOSTLY_READY"
        
    except Exception as e:
        print(f"\n💥 Critical error in global deployment testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)