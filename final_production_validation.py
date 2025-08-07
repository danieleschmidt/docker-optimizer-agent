#!/usr/bin/env python3
"""
Final Production Validation for Docker Optimizer Sentiment Analyzer
Comprehensive validation before production deployment
"""

import sys
import os
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_comprehensive_validation():
    """Run all production validation tests"""
    print("🚀 FINAL PRODUCTION VALIDATION")
    print("Docker Optimizer Sentiment Analyzer")
    print("=" * 60)
    
    validation_results = {}
    
    # Test 1: Core Functionality
    print("\n1️⃣ Testing Core Functionality...")
    try:
        from docker_optimizer.sentiment_analyzer import DockerfileSentimentAnalyzer
        
        analyzer = DockerfileSentimentAnalyzer()
        result = analyzer.analyze_sentiment("Excellent Docker security implementation!")
        
        validation_results["core_functionality"] = {
            "status": "PASS",
            "sentiment_detected": result.sentiment_score.value,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms
        }
        print("✅ Core functionality working")
        
    except Exception as e:
        validation_results["core_functionality"] = {"status": "FAIL", "error": str(e)}
        print(f"❌ Core functionality failed: {e}")
    
    # Test 2: Performance Benchmarks
    print("\n2️⃣ Testing Performance Requirements...")
    try:
        analyzer = DockerfileSentimentAnalyzer(enable_caching=True, enable_metrics=True)
        
        # Process 100 messages and measure performance
        test_messages = ["Test message for performance validation"] * 100
        start_time = time.time()
        
        for msg in test_messages:
            analyzer.analyze_sentiment(msg)
        
        total_time_ms = (time.time() - start_time) * 1000
        avg_time_ms = total_time_ms / len(test_messages)
        throughput = len(test_messages) / (total_time_ms / 1000)
        
        # Performance requirements
        performance_pass = (
            avg_time_ms < 50 and  # <50ms per message
            throughput > 1000     # >1000 messages/sec
        )
        
        validation_results["performance"] = {
            "status": "PASS" if performance_pass else "FAIL",
            "avg_processing_time_ms": round(avg_time_ms, 2),
            "throughput_per_sec": round(throughput, 0),
            "total_messages": len(test_messages)
        }
        
        if performance_pass:
            print(f"✅ Performance requirements met: {avg_time_ms:.2f}ms avg, {throughput:.0f} msg/sec")
        else:
            print(f"❌ Performance requirements not met: {avg_time_ms:.2f}ms avg, {throughput:.0f} msg/sec")
            
    except Exception as e:
        validation_results["performance"] = {"status": "FAIL", "error": str(e)}
        print(f"❌ Performance test failed: {e}")
    
    # Test 3: Caching System
    print("\n3️⃣ Testing Caching System...")
    try:
        analyzer = DockerfileSentimentAnalyzer(enable_caching=True, cache_size=100)
        
        test_message = "Cache validation message"
        
        # First call (cache miss)
        start_time = time.time()
        result1 = analyzer.analyze_sentiment(test_message)
        first_call_time = (time.time() - start_time) * 1000
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = analyzer.analyze_sentiment(test_message)
        second_call_time = (time.time() - start_time) * 1000
        
        cache_stats = analyzer.cache.get_stats()
        cache_improvement = ((first_call_time - second_call_time) / first_call_time) * 100
        
        caching_pass = (
            cache_stats['hit_rate_percent'] > 0 and
            cache_improvement > 0 and
            result1.sentiment_score == result2.sentiment_score
        )
        
        validation_results["caching"] = {
            "status": "PASS" if caching_pass else "FAIL",
            "hit_rate_percent": cache_stats['hit_rate_percent'],
            "cache_improvement_percent": round(cache_improvement, 1),
            "results_identical": result1.sentiment_score == result2.sentiment_score
        }
        
        if caching_pass:
            print(f"✅ Caching system working: {cache_stats['hit_rate_percent']}% hit rate")
        else:
            print("❌ Caching system issues detected")
            
    except Exception as e:
        validation_results["caching"] = {"status": "FAIL", "error": str(e)}
        print(f"❌ Caching test failed: {e}")
    
    # Test 4: Error Handling & Resilience
    print("\n4️⃣ Testing Error Handling...")
    try:
        analyzer = DockerfileSentimentAnalyzer()
        
        error_scenarios = [
            ("empty_string", ""),
            ("very_long_string", "x" * 20000),
            ("non_string", 12345)
        ]
        
        error_handling_results = {}
        
        for scenario_name, test_input in error_scenarios:
            try:
                if scenario_name == "non_string":
                    result = analyzer.analyze_sentiment(test_input)
                    error_handling_results[scenario_name] = "UNEXPECTED_SUCCESS"
                else:
                    result = analyzer.analyze_sentiment(test_input)
                    error_handling_results[scenario_name] = "HANDLED_GRACEFULLY"
            except Exception:
                error_handling_results[scenario_name] = "PROPERLY_REJECTED"
        
        # Check that non-string input is properly rejected
        error_handling_pass = (
            error_handling_results.get("non_string") == "PROPERLY_REJECTED" and
            error_handling_results.get("very_long_string") == "PROPERLY_REJECTED"
        )
        
        validation_results["error_handling"] = {
            "status": "PASS" if error_handling_pass else "FAIL",
            "scenarios_tested": len(error_scenarios),
            "results": error_handling_results
        }
        
        if error_handling_pass:
            print("✅ Error handling working properly")
        else:
            print("❌ Error handling issues detected")
            
    except Exception as e:
        validation_results["error_handling"] = {"status": "FAIL", "error": str(e)}
        print(f"❌ Error handling test failed: {e}")
    
    # Test 5: Global/Multilingual Features
    print("\n5️⃣ Testing Global Features...")
    try:
        analyzer = DockerfileSentimentAnalyzer(enable_global=True, target_region="EU")
        
        # Test multilingual feedback
        multilingual_feedback = analyzer.get_multilingual_feedback(
            "Great Docker optimization work!",
            "positive",
            target_languages=["es", "fr", "de"]
        )
        
        # Test regional compliance
        compliance_info = analyzer.get_regional_compliance_info()
        
        global_pass = (
            multilingual_feedback is not None and
            len(multilingual_feedback.get("translations", {})) > 0 and
            compliance_info.get("target_region") == "EU" and
            "GDPR" in str(compliance_info.get("compliance_notice", ""))
        )
        
        validation_results["global_features"] = {
            "status": "PASS" if global_pass else "FAIL",
            "multilingual_enabled": multilingual_feedback is not None,
            "translations_available": len(multilingual_feedback.get("translations", {})) if multilingual_feedback else 0,
            "regional_compliance": compliance_info.get("target_region") == "EU"
        }
        
        if global_pass:
            print("✅ Global features working properly")
        else:
            print("❌ Global features issues detected")
            
    except Exception as e:
        validation_results["global_features"] = {"status": "FAIL", "error": str(e)}
        print(f"❌ Global features test failed: {e}")
    
    # Test 6: Health Monitoring
    print("\n6️⃣ Testing Health Monitoring...")
    try:
        analyzer = DockerfileSentimentAnalyzer(enable_metrics=True)
        
        # Generate some activity
        for i in range(10):
            analyzer.analyze_sentiment(f"Test message {i}")
        
        health_status = analyzer.get_health_status()
        performance_report = analyzer.get_performance_report()
        
        monitoring_pass = (
            health_status.get("status") in ["HEALTHY", "DEGRADED"] and
            health_status.get("success_rate_percent", 0) > 90 and
            performance_report.get("performance_grade") in ["A", "B", "C"]
        )
        
        validation_results["health_monitoring"] = {
            "status": "PASS" if monitoring_pass else "FAIL",
            "health_status": health_status.get("status"),
            "success_rate": health_status.get("success_rate_percent", 0),
            "performance_grade": performance_report.get("performance_grade")
        }
        
        if monitoring_pass:
            print(f"✅ Health monitoring working: {health_status.get('status')}")
        else:
            print("❌ Health monitoring issues detected")
            
    except Exception as e:
        validation_results["health_monitoring"] = {"status": "FAIL", "error": str(e)}
        print(f"❌ Health monitoring test failed: {e}")
    
    # Test 7: Integration Readiness
    print("\n7️⃣ Testing Integration Readiness...")
    try:
        # Test CLI integration
        cli_integration = True
        try:
            from docker_optimizer.cli import main
        except ImportError as e:
            cli_integration = False
            cli_error = str(e)
        
        # Test configuration validation
        analyzer = DockerfileSentimentAnalyzer()
        config_validation = analyzer.validate_configuration()
        
        integration_pass = (
            cli_integration and
            all(config_validation.values())
        )
        
        validation_results["integration"] = {
            "status": "PASS" if integration_pass else "FAIL",
            "cli_importable": cli_integration,
            "configuration_valid": all(config_validation.values()) if config_validation else False,
            "config_details": config_validation
        }
        
        if integration_pass:
            print("✅ Integration readiness confirmed")
        else:
            print("❌ Integration issues detected")
            if not cli_integration:
                print(f"  CLI integration issue: {cli_error if 'cli_error' in locals() else 'Unknown'}")
            
    except Exception as e:
        validation_results["integration"] = {"status": "FAIL", "error": str(e)}
        print(f"❌ Integration test failed: {e}")
    
    # Final Assessment
    print("\n" + "=" * 60)
    print("📊 PRODUCTION READINESS ASSESSMENT")
    print("=" * 60)
    
    total_tests = len(validation_results)
    passed_tests = sum(1 for result in validation_results.values() if result.get("status") == "PASS")
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Detailed results
    for test_name, result in validation_results.items():
        status_icon = "✅" if result.get("status") == "PASS" else "❌"
        print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result.get('status')}")
        if result.get("status") == "FAIL" and "error" in result:
            print(f"    Error: {result['error']}")
    
    # Production readiness determination
    if success_rate >= 95:
        readiness_level = "🎉 PRODUCTION READY"
        readiness_color = "GREEN"
        deployment_recommendation = "✅ APPROVED FOR PRODUCTION DEPLOYMENT"
    elif success_rate >= 85:
        readiness_level = "🟡 MOSTLY READY"
        readiness_color = "YELLOW"
        deployment_recommendation = "⚠️  APPROVED WITH MINOR FIXES NEEDED"
    elif success_rate >= 70:
        readiness_level = "🔴 NEEDS WORK"
        readiness_color = "RED"
        deployment_recommendation = "❌ NOT RECOMMENDED FOR PRODUCTION"
    else:
        readiness_level = "💥 CRITICAL ISSUES"
        readiness_color = "CRITICAL"
        deployment_recommendation = "🚫 DO NOT DEPLOY TO PRODUCTION"
    
    print(f"\n{readiness_level}")
    print(f"Status: {readiness_color}")
    print(f"Recommendation: {deployment_recommendation}")
    
    # Save detailed results
    with open("production_validation_report.json", "w") as f:
        json.dump({
            "validation_timestamp": time.time(),
            "success_rate_percent": success_rate,
            "readiness_level": readiness_level,
            "deployment_recommendation": deployment_recommendation,
            "test_results": validation_results
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: production_validation_report.json")
    
    return success_rate >= 85

if __name__ == "__main__":
    try:
        production_ready = run_comprehensive_validation()
        sys.exit(0 if production_ready else 1)
    except Exception as e:
        print(f"\n💥 Critical validation failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)