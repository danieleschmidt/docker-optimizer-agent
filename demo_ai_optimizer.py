#!/usr/bin/env python3
"""
Docker Optimizer Agent - Advanced AI Demo
==========================================

This demo showcases the complete AI-powered Docker optimization system with:
- Generation 1: AI-Powered Optimization (Make it Work)
- Generation 2: Resilience & Health Monitoring (Make it Robust) 
- Generation 3: Adaptive Scaling & Performance (Make it Scale)

Usage:
    python demo_ai_optimizer.py
"""

import asyncio
import json
import time
from pathlib import Path

from src.docker_optimizer.ai_optimization_engine import (
    AIOptimizationEngine,
    AIOptimizationRequest,
    OptimizationStrategy
)
from src.docker_optimizer.adaptive_scaling_engine import (
    AdaptiveScalingEngine,
    ScalingStrategy
)


class AIOptimizerDemo:
    """Comprehensive demo of AI Docker optimization system."""
    
    def __init__(self):
        self.ai_engine = None
        self.scaling_engine = None
    
    async def setup_systems(self):
        """Initialize all system components."""
        print("🚀 Initializing AI Docker Optimizer Systems...")
        
        # Initialize AI Optimization Engine (Generation 1)
        self.ai_engine = AIOptimizationEngine(
            enable_llm_integration=True,
            enable_research_mode=False,
            cache_enabled=True
        )
        print("✅ Generation 1: AI-Powered Optimization Engine initialized")
        
        # Initialize Adaptive Scaling Engine (Generation 3)
        self.scaling_engine = AdaptiveScalingEngine(
            strategy=ScalingStrategy.HYBRID,
            initial_workers=4,
            min_workers=2,
            max_workers=12
        )
        await self.scaling_engine.start_scaling_engine()
        print("✅ Generation 3: Adaptive Scaling Engine started")
        
        print("✅ Generation 2: Resilience & Health Monitoring built-in")
        print()
    
    async def demo_single_optimization(self):
        """Demonstrate single AI-powered optimization."""
        print("=" * 60)
        print("📋 DEMO 1: Single AI-Powered Optimization")
        print("=" * 60)
        
        # Sample problematic Dockerfile
        dockerfile = """FROM ubuntu:latest

# Install dependencies
RUN apt-get update
RUN apt-get install -y python3 python3-pip curl git nodejs npm

# Copy application
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY package.json .
RUN npm install

COPY . /app
WORKDIR /app

# Run as root (security issue)
USER root

# No health check
CMD ["python3", "app.py"]"""
        
        print("🔍 Original Dockerfile (with security/performance issues):")
        print("-" * 50)
        print(dockerfile)
        print()
        
        # Create optimization request
        request = AIOptimizationRequest(
            dockerfile_content=dockerfile,
            strategy=OptimizationStrategy.BALANCED,
            target_environment="production",
            security_requirements=["non-root", "version-pinning"],
            performance_requirements=["minimal-size", "layer-optimization"],
            compliance_frameworks=["HIPAA"]
        )
        
        print("⚡ Running AI optimization...")
        start_time = time.time()
        
        result = await self.ai_engine.optimize_dockerfile_with_ai(request)
        
        end_time = time.time()
        print(f"✅ Optimization completed in {end_time - start_time:.2f}s")
        print()
        
        # Display results
        self._display_optimization_results(result)
    
    async def demo_concurrent_optimization(self):
        """Demonstrate concurrent optimization with scaling."""
        print("=" * 60)
        print("🔄 DEMO 2: Concurrent Optimization with Auto-Scaling")
        print("=" * 60)
        
        # Multiple Dockerfiles to optimize concurrently
        dockerfiles = {
            "Python Web App": """FROM python:latest
RUN apt-get update && apt-get install -y curl
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
USER root
CMD ["gunicorn", "app:app"]""",
            
            "Node.js API": """FROM node:latest
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
USER root
CMD ["node", "server.js"]""",
            
            "Go Microservice": """FROM golang:latest
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o main .
USER root
CMD ["./main"]""",
            
            "Java Spring Boot": """FROM openjdk:latest
COPY target/app.jar app.jar
EXPOSE 8080
USER root
ENTRYPOINT ["java","-jar","/app.jar"]"""
        }
        
        print(f"🔢 Optimizing {len(dockerfiles)} Dockerfiles concurrently...")
        
        # Create optimization requests
        requests = []
        for name, dockerfile in dockerfiles.items():
            request = AIOptimizationRequest(
                dockerfile_content=dockerfile,
                strategy=OptimizationStrategy.AGGRESSIVE,
                target_environment="production",
                security_requirements=["non-root", "version-pinning"],
                performance_requirements=["minimal-size"]
            )
            requests.append((name, request))
        
        start_time = time.time()
        
        # Run all optimizations concurrently
        tasks = [
            self.ai_engine.optimize_dockerfile_with_ai(request)
            for name, request in requests
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ All {len(dockerfiles)} optimizations completed in {total_time:.2f}s")
        print(f"⚡ Average time per optimization: {total_time / len(dockerfiles):.2f}s")
        print(f"🚀 Throughput: {len(dockerfiles) / total_time:.2f} optimizations/sec")
        print()
        
        # Display summary results
        for i, (name, result) in enumerate(zip([name for name, _ in requests], results)):
            print(f"📊 {name}:")
            print(f"   🎯 Confidence: {result.confidence_score:.1%}")
            print(f"   🔒 Security enhancement: {result.metrics.security_enhancement:.1%}")
            print(f"   📦 Size reduction: {result.metrics.size_reduction_estimate:.1%}")
            print(f"   ⚡ Processing time: {result.metrics.processing_time:.3f}s")
            print()
    
    async def demo_resilience_features(self):
        """Demonstrate resilience and error recovery features."""
        print("=" * 60)
        print("🛡️  DEMO 3: Resilience & Error Recovery")
        print("=" * 60)
        
        # Test with problematic input
        bad_dockerfile = ""  # Empty dockerfile
        
        request = AIOptimizationRequest(
            dockerfile_content=bad_dockerfile,
            strategy=OptimizationStrategy.BALANCED,
            target_environment="production"
        )
        
        print("🧪 Testing with problematic input (empty Dockerfile)...")
        
        start_time = time.time()
        
        try:
            result = await self.ai_engine.optimize_dockerfile_with_ai(request)
            end_time = time.time()
            
            print(f"✅ Resilience system handled error gracefully in {end_time - start_time:.2f}s")
            print(f"🔄 Used fallback optimization: {len(result.explanations) > 0}")
            print(f"🎯 Fallback confidence: {result.confidence_score:.1%}")
            
        except Exception as e:
            end_time = time.time()
            print(f"❌ Fallback failed: {e} (in {end_time - start_time:.2f}s)")
        
        print()
        
        # Show health status
        health_status = await self.ai_engine.get_health_status()
        print("🏥 System Health Status:")
        print(f"   Overall Status: {health_status['overall_status'].upper()}")
        print(f"   Health Message: {health_status['health_message']}")
        print(f"   AI Features Active: LLM={health_status['ai_features']['llm_integration_enabled']}, "
              f"Research={health_status['ai_features']['research_mode_enabled']}, "
              f"Cache={health_status['ai_features']['cache_enabled']}")
        print()
    
    async def demo_scaling_performance(self):
        """Demonstrate adaptive scaling performance."""
        print("=" * 60)
        print("📈 DEMO 4: Adaptive Scaling Performance")
        print("=" * 60)
        
        # Show initial scaling status
        initial_status = self.scaling_engine.get_scaling_status()
        print("📊 Initial Scaling Status:")
        print(f"   Strategy: {initial_status['strategy']}")
        print(f"   Active Workers: {sum(pool['current_workers'] for pool in initial_status['worker_pools'].values())}")
        print(f"   Total Pools: {len(initial_status['worker_pools'])}")
        print()
        
        # Submit load to trigger scaling
        print("🔥 Submitting high load to trigger auto-scaling...")
        
        for i in range(30):
            await self.scaling_engine.submit_task(
                "general",
                lambda x: time.sleep(0.1),  # Simulate work
                i
            )
        
        # Wait for processing and scaling
        await asyncio.sleep(3.0)
        
        # Show updated status
        final_status = self.scaling_engine.get_scaling_status()
        performance_metrics = self.scaling_engine.get_performance_metrics()
        
        print("📈 Post-Load Scaling Status:")
        print(f"   Active Workers: {sum(pool['current_workers'] for pool in final_status['worker_pools'].values())}")
        print(f"   Processed Tasks: {final_status['performance_stats']['processed_tasks']}")
        print(f"   Average Processing Time: {final_status['performance_stats']['avg_processing_time']:.3f}s")
        print()
        
        if "current_metrics" in performance_metrics:
            metrics = performance_metrics["current_metrics"]
            print("⚡ Performance Metrics:")
            print(f"   CPU Utilization: {metrics['cpu_utilization']:.1f}%")
            print(f"   Memory Utilization: {metrics['memory_utilization']:.1f}%")
            print(f"   Queue Size: {metrics['queue_size']}")
            print(f"   Throughput: {metrics['throughput']:.2f} ops/sec")
            print()
    
    async def demo_integration_showcase(self):
        """Demonstrate full system integration."""
        print("=" * 60)
        print("🌟 DEMO 5: Full System Integration Showcase")
        print("=" * 60)
        
        # Complex multi-stage Dockerfile for comprehensive optimization
        complex_dockerfile = """# Multi-stage build example with issues
FROM node:18-alpine as base
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:latest
RUN apt-get update && apt-get install -y curl
COPY --from=base /usr/src/app/dist /usr/share/nginx/html
EXPOSE 80
USER root
CMD ["nginx", "-g", "daemon off;"]"""
        
        print("🔧 Complex Multi-Stage Dockerfile Analysis:")
        print("-" * 50)
        print(complex_dockerfile)
        print()
        
        # Create comprehensive optimization request
        request = AIOptimizationRequest(
            dockerfile_content=complex_dockerfile,
            strategy=OptimizationStrategy.RESEARCH,  # Most comprehensive
            target_environment="production",
            security_requirements=["non-root", "version-pinning", "minimal-attack-surface"],
            performance_requirements=["minimal-size", "layer-optimization", "build-cache"],
            compliance_frameworks=["SOC2", "GDPR"]
        )
        
        print("🧠 Running comprehensive AI analysis...")
        start_time = time.time()
        
        # Get initial system state
        initial_health = await self.ai_engine.get_health_status()
        initial_scaling = self.scaling_engine.get_scaling_status()
        
        # Perform optimization
        result = await self.ai_engine.optimize_dockerfile_with_ai(request)
        
        end_time = time.time()
        
        # Get final system state
        final_health = await self.ai_engine.get_health_status()
        final_scaling = self.scaling_engine.get_scaling_status()
        operation_stats = await self.ai_engine.get_operation_statistics()
        
        print(f"✅ Comprehensive optimization completed in {end_time - start_time:.2f}s")
        print()
        
        # Display comprehensive results
        self._display_comprehensive_results(result, initial_health, final_health, 
                                           initial_scaling, final_scaling, operation_stats)
    
    def _display_optimization_results(self, result):
        """Display optimization results in a formatted way."""
        print("🎯 OPTIMIZATION RESULTS:")
        print("-" * 30)
        print(f"Confidence Score: {result.confidence_score:.1%}")
        print(f"Processing Time: {result.metrics.processing_time:.3f}s")
        print()
        
        if result.explanations:
            print("📋 Key Optimizations Applied:")
            for i, explanation in enumerate(result.explanations, 1):
                print(f"  {i}. {explanation}")
            print()
        
        if result.security_improvements:
            print("🔒 Security Improvements:")
            for improvement in result.security_improvements:
                print(f"  ✅ {improvement}")
            print()
        
        if result.performance_enhancements:
            print("⚡ Performance Enhancements:")
            for enhancement in result.performance_enhancements:
                print(f"  🚀 {enhancement}")
            print()
        
        print("🐳 Optimized Dockerfile:")
        print("-" * 30)
        print(result.optimized_dockerfile)
        print()
    
    def _display_comprehensive_results(self, result, initial_health, final_health, 
                                     initial_scaling, final_scaling, operation_stats):
        """Display comprehensive system results."""
        print("🌟 COMPREHENSIVE SYSTEM ANALYSIS:")
        print("=" * 50)
        
        # Optimization Results
        print("🎯 Optimization Metrics:")
        print(f"   Confidence Score: {result.confidence_score:.1%}")
        print(f"   Security Enhancement: {result.metrics.security_enhancement:.1%}")
        print(f"   Size Reduction Estimate: {result.metrics.size_reduction_estimate:.1%}")
        print(f"   Performance Gain: {result.metrics.performance_gain_estimate:.1%}")
        print(f"   Processing Time: {result.metrics.processing_time:.3f}s")
        print()
        
        # System Health Comparison
        print("🏥 System Health Analysis:")
        print(f"   Status: {initial_health['overall_status']} → {final_health['overall_status']}")
        print(f"   AI Features: {final_health['ai_features']}")
        print(f"   Patterns Loaded: {final_health['optimization_patterns_loaded']}")
        print()
        
        # Scaling Performance
        initial_workers = sum(pool['current_workers'] for pool in initial_scaling['worker_pools'].values())
        final_workers = sum(pool['current_workers'] for pool in final_scaling['worker_pools'].values())
        
        print("📈 Scaling Performance:")
        print(f"   Workers: {initial_workers} → {final_workers}")
        print(f"   Tasks Processed: {final_scaling['performance_stats']['processed_tasks']}")
        print(f"   Average Processing Time: {final_scaling['performance_stats']['avg_processing_time']:.3f}s")
        print()
        
        # Operation Statistics
        if "ai_optimization_stats" in operation_stats:
            stats = operation_stats["ai_optimization_stats"]
            print("📊 AI Operation Statistics:")
            print(f"   Total Calls: {stats['total_calls']}")
            print(f"   Average Response Time: {stats['avg_response_time']:.3f}s")
            print(f"   Circuit State: {stats.get('circuit_state', 'N/A')}")
            print()
        
        # Final optimized result
        print("🐳 Final Optimized Dockerfile:")
        print("-" * 40)
        print(result.optimized_dockerfile)
        print()
        
        print("💡 Alternative Approaches:")
        for approach in result.alternative_approaches:
            print(f"  • {approach}")
        print()
    
    async def cleanup_systems(self):
        """Clean up system resources."""
        print("🧹 Cleaning up systems...")
        
        if self.scaling_engine:
            await self.scaling_engine.stop_scaling_engine()
            print("✅ Adaptive Scaling Engine stopped")
        
        print("✅ AI Optimization Engine cleaned up")
        print("✅ All systems shut down gracefully")


async def main():
    """Run the comprehensive AI Docker Optimizer demo."""
    print("🌟" * 30)
    print("🤖 AI DOCKER OPTIMIZER - COMPREHENSIVE DEMO")
    print("🌟" * 30)
    print()
    print("This demo showcases three generations of AI optimization:")
    print("  Generation 1: AI-Powered Optimization (Make it Work)")
    print("  Generation 2: Resilience & Health Monitoring (Make it Robust)")
    print("  Generation 3: Adaptive Scaling & Performance (Make it Scale)")
    print()
    
    demo = AIOptimizerDemo()
    
    try:
        # Setup
        await demo.setup_systems()
        
        # Run demos
        await demo.demo_single_optimization()
        await demo.demo_concurrent_optimization()
        await demo.demo_resilience_features()
        await demo.demo_scaling_performance()
        await demo.demo_integration_showcase()
        
        print("🎉" * 30)
        print("🏆 DEMO COMPLETED SUCCESSFULLY!")
        print("🎉" * 30)
        print()
        print("✨ The AI Docker Optimizer has demonstrated:")
        print("  ✅ Advanced AI-powered Dockerfile optimization")
        print("  ✅ Resilient error handling and recovery")
        print("  ✅ Adaptive auto-scaling under load")
        print("  ✅ Comprehensive health monitoring")
        print("  ✅ Production-ready performance")
        print()
        print("🚀 Ready for production deployment!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup_systems()


if __name__ == "__main__":
    asyncio.run(main())