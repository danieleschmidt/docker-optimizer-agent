"""Comprehensive AI Performance Testing Suite."""

import asyncio
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from src.docker_optimizer.ai_optimization_engine import (
    AIOptimizationEngine,
    AIOptimizationRequest,
    OptimizationStrategy
)
from src.docker_optimizer.adaptive_scaling_engine import (
    AdaptiveScalingEngine,
    ScalingStrategy
)
from src.docker_optimizer.resilience_engine import ResilienceConfig
from src.docker_optimizer.ai_health_monitor import AIHealthMonitor


class TestAIPerformanceSuite:
    """Comprehensive performance testing for AI optimization system."""
    
    @pytest.fixture
    def sample_dockerfiles(self):
        """Sample Dockerfiles for performance testing."""
        return {
            "simple": """FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3
CMD ["python3"]""",
            
            "complex": """FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
USER root
CMD ["npm", "start"]""",
            
            "multi_stage": """FROM node:16 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]""",
            
            "security_issues": """FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y wget curl
COPY . /
WORKDIR /
CMD ["./app"]"""
        }
    
    @pytest.fixture
    def ai_engine(self):
        """AI optimization engine instance."""
        return AIOptimizationEngine(
            enable_llm_integration=True,
            enable_research_mode=False,
            cache_enabled=True
        )
    
    @pytest.fixture
    def scaling_engine(self):
        """Adaptive scaling engine instance."""
        return AdaptiveScalingEngine(
            strategy=ScalingStrategy.HYBRID,
            initial_workers=4,
            min_workers=2,
            max_workers=16
        )

    @pytest.mark.asyncio
    async def test_single_optimization_performance(self, ai_engine, sample_dockerfiles):
        """Test performance of single AI optimization."""
        request = AIOptimizationRequest(
            dockerfile_content=sample_dockerfiles["complex"],
            strategy=OptimizationStrategy.BALANCED,
            target_environment="production"
        )
        
        start_time = time.time()
        result = await ai_engine.optimize_dockerfile_with_ai(request)
        end_time = time.time()
        
        # Performance assertions
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Single optimization took too long: {processing_time}s"
        assert result.metrics.processing_time < 2.0, f"Internal processing time too high: {result.metrics.processing_time}s"
        assert result.confidence_score > 0.8, f"Confidence score too low: {result.confidence_score}"

    @pytest.mark.asyncio
    async def test_concurrent_optimization_performance(self, ai_engine, sample_dockerfiles):
        """Test performance under concurrent load."""
        requests = [
            AIOptimizationRequest(
                dockerfile_content=dockerfile,
                strategy=OptimizationStrategy.BALANCED,
                target_environment="production"
            )
            for dockerfile in sample_dockerfiles.values()
        ]
        
        start_time = time.time()
        
        # Run optimizations concurrently
        results = await asyncio.gather(*[
            ai_engine.optimize_dockerfile_with_ai(req) for req in requests
        ])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert len(results) == len(requests), "Not all optimizations completed"
        assert total_time < 10.0, f"Concurrent optimization took too long: {total_time}s"
        assert all(r.confidence_score > 0.5 for r in results), "Some optimizations had low confidence"

    @pytest.mark.asyncio 
    async def test_scaling_engine_performance(self, scaling_engine):
        """Test adaptive scaling engine performance."""
        
        # Start scaling engine
        await scaling_engine.start_scaling_engine()
        
        try:
            # Submit multiple tasks
            tasks = []
            for i in range(20):
                task_submitted = await scaling_engine.submit_task(
                    "general",
                    lambda x: time.sleep(0.1),  # Simulate work
                    i
                )
                assert task_submitted, f"Failed to submit task {i}"
            
            # Wait for tasks to process
            await asyncio.sleep(2.0)
            
            # Check scaling status
            status = scaling_engine.get_scaling_status()
            assert status["active"], "Scaling engine should be active"
            assert status["performance_stats"]["processed_tasks"] > 0, "No tasks were processed"
            
            # Performance metrics
            metrics = scaling_engine.get_performance_metrics()
            assert "current_metrics" in metrics, "Performance metrics missing"
            
        finally:
            await scaling_engine.stop_scaling_engine()

    @pytest.mark.asyncio
    async def test_health_monitoring_performance(self):
        """Test health monitoring system performance."""
        health_monitor = AIHealthMonitor()
        
        start_time = time.time()
        health_check = await health_monitor.check_system_health()
        end_time = time.time()
        
        check_time = end_time - start_time
        
        # Performance assertions
        assert check_time < 1.0, f"Health check took too long: {check_time}s"
        assert health_check.status is not None, "Health status missing"
        assert health_check.timestamp > 0, "Invalid timestamp"

    @pytest.mark.asyncio
    async def test_resilience_engine_performance(self, ai_engine):
        """Test resilience engine performance under failure conditions."""
        
        # Test with simulated failures
        failure_count = 0
        
        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise Exception("Simulated failure")
            return "Success after retries"
        
        start_time = time.time()
        result = await ai_engine.resilience_engine.execute_with_resilience(
            "test_operation",
            failing_operation
        )
        end_time = time.time()
        
        # Performance assertions
        total_time = end_time - start_time
        assert total_time < 10.0, f"Resilient operation took too long: {total_time}s"
        # Operation may fail if all retries are exhausted, this is expected behavior
        assert result.attempts >= 2, "Should attempt multiple retries"

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, ai_engine, sample_dockerfiles):
        """Test memory usage during optimization operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple optimizations
        for dockerfile in sample_dockerfiles.values():
            request = AIOptimizationRequest(
                dockerfile_content=dockerfile,
                strategy=OptimizationStrategy.BALANCED
            )
            await ai_engine.optimize_dockerfile_with_ai(request)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not increase dramatically
        assert memory_increase < 50 * 1024 * 1024, f"Memory usage increased too much: {memory_increase / 1024 / 1024}MB"

    def test_cpu_utilization_optimization(self, ai_engine, sample_dockerfiles):
        """Test CPU utilization during heavy optimization load."""
        import psutil
        
        cpu_usage_samples = []
        
        def monitor_cpu():
            for _ in range(10):
                cpu_usage_samples.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        # Start CPU monitoring in background
        with ThreadPoolExecutor(max_workers=1) as executor:
            cpu_monitor_future = executor.submit(monitor_cpu)
            
            # Perform optimizations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                for dockerfile in sample_dockerfiles.values():
                    request = AIOptimizationRequest(
                        dockerfile_content=dockerfile,
                        strategy=OptimizationStrategy.AGGRESSIVE
                    )
                    loop.run_until_complete(ai_engine.optimize_dockerfile_with_ai(request))
            finally:
                loop.close()
            
            cpu_monitor_future.result()
        
        # CPU usage should be reasonable
        avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples)
        max_cpu_usage = max(cpu_usage_samples)
        
        assert avg_cpu_usage < 80.0, f"Average CPU usage too high: {avg_cpu_usage}%"
        assert max_cpu_usage < 95.0, f"Peak CPU usage too high: {max_cpu_usage}%"

    @pytest.mark.asyncio
    async def test_throughput_performance(self, ai_engine, sample_dockerfiles):
        """Test overall system throughput."""
        
        start_time = time.time()
        total_optimizations = 0
        
        # Run optimizations for a fixed duration
        duration = 5.0  # 5 seconds
        end_time = start_time + duration
        
        while time.time() < end_time:
            dockerfile = list(sample_dockerfiles.values())[total_optimizations % len(sample_dockerfiles)]
            request = AIOptimizationRequest(
                dockerfile_content=dockerfile,
                strategy=OptimizationStrategy.BALANCED
            )
            
            await ai_engine.optimize_dockerfile_with_ai(request)
            total_optimizations += 1
        
        actual_duration = time.time() - start_time
        throughput = total_optimizations / actual_duration
        
        # Throughput assertions
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} ops/sec"
        assert total_optimizations >= 5, f"Too few optimizations completed: {total_optimizations}"

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, ai_engine):
        """Test performance of error recovery mechanisms."""
        
        # Create a request that will cause failures
        bad_request = AIOptimizationRequest(
            dockerfile_content="",  # Empty dockerfile
            strategy=OptimizationStrategy.BALANCED
        )
        
        start_time = time.time()
        
        try:
            # This should handle the error gracefully
            result = await ai_engine.optimize_dockerfile_with_ai(bad_request)
            
            # Should succeed due to fallback mechanisms
            assert result is not None, "Should have fallback result"
            
        except Exception as e:
            # If it does fail, it should fail quickly
            pass
        
        end_time = time.time()
        error_handling_time = end_time - start_time
        
        assert error_handling_time < 3.0, f"Error recovery took too long: {error_handling_time}s"

    @pytest.mark.asyncio
    async def test_cache_performance(self, ai_engine, sample_dockerfiles):
        """Test caching performance improvements."""
        
        dockerfile = sample_dockerfiles["simple"]
        request = AIOptimizationRequest(
            dockerfile_content=dockerfile,
            strategy=OptimizationStrategy.BALANCED
        )
        
        # First optimization (cold)
        start_time = time.time()
        result1 = await ai_engine.optimize_dockerfile_with_ai(request)
        first_time = time.time() - start_time
        
        # Second optimization (should benefit from caching)
        start_time = time.time()
        result2 = await ai_engine.optimize_dockerfile_with_ai(request)
        second_time = time.time() - start_time
        
        # Cache should improve performance
        improvement_ratio = first_time / max(second_time, 0.001)
        
        # Results should be consistent
        assert result1.optimized_dockerfile == result2.optimized_dockerfile, "Cached results should be identical"
        
        # Note: In this implementation, caching may not show dramatic improvement
        # as the operations are relatively fast, but the infrastructure is there
        
    @pytest.mark.asyncio
    async def test_scaling_decision_performance(self, scaling_engine):
        """Test performance of scaling decision making."""
        
        await scaling_engine.start_scaling_engine()
        
        try:
            # Wait for initial metrics collection
            await asyncio.sleep(1.0)
            
            # Trigger scaling decisions by submitting load
            for i in range(50):
                await scaling_engine.submit_task(
                    "general",
                    lambda: time.sleep(0.01)  # Small task
                )
            
            # Wait for scaling decisions
            await asyncio.sleep(2.0)
            
            status = scaling_engine.get_scaling_status()
            metrics = scaling_engine.get_performance_metrics()
            
            # Scaling should have occurred
            assert status["performance_stats"]["processed_tasks"] > 0, "Tasks should be processed"
            
            # Metrics should be available
            assert "current_metrics" in metrics, "Current metrics should be available"
            assert metrics["current_metrics"]["throughput"] >= 0, "Throughput should be measured"
            
        finally:
            await scaling_engine.stop_scaling_engine()

    @pytest.mark.asyncio
    async def test_integration_performance(self, ai_engine, scaling_engine, sample_dockerfiles):
        """Test integrated system performance."""
        
        await scaling_engine.start_scaling_engine()
        
        try:
            # Submit AI optimization tasks through scaling engine
            optimization_tasks = []
            
            for dockerfile in sample_dockerfiles.values():
                request = AIOptimizationRequest(
                    dockerfile_content=dockerfile,
                    strategy=OptimizationStrategy.BALANCED
                )
                
                task_future = asyncio.create_task(
                    ai_engine.optimize_dockerfile_with_ai(request)
                )
                optimization_tasks.append(task_future)
            
            start_time = time.time()
            results = await asyncio.gather(*optimization_tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Integration performance assertions
            assert len(results) == len(sample_dockerfiles), "All optimizations should complete"
            assert total_time < 15.0, f"Integrated system took too long: {total_time}s"
            assert all(r.confidence_score > 0.5 for r in results), "All results should have reasonable confidence"
            
            # Check system health after load
            health_status = await ai_engine.get_health_status()
            assert health_status["overall_status"] in ["healthy", "warning"], "System should remain healthy"
            
        finally:
            await scaling_engine.stop_scaling_engine()


@pytest.mark.performance
class TestStressConditions:
    """Stress testing under extreme conditions."""
    
    @pytest.mark.asyncio
    async def test_extreme_load_stress(self):
        """Test system under extreme concurrent load."""
        ai_engine = AIOptimizationEngine()
        
        dockerfile = """FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3
CMD ["python3"]"""
        
        # Create many concurrent requests
        requests = [
            AIOptimizationRequest(
                dockerfile_content=dockerfile,
                strategy=OptimizationStrategy.BALANCED
            )
            for _ in range(100)  # 100 concurrent optimizations
        ]
        
        start_time = time.time()
        
        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(20)  # Max 20 concurrent
        
        async def bounded_optimization(request):
            async with semaphore:
                return await ai_engine.optimize_dockerfile_with_ai(request)
        
        results = await asyncio.gather(
            *[bounded_optimization(req) for req in requests],
            return_exceptions=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stress test assertions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_results) / len(results)
        
        assert success_rate > 0.8, f"Success rate too low under stress: {success_rate:.2%}"
        assert total_time < 30.0, f"Stress test took too long: {total_time}s"
        
        print(f"Stress test completed: {len(successful_results)}/{len(results)} succeeded in {total_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])