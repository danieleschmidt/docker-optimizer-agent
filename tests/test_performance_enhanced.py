"""Tests for enhanced performance features."""

import pytest
import time
from unittest.mock import patch, MagicMock

from docker_optimizer.performance import (
    AutoScalingManager,
    AutoScalingConfig,
    AdaptiveLoadBalancer,
    PerformanceMetrics,
    OptimizationCache
)


class TestAutoScalingManager:
    """Test auto-scaling functionality."""

    def test_auto_scaling_initialization(self):
        """Test auto-scaling manager initialization."""
        config = AutoScalingConfig(min_workers=2, max_workers=8)
        scaler = AutoScalingManager(config)
        
        assert scaler.current_workers == 2
        assert scaler.config.min_workers == 2
        assert scaler.config.max_workers == 8
        assert scaler.worker_pool is not None
        
        scaler.shutdown()

    @patch('psutil.cpu_percent')
    def test_scale_up_trigger(self, mock_cpu):
        """Test scale up triggering."""
        mock_cpu.return_value = 90.0  # High CPU usage
        
        config = AutoScalingConfig(min_workers=2, max_workers=8, scale_up_threshold=0.8)
        scaler = AutoScalingManager(config)
        
        # Force metrics collection
        scaler._collect_metrics()
        
        # Should trigger scale up
        initial_workers = scaler.current_workers
        scaler.auto_scale()
        
        # Should have scaled up
        assert scaler.current_workers > initial_workers
        
        scaler.shutdown()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_scale_down_trigger(self, mock_memory, mock_cpu):
        """Test scale down triggering."""
        mock_cpu.return_value = 20.0  # Low CPU usage
        mock_memory.return_value.percent = 30.0
        
        config = AutoScalingConfig(
            min_workers=2, 
            max_workers=8, 
            scale_down_threshold=0.3,
            scale_down_cooldown=0  # Disable cooldown for test
        )
        scaler = AutoScalingManager(config)
        
        # Start with more workers
        scaler.current_workers = 6
        scaler._create_worker_pool()
        scaler.last_scale_time = 0  # Reset cooldown
        
        # Set conditions for scale down
        scaler._metrics["active_tasks"] = 0
        scaler._metrics["queue_length"] = 0
        
        # Force metrics collection
        scaler._collect_metrics()
        
        initial_workers = scaler.current_workers
        scaler.auto_scale()
        
        # Should have scaled down
        assert scaler.current_workers < initial_workers
        
        scaler.shutdown()

    def test_scaling_bounds(self):
        """Test that scaling respects min/max bounds."""
        config = AutoScalingConfig(min_workers=2, max_workers=4)
        scaler = AutoScalingManager(config)
        
        # Test max bound
        scaler.current_workers = 4
        scaler._scale_up()
        assert scaler.current_workers == 4  # Should not exceed max
        
        # Test min bound  
        scaler.current_workers = 2
        scaler._scale_down()
        assert scaler.current_workers == 2  # Should not go below min
        
        scaler.shutdown()

    def test_cooldown_periods(self):
        """Test cooldown periods prevent rapid scaling."""
        config = AutoScalingConfig(scale_up_cooldown=60, scale_down_cooldown=300)
        scaler = AutoScalingManager(config)
        
        # Set recent scale time
        scaler.last_scale_time = time.time()
        
        # Should not scale up due to cooldown
        assert not scaler._should_scale_up()
        
        # Should not scale down due to cooldown
        assert not scaler._should_scale_down()
        
        scaler.shutdown()

    def test_task_submission_with_auto_scaling(self):
        """Test task submission triggers auto-scaling checks."""
        scaler = AutoScalingManager()
        
        def dummy_task():
            time.sleep(0.01)
            return "completed"
        
        # Submit a task
        future = scaler.submit_task(dummy_task)
        result = future.result(timeout=1)
        
        assert result == "completed"
        assert scaler._metrics["completed_tasks"] >= 1
        
        scaler.shutdown()

    def test_scaling_metrics(self):
        """Test scaling metrics collection."""
        scaler = AutoScalingManager()
        metrics = scaler.get_scaling_metrics()
        
        expected_keys = {
            "current_workers", "min_workers", "max_workers",
            "avg_cpu_usage", "queue_length", "active_tasks",
            "completed_tasks", "last_scale_time"
        }
        
        assert set(metrics.keys()) == expected_keys
        assert isinstance(metrics["current_workers"], int)
        assert isinstance(metrics["avg_cpu_usage"], float)
        
        scaler.shutdown()


class TestAdaptiveLoadBalancer:
    """Test adaptive load balancing functionality."""

    def test_load_balancer_initialization(self):
        """Test load balancer initialization."""
        lb = AdaptiveLoadBalancer()
        assert len(lb.node_metrics) == 0
        assert len(lb.routing_weights) == 0

    def test_node_registration(self):
        """Test node registration."""
        lb = AdaptiveLoadBalancer()
        lb.register_node("node-1", initial_weight=1.0)
        lb.register_node("node-2", initial_weight=0.5)
        
        assert "node-1" in lb.node_metrics
        assert "node-2" in lb.node_metrics
        assert lb.routing_weights["node-1"] == 1.0
        assert lb.routing_weights["node-2"] == 0.5

    def test_node_metrics_update(self):
        """Test node metrics updates."""
        lb = AdaptiveLoadBalancer()
        lb.register_node("node-1")
        
        # Update with successful response
        lb.update_node_metrics("node-1", response_time=0.1, success=True)
        
        metrics = lb.node_metrics["node-1"]
        assert 0.1 in metrics["response_time"]
        assert metrics["success_rate"] > 0.9  # Should be high due to success

    def test_weight_adaptation(self):
        """Test adaptive weight calculation."""
        lb = AdaptiveLoadBalancer()
        lb.register_node("fast-node")
        lb.register_node("slow-node")
        
        # Simulate fast node performance
        for _ in range(10):
            lb.update_node_metrics("fast-node", response_time=0.1, success=True)
        
        # Simulate slow node performance
        for _ in range(10):
            lb.update_node_metrics("slow-node", response_time=1.0, success=True)
        
        # Force weight update
        lb._update_interval = 0
        lb._update_routing_weights()
        
        # Fast node should have higher weight
        assert lb.routing_weights["fast-node"] > lb.routing_weights["slow-node"]

    def test_node_selection(self):
        """Test node selection based on weights."""
        lb = AdaptiveLoadBalancer()
        lb.register_node("node-1", initial_weight=0.8)
        lb.register_node("node-2", initial_weight=0.2)
        
        # Test multiple selections (should favor node-1)
        selections = []
        for _ in range(100):
            selected = lb.select_node()
            selections.append(selected)
        
        node1_count = selections.count("node-1")
        node2_count = selections.count("node-2")
        
        # node-1 should be selected more often due to higher weight
        assert node1_count > node2_count

    def test_stale_node_handling(self):
        """Test handling of stale nodes."""
        lb = AdaptiveLoadBalancer()
        lb.register_node("stale-node")
        
        # Simulate stale node (old last_seen timestamp)
        lb.node_metrics["stale-node"]["last_seen"] = time.time() - 400  # 400 seconds ago
        lb.node_metrics["stale-node"]["response_time"] = [0.1]
        
        # Force weight update
        lb._update_interval = 0
        lb._update_routing_weights()
        
        # Stale node should have very low weight
        assert lb.routing_weights["stale-node"] < 0.1

    def test_load_balancer_metrics(self):
        """Test load balancer metrics."""
        lb = AdaptiveLoadBalancer()
        lb.register_node("node-1")
        lb.update_node_metrics("node-1", response_time=0.2, success=True)
        
        metrics = lb.get_load_balancer_metrics()
        
        assert "registered_nodes" in metrics
        assert "routing_weights" in metrics
        assert "node_metrics" in metrics
        assert metrics["registered_nodes"] == 1
        assert "node-1" in metrics["routing_weights"]


class TestPerformanceMetrics:
    """Test performance metrics functionality."""

    def test_performance_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        assert metrics.processing_time == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.dockerfiles_processed == 0

    def test_timing_context_manager(self):
        """Test timing context manager."""
        metrics = PerformanceMetrics()
        
        with metrics.timer():
            time.sleep(0.01)  # Small delay
        
        assert metrics.processing_time > 0
        assert metrics.start_time is not None
        assert metrics.end_time is not None

    @patch('psutil.Process')
    def test_memory_usage_tracking(self, mock_process):
        """Test memory usage tracking."""
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        metrics = PerformanceMetrics()
        memory_mb = metrics.get_memory_usage()
        
        assert memory_mb == 100.0  # Should be 100MB

    def test_performance_report(self):
        """Test performance report generation."""
        metrics = PerformanceMetrics()
        metrics.processing_time = 1.5
        metrics.dockerfiles_processed = 10
        metrics.cache_hits = 7
        metrics.cache_misses = 3
        
        report = metrics.get_report()
        
        assert report["processing_time"] == 1.5
        assert report["dockerfiles_processed"] == 10
        assert report["cache_hits"] == 7
        assert report["cache_misses"] == 3
        assert report["cache_hit_ratio"] == 0.7  # 7/(7+3)


class TestOptimizationCache:
    """Test optimization cache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = OptimizationCache(max_size=100, ttl_seconds=300)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
        assert cache.size() == 0

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = OptimizationCache()
        
        content1 = "FROM ubuntu:20.04\nRUN apt-get update"
        content2 = "FROM ubuntu:22.04\nRUN apt-get update"
        
        key1 = cache._generate_key(content1)
        key2 = cache._generate_key(content2)
        
        assert key1 != key2  # Different content should generate different keys
        assert len(key1) == 64  # SHA256 hash length
        assert len(key2) == 64

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        from docker_optimizer.models import OptimizationResult
        
        cache = OptimizationCache()
        content = "FROM ubuntu:20.04"
        
        # Mock optimization result
        result = OptimizationResult(
            optimized_dockerfile=content,
            explanation="Test result",
            original_size=100,
            optimized_size=80
        )
        
        # Test cache miss
        cached_result = cache.get(content)
        assert cached_result is None
        
        # Test cache set and hit
        cache.set(content, result)
        cached_result = cache.get(content)
        assert cached_result is not None
        assert cached_result.optimized_dockerfile == content

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        from docker_optimizer.models import OptimizationResult
        
        cache = OptimizationCache(ttl_seconds=0.1)  # Very short TTL
        content = "FROM ubuntu:20.04"
        
        result = OptimizationResult(
            optimized_dockerfile=content,
            explanation="Test result",
            original_size=100,
            optimized_size=80
        )
        
        # Set and immediately get
        cache.set(content, result)
        cached_result = cache.get(content)
        assert cached_result is not None
        
        # Wait for expiration
        time.sleep(0.2)
        cached_result = cache.get(content)
        assert cached_result is None  # Should be expired

    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        from docker_optimizer.models import OptimizationResult
        
        cache = OptimizationCache(max_size=2)  # Small cache size
        
        result1 = OptimizationResult(
            optimized_dockerfile="content1",
            explanation="Test1",
            original_size=100,
            optimized_size=80
        )
        result2 = OptimizationResult(
            optimized_dockerfile="content2",
            explanation="Test2",
            original_size=100,
            optimized_size=80
        )
        result3 = OptimizationResult(
            optimized_dockerfile="content3",
            explanation="Test3",
            original_size=100,
            optimized_size=80
        )
        
        # Fill cache to capacity
        cache.set("content1", result1)
        cache.set("content2", result2)
        assert cache.size() == 2
        
        # Add third item should evict first
        cache.set("content3", result3)
        assert cache.size() == 2
        assert cache.get("content1") is None  # Should be evicted
        assert cache.get("content2") is not None
        assert cache.get("content3") is not None

    def test_cache_clear(self):
        """Test cache clearing."""
        from docker_optimizer.models import OptimizationResult
        
        cache = OptimizationCache()
        result = OptimizationResult(
            optimized_dockerfile="content",
            explanation="Test",
            original_size=100,
            optimized_size=80
        )
        
        cache.set("content", result)
        assert cache.size() == 1
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("content") is None