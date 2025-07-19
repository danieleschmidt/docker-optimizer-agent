"""Tests for performance optimization features."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docker_optimizer.performance import (
    ParallelAnalyzer,
    OptimizationCache,
    LargeDockerfileHandler,
    PerformanceMetrics,
    CacheEntry,
)
from docker_optimizer.models import OptimizationResult


class TestParallelAnalyzer:
    """Test parallel analysis processing."""
    
    def test_parallel_analyzer_initialization(self):
        """Test ParallelAnalyzer can be initialized."""
        analyzer = ParallelAnalyzer(max_workers=4)
        assert analyzer.max_workers == 4
        assert analyzer.executor is not None
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_dockerfiles_parallel(self):
        """Test parallel analysis of multiple Dockerfiles."""
        analyzer = ParallelAnalyzer(max_workers=2)
        
        dockerfiles = [
            "FROM python:3.9\nRUN pip install requests",
            "FROM node:16\nRUN npm install express",
            "FROM golang:1.19\nRUN go mod download"
        ]
        
        start_time = time.time()
        results = await analyzer.analyze_multiple(dockerfiles)
        end_time = time.time()
        
        assert len(results) == 3
        assert all(isinstance(result, OptimizationResult) for result in results)
        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
    
    @pytest.mark.asyncio
    async def test_analyze_with_error_handling(self):
        """Test parallel analysis handles errors gracefully."""
        analyzer = ParallelAnalyzer(max_workers=2)
        
        # Include one invalid Dockerfile
        dockerfiles = [
            "FROM python:3.9\nRUN pip install requests",
            "INVALID DOCKERFILE CONTENT",
            "FROM node:16\nRUN npm install express"
        ]
        
        results = await analyzer.analyze_multiple(dockerfiles)
        
        # Should still return results for all Dockerfiles
        assert len(results) == 3
        # All results should be valid OptimizationResult instances
        assert all(isinstance(result, OptimizationResult) for result in results)


class TestOptimizationCache:
    """Test caching mechanisms."""
    
    def test_cache_initialization(self):
        """Test OptimizationCache can be initialized."""
        cache = OptimizationCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache._cache) == 0
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = OptimizationCache(max_size=10)
        
        dockerfile_content = "FROM python:3.9\nRUN pip install requests"
        result = OptimizationResult(
            original_size="100 MB",
            optimized_size="80 MB",
            explanation="Use specific Python version",
            optimized_dockerfile="FROM python:3.9-slim\nRUN pip install requests",
            security_fixes=[]
        )
        
        # Set cache entry
        cache.set(dockerfile_content, result)
        
        # Get cache entry
        cached_result = cache.get(dockerfile_content)
        assert cached_result is not None
        assert cached_result.original_size == "100 MB"
        assert cached_result.optimized_size == "80 MB"
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = OptimizationCache(max_size=10)
        result = cache.get("non-existent dockerfile")
        assert result is None
    
    def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        cache = OptimizationCache(max_size=10, ttl_seconds=0.1)  # Very short TTL
        
        dockerfile_content = "FROM python:3.9"
        result = OptimizationResult(
            original_size="100 MB",
            optimized_size="80 MB",
            explanation="Basic optimization",
            optimized_dockerfile="FROM python:3.9-slim",
            security_fixes=[]
        )
        
        cache.set(dockerfile_content, result)
        
        # Should be available immediately
        assert cache.get(dockerfile_content) is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired now
        assert cache.get(dockerfile_content) is None
    
    def test_cache_size_limit(self):
        """Test cache respects size limits."""
        cache = OptimizationCache(max_size=2)
        
        result = OptimizationResult(
            original_size="100 MB",
            optimized_size="80 MB",
            explanation="Basic optimization",
            optimized_dockerfile="FROM python:3.9-slim",
            security_fixes=[]
        )
        
        # Add 3 entries (exceeding max_size of 2)
        cache.set("dockerfile1", result)
        cache.set("dockerfile2", result)
        cache.set("dockerfile3", result)
        
        # Cache should only contain 2 entries
        assert len(cache._cache) == 2
        
        # First entry should be evicted (LRU)
        assert cache.get("dockerfile1") is None
        assert cache.get("dockerfile2") is not None
        assert cache.get("dockerfile3") is not None


class TestLargeDockerfileHandler:
    """Test large Dockerfile handling."""
    
    def test_handler_initialization(self):
        """Test LargeDockerfileHandler can be initialized."""
        handler = LargeDockerfileHandler(chunk_size=1000)
        assert handler.chunk_size == 1000
    
    def test_detect_large_dockerfile(self):
        """Test detection of large Dockerfiles."""
        handler = LargeDockerfileHandler(size_threshold=100)
        
        small_dockerfile = "FROM python:3.9\nRUN pip install requests"
        large_dockerfile = "FROM python:3.9\n" + "RUN echo 'line'\n" * 200
        
        assert not handler.is_large_dockerfile(small_dockerfile)
        assert handler.is_large_dockerfile(large_dockerfile)
    
    def test_chunk_large_dockerfile(self):
        """Test chunking of large Dockerfiles."""
        handler = LargeDockerfileHandler(chunk_size=50)
        
        # Create a large Dockerfile
        large_dockerfile = "FROM python:3.9\n" + "RUN echo 'test'\n" * 20
        
        chunks = handler.chunk_dockerfile(large_dockerfile)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)
        
        # Verify chunks can be reassembled
        reassembled = "\n".join(chunks)
        assert reassembled == large_dockerfile
    
    def test_process_large_dockerfile_in_chunks(self):
        """Test processing large Dockerfile in chunks."""
        handler = LargeDockerfileHandler(chunk_size=100, size_threshold=500)  # Lower threshold
        
        # Create a large Dockerfile that exceeds the threshold
        large_dockerfile = "FROM python:3.9\n" + "RUN pip install package\n" * 50
        
        # Ensure it's actually considered large
        assert handler.is_large_dockerfile(large_dockerfile)
        
        # Mock the chunk processor
        with patch.object(handler, '_process_chunk') as mock_process:
            mock_process.return_value = "optimization suggestion"
            
            result = handler.process_large_dockerfile(large_dockerfile)
            
            assert "optimization suggestion" in result.explanation
            assert mock_process.call_count > 1  # Should be called multiple times


class TestPerformanceMetrics:
    """Test performance metrics collection."""
    
    def test_metrics_initialization(self):
        """Test PerformanceMetrics can be initialized."""
        metrics = PerformanceMetrics()
        assert metrics.start_time is None
        assert metrics.end_time is None
        assert metrics.processing_time == 0.0
    
    def test_timing_context_manager(self):
        """Test metrics timing as context manager."""
        metrics = PerformanceMetrics()
        
        with metrics.timer():
            time.sleep(0.1)  # Simulate work
        
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.processing_time >= 0.1
        assert metrics.processing_time < 0.2  # Should be close to 0.1
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        metrics = PerformanceMetrics()
        
        initial_memory = metrics.get_memory_usage()
        
        # Allocate some memory
        large_list = [i for i in range(100000)]
        
        after_memory = metrics.get_memory_usage()
        
        # Memory usage should have increased
        assert after_memory > initial_memory
        
        # Clean up
        del large_list
    
    def test_metrics_report(self):
        """Test metrics report generation."""
        metrics = PerformanceMetrics()
        
        with metrics.timer():
            time.sleep(0.1)
        
        report = metrics.get_report()
        
        assert "processing_time" in report
        assert "memory_usage_mb" in report
        assert report["processing_time"] >= 0.1


class TestCacheEntry:
    """Test cache entry model."""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry can be created."""
        result = OptimizationResult(
            original_size="100 MB",
            optimized_size="80 MB",
            explanation="test optimization",
            optimized_dockerfile="FROM python:3.9-slim",
            security_fixes=[]
        )
        
        entry = CacheEntry(
            result=result,
            created_at=time.time()
        )
        
        assert entry.result == result
        assert entry.created_at > 0
    
    def test_cache_entry_is_expired(self):
        """Test cache entry expiration check."""
        result = OptimizationResult(
            original_size="100 MB",
            optimized_size="80 MB",
            explanation="test optimization",
            optimized_dockerfile="FROM python:3.9-slim",
            security_fixes=[]
        )
        
        # Create expired entry
        old_time = time.time() - 3600  # 1 hour ago
        entry = CacheEntry(result=result, created_at=old_time)
        
        assert entry.is_expired(ttl_seconds=1800)  # 30 minute TTL
        
        # Create fresh entry
        fresh_entry = CacheEntry(result=result, created_at=time.time())
        assert not fresh_entry.is_expired(ttl_seconds=1800)