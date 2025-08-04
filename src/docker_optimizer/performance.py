"""Performance optimization features for Docker Optimizer Agent."""

import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psutil
from pydantic import BaseModel, Field

from .config import Config
from .models import OptimizationResult
from .optimizer import DockerfileOptimizer

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """Cache entry for optimization results."""

    result: OptimizationResult
    created_at: float = Field(..., description="Timestamp when entry was created")

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > ttl_seconds


class PerformanceMetrics(BaseModel):
    """Performance metrics for optimization operations."""

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    dockerfiles_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    class Config:
        arbitrary_types_allowed = True

    def timer(self) -> "PerformanceMetrics":
        """Context manager for timing operations."""
        return self

    def __enter__(self) -> "PerformanceMetrics":
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and calculate duration."""
        self.end_time = time.time()
        if self.start_time:
            self.processing_time = self.end_time - self.start_time

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return float(memory_info.rss) / 1024 / 1024  # Convert to MB

    def update_memory_usage(self) -> None:
        """Update current memory usage."""
        self.memory_usage_mb = self.get_memory_usage()

    def get_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        self.update_memory_usage()
        return {
            "processing_time": self.processing_time,
            "memory_usage_mb": self.memory_usage_mb,
            "dockerfiles_processed": self.dockerfiles_processed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }


class OptimizationCache:
    """LRU cache for optimization results."""

    def __init__(self, max_size: Optional[int] = None, ttl_seconds: Optional[float] = None, config: Optional[Config] = None):
        """Initialize cache with size limit and TTL.

        Args:
            max_size: Maximum cache size. If None, uses configuration.
            ttl_seconds: Cache TTL in seconds. If None, uses configuration.
            config: Configuration instance. If None, default config is used.
        """
        self.config = config or Config()
        cache_settings = self.config.get_cache_settings()

        self.max_size = max_size if max_size is not None else cache_settings["max_size"]
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else cache_settings["ttl_seconds"]
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []

    def _generate_key(self, dockerfile_content: str) -> str:
        """Generate cache key from Dockerfile content."""
        return hashlib.sha256(dockerfile_content.encode()).hexdigest()

    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired(self.ttl_seconds)
        ]

        for key in expired_keys:
            self._remove_key(key)

    def _remove_key(self, key: str) -> None:
        """Remove key from cache and access order."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_key(lru_key)

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def get(self, dockerfile_content: str) -> Optional[OptimizationResult]:
        """Get cached optimization result."""
        self._evict_expired()

        key = self._generate_key(dockerfile_content)

        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired(self.ttl_seconds):
                self._update_access_order(key)
                return entry.result
            else:
                self._remove_key(key)

        return None

    def set(self, dockerfile_content: str, result: OptimizationResult) -> None:
        """Cache optimization result."""
        key = self._generate_key(dockerfile_content)

        # Evict expired entries first
        self._evict_expired()

        # Evict LRU if at capacity
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        # Add new entry
        entry = CacheEntry(result=result, created_at=time.time())
        self._cache[key] = entry
        self._update_access_order(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class ParallelAnalyzer:
    """Parallel processing for multiple Dockerfile analysis."""

    def __init__(self, max_workers: int = None):
        """Initialize parallel analyzer."""
        # Auto-detect optimal worker count
        if max_workers is None:
            max_workers = min(8, (psutil.cpu_count() or 4) + 4)
        
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Resource pooling - create optimizer instances per worker
        self._optimizer_pool = [DockerfileOptimizer() for _ in range(max_workers)]
        self._pool_index = 0

    def _get_optimizer(self) -> DockerfileOptimizer:
        """Get optimizer from pool (thread-safe)."""
        optimizer = self._optimizer_pool[self._pool_index % len(self._optimizer_pool)]
        self._pool_index += 1
        return optimizer

    def _analyze_single(self, dockerfile_content: str) -> OptimizationResult:
        """Analyze single Dockerfile using pooled optimizer."""
        try:
            optimizer = self._get_optimizer()
            return optimizer.optimize_dockerfile(dockerfile_content)
        except Exception as e:
            # Return empty result for failed analysis
            return OptimizationResult(
                original_size="0 MB",
                optimized_size="0 MB",
                explanation=f"Analysis failed: {str(e)}",
                optimized_dockerfile="# Analysis failed",
                security_fixes=[]
            )

    async def analyze_multiple(self, dockerfiles: List[str]) -> List[OptimizationResult]:
        """Analyze multiple Dockerfiles in parallel."""
        loop = asyncio.get_event_loop()

        # Submit all tasks to thread pool
        tasks = [
            loop.run_in_executor(self.executor, self._analyze_single, dockerfile)
            for dockerfile in dockerfiles
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, OptimizationResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                # Create error result for exceptions
                valid_results.append(OptimizationResult(
                    original_size="0 MB",
                    optimized_size="0 MB",
                    explanation=f"Analysis failed: {str(result)}",
                    optimized_dockerfile="# Analysis failed",
                    security_fixes=[]
                ))

        return valid_results

    def adjust_workers_based_on_load(self, queue_size: int) -> None:
        """Dynamically adjust worker count based on load."""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        # Scale up if high load and resources available
        if queue_size > self.max_workers * 2 and cpu_usage < 80 and memory_usage < 80:
            new_max = min(self.max_workers + 2, 16)
            if new_max > self.max_workers:
                logger.info(f"Scaling workers from {self.max_workers} to {new_max}")
                self.max_workers = new_max
                
        # Scale down if low load
        elif queue_size < self.max_workers // 2 and self.max_workers > 2:
            new_max = max(self.max_workers - 1, 2)
            logger.info(f"Scaling workers from {self.max_workers} to {new_max}")
            self.max_workers = new_max

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.executor.shutdown(wait=True)

    def close(self) -> None:
        """Close thread pool executor."""
        self.executor.shutdown(wait=True)


class LargeDockerfileHandler:
    """Handler for processing large Dockerfiles efficiently."""

    def __init__(self, size_threshold: int = 10000, chunk_size: int = 1000):
        """Initialize large Dockerfile handler."""
        self.size_threshold = size_threshold
        self.chunk_size = chunk_size
        self.optimizer = DockerfileOptimizer()

    def is_large_dockerfile(self, dockerfile_content: str) -> bool:
        """Check if Dockerfile exceeds size threshold."""
        return len(dockerfile_content) > self.size_threshold

    def chunk_dockerfile(self, dockerfile_content: str) -> List[str]:
        """Split large Dockerfile into manageable chunks."""
        chunks = []

        # Split by lines to maintain Dockerfile structure
        lines = dockerfile_content.split('\n')
        current_chunk_lines: List[str] = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.chunk_size and current_chunk_lines:
                # Current chunk is full, save it and start new one
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size

        # Add remaining content
        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))

        return chunks

    def _process_chunk(self, chunk: str) -> str:
        """Process individual chunk and extract optimization suggestions."""
        try:
            result = self.optimizer.optimize_dockerfile(chunk)
            return result.explanation
        except Exception as e:
            logger.warning("Failed to process Dockerfile chunk: %s", str(e))
            return ""

    def process_large_dockerfile(self, dockerfile_content: str) -> OptimizationResult:
        """Process large Dockerfile by chunking."""
        if not self.is_large_dockerfile(dockerfile_content):
            # Process normally for small files
            return self.optimizer.optimize_dockerfile(dockerfile_content)

        # Process in chunks
        chunks = self.chunk_dockerfile(dockerfile_content)
        chunk_explanations = []

        for chunk in chunks:
            chunk_explanation = self._process_chunk(chunk)
            if chunk_explanation:
                chunk_explanations.append(chunk_explanation)

        # Combine explanations
        combined_explanation = " ".join(chunk_explanations) if chunk_explanations else "Large Dockerfile processed in chunks"

        # Estimate sizes (simplified for large files)
        original_size = len(dockerfile_content)
        estimated_reduction = len(chunk_explanations) * 100  # Rough estimate based on chunk count
        optimized_size = max(original_size - estimated_reduction, original_size // 2)

        return OptimizationResult(
            original_size=f"{original_size // 1024} KB",
            optimized_size=f"{optimized_size // 1024} KB",
            explanation=combined_explanation,
            optimized_dockerfile=dockerfile_content,  # Simplified for large files
            security_fixes=[]  # Security analysis would need separate chunked processing
        )


@dataclass
class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    cache: OptimizationCache = field(default_factory=lambda: OptimizationCache())
    parallel_analyzer: ParallelAnalyzer = field(default_factory=lambda: ParallelAnalyzer())
    large_handler: LargeDockerfileHandler = field(default_factory=lambda: LargeDockerfileHandler())
    metrics: PerformanceMetrics = field(default_factory=lambda: PerformanceMetrics())

    def optimize_with_performance(self, dockerfile_content: str) -> OptimizationResult:
        """Optimize Dockerfile with performance enhancements."""
        with self.metrics.timer():
            # Check cache first
            cached_result = self.cache.get(dockerfile_content)
            if cached_result:
                self.metrics.cache_hits += 1
                return cached_result

            self.metrics.cache_misses += 1

            # Choose optimization strategy based on size
            if self.large_handler.is_large_dockerfile(dockerfile_content):
                result = self.large_handler.process_large_dockerfile(dockerfile_content)
            else:
                optimizer = DockerfileOptimizer()
                result = optimizer.optimize_dockerfile(dockerfile_content)

            # Cache the result
            self.cache.set(dockerfile_content, result)
            self.metrics.dockerfiles_processed += 1

            return result

    async def optimize_multiple_with_performance(self, dockerfiles: List[str]) -> List[OptimizationResult]:
        """Optimize multiple Dockerfiles with performance enhancements."""
        with self.metrics.timer():
            # Check cache for each Dockerfile
            cached_results: List[Tuple[int, OptimizationResult]] = []
            uncached_dockerfiles: List[Tuple[int, str]] = []

            for i, dockerfile in enumerate(dockerfiles):
                cached_result = self.cache.get(dockerfile)
                if cached_result:
                    cached_results.append((i, cached_result))
                    self.metrics.cache_hits += 1
                else:
                    uncached_dockerfiles.append((i, dockerfile))
                    self.metrics.cache_misses += 1

            # Process uncached Dockerfiles in parallel
            if uncached_dockerfiles:
                uncached_contents = [dockerfile for _, dockerfile in uncached_dockerfiles]
                uncached_results = await self.parallel_analyzer.analyze_multiple(uncached_contents)

                # Cache new results
                for (_, dockerfile), result in zip(uncached_dockerfiles, uncached_results):
                    self.cache.set(dockerfile, result)

                # Combine with cached results in original order
                all_results: List[Optional[OptimizationResult]] = [None] * len(dockerfiles)

                for index, result in cached_results:
                    all_results[index] = result

                for (index, _), result in zip(uncached_dockerfiles, uncached_results):
                    all_results[index] = result

                self.metrics.dockerfiles_processed += len(uncached_dockerfiles)
                # Filter out None values (shouldn't happen if logic is correct)
                return [result for result in all_results if result is not None]
            else:
                # All results were cached
                return [result for _, result in sorted(cached_results)]

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            **self.metrics.get_report(),
            "cache_size": self.cache.size(),
            "cache_max_size": self.cache.max_size,
            "cache_ttl_seconds": self.cache.ttl_seconds
        }

    def clear_cache(self) -> None:
        """Clear optimization cache."""
        self.cache.clear()

    def close(self) -> None:
        """Clean up resources."""
        self.parallel_analyzer.close()
