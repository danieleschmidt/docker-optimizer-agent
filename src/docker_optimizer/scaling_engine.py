"""Advanced Scaling and Performance Optimization Engine.

This module provides intelligent scaling, performance optimization, and resource
management capabilities for high-throughput Docker optimization workloads.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import threading
from pathlib import Path
from queue import Queue, Empty
import multiprocessing as mp
from functools import lru_cache
import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization operations."""
    start_time: float
    end_time: Optional[float] = None
    execution_time_ms: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    operations_per_second: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    throughput_files_per_minute: float = 0.0

    def mark_complete(self) -> None:
        """Mark operation as complete and calculate metrics."""
        self.end_time = time.perf_counter()
        if self.end_time and self.start_time:
            self.execution_time_ms = (self.end_time - self.start_time) * 1000
            if self.execution_time_ms > 0:
                self.operations_per_second = 1000 / self.execution_time_ms


@dataclass 
class ScalingConfiguration:
    """Configuration for auto-scaling behavior."""
    min_workers: int = 1
    max_workers: int = mp.cpu_count()
    scale_up_threshold: float = 0.8  # Scale up when CPU > 80%
    scale_down_threshold: float = 0.3  # Scale down when CPU < 30%
    scale_up_delay_seconds: float = 30.0
    scale_down_delay_seconds: float = 60.0
    memory_threshold_mb: float = 1024.0
    queue_size_threshold: int = 100
    enable_process_pool: bool = True
    enable_thread_pool: bool = True
    adaptive_batch_sizing: bool = True
    performance_monitoring: bool = True


@dataclass
class WorkerPool:
    """Manages worker processes and threads for optimization tasks."""
    thread_executor: Optional[ThreadPoolExecutor] = None
    process_executor: Optional[ProcessPoolExecutor] = None
    active_workers: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_scale_event: float = 0.0


class AdaptiveLoadBalancer:
    """Intelligent load balancing for optimization workloads."""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.worker_pool = WorkerPool()
        self.metrics_history: List[PerformanceMetrics] = []
        self.task_queue: Queue = Queue(maxsize=config.queue_size_threshold * 2)
        self.results_queue: Queue = Queue()
        self.scaling_lock = threading.Lock()
        self.performance_monitor = PerformanceMonitor()
        self._initialize_workers()
    
    def _initialize_workers(self) -> None:
        """Initialize worker pools based on configuration."""
        if self.config.enable_thread_pool:
            self.worker_pool.thread_executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers // 2,
                thread_name_prefix="docker-optimizer-thread"
            )
            logger.info(f"Initialized thread pool with {self.config.max_workers // 2} workers")
        
        if self.config.enable_process_pool:
            self.worker_pool.process_executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers // 2,
                mp_context=mp.get_context('spawn')
            )
            logger.info(f"Initialized process pool with {self.config.max_workers // 2} workers")
    
    async def submit_optimization_batch(self, 
                                      optimization_tasks: List[Callable],
                                      batch_size: Optional[int] = None) -> List[Any]:
        """Submit a batch of optimization tasks for processing."""
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(optimization_tasks))
        
        logger.info(f"Submitting batch of {len(optimization_tasks)} optimization tasks (batch_size={batch_size})")
        
        # Monitor system resources before scaling
        await self._check_and_scale()
        
        results = []
        
        # Process tasks in batches
        for i in range(0, len(optimization_tasks), batch_size):
            batch = optimization_tasks[i:i + batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            # Update performance metrics
            await self._update_performance_metrics(len(batch))
        
        logger.info(f"Completed batch processing: {len(results)} results, {self._get_success_rate():.2%} success rate")
        return results
    
    async def _process_batch(self, tasks: List[Callable]) -> List[Any]:
        """Process a single batch of optimization tasks."""
        start_time = time.perf_counter()
        
        # Choose optimal executor based on task characteristics
        executor = self._select_optimal_executor(tasks)
        
        # Submit all tasks concurrently
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in tasks:
            if executor == self.worker_pool.thread_executor:
                future = loop.run_in_executor(executor, task)
            else:
                # For CPU-bound tasks, use process pool
                future = loop.run_in_executor(executor, task)
            futures.append(future)
        
        # Wait for all tasks to complete with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=300.0  # 5 minute timeout
            )
            
            # Process results and handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Task failed: {result}")
                    self.worker_pool.failed_tasks += 1
                    processed_results.append({"error": str(result), "success": False})
                else:
                    self.worker_pool.completed_tasks += 1
                    # Convert pydantic models to dict for consistent handling
                    if hasattr(result, 'model_dump'):
                        result_dict = result.model_dump()
                        result_dict["success"] = True
                        processed_results.append(result_dict)
                    elif hasattr(result, 'dict'):
                        result_dict = result.dict()
                        result_dict["success"] = True
                        processed_results.append(result_dict)
                    else:
                        processed_results.append({"result": str(result), "success": True})
            
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Batch completed in {execution_time:.2f}ms")
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.error("Batch processing timeout")
            return [{"error": "timeout", "success": False}] * len(tasks)
    
    def _select_optimal_executor(self, tasks: List[Callable]) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Select the optimal executor based on task characteristics."""
        # For CPU-intensive tasks (like complex optimizations), prefer processes
        # For I/O-bound tasks (like file operations), prefer threads
        
        system_load = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        if (system_load > 70 or memory_usage > 80) and self.worker_pool.process_executor:
            # High system load - use processes for better isolation
            return self.worker_pool.process_executor
        elif self.worker_pool.thread_executor:
            # Normal load - use threads for faster task switching
            return self.worker_pool.thread_executor
        else:
            # Fallback
            return self.worker_pool.process_executor or self.worker_pool.thread_executor
    
    def _calculate_optimal_batch_size(self, total_tasks: int) -> int:
        """Calculate optimal batch size based on system resources and task history."""
        if not self.config.adaptive_batch_sizing:
            return min(32, total_tasks)  # Default batch size
        
        # Analyze performance history
        if len(self.metrics_history) >= 5:
            recent_metrics = self.metrics_history[-5:]
            avg_exec_time = sum(m.execution_time_ms or 0 for m in recent_metrics) / len(recent_metrics)
            
            # Adjust batch size based on average execution time
            if avg_exec_time < 100:  # Fast tasks
                batch_size = min(64, total_tasks)
            elif avg_exec_time < 1000:  # Medium tasks
                batch_size = min(32, total_tasks)
            else:  # Slow tasks
                batch_size = min(16, total_tasks)
        else:
            batch_size = min(32, total_tasks)
        
        # Adjust for system resources
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Scale batch size with available resources
        resource_multiplier = min(2.0, max(0.5, (cpu_count * memory_gb) / 16))
        batch_size = int(batch_size * resource_multiplier)
        
        return max(1, min(batch_size, total_tasks))
    
    async def _check_and_scale(self) -> None:
        """Check system metrics and scale workers if necessary."""
        if not self.config.performance_monitoring:
            return
        
        current_time = time.time()
        
        # Avoid frequent scaling
        if current_time - self.worker_pool.last_scale_event < self.config.scale_up_delay_seconds:
            return
        
        with self.scaling_lock:
            system_metrics = self._get_system_metrics()
            
            should_scale_up = (
                system_metrics['cpu_percent'] > self.config.scale_up_threshold * 100 or
                system_metrics['memory_percent'] > 80 or
                self.task_queue.qsize() > self.config.queue_size_threshold
            )
            
            should_scale_down = (
                system_metrics['cpu_percent'] < self.config.scale_down_threshold * 100 and
                system_metrics['memory_percent'] < 50 and
                self.task_queue.qsize() < self.config.queue_size_threshold // 4
            )
            
            if should_scale_up and self.worker_pool.active_workers < self.config.max_workers:
                await self._scale_up()
                self.worker_pool.last_scale_event = current_time
            elif should_scale_down and self.worker_pool.active_workers > self.config.min_workers:
                await self._scale_down()
                self.worker_pool.last_scale_event = current_time
    
    async def _scale_up(self) -> None:
        """Scale up worker capacity."""
        if self.worker_pool.thread_executor and self.worker_pool.active_workers < self.config.max_workers:
            # In practice, ThreadPoolExecutor doesn't support dynamic scaling
            # This would require implementing custom worker pool management
            logger.info("Scaling up requested - increasing worker priority")
            self.worker_pool.active_workers += 1
    
    async def _scale_down(self) -> None:
        """Scale down worker capacity."""
        if self.worker_pool.active_workers > self.config.min_workers:
            logger.info("Scaling down workers due to low resource utilization")
            self.worker_pool.active_workers -= 1
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes,
            'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }
    
    def _get_success_rate(self) -> float:
        """Calculate current success rate."""
        total_tasks = self.worker_pool.completed_tasks + self.worker_pool.failed_tasks
        if total_tasks == 0:
            return 1.0
        return self.worker_pool.completed_tasks / total_tasks
    
    async def _update_performance_metrics(self, processed_count: int) -> None:
        """Update performance metrics after processing tasks."""
        current_metrics = PerformanceMetrics(
            start_time=time.perf_counter(),
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent(),
            throughput_files_per_minute=processed_count * 60,  # Approximate
            success_rate=self._get_success_rate()
        )
        current_metrics.mark_complete()
        
        self.metrics_history.append(current_metrics)
        
        # Keep only recent metrics (last 100 operations)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        return {
            "active_workers": self.worker_pool.active_workers,
            "completed_tasks": self.worker_pool.completed_tasks,
            "failed_tasks": self.worker_pool.failed_tasks,
            "success_rate": self._get_success_rate(),
            "average_execution_time_ms": sum(m.execution_time_ms or 0 for m in recent_metrics) / len(recent_metrics),
            "average_memory_usage_mb": sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            "average_cpu_usage_percent": sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
            "total_operations": len(self.metrics_history),
            "queue_size": self.task_queue.qsize(),
            "system_metrics": self._get_system_metrics()
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all worker pools."""
        logger.info("Shutting down adaptive load balancer")
        
        if self.worker_pool.thread_executor:
            self.worker_pool.thread_executor.shutdown(wait=True)
        
        if self.worker_pool.process_executor:
            self.worker_pool.process_executor.shutdown(wait=True)


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self):
        self.alerts_enabled = True
        self.performance_thresholds = {
            'max_execution_time_ms': 5000,
            'max_memory_usage_mb': 1024,
            'min_success_rate': 0.95,
            'max_cpu_usage_percent': 90,
            'max_queue_size': 1000
        }
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def check_performance_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Check metrics against performance thresholds."""
        alerts = []
        
        for metric, threshold in self.performance_thresholds.items():
            if metric.startswith('max_') and metrics.get(metric.replace('max_', ''), 0) > threshold:
                alerts.append(f"Performance alert: {metric.replace('max_', '')} exceeded threshold ({threshold})")
            elif metric.startswith('min_') and metrics.get(metric.replace('min_', ''), 1) < threshold:
                alerts.append(f"Performance alert: {metric.replace('min_', '')} below threshold ({threshold})")
        
        return alerts
    
    def trigger_alerts(self, alerts: List[str], context: Dict[str, Any]) -> None:
        """Trigger performance alerts."""
        if not self.alerts_enabled or not alerts:
            return
        
        for alert in alerts:
            logger.warning(alert)
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert, context)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")


class DistributedOptimizationCache:
    """High-performance distributed cache for optimization results."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}  # value, timestamp
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    @lru_cache(maxsize=1000)
    def _hash_dockerfile(self, content: str) -> str:
        """Create hash key for dockerfile content."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, dockerfile_content: str) -> Optional[Any]:
        """Get cached optimization result."""
        key = self._hash_dockerfile(dockerfile_content)
        
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self.misses += 1
                return None
            
            self._access_times[key] = time.time()
            self.hits += 1
            logger.debug(f"Cache hit for dockerfile hash {key}")
            return value
    
    def put(self, dockerfile_content: str, result: Any) -> None:
        """Cache optimization result."""
        key = self._hash_dockerfile(dockerfile_content)
        
        with self._lock:
            # Evict old entries if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = (result, time.time())
            self._access_times[key] = time.time()
            logger.debug(f"Cached result for dockerfile hash {key}")
    
    def _evict_lru(self) -> None:
        """Evict least recently used cache entry."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        
        if lru_key in self._cache:
            del self._cache[lru_key]
        del self._access_times[lru_key]
        
        logger.debug(f"Evicted LRU cache entry {lru_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")


class HighThroughputOptimizer:
    """High-throughput Docker optimization engine with intelligent scaling."""
    
    def __init__(self, scaling_config: Optional[ScalingConfiguration] = None):
        self.scaling_config = scaling_config or ScalingConfiguration()
        self.load_balancer = AdaptiveLoadBalancer(self.scaling_config)
        self.cache = DistributedOptimizationCache()
        self.performance_monitor = PerformanceMonitor()
        
        # Setup performance alerts
        self.performance_monitor.add_alert_callback(self._handle_performance_alert)
        
        logger.info("High-throughput optimizer initialized with advanced scaling")
    
    async def optimize_dockerfiles_batch(self, 
                                       dockerfile_contents: List[str],
                                       enable_caching: bool = True) -> List[Dict[str, Any]]:
        """Optimize a batch of dockerfiles with high throughput."""
        logger.info(f"Starting high-throughput optimization of {len(dockerfile_contents)} dockerfiles")
        
        start_time = time.perf_counter()
        results = []
        
        # Check cache first if enabled
        cached_results = {}
        uncached_contents = []
        
        if enable_caching:
            for i, content in enumerate(dockerfile_contents):
                cached_result = self.cache.get(content)
                if cached_result:
                    cached_results[i] = cached_result
                else:
                    uncached_contents.append((i, content))
        else:
            uncached_contents = list(enumerate(dockerfile_contents))
        
        logger.info(f"Cache hits: {len(cached_results)}, cache misses: {len(uncached_contents)}")
        
        # Process uncached dockerfiles
        if uncached_contents:
            # Create optimization tasks
            from .optimizer import DockerfileOptimizer
            optimizer = DockerfileOptimizer()
            
            optimization_tasks = []
            for i, content in uncached_contents:
                task = lambda c=content: optimizer.optimize_dockerfile(c)
                optimization_tasks.append(task)
            
            # Submit for high-throughput processing
            uncached_results = await self.load_balancer.submit_optimization_batch(optimization_tasks)
            
            # Cache results and merge with cached results
            for (i, content), result in zip(uncached_contents, uncached_results):
                if enable_caching and isinstance(result, dict) and result.get('success', True):
                    self.cache.put(content, result)
                cached_results[i] = result
        
        # Reconstruct results in original order
        for i in range(len(dockerfile_contents)):
            results.append(cached_results.get(i, {"error": "processing_failed", "success": False}))
        
        # Update performance metrics
        execution_time = (time.perf_counter() - start_time) * 1000
        throughput = len(dockerfile_contents) / (execution_time / 1000) * 60  # files per minute
        
        performance_summary = self.load_balancer.get_performance_summary()
        performance_summary.update({
            "batch_execution_time_ms": execution_time,
            "batch_throughput_files_per_minute": throughput,
            "cache_stats": self.cache.get_stats()
        })
        
        # Check performance thresholds
        alerts = self.performance_monitor.check_performance_thresholds(performance_summary)
        if alerts:
            self.performance_monitor.trigger_alerts(alerts, performance_summary)
        
        logger.info(f"High-throughput optimization completed: {len(results)} results, {throughput:.1f} files/min")
        
        return results
    
    def _handle_performance_alert(self, alert: str, context: Dict[str, Any]) -> None:
        """Handle performance alerts."""
        logger.warning(f"Performance alert triggered: {alert}")
        
        # Implement automatic performance tuning
        if "memory" in alert.lower():
            # Clear cache to free memory
            logger.info("Clearing cache due to memory pressure")
            self.cache.clear()
        
        if "cpu" in alert.lower():
            # Reduce worker count temporarily
            logger.info("Reducing worker load due to CPU pressure")
            # Implementation would adjust worker pool size
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "timestamp": time.time(),
            "load_balancer": self.load_balancer.get_performance_summary(),
            "cache": self.cache.get_stats(),
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "network_connections": len(psutil.net_connections())
            },
            "performance_thresholds": self.performance_monitor.performance_thresholds
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the high-throughput optimizer."""
        logger.info("Shutting down high-throughput optimizer")
        await self.load_balancer.shutdown()
        self.cache.clear()


# Utility functions for scaling and performance optimization
async def benchmark_scaling_performance(dockerfile_samples: List[str], 
                                      scaling_configs: List[ScalingConfiguration]) -> Dict[str, Any]:
    """Benchmark different scaling configurations."""
    logger.info(f"Benchmarking {len(scaling_configs)} scaling configurations")
    
    benchmark_results = {}
    
    for i, config in enumerate(scaling_configs):
        config_name = f"config_{i+1}"
        logger.info(f"Testing scaling configuration: {config_name}")
        
        optimizer = HighThroughputOptimizer(config)
        
        start_time = time.perf_counter()
        results = await optimizer.optimize_dockerfiles_batch(dockerfile_samples)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000
        throughput = len(dockerfile_samples) / ((end_time - start_time) * 60)  # files per minute
        success_rate = sum(1 for r in results if r.get('success', True)) / len(results)
        
        benchmark_results[config_name] = {
            "config": config,
            "execution_time_ms": execution_time,
            "throughput_files_per_minute": throughput,
            "success_rate": success_rate,
            "system_status": await optimizer.get_system_status()
        }
        
        await optimizer.shutdown()
        
        # Brief pause between configurations
        await asyncio.sleep(2)
    
    return benchmark_results


def create_high_performance_config() -> ScalingConfiguration:
    """Create a high-performance scaling configuration."""
    return ScalingConfiguration(
        min_workers=4,
        max_workers=mp.cpu_count() * 2,
        scale_up_threshold=0.7,
        scale_down_threshold=0.2,
        scale_up_delay_seconds=15.0,
        scale_down_delay_seconds=30.0,
        memory_threshold_mb=2048.0,
        queue_size_threshold=200,
        enable_process_pool=True,
        enable_thread_pool=True,
        adaptive_batch_sizing=True,
        performance_monitoring=True
    )


def create_memory_optimized_config() -> ScalingConfiguration:
    """Create a memory-optimized scaling configuration."""
    return ScalingConfiguration(
        min_workers=2,
        max_workers=mp.cpu_count(),
        scale_up_threshold=0.6,
        scale_down_threshold=0.4,
        memory_threshold_mb=512.0,
        queue_size_threshold=50,
        enable_process_pool=False,  # Use threads to save memory
        enable_thread_pool=True,
        adaptive_batch_sizing=True,
        performance_monitoring=True
    )