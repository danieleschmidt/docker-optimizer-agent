"""Adaptive performance optimization engine for Docker Optimizer Agent.

Provides intelligent performance optimization, adaptive caching, resource
optimization, and autonomous performance tuning capabilities.
"""

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import psutil

from .enhanced_error_handling import retry_on_failure, with_circuit_breaker


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class CacheStrategy(Enum):
    """Caching strategy types."""
    LRU = "lru"
    FIFO = "fifo"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ResourceProfile(Enum):
    """System resource profiles for optimization."""
    LOW_RESOURCE = "low_resource"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    MEMORY_OPTIMIZED = "memory_optimized"
    CPU_OPTIMIZED = "cpu_optimized"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_usage_before: float = 0.0
    cpu_usage_after: float = 0.0
    memory_usage_before: float = 0.0
    memory_usage_after: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    throughput: float = 0.0  # Operations per second
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if self.duration and self.duration > 0:
            self.throughput = 1.0 / self.duration


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> None:
        """Update access tracking."""
        self.last_accessed = time.time()
        self.access_count += 1


class AdaptiveCache:
    """Adaptive caching system with multiple strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600,  # 1 hour
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque(maxlen=max_size)
        self.frequency_counter: defaultdict = defaultdict(int)
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive strategy learning
        self.strategy_performance: Dict[CacheStrategy, float] = {
            strategy: 0.0 for strategy in CacheStrategy if strategy != CacheStrategy.ADAPTIVE
        }
        self.current_strategy = CacheStrategy.LRU
        self.strategy_switch_threshold = 100  # Operations before strategy evaluation
        self.operations_since_evaluation = 0
        
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Update access patterns
                entry.access()
                self.hits += 1
                
                # Update strategy-specific data structures
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.frequency_counter[key] += 1
                
                return entry.value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Calculate size (rough estimation)
            try:
                size_bytes = len(json.dumps(value, default=str).encode('utf-8'))
            except:
                size_bytes = 1024  # Default estimate
            
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # If cache is full, evict based on current strategy
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict()
            
            self.cache[key] = entry
            
            # Update access patterns
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.frequency_counter[key] += 1
            
            # Adaptive strategy evaluation
            self.operations_since_evaluation += 1
            if self.operations_since_evaluation >= self.strategy_switch_threshold:
                self._evaluate_and_adapt_strategy()
    
    def _evict(self) -> None:
        """Evict entry based on current strategy."""
        if not self.cache:
            return
        
        key_to_evict = None
        
        if self.current_strategy == CacheStrategy.LRU:
            # Least Recently Used
            key_to_evict = next(iter(self.access_order))
        
        elif self.current_strategy == CacheStrategy.FIFO:
            # First In, First Out
            key_to_evict = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        
        elif self.current_strategy == CacheStrategy.LFU:
            # Least Frequently Used
            key_to_evict = min(self.cache.keys(), key=lambda k: self.frequency_counter[k])
        
        elif self.current_strategy == CacheStrategy.TTL:
            # TTL-based eviction (oldest expiring first)
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                # Fallback to LRU if no expired entries
                key_to_evict = next(iter(self.access_order))
        
        if key_to_evict and key_to_evict in self.cache:
            del self.cache[key_to_evict]
            if key_to_evict in self.access_order:
                self.access_order.remove(key_to_evict)
            del self.frequency_counter[key_to_evict]
            self.evictions += 1
    
    def _evaluate_and_adapt_strategy(self) -> None:
        """Evaluate current strategy performance and adapt if needed."""
        current_hit_rate = self.get_hit_rate()
        
        # Record current strategy performance
        self.strategy_performance[self.current_strategy] = current_hit_rate
        
        # Find best performing strategy
        best_strategy = max(
            self.strategy_performance.keys(),
            key=lambda s: self.strategy_performance[s]
        )
        
        # Switch strategy if significant improvement is possible
        improvement_threshold = 0.05  # 5% improvement required
        if (self.strategy_performance[best_strategy] - current_hit_rate) > improvement_threshold:
            self.current_strategy = best_strategy
            logging.info(f"Switched cache strategy to {best_strategy.value} (hit rate: {current_hit_rate:.3f})")
        
        self.operations_since_evaluation = 0
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_operations = self.hits + self.misses
        return self.hits / total_operations if total_operations > 0 else 0.0
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            "strategy": self.current_strategy.value,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.get_hit_rate(),
            "total_size_bytes": total_size,
            "average_entry_size": total_size / len(self.cache) if self.cache else 0,
            "strategy_performance": {
                s.value: perf for s, perf in self.strategy_performance.items()
            }
        }


class ResourceOptimizer:
    """System resource optimization and monitoring."""
    
    def __init__(self):
        self.resource_profile = ResourceProfile.BALANCED
        self.monitoring_interval = 5.0
        self.optimization_history: List[Dict[str, Any]] = []
        self.resource_thresholds = {
            ResourceProfile.LOW_RESOURCE: {
                'cpu_limit': 50.0,
                'memory_limit': 60.0,
                'max_workers': 2
            },
            ResourceProfile.BALANCED: {
                'cpu_limit': 70.0,
                'memory_limit': 75.0,
                'max_workers': 4
            },
            ResourceProfile.HIGH_PERFORMANCE: {
                'cpu_limit': 90.0,
                'memory_limit': 85.0,
                'max_workers': 8
            }
        }
        
        self._monitoring = False
        self._monitor_task = None
    
    def detect_resource_profile(self) -> ResourceProfile:
        """Automatically detect optimal resource profile."""
        try:
            # Get system information
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Simple heuristics for profile detection
            if cpu_count <= 2 or memory_gb <= 4:
                return ResourceProfile.LOW_RESOURCE
            elif cpu_count >= 8 and memory_gb >= 16:
                return ResourceProfile.HIGH_PERFORMANCE
            else:
                return ResourceProfile.BALANCED
                
        except Exception:
            return ResourceProfile.BALANCED
    
    def get_optimal_worker_count(self) -> int:
        """Get optimal worker count for current system."""
        profile_settings = self.resource_thresholds[self.resource_profile]
        base_workers = profile_settings['max_workers']
        
        # Adjust based on current system load
        current_cpu = psutil.cpu_percent(interval=1)
        current_memory = psutil.virtual_memory().percent
        
        cpu_limit = profile_settings['cpu_limit']
        memory_limit = profile_settings['memory_limit']
        
        # Scale down workers if resources are constrained
        if current_cpu > cpu_limit or current_memory > memory_limit:
            return max(1, base_workers // 2)
        
        return base_workers
    
    def should_use_multiprocessing(self, task_type: str = "default") -> bool:
        """Determine if multiprocessing should be used."""
        # CPU-intensive tasks benefit from multiprocessing
        cpu_intensive_tasks = {"docker_build", "security_scan", "optimization_analysis"}
        
        if task_type in cpu_intensive_tasks:
            return psutil.cpu_count() > 2
        
        return False
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_resources())
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
    
    async def _monitor_resources(self) -> None:
        """Monitor system resources continuously."""
        while self._monitoring:
            try:
                # Collect metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'profile': self.resource_profile.value
                }
                
                self.optimization_history.append(metrics)
                
                # Keep only recent history
                if len(self.optimization_history) > 1000:
                    self.optimization_history = self.optimization_history[-500:]
                
                # Adaptive profile adjustment
                self._adapt_resource_profile(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    def _adapt_resource_profile(self, current_metrics: Dict[str, Any]) -> None:
        """Adapt resource profile based on current conditions."""
        cpu_percent = current_metrics['cpu_percent']
        memory_percent = current_metrics['memory_percent']
        
        # Get recent average
        if len(self.optimization_history) >= 10:
            recent_cpu = sum(m['cpu_percent'] for m in self.optimization_history[-10:]) / 10
            recent_memory = sum(m['memory_percent'] for m in self.optimization_history[-10:]) / 10
            
            # Adaptive switching logic
            if recent_cpu > 80 or recent_memory > 80:
                # System under stress, switch to conservative profile
                if self.resource_profile != ResourceProfile.LOW_RESOURCE:
                    self.resource_profile = ResourceProfile.LOW_RESOURCE
                    logging.info("Switched to LOW_RESOURCE profile due to high system load")
            
            elif recent_cpu < 40 and recent_memory < 50:
                # System has capacity, switch to performance profile
                if self.resource_profile != ResourceProfile.HIGH_PERFORMANCE:
                    self.resource_profile = ResourceProfile.HIGH_PERFORMANCE
                    logging.info("Switched to HIGH_PERFORMANCE profile due to available resources")


class AdaptivePerformanceEngine:
    """Main adaptive performance optimization engine."""
    
    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE,
        cache_size: int = 1000,
        enable_monitoring: bool = True
    ):
        self.optimization_level = optimization_level
        self.enable_monitoring = enable_monitoring
        
        # Core components
        self.cache = AdaptiveCache(max_size=cache_size)
        self.resource_optimizer = ResourceOptimizer()
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_counts: defaultdict = defaultdict(int)
        
        # Execution pools
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # Configuration
        self.batch_size_limits = {
            OptimizationLevel.BASIC: 5,
            OptimizationLevel.STANDARD: 10,
            OptimizationLevel.AGGRESSIVE: 25,
            OptimizationLevel.ADAPTIVE: 50
        }
        
        self.parallelism_limits = {
            OptimizationLevel.BASIC: 2,
            OptimizationLevel.STANDARD: 4,
            OptimizationLevel.AGGRESSIVE: 8,
            OptimizationLevel.ADAPTIVE: 16
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the performance engine."""
        self.logger.info("Initializing Adaptive Performance Engine")
        
        # Auto-detect resource profile
        detected_profile = self.resource_optimizer.detect_resource_profile()
        self.resource_optimizer.resource_profile = detected_profile
        self.logger.info(f"Detected resource profile: {detected_profile.value}")
        
        # Initialize execution pools
        max_workers = self.resource_optimizer.get_optimal_worker_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        
        # Start resource monitoring
        if self.enable_monitoring:
            await self.resource_optimizer.start_monitoring()
    
    async def shutdown(self) -> None:
        """Shutdown the performance engine gracefully."""
        self.logger.info("Shutting down Adaptive Performance Engine")
        
        # Stop monitoring
        if self.enable_monitoring:
            self.resource_optimizer.stop_monitoring()
        
        # Shutdown execution pools
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
    
    def with_performance_tracking(self, operation_name: str):
        """Decorator to track operation performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                metrics = PerformanceMetrics(operation_name=operation_name)
                
                # Capture initial resource usage
                metrics.cpu_usage_before = psutil.cpu_percent(interval=None)
                metrics.memory_usage_before = psutil.virtual_memory().percent
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    return result
                    
                except Exception as e:
                    metrics.error_count += 1
                    raise
                    
                finally:
                    # Capture final resource usage
                    metrics.cpu_usage_after = psutil.cpu_percent(interval=None)
                    metrics.memory_usage_after = psutil.virtual_memory().percent
                    metrics.finalize()
                    
                    # Track operation
                    self.operation_counts[operation_name] += 1
                    self.metrics_history.append(metrics)
                    
                    # Keep history limited
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-500:]
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    @retry_on_failure(max_attempts=3, delay=1.0)
    async def optimize_batch_processing(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Optimize batch processing with adaptive sizing."""
        if not items:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(items))
        
        # Determine processing strategy
        use_multiprocessing = self.resource_optimizer.should_use_multiprocessing("batch_processing")
        max_workers = self.resource_optimizer.get_optimal_worker_count()
        
        results = []
        
        if use_multiprocessing and self.process_executor:
            # Process-based parallelism for CPU-intensive tasks
            futures = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                future = self.process_executor.submit(self._process_batch, batch, processor_func)
                futures.append(future)
            
            # Collect results as they complete
            for future in futures:
                batch_result = await asyncio.wrap_future(future)
                results.extend(batch_result)
        
        else:
            # Thread-based parallelism for I/O-bound tasks
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_batch_with_semaphore(batch):
                async with semaphore:
                    return await self._process_batch_async(batch, processor_func)
            
            tasks = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                task = asyncio.create_task(process_batch_with_semaphore(batch))
                tasks.append(task)
            
            # Wait for all tasks to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    self.logger.error(f"Batch processing error: {batch_result}")
                    continue
                results.extend(batch_result)
        
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on system resources and optimization level."""
        base_batch_size = self.batch_size_limits[self.optimization_level]
        
        # Adjust based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb > 8:
            multiplier = 2
        elif available_memory_gb > 4:
            multiplier = 1.5
        else:
            multiplier = 1
        
        optimal_size = int(base_batch_size * multiplier)
        
        # Don't exceed total items or reasonable limits
        return min(optimal_size, total_items, 100)
    
    def _process_batch(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process batch synchronously (for ProcessPoolExecutor)."""
        results = []
        for item in batch:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing item: {e}")
                results.append(None)  # Or handle error appropriately
        return results
    
    async def _process_batch_async(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process batch asynchronously."""
        results = []
        for item in batch:
            try:
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(item)
                else:
                    result = processor_func(item)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing item: {e}")
                results.append(None)  # Or handle error appropriately
        return results
    
    @with_circuit_breaker("cache_operation")
    def cached_operation(self, key: str, operation: Callable, ttl: Optional[float] = None) -> Any:
        """Execute operation with caching."""
        # Try to get from cache first
        cached_result = self.cache.get(key)
        if cached_result is not None:
            return cached_result
        
        # Execute operation
        result = operation()
        
        # Cache the result
        self.cache.put(key, result, ttl=ttl)
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        # Calculate aggregated metrics
        total_operations = len(self.metrics_history)
        avg_duration = sum(m.duration or 0 for m in self.metrics_history) / total_operations
        
        # Resource usage trends
        avg_cpu_before = sum(m.cpu_usage_before for m in self.metrics_history) / total_operations
        avg_cpu_after = sum(m.cpu_usage_after for m in self.metrics_history) / total_operations
        avg_memory_before = sum(m.memory_usage_before for m in self.metrics_history) / total_operations
        avg_memory_after = sum(m.memory_usage_after for m in self.metrics_history) / total_operations
        
        # Error rates
        total_errors = sum(m.error_count for m in self.metrics_history)
        error_rate = total_errors / total_operations if total_operations > 0 else 0
        
        # Throughput calculation
        recent_metrics = self.metrics_history[-100:]  # Last 100 operations
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            "summary": {
                "total_operations": total_operations,
                "average_duration_seconds": avg_duration,
                "average_throughput_ops_per_second": avg_throughput,
                "error_rate": error_rate,
                "optimization_level": self.optimization_level.value
            },
            "resource_usage": {
                "average_cpu_before": avg_cpu_before,
                "average_cpu_after": avg_cpu_after,
                "cpu_impact": avg_cpu_after - avg_cpu_before,
                "average_memory_before": avg_memory_before,
                "average_memory_after": avg_memory_after,
                "memory_impact": avg_memory_after - avg_memory_before
            },
            "operation_counts": dict(self.operation_counts),
            "cache_stats": self.cache.get_stats(),
            "resource_profile": self.resource_optimizer.resource_profile.value,
            "timestamp": time.time()
        }
    
    async def auto_tune_performance(self) -> Dict[str, Any]:
        """Automatically tune performance settings based on historical data."""
        if len(self.metrics_history) < 10:
            return {"message": "Insufficient data for auto-tuning"}
        
        tuning_results = {}
        
        # Analyze recent performance trends
        recent_metrics = self.metrics_history[-100:]
        avg_duration = sum(m.duration or 0 for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_count for m in recent_metrics) / len(recent_metrics)
        
        # Tune optimization level based on performance
        if avg_error_rate > 0.1:  # High error rate
            if self.optimization_level != OptimizationLevel.BASIC:
                self.optimization_level = OptimizationLevel.BASIC
                tuning_results["optimization_level"] = "Reduced to BASIC due to high error rate"
        
        elif avg_duration > 10.0:  # Slow operations
            if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                self.optimization_level = OptimizationLevel.STANDARD
                tuning_results["optimization_level"] = "Reduced to STANDARD due to slow performance"
        
        else:  # Good performance
            if self.optimization_level == OptimizationLevel.BASIC:
                self.optimization_level = OptimizationLevel.STANDARD
                tuning_results["optimization_level"] = "Increased to STANDARD due to stable performance"
        
        # Tune cache settings
        hit_rate = self.cache.get_hit_rate()
        if hit_rate < 0.5 and self.cache.max_size < 2000:
            self.cache.max_size = min(self.cache.max_size * 2, 2000)
            tuning_results["cache_size"] = f"Increased cache size to {self.cache.max_size}"
        
        return {
            "tuning_results": tuning_results,
            "new_optimization_level": self.optimization_level.value,
            "cache_hit_rate": hit_rate,
            "average_duration": avg_duration,
            "timestamp": time.time()
        }