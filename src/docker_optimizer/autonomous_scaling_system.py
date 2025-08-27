"""Autonomous scaling system for Docker Optimizer Agent.

Provides intelligent auto-scaling, load balancing, and resource management
for high-throughput optimization workloads.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Coroutine
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import psutil

from .adaptive_performance_engine import AdaptivePerformanceEngine
from .enhanced_error_handling import EnhancedErrorHandler, retry_on_failure


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadMetricType(Enum):
    """Types of load metrics for scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class WorkerType(Enum):
    """Types of workers that can be scaled."""
    DOCKERFILE_PROCESSOR = "dockerfile_processor"
    SECURITY_SCANNER = "security_scanner"
    OPTIMIZATION_ENGINE = "optimization_engine"
    BATCH_PROCESSOR = "batch_processor"
    VALIDATION_ENGINE = "validation_engine"


@dataclass
class ScalingMetric:
    """Individual scaling metric data point."""
    metric_type: LoadMetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    threshold_exceeded: bool = False
    weight: float = 1.0  # Importance weight for decision making


@dataclass
class ScalingDecision:
    """Scaling decision with justification."""
    direction: ScalingDirection
    magnitude: int  # Number of workers to add/remove
    confidence: float  # Confidence in the decision (0.0 - 1.0)
    reason: str
    metrics_snapshot: Dict[LoadMetricType, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerPool:
    """Dynamic worker pool for specific task types."""
    worker_type: WorkerType
    min_workers: int = 1
    max_workers: int = 10
    current_workers: int = 1
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    workers: List[asyncio.Task] = field(default_factory=list)
    
    @property
    def utilization(self) -> float:
        """Calculate current worker utilization."""
        if self.current_workers == 0:
            return 0.0
        return self.active_tasks / self.current_workers
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average task execution time."""
        if self.completed_tasks == 0:
            return 0.0
        return self.total_execution_time / self.completed_tasks
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total_tasks = self.completed_tasks + self.failed_tasks
        if total_tasks == 0:
            return 1.0
        return self.completed_tasks / total_tasks


class AutoScalingController:
    """Autonomous scaling controller for worker pools."""
    
    def __init__(
        self,
        check_interval: float = 10.0,
        scaling_cooldown: float = 60.0,
        metric_window_size: int = 12  # Number of metric samples to consider
    ):
        self.check_interval = check_interval
        self.scaling_cooldown = scaling_cooldown
        self.metric_window_size = metric_window_size
        
        # Scaling thresholds
        self.scale_up_thresholds = {
            LoadMetricType.CPU_USAGE: 75.0,
            LoadMetricType.MEMORY_USAGE: 80.0,
            LoadMetricType.QUEUE_LENGTH: 10.0,
            LoadMetricType.RESPONSE_TIME: 5.0,  # seconds
            LoadMetricType.ERROR_RATE: 0.05,  # 5%
            LoadMetricType.THROUGHPUT: 0.5  # Inverse threshold (scale up when throughput is low)
        }
        
        self.scale_down_thresholds = {
            LoadMetricType.CPU_USAGE: 30.0,
            LoadMetricType.MEMORY_USAGE: 40.0,
            LoadMetricType.QUEUE_LENGTH: 2.0,
            LoadMetricType.RESPONSE_TIME: 1.0,  # seconds
            LoadMetricType.ERROR_RATE: 0.01,  # 1%
            LoadMetricType.THROUGHPUT: 2.0
        }
        
        # Metric weights for decision making
        self.metric_weights = {
            LoadMetricType.CPU_USAGE: 1.0,
            LoadMetricType.MEMORY_USAGE: 1.0,
            LoadMetricType.QUEUE_LENGTH: 1.5,  # Higher weight for queue length
            LoadMetricType.RESPONSE_TIME: 1.2,
            LoadMetricType.ERROR_RATE: 2.0,  # Highest weight for error rate
            LoadMetricType.THROUGHPUT: 1.0
        }
        
        # Metrics history
        self.metrics_history: Dict[LoadMetricType, deque] = {
            metric_type: deque(maxlen=metric_window_size)
            for metric_type in LoadMetricType
        }
        
        self.scaling_history: List[ScalingDecision] = []
        self.last_scaling_time = 0.0
        self.monitoring = False
        self.logger = logging.getLogger(__name__)
    
    def add_metric(self, metric: ScalingMetric) -> None:
        """Add a new metric data point."""
        self.metrics_history[metric.metric_type].append(metric)
    
    def get_metric_trend(self, metric_type: LoadMetricType) -> Tuple[float, str]:
        """Calculate trend for a specific metric type."""
        history = self.metrics_history[metric_type]
        if len(history) < 3:
            return 0.0, "insufficient_data"
        
        # Simple linear trend calculation
        values = [m.value for m in list(history)[-5:]]  # Last 5 values
        if len(values) < 2:
            return 0.0, "insufficient_data"
        
        # Calculate slope
        x = list(range(len(values)))
        y = values
        
        n = len(values)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0, "no_variance"
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.1:
            return slope, "increasing"
        elif slope < -0.1:
            return slope, "decreasing"
        else:
            return slope, "stable"
    
    def make_scaling_decision(self, worker_pool: WorkerPool) -> Optional[ScalingDecision]:
        """Make scaling decision based on current metrics."""
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return None  # Still in cooldown period
        
        # Collect current metrics
        current_metrics = {}
        scale_up_votes = 0
        scale_down_votes = 0
        total_weighted_score = 0.0
        
        for metric_type in LoadMetricType:
            history = self.metrics_history[metric_type]
            if not history:
                continue
            
            # Get recent average
            recent_values = [m.value for m in list(history)[-3:]]
            if not recent_values:
                continue
            
            avg_value = sum(recent_values) / len(recent_values)
            current_metrics[metric_type] = avg_value
            
            # Check thresholds
            weight = self.metric_weights[metric_type]
            
            # Scale up conditions
            if metric_type in self.scale_up_thresholds:
                if avg_value > self.scale_up_thresholds[metric_type]:
                    scale_up_votes += weight
                    total_weighted_score += weight
            
            # Scale down conditions  
            if metric_type in self.scale_down_thresholds:
                if avg_value < self.scale_down_thresholds[metric_type]:
                    scale_down_votes += weight
                    total_weighted_score += weight
        
        if total_weighted_score == 0:
            return None  # No clear signal
        
        # Make decision based on weighted votes
        scale_up_ratio = scale_up_votes / total_weighted_score
        scale_down_ratio = scale_down_votes / total_weighted_score
        
        decision_threshold = 0.6  # 60% confidence required
        
        if scale_up_ratio > decision_threshold and worker_pool.current_workers < worker_pool.max_workers:
            # Calculate scaling magnitude
            magnitude = self._calculate_scaling_magnitude(
                worker_pool, ScalingDirection.UP, scale_up_ratio
            )
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                magnitude=magnitude,
                confidence=scale_up_ratio,
                reason=f"High load detected (confidence: {scale_up_ratio:.2f})",
                metrics_snapshot=current_metrics.copy()
            )
        
        elif scale_down_ratio > decision_threshold and worker_pool.current_workers > worker_pool.min_workers:
            # Calculate scaling magnitude
            magnitude = self._calculate_scaling_magnitude(
                worker_pool, ScalingDirection.DOWN, scale_down_ratio
            )
            
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                magnitude=magnitude,
                confidence=scale_down_ratio,
                reason=f"Low load detected (confidence: {scale_down_ratio:.2f})",
                metrics_snapshot=current_metrics.copy()
            )
        
        return None  # No scaling needed
    
    def _calculate_scaling_magnitude(
        self, 
        worker_pool: WorkerPool, 
        direction: ScalingDirection, 
        confidence: float
    ) -> int:
        """Calculate how many workers to add/remove."""
        base_magnitude = 1
        
        # Adjust magnitude based on confidence and current pool size
        if confidence > 0.8:
            base_magnitude = max(2, worker_pool.current_workers // 4)
        elif confidence > 0.7:
            base_magnitude = max(1, worker_pool.current_workers // 6)
        
        # Ensure we don't exceed pool limits
        if direction == ScalingDirection.UP:
            return min(base_magnitude, worker_pool.max_workers - worker_pool.current_workers)
        else:
            return min(base_magnitude, worker_pool.current_workers - worker_pool.min_workers)


class AutonomousScalingSystem:
    """Complete autonomous scaling system for Docker optimization workloads."""
    
    def __init__(
        self,
        performance_engine: Optional[AdaptivePerformanceEngine] = None,
        error_handler: Optional[EnhancedErrorHandler] = None
    ):
        self.performance_engine = performance_engine or AdaptivePerformanceEngine()
        self.error_handler = error_handler or EnhancedErrorHandler()
        
        # Worker pools for different task types
        self.worker_pools: Dict[WorkerType, WorkerPool] = {
            WorkerType.DOCKERFILE_PROCESSOR: WorkerPool(
                worker_type=WorkerType.DOCKERFILE_PROCESSOR,
                min_workers=2,
                max_workers=8
            ),
            WorkerType.SECURITY_SCANNER: WorkerPool(
                worker_type=WorkerType.SECURITY_SCANNER,
                min_workers=1,
                max_workers=4
            ),
            WorkerType.OPTIMIZATION_ENGINE: WorkerPool(
                worker_type=WorkerType.OPTIMIZATION_ENGINE,
                min_workers=2,
                max_workers=6
            ),
            WorkerType.BATCH_PROCESSOR: WorkerPool(
                worker_type=WorkerType.BATCH_PROCESSOR,
                min_workers=1,
                max_workers=10
            ),
            WorkerType.VALIDATION_ENGINE: WorkerPool(
                worker_type=WorkerType.VALIDATION_ENGINE,
                min_workers=1,
                max_workers=4
            )
        }
        
        # Scaling controllers
        self.scaling_controllers: Dict[WorkerType, AutoScalingController] = {
            worker_type: AutoScalingController()
            for worker_type in WorkerType
        }
        
        # System-wide metrics
        self.system_metrics = {
            "total_tasks_processed": 0,
            "total_scaling_actions": 0,
            "average_response_time": 0.0,
            "current_total_workers": 0,
            "peak_workers": 0
        }
        
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the autonomous scaling system."""
        self.logger.info("Starting Autonomous Scaling System")
        
        # Initialize performance engine
        await self.performance_engine.initialize()
        
        # Start initial workers for each pool
        for worker_type, pool in self.worker_pools.items():
            await self._scale_worker_pool(pool, pool.min_workers)
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Autonomous Scaling System started successfully")
    
    async def stop(self) -> None:
        """Stop the scaling system gracefully."""
        self.logger.info("Stopping Autonomous Scaling System")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all workers
        for pool in self.worker_pools.values():
            await self._stop_worker_pool(pool)
        
        # Shutdown performance engine
        await self.performance_engine.shutdown()
        
        self.logger.info("Autonomous Scaling System stopped")
    
    async def submit_task(
        self, 
        worker_type: WorkerType, 
        task_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Submit a task to the appropriate worker pool."""
        pool = self.worker_pools[worker_type]
        
        # Create task wrapper
        async def task_wrapper():
            start_time = time.time()
            pool.active_tasks += 1
            
            try:
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*args, **kwargs)
                else:
                    result = task_func(*args, **kwargs)
                
                # Update metrics
                execution_time = time.time() - start_time
                pool.total_execution_time += execution_time
                pool.completed_tasks += 1
                
                return result
                
            except Exception as e:
                pool.failed_tasks += 1
                self.logger.error(f"Task failed in {worker_type.value} pool: {e}")
                raise
                
            finally:
                pool.active_tasks -= 1
                self.system_metrics["total_tasks_processed"] += 1
        
        # Submit to queue
        await pool.queue.put(task_wrapper())
        
        # Trigger metrics update
        await self._update_pool_metrics(pool)
        
        return await pool.queue.get()
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring and scaling loop."""
        while self.monitoring_active:
            try:
                # Update system metrics
                await self._collect_system_metrics()
                
                # Check each worker pool for scaling opportunities
                for worker_type, pool in self.worker_pools.items():
                    controller = self.scaling_controllers[worker_type]
                    
                    # Update pool-specific metrics
                    await self._update_pool_metrics(pool)
                    
                    # Make scaling decision
                    decision = controller.make_scaling_decision(pool)
                    
                    if decision:
                        await self._execute_scaling_decision(pool, decision)
                        controller.scaling_history.append(decision)
                        controller.last_scaling_time = time.time()
                        self.system_metrics["total_scaling_actions"] += 1
                
                # Update global metrics
                self._update_global_metrics()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics for scaling decisions."""
        # CPU and Memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        # Add metrics to all controllers
        for controller in self.scaling_controllers.values():
            controller.add_metric(ScalingMetric(
                metric_type=LoadMetricType.CPU_USAGE,
                value=cpu_usage
            ))
            controller.add_metric(ScalingMetric(
                metric_type=LoadMetricType.MEMORY_USAGE,
                value=memory_usage
            ))
    
    async def _update_pool_metrics(self, pool: WorkerPool) -> None:
        """Update metrics for a specific worker pool."""
        controller = self.scaling_controllers[pool.worker_type]
        
        # Queue length metric
        controller.add_metric(ScalingMetric(
            metric_type=LoadMetricType.QUEUE_LENGTH,
            value=pool.queue.qsize()
        ))
        
        # Response time metric
        if pool.completed_tasks > 0:
            controller.add_metric(ScalingMetric(
                metric_type=LoadMetricType.RESPONSE_TIME,
                value=pool.average_execution_time
            ))
        
        # Error rate metric
        total_tasks = pool.completed_tasks + pool.failed_tasks
        if total_tasks > 0:
            error_rate = pool.failed_tasks / total_tasks
            controller.add_metric(ScalingMetric(
                metric_type=LoadMetricType.ERROR_RATE,
                value=error_rate
            ))
        
        # Throughput metric (tasks per second)
        if pool.total_execution_time > 0:
            throughput = pool.completed_tasks / pool.total_execution_time
            controller.add_metric(ScalingMetric(
                metric_type=LoadMetricType.THROUGHPUT,
                value=throughput
            ))
    
    async def _execute_scaling_decision(self, pool: WorkerPool, decision: ScalingDecision) -> None:
        """Execute a scaling decision."""
        self.logger.info(
            f"Scaling {pool.worker_type.value} pool {decision.direction.value} "
            f"by {decision.magnitude} workers: {decision.reason}"
        )
        
        if decision.direction == ScalingDirection.UP:
            await self._scale_worker_pool(pool, decision.magnitude)
        elif decision.direction == ScalingDirection.DOWN:
            await self._scale_worker_pool(pool, -decision.magnitude)
    
    async def _scale_worker_pool(self, pool: WorkerPool, delta: int) -> None:
        """Scale worker pool by delta workers (positive=add, negative=remove)."""
        if delta > 0:
            # Add workers
            for _ in range(delta):
                if pool.current_workers >= pool.max_workers:
                    break
                
                worker = asyncio.create_task(self._worker_loop(pool))
                pool.workers.append(worker)
                pool.current_workers += 1
        
        elif delta < 0:
            # Remove workers
            workers_to_remove = min(abs(delta), pool.current_workers - pool.min_workers)
            
            for _ in range(workers_to_remove):
                if pool.workers:
                    worker = pool.workers.pop()
                    worker.cancel()
                    pool.current_workers -= 1
                    
                    try:
                        await worker
                    except asyncio.CancelledError:
                        pass
    
    async def _worker_loop(self, pool: WorkerPool) -> None:
        """Worker loop for processing tasks from pool queue."""
        while True:
            try:
                # Get task with timeout to allow graceful shutdown
                task = await asyncio.wait_for(pool.queue.get(), timeout=1.0)
                
                # Execute task
                await task
                
            except asyncio.TimeoutError:
                # No task available, continue
                continue
                
            except asyncio.CancelledError:
                # Worker is being shut down
                break
                
            except Exception as e:
                self.logger.error(f"Worker error in {pool.worker_type.value}: {e}")
                # Continue processing other tasks
    
    async def _stop_worker_pool(self, pool: WorkerPool) -> None:
        """Stop all workers in a pool."""
        # Cancel all workers
        for worker in pool.workers:
            worker.cancel()
        
        # Wait for workers to stop
        if pool.workers:
            await asyncio.gather(*pool.workers, return_exceptions=True)
        
        pool.workers.clear()
        pool.current_workers = 0
    
    def _update_global_metrics(self) -> None:
        """Update global system metrics."""
        total_workers = sum(pool.current_workers for pool in self.worker_pools.values())
        self.system_metrics["current_total_workers"] = total_workers
        
        if total_workers > self.system_metrics["peak_workers"]:
            self.system_metrics["peak_workers"] = total_workers
        
        # Calculate average response time across all pools
        total_exec_time = sum(pool.total_execution_time for pool in self.worker_pools.values())
        total_completed = sum(pool.completed_tasks for pool in self.worker_pools.values())
        
        if total_completed > 0:
            self.system_metrics["average_response_time"] = total_exec_time / total_completed
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        pool_status = {}
        
        for worker_type, pool in self.worker_pools.items():
            controller = self.scaling_controllers[worker_type]
            
            pool_status[worker_type.value] = {
                "current_workers": pool.current_workers,
                "min_workers": pool.min_workers,
                "max_workers": pool.max_workers,
                "active_tasks": pool.active_tasks,
                "queue_size": pool.queue.qsize(),
                "completed_tasks": pool.completed_tasks,
                "failed_tasks": pool.failed_tasks,
                "success_rate": pool.success_rate,
                "utilization": pool.utilization,
                "average_execution_time": pool.average_execution_time,
                "recent_scaling_actions": len(controller.scaling_history)
            }
        
        return {
            "system_metrics": self.system_metrics.copy(),
            "worker_pools": pool_status,
            "monitoring_active": self.monitoring_active,
            "timestamp": time.time()
        }
    
    async def optimize_scaling_parameters(self) -> Dict[str, Any]:
        """Optimize scaling parameters based on historical performance."""
        optimization_results = {}
        
        for worker_type, controller in self.scaling_controllers.items():
            if len(controller.scaling_history) < 5:
                continue  # Need more data
            
            # Analyze scaling effectiveness
            successful_scalings = 0
            total_scalings = len(controller.scaling_history)
            
            for decision in controller.scaling_history[-20:]:  # Last 20 decisions
                # Simple effectiveness heuristic: high confidence decisions are considered successful
                if decision.confidence > 0.7:
                    successful_scalings += 1
            
            effectiveness = successful_scalings / min(20, total_scalings) if total_scalings > 0 else 0
            
            # Adjust thresholds based on effectiveness
            if effectiveness < 0.6:  # Less than 60% effective
                # Make scaling more conservative
                for metric_type in controller.scale_up_thresholds:
                    controller.scale_up_thresholds[metric_type] *= 1.1
                    controller.scale_down_thresholds[metric_type] *= 0.9
                
                optimization_results[worker_type.value] = "Made scaling more conservative"
            
            elif effectiveness > 0.8:  # More than 80% effective
                # Make scaling more aggressive
                for metric_type in controller.scale_up_thresholds:
                    controller.scale_up_thresholds[metric_type] *= 0.95
                    controller.scale_down_thresholds[metric_type] *= 1.05
                
                optimization_results[worker_type.value] = "Made scaling more aggressive"
        
        return {
            "optimization_results": optimization_results,
            "timestamp": time.time()
        }