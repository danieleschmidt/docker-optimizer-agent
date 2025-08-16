"""Adaptive Scaling Engine with Dynamic Load Balancing and Auto-Scaling."""

import asyncio
import logging
import queue
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import psutil

from .ai_health_monitor import AIHealthMonitor


class ScalingStrategy(str, Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"          # Scale based on current load
    PREDICTIVE = "predictive"      # Scale based on predicted load
    HYBRID = "hybrid"              # Combination of reactive and predictive
    AGGRESSIVE = "aggressive"      # Aggressive scaling for maximum performance


class WorkerType(str, Enum):
    """Types of workers."""
    THREAD = "thread"     # Thread-based workers (I/O bound tasks)
    PROCESS = "process"   # Process-based workers (CPU bound tasks)
    ASYNC = "async"       # Async workers (concurrent I/O)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    current_workers: int
    target_workers: int
    cpu_utilization: float
    memory_utilization: float
    queue_size: int
    avg_processing_time: float
    throughput_ops_per_sec: float
    error_rate_percent: float
    timestamp: float


@dataclass
class WorkerPool:
    """Worker pool configuration and state."""
    worker_type: WorkerType
    min_workers: int = 2
    max_workers: int = 16
    current_workers: int = 2
    executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
    work_queue: Optional[queue.Queue] = None
    active: bool = False


class AdaptiveScalingEngine:
    """Advanced adaptive scaling engine with dynamic load balancing."""

    def __init__(
        self,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        initial_workers: int = 4,
        min_workers: int = 2,
        max_workers: int = 32,
        scale_up_threshold: float = 0.7,      # Scale up when utilization > 70%
        scale_down_threshold: float = 0.3,     # Scale down when utilization < 30%
        scaling_cooldown: int = 30            # Cooldown period in seconds
    ):
        self.strategy = strategy
        self.initial_workers = initial_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_cooldown = scaling_cooldown

        self.logger = logging.getLogger(__name__)
        self.health_monitor = AIHealthMonitor()

        # Worker pools
        self.worker_pools: Dict[str, WorkerPool] = {}

        # Scaling state
        self.last_scale_time = 0
        self.scaling_metrics_history: List[ScalingMetrics] = []
        self.pending_tasks: Dict[str, queue.Queue] = {}

        # Performance tracking
        self.processed_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0

        # Initialize worker pools
        self._initialize_worker_pools()

        # Start background tasks
        self._background_tasks = []
        self._scaling_active = False

    def _initialize_worker_pools(self) -> None:
        """Initialize different types of worker pools."""

        # AI Optimization pool (CPU-intensive)
        self.worker_pools["ai_optimization"] = WorkerPool(
            worker_type=WorkerType.PROCESS,
            min_workers=2,
            max_workers=8,
            current_workers=self.initial_workers,
            work_queue=queue.Queue()
        )

        # Security Scanning pool (I/O-intensive)
        self.worker_pools["security_scan"] = WorkerPool(
            worker_type=WorkerType.THREAD,
            min_workers=2,
            max_workers=16,
            current_workers=self.initial_workers,
            work_queue=queue.Queue()
        )

        # General Processing pool (Mixed workload)
        self.worker_pools["general"] = WorkerPool(
            worker_type=WorkerType.THREAD,
            min_workers=2,
            max_workers=12,
            current_workers=self.initial_workers,
            work_queue=queue.Queue()
        )

    async def start_scaling_engine(self) -> None:
        """Start the adaptive scaling engine."""
        self.logger.info("Starting adaptive scaling engine")
        self._scaling_active = True

        # Start worker pools
        for pool_name, pool in self.worker_pools.items():
            await self._start_worker_pool(pool_name, pool)

        # Start background monitoring and scaling
        self._background_tasks = [
            asyncio.create_task(self._monitor_and_scale()),
            asyncio.create_task(self._collect_metrics()),
            asyncio.create_task(self._health_monitoring())
        ]

        self.logger.info(f"Scaling engine started with strategy: {self.strategy}")

    async def stop_scaling_engine(self) -> None:
        """Stop the adaptive scaling engine."""
        self.logger.info("Stopping adaptive scaling engine")
        self._scaling_active = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Stop worker pools
        for pool_name, pool in self.worker_pools.items():
            await self._stop_worker_pool(pool_name, pool)

        self.logger.info("Scaling engine stopped")

    async def _start_worker_pool(self, pool_name: str, pool: WorkerPool) -> None:
        """Start a worker pool."""
        self.logger.info(f"Starting {pool_name} worker pool ({pool.worker_type.value})")

        if pool.worker_type == WorkerType.PROCESS:
            pool.executor = ProcessPoolExecutor(max_workers=pool.current_workers)
        elif pool.worker_type == WorkerType.THREAD:
            pool.executor = ThreadPoolExecutor(max_workers=pool.current_workers)

        pool.active = True

        # Start worker processing task
        self._background_tasks.append(
            asyncio.create_task(self._process_pool_tasks(pool_name, pool))
        )

    async def _stop_worker_pool(self, pool_name: str, pool: WorkerPool) -> None:
        """Stop a worker pool."""
        self.logger.info(f"Stopping {pool_name} worker pool")
        pool.active = False

        if pool.executor:
            pool.executor.shutdown(wait=True)

    async def _process_pool_tasks(self, pool_name: str, pool: WorkerPool) -> None:
        """Process tasks from a worker pool queue."""
        while pool.active:
            try:
                if not pool.work_queue.empty():
                    task_data = pool.work_queue.get(timeout=1)

                    if task_data:
                        # Execute task based on pool type
                        await self._execute_task(pool_name, pool, task_data)
                else:
                    await asyncio.sleep(0.1)  # Brief pause when no tasks

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing {pool_name} tasks: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, pool_name: str, pool: WorkerPool, task_data: Dict) -> None:
        """Execute a task using the appropriate worker pool."""
        start_time = time.time()

        try:
            task_func = task_data["func"]
            args = task_data.get("args", ())
            kwargs = task_data.get("kwargs", {})
            callback = task_data.get("callback")

            # Execute based on worker type
            if pool.worker_type == WorkerType.ASYNC:
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*args, **kwargs)
                else:
                    result = task_func(*args, **kwargs)
            else:
                # Use executor for thread/process pools
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    pool.executor,
                    lambda: task_func(*args, **kwargs)
                )

            # Record success
            processing_time = time.time() - start_time
            self.processed_tasks += 1
            self.total_processing_time += processing_time

            # Call callback with result if provided
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)

        except Exception as e:
            self.failed_tasks += 1
            self.logger.error(f"Task execution failed in {pool_name}: {e}")

    async def submit_task(
        self,
        pool_name: str,
        func,
        *args,
        callback=None,
        **kwargs
    ) -> bool:
        """Submit a task to a worker pool."""
        if pool_name not in self.worker_pools:
            self.logger.error(f"Unknown worker pool: {pool_name}")
            return False

        pool = self.worker_pools[pool_name]

        task_data = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "callback": callback
        }

        try:
            pool.work_queue.put(task_data, block=False)
            return True
        except queue.Full:
            self.logger.warning(f"Worker pool {pool_name} queue is full")
            return False

    async def _monitor_and_scale(self) -> None:
        """Background task for monitoring and auto-scaling."""
        while self._scaling_active:
            try:
                # Collect current metrics
                metrics = await self._collect_current_metrics()

                # Store metrics history
                self.scaling_metrics_history.append(metrics)
                if len(self.scaling_metrics_history) > 100:
                    self.scaling_metrics_history.pop(0)

                # Make scaling decisions
                await self._make_scaling_decisions(metrics)

                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Error in scaling monitor: {e}")
                await asyncio.sleep(5)

    async def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system and pool metrics."""

        # System metrics
        cpu_util = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_util = memory.percent

        # Calculate aggregate metrics across pools
        total_current_workers = sum(pool.current_workers for pool in self.worker_pools.values())
        total_queue_size = sum(pool.work_queue.qsize() for pool in self.worker_pools.values())

        # Performance metrics
        avg_processing_time = (
            self.total_processing_time / max(self.processed_tasks, 1)
        )

        total_tasks = self.processed_tasks + self.failed_tasks
        error_rate = (self.failed_tasks / max(total_tasks, 1)) * 100

        # Throughput calculation (tasks per second)
        current_time = time.time()
        if hasattr(self, '_last_throughput_check'):
            time_diff = current_time - self._last_throughput_check
            task_diff = self.processed_tasks - self._last_processed_count
            throughput = task_diff / max(time_diff, 1)
        else:
            throughput = 0.0

        self._last_throughput_check = current_time
        self._last_processed_count = self.processed_tasks

        return ScalingMetrics(
            current_workers=total_current_workers,
            target_workers=total_current_workers,  # Will be updated by scaling logic
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            queue_size=total_queue_size,
            avg_processing_time=avg_processing_time,
            throughput_ops_per_sec=throughput,
            error_rate_percent=error_rate,
            timestamp=current_time
        )

    async def _make_scaling_decisions(self, metrics: ScalingMetrics) -> None:
        """Make auto-scaling decisions based on metrics."""

        # Check if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_scale_time < self.scaling_cooldown:
            return

        # Determine if scaling is needed
        should_scale_up = (
            metrics.cpu_utilization > self.scale_up_threshold * 100 or
            metrics.queue_size > 10 or
            metrics.avg_processing_time > 5.0
        )

        should_scale_down = (
            metrics.cpu_utilization < self.scale_down_threshold * 100 and
            metrics.queue_size < 2 and
            metrics.avg_processing_time < 1.0
        )

        if should_scale_up:
            await self._scale_up()
            self.last_scale_time = current_time
        elif should_scale_down:
            await self._scale_down()
            self.last_scale_time = current_time

    async def _scale_up(self) -> None:
        """Scale up worker pools."""
        self.logger.info("Scaling up worker pools")

        for pool_name, pool in self.worker_pools.items():
            if pool.current_workers < pool.max_workers:
                new_worker_count = min(
                    pool.current_workers + 2,  # Add 2 workers at a time
                    pool.max_workers
                )

                await self._resize_pool(pool_name, pool, new_worker_count)

    async def _scale_down(self) -> None:
        """Scale down worker pools."""
        self.logger.info("Scaling down worker pools")

        for pool_name, pool in self.worker_pools.items():
            if pool.current_workers > pool.min_workers:
                new_worker_count = max(
                    pool.current_workers - 1,  # Remove 1 worker at a time
                    pool.min_workers
                )

                await self._resize_pool(pool_name, pool, new_worker_count)

    async def _resize_pool(self, pool_name: str, pool: WorkerPool, new_size: int) -> None:
        """Resize a worker pool."""
        if new_size == pool.current_workers:
            return

        self.logger.info(f"Resizing {pool_name} pool: {pool.current_workers} -> {new_size}")

        # Stop current executor
        if pool.executor:
            pool.executor.shutdown(wait=False)

        # Create new executor with new size
        if pool.worker_type == WorkerType.PROCESS:
            pool.executor = ProcessPoolExecutor(max_workers=new_size)
        elif pool.worker_type == WorkerType.THREAD:
            pool.executor = ThreadPoolExecutor(max_workers=new_size)

        pool.current_workers = new_size

    async def _collect_metrics(self) -> None:
        """Background task for collecting detailed metrics."""
        while self._scaling_active:
            try:
                # This could collect more detailed metrics for analysis
                await asyncio.sleep(30)  # Collect detailed metrics every 30 seconds
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)

    async def _health_monitoring(self) -> None:
        """Background health monitoring task."""
        while self._scaling_active:
            try:
                health_check = await self.health_monitor.check_system_health()

                if health_check.status.value != "healthy":
                    self.logger.warning(f"Health issue detected: {health_check.message}")

                    # Implement health-based scaling adjustments
                    await self._handle_health_issues(health_check)

                await asyncio.sleep(60)  # Health check every minute

            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(30)

    async def _handle_health_issues(self, health_check) -> None:
        """Handle health issues by adjusting scaling."""

        if health_check.status.value == "critical":
            # Scale down to reduce load
            self.logger.info("Critical health issue - scaling down")
            await self._emergency_scale_down()
        elif health_check.status.value == "warning":
            # Pause scaling up temporarily
            self.last_scale_time = time.time()

    async def _emergency_scale_down(self) -> None:
        """Emergency scale down to minimum workers."""
        for pool_name, pool in self.worker_pools.items():
            await self._resize_pool(pool_name, pool, pool.min_workers)

    def get_scaling_status(self) -> Dict:
        """Get current scaling engine status."""
        return {
            "active": self._scaling_active,
            "strategy": self.strategy.value,
            "worker_pools": {
                name: {
                    "type": pool.worker_type.value,
                    "current_workers": pool.current_workers,
                    "min_workers": pool.min_workers,
                    "max_workers": pool.max_workers,
                    "queue_size": pool.work_queue.qsize() if pool.work_queue else 0,
                    "active": pool.active
                }
                for name, pool in self.worker_pools.items()
            },
            "performance_stats": {
                "processed_tasks": self.processed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_processing_time": self.total_processing_time,
                "avg_processing_time": self.total_processing_time / max(self.processed_tasks, 1)
            },
            "scaling_history": len(self.scaling_metrics_history)
        }

    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics."""
        if not self.scaling_metrics_history:
            return {}

        latest_metrics = self.scaling_metrics_history[-1]

        return {
            "current_metrics": {
                "workers": latest_metrics.current_workers,
                "cpu_utilization": latest_metrics.cpu_utilization,
                "memory_utilization": latest_metrics.memory_utilization,
                "queue_size": latest_metrics.queue_size,
                "throughput": latest_metrics.throughput_ops_per_sec,
                "error_rate": latest_metrics.error_rate_percent
            },
            "historical_averages": self._calculate_historical_averages(),
            "scaling_events": self._get_recent_scaling_events()
        }

    def _calculate_historical_averages(self) -> Dict:
        """Calculate historical metric averages."""
        if not self.scaling_metrics_history:
            return {}

        metrics = self.scaling_metrics_history[-20:]  # Last 20 data points

        return {
            "avg_cpu_utilization": sum(m.cpu_utilization for m in metrics) / len(metrics),
            "avg_memory_utilization": sum(m.memory_utilization for m in metrics) / len(metrics),
            "avg_queue_size": sum(m.queue_size for m in metrics) / len(metrics),
            "avg_throughput": sum(m.throughput_ops_per_sec for m in metrics) / len(metrics),
            "avg_workers": sum(m.current_workers for m in metrics) / len(metrics)
        }

    def _get_recent_scaling_events(self) -> List[Dict]:
        """Get recent scaling events."""
        events = []

        if len(self.scaling_metrics_history) > 1:
            for i in range(1, len(self.scaling_metrics_history)):
                prev_metrics = self.scaling_metrics_history[i-1]
                curr_metrics = self.scaling_metrics_history[i]

                if prev_metrics.current_workers != curr_metrics.current_workers:
                    events.append({
                        "timestamp": curr_metrics.timestamp,
                        "action": "scale_up" if curr_metrics.current_workers > prev_metrics.current_workers else "scale_down",
                        "from_workers": prev_metrics.current_workers,
                        "to_workers": curr_metrics.current_workers,
                        "cpu_utilization": curr_metrics.cpu_utilization,
                        "queue_size": curr_metrics.queue_size
                    })

        return events[-10:]  # Return last 10 scaling events


# Export for use in other modules
__all__ = [
    "AdaptiveScalingEngine",
    "ScalingStrategy",
    "WorkerType",
    "ScalingMetrics",
    "WorkerPool"
]
