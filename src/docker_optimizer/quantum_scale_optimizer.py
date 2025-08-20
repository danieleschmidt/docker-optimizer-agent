"""Quantum-Scale Optimization Engine for Massive Parallel Docker Processing."""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from .adaptive_intelligence import AdaptiveOptimizationEngine
from .advanced_resilience import AdvancedResilienceEngine
from .enhanced_observability import EnhancedObservabilityEngine
from .models import OptimizationResult
from .multilingual_optimizer import MultilingualOptimizationEngine, SupportedLanguage

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for optimization."""

    VERTICAL = "vertical"          # Scale up single instance
    HORIZONTAL = "horizontal"      # Scale out multiple instances
    ADAPTIVE = "adaptive"          # Smart scaling based on load
    QUANTUM = "quantum"           # Quantum-inspired parallel processing
    HYBRID = "hybrid"             # Combination of strategies


class ProcessingMode(Enum):
    """Processing modes for different workloads."""

    REAL_TIME = "real_time"       # Low latency, immediate processing
    BATCH = "batch"               # High throughput batch processing
    STREAMING = "streaming"       # Continuous stream processing
    BURST = "burst"               # Handle traffic spikes
    RESEARCH = "research"         # Experimental/research workloads


class WorkloadPattern(BaseModel):
    """Workload pattern analysis."""

    avg_request_rate: float
    peak_request_rate: float
    request_size_bytes: int
    processing_complexity: float  # 0-1 scale
    response_time_requirement: float  # milliseconds
    batch_size_preference: int
    resource_requirements: Dict[str, float]


class ScalingConfiguration(BaseModel):
    """Configuration for quantum scaling engine."""

    strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    max_workers: int = Field(default_factory=lambda: cpu_count() * 2)
    min_workers: int = 2
    worker_pool_size: int = 100
    queue_size: int = 10000
    timeout_seconds: int = 300
    memory_limit_mb: int = 4096
    enable_gpu_acceleration: bool = False
    enable_distributed_processing: bool = False
    load_balancing_algorithm: str = "round_robin"
    auto_scaling_enabled: bool = True
    metrics_collection_interval: int = 5


@dataclass
class ProcessingTask:
    """A processing task for optimization."""

    task_id: str
    dockerfile_content: str
    priority: int = 1  # 1-10, higher is more urgent
    timeout: Optional[float] = None
    language: Optional[SupportedLanguage] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[OptimizationResult] = None
    error: Optional[str] = None


class QuantumScaleOptimizer:
    """Quantum-scale optimization engine for massive parallel processing."""

    def __init__(self, config: Optional[ScalingConfiguration] = None):
        """Initialize the quantum scale optimizer."""
        self.config = config or ScalingConfiguration()

        # Core components
        self.adaptive_engine = AdaptiveOptimizationEngine()
        self.resilience_engine = AdvancedResilienceEngine()
        self.observability_engine = EnhancedObservabilityEngine()
        self.multilingual_engine = MultilingualOptimizationEngine()

        # Processing infrastructure
        self.process_executor: Optional[ProcessPoolExecutor] = None
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_size)
        self.result_cache: Dict[str, Any] = {}

        # Worker management
        self.active_workers: List[asyncio.Task] = []
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.current_workers = 0

        # Load balancing and routing
        self.load_balancer = QuantumLoadBalancer(self.config)
        self.task_router = IntelligentTaskRouter()

        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = {
            'throughput': [],
            'latency': [],
            'resource_utilization': [],
            'error_rate': [],
            'queue_depth': []
        }

        # Auto-scaling state
        self.last_scale_time = datetime.now()
        self.scale_cooldown = timedelta(minutes=2)

        # Distributed processing
        self.cluster_nodes: List[str] = []
        self.distributed_enabled = self.config.enable_distributed_processing

        logger.info(f"Quantum scale optimizer initialized with {self.config.strategy.value} strategy")

    async def start(self) -> None:
        """Start the quantum scale optimizer."""
        # Initialize executors
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.config.max_workers
        )
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.config.worker_pool_size
        )

        # Start core engines
        await self.resilience_engine.start()
        await self.observability_engine.start_monitoring()

        # Start initial workers
        await self._scale_workers(self.config.min_workers)

        # Start monitoring and auto-scaling
        if self.config.auto_scaling_enabled:
            asyncio.create_task(self._auto_scaling_loop())

        asyncio.create_task(self._metrics_collection_loop())

        logger.info("Quantum scale optimizer started")

    async def stop(self) -> None:
        """Stop the quantum scale optimizer."""
        # Stop all workers
        for worker in self.active_workers:
            worker.cancel()

        if self.active_workers:
            await asyncio.gather(*self.active_workers, return_exceptions=True)

        # Shutdown executors
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)

        # Stop engines
        await self.resilience_engine.stop()
        await self.observability_engine.stop_monitoring()

        logger.info("Quantum scale optimizer stopped")

    async def submit_optimization(self,
                                dockerfile_content: str,
                                priority: int = 1,
                                timeout: Optional[float] = None,
                                language: Optional[SupportedLanguage] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a Dockerfile optimization task."""
        task_id = f"task_{int(time.time() * 1000000)}"

        task = ProcessingTask(
            task_id=task_id,
            dockerfile_content=dockerfile_content,
            priority=priority,
            timeout=timeout,
            language=language,
            metadata=metadata
        )

        # Route task to appropriate queue/worker
        await self.task_router.route_task(task, self.task_queue)

        # Record metrics
        self.observability_engine.record_metric("optimization_requests_total", 1)
        self.observability_engine.record_metric("queue_depth", self.task_queue.qsize())

        logger.debug(f"Task {task_id} submitted with priority {priority}")
        return task_id

    async def submit_batch_optimization(self,
                                      dockerfiles: List[Dict[str, Any]],
                                      batch_mode: str = "parallel") -> List[str]:
        """Submit batch optimization with intelligent batching."""
        if batch_mode == "sequential":
            task_ids = []
            for dockerfile_data in dockerfiles:
                task_id = await self.submit_optimization(**dockerfile_data)
                task_ids.append(task_id)
            return task_ids

        elif batch_mode == "parallel":
            # Submit all tasks in parallel
            tasks = []
            for dockerfile_data in dockerfiles:
                task = self.submit_optimization(**dockerfile_data)
                tasks.append(task)

            return await asyncio.gather(*tasks)

        elif batch_mode == "quantum":
            # Quantum-inspired batch processing with intelligent grouping
            return await self._quantum_batch_processing(dockerfiles)

        else:
            raise ValueError(f"Unknown batch mode: {batch_mode}")

    async def get_optimization_result(self, task_id: str, timeout: float = 30.0) -> OptimizationResult:
        """Get optimization result for a task."""
        # Check cache first
        if task_id in self.result_cache:
            return self.result_cache[task_id]

        # Wait for result with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if task_id in self.result_cache:
                return self.result_cache[task_id]
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            'quantum_optimizer': {
                'active_workers': len(self.active_workers),
                'queue_depth': self.task_queue.qsize(),
                'tasks_processed': len(self.result_cache),
                'current_throughput': self._calculate_current_throughput(),
                'avg_latency': self._calculate_avg_latency(),
                'resource_utilization': await self._get_resource_utilization(),
                'error_rate': self._calculate_error_rate(),
                'scaling_strategy': self.config.strategy.value,
                'processing_mode': self.config.processing_mode.value
            },
            'load_balancer': self.load_balancer.get_metrics(),
            'observability': self.observability_engine.get_system_overview(),
            'resilience': self.resilience_engine.get_system_health()
        }

    async def _quantum_batch_processing(self, dockerfiles: List[Dict[str, Any]]) -> List[str]:
        """Quantum-inspired batch processing with intelligent optimization."""
        # Analyze batch characteristics
        batch_analysis = self._analyze_batch_characteristics(dockerfiles)

        # Group similar dockerfiles for optimization
        groups = self._quantum_grouping(dockerfiles, batch_analysis)

        # Process groups in quantum-parallel fashion
        task_ids = []
        for group in groups:
            # Process each group with optimal parallelism
            group_tasks = []
            for dockerfile_data in group:
                task = self.submit_optimization(**dockerfile_data)
                group_tasks.append(task)

            # Wait for group completion before starting next group
            group_task_ids = await asyncio.gather(*group_tasks)
            task_ids.extend(group_task_ids)

        return task_ids

    def _analyze_batch_characteristics(self, dockerfiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of a batch for optimal processing."""
        total_size = sum(len(d.get('dockerfile_content', '')) for d in dockerfiles)
        avg_size = total_size / len(dockerfiles) if dockerfiles else 0

        # Detect complexity patterns
        complexity_scores = []
        for dockerfile_data in dockerfiles:
            content = dockerfile_data.get('dockerfile_content', '')
            complexity = self._estimate_dockerfile_complexity(content)
            complexity_scores.append(complexity)

        return {
            'batch_size': len(dockerfiles),
            'total_content_size': total_size,
            'avg_content_size': avg_size,
            'complexity_distribution': {
                'min': min(complexity_scores) if complexity_scores else 0,
                'max': max(complexity_scores) if complexity_scores else 0,
                'avg': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
            },
            'estimated_processing_time': self._estimate_batch_processing_time(dockerfiles),
            'recommended_parallelism': self._calculate_optimal_parallelism(complexity_scores)
        }

    def _quantum_grouping(self, dockerfiles: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Group dockerfiles using quantum-inspired clustering."""
        # Simple clustering based on complexity and size
        groups = []
        current_group = []
        group_complexity = 0
        max_group_complexity = 5.0  # Threshold for group complexity

        sorted_dockerfiles = sorted(
            dockerfiles,
            key=lambda x: self._estimate_dockerfile_complexity(x.get('dockerfile_content', ''))
        )

        for dockerfile_data in sorted_dockerfiles:
            complexity = self._estimate_dockerfile_complexity(dockerfile_data.get('dockerfile_content', ''))

            if group_complexity + complexity <= max_group_complexity and len(current_group) < 10:
                current_group.append(dockerfile_data)
                group_complexity += complexity
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [dockerfile_data]
                group_complexity = complexity

        if current_group:
            groups.append(current_group)

        return groups

    def _estimate_dockerfile_complexity(self, content: str) -> float:
        """Estimate complexity of a Dockerfile."""
        lines = content.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]

        complexity = len(non_empty_lines) * 0.1
        complexity += content.count('RUN') * 0.3
        complexity += content.count('COPY') * 0.2
        complexity += content.count('ADD') * 0.2
        complexity += len([line for line in lines if '&&' in line]) * 0.4

        return complexity

    def _estimate_batch_processing_time(self, dockerfiles: List[Dict[str, Any]]) -> float:
        """Estimate total processing time for batch."""
        total_complexity = sum(
            self._estimate_dockerfile_complexity(d.get('dockerfile_content', ''))
            for d in dockerfiles
        )

        # Base processing time per complexity unit (seconds)
        base_time_per_unit = 0.5

        # Factor in parallelism
        parallelism = min(len(dockerfiles), self.config.max_workers)

        return (total_complexity * base_time_per_unit) / parallelism

    def _calculate_optimal_parallelism(self, complexity_scores: List[float]) -> int:
        """Calculate optimal parallelism for given complexity distribution."""
        if not complexity_scores:
            return self.config.min_workers

        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        max_complexity = max(complexity_scores)

        # More complex tasks need fewer parallel workers to avoid resource contention
        if avg_complexity > 3.0:
            return min(self.config.max_workers // 2, len(complexity_scores))
        elif avg_complexity > 1.5:
            return min(self.config.max_workers, len(complexity_scores))
        else:
            return min(self.config.max_workers * 2, len(complexity_scores))

    async def _scale_workers(self, target_workers: int) -> None:
        """Scale the number of workers."""
        current_count = len(self.active_workers)

        if target_workers > current_count:
            # Scale up
            for _ in range(target_workers - current_count):
                worker = asyncio.create_task(self._worker_loop())
                self.active_workers.append(worker)

            logger.info(f"Scaled up to {target_workers} workers")

        elif target_workers < current_count:
            # Scale down
            workers_to_remove = self.active_workers[target_workers:]
            self.active_workers = self.active_workers[:target_workers]

            for worker in workers_to_remove:
                worker.cancel()

            logger.info(f"Scaled down to {target_workers} workers")

        self.current_workers = target_workers

    async def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        worker_id = f"worker_{len(self.active_workers)}"

        while True:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Process task with resilience
                await self._process_task_with_resilience(task, worker_id)

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def _process_task_with_resilience(self, task: ProcessingTask, worker_id: str) -> None:
        """Process a task with resilience patterns."""
        task.started_at = datetime.now()

        try:
            async with self.resilience_engine.circuit_protected(f"optimization_{worker_id}"):
                # Record start metrics
                start_time = time.time()
                self.observability_engine.record_metric("optimization_duration_seconds", 0)

                # Process the optimization
                result = await self._execute_optimization(task)

                # Calculate processing time
                processing_time = time.time() - start_time

                # Store result
                task.result = result
                task.completed_at = datetime.now()
                self.result_cache[task.task_id] = result

                # Record success metrics
                self.observability_engine.record_metric("optimization_duration_seconds", processing_time)
                self._update_worker_stats(worker_id, "success", processing_time)

                logger.debug(f"Task {task.task_id} completed successfully in {processing_time:.2f}s")

        except Exception as e:
            # Handle failure
            task.error = str(e)
            task.completed_at = datetime.now()

            # Record failure metrics
            self.observability_engine.record_metric("error_rate", 1)
            self._update_worker_stats(worker_id, "error", 0)

            logger.error(f"Task {task.task_id} failed: {e}")

    async def _execute_optimization(self, task: ProcessingTask) -> OptimizationResult:
        """Execute the actual optimization."""
        # Use adaptive intelligence for optimization strategy
        adaptive_suggestions = self.adaptive_engine.suggest_adaptive_optimization(
            task.dockerfile_content,
            user_goals=['security', 'size', 'performance']
        )

        # Apply multilingual optimization if language specified
        if task.language:
            multilingual_result = self.multilingual_engine.optimize_with_localization(
                task.dockerfile_content,
                task.language
            )
            # Use the optimized dockerfile from multilingual result
            optimized_content = multilingual_result.get('localized_dockerfile', task.dockerfile_content)
        else:
            # Use standard optimization
            from .optimizer import DockerfileOptimizer
            optimizer = DockerfileOptimizer()
            result = optimizer.optimize_dockerfile(task.dockerfile_content)
            optimized_content = result.optimized_dockerfile

        # Create enhanced result with adaptive insights
        enhanced_result = OptimizationResult(
            original_size="Unknown",  # Would be calculated in real implementation
            optimized_size="Unknown",  # Would be calculated in real implementation
            security_fixes=[],
            explanation=f"Adaptive optimization applied using {adaptive_suggestions['primary_strategy']} strategy",
            optimized_dockerfile=optimized_content,
            layer_optimizations=[]
        )

        # Learn from the optimization
        self.adaptive_engine.learn_from_optimization(
            task.dockerfile_content,
            enhanced_result,
            user_feedback=task.metadata
        )

        return enhanced_result

    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling loop for dynamic worker adjustment."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)

                # Get current metrics
                queue_depth = self.task_queue.qsize()
                current_throughput = self._calculate_current_throughput()
                avg_latency = self._calculate_avg_latency()
                resource_utilization = await self._get_resource_utilization()

                # Determine if scaling is needed
                scale_decision = self._make_scaling_decision(
                    queue_depth, current_throughput, avg_latency, resource_utilization
                )

                if scale_decision != 0:
                    # Check cooldown
                    if datetime.now() - self.last_scale_time > self.scale_cooldown:
                        new_worker_count = max(
                            self.config.min_workers,
                            min(self.config.max_workers, self.current_workers + scale_decision)
                        )

                        if new_worker_count != self.current_workers:
                            await self._scale_workers(new_worker_count)
                            self.last_scale_time = datetime.now()

            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(30)  # Wait longer after error

    def _make_scaling_decision(self,
                             queue_depth: int,
                             throughput: float,
                             latency: float,
                             resource_util: Dict[str, float]) -> int:
        """Make intelligent scaling decision based on metrics."""
        scale_up_score = 0
        scale_down_score = 0

        # Queue depth factor
        if queue_depth > 50:
            scale_up_score += 3
        elif queue_depth > 20:
            scale_up_score += 1
        elif queue_depth < 5:
            scale_down_score += 1

        # Latency factor
        if latency > 5000:  # 5 seconds
            scale_up_score += 2
        elif latency < 1000:  # 1 second
            scale_down_score += 1

        # Resource utilization factor
        cpu_util = resource_util.get('cpu', 0)
        memory_util = resource_util.get('memory', 0)

        if cpu_util > 80 or memory_util > 80:
            scale_up_score += 2
        elif cpu_util < 30 and memory_util < 30:
            scale_down_score += 2

        # Make decision
        if scale_up_score > scale_down_score:
            return min(scale_up_score, 3)  # Scale up by 1-3 workers
        elif scale_down_score > scale_up_score:
            return -min(scale_down_score, 2)  # Scale down by 1-2 workers
        else:
            return 0  # No scaling needed

    async def _metrics_collection_loop(self) -> None:
        """Continuously collect performance metrics."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)

                # Collect metrics
                throughput = self._calculate_current_throughput()
                latency = self._calculate_avg_latency()
                resource_util = await self._get_resource_utilization()
                error_rate = self._calculate_error_rate()
                queue_depth = self.task_queue.qsize()

                # Store in time series
                self.performance_metrics['throughput'].append(throughput)
                self.performance_metrics['latency'].append(latency)
                self.performance_metrics['resource_utilization'].append(resource_util.get('cpu', 0))
                self.performance_metrics['error_rate'].append(error_rate)
                self.performance_metrics['queue_depth'].append(queue_depth)

                # Trim old metrics (keep last 100 readings)
                for metric_name in self.performance_metrics:
                    if len(self.performance_metrics[metric_name]) > 100:
                        self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-100:]

                # Record in observability engine
                self.observability_engine.record_metric("throughput", throughput)
                self.observability_engine.record_metric("response_time_ms", latency)
                self.observability_engine.record_metric("cpu_usage_percent", resource_util.get('cpu', 0))
                self.observability_engine.record_metric("memory_usage_bytes", resource_util.get('memory_mb', 0) * 1024 * 1024)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)

    def _calculate_current_throughput(self) -> float:
        """Calculate current throughput (tasks per second)."""
        if len(self.performance_metrics['throughput']) < 2:
            return 0.0

        # Simple throughput calculation
        completed_tasks = len([task for task in self.result_cache.values() if task])
        time_window = self.config.metrics_collection_interval * len(self.performance_metrics['throughput'])

        return completed_tasks / max(time_window, 1)

    def _calculate_avg_latency(self) -> float:
        """Calculate average latency in milliseconds."""
        if not self.performance_metrics['latency']:
            return 0.0

        return sum(self.performance_metrics['latency']) / len(self.performance_metrics['latency'])

    async def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        # Simplified implementation - in practice would use psutil or similar
        return {
            'cpu': np.random.uniform(20, 80),  # Placeholder
            'memory': np.random.uniform(30, 70),  # Placeholder
            'memory_mb': np.random.uniform(500, 2000)  # Placeholder
        }

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        if not self.performance_metrics['error_rate']:
            return 0.0

        recent_errors = self.performance_metrics['error_rate'][-10:]  # Last 10 measurements
        return sum(recent_errors) / len(recent_errors)

    def _update_worker_stats(self, worker_id: str, result_type: str, processing_time: float) -> None:
        """Update worker statistics."""
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = {
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_processing_time': 0.0,
                'avg_processing_time': 0.0
            }

        stats = self.worker_stats[worker_id]

        if result_type == "success":
            stats['tasks_completed'] += 1
            stats['total_processing_time'] += processing_time
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['tasks_completed']
        elif result_type == "error":
            stats['tasks_failed'] += 1


class QuantumLoadBalancer:
    """Quantum-inspired load balancer for optimal task distribution."""

    def __init__(self, config: ScalingConfiguration):
        """Initialize the load balancer."""
        self.config = config
        self.algorithm = config.load_balancing_algorithm
        self.worker_loads: Dict[str, float] = {}
        self.task_assignments: Dict[str, str] = {}

    def assign_task(self, task: ProcessingTask, available_workers: List[str]) -> str:
        """Assign task to optimal worker."""
        if self.algorithm == "round_robin":
            return self._round_robin_assignment(available_workers)
        elif self.algorithm == "least_loaded":
            return self._least_loaded_assignment(available_workers)
        elif self.algorithm == "quantum_optimal":
            return self._quantum_optimal_assignment(task, available_workers)
        else:
            return available_workers[0] if available_workers else "default"

    def _round_robin_assignment(self, workers: List[str]) -> str:
        """Simple round-robin assignment."""
        if not workers:
            return "default"

        return workers[len(self.task_assignments) % len(workers)]

    def _least_loaded_assignment(self, workers: List[str]) -> str:
        """Assign to least loaded worker."""
        if not workers:
            return "default"

        return min(workers, key=lambda w: self.worker_loads.get(w, 0))

    def _quantum_optimal_assignment(self, task: ProcessingTask, workers: List[str]) -> str:
        """Quantum-inspired optimal assignment based on task characteristics."""
        if not workers:
            return "default"

        # Score each worker based on task complexity and current load
        task_complexity = len(task.dockerfile_content) / 1000.0  # Simplified complexity

        best_worker = workers[0]
        best_score = float('inf')

        for worker in workers:
            load = self.worker_loads.get(worker, 0)
            # Score combines load and suitability for task complexity
            score = load + (task_complexity * 0.1)

            if score < best_score:
                best_score = score
                best_worker = worker

        return best_worker

    def update_worker_load(self, worker_id: str, load: float) -> None:
        """Update worker load information."""
        self.worker_loads[worker_id] = load

    def get_metrics(self) -> Dict[str, Any]:
        """Get load balancer metrics."""
        return {
            'algorithm': self.algorithm,
            'worker_count': len(self.worker_loads),
            'avg_load': sum(self.worker_loads.values()) / len(self.worker_loads) if self.worker_loads else 0,
            'max_load': max(self.worker_loads.values()) if self.worker_loads else 0,
            'task_assignments': len(self.task_assignments)
        }


class IntelligentTaskRouter:
    """Intelligent task router for optimal queue management."""

    def __init__(self):
        """Initialize the task router."""
        self.priority_queues: Dict[int, asyncio.Queue] = {}
        self.routing_rules: Dict[str, Callable] = {}

    async def route_task(self, task: ProcessingTask, default_queue: asyncio.Queue) -> None:
        """Route task to appropriate queue based on characteristics."""
        # Route by priority
        if task.priority >= 8:  # High priority
            await self._get_priority_queue(task.priority).put(task)
        else:
            await default_queue.put(task)

    def _get_priority_queue(self, priority: int) -> asyncio.Queue:
        """Get or create priority queue."""
        if priority not in self.priority_queues:
            self.priority_queues[priority] = asyncio.Queue(maxsize=1000)
        return self.priority_queues[priority]
