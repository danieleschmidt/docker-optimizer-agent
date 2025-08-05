"""Core quantum task planner with robust error handling and monitoring."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import traceback
from pathlib import Path
import json

from ..models.task import Task, TaskStatus, TaskPriority
from ..models.resource import Resource, ResourceStatus
from ..models.schedule import Schedule, ScheduleStatus, OptimizationObjective, ScheduleOptimization
from ..algorithms.quantum_annealing import QuantumAnnealingScheduler
from ..algorithms.qaoa_allocator import QAOAResourceAllocator, QAOAParameters
from .exceptions import (
    QuantumTaskPlannerError, ValidationError, OptimizationError, 
    SchedulingError, TimeoutError, ConfigurationError
)


logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    """Configuration for quantum task planner."""
    
    # Algorithm selection
    default_algorithm: str = "quantum_annealing"
    fallback_algorithm: str = "qaoa"
    enable_hybrid_optimization: bool = True
    
    # Performance settings  
    max_concurrent_optimizations: int = 4
    optimization_timeout_seconds: int = 300
    enable_parallel_processing: bool = True
    
    # Monitoring and logging
    enable_metrics_collection: bool = True
    metrics_export_path: Optional[Path] = None
    log_level: str = "INFO"
    enable_performance_profiling: bool = False
    
    # Error handling
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    enable_graceful_degradation: bool = True
    
    # Validation
    strict_validation: bool = True
    validate_quantum_parameters: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_concurrent_optimizations < 1:
            raise ConfigurationError("max_concurrent_optimizations must be >= 1")
        if self.optimization_timeout_seconds < 1:
            raise ConfigurationError("optimization_timeout_seconds must be >= 1")
        if not isinstance(self.enable_parallel_processing, bool):
            raise ConfigurationError("enable_parallel_processing must be boolean")


@dataclass
class PlannerMetrics:
    """Metrics tracking for planner operations."""
    
    total_optimizations: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    
    total_tasks_scheduled: int = 0
    total_resources_allocated: int = 0
    
    average_optimization_time: float = 0.0
    optimization_times: List[float] = field(default_factory=list)
    
    algorithm_usage: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def record_optimization(self, algorithm: str, duration: float, success: bool, error_type: Optional[str] = None):
        """Record optimization attempt."""
        self.total_optimizations += 1
        self.updated_at = datetime.utcnow()
        
        if success:
            self.successful_optimizations += 1
        else:
            self.failed_optimizations += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.optimization_times.append(duration)
        self.average_optimization_time = sum(self.optimization_times) / len(self.optimization_times)
        
        self.algorithm_usage[algorithm] = self.algorithm_usage.get(algorithm, 0) + 1
    
    def get_success_rate(self) -> float:
        """Calculate optimization success rate."""
        if self.total_optimizations == 0:
            return 0.0
        return self.successful_optimizations / self.total_optimizations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "failed_optimizations": self.failed_optimizations,
            "success_rate": self.get_success_rate(),
            "total_tasks_scheduled": self.total_tasks_scheduled,
            "total_resources_allocated": self.total_resources_allocated,
            "average_optimization_time": self.average_optimization_time,
            "algorithm_usage": self.algorithm_usage,
            "error_counts": self.error_counts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class QuantumTaskPlanner:
    """Quantum-inspired task planner with comprehensive error handling."""
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        """Initialize quantum task planner.
        
        Args:
            config: Planner configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self.config = config or PlannerConfig()
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
            
            # Initialize algorithms
            self.algorithms = self._initialize_algorithms()
            
            # Initialize metrics
            self.metrics = PlannerMetrics()
            
            # Active schedules tracking
            self.active_schedules: Dict[str, Schedule] = {}
            self.optimization_futures: Dict[str, Any] = {}
            
            # Executor for parallel processing
            self.executor = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_optimizations,
                thread_name_prefix="quantum-planner"
            ) if self.config.enable_parallel_processing else None
            
            # Performance profiling
            self.profiling_data: Dict[str, List[float]] = {} if self.config.enable_performance_profiling else None
            
            self.logger.info(f"Quantum task planner initialized with {len(self.algorithms)} algorithms")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum task planner: {e}")
            raise ConfigurationError(f"Planner initialization failed: {e}") from e
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
    
    def shutdown(self):
        """Shutdown planner and cleanup resources."""
        try:
            self.logger.info("Shutting down quantum task planner")
            
            # Cancel active optimizations
            for schedule_id, future in self.optimization_futures.items():
                if not future.done():
                    future.cancel()
                    self.logger.info(f"Cancelled optimization for schedule {schedule_id}")
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True, timeout=30)
            
            # Export metrics if configured
            if self.config.enable_metrics_collection and self.config.metrics_export_path:
                self._export_metrics()
            
            self.logger.info("Quantum task planner shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def create_schedule(self, schedule_id: str, name: str, start_time: datetime,
                       description: Optional[str] = None,
                       objectives: Optional[List[OptimizationObjective]] = None) -> Schedule:
        """Create new schedule with validation.
        
        Args:
            schedule_id: Unique schedule identifier
            name: Schedule name
            start_time: Schedule start time
            description: Optional description
            objectives: Optimization objectives
            
        Returns:
            Created schedule
            
        Raises:
            ValidationError: If parameters are invalid
            SchedulingError: If schedule creation fails
        """
        try:
            self.logger.info(f"Creating schedule {schedule_id}")
            
            # Validate inputs
            if not schedule_id or not schedule_id.strip():
                raise ValidationError("Schedule ID cannot be empty")
            
            if schedule_id in self.active_schedules:
                raise ValidationError(f"Schedule {schedule_id} already exists")
            
            if not name or not name.strip():
                raise ValidationError("Schedule name cannot be empty")
            
            if start_time < datetime.utcnow() - timedelta(hours=1):
                self.logger.warning(f"Schedule {schedule_id} has start time in the past")
            
            # Create schedule
            schedule = Schedule(
                id=schedule_id,
                name=name.strip(),
                description=description,
                start_time=start_time,
                objectives=objectives or [OptimizationObjective.MINIMIZE_MAKESPAN],
                status=ScheduleStatus.DRAFT
            )
            
            # Store schedule
            self.active_schedules[schedule_id] = schedule
            
            self.logger.info(f"Schedule {schedule_id} created successfully")
            return schedule
            
        except Exception as e:
            self.logger.error(f"Failed to create schedule {schedule_id}: {e}")
            if isinstance(e, (ValidationError, SchedulingError)):
                raise
            raise SchedulingError(f"Schedule creation failed: {e}", schedule_id=schedule_id) from e
    
    def add_task(self, schedule_id: str, task: Task) -> None:
        """Add task to schedule with validation.
        
        Args:
            schedule_id: Schedule identifier
            task: Task to add
            
        Raises:
            ValidationError: If task is invalid
            SchedulingError: If adding task fails
        """
        try:
            schedule = self._get_schedule(schedule_id)
            
            # Validate task
            self._validate_task(task, schedule)
            
            # Add task
            schedule.add_task(task)
            
            self.logger.debug(f"Added task {task.id} to schedule {schedule_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add task {task.id} to schedule {schedule_id}: {e}")
            if isinstance(e, (ValidationError, SchedulingError)):
                raise
            raise SchedulingError(f"Failed to add task: {e}", schedule_id=schedule_id) from e
    
    def add_resource(self, schedule_id: str, resource: Resource) -> None:
        """Add resource to schedule with validation.
        
        Args:
            schedule_id: Schedule identifier
            resource: Resource to add
            
        Raises:
            ValidationError: If resource is invalid
            SchedulingError: If adding resource fails
        """
        try:
            schedule = self._get_schedule(schedule_id)
            
            # Validate resource
            self._validate_resource(resource)
            
            # Add resource
            schedule.add_resource(resource)
            
            self.logger.debug(f"Added resource {resource.id} to schedule {schedule_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add resource {resource.id} to schedule {schedule_id}: {e}")
            if isinstance(e, (ValidationError, SchedulingError)):
                raise
            raise SchedulingError(f"Failed to add resource: {e}", schedule_id=schedule_id) from e
    
    def optimize_schedule(self, schedule_id: str, 
                         algorithm: Optional[str] = None,
                         async_optimization: bool = False,
                         optimization_config: Optional[ScheduleOptimization] = None) -> Union[Schedule, Any]:
        """Optimize schedule using quantum algorithms.
        
        Args:
            schedule_id: Schedule to optimize
            algorithm: Algorithm to use (default: config.default_algorithm)
            async_optimization: Whether to run optimization asynchronously
            optimization_config: Optimization configuration
            
        Returns:
            Optimized schedule (sync) or future (async)
            
        Raises:
            OptimizationError: If optimization fails
        """
        start_time = time.time()
        algorithm_name = algorithm or self.config.default_algorithm
        
        try:
            self.logger.info(f"Starting optimization of schedule {schedule_id} with {algorithm_name}")
            
            schedule = self._get_schedule(schedule_id)
            
            # Validate schedule for optimization
            self._validate_schedule_for_optimization(schedule)
            
            if async_optimization:
                # Submit async optimization
                future = self._submit_async_optimization(schedule, algorithm_name, optimization_config)
                self.optimization_futures[schedule_id] = future
                return future
            else:
                # Run synchronous optimization
                optimized_schedule = self._run_optimization(schedule, algorithm_name, optimization_config)
                
                # Record metrics
                duration = time.time() - start_time
                self.metrics.record_optimization(algorithm_name, duration, True)
                self.metrics.total_tasks_scheduled += len(optimized_schedule.tasks)
                self.metrics.total_resources_allocated += len(optimized_schedule.resources)
                
                self.logger.info(f"Schedule {schedule_id} optimized successfully in {duration:.2f}s")
                return optimized_schedule
                
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            self.metrics.record_optimization(algorithm_name, duration, False, error_type)
            
            self.logger.error(f"Optimization failed for schedule {schedule_id}: {e}")
            
            # Attempt graceful degradation
            if self.config.enable_graceful_degradation and algorithm_name != self.config.fallback_algorithm:
                self.logger.info(f"Attempting fallback optimization with {self.config.fallback_algorithm}")
                try:
                    return self.optimize_schedule(schedule_id, self.config.fallback_algorithm, 
                                                async_optimization, optimization_config)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback optimization also failed: {fallback_error}")
            
            if isinstance(e, OptimizationError):
                raise
            raise OptimizationError(f"Schedule optimization failed: {e}", 
                                  algorithm=algorithm_name) from e
    
    def get_schedule(self, schedule_id: str) -> Schedule:
        """Get schedule by ID.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            Schedule object
            
        Raises:
            SchedulingError: If schedule not found
        """
        return self._get_schedule(schedule_id)
    
    def get_optimization_status(self, schedule_id: str) -> Dict[str, Any]:
        """Get optimization status for schedule.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            Status information
        """
        try:
            schedule = self._get_schedule(schedule_id)
            
            status = {
                "schedule_id": schedule_id,
                "status": schedule.status.value,
                "created_at": schedule.created_at.isoformat(),
                "updated_at": schedule.updated_at.isoformat(),
                "optimized_at": schedule.optimized_at.isoformat() if schedule.optimized_at else None,
                "task_count": len(schedule.tasks),
                "resource_count": len(schedule.resources),
                "assignment_count": len(schedule.assignments)
            }
            
            # Add optimization future status if exists
            if schedule_id in self.optimization_futures:
                future = self.optimization_futures[schedule_id]
                status["async_optimization"] = {
                    "running": not future.done(),
                    "done": future.done(),
                    "cancelled": future.cancelled() if hasattr(future, 'cancelled') else False
                }
            
            # Add metrics if available
            if schedule.metrics:
                status["metrics"] = {
                    "makespan_seconds": schedule.metrics.makespan.total_seconds(),
                    "total_cost": schedule.metrics.total_cost,
                    "constraint_violations": schedule.metrics.constraint_violations,
                    "quantum_energy": schedule.metrics.quantum_energy,
                    "optimization_time_seconds": schedule.metrics.optimization_time.total_seconds(),
                    "iterations": schedule.metrics.iterations,
                    "converged": schedule.metrics.convergence_achieved
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization status for {schedule_id}: {e}")
            return {
                "schedule_id": schedule_id,
                "error": str(e),
                "status": "error"
            }
    
    def get_planner_metrics(self) -> Dict[str, Any]:
        """Get planner performance metrics.
        
        Returns:
            Metrics dictionary
        """
        return self.metrics.to_dict()
    
    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all active schedules.
        
        Returns:
            List of schedule summaries
        """
        try:
            schedules = []
            for schedule_id, schedule in self.active_schedules.items():
                schedules.append({
                    "id": schedule.id,
                    "name": schedule.name,
                    "status": schedule.status.value,
                    "task_count": len(schedule.tasks),
                    "resource_count": len(schedule.resources),
                    "created_at": schedule.created_at.isoformat(),
                    "updated_at": schedule.updated_at.isoformat()
                })
            
            return sorted(schedules, key=lambda x: x["updated_at"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to list schedules: {e}")
            return []
    
    def _initialize_algorithms(self) -> Dict[str, Any]:
        """Initialize available quantum algorithms."""
        algorithms = {}
        
        try:
            # Quantum Annealing
            algorithms["quantum_annealing"] = QuantumAnnealingScheduler()
            
            # QAOA Resource Allocator
            qaoa_params = QAOAParameters(
                layers=2,
                max_iterations=self.config.optimization_timeout_seconds // 2,
                parallel_workers=self.config.max_concurrent_optimizations
            )
            algorithms["qaoa"] = QAOAResourceAllocator(qaoa_params)
            
            self.logger.info(f"Initialized {len(algorithms)} quantum algorithms")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize algorithms: {e}")
            raise ConfigurationError(f"Algorithm initialization failed: {e}") from e
        
        return algorithms
    
    def _get_schedule(self, schedule_id: str) -> Schedule:
        """Get schedule with error handling."""
        if schedule_id not in self.active_schedules:
            raise SchedulingError(f"Schedule {schedule_id} not found", schedule_id=schedule_id)
        return self.active_schedules[schedule_id]
    
    def _validate_task(self, task: Task, schedule: Schedule) -> None:
        """Validate task for addition to schedule."""
        errors = []
        
        if not task.id or not task.id.strip():
            errors.append("Task ID cannot be empty")
        
        # Check for duplicate task ID
        if any(t.id == task.id for t in schedule.tasks):
            errors.append(f"Task {task.id} already exists in schedule")
        
        if not task.name or not task.name.strip():
            errors.append("Task name cannot be empty")
        
        if task.duration.total_seconds() <= 0:
            errors.append("Task duration must be positive")
        
        # Validate quantum parameters if enabled
        if self.config.validate_quantum_parameters:
            if not 0 <= task.quantum_weight <= 100:
                errors.append("Task quantum_weight must be between 0 and 100")
            
            if not 0 <= task.entanglement_factor <= 1:
                errors.append("Task entanglement_factor must be between 0 and 1")
        
        if errors:
            raise ValidationError(f"Invalid task {task.id}: {'; '.join(errors)}", errors)
    
    def _validate_resource(self, resource: Resource) -> None:
        """Validate resource for addition to schedule."""
        errors = []
        
        if not resource.id or not resource.id.strip():
            errors.append("Resource ID cannot be empty")
        
        if not resource.name or not resource.name.strip():
            errors.append("Resource name cannot be empty")
        
        if resource.total_capacity <= 0:
            errors.append("Resource total_capacity must be positive")
        
        if resource.available_capacity < 0:
            errors.append("Resource available_capacity cannot be negative")
        
        if resource.available_capacity > resource.total_capacity:
            errors.append("Resource available_capacity cannot exceed total_capacity")
        
        # Validate quantum parameters if enabled
        if self.config.validate_quantum_parameters:
            if not 0 <= resource.quantum_coherence <= 1:
                errors.append("Resource quantum_coherence must be between 0 and 1")
            
            if not 0 <= resource.superposition_factor <= 1:
                errors.append("Resource superposition_factor must be between 0 and 1")
        
        if errors:
            raise ValidationError(f"Invalid resource {resource.id}: {'; '.join(errors)}", errors)
    
    def _validate_schedule_for_optimization(self, schedule: Schedule) -> None:
        """Validate schedule is ready for optimization."""
        errors = []
        
        if not schedule.tasks:
            errors.append("Schedule has no tasks to optimize")
        
        if not schedule.resources:
            errors.append("Schedule has no resources available")
        
        # Check for available resources
        available_resources = [r for r in schedule.resources if r.status == ResourceStatus.AVAILABLE]
        if not available_resources:
            errors.append("No resources are available for allocation")
        
        # Validate dependencies
        task_ids = {task.id for task in schedule.tasks}
        for task in schedule.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    errors.append(f"Task {task.id} has invalid dependency {dep_id}")
        
        if errors:
            raise ValidationError(f"Schedule {schedule.id} not ready for optimization: {'; '.join(errors)}", errors)
    
    def _run_optimization(self, schedule: Schedule, algorithm: str, 
                         config: Optional[ScheduleOptimization] = None) -> Schedule:
        """Run optimization with retry logic."""
        for attempt in range(self.config.max_retry_attempts):
            try:
                if algorithm not in self.algorithms:
                    raise OptimizationError(f"Algorithm {algorithm} not available")
                
                optimizer = self.algorithms[algorithm]
                
                # Set timeout
                timeout_start = time.time()
                
                # Run optimization based on algorithm type
                if algorithm == "quantum_annealing":
                    metrics = optimizer.optimize_schedule(schedule)
                elif algorithm == "qaoa":
                    metrics = optimizer.allocate_resources(schedule)
                else:
                    raise OptimizationError(f"Unknown algorithm: {algorithm}")
                
                # Check timeout
                if time.time() - timeout_start > self.config.optimization_timeout_seconds:
                    raise TimeoutError(f"Optimization timeout after {self.config.optimization_timeout_seconds}s")
                
                return schedule
                
            except Exception as e:
                self.logger.warning(f"Optimization attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    raise OptimizationError(f"Optimization failed after {self.config.max_retry_attempts} attempts: {e}")
    
    def _submit_async_optimization(self, schedule: Schedule, algorithm: str,
                                  config: Optional[ScheduleOptimization] = None) -> Any:
        """Submit optimization for async execution."""
        if not self.executor:
            raise ConfigurationError("Async optimization requires parallel processing to be enabled")
        
        return self.executor.submit(self._run_optimization, schedule, algorithm, config)
    
    def _export_metrics(self) -> None:
        """Export metrics to configured path."""
        try:
            if not self.config.metrics_export_path:
                return
            
            metrics_path = self.config.metrics_export_path
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {metrics_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")