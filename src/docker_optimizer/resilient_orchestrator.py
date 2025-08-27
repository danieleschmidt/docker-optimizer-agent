"""Resilient orchestration system for Docker optimization workflows.

Provides fault-tolerant workflow execution, resource management, and
adaptive scaling capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil

from .enhanced_error_handling import (
    EnhancedErrorHandler,
    ErrorSeverity,
    RecoveryStrategy,
    with_circuit_breaker,
    retry_on_failure
)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEGRADED = "degraded"


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DOCKER_DAEMON = "docker_daemon"
    EXTERNAL_SERVICE = "external_service"


class WorkflowPriority(Enum):
    """Workflow execution priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResourceConstraint:
    """Resource usage constraint for workflows."""
    resource_type: ResourceType
    max_usage: float  # Percentage or absolute value
    reserved: float = 0.0  # Reserved amount
    burst_allowed: bool = True  # Allow temporary bursts
    
    
@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = time.time()
        self.duration = self.end_time - self.start_time


@dataclass
class WorkflowTask:
    """Individual task within a workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    function: Optional[callable] = None
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    resource_constraints: List[ResourceConstraint] = field(default_factory=list)
    metrics: WorkflowMetrics = field(default_factory=WorkflowMetrics)


@dataclass 
class Workflow:
    """Complete workflow definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    tasks: List[WorkflowTask] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: WorkflowMetrics = field(default_factory=WorkflowMetrics)
    max_parallel_tasks: int = 4
    
    def add_task(self, task: WorkflowTask) -> None:
        """Add task to workflow."""
        self.tasks.append(task)
    
    def get_ready_tasks(self) -> List[WorkflowTask]:
        """Get tasks that are ready to execute."""
        completed_task_ids = {
            task.id for task in self.tasks 
            if task.status == WorkflowStatus.COMPLETED
        }
        
        return [
            task for task in self.tasks
            if (task.status == WorkflowStatus.PENDING and 
                task.dependencies.issubset(completed_task_ids))
        ]
    
    def is_completed(self) -> bool:
        """Check if workflow is completed."""
        return all(
            task.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
            for task in self.tasks
        )
    
    def has_failed_tasks(self) -> bool:
        """Check if workflow has any failed tasks."""
        return any(task.status == WorkflowStatus.FAILED for task in self.tasks)


class ResourceMonitor:
    """System resource monitoring and constraint enforcement."""
    
    def __init__(self):
        self.constraints: List[ResourceConstraint] = []
        self.monitoring = False
        self.metrics_history: List[Dict[str, float]] = []
        
    def add_constraint(self, constraint: ResourceConstraint) -> None:
        """Add resource constraint."""
        self.constraints.append(constraint)
    
    def get_current_usage(self) -> Dict[ResourceType, float]:
        """Get current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Network I/O (simplified)
        net_io = psutil.net_io_counters()
        network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
        
        return {
            ResourceType.CPU: cpu_percent,
            ResourceType.MEMORY: memory_percent,
            ResourceType.DISK: disk_usage,
            ResourceType.NETWORK: network_usage
        }
    
    def can_execute_task(self, task: WorkflowTask) -> Tuple[bool, List[str]]:
        """Check if task can be executed given resource constraints."""
        current_usage = self.get_current_usage()
        violations = []
        
        for constraint in task.resource_constraints:
            if constraint.resource_type in current_usage:
                usage = current_usage[constraint.resource_type]
                if usage > constraint.max_usage:
                    violations.append(
                        f"{constraint.resource_type.value} usage ({usage:.1f}%) "
                        f"exceeds limit ({constraint.max_usage:.1f}%)"
                    )
        
        return len(violations) == 0, violations
    
    async def monitor_resources(self, interval: float = 5.0) -> None:
        """Monitor system resources continuously."""
        self.monitoring = True
        
        while self.monitoring:
            current_usage = self.get_current_usage()
            timestamp = time.time()
            
            metrics = {
                "timestamp": timestamp,
                **{rt.value: usage for rt, usage in current_usage.items()}
            }
            
            self.metrics_history.append(metrics)
            
            # Keep only last 100 metrics
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False


class ResilientOrchestrator:
    """Resilient workflow orchestration system."""
    
    def __init__(
        self,
        max_concurrent_workflows: int = 3,
        resource_monitor: Optional[ResourceMonitor] = None
    ):
        self.max_concurrent_workflows = max_concurrent_workflows
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.error_handler = EnhancedErrorHandler()
        self.logger = logging.getLogger(__name__)
        
        # Workflow management
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_queue: List[Workflow] = []
        self.workflow_history: List[Workflow] = []
        
        # Task execution
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_semaphore = asyncio.Semaphore(10)  # Max concurrent tasks
        
        # Metrics and monitoring
        self.orchestrator_metrics = {
            "workflows_executed": 0,
            "tasks_executed": 0,
            "total_errors": 0,
            "average_workflow_duration": 0.0,
            "resource_violations": 0
        }
        
    async def start(self) -> None:
        """Start the orchestrator."""
        self.logger.info("Starting resilient orchestrator")
        
        # Start resource monitoring
        asyncio.create_task(self.resource_monitor.monitor_resources())
        
        # Start workflow processing
        asyncio.create_task(self._process_workflow_queue())
        
    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        self.logger.info("Stopping resilient orchestrator")
        
        # Cancel running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
    
    def submit_workflow(self, workflow: Workflow) -> str:
        """Submit workflow for execution."""
        self.workflow_queue.append(workflow)
        self.logger.info(f"Workflow {workflow.id} submitted: {workflow.name}")
        return workflow.id
    
    async def _process_workflow_queue(self) -> None:
        """Process the workflow queue continuously."""
        while True:
            try:
                # Check if we can start new workflows
                if (len(self.active_workflows) < self.max_concurrent_workflows 
                    and self.workflow_queue):
                    
                    # Sort queue by priority
                    self.workflow_queue.sort(key=lambda w: w.priority.value, reverse=True)
                    workflow = self.workflow_queue.pop(0)
                    
                    # Start workflow
                    asyncio.create_task(self._execute_workflow(workflow))
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in workflow queue processing: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _execute_workflow(self, workflow: Workflow) -> None:
        """Execute a complete workflow."""
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = time.time()
        workflow.metrics.start_time = workflow.started_at
        
        self.active_workflows[workflow.id] = workflow
        
        try:
            self.logger.info(f"Starting workflow {workflow.id}: {workflow.name}")
            
            while not workflow.is_completed():
                ready_tasks = workflow.get_ready_tasks()
                
                if not ready_tasks:
                    # Check if workflow is stuck
                    pending_tasks = [t for t in workflow.tasks if t.status == WorkflowStatus.PENDING]
                    if pending_tasks:
                        self.logger.error(f"Workflow {workflow.id} appears stuck with pending tasks")
                        workflow.status = WorkflowStatus.FAILED
                        break
                    else:
                        break  # All tasks completed
                
                # Execute ready tasks (respecting parallelism limits)
                running_task_count = len([
                    t for t in workflow.tasks 
                    if t.status == WorkflowStatus.RUNNING
                ])
                
                tasks_to_start = ready_tasks[:workflow.max_parallel_tasks - running_task_count]
                
                for task in tasks_to_start:
                    asyncio.create_task(self._execute_task(workflow, task))
                
                await asyncio.sleep(0.5)  # Brief pause between checks
            
            # Finalize workflow
            if workflow.has_failed_tasks():
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED
            
            workflow.completed_at = time.time()
            workflow.metrics.finalize()
            
            self.logger.info(
                f"Workflow {workflow.id} {workflow.status.value} in "
                f"{workflow.metrics.duration:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow.id} failed with error: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = time.time()
            workflow.metrics.finalize()
            
            await self.error_handler.handle_error_async(e, f"workflow_{workflow.id}")
            
        finally:
            # Move workflow to history
            if workflow.id in self.active_workflows:
                del self.active_workflows[workflow.id]
            
            self.workflow_history.append(workflow)
            
            # Keep only last 50 workflows in history
            if len(self.workflow_history) > 50:
                self.workflow_history = self.workflow_history[-50:]
            
            # Update metrics
            self.orchestrator_metrics["workflows_executed"] += 1
            if workflow.metrics.duration:
                self._update_average_duration(workflow.metrics.duration)
    
    async def _execute_task(self, workflow: Workflow, task: WorkflowTask) -> None:
        """Execute individual task with resilience."""
        async with self.task_semaphore:
            task.status = WorkflowStatus.RUNNING
            task.metrics.start_time = time.time()
            
            try:
                self.logger.debug(f"Starting task {task.id}: {task.name}")
                
                # Check resource constraints
                can_execute, violations = self.resource_monitor.can_execute_task(task)
                if not can_execute:
                    self.logger.warning(f"Task {task.id} delayed due to resource constraints: {violations}")
                    self.orchestrator_metrics["resource_violations"] += 1
                    
                    # Wait and retry
                    await asyncio.sleep(5)
                    can_execute, _ = self.resource_monitor.can_execute_task(task)
                    
                    if not can_execute:
                        raise Exception(f"Resource constraints not met: {violations}")
                
                # Execute task function
                if task.function:
                    if asyncio.iscoroutinefunction(task.function):
                        if task.timeout:
                            task.result = await asyncio.wait_for(
                                task.function(*task.args, **task.kwargs),
                                timeout=task.timeout
                            )
                        else:
                            task.result = await task.function(*task.args, **task.kwargs)
                    else:
                        task.result = task.function(*task.args, **task.kwargs)
                
                task.status = WorkflowStatus.COMPLETED
                task.metrics.finalize()
                
                self.logger.debug(
                    f"Task {task.id} completed in {task.metrics.duration:.2f}s"
                )
                
            except asyncio.TimeoutError:
                task.error = TimeoutError(f"Task {task.id} timed out after {task.timeout}s")
                await self._handle_task_failure(task)
                
            except Exception as e:
                task.error = e
                await self._handle_task_failure(task)
            
            finally:
                self.orchestrator_metrics["tasks_executed"] += 1
    
    async def _handle_task_failure(self, task: WorkflowTask) -> None:
        """Handle task failure with retry logic."""
        task.retry_count += 1
        self.orchestrator_metrics["total_errors"] += 1
        
        self.logger.warning(
            f"Task {task.id} failed (attempt {task.retry_count}/{task.max_retries}): {task.error}"
        )
        
        if task.retry_count <= task.max_retries:
            task.status = WorkflowStatus.RETRYING
            
            # Exponential backoff
            delay = min(2 ** task.retry_count, 30)  # Max 30 seconds
            await asyncio.sleep(delay)
            
            # Reset for retry
            task.status = WorkflowStatus.PENDING
            task.error = None
            
        else:
            task.status = WorkflowStatus.FAILED
            self.logger.error(f"Task {task.id} failed permanently after {task.max_retries} retries")
    
    def _update_average_duration(self, duration: float) -> None:
        """Update average workflow duration metric."""
        current_avg = self.orchestrator_metrics["average_workflow_duration"]
        executed_count = self.orchestrator_metrics["workflows_executed"]
        
        # Weighted average
        new_avg = ((current_avg * (executed_count - 1)) + duration) / executed_count
        self.orchestrator_metrics["average_workflow_duration"] = new_avg
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed workflow status."""
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
        else:
            # Check history
            workflow = next(
                (w for w in self.workflow_history if w.id == workflow_id),
                None
            )
        
        if not workflow:
            return None
        
        task_statuses = {
            status.value: len([t for t in workflow.tasks if t.status == status])
            for status in WorkflowStatus
        }
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "duration": workflow.metrics.duration,
            "task_count": len(workflow.tasks),
            "task_statuses": task_statuses,
            "priority": workflow.priority.value,
            "metadata": workflow.metadata
        }
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator metrics."""
        current_usage = self.resource_monitor.get_current_usage()
        
        return {
            "orchestrator": self.orchestrator_metrics.copy(),
            "active_workflows": len(self.active_workflows),
            "queued_workflows": len(self.workflow_queue),
            "running_tasks": len(self.running_tasks),
            "resource_usage": {
                rt.value: usage for rt, usage in current_usage.items()
            },
            "resource_constraints": len(self.resource_monitor.constraints),
            "timestamp": time.time()
        }
    
    async def create_optimization_workflow(
        self,
        dockerfile_path: str,
        config: Dict[str, Any]
    ) -> Workflow:
        """Create a complete Docker optimization workflow."""
        workflow = Workflow(
            name=f"Optimize {Path(dockerfile_path).name}",
            priority=WorkflowPriority.NORMAL,
            metadata={
                "dockerfile_path": dockerfile_path,
                "config": config
            }
        )
        
        # Task 1: Parse and validate Dockerfile
        parse_task = WorkflowTask(
            name="Parse Dockerfile",
            function=self._parse_dockerfile,
            args=(dockerfile_path,),
            timeout=30.0,
            resource_constraints=[
                ResourceConstraint(ResourceType.MEMORY, max_usage=70.0)
            ]
        )
        
        # Task 2: Security scan
        security_task = WorkflowTask(
            name="Security Scan",
            function=self._security_scan,
            args=(dockerfile_path,),
            dependencies={parse_task.id},
            timeout=120.0,
            resource_constraints=[
                ResourceConstraint(ResourceType.CPU, max_usage=80.0),
                ResourceConstraint(ResourceType.NETWORK, max_usage=50.0)
            ]
        )
        
        # Task 3: Optimization analysis
        optimize_task = WorkflowTask(
            name="Optimization Analysis",
            function=self._optimize_dockerfile,
            args=(dockerfile_path, config),
            dependencies={parse_task.id, security_task.id},
            timeout=60.0,
            resource_constraints=[
                ResourceConstraint(ResourceType.CPU, max_usage=70.0),
                ResourceConstraint(ResourceType.MEMORY, max_usage=60.0)
            ]
        )
        
        # Task 4: Generate report
        report_task = WorkflowTask(
            name="Generate Report",
            function=self._generate_report,
            dependencies={optimize_task.id},
            timeout=30.0
        )
        
        workflow.add_task(parse_task)
        workflow.add_task(security_task)
        workflow.add_task(optimize_task)
        workflow.add_task(report_task)
        
        return workflow
    
    # Placeholder task functions (to be implemented with actual functionality)
    async def _parse_dockerfile(self, dockerfile_path: str) -> Dict[str, Any]:
        """Parse Dockerfile task implementation."""
        # Simulate work
        await asyncio.sleep(1)
        return {"status": "parsed", "instructions": 10}
    
    async def _security_scan(self, dockerfile_path: str) -> Dict[str, Any]:
        """Security scan task implementation."""
        # Simulate work
        await asyncio.sleep(2)
        return {"status": "scanned", "vulnerabilities": 0}
    
    async def _optimize_dockerfile(self, dockerfile_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Dockerfile optimization task implementation."""
        # Simulate work
        await asyncio.sleep(3)
        return {"status": "optimized", "size_reduction": "40%"}
    
    async def _generate_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        # Simulate work
        await asyncio.sleep(1)
        return {"status": "completed", "report_path": "/tmp/report.json"}