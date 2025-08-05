"""Schedule models for quantum-inspired task planning."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .task import Task, TaskStatus
from .resource import Resource


class ScheduleStatus(str, Enum):
    """Schedule optimization status."""
    DRAFT = "draft"
    OPTIMIZING = "optimizing"
    OPTIMIZED = "optimized"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCE_LOAD = "balance_load"
    QUANTUM_OPTIMAL = "quantum_optimal"


@dataclass
class TaskAssignment:
    """Task-to-resource assignment with timing."""
    task_id: str
    resource_id: str
    start_time: datetime
    end_time: datetime
    allocated_capacity: float
    priority: int = 1


@dataclass
class OptimizationMetrics:
    """Schedule optimization metrics."""
    makespan: timedelta
    total_cost: float
    resource_utilization: Dict[str, float]
    constraint_violations: int
    quantum_energy: float
    optimization_time: timedelta
    iterations: int
    convergence_achieved: bool


class Schedule(BaseModel):
    """Quantum-inspired schedule with optimization tracking."""
    
    id: str = Field(..., description="Unique schedule identifier")
    name: str = Field(..., description="Schedule name")
    description: Optional[str] = Field(None, description="Schedule description")
    
    # Schedule content
    tasks: List[Task] = Field(default_factory=list, description="Tasks in schedule")
    resources: List[Resource] = Field(default_factory=list, description="Available resources")
    assignments: List[TaskAssignment] = Field(default_factory=list, description="Task-resource assignments")
    
    # Schedule properties
    status: ScheduleStatus = Field(ScheduleStatus.DRAFT, description="Current status")
    start_time: datetime = Field(..., description="Schedule start time")
    end_time: Optional[datetime] = Field(None, description="Schedule end time")
    
    # Optimization configuration
    objectives: List[OptimizationObjective] = Field(
        default_factory=lambda: [OptimizationObjective.MINIMIZE_MAKESPAN],
        description="Optimization objectives"
    )
    objective_weights: Dict[str, float] = Field(
        default_factory=dict, 
        description="Weights for multi-objective optimization"
    )
    
    # Quantum-inspired properties
    quantum_temperature: float = Field(1.0, description="Quantum annealing temperature", ge=0)
    entanglement_strength: float = Field(0.5, description="Task entanglement strength", ge=0, le=1)
    superposition_exploration: bool = Field(True, description="Enable superposition exploration")
    coherence_time: timedelta = Field(
        default_factory=lambda: timedelta(minutes=30),
        description="Quantum coherence duration"
    )
    
    # Constraints
    hard_constraints: List[str] = Field(default_factory=list, description="Hard constraints")
    soft_constraints: List[str] = Field(default_factory=list, description="Soft constraints")
    constraint_weights: Dict[str, float] = Field(default_factory=dict, description="Soft constraint weights")
    
    # Metrics and tracking
    metrics: Optional[OptimizationMetrics] = Field(None, description="Optimization metrics")
    optimization_history: List[Dict] = Field(default_factory=list, description="Optimization history")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    optimized_at: Optional[datetime] = Field(None, description="Last optimization time")
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
    
    def add_task(self, task: Task) -> None:
        """Add task to schedule."""
        if not any(t.id == task.id for t in self.tasks):
            self.tasks.append(task)
            self.updated_at = datetime.utcnow()
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from schedule."""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                del self.tasks[i]
                # Remove related assignments
                self.assignments = [a for a in self.assignments if a.task_id != task_id]
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def add_resource(self, resource: Resource) -> None:
        """Add resource to schedule."""
        if not any(r.id == resource.id for r in self.resources):
            self.resources.append(resource)
            self.updated_at = datetime.utcnow()
    
    def assign_task(self, task_id: str, resource_id: str, start_time: datetime,
                    allocated_capacity: float = 1.0, priority: int = 1) -> bool:
        """Assign task to resource."""
        # Validate task exists
        task = self.get_task(task_id)
        if not task:
            return False
        
        # Validate resource exists
        resource = self.get_resource(resource_id)
        if not resource:
            return False
        
        # Calculate end time
        end_time = start_time + task.duration
        
        # Check resource availability
        if not resource.can_allocate(allocated_capacity, start_time, task.duration):
            return False
        
        # Remove existing assignment if present
        self.unassign_task(task_id)
        
        # Create assignment
        assignment = TaskAssignment(
            task_id=task_id,
            resource_id=resource_id,
            start_time=start_time,
            end_time=end_time,
            allocated_capacity=allocated_capacity,
            priority=priority
        )
        
        self.assignments.append(assignment)
        
        # Update resource allocation
        resource.allocate(task_id, allocated_capacity, start_time, task.duration, priority)
        
        # Update task schedule
        task.scheduled_start = start_time
        task.scheduled_finish = end_time
        task.status = TaskStatus.READY
        
        self.updated_at = datetime.utcnow()
        return True
    
    def unassign_task(self, task_id: str) -> bool:
        """Remove task assignment."""
        for i, assignment in enumerate(self.assignments):
            if assignment.task_id == task_id:
                # Free resource allocation
                resource = self.get_resource(assignment.resource_id)
                if resource:
                    resource.deallocate(task_id)
                
                # Update task
                task = self.get_task(task_id)
                if task:
                    task.scheduled_start = None
                    task.scheduled_finish = None
                    task.status = TaskStatus.PENDING
                
                del self.assignments[i]
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get resource by ID."""
        for resource in self.resources:
            if resource.id == resource_id:
                return resource
        return None
    
    def get_assignment(self, task_id: str) -> Optional[TaskAssignment]:
        """Get task assignment."""
        for assignment in self.assignments:
            if assignment.task_id == task_id:
                return assignment
        return None
    
    def calculate_makespan(self) -> timedelta:
        """Calculate schedule makespan."""
        if not self.assignments:
            return timedelta(0)
        
        max_end_time = max(assignment.end_time for assignment in self.assignments)
        return max_end_time - self.start_time
    
    def calculate_total_cost(self) -> float:
        """Calculate total schedule cost."""
        total_cost = 0.0
        
        for assignment in self.assignments:
            resource = self.get_resource(assignment.resource_id)
            if resource:
                duration_hours = (assignment.end_time - assignment.start_time).total_seconds() / 3600
                cost = duration_hours * resource.cost_per_unit * assignment.allocated_capacity
                total_cost += cost
        
        return total_cost
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization rates."""
        utilization = {}
        
        for resource in self.resources:
            total_time = self.calculate_makespan().total_seconds()
            if total_time == 0:
                utilization[resource.id] = 0.0
                continue
            
            allocated_time = 0.0
            for assignment in self.assignments:
                if assignment.resource_id == resource.id:
                    duration = (assignment.end_time - assignment.start_time).total_seconds()
                    allocated_time += duration * assignment.allocated_capacity
            
            utilization[resource.id] = allocated_time / (total_time * resource.total_capacity)
        
        return utilization
    
    def calculate_quantum_energy(self) -> float:
        """Calculate total quantum energy of schedule."""
        total_energy = 0.0
        current_time = datetime.utcnow()
        
        for task in self.tasks:
            task_energy = task.calculate_quantum_energy(current_time)
            
            # Apply entanglement effects
            if task.entanglement_factor > 0:
                entangled_tasks = [t for t in self.tasks if t.id in task.dependencies or task.id in t.dependencies]
                entanglement_energy = sum(t.quantum_weight for t in entangled_tasks) * task.entanglement_factor
                task_energy += entanglement_energy
            
            total_energy += task_energy
        
        # Apply global quantum effects
        total_energy *= (1 + self.entanglement_strength * len(self.tasks) * 0.1)
        
        return total_energy
    
    def validate_dependencies(self) -> List[str]:
        """Validate task dependencies are satisfied in schedule."""
        violations = []
        
        for assignment in self.assignments:
            task = self.get_task(assignment.task_id)
            if not task:
                continue
            
            for dep_id in task.dependencies:
                dep_assignment = self.get_assignment(dep_id)
                if not dep_assignment:
                    violations.append(f"Task {task.id} depends on unscheduled task {dep_id}")
                elif dep_assignment.end_time > assignment.start_time:
                    violations.append(f"Task {task.id} starts before dependency {dep_id} completes")
        
        return violations
    
    def validate_resource_constraints(self) -> List[str]:
        """Validate resource constraints are satisfied."""
        violations = []
        
        # Check for resource over-allocation
        for resource in self.resources:
            time_slots = {}
            
            for assignment in self.assignments:
                if assignment.resource_id == resource.id:
                    # Discretize time into slots for checking
                    current_time = assignment.start_time
                    while current_time < assignment.end_time:
                        slot_key = current_time.strftime("%Y-%m-%d %H:%M")
                        if slot_key not in time_slots:
                            time_slots[slot_key] = 0.0
                        time_slots[slot_key] += assignment.allocated_capacity
                        current_time += timedelta(minutes=15)  # 15-minute slots
            
            # Check for over-allocation
            for slot, allocation in time_slots.items():
                if allocation > resource.total_capacity:
                    violations.append(
                        f"Resource {resource.id} over-allocated at {slot}: "
                        f"{allocation:.2f}/{resource.total_capacity:.2f}"
                    )
        
        return violations
    
    def is_valid(self) -> Tuple[bool, List[str]]:
        """Check if schedule is valid."""
        violations = []
        violations.extend(self.validate_dependencies())
        violations.extend(self.validate_resource_constraints())
        
        return len(violations) == 0, violations
    
    def to_dict(self) -> Dict:
        """Convert schedule to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "task_count": len(self.tasks),
            "resource_count": len(self.resources),
            "assignment_count": len(self.assignments),
            "makespan_seconds": self.calculate_makespan().total_seconds(),
            "total_cost": self.calculate_total_cost(),
            "quantum_energy": self.calculate_quantum_energy(),
            "objectives": [obj.value for obj in self.objectives]
        }


class ScheduleOptimization(BaseModel):
    """Quantum-inspired schedule optimization configuration."""
    
    # Algorithm selection
    algorithm: str = Field("quantum_annealing", description="Optimization algorithm")
    max_iterations: int = Field(1000, description="Maximum optimization iterations", gt=0)
    convergence_threshold: float = Field(0.001, description="Convergence threshold", gt=0)
    
    # Quantum-inspired parameters
    initial_temperature: float = Field(10.0, description="Initial quantum temperature", gt=0)
    final_temperature: float = Field(0.01, description="Final quantum temperature", gt=0)
    cooling_rate: float = Field(0.95, description="Temperature cooling rate", gt=0, lt=1)
    
    # Multi-objective optimization
    objective_weights: Dict[str, float] = Field(
        default_factory=lambda: {"makespan": 0.4, "cost": 0.3, "utilization": 0.3},
        description="Objective weights"
    )
    
    # Quantum exploration
    tunneling_probability: float = Field(0.1, description="Quantum tunneling probability", ge=0, le=1)
    superposition_factor: float = Field(0.2, description="Superposition exploration factor", ge=0, le=1)
    entanglement_enabled: bool = Field(True, description="Enable quantum entanglement effects")
    
    # Constraints
    respect_hard_constraints: bool = Field(True, description="Strictly enforce hard constraints")
    soft_constraint_penalty: float = Field(100.0, description="Soft constraint violation penalty", ge=0)
    
    # Performance
    parallel_optimization: bool = Field(True, description="Enable parallel optimization")
    max_workers: int = Field(4, description="Maximum parallel workers", gt=0)
    
    class Config:
        use_enum_values = True