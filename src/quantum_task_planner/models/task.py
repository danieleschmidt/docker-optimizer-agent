"""Task models for quantum-inspired task planning."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TaskConstraint:
    """Task execution constraint."""
    type: str
    value: str
    weight: float = 1.0
    is_hard: bool = True


class Task(BaseModel):
    """Quantum-inspired task model with optimization features."""

    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: Optional[str] = Field(None, description="Task description")

    # Execution properties
    duration: timedelta = Field(..., description="Estimated execution duration")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current status")

    # Dependencies
    dependencies: Set[str] = Field(default_factory=set, description="Task IDs this task depends on")
    dependents: Set[str] = Field(default_factory=set, description="Task IDs that depend on this task")

    # Resource requirements
    resource_requirements: Dict[str, float] = Field(default_factory=dict, description="Resource type to quantity mapping")
    resource_preferences: Dict[str, float] = Field(default_factory=dict, description="Preferred resource allocation")

    # Constraints
    constraints: List[TaskConstraint] = Field(default_factory=list, description="Task constraints")

    # Quantum-inspired properties
    quantum_weight: float = Field(1.0, description="Quantum optimization weight", ge=0.0)
    entanglement_factor: float = Field(0.0, description="Task interdependency strength", ge=0.0, le=1.0)
    superposition_states: List[str] = Field(default_factory=list, description="Alternative execution states")

    # Timing
    earliest_start: Optional[datetime] = Field(None, description="Earliest possible start time")
    latest_finish: Optional[datetime] = Field(None, description="Latest acceptable finish time")
    scheduled_start: Optional[datetime] = Field(None, description="Scheduled start time")
    scheduled_finish: Optional[datetime] = Field(None, description="Scheduled finish time")
    actual_start: Optional[datetime] = Field(None, description="Actual start time")
    actual_finish: Optional[datetime] = Field(None, description="Actual finish time")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: Set[str] = Field(default_factory=set, description="Task tags for categorization")

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    @validator('duration')
    def validate_duration(cls, v):
        if v.total_seconds() <= 0:
            raise ValueError("Duration must be positive")
        return v

    @validator('dependencies', 'dependents')
    def validate_no_self_dependency(cls, v, values):
        if 'id' in values and values['id'] in v:
            raise ValueError("Task cannot depend on itself")
        return v

    def add_dependency(self, task_id: str) -> None:
        """Add a dependency to this task."""
        if task_id != self.id:
            self.dependencies.add(task_id)
            self.updated_at = datetime.utcnow()

    def remove_dependency(self, task_id: str) -> None:
        """Remove a dependency from this task."""
        self.dependencies.discard(task_id)
        self.updated_at = datetime.utcnow()

    def add_dependent(self, task_id: str) -> None:
        """Add a dependent task."""
        if task_id != self.id:
            self.dependents.add(task_id)
            self.updated_at = datetime.utcnow()

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute based on dependencies."""
        return (
            self.status == TaskStatus.PENDING and
            self.dependencies.issubset(completed_tasks)
        )

    def can_start_at(self, start_time: datetime) -> bool:
        """Check if task can start at given time considering constraints."""
        if self.earliest_start and start_time < self.earliest_start:
            return False

        if self.latest_finish:
            projected_finish = start_time + self.duration
            if projected_finish > self.latest_finish:
                return False

        return True

    def calculate_quantum_energy(self, current_time: datetime, system_load: float = 0.0) -> float:
        """Calculate quantum energy for optimization purposes."""
        base_energy = 1.0

        # Priority influence
        priority_weights = {
            TaskPriority.CRITICAL: 4.0,
            TaskPriority.HIGH: 2.0,
            TaskPriority.MEDIUM: 1.0,
            TaskPriority.LOW: 0.5
        }
        base_energy *= priority_weights[self.priority]

        # Time urgency
        if self.latest_finish:
            time_remaining = (self.latest_finish - current_time).total_seconds()
            duration_seconds = self.duration.total_seconds()
            urgency = duration_seconds / max(time_remaining, 1)
            base_energy *= (1 + urgency)

        # System load influence
        base_energy *= (1 + system_load * 0.5)

        # Quantum weight
        base_energy *= self.quantum_weight

        return base_energy

    def to_dict(self) -> Dict:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "duration_seconds": self.duration.total_seconds(),
            "priority": self.priority.value,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "resource_requirements": self.resource_requirements,
            "quantum_weight": self.quantum_weight,
            "entanglement_factor": self.entanglement_factor,
            "tags": list(self.tags)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create task from dictionary representation."""
        data = data.copy()
        if 'duration_seconds' in data:
            data['duration'] = timedelta(seconds=data.pop('duration_seconds'))
        if 'dependencies' in data and isinstance(data['dependencies'], list):
            data['dependencies'] = set(data['dependencies'])
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = set(data['tags'])
        return cls(**data)
