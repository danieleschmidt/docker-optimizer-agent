"""Resource models for quantum-inspired task planning."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator


class ResourceType(str, Enum):
    """Resource type classifications."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    WORKER = "worker"
    LICENSE = "license"
    CUSTOM = "custom"


class ResourceStatus(str, Enum):
    """Resource availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class ResourceAllocation:
    """Resource allocation record."""
    task_id: str
    amount: float
    start_time: datetime
    end_time: datetime
    priority: int = 1


class Resource(BaseModel):
    """Quantum-inspired resource model with optimization features."""
    
    id: str = Field(..., description="Unique resource identifier")
    name: str = Field(..., description="Human-readable resource name")
    type: ResourceType = Field(..., description="Resource type")
    
    # Capacity management
    total_capacity: float = Field(..., description="Total resource capacity", gt=0)
    available_capacity: float = Field(..., description="Currently available capacity", ge=0)
    reserved_capacity: float = Field(0.0, description="Reserved capacity", ge=0)
    
    # Status and scheduling
    status: ResourceStatus = Field(ResourceStatus.AVAILABLE, description="Current status")
    allocations: List[ResourceAllocation] = Field(default_factory=list, description="Current allocations")
    
    # Cost and efficiency
    cost_per_unit: float = Field(0.0, description="Cost per unit time", ge=0)
    efficiency_rating: float = Field(1.0, description="Resource efficiency rating", ge=0, le=1)
    
    # Quantum-inspired properties
    quantum_coherence: float = Field(1.0, description="Resource coherence for optimization", ge=0, le=1)
    entanglement_capacity: int = Field(1, description="Max concurrent task entanglement", ge=1)
    superposition_factor: float = Field(0.0, description="Multi-state processing capability", ge=0, le=1)
    
    # Physical properties
    location: Optional[str] = Field(None, description="Resource location/zone")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Custom attributes")
    
    # Constraints and preferences
    compatible_tasks: Set[str] = Field(default_factory=set, description="Compatible task types")
    incompatible_tasks: Set[str] = Field(default_factory=set, description="Incompatible task types")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: Set[str] = Field(default_factory=set, description="Resource tags")
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
    
    @validator('available_capacity')
    def validate_available_capacity(cls, v, values):
        if 'total_capacity' in values and v > values['total_capacity']:
            raise ValueError("Available capacity cannot exceed total capacity")
        return v
    
    @validator('reserved_capacity')
    def validate_reserved_capacity(cls, v, values):
        if 'total_capacity' in values and v > values['total_capacity']:
            raise ValueError("Reserved capacity cannot exceed total capacity")
        return v
    
    def allocate(self, task_id: str, amount: float, start_time: datetime, 
                 duration: timedelta, priority: int = 1) -> bool:
        """Allocate resource capacity to a task."""
        if not self.can_allocate(amount, start_time, duration):
            return False
        
        end_time = start_time + duration
        allocation = ResourceAllocation(
            task_id=task_id,
            amount=amount,
            start_time=start_time,
            end_time=end_time,
            priority=priority
        )
        
        self.allocations.append(allocation)
        self.available_capacity -= amount
        self.updated_at = datetime.utcnow()
        
        # Update status if fully allocated
        if self.available_capacity <= 0:
            self.status = ResourceStatus.BUSY
        
        return True
    
    def deallocate(self, task_id: str) -> bool:
        """Remove task allocation from resource."""
        for i, allocation in enumerate(self.allocations):
            if allocation.task_id == task_id:
                self.available_capacity += allocation.amount
                del self.allocations[i]
                self.updated_at = datetime.utcnow()
                
                # Update status if capacity becomes available
                if self.available_capacity > 0 and self.status == ResourceStatus.BUSY:
                    self.status = ResourceStatus.AVAILABLE
                
                return True
        return False
    
    def can_allocate(self, amount: float, start_time: datetime, 
                     duration: timedelta) -> bool:
        """Check if resource can accommodate allocation."""
        if self.status != ResourceStatus.AVAILABLE:
            return False
        
        if amount > self.available_capacity:
            return False
        
        end_time = start_time + duration
        
        # Check for conflicts with existing allocations
        for allocation in self.allocations:
            if (start_time < allocation.end_time and 
                end_time > allocation.start_time):
                # Overlapping allocation
                if self.available_capacity - amount < 0:
                    return False
        
        return True
    
    def get_utilization(self, time_window: Optional[timedelta] = None) -> float:
        """Calculate resource utilization percentage."""
        if time_window is None:
            # Current utilization
            return 1.0 - (self.available_capacity / self.total_capacity)
        
        # Historical utilization over time window
        current_time = datetime.utcnow()
        start_time = current_time - time_window
        
        total_allocated_time = 0.0
        total_time = time_window.total_seconds()
        
        for allocation in self.allocations:
            # Calculate overlap with time window
            overlap_start = max(allocation.start_time, start_time)
            overlap_end = min(allocation.end_time, current_time)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                utilization_fraction = allocation.amount / self.total_capacity
                total_allocated_time += overlap_duration * utilization_fraction
        
        return min(1.0, total_allocated_time / total_time)
    
    def calculate_quantum_affinity(self, task_quantum_weight: float, 
                                   task_entanglement: float) -> float:
        """Calculate quantum affinity between resource and task."""
        # Base compatibility
        base_affinity = self.efficiency_rating * self.quantum_coherence
        
        # Quantum resonance
        weight_resonance = 1.0 - abs(self.quantum_coherence - task_quantum_weight)
        entanglement_bonus = self.superposition_factor * task_entanglement
        
        # Capacity factor
        capacity_factor = self.available_capacity / self.total_capacity
        
        return base_affinity * weight_resonance * (1 + entanglement_bonus) * capacity_factor
    
    def cleanup_expired_allocations(self, current_time: Optional[datetime] = None) -> int:
        """Remove expired allocations and update capacity."""
        if current_time is None:
            current_time = datetime.utcnow()
        
        expired_count = 0
        updated_allocations = []
        
        for allocation in self.allocations:
            if allocation.end_time <= current_time:
                self.available_capacity += allocation.amount
                expired_count += 1
            else:
                updated_allocations.append(allocation)
        
        self.allocations = updated_allocations
        
        if expired_count > 0:
            self.updated_at = current_time
            # Update status if capacity becomes available
            if self.available_capacity > 0 and self.status == ResourceStatus.BUSY:
                self.status = ResourceStatus.AVAILABLE
        
        return expired_count
    
    def to_dict(self) -> Dict:
        """Convert resource to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "total_capacity": self.total_capacity,
            "available_capacity": self.available_capacity,
            "status": self.status.value,
            "cost_per_unit": self.cost_per_unit,
            "efficiency_rating": self.efficiency_rating,
            "quantum_coherence": self.quantum_coherence,
            "location": self.location,
            "tags": list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Resource':
        """Create resource from dictionary representation."""
        data = data.copy()
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = set(data['tags'])
        if 'compatible_tasks' in data and isinstance(data['compatible_tasks'], list):
            data['compatible_tasks'] = set(data['compatible_tasks'])
        if 'incompatible_tasks' in data and isinstance(data['incompatible_tasks'], list):
            data['incompatible_tasks'] = set(data['incompatible_tasks'])
        return cls(**data)