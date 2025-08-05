"""Tests for quantum task planner models."""

import pytest
from datetime import datetime, timedelta

from src.quantum_task_planner.models.task import Task, TaskStatus, TaskPriority
from src.quantum_task_planner.models.resource import Resource, ResourceType, ResourceStatus
from src.quantum_task_planner.models.schedule import Schedule, ScheduleStatus, OptimizationObjective


class TestTask:
    """Test Task model."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            id="test_task",
            name="Test Task",
            duration=timedelta(hours=2),
            priority=TaskPriority.HIGH
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task" 
        assert task.duration == timedelta(hours=2)
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert len(task.dependencies) == 0
    
    def test_task_validation_errors(self):
        """Test task validation."""
        # Invalid duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            Task(
                id="invalid",
                name="Invalid",
                duration=timedelta(seconds=-1)
            )
    
    def test_is_ready(self):
        """Test task readiness check."""
        task = Task(
            id="test",
            name="Test", 
            duration=timedelta(hours=1),
            dependencies={"dep1", "dep2"}
        )
        
        # Not ready - missing dependencies
        assert not task.is_ready(set())
        assert not task.is_ready({"dep1"})
        
        # Ready - all dependencies completed
        assert task.is_ready({"dep1", "dep2"})


class TestResource:
    """Test Resource model."""
    
    def test_resource_creation(self):
        """Test basic resource creation."""
        resource = Resource(
            id="test_resource",
            name="Test Resource",
            type=ResourceType.CPU,
            total_capacity=4.0,
            available_capacity=3.0
        )
        
        assert resource.id == "test_resource"
        assert resource.type == ResourceType.CPU
        assert resource.total_capacity == 4.0
        assert resource.available_capacity == 3.0
        assert resource.status == ResourceStatus.AVAILABLE
    
    def test_allocate(self):
        """Test resource allocation."""
        resource = Resource(
            id="test",
            name="Test",
            type=ResourceType.CPU,
            total_capacity=4.0,
            available_capacity=4.0
        )
        
        start_time = datetime.utcnow()
        duration = timedelta(hours=1)
        
        # Successful allocation
        success = resource.allocate("task1", 2.0, start_time, duration)
        assert success
        assert resource.available_capacity == 2.0
        assert len(resource.allocations) == 1


class TestSchedule:
    """Test Schedule model."""
    
    def test_schedule_creation(self):
        """Test schedule creation."""
        start_time = datetime.utcnow()
        schedule = Schedule(
            id="test_schedule",
            name="Test Schedule",
            start_time=start_time,
            objectives=[OptimizationObjective.MINIMIZE_MAKESPAN]
        )
        
        assert schedule.id == "test_schedule"
        assert schedule.name == "Test Schedule"
        assert schedule.start_time == start_time
        assert schedule.status == ScheduleStatus.DRAFT
        assert OptimizationObjective.MINIMIZE_MAKESPAN in schedule.objectives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])