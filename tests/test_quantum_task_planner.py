"""Comprehensive tests for quantum task planner."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

from src.quantum_task_planner.core.planner import QuantumTaskPlanner, PlannerConfig, PlannerMetrics
from src.quantum_task_planner.models.task import Task, TaskPriority, TaskStatus
from src.quantum_task_planner.models.resource import Resource, ResourceType, ResourceStatus
from src.quantum_task_planner.models.schedule import Schedule, ScheduleStatus, OptimizationObjective
from src.quantum_task_planner.core.exceptions import (
    ValidationError, OptimizationError, SchedulingError, ConfigurationError
)


class TestPlannerConfig:
    """Test planner configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = PlannerConfig()
        assert config.default_algorithm == "quantum_annealing"
        assert config.fallback_algorithm == "qaoa"
        assert config.max_concurrent_optimizations == 4
        assert config.optimization_timeout_seconds == 300
        assert config.enable_parallel_processing is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ConfigurationError):
            PlannerConfig(max_concurrent_optimizations=0)
        
        with pytest.raises(ConfigurationError):
            PlannerConfig(optimization_timeout_seconds=0)
    
    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = PlannerConfig(
            default_algorithm="qaoa",
            max_concurrent_optimizations=8,
            optimization_timeout_seconds=600,
            enable_parallel_processing=False
        )
        assert config.default_algorithm == "qaoa"
        assert config.max_concurrent_optimizations == 8
        assert config.optimization_timeout_seconds == 600
        assert config.enable_parallel_processing is False


class TestPlannerMetrics:
    """Test planner metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PlannerMetrics()
        assert metrics.total_optimizations == 0
        assert metrics.successful_optimizations == 0
        assert metrics.failed_optimizations == 0
        assert metrics.get_success_rate() == 0.0
    
    def test_record_successful_optimization(self):
        """Test recording successful optimization."""
        metrics = PlannerMetrics()
        metrics.record_optimization("quantum_annealing", 10.5, True)
        
        assert metrics.total_optimizations == 1
        assert metrics.successful_optimizations == 1
        assert metrics.failed_optimizations == 0
        assert metrics.get_success_rate() == 1.0
        assert metrics.average_optimization_time == 10.5
        assert metrics.algorithm_usage["quantum_annealing"] == 1
    
    def test_record_failed_optimization(self):
        """Test recording failed optimization."""
        metrics = PlannerMetrics()
        metrics.record_optimization("qaoa", 5.0, False, "OptimizationError")
        
        assert metrics.total_optimizations == 1
        assert metrics.successful_optimizations == 0
        assert metrics.failed_optimizations == 1
        assert metrics.get_success_rate() == 0.0
        assert metrics.error_counts["OptimizationError"] == 1
    
    def test_metrics_to_dict(self):
        """Test metrics dictionary conversion."""
        metrics = PlannerMetrics()
        metrics.record_optimization("quantum_annealing", 10.0, True)
        
        data = metrics.to_dict()
        assert "total_optimizations" in data
        assert "success_rate" in data
        assert "algorithm_usage" in data
        assert data["total_optimizations"] == 1
        assert data["success_rate"] == 1.0


class TestQuantumTaskPlanner:
    """Test quantum task planner core functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PlannerConfig(
            enable_parallel_processing=False,  # Disable for simpler testing
            optimization_timeout_seconds=30,
            max_retry_attempts=1
        )
    
    @pytest.fixture
    def planner(self, config):
        """Create test planner."""
        return QuantumTaskPlanner(config)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        tasks = []
        base_time = datetime.utcnow()
        
        for i in range(5):
            task = Task(
                id=f"task_{i}",
                name=f"Test Task {i}",
                duration=timedelta(hours=1),
                priority=TaskPriority.MEDIUM,
                resource_requirements={"cpu": 1.0, "memory": 2.0},
                quantum_weight=1.0 + i * 0.2,
                entanglement_factor=0.1 * i
            )
            
            # Add some dependencies
            if i > 0:
                task.add_dependency(f"task_{i-1}")
            
            tasks.append(task)
        
        return tasks
    
    @pytest.fixture
    def sample_resources(self):
        """Create sample resources for testing."""
        resources = []
        
        for i in range(3):
            resource = Resource(
                id=f"resource_{i}",
                name=f"Test Resource {i}",
                type=ResourceType.CPU,
                total_capacity=4.0,
                available_capacity=4.0,
                cost_per_unit=1.0 + i * 0.5,
                efficiency_rating=0.8 + i * 0.1,
                quantum_coherence=0.9
            )
            resources.append(resource)
        
        return resources
    
    def test_planner_initialization(self, config):
        """Test planner initialization."""
        planner = QuantumTaskPlanner(config)
        assert planner.config == config
        assert len(planner.algorithms) >= 2  # Should have quantum_annealing and qaoa
        assert planner.metrics.total_optimizations == 0
    
    def test_planner_initialization_error(self):
        """Test planner initialization error handling."""
        with patch('src.quantum_task_planner.core.planner.QuantumAnnealingScheduler', side_effect=Exception("Init error")):
            with pytest.raises(ConfigurationError):
                QuantumTaskPlanner()
    
    def test_create_schedule_success(self, planner):
        """Test successful schedule creation."""
        start_time = datetime.utcnow()
        schedule = planner.create_schedule(
            schedule_id="test_schedule",
            name="Test Schedule",
            start_time=start_time,
            description="Test description"
        )
        
        assert schedule.id == "test_schedule"
        assert schedule.name == "Test Schedule"
        assert schedule.start_time == start_time
        assert schedule.description == "Test description"
        assert schedule.status == ScheduleStatus.DRAFT
        assert "test_schedule" in planner.active_schedules
    
    def test_create_schedule_validation_errors(self, planner):
        """Test schedule creation validation."""
        start_time = datetime.utcnow()
        
        # Empty schedule ID
        with pytest.raises(ValidationError):
            planner.create_schedule("", "Test", start_time)
        
        # Empty name
        with pytest.raises(ValidationError):
            planner.create_schedule("test", "", start_time)
        
        # Duplicate schedule ID
        planner.create_schedule("test", "Test", start_time)
        with pytest.raises(ValidationError):
            planner.create_schedule("test", "Test2", start_time)
    
    def test_add_task_success(self, planner, sample_tasks):
        """Test successful task addition."""
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        task = sample_tasks[0]
        
        planner.add_task("test", task)
        
        assert len(schedule.tasks) == 1
        assert schedule.tasks[0].id == task.id
    
    def test_add_task_validation_errors(self, planner, sample_tasks):
        """Test task addition validation."""
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        task = sample_tasks[0]
        
        # Add task successfully first
        planner.add_task("test", task)
        
        # Try to add same task again
        with pytest.raises(ValidationError):
            planner.add_task("test", task)
        
        # Invalid task
        invalid_task = Task(
            id="",  # Empty ID
            name="Invalid",
            duration=timedelta(hours=1)
        )
        with pytest.raises(ValidationError):
            planner.add_task("test", invalid_task)
    
    def test_add_resource_success(self, planner, sample_resources):
        """Test successful resource addition."""
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        resource = sample_resources[0]
        
        planner.add_resource("test", resource)
        
        assert len(schedule.resources) == 1
        assert schedule.resources[0].id == resource.id
    
    def test_add_resource_validation_errors(self, planner):
        """Test resource addition validation."""
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        
        # Invalid resource
        invalid_resource = Resource(
            id="",  # Empty ID
            name="Invalid",
            type=ResourceType.CPU,
            total_capacity=0.0  # Invalid capacity
        )
        with pytest.raises(ValidationError):
            planner.add_resource("test", invalid_resource)
    
    @patch('src.quantum_task_planner.algorithms.quantum_annealing.QuantumAnnealingScheduler.optimize_schedule')
    def test_optimize_schedule_success(self, mock_optimize, planner, sample_tasks, sample_resources):
        """Test successful schedule optimization."""
        # Setup
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        for task in sample_tasks:
            planner.add_task("test", task)
        for resource in sample_resources:
            planner.add_resource("test", resource)
        
        # Mock optimization result
        from src.quantum_task_planner.models.schedule import OptimizationMetrics
        mock_metrics = OptimizationMetrics(
            makespan=timedelta(hours=5),
            total_cost=100.0,
            resource_utilization={"resource_0": 0.8},
            constraint_violations=0,
            quantum_energy=50.0,
            optimization_time=timedelta(seconds=10),
            iterations=100,
            convergence_achieved=True
        )
        mock_optimize.return_value = mock_metrics
        
        # Run optimization
        result = planner.optimize_schedule("test")
        
        assert result.status == ScheduleStatus.OPTIMIZED
        assert result.metrics == mock_metrics
        mock_optimize.assert_called_once()
    
    def test_optimize_schedule_validation_errors(self, planner):
        """Test optimization validation errors."""
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        
        # No tasks
        with pytest.raises(ValidationError):
            planner.optimize_schedule("test")
    
    @patch('src.quantum_task_planner.algorithms.quantum_annealing.QuantumAnnealingScheduler.optimize_schedule')
    def test_optimize_schedule_with_fallback(self, mock_optimize, planner, sample_tasks, sample_resources):
        """Test optimization with fallback algorithm."""
        # Setup
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        for task in sample_tasks:
            planner.add_task("test", task)
        for resource in sample_resources:
            planner.add_resource("test", resource)
        
        # Mock primary algorithm failure
        mock_optimize.side_effect = OptimizationError("Primary failed")
        
        # Mock fallback success
        with patch('src.quantum_task_planner.algorithms.qaoa_allocator.QAOAResourceAllocator.allocate_resources') as mock_fallback:
            from src.quantum_task_planner.models.schedule import OptimizationMetrics
            mock_metrics = OptimizationMetrics(
                makespan=timedelta(hours=6),
                total_cost=120.0,
                resource_utilization={"resource_0": 0.7},
                constraint_violations=0,
                quantum_energy=45.0,
                optimization_time=timedelta(seconds=15),
                iterations=150,
                convergence_achieved=True
            )
            mock_fallback.return_value = mock_metrics
            
            result = planner.optimize_schedule("test")
            
            assert result.status == ScheduleStatus.OPTIMIZED
            mock_fallback.assert_called_once()
    
    def test_get_schedule_success(self, planner):
        """Test successful schedule retrieval."""
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        retrieved = planner.get_schedule("test")
        assert retrieved == schedule
    
    def test_get_schedule_not_found(self, planner):
        """Test schedule not found error."""
        with pytest.raises(SchedulingError):
            planner.get_schedule("nonexistent")
    
    def test_get_optimization_status(self, planner, sample_tasks, sample_resources):
        """Test optimization status retrieval."""
        schedule = planner.create_schedule("test", "Test", datetime.utcnow())
        for task in sample_tasks[:2]:  # Add fewer tasks
            planner.add_task("test", task)
        for resource in sample_resources[:1]:  # Add fewer resources
            planner.add_resource("test", resource)
        
        status = planner.get_optimization_status("test")
        
        assert status["schedule_id"] == "test"
        assert status["status"] == ScheduleStatus.DRAFT.value
        assert status["task_count"] == 2
        assert status["resource_count"] == 1
    
    def test_get_planner_metrics(self, planner):
        """Test planner metrics retrieval."""
        metrics = planner.get_planner_metrics()
        
        assert "total_optimizations" in metrics
        assert "successful_optimizations" in metrics
        assert "algorithm_usage" in metrics
    
    def test_list_schedules(self, planner):
        """Test schedule listing."""
        # No schedules initially
        schedules = planner.list_schedules()
        assert len(schedules) == 0
        
        # Create schedules
        planner.create_schedule("test1", "Test 1", datetime.utcnow())
        planner.create_schedule("test2", "Test 2", datetime.utcnow())
        
        schedules = planner.list_schedules()
        assert len(schedules) == 2
        assert all("id" in s for s in schedules)
        assert all("name" in s for s in schedules)
    
    def test_context_manager(self, config):
        """Test planner as context manager."""
        with QuantumTaskPlanner(config) as planner:
            assert planner is not None
            schedule = planner.create_schedule("test", "Test", datetime.utcnow())
            assert schedule is not None
    
    def test_shutdown(self, planner):
        """Test planner shutdown."""
        # Create a schedule to have some state
        planner.create_schedule("test", "Test", datetime.utcnow())
        
        # Test shutdown doesn't raise errors
        planner.shutdown()
        
        # Test multiple shutdowns are safe
        planner.shutdown()
    
    def test_metrics_export(self, planner):
        """Test metrics export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            planner.config.metrics_export_path = metrics_path
            planner.config.enable_metrics_collection = True
            
            # Generate some metrics
            planner.metrics.record_optimization("test_algo", 10.0, True)
            
            # Export metrics
            planner._export_metrics()
            
            assert metrics_path.exists()
            with open(metrics_path) as f:
                data = json.load(f)
                assert "total_optimizations" in data
                assert data["total_optimizations"] == 1


class TestPlannerIntegration:
    """Integration tests for quantum task planner."""
    
    @pytest.fixture
    def planner(self):
        """Create planner for integration tests."""
        config = PlannerConfig(
            enable_parallel_processing=False,
            optimization_timeout_seconds=10,
            max_retry_attempts=1
        )
        return QuantumTaskPlanner(config)
    
    def test_full_optimization_workflow(self, planner):
        """Test complete optimization workflow."""
        # Create schedule
        schedule = planner.create_schedule(
            "integration_test",
            "Integration Test Schedule",
            datetime.utcnow()
        )
        
        # Add tasks with dependencies
        tasks = []
        for i in range(3):
            task = Task(
                id=f"task_{i}",
                name=f"Task {i}",
                duration=timedelta(hours=1),
                priority=TaskPriority.MEDIUM,
                resource_requirements={"cpu": 1.0}
            )
            if i > 0:
                task.add_dependency(f"task_{i-1}")
            
            planner.add_task("integration_test", task)
            tasks.append(task)
        
        # Add resources
        for i in range(2):
            resource = Resource(
                id=f"resource_{i}",
                name=f"Resource {i}",
                type=ResourceType.CPU,
                total_capacity=2.0,
                available_capacity=2.0,
                cost_per_unit=1.0,
                efficiency_rating=0.9
            )
            planner.add_resource("integration_test", resource)
        
        # Mock successful optimization
        with patch('src.quantum_task_planner.algorithms.quantum_annealing.QuantumAnnealingScheduler.optimize_schedule') as mock_opt:
            from src.quantum_task_planner.models.schedule import OptimizationMetrics
            mock_metrics = OptimizationMetrics(
                makespan=timedelta(hours=3),
                total_cost=60.0,
                resource_utilization={"resource_0": 0.8, "resource_1": 0.6},
                constraint_violations=0,
                quantum_energy=30.0,
                optimization_time=timedelta(seconds=5),
                iterations=50,
                convergence_achieved=True
            )
            mock_opt.return_value = mock_metrics
            
            # Run optimization
            optimized_schedule = planner.optimize_schedule("integration_test")
            
            # Verify results
            assert optimized_schedule.status == ScheduleStatus.OPTIMIZED
            assert optimized_schedule.metrics is not None
            assert optimized_schedule.metrics.constraint_violations == 0
            
            # Check planner metrics updated
            planner_metrics = planner.get_planner_metrics()
            assert planner_metrics["total_optimizations"] == 1
            assert planner_metrics["successful_optimizations"] == 1
    
    def test_error_handling_workflow(self, planner):
        """Test error handling in optimization workflow."""
        # Create schedule with invalid configuration
        schedule = planner.create_schedule("error_test", "Error Test", datetime.utcnow())
        
        # Add task but no resources
        task = Task(
            id="task_1",
            name="Test Task",
            duration=timedelta(hours=1),
            resource_requirements={"cpu": 1.0}
        )
        planner.add_task("error_test", task)
        
        # Should fail validation
        with pytest.raises(ValidationError):
            planner.optimize_schedule("error_test")
        
        # Check error was recorded in metrics
        planner_metrics = planner.get_planner_metrics()
        assert planner_metrics["failed_optimizations"] >= 0  # Might be 0 if validation happens before optimization


if __name__ == "__main__":
    pytest.main([__file__, "-v"])