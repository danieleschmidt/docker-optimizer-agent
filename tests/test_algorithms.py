"""Tests for quantum algorithms."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.quantum_task_planner.algorithms.quantum_annealing import QuantumAnnealingScheduler
from src.quantum_task_planner.algorithms.qaoa_allocator import QAOAResourceAllocator, QAOAParameters
from src.quantum_task_planner.models.task import Task, TaskStatus, TaskPriority
from src.quantum_task_planner.models.resource import Resource, ResourceType, ResourceStatus
from src.quantum_task_planner.models.schedule import Schedule, ScheduleStatus
from src.quantum_task_planner.core.exceptions import OptimizationError, ValidationError


class TestQuantumAnnealingScheduler:
    """Test quantum annealing scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create scheduler instance."""
        return QuantumAnnealingScheduler(
            initial_temperature=5.0,
            final_temperature=0.1,
            max_iterations=50,  # Reduced for testing
            cooling_rate=0.9
        )
    
    @pytest.fixture
    def sample_schedule(self):
        """Create sample schedule for testing."""
        schedule = Schedule(
            id="test",
            name="Test Schedule",
            start_time=datetime.utcnow()
        )
        
        # Add tasks
        tasks = [
            Task(id="task1", name="Task 1", duration=timedelta(hours=1), priority=TaskPriority.HIGH),
            Task(id="task2", name="Task 2", duration=timedelta(hours=2), priority=TaskPriority.MEDIUM),
            Task(id="task3", name="Task 3", duration=timedelta(hours=1), priority=TaskPriority.LOW)
        ]
        
        # Add dependencies
        tasks[1].add_dependency("task1")
        
        for task in tasks:
            schedule.add_task(task)
        
        # Add resources
        resources = [
            Resource(id="res1", name="Resource 1", type=ResourceType.CPU, 
                    total_capacity=4.0, available_capacity=4.0, cost_per_unit=1.0),
            Resource(id="res2", name="Resource 2", type=ResourceType.CPU,
                    total_capacity=4.0, available_capacity=4.0, cost_per_unit=1.5)
        ]
        
        for resource in resources:
            schedule.add_resource(resource)
        
        return schedule
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = QuantumAnnealingScheduler()
        
        assert scheduler.initial_temperature == 10.0
        assert scheduler.final_temperature == 0.01
        assert scheduler.max_iterations == 1000
        assert scheduler.cooling_rate == 0.95
    
    def test_optimize_schedule_success(self, scheduler, sample_schedule):
        """Test successful schedule optimization."""
        # Mock some internal methods to avoid complex calculations
        with patch.object(scheduler, '_generate_initial_state') as mock_initial:
            with patch.object(scheduler, '_apply_solution_to_schedule') as mock_apply:
                with patch.object(scheduler, '_generate_neighbor_state') as mock_neighbor:
                    # Setup mock state
                    mock_state = Mock()
                    mock_state.energy = 100.0
                    mock_state.temperature = 1000.0  # Add temperature attribute
                    mock_state.assignments = {"task1": "res1", "task2": "res2", "task3": "res1"}
                    mock_initial.return_value = mock_state
                    
                    # Setup neighbor state mock
                    mock_neighbor_state = Mock()
                    mock_neighbor_state.energy = 90.0
                    mock_neighbor_state.temperature = 950.0  # Add temperature attribute
                    mock_neighbor_state.assignments = {"task1": "res1", "task2": "res1", "task3": "res2"}
                    mock_neighbor.return_value = mock_neighbor_state
                    
                    scheduler.best_state = mock_state
                    
                    # Run optimization
                    metrics = scheduler.optimize_schedule(sample_schedule)
                    
                    # Verify results
                    assert metrics is not None
                    assert hasattr(metrics, 'makespan')
                    assert hasattr(metrics, 'total_cost')
                    assert sample_schedule.status == ScheduleStatus.OPTIMIZED
                    
                    mock_initial.assert_called_once()
                    mock_apply.assert_called_once()
    
    def test_generate_initial_state(self, scheduler, sample_schedule):
        """Test initial state generation."""
        state = scheduler._generate_initial_state(sample_schedule)
        
        assert hasattr(state, 'assignments')
        assert hasattr(state, 'start_times')
        assert hasattr(state, 'energy')
        assert len(state.assignments) == len(sample_schedule.tasks)
    
    def test_calculate_temperature(self, scheduler):
        """Test temperature calculation."""
        temp_0 = scheduler._calculate_temperature(0)
        temp_10 = scheduler._calculate_temperature(10)
        temp_100 = scheduler._calculate_temperature(100)
        
        # Temperature should decrease over iterations
        assert temp_0 > temp_10 > temp_100
        assert temp_100 >= scheduler.final_temperature
    
    def test_topological_sort_tasks(self, scheduler, sample_schedule):
        """Test topological sorting of tasks."""
        sorted_tasks = scheduler._topological_sort_tasks(sample_schedule.tasks)
        
        # Should return all tasks
        assert len(sorted_tasks) == len(sample_schedule.tasks)
        
        # Task1 should come before task2 (dependency)
        task1_index = next(i for i, t in enumerate(sorted_tasks) if t.id == "task1")
        task2_index = next(i for i, t in enumerate(sorted_tasks) if t.id == "task2")
        assert task1_index < task2_index


class TestQAOAResourceAllocator:
    """Test QAOA resource allocator."""
    
    @pytest.fixture
    def allocator(self):
        """Create allocator instance."""
        params = QAOAParameters(
            layers=1,  # Reduced for testing
            max_iterations=20,  # Reduced for testing
            timeout_seconds=30
        )
        return QAOAResourceAllocator(params)
    
    @pytest.fixture
    def sample_schedule(self):
        """Create sample schedule for testing."""
        schedule = Schedule(
            id="test",
            name="Test Schedule", 
            start_time=datetime.utcnow()
        )
        
        # Add tasks
        tasks = [
            Task(id="task1", name="Task 1", duration=timedelta(hours=1),
                 resource_requirements={"cpu": 2.0}),
            Task(id="task2", name="Task 2", duration=timedelta(hours=1),
                 resource_requirements={"cpu": 1.0})
        ]
        
        for task in tasks:
            schedule.add_task(task)
        
        # Add resources
        resources = [
            Resource(id="res1", name="Resource 1", type=ResourceType.CPU,
                    total_capacity=4.0, available_capacity=4.0, cost_per_unit=1.0),
            Resource(id="res2", name="Resource 2", type=ResourceType.CPU,
                    total_capacity=2.0, available_capacity=2.0, cost_per_unit=2.0)
        ]
        
        for resource in resources:
            schedule.add_resource(resource)
        
        return schedule
    
    def test_qaoa_parameters_validation(self):
        """Test QAOA parameters validation."""
        # Valid parameters
        params = QAOAParameters(layers=2, max_iterations=100)
        assert params.layers == 2
        
        # Invalid parameters
        with pytest.raises(ValueError):
            QAOAParameters(layers=0)
        
        with pytest.raises(ValueError):
            QAOAParameters(max_iterations=0)
    
    def test_allocator_initialization(self):
        """Test allocator initialization."""
        params = QAOAParameters()
        allocator = QAOAResourceAllocator(params)
        
        assert allocator.params == params
        assert hasattr(allocator, 'best_state')
        assert hasattr(allocator, 'optimization_history')
    
    def test_allocate_resources_success(self, allocator, sample_schedule):
        """Test successful resource allocation."""
        # Mock some complex methods
        with patch.object(allocator, '_run_optimization') as mock_run:
            mock_run.return_value = sample_schedule
            
            metrics = allocator.allocate_resources(sample_schedule)
            
            assert metrics is not None
            assert hasattr(metrics, 'total_cost')
            assert hasattr(metrics, 'optimization_time')
            mock_run.assert_called_once()
    
    def test_validate_schedule(self, allocator, sample_schedule):
        """Test schedule validation."""
        # Valid schedule should not raise
        allocator._validate_schedule(sample_schedule)
        
        # Empty schedule should raise ValidationError
        empty_schedule = Schedule(id="empty", name="Empty", start_time=datetime.utcnow())
        with pytest.raises(ValidationError):
            allocator._validate_schedule(empty_schedule)
    
    def test_generate_initial_state(self, allocator, sample_schedule):
        """Test initial state generation."""
        state = allocator._generate_initial_state(sample_schedule)
        
        assert hasattr(state, 'resource_assignments')
        assert hasattr(state, 'allocation_amounts')
        assert hasattr(state, 'cost_expectation')
        assert len(state.resource_assignments) == len(sample_schedule.tasks)
    
    def test_calculate_cost_expectation(self, allocator, sample_schedule):
        """Test cost expectation calculation."""
        assignments = {"task1": "res1", "task2": "res2"}
        allocations = {"task1": 2.0, "task2": 1.0}
        
        cost = allocator._calculate_cost_expectation(assignments, allocations, sample_schedule)
        
        assert isinstance(cost, float)
        assert cost >= 0
    
    def test_calculate_allocation_amount(self, allocator, sample_schedule):
        """Test allocation amount calculation."""
        task = sample_schedule.get_task("task1")
        resource = sample_schedule.get_resource("res1")
        
        amount = allocator._calculate_allocation_amount(task, resource)
        
        assert isinstance(amount, float)
        assert amount > 0
        assert amount <= resource.available_capacity


class TestVQEDependencyResolver:
    """Test VQE dependency resolver."""
    
    @pytest.fixture
    def resolver(self):
        """Create resolver instance."""
        # Import here to avoid circular imports during test collection
        from src.quantum_task_planner.algorithms.vqe_dependencies import VQEDependencyResolver
        return VQEDependencyResolver(
            max_iterations=20,  # Reduced for testing
            cluster_size_limit=10,
            parallel_clusters=False  # Disable for simpler testing
        )
    
    @pytest.fixture
    def sample_schedule_with_deps(self):
        """Create schedule with complex dependencies."""
        schedule = Schedule(
            id="test",
            name="Test Schedule",
            start_time=datetime.utcnow()
        )
        
        # Add tasks with dependency chain
        tasks = []
        for i in range(5):
            task = Task(
                id=f"task_{i}",
                name=f"Task {i}",
                duration=timedelta(hours=1),
                priority=TaskPriority.MEDIUM,
                quantum_weight=1.0 + i * 0.2
            )
            
            # Create dependency chain: task_i depends on task_{i-1}
            if i > 0:
                task.add_dependency(f"task_{i-1}")
            
            # Add some parallel branches
            if i > 2:
                task.add_dependency(f"task_{i-2}")
            
            tasks.append(task)
            schedule.add_task(task)
        
        # Add resources
        for i in range(2):
            resource = Resource(
                id=f"res_{i}",
                name=f"Resource {i}",
                type=ResourceType.CPU,
                total_capacity=4.0,
                available_capacity=4.0,
                cost_per_unit=1.0
            )
            schedule.add_resource(resource)
        
        return schedule
    
    def test_resolver_initialization(self):
        """Test VQE resolver initialization."""
        from src.quantum_task_planner.algorithms.vqe_dependencies import VQEDependencyResolver
        resolver = VQEDependencyResolver()
        
        assert resolver.max_iterations == 500
        assert resolver.cluster_size_limit == 20
        assert hasattr(resolver, 'energy_cache')
        assert hasattr(resolver, 'optimization_metrics')
    
    def test_validate_dependencies_success(self, resolver, sample_schedule_with_deps):
        """Test successful dependency validation."""
        # Should not raise for valid dependencies
        resolver._validate_dependencies(sample_schedule_with_deps)
    
    def test_validate_dependencies_invalid(self, resolver):
        """Test dependency validation with invalid dependencies."""
        schedule = Schedule(id="test", name="Test", start_time=datetime.utcnow())
        
        # Task with invalid dependency
        task = Task(
            id="task1",
            name="Task 1", 
            duration=timedelta(hours=1),
            dependencies={"nonexistent_task"}
        )
        schedule.add_task(task)
        
        from src.quantum_task_planner.core.exceptions import DependencyError
        with pytest.raises(DependencyError):
            resolver._validate_dependencies(schedule)
    
    def test_build_dependency_graph(self, resolver, sample_schedule_with_deps):
        """Test dependency graph building."""
        graph = resolver._build_dependency_graph(sample_schedule_with_deps.tasks)
        
        assert 'adjacency' in graph
        assert 'reverse_adjacency' in graph
        assert 'in_degree' in graph
        assert 'out_degree' in graph
        assert 'task_weights' in graph
        
        # Check that dependencies are correctly represented
        assert len(graph['task_weights']) == len(sample_schedule_with_deps.tasks)
    
    def test_create_dependency_clusters(self, resolver, sample_schedule_with_deps):
        """Test dependency clustering."""
        graph = resolver._build_dependency_graph(sample_schedule_with_deps.tasks)
        clusters = resolver._create_dependency_clusters(sample_schedule_with_deps.tasks, graph)
        
        assert len(clusters) >= 1
        assert all(len(cluster.tasks) > 0 for cluster in clusters)
        assert all(hasattr(cluster, 'internal_dependencies') for cluster in clusters)
    
    def test_calculate_task_weight(self, resolver):
        """Test task weight calculation."""
        task = Task(
            id="test",
            name="Test",
            duration=timedelta(hours=2),
            priority=TaskPriority.HIGH,
            quantum_weight=2.0,
            entanglement_factor=0.5
        )
        
        weight = resolver._calculate_task_weight(task)
        
        assert isinstance(weight, float)
        assert weight > 0
    
    def test_is_already_ordered(self, resolver):
        """Test checking if tasks are already ordered."""
        # Create tasks in correct dependency order
        task1 = Task(id="task1", name="Task 1", duration=timedelta(hours=1))
        task2 = Task(id="task2", name="Task 2", duration=timedelta(hours=1), dependencies={"task1"})
        task3 = Task(id="task3", name="Task 3", duration=timedelta(hours=1), dependencies={"task2"})
        
        ordered_tasks = [task1, task2, task3]
        assert resolver._is_already_ordered(ordered_tasks)
        
        # Create tasks in wrong order
        wrong_order_tasks = [task2, task1, task3]  # task2 before task1
        assert not resolver._is_already_ordered(wrong_order_tasks)


class TestAlgorithmIntegration:
    """Integration tests for quantum algorithms."""
    
    def test_all_algorithms_available(self):
        """Test that all algorithms can be imported and instantiated."""
        # Test quantum annealing
        scheduler = QuantumAnnealingScheduler()
        assert scheduler is not None
        
        # Test QAOA
        params = QAOAParameters()
        allocator = QAOAResourceAllocator(params)
        assert allocator is not None
        
        # Test VQE (with error handling for optional dependencies)
        try:
            from src.quantum_task_planner.algorithms.vqe_dependencies import VQEDependencyResolver
            resolver = VQEDependencyResolver()
            assert resolver is not None
        except ImportError:
            pytest.skip("VQE dependencies not available")
    
    def test_algorithm_error_handling(self):
        """Test algorithm error handling."""
        scheduler = QuantumAnnealingScheduler()
        
        # Empty schedule should fail
        empty_schedule = Schedule(id="empty", name="Empty", start_time=datetime.utcnow())
        
        # Should handle gracefully without crashing
        try:
            scheduler.optimize_schedule(empty_schedule)
        except Exception as e:
            # Should be a specific optimization error, not a generic crash
            assert "optimization" in str(e).lower() or "schedule" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])