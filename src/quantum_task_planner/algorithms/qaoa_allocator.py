"""QAOA-inspired resource allocation algorithm with robust error handling."""

import math
import random
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from ..models.task import Task
from ..models.resource import Resource, ResourceStatus
from ..models.schedule import Schedule, TaskAssignment, OptimizationMetrics, ScheduleStatus
from ..core.exceptions import OptimizationError, ResourceAllocationError, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class QAOAState:
    """QAOA optimization state with error tracking."""
    resource_assignments: Dict[str, str]  # task_id -> resource_id
    allocation_amounts: Dict[str, float]  # task_id -> allocated amount
    cost_expectation: float
    layer: int
    parameters: List[float]
    validation_errors: List[str]
    is_valid: bool = True


@dataclass
class QAOAParameters:
    """QAOA algorithm parameters with validation."""
    layers: int = 2
    max_iterations: int = 500
    convergence_threshold: float = 0.001
    cost_weight: float = 0.4
    utilization_weight: float = 0.3
    fairness_weight: float = 0.3
    parallel_workers: int = 4
    timeout_seconds: int = 300
    
    def __post_init__(self):
        """Validate parameters."""
        if self.layers < 1:
            raise ValueError("QAOA layers must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("Max iterations must be >= 1")
        if not 0 < self.convergence_threshold < 1:
            raise ValueError("Convergence threshold must be between 0 and 1")
        if abs(self.cost_weight + self.utilization_weight + self.fairness_weight - 1.0) > 0.001:
            raise ValueError("Objective weights must sum to 1.0")


class QAOAResourceAllocator:
    """QAOA-inspired resource allocator with comprehensive error handling."""
    
    def __init__(self, parameters: Optional[QAOAParameters] = None):
        """Initialize QAOA allocator with robust configuration.
        
        Args:
            parameters: QAOA algorithm parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        try:
            self.params = parameters or QAOAParameters()
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            
            # Optimization tracking
            self.best_state: Optional[QAOAState] = None
            self.optimization_history: List[Dict[str, Any]] = []
            self.error_count: int = 0
            self.warnings: List[str] = []
            
            # Performance monitoring
            self.start_time: Optional[datetime] = None
            self.iteration_times: List[float] = []
            
            self.logger.info(f"QAOA allocator initialized with {self.params.layers} layers")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize QAOA allocator: {e}")
            raise OptimizationError(f"Initialization failed: {e}") from e
    
    def allocate_resources(self, schedule: Schedule) -> OptimizationMetrics:
        """Allocate resources using QAOA-inspired optimization.
        
        Args:
            schedule: Schedule to optimize
            
        Returns:
            Optimization metrics
            
        Raises:
            OptimizationError: If optimization fails
            ValidationError: If schedule is invalid
        """
        try:
            self.start_time = datetime.utcnow()
            self.logger.info(f"Starting QAOA resource allocation for schedule {schedule.id}")
            
            # Validate input
            self._validate_schedule(schedule)
            
            # Set schedule status
            schedule.status = ScheduleStatus.OPTIMIZING
            
            # Generate initial state
            initial_state = self._generate_initial_state(schedule)
            current_state = initial_state
            
            best_cost = float('inf')
            no_improvement_count = 0
            max_no_improvement = 50
            
            # QAOA optimization loop
            for iteration in range(self.params.max_iterations):
                iteration_start = time.time()
                
                try:
                    # Apply QAOA layers
                    new_state = self._apply_qaoa_layers(current_state, schedule)
                    
                    # Validate state
                    if self._validate_state(new_state, schedule):
                        current_state = new_state
                        
                        # Track best solution
                        if new_state.cost_expectation < best_cost:
                            best_cost = new_state.cost_expectation
                            self.best_state = new_state
                            no_improvement_count = 0
                            self.logger.debug(f"New best cost: {best_cost:.4f} at iteration {iteration}")
                        else:
                            no_improvement_count += 1
                    else:
                        self.logger.warning(f"Invalid state generated at iteration {iteration}")
                        self.error_count += 1
                        
                        # Generate fallback state
                        current_state = self._generate_fallback_state(current_state, schedule)
                
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration}: {e}")
                    self.error_count += 1
                    
                    # Attempt recovery
                    if self.error_count > 10:
                        raise OptimizationError(f"Too many errors during optimization: {e}")
                    
                    current_state = self._generate_fallback_state(current_state, schedule)
                
                # Track iteration performance
                iteration_time = time.time() - iteration_start
                self.iteration_times.append(iteration_time)
                
                # Record optimization history
                self.optimization_history.append({
                    'iteration': iteration,
                    'cost': current_state.cost_expectation,
                    'valid': current_state.is_valid,
                    'errors': len(current_state.validation_errors),
                    'time': iteration_time
                })
                
                # Check convergence
                if self._check_convergence(iteration) or no_improvement_count >= max_no_improvement:
                    self.logger.info(f"Convergence achieved at iteration {iteration}")
                    break
                
                # Check timeout
                if self._check_timeout():
                    self.logger.warning("Optimization timeout reached")
                    break
            
            # Apply best solution
            if self.best_state:
                self._apply_solution_to_schedule(schedule, self.best_state)
            else:
                raise OptimizationError("No valid solution found")
            
            # Calculate metrics
            optimization_time = datetime.utcnow() - self.start_time
            metrics = self._calculate_metrics(schedule, iteration + 1, optimization_time)
            
            schedule.status = ScheduleStatus.OPTIMIZED
            schedule.optimized_at = datetime.utcnow()
            schedule.metrics = metrics
            
            self.logger.info(f"QAOA optimization completed successfully in {optimization_time}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"QAOA optimization failed: {e}")
            schedule.status = ScheduleStatus.FAILED
            raise OptimizationError(f"Resource allocation failed: {e}") from e
    
    def _validate_schedule(self, schedule: Schedule) -> None:
        """Validate schedule for optimization.
        
        Args:
            schedule: Schedule to validate
            
        Raises:
            ValidationError: If schedule is invalid
        """
        errors = []
        
        if not schedule.tasks:
            errors.append("Schedule has no tasks")
        
        if not schedule.resources:
            errors.append("Schedule has no resources")
        
        # Check resource availability
        available_resources = [r for r in schedule.resources 
                             if r.status == ResourceStatus.AVAILABLE]
        if not available_resources:
            errors.append("No available resources")
        
        # Validate task dependencies
        task_ids = {task.id for task in schedule.tasks}
        for task in schedule.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    errors.append(f"Task {task.id} has invalid dependency {dep_id}")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(schedule.tasks):
            errors.append("Circular dependencies detected")
        
        # Validate resource requirements
        for task in schedule.tasks:
            if not task.resource_requirements:
                self.warnings.append(f"Task {task.id} has no resource requirements")
            
            for resource_type, amount in task.resource_requirements.items():
                if amount <= 0:
                    errors.append(f"Task {task.id} has invalid resource requirement: {resource_type}={amount}")
        
        if errors:
            error_msg = "; ".join(errors)
            self.logger.error(f"Schedule validation failed: {error_msg}")
            raise ValidationError(f"Invalid schedule: {error_msg}")
        
        if self.warnings:
            for warning in self.warnings:
                self.logger.warning(warning)
    
    def _generate_initial_state(self, schedule: Schedule) -> QAOAState:
        """Generate initial QAOA state using greedy heuristic.
        
        Args:
            schedule: Schedule to optimize
            
        Returns:
            Initial QAOA state
        """
        try:
            resource_assignments = {}
            allocation_amounts = {}
            
            # Sort tasks by priority and resource requirements
            sorted_tasks = sorted(schedule.tasks, 
                                key=lambda t: (t.priority.value, sum(t.resource_requirements.values())),
                                reverse=True)
            
            # Greedy assignment with error handling
            for task in sorted_tasks:
                best_resource = None
                best_allocation = 0.0
                best_score = float('inf')
                
                available_resources = [r for r in schedule.resources 
                                     if r.status == ResourceStatus.AVAILABLE]
                
                for resource in available_resources:
                    try:
                        # Calculate allocation amount
                        allocation = self._calculate_allocation_amount(task, resource)
                        
                        if allocation > 0 and resource.available_capacity >= allocation:
                            # Calculate assignment score
                            score = self._calculate_assignment_score(task, resource, allocation)
                            
                            if score < best_score:
                                best_score = score
                                best_resource = resource
                                best_allocation = allocation
                    
                    except Exception as e:
                        self.logger.warning(f"Error calculating assignment for task {task.id} to resource {resource.id}: {e}")
                        continue
                
                # Assign to best resource or fallback
                if best_resource:
                    resource_assignments[task.id] = best_resource.id
                    allocation_amounts[task.id] = best_allocation
                    best_resource.available_capacity -= best_allocation
                else:
                    # Fallback assignment
                    fallback_resource = available_resources[0] if available_resources else schedule.resources[0]
                    resource_assignments[task.id] = fallback_resource.id
                    allocation_amounts[task.id] = min(1.0, fallback_resource.available_capacity)
                    
                    self.warnings.append(f"Fallback assignment for task {task.id}")
            
            # Calculate initial cost
            cost = self._calculate_cost_expectation(resource_assignments, allocation_amounts, schedule)
            
            state = QAOAState(
                resource_assignments=resource_assignments,
                allocation_amounts=allocation_amounts,
                cost_expectation=cost,
                layer=0,
                parameters=self._initialize_parameters(),
                validation_errors=[],
                is_valid=True
            )
            
            self.logger.debug(f"Generated initial state with cost {cost:.4f}")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to generate initial state: {e}")
            raise OptimizationError(f"Initial state generation failed: {e}") from e
    
    def _apply_qaoa_layers(self, state: QAOAState, schedule: Schedule) -> QAOAState:
        """Apply QAOA layers to current state.
        
        Args:
            state: Current QAOA state
            schedule: Schedule being optimized
            
        Returns:
            New state after QAOA layers
        """
        try:
            new_assignments = state.resource_assignments.copy()
            new_allocations = state.allocation_amounts.copy()
            
            # Apply parameterized layers
            for layer in range(self.params.layers):
                # Cost layer: optimize based on current cost function
                new_assignments, new_allocations = self._apply_cost_layer(
                    new_assignments, new_allocations, schedule, 
                    state.parameters[layer * 2]
                )
                
                # Mixer layer: explore neighboring solutions
                new_assignments, new_allocations = self._apply_mixer_layer(
                    new_assignments, new_allocations, schedule,
                    state.parameters[layer * 2 + 1]
                )
            
            # Calculate new cost
            new_cost = self._calculate_cost_expectation(new_assignments, new_allocations, schedule)
            
            # Update parameters using gradient-free optimization
            new_parameters = self._update_parameters(state.parameters, new_cost, state.cost_expectation)
            
            new_state = QAOAState(
                resource_assignments=new_assignments,
                allocation_amounts=new_allocations,
                cost_expectation=new_cost,
                layer=state.layer + 1,
                parameters=new_parameters,
                validation_errors=[],
                is_valid=True
            )
            
            return new_state
            
        except Exception as e:
            self.logger.error(f"Error applying QAOA layers: {e}")
            # Return original state with error flag
            error_state = QAOAState(
                resource_assignments=state.resource_assignments,
                allocation_amounts=state.allocation_amounts,
                cost_expectation=state.cost_expectation,
                layer=state.layer,
                parameters=state.parameters,
                validation_errors=[str(e)],
                is_valid=False
            )
            return error_state
    
    def _apply_cost_layer(self, assignments: Dict[str, str], allocations: Dict[str, float],
                          schedule: Schedule, gamma: float) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Apply QAOA cost layer with error handling."""
        try:
            new_assignments = assignments.copy()
            new_allocations = allocations.copy()
            
            # Optimize assignments based on cost gradient
            for task_id in assignments.keys():
                task = schedule.get_task(task_id)
                if not task:
                    continue
                
                current_resource_id = assignments[task_id]
                current_resource = schedule.get_resource(current_resource_id)
                
                if not current_resource:
                    continue
                
                # Evaluate alternative resources
                best_alternative = None
                best_cost = float('inf')
                
                for resource in schedule.resources:
                    if resource.id == current_resource_id or resource.status != ResourceStatus.AVAILABLE:
                        continue
                    
                    try:
                        # Calculate potential allocation
                        potential_allocation = self._calculate_allocation_amount(task, resource)
                        
                        if potential_allocation > 0 and resource.available_capacity >= potential_allocation:
                            # Calculate cost change
                            temp_assignments = new_assignments.copy()
                            temp_allocations = new_allocations.copy()
                            temp_assignments[task_id] = resource.id
                            temp_allocations[task_id] = potential_allocation
                            
                            cost = self._calculate_cost_expectation(temp_assignments, temp_allocations, schedule)
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_alternative = (resource.id, potential_allocation)
                    
                    except Exception as e:
                        self.logger.debug(f"Error evaluating resource {resource.id} for task {task_id}: {e}")
                        continue
                
                # Apply change based on gamma parameter
                if best_alternative and random.random() < abs(math.sin(gamma)):
                    new_resource_id, new_allocation = best_alternative
                    new_assignments[task_id] = new_resource_id
                    new_allocations[task_id] = new_allocation
            
            return new_assignments, new_allocations
            
        except Exception as e:
            self.logger.warning(f"Error in cost layer: {e}")
            return assignments, allocations
    
    def _apply_mixer_layer(self, assignments: Dict[str, str], allocations: Dict[str, float],
                           schedule: Schedule, beta: float) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Apply QAOA mixer layer for exploration."""
        try:
            new_assignments = assignments.copy()
            new_allocations = allocations.copy()
            
            # Random swaps and modifications based on beta
            num_swaps = max(1, int(len(assignments) * abs(math.sin(beta)) * 0.1))
            
            for _ in range(num_swaps):
                try:
                    # Select random task
                    task_ids = list(assignments.keys())
                    if not task_ids:
                        break
                    
                    task_id = random.choice(task_ids)
                    task = schedule.get_task(task_id)
                    
                    if not task:
                        continue
                    
                    # Select random alternative resource
                    available_resources = [r for r in schedule.resources 
                                         if r.status == ResourceStatus.AVAILABLE 
                                         and r.id != assignments[task_id]]
                    
                    if available_resources:
                        new_resource = random.choice(available_resources)
                        new_allocation = self._calculate_allocation_amount(task, new_resource)
                        
                        if new_allocation > 0 and new_resource.available_capacity >= new_allocation:
                            new_assignments[task_id] = new_resource.id
                            new_allocations[task_id] = new_allocation
                
                except Exception as e:
                    self.logger.debug(f"Error in mixer operation: {e}")
                    continue
            
            return new_assignments, new_allocations
            
        except Exception as e:
            self.logger.warning(f"Error in mixer layer: {e}")
            return assignments, allocations
    
    def _calculate_cost_expectation(self, assignments: Dict[str, str], 
                                   allocations: Dict[str, float],
                                   schedule: Schedule) -> float:
        """Calculate QAOA cost expectation with robust error handling."""
        try:
            total_cost = 0.0
            
            # Resource usage cost
            resource_costs = {}
            for task_id, resource_id in assignments.items():
                task = schedule.get_task(task_id)
                resource = schedule.get_resource(resource_id)
                
                if task and resource:
                    allocation = allocations.get(task_id, 1.0)
                    duration_hours = task.duration.total_seconds() / 3600
                    cost = duration_hours * resource.cost_per_unit * allocation
                    
                    if resource_id not in resource_costs:
                        resource_costs[resource_id] = 0.0
                    resource_costs[resource_id] += cost
            
            weighted_cost = sum(resource_costs.values()) * self.params.cost_weight
            total_cost += weighted_cost
            
            # Resource utilization balance
            utilization_variance = 0.0
            if schedule.resources:
                utilizations = []
                for resource in schedule.resources:
                    resource_load = resource_costs.get(resource.id, 0.0)
                    max_capacity_cost = resource.total_capacity * resource.cost_per_unit
                    utilization = resource_load / max(max_capacity_cost, 0.001)
                    utilizations.append(utilization)
                
                if utilizations:
                    mean_util = sum(utilizations) / len(utilizations)
                    utilization_variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
            
            total_cost += utilization_variance * self.params.utilization_weight * 100
            
            # Fairness penalty (ensure all tasks get resources)
            unallocated_penalty = 0.0
            for task in schedule.tasks:
                if task.id not in assignments:
                    unallocated_penalty += 1000  # Heavy penalty
                else:
                    allocation = allocations.get(task.id, 0.0)
                    if allocation <= 0:
                        unallocated_penalty += 500
            
            total_cost += unallocated_penalty * self.params.fairness_weight
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating cost expectation: {e}")
            return float('inf')  # Return high cost on error
    
    def _calculate_allocation_amount(self, task: Task, resource: Resource) -> float:
        """Calculate appropriate allocation amount for task on resource."""
        try:
            # Start with resource requirements
            if task.resource_requirements:
                # Get requirement for resource type
                resource_type = resource.type.value
                required_amount = task.resource_requirements.get(resource_type, 1.0)
                
                # Adjust based on resource efficiency
                adjusted_amount = required_amount / max(resource.efficiency_rating, 0.1)
                
                # Ensure within resource capacity
                return min(adjusted_amount, resource.available_capacity)
            else:
                # Default allocation
                return min(1.0, resource.available_capacity)
                
        except Exception as e:
            self.logger.debug(f"Error calculating allocation amount: {e}")
            return 1.0  # Safe default
    
    def _calculate_assignment_score(self, task: Task, resource: Resource, allocation: float) -> float:
        """Calculate assignment score for task-resource pair."""
        try:
            score = 0.0
            
            # Cost component
            duration_hours = task.duration.total_seconds() / 3600
            cost = duration_hours * resource.cost_per_unit * allocation
            score += cost
            
            # Efficiency component
            efficiency_bonus = (1.0 - resource.efficiency_rating) * 10
            score += efficiency_bonus
            
            # Quantum affinity
            quantum_affinity = resource.calculate_quantum_affinity(
                task.quantum_weight, task.entanglement_factor
            )
            score -= quantum_affinity * 5  # Lower score is better
            
            return score
            
        except Exception as e:
            self.logger.debug(f"Error calculating assignment score: {e}")
            return 1000.0  # High score on error
    
    def _initialize_parameters(self) -> List[float]:
        """Initialize QAOA parameters."""
        # Initialize with small random values
        return [random.uniform(-0.5, 0.5) for _ in range(self.params.layers * 2)]
    
    def _update_parameters(self, old_params: List[float], new_cost: float, old_cost: float) -> List[float]:
        """Update QAOA parameters using simple gradient-free method."""
        try:
            new_params = old_params.copy()
            
            # Simple parameter update based on cost improvement
            if new_cost < old_cost:
                # Small improvement in the same direction
                for i in range(len(new_params)):
                    new_params[i] += random.uniform(-0.1, 0.1)
            else:
                # Larger exploration step
                for i in range(len(new_params)):
                    new_params[i] += random.uniform(-0.3, 0.3)
            
            # Keep parameters bounded
            new_params = [max(-math.pi, min(math.pi, p)) for p in new_params]
            
            return new_params
            
        except Exception as e:
            self.logger.debug(f"Error updating parameters: {e}")
            return old_params
    
    def _validate_state(self, state: QAOAState, schedule: Schedule) -> bool:
        """Validate QAOA state for feasibility."""
        try:
            errors = []
            
            # Check all tasks are assigned
            task_ids = {task.id for task in schedule.tasks}
            assigned_tasks = set(state.resource_assignments.keys())
            
            if not task_ids.issubset(assigned_tasks):
                errors.append("Not all tasks are assigned")
            
            # Check resource capacity constraints
            resource_loads = {}
            for task_id, resource_id in state.resource_assignments.items():
                if resource_id not in resource_loads:
                    resource_loads[resource_id] = 0.0
                
                allocation = state.allocation_amounts.get(task_id, 1.0)
                resource_loads[resource_id] += allocation
            
            for resource_id, load in resource_loads.items():
                resource = schedule.get_resource(resource_id)
                if resource and load > resource.total_capacity:
                    errors.append(f"Resource {resource_id} over-allocated: {load}/{resource.total_capacity}")
            
            # Check allocation amounts are positive
            for task_id, allocation in state.allocation_amounts.items():
                if allocation <= 0:
                    errors.append(f"Task {task_id} has invalid allocation: {allocation}")
            
            state.validation_errors = errors
            state.is_valid = len(errors) == 0
            
            return state.is_valid
            
        except Exception as e:
            self.logger.error(f"Error validating state: {e}")
            state.validation_errors = [f"Validation error: {e}"]
            state.is_valid = False
            return False
    
    def _generate_fallback_state(self, current_state: QAOAState, schedule: Schedule) -> QAOAState:
        """Generate fallback state when optimization fails."""
        try:
            self.logger.info("Generating fallback state")
            
            # Use simple round-robin assignment as fallback
            assignments = {}
            allocations = {}
            
            resource_index = 0
            available_resources = [r for r in schedule.resources 
                                 if r.status == ResourceStatus.AVAILABLE]
            
            if not available_resources:
                available_resources = schedule.resources
            
            for task in schedule.tasks:
                resource = available_resources[resource_index % len(available_resources)]
                assignments[task.id] = resource.id
                allocations[task.id] = min(1.0, resource.available_capacity)
                resource_index += 1
            
            cost = self._calculate_cost_expectation(assignments, allocations, schedule)
            
            return QAOAState(
                resource_assignments=assignments,
                allocation_amounts=allocations,
                cost_expectation=cost,
                layer=current_state.layer,
                parameters=current_state.parameters,
                validation_errors=[],
                is_valid=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate fallback state: {e}")
            return current_state
    
    def _check_convergence(self, iteration: int) -> bool:
        """Check optimization convergence."""
        if iteration < 20 or len(self.optimization_history) < 20:
            return False
        
        recent_costs = [entry['cost'] for entry in self.optimization_history[-20:]]
        cost_variance = sum((c - sum(recent_costs) / len(recent_costs)) ** 2 
                           for c in recent_costs) / len(recent_costs)
        
        return cost_variance < self.params.convergence_threshold
    
    def _check_timeout(self) -> bool:
        """Check if optimization has exceeded timeout."""
        if not self.start_time:
            return False
        
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        return elapsed > self.params.timeout_seconds
    
    def _has_circular_dependencies(self, tasks: List[Task]) -> bool:
        """Check for circular dependencies in task list."""
        try:
            visited = set()
            rec_stack = set()
            
            def has_cycle(task_id: str, task_map: Dict[str, Task]) -> bool:
                if task_id in rec_stack:
                    return True
                if task_id in visited:
                    return False
                
                visited.add(task_id)
                rec_stack.add(task_id)
                
                task = task_map.get(task_id)
                if task:
                    for dep_id in task.dependencies:
                        if has_cycle(dep_id, task_map):
                            return True
                
                rec_stack.remove(task_id)
                return False
            
            task_map = {task.id: task for task in tasks}
            
            for task in tasks:
                if task.id not in visited:
                    if has_cycle(task.id, task_map):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking circular dependencies: {e}")
            return True  # Assume circular dependency on error for safety
    
    def _apply_solution_to_schedule(self, schedule: Schedule, state: QAOAState) -> None:
        """Apply QAOA solution to schedule with error handling."""
        try:
            # Clear existing assignments
            schedule.assignments.clear()
            
            # Apply new assignments
            for task_id, resource_id in state.resource_assignments.items():
                task = schedule.get_task(task_id)
                resource = schedule.get_resource(resource_id)
                
                if task and resource:
                    allocation = state.allocation_amounts.get(task_id, 1.0)
                    
                    # Create assignment
                    assignment = TaskAssignment(
                        task_id=task_id,
                        resource_id=resource_id,
                        start_time=schedule.start_time,  # Simplified - would need proper scheduling
                        end_time=schedule.start_time + task.duration,
                        allocated_capacity=allocation,
                        priority=1
                    )
                    
                    schedule.assignments.append(assignment)
                    
                    # Update task status
                    task.status = TaskStatus.READY
                    task.scheduled_start = assignment.start_time
                    task.scheduled_finish = assignment.end_time
            
            self.logger.info(f"Applied {len(state.resource_assignments)} assignments to schedule")
            
        except Exception as e:
            self.logger.error(f"Error applying solution to schedule: {e}")
            raise ResourceAllocationError(f"Failed to apply solution: {e}") from e
    
    def _calculate_metrics(self, schedule: Schedule, iterations: int,
                          optimization_time: timedelta) -> OptimizationMetrics:
        """Calculate optimization metrics with error handling."""
        try:
            return OptimizationMetrics(
                makespan=schedule.calculate_makespan(),
                total_cost=schedule.calculate_total_cost(),
                resource_utilization=schedule.get_resource_utilization(),
                constraint_violations=len(schedule.validate_dependencies()),
                quantum_energy=schedule.calculate_quantum_energy(),
                optimization_time=optimization_time,
                iterations=iterations,
                convergence_achieved=self._check_convergence(iterations)
            )
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            # Return default metrics on error
            return OptimizationMetrics(
                makespan=timedelta(0),
                total_cost=0.0,
                resource_utilization={},
                constraint_violations=0,
                quantum_energy=0.0,
                optimization_time=optimization_time,
                iterations=iterations,
                convergence_achieved=False
            )