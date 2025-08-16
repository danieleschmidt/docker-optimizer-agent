"""Quantum annealing algorithm for task scheduling optimization."""

import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..models.resource import Resource
from ..models.schedule import (
    OptimizationMetrics,
    Schedule,
    ScheduleStatus,
    TaskAssignment,
)
from ..models.task import Task, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class AnnealingState:
    """Quantum annealing state representation."""
    assignments: Dict[str, str]  # task_id -> resource_id
    start_times: Dict[str, datetime]  # task_id -> start_time
    energy: float
    temperature: float
    iteration: int


class QuantumAnnealingScheduler:
    """Quantum-inspired annealing scheduler for task optimization."""

    def __init__(self,
                 initial_temperature: float = 10.0,
                 final_temperature: float = 0.01,
                 cooling_rate: float = 0.95,
                 max_iterations: int = 1000,
                 tunneling_probability: float = 0.1,
                 convergence_threshold: float = 0.001):
        """Initialize quantum annealing scheduler.
        
        Args:
            initial_temperature: Starting quantum temperature
            final_temperature: Final quantum temperature
            cooling_rate: Temperature cooling rate per iteration
            max_iterations: Maximum optimization iterations
            tunneling_probability: Quantum tunneling probability
            convergence_threshold: Energy convergence threshold
        """
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.tunneling_probability = tunneling_probability
        self.convergence_threshold = convergence_threshold

        # Optimization tracking
        self.best_state: Optional[AnnealingState] = None
        self.energy_history: List[float] = []
        self.temperature_history: List[float] = []
        self.optimization_start_time: Optional[datetime] = None

    def optimize_schedule(self, schedule: Schedule) -> OptimizationMetrics:
        """Optimize schedule using quantum annealing.
        
        Args:
            schedule: Schedule to optimize
            
        Returns:
            Optimization metrics and results
        """
        logger.info(f"Starting quantum annealing optimization for schedule {schedule.id}")
        self.optimization_start_time = datetime.utcnow()

        # Initialize schedule status
        schedule.status = ScheduleStatus.OPTIMIZING

        # Generate initial state
        current_state = self._generate_initial_state(schedule)
        self.best_state = current_state

        # Optimization loop
        converged = False
        for iteration in range(self.max_iterations):
            # Update temperature (quantum cooling schedule)
            current_state.temperature = self._calculate_temperature(iteration)
            current_state.iteration = iteration

            # Generate neighbor state using quantum-inspired moves
            neighbor_state = self._generate_neighbor_state(current_state, schedule)

            # Accept or reject based on quantum criteria
            if self._accept_state(current_state, neighbor_state):
                current_state = neighbor_state

                # Track best state found
                if neighbor_state.energy < self.best_state.energy:
                    self.best_state = neighbor_state
                    logger.debug(f"New best energy: {neighbor_state.energy:.4f} at iteration {iteration}")

            # Track optimization progress
            self.energy_history.append(current_state.energy)
            self.temperature_history.append(current_state.temperature)

            # Check convergence
            if self._check_convergence(iteration):
                converged = True
                logger.info(f"Convergence achieved at iteration {iteration}")
                break

            # Early stopping if temperature is too low
            if current_state.temperature < self.final_temperature:
                logger.info(f"Final temperature reached at iteration {iteration}")
                break

        # Apply best solution to schedule
        self._apply_solution_to_schedule(schedule, self.best_state)

        # Calculate final metrics
        optimization_time = datetime.utcnow() - self.optimization_start_time
        metrics = self._calculate_metrics(schedule, iteration + 1, optimization_time, converged)

        schedule.status = ScheduleStatus.OPTIMIZED
        schedule.optimized_at = datetime.utcnow()
        schedule.metrics = metrics

        logger.info(f"Optimization completed. Final energy: {self.best_state.energy:.4f}")
        return metrics

    def _generate_initial_state(self, schedule: Schedule) -> AnnealingState:
        """Generate initial quantum state using heuristic assignment."""
        assignments = {}
        start_times = {}

        # Sort tasks by priority and dependencies
        sorted_tasks = self._topological_sort_tasks(schedule.tasks)

        # Greedy initial assignment
        for task in sorted_tasks:
            best_resource = None
            best_start_time = None
            best_energy = float('inf')

            for resource in schedule.resources:
                # Find earliest feasible start time
                earliest_start = self._find_earliest_start_time(
                    task, resource, assignments, start_times, schedule
                )

                if earliest_start is not None:
                    # Calculate energy for this assignment
                    temp_assignments = assignments.copy()
                    temp_start_times = start_times.copy()
                    temp_assignments[task.id] = resource.id
                    temp_start_times[task.id] = earliest_start

                    energy = self._calculate_energy(temp_assignments, temp_start_times, schedule)

                    if energy < best_energy:
                        best_energy = energy
                        best_resource = resource
                        best_start_time = earliest_start

            # Assign task to best resource
            if best_resource:
                assignments[task.id] = best_resource.id
                start_times[task.id] = best_start_time
            else:
                # Fallback: assign to first available resource
                assignments[task.id] = schedule.resources[0].id
                start_times[task.id] = schedule.start_time

        energy = self._calculate_energy(assignments, start_times, schedule)

        return AnnealingState(
            assignments=assignments,
            start_times=start_times,
            energy=energy,
            temperature=self.initial_temperature,
            iteration=0
        )

    def _generate_neighbor_state(self, current_state: AnnealingState,
                                 schedule: Schedule) -> AnnealingState:
        """Generate neighbor state using quantum-inspired moves."""
        new_assignments = current_state.assignments.copy()
        new_start_times = current_state.start_times.copy()

        # Select quantum move type
        move_type = random.choice(['reassign', 'reschedule', 'swap', 'quantum_tunnel'])

        if move_type == 'reassign':
            # Reassign random task to different resource
            task_id = random.choice(list(new_assignments.keys()))
            available_resources = [r.id for r in schedule.resources
                                 if r.id != new_assignments[task_id]]
            if available_resources:
                new_resource_id = random.choice(available_resources)
                new_assignments[task_id] = new_resource_id

                # Recalculate start time
                task = schedule.get_task(task_id)
                resource = schedule.get_resource(new_resource_id)
                new_start_time = self._find_earliest_start_time(
                    task, resource, new_assignments, new_start_times, schedule
                )
                if new_start_time:
                    new_start_times[task_id] = new_start_time

        elif move_type == 'reschedule':
            # Adjust start time of random task
            task_id = random.choice(list(new_start_times.keys()))
            current_start = new_start_times[task_id]

            # Random time adjustment within reasonable bounds
            max_shift = timedelta(hours=2)
            shift = timedelta(seconds=random.randint(-max_shift.total_seconds(),
                                                   max_shift.total_seconds()))
            new_start_times[task_id] = max(schedule.start_time, current_start + shift)

        elif move_type == 'swap':
            # Swap assignments of two tasks
            task_ids = list(new_assignments.keys())
            if len(task_ids) >= 2:
                task1, task2 = random.sample(task_ids, 2)
                new_assignments[task1], new_assignments[task2] = \
                    new_assignments[task2], new_assignments[task1]

        elif move_type == 'quantum_tunnel':
            # Quantum tunneling: large energy barrier jump
            task_id = random.choice(list(new_assignments.keys()))

            # Find resource with different characteristics
            current_resource_id = new_assignments[task_id]
            tunnel_candidates = []

            for resource in schedule.resources:
                if resource.id != current_resource_id:
                    # Calculate quantum affinity
                    task = schedule.get_task(task_id)
                    affinity = resource.calculate_quantum_affinity(
                        task.quantum_weight, task.entanglement_factor
                    )
                    tunnel_candidates.append((resource.id, affinity))

            if tunnel_candidates:
                # Select based on quantum probability
                weights = [affinity for _, affinity in tunnel_candidates]
                total_weight = sum(weights)
                if total_weight > 0:
                    probabilities = [w / total_weight for w in weights]
                    selected_resource = random.choices(
                        [res_id for res_id, _ in tunnel_candidates],
                        weights=probabilities
                    )[0]
                    new_assignments[task_id] = selected_resource

        # Calculate energy for new state
        new_energy = self._calculate_energy(new_assignments, new_start_times, schedule)

        return AnnealingState(
            assignments=new_assignments,
            start_times=new_start_times,
            energy=new_energy,
            temperature=current_state.temperature,
            iteration=current_state.iteration
        )

    def _accept_state(self, current_state: AnnealingState,
                      neighbor_state: AnnealingState) -> bool:
        """Determine state acceptance using quantum criteria."""
        energy_diff = neighbor_state.energy - current_state.energy

        # Always accept better solutions
        if energy_diff < 0:
            return True

        # Quantum acceptance probability
        if current_state.temperature > 0:
            # Classical Boltzmann factor
            classical_prob = math.exp(-energy_diff / current_state.temperature)

            # Quantum tunneling enhancement
            quantum_factor = 1 + self.tunneling_probability * math.exp(-current_state.iteration / 100)
            acceptance_prob = min(1.0, classical_prob * quantum_factor)

            return random.random() < acceptance_prob

        return False

    def _calculate_temperature(self, iteration: int) -> float:
        """Calculate quantum temperature using cooling schedule."""
        # Exponential cooling with quantum corrections
        base_temp = self.initial_temperature * (self.cooling_rate ** iteration)

        # Quantum fluctuations
        quantum_noise = 0.1 * self.initial_temperature * math.exp(-iteration / 500)
        fluctuation = random.gauss(0, quantum_noise)

        return max(self.final_temperature, base_temp + fluctuation)

    def _calculate_energy(self, assignments: Dict[str, str],
                          start_times: Dict[str, datetime],
                          schedule: Schedule) -> float:
        """Calculate total energy of schedule state."""
        energy = 0.0

        # Makespan penalty (primary objective)
        max_end_time = schedule.start_time
        for task_id, start_time in start_times.items():
            task = schedule.get_task(task_id)
            if task:
                end_time = start_time + task.duration
                max_end_time = max(max_end_time, end_time)

        makespan = (max_end_time - schedule.start_time).total_seconds() / 3600  # hours
        energy += makespan * 100  # Weight makespan heavily

        # Resource cost
        for task_id, resource_id in assignments.items():
            task = schedule.get_task(task_id)
            resource = schedule.get_resource(resource_id)
            if task and resource:
                duration_hours = task.duration.total_seconds() / 3600
                cost = duration_hours * resource.cost_per_unit
                energy += cost

        # Dependency violations (hard constraint)
        for task_id, start_time in start_times.items():
            task = schedule.get_task(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id in start_times:
                        dep_task = schedule.get_task(dep_id)
                        dep_end_time = start_times[dep_id] + dep_task.duration
                        if start_time < dep_end_time:
                            # Heavy penalty for dependency violations
                            violation_time = (dep_end_time - start_time).total_seconds() / 3600
                            energy += violation_time * 1000

        # Resource over-allocation penalty
        resource_loads = {}
        for task_id, resource_id in assignments.items():
            if resource_id not in resource_loads:
                resource_loads[resource_id] = []

            task = schedule.get_task(task_id)
            start_time = start_times[task_id]
            end_time = start_time + task.duration
            resource_loads[resource_id].append((start_time, end_time, 1.0))  # Assume unit capacity

        for resource_id, loads in resource_loads.items():
            resource = schedule.get_resource(resource_id)
            if resource:
                # Check for time overlaps
                sorted_loads = sorted(loads, key=lambda x: x[0])
                for i in range(len(sorted_loads)):
                    current_load = 0.0
                    for j in range(len(sorted_loads)):
                        if i != j:
                            # Check for overlap
                            start1, end1, cap1 = sorted_loads[i]
                            start2, end2, cap2 = sorted_loads[j]
                            if start1 < end2 and start2 < end1:
                                current_load += cap2

                    if current_load > resource.total_capacity:
                        energy += (current_load - resource.total_capacity) * 500

        # Quantum energy contributions
        quantum_energy = 0.0
        for task_id in assignments.keys():
            task = schedule.get_task(task_id)
            if task:
                task_quantum_energy = task.calculate_quantum_energy(datetime.utcnow())
                quantum_energy += task_quantum_energy

        energy += quantum_energy * 0.1  # Small weight for quantum effects

        return energy

    def _check_convergence(self, iteration: int) -> bool:
        """Check if optimization has converged."""
        if iteration < 50:  # Minimum iterations
            return False

        # Check energy stability over last 20 iterations
        recent_energies = self.energy_history[-20:]
        if len(recent_energies) < 20:
            return False

        energy_variance = sum((e - sum(recent_energies) / len(recent_energies)) ** 2
                             for e in recent_energies) / len(recent_energies)

        return energy_variance < self.convergence_threshold

    def _apply_solution_to_schedule(self, schedule: Schedule, state: AnnealingState) -> None:
        """Apply optimized solution to schedule."""
        # Clear existing assignments
        schedule.assignments.clear()

        # Apply new assignments
        for task_id, resource_id in state.assignments.items():
            start_time = state.start_times[task_id]
            task = schedule.get_task(task_id)

            if task:
                end_time = start_time + task.duration
                assignment = TaskAssignment(
                    task_id=task_id,
                    resource_id=resource_id,
                    start_time=start_time,
                    end_time=end_time,
                    allocated_capacity=1.0,  # Assume full capacity
                    priority=1
                )
                schedule.assignments.append(assignment)

                # Update task
                task.scheduled_start = start_time
                task.scheduled_finish = end_time
                task.status = TaskStatus.READY

    def _calculate_metrics(self, schedule: Schedule, iterations: int,
                          optimization_time: timedelta, converged: bool) -> OptimizationMetrics:
        """Calculate optimization metrics."""
        return OptimizationMetrics(
            makespan=schedule.calculate_makespan(),
            total_cost=schedule.calculate_total_cost(),
            resource_utilization=schedule.get_resource_utilization(),
            constraint_violations=len(schedule.validate_dependencies()),
            quantum_energy=schedule.calculate_quantum_energy(),
            optimization_time=optimization_time,
            iterations=iterations,
            convergence_achieved=converged
        )

    def _topological_sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks in topological order respecting dependencies."""
        # Simple topological sort implementation
        in_degree = {task.id: 0 for task in tasks}
        task_map = {task.id: task for task in tasks}

        # Calculate in-degrees
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in in_degree:
                    in_degree[task.id] += 1

        # Sort by priority and in-degree
        sorted_tasks = sorted(tasks,
                            key=lambda t: (in_degree[t.id], -t.priority.value))

        return sorted_tasks

    def _find_earliest_start_time(self, task: Task, resource: Resource,
                                  assignments: Dict[str, str],
                                  start_times: Dict[str, datetime],
                                  schedule: Schedule) -> Optional[datetime]:
        """Find earliest feasible start time for task on resource."""
        # Start from dependency completion or schedule start
        earliest = schedule.start_time

        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id in start_times and dep_id in assignments:
                dep_task = schedule.get_task(dep_id)
                if dep_task:
                    dep_end = start_times[dep_id] + dep_task.duration
                    earliest = max(earliest, dep_end)

        # Check task constraints
        if task.earliest_start:
            earliest = max(earliest, task.earliest_start)

        # Check resource availability (simplified)
        # In a full implementation, this would check for resource conflicts

        return earliest
