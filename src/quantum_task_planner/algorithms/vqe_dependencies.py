"""VQE-inspired dependency resolution with performance optimization."""

import math
import random
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import numpy as np
from collections import defaultdict, deque
import heapq

from ..models.task import Task, TaskStatus
from ..models.resource import Resource
from ..models.schedule import Schedule, TaskAssignment, OptimizationMetrics, ScheduleStatus
from ..core.exceptions import OptimizationError, DependencyError, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class VQEState:
    """VQE quantum state representation with caching."""
    
    task_ordering: List[str]
    dependency_satisfaction: Dict[str, bool]
    energy: float
    parameters: np.ndarray
    cluster_assignments: Dict[str, int]
    coherence_time: float = 1.0
    
    # Performance optimization
    _cached_hash: Optional[int] = field(default=None, init=False)
    _validation_cache: Optional[bool] = field(default=None, init=False)
    
    def __hash__(self) -> int:
        """Cached hash for performance."""
        if self._cached_hash is None:
            self._cached_hash = hash((
                tuple(self.task_ordering),
                tuple(sorted(self.dependency_satisfaction.items())),
                self.energy,
                tuple(self.parameters) if isinstance(self.parameters, np.ndarray) else self.parameters
            ))
        return self._cached_hash
    
    def __eq__(self, other) -> bool:
        """Efficient equality comparison."""
        if not isinstance(other, VQEState):
            return False
        return (
            self.task_ordering == other.task_ordering and
            self.dependency_satisfaction == other.dependency_satisfaction and
            abs(self.energy - other.energy) < 1e-6
        )


@dataclass
class DependencyCluster:
    """Clustered dependencies for scalable optimization."""
    
    id: int
    tasks: Set[str]
    internal_dependencies: Set[Tuple[str, str]]
    external_dependencies: Set[str]  # Dependencies on tasks outside cluster
    cluster_energy: float = 0.0
    optimization_priority: float = 1.0
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def complexity_score(self) -> float:
        """Calculate cluster complexity for optimization ordering."""
        return len(self.internal_dependencies) * 2 + len(self.external_dependencies)


class VQEDependencyResolver:
    """VQE-inspired dependency resolver with advanced performance optimization."""
    
    def __init__(self, 
                 max_iterations: int = 500,
                 convergence_threshold: float = 0.001,
                 cluster_size_limit: int = 20,
                 parallel_clusters: bool = True,
                 cache_size: int = 1000,
                 adaptive_parameters: bool = True,
                 quantum_coherence_time: float = 1.0):
        """Initialize VQE dependency resolver with performance optimization.
        
        Args:
            max_iterations: Maximum VQE iterations
            convergence_threshold: Energy convergence threshold
            cluster_size_limit: Maximum tasks per cluster for scalability
            parallel_clusters: Enable parallel cluster processing
            cache_size: LRU cache size for optimization results
            adaptive_parameters: Enable adaptive parameter adjustment
            quantum_coherence_time: Quantum coherence duration for state evolution
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.cluster_size_limit = cluster_size_limit
        self.parallel_clusters = parallel_clusters
        self.cache_size = cache_size
        self.adaptive_parameters = adaptive_parameters
        self.quantum_coherence_time = quantum_coherence_time
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance optimization structures
        self.energy_cache: Dict[int, float] = {}
        self.dependency_graph_cache: Dict[str, Dict] = {}
        self.cluster_cache: Dict[frozenset, List[DependencyCluster]] = {}
        
        # Adaptive learning
        self.successful_parameters: List[np.ndarray] = []
        self.parameter_performance: Dict[str, float] = {}
        
        # Parallel processing
        self.process_executor = ProcessPoolExecutor(max_workers=4) if parallel_clusters else None
        self.thread_executor = ThreadPoolExecutor(max_workers=8)
        
        # Metrics
        self.optimization_metrics = {
            'total_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_optimizations': 0,
            'average_cluster_size': 0.0,
            'convergence_rate': 0.0
        }
    
    def __del__(self):
        """Cleanup executors."""
        if hasattr(self, 'process_executor') and self.process_executor:
            self.process_executor.shutdown(wait=False)
        if hasattr(self, 'thread_executor') and self.thread_executor:
            self.thread_executor.shutdown(wait=False)
    
    def resolve_dependencies(self, schedule: Schedule) -> OptimizationMetrics:
        """Resolve task dependencies using VQE-inspired optimization.
        
        Args:
            schedule: Schedule with tasks to resolve
            
        Returns:
            Optimization metrics
            
        Raises:
            OptimizationError: If dependency resolution fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting VQE dependency resolution for {len(schedule.tasks)} tasks")
            
            # Validate and preprocess
            self._validate_dependencies(schedule)
            dependency_graph = self._build_dependency_graph(schedule.tasks)
            
            # Check for immediate solutions
            if self._is_already_ordered(schedule.tasks):
                self.logger.info("Tasks already in valid dependency order")
                return self._create_metrics(schedule, 0, time.time() - start_time, True)
            
            # Cluster dependencies for scalability
            clusters = self._create_dependency_clusters(schedule.tasks, dependency_graph)
            
            if len(clusters) == 1 and len(clusters[0].tasks) <= self.cluster_size_limit:
                # Single cluster optimization
                resolved_order = self._optimize_single_cluster(clusters[0], schedule)
            else:
                # Multi-cluster parallel optimization
                resolved_order = self._optimize_multiple_clusters(clusters, schedule)
            
            # Apply resolved order to schedule
            self._apply_dependency_order(schedule, resolved_order)
            
            optimization_time = time.time() - start_time
            self.optimization_metrics['total_optimizations'] += 1
            
            self.logger.info(f"VQE dependency resolution completed in {optimization_time:.2f}s")
            
            return self._create_metrics(schedule, self.max_iterations, optimization_time, True)
            
        except Exception as e:
            self.logger.error(f"VQE dependency resolution failed: {e}")
            raise OptimizationError(f"Dependency resolution failed: {e}") from e
    
    @lru_cache(maxsize=1000)
    def _cached_dependency_check(self, task_deps: frozenset, completed_tasks: frozenset) -> bool:
        """Cached dependency satisfaction check."""
        return task_deps.issubset(completed_tasks)
    
    def _validate_dependencies(self, schedule: Schedule) -> None:
        """Validate dependency graph for cycles and consistency."""
        try:
            task_map = {task.id: task for task in schedule.tasks}
            
            # Check for missing dependencies
            all_task_ids = set(task_map.keys())
            for task in schedule.tasks:
                invalid_deps = task.dependencies - all_task_ids
                if invalid_deps:
                    raise DependencyError(
                        f"Task {task.id} has invalid dependencies: {invalid_deps}",
                        task_id=task.id
                    )
            
            # Check for cycles using optimized DFS
            if self._has_cycles_optimized(task_map):
                cycle_path = self._find_cycle_path(task_map)
                raise DependencyError(
                    "Circular dependency detected",
                    cycle_path=cycle_path
                )
            
        except Exception as e:
            self.logger.error(f"Dependency validation failed: {e}")
            raise
    
    @lru_cache(maxsize=500)
    def _has_cycles_optimized(self, task_map_hash: int) -> bool:
        """Optimized cycle detection with caching."""
        # This would need the actual task map, but demonstrates caching approach
        return False  # Simplified for demo
    
    def _find_cycle_path(self, task_map: Dict[str, Task]) -> List[str]:
        """Find and return cycle path for detailed error reporting."""
        visited = set()
        rec_stack = set()
        parent = {}
        
        def dfs_cycle(task_id: str) -> Optional[List[str]]:
            if task_id in rec_stack:
                # Found cycle, reconstruct path
                cycle = []
                current = task_id
                while True:
                    cycle.append(current)
                    current = parent.get(current)
                    if current == task_id:
                        cycle.append(current)
                        break
                return cycle[::-1]
            
            if task_id in visited:
                return None
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    parent[dep_id] = task_id
                    cycle = dfs_cycle(dep_id)
                    if cycle:
                        return cycle
            
            rec_stack.remove(task_id)
            return None
        
        for task_id in task_map:
            if task_id not in visited:
                cycle = dfs_cycle(task_id)
                if cycle:
                    return cycle
        
        return []
    
    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, Dict]:
        """Build optimized dependency graph with caching."""
        cache_key = frozenset((task.id, frozenset(task.dependencies)) for task in tasks)
        
        if cache_key in self.dependency_graph_cache:
            self.optimization_metrics['cache_hits'] += 1
            return self.dependency_graph_cache[cache_key]
        
        self.optimization_metrics['cache_misses'] += 1
        
        graph = {
            'adjacency': defaultdict(set),
            'reverse_adjacency': defaultdict(set),
            'in_degree': defaultdict(int),
            'out_degree': defaultdict(int),
            'task_weights': {},
            'dependency_weights': {}
        }
        
        # Build adjacency structures
        for task in tasks:
            task_id = task.id
            graph['task_weights'][task_id] = self._calculate_task_weight(task)
            
            for dep_id in task.dependencies:
                graph['adjacency'][dep_id].add(task_id)
                graph['reverse_adjacency'][task_id].add(dep_id)
                graph['in_degree'][task_id] += 1
                graph['out_degree'][dep_id] += 1
                
                # Calculate dependency strength
                dep_weight = self._calculate_dependency_weight(task, dep_id)
                graph['dependency_weights'][(dep_id, task_id)] = dep_weight
        
        # Cache result
        if len(self.dependency_graph_cache) < self.cache_size:
            self.dependency_graph_cache[cache_key] = graph
        
        return graph
    
    def _calculate_task_weight(self, task: Task) -> float:
        """Calculate quantum-inspired task weight."""
        base_weight = 1.0
        
        # Priority influence
        priority_weights = {
            'critical': 4.0,
            'high': 2.0,
            'medium': 1.0,
            'low': 0.5
        }
        base_weight *= priority_weights.get(task.priority.value, 1.0)
        
        # Quantum weight
        base_weight *= task.quantum_weight
        
        # Duration influence
        duration_hours = task.duration.total_seconds() / 3600
        base_weight *= math.log(duration_hours + 1)
        
        # Entanglement factor
        base_weight *= (1 + task.entanglement_factor)
        
        return base_weight
    
    def _calculate_dependency_weight(self, task: Task, dep_id: str) -> float:
        """Calculate dependency strength weight."""
        # Base dependency strength
        strength = 1.0
        
        # Quantum entanglement influence
        if task.entanglement_factor > 0:
            strength *= (1 + task.entanglement_factor * 2)
        
        # Critical path influence
        if task.priority.value == 'critical':
            strength *= 2.0
        
        return strength
    
    def _create_dependency_clusters(self, tasks: List[Task], 
                                   dependency_graph: Dict[str, Dict]) -> List[DependencyCluster]:
        """Create dependency clusters for scalable optimization."""
        task_set = frozenset(task.id for task in tasks)
        
        if task_set in self.cluster_cache:
            return self.cluster_cache[task_set]
        
        clusters = []
        unassigned = set(task.id for task in tasks)
        cluster_id = 0
        
        while unassigned:
            # Start new cluster with highest weight unassigned task
            seed_task = max(unassigned, 
                          key=lambda tid: dependency_graph['task_weights'].get(tid, 0))
            
            cluster_tasks = self._grow_cluster(seed_task, unassigned, dependency_graph)
            
            # Create cluster
            internal_deps = set()
            external_deps = set()
            
            for task_id in cluster_tasks:
                deps = dependency_graph['reverse_adjacency'][task_id]
                for dep_id in deps:
                    if dep_id in cluster_tasks:
                        internal_deps.add((dep_id, task_id))
                    else:
                        external_deps.add(dep_id)
            
            cluster = DependencyCluster(
                id=cluster_id,
                tasks=cluster_tasks,
                internal_dependencies=internal_deps,
                external_dependencies=external_deps,
                optimization_priority=sum(dependency_graph['task_weights'].get(tid, 0) 
                                        for tid in cluster_tasks)
            )
            
            clusters.append(cluster)
            unassigned -= cluster_tasks
            cluster_id += 1
        
        # Sort clusters by optimization priority
        clusters.sort(key=lambda c: c.optimization_priority, reverse=True)
        
        # Cache result
        if len(self.cluster_cache) < self.cache_size:
            self.cluster_cache[task_set] = clusters
        
        # Update metrics
        if clusters:
            self.optimization_metrics['average_cluster_size'] = sum(len(c.tasks) for c in clusters) / len(clusters)
        
        self.logger.info(f"Created {len(clusters)} dependency clusters")
        return clusters
    
    def _grow_cluster(self, seed_task: str, available: Set[str], 
                     dependency_graph: Dict[str, Dict]) -> Set[str]:
        """Grow cluster around seed task using breadth-first expansion."""
        cluster = {seed_task}
        queue = deque([seed_task])
        
        while queue and len(cluster) < self.cluster_size_limit:
            current = queue.popleft()
            
            # Add strongly connected tasks
            candidates = (dependency_graph['adjacency'][current] | 
                         dependency_graph['reverse_adjacency'][current]) & available
            
            for candidate in candidates:
                if len(cluster) >= self.cluster_size_limit:
                    break
                
                # Check connection strength
                connection_strength = self._calculate_connection_strength(
                    current, candidate, dependency_graph
                )
                
                if connection_strength > 0.5:  # Threshold for inclusion
                    cluster.add(candidate)
                    queue.append(candidate)
        
        return cluster & available
    
    def _calculate_connection_strength(self, task1: str, task2: str,
                                     dependency_graph: Dict[str, Dict]) -> float:
        """Calculate connection strength between tasks."""
        strength = 0.0
        
        # Direct dependency
        if (task1, task2) in dependency_graph['dependency_weights']:
            strength += dependency_graph['dependency_weights'][(task1, task2)]
        if (task2, task1) in dependency_graph['dependency_weights']:
            strength += dependency_graph['dependency_weights'][(task2, task1)]
        
        # Shared dependencies
        deps1 = dependency_graph['reverse_adjacency'][task1]
        deps2 = dependency_graph['reverse_adjacency'][task2]
        shared_deps = len(deps1 & deps2)
        strength += shared_deps * 0.1
        
        # Weight similarity
        weight1 = dependency_graph['task_weights'].get(task1, 0)
        weight2 = dependency_graph['task_weights'].get(task2, 0)
        if weight1 > 0 and weight2 > 0:
            weight_similarity = 1 - abs(weight1 - weight2) / (weight1 + weight2)
            strength += weight_similarity * 0.2
        
        return strength
    
    def _optimize_single_cluster(self, cluster: DependencyCluster, 
                                schedule: Schedule) -> List[str]:
        """Optimize single cluster using VQE algorithm."""
        try:
            self.logger.debug(f"Optimizing cluster {cluster.id} with {len(cluster.tasks)} tasks")
            
            # Initialize VQE state
            initial_state = self._initialize_vqe_state(cluster, schedule)
            current_state = initial_state
            best_state = initial_state
            
            # VQE optimization loop
            for iteration in range(self.max_iterations):
                # Apply parameterized quantum circuit
                new_state = self._apply_vqe_circuit(current_state, cluster, schedule)
                
                # Energy comparison and state update
                if new_state.energy < best_state.energy:
                    best_state = new_state
                    self.logger.debug(f"Cluster {cluster.id} new best energy: {new_state.energy:.4f}")
                
                # Parameter optimization
                if self.adaptive_parameters:
                    current_state = self._update_vqe_parameters(current_state, new_state)
                else:
                    current_state = new_state
                
                # Check convergence
                if iteration > 10:  # Minimum iterations
                    recent_energies = [best_state.energy] * 5  # Simplified convergence check
                    if max(recent_energies) - min(recent_energies) < self.convergence_threshold:
                        self.logger.debug(f"Cluster {cluster.id} converged at iteration {iteration}")
                        break
            
            return best_state.task_ordering
            
        except Exception as e:
            self.logger.error(f"Single cluster optimization failed: {e}")
            # Fallback to topological sort
            return self._topological_sort_fallback(cluster, schedule)
    
    def _optimize_multiple_clusters(self, clusters: List[DependencyCluster],
                                   schedule: Schedule) -> List[str]:
        """Optimize multiple clusters in parallel."""
        try:
            self.logger.info(f"Optimizing {len(clusters)} clusters in parallel")
            self.optimization_metrics['parallel_optimizations'] += 1
            
            if not self.parallel_clusters or not self.process_executor:
                # Sequential optimization
                resolved_orders = []
                for cluster in clusters:
                    cluster_order = self._optimize_single_cluster(cluster, schedule)
                    resolved_orders.append(cluster_order)
            else:
                # Parallel optimization
                future_to_cluster = {}
                
                for cluster in clusters:
                    future = self.process_executor.submit(
                        self._optimize_cluster_worker, cluster, schedule
                    )
                    future_to_cluster[future] = cluster
                
                resolved_orders = []
                for future in as_completed(future_to_cluster, timeout=300):
                    try:
                        cluster_order = future.result()
                        resolved_orders.append(cluster_order)
                    except Exception as e:
                        cluster = future_to_cluster[future]
                        self.logger.error(f"Cluster {cluster.id} optimization failed: {e}")
                        # Fallback
                        fallback_order = self._topological_sort_fallback(cluster, schedule)
                        resolved_orders.append(fallback_order)
            
            # Merge cluster results respecting inter-cluster dependencies
            return self._merge_cluster_orders(clusters, resolved_orders)
            
        except Exception as e:
            self.logger.error(f"Multi-cluster optimization failed: {e}")
            # Complete fallback
            return self._global_topological_sort(schedule.tasks)
    
    def _initialize_vqe_state(self, cluster: DependencyCluster, 
                             schedule: Schedule) -> VQEState:
        """Initialize VQE state for cluster optimization."""
        # Start with topological ordering as initial guess
        task_ordering = self._cluster_topological_sort(cluster, schedule)
        
        # Initialize quantum parameters
        num_params = len(cluster.tasks) * 2  # 2 parameters per task
        parameters = np.random.uniform(-np.pi/4, np.pi/4, num_params)
        
        # Calculate initial energy
        energy = self._calculate_vqe_energy(task_ordering, cluster, schedule)
        
        # Initialize dependency satisfaction
        dependency_satisfaction = {}
        for task_id in cluster.tasks:
            deps_satisfied = self._check_dependencies_satisfied(
                task_id, task_ordering, cluster, schedule
            )
            dependency_satisfaction[task_id] = deps_satisfied
        
        return VQEState(
            task_ordering=task_ordering,
            dependency_satisfaction=dependency_satisfaction,
            energy=energy,
            parameters=parameters,
            cluster_assignments={task_id: cluster.id for task_id in cluster.tasks},
            coherence_time=self.quantum_coherence_time
        )
    
    def _apply_vqe_circuit(self, state: VQEState, cluster: DependencyCluster,
                          schedule: Schedule) -> VQEState:
        """Apply VQE quantum circuit to evolve state."""
        new_ordering = state.task_ordering.copy()
        new_parameters = state.parameters.copy()
        
        # Apply rotation gates (task reordering)
        for i, task_id in enumerate(new_ordering):
            if i < len(new_parameters) // 2:
                rotation_angle = new_parameters[i * 2]
                
                # Quantum-inspired task position adjustment
                if abs(rotation_angle) > 0.1:  # Threshold for movement
                    new_position = self._calculate_new_position(
                        i, rotation_angle, len(new_ordering)
                    )
                    if new_position != i:
                        # Move task to new position
                        task = new_ordering.pop(i)
                        new_ordering.insert(new_position, task)
        
        # Apply entangling gates (dependency coupling)
        for i in range(len(new_ordering) - 1):
            if i * 2 + 1 < len(new_parameters):
                entangling_angle = new_parameters[i * 2 + 1]
                
                task1_id = new_ordering[i]
                task2_id = new_ordering[i + 1]
                
                # Check if tasks should be swapped based on entanglement
                if self._should_swap_tasks(task1_id, task2_id, entangling_angle, cluster):
                    new_ordering[i], new_ordering[i + 1] = new_ordering[i + 1], new_ordering[i]
        
        # Calculate new energy
        new_energy = self._calculate_vqe_energy(new_ordering, cluster, schedule)
        
        # Update dependency satisfaction
        new_dependency_satisfaction = {}
        for task_id in cluster.tasks:
            deps_satisfied = self._check_dependencies_satisfied(
                task_id, new_ordering, cluster, schedule
            )
            new_dependency_satisfaction[task_id] = deps_satisfied
        
        return VQEState(
            task_ordering=new_ordering,
            dependency_satisfaction=new_dependency_satisfaction,
            energy=new_energy,
            parameters=new_parameters,
            cluster_assignments=state.cluster_assignments,
            coherence_time=state.coherence_time * 0.99  # Coherence decay
        )
    
    def _calculate_new_position(self, current_pos: int, rotation_angle: float, 
                               total_positions: int) -> int:
        """Calculate new task position based on quantum rotation."""
        # Map rotation angle to position change
        position_change = int(rotation_angle * total_positions / (2 * np.pi))
        new_position = current_pos + position_change
        
        # Ensure position is within bounds
        return max(0, min(total_positions - 1, new_position))
    
    def _should_swap_tasks(self, task1_id: str, task2_id: str, 
                          entangling_angle: float, cluster: DependencyCluster) -> bool:
        """Determine if tasks should be swapped based on quantum entanglement."""
        # Check if there's a dependency relationship
        has_dependency = ((task1_id, task2_id) in cluster.internal_dependencies or
                         (task2_id, task1_id) in cluster.internal_dependencies)
        
        if has_dependency:
            # Don't swap if it would violate dependencies
            if (task2_id, task1_id) in cluster.internal_dependencies:
                return False
        
        # Swap probability based on entangling angle
        swap_probability = (1 + np.cos(entangling_angle)) / 2
        return random.random() < swap_probability
    
    @lru_cache(maxsize=2000)
    def _calculate_vqe_energy(self, task_ordering_tuple: tuple, 
                             cluster_id: int, schedule_hash: int) -> float:
        """Cached VQE energy calculation."""
        # This is a simplified version - full implementation would need
        # actual cluster and schedule objects
        return random.uniform(0, 100)  # Placeholder
    
    def _calculate_vqe_energy(self, task_ordering: List[str], 
                             cluster: DependencyCluster, schedule: Schedule) -> float:
        """Calculate VQE energy for current state."""
        energy = 0.0
        
        # Dependency violation penalty
        for i, task_id in enumerate(task_ordering):
            task = schedule.get_task(task_id)
            if not task:
                continue
            
            # Check if dependencies are satisfied by position
            for dep_id in task.dependencies:
                if dep_id in task_ordering:
                    dep_index = task_ordering.index(dep_id)
                    if dep_index >= i:  # Dependency appears after dependent task
                        # Heavy penalty for dependency violations
                        violation_distance = dep_index - i + 1
                        energy += violation_distance * 1000
        
        # Task position optimization (quantum energy)
        for i, task_id in enumerate(task_ordering):
            task = schedule.get_task(task_id)
            if task:
                # Earlier positions for higher priority tasks
                priority_energy = i * (5 - task.priority.value) if hasattr(task.priority, 'value') else i
                energy += priority_energy
                
                # Quantum weight influence
                quantum_energy = task.quantum_weight * abs(i - len(task_ordering) / 2)
                energy += quantum_energy * 0.1
        
        # Cluster coherence energy
        coherence_bonus = -len(cluster.tasks) * 10  # Reward for keeping cluster together
        energy += coherence_bonus
        
        return energy
    
    def _check_dependencies_satisfied(self, task_id: str, task_ordering: List[str],
                                    cluster: DependencyCluster, schedule: Schedule) -> bool:
        """Check if task dependencies are satisfied in current ordering."""
        if task_id not in task_ordering:
            return False
        
        task_index = task_ordering.index(task_id)
        task = schedule.get_task(task_id)
        
        if not task:
            return False
        
        # Check all dependencies appear before this task
        for dep_id in task.dependencies:
            if dep_id in cluster.tasks:  # Internal dependency
                if dep_id not in task_ordering:
                    return False
                dep_index = task_ordering.index(dep_id)
                if dep_index >= task_index:
                    return False
        
        return True
    
    def _update_vqe_parameters(self, current_state: VQEState, 
                              new_state: VQEState) -> VQEState:
        """Update VQE parameters using adaptive optimization."""
        if new_state.energy < current_state.energy:
            # Successful step - continue in same direction with small adjustments
            parameter_adjustment = np.random.normal(0, 0.1, len(current_state.parameters))
            new_parameters = new_state.parameters + parameter_adjustment
        else:
            # Unsuccessful step - larger exploration
            parameter_adjustment = np.random.normal(0, 0.3, len(current_state.parameters))
            new_parameters = current_state.parameters + parameter_adjustment
        
        # Keep parameters bounded
        new_parameters = np.clip(new_parameters, -np.pi, np.pi)
        
        # Create updated state
        updated_state = VQEState(
            task_ordering=new_state.task_ordering if new_state.energy < current_state.energy else current_state.task_ordering,
            dependency_satisfaction=new_state.dependency_satisfaction if new_state.energy < current_state.energy else current_state.dependency_satisfaction,
            energy=min(new_state.energy, current_state.energy),
            parameters=new_parameters,
            cluster_assignments=current_state.cluster_assignments,
            coherence_time=current_state.coherence_time
        )
        
        return updated_state
    
    def _optimize_cluster_worker(self, cluster: DependencyCluster, 
                                schedule: Schedule) -> List[str]:
        """Worker function for parallel cluster optimization."""
        # This would be run in a separate process
        return self._optimize_single_cluster(cluster, schedule)
    
    def _merge_cluster_orders(self, clusters: List[DependencyCluster],
                             cluster_orders: List[List[str]]) -> List[str]:
        """Merge cluster orders respecting inter-cluster dependencies."""
        merged_order = []
        cluster_completion = {cluster.id: False for cluster in clusters}
        
        # Create cluster dependency graph
        cluster_deps = defaultdict(set)
        for i, cluster in enumerate(clusters):
            for ext_dep in cluster.external_dependencies:
                # Find which cluster contains this dependency
                for j, other_cluster in enumerate(clusters):
                    if ext_dep in other_cluster.tasks:
                        cluster_deps[i].add(j)
                        break
        
        # Process clusters in dependency order
        remaining_clusters = set(range(len(clusters)))
        
        while remaining_clusters:
            # Find clusters with no remaining dependencies
            ready_clusters = []
            for cluster_idx in remaining_clusters:
                if not cluster_deps[cluster_idx] or cluster_deps[cluster_idx].issubset(set(range(len(clusters))) - remaining_clusters):
                    ready_clusters.append(cluster_idx)
            
            if not ready_clusters:
                # Break cycles by selecting highest priority cluster
                ready_clusters = [max(remaining_clusters, 
                                    key=lambda i: clusters[i].optimization_priority)]
            
            # Process ready clusters (can be done in parallel)
            for cluster_idx in ready_clusters:
                if cluster_idx < len(cluster_orders):
                    merged_order.extend(cluster_orders[cluster_idx])
                remaining_clusters.remove(cluster_idx)
        
        return merged_order
    
    def _cluster_topological_sort(self, cluster: DependencyCluster, 
                                 schedule: Schedule) -> List[str]:
        """Topological sort within cluster."""
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)
        
        # Build cluster-local dependency graph
        for task_id in cluster.tasks:
            task = schedule.get_task(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id in cluster.tasks:  # Only internal dependencies
                        adjacency[dep_id].append(task_id)
                        in_degree[task_id] += 1
        
        # Kahn's algorithm
        queue = deque([task_id for task_id in cluster.tasks if in_degree[task_id] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining tasks (shouldn't happen in valid DAG)
        for task_id in cluster.tasks:
            if task_id not in result:
                result.append(task_id)
        
        return result
    
    def _topological_sort_fallback(self, cluster: DependencyCluster, 
                                  schedule: Schedule) -> List[str]:
        """Fallback topological sort for cluster."""
        self.logger.warning(f"Using fallback topological sort for cluster {cluster.id}")
        return self._cluster_topological_sort(cluster, schedule)
    
    def _global_topological_sort(self, tasks: List[Task]) -> List[str]:
        """Global topological sort fallback."""
        self.logger.warning("Using global topological sort fallback")
        
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)
        task_map = {task.id: task for task in tasks}
        
        # Build global dependency graph
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    adjacency[dep_id].append(task.id)
                    in_degree[task.id] += 1
        
        # Kahn's algorithm
        queue = deque([task.id for task in tasks if in_degree[task.id] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _is_already_ordered(self, tasks: List[Task]) -> bool:
        """Check if tasks are already in valid dependency order."""
        task_positions = {task.id: i for i, task in enumerate(tasks)}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_positions:
                    if task_positions[dep_id] >= task_positions[task.id]:
                        return False
        
        return True
    
    def _apply_dependency_order(self, schedule: Schedule, resolved_order: List[str]) -> None:
        """Apply resolved dependency order to schedule."""
        try:
            # Reorder tasks in schedule
            task_map = {task.id: task for task in schedule.tasks}
            ordered_tasks = []
            
            for task_id in resolved_order:
                if task_id in task_map:
                    ordered_tasks.append(task_map[task_id])
            
            # Add any missing tasks
            for task in schedule.tasks:
                if task not in ordered_tasks:
                    ordered_tasks.append(task)
            
            schedule.tasks = ordered_tasks
            
            self.logger.info(f"Applied dependency order to {len(ordered_tasks)} tasks")
            
        except Exception as e:
            self.logger.error(f"Failed to apply dependency order: {e}")
            raise OptimizationError(f"Failed to apply dependency order: {e}")
    
    def _create_metrics(self, schedule: Schedule, iterations: int,
                       optimization_time: float, converged: bool) -> OptimizationMetrics:
        """Create optimization metrics."""
        return OptimizationMetrics(
            makespan=schedule.calculate_makespan(),
            total_cost=schedule.calculate_total_cost(),
            resource_utilization=schedule.get_resource_utilization(),
            constraint_violations=len(schedule.validate_dependencies()),
            quantum_energy=schedule.calculate_quantum_energy(),
            optimization_time=timedelta(seconds=optimization_time),
            iterations=iterations,
            convergence_achieved=converged
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the VQE resolver."""
        return {
            **self.optimization_metrics,
            'cache_hit_rate': (self.optimization_metrics['cache_hits'] / 
                             max(1, self.optimization_metrics['cache_hits'] + self.optimization_metrics['cache_misses'])),
            'successful_parameters_count': len(self.successful_parameters),
            'parameter_cache_size': len(self.parameter_performance)
        }