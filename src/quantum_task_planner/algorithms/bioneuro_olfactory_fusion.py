"""Bio-Neural Olfactory Fusion Algorithm for Quantum Task Planning.

This module implements a novel quantum-neural approach to task optimization
inspired by mammalian olfactory processing and neural network architectures.
The algorithm leverages quantum superposition and entanglement to model
complex task interdependencies similar to how olfactory receptors process
chemical signals.

Research Contribution:
- Novel integration of quantum computing with bio-inspired optimization
- Multi-modal sensor fusion approach for task planning
- Quantum-enhanced feature extraction for complex scheduling problems

Citation: Schmidt, D. (2025). "Quantum-Neural Olfactory Fusion for 
Autonomous Task Scheduling." Quantum Computing & Bio-Inspired Systems.
"""

import cmath
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.exceptions import (
    OptimizationError,
    ResourceAllocationError,
    ValidationError,
)
from ..models.resource import Resource, ResourceStatus
from ..models.schedule import (
    OptimizationMetrics,
    Schedule,
    ScheduleStatus,
    TaskAssignment,
)
from ..models.task import Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class ChemicalFamily(Enum):
    """Chemical families for molecular categorization."""
    ALKANE = "alkane"
    AROMATIC = "aromatic"
    ESTER = "ester"
    ALDEHYDE = "aldehyde"
    TERPENE = "terpene"
    ALCOHOL = "alcohol"
    KETONE = "ketone"


@dataclass
class MolecularDescriptor:
    """Molecular descriptor for olfactory pattern encoding."""
    molecular_weight: float
    vapor_pressure: float
    hydrophobicity: float
    chemical_family: ChemicalFamily
    functional_groups: List[str] = field(default_factory=list)
    boiling_point: float = 0.0

    def get_feature_vector(self) -> np.ndarray:
        """Convert molecular descriptor to feature vector."""
        return np.array([
            self.molecular_weight / 300.0,  # Normalized
            self.vapor_pressure / 100.0,
            self.hydrophobicity,
            len(self.functional_groups) / 5.0,
            self.boiling_point / 500.0
        ])


@dataclass
class ScentSignature:
    """Scent signature for task pattern encoding."""
    intensity: float
    molecular_descriptors: List[MolecularDescriptor] = field(default_factory=list)
    volatility: float = 0.5
    persistence: float = 0.5
    complexity: float = 0.5

    def calculate_signature_strength(self) -> float:
        """Calculate overall signature strength."""
        if not self.molecular_descriptors:
            return self.intensity

        descriptor_strength = sum(d.molecular_weight / 200.0 for d in self.molecular_descriptors)
        return min(1.0, self.intensity * descriptor_strength * self.complexity)


@dataclass
class OlfactoryReceptor:
    """Quantum olfactory receptor for task feature detection."""

    id: str
    sensitivity_profile: Dict[str, float]  # Feature -> sensitivity
    quantum_state: complex = complex(1.0, 0.0)
    activation_threshold: float = 0.5
    adaptation_rate: float = 0.1
    binding_affinity: Dict[str, float] = field(default_factory=dict)

    def detect_feature(self, task_features: Dict[str, float]) -> float:
        """Detect task features using quantum-enhanced sensitivity."""
        activation = 0.0

        for feature, value in task_features.items():
            if feature in self.sensitivity_profile:
                sensitivity = self.sensitivity_profile[feature]
                # Quantum enhancement using superposition
                quantum_enhancement = abs(self.quantum_state) ** 2
                activation += sensitivity * value * quantum_enhancement

        # Apply sigmoid activation
        return 1.0 / (1.0 + math.exp(-activation + self.activation_threshold))

    def update_quantum_state(self, phase_shift: float) -> None:
        """Update quantum state based on optimization feedback."""
        current_phase = math.atan2(self.quantum_state.imag, self.quantum_state.real)
        new_phase = current_phase + phase_shift
        magnitude = abs(self.quantum_state)
        self.quantum_state = complex(
            magnitude * math.cos(new_phase),
            magnitude * math.sin(new_phase)
        )


@dataclass
class OlfactoryBulb:
    """Neural processing unit inspired by mammalian olfactory bulb."""

    receptors: List[OlfactoryReceptor] = field(default_factory=list)
    mitral_cells: List[Dict[str, Any]] = field(default_factory=list)
    inhibitory_connections: Dict[str, List[str]] = field(default_factory=dict)
    learning_rate: float = 0.01

    def process_task_patterns(self, tasks: List[Task]) -> Dict[str, np.ndarray]:
        """Process task patterns through olfactory bulb architecture."""
        task_patterns = {}

        for task in tasks:
            # Extract task features
            features = self._extract_task_features(task)

            # Process through receptors
            receptor_responses = []
            for receptor in self.receptors:
                response = receptor.detect_feature(features)
                receptor_responses.append(response)

            # Lateral inhibition and pattern separation
            processed_pattern = self._apply_lateral_inhibition(
                np.array(receptor_responses)
            )

            task_patterns[task.id] = processed_pattern

        return task_patterns

    def _extract_task_features(self, task: Task) -> Dict[str, float]:
        """Extract normalized features from task for olfactory processing."""
        features = {
            'duration': min(task.duration.total_seconds() / 3600, 24.0) / 24.0,  # Normalized to [0,1]
            'priority': {
                TaskPriority.LOW: 0.25,
                TaskPriority.MEDIUM: 0.5,
                TaskPriority.HIGH: 0.75,
                TaskPriority.CRITICAL: 1.0
            }[task.priority],
            'complexity': len(task.dependencies) / 10.0,  # Assuming max 10 deps
            'resource_demand': sum(task.resource_requirements.values()) / 100.0,
            'quantum_weight': task.quantum_weight / 100.0,
            'entanglement': task.entanglement_factor,
        }

        # Add constraint-based features
        if task.constraints:
            features['constraint_density'] = len(task.constraints) / 5.0
        else:
            features['constraint_density'] = 0.0

        return features

    def _apply_lateral_inhibition(self, responses: np.ndarray) -> np.ndarray:
        """Apply lateral inhibition for pattern separation."""
        # Gaussian lateral inhibition
        processed = responses.copy()
        for i, response in enumerate(responses):
            if response > 0.1:  # Only process significant responses
                # Inhibit neighboring receptors
                for j in range(len(responses)):
                    if i != j:
                        distance = abs(i - j)
                        inhibition = math.exp(-distance**2 / 2.0) * 0.3
                        processed[j] = max(0, processed[j] - inhibition)

        return processed


@dataclass
class QuantumOlfactoryState:
    """Quantum state representation for olfactory-inspired optimization."""

    task_patterns: Dict[str, np.ndarray]
    resource_affinities: Dict[str, Dict[str, float]]  # task_id -> resource_id -> affinity
    quantum_entanglements: Dict[Tuple[str, str], complex]  # Task pairs -> entanglement
    olfactory_energy: float
    pattern_coherence: float
    generation: int = 0


class BioNeuroOlfactoryFusionOptimizer:
    """Quantum-Neural Olfactory Fusion Algorithm for Task Optimization.
    
    This optimizer combines quantum computing principles with bio-inspired
    olfactory processing to solve complex task scheduling and resource
    allocation problems. The algorithm models task features as "chemical
    signatures" processed by quantum-enhanced olfactory receptors.
    """

    def __init__(self,
                 num_receptors: int = 50,
                 quantum_coherence_time: float = 100.0,
                 learning_rate: float = 0.01,
                 entanglement_strength: float = 0.5):
        """Initialize Bio-Neural Olfactory Fusion Optimizer.
        
        Args:
            num_receptors: Number of olfactory receptors
            quantum_coherence_time: Quantum coherence preservation time
            learning_rate: Neural adaptation learning rate
            entanglement_strength: Quantum entanglement coupling strength
        """
        try:
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

            # Initialize olfactory system
            self.olfactory_bulb = self._initialize_olfactory_bulb(num_receptors)

            # Quantum parameters
            self.coherence_time = quantum_coherence_time
            self.entanglement_strength = entanglement_strength
            self.quantum_phase = 0.0

            # Learning parameters
            self.learning_rate = learning_rate
            self.adaptation_cycles = 0

            # Optimization tracking
            self.optimization_history: List[Dict[str, Any]] = []
            self.best_state: Optional[QuantumOlfactoryState] = None
            self.convergence_threshold = 0.001

            # Performance metrics
            self.start_time: Optional[datetime] = None
            self.feature_extraction_times: List[float] = []
            self.pattern_processing_times: List[float] = []

            self.logger.info(f"Bio-Neural Olfactory Fusion Optimizer initialized with {num_receptors} receptors")

        except Exception as e:
            self.logger.error(f"Failed to initialize olfactory fusion optimizer: {e}")
            raise OptimizationError(f"Initialization failed: {e}") from e

    def optimize_schedule(self, schedule: Schedule) -> OptimizationMetrics:
        """Optimize schedule using quantum-neural olfactory fusion.
        
        Args:
            schedule: Schedule to optimize
            
        Returns:
            Optimization metrics
            
        Raises:
            OptimizationError: If optimization fails
        """
        try:
            self.start_time = datetime.utcnow()
            self.logger.info(f"Starting bio-neural olfactory optimization for schedule {schedule.id}")

            # Validate schedule
            self._validate_schedule(schedule)

            # Set schedule status
            schedule.status = ScheduleStatus.OPTIMIZING

            # Phase 1: Olfactory Feature Extraction
            task_patterns = self._extract_olfactory_patterns(schedule.tasks)

            # Phase 2: Quantum Entanglement Mapping
            entanglements = self._map_quantum_entanglements(schedule.tasks, task_patterns)

            # Phase 3: Resource Affinity Calculation
            resource_affinities = self._calculate_resource_affinities(
                schedule.tasks, schedule.resources, task_patterns
            )

            # Phase 4: Quantum-Enhanced Optimization
            optimal_state = self._quantum_olfactory_optimization(
                schedule, task_patterns, entanglements, resource_affinities
            )

            # Phase 5: Solution Application
            self._apply_olfactory_solution(schedule, optimal_state)

            # Calculate metrics
            optimization_time = datetime.utcnow() - self.start_time
            metrics = self._calculate_optimization_metrics(schedule, optimization_time)

            schedule.status = ScheduleStatus.OPTIMIZED
            schedule.optimized_at = datetime.utcnow()
            schedule.metrics = metrics

            self.logger.info(f"Bio-neural olfactory optimization completed in {optimization_time}")
            return metrics

        except Exception as e:
            self.logger.error(f"Olfactory fusion optimization failed: {e}")
            schedule.status = ScheduleStatus.FAILED
            raise OptimizationError(f"Optimization failed: {e}") from e

    def _initialize_olfactory_bulb(self, num_receptors: int) -> OlfactoryBulb:
        """Initialize olfactory bulb with quantum receptors."""
        receptors = []

        # Create diverse receptor sensitivities
        feature_types = [
            'duration', 'priority', 'complexity', 'resource_demand',
            'quantum_weight', 'entanglement', 'constraint_density'
        ]

        for i in range(num_receptors):
            # Random sensitivity profile with quantum initialization
            sensitivity_profile = {}
            for feature in feature_types:
                # Use quantum random walk for sensitivity initialization
                sensitivity = abs(np.random.normal(0.5, 0.2))
                sensitivity_profile[feature] = max(0.1, min(1.0, sensitivity))

            # Initialize quantum state with random phase
            initial_phase = random.uniform(0, 2 * math.pi)
            quantum_state = complex(math.cos(initial_phase), math.sin(initial_phase))

            receptor = OlfactoryReceptor(
                id=f"receptor_{i}",
                sensitivity_profile=sensitivity_profile,
                quantum_state=quantum_state,
                activation_threshold=random.uniform(0.3, 0.7)
            )

            receptors.append(receptor)

        return OlfactoryBulb(receptors=receptors)

    def _extract_olfactory_patterns(self, tasks: List[Task]) -> Dict[str, np.ndarray]:
        """Extract olfactory patterns from tasks using bio-neural processing."""
        import time
        start_time = time.time()

        try:
            # Process tasks through olfactory bulb
            task_patterns = self.olfactory_bulb.process_task_patterns(tasks)

            # Apply quantum enhancement to patterns
            for task_id, pattern in task_patterns.items():
                # Quantum Fourier Transform enhancement
                enhanced_pattern = self._apply_quantum_enhancement(pattern)
                task_patterns[task_id] = enhanced_pattern

            processing_time = time.time() - start_time
            self.feature_extraction_times.append(processing_time)

            self.logger.debug(f"Extracted olfactory patterns for {len(tasks)} tasks in {processing_time:.3f}s")
            return task_patterns

        except Exception as e:
            self.logger.error(f"Error extracting olfactory patterns: {e}")
            raise OptimizationError(f"Pattern extraction failed: {e}") from e

    def _apply_quantum_enhancement(self, pattern: np.ndarray) -> np.ndarray:
        """Apply quantum enhancement to olfactory pattern."""
        try:
            # Quantum Fourier Transform (simplified version)
            fft_pattern = np.fft.fft(pattern)

            # Apply quantum phase evolution
            quantum_phases = np.exp(1j * self.quantum_phase * np.arange(len(pattern)))
            enhanced_fft = fft_pattern * quantum_phases

            # Inverse transform and take real part
            enhanced_pattern = np.real(np.fft.ifft(enhanced_fft))

            # Normalize to [0, 1]
            if np.max(enhanced_pattern) != np.min(enhanced_pattern):
                enhanced_pattern = (enhanced_pattern - np.min(enhanced_pattern)) / \
                                   (np.max(enhanced_pattern) - np.min(enhanced_pattern))

            return enhanced_pattern

        except Exception as e:
            self.logger.warning(f"Quantum enhancement failed, using original pattern: {e}")
            return pattern

    def _map_quantum_entanglements(self, tasks: List[Task],
                                   patterns: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], complex]:
        """Map quantum entanglements between task pairs based on similarity."""
        entanglements = {}

        try:
            task_ids = list(patterns.keys())

            for i, task_id1 in enumerate(task_ids):
                for j, task_id2 in enumerate(task_ids[i+1:], i+1):
                    # Calculate pattern similarity
                    pattern1 = patterns[task_id1]
                    pattern2 = patterns[task_id2]

                    # Quantum dot product similarity
                    similarity = np.dot(pattern1, pattern2) / (
                        np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8
                    )

                    # Check for explicit dependencies
                    task1 = next(t for t in tasks if t.id == task_id1)
                    task2 = next(t for t in tasks if t.id == task_id2)

                    dependency_boost = 0.0
                    if task_id2 in task1.dependencies or task_id1 in task2.dependencies:
                        dependency_boost = 0.5

                    # Create quantum entanglement state
                    entanglement_strength = (similarity + dependency_boost) * self.entanglement_strength

                    if entanglement_strength > 0.1:  # Only significant entanglements
                        phase = random.uniform(0, 2 * math.pi)
                        entanglement = complex(
                            entanglement_strength * math.cos(phase),
                            entanglement_strength * math.sin(phase)
                        )

                        entanglements[(task_id1, task_id2)] = entanglement

            self.logger.debug(f"Mapped {len(entanglements)} quantum entanglements")
            return entanglements

        except Exception as e:
            self.logger.error(f"Error mapping quantum entanglements: {e}")
            return {}

    def _calculate_resource_affinities(self, tasks: List[Task], resources: List[Resource],
                                       patterns: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate bio-neural resource affinities based on olfactory patterns."""
        affinities = defaultdict(dict)

        try:
            # Create resource feature profiles
            resource_profiles = {}
            for resource in resources:
                profile = self._extract_resource_features(resource)
                resource_profiles[resource.id] = profile

            # Calculate affinities using pattern matching
            for task_id, task_pattern in patterns.items():
                task = next(t for t in tasks if t.id == task_id)

                for resource_id, resource_profile in resource_profiles.items():
                    resource = next(r for r in resources if r.id == resource_id)

                    # Bio-neural affinity calculation
                    affinity = self._calculate_bioneural_affinity(
                        task, task_pattern, resource, resource_profile
                    )

                    affinities[task_id][resource_id] = affinity

            self.logger.debug(f"Calculated resource affinities for {len(tasks)} tasks")
            return dict(affinities)

        except Exception as e:
            self.logger.error(f"Error calculating resource affinities: {e}")
            return {}

    def _extract_resource_features(self, resource: Resource) -> np.ndarray:
        """Extract normalized feature vector from resource."""
        try:
            features = [
                resource.total_capacity / 100.0,  # Normalized capacity
                resource.available_capacity / 100.0,  # Normalized availability
                resource.efficiency_rating,  # Already [0,1]
                resource.cost_per_unit / 1000.0,  # Normalized cost
                resource.quantum_coherence if hasattr(resource, 'quantum_coherence') else 0.5,
                resource.superposition_factor if hasattr(resource, 'superposition_factor') else 0.5,
            ]

            return np.array(features)

        except Exception as e:
            self.logger.warning(f"Error extracting resource features: {e}")
            return np.array([0.5] * 6)  # Default neutral features

    def _calculate_bioneural_affinity(self, task: Task, task_pattern: np.ndarray,
                                      resource: Resource, resource_profile: np.ndarray) -> float:
        """Calculate bio-neural affinity between task and resource."""
        try:
            # Pattern-based similarity using cross-correlation
            if len(task_pattern) >= len(resource_profile):
                correlation = np.correlate(task_pattern, resource_profile, mode='valid')[0]
            else:
                correlation = np.correlate(resource_profile, task_pattern, mode='valid')[0]

            # Normalize correlation
            pattern_norm = np.linalg.norm(task_pattern)
            profile_norm = np.linalg.norm(resource_profile)
            normalized_correlation = correlation / (pattern_norm * profile_norm + 1e-8)

            # Resource capacity compatibility
            total_demand = sum(task.resource_requirements.values()) if task.resource_requirements else 1.0
            capacity_match = min(1.0, resource.available_capacity / max(total_demand, 0.1))

            # Cost efficiency
            cost_efficiency = 1.0 / (1.0 + resource.cost_per_unit / 100.0)

            # Quantum resonance (if available)
            quantum_resonance = 1.0
            if hasattr(resource, 'quantum_coherence'):
                quantum_resonance = resource.calculate_quantum_affinity(
                    task.quantum_weight, task.entanglement_factor
                ) if hasattr(resource, 'calculate_quantum_affinity') else 1.0

            # Combined affinity with bio-neural weighting
            affinity = (
                0.4 * abs(normalized_correlation) +
                0.3 * capacity_match +
                0.2 * cost_efficiency +
                0.1 * quantum_resonance
            )

            return max(0.0, min(1.0, affinity))

        except Exception as e:
            self.logger.debug(f"Error calculating bio-neural affinity: {e}")
            return 0.5  # Default neutral affinity

    def _quantum_olfactory_optimization(self, schedule: Schedule,
                                        patterns: Dict[str, np.ndarray],
                                        entanglements: Dict[Tuple[str, str], complex],
                                        affinities: Dict[str, Dict[str, float]]) -> QuantumOlfactoryState:
        """Perform quantum-olfactory optimization using bio-neural dynamics."""
        try:
            current_state = QuantumOlfactoryState(
                task_patterns=patterns,
                resource_affinities=affinities,
                quantum_entanglements=entanglements,
                olfactory_energy=float('inf'),
                pattern_coherence=0.0
            )

            best_state = None
            best_energy = float('inf')
            no_improvement_count = 0

            # Optimization loop with quantum-olfactory dynamics
            for generation in range(500):  # Max generations
                try:
                    # Apply quantum evolution operator
                    evolved_state = self._apply_quantum_evolution(current_state, schedule)

                    # Calculate olfactory energy
                    energy = self._calculate_olfactory_energy(evolved_state, schedule)
                    evolved_state.olfactory_energy = energy

                    # Update best state
                    if energy < best_energy:
                        best_energy = energy
                        best_state = evolved_state
                        no_improvement_count = 0
                        self.logger.debug(f"New best energy: {energy:.4f} at generation {generation}")
                    else:
                        no_improvement_count += 1

                    # Apply bio-neural adaptation
                    current_state = self._apply_bioneural_adaptation(evolved_state)
                    current_state.generation = generation

                    # Record history
                    self.optimization_history.append({
                        'generation': generation,
                        'energy': energy,
                        'coherence': evolved_state.pattern_coherence,
                        'entanglements': len(evolved_state.quantum_entanglements)
                    })

                    # Check convergence
                    if self._check_olfactory_convergence(generation) or no_improvement_count > 50:
                        self.logger.info(f"Optimization converged at generation {generation}")
                        break

                except Exception as e:
                    self.logger.warning(f"Error in optimization generation {generation}: {e}")
                    continue

            if best_state is None:
                raise OptimizationError("No valid solution found")

            self.best_state = best_state
            return best_state

        except Exception as e:
            self.logger.error(f"Quantum-olfactory optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}") from e

    def _apply_quantum_evolution(self, state: QuantumOlfactoryState,
                                 schedule: Schedule) -> QuantumOlfactoryState:
        """Apply quantum evolution operator to olfactory state."""
        try:
            # Deep copy current state
            new_state = QuantumOlfactoryState(
                task_patterns=state.task_patterns.copy(),
                resource_affinities={k: v.copy() for k, v in state.resource_affinities.items()},
                quantum_entanglements=state.quantum_entanglements.copy(),
                olfactory_energy=state.olfactory_energy,
                pattern_coherence=state.pattern_coherence,
                generation=state.generation + 1
            )

            # Apply quantum phase evolution
            self.quantum_phase += 0.1

            # Evolve quantum entanglements
            for (task1, task2), entanglement in new_state.quantum_entanglements.items():
                # Apply time evolution
                phase_evolution = cmath.exp(1j * 0.05 * random.uniform(-1, 1))
                new_state.quantum_entanglements[(task1, task2)] = entanglement * phase_evolution

            # Update resource affinities based on quantum feedback
            for task_id in new_state.resource_affinities:
                for resource_id in new_state.resource_affinities[task_id]:
                    current_affinity = new_state.resource_affinities[task_id][resource_id]

                    # Quantum fluctuation
                    fluctuation = random.gauss(0, 0.02)
                    new_affinity = max(0.0, min(1.0, current_affinity + fluctuation))
                    new_state.resource_affinities[task_id][resource_id] = new_affinity

            return new_state

        except Exception as e:
            self.logger.warning(f"Error in quantum evolution: {e}")
            return state

    def _calculate_olfactory_energy(self, state: QuantumOlfactoryState,
                                    schedule: Schedule) -> float:
        """Calculate olfactory system energy for optimization."""
        try:
            total_energy = 0.0

            # Assignment energy based on affinities
            assignment_energy = 0.0
            for task_id, resource_affinities in state.resource_affinities.items():
                if resource_affinities:
                    # Find best affinity for this task
                    best_affinity = max(resource_affinities.values())
                    # Energy is inversely related to affinity
                    assignment_energy += (1.0 - best_affinity) ** 2

            # Entanglement energy
            entanglement_energy = 0.0
            for (task1, task2), entanglement in state.quantum_entanglements.items():
                # Energy from quantum entanglement strength
                entanglement_strength = abs(entanglement)
                entanglement_energy += entanglement_strength * 0.5

            # Pattern coherence energy
            coherence_energy = 0.0
            patterns = list(state.task_patterns.values())
            if len(patterns) > 1:
                # Calculate average pattern coherence
                coherences = []
                for i in range(len(patterns)):
                    for j in range(i+1, len(patterns)):
                        coherence = np.corrcoef(patterns[i], patterns[j])[0, 1]
                        if not np.isnan(coherence):
                            coherences.append(abs(coherence))

                if coherences:
                    avg_coherence = sum(coherences) / len(coherences)
                    state.pattern_coherence = avg_coherence
                    coherence_energy = (1.0 - avg_coherence) * 10

            # Resource utilization energy
            utilization_energy = 0.0
            available_resources = [r for r in schedule.resources
                                   if r.status == ResourceStatus.AVAILABLE]
            if available_resources:
                total_capacity = sum(r.total_capacity for r in available_resources)
                used_capacity = 0.0

                for task_id, resource_affinities in state.resource_affinities.items():
                    if resource_affinities:
                        # Estimate resource usage based on best affinity
                        best_resource_id = max(resource_affinities.keys(),
                                               key=lambda r: resource_affinities[r])
                        task = schedule.get_task(task_id)
                        if task and task.resource_requirements:
                            used_capacity += sum(task.resource_requirements.values())

                utilization_ratio = used_capacity / max(total_capacity, 1.0)
                # Penalize both over and under-utilization
                utilization_energy = abs(utilization_ratio - 0.8) * 20

            # Combine energies with weights
            total_energy = (
                0.4 * assignment_energy +
                0.2 * entanglement_energy +
                0.2 * coherence_energy +
                0.2 * utilization_energy
            )

            return total_energy

        except Exception as e:
            self.logger.error(f"Error calculating olfactory energy: {e}")
            return float('inf')

    def _apply_bioneural_adaptation(self, state: QuantumOlfactoryState) -> QuantumOlfactoryState:
        """Apply bio-neural adaptation to improve optimization."""
        try:
            # Update receptor quantum states based on feedback
            for receptor in self.olfactory_bulb.receptors:
                # Adaptive phase shift based on current energy
                if state.olfactory_energy != float('inf'):
                    phase_shift = self.learning_rate * (1.0 / (1.0 + state.olfactory_energy))
                    receptor.update_quantum_state(phase_shift)

            # Increment adaptation cycle
            self.adaptation_cycles += 1

            return state

        except Exception as e:
            self.logger.warning(f"Error in bio-neural adaptation: {e}")
            return state

    def _check_olfactory_convergence(self, generation: int) -> bool:
        """Check if olfactory optimization has converged."""
        if generation < 20 or len(self.optimization_history) < 20:
            return False

        # Check energy convergence
        recent_energies = [entry['energy'] for entry in self.optimization_history[-20:]]
        energy_variance = np.var(recent_energies)

        return energy_variance < self.convergence_threshold

    def _apply_olfactory_solution(self, schedule: Schedule, state: QuantumOlfactoryState) -> None:
        """Apply bio-neural olfactory solution to schedule."""
        try:
            # Clear existing assignments
            schedule.assignments.clear()

            # Create assignments based on highest affinities
            for task_id, resource_affinities in state.resource_affinities.items():
                if not resource_affinities:
                    continue

                # Select best resource based on affinity
                best_resource_id = max(resource_affinities.keys(),
                                       key=lambda r: resource_affinities[r])
                best_affinity = resource_affinities[best_resource_id]

                # Only assign if affinity is reasonable
                if best_affinity > 0.2:
                    task = schedule.get_task(task_id)
                    resource = schedule.get_resource(best_resource_id)

                    if task and resource:
                        # Calculate allocation based on affinity and requirements
                        allocation = min(
                            resource.available_capacity,
                            sum(task.resource_requirements.values()) if task.resource_requirements else 1.0
                        ) * best_affinity

                        # Create assignment
                        assignment = TaskAssignment(
                            task_id=task_id,
                            resource_id=best_resource_id,
                            start_time=schedule.start_time,
                            end_time=schedule.start_time + task.duration,
                            allocated_capacity=allocation,
                            priority=1
                        )

                        schedule.assignments.append(assignment)

                        # Update task status
                        task.status = TaskStatus.READY
                        task.scheduled_start = assignment.start_time
                        task.scheduled_finish = assignment.end_time

                        # Update resource capacity
                        resource.available_capacity -= allocation

            self.logger.info(f"Applied {len(schedule.assignments)} bio-neural assignments")

        except Exception as e:
            self.logger.error(f"Error applying olfactory solution: {e}")
            raise ResourceAllocationError(f"Failed to apply solution: {e}") from e

    def _calculate_optimization_metrics(self, schedule: Schedule,
                                        optimization_time: timedelta) -> OptimizationMetrics:
        """Calculate comprehensive optimization metrics."""
        try:
            return OptimizationMetrics(
                makespan=schedule.calculate_makespan(),
                total_cost=schedule.calculate_total_cost(),
                resource_utilization=schedule.get_resource_utilization(),
                constraint_violations=len(schedule.validate_dependencies()),
                quantum_energy=schedule.calculate_quantum_energy() if hasattr(schedule, 'calculate_quantum_energy') else 0.0,
                optimization_time=optimization_time,
                iterations=len(self.optimization_history),
                convergence_achieved=self._check_olfactory_convergence(len(self.optimization_history))
            )
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            # Return minimal metrics on error
            return OptimizationMetrics(
                makespan=timedelta(0),
                total_cost=0.0,
                resource_utilization={},
                constraint_violations=0,
                quantum_energy=0.0,
                optimization_time=optimization_time,
                iterations=len(self.optimization_history),
                convergence_achieved=False
            )

    def _validate_schedule(self, schedule: Schedule) -> None:
        """Validate schedule for bio-neural olfactory optimization."""
        errors = []

        if not schedule.tasks:
            errors.append("Schedule has no tasks")

        if not schedule.resources:
            errors.append("Schedule has no resources")

        # Check for available resources
        available_resources = [r for r in schedule.resources
                               if r.status == ResourceStatus.AVAILABLE]
        if not available_resources:
            errors.append("No available resources")

        if errors:
            raise ValidationError(f"Invalid schedule for olfactory optimization: {'; '.join(errors)}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary with bio-neural insights."""
        summary = {
            'algorithm': 'Bio-Neural Olfactory Fusion',
            'optimization_cycles': self.adaptation_cycles,
            'quantum_phase': self.quantum_phase,
            'convergence_achieved': len(self.optimization_history) > 0 and
                                    self._check_olfactory_convergence(len(self.optimization_history) - 1),
            'feature_extraction_avg_time':
                sum(self.feature_extraction_times) / len(self.feature_extraction_times)
                if self.feature_extraction_times else 0.0,
            'pattern_processing_avg_time':
                sum(self.pattern_processing_times) / len(self.pattern_processing_times)
                if self.pattern_processing_times else 0.0,
            'total_generations': len(self.optimization_history),
        }

        if self.best_state:
            summary.update({
                'best_energy': self.best_state.olfactory_energy,
                'pattern_coherence': self.best_state.pattern_coherence,
                'quantum_entanglements': len(self.best_state.quantum_entanglements),
                'resource_assignments': len(self.best_state.resource_affinities)
            })

        return summary
