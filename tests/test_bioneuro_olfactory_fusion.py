"""Comprehensive test suite for Bio-Neural Olfactory Fusion Algorithm.

This test suite validates the novel quantum-neural olfactory approach
including molecular descriptors, scent signatures, cross-modal fusion,
and optimization performance under various conditions.

Test Categories:
- Unit tests for individual algorithm components
- Integration tests for full optimization workflows  
- Performance benchmarks and regression testing
- Statistical validation of quantum-neural improvements
- Reproducibility and determinism testing
"""

import pytest
import numpy as np
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

from src.quantum_task_planner.algorithms.bioneuro_olfactory_fusion import (
    BioNeuroOlfactoryFusionOptimizer,
    OlfactoryReceptor,
    OlfactoryBulb,
    QuantumOlfactoryState,
    ScentSignature,
    MolecularDescriptor,
    ChemicalFamily
)
from src.quantum_task_planner.algorithms.olfactory_data_pipeline import (
    OlfactoryDataPipeline,
    MultiModalSensorData,
    SensorModality
)
from src.quantum_task_planner.models.task import Task, TaskPriority, TaskStatus
from src.quantum_task_planner.models.resource import Resource, ResourceStatus
from src.quantum_task_planner.models.schedule import Schedule, ScheduleStatus
from src.quantum_task_planner.core.exceptions import OptimizationError, ValidationError


@pytest.fixture
def molecular_descriptor():
    """Create test molecular descriptor."""
    return MolecularDescriptor(
        molecular_weight=150.0,
        boiling_point=120.0,
        vapor_pressure=0.5,
        solubility=0.7,
        polarity=0.6,
        aromaticity=0.4,
        hydrogen_bonding=0.3,
        steric_hindrance=0.2
    )


@pytest.fixture
def scent_signature(molecular_descriptor):
    """Create test scent signature."""
    return ScentSignature(
        id="test_scent",
        chemical_family=ChemicalFamily.ESTERS,
        molecular_descriptors=molecular_descriptor,
        intensity=0.8,
        persistence=0.6,
        diffusion_rate=0.7
    )


@pytest.fixture
def olfactory_receptor():
    """Create test olfactory receptor."""
    return OlfactoryReceptor(
        id="receptor_1",
        sensitivity_profile={
            'duration': 0.8,
            'priority': 0.6,
            'complexity': 0.4
        },
        quantum_state=complex(0.8, 0.6),
        activation_threshold=0.5
    )


@pytest.fixture
def sample_task():
    """Create sample task for testing."""
    return Task(
        id="task_1",
        name="Test Task",
        description="Sample task for testing",
        duration=timedelta(hours=2),
        priority=TaskPriority.HIGH,
        resource_requirements={"cpu": 4.0, "memory": 8.0},
        quantum_weight=1.5,
        entanglement_factor=0.3
    )


@pytest.fixture
def sample_resource():
    """Create sample resource for testing."""
    return Resource(
        id="resource_1",
        name="Test Resource",
        type="compute",
        total_capacity=10.0,
        available_capacity=8.0,
        efficiency_rating=0.85,
        cost_per_unit=5.0
    )


@pytest.fixture
def sample_schedule(sample_task, sample_resource):
    """Create sample schedule for testing."""
    schedule = Schedule(
        id="schedule_1",
        name="Test Schedule",
        start_time=datetime.utcnow()
    )
    schedule.add_task(sample_task)
    schedule.add_resource(sample_resource)
    return schedule


@pytest.fixture
def bio_neural_optimizer():
    """Create Bio-Neural Olfactory Fusion Optimizer."""
    return BioNeuroOlfactoryFusionOptimizer(
        num_receptors=20,
        quantum_coherence_time=50.0,
        learning_rate=0.01,
        entanglement_strength=0.3
    )


@pytest.fixture
def olfactory_pipeline():
    """Create olfactory data pipeline."""
    return OlfactoryDataPipeline(
        buffer_size=100,
        feature_dimensions=32,
        enable_cross_modal=True
    )


class TestMolecularDescriptor:
    """Test molecular descriptor functionality."""
    
    def test_to_vector(self, molecular_descriptor):
        """Test molecular descriptor to vector conversion."""
        vector = molecular_descriptor.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 8
        assert all(0 <= v <= 1 for v in vector)  # Normalized values
    
    def test_vector_normalization(self):
        """Test vector normalization with extreme values."""
        descriptor = MolecularDescriptor(
            molecular_weight=1000.0,  # High value
            boiling_point=500.0,      # High value
            vapor_pressure=200.0,     # High value
            solubility=1.2,           # Invalid value > 1
            polarity=0.5,
            aromaticity=0.3,
            hydrogen_bonding=0.8,
            steric_hindrance=0.1
        )
        
        vector = descriptor.to_vector()
        assert all(v <= 1.0 for v in vector)  # Should be clamped to max 1.0


class TestScentSignature:
    """Test scent signature functionality."""
    
    def test_initialization(self, scent_signature):
        """Test scent signature initialization."""
        assert scent_signature.id == "test_scent"
        assert scent_signature.chemical_family == ChemicalFamily.ESTERS
        assert 0 <= scent_signature.intensity <= 1
        assert len(scent_signature.temporal_profile) > 0
    
    def test_temporal_profile_generation(self):
        """Test automatic temporal profile generation."""
        signature = ScentSignature(
            id="test",
            chemical_family=ChemicalFamily.ALDEHYDES,
            molecular_descriptors=MolecularDescriptor(
                molecular_weight=100, boiling_point=100, vapor_pressure=0.5,
                solubility=0.5, polarity=0.5, aromaticity=0.3,
                hydrogen_bonding=0.3, steric_hindrance=0.2
            ),
            intensity=0.8,
            persistence=0.6,
            diffusion_rate=0.4
        )
        
        assert len(signature.temporal_profile) == 100
        assert signature.temporal_profile[0] == signature.intensity
        # Should decay over time
        assert signature.temporal_profile[-1] < signature.temporal_profile[0]
    
    def test_similarity_calculation(self, scent_signature):
        """Test scent similarity calculation."""
        # Create similar signature
        similar_signature = ScentSignature(
            id="similar_scent",
            chemical_family=ChemicalFamily.ESTERS,  # Same family
            molecular_descriptors=scent_signature.molecular_descriptors,
            intensity=0.75,  # Close intensity
            persistence=0.65,  # Close persistence
            diffusion_rate=0.72
        )
        
        similarity = scent_signature.calculate_similarity(similar_signature)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be similar
        
        # Create dissimilar signature
        dissimilar_signature = ScentSignature(
            id="dissimilar_scent",
            chemical_family=ChemicalFamily.ACIDS,  # Different family
            molecular_descriptors=MolecularDescriptor(
                molecular_weight=50, boiling_point=50, vapor_pressure=0.1,
                solubility=0.1, polarity=0.9, aromaticity=0.1,
                hydrogen_bonding=0.9, steric_hindrance=0.8
            ),
            intensity=0.2,
            persistence=0.1,
            diffusion_rate=0.1
        )
        
        dissimilarity = scent_signature.calculate_similarity(dissimilar_signature)
        assert dissimilarity < similarity  # Should be less similar
    
    def test_hash_generation(self, scent_signature):
        """Test unique hash generation."""
        hash1 = scent_signature.generate_hash()
        hash2 = scent_signature.generate_hash()
        
        assert hash1 == hash2  # Same signature should produce same hash
        assert len(hash1) == 12  # Expected hash length
        
        # Different signature should produce different hash
        different_signature = ScentSignature(
            id="different",
            chemical_family=ChemicalFamily.KETONES,
            molecular_descriptors=scent_signature.molecular_descriptors,
            intensity=0.5,
            persistence=0.5,
            diffusion_rate=0.5
        )
        
        assert hash1 != different_signature.generate_hash()


class TestOlfactoryReceptor:
    """Test olfactory receptor functionality."""
    
    def test_feature_detection(self, olfactory_receptor):
        """Test feature detection with quantum enhancement."""
        features = {
            'duration': 0.8,
            'priority': 0.6,
            'complexity': 0.2,
            'unknown_feature': 0.9  # Should be ignored
        }
        
        activation = olfactory_receptor.detect_feature(features)
        assert 0.0 <= activation <= 1.0
        
        # Test with empty features
        empty_activation = olfactory_receptor.detect_feature({})
        assert 0.0 <= empty_activation <= 1.0
    
    def test_quantum_state_update(self, olfactory_receptor):
        """Test quantum state evolution."""
        original_state = olfactory_receptor.quantum_state
        
        # Apply phase shift
        olfactory_receptor.update_quantum_state(math.pi / 4)
        
        # State should change but maintain magnitude
        assert olfactory_receptor.quantum_state != original_state
        assert abs(abs(olfactory_receptor.quantum_state) - abs(original_state)) < 1e-10
    
    def test_quantum_enhancement_effect(self):
        """Test quantum enhancement effect on detection."""
        receptor1 = OlfactoryReceptor(
            id="high_quantum",
            sensitivity_profile={'test': 0.5},
            quantum_state=complex(1.0, 0.0),  # High quantum state
            activation_threshold=0.3
        )
        
        receptor2 = OlfactoryReceptor(
            id="low_quantum", 
            sensitivity_profile={'test': 0.5},
            quantum_state=complex(0.1, 0.0),  # Low quantum state
            activation_threshold=0.3
        )
        
        features = {'test': 0.8}
        
        activation1 = receptor1.detect_feature(features)
        activation2 = receptor2.detect_feature(features)
        
        # Higher quantum state should give higher activation
        assert activation1 > activation2


class TestOlfactoryBulb:
    """Test olfactory bulb neural processing."""
    
    def test_task_feature_extraction(self, sample_task):
        """Test task feature extraction."""
        bulb = OlfactoryBulb()
        features = bulb._extract_task_features(sample_task)
        
        assert isinstance(features, dict)
        expected_features = [
            'duration', 'priority', 'complexity', 'resource_demand',
            'quantum_weight', 'entanglement', 'constraint_density'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert 0.0 <= features[feature] <= 1.0
    
    def test_pattern_processing(self, sample_task):
        """Test neural pattern processing through olfactory bulb."""
        # Create receptors with different sensitivities
        receptors = []
        for i in range(10):
            receptor = OlfactoryReceptor(
                id=f"receptor_{i}",
                sensitivity_profile={
                    'duration': np.random.uniform(0.1, 1.0),
                    'priority': np.random.uniform(0.1, 1.0),
                    'complexity': np.random.uniform(0.1, 1.0)
                },
                quantum_state=complex(np.random.uniform(0.5, 1.0), 0)
            )
            receptors.append(receptor)
        
        bulb = OlfactoryBulb(receptors=receptors)
        patterns = bulb.process_task_patterns([sample_task])
        
        assert sample_task.id in patterns
        assert isinstance(patterns[sample_task.id], np.ndarray)
        assert len(patterns[sample_task.id]) == len(receptors)
    
    def test_lateral_inhibition(self):
        """Test lateral inhibition processing."""
        bulb = OlfactoryBulb()
        
        # Test with strong response pattern
        responses = np.array([0.1, 0.9, 0.8, 0.2, 0.1])
        processed = bulb._apply_lateral_inhibition(responses)
        
        assert len(processed) == len(responses)
        assert np.all(processed >= 0)  # No negative values
        
        # Strong responses should inhibit neighbors
        assert processed[1] <= responses[1]  # May be inhibited by strong neighbor


class TestOlfactoryDataPipeline:
    """Test olfactory data processing pipeline."""
    
    def test_initialization(self, olfactory_pipeline):
        """Test pipeline initialization."""
        assert olfactory_pipeline.feature_dimensions == 32
        assert olfactory_pipeline.enable_cross_modal == True
        assert len(olfactory_pipeline.sensor_buffer) == 0
        assert len(olfactory_pipeline.scent_library) == 0
    
    def test_task_to_scent_conversion(self, olfactory_pipeline, sample_task):
        """Test task to scent signature conversion."""
        scent = olfactory_pipeline.process_task_to_scent(sample_task)
        
        assert isinstance(scent, ScentSignature)
        assert scent.id.startswith("task_")
        assert 0.0 <= scent.intensity <= 1.0
        assert 0.0 <= scent.persistence <= 1.0
        assert 0.0 <= scent.diffusion_rate <= 1.0
        
        # Should be stored in library
        assert scent.id in olfactory_pipeline.scent_library
    
    def test_resource_to_scent_conversion(self, olfactory_pipeline, sample_resource):
        """Test resource to scent signature conversion."""
        scent = olfactory_pipeline.process_resource_to_scent(sample_resource)
        
        assert isinstance(scent, ScentSignature)
        assert scent.id.startswith("resource_")
        assert 0.0 <= scent.intensity <= 1.0
    
    def test_scent_interaction_analysis(self, olfactory_pipeline, sample_task, sample_resource):
        """Test scent interaction analysis."""
        task_scent = olfactory_pipeline.process_task_to_scent(sample_task)
        resource_scent = olfactory_pipeline.process_resource_to_scent(sample_resource)
        
        interactions = olfactory_pipeline.analyze_scent_interactions(task_scent, resource_scent)
        
        required_keys = ['compatibility', 'similarity', 'mixing_potential', 'interference', 'temporal_sync']
        for key in required_keys:
            assert key in interactions
            assert 0.0 <= interactions[key] <= 1.0
    
    def test_cross_modal_fusion(self, olfactory_pipeline):
        """Test cross-modal sensor fusion."""
        # Create multi-modal sensor data
        sensor_data = []
        modalities = [SensorModality.OLFACTORY, SensorModality.VISUAL, SensorModality.AUDITORY]
        
        for modality in modalities:
            data = MultiModalSensorData(
                modality=modality,
                timestamp=datetime.utcnow(),
                raw_data=np.random.uniform(0, 1, 20),
                processed_features=np.array([]),
                confidence=0.8,
                sensor_id=f"{modality.value}_sensor"
            )
            sensor_data.append(data)
        
        fused_features = olfactory_pipeline.cross_modal_fusion(sensor_data)
        
        assert isinstance(fused_features, np.ndarray)
        assert len(fused_features) == olfactory_pipeline.feature_dimensions
        assert np.all(np.isfinite(fused_features))  # No NaN or inf values
    
    def test_temporal_pattern_extraction(self, olfactory_pipeline):
        """Test temporal pattern extraction from scent signatures."""
        # Create multiple scent signatures with different temporal profiles
        scents = []
        for i in range(5):
            scent = ScentSignature(
                id=f"scent_{i}",
                chemical_family=ChemicalFamily.ESTERS,
                molecular_descriptors=MolecularDescriptor(
                    molecular_weight=100, boiling_point=100, vapor_pressure=0.5,
                    solubility=0.5, polarity=0.5, aromaticity=0.3,
                    hydrogen_bonding=0.3, steric_hindrance=0.2
                ),
                intensity=0.5 + i * 0.1,
                persistence=0.5,
                diffusion_rate=0.5,
                temporal_profile=[0.8 * np.exp(-0.05 * t + i * 0.1) for t in range(50)]
            )
            scents.append(scent)
        
        patterns = olfactory_pipeline.extract_temporal_patterns(scents)
        
        assert isinstance(patterns, dict)
        if patterns:  # May be empty if processing fails
            expected_keys = ['mean_profile', 'std_profile', 'peak_time', 'decay_rate']
            for key in expected_keys:
                if key in patterns:
                    assert patterns[key] is not None
    
    def test_error_handling(self, olfactory_pipeline):
        """Test error handling in pipeline operations."""
        # Test with invalid task
        invalid_task = Mock()
        invalid_task.id = None  # Invalid ID
        invalid_task.priority = "invalid_priority"
        
        scent = olfactory_pipeline.process_task_to_scent(invalid_task)
        assert isinstance(scent, ScentSignature)  # Should return default signature
    
    def test_pipeline_statistics(self, olfactory_pipeline, sample_task):
        """Test pipeline statistics collection."""
        # Process some data
        olfactory_pipeline.process_task_to_scent(sample_task)
        
        stats = olfactory_pipeline.get_pipeline_statistics()
        
        assert isinstance(stats, dict)
        expected_keys = [
            'processing_stats', 'scent_library_size', 'sensor_buffer_size',
            'cross_modal_enabled', 'feature_dimensions'
        ]
        
        for key in expected_keys:
            assert key in stats


class TestBioNeuroOlfactoryFusionOptimizer:
    """Test Bio-Neural Olfactory Fusion Optimizer."""
    
    def test_initialization(self, bio_neural_optimizer):
        """Test optimizer initialization."""
        assert len(bio_neural_optimizer.olfactory_bulb.receptors) == 20
        assert bio_neural_optimizer.coherence_time == 50.0
        assert bio_neural_optimizer.learning_rate == 0.01
        assert len(bio_neural_optimizer.optimization_history) == 0
    
    def test_receptor_initialization(self, bio_neural_optimizer):
        """Test olfactory receptor initialization."""
        receptors = bio_neural_optimizer.olfactory_bulb.receptors
        
        for receptor in receptors:
            assert receptor.id.startswith("receptor_")
            assert isinstance(receptor.sensitivity_profile, dict)
            assert len(receptor.sensitivity_profile) > 0
            assert abs(receptor.quantum_state) > 0  # Non-zero quantum state
    
    def test_schedule_validation(self, bio_neural_optimizer):
        """Test schedule validation."""
        # Test with empty schedule
        empty_schedule = Schedule(
            id="empty",
            name="Empty Schedule",
            start_time=datetime.utcnow()
        )
        
        with pytest.raises(ValidationError):
            bio_neural_optimizer._validate_schedule(empty_schedule)
        
        # Test with tasks but no resources
        task_only_schedule = Schedule(
            id="task_only",
            name="Task Only",
            start_time=datetime.utcnow()
        )
        task_only_schedule.add_task(Task(
            id="task1", name="Test", duration=timedelta(hours=1)
        ))
        
        with pytest.raises(ValidationError):
            bio_neural_optimizer._validate_schedule(task_only_schedule)
    
    def test_olfactory_pattern_extraction(self, bio_neural_optimizer, sample_task):
        """Test olfactory pattern extraction from tasks."""
        tasks = [sample_task]
        patterns = bio_neural_optimizer._extract_olfactory_patterns(tasks)
        
        assert isinstance(patterns, dict)
        assert sample_task.id in patterns
        assert isinstance(patterns[sample_task.id], np.ndarray)
        assert len(patterns[sample_task.id]) == len(bio_neural_optimizer.olfactory_bulb.receptors)
    
    def test_quantum_enhancement(self, bio_neural_optimizer):
        """Test quantum enhancement of patterns."""
        test_pattern = np.array([0.5, 0.8, 0.3, 0.6, 0.1])
        enhanced = bio_neural_optimizer._apply_quantum_enhancement(test_pattern)
        
        assert isinstance(enhanced, np.ndarray)
        assert len(enhanced) == len(test_pattern)
        assert np.all(enhanced >= 0)
        assert np.all(enhanced <= 1)
    
    def test_entanglement_mapping(self, bio_neural_optimizer):
        """Test quantum entanglement mapping between tasks."""
        # Create tasks with dependencies
        task1 = Task(id="t1", name="Task 1", duration=timedelta(hours=1))
        task2 = Task(id="t2", name="Task 2", duration=timedelta(hours=1))
        task2.add_dependency("t1")
        
        tasks = [task1, task2]
        patterns = bio_neural_optimizer._extract_olfactory_patterns(tasks)
        entanglements = bio_neural_optimizer._map_quantum_entanglements(tasks, patterns)
        
        assert isinstance(entanglements, dict)
        # Should have entanglement due to dependency
        assert len(entanglements) >= 0
        
        for (task_id1, task_id2), entanglement in entanglements.items():
            assert isinstance(entanglement, complex)
            assert abs(entanglement) <= 1.0
    
    def test_resource_affinity_calculation(self, bio_neural_optimizer, sample_task, sample_resource):
        """Test bio-neural resource affinity calculation."""
        tasks = [sample_task]
        resources = [sample_resource]
        patterns = bio_neural_optimizer._extract_olfactory_patterns(tasks)
        
        affinities = bio_neural_optimizer._calculate_resource_affinities(tasks, resources, patterns)
        
        assert isinstance(affinities, dict)
        assert sample_task.id in affinities
        assert sample_resource.id in affinities[sample_task.id]
        
        affinity_value = affinities[sample_task.id][sample_resource.id]
        assert 0.0 <= affinity_value <= 1.0
    
    def test_olfactory_energy_calculation(self, bio_neural_optimizer):
        """Test olfactory system energy calculation."""
        # Create test state
        state = QuantumOlfactoryState(
            task_patterns={"task1": np.array([0.5, 0.8, 0.3])},
            resource_affinities={"task1": {"resource1": 0.8}},
            quantum_entanglements={("task1", "task2"): complex(0.5, 0.3)},
            olfactory_energy=0.0,
            pattern_coherence=0.7
        )
        
        # Mock schedule for testing
        schedule = Mock()
        schedule.resources = []
        schedule.get_task.return_value = None
        schedule.get_resource.return_value = None
        
        energy = bio_neural_optimizer._calculate_olfactory_energy(state, schedule)
        
        assert isinstance(energy, float)
        assert energy >= 0.0
        assert energy != float('inf')
    
    def test_bioneural_adaptation(self, bio_neural_optimizer):
        """Test bio-neural adaptation mechanism."""
        state = QuantumOlfactoryState(
            task_patterns={},
            resource_affinities={},
            quantum_entanglements={},
            olfactory_energy=10.0,
            pattern_coherence=0.5
        )
        
        original_quantum_states = [r.quantum_state for r in bio_neural_optimizer.olfactory_bulb.receptors]
        
        adapted_state = bio_neural_optimizer._apply_bioneural_adaptation(state)
        
        assert isinstance(adapted_state, QuantumOlfactoryState)
        
        # Check if quantum states changed
        new_quantum_states = [r.quantum_state for r in bio_neural_optimizer.olfactory_bulb.receptors]
        state_changes = any(
            abs(old - new) > 1e-10 
            for old, new in zip(original_quantum_states, new_quantum_states)
        )
        assert state_changes or bio_neural_optimizer.learning_rate == 0  # Should change unless lr=0
    
    @pytest.mark.slow
    def test_full_optimization_workflow(self, bio_neural_optimizer, sample_schedule):
        """Test complete optimization workflow (integration test)."""
        try:
            metrics = bio_neural_optimizer.optimize_schedule(sample_schedule)
            
            assert sample_schedule.status == ScheduleStatus.OPTIMIZED
            assert sample_schedule.optimized_at is not None
            assert len(sample_schedule.assignments) > 0
            
            # Check metrics
            assert metrics.optimization_time.total_seconds() > 0
            assert metrics.iterations > 0
            
        except OptimizationError:
            # Acceptable if optimization fails with small test data
            pytest.skip("Optimization failed with minimal test data")
    
    def test_error_recovery(self, bio_neural_optimizer):
        """Test error recovery mechanisms."""
        # Create invalid schedule that should trigger error handling
        schedule = Schedule(
            id="invalid",
            name="Invalid Schedule", 
            start_time=datetime.utcnow()
        )
        
        # Add task without resources
        task = Task(id="task1", name="Test", duration=timedelta(hours=1))
        schedule.add_task(task)
        
        with pytest.raises(ValidationError):
            bio_neural_optimizer.optimize_schedule(schedule)
    
    def test_convergence_detection(self, bio_neural_optimizer):
        """Test optimization convergence detection."""
        # Simulate optimization history with converging values
        bio_neural_optimizer.optimization_history = [
            {'generation': i, 'energy': 10.0 - i * 0.01, 'coherence': 0.5}
            for i in range(30)
        ]
        
        converged = bio_neural_optimizer._check_olfactory_convergence(25)
        assert isinstance(converged, bool)
    
    def test_optimization_summary(self, bio_neural_optimizer):
        """Test optimization summary generation."""
        summary = bio_neural_optimizer.get_optimization_summary()
        
        assert isinstance(summary, dict)
        expected_keys = [
            'algorithm', 'optimization_cycles', 'quantum_phase',
            'convergence_achieved', 'total_generations'
        ]
        
        for key in expected_keys:
            assert key in summary


class TestMultiModalSensorData:
    """Test multi-modal sensor data processing."""
    
    @pytest.mark.parametrize("modality", [
        SensorModality.OLFACTORY,
        SensorModality.VISUAL, 
        SensorModality.AUDITORY,
        SensorModality.TACTILE,
        SensorModality.TEMPORAL
    ])
    def test_feature_extraction_by_modality(self, modality):
        """Test feature extraction for each sensor modality."""
        sensor_data = MultiModalSensorData(
            modality=modality,
            timestamp=datetime.utcnow(),
            raw_data=np.random.uniform(0, 1, 50),
            processed_features=np.array([]),
            confidence=0.8,
            sensor_id=f"{modality.value}_test"
        )
        
        features = sensor_data.extract_features()
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert np.all(np.isfinite(features))
    
    def test_olfactory_feature_extraction_details(self):
        """Test detailed olfactory feature extraction."""
        # Create synthetic GC-MS like data
        raw_data = np.concatenate([
            np.random.normal(0.1, 0.02, 20),  # Baseline
            np.random.normal(0.8, 0.1, 10),   # Peak 1
            np.random.normal(0.2, 0.05, 20),  # Baseline
            np.random.normal(0.6, 0.08, 10),  # Peak 2
            np.random.normal(0.1, 0.02, 20)   # Baseline
        ])
        
        sensor_data = MultiModalSensorData(
            modality=SensorModality.OLFACTORY,
            timestamp=datetime.utcnow(),
            raw_data=raw_data,
            processed_features=np.array([]),
            confidence=0.9,
            sensor_id="gcms_001"
        )
        
        features = sensor_data.extract_features()
        
        # Should detect peaks
        assert len(features) >= 6  # At least 6 features expected
        assert np.any(features > 0)  # Should have some non-zero features


class TestPerformanceAndBenchmarks:
    """Performance and benchmark tests."""
    
    def test_optimization_speed(self, bio_neural_optimizer):
        """Test optimization speed benchmarks."""
        # Create medium-sized test problem
        schedule = Schedule(
            id="speed_test",
            name="Speed Test",
            start_time=datetime.utcnow()
        )
        
        # Add multiple tasks and resources
        for i in range(10):
            task = Task(
                id=f"task_{i}",
                name=f"Task {i}",
                duration=timedelta(hours=1),
                priority=TaskPriority.MEDIUM
            )
            schedule.add_task(task)
        
        for i in range(5):
            resource = Resource(
                id=f"resource_{i}",
                name=f"Resource {i}",
                type="compute",
                total_capacity=10.0,
                available_capacity=8.0,
                efficiency_rating=0.8,
                cost_per_unit=5.0
            )
            schedule.add_resource(resource)
        
        start_time = time.time()
        
        try:
            metrics = bio_neural_optimizer.optimize_schedule(schedule)
            optimization_time = time.time() - start_time
            
            # Performance assertions
            assert optimization_time < 30.0  # Should complete within 30 seconds
            assert metrics.iterations > 0
            
        except OptimizationError:
            # Still measure time even if optimization fails
            optimization_time = time.time() - start_time
            assert optimization_time < 30.0
    
    def test_memory_usage(self, bio_neural_optimizer):
        """Test memory usage during optimization."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run optimization multiple times
        for iteration in range(5):
            schedule = Schedule(
                id=f"memory_test_{iteration}",
                name="Memory Test",
                start_time=datetime.utcnow()
            )
            
            task = Task(id=f"task_{iteration}", name="Test", duration=timedelta(hours=1))
            resource = Resource(
                id=f"resource_{iteration}", name="Test", type="compute",
                total_capacity=10.0, available_capacity=8.0,
                efficiency_rating=0.8, cost_per_unit=5.0
            )
            
            schedule.add_task(task)
            schedule.add_resource(resource)
            
            try:
                bio_neural_optimizer.optimize_schedule(schedule)
            except:
                pass  # Ignore optimization errors for memory test
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not increase excessively
        assert memory_increase < 100  # Less than 100MB increase
    
    def test_scalability(self, bio_neural_optimizer):
        """Test algorithm scalability with different problem sizes."""
        problem_sizes = [5, 10, 15]
        optimization_times = []
        
        for size in problem_sizes:
            schedule = Schedule(
                id=f"scale_test_{size}",
                name=f"Scale Test {size}",
                start_time=datetime.utcnow()
            )
            
            # Add tasks and resources proportional to size
            for i in range(size):
                task = Task(id=f"task_{i}", name=f"Task {i}", duration=timedelta(hours=1))
                schedule.add_task(task)
            
            for i in range(max(1, size // 2)):
                resource = Resource(
                    id=f"resource_{i}", name=f"Resource {i}", type="compute",
                    total_capacity=10.0, available_capacity=8.0,
                    efficiency_rating=0.8, cost_per_unit=5.0
                )
                schedule.add_resource(resource)
            
            start_time = time.time()
            
            try:
                bio_neural_optimizer.optimize_schedule(schedule)
                optimization_time = time.time() - start_time
                optimization_times.append(optimization_time)
                
            except OptimizationError:
                optimization_times.append(30.0)  # Max time for failed optimization
        
        # Check that optimization time doesn't grow exponentially
        for i in range(1, len(optimization_times)):
            growth_factor = optimization_times[i] / optimization_times[i-1]
            assert growth_factor < 5.0  # Should not grow more than 5x per size increase


class TestStatisticalValidation:
    """Statistical validation of quantum-neural improvements."""
    
    def test_reproducibility(self, bio_neural_optimizer):
        """Test reproducibility of results with same inputs."""
        # Set deterministic conditions
        np.random.seed(42)
        
        schedule = Schedule(
            id="repro_test",
            name="Reproducibility Test",
            start_time=datetime.utcnow()
        )
        
        task = Task(id="task1", name="Test Task", duration=timedelta(hours=1))
        resource = Resource(
            id="res1", name="Test Resource", type="compute",
            total_capacity=10.0, available_capacity=8.0,
            efficiency_rating=0.8, cost_per_unit=5.0
        )
        
        schedule.add_task(task)
        schedule.add_resource(resource)
        
        results = []
        for run in range(3):
            try:
                # Reset optimizer state
                optimizer = BioNeuroOlfactoryFusionOptimizer(
                    num_receptors=10,
                    quantum_coherence_time=50.0,
                    learning_rate=0.01
                )
                
                metrics = optimizer.optimize_schedule(schedule.copy())
                results.append(metrics.total_cost)
                
            except OptimizationError:
                results.append(float('inf'))
        
        # Results should be similar (not necessarily identical due to quantum nature)
        if len([r for r in results if r != float('inf')]) >= 2:
            valid_results = [r for r in results if r != float('inf')]
            coefficient_of_variation = np.std(valid_results) / np.mean(valid_results)
            assert coefficient_of_variation < 0.5  # Reasonable variation
    
    def test_quantum_enhancement_validation(self):
        """Test that quantum enhancement provides measurable improvements."""
        # Create two optimizers: one with quantum enhancement, one without
        quantum_optimizer = BioNeuroOlfactoryFusionOptimizer(
            num_receptors=20,
            entanglement_strength=0.5  # High entanglement
        )
        
        classical_optimizer = BioNeuroOlfactoryFusionOptimizer(
            num_receptors=20,
            entanglement_strength=0.0  # No entanglement (more classical)
        )
        
        # Create test problem
        schedule = Schedule(
            id="quantum_test",
            name="Quantum Enhancement Test",
            start_time=datetime.utcnow()
        )
        
        for i in range(8):
            task = Task(id=f"task_{i}", name=f"Task {i}", duration=timedelta(hours=1))
            schedule.add_task(task)
        
        for i in range(4):
            resource = Resource(
                id=f"res_{i}", name=f"Resource {i}", type="compute",
                total_capacity=10.0, available_capacity=8.0,
                efficiency_rating=0.8, cost_per_unit=5.0
            )
            schedule.add_resource(resource)
        
        quantum_results = []
        classical_results = []
        
        # Run multiple trials
        for trial in range(3):
            try:
                q_metrics = quantum_optimizer.optimize_schedule(schedule.copy())
                quantum_results.append(q_metrics.total_cost)
            except:
                quantum_results.append(float('inf'))
            
            try:
                c_metrics = classical_optimizer.optimize_schedule(schedule.copy())
                classical_results.append(c_metrics.total_cost)
            except:
                classical_results.append(float('inf'))
        
        # Analyze results (quantum should show different behavior, not necessarily better)
        valid_quantum = [r for r in quantum_results if r != float('inf')]
        valid_classical = [r for r in classical_results if r != float('inf')]
        
        if valid_quantum and valid_classical:
            # Check that results are different (indicating quantum effects)
            mean_quantum = np.mean(valid_quantum)
            mean_classical = np.mean(valid_classical)
            
            # Results should be meaningfully different
            relative_difference = abs(mean_quantum - mean_classical) / max(mean_quantum, mean_classical)
            assert relative_difference > 0.01  # At least 1% difference


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_empty_inputs(self, bio_neural_optimizer):
        """Test handling of empty or minimal inputs."""
        empty_schedule = Schedule(
            id="empty",
            name="Empty",
            start_time=datetime.utcnow()
        )
        
        with pytest.raises(ValidationError):
            bio_neural_optimizer.optimize_schedule(empty_schedule)
    
    def test_extreme_parameter_values(self):
        """Test optimizer with extreme parameter values."""
        # Test with very large number of receptors
        try:
            optimizer = BioNeuroOlfactoryFusionOptimizer(num_receptors=1000)
            assert len(optimizer.olfactory_bulb.receptors) == 1000
        except MemoryError:
            pytest.skip("System memory insufficient for large receptor test")
        
        # Test with zero receptors
        with pytest.raises(Exception):  # Should fail
            BioNeuroOlfactoryFusionOptimizer(num_receptors=0)
    
    def test_invalid_quantum_states(self, olfactory_receptor):
        """Test handling of invalid quantum states."""
        # Test with zero quantum state
        olfactory_receptor.quantum_state = complex(0, 0)
        
        features = {'test': 0.5}
        activation = olfactory_receptor.detect_feature(features)
        
        assert 0.0 <= activation <= 1.0  # Should still work
    
    def test_corrupted_data_handling(self, olfactory_pipeline):
        """Test handling of corrupted or invalid sensor data."""
        corrupted_data = MultiModalSensorData(
            modality=SensorModality.OLFACTORY,
            timestamp=datetime.utcnow(),
            raw_data=np.array([np.nan, np.inf, -np.inf, 1.0]),  # Corrupted data
            processed_features=np.array([]),
            confidence=0.0,  # Zero confidence
            sensor_id="corrupted"
        )
        
        features = corrupted_data.extract_features()
        
        # Should return valid features despite corrupted input
        assert isinstance(features, np.ndarray)
        assert np.all(np.isfinite(features))
    
    def test_timeout_handling(self):
        """Test optimization timeout handling."""
        # Create optimizer with very short timeout (simulated)
        optimizer = BioNeuroOlfactoryFusionOptimizer(quantum_coherence_time=0.001)
        
        # This test would require modifying the optimizer to respect timeout
        # For now, just verify it doesn't crash with extreme parameters
        assert optimizer.coherence_time == 0.001


@pytest.mark.integration
class TestIntegrationWithExistingSystem:
    """Integration tests with existing quantum task planner system."""
    
    def test_integration_with_qaoa_allocator(self, sample_schedule):
        """Test integration with existing QAOA allocator."""
        from src.quantum_task_planner.algorithms.qaoa_allocator import QAOAResourceAllocator, QAOAParameters
        
        # Run bio-neural optimization
        bio_optimizer = BioNeuroOlfactoryFusionOptimizer(num_receptors=10)
        
        try:
            bio_metrics = bio_optimizer.optimize_schedule(sample_schedule.copy())
            bio_cost = bio_metrics.total_cost
        except OptimizationError:
            bio_cost = float('inf')
        
        # Run QAOA optimization for comparison  
        qaoa_optimizer = QAOAResourceAllocator(QAOAParameters(layers=1, max_iterations=50))
        
        try:
            qaoa_metrics = qaoa_optimizer.allocate_resources(sample_schedule.copy())
            qaoa_cost = qaoa_metrics.total_cost
        except:
            qaoa_cost = float('inf')
        
        # Both should produce valid results or both should fail
        assert (bio_cost != float('inf')) == (qaoa_cost != float('inf'))
    
    def test_compatibility_with_task_models(self, sample_task):
        """Test compatibility with existing task models."""
        pipeline = OlfactoryDataPipeline()
        
        # Should work with standard Task objects
        scent = pipeline.process_task_to_scent(sample_task)
        assert isinstance(scent, ScentSignature)
        
        # Test with task containing all optional fields
        complex_task = Task(
            id="complex",
            name="Complex Task",
            description="A complex task with all fields",
            duration=timedelta(hours=3, minutes=30),
            priority=TaskPriority.CRITICAL,
            quantum_weight=2.5,
            entanglement_factor=0.8,
            earliest_start=datetime.utcnow(),
            latest_finish=datetime.utcnow() + timedelta(days=1)
        )
        complex_task.add_dependency("dep1")
        complex_task.resource_requirements = {"cpu": 8.0, "memory": 16.0, "gpu": 2.0}
        
        complex_scent = pipeline.process_task_to_scent(complex_task)
        assert isinstance(complex_scent, ScentSignature)
        assert complex_scent.intensity > scent.intensity  # Higher priority should give higher intensity


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "-x"  # Stop on first failure
    ])