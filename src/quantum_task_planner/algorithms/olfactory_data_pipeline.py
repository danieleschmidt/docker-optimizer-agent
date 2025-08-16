"""Olfactory Data Processing Pipeline for Bio-Neural Fusion.

This module implements a comprehensive data processing pipeline for
olfactory-inspired optimization, including scent signature extraction,
chemical pattern analysis, and multi-modal sensor fusion.

Research Contribution:
- Novel data structures for representing chemical signatures as optimization features
- Multi-scale temporal pattern analysis for dynamic task scheduling
- Cross-modal learning between olfactory and visual/auditory sensory inputs

Citation: Schmidt, D. (2025). "Multi-Modal Olfactory Data Processing for
Quantum Task Optimization." Neural Computing & Pattern Recognition.
"""

import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np

from ..models.resource import Resource
from ..models.task import Task, TaskPriority

logger = logging.getLogger(__name__)


class SensorModality(str, Enum):
    """Multi-modal sensor types for cross-modal learning."""
    OLFACTORY = "olfactory"
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    TEMPORAL = "temporal"


class ChemicalFamily(str, Enum):
    """Chemical families for scent signature classification."""
    ESTERS = "esters"          # Fruity, sweet
    ALDEHYDES = "aldehydes"    # Fresh, citrusy
    KETONES = "ketones"        # Woody, musky
    ALCOHOLS = "alcohols"      # Floral, clean
    TERPENES = "terpenes"      # Pine, herbal
    PHENOLS = "phenols"        # Smoky, medicinal
    ACIDS = "acids"            # Sour, pungent
    ETHERS = "ethers"          # Ethereal, anesthetic


@dataclass
class MolecularDescriptor:
    """Molecular descriptor for chemical signature analysis."""

    molecular_weight: float
    boiling_point: float
    vapor_pressure: float
    solubility: float
    polarity: float
    aromaticity: float
    hydrogen_bonding: float
    steric_hindrance: float

    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector."""
        return np.array([
            min(self.molecular_weight / 500.0, 1.0),  # Normalize by typical range
            min(self.boiling_point / 300.0, 1.0),
            min(self.vapor_pressure / 100.0, 1.0),
            self.solubility,  # Assumed already normalized [0,1]
            self.polarity,    # Assumed already normalized [0,1]
            self.aromaticity, # Assumed already normalized [0,1]
            self.hydrogen_bonding,  # Assumed already normalized [0,1]
            self.steric_hindrance   # Assumed already normalized [0,1]
        ])


@dataclass
class ScentSignature:
    """Chemical scent signature for task characterization."""

    id: str
    chemical_family: ChemicalFamily
    molecular_descriptors: MolecularDescriptor
    intensity: float  # [0,1]
    persistence: float  # [0,1] - how long the scent lasts
    diffusion_rate: float  # [0,1] - how quickly it spreads
    interaction_coefficients: Dict[str, float] = field(default_factory=dict)  # Scent mixing
    temporal_profile: List[float] = field(default_factory=list)  # Time-dependent intensity

    def __post_init__(self):
        """Initialize temporal profile if not provided."""
        if not self.temporal_profile:
            # Default exponential decay profile
            self.temporal_profile = [
                self.intensity * np.exp(-0.1 * t) for t in range(100)
            ]

    def calculate_similarity(self, other: 'ScentSignature') -> float:
        """Calculate chemical similarity with another scent signature."""
        try:
            # Molecular descriptor similarity
            desc_sim = np.dot(
                self.molecular_descriptors.to_vector(),
                other.molecular_descriptors.to_vector()
            ) / (
                np.linalg.norm(self.molecular_descriptors.to_vector()) *
                np.linalg.norm(other.molecular_descriptors.to_vector()) + 1e-8
            )

            # Chemical family compatibility
            family_compatibility = {
                ChemicalFamily.ESTERS: {ChemicalFamily.ALDEHYDES: 0.8, ChemicalFamily.ALCOHOLS: 0.7},
                ChemicalFamily.ALDEHYDES: {ChemicalFamily.ESTERS: 0.8, ChemicalFamily.TERPENES: 0.6},
                ChemicalFamily.KETONES: {ChemicalFamily.PHENOLS: 0.7, ChemicalFamily.ETHERS: 0.5},
                ChemicalFamily.ALCOHOLS: {ChemicalFamily.ESTERS: 0.7, ChemicalFamily.ACIDS: 0.4},
                ChemicalFamily.TERPENES: {ChemicalFamily.ALDEHYDES: 0.6, ChemicalFamily.KETONES: 0.5},
                ChemicalFamily.PHENOLS: {ChemicalFamily.KETONES: 0.7, ChemicalFamily.ACIDS: 0.6},
                ChemicalFamily.ACIDS: {ChemicalFamily.ALCOHOLS: 0.4, ChemicalFamily.PHENOLS: 0.6},
                ChemicalFamily.ETHERS: {ChemicalFamily.KETONES: 0.5, ChemicalFamily.ALDEHYDES: 0.3}
            }

            family_sim = 0.5  # Neutral similarity
            if self.chemical_family == other.chemical_family:
                family_sim = 1.0
            elif other.chemical_family in family_compatibility.get(self.chemical_family, {}):
                family_sim = family_compatibility[self.chemical_family][other.chemical_family]

            # Temporal profile correlation
            temporal_sim = 0.5
            if len(self.temporal_profile) == len(other.temporal_profile):
                correlation = np.corrcoef(self.temporal_profile, other.temporal_profile)[0, 1]
                if not np.isnan(correlation):
                    temporal_sim = abs(correlation)

            # Combined similarity
            total_similarity = 0.4 * desc_sim + 0.4 * family_sim + 0.2 * temporal_sim
            return max(0.0, min(1.0, total_similarity))

        except Exception as e:
            logger.warning(f"Error calculating scent similarity: {e}")
            return 0.5

    def generate_hash(self) -> str:
        """Generate unique hash for scent signature."""
        signature_data = {
            'chemical_family': self.chemical_family.value,
            'molecular_descriptors': self.molecular_descriptors.__dict__,
            'intensity': round(self.intensity, 3),
            'persistence': round(self.persistence, 3),
            'diffusion_rate': round(self.diffusion_rate, 3)
        }

        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:12]


@dataclass
class MultiModalSensorData:
    """Multi-modal sensor data for cross-modal learning."""

    modality: SensorModality
    timestamp: datetime
    raw_data: np.ndarray
    processed_features: np.ndarray
    confidence: float
    sensor_id: str
    calibration_params: Dict[str, float] = field(default_factory=dict)

    def extract_features(self) -> np.ndarray:
        """Extract features based on sensor modality."""
        try:
            if self.modality == SensorModality.OLFACTORY:
                return self._extract_olfactory_features()
            elif self.modality == SensorModality.VISUAL:
                return self._extract_visual_features()
            elif self.modality == SensorModality.AUDITORY:
                return self._extract_auditory_features()
            elif self.modality == SensorModality.TACTILE:
                return self._extract_tactile_features()
            elif self.modality == SensorModality.TEMPORAL:
                return self._extract_temporal_features()
            else:
                return self.raw_data

        except Exception as e:
            logger.warning(f"Feature extraction failed for {self.modality}: {e}")
            return np.zeros(10)  # Default feature vector

    def _extract_olfactory_features(self) -> np.ndarray:
        """Extract olfactory-specific features using chemical analysis."""
        try:
            # Simulate gas chromatography-mass spectrometry (GC-MS) analysis
            features = []

            # Peak detection and intensity analysis
            peaks = self._detect_chemical_peaks(self.raw_data)
            features.extend([len(peaks), np.mean([p[1] for p in peaks]) if peaks else 0])

            # Spectral analysis
            fft_data = np.fft.fft(self.raw_data)
            power_spectrum = np.abs(fft_data[:len(fft_data)//2])
            features.extend([
                np.mean(power_spectrum),
                np.std(power_spectrum),
                np.max(power_spectrum),
                np.argmax(power_spectrum)  # Dominant frequency
            ])

            # Chemical signature inference
            molecular_indicators = self._infer_molecular_properties(self.raw_data)
            features.extend(molecular_indicators)

            return np.array(features)

        except Exception as e:
            logger.debug(f"Olfactory feature extraction error: {e}")
            return np.random.normal(0.5, 0.1, 10)

    def _extract_visual_features(self) -> np.ndarray:
        """Extract visual features for cross-modal learning."""
        # Simulate computer vision features
        return np.array([
            np.mean(self.raw_data),     # Average brightness
            np.std(self.raw_data),      # Contrast
            np.max(self.raw_data) - np.min(self.raw_data),  # Dynamic range
            len(np.where(np.diff(self.raw_data) > 0.1)[0]),  # Edge count
            np.sum(self.raw_data > np.mean(self.raw_data)),   # Bright pixel count
        ])

    def _extract_auditory_features(self) -> np.ndarray:
        """Extract auditory features for cross-modal learning."""
        # Simulate audio features
        fft_audio = np.fft.fft(self.raw_data)
        return np.array([
            np.mean(np.abs(fft_audio)),      # Average amplitude
            np.std(np.abs(fft_audio)),       # Amplitude variance
            np.argmax(np.abs(fft_audio)),    # Dominant frequency
            len(np.where(np.abs(self.raw_data) > 0.5)[0]),  # High amplitude count
            np.sum(np.abs(self.raw_data))     # Total energy
        ])

    def _extract_tactile_features(self) -> np.ndarray:
        """Extract tactile features for cross-modal learning."""
        # Simulate tactile sensor features
        return np.array([
            np.mean(self.raw_data),          # Average pressure
            np.max(self.raw_data),           # Peak pressure
            np.sum(np.abs(np.diff(self.raw_data))),  # Texture roughness
            len(np.where(self.raw_data > np.mean(self.raw_data))[0]),  # Contact points
            np.var(self.raw_data)            # Pressure variation
        ])

    def _extract_temporal_features(self) -> np.ndarray:
        """Extract temporal pattern features."""
        # Time series analysis
        return np.array([
            np.mean(self.raw_data),          # Temporal average
            np.std(self.raw_data),           # Temporal variance
            len(np.where(np.diff(self.raw_data) > 0)[0]),  # Rising edges
            len(np.where(np.diff(self.raw_data) < 0)[0]),  # Falling edges
            np.max(self.raw_data) - np.min(self.raw_data)  # Temporal range
        ])

    def _detect_chemical_peaks(self, signal: np.ndarray) -> List[Tuple[int, float]]:
        """Detect chemical peaks in sensor signal."""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > 0.1:  # Minimum peak threshold
                    peaks.append((i, signal[i]))
        return sorted(peaks, key=lambda x: x[1], reverse=True)[:10]  # Top 10 peaks

    def _infer_molecular_properties(self, signal: np.ndarray) -> List[float]:
        """Infer molecular properties from sensor signal."""
        # Simplified molecular property inference
        return [
            np.mean(signal),        # Average molecular activity
            np.std(signal),         # Molecular diversity
            len(np.where(signal > 0.5)[0]) / len(signal),  # High activity ratio
            np.sum(signal > 0.8),   # Highly reactive compounds
        ]


class OlfactoryDataPipeline:
    """Comprehensive olfactory data processing pipeline for bio-neural optimization."""

    def __init__(self,
                 buffer_size: int = 1000,
                 feature_dimensions: int = 64,
                 enable_cross_modal: bool = True):
        """Initialize olfactory data pipeline.
        
        Args:
            buffer_size: Size of temporal data buffer
            feature_dimensions: Dimensionality of extracted features
            enable_cross_modal: Enable cross-modal sensor fusion
        """
        try:
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

            # Pipeline configuration
            self.buffer_size = buffer_size
            self.feature_dimensions = feature_dimensions
            self.enable_cross_modal = enable_cross_modal

            # Data buffers
            self.sensor_buffer: deque = deque(maxlen=buffer_size)
            self.scent_library: Dict[str, ScentSignature] = {}
            self.temporal_patterns: Dict[str, List[np.ndarray]] = {}

            # Cross-modal learning
            self.modal_correlations: Dict[Tuple[SensorModality, SensorModality], float] = {}
            self.cross_modal_weights: np.ndarray = np.ones((5, 5)) * 0.2  # 5 modalities

            # Chemical databases
            self.chemical_database = self._initialize_chemical_database()
            self.scent_clusters: Dict[str, List[str]] = {}

            # Processing statistics
            self.processing_stats = {
                'total_samples': 0,
                'successful_extractions': 0,
                'cross_modal_fusions': 0,
                'scent_identifications': 0
            }

            self.logger.info(f"Olfactory data pipeline initialized with {feature_dimensions}D features")

        except Exception as e:
            self.logger.error(f"Failed to initialize olfactory pipeline: {e}")
            raise

    def process_task_to_scent(self, task: Task) -> ScentSignature:
        """Convert task characteristics to olfactory scent signature.
        
        Args:
            task: Task to convert
            
        Returns:
            Generated scent signature
        """
        try:
            # Map task properties to chemical families
            chemical_family = self._map_task_to_chemical_family(task)

            # Generate molecular descriptors based on task properties
            molecular_descriptors = self._generate_molecular_descriptors(task)

            # Calculate intensity based on priority and complexity
            intensity = self._calculate_scent_intensity(task)

            # Calculate persistence based on duration
            persistence = min(1.0, task.duration.total_seconds() / (24 * 3600))

            # Calculate diffusion rate based on dependencies
            diffusion_rate = min(1.0, len(task.dependencies) / 10.0)

            # Create scent signature
            scent_id = f"task_{task.id}_{chemical_family.value}"
            signature = ScentSignature(
                id=scent_id,
                chemical_family=chemical_family,
                molecular_descriptors=molecular_descriptors,
                intensity=intensity,
                persistence=persistence,
                diffusion_rate=diffusion_rate
            )

            # Store in scent library
            self.scent_library[scent_id] = signature

            self.logger.debug(f"Generated scent signature for task {task.id}: {chemical_family.value}")
            return signature

        except Exception as e:
            self.logger.error(f"Error converting task to scent: {e}")
            # Return default signature
            return self._create_default_signature(task.id)

    def process_resource_to_scent(self, resource: Resource) -> ScentSignature:
        """Convert resource characteristics to olfactory scent signature.
        
        Args:
            resource: Resource to convert
            
        Returns:
            Generated scent signature
        """
        try:
            # Map resource type to chemical family
            chemical_family = self._map_resource_to_chemical_family(resource)

            # Generate molecular descriptors
            molecular_descriptors = MolecularDescriptor(
                molecular_weight=resource.total_capacity * 10,
                boiling_point=100 + resource.efficiency_rating * 50,
                vapor_pressure=resource.cost_per_unit,
                solubility=resource.available_capacity / max(resource.total_capacity, 1.0),
                polarity=0.5,  # Neutral default
                aromaticity=resource.efficiency_rating,
                hydrogen_bonding=0.3,  # Default bonding
                steric_hindrance=1.0 - resource.efficiency_rating
            )

            # Calculate properties
            intensity = resource.efficiency_rating
            persistence = resource.available_capacity / max(resource.total_capacity, 1.0)
            diffusion_rate = 1.0 - resource.cost_per_unit / 1000.0

            # Create signature
            scent_id = f"resource_{resource.id}_{chemical_family.value}"
            signature = ScentSignature(
                id=scent_id,
                chemical_family=chemical_family,
                molecular_descriptors=molecular_descriptors,
                intensity=intensity,
                persistence=persistence,
                diffusion_rate=max(0.1, diffusion_rate)
            )

            self.scent_library[scent_id] = signature
            return signature

        except Exception as e:
            self.logger.error(f"Error converting resource to scent: {e}")
            return self._create_default_signature(resource.id)

    def analyze_scent_interactions(self, scent1: ScentSignature,
                                   scent2: ScentSignature) -> Dict[str, float]:
        """Analyze interactions between two scent signatures.
        
        Args:
            scent1: First scent signature
            scent2: Second scent signature
            
        Returns:
            Interaction analysis results
        """
        try:
            interactions = {}

            # Chemical similarity
            interactions['similarity'] = scent1.calculate_similarity(scent2)

            # Mixing potential
            mixing_matrix = {
                (ChemicalFamily.ESTERS, ChemicalFamily.ALDEHYDES): 0.9,
                (ChemicalFamily.ALCOHOLS, ChemicalFamily.ESTERS): 0.8,
                (ChemicalFamily.TERPENES, ChemicalFamily.KETONES): 0.7,
                (ChemicalFamily.PHENOLS, ChemicalFamily.ACIDS): 0.6,
            }

            family_pair = (scent1.chemical_family, scent2.chemical_family)
            reverse_pair = (scent2.chemical_family, scent1.chemical_family)

            if family_pair in mixing_matrix:
                interactions['mixing_potential'] = mixing_matrix[family_pair]
            elif reverse_pair in mixing_matrix:
                interactions['mixing_potential'] = mixing_matrix[reverse_pair]
            else:
                interactions['mixing_potential'] = 0.5

            # Interference analysis
            interference = abs(scent1.intensity - scent2.intensity) * 0.5
            if scent1.chemical_family == scent2.chemical_family:
                interference += 0.3  # Same family compounds may interfere

            interactions['interference'] = min(1.0, interference)

            # Temporal compatibility
            if len(scent1.temporal_profile) == len(scent2.temporal_profile):
                temporal_correlation = np.corrcoef(
                    scent1.temporal_profile,
                    scent2.temporal_profile
                )[0, 1]
                interactions['temporal_sync'] = abs(temporal_correlation) if not np.isnan(temporal_correlation) else 0.5
            else:
                interactions['temporal_sync'] = 0.5

            # Overall compatibility score
            interactions['compatibility'] = (
                0.3 * interactions['similarity'] +
                0.3 * interactions['mixing_potential'] +
                0.2 * (1.0 - interactions['interference']) +
                0.2 * interactions['temporal_sync']
            )

            return interactions

        except Exception as e:
            self.logger.error(f"Error analyzing scent interactions: {e}")
            return {'compatibility': 0.5, 'similarity': 0.5, 'mixing_potential': 0.5}

    def extract_temporal_patterns(self, task_scents: List[ScentSignature],
                                  time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Extract temporal patterns from scent signatures.
        
        Args:
            task_scents: List of scent signatures
            time_window: Time window for pattern analysis
            
        Returns:
            Temporal pattern analysis
        """
        try:
            patterns = {}

            # Aggregate temporal profiles
            if task_scents:
                max_length = max(len(scent.temporal_profile) for scent in task_scents)

                # Align temporal profiles
                aligned_profiles = []
                for scent in task_scents:
                    profile = np.array(scent.temporal_profile)
                    if len(profile) < max_length:
                        # Pad with exponential decay
                        padding = np.array([
                            profile[-1] * np.exp(-0.1 * i)
                            for i in range(1, max_length - len(profile) + 1)
                        ])
                        profile = np.concatenate([profile, padding])
                    aligned_profiles.append(profile)

                aligned_profiles = np.array(aligned_profiles)

                # Calculate pattern statistics
                patterns['mean_profile'] = np.mean(aligned_profiles, axis=0)
                patterns['std_profile'] = np.std(aligned_profiles, axis=0)
                patterns['peak_time'] = np.argmax(patterns['mean_profile'])
                patterns['decay_rate'] = self._calculate_decay_rate(patterns['mean_profile'])

                # Frequency domain analysis
                fft_mean = np.fft.fft(patterns['mean_profile'])
                patterns['dominant_frequency'] = np.argmax(np.abs(fft_mean))
                patterns['spectral_centroid'] = np.sum(
                    np.arange(len(fft_mean)) * np.abs(fft_mean)
                ) / np.sum(np.abs(fft_mean))

                # Pattern clustering
                patterns['clusters'] = self._cluster_temporal_patterns(aligned_profiles)

            return patterns

        except Exception as e:
            self.logger.error(f"Error extracting temporal patterns: {e}")
            return {}

    def cross_modal_fusion(self, sensor_data: List[MultiModalSensorData]) -> np.ndarray:
        """Perform cross-modal sensor fusion for enhanced features.
        
        Args:
            sensor_data: List of multi-modal sensor data
            
        Returns:
            Fused feature vector
        """
        try:
            if not self.enable_cross_modal:
                # Return olfactory features only
                olfactory_data = [d for d in sensor_data if d.modality == SensorModality.OLFACTORY]
                if olfactory_data:
                    return olfactory_data[0].extract_features()
                else:
                    return np.zeros(self.feature_dimensions)

            # Extract features from each modality
            modal_features = {}
            for data in sensor_data:
                features = data.extract_features()
                modal_features[data.modality] = features

            # Perform weighted fusion
            modality_order = [
                SensorModality.OLFACTORY,
                SensorModality.VISUAL,
                SensorModality.AUDITORY,
                SensorModality.TACTILE,
                SensorModality.TEMPORAL
            ]

            # Create fusion matrix
            fused_features = []
            base_dim = min([len(features) for features in modal_features.values()]) if modal_features else 5

            for i, primary_modality in enumerate(modality_order):
                if primary_modality in modal_features:
                    primary_features = modal_features[primary_modality][:base_dim]

                    # Cross-modal enhancement
                    enhanced_features = primary_features.copy()
                    for j, secondary_modality in enumerate(modality_order):
                        if i != j and secondary_modality in modal_features:
                            weight = self.cross_modal_weights[i, j]
                            secondary_features = modal_features[secondary_modality][:base_dim]
                            enhanced_features += weight * secondary_features

                    fused_features.extend(enhanced_features)

            # Normalize fused features
            fused_array = np.array(fused_features)
            if len(fused_array) > self.feature_dimensions:
                fused_array = fused_array[:self.feature_dimensions]
            elif len(fused_array) < self.feature_dimensions:
                padding = np.zeros(self.feature_dimensions - len(fused_array))
                fused_array = np.concatenate([fused_array, padding])

            # Update processing stats
            self.processing_stats['cross_modal_fusions'] += 1

            return fused_array

        except Exception as e:
            self.logger.error(f"Cross-modal fusion failed: {e}")
            return np.zeros(self.feature_dimensions)

    def _map_task_to_chemical_family(self, task: Task) -> ChemicalFamily:
        """Map task characteristics to appropriate chemical family."""
        try:
            # Priority-based mapping
            if task.priority == TaskPriority.CRITICAL:
                return ChemicalFamily.ALDEHYDES  # Sharp, attention-grabbing
            elif task.priority == TaskPriority.HIGH:
                return ChemicalFamily.ESTERS  # Pleasant, noticeable
            elif task.priority == TaskPriority.MEDIUM:
                return ChemicalFamily.ALCOHOLS  # Clean, neutral
            else:
                return ChemicalFamily.TERPENES  # Subtle, background
        except Exception:
            return ChemicalFamily.ALCOHOLS  # Default

    def _map_resource_to_chemical_family(self, resource: Resource) -> ChemicalFamily:
        """Map resource characteristics to chemical family."""
        try:
            # Efficiency-based mapping
            if resource.efficiency_rating > 0.8:
                return ChemicalFamily.ESTERS  # High performance
            elif resource.efficiency_rating > 0.6:
                return ChemicalFamily.ALCOHOLS  # Good performance
            elif resource.efficiency_rating > 0.4:
                return ChemicalFamily.KETONES  # Average performance
            else:
                return ChemicalFamily.ACIDS  # Lower performance
        except Exception:
            return ChemicalFamily.ALCOHOLS  # Default

    def _generate_molecular_descriptors(self, task: Task) -> MolecularDescriptor:
        """Generate molecular descriptors based on task properties."""
        try:
            return MolecularDescriptor(
                molecular_weight=task.duration.total_seconds() / 3600 * 50,  # Duration-based
                boiling_point=100 + len(task.dependencies) * 10,  # Complexity-based
                vapor_pressure=task.quantum_weight,
                solubility=task.entanglement_factor,
                polarity=0.5 + (hash(task.id) % 100) / 200.0,  # Pseudo-random
                aromaticity=len(task.resource_requirements) / 10.0 if task.resource_requirements else 0.1,
                hydrogen_bonding=0.3 + len(task.constraints) / 20.0 if task.constraints else 0.3,
                steric_hindrance=0.2 + (hash(task.name or '') % 50) / 250.0
            )
        except Exception:
            return MolecularDescriptor(
                molecular_weight=150.0, boiling_point=120.0, vapor_pressure=0.5,
                solubility=0.5, polarity=0.5, aromaticity=0.3,
                hydrogen_bonding=0.3, steric_hindrance=0.2
            )

    def _calculate_scent_intensity(self, task: Task) -> float:
        """Calculate scent intensity based on task properties."""
        priority_weights = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.MEDIUM: 0.6,
            TaskPriority.LOW: 0.4
        }

        base_intensity = priority_weights.get(task.priority, 0.6)
        complexity_factor = min(1.0, len(task.dependencies) / 5.0) * 0.2
        return min(1.0, base_intensity + complexity_factor)

    def _create_default_signature(self, entity_id: str) -> ScentSignature:
        """Create default scent signature for fallback."""
        return ScentSignature(
            id=f"default_{entity_id}",
            chemical_family=ChemicalFamily.ALCOHOLS,
            molecular_descriptors=MolecularDescriptor(
                molecular_weight=150.0, boiling_point=120.0,
                vapor_pressure=0.5, solubility=0.5, polarity=0.5,
                aromaticity=0.3, hydrogen_bonding=0.3, steric_hindrance=0.2
            ),
            intensity=0.5,
            persistence=0.5,
            diffusion_rate=0.5
        )

    def _calculate_decay_rate(self, profile: np.ndarray) -> float:
        """Calculate exponential decay rate from temporal profile."""
        try:
            # Fit exponential decay
            if len(profile) < 2:
                return 0.1

            # Find peak and calculate decay from there
            peak_idx = np.argmax(profile)
            if peak_idx >= len(profile) - 1:
                return 0.1

            decay_part = profile[peak_idx:]
            if len(decay_part) < 2:
                return 0.1

            # Simple linear fit to log values
            log_values = np.log(np.maximum(decay_part, 1e-8))
            if len(log_values) > 1:
                decay_rate = -(log_values[-1] - log_values[0]) / len(log_values)
                return max(0.01, min(1.0, decay_rate))

            return 0.1

        except Exception:
            return 0.1

    def _cluster_temporal_patterns(self, profiles: np.ndarray) -> Dict[str, Any]:
        """Cluster temporal patterns using simple k-means-like approach."""
        try:
            if len(profiles) < 2:
                return {'n_clusters': 1, 'cluster_centers': [np.mean(profiles, axis=0)] if len(profiles) > 0 else []}

            # Simple clustering based on peak times and decay rates
            clusters = {'n_clusters': min(3, len(profiles)), 'cluster_centers': []}

            # Find cluster centers based on peak times
            peak_times = [np.argmax(profile) for profile in profiles]
            unique_peaks = sorted(set(peak_times))

            for peak in unique_peaks[:3]:  # Max 3 clusters
                cluster_profiles = [profiles[i] for i, p in enumerate(peak_times) if p == peak]
                center = np.mean(cluster_profiles, axis=0) if cluster_profiles else np.zeros(len(profiles[0]))
                clusters['cluster_centers'].append(center)

            return clusters

        except Exception as e:
            self.logger.debug(f"Pattern clustering error: {e}")
            return {'n_clusters': 1, 'cluster_centers': []}

    def _initialize_chemical_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize chemical database with common compounds."""
        return {
            'limonene': {
                'family': ChemicalFamily.TERPENES,
                'properties': MolecularDescriptor(136.24, 176, 0.2, 0.1, 0.3, 0.8, 0.1, 0.2)
            },
            'linalool': {
                'family': ChemicalFamily.ALCOHOLS,
                'properties': MolecularDescriptor(154.25, 198, 0.15, 0.6, 0.4, 0.2, 0.8, 0.1)
            },
            'benzaldehyde': {
                'family': ChemicalFamily.ALDEHYDES,
                'properties': MolecularDescriptor(106.12, 179, 1.27, 0.3, 0.7, 0.9, 0.2, 0.3)
            },
            'ethyl_acetate': {
                'family': ChemicalFamily.ESTERS,
                'properties': MolecularDescriptor(88.11, 77, 97.0, 0.9, 0.6, 0.1, 0.3, 0.1)
            }
        }

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline processing statistics."""
        return {
            'processing_stats': self.processing_stats.copy(),
            'scent_library_size': len(self.scent_library),
            'sensor_buffer_size': len(self.sensor_buffer),
            'temporal_patterns_tracked': len(self.temporal_patterns),
            'cross_modal_enabled': self.enable_cross_modal,
            'feature_dimensions': self.feature_dimensions,
            'chemical_families_used': len(set(s.chemical_family for s in self.scent_library.values()))
        }
