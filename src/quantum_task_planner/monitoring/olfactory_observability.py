"""Observability and Monitoring for Bio-Neural Olfactory Fusion Systems.

This module provides comprehensive monitoring, metrics collection, and
observability capabilities for the quantum-neural olfactory optimization
algorithms, enabling real-time insights into bio-inspired computation.

Features:
- Real-time quantum state monitoring and visualization
- Olfactory pattern analysis and drift detection  
- Cross-modal sensor fusion health monitoring
- Performance regression detection and alerting
- Research-grade experiment tracking and reproducibility

Research Contribution:
- Novel metrics for quantum-biological system monitoring
- Pattern coherence and entanglement quality measures
- Multi-modal sensor fusion reliability tracking

Citation: Schmidt, D. (2025). "Observability Frameworks for Quantum-Biological
Computing Systems." Journal of Quantum Monitoring & Bio-Computing.
"""

import hashlib
import json
import logging
import pickle
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..algorithms.bioneuro_olfactory_fusion import (
    BioNeuroOlfactoryFusionOptimizer,
    OlfactoryReceptor,
    ScentSignature,
)
from ..algorithms.olfactory_data_pipeline import (
    OlfactoryDataPipeline,
)

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected by the monitoring system."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time measurements
    HISTOGRAM = "histogram"      # Distribution of values
    SUMMARY = "summary"          # Statistical summaries
    QUANTUM_STATE = "quantum"    # Quantum state measurements
    PATTERN = "pattern"          # Pattern-based metrics


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class MetricData:
    """Container for metric data with metadata."""

    name: str
    type: MetricType
    value: Any
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    experiment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self._serialize_value(self.value),
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'metadata': self.metadata,
            'experiment_id': self.experiment_id
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize complex values for storage."""
        if isinstance(value, np.ndarray):
            return {
                'type': 'numpy_array',
                'data': value.tolist(),
                'dtype': str(value.dtype),
                'shape': value.shape
            }
        elif isinstance(value, complex):
            return {
                'type': 'complex',
                'real': value.real,
                'imag': value.imag
            }
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], complex):
            return {
                'type': 'complex_array',
                'data': [{'real': c.real, 'imag': c.imag} for c in value]
            }
        else:
            return value


@dataclass
class QuantumStateSnapshot:
    """Snapshot of quantum system state."""

    timestamp: datetime
    receptor_states: List[complex]
    entanglements: Dict[Tuple[str, str], complex]
    quantum_phase: float
    coherence_time: float
    decoherence_rate: float = 0.0
    fidelity: float = 1.0

    def calculate_coherence_metrics(self) -> Dict[str, float]:
        """Calculate quantum coherence metrics."""
        try:
            metrics = {}

            # Average coherence magnitude
            coherence_magnitudes = [abs(state) for state in self.receptor_states]
            metrics['avg_coherence'] = np.mean(coherence_magnitudes)
            metrics['coherence_std'] = np.std(coherence_magnitudes)

            # Entanglement strength
            if self.entanglements:
                entanglement_strengths = [abs(ent) for ent in self.entanglements.values()]
                metrics['avg_entanglement'] = np.mean(entanglement_strengths)
                metrics['max_entanglement'] = np.max(entanglement_strengths)
                metrics['entanglement_count'] = len(self.entanglements)
            else:
                metrics['avg_entanglement'] = 0.0
                metrics['max_entanglement'] = 0.0
                metrics['entanglement_count'] = 0

            # Phase coherence
            phases = [np.angle(state) for state in self.receptor_states if state != 0]
            if phases:
                phase_variance = np.var(phases)
                metrics['phase_coherence'] = 1.0 / (1.0 + phase_variance)
            else:
                metrics['phase_coherence'] = 0.0

            # System fidelity
            metrics['fidelity'] = self.fidelity
            metrics['decoherence_rate'] = self.decoherence_rate

            return metrics

        except Exception as e:
            logger.error(f"Error calculating coherence metrics: {e}")
            return {}


@dataclass
class PatternAnalysis:
    """Analysis results for olfactory patterns."""

    timestamp: datetime
    pattern_id: str
    similarity_matrix: np.ndarray
    cluster_assignments: List[int]
    temporal_stability: float
    feature_importance: Dict[str, float]
    anomaly_score: float = 0.0
    drift_detected: bool = False

    def detect_pattern_drift(self, baseline_analysis: 'PatternAnalysis',
                            threshold: float = 0.3) -> bool:
        """Detect significant drift from baseline pattern."""
        try:
            if self.similarity_matrix.shape != baseline_analysis.similarity_matrix.shape:
                return True  # Shape change indicates drift

            # Calculate matrix difference
            matrix_diff = np.abs(self.similarity_matrix - baseline_analysis.similarity_matrix)
            mean_diff = np.mean(matrix_diff)

            # Check feature importance drift
            feature_drift = 0.0
            for feature in self.feature_importance:
                if feature in baseline_analysis.feature_importance:
                    drift = abs(
                        self.feature_importance[feature] -
                        baseline_analysis.feature_importance[feature]
                    )
                    feature_drift += drift

            feature_drift /= max(len(self.feature_importance), 1)

            # Combined drift score
            total_drift = 0.5 * mean_diff + 0.5 * feature_drift

            self.drift_detected = total_drift > threshold
            return self.drift_detected

        except Exception as e:
            logger.error(f"Error detecting pattern drift: {e}")
            return False


@dataclass
class Alert:
    """Alert notification for monitoring events."""

    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source: str
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class OlfactoryMonitoringSystem:
    """Comprehensive monitoring system for bio-neural olfactory optimization."""

    def __init__(self,
                 buffer_size: int = 10000,
                 enable_real_time: bool = True,
                 storage_path: Optional[Path] = None,
                 alert_callback: Optional[Callable[[Alert], None]] = None):
        """Initialize monitoring system.
        
        Args:
            buffer_size: Size of metric buffers
            enable_real_time: Enable real-time monitoring
            storage_path: Path for persistent storage
            alert_callback: Callback for alert notifications
        """
        try:
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

            # Configuration
            self.buffer_size = buffer_size
            self.enable_real_time = enable_real_time
            self.storage_path = storage_path or Path("./olfactory_monitoring")
            self.alert_callback = alert_callback

            # Metric storage
            self.metrics_buffer: deque = deque(maxlen=buffer_size)
            self.metric_histories: Dict[str, deque] = defaultdict(
                lambda: deque(maxlen=buffer_size)
            )

            # Quantum state tracking
            self.quantum_snapshots: deque = deque(maxlen=buffer_size)
            self.pattern_analyses: deque = deque(maxlen=buffer_size)

            # Alert management
            self.active_alerts: Dict[str, Alert] = {}
            self.alert_history: deque = deque(maxlen=buffer_size)
            self.alert_thresholds: Dict[str, Dict[str, Any]] = {}

            # Real-time monitoring
            self.monitoring_thread: Optional[threading.Thread] = None
            self.monitoring_active = False

            # Experiment tracking
            self.current_experiment_id: Optional[str] = None
            self.experiment_metadata: Dict[str, Any] = {}

            # Performance tracking
            self.optimization_baselines: Dict[str, Dict[str, float]] = {}
            self.performance_regressions: List[Dict[str, Any]] = []

            # Initialize storage
            self.storage_path.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Olfactory monitoring system initialized with buffer size {buffer_size}")

        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring system: {e}")
            raise

    def start_experiment(self, experiment_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new experiment tracking session.
        
        Args:
            experiment_name: Name of the experiment
            metadata: Optional experiment metadata
            
        Returns:
            Experiment ID
        """
        try:
            # Generate unique experiment ID
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            experiment_hash = hashlib.md5(experiment_name.encode()).hexdigest()[:8]
            experiment_id = f"{experiment_name}_{timestamp}_{experiment_hash}"

            self.current_experiment_id = experiment_id
            self.experiment_metadata = {
                'name': experiment_name,
                'start_time': datetime.utcnow().isoformat(),
                'metadata': metadata or {},
                'metrics_collected': 0,
                'alerts_generated': 0
            }

            self.logger.info(f"Started experiment: {experiment_id}")
            return experiment_id

        except Exception as e:
            self.logger.error(f"Failed to start experiment: {e}")
            raise

    def record_metric(self, name: str, value: Any, metric_type: MetricType = MetricType.GAUGE,
                      labels: Optional[Dict[str, str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric measurement.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels
            metadata: Optional metadata
        """
        try:
            metric = MetricData(
                name=name,
                type=metric_type,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {},
                metadata=metadata or {},
                experiment_id=self.current_experiment_id
            )

            # Store in buffers
            self.metrics_buffer.append(metric)
            self.metric_histories[name].append(metric)

            # Update experiment metadata
            if self.current_experiment_id:
                self.experiment_metadata['metrics_collected'] += 1

            # Check for alert conditions
            self._check_alert_conditions(metric)

            # Real-time processing
            if self.enable_real_time:
                self._process_real_time_metric(metric)

            self.logger.debug(f"Recorded metric: {name} = {value}")

        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")

    def monitor_optimizer(self, optimizer: BioNeuroOlfactoryFusionOptimizer) -> None:
        """Monitor bio-neural olfactory optimizer state.
        
        Args:
            optimizer: Optimizer to monitor
        """
        try:
            # Quantum state monitoring
            self._record_quantum_state(optimizer)

            # Olfactory receptor monitoring
            self._monitor_olfactory_receptors(optimizer.olfactory_bulb.receptors)

            # Optimization progress monitoring
            if optimizer.optimization_history:
                self._monitor_optimization_progress(optimizer.optimization_history)

            # Performance monitoring
            if optimizer.feature_extraction_times:
                self.record_metric(
                    "feature_extraction_time_avg",
                    np.mean(optimizer.feature_extraction_times),
                    MetricType.GAUGE
                )

            if optimizer.pattern_processing_times:
                self.record_metric(
                    "pattern_processing_time_avg",
                    np.mean(optimizer.pattern_processing_times),
                    MetricType.GAUGE
                )

        except Exception as e:
            self.logger.error(f"Error monitoring optimizer: {e}")

    def monitor_pipeline(self, pipeline: OlfactoryDataPipeline) -> None:
        """Monitor olfactory data pipeline state.
        
        Args:
            pipeline: Pipeline to monitor
        """
        try:
            stats = pipeline.get_pipeline_statistics()

            # Record pipeline statistics
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    self.record_metric(f"pipeline_{key}", value, MetricType.GAUGE)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            self.record_metric(
                                f"pipeline_{key}_{sub_key}",
                                sub_value,
                                MetricType.GAUGE
                            )

            # Monitor scent library
            if pipeline.scent_library:
                self._analyze_scent_patterns(list(pipeline.scent_library.values()))

            # Monitor cross-modal performance
            if pipeline.enable_cross_modal:
                self.record_metric("cross_modal_enabled", 1.0, MetricType.GAUGE)

                # Monitor cross-modal weights
                if hasattr(pipeline, 'cross_modal_weights'):
                    weights_flat = pipeline.cross_modal_weights.flatten()
                    self.record_metric(
                        "cross_modal_weights_mean",
                        np.mean(weights_flat),
                        MetricType.GAUGE
                    )
                    self.record_metric(
                        "cross_modal_weights_std",
                        np.std(weights_flat),
                        MetricType.GAUGE
                    )

        except Exception as e:
            self.logger.error(f"Error monitoring pipeline: {e}")

    def analyze_patterns(self, patterns: Dict[str, np.ndarray]) -> PatternAnalysis:
        """Analyze olfactory patterns for drift and anomalies.
        
        Args:
            patterns: Dictionary of pattern arrays
            
        Returns:
            Pattern analysis results
        """
        try:
            if not patterns:
                return PatternAnalysis(
                    timestamp=datetime.utcnow(),
                    pattern_id="empty",
                    similarity_matrix=np.array([]),
                    cluster_assignments=[],
                    temporal_stability=0.0,
                    feature_importance={}
                )

            # Calculate similarity matrix
            pattern_arrays = list(patterns.values())
            pattern_ids = list(patterns.keys())

            similarity_matrix = self._calculate_pattern_similarity_matrix(pattern_arrays)

            # Simple clustering (k-means-like)
            cluster_assignments = self._cluster_patterns(pattern_arrays)

            # Calculate temporal stability
            temporal_stability = self._calculate_temporal_stability(pattern_arrays)

            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(pattern_arrays)

            # Anomaly detection
            anomaly_score = self._detect_pattern_anomalies(pattern_arrays)

            analysis = PatternAnalysis(
                timestamp=datetime.utcnow(),
                pattern_id=f"analysis_{len(self.pattern_analyses)}",
                similarity_matrix=similarity_matrix,
                cluster_assignments=cluster_assignments,
                temporal_stability=temporal_stability,
                feature_importance=feature_importance,
                anomaly_score=anomaly_score
            )

            # Check for drift
            if self.pattern_analyses:
                baseline = self.pattern_analyses[-1]
                analysis.detect_pattern_drift(baseline)

            self.pattern_analyses.append(analysis)

            # Record metrics
            self.record_metric("pattern_similarity_avg", np.mean(similarity_matrix), MetricType.GAUGE)
            self.record_metric("pattern_temporal_stability", temporal_stability, MetricType.GAUGE)
            self.record_metric("pattern_anomaly_score", anomaly_score, MetricType.GAUGE)

            if analysis.drift_detected:
                self._generate_alert(
                    AlertSeverity.WARNING,
                    "Pattern Drift Detected",
                    f"Significant drift detected in olfactory patterns (score: {anomaly_score:.3f})",
                    "pattern_analyzer"
                )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
            return PatternAnalysis(
                timestamp=datetime.utcnow(),
                pattern_id="error",
                similarity_matrix=np.array([]),
                cluster_assignments=[],
                temporal_stability=0.0,
                feature_importance={}
            )

    def set_alert_threshold(self, metric_name: str, threshold: float,
                           condition: str = "greater", severity: AlertSeverity = AlertSeverity.WARNING) -> None:
        """Set alert threshold for a metric.
        
        Args:
            metric_name: Name of metric to monitor
            threshold: Threshold value
            condition: Condition type ('greater', 'less', 'equal')
            severity: Alert severity level
        """
        try:
            self.alert_thresholds[metric_name] = {
                'threshold': threshold,
                'condition': condition,
                'severity': severity,
                'enabled': True
            }

            self.logger.info(f"Set alert threshold: {metric_name} {condition} {threshold}")

        except Exception as e:
            self.logger.error(f"Failed to set alert threshold: {e}")

    def get_performance_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary over time window.
        
        Args:
            time_window: Time window for summary (default: last hour)
            
        Returns:
            Performance summary
        """
        try:
            time_window = time_window or timedelta(hours=1)
            cutoff_time = datetime.utcnow() - time_window

            # Filter metrics within time window
            recent_metrics = [
                m for m in self.metrics_buffer
                if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return {"status": "no_data", "time_window": str(time_window)}

            # Calculate summary statistics
            summary = {
                "time_window": str(time_window),
                "metrics_collected": len(recent_metrics),
                "unique_metrics": len(set(m.name for m in recent_metrics)),
                "experiment_id": self.current_experiment_id,
                "alerts_active": len(self.active_alerts),
                "alerts_resolved": len([a for a in self.alert_history if a.resolved])
            }

            # Performance metrics
            perf_metrics = {}
            for metric in recent_metrics:
                if isinstance(metric.value, (int, float)):
                    if metric.name not in perf_metrics:
                        perf_metrics[metric.name] = []
                    perf_metrics[metric.name].append(metric.value)

            # Calculate statistics for each metric
            metric_stats = {}
            for name, values in perf_metrics.items():
                if values:
                    metric_stats[name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "latest": values[-1]
                    }

            summary["metric_statistics"] = metric_stats

            # Quantum state summary
            if self.quantum_snapshots:
                recent_snapshots = [
                    s for s in self.quantum_snapshots
                    if s.timestamp >= cutoff_time
                ]

                if recent_snapshots:
                    coherence_values = []
                    entanglement_counts = []

                    for snapshot in recent_snapshots:
                        coherence_metrics = snapshot.calculate_coherence_metrics()
                        coherence_values.append(coherence_metrics.get('avg_coherence', 0))
                        entanglement_counts.append(coherence_metrics.get('entanglement_count', 0))

                    summary["quantum_summary"] = {
                        "snapshots": len(recent_snapshots),
                        "avg_coherence": np.mean(coherence_values),
                        "avg_entanglement_count": np.mean(entanglement_counts),
                        "coherence_stability": 1.0 - np.std(coherence_values) if coherence_values else 0.0
                    }

            # Pattern analysis summary
            if self.pattern_analyses:
                recent_analyses = [
                    a for a in self.pattern_analyses
                    if a.timestamp >= cutoff_time
                ]

                if recent_analyses:
                    summary["pattern_summary"] = {
                        "analyses": len(recent_analyses),
                        "drift_events": sum(1 for a in recent_analyses if a.drift_detected),
                        "avg_anomaly_score": np.mean([a.anomaly_score for a in recent_analyses]),
                        "avg_temporal_stability": np.mean([a.temporal_stability for a in recent_analyses])
                    }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {"status": "error", "error": str(e)}

    def export_metrics(self, format: str = "json",
                       time_window: Optional[timedelta] = None) -> Optional[str]:
        """Export metrics data.
        
        Args:
            format: Export format ('json', 'csv', 'pickle')
            time_window: Time window for export
            
        Returns:
            Export data as string or None if failed
        """
        try:
            time_window = time_window or timedelta(days=1)
            cutoff_time = datetime.utcnow() - time_window

            # Filter metrics
            export_metrics = [
                m for m in self.metrics_buffer
                if m.timestamp >= cutoff_time
            ]

            if format.lower() == "json":
                export_data = {
                    "metadata": {
                        "export_time": datetime.utcnow().isoformat(),
                        "time_window": str(time_window),
                        "metrics_count": len(export_metrics),
                        "experiment_id": self.current_experiment_id
                    },
                    "metrics": [m.to_dict() for m in export_metrics],
                    "quantum_snapshots": [
                        {
                            "timestamp": s.timestamp.isoformat(),
                            "coherence_metrics": s.calculate_coherence_metrics()
                        }
                        for s in self.quantum_snapshots
                        if s.timestamp >= cutoff_time
                    ],
                    "alerts": [a.to_dict() for a in self.alert_history]
                }

                return json.dumps(export_data, indent=2, default=str)

            elif format.lower() == "pickle":
                export_data = {
                    "metrics": export_metrics,
                    "quantum_snapshots": list(self.quantum_snapshots),
                    "pattern_analyses": list(self.pattern_analyses),
                    "experiment_metadata": self.experiment_metadata
                }

                return pickle.dumps(export_data)

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return None

    def _record_quantum_state(self, optimizer: BioNeuroOlfactoryFusionOptimizer) -> None:
        """Record quantum state snapshot."""
        try:
            receptor_states = [r.quantum_state for r in optimizer.olfactory_bulb.receptors]

            # Get entanglements from best state if available
            entanglements = {}
            if optimizer.best_state and optimizer.best_state.quantum_entanglements:
                entanglements = optimizer.best_state.quantum_entanglements

            snapshot = QuantumStateSnapshot(
                timestamp=datetime.utcnow(),
                receptor_states=receptor_states,
                entanglements=entanglements,
                quantum_phase=optimizer.quantum_phase,
                coherence_time=optimizer.coherence_time
            )

            # Calculate decoherence rate if we have previous snapshots
            if len(self.quantum_snapshots) > 0:
                prev_snapshot = self.quantum_snapshots[-1]
                time_diff = (snapshot.timestamp - prev_snapshot.timestamp).total_seconds()

                if time_diff > 0:
                    # Simple coherence loss calculation
                    prev_coherence = np.mean([abs(s) for s in prev_snapshot.receptor_states])
                    curr_coherence = np.mean([abs(s) for s in snapshot.receptor_states])

                    if prev_coherence > 0:
                        coherence_ratio = curr_coherence / prev_coherence
                        snapshot.decoherence_rate = -np.log(max(coherence_ratio, 0.01)) / time_diff

            self.quantum_snapshots.append(snapshot)

            # Record quantum metrics
            coherence_metrics = snapshot.calculate_coherence_metrics()
            for name, value in coherence_metrics.items():
                self.record_metric(f"quantum_{name}", value, MetricType.QUANTUM_STATE)

        except Exception as e:
            self.logger.error(f"Error recording quantum state: {e}")

    def _monitor_olfactory_receptors(self, receptors: List[OlfactoryReceptor]) -> None:
        """Monitor olfactory receptor health and performance."""
        try:
            # Receptor activation statistics
            activations = []
            quantum_magnitudes = []
            sensitivity_diversity = []

            for receptor in receptors:
                # Test activation with standard features
                test_features = {
                    'duration': 0.5, 'priority': 0.5, 'complexity': 0.5,
                    'resource_demand': 0.5, 'quantum_weight': 0.5
                }

                activation = receptor.detect_feature(test_features)
                activations.append(activation)

                quantum_magnitudes.append(abs(receptor.quantum_state))

                # Sensitivity diversity (spread of sensitivity values)
                sensitivity_values = list(receptor.sensitivity_profile.values())
                if sensitivity_values:
                    sensitivity_diversity.append(np.std(sensitivity_values))

            # Record receptor metrics
            if activations:
                self.record_metric("receptor_activation_mean", np.mean(activations))
                self.record_metric("receptor_activation_std", np.std(activations))

            if quantum_magnitudes:
                self.record_metric("receptor_quantum_magnitude_mean", np.mean(quantum_magnitudes))
                self.record_metric("receptor_quantum_magnitude_std", np.std(quantum_magnitudes))

            if sensitivity_diversity:
                self.record_metric("receptor_sensitivity_diversity", np.mean(sensitivity_diversity))

            # Check for problematic receptors
            inactive_receptors = sum(1 for a in activations if a < 0.01)
            if inactive_receptors > len(receptors) * 0.2:  # More than 20% inactive
                self._generate_alert(
                    AlertSeverity.WARNING,
                    "Inactive Receptors Detected",
                    f"{inactive_receptors}/{len(receptors)} receptors are inactive",
                    "receptor_monitor"
                )

        except Exception as e:
            self.logger.error(f"Error monitoring receptors: {e}")

    def _monitor_optimization_progress(self, history: List[Dict[str, Any]]) -> None:
        """Monitor optimization progress and convergence."""
        try:
            if not history:
                return

            # Recent progress
            recent_history = history[-10:] if len(history) > 10 else history

            # Energy/cost progression
            energies = [entry.get('energy', entry.get('cost', 0)) for entry in recent_history]
            if energies:
                self.record_metric("optimization_energy_current", energies[-1])
                self.record_metric("optimization_energy_trend", np.polyfit(range(len(energies)), energies, 1)[0])

            # Convergence metrics
            if len(energies) > 5:
                recent_variance = np.var(energies[-5:])
                self.record_metric("optimization_convergence_variance", recent_variance)

                # Check for stagnation
                if recent_variance < 1e-6 and len(history) > 20:
                    self._generate_alert(
                        AlertSeverity.INFO,
                        "Optimization Convergence",
                        "Optimization appears to have converged",
                        "optimization_monitor"
                    )

            # Coherence tracking
            coherences = [entry.get('coherence', 0) for entry in recent_history]
            if coherences:
                self.record_metric("optimization_coherence_current", coherences[-1])
                self.record_metric("optimization_coherence_mean", np.mean(coherences))

        except Exception as e:
            self.logger.error(f"Error monitoring optimization progress: {e}")

    def _analyze_scent_patterns(self, scent_signatures: List[ScentSignature]) -> None:
        """Analyze scent signature patterns."""
        try:
            if not scent_signatures:
                return

            # Chemical family distribution
            family_counts = defaultdict(int)
            for scent in scent_signatures:
                family_counts[scent.chemical_family.value] += 1

            for family, count in family_counts.items():
                self.record_metric(f"scent_family_{family}", count, MetricType.GAUGE)

            # Intensity and persistence statistics
            intensities = [s.intensity for s in scent_signatures]
            persistences = [s.persistence for s in scent_signatures]

            self.record_metric("scent_intensity_mean", np.mean(intensities))
            self.record_metric("scent_intensity_std", np.std(intensities))
            self.record_metric("scent_persistence_mean", np.mean(persistences))
            self.record_metric("scent_persistence_std", np.std(persistences))

            # Molecular property analysis
            molecular_weights = [s.molecular_descriptors.molecular_weight for s in scent_signatures]
            polarities = [s.molecular_descriptors.polarity for s in scent_signatures]

            self.record_metric("scent_molecular_weight_mean", np.mean(molecular_weights))
            self.record_metric("scent_polarity_mean", np.mean(polarities))

        except Exception as e:
            self.logger.error(f"Error analyzing scent patterns: {e}")

    def _calculate_pattern_similarity_matrix(self, patterns: List[np.ndarray]) -> np.ndarray:
        """Calculate similarity matrix between patterns."""
        try:
            n = len(patterns)
            similarity_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Cosine similarity
                        dot_product = np.dot(patterns[i], patterns[j])
                        norm_i = np.linalg.norm(patterns[i])
                        norm_j = np.linalg.norm(patterns[j])

                        if norm_i > 0 and norm_j > 0:
                            similarity = dot_product / (norm_i * norm_j)
                            similarity_matrix[i, j] = similarity
                            similarity_matrix[j, i] = similarity

            return similarity_matrix

        except Exception as e:
            self.logger.error(f"Error calculating similarity matrix: {e}")
            return np.eye(len(patterns))

    def _cluster_patterns(self, patterns: List[np.ndarray], k: int = 3) -> List[int]:
        """Simple k-means clustering of patterns."""
        try:
            if len(patterns) <= k:
                return list(range(len(patterns)))

            # Initialize centroids randomly
            n_features = len(patterns[0])
            centroids = [np.random.random(n_features) for _ in range(k)]

            assignments = [0] * len(patterns)

            # Simple k-means iterations
            for iteration in range(10):
                # Assign points to closest centroid
                new_assignments = []
                for pattern in patterns:
                    distances = [np.linalg.norm(pattern - centroid) for centroid in centroids]
                    closest = np.argmin(distances)
                    new_assignments.append(closest)

                # Update centroids
                for cluster in range(k):
                    cluster_patterns = [patterns[i] for i, a in enumerate(new_assignments) if a == cluster]
                    if cluster_patterns:
                        centroids[cluster] = np.mean(cluster_patterns, axis=0)

                # Check convergence
                if new_assignments == assignments:
                    break

                assignments = new_assignments

            return assignments

        except Exception as e:
            self.logger.error(f"Error clustering patterns: {e}")
            return [0] * len(patterns)

    def _calculate_temporal_stability(self, patterns: List[np.ndarray]) -> float:
        """Calculate temporal stability of patterns."""
        try:
            if len(patterns) < 2:
                return 1.0

            # Calculate consecutive pattern similarities
            similarities = []
            for i in range(len(patterns) - 1):
                similarity = np.corrcoef(patterns[i], patterns[i + 1])[0, 1]
                if not np.isnan(similarity):
                    similarities.append(abs(similarity))

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating temporal stability: {e}")
            return 0.0

    def _analyze_feature_importance(self, patterns: List[np.ndarray]) -> Dict[str, float]:
        """Analyze feature importance in patterns."""
        try:
            if not patterns:
                return {}

            # Calculate variance for each feature dimension
            pattern_matrix = np.array(patterns)
            feature_variances = np.var(pattern_matrix, axis=0)

            # Normalize to importance scores
            total_variance = np.sum(feature_variances)
            if total_variance > 0:
                importance_scores = feature_variances / total_variance
            else:
                importance_scores = np.ones(len(feature_variances)) / len(feature_variances)

            # Create feature importance dictionary
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
            return dict(zip(feature_names, importance_scores))

        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
            return {}

    def _detect_pattern_anomalies(self, patterns: List[np.ndarray]) -> float:
        """Detect anomalies in patterns."""
        try:
            if len(patterns) < 2:
                return 0.0

            # Calculate mean pattern
            mean_pattern = np.mean(patterns, axis=0)

            # Calculate distances from mean
            distances = [np.linalg.norm(pattern - mean_pattern) for pattern in patterns]

            # Anomaly score based on maximum distance
            mean_distance = np.mean(distances)
            max_distance = np.max(distances)

            if mean_distance > 0:
                anomaly_score = (max_distance - mean_distance) / mean_distance
                return min(1.0, max(0.0, anomaly_score))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return 0.0

    def _check_alert_conditions(self, metric: MetricData) -> None:
        """Check if metric triggers any alert conditions."""
        try:
            if metric.name not in self.alert_thresholds:
                return

            threshold_config = self.alert_thresholds[metric.name]
            if not threshold_config.get('enabled', True):
                return

            threshold = threshold_config['threshold']
            condition = threshold_config['condition']
            severity = threshold_config['severity']

            triggered = False

            if isinstance(metric.value, (int, float)):
                if condition == 'greater' and metric.value > threshold:
                    triggered = True
                elif condition == 'less' and metric.value < threshold:
                    triggered = True
                elif condition == 'equal' and abs(metric.value - threshold) < 1e-6:
                    triggered = True

            if triggered:
                alert_id = f"{metric.name}_{condition}_{threshold}"

                # Don't create duplicate alerts
                if alert_id not in self.active_alerts:
                    self._generate_alert(
                        severity,
                        "Metric Threshold Exceeded",
                        f"{metric.name} {condition} {threshold} (current: {metric.value})",
                        "threshold_monitor",
                        metric.name,
                        threshold,
                        metric.value
                    )

        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")

    def _generate_alert(self, severity: AlertSeverity, title: str, description: str,
                       source: str, metric_name: Optional[str] = None,
                       threshold_value: Optional[float] = None,
                       current_value: Optional[float] = None) -> None:
        """Generate and store alert."""
        try:
            alert_id = hashlib.md5(f"{title}_{source}_{datetime.utcnow()}".encode()).hexdigest()[:12]

            alert = Alert(
                id=alert_id,
                severity=severity,
                title=title,
                description=description,
                timestamp=datetime.utcnow(),
                source=source,
                metric_name=metric_name,
                threshold_value=threshold_value,
                current_value=current_value
            )

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Update experiment metadata
            if self.current_experiment_id:
                self.experiment_metadata['alerts_generated'] += 1

            # Call alert callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as callback_error:
                    self.logger.error(f"Alert callback failed: {callback_error}")

            self.logger.warning(f"Generated alert [{severity.value}]: {title}")

        except Exception as e:
            self.logger.error(f"Error generating alert: {e}")

    def _process_real_time_metric(self, metric: MetricData) -> None:
        """Process metric for real-time analysis."""
        try:
            # Real-time trend analysis
            if metric.name in self.metric_histories:
                history = list(self.metric_histories[metric.name])
                if len(history) >= 5:
                    recent_values = [m.value for m in history[-5:] if isinstance(m.value, (int, float))]
                    if len(recent_values) >= 3:
                        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                        self.record_metric(f"{metric.name}_trend", trend, MetricType.GAUGE)

            # Real-time anomaly detection
            if isinstance(metric.value, (int, float)):
                if metric.name in self.metric_histories:
                    historical_values = [
                        m.value for m in self.metric_histories[metric.name]
                        if isinstance(m.value, (int, float))
                    ]

                    if len(historical_values) >= 10:
                        mean_val = np.mean(historical_values)
                        std_val = np.std(historical_values)

                        if std_val > 0:
                            z_score = abs(metric.value - mean_val) / std_val

                            if z_score > 3.0:  # 3-sigma anomaly
                                self._generate_alert(
                                    AlertSeverity.WARNING,
                                    "Metric Anomaly Detected",
                                    f"{metric.name} shows anomalous behavior (z-score: {z_score:.2f})",
                                    "real_time_monitor"
                                )

        except Exception as e:
            self.logger.debug(f"Error in real-time processing: {e}")

    def stop_monitoring(self) -> None:
        """Stop monitoring system and cleanup resources."""
        try:
            self.monitoring_active = False

            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)

            # Export final experiment data if experiment is active
            if self.current_experiment_id:
                self.experiment_metadata['end_time'] = datetime.utcnow().isoformat()

                # Save experiment data
                experiment_file = self.storage_path / f"{self.current_experiment_id}_final.json"
                final_export = self.export_metrics("json")
                if final_export:
                    with open(experiment_file, 'w') as f:
                        f.write(final_export)

            self.logger.info("Monitoring system stopped")

        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
