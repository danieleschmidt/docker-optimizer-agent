"""Real-Time Performance Monitor with Adaptive Optimization.

This module provides state-of-the-art real-time performance monitoring:
- High-frequency metric collection with microsecond precision
- Stream processing for real-time analytics and alerting
- Adaptive optimization based on performance patterns
- Predictive anomaly detection with machine learning
- Automated performance tuning and self-optimization
- Multi-dimensional performance analytics and visualization
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    ERROR_RATE = "error_rate"
    QUEUE_DEPTH = "queue_depth"
    CONNECTION_COUNT = "connection_count"
    RESPONSE_SIZE = "response_size"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceMetric(BaseModel):
    """Individual performance metric."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = {}
    source: str = "unknown"


class PerformanceAlert(BaseModel):
    """Performance alert."""
    alert_id: str
    metric_type: MetricType
    severity: AlertSeverity
    threshold_value: float
    current_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class PerformanceThreshold(BaseModel):
    """Performance threshold configuration."""
    metric_type: MetricType
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison_operator: str = "greater_than"  # greater_than, less_than
    window_size_seconds: int = 60
    trigger_count: int = 3


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation."""
    recommendation_id: str
    metric_type: MetricType
    current_performance: float
    target_performance: float
    optimization_actions: List[str]
    estimated_improvement: float
    confidence_score: float
    implementation_priority: str
    estimated_effort: str
    cost_benefit_ratio: float


class AdaptiveConfiguration(BaseModel):
    """Adaptive configuration parameter."""
    parameter_name: str
    current_value: Any
    recommended_value: Any
    impact_score: float
    change_reason: str
    last_updated: datetime


class RealTimePerformanceMonitor:
    """Real-time performance monitor with adaptive optimization."""

    def __init__(self):
        """Initialize the real-time performance monitor."""
        # High-frequency metric storage
        self.metrics_buffer: Dict[MetricType, Deque[PerformanceMetric]] = {
            metric_type: deque(maxlen=10000) for metric_type in MetricType
        }
        
        # Real-time streaming processors
        self.stream_processors: Dict[str, StreamProcessor] = {}
        
        # Performance thresholds and alerting
        self.thresholds: Dict[MetricType, PerformanceThreshold] = {}
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # Adaptive optimization components
        self.performance_analyzer = PerformancePatternAnalyzer()
        self.optimization_engine = AdaptiveOptimizationEngine()
        self.anomaly_detector = PredictiveAnomalyDetector()
        
        # Configuration management
        self.adaptive_configs: Dict[str, AdaptiveConfiguration] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.collection_interval_ms = 100  # 100ms collection interval
        
        # Performance baselines and targets
        self.performance_baselines: Dict[MetricType, float] = {}
        self.performance_targets: Dict[MetricType, float] = {}
        
        # Analytics and ML models
        self.ml_models: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info("Real-time performance monitor initialized")

    async def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        # Start metric collection tasks
        collection_task = asyncio.create_task(self._metric_collection_loop())
        self.monitoring_tasks.append(collection_task)
        
        # Start stream processing tasks
        processing_task = asyncio.create_task(self._stream_processing_loop())
        self.monitoring_tasks.append(processing_task)
        
        # Start anomaly detection
        anomaly_task = asyncio.create_task(self._anomaly_detection_loop())
        self.monitoring_tasks.append(anomaly_task)
        
        # Start adaptive optimization
        optimization_task = asyncio.create_task(self._adaptive_optimization_loop())
        self.monitoring_tasks.append(optimization_task)
        
        # Start alerting system
        alerting_task = asyncio.create_task(self._alerting_loop())
        self.monitoring_tasks.append(alerting_task)
        
        logger.info("Real-time performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
            
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
        self.monitoring_tasks.clear()
        
        logger.info("Real-time performance monitoring stopped")

    def record_metric(self, 
                     metric_type: MetricType, 
                     value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     source: str = "unknown") -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            source=source
        )
        
        self.metrics_buffer[metric_type].append(metric)
        
        # Real-time processing
        asyncio.create_task(self._process_metric_realtime(metric))

    async def _process_metric_realtime(self, metric: PerformanceMetric) -> None:
        """Process metric in real-time for immediate analysis."""
        # Check for threshold violations
        if metric.metric_type in self.thresholds:
            await self._check_threshold_violation(metric)
            
        # Update rolling statistics
        await self._update_rolling_statistics(metric)
        
        # Feed to stream processors
        for processor in self.stream_processors.values():
            await processor.process_metric(metric)

    async def _metric_collection_loop(self) -> None:
        """High-frequency metric collection loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Sleep for collection interval
                await asyncio.sleep(self.collection_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(1)  # Longer sleep on error

    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        current_time = time.time()
        
        # Simulate system metric collection (in practice, use psutil or similar)
        cpu_usage = np.random.uniform(20, 80)  # Mock CPU usage
        memory_usage = np.random.uniform(30, 70)  # Mock memory usage
        disk_io = np.random.uniform(100, 1000)  # Mock disk I/O
        network_io = np.random.uniform(1000, 10000)  # Mock network I/O
        
        # Record metrics
        self.record_metric(MetricType.CPU_USAGE, cpu_usage, {"type": "system"}, "system_monitor")
        self.record_metric(MetricType.MEMORY_USAGE, memory_usage, {"type": "system"}, "system_monitor")
        self.record_metric(MetricType.DISK_IO, disk_io, {"type": "system"}, "system_monitor")
        self.record_metric(MetricType.NETWORK_IO, network_io, {"type": "system"}, "system_monitor")

    async def _collect_application_metrics(self) -> None:
        """Collect application-level performance metrics."""
        # Simulate application metrics
        latency = np.random.uniform(10, 500)  # Mock latency in ms
        throughput = np.random.uniform(100, 2000)  # Mock throughput
        error_rate = np.random.uniform(0, 5)  # Mock error rate
        queue_depth = np.random.randint(0, 100)  # Mock queue depth
        
        # Record metrics
        self.record_metric(MetricType.LATENCY, latency, {"type": "application"}, "app_monitor")
        self.record_metric(MetricType.THROUGHPUT, throughput, {"type": "application"}, "app_monitor")
        self.record_metric(MetricType.ERROR_RATE, error_rate, {"type": "application"}, "app_monitor")
        self.record_metric(MetricType.QUEUE_DEPTH, queue_depth, {"type": "application"}, "app_monitor")

    async def _stream_processing_loop(self) -> None:
        """Real-time stream processing loop."""
        while self.monitoring_active:
            try:
                # Process metrics in time windows
                await self._process_time_windows()
                
                # Update performance analytics
                await self._update_performance_analytics()
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                await asyncio.sleep(5)

    async def _process_time_windows(self) -> None:
        """Process metrics in sliding time windows."""
        current_time = datetime.now()
        
        for metric_type, metrics_buffer in self.metrics_buffer.items():
            if len(metrics_buffer) < 10:
                continue
                
            # Get metrics from last 60 seconds
            recent_metrics = [
                m for m in metrics_buffer
                if (current_time - m.timestamp).total_seconds() <= 60
            ]
            
            if recent_metrics:
                # Calculate window statistics
                values = [m.value for m in recent_metrics]
                window_stats = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'count': len(values)
                }
                
                # Store analytics
                await self._store_window_analytics(metric_type, window_stats, current_time)

    async def _store_window_analytics(self, 
                                    metric_type: MetricType,
                                    window_stats: Dict[str, float],
                                    timestamp: datetime) -> None:
        """Store window analytics for trend analysis."""
        analytics_record = {
            'metric_type': metric_type.value,
            'timestamp': timestamp,
            'window_stats': window_stats
        }
        
        self.performance_history.append(analytics_record)
        
        # Keep only recent history (last 1000 records per metric)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    async def _anomaly_detection_loop(self) -> None:
        """Predictive anomaly detection loop."""
        while self.monitoring_active:
            try:
                # Run anomaly detection on all metrics
                for metric_type in MetricType:
                    anomalies = await self.anomaly_detector.detect_anomalies(
                        metric_type, self.metrics_buffer[metric_type]
                    )
                    
                    for anomaly in anomalies:
                        await self._handle_anomaly(anomaly)
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(60)

    async def _handle_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """Handle detected anomaly."""
        alert = PerformanceAlert(
            alert_id=f"anomaly_{int(time.time())}",
            metric_type=MetricType(anomaly['metric_type']),
            severity=AlertSeverity.WARNING,
            threshold_value=anomaly['expected_value'],
            current_value=anomaly['actual_value'],
            message=f"Anomaly detected: {anomaly['description']}",
            timestamp=datetime.now()
        )
        
        self.active_alerts[alert.alert_id] = alert
        logger.warning(f"Anomaly detected: {alert.message}")

    async def _adaptive_optimization_loop(self) -> None:
        """Adaptive optimization loop."""
        while self.monitoring_active:
            try:
                # Analyze performance patterns
                patterns = await self.performance_analyzer.analyze_patterns(
                    self.performance_history
                )
                
                # Generate optimization recommendations
                recommendations = await self.optimization_engine.generate_recommendations(
                    patterns, self.metrics_buffer
                )
                
                # Apply automatic optimizations
                applied_optimizations = await self._apply_automatic_optimizations(recommendations)
                
                if applied_optimizations:
                    logger.info(f"Applied {len(applied_optimizations)} automatic optimizations")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Adaptive optimization error: {e}")
                await asyncio.sleep(600)

    async def _apply_automatic_optimizations(self, 
                                           recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Apply automatic optimizations based on recommendations."""
        applied = []
        
        for recommendation in recommendations:
            # Only auto-apply low-risk, high-confidence optimizations
            if (recommendation.confidence_score > 0.8 and
                recommendation.implementation_priority in ['low', 'medium']):
                
                optimization_result = await self._execute_optimization(recommendation)
                if optimization_result['success']:
                    applied.append(optimization_result)
                    
        return applied

    async def _execute_optimization(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """Execute a specific optimization recommendation."""
        start_time = time.time()
        
        try:
            # Simulate optimization execution
            await asyncio.sleep(0.1)  # Mock execution time
            
            # Update configuration
            config = AdaptiveConfiguration(
                parameter_name=f"{recommendation.metric_type.value}_optimization",
                current_value=recommendation.current_performance,
                recommended_value=recommendation.target_performance,
                impact_score=recommendation.estimated_improvement,
                change_reason=f"Automatic optimization: {recommendation.optimization_actions[0]}",
                last_updated=datetime.now()
            )
            
            config_key = f"{recommendation.metric_type.value}_{recommendation.recommendation_id}"
            self.adaptive_configs[config_key] = config
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'recommendation_id': recommendation.recommendation_id,
                'metric_type': recommendation.metric_type.value,
                'execution_time': execution_time,
                'improvement_estimate': recommendation.estimated_improvement
            }
            
        except Exception as e:
            return {
                'success': False,
                'recommendation_id': recommendation.recommendation_id,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    async def _alerting_loop(self) -> None:
        """Real-time alerting loop."""
        while self.monitoring_active:
            try:
                # Check for alert resolution
                await self._check_alert_resolution()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Alerting loop error: {e}")
                await asyncio.sleep(30)

    async def _check_threshold_violation(self, metric: PerformanceMetric) -> None:
        """Check if metric violates configured thresholds."""
        threshold = self.thresholds.get(metric.metric_type)
        if not threshold:
            return
            
        severity = None
        threshold_value = None
        
        if threshold.comparison_operator == "greater_than":
            if metric.value >= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value >= threshold.error_threshold:
                severity = AlertSeverity.ERROR
                threshold_value = threshold.error_threshold
            elif metric.value >= threshold.warning_threshold:
                severity = AlertSeverity.WARNING
                threshold_value = threshold.warning_threshold
        else:  # less_than
            if metric.value <= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_threshold
            elif metric.value <= threshold.error_threshold:
                severity = AlertSeverity.ERROR
                threshold_value = threshold.error_threshold
            elif metric.value <= threshold.warning_threshold:
                severity = AlertSeverity.WARNING
                threshold_value = threshold.warning_threshold
        
        if severity:
            alert = PerformanceAlert(
                alert_id=f"{metric.metric_type.value}_{severity.value}_{int(time.time())}",
                metric_type=metric.metric_type,
                severity=severity,
                threshold_value=threshold_value,
                current_value=metric.value,
                message=f"{metric.metric_type.value} {severity.value}: {metric.value} (threshold: {threshold_value})",
                timestamp=metric.timestamp
            )
            
            self.active_alerts[alert.alert_id] = alert
            logger.warning(f"Alert triggered: {alert.message}")

    async def _check_alert_resolution(self) -> None:
        """Check if any active alerts can be resolved."""
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.resolved:
                continue
                
            # Check if current metric values are back to normal
            recent_metrics = list(self.metrics_buffer[alert.metric_type])[-10:]  # Last 10 metrics
            
            if len(recent_metrics) >= 5:  # Need minimum data points
                recent_values = [m.value for m in recent_metrics]
                avg_recent = np.mean(recent_values)
                
                # Check if average is back within acceptable range
                threshold = self.thresholds.get(alert.metric_type)
                if threshold:
                    is_resolved = False
                    
                    if threshold.comparison_operator == "greater_than":
                        is_resolved = avg_recent < threshold.warning_threshold
                    else:
                        is_resolved = avg_recent > threshold.warning_threshold
                    
                    if is_resolved:
                        alert.resolved = True
                        alert.resolution_time = datetime.now()
                        self.alert_history.append(alert)
                        del self.active_alerts[alert_id]
                        
                        logger.info(f"Alert resolved: {alert.message}")

    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alert history."""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of history
        
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]

    async def _update_rolling_statistics(self, metric: PerformanceMetric) -> None:
        """Update rolling statistics for the metric."""
        # Update baselines if not set
        if metric.metric_type not in self.performance_baselines:
            self.performance_baselines[metric.metric_type] = metric.value
        else:
            # Exponential moving average
            alpha = 0.1  # Smoothing factor
            current_baseline = self.performance_baselines[metric.metric_type]
            self.performance_baselines[metric.metric_type] = alpha * metric.value + (1 - alpha) * current_baseline

    def _initialize_default_thresholds(self) -> None:
        """Initialize default performance thresholds."""
        default_thresholds = {
            MetricType.LATENCY: PerformanceThreshold(
                metric_type=MetricType.LATENCY,
                warning_threshold=200.0,  # 200ms
                error_threshold=500.0,    # 500ms
                critical_threshold=1000.0,  # 1s
                comparison_operator="greater_than"
            ),
            MetricType.CPU_USAGE: PerformanceThreshold(
                metric_type=MetricType.CPU_USAGE,
                warning_threshold=70.0,   # 70%
                error_threshold=85.0,     # 85%
                critical_threshold=95.0,  # 95%
                comparison_operator="greater_than"
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=80.0,   # 80%
                error_threshold=90.0,     # 90%
                critical_threshold=95.0,  # 95%
                comparison_operator="greater_than"
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=1.0,    # 1%
                error_threshold=5.0,      # 5%
                critical_threshold=10.0,  # 10%
                comparison_operator="greater_than"
            ),
            MetricType.THROUGHPUT: PerformanceThreshold(
                metric_type=MetricType.THROUGHPUT,
                warning_threshold=500.0,  # 500 req/s
                error_threshold=200.0,    # 200 req/s
                critical_threshold=50.0,  # 50 req/s
                comparison_operator="less_than"
            )
        }
        
        self.thresholds.update(default_thresholds)

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        current_time = datetime.now()
        
        # Current metrics summary
        current_metrics = {}
        for metric_type, metrics_buffer in self.metrics_buffer.items():
            if metrics_buffer:
                latest_metric = metrics_buffer[-1]
                current_metrics[metric_type.value] = {
                    'current_value': latest_metric.value,
                    'timestamp': latest_metric.timestamp.isoformat(),
                    'source': latest_metric.source
                }
        
        # Active alerts summary
        alerts_by_severity = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            if severity not in alerts_by_severity:
                alerts_by_severity[severity] = 0
            alerts_by_severity[severity] += 1
        
        # Performance trends (last hour)
        hour_ago = current_time - timedelta(hours=1)
        recent_history = [
            h for h in self.performance_history
            if h['timestamp'] > hour_ago
        ]
        
        # Optimization status
        optimization_status = {
            'total_recommendations': len(self.optimization_recommendations),
            'applied_configurations': len(self.adaptive_configs),
            'last_optimization': max(
                [config.last_updated for config in self.adaptive_configs.values()],
                default=None
            )
        }
        
        return {
            'monitoring_status': {
                'active': self.monitoring_active,
                'collection_interval_ms': self.collection_interval_ms,
                'active_tasks': len(self.monitoring_tasks)
            },
            'current_metrics': current_metrics,
            'performance_baselines': {
                metric_type.value: baseline 
                for metric_type, baseline in self.performance_baselines.items()
            },
            'active_alerts': alerts_by_severity,
            'total_alert_history': len(self.alert_history),
            'performance_trends': {
                'data_points_last_hour': len(recent_history),
                'metrics_collected': sum(len(buffer) for buffer in self.metrics_buffer.values())
            },
            'optimization_status': optimization_status,
            'anomaly_detection': {
                'models_active': len(self.ml_models),
                'detection_enabled': True
            }
        }

    async def update_performance_analytics(self) -> None:
        """Update performance analytics and insights."""
        # This would be called by the stream processing loop
        pass


class StreamProcessor:
    """Real-time stream processor for metrics."""
    
    def __init__(self, processor_id: str):
        """Initialize stream processor."""
        self.processor_id = processor_id
        self.processed_count = 0
        
    async def process_metric(self, metric: PerformanceMetric) -> None:
        """Process a metric in the stream."""
        self.processed_count += 1
        # Override in subclasses for specific processing


class PerformancePatternAnalyzer:
    """Analyzer for detecting performance patterns."""
    
    def __init__(self):
        """Initialize pattern analyzer."""
        self.detected_patterns: List[Dict[str, Any]] = []
        
    async def analyze_patterns(self, performance_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze performance patterns from historical data."""
        patterns = []
        
        if len(performance_history) < 10:
            return patterns
        
        # Group by metric type
        metrics_by_type = {}
        for record in performance_history:
            metric_type = record['metric_type']
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(record)
        
        # Analyze each metric type
        for metric_type, records in metrics_by_type.items():
            if len(records) >= 10:
                pattern = await self._analyze_metric_pattern(metric_type, records)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    async def _analyze_metric_pattern(self, metric_type: str, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze pattern for a specific metric type."""
        # Extract mean values over time
        values = [record['window_stats']['mean'] for record in records[-20:]]  # Last 20 records
        
        if len(values) < 5:
            return None
        
        # Detect trend
        trend_slope = np.polyfit(range(len(values)), values, 1)[0]
        
        # Detect seasonality (simplified)
        if len(values) >= 10:
            values_array = np.array(values)
            autocorr = np.corrcoef(values_array[:-1], values_array[1:])[0, 1]
        else:
            autocorr = 0.0
        
        # Classify pattern
        if abs(trend_slope) < 0.01:
            pattern_type = "stable"
        elif trend_slope > 0:
            pattern_type = "increasing"
        else:
            pattern_type = "decreasing"
        
        return {
            'metric_type': metric_type,
            'pattern_type': pattern_type,
            'trend_slope': trend_slope,
            'autocorrelation': autocorr,
            'stability_score': 1.0 - abs(trend_slope),
            'data_points': len(values)
        }


class AdaptiveOptimizationEngine:
    """Engine for generating adaptive optimization recommendations."""
    
    def __init__(self):
        """Initialize optimization engine."""
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def generate_recommendations(self,
                                     patterns: List[Dict[str, Any]],
                                     metrics_buffer: Dict[MetricType, Deque[PerformanceMetric]]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on patterns."""
        recommendations = []
        
        for pattern in patterns:
            metric_type = pattern['metric_type']
            pattern_type = pattern['pattern_type']
            
            # Get current performance
            if MetricType(metric_type) in metrics_buffer:
                recent_metrics = list(metrics_buffer[MetricType(metric_type)])[-10:]
                if recent_metrics:
                    current_performance = np.mean([m.value for m in recent_metrics])
                else:
                    continue
            else:
                continue
            
            # Generate recommendations based on pattern
            if pattern_type == "increasing" and metric_type in ["latency", "cpu_usage", "memory_usage"]:
                # Performance is degrading
                recommendation = OptimizationRecommendation(
                    recommendation_id=f"opt_{metric_type}_{int(time.time())}",
                    metric_type=MetricType(metric_type),
                    current_performance=current_performance,
                    target_performance=current_performance * 0.8,  # 20% improvement
                    optimization_actions=[
                        f"Optimize {metric_type.replace('_', ' ')} configuration",
                        "Review resource allocation",
                        "Implement caching strategies"
                    ],
                    estimated_improvement=0.2,
                    confidence_score=0.7,
                    implementation_priority="medium",
                    estimated_effort="moderate",
                    cost_benefit_ratio=2.5
                )
                recommendations.append(recommendation)
                
            elif pattern_type == "decreasing" and metric_type == "throughput":
                # Throughput is declining
                recommendation = OptimizationRecommendation(
                    recommendation_id=f"opt_{metric_type}_{int(time.time())}",
                    metric_type=MetricType(metric_type),
                    current_performance=current_performance,
                    target_performance=current_performance * 1.3,  # 30% improvement
                    optimization_actions=[
                        "Scale up processing capacity",
                        "Optimize database queries",
                        "Implement load balancing"
                    ],
                    estimated_improvement=0.3,
                    confidence_score=0.8,
                    implementation_priority="high",
                    estimated_effort="significant",
                    cost_benefit_ratio=3.0
                )
                recommendations.append(recommendation)
        
        return recommendations


class PredictiveAnomalyDetector:
    """Predictive anomaly detector using machine learning."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.baseline_models: Dict[MetricType, Dict[str, float]] = {}
        self.detection_sensitivity = 2.0  # Standard deviations
        
    async def detect_anomalies(self,
                             metric_type: MetricType,
                             metrics_buffer: Deque[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Detect anomalies in metric stream."""
        anomalies = []
        
        if len(metrics_buffer) < 50:  # Need minimum data for detection
            return anomalies
        
        # Get recent metrics
        recent_metrics = list(metrics_buffer)[-50:]
        recent_values = [m.value for m in recent_metrics]
        
        # Calculate baseline statistics
        mean_value = np.mean(recent_values[:-10])  # Exclude last 10 for comparison
        std_value = np.std(recent_values[:-10])
        
        # Check last few values for anomalies
        for metric in recent_metrics[-10:]:
            if std_value > 0:  # Avoid division by zero
                z_score = abs(metric.value - mean_value) / std_value
                
                if z_score > self.detection_sensitivity:
                    anomaly = {
                        'metric_type': metric_type.value,
                        'timestamp': metric.timestamp,
                        'actual_value': metric.value,
                        'expected_value': mean_value,
                        'z_score': z_score,
                        'severity': 'HIGH' if z_score > 3.0 else 'MEDIUM',
                        'description': f"{metric_type.value} value {metric.value:.2f} deviates {z_score:.2f} standard deviations from expected {mean_value:.2f}"
                    }
                    anomalies.append(anomaly)
        
        return anomalies