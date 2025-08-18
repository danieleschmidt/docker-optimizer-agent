"""Enhanced Observability Engine with Real-time Monitoring and Analytics."""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricValue(BaseModel):
    """A metric value with timestamp."""
    
    timestamp: datetime
    value: float
    tags: Dict[str, str] = Field(default_factory=dict)


class Metric(BaseModel):
    """A metric definition."""
    
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    values: List[MetricValue] = Field(default_factory=list)
    max_values: int = 1000  # Maximum number of values to retain


class Alert(BaseModel):
    """An alert definition."""
    
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # String representation of condition
    threshold: float
    metric_name: str
    duration: int = 0  # Duration in seconds before firing
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    is_active: bool = False


class Dashboard(BaseModel):
    """A monitoring dashboard."""
    
    name: str
    description: str
    metrics: List[str]
    refresh_interval: int = 30
    layout: Dict[str, Any] = Field(default_factory=dict)


class PerformanceInsight(BaseModel):
    """Performance insight derived from metrics."""
    
    insight_type: str
    title: str
    description: str
    impact_level: str  # low, medium, high, critical
    recommendation: str
    confidence_score: float
    metrics_evidence: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class EnhancedObservabilityEngine:
    """Advanced observability engine with real-time monitoring."""
    
    def __init__(self, retention_days: int = 7):
        """Initialize the observability engine."""
        self.retention_days = retention_days
        
        # Metrics storage
        self.metrics: Dict[str, Metric] = {}
        self.metric_collectors: Dict[str, Callable] = {}
        
        # Alerts
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Dashboards
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Performance insights
        self.insights: List[PerformanceInsight] = []
        self.insight_generators: List[Callable] = []
        
        # Real-time monitoring
        self.monitoring_tasks: List[asyncio.Task] = []
        self.monitoring_active = False
        
        # Analytics cache
        self.analytics_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        
        # Initialize default metrics and alerts
        self._initialize_default_metrics()
        self._initialize_default_alerts()
        self._initialize_default_dashboards()
        self._initialize_insight_generators()
        
        logger.info("Enhanced observability engine initialized")
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start metric collection tasks
        for metric_name, collector in self.metric_collectors.items():
            task = asyncio.create_task(self._collect_metric_continuously(metric_name, collector))
            self.monitoring_tasks.append(task)
        
        # Start alert monitoring
        alert_task = asyncio.create_task(self._monitor_alerts())
        self.monitoring_tasks.append(alert_task)
        
        # Start insight generation
        insight_task = asyncio.create_task(self._generate_insights_continuously())
        self.monitoring_tasks.append(insight_task)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_old_data())
        self.monitoring_tasks.append(cleanup_task)
        
        logger.info("Real-time monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Real-time monitoring stopped")
    
    def register_metric(self, 
                       name: str, 
                       metric_type: MetricType,
                       description: str = "",
                       unit: str = "",
                       collector: Optional[Callable] = None) -> None:
        """Register a new metric."""
        self.metrics[name] = Metric(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit
        )
        
        if collector:
            self.metric_collectors[name] = collector
        
        logger.info(f"Metric registered: {name} ({metric_type.value})")
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     timestamp: Optional[datetime] = None) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            logger.warning(f"Metric {name} not registered, creating with default settings")
            self.register_metric(name, MetricType.GAUGE)
        
        metric = self.metrics[name]
        timestamp = timestamp or datetime.now()
        tags = tags or {}
        
        metric_value = MetricValue(
            timestamp=timestamp,
            value=value,
            tags=tags
        )
        
        metric.values.append(metric_value)
        
        # Trim old values
        if len(metric.values) > metric.max_values:
            metric.values = metric.values[-metric.max_values:]
        
        # Invalidate related cache
        self._invalidate_cache(name)
    
    def register_alert(self,
                      alert_id: str,
                      name: str,
                      metric_name: str,
                      condition: str,
                      threshold: float,
                      severity: AlertSeverity = AlertSeverity.WARNING,
                      duration: int = 0) -> None:
        """Register a new alert."""
        alert = Alert(
            id=alert_id,
            name=name,
            description=f"Alert when {metric_name} {condition} {threshold}",
            severity=severity,
            condition=condition,
            threshold=threshold,
            metric_name=metric_name,
            duration=duration
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"Alert registered: {name}")
    
    def create_dashboard(self,
                        name: str,
                        metrics: List[str],
                        description: str = "",
                        refresh_interval: int = 30) -> None:
        """Create a monitoring dashboard."""
        dashboard = Dashboard(
            name=name,
            description=description,
            metrics=metrics,
            refresh_interval=refresh_interval
        )
        
        self.dashboards[name] = dashboard
        logger.info(f"Dashboard created: {name}")
    
    def get_metric_statistics(self, 
                            metric_name: str, 
                            time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get statistical analysis of a metric."""
        if metric_name not in self.metrics:
            return {}
        
        cache_key = f"stats_{metric_name}_{time_range}"
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]
        
        metric = self.metrics[metric_name]
        time_range = time_range or timedelta(hours=1)
        cutoff = datetime.now() - time_range
        
        # Filter values by time range
        values = [
            mv.value for mv in metric.values 
            if mv.timestamp >= cutoff
        ]
        
        if not values:
            return {}
        
        stats = {
            'count': len(values),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std_dev': float(np.std(values)),
            'percentiles': {
                'p50': float(np.percentile(values, 50)),
                'p90': float(np.percentile(values, 90)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            },
            'trend': self._calculate_trend(values),
            'anomalies': self._detect_anomalies(values)
        }
        
        # Cache the result
        self.analytics_cache[cache_key] = stats
        self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=5)
        
        return stats
    
    def get_metric_time_series(self, 
                             metric_name: str,
                             time_range: Optional[timedelta] = None,
                             aggregation: str = "none") -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        if metric_name not in self.metrics:
            return []
        
        metric = self.metrics[metric_name]
        time_range = time_range or timedelta(hours=1)
        cutoff = datetime.now() - time_range
        
        # Filter values by time range
        filtered_values = [
            mv for mv in metric.values 
            if mv.timestamp >= cutoff
        ]
        
        if aggregation == "none":
            return [
                {
                    'timestamp': mv.timestamp.isoformat(),
                    'value': mv.value,
                    'tags': mv.tags
                }
                for mv in filtered_values
            ]
        else:
            # Aggregate by time buckets
            return self._aggregate_time_series(filtered_values, aggregation)
    
    def get_dashboard_data(self, dashboard_name: str) -> Dict[str, Any]:
        """Get data for a dashboard."""
        if dashboard_name not in self.dashboards:
            return {}
        
        dashboard = self.dashboards[dashboard_name]
        
        data = {
            'dashboard': dashboard.dict(),
            'metrics_data': {},
            'generated_at': datetime.now().isoformat()
        }
        
        for metric_name in dashboard.metrics:
            if metric_name in self.metrics:
                data['metrics_data'][metric_name] = {
                    'current_value': self._get_current_metric_value(metric_name),
                    'statistics': self.get_metric_statistics(metric_name),
                    'time_series': self.get_metric_time_series(metric_name, timedelta(hours=1))
                }
        
        return data
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [
            {
                'id': alert.id,
                'name': alert.name,
                'severity': alert.severity.value,
                'metric_name': alert.metric_name,
                'triggered_at': alert.triggered_at.isoformat() if alert.triggered_at else None,
                'duration': (datetime.now() - alert.triggered_at).total_seconds() if alert.triggered_at else 0
            }
            for alert in self.alerts.values()
            if alert.is_active
        ]
    
    def get_performance_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance insights."""
        sorted_insights = sorted(
            self.insights, 
            key=lambda x: (x.timestamp, x.confidence_score), 
            reverse=True
        )
        
        return [insight.dict() for insight in sorted_insights[:limit]]
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        now = datetime.now()
        
        # Calculate health score
        health_score = self._calculate_system_health_score()
        
        # Get metric summary
        metric_summary = {}
        for name, metric in self.metrics.items():
            if metric.values:
                latest_value = metric.values[-1].value
                metric_summary[name] = {
                    'current_value': latest_value,
                    'unit': metric.unit,
                    'trend': self._get_metric_trend(name)
                }
        
        # Get alert summary
        alert_summary = {
            'total_alerts': len(self.alerts),
            'active_alerts': len([a for a in self.alerts.values() if a.is_active]),
            'critical_alerts': len([a for a in self.alerts.values() if a.is_active and a.severity == AlertSeverity.CRITICAL])
        }
        
        return {
            'timestamp': now.isoformat(),
            'health_score': health_score,
            'system_status': self._get_system_status(health_score),
            'metrics_summary': metric_summary,
            'alerts_summary': alert_summary,
            'insights_count': len(self.insights),
            'monitoring_active': self.monitoring_active,
            'uptime': self._calculate_uptime()
        }
    
    async def _collect_metric_continuously(self, metric_name: str, collector: Callable) -> None:
        """Continuously collect a metric."""
        while self.monitoring_active:
            try:
                # Collect metric value
                if asyncio.iscoroutinefunction(collector):
                    value = await collector()
                else:
                    value = collector()
                
                if value is not None:
                    self.record_metric(metric_name, value)
                
                # Wait for next collection
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metric {metric_name}: {e}")
                await asyncio.sleep(30)  # Wait longer after error
    
    async def _monitor_alerts(self) -> None:
        """Monitor all alerts continuously."""
        while self.monitoring_active:
            try:
                for alert in self.alerts.values():
                    await self._check_alert(alert)
                
                await asyncio.sleep(5)  # Check alerts every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring alerts: {e}")
                await asyncio.sleep(10)
    
    async def _check_alert(self, alert: Alert) -> None:
        """Check if an alert should fire or resolve."""
        metric_name = alert.metric_name
        if metric_name not in self.metrics:
            return
        
        metric = self.metrics[metric_name]
        if not metric.values:
            return
        
        current_value = metric.values[-1].value
        threshold = alert.threshold
        
        # Check condition
        condition_met = False
        if alert.condition == "greater_than":
            condition_met = current_value > threshold
        elif alert.condition == "less_than":
            condition_met = current_value < threshold
        elif alert.condition == "equals":
            condition_met = abs(current_value - threshold) < 0.001
        elif alert.condition == "not_equals":
            condition_met = abs(current_value - threshold) >= 0.001
        
        now = datetime.now()
        
        if condition_met and not alert.is_active:
            # Alert should fire
            if alert.duration == 0 or self._condition_met_for_duration(alert, alert.duration):
                alert.is_active = True
                alert.triggered_at = now
                logger.warning(f"Alert fired: {alert.name} (value: {current_value}, threshold: {threshold})")
                
                # Add to history
                self.alert_history.append(alert.copy(deep=True))
        
        elif not condition_met and alert.is_active:
            # Alert should resolve
            alert.is_active = False
            alert.resolved_at = now
            logger.info(f"Alert resolved: {alert.name} (value: {current_value})")
    
    def _condition_met_for_duration(self, alert: Alert, duration: int) -> bool:
        """Check if condition has been met for the required duration."""
        metric = self.metrics[alert.metric_name]
        cutoff = datetime.now() - timedelta(seconds=duration)
        
        recent_values = [
            mv for mv in metric.values 
            if mv.timestamp >= cutoff
        ]
        
        if not recent_values:
            return False
        
        # Check if all recent values meet the condition
        for mv in recent_values:
            condition_met = False
            if alert.condition == "greater_than":
                condition_met = mv.value > alert.threshold
            elif alert.condition == "less_than":
                condition_met = mv.value < alert.threshold
            
            if not condition_met:
                return False
        
        return True
    
    async def _generate_insights_continuously(self) -> None:
        """Generate performance insights continuously."""
        while self.monitoring_active:
            try:
                for generator in self.insight_generators:
                    if asyncio.iscoroutinefunction(generator):
                        insights = await generator()
                    else:
                        insights = generator()
                    
                    if insights:
                        for insight in insights:
                            self.insights.append(insight)
                
                # Trim old insights
                cutoff = datetime.now() - timedelta(days=self.retention_days)
                self.insights = [i for i in self.insights if i.timestamp >= cutoff]
                
                await asyncio.sleep(300)  # Generate insights every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating insights: {e}")
                await asyncio.sleep(600)  # Wait longer after error
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data periodically."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Run cleanup every hour
                
                cutoff = datetime.now() - timedelta(days=self.retention_days)
                
                # Clean up old metric values
                for metric in self.metrics.values():
                    metric.values = [mv for mv in metric.values if mv.timestamp >= cutoff]
                
                # Clean up alert history
                self.alert_history = [a for a in self.alert_history if (a.triggered_at or datetime.now()) >= cutoff]
                
                # Clean up cache
                now = datetime.now()
                expired_keys = [k for k, ttl in self.cache_ttl.items() if ttl < now]
                for key in expired_keys:
                    self.analytics_cache.pop(key, None)
                    self.cache_ttl.pop(key, None)
                
                logger.debug("Completed data cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default metrics."""
        # Docker optimization metrics
        self.register_metric("optimization_requests_total", MetricType.COUNTER, "Total optimization requests")
        self.register_metric("optimization_duration_seconds", MetricType.TIMER, "Optimization duration")
        self.register_metric("docker_size_reduction_percent", MetricType.GAUGE, "Size reduction percentage")
        self.register_metric("security_issues_found", MetricType.GAUGE, "Security issues found")
        self.register_metric("layer_count_before", MetricType.GAUGE, "Layer count before optimization")
        self.register_metric("layer_count_after", MetricType.GAUGE, "Layer count after optimization")
        
        # System metrics
        self.register_metric("memory_usage_bytes", MetricType.GAUGE, "Memory usage", "bytes")
        self.register_metric("cpu_usage_percent", MetricType.GAUGE, "CPU usage", "%")
        self.register_metric("error_rate", MetricType.RATE, "Error rate")
        self.register_metric("response_time_ms", MetricType.TIMER, "Response time", "ms")
    
    def _initialize_default_alerts(self) -> None:
        """Initialize default alerts."""
        self.register_alert(
            "high_error_rate",
            "High Error Rate",
            "error_rate",
            "greater_than",
            0.1,
            AlertSeverity.ERROR,
            duration=60
        )
        
        self.register_alert(
            "slow_response_time",
            "Slow Response Time",
            "response_time_ms",
            "greater_than",
            5000,
            AlertSeverity.WARNING,
            duration=300
        )
        
        self.register_alert(
            "high_memory_usage",
            "High Memory Usage",
            "memory_usage_bytes",
            "greater_than",
            1073741824,  # 1GB
            AlertSeverity.WARNING
        )
    
    def _initialize_default_dashboards(self) -> None:
        """Initialize default dashboards."""
        self.create_dashboard(
            "optimization_performance",
            [
                "optimization_requests_total",
                "optimization_duration_seconds",
                "docker_size_reduction_percent",
                "security_issues_found"
            ],
            "Docker optimization performance metrics"
        )
        
        self.create_dashboard(
            "system_health",
            [
                "memory_usage_bytes",
                "cpu_usage_percent", 
                "error_rate",
                "response_time_ms"
            ],
            "System health and performance metrics"
        )
    
    def _initialize_insight_generators(self) -> None:
        """Initialize insight generators."""
        self.insight_generators = [
            self._generate_performance_insights,
            self._generate_optimization_insights,
            self._generate_security_insights
        ]
    
    def _generate_performance_insights(self) -> List[PerformanceInsight]:
        """Generate performance-related insights."""
        insights = []
        
        # Check response time trends
        response_time_stats = self.get_metric_statistics("response_time_ms", timedelta(hours=1))
        if response_time_stats and response_time_stats.get('trend') == 'increasing':
            insights.append(PerformanceInsight(
                insight_type="performance_degradation",
                title="Response Time Increasing",
                description=f"Response time has increased to {response_time_stats['mean']:.1f}ms average",
                impact_level="medium",
                recommendation="Investigate for performance bottlenecks or increase resources",
                confidence_score=0.8,
                metrics_evidence=["response_time_ms"]
            ))
        
        # Check error rate
        error_rate_stats = self.get_metric_statistics("error_rate", timedelta(hours=1))
        if error_rate_stats and error_rate_stats['mean'] > 0.05:  # 5% error rate
            insights.append(PerformanceInsight(
                insight_type="reliability_issue",
                title="High Error Rate Detected",
                description=f"Error rate is {error_rate_stats['mean']*100:.1f}%, above acceptable threshold",
                impact_level="high",
                recommendation="Investigate error causes and implement fixes",
                confidence_score=0.9,
                metrics_evidence=["error_rate"]
            ))
        
        return insights
    
    def _generate_optimization_insights(self) -> List[PerformanceInsight]:
        """Generate optimization-related insights."""
        insights = []
        
        # Check optimization effectiveness
        size_reduction_stats = self.get_metric_statistics("docker_size_reduction_percent", timedelta(days=1))
        if size_reduction_stats and size_reduction_stats['mean'] < 20:  # Less than 20% reduction
            insights.append(PerformanceInsight(
                insight_type="optimization_opportunity",
                title="Low Docker Size Reduction",
                description=f"Average size reduction is only {size_reduction_stats['mean']:.1f}%",
                impact_level="medium",
                recommendation="Review optimization strategies for better size reduction",
                confidence_score=0.7,
                metrics_evidence=["docker_size_reduction_percent"]
            ))
        
        return insights
    
    def _generate_security_insights(self) -> List[PerformanceInsight]:
        """Generate security-related insights."""
        insights = []
        
        # Check security issues trend
        security_stats = self.get_metric_statistics("security_issues_found", timedelta(days=1))
        if security_stats and security_stats.get('trend') == 'increasing':
            insights.append(PerformanceInsight(
                insight_type="security_concern",
                title="Increasing Security Issues",
                description=f"Security issues trending upward, average {security_stats['mean']:.1f} per optimization",
                impact_level="high",
                recommendation="Review security scanning rules and update base images",
                confidence_score=0.8,
                metrics_evidence=["security_issues_found"]
            ))
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """Detect anomalies in metric values."""
        if len(values) < 10:
            return []
        
        # Use simple statistical method (values outside 2 standard deviations)
        mean = np.mean(values)
        std = np.std(values)
        threshold = 2 * std
        
        anomalies = []
        for i, value in enumerate(values):
            if abs(value - mean) > threshold:
                anomalies.append(i)
        
        return anomalies
    
    def _aggregate_time_series(self, values: List[MetricValue], aggregation: str) -> List[Dict[str, Any]]:
        """Aggregate time series data."""
        # Group by time buckets (simplified implementation)
        bucket_size = timedelta(minutes=5)  # 5-minute buckets
        buckets = defaultdict(list)
        
        for value in values:
            bucket_time = value.timestamp.replace(second=0, microsecond=0)
            bucket_time = bucket_time.replace(minute=(bucket_time.minute // 5) * 5)
            buckets[bucket_time].append(value.value)
        
        aggregated = []
        for bucket_time, bucket_values in sorted(buckets.items()):
            if aggregation == "mean":
                agg_value = np.mean(bucket_values)
            elif aggregation == "sum":
                agg_value = np.sum(bucket_values)
            elif aggregation == "max":
                agg_value = np.max(bucket_values)
            elif aggregation == "min":
                agg_value = np.min(bucket_values)
            else:
                agg_value = np.mean(bucket_values)  # Default to mean
            
            aggregated.append({
                'timestamp': bucket_time.isoformat(),
                'value': float(agg_value),
                'count': len(bucket_values)
            })
        
        return aggregated
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get the current value of a metric."""
        if metric_name not in self.metrics:
            return None
        
        metric = self.metrics[metric_name]
        if not metric.values:
            return None
        
        return metric.values[-1].value
    
    def _get_metric_trend(self, metric_name: str) -> str:
        """Get trend for a specific metric."""
        stats = self.get_metric_statistics(metric_name, timedelta(hours=1))
        return stats.get('trend', 'unknown')
    
    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.metrics:
            return 100.0
        
        # Weight different factors
        error_rate = self._get_current_metric_value("error_rate") or 0
        response_time = self._get_current_metric_value("response_time_ms") or 0
        active_alerts = len([a for a in self.alerts.values() if a.is_active])
        
        # Calculate health score (simplified)
        health_score = 100.0
        
        # Penalize high error rate
        health_score -= min(error_rate * 1000, 50)  # Max 50 point penalty
        
        # Penalize slow response times
        if response_time > 1000:  # Over 1 second
            health_score -= min((response_time - 1000) / 100, 30)  # Max 30 point penalty
        
        # Penalize active alerts
        health_score -= min(active_alerts * 10, 20)  # Max 20 point penalty
        
        return max(health_score, 0.0)
    
    def _get_system_status(self, health_score: float) -> str:
        """Get system status based on health score."""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 70:
            return "good"
        elif health_score >= 50:
            return "fair"
        elif health_score >= 30:
            return "poor"
        else:
            return "critical"
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime (simplified)."""
        # This would track actual start time in a real implementation
        return 99.9  # Placeholder
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.analytics_cache:
            return False
        
        ttl = self.cache_ttl.get(cache_key)
        if ttl is None or ttl < datetime.now():
            return False
        
        return True
    
    def _invalidate_cache(self, metric_name: str) -> None:
        """Invalidate cache entries related to a metric."""
        keys_to_remove = [
            key for key in self.analytics_cache.keys()
            if metric_name in key
        ]
        
        for key in keys_to_remove:
            self.analytics_cache.pop(key, None)
            self.cache_ttl.pop(key, None)