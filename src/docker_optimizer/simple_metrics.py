"""Simple metrics collection for Docker Optimizer Agent."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: Union[float, int, str]
    timestamp: float
    labels: Optional[Dict[str, str]] = None


class SimpleMetricsCollector:
    """Simple metrics collection and aggregation."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
    
    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.counters[name] += value
        self._record_metric(name, self.counters[name], labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self.gauges[name] = value
        self._record_metric(name, value, labels)
    
    def record_timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self._record_metric(f"{name}_duration", duration, labels)
    
    def _record_metric(self, name: str, value: Union[float, int, str], labels: Optional[Dict[str, str]]):
        """Record a metric internally."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels
        )
        self.metrics[name].append(metric)
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        return self.counters[name]
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self.gauges[name]
    
    def get_recent_metrics(self, name: str, seconds: int = 300) -> List[Metric]:
        """Get recent metrics for a given name."""
        if name not in self.metrics:
            return []
        
        cutoff = time.time() - seconds
        return [m for m in self.metrics[name] if m.timestamp >= cutoff]
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get a summary of all metrics."""
        summary = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'metric_counts': {name: len(values) for name, values in self.metrics.items()}
        }
        return summary
    
    def get_optimization_metrics(self) -> Dict[str, float]:
        """Get key optimization metrics."""
        return {
            'dockerfiles_optimized': self.get_counter('dockerfiles_optimized'),
            'security_fixes_applied': self.get_counter('security_fixes_applied'),
            'layer_optimizations': self.get_counter('layer_optimizations'),
            'average_size_reduction': self.get_gauge('average_size_reduction_percent'),
            'total_processing_time': self.get_gauge('total_processing_time_seconds')
        }