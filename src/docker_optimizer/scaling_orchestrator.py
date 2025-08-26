"""Scaling Orchestrator for Docker Optimizer Agent.

This module implements intelligent auto-scaling, performance optimization,
and resource management for production workloads.
"""

import asyncio
import time
import psutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import threading

from .logging_observability import ObservabilityManager, LogLevel


class ScalingDecision(Enum):
    """Scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


class ResourceType(Enum):
    """Resource types to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CONCURRENT_OPERATIONS = "concurrent_operations"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_rate: float
    network_io_rate: float
    concurrent_operations: int
    queue_length: int
    response_time_ms: float
    error_rate: float
    throughput: float


@dataclass
class ScalingThresholds:
    """Thresholds for scaling decisions."""
    cpu_scale_up: float = 70.0
    cpu_scale_down: float = 30.0
    memory_scale_up: float = 75.0
    memory_scale_down: float = 40.0
    response_time_scale_up_ms: float = 5000.0
    error_rate_scale_up: float = 0.05  # 5%
    queue_length_scale_up: int = 10
    emergency_cpu_threshold: float = 90.0
    emergency_memory_threshold: float = 90.0


@dataclass
class ScalingConfiguration:
    """Configuration for auto-scaling."""
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown_seconds: int = 300  # 5 minutes
    scale_down_cooldown_seconds: int = 600  # 10 minutes
    metrics_window_seconds: int = 300  # 5 minutes
    decision_interval_seconds: int = 60  # 1 minute
    predictive_scaling: bool = True
    aggressive_scaling: bool = False


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies."""
    
    @abstractmethod
    async def make_scaling_decision(
        self, 
        current_metrics: ScalingMetrics, 
        historical_metrics: List[ScalingMetrics],
        current_instances: int
    ) -> ScalingDecision:
        """Make a scaling decision based on metrics."""
        pass


class ReactiveScalingStrategy(ScalingStrategy):
    """Reactive scaling based on current metrics."""
    
    def __init__(self, thresholds: ScalingThresholds):
        self.thresholds = thresholds
    
    async def make_scaling_decision(
        self, 
        current_metrics: ScalingMetrics, 
        historical_metrics: List[ScalingMetrics],
        current_instances: int
    ) -> ScalingDecision:
        """Make scaling decision based on current thresholds."""
        
        # Emergency scaling
        if (current_metrics.cpu_percent > self.thresholds.emergency_cpu_threshold or
            current_metrics.memory_percent > self.thresholds.emergency_memory_threshold):
            return ScalingDecision.EMERGENCY_SCALE
        
        # Scale up conditions
        scale_up_signals = 0
        if current_metrics.cpu_percent > self.thresholds.cpu_scale_up:
            scale_up_signals += 1
        if current_metrics.memory_percent > self.thresholds.memory_scale_up:
            scale_up_signals += 1
        if current_metrics.response_time_ms > self.thresholds.response_time_scale_up_ms:
            scale_up_signals += 1
        if current_metrics.error_rate > self.thresholds.error_rate_scale_up:
            scale_up_signals += 2  # Errors are more important
        if current_metrics.queue_length > self.thresholds.queue_length_scale_up:
            scale_up_signals += 1
        
        if scale_up_signals >= 2:  # Need multiple signals to scale up
            return ScalingDecision.SCALE_UP
        
        # Scale down conditions (only if no scale up signals)
        if scale_up_signals == 0:
            if (current_metrics.cpu_percent < self.thresholds.cpu_scale_down and
                current_metrics.memory_percent < self.thresholds.memory_scale_down and
                current_metrics.queue_length == 0):
                return ScalingDecision.SCALE_DOWN
        
        return ScalingDecision.MAINTAIN


class PredictiveScalingStrategy(ScalingStrategy):
    """Predictive scaling using historical patterns."""
    
    def __init__(self, thresholds: ScalingThresholds):
        self.thresholds = thresholds
        self.reactive_strategy = ReactiveScalingStrategy(thresholds)
    
    async def make_scaling_decision(
        self, 
        current_metrics: ScalingMetrics, 
        historical_metrics: List[ScalingMetrics],
        current_instances: int
    ) -> ScalingDecision:
        """Make predictive scaling decision."""
        
        # Fall back to reactive if not enough historical data
        if len(historical_metrics) < 10:
            return await self.reactive_strategy.make_scaling_decision(
                current_metrics, historical_metrics, current_instances
            )
        
        # Analyze trends in last 10 minutes
        recent_metrics = historical_metrics[-10:]
        
        # Calculate trend for key metrics
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.response_time_ms for m in recent_metrics])
        
        # Predict values in next 5 minutes
        predicted_cpu = current_metrics.cpu_percent + (cpu_trend * 5)
        predicted_memory = current_metrics.memory_percent + (memory_trend * 5)
        predicted_response_time = current_metrics.response_time_ms + (response_time_trend * 5)
        
        # Make decision based on predictions
        if (predicted_cpu > self.thresholds.cpu_scale_up or
            predicted_memory > self.thresholds.memory_scale_up or
            predicted_response_time > self.thresholds.response_time_scale_up_ms):
            return ScalingDecision.SCALE_UP
        
        # Use reactive strategy for scale down and emergency
        return await self.reactive_strategy.make_scaling_decision(
            current_metrics, historical_metrics, current_instances
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class ScalingOrchestrator:
    """Orchestrates auto-scaling and performance optimization."""
    
    def __init__(self, config: Optional[ScalingConfiguration] = None):
        self.config = config or ScalingConfiguration()
        self.thresholds = ScalingThresholds()
        
        self.obs_manager = ObservabilityManager(
            log_level=LogLevel.INFO,
            service_name="scaling-orchestrator"
        )
        
        # Initialize scaling strategy
        if self.config.predictive_scaling:
            self.scaling_strategy = PredictiveScalingStrategy(self.thresholds)
        else:
            self.scaling_strategy = ReactiveScalingStrategy(self.thresholds)
        
        # State tracking
        self.current_instances = 1
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scale_up_time: Optional[datetime] = None
        self.last_scale_down_time: Optional[datetime] = None
        
        # Monitoring
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance optimization
        self.performance_optimizations: Dict[str, bool] = {
            "connection_pooling": False,
            "request_batching": False,
            "caching": False,
            "compression": False,
            "lazy_loading": False
        }
    
    def start_monitoring(self):
        """Start scaling monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.obs_manager.logger.info("Scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop scaling monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.obs_manager.logger.info("Scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                current_metrics = self._collect_metrics()
                self.metrics_history.append(current_metrics)
                
                # Clean old metrics (keep only last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Make scaling decision
                asyncio.create_task(self._evaluate_scaling(current_metrics))
                
                # Apply performance optimizations
                self._apply_performance_optimizations(current_metrics)
                
            except Exception as e:
                self.obs_manager.logger.error(f"Error in scaling monitoring loop: {e}")
            
            # Wait for next decision interval
            self.stop_event.wait(self.config.decision_interval_seconds)
    
    async def _evaluate_scaling(self, current_metrics: ScalingMetrics):
        """Evaluate and execute scaling decisions."""
        try:
            # Get recent metrics for decision making
            recent_metrics = [
                m for m in self.metrics_history 
                if (datetime.now() - m.timestamp).total_seconds() < self.config.metrics_window_seconds
            ]
            
            # Make scaling decision
            decision = await self.scaling_strategy.make_scaling_decision(
                current_metrics, recent_metrics, self.current_instances
            )
            
            # Execute scaling decision
            await self._execute_scaling_decision(decision, current_metrics)
            
        except Exception as e:
            self.obs_manager.logger.error(f"Error evaluating scaling: {e}")
    
    async def _execute_scaling_decision(self, decision: ScalingDecision, metrics: ScalingMetrics):
        """Execute a scaling decision."""
        now = datetime.now()
        
        if decision == ScalingDecision.SCALE_UP:
            # Check cooldown
            if (self.last_scale_up_time and 
                (now - self.last_scale_up_time).total_seconds() < self.config.scale_up_cooldown_seconds):
                return
            
            # Check max instances
            if self.current_instances >= self.config.max_instances:
                self.obs_manager.logger.warning("Cannot scale up: at maximum instances")
                return
            
            # Scale up
            new_instances = min(self.current_instances + 1, self.config.max_instances)
            await self._scale_to_instances(new_instances, "SCALE_UP", metrics)
            self.last_scale_up_time = now
            
        elif decision == ScalingDecision.SCALE_DOWN:
            # Check cooldown
            if (self.last_scale_down_time and 
                (now - self.last_scale_down_time).total_seconds() < self.config.scale_down_cooldown_seconds):
                return
            
            # Check min instances
            if self.current_instances <= self.config.min_instances:
                return
            
            # Scale down
            new_instances = max(self.current_instances - 1, self.config.min_instances)
            await self._scale_to_instances(new_instances, "SCALE_DOWN", metrics)
            self.last_scale_down_time = now
            
        elif decision == ScalingDecision.EMERGENCY_SCALE:
            # Emergency scaling ignores cooldowns
            emergency_instances = min(self.current_instances + 2, self.config.max_instances)
            await self._scale_to_instances(emergency_instances, "EMERGENCY_SCALE", metrics)
            self.last_scale_up_time = now
    
    async def _scale_to_instances(self, target_instances: int, reason: str, metrics: ScalingMetrics):
        """Scale to target number of instances."""
        previous_instances = self.current_instances
        self.current_instances = target_instances
        
        self.obs_manager.logger.info(
            f"Scaling {reason}: {previous_instances} → {target_instances} instances",
            extra={
                "previous_instances": previous_instances,
                "target_instances": target_instances,
                "reason": reason,
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "response_time_ms": metrics.response_time_ms,
                "error_rate": metrics.error_rate
            }
        )
        
        # Here you would integrate with actual scaling mechanisms
        # (e.g., Kubernetes HPA, Docker Swarm, AWS Auto Scaling, etc.)
        
        # Simulate scaling delay
        await asyncio.sleep(0.1)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system and application metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_rate = getattr(disk_io, 'read_bytes', 0) + getattr(disk_io, 'write_bytes', 0)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_rate = network_io.bytes_sent + network_io.bytes_recv
            
            # Mock application metrics (in real implementation, these would come from the application)
            concurrent_operations = len(threading.enumerate())  # Simplified
            queue_length = 0  # Would be actual queue length
            response_time_ms = 100.0  # Would be measured response time
            error_rate = 0.01  # Would be calculated error rate
            throughput = 10.0  # Would be measured throughput
            
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io_rate=disk_io_rate,
                network_io_rate=network_io_rate,
                concurrent_operations=concurrent_operations,
                queue_length=queue_length,
                response_time_ms=response_time_ms,
                error_rate=error_rate,
                throughput=throughput
            )
            
        except Exception as e:
            self.obs_manager.logger.error(f"Error collecting metrics: {e}")
            # Return safe default metrics
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_io_rate=0.0,
                network_io_rate=0.0,
                concurrent_operations=1,
                queue_length=0,
                response_time_ms=100.0,
                error_rate=0.0,
                throughput=1.0
            )
    
    def _apply_performance_optimizations(self, metrics: ScalingMetrics):
        """Apply performance optimizations based on current metrics."""
        
        # Enable connection pooling under high load
        if metrics.concurrent_operations > 5 and not self.performance_optimizations["connection_pooling"]:
            self.performance_optimizations["connection_pooling"] = True
            self.obs_manager.logger.info("Enabled connection pooling")
        
        # Enable request batching for high throughput
        if metrics.throughput > 50 and not self.performance_optimizations["request_batching"]:
            self.performance_optimizations["request_batching"] = True
            self.obs_manager.logger.info("Enabled request batching")
        
        # Enable caching for high response times
        if metrics.response_time_ms > 1000 and not self.performance_optimizations["caching"]:
            self.performance_optimizations["caching"] = True
            self.obs_manager.logger.info("Enabled intelligent caching")
        
        # Enable compression for high network I/O
        if metrics.network_io_rate > 1000000 and not self.performance_optimizations["compression"]:  # 1MB
            self.performance_optimizations["compression"] = True
            self.obs_manager.logger.info("Enabled response compression")
        
        # Enable lazy loading for high memory usage
        if metrics.memory_percent > 60 and not self.performance_optimizations["lazy_loading"]:
            self.performance_optimizations["lazy_loading"] = True
            self.obs_manager.logger.info("Enabled lazy loading")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        if not self.metrics_history:
            return {"status": "no_metrics_available"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate average metrics over last 5 minutes
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > recent_cutoff]
        
        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = latest_metrics.cpu_percent
            avg_memory = latest_metrics.memory_percent
            avg_response_time = latest_metrics.response_time_ms
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.config.min_instances,
            "max_instances": self.config.max_instances,
            "is_monitoring": self.is_monitoring,
            "current_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "response_time_ms": latest_metrics.response_time_ms,
                "error_rate": latest_metrics.error_rate,
                "throughput": latest_metrics.throughput
            },
            "average_metrics_5m": {
                "cpu_percent": round(avg_cpu, 1),
                "memory_percent": round(avg_memory, 1),
                "response_time_ms": round(avg_response_time, 1)
            },
            "performance_optimizations": self.performance_optimizations,
            "scaling_strategy": self.scaling_strategy.__class__.__name__,
            "last_scale_up": self.last_scale_up_time.isoformat() if self.last_scale_up_time else None,
            "last_scale_down": self.last_scale_down_time.isoformat() if self.last_scale_down_time else None,
            "metrics_history_count": len(self.metrics_history)
        }
    
    def update_configuration(self, new_config: ScalingConfiguration):
        """Update scaling configuration."""
        self.config = new_config
        self.obs_manager.logger.info("Scaling configuration updated")
    
    def update_thresholds(self, new_thresholds: ScalingThresholds):
        """Update scaling thresholds."""
        self.thresholds = new_thresholds
        # Recreate strategy with new thresholds
        if self.config.predictive_scaling:
            self.scaling_strategy = PredictiveScalingStrategy(self.thresholds)
        else:
            self.scaling_strategy = ReactiveScalingStrategy(self.thresholds)
        
        self.obs_manager.logger.info("Scaling thresholds updated")