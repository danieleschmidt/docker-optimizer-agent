"""Auto-scaling Engine for Dynamic Resource Management."""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import psutil

from .health_monitor import get_health_monitor
from .intelligent_caching import get_cache_manager
from .monitoring_integration import get_monitoring_integration

logger = logging.getLogger(__name__)


class ScalingDirection(str, Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingMetric(str, Enum):
    """Metrics for scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"


@dataclass
class ScalingRule:
    """Defines a scaling rule."""
    metric: ScalingMetric
    threshold_up: float
    threshold_down: float
    cooldown_minutes: int = 5
    min_instances: int = 1
    max_instances: int = 10
    scale_factor: float = 1.5


@dataclass
class ScalingEvent:
    """Records a scaling event."""
    timestamp: datetime
    direction: ScalingDirection
    trigger_metric: ScalingMetric
    metric_value: float
    instances_before: int
    instances_after: int
    reason: str


@dataclass
class ResourcePool:
    """Manages a pool of resources."""
    name: str
    current_instances: int = 1
    target_instances: int = 1
    min_instances: int = 1
    max_instances: int = 10
    last_scaled: datetime = field(default_factory=datetime.now)
    scaling_rules: List[ScalingRule] = field(default_factory=list)
    scaling_history: List[ScalingEvent] = field(default_factory=list)


class AutoScaler:
    """Intelligent auto-scaling engine."""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self.scaling_thread: Optional[threading.Thread] = None

        # Resource pools
        self.resource_pools: Dict[str, ResourcePool] = {}

        # Monitoring
        self.health_monitor = get_health_monitor()
        self.monitoring = get_monitoring_integration()
        self.cache_manager = get_cache_manager()

        # Scaling configuration
        self.global_cooldown = 300  # 5 minutes
        self.prediction_window = 600  # 10 minutes

        # Initialize default pools
        self._initialize_default_pools()

        # Predictive scaling
        self.enable_predictive_scaling = True
        self.load_predictions: Dict[str, List[Tuple[datetime, float]]] = {}

    def _initialize_default_pools(self) -> None:
        """Initialize default resource pools."""
        # Optimization workers pool
        optimization_rules = [
            ScalingRule(
                metric=ScalingMetric.CPU_USAGE,
                threshold_up=70.0,
                threshold_down=30.0,
                cooldown_minutes=3,
                max_instances=8
            ),
            ScalingRule(
                metric=ScalingMetric.REQUEST_RATE,
                threshold_up=50.0,  # requests per minute
                threshold_down=10.0,
                cooldown_minutes=5,
                max_instances=10
            ),
            ScalingRule(
                metric=ScalingMetric.RESPONSE_TIME,
                threshold_up=5000.0,  # milliseconds
                threshold_down=1000.0,
                cooldown_minutes=2,
                max_instances=6
            )
        ]

        self.resource_pools["optimization_workers"] = ResourcePool(
            name="optimization_workers",
            current_instances=2,
            target_instances=2,
            min_instances=1,
            max_instances=10,
            scaling_rules=optimization_rules
        )

        # Security scanning pool
        security_rules = [
            ScalingRule(
                metric=ScalingMetric.CPU_USAGE,
                threshold_up=80.0,
                threshold_down=40.0,
                cooldown_minutes=5,
                max_instances=4
            ),
            ScalingRule(
                metric=ScalingMetric.QUEUE_LENGTH,
                threshold_up=20.0,
                threshold_down=5.0,
                cooldown_minutes=3,
                max_instances=6
            )
        ]

        self.resource_pools["security_scanners"] = ResourcePool(
            name="security_scanners",
            current_instances=1,
            target_instances=1,
            min_instances=1,
            max_instances=4,
            scaling_rules=security_rules
        )

        # Cache workers pool
        cache_rules = [
            ScalingRule(
                metric=ScalingMetric.MEMORY_USAGE,
                threshold_up=85.0,
                threshold_down=50.0,
                cooldown_minutes=10,
                max_instances=3
            )
        ]

        self.resource_pools["cache_workers"] = ResourcePool(
            name="cache_workers",
            current_instances=1,
            target_instances=1,
            min_instances=1,
            max_instances=3,
            scaling_rules=cache_rules
        )

    def start_autoscaling(self) -> None:
        """Start auto-scaling monitoring."""
        if self.running:
            return

        self.running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling started")

    def stop_autoscaling(self) -> None:
        """Stop auto-scaling monitoring."""
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
        logger.info("Auto-scaling stopped")

    def _scaling_loop(self) -> None:
        """Main auto-scaling loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()

                # Process each resource pool
                for pool_name, pool in self.resource_pools.items():
                    self._process_pool_scaling(pool, metrics)

                # Predictive scaling
                if self.enable_predictive_scaling:
                    self._predictive_scaling()

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                time.sleep(self.check_interval)

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}

        try:
            # System metrics
            metrics[ScalingMetric.CPU_USAGE.value] = psutil.cpu_percent(interval=1)
            metrics[ScalingMetric.MEMORY_USAGE.value] = psutil.virtual_memory().percent

            # Application metrics from monitoring
            monitoring_metrics = self.monitoring.get_performance_metrics()

            # Request rate (requests per minute)
            total_requests = monitoring_metrics["counters"].get("optimization_requests", 0)
            metrics[ScalingMetric.REQUEST_RATE.value] = self._calculate_request_rate(total_requests)

            # Response time
            response_times = monitoring_metrics.get("response_times", {})
            metrics[ScalingMetric.RESPONSE_TIME.value] = response_times.get("avg", 0) * 1000  # Convert to ms

            # Error rate
            metrics[ScalingMetric.ERROR_RATE.value] = monitoring_metrics.get("error_rate", 0)

            # Simulated queue length (in real scenario, this would come from actual queues)
            metrics[ScalingMetric.QUEUE_LENGTH.value] = self._estimate_queue_length()

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return default metrics to prevent scaling errors
            metrics = {metric.value: 0.0 for metric in ScalingMetric}

        return metrics

    def _calculate_request_rate(self, total_requests: int) -> float:
        """Calculate requests per minute."""
        # This is a simplified calculation
        # In practice, you'd track requests over time windows
        current_time = time.time()

        # Store request counts with timestamps
        if not hasattr(self, '_request_history'):
            self._request_history = []

        # Add current measurement
        self._request_history.append((current_time, total_requests))

        # Keep only last 5 minutes of data
        cutoff_time = current_time - 300
        self._request_history = [(t, r) for t, r in self._request_history if t > cutoff_time]

        if len(self._request_history) < 2:
            return 0.0

        # Calculate rate
        time_diff = self._request_history[-1][0] - self._request_history[0][0]
        request_diff = self._request_history[-1][1] - self._request_history[0][1]

        if time_diff > 0:
            return (request_diff / time_diff) * 60  # Convert to per minute

        return 0.0

    def _estimate_queue_length(self) -> float:
        """Estimate current queue length."""
        # This would typically come from actual queue monitoring
        # For demonstration, we'll estimate based on response times and CPU usage

        metrics = self.monitoring.get_performance_metrics()
        response_time = metrics.get("response_times", {}).get("avg", 0)
        cpu_usage = psutil.cpu_percent()

        # Simple heuristic: higher response times and CPU usage suggest longer queues
        if response_time > 3.0 and cpu_usage > 70:
            return min(50.0, response_time * cpu_usage / 10)

        return max(0.0, (response_time - 1.0) * 5)

    def _process_pool_scaling(self, pool: ResourcePool, metrics: Dict[str, float]) -> None:
        """Process scaling decisions for a resource pool."""
        now = datetime.now()

        # Check cooldown
        if (now - pool.last_scaled).total_seconds() < self.global_cooldown:
            return

        # Evaluate scaling rules
        scale_decisions = []

        for rule in pool.scaling_rules:
            metric_value = metrics.get(rule.metric.value, 0.0)

            if metric_value > rule.threshold_up:
                # Scale up
                new_instances = min(
                    pool.max_instances,
                    int(pool.current_instances * rule.scale_factor)
                )
                if new_instances > pool.current_instances:
                    scale_decisions.append((
                        ScalingDirection.UP,
                        rule,
                        metric_value,
                        new_instances,
                        f"{rule.metric.value} ({metric_value:.1f}) > {rule.threshold_up}"
                    ))

            elif metric_value < rule.threshold_down and pool.current_instances > pool.min_instances:
                # Scale down
                new_instances = max(
                    pool.min_instances,
                    int(pool.current_instances / rule.scale_factor)
                )
                if new_instances < pool.current_instances:
                    scale_decisions.append((
                        ScalingDirection.DOWN,
                        rule,
                        metric_value,
                        new_instances,
                        f"{rule.metric.value} ({metric_value:.1f}) < {rule.threshold_down}"
                    ))

        # Execute scaling decision
        if scale_decisions:
            # Choose the most aggressive scaling decision
            scale_decisions.sort(key=lambda x: abs(x[3] - pool.current_instances), reverse=True)

            direction, rule, metric_value, new_instances, reason = scale_decisions[0]

            # Execute scaling
            self._execute_scaling(pool, direction, new_instances, rule.metric, metric_value, reason)

    def _execute_scaling(self,
                        pool: ResourcePool,
                        direction: ScalingDirection,
                        new_instances: int,
                        trigger_metric: ScalingMetric,
                        metric_value: float,
                        reason: str) -> None:
        """Execute scaling action."""
        old_instances = pool.current_instances

        # Update pool
        pool.current_instances = new_instances
        pool.target_instances = new_instances
        pool.last_scaled = datetime.now()

        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            direction=direction,
            trigger_metric=trigger_metric,
            metric_value=metric_value,
            instances_before=old_instances,
            instances_after=new_instances,
            reason=reason
        )

        pool.scaling_history.append(event)

        # Keep only last 100 scaling events
        if len(pool.scaling_history) > 100:
            pool.scaling_history = pool.scaling_history[-100:]

        # Log scaling action
        logger.info(
            f"Scaled {pool.name} {direction.value}: "
            f"{old_instances} -> {new_instances} instances. "
            f"Reason: {reason}"
        )

        # Actually perform scaling (this would integrate with container orchestration)
        self._perform_scaling_action(pool.name, direction, old_instances, new_instances)

    def _perform_scaling_action(self,
                               pool_name: str,
                               direction: ScalingDirection,
                               old_instances: int,
                               new_instances: int) -> None:
        """Perform the actual scaling action."""
        # This is where you would integrate with:
        # - Kubernetes HPA/VPA
        # - Docker Swarm
        # - Cloud provider auto-scaling groups
        # - Process pools, thread pools, etc.

        logger.info(f"Performing {direction.value} scaling for {pool_name}")

        if pool_name == "optimization_workers":
            # Adjust worker pool size
            self._scale_worker_pool("optimization", new_instances)

        elif pool_name == "security_scanners":
            # Adjust security scanner pool
            self._scale_worker_pool("security", new_instances)

        elif pool_name == "cache_workers":
            # Adjust cache worker resources
            self._scale_cache_resources(new_instances)

    def _scale_worker_pool(self, pool_type: str, target_instances: int) -> None:
        """Scale worker pool (simulation)."""
        # In a real implementation, this would:
        # - Start/stop worker processes
        # - Adjust thread pool sizes
        # - Scale container replicas
        logger.info(f"Scaling {pool_type} worker pool to {target_instances} instances")

    def _scale_cache_resources(self, target_instances: int) -> None:
        """Scale cache resources."""
        # Adjust cache sizes based on instance count
        base_cache_size = 50  # MB
        new_cache_size = base_cache_size * target_instances

        # Update cache configurations
        cache_manager = self.cache_manager
        for cache_name, cache in cache_manager.caches.items():
            cache.max_size_bytes = new_cache_size * 1024 * 1024

        logger.info(f"Scaled cache resources: {new_cache_size}MB per cache")

    def _predictive_scaling(self) -> None:
        """Implement predictive scaling based on historical patterns."""
        if not self.enable_predictive_scaling:
            return

        now = datetime.now()

        # Analyze historical scaling patterns
        for pool_name, pool in self.resource_pools.items():
            if len(pool.scaling_history) < 5:
                continue

            # Look for patterns in scaling events
            predictions = self._analyze_scaling_patterns(pool)

            if predictions:
                self._apply_predictive_scaling(pool, predictions)

    def _analyze_scaling_patterns(self, pool: ResourcePool) -> List[Tuple[datetime, int]]:
        """Analyze historical patterns to predict future scaling needs."""
        predictions = []

        # Simple pattern: if we consistently scale up at certain times,
        # predict similar scaling needs

        scaling_events = pool.scaling_history[-20:]  # Last 20 events
        up_events = [e for e in scaling_events if e.direction == ScalingDirection.UP]

        if len(up_events) >= 3:
            # Look for time-of-day patterns
            hours = [e.timestamp.hour for e in up_events]
            most_common_hour = max(set(hours), key=hours.count)

            # Predict scaling up in the next occurrence of this hour
            next_scale_time = now.replace(hour=most_common_hour, minute=0, second=0, microsecond=0)
            if next_scale_time <= now:
                next_scale_time += timedelta(days=1)

            # If we're within 30 minutes of the predicted time, suggest pre-scaling
            if (next_scale_time - now).total_seconds() <= 1800:  # 30 minutes
                target_instances = min(pool.max_instances, pool.current_instances + 1)
                predictions.append((next_scale_time, target_instances))

        return predictions

    def _apply_predictive_scaling(self, pool: ResourcePool, predictions: List[Tuple[datetime, int]]) -> None:
        """Apply predictive scaling decisions."""
        now = datetime.now()

        for pred_time, pred_instances in predictions:
            if pred_instances > pool.current_instances:
                # Only scale up predictively if we're not already at target
                time_until = (pred_time - now).total_seconds()

                if 0 <= time_until <= 1800:  # Within next 30 minutes
                    logger.info(
                        f"Predictive scaling: scaling {pool.name} to {pred_instances} "
                        f"instances in anticipation of load increase"
                    )

                    # Execute predictive scaling
                    self._execute_scaling(
                        pool,
                        ScalingDirection.UP,
                        pred_instances,
                        ScalingMetric.REQUEST_RATE,  # Default metric for predictive scaling
                        0.0,
                        "Predictive scaling based on historical patterns"
                    )
                    break

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        status = {
            "enabled": self.running,
            "global_cooldown_seconds": self.global_cooldown,
            "predictive_scaling_enabled": self.enable_predictive_scaling,
            "pools": {}
        }

        for pool_name, pool in self.resource_pools.items():
            pool_status = {
                "current_instances": pool.current_instances,
                "target_instances": pool.target_instances,
                "min_instances": pool.min_instances,
                "max_instances": pool.max_instances,
                "last_scaled": pool.last_scaled.isoformat(),
                "scaling_rules_count": len(pool.scaling_rules),
                "scaling_events_count": len(pool.scaling_history)
            }

            # Recent scaling events
            if pool.scaling_history:
                recent_events = pool.scaling_history[-5:]
                pool_status["recent_events"] = [
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "direction": event.direction.value,
                        "trigger": event.trigger_metric.value,
                        "metric_value": event.metric_value,
                        "instances_before": event.instances_before,
                        "instances_after": event.instances_after,
                        "reason": event.reason
                    }
                    for event in recent_events
                ]
            else:
                pool_status["recent_events"] = []

            status["pools"][pool_name] = pool_status

        return status

    def add_scaling_rule(self, pool_name: str, rule: ScalingRule) -> bool:
        """Add a new scaling rule to a pool."""
        if pool_name in self.resource_pools:
            self.resource_pools[pool_name].scaling_rules.append(rule)
            logger.info(f"Added scaling rule to {pool_name}: {rule.metric.value}")
            return True
        return False

    def remove_scaling_rule(self, pool_name: str, metric: ScalingMetric) -> bool:
        """Remove scaling rule for specific metric."""
        if pool_name in self.resource_pools:
            pool = self.resource_pools[pool_name]
            original_count = len(pool.scaling_rules)
            pool.scaling_rules = [r for r in pool.scaling_rules if r.metric != metric]
            return len(pool.scaling_rules) < original_count
        return False

    def force_scale(self, pool_name: str, target_instances: int, reason: str = "Manual scaling") -> bool:
        """Manually force scaling of a pool."""
        if pool_name not in self.resource_pools:
            return False

        pool = self.resource_pools[pool_name]

        # Validate target
        if target_instances < pool.min_instances or target_instances > pool.max_instances:
            logger.warning(
                f"Target instances {target_instances} outside valid range "
                f"[{pool.min_instances}, {pool.max_instances}] for {pool_name}"
            )
            return False

        # Determine direction
        if target_instances > pool.current_instances:
            direction = ScalingDirection.UP
        elif target_instances < pool.current_instances:
            direction = ScalingDirection.DOWN
        else:
            return True  # Already at target

        # Execute scaling
        self._execute_scaling(
            pool,
            direction,
            target_instances,
            ScalingMetric.CPU_USAGE,  # Default metric for manual scaling
            0.0,
            reason
        )

        return True

    def export_metrics(self) -> str:
        """Export auto-scaling metrics in Prometheus format."""
        metrics_lines = []

        for pool_name, pool in self.resource_pools.items():
            # Current instances
            metrics_lines.extend([
                f"# HELP docker_optimizer_autoscaler_instances Current instances for {pool_name}",
                "# TYPE docker_optimizer_autoscaler_instances gauge",
                f'docker_optimizer_autoscaler_instances{{pool="{pool_name}"}} {pool.current_instances}',
                ""
            ])

            # Scaling events
            recent_events = len([e for e in pool.scaling_history
                               if (datetime.now() - e.timestamp).total_seconds() < 3600])
            metrics_lines.extend([
                f"# HELP docker_optimizer_autoscaler_events_1h Scaling events in last hour for {pool_name}",
                "# TYPE docker_optimizer_autoscaler_events_1h counter",
                f'docker_optimizer_autoscaler_events_1h{{pool="{pool_name}"}} {recent_events}',
                ""
            ])

        return "\n".join(metrics_lines)


# Global auto-scaler instance
_autoscaler: Optional[AutoScaler] = None


def get_autoscaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _autoscaler
    if _autoscaler is None:
        _autoscaler = AutoScaler()
    return _autoscaler


def start_autoscaling() -> None:
    """Start auto-scaling."""
    scaler = get_autoscaler()
    scaler.start_autoscaling()


def stop_autoscaling() -> None:
    """Stop auto-scaling."""
    global _autoscaler
    if _autoscaler:
        _autoscaler.stop_autoscaling()


def get_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status."""
    scaler = get_autoscaler()
    return scaler.get_scaling_status()


def force_scale_pool(pool_name: str, target_instances: int, reason: str = "Manual") -> bool:
    """Force scale a specific pool."""
    scaler = get_autoscaler()
    return scaler.force_scale(pool_name, target_instances, reason)
