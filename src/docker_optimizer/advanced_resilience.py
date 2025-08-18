"""Advanced Resilience Engine for Self-Healing Docker Optimization."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur."""
    
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INVALID_INPUT = "invalid_input"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class RetryStrategy(Enum):
    """Retry strategies for different failure types."""
    
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    NO_RETRY = "no_retry"
    ADAPTIVE_RETRY = "adaptive_retry"


class HealthStatus(Enum):
    """Health status for components."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureMetrics(BaseModel):
    """Metrics for tracking failures."""
    
    failure_type: FailureType
    count: int = 0
    last_occurrence: Optional[datetime] = None
    avg_recovery_time: float = 0.0
    success_rate: float = 1.0
    trend: str = "stable"


class CircuitBreaker(BaseModel):
    """Circuit breaker for protecting against cascading failures."""
    
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0
    success_count: int = 0
    total_calls: int = 0


class HealthCheck(BaseModel):
    """Health check configuration."""
    
    name: str
    check_function: str  # Function name as string for serialization
    interval: int = 30  # seconds
    timeout: int = 10   # seconds
    failure_threshold: int = 3
    success_threshold: int = 2
    current_failures: int = 0
    current_successes: int = 0
    last_check: Optional[datetime] = None
    status: HealthStatus = HealthStatus.UNKNOWN


class ResilienceConfiguration(BaseModel):
    """Configuration for resilience engine."""
    
    default_retry_attempts: int = 3
    default_timeout: int = 30
    circuit_breaker_enabled: bool = True
    health_checks_enabled: bool = True
    auto_recovery_enabled: bool = True
    failure_rate_threshold: float = 0.5
    degraded_mode_enabled: bool = True
    metrics_retention_days: int = 7


class AdvancedResilienceEngine:
    """Advanced resilience engine with self-healing capabilities."""
    
    def __init__(self, config: Optional[ResilienceConfiguration] = None):
        """Initialize the resilience engine."""
        self.config = config or ResilienceConfiguration()
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_check_tasks: List[asyncio.Task] = []
        
        # Failure tracking
        self.failure_metrics: Dict[str, FailureMetrics] = {}
        self.failure_history: List[Tuple[datetime, str, FailureType]] = []
        
        # Recovery strategies
        self.recovery_strategies: Dict[FailureType, Callable] = {}
        
        # System state
        self.overall_health: HealthStatus = HealthStatus.HEALTHY
        self.degraded_mode: bool = False
        self.last_health_assessment: Optional[datetime] = None
        
        # Initialize default recovery strategies
        self._initialize_recovery_strategies()
        
        logger.info("Advanced resilience engine initialized")
    
    async def start(self) -> None:
        """Start the resilience engine."""
        if self.config.health_checks_enabled:
            await self._start_health_checks()
        
        logger.info("Resilience engine started")
    
    async def stop(self) -> None:
        """Stop the resilience engine."""
        # Cancel all health check tasks
        for task in self.health_check_tasks:
            task.cancel()
        
        if self.health_check_tasks:
            await asyncio.gather(*self.health_check_tasks, return_exceptions=True)
        
        logger.info("Resilience engine stopped")
    
    def register_circuit_breaker(self, 
                                name: str, 
                                failure_threshold: int = 5,
                                recovery_timeout: int = 60) -> None:
        """Register a circuit breaker for a component."""
        self.circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        logger.info(f"Circuit breaker registered: {name}")
    
    def register_health_check(self,
                            name: str,
                            check_function: Callable,
                            interval: int = 30,
                            timeout: int = 10) -> None:
        """Register a health check."""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function.__name__,
            interval=interval,
            timeout=timeout
        )
        
        logger.info(f"Health check registered: {name}")
    
    @asynccontextmanager
    async def circuit_protected(self, circuit_name: str):
        """Context manager for circuit breaker protection."""
        if circuit_name not in self.circuit_breakers:
            self.register_circuit_breaker(circuit_name)
        
        circuit = self.circuit_breakers[circuit_name]
        
        # Check if circuit is open
        if circuit.state == CircuitState.OPEN:
            if self._should_attempt_reset(circuit):
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_calls = 0
            else:
                raise Exception(f"Circuit breaker {circuit_name} is OPEN")
        
        # Check if circuit is half-open and limit calls
        if circuit.state == CircuitState.HALF_OPEN:
            if circuit.half_open_calls >= circuit.half_open_max_calls:
                raise Exception(f"Circuit breaker {circuit_name} is HALF_OPEN and at call limit")
            circuit.half_open_calls += 1
        
        circuit.total_calls += 1
        start_time = time.time()
        
        try:
            yield
            
            # Success - update circuit breaker
            circuit.success_count += 1
            execution_time = time.time() - start_time
            
            if circuit.state == CircuitState.HALF_OPEN:
                if circuit.half_open_calls >= circuit.half_open_max_calls:
                    # Enough successful calls, close the circuit
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    logger.info(f"Circuit breaker {circuit_name} CLOSED after successful recovery")
            
        except Exception as e:
            # Failure - update circuit breaker
            circuit.failure_count += 1
            circuit.last_failure_time = datetime.now()
            
            if circuit.state == CircuitState.HALF_OPEN:
                # Failure during half-open, go back to open
                circuit.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {circuit_name} back to OPEN after failure during recovery")
            elif circuit.failure_count >= circuit.failure_threshold:
                # Too many failures, open the circuit
                circuit.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {circuit_name} OPENED due to failure threshold")
            
            # Record failure
            await self._record_failure(circuit_name, self._classify_failure(e))
            raise
    
    async def resilient_operation(self,
                                operation: Callable,
                                operation_name: str,
                                retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
                                max_retries: Optional[int] = None,
                                timeout: Optional[int] = None,
                                circuit_breaker: bool = True) -> Any:
        """Execute an operation with resilience patterns."""
        max_retries = max_retries or self.config.default_retry_attempts
        timeout = timeout or self.config.default_timeout
        
        async def _execute_with_timeout():
            if asyncio.iscoroutinefunction(operation):
                return await asyncio.wait_for(operation(), timeout=timeout)
            else:
                # Run in executor for sync functions
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, operation),
                    timeout=timeout
                )
        
        # Execute with circuit breaker if enabled
        if circuit_breaker and self.config.circuit_breaker_enabled:
            async with self.circuit_protected(operation_name):
                return await self._execute_with_retry(
                    _execute_with_timeout,
                    operation_name,
                    retry_strategy,
                    max_retries
                )
        else:
            return await self._execute_with_retry(
                _execute_with_timeout,
                operation_name,
                retry_strategy,
                max_retries
            )
    
    async def graceful_degradation(self, 
                                 primary_operation: Callable,
                                 fallback_operation: Optional[Callable] = None,
                                 operation_name: str = "unnamed_operation") -> Any:
        """Execute operation with graceful degradation."""
        try:
            # Try primary operation
            return await self.resilient_operation(
                primary_operation,
                f"{operation_name}_primary"
            )
        except Exception as e:
            logger.warning(f"Primary operation {operation_name} failed: {e}")
            
            if fallback_operation and self.config.degraded_mode_enabled:
                try:
                    # Try fallback operation
                    logger.info(f"Attempting fallback for {operation_name}")
                    result = await self.resilient_operation(
                        fallback_operation,
                        f"{operation_name}_fallback",
                        max_retries=1  # Fewer retries for fallback
                    )
                    
                    # Mark system as degraded
                    self.degraded_mode = True
                    logger.warning(f"System running in degraded mode for {operation_name}")
                    
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback operation also failed: {fallback_error}")
                    raise
            else:
                raise
    
    async def self_heal(self, component_name: str) -> bool:
        """Attempt to self-heal a failed component."""
        if component_name in self.recovery_strategies:
            try:
                recovery_function = self.recovery_strategies[component_name]
                await recovery_function()
                
                logger.info(f"Self-healing successful for {component_name}")
                return True
            except Exception as e:
                logger.error(f"Self-healing failed for {component_name}: {e}")
                return False
        else:
            logger.warning(f"No recovery strategy available for {component_name}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        now = datetime.now()
        
        # Update overall health
        self._assess_overall_health()
        
        health_report = {
            'overall_status': self.overall_health.value,
            'degraded_mode': self.degraded_mode,
            'last_assessment': self.last_health_assessment.isoformat() if self.last_health_assessment else None,
            'circuit_breakers': self._get_circuit_breaker_status(),
            'health_checks': self._get_health_check_status(),
            'failure_metrics': self._get_failure_metrics_summary(),
            'system_metrics': {
                'total_operations': sum(cb.total_calls for cb in self.circuit_breakers.values()),
                'total_failures': sum(cb.failure_count for cb in self.circuit_breakers.values()),
                'avg_success_rate': self._calculate_avg_success_rate(),
                'recovery_attempts': len([f for f in self.failure_history if (now - f[0]).hours < 24])
            }
        }
        
        return health_report
    
    async def _execute_with_retry(self,
                                operation: Callable,
                                operation_name: str,
                                retry_strategy: RetryStrategy,
                                max_retries: int) -> Any:
        """Execute operation with retry logic."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_retry_delay(retry_strategy, attempt)
                    logger.info(f"Retrying {operation_name} (attempt {attempt + 1}/{max_retries + 1}) after {delay}s")
                    await asyncio.sleep(delay)
                
                result = await operation()
                
                if attempt > 0:
                    logger.info(f"Operation {operation_name} succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)
                
                await self._record_failure(operation_name, failure_type)
                
                if attempt < max_retries:
                    logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}): {e}")
                else:
                    logger.error(f"Operation {operation_name} failed after {max_retries} retries: {e}")
        
        raise last_exception
    
    def _calculate_retry_delay(self, strategy: RetryStrategy, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        base_delay = 1.0
        
        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(base_delay * (2 ** attempt), 60)  # Max 60 seconds
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(base_delay * attempt, 30)  # Max 30 seconds
        elif strategy == RetryStrategy.FIXED_INTERVAL:
            return base_delay
        elif strategy == RetryStrategy.ADAPTIVE_RETRY:
            # Adaptive based on recent failure patterns
            return self._calculate_adaptive_delay(attempt)
        else:
            return 0
    
    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Calculate adaptive retry delay based on failure patterns."""
        # Analyze recent failure patterns to determine optimal delay
        recent_failures = [
            f for f in self.failure_history 
            if (datetime.now() - f[0]).total_seconds() < 300  # Last 5 minutes
        ]
        
        if len(recent_failures) > 5:
            # High failure rate, use longer delays
            return min(2.0 * (1.5 ** attempt), 60)
        else:
            # Normal failure rate, use standard exponential backoff
            return min(1.0 * (2 ** attempt), 30)
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure."""
        if isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT_ERROR
        elif isinstance(exception, ConnectionError):
            return FailureType.NETWORK_ERROR
        elif isinstance(exception, MemoryError):
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(exception, ValueError):
            return FailureType.INVALID_INPUT
        elif "service" in str(exception).lower():
            return FailureType.EXTERNAL_SERVICE_ERROR
        elif isinstance(exception, (OSError, SystemError)):
            return FailureType.SYSTEM_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    async def _record_failure(self, operation_name: str, failure_type: FailureType) -> None:
        """Record a failure for metrics and analysis."""
        now = datetime.now()
        
        # Add to failure history
        self.failure_history.append((now, operation_name, failure_type))
        
        # Update failure metrics
        metric_key = f"{operation_name}_{failure_type.value}"
        if metric_key not in self.failure_metrics:
            self.failure_metrics[metric_key] = FailureMetrics(failure_type=failure_type)
        
        metric = self.failure_metrics[metric_key]
        metric.count += 1
        metric.last_occurrence = now
        
        # Clean old failures (keep only last N days)
        cutoff = now - timedelta(days=self.config.metrics_retention_days)
        self.failure_history = [f for f in self.failure_history if f[0] > cutoff]
        
        # Trigger self-healing if pattern detected
        if self.config.auto_recovery_enabled:
            await self._check_for_recovery_trigger(operation_name, failure_type)
    
    async def _check_for_recovery_trigger(self, operation_name: str, failure_type: FailureType) -> None:
        """Check if failure pattern triggers recovery action."""
        # Count recent failures for this operation
        recent_failures = [
            f for f in self.failure_history 
            if f[1] == operation_name and (datetime.now() - f[0]).total_seconds() < 300
        ]
        
        if len(recent_failures) >= 3:  # 3 failures in 5 minutes
            logger.warning(f"Failure pattern detected for {operation_name}, triggering recovery")
            await self.self_heal(operation_name)
    
    def _should_attempt_reset(self, circuit: CircuitBreaker) -> bool:
        """Check if circuit breaker should attempt reset."""
        if circuit.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - circuit.last_failure_time).total_seconds()
        return time_since_failure >= circuit.recovery_timeout
    
    async def _start_health_checks(self) -> None:
        """Start health check tasks."""
        for check in self.health_checks.values():
            task = asyncio.create_task(self._run_health_check(check))
            self.health_check_tasks.append(task)
    
    async def _run_health_check(self, check: HealthCheck) -> None:
        """Run a single health check continuously."""
        while True:
            try:
                await asyncio.sleep(check.interval)
                
                # Execute health check with timeout
                start_time = time.time()
                
                try:
                    # In a real implementation, you'd call the actual check function
                    # For now, simulate a health check
                    await asyncio.sleep(0.1)  # Simulate check
                    success = True  # Placeholder
                    
                    execution_time = time.time() - start_time
                    
                    if success:
                        check.current_successes += 1
                        check.current_failures = 0
                        
                        if check.current_successes >= check.success_threshold:
                            if check.status != HealthStatus.HEALTHY:
                                logger.info(f"Health check {check.name} recovered")
                            check.status = HealthStatus.HEALTHY
                    else:
                        check.current_failures += 1
                        check.current_successes = 0
                        
                        if check.current_failures >= check.failure_threshold:
                            check.status = HealthStatus.UNHEALTHY
                            logger.warning(f"Health check {check.name} failed")
                
                except asyncio.TimeoutError:
                    check.current_failures += 1
                    check.current_successes = 0
                    check.status = HealthStatus.DEGRADED
                    logger.warning(f"Health check {check.name} timed out")
                
                check.last_check = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check {check.name} encountered error: {e}")
    
    def _assess_overall_health(self) -> None:
        """Assess overall system health."""
        now = datetime.now()
        
        # Count healthy vs unhealthy components
        health_statuses = [check.status for check in self.health_checks.values()]
        open_circuits = [cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN]
        
        unhealthy_count = health_statuses.count(HealthStatus.UNHEALTHY)
        degraded_count = health_statuses.count(HealthStatus.DEGRADED)
        total_checks = len(health_statuses)
        
        if len(open_circuits) > 0 or (total_checks > 0 and unhealthy_count / total_checks > 0.5):
            self.overall_health = HealthStatus.CRITICAL
        elif unhealthy_count > 0 or degraded_count > 0 or self.degraded_mode:
            self.overall_health = HealthStatus.DEGRADED
        elif total_checks > 0:
            self.overall_health = HealthStatus.HEALTHY
        else:
            self.overall_health = HealthStatus.UNKNOWN
        
        self.last_health_assessment = now
    
    def _get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            name: {
                'state': circuit.state.value,
                'failure_count': circuit.failure_count,
                'success_rate': circuit.success_count / max(circuit.total_calls, 1),
                'last_failure': circuit.last_failure_time.isoformat() if circuit.last_failure_time else None
            }
            for name, circuit in self.circuit_breakers.items()
        }
    
    def _get_health_check_status(self) -> Dict[str, Any]:
        """Get status of all health checks."""
        return {
            name: {
                'status': check.status.value,
                'last_check': check.last_check.isoformat() if check.last_check else None,
                'consecutive_failures': check.current_failures,
                'consecutive_successes': check.current_successes
            }
            for name, check in self.health_checks.items()
        }
    
    def _get_failure_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of failure metrics."""
        now = datetime.now()
        
        # Recent failures (last hour)
        recent_failures = [
            f for f in self.failure_history 
            if (now - f[0]).total_seconds() < 3600
        ]
        
        failure_by_type = {}
        for failure_type in FailureType:
            count = len([f for f in recent_failures if f[2] == failure_type])
            failure_by_type[failure_type.value] = count
        
        return {
            'total_recent_failures': len(recent_failures),
            'failures_by_type': failure_by_type,
            'failure_rate_per_hour': len(recent_failures),
            'most_common_failure': max(failure_by_type, key=failure_by_type.get) if failure_by_type else None
        }
    
    def _calculate_avg_success_rate(self) -> float:
        """Calculate average success rate across all circuit breakers."""
        if not self.circuit_breakers:
            return 1.0
        
        total_calls = sum(cb.total_calls for cb in self.circuit_breakers.values())
        total_successes = sum(cb.success_count for cb in self.circuit_breakers.values())
        
        return total_successes / max(total_calls, 1)
    
    def _initialize_recovery_strategies(self) -> None:
        """Initialize default recovery strategies."""
        self.recovery_strategies = {
            FailureType.NETWORK_ERROR.value: self._recover_network_error,
            FailureType.RESOURCE_EXHAUSTION.value: self._recover_resource_exhaustion,
            FailureType.EXTERNAL_SERVICE_ERROR.value: self._recover_external_service_error,
            FailureType.SYSTEM_ERROR.value: self._recover_system_error
        }
    
    async def _recover_network_error(self) -> None:
        """Recovery strategy for network errors."""
        logger.info("Attempting network error recovery")
        # In practice, this might restart network components, clear DNS cache, etc.
        await asyncio.sleep(1)  # Simulate recovery action
    
    async def _recover_resource_exhaustion(self) -> None:
        """Recovery strategy for resource exhaustion."""
        logger.info("Attempting resource exhaustion recovery")
        # In practice, this might clear caches, garbage collect, restart services, etc.
        await asyncio.sleep(1)  # Simulate recovery action
    
    async def _recover_external_service_error(self) -> None:
        """Recovery strategy for external service errors."""
        logger.info("Attempting external service error recovery")
        # In practice, this might switch to backup services, refresh credentials, etc.
        await asyncio.sleep(1)  # Simulate recovery action
    
    async def _recover_system_error(self) -> None:
        """Recovery strategy for system errors."""
        logger.info("Attempting system error recovery")
        # In practice, this might restart components, clear temporary files, etc.
        await asyncio.sleep(1)  # Simulate recovery action