"""Advanced Resilience Engine with Circuit Breaker, Retry Logic, and Fallbacks."""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import requests


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(str, Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"  
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


@dataclass
class ResilienceConfig:
    """Configuration for resilience mechanisms."""
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half-open state
    
    # Retry settings
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: bool = True
    
    # Timeout settings
    operation_timeout: float = 30.0  # seconds
    connection_timeout: float = 10.0  # seconds
    
    # Fallback settings
    enable_fallbacks: bool = True
    enable_circuit_breaker: bool = True
    enable_retries: bool = True


@dataclass
class OperationResult:
    """Result of a resilient operation."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0
    circuit_state: Optional[CircuitState] = None
    used_fallback: bool = False


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker moved to HALF_OPEN")
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker CLOSED - service recovered")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning("Circuit breaker OPEN - test failed in HALF_OPEN")


class RetryManager:
    """Retry logic manager with various backoff strategies."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RetryManager")
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.retry_strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        
        elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        
        elif self.config.retry_strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = self.config.base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, base_delay * 0.1) if self.config.jitter else 0
            delay = base_delay + jitter
        
        else:
            delay = self.config.base_delay
        
        # Ensure delay doesn't exceed max_delay
        return min(delay, self.config.max_delay)
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_retries:
            return False
        
        # Don't retry certain types of exceptions
        non_retryable_exceptions = (
            ValueError,
            TypeError,
            SyntaxError,
            KeyboardInterrupt
        )
        
        if isinstance(exception, non_retryable_exceptions):
            self.logger.debug(f"Not retrying non-retryable exception: {type(exception).__name__}")
            return False
        
        return True


class FallbackManager:
    """Fallback mechanism manager."""
    
    def __init__(self):
        self.fallbacks: Dict[str, Callable] = {}
        self.logger = logging.getLogger(f"{__name__}.FallbackManager")
    
    def register_fallback(self, operation_name: str, fallback_func: Callable) -> None:
        """Register fallback function for operation."""
        self.fallbacks[operation_name] = fallback_func
        self.logger.debug(f"Registered fallback for: {operation_name}")
    
    def get_fallback(self, operation_name: str) -> Optional[Callable]:
        """Get fallback function for operation."""
        return self.fallbacks.get(operation_name)
    
    async def execute_fallback(
        self, 
        operation_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute fallback function."""
        fallback = self.get_fallback(operation_name)
        if fallback:
            self.logger.info(f"Executing fallback for: {operation_name}")
            if asyncio.iscoroutinefunction(fallback):
                return await fallback(*args, **kwargs)
            else:
                return fallback(*args, **kwargs)
        return None


class ResilienceEngine:
    """Main resilience engine coordinating all resilience mechanisms."""
    
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager(config)
        self.fallback_manager = FallbackManager()
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.operation_metrics: Dict[str, List[float]] = {}
    
    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(self.config)
        return self.circuit_breakers[operation_name]
    
    def register_fallback(self, operation_name: str, fallback_func: Callable) -> None:
        """Register fallback function."""
        self.fallback_manager.register_fallback(operation_name, fallback_func)
    
    async def execute_with_resilience(
        self,
        operation_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> OperationResult:
        """Execute operation with full resilience mechanisms."""
        start_time = time.time()
        attempts = 0
        last_exception = None
        
        # Get circuit breaker for this operation
        circuit_breaker = self.get_circuit_breaker(operation_name)
        
        while attempts < self.config.max_retries + 1:
            attempts += 1
            
            # Check circuit breaker
            if self.config.enable_circuit_breaker and not circuit_breaker.can_execute():
                self.logger.warning(f"Circuit breaker OPEN for {operation_name}")
                break
            
            try:
                # Execute with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.operation_timeout
                    )
                else:
                    result = func(*args, **kwargs)
                
                # Success - record metrics
                execution_time = time.time() - start_time
                self._record_success_metrics(operation_name, execution_time)
                
                if self.config.enable_circuit_breaker:
                    circuit_breaker.record_success()
                
                return OperationResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_time=execution_time,
                    circuit_state=circuit_breaker.state,
                    used_fallback=False
                )
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Operation {operation_name} failed (attempt {attempts}): {e}")
                
                if self.config.enable_circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Check if we should retry
                if (self.config.enable_retries and 
                    attempts <= self.config.max_retries and 
                    self.retry_manager.should_retry(attempts, e)):
                    
                    delay = self.retry_manager.calculate_delay(attempts)
                    self.logger.info(f"Retrying {operation_name} in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    break
        
        # All retries exhausted or circuit open - try fallback
        if self.config.enable_fallbacks:
            try:
                fallback_result = await self.fallback_manager.execute_fallback(
                    operation_name, *args, **kwargs
                )
                if fallback_result is not None:
                    total_time = time.time() - start_time
                    return OperationResult(
                        success=True,
                        result=fallback_result,
                        attempts=attempts,
                        total_time=total_time,
                        circuit_state=circuit_breaker.state,
                        used_fallback=True
                    )
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed for {operation_name}: {fallback_error}")
        
        # Complete failure
        total_time = time.time() - start_time
        self._record_failure_metrics(operation_name, total_time)
        
        return OperationResult(
            success=False,
            error=last_exception,
            attempts=attempts,
            total_time=total_time,
            circuit_state=circuit_breaker.state,
            used_fallback=False
        )
    
    def _record_success_metrics(self, operation_name: str, execution_time: float) -> None:
        """Record successful operation metrics."""
        if operation_name not in self.operation_metrics:
            self.operation_metrics[operation_name] = []
        
        metrics = self.operation_metrics[operation_name]
        metrics.append(execution_time)
        
        # Keep only last 1000 measurements
        if len(metrics) > 1000:
            metrics.pop(0)
    
    def _record_failure_metrics(self, operation_name: str, execution_time: float) -> None:
        """Record failed operation metrics."""
        # For now, just log the failure
        self.logger.error(f"Operation {operation_name} failed after {execution_time:.2f}s")
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for operation."""
        metrics = self.operation_metrics.get(operation_name, [])
        
        if not metrics:
            return {
                "total_calls": 0,
                "avg_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0
            }
        
        sorted_metrics = sorted(metrics)
        total_calls = len(metrics)
        avg_response_time = sum(metrics) / total_calls
        
        p95_index = int(total_calls * 0.95)
        p99_index = int(total_calls * 0.99)
        
        return {
            "total_calls": total_calls,
            "avg_response_time": avg_response_time,
            "p95_response_time": sorted_metrics[p95_index] if p95_index < total_calls else 0.0,
            "p99_response_time": sorted_metrics[p99_index] if p99_index < total_calls else 0.0,
            "circuit_state": self.circuit_breakers.get(operation_name, {}).state if operation_name in self.circuit_breakers else "N/A"
        }
    
    def get_system_resilience_status(self) -> Dict[str, Any]:
        """Get overall system resilience status."""
        circuit_states = {
            name: cb.state.value 
            for name, cb in self.circuit_breakers.items()
        }
        
        total_operations = sum(len(metrics) for metrics in self.operation_metrics.values())
        
        return {
            "circuit_breakers": circuit_states,
            "total_operations": total_operations,
            "active_circuits": len(self.circuit_breakers),
            "registered_fallbacks": len(self.fallback_manager.fallbacks),
            "config": {
                "max_retries": self.config.max_retries,
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "operation_timeout": self.config.operation_timeout
            }
        }


# Decorator for easy resilience application
def resilient_operation(
    operation_name: str,
    config: Optional[ResilienceConfig] = None,
    fallback_func: Optional[Callable] = None
):
    """Decorator to make any function resilient."""
    if config is None:
        config = ResilienceConfig()
    
    engine = ResilienceEngine(config)
    
    if fallback_func:
        engine.register_fallback(operation_name, fallback_func)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await engine.execute_with_resilience(
                operation_name, func, *args, **kwargs
            )
            if result.success:
                return result.result
            else:
                raise result.error or Exception("Operation failed")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Export for use in other modules
__all__ = [
    "ResilienceEngine",
    "ResilienceConfig", 
    "CircuitBreaker",
    "RetryManager",
    "FallbackManager",
    "OperationResult",
    "CircuitState",
    "RetryStrategy",
    "resilient_operation"
]