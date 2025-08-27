"""Enhanced error handling and recovery system for Docker Optimizer Agent.

This module provides comprehensive error handling, circuit breaker patterns,
retry mechanisms, and graceful degradation capabilities.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import psutil


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

    def can_retry(self) -> bool:
        """Check if error can be retried."""
        return self.recovery_attempts < self.max_recovery_attempts


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation for service resilience."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count_in_half_open: int = 0
    success_threshold: int = 2

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count_in_half_open = 0
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count_in_half_open += 1
            if self.success_count_in_half_open >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count_in_half_open = 0


class EnhancedErrorHandler:
    """Enhanced error handling system with multiple recovery strategies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
        self.error_patterns: Dict[str, RecoveryStrategy] = {
            "docker.errors.APIError": RecoveryStrategy.RETRY,
            "requests.exceptions.ConnectionError": RecoveryStrategy.CIRCUIT_BREAKER,
            "docker_optimizer.SecurityScanError": RecoveryStrategy.FALLBACK,
            "MemoryError": RecoveryStrategy.GRACEFUL_DEGRADATION,
            "FileNotFoundError": RecoveryStrategy.FAIL_FAST,
        }

    def register_fallback(self, error_type: str, fallback_func: Callable) -> None:
        """Register fallback function for specific error type."""
        self.fallback_strategies[error_type] = fallback_func

    def get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker()
        return self.circuit_breakers[operation]

    def classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and context."""
        error_type = type(error).__name__
        
        critical_errors = {
            "MemoryError", "SystemError", "KeyboardInterrupt",
            "SystemExit", "OSError"
        }
        
        high_errors = {
            "SecurityScanError", "DockerBuildError", "ValidationError",
            "PermissionError", "TimeoutError"
        }
        
        medium_errors = {
            "ConnectionError", "HTTPError", "FileNotFoundError",
            "ConfigurationError", "ParseError"
        }
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def get_recovery_strategy(self, error: Exception) -> RecoveryStrategy:
        """Determine recovery strategy for error."""
        error_module = error.__class__.__module__
        error_name = error.__class__.__name__
        
        # Check full module path
        full_path = f"{error_module}.{error_name}"
        if full_path in self.error_patterns:
            return self.error_patterns[full_path]
        
        # Check just class name
        if error_name in self.error_patterns:
            return self.error_patterns[error_name]
        
        # Default strategy based on severity
        severity = self.classify_error_severity(error)
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAIL_FAST
        elif severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.CIRCUIT_BREAKER
        else:
            return RecoveryStrategy.RETRY

    async def handle_error_async(
        self,
        error: Exception,
        operation: str = "",
        context: Optional[ErrorContext] = None
    ) -> Optional[Any]:
        """Handle error asynchronously with appropriate recovery strategy."""
        if context is None:
            context = ErrorContext(
                error_type=type(error).__name__,
                severity=self.classify_error_severity(error),
                operation=operation
            )
        
        self.error_history.append(context)
        
        recovery_strategy = self.get_recovery_strategy(error)
        
        self.logger.error(
            f"Error in {operation}: {error} (Severity: {context.severity.value}, "
            f"Strategy: {recovery_strategy.value})"
        )
        
        if recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            circuit_breaker = self.get_circuit_breaker(operation)
            if not circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for {operation}")
            
            circuit_breaker.record_failure()
            
        elif recovery_strategy == RecoveryStrategy.FALLBACK:
            error_type = type(error).__name__
            if error_type in self.fallback_strategies:
                try:
                    return await self._run_async_fallback(
                        self.fallback_strategies[error_type]
                    )
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback failed for {error_type}: {fallback_error}"
                    )
        
        elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation(error, context)
        
        # Re-raise for FAIL_FAST or unhandled cases
        raise error

    async def _run_async_fallback(self, fallback_func: Callable) -> Any:
        """Run fallback function asynchronously."""
        if asyncio.iscoroutinefunction(fallback_func):
            return await fallback_func()
        else:
            return fallback_func()

    async def _graceful_degradation(
        self, error: Exception, context: ErrorContext
    ) -> Dict[str, Any]:
        """Implement graceful degradation strategy."""
        self.logger.warning(
            f"Implementing graceful degradation for {context.operation}"
        )
        
        # Return minimal safe response
        return {
            "status": "degraded",
            "error": str(error),
            "timestamp": time.time(),
            "operation": context.operation
        }

    def with_retry(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """Decorator for retry logic."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            wait_time = delay * (backoff_factor ** attempt)
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            self.logger.error(
                                f"All {max_attempts} attempts failed: {e}"
                            )
                
                raise last_exception
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            wait_time = delay * (backoff_factor ** attempt)
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                            )
                            time.sleep(wait_time)
                        else:
                            self.logger.error(
                                f"All {max_attempts} attempts failed: {e}"
                            )
                
                raise last_exception
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    @contextmanager
    def error_context(self, operation: str, metadata: Optional[Dict] = None):
        """Context manager for error handling."""
        context = ErrorContext(
            error_type="",
            severity=ErrorSeverity.LOW,
            operation=operation,
            metadata=metadata or {}
        )
        
        try:
            yield context
        except Exception as e:
            context.error_type = type(e).__name__
            context.severity = self.classify_error_severity(e)
            
            # Handle based on strategy
            strategy = self.get_recovery_strategy(e)
            
            if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                circuit_breaker = self.get_circuit_breaker(operation)
                circuit_breaker.record_failure()
            
            self.error_history.append(context)
            raise

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics for monitoring."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        recent_errors = [
            error for error in self.error_history
            if time.time() - error.timestamp < 3600  # Last hour
        ]
        
        circuit_breaker_status = {
            name: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time
            }
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            "timestamp": time.time(),
            "system": {
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "available_memory_gb": memory.available / (1024**3)
            },
            "errors": {
                "recent_count": len(recent_errors),
                "total_count": len(self.error_history),
                "severity_distribution": {
                    severity.value: len([
                        e for e in recent_errors 
                        if e.severity == severity
                    ]) for severity in ErrorSeverity
                }
            },
            "circuit_breakers": circuit_breaker_status
        }

    def reset_error_history(self, older_than_hours: int = 24) -> None:
        """Reset error history older than specified hours."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        self.error_history = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]


# Global error handler instance
error_handler = EnhancedErrorHandler()


# Convenience decorators
def with_circuit_breaker(operation: str):
    """Decorator to add circuit breaker protection."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            circuit_breaker = error_handler.get_circuit_breaker(operation)
            
            if not circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for {operation}")
            
            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                raise
        
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retry logic."""
    return error_handler.with_retry(
        max_attempts=max_attempts,
        delay=delay,
        backoff_factor=backoff_factor,
        exceptions=exceptions
    )
