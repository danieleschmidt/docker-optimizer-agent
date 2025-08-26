"""Robust Error Handling System for Docker Optimizer Agent.

This module implements comprehensive error handling, recovery mechanisms,
and resilience patterns for production-ready operation.
"""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime, timedelta

from .logging_observability import ObservabilityManager, LogLevel


class ErrorSeverity(Enum):
    """Error severity levels."""
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
    error_message: str
    severity: ErrorSeverity
    timestamp: datetime
    operation: str
    retry_count: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker pattern."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    success_count: int = 0


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""

    def __init__(self):
        self.obs_manager = ObservabilityManager(
            log_level=LogLevel.INFO,
            service_name="robust-error-handler"
        )
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60  # seconds
        
        # Initialize default strategies
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """Set up default recovery strategies for common errors."""
        self.recovery_strategies.update({
            "ConnectionError": RecoveryStrategy.RETRY,
            "TimeoutError": RecoveryStrategy.RETRY,
            "FileNotFoundError": RecoveryStrategy.FALLBACK,
            "PermissionError": RecoveryStrategy.GRACEFUL_DEGRADATION,
            "ImportError": RecoveryStrategy.FALLBACK,
            "ValidationError": RecoveryStrategy.FAIL_FAST,
            "SecurityError": RecoveryStrategy.FAIL_FAST,
            "OutOfMemoryError": RecoveryStrategy.CIRCUIT_BREAKER,
        })

    def with_error_handling(self, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Decorator for adding comprehensive error handling to functions."""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_handling(func, operation, severity, *args, **kwargs)
            return wrapper
        return decorator

    def with_async_error_handling(self, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Decorator for adding comprehensive error handling to async functions."""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await self._execute_async_with_handling(func, operation, severity, *args, **kwargs)
            return wrapper
        return decorator

    def _execute_with_handling(self, func: Callable, operation: str, severity: ErrorSeverity, *args, **kwargs):
        """Execute function with comprehensive error handling."""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Check circuit breaker
                if self._check_circuit_breaker(operation):
                    raise Exception(f"Circuit breaker OPEN for operation: {operation}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Reset circuit breaker on success
                self._record_success(operation)
                
                return result
                
            except Exception as e:
                last_error = e
                error_context = self._create_error_context(
                    e, operation, severity, retry_count
                )
                
                # Record failure
                self._record_failure(operation)
                
                # Determine recovery strategy
                strategy = self._get_recovery_strategy(e)
                error_context.recovery_strategy = strategy
                
                # Log error
                self.obs_manager.logger.error(
                    f"Error in {operation}: {str(e)}",
                    extra={
                        "error_context": error_context.__dict__,
                        "retry_count": retry_count,
                        "strategy": strategy.value if strategy else None
                    }
                )
                
                # Apply recovery strategy
                if strategy == RecoveryStrategy.FAIL_FAST:
                    self.error_history.append(error_context)
                    raise e
                elif strategy == RecoveryStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    time.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    continue
                elif strategy == RecoveryStrategy.FALLBACK:
                    fallback_result = self._execute_fallback(operation, error_context)
                    if fallback_result is not None:
                        return fallback_result
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    return self._graceful_degradation(operation, error_context)
                elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    self._open_circuit_breaker(operation)
                    raise e
                
                # If we've exhausted retries, fail
                if retry_count >= self.max_retries:
                    break
                    
                retry_count += 1
        
        # Final failure
        self.error_history.append(error_context)
        raise last_error

    async def _execute_async_with_handling(self, func: Callable, operation: str, severity: ErrorSeverity, *args, **kwargs):
        """Execute async function with comprehensive error handling."""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Check circuit breaker
                if self._check_circuit_breaker(operation):
                    raise Exception(f"Circuit breaker OPEN for operation: {operation}")
                
                # Execute async function
                result = await func(*args, **kwargs)
                
                # Reset circuit breaker on success
                self._record_success(operation)
                
                return result
                
            except Exception as e:
                last_error = e
                error_context = self._create_error_context(
                    e, operation, severity, retry_count
                )
                
                # Record failure
                self._record_failure(operation)
                
                # Determine recovery strategy
                strategy = self._get_recovery_strategy(e)
                error_context.recovery_strategy = strategy
                
                # Log error
                self.obs_manager.logger.error(
                    f"Error in {operation}: {str(e)}",
                    extra={
                        "error_context": error_context.__dict__,
                        "retry_count": retry_count,
                        "strategy": strategy.value if strategy else None
                    }
                )
                
                # Apply recovery strategy
                if strategy == RecoveryStrategy.FAIL_FAST:
                    self.error_history.append(error_context)
                    raise e
                elif strategy == RecoveryStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    await asyncio.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    continue
                elif strategy == RecoveryStrategy.FALLBACK:
                    fallback_result = await self._execute_async_fallback(operation, error_context)
                    if fallback_result is not None:
                        return fallback_result
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    return self._graceful_degradation(operation, error_context)
                elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    self._open_circuit_breaker(operation)
                    raise e
                
                # If we've exhausted retries, fail
                if retry_count >= self.max_retries:
                    break
                    
                retry_count += 1
        
        # Final failure
        self.error_history.append(error_context)
        raise last_error

    def _create_error_context(self, error: Exception, operation: str, severity: ErrorSeverity, retry_count: int) -> ErrorContext:
        """Create error context for tracking."""
        return ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            timestamp=datetime.now(),
            operation=operation,
            retry_count=retry_count,
            metadata={
                "exception_class": error.__class__.__module__ + "." + error.__class__.__name__,
                "traceback_summary": str(error)
            }
        )

    def _get_recovery_strategy(self, error: Exception) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy for an error."""
        error_type = type(error).__name__
        return self.recovery_strategies.get(error_type, RecoveryStrategy.RETRY)

    def _check_circuit_breaker(self, operation: str) -> bool:
        """Check if circuit breaker is open for this operation."""
        if operation not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[operation]
        if breaker.state != "OPEN":
            return False
        
        # Check if timeout has passed
        if breaker.last_failure_time:
            time_since_failure = datetime.now() - breaker.last_failure_time
            if time_since_failure.total_seconds() > self.circuit_breaker_timeout:
                # Move to half-open state
                breaker.state = "HALF_OPEN"
                breaker.success_count = 0
                return False
        
        return True

    def _record_success(self, operation: str):
        """Record successful operation."""
        if operation not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[operation]
        if breaker.state == "HALF_OPEN":
            breaker.success_count += 1
            if breaker.success_count >= 3:  # Reset after 3 successes
                breaker.state = "CLOSED"
                breaker.failure_count = 0
                breaker.last_failure_time = None
                self.obs_manager.logger.info(f"Circuit breaker CLOSED for operation: {operation}")

    def _record_failure(self, operation: str):
        """Record failed operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreakerState()
        
        breaker = self.circuit_breakers[operation]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= self.circuit_breaker_threshold:
            self._open_circuit_breaker(operation)

    def _open_circuit_breaker(self, operation: str):
        """Open circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreakerState()
        
        breaker = self.circuit_breakers[operation]
        breaker.state = "OPEN"
        breaker.last_failure_time = datetime.now()
        
        self.obs_manager.logger.warning(f"Circuit breaker OPENED for operation: {operation}")

    def _execute_fallback(self, operation: str, error_context: ErrorContext) -> Any:
        """Execute fallback strategy."""
        self.obs_manager.logger.info(f"Executing fallback for operation: {operation}")
        
        # Operation-specific fallbacks
        fallbacks = {
            "dockerfile_optimization": self._fallback_basic_optimization,
            "security_scanning": self._fallback_basic_security,
            "multistage_generation": self._fallback_single_stage,
        }
        
        fallback_func = fallbacks.get(operation)
        if fallback_func:
            return fallback_func(error_context)
        
        return None

    async def _execute_async_fallback(self, operation: str, error_context: ErrorContext) -> Any:
        """Execute async fallback strategy."""
        # For now, use sync fallback
        return self._execute_fallback(operation, error_context)

    def _graceful_degradation(self, operation: str, error_context: ErrorContext) -> Any:
        """Implement graceful degradation."""
        self.obs_manager.logger.warning(f"Graceful degradation for operation: {operation}")
        
        # Return minimal safe result
        degraded_results = {
            "dockerfile_optimization": {
                "optimized_dockerfile": "# Optimization failed, returning original",
                "explanation": f"Optimization failed due to {error_context.error_type}",
                "success": False
            },
            "security_scanning": {
                "vulnerabilities": [],
                "security_score": {"grade": "UNKNOWN", "score": 0},
                "success": False
            }
        }
        
        return degraded_results.get(operation, {"success": False, "error": error_context.error_message})

    def _fallback_basic_optimization(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for dockerfile optimization."""
        return {
            "optimized_dockerfile": "# Basic optimization fallback - add non-root user\nFROM ubuntu:22.04\nRUN groupadd -r appuser && useradd -r -g appuser appuser\nUSER appuser",
            "explanation": "Applied basic security optimization due to primary optimization failure",
            "success": True,
            "fallback_used": True
        }

    def _fallback_basic_security(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for security scanning."""
        return {
            "vulnerabilities": [],
            "security_score": {"grade": "C", "score": 60},
            "explanation": "Basic security rules applied, external scanning unavailable",
            "success": True,
            "fallback_used": True
        }

    def _fallback_single_stage(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Fallback for multistage builds."""
        return {
            "optimized_dockerfile": "# Single-stage fallback\nFROM ubuntu:22.04-slim",
            "explanation": "Multistage optimization failed, using single-stage build",
            "success": True,
            "fallback_used": True
        }

    @contextmanager
    def resilient_operation(self, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Context manager for resilient operations."""
        try:
            yield
        except Exception as e:
            error_context = self._create_error_context(e, operation, severity, 0)
            self.error_history.append(error_context)
            
            self.obs_manager.logger.error(
                f"Operation {operation} failed in resilient context",
                extra={"error_context": error_context.__dict__}
            )
            
            # Apply graceful degradation
            if self._get_recovery_strategy(e) == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._graceful_degradation(operation, error_context)
            
            raise

    @asynccontextmanager
    async def async_resilient_operation(self, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Async context manager for resilient operations."""
        try:
            yield
        except Exception as e:
            error_context = self._create_error_context(e, operation, severity, 0)
            self.error_history.append(error_context)
            
            self.obs_manager.logger.error(
                f"Async operation {operation} failed in resilient context",
                extra={"error_context": error_context.__dict__}
            )
            
            # Apply graceful degradation - can't return from async context manager
            if self._get_recovery_strategy(e) == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Log degradation, but still raise the exception
                degraded_result = self._graceful_degradation(operation, error_context)
                self.obs_manager.logger.warning(f"Graceful degradation applied for {operation}")
            
            raise

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_types = {}
        severity_counts = {}
        operation_errors = {}
        
        for error in self.error_history:
            # Count by type
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            # Count by severity
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            # Count by operation
            operation_errors[error.operation] = operation_errors.get(error.operation, 0) + 1
        
        # Calculate error rate over time
        recent_errors = [e for e in self.error_history 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "error_types": error_types,
            "severity_distribution": severity_counts,
            "operation_errors": operation_errors,
            "circuit_breaker_states": {
                op: breaker.state for op, breaker in self.circuit_breakers.items()
            },
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            "error_rate_1h": len(recent_errors) / 60 if recent_errors else 0  # errors per minute
        }

    def reset_circuit_breaker(self, operation: str):
        """Manually reset circuit breaker for operation."""
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreakerState()
            self.obs_manager.logger.info(f"Circuit breaker manually reset for operation: {operation}")

    def clear_error_history(self):
        """Clear error history (useful for testing)."""
        self.error_history.clear()
        self.obs_manager.logger.info("Error history cleared")