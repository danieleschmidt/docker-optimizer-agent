"""Tests for enhanced error handling system."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from src.docker_optimizer.enhanced_error_handling import (
    EnhancedErrorHandler,
    ErrorSeverity,
    RecoveryStrategy,
    CircuitBreaker,
    CircuitBreakerState,
    ErrorContext,
    error_handler,
    with_circuit_breaker,
    retry_on_failure
)


class TestErrorContext:
    """Test ErrorContext functionality."""
    
    def test_error_context_creation(self):
        """Test error context creation."""
        context = ErrorContext(
            error_type="TestError",
            severity=ErrorSeverity.HIGH,
            operation="test_operation"
        )
        
        assert context.error_type == "TestError"
        assert context.severity == ErrorSeverity.HIGH
        assert context.operation == "test_operation"
        assert context.recovery_attempts == 0
        assert context.can_retry()
    
    def test_can_retry_logic(self):
        """Test retry logic."""
        context = ErrorContext(
            error_type="TestError",
            severity=ErrorSeverity.MEDIUM,
            max_recovery_attempts=2
        )
        
        assert context.can_retry()
        
        context.recovery_attempts = 1
        assert context.can_retry()
        
        context.recovery_attempts = 2
        assert not context.can_retry()


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_initial_state(self):
        """Test initial circuit breaker state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute()
        assert cb.failure_count == 0
    
    def test_failure_threshold(self):
        """Test failure threshold behavior."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        for i in range(3):
            cb.record_failure()
            if i < 2:
                assert cb.state == CircuitBreakerState.CLOSED
            else:
                assert cb.state == CircuitBreakerState.OPEN
                assert not cb.can_execute()
    
    def test_recovery_timeout(self):
        """Test recovery timeout behavior."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # Trip the breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        assert cb.can_execute()  # Should transition to HALF_OPEN
        assert cb.state == CircuitBreakerState.HALF_OPEN
    
    def test_half_open_success(self):
        """Test successful recovery from HALF_OPEN state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, success_threshold=2)
        
        # Trip the breaker
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait and transition to HALF_OPEN
        time.sleep(0.02)
        assert cb.can_execute()
        
        # Record successes to close the breaker
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_half_open_failure(self):
        """Test failure in HALF_OPEN state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        
        # Trip the breaker
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait and transition to HALF_OPEN
        time.sleep(0.02)
        assert cb.can_execute()
        
        # Record failure in HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestEnhancedErrorHandler:
    """Test EnhancedErrorHandler functionality."""
    
    def setup_method(self):
        """Setup test method."""
        self.handler = EnhancedErrorHandler()
    
    def test_error_severity_classification(self):
        """Test error severity classification."""
        # Test critical errors
        memory_error = MemoryError("Out of memory")
        assert self.handler.classify_error_severity(memory_error) == ErrorSeverity.CRITICAL
        
        # Test medium errors
        connection_error = ConnectionError("Connection failed")
        assert self.handler.classify_error_severity(connection_error) == ErrorSeverity.MEDIUM
        
        # Test low errors (unknown error type)
        custom_error = Exception("Custom error")
        assert self.handler.classify_error_severity(custom_error) == ErrorSeverity.LOW
    
    def test_recovery_strategy_detection(self):
        """Test recovery strategy detection."""
        # Test specific error types
        connection_error = ConnectionError("Connection failed")
        strategy = self.handler.get_recovery_strategy(connection_error)
        assert strategy == RecoveryStrategy.CIRCUIT_BREAKER
        
        # Test critical errors
        memory_error = MemoryError("Out of memory")
        strategy = self.handler.get_recovery_strategy(memory_error)
        assert strategy == RecoveryStrategy.FAIL_FAST
    
    def test_circuit_breaker_registration(self):
        """Test circuit breaker registration and retrieval."""
        operation = "test_operation"
        cb1 = self.handler.get_circuit_breaker(operation)
        cb2 = self.handler.get_circuit_breaker(operation)
        
        # Should return the same instance
        assert cb1 is cb2
        assert operation in self.handler.circuit_breakers
    
    def test_fallback_registration(self):
        """Test fallback function registration."""
        def test_fallback():
            return "fallback_result"
        
        self.handler.register_fallback("TestError", test_fallback)
        assert "TestError" in self.handler.fallback_strategies
        assert self.handler.fallback_strategies["TestError"] == test_fallback
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test async error handling."""
        test_error = ValueError("Test error")
        
        with pytest.raises(ValueError):
            await self.handler.handle_error_async(test_error, "test_operation")
        
        # Check that error was recorded
        assert len(self.handler.error_history) > 0
        assert self.handler.error_history[-1].error_type == "ValueError"
    
    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        """Test fallback execution."""
        def sync_fallback():
            return "sync_fallback_result"
        
        async def async_fallback():
            return "async_fallback_result"
        
        self.handler.register_fallback("TestError", sync_fallback)
        
        test_error = Exception("Test error")
        test_error.__class__.__name__ = "TestError"
        
        # Test with fallback (should not raise)
        result = await self.handler.handle_error_async(test_error)
        assert result == "sync_fallback_result"
        
        # Test async fallback
        self.handler.register_fallback("AsyncTestError", async_fallback)
        async_test_error = Exception("Async test error")
        async_test_error.__class__.__name__ = "AsyncTestError"
        
        result = await self.handler.handle_error_async(async_test_error)
        assert result == "async_fallback_result"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation."""
        self.handler.error_patterns["TestError"] = RecoveryStrategy.GRACEFUL_DEGRADATION
        
        test_error = Exception("Test error")
        test_error.__class__.__name__ = "TestError"
        
        result = await self.handler.handle_error_async(test_error, "test_operation")
        
        assert isinstance(result, dict)
        assert result["status"] == "degraded"
        assert result["operation"] == "test_operation"
    
    def test_error_context_manager(self):
        """Test error context manager."""
        with pytest.raises(ValueError):
            with self.handler.error_context("test_operation") as context:
                assert context.operation == "test_operation"
                raise ValueError("Test error")
        
        # Check that error was recorded
        assert len(self.handler.error_history) > 0
        last_error = self.handler.error_history[-1]
        assert last_error.error_type == "ValueError"
        assert last_error.operation == "test_operation"
    
    def test_system_health_metrics(self):
        """Test system health metrics."""
        # Add some error history
        context = ErrorContext(
            error_type="TestError",
            severity=ErrorSeverity.HIGH,
            operation="test_op"
        )
        self.handler.error_history.append(context)
        
        health = self.handler.get_system_health()
        
        assert "timestamp" in health
        assert "system" in health
        assert "errors" in health
        assert "circuit_breakers" in health
        
        assert health["errors"]["total_count"] >= 1
        assert health["system"]["memory_percent"] > 0
    
    def test_error_history_cleanup(self):
        """Test error history cleanup."""
        # Add old errors
        old_context = ErrorContext(
            error_type="OldError",
            severity=ErrorSeverity.LOW,
            timestamp=time.time() - 25 * 3600  # 25 hours ago
        )
        
        recent_context = ErrorContext(
            error_type="RecentError",
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time() - 1 * 3600  # 1 hour ago
        )
        
        self.handler.error_history.extend([old_context, recent_context])
        
        # Clean up errors older than 24 hours
        self.handler.reset_error_history(older_than_hours=24)
        
        # Should only have recent error
        assert len(self.handler.error_history) == 1
        assert self.handler.error_history[0].error_type == "RecentError"


class TestDecorators:
    """Test error handling decorators."""
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        
        @with_circuit_breaker("test_operation")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Check that circuit breaker was created
        cb = error_handler.get_circuit_breaker("test_operation")
        assert cb is not None
    
    def test_circuit_breaker_decorator_with_failure(self):
        """Test circuit breaker decorator with failure."""
        
        @with_circuit_breaker("failing_operation")
        def failing_function():
            raise ValueError("Test failure")
        
        cb = error_handler.get_circuit_breaker("failing_operation")
        
        # Should fail and record failure
        with pytest.raises(ValueError):
            failing_function()
        
        assert cb.failure_count > 0
    
    @pytest.mark.asyncio
    async def test_retry_decorator_async(self):
        """Test retry decorator with async function."""
        call_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.01)
        async def flaky_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await flaky_async_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_decorator_sync(self):
        """Test retry decorator with sync function."""
        call_count = 0
        
        @retry_on_failure(max_attempts=3, delay=0.01)
        def flaky_sync_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_sync_function()
        assert result == "success"
        assert call_count == 2
    
    def test_retry_decorator_max_attempts_exceeded(self):
        """Test retry decorator when max attempts are exceeded."""
        
        @retry_on_failure(max_attempts=2, delay=0.01)
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_failing_function()


@pytest.mark.integration
class TestErrorHandlerIntegration:
    """Integration tests for error handling system."""
    
    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow."""
        handler = EnhancedErrorHandler()
        
        # Register fallback
        def fallback_function():
            return "fallback_executed"
        
        handler.register_fallback("TestError", fallback_function)
        
        # Test with circuit breaker and fallback
        @with_circuit_breaker("integration_test")
        def test_operation():
            error = Exception("Test error")
            error.__class__.__name__ = "TestError"
            raise error
        
        # First call should trigger fallback (if properly handled)
        # In this test, we're just testing the setup
        with pytest.raises(Exception):
            test_operation()
        
        # Verify circuit breaker was used
        cb = handler.get_circuit_breaker("integration_test")
        assert cb.failure_count > 0