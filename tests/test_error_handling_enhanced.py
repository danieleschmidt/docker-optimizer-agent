"""Tests for enhanced error handling features."""

import pytest
import time
from unittest.mock import patch, MagicMock

from docker_optimizer.error_handling import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    RetryConfig,
    retry_with_exponential_backoff,
    with_resilience,
    DockerOptimizerException,
    ErrorCategory,
    ErrorSeverity
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        assert cb.name == "test"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_success_flow(self):
        """Test successful operation flow."""
        cb = CircuitBreaker("test")
        
        @cb
        def successful_operation():
            return "success"
        
        result = successful_operation()
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_failure_flow(self):
        """Test failure handling and state transitions."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1)
        cb = CircuitBreaker("test", config)
        
        @cb
        def failing_operation():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            failing_operation()
        assert cb.failure_count == 1
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            failing_operation()
        assert cb.failure_count == 2
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_open_state_blocking(self):
        """Test that open circuit breaker blocks calls."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=1)
        cb = CircuitBreaker("test", config)
        cb.state = CircuitBreakerState.OPEN
        cb.failure_count = 1
        cb.last_failure_time = time.time()
        
        @cb
        def test_operation():
            return "should not execute"
        
        with pytest.raises(DockerOptimizerException) as exc_info:
            test_operation()
        
        assert "Circuit breaker" in str(exc_info.value)
        assert "OPEN" in str(exc_info.value)

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, timeout_seconds=0.1)
        cb = CircuitBreaker("test", config)
        
        # Simulate open state
        cb.state = CircuitBreakerState.OPEN
        cb.failure_count = 1
        cb.last_failure_time = time.time() - 0.2  # Past timeout
        
        @cb
        def recovering_operation():
            return "recovered"
        
        # Should transition to half-open and allow calls
        result = recovering_operation()
        assert result == "recovered"
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.success_count == 1
        
        # Another success should close circuit
        recovering_operation()
        assert cb.state == CircuitBreakerState.CLOSED


class TestRetryMechanism:
    """Test retry mechanisms."""

    def test_retry_success_on_first_attempt(self):
        """Test successful operation on first attempt."""
        call_count = 0
        
        @retry_with_exponential_backoff(RetryConfig(max_attempts=3))
        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_operation()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test successful operation after initial failures."""
        call_count = 0
        
        @retry_with_exponential_backoff(RetryConfig(max_attempts=3, base_delay=0.01))
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhaustion(self):
        """Test retry exhaustion after max attempts."""
        call_count = 0
        
        @retry_with_exponential_backoff(RetryConfig(max_attempts=2, base_delay=0.01))
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        with pytest.raises(Exception) as exc_info:
            always_failing()
        
        assert "Always fails" in str(exc_info.value)
        assert call_count == 2

    def test_retry_non_recoverable_exception(self):
        """Test that non-recoverable exceptions are not retried."""
        call_count = 0
        
        @retry_with_exponential_backoff(RetryConfig(max_attempts=3))
        def non_recoverable_failure():
            nonlocal call_count
            call_count += 1
            raise DockerOptimizerException(
                "Critical failure",
                recoverable=False
            )
        
        with pytest.raises(DockerOptimizerException):
            non_recoverable_failure()
        
        assert call_count == 1  # Should not retry


class TestResilienceIntegration:
    """Test integration of circuit breaker and retry mechanisms."""

    def test_resilience_combination(self):
        """Test combined resilience patterns."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01)
        
        call_count = 0
        
        @with_resilience(cb, retry_config)
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt fails")
            return "success"
        
        result = flaky_operation()
        assert result == "success"
        assert call_count == 2
        assert cb.state == CircuitBreakerState.CLOSED

    @patch('time.sleep')
    def test_exponential_backoff_timing(self, mock_sleep):
        """Test exponential backoff timing."""
        call_count = 0
        
        @retry_with_exponential_backoff(
            RetryConfig(max_attempts=3, base_delay=1.0, exponential_base=2.0, jitter=False)
        )
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Fail")
            return "success"
        
        result = failing_operation()
        assert result == "success"
        assert call_count == 3
        
        # Check that sleep was called with increasing delays
        assert mock_sleep.call_count == 2
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0  # First retry delay
        assert calls[1][0][0] == 2.0  # Second retry delay

    def test_jitter_in_backoff(self):
        """Test that jitter is applied to backoff delays."""
        with patch('time.sleep') as mock_sleep:
            call_count = 0
            
            @retry_with_exponential_backoff(
                RetryConfig(max_attempts=3, base_delay=1.0, jitter=True)
            )
            def failing_operation():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise Exception("Fail")
                return "success"
            
            failing_operation()
            
            # With jitter enabled, delays should be between 0.5 and 1.5
            if mock_sleep.called:
                delay = mock_sleep.call_args[0][0]
                assert 0.5 <= delay <= 1.5


class TestErrorTypes:
    """Test custom error types and handling."""

    def test_docker_optimizer_exception_properties(self):
        """Test DockerOptimizerException properties."""
        exc = DockerOptimizerException(
            "Test error",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            recoverable=False
        )
        
        assert exc.message == "Test error"
        assert exc.category == ErrorCategory.SECURITY
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.recoverable is False

    def test_circuit_breaker_with_custom_exceptions(self):
        """Test circuit breaker behavior with custom exceptions."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
        
        @cb
        def failing_with_custom_exception():
            raise DockerOptimizerException(
                "Custom failure",
                category=ErrorCategory.EXTERNAL_DEPENDENCY,
                severity=ErrorSeverity.MEDIUM
            )
        
        with pytest.raises(DockerOptimizerException):
            failing_with_custom_exception()
        
        assert cb.failure_count == 1
        assert cb.state == CircuitBreakerState.OPEN