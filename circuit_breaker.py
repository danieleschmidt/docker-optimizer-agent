#!/usr/bin/env python3
"""Circuit breaker pattern for resilient Docker optimization."""

import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: int = 60,
                 expected_exception: type = Exception):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - failing fast")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker reset to CLOSED")
            
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

def circuit_breaker(failure_threshold: int = 5, 
                   timeout: int = 60,
                   expected_exception: type = Exception):
    """Decorator for applying circuit breaker pattern."""
    breaker = CircuitBreaker(failure_threshold, timeout, expected_exception)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

# Example usage for Docker optimization functions
@circuit_breaker(failure_threshold=3, timeout=30)
def optimize_dockerfile_with_protection(dockerfile_content: str) -> str:
    """Protected Dockerfile optimization with circuit breaker."""
    # Simulate optimization that might fail
    if "FROM scratch" in dockerfile_content:
        raise Exception("Cannot optimize scratch-based Dockerfile")
    
    return dockerfile_content.replace("latest", "22.04")

def main():
    """Test circuit breaker functionality."""
    # Test successful operation
    try:
        result = optimize_dockerfile_with_protection("FROM ubuntu:latest")
        print(f"Optimization successful: {result}")
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # Test circuit breaker with failing operation
    failing_dockerfile = "FROM scratch"
    for i in range(5):
        try:
            result = optimize_dockerfile_with_protection(failing_dockerfile)
            print(f"Attempt {i+1} successful: {result}")
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")

if __name__ == "__main__":
    main()