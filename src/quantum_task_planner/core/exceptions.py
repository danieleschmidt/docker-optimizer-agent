"""Custom exceptions for quantum task planner."""

from typing import Optional, List, Dict, Any


class QuantumTaskPlannerError(Exception):
    """Base exception for all quantum task planner errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize with message and optional details.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation with details."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ValidationError(QuantumTaskPlannerError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            validation_errors: List of specific validation errors
            details: Additional error details
        """
        super().__init__(message, details)
        self.validation_errors = validation_errors or []
    
    def __str__(self) -> str:
        """String representation with validation errors."""
        base_str = super().__str__()
        if self.validation_errors:
            errors_str = "; ".join(self.validation_errors)
            return f"{base_str} - Validation errors: {errors_str}"
        return base_str


class OptimizationError(QuantumTaskPlannerError):
    """Raised when optimization fails."""
    
    def __init__(self, message: str, algorithm: Optional[str] = None,
                 iteration: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize optimization error.
        
        Args:
            message: Error message
            algorithm: Algorithm that failed
            iteration: Iteration where failure occurred
            details: Additional error details
        """
        if not details:
            details = {}
        if algorithm:
            details['algorithm'] = algorithm
        if iteration is not None:
            details['iteration'] = iteration
        
        super().__init__(message, details)
        self.algorithm = algorithm
        self.iteration = iteration


class ResourceAllocationError(QuantumTaskPlannerError):
    """Raised when resource allocation fails."""
    
    def __init__(self, message: str, resource_id: Optional[str] = None,
                 task_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize resource allocation error.
        
        Args:
            message: Error message
            resource_id: ID of resource involved in error
            task_id: ID of task involved in error
            details: Additional error details
        """
        if not details:
            details = {}
        if resource_id:
            details['resource_id'] = resource_id
        if task_id:
            details['task_id'] = task_id
        
        super().__init__(message, details)
        self.resource_id = resource_id
        self.task_id = task_id


class SchedulingError(QuantumTaskPlannerError):
    """Raised when scheduling fails."""
    
    def __init__(self, message: str, schedule_id: Optional[str] = None,
                 constraint_violations: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        """Initialize scheduling error.
        
        Args:
            message: Error message
            schedule_id: ID of schedule that failed
            constraint_violations: List of constraint violations
            details: Additional error details
        """
        if not details:
            details = {}
        if schedule_id:
            details['schedule_id'] = schedule_id
        
        super().__init__(message, details)
        self.schedule_id = schedule_id
        self.constraint_violations = constraint_violations or []


class DependencyError(QuantumTaskPlannerError):
    """Raised when task dependency issues occur."""
    
    def __init__(self, message: str, task_id: Optional[str] = None,
                 dependency_id: Optional[str] = None, cycle_path: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        """Initialize dependency error.
        
        Args:
            message: Error message
            task_id: ID of task with dependency issue
            dependency_id: ID of problematic dependency
            cycle_path: Path of circular dependency if applicable
            details: Additional error details
        """
        if not details:
            details = {}
        if task_id:
            details['task_id'] = task_id
        if dependency_id:
            details['dependency_id'] = dependency_id
        if cycle_path:
            details['cycle_path'] = cycle_path
        
        super().__init__(message, details)
        self.task_id = task_id
        self.dependency_id = dependency_id
        self.cycle_path = cycle_path or []


class QuantumAlgorithmError(QuantumTaskPlannerError):
    """Raised when quantum algorithm operations fail."""
    
    def __init__(self, message: str, algorithm_type: Optional[str] = None,
                 quantum_parameter: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize quantum algorithm error.
        
        Args:
            message: Error message
            algorithm_type: Type of quantum algorithm
            quantum_parameter: Specific quantum parameter that caused error
            details: Additional error details
        """
        if not details:
            details = {}
        if algorithm_type:
            details['algorithm_type'] = algorithm_type
        if quantum_parameter:
            details['quantum_parameter'] = quantum_parameter
        
        super().__init__(message, details)
        self.algorithm_type = algorithm_type
        self.quantum_parameter = quantum_parameter


class ConfigurationError(QuantumTaskPlannerError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_value: Optional[Any] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused error
            config_value: Invalid configuration value
            details: Additional error details
        """
        if not details:
            details = {}
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = config_value
        
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class TimeoutError(QuantumTaskPlannerError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize timeout error.
        
        Args:
            message: Error message
            timeout_seconds: Timeout duration that was exceeded
            operation: Operation that timed out
            details: Additional error details
        """
        if not details:
            details = {}
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class ConvergenceError(QuantumTaskPlannerError):
    """Raised when optimization fails to converge."""
    
    def __init__(self, message: str, max_iterations: Optional[int] = None,
                 final_energy: Optional[float] = None, 
                 convergence_threshold: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None):
        """Initialize convergence error.
        
        Args:
            message: Error message
            max_iterations: Maximum iterations attempted
            final_energy: Final energy value achieved
            convergence_threshold: Required convergence threshold
            details: Additional error details
        """
        if not details:
            details = {}
        if max_iterations:
            details['max_iterations'] = max_iterations
        if final_energy is not None:
            details['final_energy'] = final_energy
        if convergence_threshold is not None:
            details['convergence_threshold'] = convergence_threshold
        
        super().__init__(message, details)
        self.max_iterations = max_iterations
        self.final_energy = final_energy
        self.convergence_threshold = convergence_threshold


class ResourceCapacityError(ResourceAllocationError):
    """Raised when resource capacity is exceeded."""
    
    def __init__(self, message: str, resource_id: str, 
                 requested_capacity: float, available_capacity: float,
                 details: Optional[Dict[str, Any]] = None):
        """Initialize resource capacity error.
        
        Args:
            message: Error message
            resource_id: ID of over-allocated resource
            requested_capacity: Requested capacity amount
            available_capacity: Available capacity amount
            details: Additional error details
        """
        if not details:
            details = {}
        details.update({
            'requested_capacity': requested_capacity,
            'available_capacity': available_capacity
        })
        
        super().__init__(message, resource_id=resource_id, details=details)
        self.requested_capacity = requested_capacity
        self.available_capacity = available_capacity


class TaskNotFoundError(QuantumTaskPlannerError):
    """Raised when a task cannot be found."""
    
    def __init__(self, task_id: str, details: Optional[Dict[str, Any]] = None):
        """Initialize task not found error.
        
        Args:
            task_id: ID of task that was not found
            details: Additional error details
        """
        message = f"Task not found: {task_id}"
        if not details:
            details = {}
        details['task_id'] = task_id
        
        super().__init__(message, details)
        self.task_id = task_id


class ResourceNotFoundError(QuantumTaskPlannerError):
    """Raised when a resource cannot be found."""
    
    def __init__(self, resource_id: str, details: Optional[Dict[str, Any]] = None):
        """Initialize resource not found error.
        
        Args:
            resource_id: ID of resource that was not found
            details: Additional error details
        """
        message = f"Resource not found: {resource_id}"
        if not details:
            details = {}
        details['resource_id'] = resource_id
        
        super().__init__(message, details)
        self.resource_id = resource_id


class ScheduleNotFoundError(QuantumTaskPlannerError):
    """Raised when a schedule cannot be found."""
    
    def __init__(self, schedule_id: str, details: Optional[Dict[str, Any]] = None):
        """Initialize schedule not found error.
        
        Args:
            schedule_id: ID of schedule that was not found
            details: Additional error details
        """
        message = f"Schedule not found: {schedule_id}"
        if not details:
            details = {}
        details['schedule_id'] = schedule_id
        
        super().__init__(message, details)
        self.schedule_id = schedule_id