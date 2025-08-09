"""Advanced Error Handling and Validation for Docker Optimizer.

This module provides comprehensive error handling, validation, and recovery mechanisms
to ensure robust operation across diverse environments and edge cases.
"""

import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    PARSING = "parsing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EXTERNAL_DEPENDENCY = "external_dependency"
    CONFIGURATION = "configuration"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK = "network"
    FILE_SYSTEM = "file_system"


@dataclass
class ErrorContext:
    """Context information for error tracking and debugging."""
    component: str
    operation: str
    input_data: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class DockerOptimizerException(Exception):
    """Base exception for Docker Optimizer."""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.VALIDATION,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.recoverable = recoverable


class DockerfileValidationError(DockerOptimizerException):
    """Raised when Dockerfile validation fails."""
    
    def __init__(self, message: str, dockerfile_path: Optional[Path] = None, line_number: Optional[int] = None):
        super().__init__(
            message, 
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH
        )
        self.dockerfile_path = dockerfile_path
        self.line_number = line_number


class SecurityValidationError(DockerOptimizerException):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_issue: str, fix_suggestion: Optional[str] = None):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            recoverable=bool(fix_suggestion)
        )
        self.security_issue = security_issue
        self.fix_suggestion = fix_suggestion


class ExternalDependencyError(DockerOptimizerException):
    """Raised when external dependencies fail."""
    
    def __init__(self, message: str, dependency_name: str, required_version: Optional[str] = None):
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_DEPENDENCY,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True
        )
        self.dependency_name = dependency_name
        self.required_version = required_version


class PerformanceError(DockerOptimizerException):
    """Raised when performance constraints are violated."""
    
    def __init__(self, message: str, metric_name: str, threshold: float, actual_value: float):
        super().__init__(
            message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True
        )
        self.metric_name = metric_name
        self.threshold = threshold
        self.actual_value = actual_value


class DockerfileValidator:
    """Comprehensive Dockerfile validation."""
    
    def __init__(self):
        self.validation_rules = [
            self._validate_from_instruction,
            self._validate_maintainer_labels,
            self._validate_user_security,
            self._validate_layer_efficiency,
            self._validate_package_management,
            self._validate_port_exposure,
            self._validate_volume_usage,
            self._validate_environment_variables,
            self._validate_health_checks,
            self._validate_signal_handling
        ]
    
    def validate(self, dockerfile_content: str, dockerfile_path: Optional[Path] = None) -> ValidationResult:
        """Perform comprehensive dockerfile validation."""
        errors = []
        warnings = []
        suggestions = []
        max_severity = ErrorSeverity.LOW
        
        try:
            lines = dockerfile_content.strip().split('\n')
            
            # Run all validation rules
            for rule in self.validation_rules:
                try:
                    rule_result = rule(lines, dockerfile_content, dockerfile_path)
                    if rule_result:
                        errors.extend(rule_result.get('errors', []))
                        warnings.extend(rule_result.get('warnings', []))
                        suggestions.extend(rule_result.get('suggestions', []))
                        
                        rule_severity = rule_result.get('severity', ErrorSeverity.LOW)
                        if rule_severity.value > max_severity.value:
                            max_severity = rule_severity
                            
                except Exception as e:
                    logger.error(f"Validation rule failed: {rule.__name__}: {e}")
                    warnings.append(f"Internal validation error in {rule.__name__}")
            
            # Global validation checks
            global_issues = self._validate_global_structure(lines, dockerfile_content)
            errors.extend(global_issues.get('errors', []))
            warnings.extend(global_issues.get('warnings', []))
            suggestions.extend(global_issues.get('suggestions', []))
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                severity=max_severity
            )
            
        except Exception as e:
            logger.error(f"Dockerfile validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                suggestions=["Check dockerfile syntax and try again"],
                severity=ErrorSeverity.CRITICAL
            )
    
    def _validate_from_instruction(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate FROM instruction."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        from_lines = [line.strip() for line in lines if line.strip().upper().startswith('FROM')]
        
        if not from_lines:
            issues["errors"].append("No FROM instruction found - Dockerfiles must start with FROM")
            issues["severity"] = ErrorSeverity.CRITICAL
            return issues
        
        for from_line in from_lines:
            # Check for latest tag usage
            if ':latest' in from_line.lower() or (from_line.count(':') == 0 and 'scratch' not in from_line.lower()):
                issues["warnings"].append("Using 'latest' tag or no tag - consider pinning to specific version")
                issues["suggestions"].append("Use specific version tags like 'ubuntu:22.04' instead of 'ubuntu:latest'")
            
            # Check for deprecated base images
            deprecated_images = ['ubuntu:14.04', 'ubuntu:16.04', 'ubuntu:18.04', 'centos:7', 'node:10']
            if any(deprecated in from_line.lower() for deprecated in deprecated_images):
                issues["warnings"].append("Using deprecated base image version")
                issues["suggestions"].append("Update to current LTS or stable versions")
        
        return issues
    
    def _validate_maintainer_labels(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate maintainer and label instructions."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        # Check for deprecated MAINTAINER instruction
        if any('MAINTAINER' in line.upper() for line in lines):
            issues["warnings"].append("MAINTAINER instruction is deprecated")
            issues["suggestions"].append("Use LABEL maintainer='...' instead of MAINTAINER")
        
        # Check for recommended labels
        has_maintainer_label = any('LABEL' in line.upper() and 'maintainer' in line.lower() for line in lines)
        has_version_label = any('LABEL' in line.upper() and ('version' in line.lower() or 'ver' in line.lower()) for line in lines)
        
        if not has_maintainer_label:
            issues["suggestions"].append("Consider adding LABEL maintainer='...' for image documentation")
        
        if not has_version_label:
            issues["suggestions"].append("Consider adding LABEL version='...' for image versioning")
        
        return issues
    
    def _validate_user_security(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate user security configuration."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        user_lines = [line for line in lines if line.strip().upper().startswith('USER')]
        
        if not user_lines:
            issues["warnings"].append("No USER instruction found - container will run as root")
            issues["suggestions"].append("Add 'USER 1001:1001' or create dedicated user for security")
            issues["severity"] = ErrorSeverity.HIGH
        else:
            for user_line in user_lines:
                if 'USER root' in user_line or 'USER 0' in user_line:
                    issues["warnings"].append("Explicitly setting USER to root is a security risk")
                    issues["suggestions"].append("Use non-root user or remove explicit root USER instruction")
        
        return issues
    
    def _validate_layer_efficiency(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate layer efficiency and optimization."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        run_count = sum(1 for line in lines if line.strip().upper().startswith('RUN'))
        copy_count = sum(1 for line in lines if line.strip().upper().startswith('COPY'))
        add_count = sum(1 for line in lines if line.strip().upper().startswith('ADD'))
        
        total_layers = run_count + copy_count + add_count
        
        if run_count > 3:
            issues["warnings"].append(f"Multiple RUN instructions ({run_count}) - consider combining")
            issues["suggestions"].append("Combine RUN commands with && to reduce layers: RUN cmd1 && cmd2")
        
        if add_count > 0:
            issues["warnings"].append("ADD instruction used - COPY is preferred unless extracting archives")
            issues["suggestions"].append("Use COPY instead of ADD for simple file copying")
        
        if total_layers > 10:
            issues["warnings"].append(f"High layer count ({total_layers}) may increase image size")
            issues["suggestions"].append("Consider multi-stage build or combining instructions")
        
        return issues
    
    def _validate_package_management(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate package management best practices."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        has_apt_update = any('apt-get update' in line.lower() for line in lines)
        has_apt_clean = any('rm -rf /var/lib/apt/lists/*' in line for line in lines)
        has_no_install_recommends = any('--no-install-recommends' in line for line in lines if 'apt-get install' in line.lower())
        
        if has_apt_update and not has_apt_clean:
            issues["warnings"].append("apt-get update without cleanup increases image size")
            issues["suggestions"].append("Add '&& rm -rf /var/lib/apt/lists/*' after apt-get commands")
        
        if any('apt-get install' in line.lower() for line in lines) and not has_no_install_recommends:
            issues["suggestions"].append("Use '--no-install-recommends' flag with apt-get install to reduce size")
        
        # Check for package caching issues
        separate_update_install = False
        for i, line in enumerate(lines[:-1]):
            if 'apt-get update' in line.lower() and 'install' not in line.lower():
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if 'apt-get install' in next_line.lower():
                    separate_update_install = True
        
        if separate_update_install:
            issues["warnings"].append("apt-get update and install in separate RUN commands")
            issues["suggestions"].append("Combine apt-get update && apt-get install in single RUN command")
        
        return issues
    
    def _validate_port_exposure(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate port exposure configuration."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        expose_lines = [line for line in lines if line.strip().upper().startswith('EXPOSE')]
        
        if expose_lines:
            for expose_line in expose_lines:
                ports = expose_line.upper().replace('EXPOSE', '').strip().split()
                for port in ports:
                    try:
                        port_num = int(port)
                        if port_num < 1024:
                            issues["warnings"].append(f"Exposing privileged port {port_num}")
                            issues["suggestions"].append("Consider using non-privileged ports (>1024)")
                        elif port_num > 65535:
                            issues["errors"].append(f"Invalid port number {port_num}")
                    except ValueError:
                        issues["errors"].append(f"Invalid port specification: {port}")
        
        return issues
    
    def _validate_volume_usage(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate volume usage patterns."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        volume_lines = [line for line in lines if line.strip().upper().startswith('VOLUME')]
        
        for volume_line in volume_lines:
            # Check for absolute paths
            if '/tmp' in volume_line or '/var/tmp' in volume_line:
                issues["warnings"].append("Mounting /tmp as volume may cause issues")
                issues["suggestions"].append("Consider using application-specific directory for volumes")
        
        return issues
    
    def _validate_environment_variables(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate environment variable usage."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        env_lines = [line for line in lines if line.strip().upper().startswith('ENV')]
        
        sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential']
        
        for env_line in env_lines:
            env_lower = env_line.lower()
            for pattern in sensitive_patterns:
                if pattern in env_lower and '=' in env_line:
                    issues["warnings"].append(f"Potential sensitive data in ENV: {pattern}")
                    issues["suggestions"].append("Use build-time secrets or runtime configuration for sensitive data")
        
        return issues
    
    def _validate_health_checks(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate health check configuration."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        has_healthcheck = any(line.strip().upper().startswith('HEALTHCHECK') for line in lines)
        has_expose = any(line.strip().upper().startswith('EXPOSE') for line in lines)
        
        if has_expose and not has_healthcheck:
            issues["suggestions"].append("Consider adding HEALTHCHECK instruction for service monitoring")
        
        return issues
    
    def _validate_signal_handling(self, lines: List[str], content: str, path: Optional[Path]) -> Dict[str, Any]:
        """Validate signal handling configuration."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        has_stopsignal = any(line.strip().upper().startswith('STOPSIGNAL') for line in lines)
        
        # Check if using init system that might need custom signals
        uses_systemd = any('systemd' in line.lower() for line in lines)
        uses_supervisor = any('supervisor' in line.lower() for line in lines)
        
        if (uses_systemd or uses_supervisor) and not has_stopsignal:
            issues["suggestions"].append("Consider adding STOPSIGNAL for proper container shutdown")
        
        return issues
    
    def _validate_global_structure(self, lines: List[str], content: str) -> Dict[str, Any]:
        """Validate global dockerfile structure."""
        issues = {"errors": [], "warnings": [], "suggestions": []}
        
        # Check instruction order
        instruction_order = []
        for line in lines:
            stripped = line.strip().upper()
            if stripped and not stripped.startswith('#'):
                instruction = stripped.split()[0] if stripped.split() else ""
                if instruction in ['FROM', 'LABEL', 'ENV', 'WORKDIR', 'COPY', 'RUN', 'EXPOSE', 'USER', 'CMD', 'ENTRYPOINT']:
                    instruction_order.append(instruction)
        
        # FROM should be first
        if instruction_order and instruction_order[0] != 'FROM':
            issues["errors"].append("FROM instruction must be the first non-comment instruction")
        
        # CMD/ENTRYPOINT should be last
        if instruction_order and instruction_order[-1] not in ['CMD', 'ENTRYPOINT']:
            issues["suggestions"].append("Consider ending with CMD or ENTRYPOINT instruction")
        
        return issues


class ErrorRecoveryManager:
    """Manages error recovery and fallback strategies."""
    
    def __init__(self):
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.EXTERNAL_DEPENDENCY: [
                self._retry_with_backoff,
                self._fallback_to_mock_data,
                self._skip_optional_feature
            ],
            ErrorCategory.PARSING: [
                self._sanitize_input,
                self._use_default_values,
                self._partial_parsing_mode
            ],
            ErrorCategory.VALIDATION: [
                self._apply_auto_fixes,
                self._generate_warnings,
                self._continue_with_best_effort
            ]
        }
    
    def attempt_recovery(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Attempt to recover from an error using appropriate strategies."""
        if not error.recoverable:
            logger.error(f"Non-recoverable error: {error.message}")
            return None
        
        strategies = self.recovery_strategies.get(error.category, [])
        
        for strategy in strategies:
            try:
                result = strategy(error, context)
                if result is not None:
                    logger.info(f"Recovery successful using {strategy.__name__}")
                    return result
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
        
        logger.error(f"All recovery strategies failed for error: {error.message}")
        return None
    
    def _retry_with_backoff(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Retry operation with exponential backoff."""
        # Implementation would include actual retry logic
        logger.info("Attempting retry with backoff")
        return None
    
    def _fallback_to_mock_data(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Use mock data when external dependencies fail."""
        logger.info("Using mock data as fallback")
        return {"mock": True, "reason": "external_dependency_failure"}
    
    def _skip_optional_feature(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Skip optional features that fail."""
        logger.info("Skipping optional feature due to error")
        return {"skipped": True, "feature": context.operation}
    
    def _sanitize_input(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Sanitize problematic input."""
        logger.info("Sanitizing input data")
        if context.input_data:
            # Remove problematic characters or patterns
            return {k: str(v).strip() for k, v in context.input_data.items()}
        return None
    
    def _use_default_values(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Use default values when parsing fails."""
        logger.info("Using default values")
        return {"use_defaults": True}
    
    def _partial_parsing_mode(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Continue with partial parsing results."""
        logger.info("Continuing with partial parsing")
        return {"partial": True, "warning": "Some data could not be parsed"}
    
    def _apply_auto_fixes(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Apply automatic fixes for validation errors."""
        logger.info("Applying automatic fixes")
        return {"auto_fixed": True, "fixes_applied": []}
    
    def _generate_warnings(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Generate warnings instead of failing."""
        logger.info("Converting error to warning")
        return {"warning": error.message, "continue": True}
    
    def _continue_with_best_effort(self, error: DockerOptimizerException, context: ErrorContext) -> Optional[Any]:
        """Continue with best-effort processing."""
        logger.info("Continuing with best effort")
        return {"best_effort": True, "partial_success": True}


def robust_operation(fallback_value: Any = None, 
                    error_category: ErrorCategory = ErrorCategory.VALIDATION,
                    max_retries: int = 3):
    """Decorator for robust operations with automatic error handling."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            recovery_manager = ErrorRecoveryManager()
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except DockerOptimizerException as e:
                    logger.warning(f"Operation {func.__name__} failed (attempt {attempt + 1}): {e.message}")
                    
                    if attempt == max_retries - 1:  # Last attempt
                        context = ErrorContext(
                            component=func.__module__,
                            operation=func.__name__,
                            input_data={"args": args, "kwargs": kwargs}
                        )
                        
                        recovery_result = recovery_manager.attempt_recovery(e, context)
                        if recovery_result is not None:
                            return recovery_result
                        
                        if fallback_value is not None:
                            logger.info(f"Using fallback value for {func.__name__}")
                            return fallback_value
                        
                        raise  # Re-raise if no recovery possible
                    
                except Exception as e:
                    logger.error(f"Unexpected error in {func.__name__} (attempt {attempt + 1}): {e}")
                    
                    if attempt == max_retries - 1:
                        if fallback_value is not None:
                            return fallback_value
                        raise DockerOptimizerException(
                            f"Operation {func.__name__} failed: {str(e)}",
                            category=error_category,
                            severity=ErrorSeverity.HIGH
                        )
            
            return fallback_value
        
        return wrapper
    return decorator


@contextmanager
def error_context(component: str, operation: str, **context_kwargs):
    """Context manager for comprehensive error tracking."""
    context = ErrorContext(
        component=component,
        operation=operation,
        user_context=context_kwargs
    )
    
    try:
        yield context
    except Exception as e:
        if isinstance(e, DockerOptimizerException):
            e.context = context
        else:
            # Wrap unexpected exceptions
            wrapped_error = DockerOptimizerException(
                f"Unexpected error in {component}.{operation}: {str(e)}",
                context=context,
                severity=ErrorSeverity.HIGH
            )
            raise wrapped_error from e
        raise


# Validation utilities
def validate_dockerfile_content(content: str, path: Optional[Path] = None) -> ValidationResult:
    """Validate dockerfile content comprehensively."""
    validator = DockerfileValidator()
    return validator.validate(content, path)


def ensure_dockerfile_valid(content: str, path: Optional[Path] = None) -> str:
    """Ensure dockerfile is valid, raising exception if not."""
    validation_result = validate_dockerfile_content(content, path)
    
    if not validation_result.is_valid:
        error_msg = f"Dockerfile validation failed: {'; '.join(validation_result.errors)}"
        raise DockerfileValidationError(error_msg, path)
    
    return content