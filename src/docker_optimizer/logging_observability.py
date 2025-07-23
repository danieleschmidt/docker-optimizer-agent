"""Comprehensive logging and observability infrastructure for Docker optimizer."""

import json
import logging
import logging.handlers
import subprocess
import sys
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator, Generator

import psutil


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OperationContext:
    """Context information for operations and logging."""

    def __init__(
        self,
        operation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        dockerfile_path: Optional[str] = None,
        parent_id: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> None:
        """Initialize operation context.

        Args:
            operation_id: Unique operation identifier
            user_id: User identifier
            dockerfile_path: Path to Dockerfile being processed
            parent_id: Parent operation ID for nested operations
            operation_type: Type of operation being performed
        """
        self.operation_id = operation_id or str(uuid.uuid4())
        self.user_id = user_id
        self.dockerfile_path = dockerfile_path
        self.parent_id = parent_id
        self.operation_type = operation_type
        self.start_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        context = {
            "operation_id": self.operation_id,
            "start_time": self.start_time
        }

        if self.user_id:
            context["user_id"] = self.user_id
        if self.dockerfile_path:
            context["dockerfile_path"] = self.dockerfile_path
        if self.parent_id:
            context["parent_id"] = self.parent_id
        if self.operation_type:
            context["operation_type"] = self.operation_type

        return context


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add context information if available
        if hasattr(record, 'context') and record.context:
            log_data.update(record.context.to_dict())

        # Add extra fields
        if hasattr(record, 'extra_fields') and record.extra_fields:
            log_data.update(record.extra_fields)

        # Add exception information
        if record.exc_info and record.exc_info[0]:
            log_data.update({
                "exception_type": record.exc_info[0].__name__,
                "exception_message": str(record.exc_info[1]),
                "stack_trace": traceback.format_exception(*record.exc_info)
            })

        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with JSON output and context support."""

    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = True
    ) -> None:
        """Initialize structured logger.

        Args:
            name: Logger name
            log_file: Optional log file path
            level: Log level
            enable_console: Whether to enable console logging
        """
        self.name = name
        self.level = level
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))

        # Clear existing handlers
        self._logger.handlers.clear()

        # Set up JSON formatter
        formatter = JSONFormatter()

        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, level.value))
            self._logger.addHandler(file_handler)

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, level.value))
            self._logger.addHandler(console_handler)

    def _log(
        self,
        level: str,
        message: str,
        context: Optional[OperationContext] = None,
        exception: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Internal logging method."""
        extra_fields = extra or {}

        # Create log record
        if exception:
            self._logger._log(
                getattr(logging, level),
                message,
                (),
                exc_info=(type(exception), exception, exception.__traceback__),
                extra={"context": context, "extra_fields": extra_fields}
            )
        else:
            self._logger._log(
                getattr(logging, level),
                message,
                (),
                extra={"context": context, "extra_fields": extra_fields}
            )

    def debug(
        self,
        message: str,
        context: Optional[OperationContext] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log debug message."""
        self._log("DEBUG", message, context, extra=extra)

    def info(
        self,
        message: str,
        context: Optional[OperationContext] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log info message."""
        self._log("INFO", message, context, extra=extra)

    def warning(
        self,
        message: str,
        context: Optional[OperationContext] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log warning message."""
        self._log("WARNING", message, context, extra=extra)

    def error(
        self,
        message: str,
        context: Optional[OperationContext] = None,
        exception: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error message."""
        self._log("ERROR", message, context, exception, extra)

    def critical(
        self,
        message: str,
        context: Optional[OperationContext] = None,
        exception: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, context, exception, extra)


class MetricsCollector:
    """Metrics collection for performance monitoring."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._metrics: Dict[str, Any] = {}
        self._counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], int]] = defaultdict(lambda: defaultdict(int))
        self._gauges: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[Tuple[Tuple[str, str], ...], List[float]]] = defaultdict(lambda: defaultdict(list))

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.

        Args:
            name: Counter name
            value: Value to increment by
            tags: Optional tags for the metric
        """
        tag_tuple = tuple(sorted((tags or {}).items()))
        self._counters[name][tag_tuple] += value

    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric.

        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags for the metric
        """
        tag_tuple = tuple(sorted((tags or {}).items()))
        self._gauges[name][tag_tuple] = value

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric.

        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags for the metric
        """
        tag_tuple = tuple(sorted((tags or {}).items()))
        self._histograms[name][tag_tuple].append(value)

    @contextmanager
    def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None) -> Generator[None, None, None]:
        """Context manager for timing operations.

        Args:
            name: Operation name
            tags: Optional tags for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_histogram(name, duration_ms, tags)

    def get_counter(self, name: str) -> Dict[Tuple[Tuple[str, str], ...], int]:
        """Get counter values."""
        return dict(self._counters[name])

    def get_gauge(self, name: str) -> Dict[Tuple[Tuple[str, str], ...], float]:
        """Get gauge values."""
        return dict(self._gauges[name])

    def get_histogram(self, name: str) -> Dict[Tuple[Tuple[str, str], ...], List[float]]:
        """Get histogram values."""
        return dict(self._histograms[name])

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a summary."""
        def serialize_metric_dict(metric_dict: Dict[str, Dict[Tuple[Tuple[str, str], ...], Any]]) -> Dict[str, Dict[str, Any]]:
            """Convert tuple keys to string representation for JSON serialization."""
            result: Dict[str, Dict[str, Any]] = {}
            for name, tag_dict in metric_dict.items():
                result[name] = {}
                for tag_tuple, value in tag_dict.items():
                    # Convert tuple of (key, value) pairs to string
                    if tag_tuple:
                        tag_str = ",".join(f"{k}={v}" for k, v in tag_tuple)
                    else:
                        tag_str = "default"
                    result[name][tag_str] = value
            return result

        return {
            "counters": serialize_metric_dict(self._counters),
            "gauges": serialize_metric_dict(self._gauges),
            "histograms": serialize_metric_dict(self._histograms)
        }


class HealthChecker:
    """Health checking functionality."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], Tuple[bool, str]]] = {}
        self._status = "unknown"

        # Register default checks
        self.register_check("docker", self.check_docker_health)
        self.register_check("memory", self.check_memory_health)
        self.register_check("disk", self.check_disk_health)

    def register_check(self, name: str, check_func: Callable[[], Tuple[bool, str]]) -> None:
        """Register a health check.

        Args:
            name: Check name
            check_func: Function that returns (is_healthy, message)
        """
        self._checks[name] = check_func

    def check_health(self) -> Dict[str, Any]:
        """Run all health checks and return status.

        Returns:
            Health status dictionary
        """
        checks = {}
        overall_healthy = True

        for name, check_func in self._checks.items():
            try:
                is_healthy, message = check_func()
                checks[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "message": message
                }
                if not is_healthy:
                    overall_healthy = False
            except Exception as e:
                checks[name] = {
                    "status": "unhealthy",
                    "message": f"Exception during health check: {str(e)}"
                }
                overall_healthy = False

        self._status = "healthy" if overall_healthy else "unhealthy"

        return {
            "status": self._status,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }

    def check_docker_health(self) -> Tuple[bool, str]:
        """Check Docker daemon health."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "Docker daemon is healthy"
            else:
                return False, f"Docker daemon is not available: {result.stderr}"
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            return False, f"Docker daemon is not available: {str(e)}"

    def check_memory_health(self) -> Tuple[bool, str]:
        """Check memory usage health."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            if usage_percent < 80:
                return True, f"Memory usage: {usage_percent:.1f}%"
            elif usage_percent < 90:
                return True, f"Memory usage high: {usage_percent:.1f}%"
            else:
                return False, f"Memory usage critical: {usage_percent:.1f}%"
        except Exception as e:
            return False, f"Failed to check memory: {str(e)}"

    def check_disk_health(self) -> Tuple[bool, str]:
        """Check disk space health."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100

            if usage_percent < 80:
                return True, f"Disk usage: {usage_percent:.1f}%"
            elif usage_percent < 90:
                return True, f"Disk usage high: {usage_percent:.1f}%"
            else:
                return False, f"Disk usage critical: {usage_percent:.1f}%"
        except Exception as e:
            return False, f"Failed to check disk: {str(e)}"


class ObservabilityManager:
    """Comprehensive observability management."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_level: LogLevel = LogLevel.INFO,
        enable_metrics: bool = True,
        enable_health_checks: bool = True,
        service_name: str = "docker-optimizer"
    ) -> None:
        """Initialize observability manager.

        Args:
            log_dir: Directory for log files
            log_level: Logging level
            enable_metrics: Whether to enable metrics collection
            enable_health_checks: Whether to enable health checks
            service_name: Name of the service
        """
        self.service_name = service_name
        self.log_dir = log_dir or Path.cwd() / "logs"

        # Initialize structured logger
        log_file = self.log_dir / f"{service_name}.log"
        self.logger = StructuredLogger(
            name=service_name,
            log_file=log_file,
            level=log_level
        )

        # Initialize metrics collector
        self.metrics = MetricsCollector() if enable_metrics else None

        # Initialize health checker
        self.health_checker = HealthChecker() if enable_health_checks else None

        # Log initialization
        self.logger.info("Observability manager initialized", extra={
            "service_name": service_name,
            "log_level": log_level.value,
            "metrics_enabled": enable_metrics,
            "health_checks_enabled": enable_health_checks
        })

    @contextmanager
    def track_operation(
        self,
        operation_type: str,
        user_id: Optional[str] = None,
        dockerfile_path: Optional[str] = None,
        parent_id: Optional[str] = None
    ) -> Generator[OperationContext, None, None]:
        """Context manager for tracking operations.

        Args:
            operation_type: Type of operation
            user_id: Optional user ID
            dockerfile_path: Optional Dockerfile path
            parent_id: Optional parent operation ID
        """
        context = OperationContext(
            operation_type=operation_type,
            user_id=user_id,
            dockerfile_path=dockerfile_path,
            parent_id=parent_id
        )

        self.logger.info(f"Starting operation: {operation_type}", context=context)

        start_time = time.time()
        error_occurred = False

        try:
            if self.metrics:
                self.metrics.increment_counter("operation_started", tags={"type": operation_type})
            yield context
        except Exception as e:
            error_occurred = True
            self.logger.error(f"Operation failed: {operation_type}", context=context, exception=e)
            if self.metrics:
                self.metrics.increment_counter("operation_errors", tags={"type": operation_type})
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000

            if not error_occurred:
                self.logger.info(
                    f"Operation completed: {operation_type}",
                    context=context,
                    extra={"duration_ms": duration_ms}
                )
                if self.metrics:
                    self.metrics.increment_counter("operation_completed", tags={"type": operation_type})

            if self.metrics:
                self.metrics.record_histogram("operation_duration_ms", duration_ms, tags={"type": operation_type})

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status.

        Returns:
            Health status with version and timestamp
        """
        if not self.health_checker:
            return {
                "status": "unknown",
                "message": "Health checks disabled",
                "timestamp": datetime.now().isoformat()
            }

        health_status = self.health_checker.check_health()
        health_status.update({
            "service": self.service_name,
            "version": "1.0.0"  # Could be dynamically set
        })

        return health_status

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format.

        Args:
            format: Export format ("json" currently supported)

        Returns:
            Serialized metrics
        """
        if not self.metrics:
            return json.dumps({"error": "Metrics collection disabled"})

        metrics_data = self.metrics.get_all_metrics()
        metrics_data.update({
            "service": self.service_name,
            "timestamp": datetime.now().isoformat()
        })

        if format.lower() == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def log_performance_metrics(self) -> None:
        """Log current performance metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            performance_data = {
                "memory_usage_mb": memory.used / (1024 * 1024),
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent
            }

            self.logger.info("Performance metrics", extra=performance_data)

            if self.metrics:
                self.metrics.record_gauge("memory_usage_mb", performance_data["memory_usage_mb"])
                self.metrics.record_gauge("memory_percent", performance_data["memory_percent"])
                self.metrics.record_gauge("cpu_percent", performance_data["cpu_percent"])

        except Exception as e:
            self.logger.error("Failed to collect performance metrics", exception=e)
