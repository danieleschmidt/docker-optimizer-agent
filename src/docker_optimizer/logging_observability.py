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
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

# Optional psutil import for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Metric type enumeration for structured logging."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


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
        self.operation_type = operation_type or "unknown"
        self.start_time = time.time()
        self.metadata: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def add_metric(self, name: str, value: Any, metric_type: MetricType = MetricType.GAUGE) -> None:
        """Add a metric to the operation context.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
        """
        self.metrics[name] = {
            "value": value,
            "type": metric_type.value,
            "timestamp": time.time()
        }

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the operation context.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_duration(self) -> float:
        """Get operation duration in seconds.
        
        Returns:
            Duration since start time
        """
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        context = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "start_time": self.start_time,
            "duration_seconds": self.get_duration()
        }

        if self.user_id:
            context["user_id"] = self.user_id
        if self.dockerfile_path:
            context["dockerfile_path"] = self.dockerfile_path
        if self.parent_id:
            context["parent_id"] = self.parent_id

        if self.metadata:
            context["metadata"] = self.metadata
        if self.metrics:
            context["metrics"] = self.metrics

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

        # Add system metrics if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                log_data["system_metrics"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "num_threads": process.num_threads()
                }
            except Exception:
                # Ignore psutil errors
                pass

        # Add context information if available
        if hasattr(record, 'context') and record.context:
            log_data["context"] = record.context.to_dict()

        # Add extra fields
        if hasattr(record, 'extra_fields') and record.extra_fields:
            log_data["extra"] = record.extra_fields

        # Add exception information with detailed context
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stack_trace": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data, default=str)


class PerformanceMetrics:
    """Performance metrics collector for comprehensive observability."""

    def __init__(self):
        """Initialize performance metrics collector."""
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags for the metric
        """
        self.counters[name] += value
        self.metrics[name].append({
            "type": "counter",
            "value": value,
            "total": self.counters[name],
            "timestamp": time.time(),
            "tags": tags or {}
        })

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric.
        
        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags for the metric
        """
        self.gauges[name] = value
        self.metrics[name].append({
            "type": "gauge",
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        })

    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric.
        
        Args:
            name: Timer name
            duration: Duration in seconds
            tags: Optional tags for the metric
        """
        self.metrics[name].append({
            "type": "timer",
            "duration_seconds": duration,
            "timestamp": time.time(),
            "tags": tags or {}
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary.
        
        Returns:
            Dictionary containing all metrics
        """
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "recent_metrics": {
                name: metrics[-10:]  # Last 10 entries
                for name, metrics in self.metrics.items()
            },
            "timestamp": time.time()
        }

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> Generator[None, None, None]:
        """Context manager for timing operations.
        
        Args:
            name: Timer name
            tags: Optional tags for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(name, duration, tags)


class ErrorTracker:
    """Error tracking system for comprehensive error analysis."""

    def __init__(self, max_errors: int = 1000):
        """Initialize error tracker.
        
        Args:
            max_errors: Maximum number of errors to keep in memory
        """
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_rates: Dict[str, List[float]] = defaultdict(list)

    def track_error(
        self,
        error: Exception,
        context: Optional[OperationContext] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Track an error with context.
        
        Args:
            error: Exception that occurred
            context: Operation context
            tags: Optional tags for categorization
            
        Returns:
            Error ID for reference
        """
        error_id = str(uuid.uuid4())
        error_type = type(error).__name__
        timestamp = time.time()

        error_record = {
            "error_id": error_id,
            "error_type": error_type,
            "message": str(error),
            "timestamp": timestamp,
            "stack_trace": traceback.format_exception(type(error), error, error.__traceback__),
            "tags": tags or {},
            "context": context.to_dict() if context else None
        }

        # Add to error list (maintain max size)
        self.errors.append(error_record)
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)

        # Update error counts
        self.error_counts[error_type] += 1

        # Update error rates (errors per minute)
        current_minute = int(timestamp // 60)
        self.error_rates[error_type].append(current_minute)

        # Clean old rate data (keep last 60 minutes)
        cutoff = current_minute - 60
        self.error_rates[error_type] = [
            minute for minute in self.error_rates[error_type]
            if minute > cutoff
        ]

        return error_id

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error tracking summary.
        
        Returns:
            Dictionary containing error statistics
        """
        current_time = time.time()
        current_minute = int(current_time // 60)

        # Calculate error rates per minute for each error type
        error_rates = {}
        for error_type, minutes in self.error_rates.items():
            recent_minutes = [m for m in minutes if m > current_minute - 5]  # Last 5 minutes
            error_rates[error_type] = len(recent_minutes) / 5.0  # Errors per minute

        return {
            "total_errors": len(self.errors),
            "error_counts_by_type": dict(self.error_counts),
            "error_rates_per_minute": error_rates,
            "recent_errors": self.errors[-10:],  # Last 10 errors
            "timestamp": current_time
        }

    def get_error_by_id(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Get error details by ID.
        
        Args:
            error_id: Error ID to lookup
            
        Returns:
            Error record or None if not found
        """
        for error in self.errors:
            if error["error_id"] == error_id:
                return error
        return None


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

    def configure_logging(self, config: Dict[str, Any]) -> None:
        """Configure logging with advanced options.
        
        Args:
            config: Logging configuration dictionary with options:
                - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - enable_console: Enable console logging (bool)
                - log_format: Log format (json, text)
                - max_file_size_mb: Maximum log file size in MB
                - backup_count: Number of backup files to keep
                - enable_metrics: Enable metrics collection (bool)
                - metrics_interval: Metrics collection interval in seconds
        """
        # Update log level
        if 'log_level' in config:
            try:
                new_level = LogLevel(config['log_level'].upper())
                self.logger.level = new_level
                self.logger._logger.setLevel(getattr(logging, new_level.value))
                self.logger.info(f"Log level updated to {new_level.value}")
            except ValueError:
                self.logger.warning(f"Invalid log level: {config['log_level']}")

        # Configure file rotation
        if 'max_file_size_mb' in config or 'backup_count' in config:
            max_size = config.get('max_file_size_mb', 50) * 1024 * 1024
            backup_count = config.get('backup_count', 5)

            # Update existing file handlers
            for handler in self.logger._logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.maxBytes = max_size
                    handler.backupCount = backup_count

        # Log configuration update
        self.logger.info("Logging configuration updated", extra={
            "config": config,
            "applied_at": datetime.now().isoformat()
        })

    def get_logging_config(self) -> Dict[str, Any]:
        """Get current logging configuration.
        
        Returns:
            Dictionary containing current logging settings
        """
        return {
            "service_name": self.service_name,
            "log_level": self.logger.level.value,
            "log_directory": str(self.log_dir),
            "metrics_enabled": self.metrics is not None,
            "health_checks_enabled": self.health_checker is not None,
            "handler_count": len(self.logger._logger.handlers),
            "psutil_available": PSUTIL_AVAILABLE
        }
