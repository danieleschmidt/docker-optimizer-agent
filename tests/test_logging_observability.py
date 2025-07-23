"""Tests for logging and observability functionality."""

import json
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from docker_optimizer.logging_observability import (
    ObservabilityManager,
    StructuredLogger,
    MetricsCollector,
    HealthChecker,
    LogLevel,
    OperationContext,
)


class TestStructuredLogger:
    """Test structured logging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
        self.logger = StructuredLogger(
            name="test_logger",
            log_file=self.log_file,
            level=LogLevel.INFO
        )

    def test_structured_log_creation(self):
        """Test creation of structured logger."""
        assert self.logger.name == "test_logger"
        assert self.logger.level == LogLevel.INFO
        assert hasattr(self.logger, '_logger')

    def test_info_logging_with_context(self):
        """Test info level logging with context."""
        context = OperationContext(
            operation_id="op_123",
            user_id="user_456",
            dockerfile_path="/path/to/Dockerfile"
        )
        
        self.logger.info("Test message", context=context, extra={"key": "value"})
        
        # Verify log was written
        assert self.log_file.exists()
        
        # Verify log content is JSON structured
        with open(self.log_file) as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["operation_id"] == "op_123"
        assert log_data["user_id"] == "user_456"
        assert log_data["dockerfile_path"] == "/path/to/Dockerfile"
        assert log_data["key"] == "value"
        assert "timestamp" in log_data

    def test_error_logging_with_exception(self):
        """Test error logging with exception details."""
        context = OperationContext(operation_id="op_789")
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            self.logger.error("Error occurred", context=context, exception=e)
        
        with open(self.log_file) as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Error occurred"
        assert log_data["operation_id"] == "op_789"
        assert "exception_type" in log_data
        assert "exception_message" in log_data
        assert "stack_trace" in log_data

    def test_warning_logging(self):
        """Test warning level logging."""
        context = OperationContext(operation_id="warn_123")
        
        self.logger.warning("Warning message", context=context)
        
        with open(self.log_file) as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
        assert log_data["level"] == "WARNING"
        assert log_data["message"] == "Warning message"
        assert log_data["operation_id"] == "warn_123"

    def test_debug_logging_filtered_by_level(self):
        """Test that debug messages are filtered when level is INFO."""
        context = OperationContext(operation_id="debug_123")
        
        self.logger.debug("Debug message", context=context)
        
        # Debug message should not be written when level is INFO
        if self.log_file.exists():
            with open(self.log_file) as f:
                content = f.read()
                assert content.strip() == ""
        else:
            # File might not even be created
            assert True


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()

    def test_metrics_collector_creation(self):
        """Test creation of metrics collector."""
        assert hasattr(self.collector, '_metrics')
        assert hasattr(self.collector, '_counters')
        assert hasattr(self.collector, '_histograms')

    def test_increment_counter(self):
        """Test incrementing a counter metric."""
        self.collector.increment_counter("dockerfiles_processed", tags={"status": "success"})
        self.collector.increment_counter("dockerfiles_processed", tags={"status": "success"})
        self.collector.increment_counter("dockerfiles_processed", tags={"status": "error"})
        
        metrics = self.collector.get_counter("dockerfiles_processed")
        assert metrics[(("status", "success"),)] == 2
        assert metrics[(("status", "error"),)] == 1

    def test_record_gauge(self):
        """Test recording a gauge metric."""
        self.collector.record_gauge("memory_usage_mb", 256.5, tags={"component": "optimizer"})
        
        metrics = self.collector.get_gauge("memory_usage_mb")
        assert metrics[(("component", "optimizer"),)] == 256.5

    def test_record_histogram(self):
        """Test recording histogram metrics."""
        values = [100, 200, 150, 300, 250]
        for value in values:
            self.collector.record_histogram("processing_time_ms", value, tags={"operation": "optimize"})
        
        histogram = self.collector.get_histogram("processing_time_ms")
        operation_data = histogram[(("operation", "optimize"),)]
        
        assert len(operation_data) == 5
        assert min(operation_data) == 100
        assert max(operation_data) == 300
        assert sum(operation_data) / len(operation_data) == 200

    def test_record_timing(self):
        """Test timing context manager."""
        with self.collector.time_operation("test_operation", tags={"type": "unit_test"}):
            time.sleep(0.01)  # Small sleep to ensure measurable time
        
        histogram = self.collector.get_histogram("test_operation")
        timing_data = histogram[(("type", "unit_test"),)]
        
        assert len(timing_data) == 1
        assert timing_data[0] > 0  # Should have recorded some timing

    def test_get_all_metrics(self):
        """Test getting all metrics as a summary."""
        self.collector.increment_counter("test_counter")
        self.collector.record_gauge("test_gauge", 42.0)
        self.collector.record_histogram("test_histogram", 100)
        
        all_metrics = self.collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "test_counter" in all_metrics["counters"]
        assert "test_gauge" in all_metrics["gauges"]
        assert "test_histogram" in all_metrics["histograms"]


class TestHealthChecker:
    """Test health checking functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.health_checker = HealthChecker()

    def test_health_checker_creation(self):
        """Test creation of health checker."""
        assert hasattr(self.health_checker, '_checks')
        assert hasattr(self.health_checker, '_status')

    def test_register_health_check(self):
        """Test registering a health check."""
        def dummy_check():
            return True, "OK"
        
        self.health_checker.register_check("dummy", dummy_check)
        
        assert "dummy" in self.health_checker._checks

    def test_successful_health_check(self):
        """Test successful health check execution."""
        def healthy_check():
            return True, "Service is healthy"
        
        self.health_checker.register_check("healthy_service", healthy_check)
        
        result = self.health_checker.check_health()
        
        assert result["status"] == "healthy"
        assert "healthy_service" in result["checks"]
        assert result["checks"]["healthy_service"]["status"] == "healthy"
        assert result["checks"]["healthy_service"]["message"] == "Service is healthy"

    def test_failed_health_check(self):
        """Test failed health check execution."""
        def unhealthy_check():
            return False, "Service is down"
        
        self.health_checker.register_check("unhealthy_service", unhealthy_check)
        
        result = self.health_checker.check_health()
        
        assert result["status"] == "unhealthy"
        assert "unhealthy_service" in result["checks"]
        assert result["checks"]["unhealthy_service"]["status"] == "unhealthy"
        assert result["checks"]["unhealthy_service"]["message"] == "Service is down"

    def test_mixed_health_checks(self):
        """Test mixed healthy and unhealthy checks."""
        def healthy_check():
            return True, "OK"
        
        def unhealthy_check():
            return False, "Error"
        
        self.health_checker.register_check("healthy", healthy_check)
        self.health_checker.register_check("unhealthy", unhealthy_check)
        
        result = self.health_checker.check_health()
        
        assert result["status"] == "unhealthy"  # Overall status should be unhealthy
        assert result["checks"]["healthy"]["status"] == "healthy"
        assert result["checks"]["unhealthy"]["status"] == "unhealthy"

    def test_health_check_with_exception(self):
        """Test health check that raises an exception."""
        def failing_check():
            raise RuntimeError("Check failed")
        
        self.health_checker.register_check("failing", failing_check)
        
        result = self.health_checker.check_health()
        
        assert result["status"] == "unhealthy"
        assert "failing" in result["checks"]
        assert result["checks"]["failing"]["status"] == "unhealthy"
        assert "Exception" in result["checks"]["failing"]["message"]

    @patch('docker_optimizer.logging_observability.subprocess.run')
    def test_docker_health_check(self, mock_run):
        """Test Docker daemon health check."""
        # Mock successful Docker command
        mock_run.return_value = Mock(returncode=0, stdout="Docker is running")
        
        result = self.health_checker.check_docker_health()
        
        assert result[0] is True
        assert "Docker daemon is healthy" in result[1]

    @patch('docker_optimizer.logging_observability.subprocess.run')
    def test_docker_health_check_failure(self, mock_run):
        """Test Docker daemon health check failure."""
        # Mock failed Docker command
        mock_run.return_value = Mock(returncode=1, stderr="Docker not found")
        
        result = self.health_checker.check_docker_health()
        
        assert result[0] is False
        assert "Docker daemon is not available" in result[1]


class TestObservabilityManager:
    """Test comprehensive observability management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)
        self.manager = ObservabilityManager(
            log_dir=self.log_dir,
            log_level=LogLevel.INFO,
            enable_metrics=True,
            enable_health_checks=True
        )

    def test_observability_manager_creation(self):
        """Test creation of observability manager."""
        assert hasattr(self.manager, 'logger')
        assert hasattr(self.manager, 'metrics')
        assert hasattr(self.manager, 'health_checker')

    def test_operation_context_manager(self):
        """Test operation context manager for automatic tracking."""
        with self.manager.track_operation("optimize_dockerfile", user_id="test_user") as context:
            assert context.operation_id is not None
            assert context.user_id == "test_user"
            
            # Simulate some work
            time.sleep(0.01)
            
            # Log something during the operation
            self.manager.logger.info("Processing dockerfile", context=context)
        
        # Verify metrics were collected
        metrics = self.manager.metrics.get_all_metrics()
        assert "operation_duration_ms" in metrics["histograms"]

    def test_error_tracking(self):
        """Test automatic error tracking."""
        try:
            with self.manager.track_operation("failing_operation") as context:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify error was tracked
        metrics = self.manager.metrics.get_all_metrics()
        assert "operation_errors" in metrics["counters"]

    def test_comprehensive_health_check(self):
        """Test comprehensive health check including Docker."""
        result = self.manager.get_health_status()
        
        assert "status" in result
        assert "checks" in result
        assert "timestamp" in result
        assert "version" in result

    def test_metrics_export(self):
        """Test exporting metrics in different formats."""
        # Record some test metrics
        self.manager.metrics.increment_counter("test_counter")
        self.manager.metrics.record_gauge("test_gauge", 42.0)
        
        # Export as JSON
        json_metrics = self.manager.export_metrics(format="json")
        metrics_data = json.loads(json_metrics)
        
        assert "counters" in metrics_data
        assert "gauges" in metrics_data
        assert "test_counter" in metrics_data["counters"]
        assert "test_gauge" in metrics_data["gauges"]

    def test_log_correlation(self):
        """Test log correlation across operations."""
        # Start first operation
        with self.manager.track_operation("operation_1") as context1:
            self.manager.logger.info("Starting operation 1", context=context1)
            
            # Start nested operation
            with self.manager.track_operation("operation_2", parent_id=context1.operation_id) as context2:
                self.manager.logger.info("Starting operation 2", context=context2)
                assert context2.parent_id == context1.operation_id

    def test_performance_tracking(self):
        """Test automatic performance tracking."""
        with self.manager.track_operation("performance_test") as context:
            # Simulate work
            time.sleep(0.01)
            self.manager.logger.info("Work completed", context=context)
        
        # Check that performance metrics were recorded
        histogram = self.manager.metrics.get_histogram("operation_duration_ms")
        assert len(histogram) > 0