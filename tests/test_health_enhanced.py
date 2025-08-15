"""Tests for enhanced health check features."""

import pytest
from unittest.mock import patch, MagicMock

from docker_optimizer.health import HealthCheck, format_prometheus_metrics
from docker_optimizer.config import Config
from docker_optimizer.error_handling import CircuitBreakerState


class TestHealthCheck:
    """Test enhanced health check functionality."""

    def test_health_check_initialization(self):
        """Test health check initialization."""
        config = Config()
        health = HealthCheck(config)
        
        assert health.config == config
        assert health.start_time > 0
        assert isinstance(health._last_check, dict)

    def test_get_health_status_structure(self):
        """Test health status response structure."""
        config = Config()
        health = HealthCheck(config)
        
        status = health.get_health_status()
        
        expected_keys = {"status", "timestamp", "uptime_seconds", "version", "checks"}
        assert set(status.keys()) == expected_keys
        
        # Check nested structure
        checks = status["checks"]
        expected_check_keys = {
            "system", "dependencies", "resources", "external_services", "circuit_breakers"
        }
        assert set(checks.keys()) == expected_check_keys

    def test_readiness_check(self):
        """Test readiness check functionality."""
        config = Config()
        health = HealthCheck(config)
        
        readiness = health.get_readiness_status()
        
        assert "ready" in readiness
        assert "timestamp" in readiness
        assert "checks" in readiness
        assert isinstance(readiness["ready"], bool)

    def test_liveness_check(self):
        """Test liveness check functionality."""
        config = Config()
        health = HealthCheck(config)
        
        liveness = health.get_liveness_status()
        
        assert "alive" in liveness
        assert "timestamp" in liveness
        assert "pid" in liveness
        assert "uptime_seconds" in liveness
        assert liveness["alive"] is True

    @patch('psutil.Process')
    def test_get_metrics(self, mock_process):
        """Test metrics collection."""
        # Mock process metrics
        mock_proc_instance = MagicMock()
        mock_proc_instance.cpu_percent.return_value = 25.0
        mock_proc_instance.memory_info.return_value.rss = 1024 * 1024 * 50  # 50MB
        mock_proc_instance.memory_info.return_value.vms = 1024 * 1024 * 100  # 100MB
        mock_proc_instance.num_threads.return_value = 8
        mock_process.return_value = mock_proc_instance
        
        config = Config()
        health = HealthCheck(config)
        
        metrics = health.get_metrics()
        
        # Check process metrics
        assert "process_cpu_percent" in metrics
        assert "process_memory_rss" in metrics
        assert "process_memory_vms" in metrics
        assert "process_num_threads" in metrics
        
        assert metrics["process_cpu_percent"] == 25.0
        assert metrics["process_memory_rss"] == 1024 * 1024 * 50
        assert metrics["process_num_threads"] == 8

    def test_circuit_breaker_monitoring(self):
        """Test circuit breaker status monitoring."""
        config = Config()
        health = HealthCheck(config)
        
        # Get circuit breaker status
        status = health.get_health_status()
        cb_status = status["checks"]["circuit_breakers"]
        
        assert "status" in cb_status
        assert "breakers" in cb_status
        
        breakers = cb_status["breakers"]
        expected_breakers = {"trivy_scanner", "registry_api", "external_api"}
        assert set(breakers.keys()) == expected_breakers
        
        # Check structure of individual breaker status
        for breaker_name, breaker_info in breakers.items():
            expected_keys = {"state", "failure_count", "success_count", "timeout_seconds", "healthy"}
            assert set(breaker_info.keys()) == expected_keys
            assert isinstance(breaker_info["healthy"], bool)

    @patch('subprocess.run')
    def test_external_services_check(self, mock_subprocess):
        """Test external services health checking."""
        # Mock successful Trivy check
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0.18.3"
        mock_subprocess.return_value = mock_result
        
        config = Config()
        health = HealthCheck(config)
        
        status = health.get_health_status()
        external_services = status["checks"]["external_services"]
        
        assert "status" in external_services
        assert "services" in external_services
        
        services = external_services["services"]
        if "trivy" in services:
            trivy_status = services["trivy"]
            assert "status" in trivy_status
            assert trivy_status["status"] in ["ok", "error", "unavailable"]

    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_percent')
    def test_system_resource_monitoring(self, mock_cpu, mock_disk, mock_memory):
        """Test system resource monitoring."""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value.percent = 60.0
        mock_disk.return_value.percent = 75.0
        
        config = Config()
        health = HealthCheck(config)
        
        status = health.get_health_status()
        resources = status["checks"]["resources"]
        
        assert "status" in resources
        assert resources["status"] in ["ok", "warning", "critical"]

    def test_version_detection(self):
        """Test version detection."""
        config = Config()
        health = HealthCheck(config)
        
        version = health._get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    @patch('docker_optimizer.health.trivy_circuit_breaker')
    def test_circuit_breaker_status_reflection(self, mock_cb):
        """Test that circuit breaker status is properly reflected."""
        # Mock circuit breaker in different states
        mock_cb.state = CircuitBreakerState.OPEN
        mock_cb.failure_count = 5
        mock_cb.success_count = 0
        mock_cb.timeout = 60
        
        config = Config()
        health = HealthCheck(config)
        
        cb_status = health._check_circuit_breakers()
        
        assert cb_status["status"] == "degraded"  # Should be degraded when breaker is open
        trivy_breaker = cb_status["breakers"]["trivy_scanner"]
        assert trivy_breaker["state"] == "open"
        assert trivy_breaker["failure_count"] == 5
        assert trivy_breaker["healthy"] is False


class TestPrometheusMetrics:
    """Test Prometheus metrics formatting."""

    def test_prometheus_metrics_formatting(self):
        """Test Prometheus metrics format."""
        metrics = {
            "docker_optimizer_requests_total": 100,
            "docker_optimizer_response_time": 0.5,
            "process_cpu_percent": 25.0,
            "other_metric": 42
        }
        
        formatted = format_prometheus_metrics(metrics)
        lines = formatted.strip().split('\n')
        
        # Check that docker_optimizer metrics have help and type comments
        help_lines = [line for line in lines if line.startswith('# HELP docker_optimizer_')]
        type_lines = [line for line in lines if line.startswith('# TYPE docker_optimizer_')]
        
        assert len(help_lines) >= 2
        assert len(type_lines) >= 2
        
        # Check metric values
        assert "docker_optimizer_requests_total 100" in formatted
        assert "docker_optimizer_response_time 0.5" in formatted
        assert "process_cpu_percent 25.0" in formatted
        assert "other_metric 42" in formatted

    def test_prometheus_metrics_edge_cases(self):
        """Test Prometheus metrics with edge cases."""
        metrics = {
            "zero_value": 0,
            "negative_value": -1,
            "float_value": 3.14159,
            "string_value": "should_be_ignored"
        }
        
        formatted = format_prometheus_metrics(metrics)
        
        # Only numeric values should be included
        assert "zero_value 0" in formatted
        assert "negative_value -1" in formatted
        assert "float_value 3.14159" in formatted
        assert "string_value" not in formatted

    def test_empty_metrics(self):
        """Test formatting empty metrics."""
        formatted = format_prometheus_metrics({})
        assert formatted == "\n"


class TestHealthRoutes:
    """Test health route creation."""

    def test_create_health_routes(self):
        """Test health route creation."""
        from docker_optimizer.health import create_health_routes
        
        config = Config()
        health = HealthCheck(config)
        
        routes = create_health_routes(health_check=health)
        
        expected_routes = {"/health", "/ready", "/live", "/metrics"}
        assert set(routes.keys()) == expected_routes
        
        # Test that routes return callable functions
        for route_path, route_func in routes.items():
            assert callable(route_func)
            
            # Test calling the route function (except /metrics which returns string)
            if route_path == "/metrics":
                result = route_func()
                assert isinstance(result, str)
            else:
                result = route_func()
                assert isinstance(result, dict)

    def test_health_routes_without_health_check(self):
        """Test route creation with default health check."""
        from docker_optimizer.health import create_health_routes
        
        routes = create_health_routes()
        
        # Should still create routes with default health check
        assert "/health" in routes
        assert "/ready" in routes
        assert "/live" in routes
        assert "/metrics" in routes