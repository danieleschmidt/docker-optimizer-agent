"""Production Readiness and Deployment Assessment System."""

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pkg_resources

from .health_monitor import get_health_monitor
from .monitoring_integration import get_monitoring_integration
from .auto_scaling import get_autoscaler
from .intelligent_caching import get_cache_manager

logger = logging.getLogger(__name__)


class ReadinessLevel(str, Enum):
    """Production readiness levels."""
    DEVELOPMENT = "development"
    STAGING = "staging"  
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class CheckStatus(str, Enum):
    """Status of readiness checks."""
    PASS = "pass"
    WARN = "warning" 
    FAIL = "critical"
    SKIP = "skipped"


@dataclass
class ReadinessCheck:
    """Individual readiness check result."""
    name: str
    category: str
    status: CheckStatus
    message: str
    recommendation: str = ""
    impact: str = "medium"
    fix_command: Optional[str] = None
    documentation_url: Optional[str] = None


@dataclass
class ReadinessReport:
    """Complete production readiness assessment."""
    overall_status: CheckStatus
    readiness_level: ReadinessLevel
    score: float  # 0-100
    checks: List[ReadinessCheck] = field(default_factory=list)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    deployment_recommendations: List[str] = field(default_factory=list)
    security_recommendations: List[str] = field(default_factory=list)
    performance_recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ProductionReadinessAssessment:
    """Comprehensive production readiness assessment system."""
    
    def __init__(self):
        self.checks = []
        self.health_monitor = get_health_monitor()
        self.monitoring = get_monitoring_integration()
        self.autoscaler = get_autoscaler()
        self.cache_manager = get_cache_manager()
        
        # Readiness criteria by level
        self.level_requirements = {
            ReadinessLevel.DEVELOPMENT: {"min_score": 60, "critical_max": 2},
            ReadinessLevel.STAGING: {"min_score": 75, "critical_max": 1},
            ReadinessLevel.PRODUCTION: {"min_score": 85, "critical_max": 0},
            ReadinessLevel.ENTERPRISE: {"min_score": 95, "critical_max": 0}
        }
    
    def assess_readiness(self, target_level: ReadinessLevel = ReadinessLevel.PRODUCTION) -> ReadinessReport:
        """Perform comprehensive production readiness assessment."""
        logger.info(f"Starting production readiness assessment for {target_level.value} level")
        
        self.checks = []
        
        # Core infrastructure checks
        self._check_system_requirements()
        self._check_dependencies()
        self._check_configuration()
        
        # Security checks
        self._check_security_configuration()
        self._check_secrets_management()
        self._check_network_security()
        
        # Performance checks
        self._check_performance_configuration()
        self._check_caching_readiness()
        self._check_scaling_readiness()
        
        # Monitoring and observability
        self._check_monitoring_setup()
        self._check_health_checks()
        self._check_logging_configuration()
        
        # Deployment checks
        self._check_deployment_artifacts()
        self._check_database_readiness()
        self._check_external_dependencies()
        
        # Generate report
        return self._generate_report(target_level)
    
    def _check_system_requirements(self) -> None:
        """Check system requirements and compatibility."""
        # Python version check
        python_version = sys.version_info
        if python_version >= (3, 9):
            self.checks.append(ReadinessCheck(
                name="Python Version",
                category="system",
                status=CheckStatus.PASS,
                message=f"Python {python_version.major}.{python_version.minor} supported",
                impact="high"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Python Version",
                category="system",
                status=CheckStatus.FAIL,
                message=f"Python {python_version.major}.{python_version.minor} not supported",
                recommendation="Upgrade to Python 3.9+",
                impact="critical",
                fix_command="# Update Python to 3.9 or later"
            ))
        
        # Platform check
        current_platform = platform.system()
        supported_platforms = ["Linux", "Darwin", "Windows"]
        if current_platform in supported_platforms:
            self.checks.append(ReadinessCheck(
                name="Platform Compatibility",
                category="system",
                status=CheckStatus.PASS,
                message=f"{current_platform} platform supported",
                impact="high"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Platform Compatibility",
                category="system",
                status=CheckStatus.WARN,
                message=f"{current_platform} platform not fully tested",
                recommendation="Test thoroughly on target platform",
                impact="medium"
            ))
        
        # Memory check
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            if total_memory_gb >= 4:
                self.checks.append(ReadinessCheck(
                    name="System Memory",
                    category="system", 
                    status=CheckStatus.PASS,
                    message=f"{total_memory_gb:.1f}GB memory available",
                    impact="medium"
                ))
            else:
                self.checks.append(ReadinessCheck(
                    name="System Memory",
                    category="system",
                    status=CheckStatus.WARN,
                    message=f"Only {total_memory_gb:.1f}GB memory available",
                    recommendation="Consider increasing memory for production workloads",
                    impact="medium"
                ))
        except ImportError:
            self.checks.append(ReadinessCheck(
                name="System Memory",
                category="system",
                status=CheckStatus.SKIP,
                message="psutil not available for memory check",
                impact="low"
            ))
    
    def _check_dependencies(self) -> None:
        """Check required dependencies and versions."""
        required_packages = [
            ("click", "8.0.0"),
            ("pydantic", "2.0.0"),
            ("requests", "2.28.0"),
            ("pyyaml", "6.0"),
            ("rich", "13.0.0")
        ]
        
        for package_name, min_version in required_packages:
            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                if pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(min_version):
                    self.checks.append(ReadinessCheck(
                        name=f"Dependency: {package_name}",
                        category="dependencies",
                        status=CheckStatus.PASS,
                        message=f"Version {installed_version} meets requirement {min_version}+",
                        impact="high"
                    ))
                else:
                    self.checks.append(ReadinessCheck(
                        name=f"Dependency: {package_name}",
                        category="dependencies",
                        status=CheckStatus.FAIL,
                        message=f"Version {installed_version} below requirement {min_version}",
                        recommendation=f"Upgrade {package_name} to {min_version}+",
                        impact="critical",
                        fix_command=f"pip install '{package_name}>={min_version}'"
                    ))
            except pkg_resources.DistributionNotFound:
                self.checks.append(ReadinessCheck(
                    name=f"Dependency: {package_name}",
                    category="dependencies",
                    status=CheckStatus.FAIL,
                    message=f"Package {package_name} not found",
                    recommendation=f"Install {package_name}",
                    impact="critical",
                    fix_command=f"pip install '{package_name}>={min_version}'"
                ))
        
        # Optional dependencies check
        optional_packages = ["docker", "numpy", "matplotlib", "scikit-learn"]
        for package_name in optional_packages:
            try:
                version = pkg_resources.get_distribution(package_name).version
                self.checks.append(ReadinessCheck(
                    name=f"Optional: {package_name}",
                    category="dependencies",
                    status=CheckStatus.PASS,
                    message=f"Version {version} available (optional)",
                    impact="low"
                ))
            except pkg_resources.DistributionNotFound:
                self.checks.append(ReadinessCheck(
                    name=f"Optional: {package_name}",
                    category="dependencies",
                    status=CheckStatus.WARN,
                    message=f"Optional package {package_name} not available",
                    recommendation=f"Install {package_name} for enhanced functionality",
                    impact="low",
                    fix_command=f"pip install {package_name}"
                ))
    
    def _check_configuration(self) -> None:
        """Check configuration and environment setup."""
        # Environment variables check
        important_env_vars = [
            ("DOCKER_OPTIMIZER_LOG_LEVEL", "INFO", "Logging configuration"),
            ("DOCKER_OPTIMIZER_CACHE_SIZE", "100", "Cache size configuration"),
            ("DOCKER_OPTIMIZER_MAX_WORKERS", "4", "Worker pool configuration")
        ]
        
        for env_var, default_value, description in important_env_vars:
            if env_var in os.environ:
                self.checks.append(ReadinessCheck(
                    name=f"Environment: {env_var}",
                    category="configuration",
                    status=CheckStatus.PASS,
                    message=f"Set to: {os.environ[env_var]}",
                    impact="low"
                ))
            else:
                self.checks.append(ReadinessCheck(
                    name=f"Environment: {env_var}",
                    category="configuration",
                    status=CheckStatus.WARN,
                    message=f"Not set, using default: {default_value}",
                    recommendation=f"Set {env_var} for {description}",
                    impact="low",
                    fix_command=f"export {env_var}={default_value}"
                ))
        
        # Configuration files check
        config_files = [
            Path("config/production.yml"),
            Path("config/logging.yml"),
            Path("docker-compose.yml")
        ]
        
        for config_file in config_files:
            if config_file.exists():
                self.checks.append(ReadinessCheck(
                    name=f"Config File: {config_file.name}",
                    category="configuration",
                    status=CheckStatus.PASS,
                    message=f"Configuration file present: {config_file}",
                    impact="medium"
                ))
            else:
                self.checks.append(ReadinessCheck(
                    name=f"Config File: {config_file.name}",
                    category="configuration",
                    status=CheckStatus.WARN,
                    message=f"Configuration file missing: {config_file}",
                    recommendation=f"Create {config_file} for production deployment",
                    impact="medium"
                ))
    
    def _check_security_configuration(self) -> None:
        """Check security configuration and settings."""
        # Debug mode check
        debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
        if not debug_mode:
            self.checks.append(ReadinessCheck(
                name="Debug Mode",
                category="security",
                status=CheckStatus.PASS,
                message="Debug mode disabled",
                impact="high"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Debug Mode",
                category="security",
                status=CheckStatus.FAIL,
                message="Debug mode enabled in production",
                recommendation="Disable debug mode for production",
                impact="critical",
                fix_command="export DEBUG=false"
            ))
        
        # SSL/TLS check
        ssl_enabled = os.environ.get("SSL_ENABLED", "false").lower() == "true"
        if ssl_enabled:
            self.checks.append(ReadinessCheck(
                name="SSL/TLS",
                category="security",
                status=CheckStatus.PASS,
                message="SSL/TLS enabled",
                impact="high"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="SSL/TLS",
                category="security",
                status=CheckStatus.WARN,
                message="SSL/TLS not explicitly enabled",
                recommendation="Enable SSL/TLS for production traffic",
                impact="high",
                fix_command="export SSL_ENABLED=true"
            ))
    
    def _check_secrets_management(self) -> None:
        """Check secrets management configuration."""
        # Check for hardcoded secrets in environment
        sensitive_patterns = ["password", "secret", "key", "token", "credential"]
        env_vars = dict(os.environ)
        
        for var_name, value in env_vars.items():
            if any(pattern in var_name.lower() for pattern in sensitive_patterns):
                if len(value) < 20:  # Likely a placeholder or weak secret
                    self.checks.append(ReadinessCheck(
                        name=f"Secret: {var_name}",
                        category="security",
                        status=CheckStatus.WARN,
                        message="Short secret detected - may be placeholder",
                        recommendation="Use strong, randomly generated secrets",
                        impact="high"
                    ))
                else:
                    self.checks.append(ReadinessCheck(
                        name=f"Secret: {var_name}",
                        category="security",
                        status=CheckStatus.PASS,
                        message="Secret appears properly configured",
                        impact="high"
                    ))
        
        # Secrets management system check
        secrets_provider = os.environ.get("SECRETS_PROVIDER", "environment")
        if secrets_provider in ["vault", "k8s-secrets", "aws-secrets", "azure-keyvault"]:
            self.checks.append(ReadinessCheck(
                name="Secrets Provider",
                category="security", 
                status=CheckStatus.PASS,
                message=f"Using external secrets provider: {secrets_provider}",
                impact="high"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Secrets Provider",
                category="security",
                status=CheckStatus.WARN,
                message="Using environment variables for secrets",
                recommendation="Consider external secrets management for production",
                impact="medium"
            ))
    
    def _check_network_security(self) -> None:
        """Check network security configuration."""
        # Firewall check (basic)
        self.checks.append(ReadinessCheck(
            name="Network Security",
            category="security",
            status=CheckStatus.WARN,
            message="Network security configuration should be verified",
            recommendation="Ensure firewall rules and network policies are configured",
            impact="high"
        ))
    
    def _check_performance_configuration(self) -> None:
        """Check performance-related configuration."""
        # Worker pool configuration
        max_workers = int(os.environ.get("DOCKER_OPTIMIZER_MAX_WORKERS", "4"))
        if max_workers >= 2:
            self.checks.append(ReadinessCheck(
                name="Worker Pool Size",
                category="performance",
                status=CheckStatus.PASS,
                message=f"Worker pool configured with {max_workers} workers",
                impact="medium"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Worker Pool Size",
                category="performance",
                status=CheckStatus.WARN,
                message=f"Only {max_workers} worker(s) configured",
                recommendation="Increase worker pool size for better performance",
                impact="medium",
                fix_command="export DOCKER_OPTIMIZER_MAX_WORKERS=4"
            ))
        
        # Memory limits check
        memory_limit = os.environ.get("MEMORY_LIMIT", None)
        if memory_limit:
            self.checks.append(ReadinessCheck(
                name="Memory Limits",
                category="performance",
                status=CheckStatus.PASS,
                message=f"Memory limit configured: {memory_limit}",
                impact="medium"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Memory Limits",
                category="performance",
                status=CheckStatus.WARN,
                message="No memory limits configured",
                recommendation="Set memory limits for production deployment",
                impact="medium"
            ))
    
    def _check_caching_readiness(self) -> None:
        """Check caching system readiness."""
        try:
            cache_stats = self.cache_manager.get_overall_stats()
            if cache_stats["total_entries"] == 0:
                # Cache is available but empty
                self.checks.append(ReadinessCheck(
                    name="Caching System",
                    category="performance",
                    status=CheckStatus.PASS,
                    message="Caching system initialized and ready",
                    impact="medium"
                ))
            else:
                # Cache has data
                hit_rate = cache_stats.get("overall_hit_rate", 0)
                self.checks.append(ReadinessCheck(
                    name="Caching System",
                    category="performance",
                    status=CheckStatus.PASS,
                    message=f"Caching active with {hit_rate:.1f}% hit rate",
                    impact="medium"
                ))
        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Caching System",
                category="performance",
                status=CheckStatus.FAIL,
                message=f"Caching system error: {str(e)}",
                recommendation="Fix caching system configuration",
                impact="medium"
            ))
    
    def _check_scaling_readiness(self) -> None:
        """Check auto-scaling readiness."""
        try:
            scaling_status = self.autoscaler.get_scaling_status()
            if scaling_status["enabled"]:
                pool_count = len(scaling_status["pools"])
                self.checks.append(ReadinessCheck(
                    name="Auto-scaling",
                    category="performance",
                    status=CheckStatus.PASS,
                    message=f"Auto-scaling enabled with {pool_count} pools",
                    impact="high"
                ))
            else:
                self.checks.append(ReadinessCheck(
                    name="Auto-scaling",
                    category="performance",
                    status=CheckStatus.WARN,
                    message="Auto-scaling not enabled",
                    recommendation="Enable auto-scaling for production workloads",
                    impact="medium"
                ))
        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Auto-scaling",
                category="performance",
                status=CheckStatus.FAIL,
                message=f"Auto-scaling error: {str(e)}",
                recommendation="Fix auto-scaling configuration",
                impact="medium"
            ))
    
    def _check_monitoring_setup(self) -> None:
        """Check monitoring and observability setup."""
        try:
            metrics = self.monitoring.get_performance_metrics()
            self.checks.append(ReadinessCheck(
                name="Monitoring System",
                category="monitoring",
                status=CheckStatus.PASS,
                message="Monitoring system active and collecting metrics",
                impact="high"
            ))
        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Monitoring System",
                category="monitoring",
                status=CheckStatus.FAIL,
                message=f"Monitoring system error: {str(e)}",
                recommendation="Fix monitoring system configuration",
                impact="high"
            ))
        
        # Prometheus metrics check
        try:
            prometheus_metrics = self.monitoring.export_prometheus_metrics()
            if len(prometheus_metrics) > 100:  # Has meaningful metrics
                self.checks.append(ReadinessCheck(
                    name="Prometheus Metrics",
                    category="monitoring",
                    status=CheckStatus.PASS,
                    message="Prometheus metrics endpoint ready",
                    impact="medium"
                ))
            else:
                self.checks.append(ReadinessCheck(
                    name="Prometheus Metrics",
                    category="monitoring",
                    status=CheckStatus.WARN,
                    message="Limited Prometheus metrics available",
                    recommendation="Verify metrics collection configuration",
                    impact="medium"
                ))
        except Exception:
            self.checks.append(ReadinessCheck(
                name="Prometheus Metrics",
                category="monitoring",
                status=CheckStatus.FAIL,
                message="Prometheus metrics not available",
                recommendation="Configure Prometheus metrics endpoint",
                impact="medium"
            ))
    
    def _check_health_checks(self) -> None:
        """Check health monitoring configuration."""
        try:
            health_report = self.health_monitor.get_health_report()
            if health_report.get("overall_status") == "healthy":
                self.checks.append(ReadinessCheck(
                    name="Health Monitoring",
                    category="monitoring",
                    status=CheckStatus.PASS,
                    message="Health monitoring active and system healthy",
                    impact="high"
                ))
            else:
                status = health_report.get("overall_status", "unknown")
                self.checks.append(ReadinessCheck(
                    name="Health Monitoring",
                    category="monitoring",
                    status=CheckStatus.WARN,
                    message=f"Health monitoring reports: {status}",
                    recommendation="Investigate and resolve health issues",
                    impact="high"
                ))
        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Health Monitoring",
                category="monitoring",
                status=CheckStatus.FAIL,
                message=f"Health monitoring error: {str(e)}",
                recommendation="Fix health monitoring configuration",
                impact="high"
            ))
    
    def _check_logging_configuration(self) -> None:
        """Check logging configuration."""
        log_level = os.environ.get("DOCKER_OPTIMIZER_LOG_LEVEL", "INFO")
        if log_level.upper() in ["INFO", "WARNING", "ERROR"]:
            self.checks.append(ReadinessCheck(
                name="Logging Level",
                category="monitoring",
                status=CheckStatus.PASS,
                message=f"Logging level set to {log_level}",
                impact="medium"
            ))
        elif log_level.upper() == "DEBUG":
            self.checks.append(ReadinessCheck(
                name="Logging Level",
                category="monitoring",
                status=CheckStatus.WARN,
                message="Debug logging enabled - high verbosity for production",
                recommendation="Set log level to INFO or WARNING for production",
                impact="medium",
                fix_command="export DOCKER_OPTIMIZER_LOG_LEVEL=INFO"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Logging Level",
                category="monitoring",
                status=CheckStatus.WARN,
                message=f"Unknown log level: {log_level}",
                recommendation="Set valid log level (INFO, WARNING, ERROR)",
                impact="medium"
            ))
        
        # Log aggregation check
        log_aggregation = os.environ.get("LOG_AGGREGATION", None)
        if log_aggregation:
            self.checks.append(ReadinessCheck(
                name="Log Aggregation",
                category="monitoring",
                status=CheckStatus.PASS,
                message=f"Log aggregation configured: {log_aggregation}",
                impact="medium"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Log Aggregation",
                category="monitoring",
                status=CheckStatus.WARN,
                message="No log aggregation configured",
                recommendation="Configure log aggregation for production",
                impact="medium"
            ))
    
    def _check_deployment_artifacts(self) -> None:
        """Check deployment artifacts and files."""
        required_files = [
            ("Dockerfile", "Container deployment"),
            ("docker-compose.yml", "Container orchestration"),
            ("requirements.txt", "Python dependencies"),
            ("pyproject.toml", "Package configuration")
        ]
        
        for filename, description in required_files:
            if Path(filename).exists():
                self.checks.append(ReadinessCheck(
                    name=f"Deployment File: {filename}",
                    category="deployment",
                    status=CheckStatus.PASS,
                    message=f"{description} file present",
                    impact="high"
                ))
            else:
                impact = "critical" if filename in ["Dockerfile", "requirements.txt"] else "medium"
                self.checks.append(ReadinessCheck(
                    name=f"Deployment File: {filename}",
                    category="deployment",
                    status=CheckStatus.FAIL if impact == "critical" else CheckStatus.WARN,
                    message=f"{description} file missing",
                    recommendation=f"Create {filename} for deployment",
                    impact=impact
                ))
    
    def _check_database_readiness(self) -> None:
        """Check database and persistence readiness."""
        # For this application, we don't use a traditional database
        # but we can check for data persistence configuration
        
        data_dir = os.environ.get("DATA_DIRECTORY", "./data")
        if Path(data_dir).exists():
            self.checks.append(ReadinessCheck(
                name="Data Directory",
                category="deployment",
                status=CheckStatus.PASS,
                message=f"Data directory exists: {data_dir}",
                impact="medium"
            ))
        else:
            self.checks.append(ReadinessCheck(
                name="Data Directory",
                category="deployment",
                status=CheckStatus.WARN,
                message=f"Data directory missing: {data_dir}",
                recommendation="Create data directory for persistent storage",
                impact="medium",
                fix_command=f"mkdir -p {data_dir}"
            ))
    
    def _check_external_dependencies(self) -> None:
        """Check external service dependencies."""
        # Docker daemon check
        try:
            result = subprocess.run(["docker", "version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.checks.append(ReadinessCheck(
                    name="Docker Daemon",
                    category="dependencies",
                    status=CheckStatus.PASS,
                    message="Docker daemon accessible",
                    impact="high"
                ))
            else:
                self.checks.append(ReadinessCheck(
                    name="Docker Daemon",
                    category="dependencies",
                    status=CheckStatus.FAIL,
                    message="Docker daemon not accessible",
                    recommendation="Start Docker daemon",
                    impact="critical"
                ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.checks.append(ReadinessCheck(
                name="Docker Daemon",
                category="dependencies",
                status=CheckStatus.WARN,
                message="Docker daemon check failed",
                recommendation="Verify Docker installation and daemon status",
                impact="high"
            ))
        
        # Network connectivity check
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.checks.append(ReadinessCheck(
                name="Network Connectivity",
                category="dependencies",
                status=CheckStatus.PASS,
                message="External network connectivity verified",
                impact="medium"
            ))
        except Exception:
            self.checks.append(ReadinessCheck(
                name="Network Connectivity",
                category="dependencies",
                status=CheckStatus.WARN,
                message="External network connectivity issues",
                recommendation="Verify network configuration and firewall rules",
                impact="medium"
            ))
    
    def _generate_report(self, target_level: ReadinessLevel) -> ReadinessReport:
        """Generate comprehensive readiness report."""
        # Calculate scores and status
        total_checks = len(self.checks)
        pass_count = len([c for c in self.checks if c.status == CheckStatus.PASS])
        warn_count = len([c for c in self.checks if c.status == CheckStatus.WARN])
        fail_count = len([c for c in self.checks if c.status == CheckStatus.FAIL])
        
        # Calculate score (0-100)
        if total_checks == 0:
            score = 0.0
        else:
            score = (pass_count + (warn_count * 0.5)) / total_checks * 100
        
        # Determine overall status
        requirements = self.level_requirements[target_level]
        if fail_count > requirements["critical_max"]:
            overall_status = CheckStatus.FAIL
        elif score < requirements["min_score"]:
            overall_status = CheckStatus.WARN
        else:
            overall_status = CheckStatus.PASS
        
        # Generate recommendations
        deployment_recs = []
        security_recs = []
        performance_recs = []
        
        for check in self.checks:
            if check.status in [CheckStatus.FAIL, CheckStatus.WARN] and check.recommendation:
                if check.category == "security":
                    security_recs.append(check.recommendation)
                elif check.category in ["performance", "monitoring"]:
                    performance_recs.append(check.recommendation)
                else:
                    deployment_recs.append(check.recommendation)
        
        # Environment info
        environment_info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "total_checks": total_checks,
            "checks_by_status": {
                "pass": pass_count,
                "warning": warn_count,
                "critical": fail_count,
                "skipped": len([c for c in self.checks if c.status == CheckStatus.SKIP])
            }
        }
        
        return ReadinessReport(
            overall_status=overall_status,
            readiness_level=target_level,
            score=score,
            checks=self.checks,
            environment_info=environment_info,
            deployment_recommendations=deployment_recs,
            security_recommendations=security_recs,
            performance_recommendations=performance_recs
        )
    
    def export_report(self, report: ReadinessReport, output_path: Path) -> None:
        """Export readiness report to file."""
        report_data = {
            "assessment_timestamp": report.timestamp.isoformat(),
            "overall_status": report.overall_status.value,
            "readiness_level": report.readiness_level.value,
            "score": report.score,
            "environment_info": report.environment_info,
            "checks": [
                {
                    "name": check.name,
                    "category": check.category,
                    "status": check.status.value,
                    "message": check.message,
                    "recommendation": check.recommendation,
                    "impact": check.impact,
                    "fix_command": check.fix_command,
                    "documentation_url": check.documentation_url
                }
                for check in report.checks
            ],
            "recommendations": {
                "deployment": report.deployment_recommendations,
                "security": report.security_recommendations,
                "performance": report.performance_recommendations
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Production readiness report exported to: {output_path}")


def assess_production_readiness(target_level: ReadinessLevel = ReadinessLevel.PRODUCTION) -> ReadinessReport:
    """Perform production readiness assessment."""
    assessor = ProductionReadinessAssessment()
    return assessor.assess_readiness(target_level)


def generate_deployment_checklist(report: ReadinessReport) -> List[str]:
    """Generate deployment checklist from readiness report."""
    checklist = []
    
    # Critical issues first
    critical_checks = [c for c in report.checks if c.status == CheckStatus.FAIL]
    if critical_checks:
        checklist.append("🚨 CRITICAL ISSUES - Must be resolved before deployment:")
        for check in critical_checks:
            checklist.append(f"   ❌ {check.name}: {check.message}")
            if check.recommendation:
                checklist.append(f"      → {check.recommendation}")
            if check.fix_command:
                checklist.append(f"      💻 {check.fix_command}")
        checklist.append("")
    
    # Warnings
    warning_checks = [c for c in report.checks if c.status == CheckStatus.WARN]
    if warning_checks:
        checklist.append("⚠️  WARNINGS - Should be addressed:")
        for check in warning_checks:
            checklist.append(f"   ⚠️  {check.name}: {check.message}")
            if check.recommendation:
                checklist.append(f"      → {check.recommendation}")
        checklist.append("")
    
    # Final recommendations
    if report.deployment_recommendations:
        checklist.append("📋 DEPLOYMENT RECOMMENDATIONS:")
        for rec in report.deployment_recommendations:
            checklist.append(f"   • {rec}")
        checklist.append("")
    
    if report.security_recommendations:
        checklist.append("🔒 SECURITY RECOMMENDATIONS:")
        for rec in report.security_recommendations:
            checklist.append(f"   • {rec}")
        checklist.append("")
    
    if report.performance_recommendations:
        checklist.append("⚡ PERFORMANCE RECOMMENDATIONS:")
        for rec in report.performance_recommendations:
            checklist.append(f"   • {rec}")
        checklist.append("")
    
    # Summary
    checklist.append(f"📊 READINESS SUMMARY:")
    checklist.append(f"   Score: {report.score:.1f}/100")
    checklist.append(f"   Target Level: {report.readiness_level.value}")
    checklist.append(f"   Overall Status: {report.overall_status.value.upper()}")
    
    return checklist