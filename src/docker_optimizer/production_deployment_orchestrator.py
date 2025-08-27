"""Production deployment orchestrator for Docker Optimizer Agent.

Handles production-ready deployment configurations, health monitoring,
observability, and operational readiness verification.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from .comprehensive_quality_gates import ComprehensiveQualityGates, QualityGateStatus
from .adaptive_performance_engine import AdaptivePerformanceEngine
from .autonomous_scaling_system import AutonomousScalingSystem
from .enhanced_error_handling import EnhancedErrorHandler


class DeploymentStage(Enum):
    """Deployment stage definitions."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ReadinessCheck(Enum):
    """Production readiness check types."""
    HEALTH_ENDPOINTS = "health_endpoints"
    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MONITORING = "monitoring"
    BACKUP = "backup"
    DISASTER_RECOVERY = "disaster_recovery"


@dataclass
class DeploymentConfiguration:
    """Complete deployment configuration."""
    stage: DeploymentStage
    name: str
    version: str
    namespace: str = "default"
    replicas: int = 3
    resources: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "requests": {"cpu": "100m", "memory": "256Mi"},
        "limits": {"cpu": "1000m", "memory": "1Gi"}
    })
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    config_maps: List[str] = field(default_factory=list)
    service_ports: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "http", "port": 8080, "targetPort": 8080}
    ])
    health_checks: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "liveness": {
            "httpGet": {"path": "/health", "port": 8080},
            "initialDelaySeconds": 30,
            "periodSeconds": 10
        },
        "readiness": {
            "httpGet": {"path": "/ready", "port": 8080},
            "initialDelaySeconds": 5,
            "periodSeconds": 5
        }
    })
    ingress_config: Optional[Dict[str, Any]] = None
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        "metrics_enabled": True,
        "metrics_port": 9090,
        "alerts_enabled": True,
        "log_level": "INFO"
    })
    security_context: Dict[str, Any] = field(default_factory=lambda: {
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "readOnlyRootFilesystem": True,
        "allowPrivilegeEscalation": False
    })


@dataclass
class ReadinessCheckResult:
    """Result of a production readiness check."""
    check_type: ReadinessCheck
    passed: bool
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DeploymentReport:
    """Comprehensive deployment readiness report."""
    stage: DeploymentStage
    overall_readiness: bool
    overall_score: float
    readiness_checks: List[ReadinessCheckResult]
    quality_gates_report: Optional[Dict[str, Any]] = None
    deployment_config: Optional[DeploymentConfiguration] = None
    recommendations: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_deployment_time: int = 300  # seconds
    timestamp: float = field(default_factory=time.time)


class ProductionDeploymentOrchestrator:
    """Orchestrates production deployment with comprehensive readiness checks."""
    
    def __init__(
        self,
        quality_gates: Optional[ComprehensiveQualityGates] = None,
        performance_engine: Optional[AdaptivePerformanceEngine] = None,
        scaling_system: Optional[AutonomousScalingSystem] = None,
        error_handler: Optional[EnhancedErrorHandler] = None
    ):
        self.quality_gates = quality_gates or ComprehensiveQualityGates()
        self.performance_engine = performance_engine or AdaptivePerformanceEngine()
        self.scaling_system = scaling_system or AutonomousScalingSystem()
        self.error_handler = error_handler or EnhancedErrorHandler()
        
        self.logger = logging.getLogger(__name__)
        
        # Deployment templates
        self.deployment_templates = {
            DeploymentStage.DEVELOPMENT: self._get_dev_template(),
            DeploymentStage.TESTING: self._get_test_template(), 
            DeploymentStage.STAGING: self._get_staging_template(),
            DeploymentStage.PRODUCTION: self._get_production_template()
        }
        
        # Readiness check weights
        self.check_weights = {
            ReadinessCheck.HEALTH_ENDPOINTS: 0.20,
            ReadinessCheck.CONFIGURATION: 0.15,
            ReadinessCheck.DEPENDENCIES: 0.15,
            ReadinessCheck.SECURITY: 0.20,
            ReadinessCheck.PERFORMANCE: 0.15,
            ReadinessCheck.MONITORING: 0.10,
            ReadinessCheck.BACKUP: 0.03,
            ReadinessCheck.DISASTER_RECOVERY: 0.02
        }
        
        # Stage-specific requirements
        self.stage_requirements = {
            DeploymentStage.DEVELOPMENT: {
                "min_score": 60.0,
                "required_checks": [ReadinessCheck.HEALTH_ENDPOINTS, ReadinessCheck.CONFIGURATION]
            },
            DeploymentStage.TESTING: {
                "min_score": 70.0,
                "required_checks": [
                    ReadinessCheck.HEALTH_ENDPOINTS,
                    ReadinessCheck.CONFIGURATION,
                    ReadinessCheck.DEPENDENCIES,
                    ReadinessCheck.SECURITY
                ]
            },
            DeploymentStage.STAGING: {
                "min_score": 80.0,
                "required_checks": [
                    ReadinessCheck.HEALTH_ENDPOINTS,
                    ReadinessCheck.CONFIGURATION,
                    ReadinessCheck.DEPENDENCIES,
                    ReadinessCheck.SECURITY,
                    ReadinessCheck.PERFORMANCE,
                    ReadinessCheck.MONITORING
                ]
            },
            DeploymentStage.PRODUCTION: {
                "min_score": 90.0,
                "required_checks": list(ReadinessCheck)
            }
        }
    
    async def assess_deployment_readiness(
        self,
        stage: DeploymentStage,
        dockerfile_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DeploymentReport:
        """Perform comprehensive deployment readiness assessment."""
        self.logger.info(f"Starting deployment readiness assessment for {stage.value}")
        
        if context is None:
            context = {}
        
        start_time = time.time()
        
        # Run quality gates first
        quality_report = None
        try:
            quality_report = await self.quality_gates.execute_all_gates(dockerfile_content, context)
        except Exception as e:
            self.logger.error(f"Quality gates execution failed: {e}")
        
        # Perform readiness checks
        readiness_checks = await self._perform_readiness_checks(stage, dockerfile_content, context)
        
        # Calculate overall readiness
        overall_score = self._calculate_overall_score(readiness_checks)
        stage_requirements = self.stage_requirements[stage]
        
        overall_readiness = (
            overall_score >= stage_requirements["min_score"] and
            all(
                any(check.check_type == required_check and check.passed 
                   for check in readiness_checks)
                for required_check in stage_requirements["required_checks"]
            )
        )
        
        # Generate deployment configuration
        deployment_config = self._generate_deployment_config(stage, context)
        
        # Compile recommendations and blockers
        recommendations, blockers, warnings = self._analyze_readiness_results(
            readiness_checks, quality_report, overall_readiness, stage
        )
        
        # Estimate deployment time
        estimated_time = self._estimate_deployment_time(stage, deployment_config, overall_score)
        
        report = DeploymentReport(
            stage=stage,
            overall_readiness=overall_readiness,
            overall_score=overall_score,
            readiness_checks=readiness_checks,
            quality_gates_report=quality_report,
            deployment_config=deployment_config,
            recommendations=recommendations,
            blockers=blockers,
            warnings=warnings,
            estimated_deployment_time=estimated_time
        )
        
        execution_time = time.time() - start_time
        self.logger.info(
            f"Deployment readiness assessment completed in {execution_time:.2f}s: "
            f"{'READY' if overall_readiness else 'NOT READY'} (score: {overall_score:.1f})"
        )
        
        return report
    
    async def _perform_readiness_checks(
        self,
        stage: DeploymentStage,
        dockerfile_content: str,
        context: Dict[str, Any]
    ) -> List[ReadinessCheckResult]:
        """Perform all relevant readiness checks."""
        checks = []
        required_checks = self.stage_requirements[stage]["required_checks"]
        
        # Perform checks in parallel
        tasks = []
        
        if ReadinessCheck.HEALTH_ENDPOINTS in required_checks:
            tasks.append(self._check_health_endpoints(dockerfile_content, context))
        
        if ReadinessCheck.CONFIGURATION in required_checks:
            tasks.append(self._check_configuration(dockerfile_content, context))
        
        if ReadinessCheck.DEPENDENCIES in required_checks:
            tasks.append(self._check_dependencies(dockerfile_content, context))
        
        if ReadinessCheck.SECURITY in required_checks:
            tasks.append(self._check_security_readiness(dockerfile_content, context))
        
        if ReadinessCheck.PERFORMANCE in required_checks:
            tasks.append(self._check_performance_readiness(dockerfile_content, context))
        
        if ReadinessCheck.MONITORING in required_checks:
            tasks.append(self._check_monitoring_setup(dockerfile_content, context))
        
        if ReadinessCheck.BACKUP in required_checks:
            tasks.append(self._check_backup_strategy(dockerfile_content, context))
        
        if ReadinessCheck.DISASTER_RECOVERY in required_checks:
            tasks.append(self._check_disaster_recovery(dockerfile_content, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Readiness check failed: {result}")
                checks.append(ReadinessCheckResult(
                    check_type=ReadinessCheck.CONFIGURATION,  # Default
                    passed=False,
                    score=0.0,
                    message=f"Check failed: {str(result)}"
                ))
            else:
                checks.append(result)
        
        return checks
    
    async def _check_health_endpoints(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check health endpoint configuration."""
        score = 100.0
        issues = []
        recommendations = []
        
        # Check for health check endpoints
        has_health_endpoint = False
        has_ready_endpoint = False
        
        # Simple heuristic check - in production, this would be more sophisticated
        if 'HEALTHCHECK' in dockerfile_content:
            has_health_endpoint = True
            score += 10
        
        if '/health' in dockerfile_content or '/ready' in dockerfile_content:
            has_ready_endpoint = True
            score += 10
        
        if not has_health_endpoint:
            score -= 30
            issues.append("No HEALTHCHECK instruction found in Dockerfile")
            recommendations.append("Add HEALTHCHECK instruction for container health monitoring")
        
        if not has_ready_endpoint:
            score -= 20
            issues.append("No readiness endpoint detected")
            recommendations.append("Implement /health and /ready endpoints")
        
        # Check for proper health check configuration
        if 'HEALTHCHECK' in dockerfile_content:
            if '--interval=' not in dockerfile_content:
                score -= 10
                recommendations.append("Configure health check intervals")
            
            if '--retries=' not in dockerfile_content:
                score -= 10
                recommendations.append("Configure health check retry limits")
        
        score = max(0, min(100, score))
        passed = score >= 70 and has_health_endpoint
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.HEALTH_ENDPOINTS,
            passed=passed,
            score=score,
            message="Health endpoints configuration" + (f" - {', '.join(issues)}" if issues else " - OK"),
            details={
                "has_health_endpoint": has_health_endpoint,
                "has_ready_endpoint": has_ready_endpoint,
                "issues": issues
            },
            recommendations=recommendations
        )
    
    async def _check_configuration(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check configuration management."""
        score = 100.0
        issues = []
        recommendations = []
        
        # Check for proper configuration practices
        has_env_vars = 'ENV ' in dockerfile_content
        has_labels = 'LABEL ' in dockerfile_content
        has_proper_user = 'USER ' in dockerfile_content and 'USER root' not in dockerfile_content
        has_workdir = 'WORKDIR ' in dockerfile_content
        
        if not has_env_vars:
            score -= 15
            recommendations.append("Use ENV instructions for configuration")
        
        if not has_labels:
            score -= 10
            issues.append("Missing metadata labels")
            recommendations.append("Add LABEL instructions for image metadata")
        
        if not has_proper_user:
            score -= 25
            issues.append("Running as root user")
            recommendations.append("Configure non-root user with USER instruction")
        
        if not has_workdir:
            score -= 10
            recommendations.append("Set explicit WORKDIR")
        
        # Check for hardcoded values
        hardcoded_patterns = ['password=', 'secret=', 'key=', 'token=']
        hardcoded_found = any(pattern in dockerfile_content.lower() for pattern in hardcoded_patterns)
        
        if hardcoded_found:
            score -= 30
            issues.append("Hardcoded secrets or credentials detected")
            recommendations.append("Use environment variables or secrets management")
        
        score = max(0, min(100, score))
        passed = score >= 70 and has_proper_user and not hardcoded_found
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.CONFIGURATION,
            passed=passed,
            score=score,
            message="Configuration management" + (f" - {', '.join(issues)}" if issues else " - OK"),
            details={
                "has_env_vars": has_env_vars,
                "has_labels": has_labels,
                "has_proper_user": has_proper_user,
                "has_workdir": has_workdir,
                "hardcoded_found": hardcoded_found
            },
            recommendations=recommendations
        )
    
    async def _check_dependencies(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check dependency management."""
        score = 100.0
        issues = []
        recommendations = []
        
        # Check for version pinning
        run_lines = [line for line in dockerfile_content.split('\\n') if 'RUN ' in line]
        install_lines = [line for line in run_lines if 'install' in line or 'add' in line]
        
        if install_lines:
            versioned_installs = sum(1 for line in install_lines if '=' in line or ':' in line)
            version_ratio = versioned_installs / len(install_lines)
            
            if version_ratio < 0.5:
                score -= 20
                issues.append("Less than 50% of packages have pinned versions")
                recommendations.append("Pin package versions for reproducible builds")
            elif version_ratio < 0.8:
                score -= 10
                recommendations.append("Consider pinning more package versions")
        
        # Check for package manager best practices
        if 'apt-get update' in dockerfile_content:
            if 'rm -rf /var/lib/apt/lists/*' not in dockerfile_content:
                score -= 15
                issues.append("Package manager cache not cleaned")
                recommendations.append("Clean package manager caches to reduce image size")
            
            if '--no-install-recommends' not in dockerfile_content:
                score -= 10
                recommendations.append("Use --no-install-recommends to minimize dependencies")
        
        # Check for security updates
        if 'upgrade' not in dockerfile_content and 'update' in dockerfile_content:
            score -= 10
            recommendations.append("Consider running security updates")
        
        score = max(0, min(100, score))
        passed = score >= 70
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.DEPENDENCIES,
            passed=passed,
            score=score,
            message="Dependency management" + (f" - {', '.join(issues)}" if issues else " - OK"),
            details={
                "install_lines_count": len(install_lines),
                "version_ratio": versioned_installs / len(install_lines) if install_lines else 1.0
            },
            recommendations=recommendations
        )
    
    async def _check_security_readiness(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check security readiness."""
        score = 100.0
        issues = []
        recommendations = []
        
        # Basic security checks
        runs_as_root = 'USER ' not in dockerfile_content or 'USER root' in dockerfile_content
        has_privileged_commands = any(cmd in dockerfile_content.lower() 
                                    for cmd in ['sudo', 'chmod 777', '--privileged'])
        uses_latest_tag = ':latest' in dockerfile_content or dockerfile_content.count(':') == 0
        
        if runs_as_root:
            score -= 25
            issues.append("Container runs as root")
            recommendations.append("Configure non-root user")
        
        if has_privileged_commands:
            score -= 20
            issues.append("Privileged commands detected")
            recommendations.append("Avoid privileged operations")
        
        if uses_latest_tag:
            score -= 15
            issues.append("Using latest tags")
            recommendations.append("Use specific version tags")
        
        # Check for security scanning
        scan_context = context.get('security_scan', {})
        if scan_context.get('vulnerabilities', 0) > 0:
            vuln_count = scan_context['vulnerabilities']
            if vuln_count > 10:
                score -= 30
                issues.append(f"{vuln_count} vulnerabilities found")
            elif vuln_count > 5:
                score -= 15
                issues.append(f"{vuln_count} vulnerabilities found")
            else:
                score -= 5
        
        score = max(0, min(100, score))
        passed = score >= 80 and not runs_as_root
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.SECURITY,
            passed=passed,
            score=score,
            message="Security readiness" + (f" - {', '.join(issues)}" if issues else " - OK"),
            details={
                "runs_as_root": runs_as_root,
                "has_privileged_commands": has_privileged_commands,
                "uses_latest_tag": uses_latest_tag,
                "vulnerability_count": scan_context.get('vulnerabilities', 0)
            },
            recommendations=recommendations
        )
    
    async def _check_performance_readiness(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check performance readiness."""
        score = 100.0
        issues = []
        recommendations = []
        
        # Layer count check
        layer_instructions = ['RUN', 'COPY', 'ADD']
        layer_count = sum(dockerfile_content.count(f'{instruction} ') for instruction in layer_instructions)
        
        if layer_count > 20:
            score -= 20
            issues.append(f"High layer count ({layer_count})")
            recommendations.append("Combine instructions to reduce layers")
        elif layer_count > 15:
            score -= 10
            recommendations.append("Consider optimizing layer count")
        
        # Multi-stage build check
        from_count = dockerfile_content.count('FROM ')
        is_multistage = from_count > 1
        
        # If there are build tools but no multi-stage, recommend it
        build_tools = ['gcc', 'make', 'cmake', 'build-essential']
        has_build_tools = any(tool in dockerfile_content for tool in build_tools)
        
        if has_build_tools and not is_multistage:
            score -= 15
            recommendations.append("Consider multi-stage build to reduce final image size")
        
        # Cache optimization
        if 'apt-get update' in dockerfile_content:
            combined_commands = '&&' in dockerfile_content
            if not combined_commands:
                score -= 10
                recommendations.append("Combine package manager commands for better caching")
        
        # Resource optimization hints
        has_resource_limits = context.get('resource_limits', False)
        if not has_resource_limits:
            score -= 5
            recommendations.append("Configure resource limits for better performance")
        
        score = max(0, min(100, score))
        passed = score >= 70
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.PERFORMANCE,
            passed=passed,
            score=score,
            message="Performance readiness" + (f" - {', '.join(issues)}" if issues else " - OK"),
            details={
                "layer_count": layer_count,
                "is_multistage": is_multistage,
                "has_build_tools": has_build_tools
            },
            recommendations=recommendations
        )
    
    async def _check_monitoring_setup(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check monitoring and observability setup."""
        score = 100.0
        issues = []
        recommendations = []
        
        # Check for monitoring endpoints
        has_metrics_endpoint = '/metrics' in dockerfile_content or context.get('metrics_enabled', False)
        has_health_endpoint = 'HEALTHCHECK' in dockerfile_content
        has_logging_config = context.get('logging_configured', False)
        
        if not has_metrics_endpoint:
            score -= 25
            issues.append("No metrics endpoint configured")
            recommendations.append("Expose metrics endpoint for monitoring")
        
        if not has_health_endpoint:
            score -= 20
            issues.append("No health check configured")
            recommendations.append("Add health check for monitoring")
        
        if not has_logging_config:
            score -= 15
            recommendations.append("Configure structured logging")
        
        # Check for tracing
        has_tracing = context.get('tracing_enabled', False)
        if not has_tracing:
            score -= 10
            recommendations.append("Consider adding distributed tracing")
        
        score = max(0, min(100, score))
        passed = score >= 70 and has_health_endpoint
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.MONITORING,
            passed=passed,
            score=score,
            message="Monitoring setup" + (f" - {', '.join(issues)}" if issues else " - OK"),
            details={
                "has_metrics_endpoint": has_metrics_endpoint,
                "has_health_endpoint": has_health_endpoint,
                "has_logging_config": has_logging_config,
                "has_tracing": has_tracing
            },
            recommendations=recommendations
        )
    
    async def _check_backup_strategy(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check backup strategy."""
        score = 80.0  # Start with lower baseline for optional check
        recommendations = []
        
        # Check for data volume configuration
        has_volumes = 'VOLUME' in dockerfile_content
        has_persistent_storage = context.get('persistent_storage', False)
        has_backup_policy = context.get('backup_policy', False)
        
        if has_volumes or has_persistent_storage:
            score += 10
        else:
            recommendations.append("Configure data volumes for persistent storage")
        
        if has_backup_policy:
            score += 10
        else:
            recommendations.append("Define backup and retention policies")
        
        score = max(0, min(100, score))
        passed = score >= 70
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.BACKUP,
            passed=passed,
            score=score,
            message="Backup strategy - OK",
            details={
                "has_volumes": has_volumes,
                "has_persistent_storage": has_persistent_storage,
                "has_backup_policy": has_backup_policy
            },
            recommendations=recommendations
        )
    
    async def _check_disaster_recovery(self, dockerfile_content: str, context: Dict[str, Any]) -> ReadinessCheckResult:
        """Check disaster recovery readiness."""
        score = 80.0  # Start with lower baseline for optional check
        recommendations = []
        
        # Check for multi-region deployment readiness
        has_multi_region = context.get('multi_region', False)
        has_failover_config = context.get('failover_config', False)
        has_recovery_procedures = context.get('recovery_procedures', False)
        
        if has_multi_region:
            score += 10
        else:
            recommendations.append("Consider multi-region deployment strategy")
        
        if has_failover_config:
            score += 10
        else:
            recommendations.append("Configure automatic failover mechanisms")
        
        if has_recovery_procedures:
            score += 10
        else:
            recommendations.append("Document disaster recovery procedures")
        
        score = max(0, min(100, score))
        passed = score >= 70
        
        return ReadinessCheckResult(
            check_type=ReadinessCheck.DISASTER_RECOVERY,
            passed=passed,
            score=score,
            message="Disaster recovery - OK",
            details={
                "has_multi_region": has_multi_region,
                "has_failover_config": has_failover_config,
                "has_recovery_procedures": has_recovery_procedures
            },
            recommendations=recommendations
        )
    
    def _calculate_overall_score(self, readiness_checks: List[ReadinessCheckResult]) -> float:
        """Calculate weighted overall readiness score."""
        if not readiness_checks:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for check in readiness_checks:
            weight = self.check_weights.get(check.check_type, 0.1)
            weighted_sum += check.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_deployment_config(
        self, 
        stage: DeploymentStage,
        context: Dict[str, Any]
    ) -> DeploymentConfiguration:
        """Generate deployment configuration for the stage."""
        template = self.deployment_templates[stage].copy()
        
        # Customize based on context
        if 'app_name' in context:
            template.name = context['app_name']
        
        if 'version' in context:
            template.version = context['version']
        
        if 'namespace' in context:
            template.namespace = context['namespace']
        
        # Stage-specific customizations
        if stage == DeploymentStage.PRODUCTION:
            template.replicas = max(template.replicas, 3)  # Minimum 3 replicas for production
            template.resources["limits"]["cpu"] = "2000m"
            template.resources["limits"]["memory"] = "2Gi"
        
        return template
    
    def _analyze_readiness_results(
        self,
        readiness_checks: List[ReadinessCheckResult],
        quality_report: Optional[Dict[str, Any]],
        overall_readiness: bool,
        stage: DeploymentStage
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze results to generate recommendations, blockers, and warnings."""
        recommendations = []
        blockers = []
        warnings = []
        
        # Analyze readiness checks
        for check in readiness_checks:
            if not check.passed:
                if check.check_type in self.stage_requirements[stage]["required_checks"]:
                    blockers.extend([f"{check.check_type.value}: {rec}" for rec in check.recommendations])
                else:
                    warnings.extend([f"{check.check_type.value}: {rec}" for rec in check.recommendations])
            
            recommendations.extend(check.recommendations)
        
        # Analyze quality gates
        if quality_report:
            if quality_report.get("overall_status") == "failed":
                blockers.append("Quality gates failed - must be resolved before deployment")
            elif quality_report.get("overall_status") == "warning":
                warnings.append("Quality gates show warnings - review recommended")
            
            # Add specific quality gate recommendations
            for gate_name, gate_data in quality_report.get("gates", {}).items():
                if gate_data.get("status") == "failed":
                    blockers.extend([f"{gate_name}: {error}" for error in gate_data.get("errors", [])])
                
                recommendations.extend([f"{gate_name}: {sug}" for sug in gate_data.get("suggestions", [])])
        
        # Stage-specific recommendations
        if stage == DeploymentStage.PRODUCTION:
            if not overall_readiness:
                blockers.insert(0, "Production deployment blocked - resolve all critical issues")
            
            recommendations.extend([
                "Ensure monitoring and alerting are properly configured",
                "Verify backup and disaster recovery procedures",
                "Conduct security review and penetration testing",
                "Perform load testing to validate performance"
            ])
        
        # Deduplicate
        recommendations = list(set(recommendations))
        blockers = list(set(blockers))
        warnings = list(set(warnings))
        
        return recommendations, blockers, warnings
    
    def _estimate_deployment_time(
        self,
        stage: DeploymentStage,
        config: DeploymentConfiguration,
        score: float
    ) -> int:
        """Estimate deployment time in seconds."""
        base_times = {
            DeploymentStage.DEVELOPMENT: 60,
            DeploymentStage.TESTING: 120,
            DeploymentStage.STAGING: 300,
            DeploymentStage.PRODUCTION: 600
        }
        
        base_time = base_times[stage]
        
        # Adjust based on configuration complexity
        if config.replicas > 3:
            base_time += 30 * (config.replicas - 3)
        
        if len(config.secrets) > 0:
            base_time += 15 * len(config.secrets)
        
        # Adjust based on readiness score
        if score < 80:
            base_time = int(base_time * 1.5)  # Slower deployment for lower readiness
        
        return base_time
    
    def _get_dev_template(self) -> DeploymentConfiguration:
        """Get development deployment template."""
        return DeploymentConfiguration(
            stage=DeploymentStage.DEVELOPMENT,
            name="docker-optimizer-dev",
            version="latest",
            namespace="development",
            replicas=1,
            resources={
                "requests": {"cpu": "100m", "memory": "256Mi"},
                "limits": {"cpu": "500m", "memory": "512Mi"}
            },
            monitoring_config={"log_level": "DEBUG", "metrics_enabled": True}
        )
    
    def _get_test_template(self) -> DeploymentConfiguration:
        """Get testing deployment template."""
        return DeploymentConfiguration(
            stage=DeploymentStage.TESTING,
            name="docker-optimizer-test",
            version="test",
            namespace="testing",
            replicas=2,
            resources={
                "requests": {"cpu": "200m", "memory": "512Mi"},
                "limits": {"cpu": "1000m", "memory": "1Gi"}
            }
        )
    
    def _get_staging_template(self) -> DeploymentConfiguration:
        """Get staging deployment template."""
        return DeploymentConfiguration(
            stage=DeploymentStage.STAGING,
            name="docker-optimizer-staging",
            version="staging",
            namespace="staging",
            replicas=2,
            resources={
                "requests": {"cpu": "200m", "memory": "512Mi"},
                "limits": {"cpu": "1500m", "memory": "1.5Gi"}
            }
        )
    
    def _get_production_template(self) -> DeploymentConfiguration:
        """Get production deployment template."""
        return DeploymentConfiguration(
            stage=DeploymentStage.PRODUCTION,
            name="docker-optimizer",
            version="v1.0.0",
            namespace="production",
            replicas=3,
            resources={
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"cpu": "2000m", "memory": "2Gi"}
            },
            monitoring_config={
                "log_level": "INFO",
                "metrics_enabled": True,
                "alerts_enabled": True,
                "metrics_port": 9090
            }
        )
    
    def generate_kubernetes_manifests(self, config: DeploymentConfiguration) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        manifests = {}
        
        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "namespace": config.namespace,
                "labels": {
                    "app": config.name,
                    "version": config.version,
                    "stage": config.stage.value
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {"app": config.name}
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.name,
                            "version": config.version
                        }
                    },
                    "spec": {
                        "securityContext": config.security_context,
                        "containers": [{
                            "name": config.name,
                            "image": f"{config.name}:{config.version}",
                            "ports": [{"containerPort": port["targetPort"]} for port in config.service_ports],
                            "resources": config.resources,
                            "env": [{"name": k, "value": v} for k, v in config.environment_variables.items()],
                            "livenessProbe": config.health_checks["liveness"],
                            "readinessProbe": config.health_checks["readiness"]
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.name}-service",
                "namespace": config.namespace
            },
            "spec": {
                "selector": {"app": config.name},
                "ports": config.service_ports,
                "type": "ClusterIP"
            }
        }
        
        manifests["deployment.yaml"] = yaml.dump(deployment, default_flow_style=False)
        manifests["service.yaml"] = yaml.dump(service, default_flow_style=False)
        
        # Ingress manifest (if configured)
        if config.ingress_config:
            ingress = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "metadata": {
                    "name": f"{config.name}-ingress",
                    "namespace": config.namespace
                },
                "spec": config.ingress_config
            }
            manifests["ingress.yaml"] = yaml.dump(ingress, default_flow_style=False)
        
        return manifests
    
    async def validate_deployment_prerequisites(self, stage: DeploymentStage) -> Dict[str, Any]:
        """Validate deployment prerequisites for the stage."""
        prerequisites = {
            "cluster_access": False,
            "namespace_exists": False,
            "secrets_available": False,
            "registry_access": False,
            "monitoring_setup": False
        }
        
        validation_results = []
        
        # In a real implementation, these would be actual checks
        # For now, we'll simulate the validation
        
        if stage == DeploymentStage.PRODUCTION:
            prerequisites.update({
                "backup_configured": False,
                "monitoring_setup": True,
                "security_scan_passed": False,
                "load_testing_completed": False
            })
        
        return {
            "stage": stage.value,
            "prerequisites": prerequisites,
            "all_met": all(prerequisites.values()),
            "validation_time": time.time()
        }