"""Comprehensive quality gates system for Docker Optimizer Agent.

Implements automated quality assurance, security validation, performance
benchmarking, and compliance checking with detailed reporting.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .enhanced_error_handling import EnhancedErrorHandler, retry_on_failure
from .enhanced_validation import EnhancedValidator, ValidationLevel
from .adaptive_performance_engine import AdaptivePerformanceEngine


class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class QualityGateType(Enum):
    """Types of quality gates."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    FUNCTIONAL = "functional"
    INTEGRATION = "integration"
    DEPLOYMENT = "deployment"


class SeverityLevel(Enum):
    """Severity levels for quality gate failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """Individual quality metric measurement."""
    name: str
    value: Union[float, int, str, bool]
    unit: str = ""
    threshold: Optional[Union[float, int]] = None
    threshold_operator: str = ">="  # >=, <=, ==, !=
    passed: Optional[bool] = None
    message: str = ""
    
    def evaluate(self) -> bool:
        """Evaluate metric against threshold."""
        if self.threshold is None:
            return True
        
        try:
            if self.threshold_operator == ">=":
                self.passed = float(self.value) >= self.threshold
            elif self.threshold_operator == "<=":
                self.passed = float(self.value) <= self.threshold
            elif self.threshold_operator == "==":
                self.passed = float(self.value) == self.threshold
            elif self.threshold_operator == "!=":
                self.passed = float(self.value) != self.threshold
            elif self.threshold_operator == ">":
                self.passed = float(self.value) > self.threshold
            elif self.threshold_operator == "<":
                self.passed = float(self.value) < self.threshold
            else:
                self.passed = True  # Unknown operator, assume pass
                
        except (ValueError, TypeError):
            # Non-numeric comparison
            if self.threshold_operator == "==":
                self.passed = self.value == self.threshold
            elif self.threshold_operator == "!=":
                self.passed = self.value != self.threshold
            else:
                self.passed = True
        
        return self.passed


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_type: QualityGateType
    status: QualityGateStatus
    metrics: List[QualityMetric] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    severity: SeverityLevel = SeverityLevel.LOW
    
    @property
    def passed(self) -> bool:
        """Check if quality gate passed."""
        return self.status == QualityGateStatus.PASSED
    
    @property
    def failed_metrics_count(self) -> int:
        """Count of failed metrics."""
        return sum(1 for metric in self.metrics if metric.passed is False)
    
    def calculate_score(self) -> float:
        """Calculate quality score (0-100)."""
        if not self.metrics:
            return 100.0 if self.passed else 0.0
        
        passed_metrics = sum(1 for metric in self.metrics if metric.passed is not False)
        return (passed_metrics / len(self.metrics)) * 100.0


class SecurityQualityGate:
    """Security-focused quality gate."""
    
    def __init__(self, validator: EnhancedValidator):
        self.validator = validator
        self.logger = logging.getLogger(__name__)
    
    @retry_on_failure(max_attempts=2)
    async def execute(self, dockerfile_content: str, context: Dict[str, Any]) -> QualityGateResult:
        """Execute security quality gate."""
        start_time = time.time()
        result = QualityGateResult(
            gate_type=QualityGateType.SECURITY,
            status=QualityGateStatus.RUNNING
        )
        
        try:
            # Perform validation
            validation_result = self.validator.validate_dockerfile_content(dockerfile_content)
            
            # Security-specific checks
            security_metrics = await self._check_security_metrics(dockerfile_content, context)
            result.metrics.extend(security_metrics)
            
            # Evaluate results
            critical_issues = len([e for e in validation_result.errors if "secret" in e.lower() or "credential" in e.lower()])
            high_issues = len([w for w in validation_result.warnings if "root" in w.lower() or "privileged" in w.lower()])
            
            # Add summary metrics
            result.metrics.extend([
                QualityMetric(
                    name="critical_security_issues",
                    value=critical_issues,
                    threshold=0,
                    threshold_operator="<=",
                    message="Critical security issues found"
                ),
                QualityMetric(
                    name="high_security_issues", 
                    value=high_issues,
                    threshold=2,
                    threshold_operator="<=",
                    message="High severity security issues"
                ),
                QualityMetric(
                    name="security_validation_passed",
                    value=validation_result.is_valid,
                    message="Overall security validation status"
                )
            ])
            
            # Evaluate all metrics
            failed_metrics = 0
            for metric in result.metrics:
                metric.evaluate()
                if not metric.passed:
                    failed_metrics += 1
            
            # Determine status and severity
            if critical_issues > 0:
                result.status = QualityGateStatus.FAILED
                result.severity = SeverityLevel.CRITICAL
                result.errors.extend([e for e in validation_result.errors if "secret" in e.lower() or "credential" in e.lower()])
            elif high_issues > 5:
                result.status = QualityGateStatus.FAILED
                result.severity = SeverityLevel.HIGH
            elif failed_metrics > len(result.metrics) // 2:
                result.status = QualityGateStatus.WARNING
                result.severity = SeverityLevel.MEDIUM
            else:
                result.status = QualityGateStatus.PASSED
                result.severity = SeverityLevel.LOW
            
            result.warnings.extend(validation_result.warnings)
            result.suggestions.extend(validation_result.suggestions)
            
        except Exception as e:
            self.logger.error(f"Security quality gate failed: {e}")
            result.status = QualityGateStatus.FAILED
            result.severity = SeverityLevel.CRITICAL
            result.errors.append(f"Security validation failed: {str(e)}")
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    async def _check_security_metrics(self, dockerfile_content: str, context: Dict[str, Any]) -> List[QualityMetric]:
        """Check additional security metrics."""
        metrics = []
        
        # Base image security score (simplified)
        base_image_score = self._evaluate_base_image_security(dockerfile_content)
        metrics.append(QualityMetric(
            name="base_image_security_score",
            value=base_image_score,
            unit="score",
            threshold=7.0,
            threshold_operator=">=",
            message="Base image security score (1-10)"
        ))
        
        # User privilege check
        has_user_instruction = "USER " in dockerfile_content and "USER root" not in dockerfile_content
        metrics.append(QualityMetric(
            name="non_root_user_configured",
            value=has_user_instruction,
            message="Non-root user configured"
        ))
        
        # Package manager security
        secure_package_usage = self._check_secure_package_usage(dockerfile_content)
        metrics.append(QualityMetric(
            name="secure_package_usage",
            value=secure_package_usage,
            threshold=8.0,
            threshold_operator=">=",
            message="Secure package management practices score"
        ))
        
        return metrics
    
    def _evaluate_base_image_security(self, dockerfile_content: str) -> float:
        """Evaluate base image security (simplified scoring)."""
        lines = dockerfile_content.lower().split('\\n')
        from_lines = [line for line in lines if line.strip().startswith('from ')]
        
        if not from_lines:
            return 1.0  # No FROM instruction
        
        score = 10.0
        
        for from_line in from_lines:
            # Penalize latest tags
            if ':latest' in from_line or ':' not in from_line.split()[1]:
                score -= 3.0
            
            # Reward specific security-focused images
            if any(secure in from_line for secure in ['distroless', 'scratch', 'alpine']):
                score += 1.0
            
            # Penalize potentially insecure base images
            if any(insecure in from_line for insecure in ['ubuntu:14.04', 'centos:6']):
                score -= 2.0
        
        return max(1.0, min(10.0, score))
    
    def _check_secure_package_usage(self, dockerfile_content: str) -> float:
        """Check secure package management usage."""
        score = 10.0
        
        # Check for package manager best practices
        if 'apt-get update' in dockerfile_content:
            if 'rm -rf /var/lib/apt/lists/*' not in dockerfile_content:
                score -= 2.0  # Missing cache cleanup
            
            if '--no-install-recommends' not in dockerfile_content:
                score -= 1.0  # Not using minimal packages
        
        # Check for version pinning
        run_lines = [line for line in dockerfile_content.split('\\n') if 'RUN ' in line and ('install' in line or 'add' in line)]
        if run_lines:
            version_pinned = sum(1 for line in run_lines if '=' in line or '@' in line)
            version_ratio = version_pinned / len(run_lines)
            if version_ratio < 0.5:
                score -= 3.0  # Less than 50% version pinned
        
        return max(1.0, min(10.0, score))


class PerformanceQualityGate:
    """Performance-focused quality gate."""
    
    def __init__(self, performance_engine: AdaptivePerformanceEngine):
        self.performance_engine = performance_engine
        self.logger = logging.getLogger(__name__)
    
    @retry_on_failure(max_attempts=2)
    async def execute(self, dockerfile_content: str, context: Dict[str, Any]) -> QualityGateResult:
        """Execute performance quality gate."""
        start_time = time.time()
        result = QualityGateResult(
            gate_type=QualityGateType.PERFORMANCE,
            status=QualityGateStatus.RUNNING
        )
        
        try:
            # Performance metrics collection
            performance_metrics = await self._collect_performance_metrics(dockerfile_content, context)
            result.metrics.extend(performance_metrics)
            
            # Build performance simulation
            build_metrics = await self._simulate_build_performance(dockerfile_content)
            result.metrics.extend(build_metrics)
            
            # Evaluate all metrics
            failed_metrics = 0
            critical_failures = 0
            
            for metric in result.metrics:
                metric.evaluate()
                if not metric.passed:
                    failed_metrics += 1
                    if metric.name in ['estimated_build_time', 'estimated_image_size']:
                        critical_failures += 1
            
            # Determine status
            if critical_failures > 0:
                result.status = QualityGateStatus.FAILED
                result.severity = SeverityLevel.HIGH
            elif failed_metrics > len(result.metrics) // 3:
                result.status = QualityGateStatus.WARNING
                result.severity = SeverityLevel.MEDIUM
            else:
                result.status = QualityGateStatus.PASSED
                result.severity = SeverityLevel.LOW
            
        except Exception as e:
            self.logger.error(f"Performance quality gate failed: {e}")
            result.status = QualityGateStatus.FAILED
            result.severity = SeverityLevel.HIGH
            result.errors.append(f"Performance evaluation failed: {str(e)}")
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    async def _collect_performance_metrics(self, dockerfile_content: str, context: Dict[str, Any]) -> List[QualityMetric]:
        """Collect performance-related metrics."""
        metrics = []
        
        # Layer count analysis
        layer_count = len([line for line in dockerfile_content.split('\\n') 
                          if any(line.strip().startswith(cmd) for cmd in ['RUN', 'COPY', 'ADD'])])
        
        metrics.append(QualityMetric(
            name="dockerfile_layer_count",
            value=layer_count,
            unit="layers",
            threshold=15,
            threshold_operator="<=",
            message="Number of Docker layers"
        ))
        
        # Caching optimization score
        cache_score = self._analyze_cache_optimization(dockerfile_content)
        metrics.append(QualityMetric(
            name="cache_optimization_score",
            value=cache_score,
            unit="score",
            threshold=7.0,
            threshold_operator=">=",
            message="Docker build cache optimization score (1-10)"
        ))
        
        # Multi-stage build efficiency
        multistage_efficiency = self._analyze_multistage_efficiency(dockerfile_content)
        if multistage_efficiency is not None:
            metrics.append(QualityMetric(
                name="multistage_efficiency_score",
                value=multistage_efficiency,
                unit="score", 
                threshold=6.0,
                threshold_operator=">=",
                message="Multi-stage build efficiency score (1-10)"
            ))
        
        return metrics
    
    async def _simulate_build_performance(self, dockerfile_content: str) -> List[QualityMetric]:
        """Simulate build performance metrics."""
        # This is a simplified simulation - in practice, you might use
        # Docker BuildKit or other tools for more accurate estimation
        
        metrics = []
        
        # Estimate build time based on complexity
        complexity_score = self._calculate_complexity(dockerfile_content)
        estimated_build_time = complexity_score * 30  # seconds per complexity point
        
        metrics.append(QualityMetric(
            name="estimated_build_time",
            value=estimated_build_time,
            unit="seconds",
            threshold=300,  # 5 minutes
            threshold_operator="<=",
            message="Estimated Docker build time"
        ))
        
        # Estimate final image size
        estimated_size = self._estimate_image_size(dockerfile_content)
        metrics.append(QualityMetric(
            name="estimated_image_size",
            value=estimated_size,
            unit="MB",
            threshold=500,
            threshold_operator="<=",
            message="Estimated final image size"
        ))
        
        return metrics
    
    def _analyze_cache_optimization(self, dockerfile_content: str) -> float:
        """Analyze Docker build cache optimization."""
        score = 10.0
        lines = dockerfile_content.strip().split('\\n')
        
        # Check for common cache-busting patterns
        copy_before_install = False
        run_instructions = []
        copy_instructions = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('RUN '):
                run_instructions.append((i, line))
            elif line.startswith('COPY ') or line.startswith('ADD '):
                copy_instructions.append((i, line))
        
        # Penalize COPY instructions before package installation
        for copy_idx, _ in copy_instructions:
            for run_idx, run_line in run_instructions:
                if copy_idx < run_idx and any(pkg in run_line for pkg in ['install', 'update', 'upgrade']):
                    score -= 2.0
                    break
        
        # Reward proper layer ordering
        package_files_early = any('package' in line or 'requirements' in line 
                                 for _, line in copy_instructions[:2])
        if package_files_early:
            score += 1.0
        
        # Check for combined RUN instructions
        separate_apt_commands = 0
        for _, line in run_instructions:
            if 'apt-get' in line and '&&' not in line:
                separate_apt_commands += 1
        
        if separate_apt_commands > 1:
            score -= 1.5
        
        return max(1.0, min(10.0, score))
    
    def _analyze_multistage_efficiency(self, dockerfile_content: str) -> Optional[float]:
        """Analyze multi-stage build efficiency."""
        from_count = dockerfile_content.count('FROM ')
        
        if from_count <= 1:
            return None  # Not a multi-stage build
        
        score = 10.0
        
        # Check for proper stage naming
        named_stages = dockerfile_content.count(' AS ')
        if named_stages < from_count - 1:
            score -= 2.0
        
        # Check for COPY --from usage
        copy_from_count = dockerfile_content.count('COPY --from=')
        if copy_from_count == 0:
            score -= 3.0  # Multi-stage but not copying between stages
        
        # Check for build tools in final stage
        final_stage_lines = dockerfile_content.split('FROM ')[-1].split('\\n')
        build_tools = ['gcc', 'make', 'cmake', 'build-essential', 'dev-tools']
        
        for line in final_stage_lines:
            if any(tool in line for tool in build_tools):
                score -= 2.0  # Build tools in final stage
                break
        
        return max(1.0, min(10.0, score))
    
    def _calculate_complexity(self, dockerfile_content: str) -> float:
        """Calculate Dockerfile complexity score."""
        lines = dockerfile_content.strip().split('\\n')
        complexity = 0.0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Base complexity per instruction
            complexity += 1.0
            
            # Additional complexity for certain patterns
            if '&&' in line:
                complexity += line.count('&&') * 0.5
                
            if any(cmd in line for cmd in ['curl', 'wget', 'git']):
                complexity += 1.0  # Network operations
                
            if 'compile' in line or 'build' in line or 'make' in line:
                complexity += 2.0  # Build operations
        
        return complexity
    
    def _estimate_image_size(self, dockerfile_content: str) -> float:
        """Estimate final image size in MB."""
        # Very simplified estimation based on base image and operations
        base_sizes = {
            'alpine': 5,
            'ubuntu': 70,
            'debian': 100,
            'centos': 200,
            'node': 900,
            'python': 800,
            'java': 400
        }
        
        base_size = 100  # Default
        dockerfile_lower = dockerfile_content.lower()
        
        for image, size in base_sizes.items():
            if f'from {image}' in dockerfile_lower or f':{image}' in dockerfile_lower:
                base_size = size
                break
        
        # Add size for operations
        operation_overhead = 0
        
        # Package installations add size
        if 'apt-get install' in dockerfile_lower or 'apk add' in dockerfile_lower:
            package_count = dockerfile_lower.count('install') + dockerfile_lower.count('apk add')
            operation_overhead += package_count * 50  # 50MB per package group
        
        # Application code
        if 'COPY' in dockerfile_content or 'ADD' in dockerfile_content:
            operation_overhead += 100  # Assume 100MB for application code
        
        return base_size + operation_overhead


class ComplianceQualityGate:
    """Compliance and best practices quality gate."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compliance_rules = {
            "dockerfile_standards": {
                "has_from": True,
                "has_maintainer_or_label": True,
                "no_latest_tags": True,
                "has_health_check": False,  # Optional
                "uses_non_root_user": True
            },
            "security_standards": {
                "no_secrets_hardcoded": True,
                "no_privileged_commands": True,
                "secure_base_images": True
            },
            "operational_standards": {
                "has_proper_signal_handling": False,  # Optional
                "proper_logging": False,  # Optional
                "resource_limits": False   # Optional
            }
        }
    
    @retry_on_failure(max_attempts=2)
    async def execute(self, dockerfile_content: str, context: Dict[str, Any]) -> QualityGateResult:
        """Execute compliance quality gate."""
        start_time = time.time()
        result = QualityGateResult(
            gate_type=QualityGateType.COMPLIANCE,
            status=QualityGateStatus.RUNNING
        )
        
        try:
            # Check compliance metrics
            compliance_metrics = await self._check_compliance_metrics(dockerfile_content, context)
            result.metrics.extend(compliance_metrics)
            
            # Evaluate all metrics
            failed_required = 0
            failed_optional = 0
            
            for metric in result.metrics:
                metric.evaluate()
                if not metric.passed:
                    if "required" in metric.message.lower():
                        failed_required += 1
                    else:
                        failed_optional += 1
            
            # Determine status
            if failed_required > 0:
                result.status = QualityGateStatus.FAILED
                result.severity = SeverityLevel.HIGH
            elif failed_optional > 2:
                result.status = QualityGateStatus.WARNING
                result.severity = SeverityLevel.MEDIUM
            else:
                result.status = QualityGateStatus.PASSED
                result.severity = SeverityLevel.LOW
            
            # Add recommendations
            if failed_required > 0:
                result.suggestions.append("Address required compliance issues before deployment")
            if failed_optional > 0:
                result.suggestions.append("Consider implementing optional compliance improvements")
            
        except Exception as e:
            self.logger.error(f"Compliance quality gate failed: {e}")
            result.status = QualityGateStatus.FAILED
            result.severity = SeverityLevel.HIGH
            result.errors.append(f"Compliance check failed: {str(e)}")
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    async def _check_compliance_metrics(self, dockerfile_content: str, context: Dict[str, Any]) -> List[QualityMetric]:
        """Check compliance metrics."""
        metrics = []
        
        # Dockerfile standards
        has_from = dockerfile_content.strip().startswith('FROM') or '\\nFROM' in dockerfile_content
        metrics.append(QualityMetric(
            name="has_from_instruction",
            value=has_from,
            message="FROM instruction present (required)"
        ))
        
        has_maintainer_or_label = ('MAINTAINER' in dockerfile_content or 'LABEL maintainer' in dockerfile_content)
        metrics.append(QualityMetric(
            name="has_maintainer_info",
            value=has_maintainer_or_label,
            message="Maintainer information present (required)"
        ))
        
        no_latest_tags = ':latest' not in dockerfile_content
        metrics.append(QualityMetric(
            name="no_latest_tags",
            value=no_latest_tags,
            message="No latest tags used (required)"
        ))
        
        has_user = 'USER ' in dockerfile_content and 'USER root' not in dockerfile_content
        metrics.append(QualityMetric(
            name="uses_non_root_user",
            value=has_user,
            message="Non-root user configured (required)"
        ))
        
        # Security standards
        has_secrets = any(pattern in dockerfile_content.lower() 
                         for pattern in ['password=', 'secret=', 'api_key=', 'token='])
        metrics.append(QualityMetric(
            name="no_hardcoded_secrets",
            value=not has_secrets,
            message="No hardcoded secrets (required)"
        ))
        
        # Optional standards
        has_health_check = 'HEALTHCHECK' in dockerfile_content
        metrics.append(QualityMetric(
            name="has_health_check",
            value=has_health_check,
            message="Health check configured (optional)"
        ))
        
        has_signal_handling = 'STOPSIGNAL' in dockerfile_content
        metrics.append(QualityMetric(
            name="proper_signal_handling",
            value=has_signal_handling,
            message="Proper signal handling configured (optional)"
        ))
        
        return metrics


class ComprehensiveQualityGates:
    """Main quality gates orchestrator."""
    
    def __init__(
        self,
        validator: Optional[EnhancedValidator] = None,
        performance_engine: Optional[AdaptivePerformanceEngine] = None,
        error_handler: Optional[EnhancedErrorHandler] = None
    ):
        self.validator = validator or EnhancedValidator(level=ValidationLevel.STANDARD)
        self.performance_engine = performance_engine or AdaptivePerformanceEngine()
        self.error_handler = error_handler or EnhancedErrorHandler()
        
        # Initialize quality gates
        self.security_gate = SecurityQualityGate(self.validator)
        self.performance_gate = PerformanceQualityGate(self.performance_engine)
        self.compliance_gate = ComplianceQualityGate()
        
        # Quality gate configuration
        self.enabled_gates = {
            QualityGateType.SECURITY: True,
            QualityGateType.PERFORMANCE: True,
            QualityGateType.COMPLIANCE: True,
        }
        
        self.gate_weights = {
            QualityGateType.SECURITY: 0.4,      # 40% weight
            QualityGateType.PERFORMANCE: 0.35,  # 35% weight
            QualityGateType.COMPLIANCE: 0.25,   # 25% weight
        }
        
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    async def execute_all_gates(
        self,
        dockerfile_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute all enabled quality gates."""
        if context is None:
            context = {}
        
        start_time = time.time()
        gate_results = {}
        overall_status = QualityGateStatus.PASSED
        overall_score = 0.0
        
        self.logger.info("Starting comprehensive quality gate execution")
        
        try:
            # Execute gates in parallel for better performance
            tasks = []
            
            if self.enabled_gates.get(QualityGateType.SECURITY):
                tasks.append(self._execute_gate_with_timeout(
                    self.security_gate, dockerfile_content, context, QualityGateType.SECURITY
                ))
            
            if self.enabled_gates.get(QualityGateType.PERFORMANCE):
                tasks.append(self._execute_gate_with_timeout(
                    self.performance_gate, dockerfile_content, context, QualityGateType.PERFORMANCE
                ))
            
            if self.enabled_gates.get(QualityGateType.COMPLIANCE):
                tasks.append(self._execute_gate_with_timeout(
                    self.compliance_gate, dockerfile_content, context, QualityGateType.COMPLIANCE
                ))
            
            # Wait for all gates to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            total_weight = 0.0
            weighted_score = 0.0
            
            for i, result in enumerate(results):
                gate_type = list(self.enabled_gates.keys())[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"Quality gate {gate_type.value} failed with exception: {result}")
                    gate_results[gate_type.value] = QualityGateResult(
                        gate_type=gate_type,
                        status=QualityGateStatus.FAILED,
                        errors=[f"Gate execution failed: {str(result)}"],
                        severity=SeverityLevel.CRITICAL
                    )
                else:
                    gate_results[gate_type.value] = result
                
                # Calculate weighted score
                gate_result = gate_results[gate_type.value]
                gate_score = gate_result.calculate_score()
                weight = self.gate_weights.get(gate_type, 0.0)
                
                weighted_score += gate_score * weight
                total_weight += weight
                
                # Determine overall status
                if gate_result.status == QualityGateStatus.FAILED:
                    if gate_result.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                        overall_status = QualityGateStatus.FAILED
                    elif overall_status == QualityGateStatus.PASSED:
                        overall_status = QualityGateStatus.WARNING
                elif gate_result.status == QualityGateStatus.WARNING and overall_status == QualityGateStatus.PASSED:
                    overall_status = QualityGateStatus.WARNING
            
            # Calculate final score
            if total_weight > 0:
                overall_score = weighted_score / total_weight
            
        except Exception as e:
            self.logger.error(f"Quality gates execution failed: {e}")
            overall_status = QualityGateStatus.FAILED
            overall_score = 0.0
        
        execution_time = time.time() - start_time
        
        # Compile final report
        report = {
            "overall_status": overall_status.value,
            "overall_score": round(overall_score, 2),
            "execution_time": round(execution_time, 2),
            "timestamp": time.time(),
            "gates": {
                gate_type: {
                    "status": result.status.value,
                    "score": round(result.calculate_score(), 2),
                    "execution_time": round(result.execution_time, 2),
                    "metrics": [
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "unit": metric.unit,
                            "passed": metric.passed,
                            "threshold": metric.threshold,
                            "message": metric.message
                        } for metric in result.metrics
                    ],
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "suggestions": result.suggestions,
                    "severity": result.severity.value
                } for gate_type, result in gate_results.items()
            },
            "summary": self._generate_summary(gate_results, overall_score, overall_status),
            "recommendations": self._generate_recommendations(gate_results, overall_status)
        }
        
        # Store in history
        self.execution_history.append(report)
        
        # Keep history limited
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]
        
        self.logger.info(f"Quality gates execution completed: {overall_status.value} (score: {overall_score:.1f})")
        
        return report
    
    async def _execute_gate_with_timeout(
        self, 
        gate, 
        dockerfile_content: str, 
        context: Dict[str, Any],
        gate_type: QualityGateType,
        timeout: float = 120.0
    ) -> QualityGateResult:
        """Execute quality gate with timeout protection."""
        try:
            return await asyncio.wait_for(
                gate.execute(dockerfile_content, context),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Quality gate {gate_type.value} timed out after {timeout} seconds")
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityGateStatus.FAILED,
                errors=[f"Gate execution timed out after {timeout} seconds"],
                severity=SeverityLevel.HIGH
            )
    
    def _generate_summary(
        self, 
        gate_results: Dict[str, QualityGateResult], 
        overall_score: float,
        overall_status: QualityGateStatus
    ) -> Dict[str, Any]:
        """Generate execution summary."""
        total_metrics = sum(len(result.metrics) for result in gate_results.values())
        passed_metrics = sum(
            sum(1 for metric in result.metrics if metric.passed is not False)
            for result in gate_results.values()
        )
        
        total_errors = sum(len(result.errors) for result in gate_results.values())
        total_warnings = sum(len(result.warnings) for result in gate_results.values())
        total_suggestions = sum(len(result.suggestions) for result in gate_results.values())
        
        return {
            "overall_score": overall_score,
            "overall_status": overall_status.value,
            "total_gates_executed": len(gate_results),
            "gates_passed": sum(1 for result in gate_results.values() 
                              if result.status == QualityGateStatus.PASSED),
            "gates_failed": sum(1 for result in gate_results.values() 
                              if result.status == QualityGateStatus.FAILED),
            "gates_warning": sum(1 for result in gate_results.values() 
                               if result.status == QualityGateStatus.WARNING),
            "total_metrics": total_metrics,
            "passed_metrics": passed_metrics,
            "metric_success_rate": (passed_metrics / total_metrics * 100) if total_metrics > 0 else 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_suggestions": total_suggestions
        }
    
    def _generate_recommendations(
        self,
        gate_results: Dict[str, QualityGateResult],
        overall_status: QualityGateStatus
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if overall_status == QualityGateStatus.FAILED:
            recommendations.append("❌ Critical issues detected - address before deployment")
            
            # Identify most critical issues
            for gate_type, result in gate_results.items():
                if result.status == QualityGateStatus.FAILED and result.errors:
                    recommendations.append(f"🔧 {gate_type.title()}: {result.errors[0]}")
        
        elif overall_status == QualityGateStatus.WARNING:
            recommendations.append("⚠️ Some improvements recommended")
        
        else:
            recommendations.append("✅ Quality gates passed successfully")
        
        # Add specific recommendations from gates
        all_suggestions = []
        for result in gate_results.values():
            all_suggestions.extend(result.suggestions)
        
        # Deduplicate and add top suggestions
        unique_suggestions = list(set(all_suggestions))[:5]  # Top 5 unique suggestions
        recommendations.extend([f"💡 {suggestion}" for suggestion in unique_suggestions])
        
        return recommendations
    
    def get_quality_trends(self, last_n_executions: int = 10) -> Dict[str, Any]:
        """Get quality trends over recent executions."""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        recent_executions = self.execution_history[-last_n_executions:]
        
        # Calculate trends
        scores = [exec_data["overall_score"] for exec_data in recent_executions]
        execution_times = [exec_data["execution_time"] for exec_data in recent_executions]
        
        if len(scores) > 1:
            score_trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
            time_trend = "faster" if execution_times[-1] < execution_times[0] else "slower" if execution_times[-1] > execution_times[0] else "stable"
        else:
            score_trend = "insufficient_data"
            time_trend = "insufficient_data"
        
        return {
            "executions_analyzed": len(recent_executions),
            "average_score": sum(scores) / len(scores),
            "score_trend": score_trend,
            "latest_score": scores[-1] if scores else 0,
            "average_execution_time": sum(execution_times) / len(execution_times),
            "time_trend": time_trend,
            "latest_execution_time": execution_times[-1] if execution_times else 0,
            "success_rate": sum(1 for exec_data in recent_executions 
                              if exec_data["overall_status"] == "passed") / len(recent_executions) * 100
        }