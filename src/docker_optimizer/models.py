"""Data models for Docker optimization results."""

from typing import Dict, List, Optional

# Graceful pydantic import with fallback
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Simple fallback base class if pydantic is not available
    PYDANTIC_AVAILABLE = False

    class BaseModel:
        """Fallback BaseModel when pydantic is not available."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, description="", **kwargs):
        """Fallback Field function that handles all pydantic Field parameters."""
        if 'default_factory' in kwargs:
            return kwargs['default_factory']()
        return default

    def validator(*args, **kwargs):
        """Fallback validator decorator."""
        def decorator(func):
            return func
        return decorator


class SecurityFix(BaseModel):
    """Represents a security vulnerability fix."""

    vulnerability: str = Field(..., description="CVE identifier or vulnerability name")
    severity: str = Field(
        ..., description="Severity level: LOW, MEDIUM, HIGH, CRITICAL"
    )
    description: str = Field(..., description="Description of the vulnerability")
    fix: str = Field(..., description="Description of the applied fix")

    @validator("severity")
    def validate_severity(cls, v: str) -> str:
        """Validate severity is one of the allowed values."""
        allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f'Severity must be one of: {", ".join(allowed)}')
        return v.upper()


class LayerOptimization(BaseModel):
    """Represents an optimization applied to a Dockerfile layer."""

    original_instruction: str = Field(
        ..., description="Original Dockerfile instruction"
    )
    optimized_instruction: str = Field(
        ..., description="Optimized Dockerfile instruction"
    )
    reasoning: str = Field(
        ..., description="Explanation of why this optimization was applied"
    )


class OptimizationResult(BaseModel):
    """Complete result of Dockerfile optimization analysis."""

    original_size: str = Field(..., description="Estimated original image size")
    optimized_size: str = Field(..., description="Estimated optimized image size")
    security_fixes: List[SecurityFix] = Field(
        default_factory=list, description="Security improvements applied"
    )
    explanation: str = Field(..., description="Overall explanation of optimizations")
    optimized_dockerfile: str = Field(
        ..., description="Complete optimized Dockerfile content"
    )
    layer_optimizations: List[LayerOptimization] = Field(
        default_factory=list, description="Layer-specific optimizations"
    )

    @property
    def has_security_improvements(self) -> bool:
        """Check if any security improvements were made."""
        return len(self.security_fixes) > 0

    @property
    def has_layer_optimizations(self) -> bool:
        """Check if any layer optimizations were made."""
        return len(self.layer_optimizations) > 0


class DockerfileAnalysis(BaseModel):
    """Analysis results of a Dockerfile before optimization."""

    base_image: str = Field(..., description="Base image used")
    total_layers: int = Field(..., description="Total number of layers")
    security_issues: List[str] = Field(
        default_factory=list, description="Identified security issues"
    )
    optimization_opportunities: List[str] = Field(
        default_factory=list, description="Identified optimization opportunities"
    )
    estimated_size: Optional[str] = Field(None, description="Estimated image size")

    @property
    def has_security_issues(self) -> bool:
        """Check if any security issues were found."""
        return len(self.security_issues) > 0

    @property
    def has_optimization_opportunities(self) -> bool:
        """Check if any optimization opportunities were found."""
        return len(self.optimization_opportunities) > 0


class BuildStage(BaseModel):
    """Represents a build stage in a multi-stage Dockerfile."""

    name: str = Field(..., description="Stage name or alias")
    base_image: str = Field(..., description="Base image for this stage")
    commands: List[str] = Field(
        default_factory=list, description="Commands in this stage"
    )
    purpose: str = Field(
        ..., description="Purpose: 'build', 'runtime', or 'intermediate'"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Dependencies installed"
    )

    @property
    def is_build_stage(self) -> bool:
        """Check if this is a build stage."""
        return self.purpose == "build"

    @property
    def is_runtime_stage(self) -> bool:
        """Check if this is a runtime stage."""
        return self.purpose == "runtime"


class MultiStageOpportunity(BaseModel):
    """Analysis of multi-stage build optimization opportunity."""

    recommended: bool = Field(
        ..., description="Whether multi-stage build is recommended"
    )
    has_build_dependencies: bool = Field(
        ..., description="Whether build dependencies were found"
    )
    build_dependencies: List[str] = Field(
        default_factory=list, description="Build-only dependencies"
    )
    runtime_dependencies: List[str] = Field(
        default_factory=list, description="Runtime dependencies"
    )
    benefits: List[str] = Field(default_factory=list, description="Expected benefits")
    estimated_size_reduction: str = Field(..., description="Estimated size reduction")
    complexity_score: int = Field(..., description="Implementation complexity (1-10)")


class MultiStageOptimization(BaseModel):
    """Result of multi-stage build optimization."""

    original_dockerfile: str = Field(..., description="Original Dockerfile content")
    optimized_dockerfile: str = Field(
        ..., description="Optimized multi-stage Dockerfile"
    )
    stages: List[BuildStage] = Field(
        default_factory=list, description="Build stages created"
    )
    estimated_size_reduction: int = Field(
        ..., description="Estimated size reduction in MB"
    )
    security_improvements: int = Field(
        default=0, description="Number of security improvements"
    )
    size_reduction: int = Field(..., description="Size reduction in MB")
    explanation: str = Field(..., description="Explanation of optimizations applied")

    @property
    def has_multiple_stages(self) -> bool:
        """Check if optimization resulted in multiple stages."""
        return len(self.stages) > 1


class CVEDetails(BaseModel):
    """Details of a specific CVE vulnerability."""

    cve_id: str = Field(..., description="CVE identifier")
    severity: str = Field(..., description="Vulnerability severity")
    package: str = Field(..., description="Affected package name")
    installed_version: str = Field(..., description="Currently installed version")
    fixed_version: Optional[str] = Field(
        None, description="Version that fixes the vulnerability"
    )
    description: str = Field(..., description="Vulnerability description")

    @property
    def has_fix(self) -> bool:
        """Check if a fix is available."""
        return self.fixed_version is not None


class VulnerabilityReport(BaseModel):
    """Report of vulnerabilities found in a Docker image or Dockerfile."""

    total_vulnerabilities: int = Field(
        ..., description="Total number of vulnerabilities"
    )
    critical_count: int = Field(
        default=0, description="Number of critical vulnerabilities"
    )
    high_count: int = Field(
        default=0, description="Number of high severity vulnerabilities"
    )
    medium_count: int = Field(
        default=0, description="Number of medium severity vulnerabilities"
    )
    low_count: int = Field(
        default=0, description="Number of low severity vulnerabilities"
    )
    cve_details: List[CVEDetails] = Field(
        default_factory=list, description="Detailed CVE information"
    )

    @property
    def has_critical_vulnerabilities(self) -> bool:
        """Check if there are any critical vulnerabilities."""
        return self.critical_count > 0

    @property
    def has_high_vulnerabilities(self) -> bool:
        """Check if there are any high severity vulnerabilities."""
        return self.high_count > 0

    @property
    def severity_distribution(self) -> Dict[str, int]:
        """Get distribution of vulnerabilities by severity."""
        return {
            "critical": self.critical_count,
            "high": self.high_count,
            "medium": self.medium_count,
            "low": self.low_count,
        }


class SecurityScore(BaseModel):
    """Security score assessment for a Docker image."""

    score: int = Field(..., description="Security score from 0-100 (higher is better)")
    grade: str = Field(..., description="Letter grade (A-F)")
    analysis: str = Field(..., description="Analysis explanation")
    recommendations: List[str] = Field(
        default_factory=list, description="Security recommendations"
    )

    @validator("score")
    def validate_score(cls, v: int) -> int:
        """Validate score is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return v

    @validator("grade")
    def validate_grade(cls, v: str) -> str:
        """Validate grade is a valid letter grade."""
        if v not in ["A", "B", "C", "D", "F"]:
            raise ValueError("Grade must be one of: A, B, C, D, F")
        return v


class LayerInfo(BaseModel):
    """Information about a specific Docker layer."""

    layer_id: str = Field(..., description="Layer ID or hash")
    command: str = Field(..., description="Command that created this layer")
    size_bytes: int = Field(..., description="Layer size in bytes")
    created: Optional[str] = Field(None, description="Layer creation timestamp")
    estimated_size_bytes: Optional[int] = Field(
        None, description="Estimated size for analysis"
    )

    @property
    def size_mb(self) -> float:
        """Get layer size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def size_human(self) -> str:
        """Get human-readable size string."""
        if self.size_bytes < 1024:
            return f"{self.size_bytes}B"
        elif self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f}KB"
        elif self.size_bytes < 1024 * 1024 * 1024:
            return f"{self.size_bytes / (1024 * 1024):.1f}MB"
        else:
            return f"{self.size_bytes / (1024 * 1024 * 1024):.1f}GB"


class ImageAnalysis(BaseModel):
    """Analysis of Docker image layers and sizes."""

    image_name: str = Field(..., description="Docker image name")
    layers: List[LayerInfo] = Field(
        default_factory=list, description="Layer information"
    )
    total_size: int = Field(default=0, description="Total image size in bytes")
    docker_available: bool = Field(
        default=True, description="Whether Docker is available"
    )
    analysis_method: str = Field(
        default="docker_history", description="Method used for analysis"
    )

    @property
    def total_size_mb(self) -> float:
        """Get total size in megabytes."""
        return self.total_size / (1024 * 1024)

    @property
    def layer_count(self) -> int:
        """Get number of layers."""
        return len(self.layers)

    @property
    def largest_layer(self) -> Optional[LayerInfo]:
        """Get the largest layer by size."""
        if not self.layers:
            return None
        return max(self.layers, key=lambda layer: layer.size_bytes)


class OptimizationSuggestion(BaseModel):
    """Represents a real-time optimization suggestion."""

    line_number: int = Field(
        ..., description="Line number in Dockerfile where suggestion applies"
    )
    suggestion_type: str = Field(
        ..., description="Type: security, optimization, best_practice"
    )
    priority: str = Field(
        ..., description="Priority level: LOW, MEDIUM, HIGH, CRITICAL"
    )
    message: str = Field(..., description="Brief suggestion message")
    explanation: str = Field(..., description="Detailed explanation of the suggestion")
    fix_example: str = Field(..., description="Example of how to implement the fix")

    @validator("priority")
    def validate_priority(cls, v: str) -> str:
        """Validate priority is one of the allowed values."""
        allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f'Priority must be one of: {", ".join(allowed)}')
        return v.upper()

    @property
    def type(self) -> str:
        """Backward compatibility property for suggestion_type."""
        return self.suggestion_type

    @property
    def description(self) -> str:
        """Backward compatibility property for message."""
        return self.message

    @property
    def impact(self) -> str:
        """Backward compatibility property for priority."""
        return self.priority.lower()

    @property
    def dockerfile_changes(self) -> List[str]:
        """Backward compatibility property for fix_example as list."""
        return self.fix_example.split('\n') if self.fix_example else []


class SuggestionContext(BaseModel):
    """Context information for generating targeted suggestions."""

    current_line: int = Field(default=0, description="Current line being analyzed")
    has_security_scan: bool = Field(
        default=False, description="Whether security scanning is enabled"
    )
    has_multistage: bool = Field(
        default=False, description="Whether multi-stage build is used"
    )
    project_type: Optional[str] = Field(
        default=None, description="Detected project type"
    )
    previous_suggestions: List[str] = Field(
        default_factory=list, description="Previously shown suggestion types"
    )


class SecurityRule(BaseModel):
    """Represents a custom security rule for Advanced Security Rule Engine."""

    id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(..., description="Detailed description of the rule")
    severity: str = Field(
        ..., description="Severity level: LOW, MEDIUM, HIGH, CRITICAL"
    )
    category: str = Field(..., description="Rule category: security, performance, best_practice")
    rule_type: str = Field(..., description="Rule type: pattern, function, compliance")
    pattern: Optional[str] = Field(None, description="Regex pattern for pattern-type rules")
    function_name: Optional[str] = Field(None, description="Function name for function-type rules")
    message: str = Field(..., description="Violation message to display")
    fix_example: Optional[str] = Field(None, description="Example of how to fix the violation")
    compliance_frameworks: List[str] = Field(
        default_factory=list, description="Compliance frameworks this rule applies to"
    )
    enabled: bool = Field(default=True, description="Whether this rule is enabled")

    @validator("severity")
    def validate_severity(cls, v: str) -> str:
        """Validate severity is one of the allowed values."""
        allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f'Severity must be one of: {", ".join(allowed)}')
        return v.upper()

    @validator("rule_type")
    def validate_rule_type(cls, v: str) -> str:
        """Validate rule type is one of the allowed values."""
        allowed = {"pattern", "function", "compliance"}
        if v.lower() not in allowed:
            raise ValueError(f'Rule type must be one of: {", ".join(allowed)}')
        return v.lower()

    @property
    def is_critical(self) -> bool:
        """Check if this rule has critical severity."""
        return self.severity == "CRITICAL"

    @property
    def is_high_severity(self) -> bool:
        """Check if this rule has high severity."""
        return self.severity == "HIGH"


class SecurityRuleSet(BaseModel):
    """Represents a collection of security rules (policy)."""

    name: str = Field(..., description="Policy name")
    version: str = Field(..., description="Policy version")
    description: str = Field(..., description="Policy description")
    rules: List[SecurityRule] = Field(
        default_factory=list, description="Security rules in this policy"
    )
    compliance_framework: Optional[str] = Field(
        None, description="Compliance framework this policy implements"
    )
    author: Optional[str] = Field(None, description="Policy author")
    created_at: Optional[str] = Field(None, description="Creation timestamp")

    def get_rules_by_severity(self, severity: str) -> List[SecurityRule]:
        """Get rules filtered by severity level."""
        return [rule for rule in self.rules if rule.severity == severity.upper()]

    def get_rules_by_category(self, category: str) -> List[SecurityRule]:
        """Get rules filtered by category."""
        return [rule for rule in self.rules if rule.category == category.lower()]

    def get_enabled_rules(self) -> List[SecurityRule]:
        """Get only enabled rules."""
        return [rule for rule in self.rules if rule.enabled]

    @property
    def rule_count(self) -> int:
        """Get total number of rules."""
        return len(self.rules)

    @property
    def enabled_rule_count(self) -> int:
        """Get number of enabled rules."""
        return len(self.get_enabled_rules())


class ComplianceViolation(BaseModel):
    """Represents a compliance framework violation."""

    framework: str = Field(..., description="Compliance framework name")
    rule_id: str = Field(..., description="Rule identifier that was violated")
    control_id: Optional[str] = Field(None, description="Specific control identifier")
    severity: str = Field(..., description="Violation severity")
    description: str = Field(..., description="Description of the violation")
    requirement: str = Field(..., description="Compliance requirement that was violated")
    remediation: str = Field(..., description="How to remediate this violation")
    risk_level: str = Field(default="MEDIUM", description="Business risk level")

    @validator("framework")
    def validate_framework(cls, v: str) -> str:
        """Validate compliance framework."""
        allowed = {"SOC2", "PCI-DSS", "HIPAA", "GDPR", "ISO27001"}
        if v.upper() not in allowed:
            raise ValueError(f'Framework must be one of: {", ".join(allowed)}')
        return v.upper()


class SecurityRuleEngineResult(BaseModel):
    """Complete result from Advanced Security Rule Engine analysis."""

    violations: List[SecurityFix] = Field(
        default_factory=list, description="Security rule violations found"
    )
    compliance_violations: List[ComplianceViolation] = Field(
        default_factory=list, description="Compliance violations found"
    )
    policies_applied: List[str] = Field(
        default_factory=list, description="Names of policies that were applied"
    )
    rules_evaluated: int = Field(default=0, description="Total number of rules evaluated")
    execution_time_ms: float = Field(default=0.0, description="Analysis execution time in milliseconds")
    security_score: Optional[SecurityScore] = Field(None, description="Overall security score")

    @property
    def total_violations(self) -> int:
        """Get total number of violations."""
        return len(self.violations) + len(self.compliance_violations)

    @property
    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations."""
        return any(v.severity == "CRITICAL" for v in self.violations)

    @property
    def violation_summary(self) -> Dict[str, int]:
        """Get summary of violations by severity."""
        summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for violation in self.violations:
            summary[violation.severity] += 1
        return summary


class RegistryVulnerabilityData(BaseModel):
    """Vulnerability data from a specific Docker registry."""

    registry_type: str = Field(..., description="Type of registry: ECR, ACR, GCR, etc.")
    image_name: str = Field(..., description="Full image name including tag")
    critical_count: int = Field(default=0, description="Number of critical vulnerabilities")
    high_count: int = Field(default=0, description="Number of high severity vulnerabilities")
    medium_count: int = Field(default=0, description="Number of medium severity vulnerabilities")
    low_count: int = Field(default=0, description="Number of low severity vulnerabilities")
    scan_timestamp: Optional[str] = Field(None, description="When the scan was performed")
    registry_url: Optional[str] = Field(None, description="Registry URL where image is hosted")

    @property
    def total_vulnerabilities(self) -> int:
        """Get total number of vulnerabilities."""
        return self.critical_count + self.high_count + self.medium_count + self.low_count

    @property
    def severity_score(self) -> float:
        """Calculate a severity score for comparison (higher is worse)."""
        return (self.critical_count * 10 + self.high_count * 5 +
                self.medium_count * 2 + self.low_count * 1)


class RegistryComparison(BaseModel):
    """Comparison of the same image across multiple registries."""

    image_name: str = Field(..., description="Image being compared")
    registry_data: List[RegistryVulnerabilityData] = Field(
        default_factory=list, description="Vulnerability data from each registry"
    )
    best_registry: Optional[str] = Field(None, description="Registry with fewest vulnerabilities")
    vulnerability_score: float = Field(default=0.0, description="Best available vulnerability score")

    @property
    def registries_compared(self) -> List[str]:
        """Get list of registry types that were compared."""
        return [data.registry_type for data in self.registry_data]

    def get_best_option(self) -> Optional[RegistryVulnerabilityData]:
        """Get the registry option with the lowest vulnerability score."""
        if not self.registry_data:
            return None
        return min(self.registry_data, key=lambda x: x.severity_score)


class RegistryRecommendation(BaseModel):
    """Registry-specific optimization recommendation."""

    registry_type: str = Field(..., description="Target registry type")
    recommendation_type: str = Field(..., description="Type: base_image, security, optimization")
    title: str = Field(..., description="Brief recommendation title")
    description: str = Field(..., description="Detailed recommendation description")
    dockerfile_change: str = Field(..., description="Suggested Dockerfile modification")
    security_benefit: str = Field(..., description="Security benefits of this change")
    estimated_impact: str = Field(..., description="LOW, MEDIUM, HIGH impact estimation")

    @validator("registry_type")
    def validate_registry_type(cls, v: str) -> str:
        """Validate registry type is supported."""
        allowed = {"ECR", "ACR", "GCR", "DOCKERHUB"}
        if v.upper() not in allowed:
            raise ValueError(f'Registry type must be one of: {", ".join(allowed)}')
        return v.upper()

    @validator("estimated_impact")
    def validate_impact(cls, v: str) -> str:
        """Validate impact level."""
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if v.upper() not in allowed:
            raise ValueError(f'Impact must be one of: {", ".join(allowed)}')
        return v.upper()


class OptimizationStep(BaseModel):
    """Represents a single optimization step in a preset."""

    name: str = Field(..., description="Name of the optimization step")
    description: str = Field(..., description="Description of what this step does")
    dockerfile_change: str = Field(..., description="Dockerfile modification to apply")
    reasoning: str = Field(..., description="Why this optimization is beneficial")
    priority: int = Field(default=1, description="Priority level (1=highest, 5=lowest)")

    @validator("priority")
    def validate_priority(cls, v: int) -> int:
        """Validate priority is between 1 and 5."""
        if not 1 <= v <= 5:
            raise ValueError("Priority must be between 1 and 5")
        return v


class OptimizationPreset(BaseModel):
    """Represents a complete optimization preset."""

    name: str = Field(..., description="Name of the preset")
    description: str = Field(..., description="Description of the preset")
    preset_type: str = Field(..., description="Type of preset")
    optimizations: List[OptimizationStep] = Field(
        default_factory=list, description="List of optimization steps"
    )
    target_use_case: str = Field(..., description="Target use case for this preset")
    estimated_size_reduction: str = Field(default="Unknown", description="Estimated size reduction")
    security_level: str = Field(default="Standard", description="Security level provided")

    @property
    def total_steps(self) -> int:
        """Get total number of optimization steps."""
        return len(self.optimizations)

    @property
    def high_priority_steps(self) -> List[OptimizationStep]:
        """Get high priority optimization steps (priority 1-2)."""
        return [opt for opt in self.optimizations if opt.priority <= 2]


class CustomPreset(BaseModel):
    """Represents a user-defined custom optimization preset."""

    name: str = Field(..., description="Name of the custom preset")
    description: str = Field(..., description="Description of the custom preset")
    base_preset: Optional[str] = Field(None, description="Base preset this is derived from")
    additional_optimizations: List[str] = Field(
        default_factory=list, description="Additional optimization descriptions"
    )
    disabled_optimizations: List[str] = Field(
        default_factory=list, description="Optimizations to disable from base preset"
    )
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    author: Optional[str] = Field(None, description="Author of the custom preset")

    @property
    def modification_count(self) -> int:
        """Get total number of modifications from base preset."""
        return len(self.additional_optimizations) + len(self.disabled_optimizations)
