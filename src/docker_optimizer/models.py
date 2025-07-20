"""Data models for Docker optimization results."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


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
    commands: List[str] = Field(default_factory=list, description="Commands in this stage")
    purpose: str = Field(..., description="Purpose: 'build', 'runtime', or 'intermediate'")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies installed")

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

    recommended: bool = Field(..., description="Whether multi-stage build is recommended")
    has_build_dependencies: bool = Field(..., description="Whether build dependencies were found")
    build_dependencies: List[str] = Field(default_factory=list, description="Build-only dependencies")
    runtime_dependencies: List[str] = Field(default_factory=list, description="Runtime dependencies")
    benefits: List[str] = Field(default_factory=list, description="Expected benefits")
    estimated_size_reduction: str = Field(..., description="Estimated size reduction")
    complexity_score: int = Field(..., description="Implementation complexity (1-10)")


class MultiStageOptimization(BaseModel):
    """Result of multi-stage build optimization."""

    original_dockerfile: str = Field(..., description="Original Dockerfile content")
    optimized_dockerfile: str = Field(..., description="Optimized multi-stage Dockerfile")
    stages: List[BuildStage] = Field(default_factory=list, description="Build stages created")
    estimated_size_reduction: int = Field(..., description="Estimated size reduction in MB")
    security_improvements: int = Field(default=0, description="Number of security improvements")
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
    fixed_version: Optional[str] = Field(None, description="Version that fixes the vulnerability")
    description: str = Field(..., description="Vulnerability description")

    @property
    def has_fix(self) -> bool:
        """Check if a fix is available."""
        return self.fixed_version is not None


class VulnerabilityReport(BaseModel):
    """Report of vulnerabilities found in a Docker image or Dockerfile."""

    total_vulnerabilities: int = Field(..., description="Total number of vulnerabilities")
    critical_count: int = Field(default=0, description="Number of critical vulnerabilities")
    high_count: int = Field(default=0, description="Number of high severity vulnerabilities")
    medium_count: int = Field(default=0, description="Number of medium severity vulnerabilities")
    low_count: int = Field(default=0, description="Number of low severity vulnerabilities")
    cve_details: List[CVEDetails] = Field(default_factory=list, description="Detailed CVE information")

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
            "low": self.low_count
        }


class SecurityScore(BaseModel):
    """Security score assessment for a Docker image."""

    score: int = Field(..., description="Security score from 0-100 (higher is better)")
    grade: str = Field(..., description="Letter grade (A-F)")
    analysis: str = Field(..., description="Analysis explanation")
    recommendations: List[str] = Field(default_factory=list, description="Security recommendations")

    @validator('score')
    def validate_score(cls, v: int) -> int:
        """Validate score is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v

    @validator('grade')
    def validate_grade(cls, v: str) -> str:
        """Validate grade is a valid letter grade."""
        if v not in ['A', 'B', 'C', 'D', 'F']:
            raise ValueError('Grade must be one of: A, B, C, D, F')
        return v


class LayerInfo(BaseModel):
    """Information about a specific Docker layer."""

    layer_id: str = Field(..., description="Layer ID or hash")
    command: str = Field(..., description="Command that created this layer")
    size_bytes: int = Field(..., description="Layer size in bytes")
    created: Optional[str] = Field(None, description="Layer creation timestamp")
    estimated_size_bytes: Optional[int] = Field(None, description="Estimated size for analysis")

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
    layers: List[LayerInfo] = Field(default_factory=list, description="Layer information")
    total_size: int = Field(default=0, description="Total image size in bytes")
    docker_available: bool = Field(default=True, description="Whether Docker is available")
    analysis_method: str = Field(default="docker_history", description="Method used for analysis")

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
