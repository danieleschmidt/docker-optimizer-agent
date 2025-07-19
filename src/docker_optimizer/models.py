"""Data models for Docker optimization results."""

from typing import List, Optional

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
