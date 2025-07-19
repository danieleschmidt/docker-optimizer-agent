"""Test cases for optimization result models."""

from docker_optimizer.models import LayerOptimization, OptimizationResult, SecurityFix


class TestOptimizationResult:
    """Test cases for OptimizationResult model."""

    def test_create_optimization_result(self):
        """Test creating a basic optimization result."""
        result = OptimizationResult(
            original_size="500MB",
            optimized_size="200MB",
            security_fixes=[],
            explanation="Basic optimization applied",
            optimized_dockerfile="FROM alpine:3.18\nRUN echo hello",
        )

        assert result.original_size == "500MB"
        assert result.optimized_size == "200MB"
        assert result.explanation == "Basic optimization applied"
        assert len(result.security_fixes) == 0

    def test_optimization_result_with_security_fixes(self):
        """Test optimization result with security improvements."""
        security_fix = SecurityFix(
            vulnerability="CVE-2023-1234",
            severity="HIGH",
            description="Outdated base image",
            fix="Updated to alpine:3.18",
        )

        result = OptimizationResult(
            original_size="500MB",
            optimized_size="200MB",
            security_fixes=[security_fix],
            explanation="Security and size optimization",
            optimized_dockerfile="FROM alpine:3.18\nRUN echo hello",
        )

        assert len(result.security_fixes) == 1
        assert result.security_fixes[0].vulnerability == "CVE-2023-1234"
        assert result.security_fixes[0].severity == "HIGH"

    def test_optimization_result_with_layer_optimizations(self):
        """Test optimization result with layer improvements."""
        layer_opt = LayerOptimization(
            original_instruction="RUN apt-get update && apt-get install -y curl",
            optimized_instruction="RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*",
            reasoning="Added cleanup and no-install-recommends for smaller layer",
        )

        result = OptimizationResult(
            original_size="500MB",
            optimized_size="200MB",
            security_fixes=[],
            explanation="Layer optimization applied",
            optimized_dockerfile="FROM alpine:3.18\nRUN echo hello",
            layer_optimizations=[layer_opt],
        )

        assert len(result.layer_optimizations) == 1
        assert (
            "no-install-recommends"
            in result.layer_optimizations[0].optimized_instruction
        )


class TestSecurityFix:
    """Test cases for SecurityFix model."""

    def test_create_security_fix(self):
        """Test creating a security fix."""
        fix = SecurityFix(
            vulnerability="CVE-2023-1234",
            severity="CRITICAL",
            description="Root user detected",
            fix="Added USER directive with non-root user",
        )

        assert fix.vulnerability == "CVE-2023-1234"
        assert fix.severity == "CRITICAL"
        assert "non-root" in fix.fix

    def test_security_fix_severity_validation(self):
        """Test that security fix validates severity levels."""
        # Valid severities should work
        valid_severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for severity in valid_severities:
            fix = SecurityFix(
                vulnerability="CVE-2023-1234",
                severity=severity,
                description="Test vulnerability",
                fix="Test fix",
            )
            assert fix.severity == severity


class TestLayerOptimization:
    """Test cases for LayerOptimization model."""

    def test_create_layer_optimization(self):
        """Test creating a layer optimization."""
        optimization = LayerOptimization(
            original_instruction="RUN apt-get update",
            optimized_instruction="RUN apt-get update && rm -rf /var/lib/apt/lists/*",
            reasoning="Added cleanup to reduce layer size",
        )

        assert "apt-get update" in optimization.original_instruction
        assert "rm -rf /var/lib/apt/lists/*" in optimization.optimized_instruction
        assert "reduce layer size" in optimization.reasoning
