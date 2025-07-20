"""Test cases for the main DockerfileOptimizer class."""

from unittest.mock import patch

import pytest

from docker_optimizer.models import DockerfileAnalysis, OptimizationResult
from docker_optimizer.optimizer import DockerfileOptimizer


class TestDockerfileOptimizer:
    """Test cases for DockerfileOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = DockerfileOptimizer()

    def test_optimizer_initialization(self):
        """Test that optimizer initializes correctly."""
        assert isinstance(self.optimizer, DockerfileOptimizer)

    def test_analyze_dockerfile_basic(self):
        """Test basic Dockerfile analysis."""
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl
COPY . /app
WORKDIR /app
"""

        analysis = self.optimizer.analyze_dockerfile(dockerfile_content)

        assert isinstance(analysis, DockerfileAnalysis)
        assert analysis.base_image == "ubuntu:latest"
        assert analysis.total_layers > 0
        assert (
            analysis.has_security_issues
        )  # ubuntu:latest should trigger security warning

    def test_analyze_dockerfile_with_security_issues(self):
        """Test that security issues are properly identified."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
"""

        analysis = self.optimizer.analyze_dockerfile(dockerfile_content)

        assert analysis.has_security_issues
        assert any("latest" in issue.lower() for issue in analysis.security_issues)
        assert any("root" in issue.lower() for issue in analysis.security_issues)

    def test_analyze_dockerfile_with_optimization_opportunities(self):
        """Test that optimization opportunities are identified."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
"""

        analysis = self.optimizer.analyze_dockerfile(dockerfile_content)

        assert analysis.has_optimization_opportunities
        assert any(
            "layer" in opp.lower() for opp in analysis.optimization_opportunities
        )

    def test_optimize_dockerfile_basic(self):
        """Test basic Dockerfile optimization."""
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl
"""

        result = self.optimizer.optimize_dockerfile(dockerfile_content)

        assert isinstance(result, OptimizationResult)
        # Verify optimization occurred (either size change or layer optimizations or security fixes)
        assert (result.original_size != result.optimized_size or
                len(result.layer_optimizations) > 0 or
                len(result.security_fixes) > 0)
        # Verify optimization improvements were made
        assert "ubuntu:22.04" in result.optimized_dockerfile  # Latest tag was fixed
        assert "--no-install-recommends" in result.optimized_dockerfile  # Package optimization
        assert len(result.security_fixes) > 0  # Security improvements were made

    def test_optimize_dockerfile_with_security_fixes(self):
        """Test that security fixes are applied during optimization."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
"""

        result = self.optimizer.optimize_dockerfile(dockerfile_content)

        assert result.has_security_improvements
        assert "USER" in result.optimized_dockerfile
        assert (
            "root" not in result.optimized_dockerfile
            or "USER" in result.optimized_dockerfile
        )

    def test_optimize_dockerfile_with_layer_optimizations(self):
        """Test that layer optimizations are applied."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
"""

        result = self.optimizer.optimize_dockerfile(dockerfile_content)

        assert result.has_layer_optimizations
        # Should combine RUN statements
        run_count = result.optimized_dockerfile.count("RUN")
        original_run_count = dockerfile_content.count("RUN")
        assert run_count < original_run_count

    def test_analyze_and_optimize_integration(self):
        """Test the complete analyze and optimize workflow."""
        dockerfile_path = "/tmp/test_dockerfile"
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl vim
COPY . /app
WORKDIR /app
EXPOSE 8080
"""

        # Mock file reading
        with patch("builtins.open", mock_open(read_data=dockerfile_content)):
            result = self.optimizer.analyze_and_optimize(dockerfile_path)

        assert isinstance(result, OptimizationResult)
        assert result.explanation
        assert result.optimized_dockerfile != dockerfile_content

    def test_get_size_estimation(self):
        """Test size estimation functionality."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache curl
"""

        estimated_size = self.optimizer._estimate_size(dockerfile_content)

        assert estimated_size is not None
        assert "MB" in estimated_size or "GB" in estimated_size

    @pytest.mark.parametrize(
        "base_image,expected_type",
        [
            ("ubuntu:latest", "security"),
            ("alpine:latest", "security"),
            ("node:18-alpine", "optimization"),
            ("python:3.9-slim", "optimization"),
        ],
    )
    def test_identify_issues_by_base_image(self, base_image, expected_type):
        """Test that different base images trigger appropriate issue identification."""
        dockerfile_content = f"FROM {base_image}\nRUN echo hello"

        analysis = self.optimizer.analyze_dockerfile(dockerfile_content)

        if expected_type == "security":
            assert analysis.has_security_issues
        else:
            # Should have fewer security issues for optimized base images
            assert len(analysis.security_issues) <= 1


def mock_open(read_data):
    """Helper function to mock file opening."""
    from unittest.mock import mock_open as _mock_open

    return _mock_open(read_data=read_data)
