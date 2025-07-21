"""Tests for Docker layer analysis functionality."""

from unittest.mock import patch

from src.docker_optimizer.layer_analyzer import DockerLayerAnalyzer
from src.docker_optimizer.models import ImageAnalysis, LayerInfo


class TestDockerLayerAnalyzer:
    """Test Docker layer analysis capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DockerLayerAnalyzer()

    def test_analyze_image_layers_success(self):
        """Test successful Docker image layer analysis."""
        # This should fail initially since DockerLayerAnalyzer doesn't exist yet
        mock_history_output = """
IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
abc123def456        2 hours ago         /bin/sh -c apt-get update && apt-get install   45.2MB
def456ghi789        2 hours ago         /bin/sh -c #(nop) COPY . /app                   1.23MB
ghi789jkl012        2 hours ago         /bin/sh -c #(nop) WORKDIR /app                  0B
jkl012mno345        3 hours ago         /bin/sh -c #(nop)  CMD ["python3" "app.py"]    0B
"""

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = mock_history_output
            mock_run.return_value.returncode = 0

            result = self.analyzer.analyze_image_layers("test-image:latest")

            assert isinstance(result, ImageAnalysis)
            assert len(result.layers) == 4
            assert result.total_size > 0

            # Verify layer details
            first_layer = result.layers[0]
            assert isinstance(first_layer, LayerInfo)
            assert abs(first_layer.size_bytes - 45.2 * 1024 * 1024) < 1024  # Allow small precision difference
            assert "apt-get update" in first_layer.command

    def test_analyze_image_layers_docker_not_available(self):
        """Test graceful handling when Docker is not available."""
        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = self.analyzer.analyze_image_layers("test-image:latest")

            assert isinstance(result, ImageAnalysis)
            assert len(result.layers) == 0
            assert result.total_size == 0
            assert not result.docker_available

    def test_get_layer_sizes_for_dockerfile(self):
        """Test getting layer sizes for a Dockerfile without building."""
        dockerfile_content = """
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
"""

        # Should estimate sizes based on command analysis
        result = self.analyzer.get_layer_sizes_for_dockerfile(dockerfile_content)

        assert isinstance(result, ImageAnalysis)
        assert len(result.layers) >= 4  # FROM, RUN, COPY, WORKDIR, CMD

        # RUN command with package installation should have larger estimated size
        run_layer = next((layer for layer in result.layers if "apt-get" in layer.command), None)
        assert run_layer is not None
        assert run_layer.estimated_size_bytes > 1024 * 1024  # Should be > 1MB

    def test_compare_dockerfile_efficiency(self):
        """Test comparing efficiency between original and optimized Dockerfiles."""
        original_dockerfile = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
"""

        optimized_dockerfile = """
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 pip && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
"""

        comparison = self.analyzer.compare_dockerfile_efficiency(
            original_dockerfile, optimized_dockerfile
        )

        assert comparison["original_layers"] > comparison["optimized_layers"]
        assert comparison["layer_reduction"] > 0
        assert comparison["estimated_size_reduction"] > 0
