"""Test cases for CLI registry integration."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from docker_optimizer.cli import main


class TestCLIRegistryIntegration:
    """Test CLI integration for registry vulnerability scanning and comparison."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_registry_scan_flag_ecr(self):
        """Test CLI --registry-scan flag with ECR registry."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the new --registry-scan flag
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--registry-scan", "ECR",
                "--registry-image", "my-app:latest",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            
            # Should include registry vulnerability data
            assert "registry_vulnerabilities" in output_data
            assert output_data["registry_vulnerabilities"]["registry_type"] == "ECR"
            assert output_data["registry_vulnerabilities"]["image_name"] == "my-app:latest"
            assert "critical_count" in output_data["registry_vulnerabilities"]
            
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_registry_scan_flag_acr(self):
        """Test CLI --registry-scan flag with Azure ACR registry."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y nodejs npm
COPY . /app
WORKDIR /app
CMD ["node", "server.js"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the new --registry-scan flag with ACR
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--registry-scan", "ACR",
                "--registry-image", "myregistry.azurecr.io/myapp:v1.0",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            
            # Should include registry vulnerability data
            assert "registry_vulnerabilities" in output_data
            assert output_data["registry_vulnerabilities"]["registry_type"] == "ACR"
            assert "myregistry.azurecr.io/myapp:v1.0" in output_data["registry_vulnerabilities"]["image_name"]
            
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_registry_scan_flag_gcr(self):
        """Test CLI --registry-scan flag with Google GCR registry."""
        dockerfile_content = """FROM golang:latest
COPY . /app
WORKDIR /app
RUN go build -o main .
CMD ["./main"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the new --registry-scan flag with GCR
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--registry-scan", "GCR",
                "--registry-image", "gcr.io/my-project/my-app:latest",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            
            # Should include registry vulnerability data
            assert "registry_vulnerabilities" in output_data
            assert output_data["registry_vulnerabilities"]["registry_type"] == "GCR"
            assert "gcr.io/my-project/my-app:latest" in output_data["registry_vulnerabilities"]["image_name"]
            
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_registry_compare_flag(self):
        """Test CLI --registry-compare flag to compare across registries."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the new --registry-compare flag
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--registry-compare",
                "--registry-images", "my-app:latest",
                "--registry-images", "myregistry.azurecr.io/my-app:latest", 
                "--registry-images", "gcr.io/my-project/my-app:latest",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            
            # Should include registry comparison data
            assert "registry_comparison" in output_data
            assert len(output_data["registry_comparison"]["comparisons"]) >= 3
            assert "recommendations" in output_data["registry_comparison"]
            
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_registry_scan_without_image_flag(self):
        """Test that registry scan fails gracefully without image specified."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update
COPY . /app
WORKDIR /app"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test registry scan without specifying image
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--registry-scan", "ECR",
                "--format", "json"
            ])

            # Should fail with proper error message
            assert result.exit_code != 0
            assert "registry-image" in result.output or "Error" in result.output
            
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_registry_scan_text_format(self):
        """Test CLI --registry-scan flag with text output format."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test registry scan with text format
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--registry-scan", "ECR",
                "--registry-image", "my-app:latest",
                "--format", "text"
            ])

            assert result.exit_code == 0
            
            # Should include registry analysis in text output
            assert "Registry Vulnerability Analysis:" in result.output
            assert "Registry: ECR" in result.output
            assert "Image: my-app:latest" in result.output
            assert "Vulnerability Summary:" in result.output
            
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_registry_recommendations(self):
        """Test CLI registry-specific optimization recommendations."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test registry recommendations
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--registry-scan", "ECR",
                "--registry-image", "my-app:latest",
                "--registry-recommendations",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)
            
            # Should include registry-specific recommendations
            assert "registry_recommendations" in output_data
            recommendations = output_data["registry_recommendations"]
            assert len(recommendations) > 0
            
            # Check recommendation structure
            for rec in recommendations:
                assert "type" in rec
                assert "priority" in rec
                assert "description" in rec
                assert "registry_specific" in rec
            
        finally:
            Path(dockerfile_path).unlink()