"""Test cases for CLI interface."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from docker_optimizer.cli import main


class TestCLI:
    """Test cases for CLI interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_basic_optimization(self):
        """Test basic CLI optimization."""
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dockerfile", delete=False
        ) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, ["--dockerfile", dockerfile_path])

            assert result.exit_code == 0
            assert "Optimization Results" in result.output
            assert "Optimized Dockerfile:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analysis_only(self):
        """Test CLI analysis-only mode."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dockerfile", delete=False
        ) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(
                main, ["--dockerfile", dockerfile_path, "--analysis-only"]
            )

            assert result.exit_code == 0
            assert "Dockerfile Analysis Results" in result.output
            assert "Security Issues Found" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_json_output(self):
        """Test CLI JSON output format."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache curl
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dockerfile", delete=False
        ) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(
                main, ["--dockerfile", dockerfile_path, "--format", "json"]
            )

            assert result.exit_code == 0
            # Should be valid JSON
            output_data = json.loads(result.output)
            assert "optimized_dockerfile" in output_data
            assert "explanation" in output_data
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_output_to_file(self):
        """Test CLI output to file."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache curl
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dockerfile", delete=False
        ) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".optimized", delete=False
        ) as outf:
            output_path = outf.name

        try:
            result = self.runner.invoke(
                main, ["--dockerfile", dockerfile_path, "--output", output_path]
            )

            assert result.exit_code == 0
            assert f"written to {output_path}" in result.output

            # Check output file exists and has content
            output_content = Path(output_path).read_text()
            assert "Optimization Results" in output_content
        finally:
            Path(dockerfile_path).unlink()
            Path(output_path).unlink()

    def test_cli_verbose_mode(self):
        """Test CLI verbose mode."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dockerfile", delete=False
        ) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(
                main, ["--dockerfile", dockerfile_path, "--verbose"]
            )

            assert result.exit_code == 0
            # Verbose mode should show more details
            assert len(result.output) > 200  # Should be fairly detailed
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_nonexistent_file(self):
        """Test CLI with nonexistent Dockerfile."""
        result = self.runner.invoke(main, ["--dockerfile", "/nonexistent/Dockerfile"])

        assert result.exit_code == 2  # Click uses exit code 2 for file not found
        assert "not found" in result.output or "does not exist" in result.output

    def test_cli_default_dockerfile(self):
        """Test CLI with default Dockerfile name."""
        # This should fail since no Dockerfile exists in the test directory
        result = self.runner.invoke(main, [])

        assert result.exit_code == 2  # Click uses exit code 2 for file not found
        assert "not found" in result.output or "does not exist" in result.output

    def test_cli_yaml_output(self):
        """Test CLI YAML output format."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache curl
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dockerfile", delete=False
        ) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(
                main, ["--dockerfile", dockerfile_path, "--format", "yaml"]
            )

            assert result.exit_code == 0
            # Should contain YAML-like output
            assert "optimized_dockerfile:" in result.output
            assert "explanation:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_help(self):
        """Test CLI help output."""
        result = self.runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Docker Optimizer Agent" in result.output
        assert "--dockerfile" in result.output
        assert "--output" in result.output
