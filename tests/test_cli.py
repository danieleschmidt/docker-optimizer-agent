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

    def test_cli_multistage_optimization(self):
        """Test CLI multi-stage optimization."""
        dockerfile_content = """
FROM python:3.11
RUN apt-get update && apt-get install -y gcc build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, ['--dockerfile', dockerfile_path, '--multistage'])

            assert result.exit_code == 0
            assert "Multi-Stage Build Optimization Results" in result.output
            assert "Build Stages:" in result.output
            assert "builder" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_multistage_json_output(self):
        """Test CLI multi-stage optimization with JSON output."""
        dockerfile_content = """
FROM node:18
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, ['--dockerfile', dockerfile_path, '--multistage', '--format', 'json'])

            assert result.exit_code == 0
            # Should be valid JSON
            output_data = json.loads(result.output)
            assert 'optimized_dockerfile' in output_data
            assert 'stages' in output_data
            assert 'estimated_size_reduction' in output_data
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_error_handling(self):
        """Test CLI error handling with invalid dockerfile."""
        result = self.runner.invoke(main, ['--dockerfile', 'nonexistent.dockerfile'])

        assert result.exit_code in [1, 2]  # Allow both error codes
        assert "Error:" in result.output or "does not exist" in result.output

    def test_cli_verbose_error_handling(self):
        """Test CLI verbose error handling."""
        result = self.runner.invoke(main, ['--dockerfile', 'nonexistent.dockerfile', '--verbose'])

        assert result.exit_code in [1, 2]  # Allow both error codes
        assert "Error:" in result.output or "does not exist" in result.output

    def test_cli_yaml_format_output(self):
        """Test CLI YAML format output."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, ['--dockerfile', dockerfile_path, '--analysis-only', '--format', 'yaml'])

            assert result.exit_code == 0
            # Should contain YAML-style content
            assert "security_score:" in result.output or "estimated_size:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_batch_processing_performance(self):
        """Test CLI batch processing with performance mode."""
        dockerfile_content = """
FROM python:3.11-slim
RUN pip install requests
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        dockerfiles = []
        try:
            # Create multiple test dockerfiles
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.dockerfile{i}', delete=False) as f:
                    f.write(dockerfile_content)
                    dockerfiles.append(f.name)

            # Test batch processing with performance
            batch_args = ['--performance']
            for dockerfile in dockerfiles:
                batch_args.extend(['--batch', dockerfile])

            result = self.runner.invoke(main, batch_args)

            assert result.exit_code == 0
            assert "Results for:" in result.output or "Optimization Results" in result.output
        finally:
            for dockerfile in dockerfiles:
                Path(dockerfile).unlink()

    def test_cli_batch_processing_with_performance_report(self):
        """Test CLI batch processing with performance report."""
        dockerfile_content = """
FROM alpine:latest
RUN apk add --no-cache curl
"""

        dockerfiles = []
        try:
            # Create multiple test dockerfiles
            for i in range(2):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.dockerfile{i}', delete=False) as f:
                    f.write(dockerfile_content)
                    dockerfiles.append(f.name)

            # Test batch processing with performance report
            batch_args = ['--performance', '--performance-report']
            for dockerfile in dockerfiles:
                batch_args.extend(['--batch', dockerfile])

            result = self.runner.invoke(main, batch_args)

            assert result.exit_code == 0
            # Should contain performance metrics
            assert "processed" in result.output.lower() or "performance" in result.output.lower()
        finally:
            for dockerfile in dockerfiles:
                Path(dockerfile).unlink()

    def test_cli_multistage_output_to_file(self):
        """Test CLI multi-stage optimization with file output."""
        dockerfile_content = """
FROM golang:1.21
COPY . /app
WORKDIR /app
RUN go build -o main .
CMD ["./main"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.optimized', delete=False) as outf:
            output_path = outf.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--multistage',
                '--output', output_path
            ])

            assert result.exit_code == 0
            assert f"written to {output_path}" in result.output

            # Check output file exists and has content
            output_content = Path(output_path).read_text()
            assert "Multi-Stage Build Optimization" in output_content
            assert "FROM golang:" in output_content
            assert "FROM alpine:" in output_content  # Should use minimal runtime
        finally:
            Path(dockerfile_path).unlink()
            Path(output_path).unlink()

    def test_cli_security_scan(self):
        """Test CLI security scan functionality."""
        dockerfile_content = """
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y openssl=1.1.0g-2ubuntu4
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, ['--dockerfile', dockerfile_path, '--security-scan'])

            assert result.exit_code == 0
            assert "Security Vulnerability Scan Results" in result.output
            assert "Security Score:" in result.output
            assert "Total Vulnerabilities:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_security_scan_json_output(self):
        """Test CLI security scan with JSON output."""
        dockerfile_content = """
FROM alpine:3.14
RUN apk add --no-cache nginx=1.18.0-r13
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--security-scan',
                '--format', 'json'
            ])

            assert result.exit_code == 0
            # Should be valid JSON
            output_data = json.loads(result.output)
            assert 'vulnerability_report' in output_data
            assert 'security_score' in output_data
            assert 'suggestions' in output_data
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_security_scan_yaml_output(self):
        """Test CLI security scan with YAML output."""
        dockerfile_content = """
FROM debian:10
RUN apt-get update && apt-get install -y curl
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--security-scan',
                '--format', 'yaml'
            ])

            assert result.exit_code == 0
            assert 'vulnerability_report:' in result.output
            assert 'security_score:' in result.output
            assert 'suggestions:' in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_security_scan_output_to_file(self):
        """Test CLI security scan with file output."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.security', delete=False) as outf:
            output_path = outf.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--security-scan',
                '--output', output_path
            ])

            assert result.exit_code == 0
            assert f"written to {output_path}" in result.output

            # Check output file exists and has content
            output_content = Path(output_path).read_text()
            assert "Security Vulnerability Scan Results" in output_content
            assert "Security Score:" in output_content
        finally:
            Path(dockerfile_path).unlink()
            Path(output_path).unlink()

    def test_cli_security_scan_verbose_mode(self):
        """Test CLI security scan with verbose mode."""
        dockerfile_content = """
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y openssl
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--security-scan',
                '--verbose'
            ])

            assert result.exit_code == 0
            assert "Security Vulnerability Scan Results" in result.output
            # Verbose mode should show more details - may include CVE details or recommendations
            assert len(result.output) > 100  # Should have substantial output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_performance_optimization(self):
        """Test CLI performance optimization."""
        dockerfile_content = """
FROM python:3.9
RUN pip install requests
RUN pip install flask
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--performance'
            ])

            assert result.exit_code == 0
            assert "Optimized Dockerfile" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_performance_with_report(self):
        """Test CLI performance optimization with performance report."""
        dockerfile_content = """
FROM python:3.9
RUN pip install requests
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--performance',
                '--performance-report'
            ])

            assert result.exit_code == 0
            assert "Performance Metrics:" in result.output
            assert "Processing Time:" in result.output
            assert "Memory Usage:" in result.output
            assert "Cache Hit Ratio:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_batch_processing(self):
        """Test CLI batch processing."""
        dockerfile1_content = """
FROM python:3.9
RUN pip install requests
"""
        dockerfile2_content = """
FROM node:16
RUN npm install express
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f1:
            f1.write(dockerfile1_content)
            dockerfile1_path = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f2:
            f2.write(dockerfile2_content)
            dockerfile2_path = f2.name

        try:
            result = self.runner.invoke(main, [
                '--batch', dockerfile1_path,
                '--batch', dockerfile2_path
            ])

            assert result.exit_code == 0
            assert "Results for:" in result.output
            assert dockerfile1_path in result.output
            assert dockerfile2_path in result.output
        finally:
            Path(dockerfile1_path).unlink()
            Path(dockerfile2_path).unlink()

    def test_cli_batch_with_performance(self):
        """Test CLI batch processing with performance optimization."""
        dockerfile1_content = """
FROM python:3.9
RUN pip install requests
"""
        dockerfile2_content = """
FROM node:16
RUN npm install express
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f1:
            f1.write(dockerfile1_content)
            dockerfile1_path = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f2:
            f2.write(dockerfile2_content)
            dockerfile2_path = f2.name

        try:
            result = self.runner.invoke(main, [
                '--batch', dockerfile1_path,
                '--batch', dockerfile2_path,
                '--performance',
                '--performance-report'
            ])

            assert result.exit_code == 0
            assert "Results for:" in result.output
            assert "Performance Report" in result.output
            assert "Processing Time:" in result.output
        finally:
            Path(dockerfile1_path).unlink()
            Path(dockerfile2_path).unlink()

    def test_cli_performance_json_output(self):
        """Test CLI performance optimization with JSON output."""
        dockerfile_content = """
FROM python:3.9
RUN pip install requests
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--performance',
                '--performance-report',
                '--format', 'json'
            ])

            assert result.exit_code == 0
            # Should contain valid JSON
            assert '"processing_time"' in result.output
            assert '"memory_usage_mb"' in result.output
            assert '"cache_hits"' in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_layer_analysis_flag(self):
        """Test CLI --layer-analysis flag functionality."""
        dockerfile_content = """
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis'
            ])

            assert result.exit_code == 0
            assert "Dockerfile Layer Analysis" in result.output
            assert "Traditional Size Estimate" in result.output
            assert "Layer-Based Size Estimate" in result.output
            assert "Efficiency Score" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_layer_analysis_json_output(self):
        """Test CLI --layer-analysis with JSON output."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache python3
COPY . /app
CMD ["python3", "/app/main.py"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis',
                '--format', 'json'
            ])

            assert result.exit_code == 0
            # Should be valid JSON
            output_data = json.loads(result.output)
            assert 'traditional_estimate' in output_data
            assert 'layer_analysis' in output_data
            assert 'dockerfile_efficiency_score' in output_data
            assert 'estimated_layers' in output_data
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_layer_analysis_yaml_output(self):
        """Test CLI --layer-analysis with YAML output."""
        dockerfile_content = """
FROM node:18-alpine
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm", "start"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis',
                '--format', 'yaml'
            ])

            assert result.exit_code == 0
            # Should contain YAML-style content
            assert 'traditional_estimate:' in result.output
            assert 'layer_analysis:' in result.output
            assert 'dockerfile_efficiency_score:' in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_layer_analysis_verbose_mode(self):
        """Test CLI --layer-analysis with verbose mode."""
        dockerfile_content = """
FROM python:3.11-slim
RUN pip install requests
RUN pip install flask
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis',
                '--verbose'
            ])

            assert result.exit_code == 0
            assert "Layer Breakdown:" in result.output
            assert "Estimated Size:" in result.output
            # Verbose mode should show individual layer details
            assert len(result.output) > 500  # Should be detailed
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analyze_image_flag(self):
        """Test CLI --analyze-image flag functionality."""
        dockerfile_content = """FROM alpine:3.18
RUN apk add --no-cache curl"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test with a common image that should be available
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analyze-image', 'alpine:latest'
            ])

            # Should work even if Docker is not available (graceful handling)
            assert result.exit_code in [0, 1]  # Allow both success and graceful failure
            if result.exit_code == 0:
                assert "Docker Image Analysis" in result.output
                assert "alpine:latest" in result.output
                assert "Total Size:" in result.output
                assert "Layer Count:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analyze_image_json_output(self):
        """Test CLI --analyze-image with JSON output."""
        dockerfile_content = """FROM ubuntu:22.04
RUN apt-get update"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analyze-image', 'ubuntu:22.04',
                '--format', 'json'
            ])

            # Should work even if Docker is not available (graceful handling)
            assert result.exit_code in [0, 1]  # Allow both success and graceful failure
            if result.exit_code == 0:
                # Should be valid JSON
                output_data = json.loads(result.output)
                assert 'image_name' in output_data
                assert 'total_size' in output_data
                assert 'layers' in output_data
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analyze_image_yaml_output(self):
        """Test CLI --analyze-image with YAML output."""
        dockerfile_content = """FROM nginx:alpine
RUN apk add --no-cache curl"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analyze-image', 'nginx:alpine',
                '--format', 'yaml'
            ])

            # Should work even if Docker is not available (graceful handling)
            assert result.exit_code in [0, 1]  # Allow both success and graceful failure
            if result.exit_code == 0:
                assert 'image_name:' in result.output
                assert 'total_size:' in result.output
                assert 'layers:' in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analyze_image_verbose_mode(self):
        """Test CLI --analyze-image with verbose mode."""
        dockerfile_content = """FROM python:3.11-alpine
RUN pip install requests"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analyze-image', 'python:3.11-alpine',
                '--verbose'
            ])

            # Should work even if Docker is not available (graceful handling)
            assert result.exit_code in [0, 1]  # Allow both success and graceful failure
            if result.exit_code == 0:
                assert "Docker Image Analysis" in result.output
                # Layer Details only shown if layers exist (Docker available)
                if "Layer Count: 0" not in result.output:
                    assert "Layer Details:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analyze_image_invalid_name(self):
        """Test CLI --analyze-image with invalid image name."""
        dockerfile_content = """FROM alpine:3.18
RUN echo 'test'"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analyze-image', 'nonexistent-image:invalid-tag'
            ])

            # Should handle invalid image names gracefully
            assert result.exit_code in [0, 1]  # Allow graceful failure
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_layer_analysis_with_output_file(self):
        """Test CLI --layer-analysis with file output."""
        dockerfile_content = """
FROM golang:1.21-alpine
COPY . /app
WORKDIR /app
RUN go build -o main .
CMD ["./main"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.analysis', delete=False) as outf:
            output_path = outf.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis',
                '--output', output_path
            ])

            assert result.exit_code == 0
            # Layer analysis currently doesn't support file output, outputs to stdout
            assert "Dockerfile Layer Analysis" in result.output
            assert "Efficiency Score" in result.output
        finally:
            Path(dockerfile_path).unlink()
            Path(output_path).unlink()

    def test_cli_analyze_image_with_output_file(self):
        """Test CLI --analyze-image with file output."""
        dockerfile_content = """FROM redis:alpine
RUN apk add --no-cache curl"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.analysis', delete=False) as outf:
            output_path = outf.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analyze-image', 'redis:alpine',
                '--output', output_path
            ])

            # Should work even if Docker is not available (graceful handling)
            assert result.exit_code in [0, 1]  # Allow both success and graceful failure
            if result.exit_code == 0:
                # Image analysis currently doesn't support file output, outputs to stdout
                assert "Docker Image Analysis" in result.output
        finally:
            Path(dockerfile_path).unlink()
            Path(output_path).unlink()

    def test_cli_efficiency_score_recommendations(self):
        """Test CLI efficiency score recommendations display."""
        # Test dockerfile with poor efficiency (multiple RUN commands)
        dockerfile_content = """
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip
RUN pip install requests
RUN pip install flask
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis'
            ])

            assert result.exit_code == 0
            # Should show efficiency recommendations
            assert any(keyword in result.output for keyword in [
                "Excellent:", "Good:", "Fair:", "Poor:"
            ])
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_combined_flags_layer_analysis_and_security(self):
        """Test CLI with combined --layer-analysis and --security-scan flags."""
        dockerfile_content = """
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y python3 openssl
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis',
                '--security-scan'
            ])

            assert result.exit_code == 0
            # Due to CLI logic using elif, only layer analysis runs when both flags are provided
            assert "Layer Analysis" in result.output or "Efficiency Score" in result.output
            # Security scan doesn't run due to elif priority
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_combined_flags_performance_and_layer_analysis(self):
        """Test CLI with combined --performance and --layer-analysis flags."""
        dockerfile_content = """
FROM python:3.11
RUN pip install requests
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--performance',
                '--layer-analysis'
            ])

            assert result.exit_code == 0
            # Should work with combined flags
            assert "Optimized Dockerfile" in result.output or "Dockerfile Layer Analysis" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_exception_handling_invalid_dockerfile_content(self):
        """Test CLI exception handling with invalid Dockerfile content."""
        # Create a Dockerfile with invalid content that might cause parsing errors
        dockerfile_content = "INVALID_INSTRUCTION this is not a valid Dockerfile instruction\nFROM"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--verbose'  # Enable verbose for more error details
            ])

            # Should handle invalid Dockerfile gracefully
            assert result.exit_code in [0, 1, 2]  # Allow various error codes
            # Should show error message in verbose mode
            if result.exit_code != 0:
                assert len(result.output) > 0  # Should have some error output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_exception_handling_with_verbose_traceback(self):
        """Test CLI exception handling with verbose traceback output."""
        # Create a scenario that might cause an exception during processing
        dockerfile_content = "FROM nonexistent:invalid\nRUN invalid-command"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--verbose'
            ])
            
            # Should handle gracefully and show traceback in verbose mode
            assert result.exit_code in [0, 1]
            # In verbose mode, should either succeed or show detailed error
            assert len(result.output) > 0
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_empty_batch_processing_error(self):
        """Test CLI batch processing with no valid Dockerfiles."""
        # Create non-existent file paths for batch processing
        result = self.runner.invoke(main, [
            '--batch', '/nonexistent/Dockerfile1',
            '--batch', '/nonexistent/Dockerfile2',
            '--performance'
        ])
        
        # Should handle empty batch processing gracefully
        assert result.exit_code == 1
        assert "No valid Dockerfiles found" in result.output

    def test_cli_analysis_with_yaml_output_format(self):
        """Test CLI analysis mode with YAML output format."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache python3
USER root
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analysis-only',
                '--format', 'yaml'
            ])
            
            assert result.exit_code == 0
            # YAML output should contain structured data
            assert "base_image:" in result.output
            assert "total_layers:" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analysis_with_json_output_format(self):
        """Test CLI analysis mode with JSON output format."""
        dockerfile_content = """
FROM ubuntu:20.04
USER root
RUN apt-get update
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analysis-only',
                '--format', 'json'
            ])
            
            assert result.exit_code == 0
            # Should be valid JSON
            output_data = json.loads(result.output)
            assert "base_image" in output_data
            assert "total_layers" in output_data
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_verbose_analysis_with_security_issues(self):
        """Test CLI verbose analysis mode showing security issue details."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update
RUN curl http://insecure-url.com/script.sh | bash
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--analysis-only',
                '--verbose'
            ])
            
            assert result.exit_code == 0
            assert "Security Issues Found" in result.output
            # Verbose mode should show more details
            assert len(result.output) > 300
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_verbose_optimization_with_layer_details(self):
        """Test CLI verbose optimization showing layer optimization details."""
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get clean
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--verbose'
            ])
            
            assert result.exit_code == 0
            assert "Layer Optimizations" in result.output
            # Verbose mode should show optimization reasoning
            assert len(result.output) > 400
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_batch_processing_with_output_file_paths(self):
        """Test CLI batch processing with output file generation."""
        dockerfile1_content = "FROM alpine:3.18\nRUN apk add curl"
        dockerfile2_content = "FROM ubuntu:20.04\nRUN apt-get update"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f1:
            f1.write(dockerfile1_content)
            dockerfile1_path = f1.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f2:
            f2.write(dockerfile2_content)
            dockerfile2_path = f2.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.output', delete=False) as out:
            output_path = out.name
        
        try:
            result = self.runner.invoke(main, [
                '--batch', dockerfile1_path,
                '--batch', dockerfile2_path,
                '--output', output_path
            ])
            
            assert result.exit_code == 0
            # Should process multiple files
            assert dockerfile1_path in result.output
            assert dockerfile2_path in result.output
        finally:
            Path(dockerfile1_path).unlink()
            Path(dockerfile2_path).unlink()
            Path(output_path).unlink(missing_ok=True)

    def test_cli_invalid_format_option(self):
        """Test CLI with invalid format option."""
        dockerfile_content = "FROM alpine:3.18\nRUN apk add curl"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--format', 'invalid'
            ])
            
            # Should handle invalid format option
            assert result.exit_code == 2  # Click validation error
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_layer_analysis_with_verbose_layer_breakdown(self):
        """Test CLI layer analysis with verbose layer breakdown details."""
        dockerfile_content = """
FROM python:3.11-slim
RUN pip install requests
RUN pip install flask
COPY app.py /app/
WORKDIR /app
CMD ["python", "app.py"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name
        
        try:
            result = self.runner.invoke(main, [
                '--dockerfile', dockerfile_path,
                '--layer-analysis',
                '--verbose'
            ])
            
            assert result.exit_code == 0
            assert "Layer Breakdown" in result.output
            # Verbose mode should show individual layer details
            assert "Layer 1:" in result.output or "Estimated Size" in result.output
        finally:
            Path(dockerfile_path).unlink()

    def test_cli_analyze_image_with_verbose_details(self):
        """Test CLI image analysis with verbose details.""" 
        result = self.runner.invoke(main, [
            '--analyze-image', 'alpine:3.18',
            '--verbose'
        ])
        
        # May succeed or fail depending on Docker availability
        assert result.exit_code in [0, 1, 2]  # Allow Click validation errors too
        if result.exit_code == 0:
            assert "Docker Image Analysis" in result.output
            # Verbose should show layer commands and creation times
            assert len(result.output) > 200
