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
