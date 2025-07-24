"""Test cases for CLI language detection integration."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from docker_optimizer.cli import main


class TestCLILanguageDetection:
    """Test CLI integration for language-specific optimization patterns."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_language_detect_flag_python_project(self):
        """Test CLI --language-detect flag with Python project."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            dockerfile_path = project_path / "Dockerfile"

            # Create Python project indicators
            (project_path / "requirements.txt").write_text("django>=4.0\npsycopg2>=2.8")
            (project_path / "manage.py").write_text("#!/usr/bin/env python")
            dockerfile_path.write_text(dockerfile_content)

            # Test the new --language-detect flag
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--language-detect",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should detect Python and Django
            assert "language_analysis" in output_data
            assert output_data["language_analysis"]["language"] == "python"
            assert output_data["language_analysis"]["framework"] == "django"
            assert output_data["language_analysis"]["language_confidence"] > 0.8

            # Should include language-specific suggestions
            suggestions = output_data.get("suggestions", [])
            suggestion_texts = [s["description"] for s in suggestions]
            assert any("python:" in text for text in suggestion_texts)
            assert any("collectstatic" in text for text in suggestion_texts)

    def test_cli_language_detect_flag_nodejs_project(self):
        """Test CLI --language-detect flag with Node.js project."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y nodejs npm
COPY . /app
WORKDIR /app
CMD ["node", "server.js"]"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            dockerfile_path = project_path / "Dockerfile"

            # Create Node.js project indicators
            (project_path / "package.json").write_text('{"name": "test", "version": "1.0.0"}')
            (project_path / "server.js").write_text("const express = require('express');")
            dockerfile_path.write_text(dockerfile_content)

            # Test the new --language-detect flag
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--language-detect",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should detect Node.js and Express
            assert "language_analysis" in output_data
            assert output_data["language_analysis"]["language"] == "nodejs"
            assert output_data["language_analysis"]["framework"] == "express"

            # Should include Node.js-specific suggestions
            suggestions = output_data.get("suggestions", [])
            suggestion_texts = [s["description"] for s in suggestions]
            assert any("node:" in text for text in suggestion_texts)
            assert any("npm" in text for text in suggestion_texts)

    def test_cli_language_detect_flag_go_project(self):
        """Test CLI --language-detect flag with Go project."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y golang
COPY . /app
WORKDIR /app
CMD ["./main"]"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            dockerfile_path = project_path / "Dockerfile"

            # Create Go project indicators
            (project_path / "go.mod").write_text("module test\ngo 1.20")
            (project_path / "main.go").write_text("package main\nfunc main() {}")
            dockerfile_path.write_text(dockerfile_content)

            # Test the new --language-detect flag
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--language-detect",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should detect Go
            assert "language_analysis" in output_data
            assert output_data["language_analysis"]["language"] == "go"

            # Should include Go-specific suggestions
            suggestions = output_data.get("suggestions", [])
            suggestion_texts = [s["description"] for s in suggestions]
            assert any("golang:" in text for text in suggestion_texts)
            assert any("go mod" in text for text in suggestion_texts)

    def test_cli_language_detect_without_flag(self):
        """Test that language detection doesn't run without the flag."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            dockerfile_path = project_path / "Dockerfile"

            # Create Python project indicators
            (project_path / "requirements.txt").write_text("django>=4.0")
            dockerfile_path.write_text(dockerfile_content)

            # Test without --language-detect flag
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should NOT include language analysis
            assert "language_analysis" not in output_data

    def test_cli_language_detect_unknown_project(self):
        """Test CLI --language-detect flag with unknown project type."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update
COPY . /app
WORKDIR /app
CMD ["./app"]"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            dockerfile_path = project_path / "Dockerfile"

            # Create only generic files
            (project_path / "README.md").write_text("# Test project")
            dockerfile_path.write_text(dockerfile_content)

            # Test the new --language-detect flag
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--language-detect",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should detect unknown language
            assert "language_analysis" in output_data
            assert output_data["language_analysis"]["language"] == "unknown"
            assert output_data["language_analysis"]["language_confidence"] == 0.0

            # Should still provide generic suggestions
            suggestions = output_data.get("suggestions", [])
            assert len(suggestions) > 0
            suggestion_texts = [s["description"] for s in suggestions]
            assert any("multi-stage" in text for text in suggestion_texts)

    def test_cli_language_detect_text_format(self):
        """Test CLI --language-detect flag with text output format."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            dockerfile_path = project_path / "Dockerfile"

            # Create Python project indicators
            (project_path / "requirements.txt").write_text("flask>=2.0")
            dockerfile_path.write_text(dockerfile_content)

            # Test the new --language-detect flag with text format
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--language-detect",
                "--format", "text"
            ])

            assert result.exit_code == 0

            # Should include language analysis in text output
            assert "Language Analysis:" in result.output
            assert "Detected Language: python" in result.output
            assert "Detected Framework: flask" in result.output
            assert "Language-Specific Suggestions:" in result.output
            assert "python:" in result.output

    def test_cli_language_detect_with_optimization_profile(self):
        """Test CLI --language-detect with optimization profile."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            dockerfile_path = project_path / "Dockerfile"

            # Create Python project indicators
            (project_path / "requirements.txt").write_text("django>=4.0")
            dockerfile_path.write_text(dockerfile_content)

            # Test with development profile
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--language-detect",
                "--optimization-profile", "development",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should apply development profile recommendations
            suggestions = output_data.get("suggestions", [])

            # Development profile should suggest non-slim images
            base_image_suggestions = [s for s in suggestions if s.get("type") == "base_image"]
            assert len(base_image_suggestions) > 0
            assert "python:3" in base_image_suggestions[0]["description"]
