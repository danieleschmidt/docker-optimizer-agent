"""Test cases for CLI optimization presets integration."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from docker_optimizer.cli import main


class TestCLIOptimizationPresets:
    """Test CLI integration for optimization presets and profiles."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_preset_development(self):
        """Test CLI --preset flag with development profile."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the --preset flag with development
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--preset", "DEVELOPMENT",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should include preset information
            assert "preset_applied" in output_data
            assert output_data["preset_applied"]["type"] == "DEVELOPMENT"
            assert "optimizations" in output_data["preset_applied"]

            # Should have development-specific optimizations
            optimizations = output_data["preset_applied"]["optimizations"]
            optimization_names = [opt["name"] for opt in optimizations]
            assert any("cache" in name for name in optimization_names)
            assert any("debug" in opt["description"].lower() for opt in optimizations)

        finally:
            Path(dockerfile_path).unlink()

    def test_cli_preset_production(self):
        """Test CLI --preset flag with production profile."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the --preset flag with production
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--preset", "PRODUCTION",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should include preset information
            assert "preset_applied" in output_data
            assert output_data["preset_applied"]["type"] == "PRODUCTION"

            # Should have production-specific optimizations
            optimizations = output_data["preset_applied"]["optimizations"]
            optimization_names = [opt["name"] for opt in optimizations]
            assert any("slim" in name.lower() or "size" in name.lower() for name in optimization_names)

        finally:
            Path(dockerfile_path).unlink()

    def test_cli_preset_web_app(self):
        """Test CLI --preset flag with web app profile."""
        dockerfile_content = """FROM node:latest
COPY package.json /app/
WORKDIR /app
RUN npm install
COPY . /app
CMD ["npm", "start"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the --preset flag with web app
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--preset", "WEB_APP",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should include preset information
            assert "preset_applied" in output_data
            assert output_data["preset_applied"]["type"] == "WEB_APP"

            # Should have web app-specific optimizations
            optimizations = output_data["preset_applied"]["optimizations"]
            optimization_descriptions = [opt["description"] for opt in optimizations]
            assert any("static" in desc.lower() or "nginx" in desc.lower() for desc in optimization_descriptions)

        finally:
            Path(dockerfile_path).unlink()

    def test_cli_preset_ml(self):
        """Test CLI --preset flag with ML profile."""
        dockerfile_content = """FROM python:latest
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "train.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the --preset flag with ML
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--preset", "ML",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should include preset information
            assert "preset_applied" in output_data
            assert output_data["preset_applied"]["type"] == "ML"

            # Should have ML-specific optimizations
            optimizations = output_data["preset_applied"]["optimizations"]
            optimization_descriptions = [opt["description"] for opt in optimizations]
            assert any("gpu" in desc.lower() or "nvidia" in desc.lower() or "cuda" in desc.lower() for desc in optimization_descriptions)

        finally:
            Path(dockerfile_path).unlink()

    def test_cli_preset_data_processing(self):
        """Test CLI --preset flag with data processing profile."""
        dockerfile_content = """FROM python:latest
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "process_data.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test the --preset flag with data processing
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--preset", "DATA_PROCESSING",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should include preset information
            assert "preset_applied" in output_data
            assert output_data["preset_applied"]["type"] == "DATA_PROCESSING"

            # Should have data processing-specific optimizations
            optimizations = output_data["preset_applied"]["optimizations"]
            optimization_descriptions = [opt["description"] for opt in optimizations]
            assert any("memory" in desc.lower() or "volume" in desc.lower() or "io" in desc.lower() for desc in optimization_descriptions)

        finally:
            Path(dockerfile_path).unlink()

    def test_cli_custom_preset_file(self):
        """Test CLI --custom-preset flag with custom preset file."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        custom_preset = {
            "name": "Custom Test Preset",
            "description": "A custom preset for testing",
            "target_use_case": "Testing",
            "optimizations": [
                {
                    "name": "Custom optimization",
                    "description": "Custom optimization for testing",
                    "dockerfile_change": "RUN echo 'custom optimization'",
                    "reasoning": "Testing custom presets",
                    "priority": 1
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            preset_path = Path(temp_dir) / "custom_preset.json"

            dockerfile_path.write_text(dockerfile_content)
            preset_path.write_text(json.dumps(custom_preset, indent=2))

            # Test the --custom-preset flag
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--custom-preset", str(preset_path),
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should include custom preset information
            assert "preset_applied" in output_data
            assert output_data["preset_applied"]["name"] == "Custom Test Preset"
            assert len(output_data["preset_applied"]["optimizations"]) == 1
            assert output_data["preset_applied"]["optimizations"][0]["name"] == "Custom optimization"

    def test_cli_preset_text_format(self):
        """Test CLI --preset flag with text output format."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update && apt-get install -y python3
COPY . /app
WORKDIR /app
CMD ["python3", "app.py"]"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test preset with text format
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--preset", "PRODUCTION",
                "--format", "text"
            ])

            assert result.exit_code == 0

            # Should include preset information in text output
            assert "Preset Applied:" in result.output
            assert "Type: PRODUCTION" in result.output
            assert "Optimizations Applied:" in result.output

        finally:
            Path(dockerfile_path).unlink()

    def test_cli_preset_with_language_detect(self):
        """Test CLI preset combined with language detection."""
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

            # Test preset with language detection
            result = self.runner.invoke(main, [
                "--dockerfile", str(dockerfile_path),
                "--preset", "WEB_APP",
                "--language-detect",
                "--format", "json"
            ])

            assert result.exit_code == 0
            output_data = json.loads(result.output)

            # Should include both preset and language analysis
            assert "preset_applied" in output_data
            assert "language_analysis" in output_data
            assert output_data["preset_applied"]["type"] == "WEB_APP"
            assert output_data["language_analysis"]["language"] == "python"

    def test_cli_list_presets(self):
        """Test CLI --list-presets flag to show available presets."""
        # Test the --list-presets flag
        result = self.runner.invoke(main, ["--list-presets"])

        assert result.exit_code == 0

        # Should list all available presets
        assert "Available Optimization Presets:" in result.output
        assert "DEVELOPMENT" in result.output
        assert "PRODUCTION" in result.output
        assert "WEB_APP" in result.output
        assert "ML" in result.output
        assert "DATA_PROCESSING" in result.output

    def test_cli_preset_invalid(self):
        """Test CLI --preset flag with invalid preset name."""
        dockerfile_content = """FROM ubuntu:latest
RUN apt-get update
COPY . /app
WORKDIR /app"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
            f.write(dockerfile_content)
            dockerfile_path = f.name

        try:
            # Test invalid preset name
            result = self.runner.invoke(main, [
                "--dockerfile", dockerfile_path,
                "--preset", "INVALID_PRESET",
                "--format", "json"
            ])

            # Should fail with proper error message
            assert result.exit_code != 0
            assert "Invalid preset" in result.output or "Error" in result.output

        finally:
            Path(dockerfile_path).unlink()
