"""Size estimation utilities for Docker images."""

from typing import Any, Dict, Optional

from .config import Config
from .layer_analyzer import DockerLayerAnalyzer
from .models import ImageAnalysis


class SizeEstimator:
    """Estimates Docker image sizes based on Dockerfile content."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the size estimator with configuration.

        Args:
            config: Optional configuration instance. If None, default config is used.
        """
        self.config = config or Config()
        self.layer_analyzer = DockerLayerAnalyzer()

        # Load sizes from configuration instead of hardcoded values
        self.base_image_sizes = self.config.get_base_image_sizes()
        self.package_sizes = self.config.get_package_sizes()
        self.layer_settings = self.config.get_layer_estimation_settings()

        # Legacy hardcoded fallback (kept for backward compatibility during transition)
        self._legacy_base_sizes = {
            # Alpine variants (MB)
            "alpine:3.18": 7,
            "alpine:3.17": 7,
            "alpine:latest": 7,
            "alpine": 7,
            # Ubuntu variants (MB)
            "ubuntu:22.04": 77,
            "ubuntu:20.04": 72,
            "ubuntu:22.04-slim": 30,
            "ubuntu:20.04-slim": 28,
            "ubuntu:latest": 77,
            "ubuntu": 77,
            # Debian variants (MB)
            "debian:12": 117,
            "debian:11": 124,
            "debian:12-slim": 74,
            "debian:11-slim": 80,
            "debian:latest": 117,
            "debian": 117,
            # Language-specific images (MB)
            "python:3.11": 1013,
            "python:3.11-slim": 130,
            "python:3.11-alpine": 47,
            "python:3.10": 995,
            "python:3.10-slim": 125,
            "python:3.10-alpine": 45,
            "python": 1013,
            "node:18": 993,
            "node:18-slim": 167,
            "node:18-alpine": 110,
            "node:16": 943,
            "node:16-slim": 159,
            "node:16-alpine": 109,
            "node": 993,
            "golang:1.21": 814,
            "golang:1.21-alpine": 268,
            "golang:1.20": 808,
            "golang:1.20-alpine": 260,
            "golang": 814,
            "openjdk:17": 471,
            "openjdk:17-slim": 220,
            "openjdk:17-alpine": 164,
            "openjdk:11": 390,
            "openjdk:11-slim": 179,
            "openjdk:11-alpine": 156,
            "openjdk": 471,
        }

        # Legacy package sizes (kept for backward compatibility during transition)
        self._legacy_package_sizes = {
            "curl": 2,
            "wget": 1,
            "git": 15,
            "vim": 8,
            "nano": 1,
            "build-essential": 180,
            "gcc": 90,
            "g++": 50,
            "make": 5,
            "cmake": 25,
            "nodejs": 50,
            "npm": 25,
            "python3": 25,
            "python3-pip": 15,
            "postgresql-client": 20,
            "mysql-client": 15,
            "redis-tools": 5,
            "imagemagick": 40,
            "ffmpeg": 60,
            "openssh-client": 3,
            "rsync": 3,
            "unzip": 1,
            "zip": 1,
            "jq": 2,
        }

    def estimate_size(self, dockerfile_content: str) -> str:
        """Estimate the final image size.

        Args:
            dockerfile_content: The Dockerfile content to analyze

        Returns:
            Estimated size as a string (e.g., "120MB")
        """
        base_size = self._get_base_image_size(dockerfile_content)
        package_size = self._estimate_package_size(dockerfile_content)

        total_size_mb = base_size + package_size

        if total_size_mb >= 1024:
            return f"{total_size_mb / 1024:.1f}GB"
        else:
            return f"{total_size_mb}MB"

    def _get_base_image_size(self, dockerfile_content: str) -> int:
        """Extract and estimate base image size."""
        base_image = self._extract_base_image(dockerfile_content)

        # Try exact match first
        if base_image in self.base_image_sizes:
            return self.base_image_sizes[base_image]

        # Try pattern matching
        for pattern, size in self.base_image_sizes.items():
            if self._matches_image_pattern(base_image, pattern):
                return size

        # Use configured fallback or legacy fallback
        fallback_size = self.config.get_image_size(base_image)
        if fallback_size != self.config._config["default_fallbacks"]["unknown_image_size_mb"]:
            return fallback_size

        # Legacy fallback patterns
        if "alpine" in base_image.lower():
            return 20
        elif "slim" in base_image.lower():
            return 50
        else:
            fallback: int = self.config._config["default_fallbacks"]["unknown_image_size_mb"]
            return fallback

    def _extract_base_image(self, dockerfile_content: str) -> str:
        """Extract the base image from Dockerfile."""
        lines = dockerfile_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("FROM "):
                parts = line.split()
                if len(parts) >= 2:
                    base_image = parts[1]
                    # Remove alias if present
                    if "AS" in line.upper():
                        return base_image
                    return base_image
        return "unknown"

    def _matches_image_pattern(self, image: str, pattern: str) -> bool:
        """Check if image matches a pattern."""
        # Handle wildcards and version matching
        image_lower = image.lower()
        pattern_lower = pattern.lower()

        # Extract base name
        image_base = image_lower.split(":")[0]
        pattern_base = pattern_lower.split(":")[0]

        return image_base == pattern_base

    def _estimate_package_size(self, dockerfile_content: str) -> int:
        """Estimate size added by packages and commands."""
        total_size = 0

        # Analyze RUN commands for package installations
        run_lines = [
            line
            for line in dockerfile_content.split("\n")
            if line.strip().startswith("RUN ")
        ]

        for line in run_lines:
            line_lower = line.lower()

            # Check for package manager commands
            if "apt-get install" in line_lower or "apt install" in line_lower:
                total_size += self._estimate_apt_packages(line_lower)
            elif "apk add" in line_lower:
                total_size += self._estimate_apk_packages(line_lower)
            elif "yum install" in line_lower or "dnf install" in line_lower:
                total_size += self._estimate_yum_packages(line_lower)
            elif "pip install" in line_lower:
                total_size += self._estimate_pip_packages(line_lower)
            elif "npm install" in line_lower:
                total_size += self._estimate_npm_packages(line_lower)

        # Add overhead for COPY/ADD operations
        copy_lines = [
            line
            for line in dockerfile_content.split("\n")
            if line.strip().startswith(("COPY ", "ADD "))
        ]
        total_size += len(copy_lines) * self.layer_settings["copy_layer_mb"]

        return total_size

    def _estimate_apt_packages(self, line: str) -> int:
        """Estimate size of apt packages."""
        size = 0
        for package, package_size in self.package_sizes.items():
            if package in line:
                size += package_size

        # If no specific packages found, estimate based on line complexity
        if size == 0:
            package_count = len(
                [
                    word
                    for word in line.split()
                    if not word.startswith("-")
                    and word not in ["apt-get", "install", "update", "upgrade"]
                ]
            )
            size = package_count * self.config._config["default_fallbacks"]["unknown_package_size_mb"]

        return size

    def _estimate_apk_packages(self, line: str) -> int:
        """Estimate size of apk packages (typically smaller than apt)."""
        return int(
            self._estimate_apt_packages(line) * 0.6
        )  # Alpine packages are typically smaller

    def _estimate_yum_packages(self, line: str) -> int:
        """Estimate size of yum/dnf packages."""
        return int(
            self._estimate_apt_packages(line) * 1.2
        )  # RPM packages can be larger

    def _estimate_pip_packages(self, line: str) -> int:
        """Estimate size of Python packages."""
        # Common Python packages and their approximate sizes
        python_packages = {
            "numpy": 20,
            "pandas": 40,
            "scipy": 35,
            "matplotlib": 50,
            "tensorflow": 500,
            "torch": 800,
            "django": 15,
            "flask": 5,
            "requests": 2,
            "pillow": 10,
            "opencv-python": 60,
        }

        size = 0
        for package, package_size in python_packages.items():
            if package in line:
                size += package_size

        # Estimate for unknown packages
        if "requirements.txt" in line:
            size += 50  # Estimate for typical requirements file
        elif size == 0:
            # Count package names
            package_count = len(
                [
                    word
                    for word in line.split()
                    if not word.startswith("-") and word not in ["pip", "install"]
                ]
            )
            size = package_count * 5  # 5MB per unknown Python package

        return size

    def _estimate_npm_packages(self, line: str) -> int:
        """Estimate size of npm packages."""
        if "package.json" in line or "npm install" in line and len(line.split()) == 3:
            return 100  # Typical node_modules size
        else:
            # Count individual packages
            package_count = len(
                [
                    word
                    for word in line.split()
                    if not word.startswith("-") and word not in ["npm", "install"]
                ]
            )
            return package_count * 10  # 10MB per npm package

    def analyze_dockerfile_layers(self, dockerfile_content: str) -> ImageAnalysis:
        """Analyze Dockerfile layers for size estimation."""
        return self.layer_analyzer.get_layer_sizes_for_dockerfile(dockerfile_content)

    def analyze_image_layers(self, image_name: str) -> ImageAnalysis:
        """Analyze existing Docker image layers."""
        return self.layer_analyzer.analyze_image_layers(image_name)

    def get_detailed_size_breakdown(self, dockerfile_content: str) -> Dict[str, Any]:
        """Get detailed size breakdown including layer analysis."""
        layer_analysis = self.analyze_dockerfile_layers(dockerfile_content)
        traditional_estimate = self.estimate_size(dockerfile_content)

        return {
            "traditional_estimate": traditional_estimate,
            "layer_analysis": layer_analysis,
            "estimated_layers": len(layer_analysis.layers),
            "total_estimated_size_mb": layer_analysis.total_size_mb,
            "largest_layer_mb": layer_analysis.largest_layer.size_mb if layer_analysis.largest_layer else 0,
            "dockerfile_efficiency_score": self._calculate_efficiency_score(layer_analysis)
        }

    def _calculate_efficiency_score(self, analysis: ImageAnalysis) -> int:
        """Calculate efficiency score (0-100) based on layer analysis."""
        if not analysis.layers:
            return 50  # Neutral score if no analysis available

        score = 100

        # Penalty for too many layers (more than 8 is getting inefficient)
        if analysis.layer_count > 8:
            score -= min(30, (analysis.layer_count - 8) * 4)

        # Penalty for separate RUN commands (indicates layer consolidation opportunity)
        run_commands = [layer for layer in analysis.layers
                       if layer.command.upper().startswith('RUN')]
        separate_run_commands = [layer for layer in run_commands
                               if '&&' not in layer.command]
        if len(separate_run_commands) > 1:
            score -= min(25, (len(separate_run_commands) - 1) * 8)

        # Penalty for very large individual layers (>100MB)
        large_layers = [layer for layer in analysis.layers
                       if (layer.estimated_size_bytes or 0) > 100 * 1024 * 1024]
        if large_layers:
            score -= min(20, len(large_layers) * 5)

        # Bonus for using combined RUN commands
        combined_commands = sum(1 for layer in analysis.layers
                              if '&&' in layer.command)
        if combined_commands > 0:
            score += min(10, combined_commands * 2)

        return max(0, min(100, score))
