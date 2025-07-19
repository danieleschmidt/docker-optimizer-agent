"""Size estimation utilities for Docker images."""




class SizeEstimator:
    """Estimates Docker image sizes based on Dockerfile content."""

    def __init__(self) -> None:
        """Initialize the size estimator with base image size data."""
        self.base_image_sizes = {
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

        # Package size estimates (MB)
        self.package_sizes = {
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

        # Default fallback
        if "alpine" in base_image.lower():
            return 20
        elif "slim" in base_image.lower():
            return 50
        else:
            return 100

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
        total_size += len(copy_lines) * 5  # Estimate 5MB per copy operation

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
            size = package_count * 10  # 10MB per unknown package

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
