"""Test cases for size estimator."""

from docker_optimizer.size_estimator import SizeEstimator


class TestSizeEstimator:
    """Test cases for SizeEstimator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = SizeEstimator()

    def test_estimate_alpine_size(self):
        """Test size estimation for Alpine-based images."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache curl
"""

        size = self.estimator.estimate_size(dockerfile_content)
        assert "MB" in size or "GB" in size
        # Alpine should be relatively small
        if "MB" in size:
            size_mb = int(size.replace("MB", ""))
            assert size_mb < 100

    def test_estimate_ubuntu_size(self):
        """Test size estimation for Ubuntu-based images."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y curl
"""

        size = self.estimator.estimate_size(dockerfile_content)
        assert "MB" in size or "GB" in size
        # Ubuntu should be larger than Alpine
        if "MB" in size:
            size_mb = int(size.replace("MB", ""))
            assert size_mb > 50

    def test_get_base_image_size_exact_match(self):
        """Test base image size with exact match."""
        size = self.estimator._get_base_image_size("FROM alpine:3.18")
        assert size == 7  # Should match exact Alpine 3.18 size

    def test_get_base_image_size_pattern_match(self):
        """Test base image size with pattern matching."""
        size = self.estimator._get_base_image_size("FROM python:3.11-custom")
        # Should match python pattern and return a reasonable size
        assert size > 0

    def test_extract_base_image(self):
        """Test base image extraction."""
        # Simple case
        image1 = self.estimator._extract_base_image("FROM ubuntu:20.04")
        assert image1 == "ubuntu:20.04"

        # Multi-stage build
        image2 = self.estimator._extract_base_image("FROM node:18 AS builder")
        assert image2 == "node:18"

        # Complex dockerfile
        dockerfile = """
# Comment
FROM python:3.11-slim
RUN pip install flask
"""
        image3 = self.estimator._extract_base_image(dockerfile)
        assert image3 == "python:3.11-slim"

    def test_matches_image_pattern(self):
        """Test image pattern matching."""
        # Exact match
        assert self.estimator._matches_image_pattern("ubuntu:20.04", "ubuntu:20.04")

        # Base name match
        assert self.estimator._matches_image_pattern("ubuntu:latest", "ubuntu")

        # No match
        assert not self.estimator._matches_image_pattern("alpine:3.18", "ubuntu")

    def test_estimate_package_size(self):
        """Test package size estimation."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y curl wget git
"""

        package_size = self.estimator._estimate_package_size(dockerfile_content)
        assert package_size > 0
        # Should be reasonable for curl + wget + git
        assert package_size >= 15  # curl(2) + wget(1) + git(15) minimum

    def test_estimate_apt_packages(self):
        """Test APT package size estimation."""
        line = "apt-get install -y curl git build-essential"
        size = self.estimator._estimate_apt_packages(line)

        # Should include curl(2) + git(15) + build-essential(180)
        assert size >= 190

    def test_estimate_apk_packages(self):
        """Test APK package size estimation."""
        line = "apk add curl git"
        size = self.estimator._estimate_apk_packages(line)

        # Should be smaller than apt equivalent
        apt_size = self.estimator._estimate_apt_packages(line)
        assert size < apt_size

    def test_estimate_pip_packages(self):
        """Test Python package size estimation."""
        line = "pip install numpy pandas flask"
        size = self.estimator._estimate_pip_packages(line)

        # Should include numpy(20) + pandas(40) + flask(5)
        assert size >= 60

    def test_estimate_npm_packages(self):
        """Test npm package size estimation."""
        line = "npm install express react lodash"
        size = self.estimator._estimate_npm_packages(line)

        # Should be reasonable for 3 packages
        assert size >= 30

    def test_estimate_size_with_copy_operations(self):
        """Test size estimation including COPY operations."""
        dockerfile_content = """
FROM alpine:3.18
COPY package.json /app/
COPY src/ /app/src/
ADD https://example.com/file.tar.gz /tmp/
"""

        package_size = self.estimator._estimate_package_size(dockerfile_content)
        # Should add overhead for 2 COPY operations (ADD doesn't count as COPY)
        assert package_size >= 10  # 2 * 5MB

    def test_estimate_large_image_returns_gb(self):
        """Test that large images return size in GB."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y build-essential nodejs npm python3-pip
RUN npm install -g webpack babel typescript
RUN pip install tensorflow torch numpy pandas scipy matplotlib
"""

        size = self.estimator.estimate_size(dockerfile_content)
        # With all these large packages, should exceed 1GB
        assert "GB" in size or ("MB" in size and int(size.replace("MB", "")) > 1000)

    def test_estimate_size_unknown_base_image(self):
        """Test size estimation for unknown base image."""
        dockerfile_content = "FROM some-custom-image:latest"

        size = self.estimator.estimate_size(dockerfile_content)
        assert "MB" in size
        # Should use reasonable default
        size_mb = int(size.replace("MB", ""))
        assert 50 <= size_mb <= 200

    def test_pip_requirements_file(self):
        """Test pip packages from requirements file."""
        line = "pip install -r requirements.txt"
        size = self.estimator._estimate_pip_packages(line)

        # Should estimate typical requirements file
        assert size >= 50
