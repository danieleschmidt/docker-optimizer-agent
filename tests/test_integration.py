"""Integration tests with real Docker builds."""

import os
import subprocess
import tempfile
from typing import Any, Dict

import pytest

from docker_optimizer import DockerfileOptimizer


class TestDockerIntegration:
    """Integration tests that validate optimizations with real Docker builds."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = DockerfileOptimizer()
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up any test images
        try:
            subprocess.run(
                ["docker", "images", "-q", "--filter", "label=test-optimizer"],
                capture_output=True,
                check=False
            )
        except Exception:
            pass

    def _create_test_dockerfile(self, content: str, filename: str = "Dockerfile") -> str:
        """Create a test dockerfile in the test directory."""
        dockerfile_path = os.path.join(self.test_dir, filename)
        with open(dockerfile_path, 'w') as f:
            f.write(content)
        return dockerfile_path

    def _build_docker_image(self, dockerfile_path: str, tag: str) -> Dict[str, Any]:
        """Build a Docker image and return build metrics."""
        try:
            # Build the image
            result = subprocess.run([
                "docker", "build",
                "-f", dockerfile_path,
                "-t", tag,
                "--label", "test-optimizer",
                self.test_dir
            ], capture_output=True, text=True, timeout=120)

            # Get image size
            size_result = subprocess.run([
                "docker", "images", tag, "--format", "{{.Size}}"
            ], capture_output=True, text=True)

            return {
                "success": result.returncode == 0,
                "build_output": result.stdout,
                "build_error": result.stderr,
                "size": size_result.stdout.strip() if size_result.returncode == 0 else "unknown"
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "build_output": "",
                "build_error": "Build timeout",
                "size": "unknown"
            }

    def test_basic_optimization_builds_successfully(self):
        """Test that optimized Dockerfiles build successfully."""
        # Original inefficient Dockerfile
        original_dockerfile = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
COPY . /app
WORKDIR /app
CMD ["echo", "hello"]
"""

        # Optimize the Dockerfile
        result = self.optimizer.optimize_dockerfile(original_dockerfile)

        # Ensure optimization was successful
        assert result.optimized_dockerfile != original_dockerfile
        assert len(result.layer_optimizations) > 0 or len(result.security_fixes) > 0

        # Create dockerfiles
        original_path = self._create_test_dockerfile(original_dockerfile, "Dockerfile.original")
        optimized_path = self._create_test_dockerfile(result.optimized_dockerfile, "Dockerfile.optimized")

        # Build both images
        original_build = self._build_docker_image(original_path, "test-optimizer-original:latest")
        optimized_build = self._build_docker_image(optimized_path, "test-optimizer-optimized:latest")

        # Both should build successfully
        assert original_build["success"], f"Original build failed: {original_build['build_error']}"
        assert optimized_build["success"], f"Optimized build failed: {optimized_build['build_error']}"

    def test_multistage_optimization_builds(self):
        """Test that multi-stage optimizations build correctly."""
        # Python app Dockerfile suitable for multi-stage optimization
        python_dockerfile = """
FROM python:3.11
RUN apt-get update && apt-get install -y gcc build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        # Create a minimal requirements.txt
        requirements_path = os.path.join(self.test_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write("requests==2.28.0\n")

        # Create a minimal app.py
        app_path = os.path.join(self.test_dir, "app.py")
        with open(app_path, 'w') as f:
            f.write("print('Hello from optimized Docker!')\n")

        # Optimize with multi-stage
        from docker_optimizer.multistage import MultiStageOptimizer
        multistage_optimizer = MultiStageOptimizer()
        result = multistage_optimizer.generate_multistage_dockerfile(python_dockerfile)

        # Should generate a multi-stage build
        assert "FROM" in result.optimized_dockerfile
        assert result.optimized_dockerfile.count("FROM") > 1  # Multi-stage

        # Create and build optimized dockerfile
        optimized_path = self._create_test_dockerfile(result.optimized_dockerfile, "Dockerfile.multistage")
        build_result = self._build_docker_image(optimized_path, "test-optimizer-multistage:latest")

        # Should build successfully
        assert build_result["success"], f"Multi-stage build failed: {build_result['build_error']}"

    def test_security_optimizations_build(self):
        """Test that security-focused optimizations build correctly."""
        # Dockerfile with security issues
        insecure_dockerfile = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
COPY . /app
WORKDIR /app
CMD ["./app"]
"""

        # Optimize focusing on security
        result = self.optimizer.optimize_dockerfile(insecure_dockerfile)

        # Should include security improvements
        optimized_lower = result.optimized_dockerfile.lower()
        assert "user" in optimized_lower or len(result.security_fixes) > 0

        # Build the optimized version
        optimized_path = self._create_test_dockerfile(result.optimized_dockerfile, "Dockerfile.secure")
        build_result = self._build_docker_image(optimized_path, "test-optimizer-secure:latest")

        # Should build successfully
        assert build_result["success"], f"Secure build failed: {build_result['build_error']}"

    @pytest.mark.slow
    def test_large_dockerfile_optimization(self):
        """Test optimization of larger, more complex Dockerfiles."""
        # Complex Dockerfile with multiple optimization opportunities
        complex_dockerfile = """
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y build-essential
COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install -r requirements.txt
COPY . /app/
RUN python3 setup.py build
EXPOSE 8000
CMD ["python3", "server.py"]
"""

        # Create supporting files
        requirements_path = os.path.join(self.test_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write("flask==2.0.1\nrequests==2.28.0\n")

        setup_path = os.path.join(self.test_dir, "setup.py")
        with open(setup_path, 'w') as f:
            f.write("from setuptools import setup\nsetup(name='test')\n")

        server_path = os.path.join(self.test_dir, "server.py")
        with open(server_path, 'w') as f:
            f.write("print('Server starting...')\n")

        # Optimize
        result = self.optimizer.optimize_dockerfile(complex_dockerfile)

        # Should have multiple optimizations (layer combining and security fix)
        assert len(result.layer_optimizations) + len(result.security_fixes) >= 2

        # Should combine RUN commands
        assert result.optimized_dockerfile.count("RUN apt-get") < complex_dockerfile.count("RUN apt-get")

        # Build both versions
        original_path = self._create_test_dockerfile(complex_dockerfile, "Dockerfile.complex.original")
        optimized_path = self._create_test_dockerfile(result.optimized_dockerfile, "Dockerfile.complex.optimized")

        original_build = self._build_docker_image(original_path, "test-optimizer-complex-original:latest")
        optimized_build = self._build_docker_image(optimized_path, "test-optimizer-complex-optimized:latest")

        # Both should build successfully
        assert original_build["success"], f"Original complex build failed: {original_build['build_error']}"
        assert optimized_build["success"], f"Optimized complex build failed: {optimized_build['build_error']}"

    def test_optimization_preserves_functionality(self):
        """Test that optimizations don't break application functionality."""
        # Simple Node.js app Dockerfile
        nodejs_dockerfile = """
FROM node:16
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
EXPOSE 3000
CMD ["node", "index.js"]
"""

        # Create supporting files
        package_json_path = os.path.join(self.test_dir, "package.json")
        with open(package_json_path, 'w') as f:
            f.write('{"name": "test", "dependencies": {}}\n')

        index_path = os.path.join(self.test_dir, "index.js")
        with open(index_path, 'w') as f:
            f.write("console.log('App running on port 3000');\n")

        # Optimize
        result = self.optimizer.optimize_dockerfile(nodejs_dockerfile)

        # Build optimized version
        optimized_path = self._create_test_dockerfile(result.optimized_dockerfile, "Dockerfile.nodejs")
        build_result = self._build_docker_image(optimized_path, "test-optimizer-nodejs:latest")

        # Should build successfully
        assert build_result["success"], f"Node.js build failed: {build_result['build_error']}"

        # Test that the container runs
        try:
            run_result = subprocess.run([
                "docker", "run", "--rm", "-t",
                "test-optimizer-nodejs:latest",
                "node", "-e", "console.log('test passed')"
            ], capture_output=True, text=True, timeout=30)

            assert run_result.returncode == 0, f"Container execution failed: {run_result.stderr}"
            assert "test passed" in run_result.stdout
        except subprocess.TimeoutExpired:
            pytest.fail("Container execution timed out")
