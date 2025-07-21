"""Docker layer analysis for accurate size estimation."""

import re
import subprocess
from typing import Dict, List

from .models import ImageAnalysis, LayerInfo


class DockerLayerAnalyzer:
    """Analyzes Docker images and Dockerfiles for accurate layer size information."""

    def __init__(self) -> None:
        """Initialize the layer analyzer."""
        self.docker_available = self._check_docker_availability()

    def _check_docker_availability(self) -> bool:
        """Check if Docker is available on the system."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    def analyze_image_layers(self, image_name: str) -> ImageAnalysis:
        """Analyze layers of an existing Docker image."""
        if not self.docker_available:
            return ImageAnalysis(
                image_name=image_name,
                docker_available=False,
                analysis_method="unavailable"
            )

        try:
            # Get layer history
            result = subprocess.run(
                ["docker", "history", "--no-trunc", image_name],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return ImageAnalysis(
                    image_name=image_name,
                    docker_available=True,
                    analysis_method="docker_history_failed"
                )

            layers = self._parse_docker_history(result.stdout)
            total_size = sum(layer.size_bytes for layer in layers)

            return ImageAnalysis(
                image_name=image_name,
                layers=layers,
                total_size=total_size,
                docker_available=True,
                analysis_method="docker_history"
            )

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return ImageAnalysis(
                image_name=image_name,
                docker_available=False,
                analysis_method="docker_error"
            )

    def _parse_docker_history(self, history_output: str) -> List[LayerInfo]:
        """Parse Docker history output into LayerInfo objects."""
        layers = []
        lines = history_output.strip().split('\n')

        # Skip header line if present
        if lines and 'IMAGE' in lines[0] and 'CREATED' in lines[0]:
            lines = lines[1:]

        for line in lines:
            if not line.strip():
                continue

            # Parse the standard docker history format
            # Format: IMAGE  CREATED  CREATED BY  SIZE  COMMENT
            parts = line.split(None, 4)  # Split on whitespace, max 5 parts
            if len(parts) < 4:
                continue

            layer_id = parts[0][:12]  # First column: IMAGE (layer ID)
            created = parts[1] + " " + parts[2]  # CREATED (may have spaces)

            # Find the size by looking for patterns like "45.2MB", "1.23KB", etc.
            size_match = re.search(r'(\d+(?:\.\d+)?(?:B|KB|MB|GB|TB))', line)
            size_str = size_match.group(1) if size_match else "0B"

            # Extract the command part (everything after size)
            size_pos = line.find(size_str)
            if size_pos >= 0:
                command_part = line[size_pos + len(size_str):].strip()
                # Look for CREATED BY part before the size
                created_by_start = line.find('/bin/sh -c')
                if created_by_start >= 0 and created_by_start < size_pos:
                    command = line[created_by_start:size_pos].strip()
                else:
                    command = command_part
            else:
                command = "unknown"

            # Parse size string to bytes
            size_bytes = self._parse_size_string(size_str)

            # Clean up command (remove #(nop) prefix)
            if command.startswith("#(nop)"):
                command = command[6:].strip()

            layer = LayerInfo(
                layer_id=layer_id,
                command=command,
                size_bytes=size_bytes,
                created=created,
                estimated_size_bytes=size_bytes
            )
            layers.append(layer)

        return layers

    def _parse_size_string(self, size_str: str) -> int:
        """Parse Docker size string (e.g., '45.2MB', '1.23kB') to bytes."""
        if not size_str or size_str == '0B':
            return 0

        # Extract number and unit
        match = re.match(r'([0-9.]+)\s*([A-Za-z]+)', size_str)
        if not match:
            return 0

        number = float(match.group(1))
        unit = match.group(2).upper()

        # Convert to bytes
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024,
            'TB': 1024 * 1024 * 1024 * 1024
        }

        return int(number * multipliers.get(unit, 1))

    def get_layer_sizes_for_dockerfile(self, dockerfile_content: str) -> ImageAnalysis:
        """Estimate layer sizes for a Dockerfile without building it."""
        layers = []
        instructions = self._parse_dockerfile_instructions(dockerfile_content)

        for i, instruction in enumerate(instructions):
            # Estimate size based on instruction type and content
            estimated_size = self._estimate_instruction_size(instruction)

            layer = LayerInfo(
                layer_id=f"estimated_{i}",
                command=instruction,
                size_bytes=0,  # No actual size available
                created=None,
                estimated_size_bytes=estimated_size
            )
            layers.append(layer)

        total_estimated_size = sum(layer.estimated_size_bytes or 0 for layer in layers)

        return ImageAnalysis(
            image_name="dockerfile_analysis",
            layers=layers,
            total_size=total_estimated_size,
            docker_available=self.docker_available,
            analysis_method="dockerfile_estimation"
        )

    def _parse_dockerfile_instructions(self, dockerfile_content: str) -> List[str]:
        """Parse Dockerfile content into individual instructions."""
        instructions = []

        for line in dockerfile_content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                instructions.append(line)

        return instructions

    def _estimate_instruction_size(self, instruction: str) -> int:
        """Estimate the size impact of a Dockerfile instruction."""
        instruction_upper = instruction.upper()

        # FROM instructions typically don't add size (base image size is separate)
        if instruction_upper.startswith('FROM'):
            return 0

        # RUN instructions with package management
        if instruction_upper.startswith('RUN'):
            if any(pkg_mgr in instruction.lower() for pkg_mgr in
                   ['apt-get install', 'yum install', 'apk add', 'pip install', 'npm install']):
                # Count individual RUN commands vs combined commands
                if '&&' in instruction:
                    # Combined command - more efficient, smaller total size
                    return 8 * 1024 * 1024  # 8MB for combined package install
                else:
                    # Individual commands are less efficient
                    return 12 * 1024 * 1024  # 12MB for individual package install
            elif 'apt-get update' in instruction.lower():
                return 15 * 1024 * 1024  # 15MB for apt-get update alone
            else:
                # Other RUN commands (file operations, etc.)
                return 1024 * 1024  # 1MB estimate

        # COPY/ADD instructions
        if instruction_upper.startswith(('COPY', 'ADD')):
            # Hard to estimate without actual files, use moderate estimate
            return 5 * 1024 * 1024  # 5MB estimate

        # Other instructions (WORKDIR, ENV, EXPOSE, etc.) typically add minimal size
        return 0

    def compare_dockerfile_efficiency(self, original_dockerfile: str,
                                    optimized_dockerfile: str) -> Dict[str, int]:
        """Compare efficiency between original and optimized Dockerfiles."""
        original_analysis = self.get_layer_sizes_for_dockerfile(original_dockerfile)
        optimized_analysis = self.get_layer_sizes_for_dockerfile(optimized_dockerfile)

        original_layers = len(original_analysis.layers)
        optimized_layers = len(optimized_analysis.layers)
        layer_reduction = original_layers - optimized_layers

        original_size = original_analysis.total_size
        optimized_size = optimized_analysis.total_size
        size_reduction = original_size - optimized_size

        return {
            "original_layers": original_layers,
            "optimized_layers": optimized_layers,
            "layer_reduction": layer_reduction,
            "original_estimated_size": original_size,
            "optimized_estimated_size": optimized_size,
            "estimated_size_reduction": size_reduction
        }
