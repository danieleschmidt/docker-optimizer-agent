"""Optimization presets and profiles for Docker optimization."""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from .config import Config
from .models import CustomPreset, OptimizationPreset, OptimizationStep

logger = logging.getLogger(__name__)


class PresetType(Enum):
    """Enumeration of available preset types."""

    DEVELOPMENT = "DEVELOPMENT"
    PRODUCTION = "PRODUCTION"
    WEB_APP = "WEB_APP"
    ML = "ML"
    DATA_PROCESSING = "DATA_PROCESSING"


class PresetManager:
    """Manages optimization presets and profiles."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the preset manager.

        Args:
            config: Optional configuration instance
        """
        self.config = config or Config()
        self._presets: Dict[PresetType, OptimizationPreset] = {}
        self._load_builtin_presets()

    def _load_builtin_presets(self) -> None:
        """Load built-in optimization presets."""
        # Development preset - prioritizes build speed and debugging
        dev_optimizations = [
            OptimizationStep(
                name="Use cache mounts",
                description="Use Docker buildkit cache mounts for faster builds",
                dockerfile_change="RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt",
                reasoning="Caches dependencies between builds for faster development iteration",
                priority=1
            ),
            OptimizationStep(
                name="Layer caching optimization",
                description="Order layers to maximize cache hits",
                dockerfile_change="COPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .",
                reasoning="Installing dependencies before copying source code improves layer caching",
                priority=1
            ),
            OptimizationStep(
                name="Development base image",
                description="Use full-featured base image for debugging",
                dockerfile_change="FROM python:3.11 # Full image with debugging tools",
                reasoning="Full images include debugging tools useful during development",
                priority=2
            )
        ]

        self._presets[PresetType.DEVELOPMENT] = OptimizationPreset(
            name="Development",
            description="Optimized for fast builds and debugging capabilities",
            preset_type=PresetType.DEVELOPMENT.value,
            optimizations=dev_optimizations,
            target_use_case="Local development and debugging",
            estimated_size_reduction="Minimal (focus on build speed)",
            security_level="Standard"
        )

        # Production preset - prioritizes security and size
        prod_optimizations = [
            OptimizationStep(
                name="Multi-stage build",
                description="Use multi-stage builds to reduce final image size",
                dockerfile_change="FROM python:3.11-slim AS builder\n# Build stage\nFROM python:3.11-slim AS runtime\n# Runtime stage",
                reasoning="Separates build dependencies from runtime, reducing final image size",
                priority=1
            ),
            OptimizationStep(
                name="Distroless or slim base",
                description="Use minimal base images for security",
                dockerfile_change="FROM python:3.11-slim",
                reasoning="Reduces attack surface by removing unnecessary packages",
                priority=1
            ),
            OptimizationStep(
                name="Non-root user",
                description="Run container as non-root user",
                dockerfile_change="RUN adduser --disabled-password --gecos '' appuser\nUSER appuser",
                reasoning="Improves security by following principle of least privilege",
                priority=1
            ),
            OptimizationStep(
                name="Security scanning integration",
                description="Add vulnerability scanning to build process",
                dockerfile_change="# Add security scanning in CI/CD pipeline",
                reasoning="Catches vulnerabilities early in the development cycle",
                priority=2
            )
        ]

        self._presets[PresetType.PRODUCTION] = OptimizationPreset(
            name="Production",
            description="Optimized for security, size, and performance in production",
            preset_type=PresetType.PRODUCTION.value,
            optimizations=prod_optimizations,
            target_use_case="Production deployments",
            estimated_size_reduction="30-50%",
            security_level="High"
        )

        # Web application preset
        web_optimizations = [
            OptimizationStep(
                name="Static file optimization",
                description="Optimize static file serving",
                dockerfile_change="RUN npm run build\nCOPY --from=builder /app/dist /usr/share/nginx/html",
                reasoning="Pre-built static files improve serving performance",
                priority=1
            ),
            OptimizationStep(
                name="Nginx reverse proxy",
                description="Use nginx for static file serving",
                dockerfile_change="FROM nginx:alpine\nCOPY nginx.conf /etc/nginx/nginx.conf",
                reasoning="Nginx efficiently serves static files and acts as reverse proxy",
                priority=2
            ),
            OptimizationStep(
                name="Health checks",
                description="Add health check endpoints",
                dockerfile_change="HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost/ || exit 1",
                reasoning="Health checks enable proper load balancer integration",
                priority=2
            )
        ]

        self._presets[PresetType.WEB_APP] = OptimizationPreset(
            name="Web Application",
            description="Optimized for web applications with static assets",
            preset_type=PresetType.WEB_APP.value,
            optimizations=web_optimizations,
            target_use_case="Web applications and APIs",
            estimated_size_reduction="20-40%",
            security_level="Standard"
        )

        # Machine Learning preset
        ml_optimizations = [
            OptimizationStep(
                name="CUDA base image",
                description="Use CUDA-enabled base image for GPU workloads",
                dockerfile_change="FROM nvidia/cuda:11.8-runtime-ubuntu20.04",
                reasoning="Enables GPU acceleration for ML workloads",
                priority=1
            ),
            OptimizationStep(
                name="Optimized Python packages",
                description="Use pre-compiled wheels for faster installs",
                dockerfile_change="RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                reasoning="Pre-compiled wheels reduce build time and image size",
                priority=1
            ),
            OptimizationStep(
                name="Model artifact optimization",
                description="Optimize model storage and loading",
                dockerfile_change="COPY --from=builder /app/models /app/models",
                reasoning="Separate model artifacts for efficient caching and updates",
                priority=2
            )
        ]

        self._presets[PresetType.ML] = OptimizationPreset(
            name="Machine Learning",
            description="Optimized for ML workloads with GPU support",
            preset_type=PresetType.ML.value,
            optimizations=ml_optimizations,
            target_use_case="Machine learning and AI applications",
            estimated_size_reduction="15-30%",
            security_level="Standard"
        )

        # Data processing preset
        data_optimizations = [
            OptimizationStep(
                name="JVM optimization",
                description="Optimize JVM settings for data processing",
                dockerfile_change="ENV JAVA_OPTS='-Xmx4g -XX:+UseG1GC'",
                reasoning="Tuned JVM settings improve performance for data processing",
                priority=1
            ),
            OptimizationStep(
                name="Parallel processing setup",
                description="Configure for multi-threaded processing",
                dockerfile_change="ENV SPARK_WORKER_CORES=4\nENV SPARK_WORKER_MEMORY=4g",
                reasoning="Maximizes utilization of available CPU and memory resources",
                priority=1
            ),
            OptimizationStep(
                name="Data volume optimization",
                description="Optimize data volume handling",
                dockerfile_change="VOLUME ['/data']\nWORKDIR /data",
                reasoning="Proper volume handling improves I/O performance",
                priority=2
            )
        ]

        self._presets[PresetType.DATA_PROCESSING] = OptimizationPreset(
            name="Data Processing",
            description="Optimized for big data and batch processing workloads",
            preset_type=PresetType.DATA_PROCESSING.value,
            optimizations=data_optimizations,
            target_use_case="Data processing and analytics",
            estimated_size_reduction="10-25%",
            security_level="Standard"
        )

    def get_preset(self, preset_type: PresetType) -> OptimizationPreset:
        """Get a specific optimization preset.

        Args:
            preset_type: Type of preset to retrieve

        Returns:
            The requested optimization preset

        Raises:
            ValueError: If preset type is not available
        """
        if preset_type not in self._presets:
            raise ValueError(f"Preset type {preset_type} is not available")

        return self._presets[preset_type]

    def list_presets(self) -> List[OptimizationPreset]:
        """List all available optimization presets.

        Returns:
            List of all available presets
        """
        return list(self._presets.values())

    def apply_preset(self, dockerfile_content: str, preset: OptimizationPreset) -> str:
        """Apply an optimization preset to a Dockerfile.

        Args:
            dockerfile_content: Original Dockerfile content
            preset: Preset to apply

        Returns:
            Optimized Dockerfile content
        """
        logger.info(f"Applying preset '{preset.name}' to Dockerfile")

        # Start with original content
        optimized_content = dockerfile_content

        # Apply high priority optimizations first
        high_priority_steps = preset.high_priority_steps

        for step in high_priority_steps:
            # This is a simplified application - in reality, this would be more sophisticated
            if "FROM" in step.dockerfile_change and "FROM" in optimized_content:
                # Replace the FROM line
                lines = optimized_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('FROM'):
                        lines[i] = step.dockerfile_change
                        break
                optimized_content = '\n'.join(lines)
            else:
                # Add the optimization as a comment or instruction
                optimized_content += f"\n\n# {step.name}\n{step.dockerfile_change}"

        return optimized_content

    def create_custom_preset(
        self,
        name: str,
        description: str,
        base_preset: PresetType,
        additional_optimizations: Optional[List[str]] = None,
        disabled_optimizations: Optional[List[str]] = None
    ) -> CustomPreset:
        """Create a custom optimization preset.

        Args:
            name: Name of the custom preset
            description: Description of the custom preset
            base_preset: Base preset to extend
            additional_optimizations: Additional optimizations to include
            disabled_optimizations: Optimizations to disable from base

        Returns:
            Created custom preset
        """
        return CustomPreset(
            name=name,
            description=description,
            base_preset=base_preset.value,
            additional_optimizations=additional_optimizations or [],
            disabled_optimizations=disabled_optimizations or [],
            created_at=datetime.now().isoformat(),
            author="user"
        )

    def save_custom_preset(self, preset: CustomPreset, file_path: Path) -> None:
        """Save a custom preset to file.

        Args:
            preset: Custom preset to save
            file_path: Path to save the preset file
        """
        logger.info(f"Saving custom preset '{preset.name}' to {file_path}")

        preset_data = preset.dict()

        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'w') as f:
                yaml.dump(preset_data, f, default_flow_style=False)
        else:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=2)

    def load_custom_preset(self, file_path: Path) -> CustomPreset:
        """Load a custom preset from file.

        Args:
            file_path: Path to the preset file

        Returns:
            Loaded custom preset
        """
        logger.info(f"Loading custom preset from {file_path}")

        with open(file_path) as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                preset_data = yaml.safe_load(f)
            else:
                preset_data = json.load(f)

        # Keep base_preset as string for consistency
        # preset_data['base_preset'] = PresetType(preset_data['base_preset'])

        return CustomPreset(**preset_data)

    def compare_presets(
        self,
        preset1: OptimizationPreset,
        preset2: OptimizationPreset
    ) -> Dict[str, List[str]]:
        """Compare two optimization presets.

        Args:
            preset1: First preset to compare
            preset2: Second preset to compare

        Returns:
            Dictionary with comparison results
        """
        preset1_names = {opt.name for opt in preset1.optimizations}
        preset2_names = {opt.name for opt in preset2.optimizations}

        return {
            f"{preset1.name.lower()}_only": list(preset1_names - preset2_names),
            f"{preset2.name.lower()}_only": list(preset2_names - preset1_names),
            "common": list(preset1_names & preset2_names)
        }

    def get_preset_recommendations(
        self,
        project_type: str,
        deployment_target: str,
        performance_priority: str
    ) -> List[OptimizationPreset]:
        """Get preset recommendations based on project characteristics.

        Args:
            project_type: Type of project (web, ml, data, etc.)
            deployment_target: Deployment target (cloud, on-prem, edge)
            performance_priority: Performance priority (speed, size, security)

        Returns:
            List of recommended presets
        """
        recommendations = []

        # Map project types to presets
        type_mapping = {
            "web": [PresetType.WEB_APP, PresetType.PRODUCTION],
            "ml": [PresetType.ML, PresetType.PRODUCTION],
            "data": [PresetType.DATA_PROCESSING, PresetType.PRODUCTION],
            "api": [PresetType.WEB_APP, PresetType.PRODUCTION],
            "development": [PresetType.DEVELOPMENT]
        }

        suggested_types = type_mapping.get(project_type.lower(), [PresetType.PRODUCTION])

        for preset_type in suggested_types:
            recommendations.append(self.get_preset(preset_type))

        return recommendations

    def validate_preset(self, preset: OptimizationPreset) -> Dict[str, Union[bool, List[str]]]:
        """Validate an optimization preset.

        Args:
            preset: Preset to validate

        Returns:
            Validation results
        """
        issues = []

        if not preset.optimizations:
            issues.append("Preset has no optimization steps")

        if len(preset.name) < 3:
            issues.append("Preset name is too short")

        # Check for duplicate optimization names
        names = [opt.name for opt in preset.optimizations]
        if len(names) != len(set(names)):
            issues.append("Preset contains duplicate optimization names")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }

    def merge_presets(self, presets: List[OptimizationPreset]) -> OptimizationPreset:
        """Merge multiple presets into one.

        Args:
            presets: List of presets to merge

        Returns:
            Merged optimization preset
        """
        if not presets:
            raise ValueError("Cannot merge empty list of presets")

        merged_optimizations = []
        seen_names = set()

        # Combine optimizations, avoiding duplicates
        for preset in presets:
            for optimization in preset.optimizations:
                if optimization.name not in seen_names:
                    merged_optimizations.append(optimization)
                    seen_names.add(optimization.name)

        merged_name = f"Merged ({', '.join(p.name for p in presets)})"

        return OptimizationPreset(
            name=merged_name,
            description=f"Merged preset combining: {', '.join(p.name for p in presets)}",
            preset_type="MERGED",
            optimizations=merged_optimizations,
            target_use_case="Combined use cases",
            estimated_size_reduction="Varies",
            security_level="Varies"
        )
