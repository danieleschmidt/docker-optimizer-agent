"""Main Docker optimization engine."""

import re
from typing import Any, Dict, List

from .models import (
    DockerfileAnalysis,
    LayerOptimization,
    OptimizationResult,
    SecurityFix,
)
from .parser import DockerfileParser
from .security import SecurityAnalyzer
from .size_estimator import SizeEstimator


class DockerfileOptimizer:
    """Main class for analyzing and optimizing Dockerfiles."""

    def __init__(self) -> None:
        """Initialize the optimizer with its components."""
        self.parser = DockerfileParser()
        self.security_analyzer = SecurityAnalyzer()
        self.size_estimator = SizeEstimator()

    def analyze_dockerfile(self, dockerfile_content: str) -> DockerfileAnalysis:
        """Analyze a Dockerfile for security issues and optimization opportunities.

        Args:
            dockerfile_content: The content of the Dockerfile to analyze

        Returns:
            DockerfileAnalysis: Analysis results
        """
        parsed = self.parser.parse(dockerfile_content)

        # Extract base image
        base_image = self._extract_base_image(dockerfile_content)

        # Count layers (RUN, COPY, ADD instructions create layers)
        layer_instructions = ["RUN", "COPY", "ADD"]
        total_layers = sum(
            1
            for instruction in parsed
            if instruction["instruction"] in layer_instructions
        )

        # Identify security issues
        security_issues = self._identify_security_issues(dockerfile_content, parsed)

        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            dockerfile_content, parsed
        )

        # Estimate size
        estimated_size = self._estimate_size(dockerfile_content)

        return DockerfileAnalysis(
            base_image=base_image,
            total_layers=total_layers,
            security_issues=security_issues,
            optimization_opportunities=optimization_opportunities,
            estimated_size=estimated_size,
        )

    def optimize_dockerfile(self, dockerfile_content: str) -> OptimizationResult:
        """Optimize a Dockerfile for security, size, and best practices.

        Args:
            dockerfile_content: The content of the Dockerfile to optimize

        Returns:
            OptimizationResult: Optimization results and optimized Dockerfile
        """
        # First analyze the original
        analysis = self.analyze_dockerfile(dockerfile_content)

        # Apply optimizations
        optimized_content = dockerfile_content
        security_fixes = []
        layer_optimizations = []

        # Security optimizations
        if analysis.has_security_issues:
            optimized_content, sec_fixes = self._apply_security_fixes(
                optimized_content, analysis.security_issues
            )
            security_fixes.extend(sec_fixes)

        # Layer optimizations
        if analysis.has_optimization_opportunities:
            optimized_content, layer_opts = self._apply_layer_optimizations(
                optimized_content
            )
            layer_optimizations.extend(layer_opts)

        # Base image optimization
        optimized_content = self._optimize_base_image(optimized_content)

        # Generate explanation
        explanation = self._generate_explanation(security_fixes, layer_optimizations)

        return OptimizationResult(
            original_size=analysis.estimated_size or "Unknown",
            optimized_size=self._estimate_size(optimized_content),
            security_fixes=security_fixes,
            explanation=explanation,
            optimized_dockerfile=optimized_content,
            layer_optimizations=layer_optimizations,
        )

    def analyze_and_optimize(self, dockerfile_path: str) -> OptimizationResult:
        """Analyze and optimize a Dockerfile from a file path.

        Args:
            dockerfile_path: Path to the Dockerfile

        Returns:
            OptimizationResult: Complete optimization results
        """
        with open(dockerfile_path, encoding="utf-8") as f:
            dockerfile_content = f.read()

        return self.optimize_dockerfile(dockerfile_content)

    def _extract_base_image(self, dockerfile_content: str) -> str:
        """Extract the base image from a Dockerfile."""
        lines = dockerfile_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("FROM "):
                # Handle multi-stage builds
                parts = line.split()
                if len(parts) >= 2:
                    base_image = parts[1]
                    # Remove alias if present (FROM image AS alias)
                    if "AS" in line.upper():
                        return base_image
                    return base_image
        return "unknown"

    def _identify_security_issues(
        self, dockerfile_content: str, parsed: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify security issues in the Dockerfile."""
        issues = []

        # Check for latest tag
        if (
            ":latest" in dockerfile_content
            or "FROM ubuntu" in dockerfile_content
            and ":" not in dockerfile_content
        ):
            issues.append("Using 'latest' tag is not recommended for production")

        # Check for root user
        if "USER root" in dockerfile_content or "USER 0" in dockerfile_content:
            issues.append("Running as root user poses security risks")

        # Check if USER directive is missing
        if "USER " not in dockerfile_content:
            issues.append("No USER directive found - container will run as root")

        # Check for package cache cleanup
        if (
            "apt-get update" in dockerfile_content
            and "rm -rf /var/lib/apt/lists/*" not in dockerfile_content
        ):
            issues.append(
                "Package cache not cleaned up, increases image size and attack surface"
            )

        return issues

    def _identify_optimization_opportunities(
        self, dockerfile_content: str, parsed: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify optimization opportunities in the Dockerfile."""
        opportunities = []

        # Count RUN statements
        run_count = dockerfile_content.count("RUN ")
        if run_count > 2:
            opportunities.append(
                f"Multiple RUN statements ({run_count}) can be combined to reduce layers"
            )

        # Check for package manager optimizations
        if (
            "apt-get install" in dockerfile_content
            and "--no-install-recommends" not in dockerfile_content
        ):
            opportunities.append(
                "Use --no-install-recommends flag to reduce package installation size"
            )

        # Check for multi-stage build opportunities
        if (
            "gcc" in dockerfile_content
            or "make" in dockerfile_content
            or "build-essential" in dockerfile_content
        ):
            if "FROM " in dockerfile_content and dockerfile_content.count("FROM ") == 1:
                opportunities.append(
                    "Consider multi-stage build to exclude build dependencies from final image"
                )

        return opportunities

    def _apply_security_fixes(
        self, dockerfile_content: str, security_issues: List[str]
    ) -> tuple[str, List[SecurityFix]]:
        """Apply security fixes to the Dockerfile."""
        fixes = []
        content = dockerfile_content

        # Fix latest tag
        if "latest" in security_issues[0] if security_issues else False:
            if "ubuntu:latest" in content:
                content = content.replace("ubuntu:latest", "ubuntu:22.04-slim")
                fixes.append(
                    SecurityFix(
                        vulnerability="Unspecified version tag",
                        severity="MEDIUM",
                        description="Using 'latest' tag is unpredictable and insecure",
                        fix="Changed to ubuntu:22.04-slim for specific version and smaller size",
                    )
                )
            elif "alpine:latest" in content:
                content = content.replace("alpine:latest", "alpine:3.18")
                fixes.append(
                    SecurityFix(
                        vulnerability="Unspecified version tag",
                        severity="MEDIUM",
                        description="Using 'latest' tag is unpredictable and insecure",
                        fix="Changed to alpine:3.18 for specific version",
                    )
                )

        # Add non-root user if missing
        if any("USER directive" in issue for issue in security_issues):
            # Add before the last instruction
            lines = content.strip().split("\n")
            # Insert USER directive before the last line
            lines.insert(-1, "USER 1001:1001")
            content = "\n".join(lines)
            fixes.append(
                SecurityFix(
                    vulnerability="Container running as root",
                    severity="HIGH",
                    description="Container processes running as root pose security risks",
                    fix="Added USER directive to run as non-root user (1001:1001)",
                )
            )

        return content, fixes

    def _apply_layer_optimizations(
        self, dockerfile_content: str
    ) -> tuple[str, List[LayerOptimization]]:
        """Apply layer optimizations to reduce image size."""
        optimizations = []
        content = dockerfile_content

        # Combine consecutive RUN statements
        lines = content.split("\n")
        new_lines = []
        run_commands = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("RUN "):
                command = stripped[4:]  # Remove 'RUN '
                run_commands.append(command)
            else:
                if run_commands:
                    # Combine all collected RUN commands
                    if len(run_commands) > 1:
                        combined = "RUN " + " && \\\n    ".join(run_commands)
                        new_lines.append(combined)
                        optimizations.append(
                            LayerOptimization(
                                original_instruction=f"{len(run_commands)} separate RUN statements",
                                optimized_instruction=combined,
                                reasoning=f"Combined {len(run_commands)} RUN statements into one to reduce layers",
                            )
                        )
                    else:
                        new_lines.append("RUN " + run_commands[0])
                    run_commands = []
                new_lines.append(line)

        # Handle any remaining RUN commands
        if run_commands:
            if len(run_commands) > 1:
                combined = "RUN " + " && \\\n    ".join(run_commands)
                new_lines.append(combined)
                optimizations.append(
                    LayerOptimization(
                        original_instruction=f"{len(run_commands)} separate RUN statements",
                        optimized_instruction=combined,
                        reasoning=f"Combined {len(run_commands)} RUN statements into one to reduce layers",
                    )
                )
            else:
                new_lines.append("RUN " + run_commands[0])

        content = "\n".join(new_lines)

        # Add package manager optimizations
        if "apt-get install" in content and "--no-install-recommends" not in content:
            content = content.replace(
                "apt-get install -y", "apt-get install -y --no-install-recommends"
            )
            content = content.replace(
                "apt-get install", "apt-get install --no-install-recommends"
            )
            if "&& rm -rf /var/lib/apt/lists/*" not in content:
                content = (
                    content.replace(
                        "apt-get install -y --no-install-recommends",
                        "apt-get install -y --no-install-recommends",
                    )
                    + " && rm -rf /var/lib/apt/lists/*"
                )

        return content, optimizations

    def _optimize_base_image(self, dockerfile_content: str) -> str:
        """Optimize the base image selection."""
        content = dockerfile_content

        # Suggest slimmer alternatives
        if "FROM ubuntu:" in content and "slim" not in content:
            content = re.sub(r"FROM ubuntu:(\d+\.\d+)", r"FROM ubuntu:\1-slim", content)

        return content

    def _generate_explanation(
        self,
        security_fixes: List[SecurityFix],
        layer_optimizations: List[LayerOptimization],
    ) -> str:
        """Generate a human-readable explanation of optimizations."""
        explanations = []

        if security_fixes:
            explanations.append(f"Applied {len(security_fixes)} security improvements")

        if layer_optimizations:
            explanations.append(
                f"Applied {len(layer_optimizations)} layer optimizations"
            )

        if not explanations:
            explanations.append(
                "No major optimizations needed - Dockerfile follows best practices"
            )

        return "; ".join(explanations)

    def _estimate_size(self, dockerfile_content: str) -> str:
        """Estimate the size of the resulting Docker image."""
        # Basic size estimation based on base image and packages
        base_image = self._extract_base_image(dockerfile_content)

        base_sizes = {
            "alpine": 5,
            "ubuntu": 70,
            "debian": 120,
            "centos": 200,
            "node": 150,
            "python": 100,
        }

        estimated_mb = 50  # Default
        for image, size in base_sizes.items():
            if image in base_image.lower():
                estimated_mb = size
                break

        # Add estimated size for packages
        package_indicators = ["curl", "wget", "git", "vim", "build-essential", "gcc"]
        for indicator in package_indicators:
            if indicator in dockerfile_content:
                estimated_mb += 20

        # Reduce estimate for slim/alpine variants
        if "slim" in base_image or "alpine" in base_image:
            estimated_mb = int(estimated_mb * 0.6)

        return f"{estimated_mb}MB"
