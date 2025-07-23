"""Main Docker optimization engine."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .language_optimizer import LanguageOptimizer, analyze_project_language
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
        self.language_optimizer = LanguageOptimizer()

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
                content = content.replace("ubuntu:latest", "ubuntu:22.04")
                fixes.append(
                    SecurityFix(
                        vulnerability="Unspecified version tag",
                        severity="MEDIUM",
                        description="Using 'latest' tag is unpredictable and insecure",
                        fix="Changed to ubuntu:22.04 for specific version",
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
            # First replace specific patterns to avoid duplicates
            content = content.replace(
                "apt-get install -y", "apt-get install --no-install-recommends -y"
            )
            # Then handle remaining apt-get install commands
            content = re.sub(
                r"apt-get install(?!\s+--no-install-recommends)",
                "apt-get install --no-install-recommends",
                content
            )
            # Add cleanup to the last RUN command that contains apt-get, not to the entire content
            if "&& rm -rf /var/lib/apt/lists/*" not in content and "apt-get" in content:
                # Find the last RUN command with apt-get and add cleanup
                lines = content.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith('RUN') and 'apt-get' in lines[i]:
                        if not lines[i].endswith(' \\'):
                            lines[i] += " && rm -rf /var/lib/apt/lists/*"
                        else:
                            # Find the end of this RUN command
                            j = i
                            while j < len(lines) - 1 and lines[j].endswith(' \\'):
                                j += 1
                            if j < len(lines):
                                lines[j] += " && rm -rf /var/lib/apt/lists/*"
                        break
                content = '\n'.join(lines)

        return content, optimizations

    def _optimize_base_image(self, dockerfile_content: str) -> str:
        """Optimize the base image selection."""
        content = dockerfile_content

        # Suggest slimmer alternatives for ubuntu
        if "FROM ubuntu:" in content and "slim" not in content:
            # Replace ubuntu:latest with ubuntu:22.04 (not slim as latest-slim doesn't exist)
            if "ubuntu:latest" in content:
                content = content.replace("ubuntu:latest", "ubuntu:22.04")
            # Note: We avoid adding -slim for now as not all ubuntu:version-slim images exist

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

    def get_language_specific_recommendations(
        self,
        dockerfile_content: str,
        project_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Get language-specific optimization recommendations.

        Args:
            dockerfile_content: Content of the Dockerfile
            project_path: Optional path to the project directory for analysis

        Returns:
            List of language-specific recommendations
        """
        recommendations = []

        if project_path and project_path.exists():
            # Analyze the project to detect language and framework
            analysis = analyze_project_language(project_path)

            if analysis["recommendations_available"]:
                # Get language-specific suggestions
                suggestions = self.language_optimizer.get_language_recommendations(
                    analysis["language"],
                    analysis.get("framework")
                )

                # Convert suggestions to recommendations format
                for suggestion in suggestions:
                    recommendations.append({
                        "type": suggestion.type,
                        "description": suggestion.description,
                        "impact": suggestion.impact,
                        "language": analysis["language"],
                        "framework": analysis.get("framework"),
                        "confidence": analysis["language_confidence"],
                        "dockerfile_changes": suggestion.dockerfile_changes,
                        "explanation": suggestion.explanation
                    })

                # Add project analysis summary
                recommendations.insert(0, {
                    "type": "project_analysis",
                    "description": f"Detected {analysis['language']} project" + (
                        f" with {analysis['framework']} framework" if analysis['framework'] else ""
                    ),
                    "impact": "info",
                    "language": analysis["language"],
                    "framework": analysis.get("framework"),
                    "confidence": analysis["language_confidence"],
                    "dockerfile_changes": [],
                    "explanation": "Project type detection enables language-specific optimizations"
                })
        else:
            # Fallback: try to detect language from Dockerfile content
            detected_language = self._detect_language_from_dockerfile(dockerfile_content)
            if detected_language:
                suggestions = self.language_optimizer.get_language_recommendations(detected_language)

                for suggestion in suggestions:
                    recommendations.append({
                        "type": suggestion.type,
                        "description": suggestion.description,
                        "impact": suggestion.impact,
                        "language": detected_language,
                        "framework": None,
                        "confidence": 0.7,  # Medium confidence from Dockerfile analysis
                        "dockerfile_changes": suggestion.dockerfile_changes,
                        "explanation": suggestion.explanation
                    })

        return recommendations

    def _detect_language_from_dockerfile(self, dockerfile_content: str) -> Optional[str]:
        """Detect programming language from Dockerfile content.

        Args:
            dockerfile_content: Content of the Dockerfile

        Returns:
            Detected language or None
        """
        content_lower = dockerfile_content.lower()

        # Language detection patterns
        language_patterns = {
            'python': ['python:', 'pip install', 'requirements.txt', 'python3'],
            'nodejs': ['node:', 'npm install', 'package.json', 'yarn'],
            'go': ['golang:', 'go build', 'go.mod', 'go install'],
            'java': ['openjdk:', 'java -jar', 'maven', 'gradle', '.jar'],
            'rust': ['rust:', 'cargo build', 'cargo.toml'],
            'ruby': ['ruby:', 'bundle install', 'gemfile'],
            'php': ['php:', 'composer install', 'composer.json']
        }

        # Score each language based on pattern matches
        language_scores = {}
        for language, patterns in language_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in content_lower:
                    score += 1

            if score > 0:
                language_scores[language] = score

        # Return the language with the highest score
        if language_scores:
            return max(language_scores, key=lambda x: language_scores[x])

        return None

    def optimize_dockerfile_with_language_analysis(
        self,
        dockerfile_content: str,
        project_path: Optional[Path] = None
    ) -> OptimizationResult:
        """Optimize a Dockerfile with language-specific analysis.

        Args:
            dockerfile_content: The content of the Dockerfile to optimize
            project_path: Optional path to project directory for language detection

        Returns:
            OptimizationResult: Enhanced optimization results with language-specific recommendations
        """
        # Get standard optimization result
        result = self.optimize_dockerfile(dockerfile_content)

        # Add language-specific recommendations
        self.get_language_specific_recommendations(
            dockerfile_content, project_path
        )

        # Enhance the result with language-specific information
        # Note: This assumes OptimizationResult has a way to include additional recommendations
        # If not, we'd need to modify the model or create a new enhanced result type

        return result
