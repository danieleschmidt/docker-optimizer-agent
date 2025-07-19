"""Security analysis for Dockerfiles."""

from typing import Any, Dict, List

from .models import SecurityFix


class SecurityAnalyzer:
    """Analyzes Dockerfiles for security vulnerabilities."""

    def __init__(self) -> None:
        """Initialize the security analyzer."""
        self.known_vulnerabilities = {
            "latest_tag": {
                "severity": "MEDIUM",
                "description": "Using latest tag makes builds unpredictable",
                "fix": "Use specific version tags",
            },
            "root_user": {
                "severity": "HIGH",
                "description": "Running as root increases attack surface",
                "fix": "Create and use non-root user",
            },
            "package_cache": {
                "severity": "LOW",
                "description": "Package cache increases image size and attack surface",
                "fix": "Clean package cache after installation",
            },
        }

    def analyze_security(
        self, dockerfile_content: str, parsed_instructions: List[Dict[str, Any]]
    ) -> List[SecurityFix]:
        """Analyze Dockerfile for security issues.

        Args:
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            List of identified security fixes
        """
        fixes = []

        # Check for latest tag usage
        if self._uses_latest_tag(dockerfile_content):
            fixes.append(
                SecurityFix(
                    vulnerability="Latest tag usage",
                    severity="MEDIUM",
                    description="Using 'latest' tag makes builds unpredictable and potentially insecure",
                    fix="Replace with specific version tag (e.g., ubuntu:22.04)",
                )
            )

        # Check for root user
        if self._runs_as_root(parsed_instructions):
            fixes.append(
                SecurityFix(
                    vulnerability="Root user execution",
                    severity="HIGH",
                    description="Container runs as root user by default",
                    fix="Add USER directive with non-root user",
                )
            )

        # Check for package cache cleanup
        if self._has_uncleaned_package_cache(dockerfile_content):
            fixes.append(
                SecurityFix(
                    vulnerability="Package cache retention",
                    severity="LOW",
                    description="Package manager cache not cleaned, increases attack surface",
                    fix="Add cleanup commands after package installation",
                )
            )

        return fixes

    def _uses_latest_tag(self, dockerfile_content: str) -> bool:
        """Check if Dockerfile uses latest tag."""
        return ":latest" in dockerfile_content or (
            "FROM " in dockerfile_content
            and ":" not in dockerfile_content.split("FROM ")[1].split("\n")[0]
        )

    def _runs_as_root(self, instructions: List[Dict[str, Any]]) -> bool:
        """Check if container runs as root."""
        # Look for USER directive
        user_instructions = [
            inst for inst in instructions if inst["instruction"] == "USER"
        ]

        # If no USER directive, runs as root
        if not user_instructions:
            return True

        # Check if explicitly set to root
        for inst in user_instructions:
            if inst["value"].strip() in ["root", "0"]:
                return True

        return False

    def _has_uncleaned_package_cache(self, dockerfile_content: str) -> bool:
        """Check if package cache is not cleaned."""
        has_apt_update = "apt-get update" in dockerfile_content
        has_cleanup = "rm -rf /var/lib/apt/lists/*" in dockerfile_content

        return has_apt_update and not has_cleanup
