"""Security analysis for Dockerfiles."""

import re
from typing import Any, Dict, List

from .models import SecurityFix


class SecurityAnalyzer:
    """Analyzes Dockerfiles for security vulnerabilities."""

    def __init__(self) -> None:
        """Initialize the security analyzer."""
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r'curl.*\|\s*bash',  # Pipe to bash
            r'wget.*\|\s*sh',    # Pipe to shell
            r'eval\s*\$',        # Dynamic evaluation
            r'rm\s+-rf\s+/',     # Dangerous rm commands
            r'chmod\s+777',      # Overly permissive permissions
        ]
        
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

        # Check for dangerous patterns
        dangerous_patterns_found = self._check_dangerous_patterns(dockerfile_content)
        fixes.extend(dangerous_patterns_found)

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

    def _check_dangerous_patterns(self, dockerfile_content: str) -> List[SecurityFix]:
        """Check for dangerous command patterns."""
        fixes = []
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, dockerfile_content, re.IGNORECASE):
                fixes.append(
                    SecurityFix(
                        vulnerability="Dangerous command pattern",
                        severity="CRITICAL",
                        description=f"Detected potentially dangerous pattern: {pattern}",
                        fix="Review and replace with safer alternatives",
                    )
                )
        
        return fixes

    @staticmethod
    def sanitize_dockerfile_content(content: str) -> str:
        """Sanitize Dockerfile content for safe processing."""
        # Remove any potentially malicious content
        sanitized = content
        
        # Remove comments that might contain injection attempts
        sanitized = re.sub(r'#.*$', '', sanitized, flags=re.MULTILINE)
        
        # Limit line length to prevent buffer overflows
        lines = sanitized.split('\n')
        sanitized_lines = [line[:1000] for line in lines if len(line.strip()) > 0]
        
        return '\n'.join(sanitized_lines[:500])  # Limit total lines
