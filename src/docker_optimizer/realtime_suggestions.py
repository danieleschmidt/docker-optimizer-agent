"""Real-time optimization suggestions engine."""

from enum import Enum
from typing import Dict, List

from .models import OptimizationSuggestion, SuggestionContext
from .parser import DockerfileParser


class ProjectType(Enum):
    """Supported project types for smart suggestions."""

    PYTHON = "python"
    NODEJS = "nodejs"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    UNKNOWN = "unknown"


class RealtimeSuggestionEngine:
    """Engine for generating real-time optimization suggestions."""

    def __init__(self) -> None:
        """Initialize the suggestion engine."""
        self.parser = DockerfileParser()

        # Language-specific patterns for project detection
        self.language_patterns = {
            ProjectType.PYTHON: {
                "base_images": ["python", "pypy"],
                "files": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
                "commands": ["pip install", "python", "pip3"],
            },
            ProjectType.NODEJS: {
                "base_images": ["node", "npm"],
                "files": ["package.json", "package-lock.json", "yarn.lock"],
                "commands": ["npm install", "yarn", "node"],
            },
            ProjectType.GO: {
                "base_images": ["golang", "go"],
                "files": ["go.mod", "go.sum"],
                "commands": ["go build", "go mod", "go get"],
            },
            ProjectType.JAVA: {
                "base_images": ["openjdk", "maven", "gradle"],
                "files": ["pom.xml", "build.gradle", "gradle.build"],
                "commands": ["mvn", "gradle", "java -jar"],
            },
            ProjectType.RUST: {
                "base_images": ["rust"],
                "files": ["Cargo.toml", "Cargo.lock"],
                "commands": ["cargo build", "cargo install"],
            },
        }

    def detect_project_type(self, dockerfile_content: str) -> ProjectType:
        """Detect project type based on Dockerfile content.

        Args:
            dockerfile_content: Content of the Dockerfile

        Returns:
            ProjectType: Detected project type
        """
        content_lower = dockerfile_content.lower()

        # Check base images and commands for language indicators
        for project_type, patterns in self.language_patterns.items():
            # Check base images
            for base_image in patterns["base_images"]:
                if f"from {base_image}" in content_lower:
                    return project_type

            # Check for language-specific commands
            for command in patterns["commands"]:
                if command in content_lower:
                    return project_type

            # Check for language-specific files
            for file_pattern in patterns["files"]:
                if file_pattern in content_lower:
                    return project_type

        return ProjectType.UNKNOWN

    def generate_realtime_suggestions(
        self, dockerfile_content: str
    ) -> List[OptimizationSuggestion]:
        """Generate real-time optimization suggestions for a Dockerfile.

        Args:
            dockerfile_content: Content of the Dockerfile to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Detect project type for context-aware suggestions
        project_type = self.detect_project_type(dockerfile_content)

        # Generate different types of suggestions
        suggestions.extend(self._suggest_base_image_optimization(dockerfile_content))
        suggestions.extend(self._suggest_security_improvements(dockerfile_content))
        suggestions.extend(self._suggest_layer_optimization(dockerfile_content))
        suggestions.extend(
            self._suggest_project_specific_optimizations(
                dockerfile_content, project_type
            )
        )
        suggestions.extend(self._suggest_best_practices(dockerfile_content))

        # Sort by priority (CRITICAL -> HIGH -> MEDIUM -> LOW)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 4))

        return suggestions

    def generate_interactive_suggestions(
        self, dockerfile_content: str, context: SuggestionContext
    ) -> List[OptimizationSuggestion]:
        """Generate context-aware suggestions for interactive mode.

        Args:
            dockerfile_content: Current Dockerfile content
            context: Context information for targeted suggestions

        Returns:
            List of context-aware suggestions
        """
        all_suggestions = self.generate_realtime_suggestions(dockerfile_content)

        # Filter based on context
        filtered_suggestions = []

        for suggestion in all_suggestions:
            # Skip suggestions that were already shown
            if suggestion.suggestion_type in context.previous_suggestions:
                continue

            # Apply line-based filtering for interactive mode
            if context.current_line > 0:
                # Prioritize suggestions for current and nearby lines
                line_distance = abs(suggestion.line_number - context.current_line)
                if line_distance <= 2:  # Show suggestions for current +/- 2 lines
                    filtered_suggestions.append(suggestion)
                elif suggestion.priority in ["CRITICAL", "HIGH"]:
                    # Always show critical/high priority suggestions
                    filtered_suggestions.append(suggestion)
            else:
                filtered_suggestions.append(suggestion)

        # Limit to top 5 suggestions for interactive display
        return filtered_suggestions[:5]

    def generate_smart_dockerfile(
        self, project_type: ProjectType, project_files: Dict[str, str]
    ) -> str:
        """Generate a smart, optimized Dockerfile based on project type.

        Args:
            project_type: Detected project type
            project_files: Dictionary of project files and their content

        Returns:
            Generated Dockerfile content
        """
        if project_type == ProjectType.PYTHON:
            return self._generate_python_dockerfile(project_files)
        elif project_type == ProjectType.NODEJS:
            return self._generate_nodejs_dockerfile(project_files)
        elif project_type == ProjectType.GO:
            return self._generate_go_dockerfile(project_files)
        else:
            return self._generate_generic_dockerfile()

    def _suggest_base_image_optimization(
        self, dockerfile_content: str
    ) -> List[OptimizationSuggestion]:
        """Suggest base image optimizations."""
        suggestions = []
        lines = dockerfile_content.split("\n")

        for i, line in enumerate(lines):
            if line.strip().upper().startswith("FROM"):
                # Check for latest tag
                if ":latest" in line or (":" not in line and "@" not in line):
                    suggestions.append(
                        OptimizationSuggestion(
                            line_number=i + 1,
                            suggestion_type="security",
                            priority="HIGH",
                            message="Use specific version instead of 'latest' tag",
                            explanation="Using 'latest' tag can lead to unpredictable builds and security vulnerabilities",
                            fix_example="FROM ubuntu:20.04 or FROM python:3.11-slim",
                        )
                    )

                # Suggest language-specific base images
                if "ubuntu" in line.lower() or "debian" in line.lower():
                    project_type = self.detect_project_type(dockerfile_content)
                    if project_type != ProjectType.UNKNOWN:
                        language_image = {
                            ProjectType.PYTHON: "python:3.11-slim",
                            ProjectType.NODEJS: "node:18-slim",
                            ProjectType.GO: "golang:1.20-alpine",
                            ProjectType.JAVA: "openjdk:11-jre-slim",
                        }.get(project_type)

                        if language_image:
                            suggestions.append(
                                OptimizationSuggestion(
                                    line_number=i + 1,
                                    suggestion_type="optimization",
                                    priority="MEDIUM",
                                    message=f"Consider using {language_image} for better optimization",
                                    explanation="Language-specific base images are more efficient than generic OS images",
                                    fix_example=f"FROM {language_image}",
                                )
                            )

        return suggestions

    def _suggest_security_improvements(
        self, dockerfile_content: str
    ) -> List[OptimizationSuggestion]:
        """Suggest security improvements."""
        suggestions = []
        lines = dockerfile_content.split("\n")

        has_user = any(
            "USER" in line.upper() and not line.strip().upper().startswith("#")
            for line in lines
        )
        has_root_user = any("USER root" in line for line in lines)

        for i, line in enumerate(lines):
            # Check for root user
            if "USER ROOT" in line.upper():
                suggestions.append(
                    OptimizationSuggestion(
                        line_number=i + 1,
                        suggestion_type="security",
                        priority="HIGH",
                        message="Avoid running as root user",
                        explanation="Running containers as root increases security risks",
                        fix_example="USER appuser",
                    )
                )

            # Check for insecure downloads
            if "curl" in line.lower() and "http://" in line:
                suggestions.append(
                    OptimizationSuggestion(
                        line_number=i + 1,
                        suggestion_type="security",
                        priority="CRITICAL",
                        message="Avoid insecure HTTP downloads",
                        explanation="HTTP downloads can be intercepted and modified",
                        fix_example="Use HTTPS URLs or verify downloaded files",
                    )
                )

            # Check for bash piping from curl
            if "curl" in line.lower() and "| bash" in line.lower():
                suggestions.append(
                    OptimizationSuggestion(
                        line_number=i + 1,
                        suggestion_type="security",
                        priority="CRITICAL",
                        message="Avoid piping downloads directly to bash",
                        explanation="This practice can execute malicious code without verification",
                        fix_example="Download, verify, then execute scripts separately",
                    )
                )

        # Suggest non-root user if not present
        if not has_user or has_root_user:
            suggestions.append(
                OptimizationSuggestion(
                    line_number=len(lines),
                    suggestion_type="security",
                    priority="HIGH",
                    message="Add non-root user for security",
                    explanation="Running as non-root user reduces attack surface",
                    fix_example="RUN adduser --disabled-password --gecos '' appuser\nUSER appuser",
                )
            )

        return suggestions

    def _suggest_layer_optimization(
        self, dockerfile_content: str
    ) -> List[OptimizationSuggestion]:
        """Suggest layer optimizations."""
        suggestions = []
        lines = dockerfile_content.split("\n")

        # Find consecutive RUN commands
        consecutive_runs = []
        current_run_block = []

        for i, line in enumerate(lines):
            if line.strip().upper().startswith("RUN"):
                current_run_block.append((i + 1, line))
            else:
                if len(current_run_block) > 1:
                    consecutive_runs.append(current_run_block)
                current_run_block = []

        if len(current_run_block) > 1:
            consecutive_runs.append(current_run_block)

        # Suggest combining consecutive RUN commands
        for run_block in consecutive_runs:
            if len(run_block) > 1:
                first_line = run_block[0][0]
                suggestions.append(
                    OptimizationSuggestion(
                        line_number=first_line,
                        suggestion_type="optimization",
                        priority="MEDIUM",
                        message=f"Combine {len(run_block)} RUN commands into one",
                        explanation="Combining RUN commands reduces image layers and size",
                        fix_example="RUN cmd1 && \\\n    cmd2 && \\\n    cmd3",
                    )
                )

        return suggestions

    def _suggest_project_specific_optimizations(
        self, dockerfile_content: str, project_type: ProjectType
    ) -> List[OptimizationSuggestion]:
        """Suggest project-specific optimizations."""
        suggestions = []
        lines = dockerfile_content.split("\n")

        if project_type == ProjectType.PYTHON:
            # Python-specific suggestions
            for i, line in enumerate(lines):
                if "pip install" in line.lower():
                    if "--no-cache-dir" not in line:
                        suggestions.append(
                            OptimizationSuggestion(
                                line_number=i + 1,
                                suggestion_type="optimization",
                                priority="MEDIUM",
                                message="Add --no-cache-dir to pip install",
                                explanation="Reduces image size by not storing pip cache",
                                fix_example="RUN pip install --no-cache-dir -r requirements.txt",
                            )
                        )

                    if (
                        "requirements.txt" in line
                        and "COPY requirements.txt" not in "\n".join(lines[:i])
                    ):
                        suggestions.append(
                            OptimizationSuggestion(
                                line_number=i + 1,
                                suggestion_type="optimization",
                                priority="LOW",
                                message="Copy requirements.txt before other files for better caching",
                                explanation="Copying requirements first enables Docker layer caching for dependencies",
                                fix_example="COPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .",
                            )
                        )

        elif project_type == ProjectType.NODEJS:
            # Node.js-specific suggestions
            for i, line in enumerate(lines):
                if "npm install" in line.lower():
                    if "--production" not in line and "--only=production" not in line:
                        suggestions.append(
                            OptimizationSuggestion(
                                line_number=i + 1,
                                suggestion_type="optimization",
                                priority="MEDIUM",
                                message="Use npm ci --only=production for production builds",
                                explanation="Installs only production dependencies and is faster",
                                fix_example="RUN npm ci --only=production",
                            )
                        )

        return suggestions

    def _suggest_best_practices(
        self, dockerfile_content: str
    ) -> List[OptimizationSuggestion]:
        """Suggest general best practices."""
        suggestions = []
        lines = dockerfile_content.split("\n")

        has_workdir = any("WORKDIR" in line.upper() for line in lines)
        has_healthcheck = any("HEALTHCHECK" in line.upper() for line in lines)

        # Suggest WORKDIR if not present
        if not has_workdir and any(
            "COPY" in line.upper() or "ADD" in line.upper() for line in lines
        ):
            suggestions.append(
                OptimizationSuggestion(
                    line_number=len(lines) // 2,
                    suggestion_type="best_practice",
                    priority="LOW",
                    message="Add WORKDIR instruction",
                    explanation="WORKDIR sets a clear working directory for subsequent instructions",
                    fix_example="WORKDIR /app",
                )
            )

        # Suggest health check for web applications
        if not has_healthcheck:
            if any(
                port in dockerfile_content
                for port in ["EXPOSE", "8000", "3000", "5000", "80", "443"]
            ):
                suggestions.append(
                    OptimizationSuggestion(
                        line_number=len(lines),
                        suggestion_type="best_practice",
                        priority="LOW",
                        message="Consider adding a health check",
                        explanation="Health checks help orchestrators monitor container health",
                        fix_example="HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://localhost:8000/health || exit 1",
                    )
                )

        return suggestions

    def _generate_python_dockerfile(self, project_files: Dict[str, str]) -> str:
        """Generate optimized Python Dockerfile."""
        has_requirements = "requirements.txt" in project_files

        dockerfile = """# Multi-stage Python build
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

"""

        if has_requirements:
            dockerfile += """# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

"""

        dockerfile += """# Runtime stage
FROM python:3.11-slim

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Switch to non-root user
USER appuser

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start application
CMD ["python", "app.py"]
"""

        return dockerfile

    def _generate_nodejs_dockerfile(self, project_files: Dict[str, str]) -> str:
        """Generate optimized Node.js Dockerfile."""
        has_package_json = "package.json" in project_files

        dockerfile = """# Multi-stage Node.js build
FROM node:18-slim as builder

WORKDIR /app

"""

        if has_package_json:
            dockerfile += """# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

"""

        dockerfile += """# Runtime stage
FROM node:18-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /app/node_modules ./node_modules

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:3000/health || exit 1

# Start application
CMD ["npm", "start"]
"""

        return dockerfile

    def _generate_go_dockerfile(self, project_files: Dict[str, str]) -> str:
        """Generate optimized Go Dockerfile."""
        dockerfile = """# Multi-stage Go build
FROM golang:1.20-alpine AS builder

WORKDIR /app

# Copy go modules
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

# Runtime stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -S appuser && adduser -S appuser -G appuser

WORKDIR /root/

# Copy binary from builder
COPY --from=builder /app/app .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD ./app --health-check || exit 1

# Start application
CMD ["./app"]
"""

        return dockerfile

    def _generate_generic_dockerfile(self) -> str:
        """Generate generic optimized Dockerfile."""
        return """FROM ubuntu:20.04

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Start application
CMD ["./start.sh"]
"""
