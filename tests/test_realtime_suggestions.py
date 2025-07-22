"""Test cases for real-time optimization suggestions."""


import pytest

from docker_optimizer.realtime_suggestions import (
    OptimizationSuggestion,
    ProjectType,
    RealtimeSuggestionEngine,
    SuggestionContext,
)


class TestRealtimeSuggestionEngine:
    """Test cases for real-time optimization suggestion engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RealtimeSuggestionEngine()

    def test_detect_project_type_python(self):
        """Test project type detection for Python projects."""
        dockerfile_content = """
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""
        project_type = self.engine.detect_project_type(dockerfile_content)
        assert project_type == ProjectType.PYTHON

    def test_detect_project_type_nodejs(self):
        """Test project type detection for Node.js projects."""
        dockerfile_content = """
FROM node:18
COPY package.json .
RUN npm install
COPY . .
CMD ["npm", "start"]
"""
        project_type = self.engine.detect_project_type(dockerfile_content)
        assert project_type == ProjectType.NODEJS

    def test_detect_project_type_go(self):
        """Test project type detection for Go projects."""
        dockerfile_content = """
FROM golang:1.20
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o app
CMD ["./app"]
"""
        project_type = self.engine.detect_project_type(dockerfile_content)
        assert project_type == ProjectType.GO

    def test_detect_project_type_unknown(self):
        """Test project type detection for unknown projects."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update
CMD ["echo", "hello"]
"""
        project_type = self.engine.detect_project_type(dockerfile_content)
        assert project_type == ProjectType.UNKNOWN

    def test_generate_realtime_suggestions_python(self):
        """Test real-time suggestions for Python projects."""
        dockerfile_content = """
FROM python:latest
USER root
COPY . /app
RUN pip install requirements.txt
WORKDIR /app
CMD ["python", "app.py"]
"""

        suggestions = self.engine.generate_realtime_suggestions(dockerfile_content)

        # Should detect multiple optimization opportunities
        assert len(suggestions) >= 3

        # Should suggest specific Python version
        version_suggestions = [s for s in suggestions if "specific version" in s.message.lower()]
        assert len(version_suggestions) > 0

        # Should suggest non-root user
        user_suggestions = [s for s in suggestions if "non-root" in s.message.lower()]
        assert len(user_suggestions) > 0

        # Should suggest requirements.txt optimization
        pip_suggestions = [s for s in suggestions if "requirements.txt" in s.message.lower()]
        assert len(pip_suggestions) > 0

    def test_generate_interactive_suggestions(self):
        """Test interactive suggestion generation with progressive hints."""
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3
COPY . /app
"""

        context = SuggestionContext(
            current_line=2,
            has_security_scan=False,
            has_multistage=False
        )

        suggestions = self.engine.generate_interactive_suggestions(
            dockerfile_content, context
        )

        # Should provide context-aware suggestions
        assert len(suggestions) > 0

        # Should prioritize immediate improvements
        high_priority = [s for s in suggestions if s.priority == "HIGH"]
        assert len(high_priority) > 0

    def test_suggest_base_image_optimization(self):
        """Test base image optimization suggestions."""
        dockerfile_content = "FROM ubuntu:latest\nRUN apt-get install python3"

        suggestions = self.engine._suggest_base_image_optimization(dockerfile_content)

        # Should suggest specific version and language-specific base image
        assert len(suggestions) >= 2
        assert any("specific version" in s.message for s in suggestions)
        assert any("python:" in s.message.lower() for s in suggestions)

    def test_suggest_layer_optimization(self):
        """Test layer optimization suggestions."""
        dockerfile_content = """
FROM python:3.11
RUN apt-get update
RUN apt-get install -y curl
RUN pip install requests
"""

        suggestions = self.engine._suggest_layer_optimization(dockerfile_content)

        # Should suggest combining RUN commands
        assert len(suggestions) > 0
        assert any("combine" in s.message.lower() for s in suggestions)

    def test_suggest_security_improvements(self):
        """Test security improvement suggestions."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN curl http://insecure.com/script.sh | bash
"""

        suggestions = self.engine._suggest_security_improvements(dockerfile_content)

        # Should suggest multiple security improvements
        assert len(suggestions) >= 2

        # Should suggest non-root user
        user_suggestions = [s for s in suggestions if "non-root" in s.message.lower()]
        assert len(user_suggestions) > 0

        # Should warn about insecure downloads
        security_suggestions = [s for s in suggestions if "insecure" in s.message.lower()]
        assert len(security_suggestions) > 0

    def test_suggestion_context_aware_filtering(self):
        """Test that suggestions are context-aware and avoid duplicates."""
        dockerfile_content = """
FROM python:3.11-slim
USER appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
"""

        context = SuggestionContext(
            current_line=4,
            has_security_scan=True,
            has_multistage=False
        )

        suggestions = self.engine.generate_interactive_suggestions(
            dockerfile_content, context
        )

        # Should not suggest improvements already implemented
        user_suggestions = [s for s in suggestions if "user" in s.message.lower()]
        version_suggestions = [s for s in suggestions if "version" in s.message.lower()]

        # Should be fewer suggestions since good practices are already followed
        assert len(user_suggestions) == 0  # User is already non-root
        assert len(version_suggestions) == 0  # Version is already specific

    def test_smart_dockerfile_generation(self):
        """Test smart Dockerfile generation based on project detection."""
        project_files = {
            "requirements.txt": "flask==2.3.0\nrequests==2.28.0",
            "app.py": "from flask import Flask\napp = Flask(__name__)",
        }

        dockerfile = self.engine.generate_smart_dockerfile(
            project_type=ProjectType.PYTHON,
            project_files=project_files
        )

        # Should generate a complete, optimized Dockerfile
        assert "FROM python:" in dockerfile
        assert "requirements.txt" in dockerfile
        assert "USER" in dockerfile  # Should include non-root user
        assert "WORKDIR" in dockerfile
        assert "--no-cache-dir" in dockerfile  # Should include pip optimization

    def test_progressive_suggestion_priority(self):
        """Test that suggestions are properly prioritized for progressive display."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y python3 pip
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]
"""

        suggestions = self.engine.generate_realtime_suggestions(dockerfile_content)

        # Should have suggestions with different priorities
        priorities = {s.priority for s in suggestions}
        assert "HIGH" in priorities
        assert "MEDIUM" in priorities

        # High priority should include security issues
        high_priority = [s for s in suggestions if s.priority == "HIGH"]
        security_related = [s for s in high_priority if any(
            keyword in s.message.lower()
            for keyword in ["security", "user", "root", "version"]
        )]
        assert len(security_related) > 0


class TestOptimizationSuggestion:
    """Test optimization suggestion model."""

    def test_optimization_suggestion_creation(self):
        """Test OptimizationSuggestion model creation."""
        suggestion = OptimizationSuggestion(
            line_number=3,
            suggestion_type="security",
            priority="HIGH",
            message="Use non-root user",
            explanation="Running as root increases security risk",
            fix_example="USER appuser"
        )

        assert suggestion.line_number == 3
        assert suggestion.suggestion_type == "security"
        assert suggestion.priority == "HIGH"
        assert suggestion.message == "Use non-root user"
        assert suggestion.explanation == "Running as root increases security risk"
        assert suggestion.fix_example == "USER appuser"

    def test_suggestion_priority_validation(self):
        """Test suggestion priority validation."""
        # Valid priority should work
        suggestion = OptimizationSuggestion(
            line_number=1,
            suggestion_type="optimization",
            priority="MEDIUM",
            message="Test message",
            explanation="Test explanation",
            fix_example="Test fix"
        )
        assert suggestion.priority == "MEDIUM"

        # Invalid priority should raise error
        with pytest.raises(ValueError):
            OptimizationSuggestion(
                line_number=1,
                suggestion_type="optimization",
                priority="INVALID",
                message="Test message",
                explanation="Test explanation",
                fix_example="Test fix"
            )


class TestSuggestionContext:
    """Test suggestion context model."""

    def test_suggestion_context_creation(self):
        """Test SuggestionContext model creation."""
        context = SuggestionContext(
            current_line=5,
            has_security_scan=True,
            has_multistage=False,
            project_type=ProjectType.PYTHON
        )

        assert context.current_line == 5
        assert context.has_security_scan is True
        assert context.has_multistage is False
        assert context.project_type == ProjectType.PYTHON.value
