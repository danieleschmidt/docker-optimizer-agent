"""Test cases for security analyzer."""

from docker_optimizer.parser import DockerfileParser
from docker_optimizer.security import SecurityAnalyzer


class TestSecurityAnalyzer:
    """Test cases for SecurityAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SecurityAnalyzer()
        self.parser = DockerfileParser()

    def test_analyze_security_basic(self):
        """Test basic security analysis."""
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update && apt-get install -y curl
"""

        parsed = self.parser.parse(dockerfile_content)
        fixes = self.analyzer.analyze_security(dockerfile_content, parsed)

        assert len(fixes) >= 2  # Should find latest tag and root user issues
        vulnerabilities = [fix.vulnerability for fix in fixes]
        assert any("latest" in vuln.lower() for vuln in vulnerabilities)

    def test_uses_latest_tag(self):
        """Test detection of latest tag usage."""
        # Explicit latest tag
        assert self.analyzer._uses_latest_tag("FROM ubuntu:latest")

        # Implicit latest tag
        assert self.analyzer._uses_latest_tag("FROM ubuntu")

        # No latest tag
        assert not self.analyzer._uses_latest_tag("FROM ubuntu:20.04")

    def test_runs_as_root(self):
        """Test detection of root user execution."""
        # No USER directive (defaults to root)
        instructions1 = self.parser.parse("FROM ubuntu:20.04\nRUN echo hello")
        assert self.analyzer._runs_as_root(instructions1)

        # Explicit root user
        instructions2 = self.parser.parse("FROM ubuntu:20.04\nUSER root")
        assert self.analyzer._runs_as_root(instructions2)

        # Explicit root user with UID 0
        instructions3 = self.parser.parse("FROM ubuntu:20.04\nUSER 0")
        assert self.analyzer._runs_as_root(instructions3)

        # Non-root user
        instructions4 = self.parser.parse("FROM ubuntu:20.04\nUSER 1000")
        assert not self.analyzer._runs_as_root(instructions4)

    def test_has_uncleaned_package_cache(self):
        """Test detection of uncleaned package cache."""
        # Has apt-get update but no cleanup
        content1 = "RUN apt-get update && apt-get install -y curl"
        assert self.analyzer._has_uncleaned_package_cache(content1)

        # Has apt-get update with cleanup
        content2 = "RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*"
        assert not self.analyzer._has_uncleaned_package_cache(content2)

        # No apt-get update
        content3 = "RUN echo hello"
        assert not self.analyzer._has_uncleaned_package_cache(content3)

    def test_analyze_security_comprehensive(self):
        """Test comprehensive security analysis."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl git
COPY . /app
WORKDIR /app
"""

        parsed = self.parser.parse(dockerfile_content)
        fixes = self.analyzer.analyze_security(dockerfile_content, parsed)

        # Should identify multiple issues
        assert len(fixes) >= 3

        # Check for specific vulnerabilities
        vulnerability_types = [fix.vulnerability.lower() for fix in fixes]
        assert any("latest" in vuln for vuln in vulnerability_types)
        assert any("root" in vuln for vuln in vulnerability_types)
        assert any("cache" in vuln for vuln in vulnerability_types)

    def test_analyze_security_secure_dockerfile(self):
        """Test analysis of already secure Dockerfile."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache curl && rm -rf /var/cache/apk/*
USER 1000
"""

        parsed = self.parser.parse(dockerfile_content)
        fixes = self.analyzer.analyze_security(dockerfile_content, parsed)

        # Should find minimal or no issues
        assert len(fixes) <= 1  # Might still find some minor issues

    def test_security_fix_severities(self):
        """Test that security fixes have appropriate severities."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
"""

        parsed = self.parser.parse(dockerfile_content)
        fixes = self.analyzer.analyze_security(dockerfile_content, parsed)

        # Check that severities are valid
        valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for fix in fixes:
            assert fix.severity in valid_severities

    def test_security_fix_descriptions(self):
        """Test that security fixes have meaningful descriptions."""
        dockerfile_content = "FROM ubuntu:latest"

        parsed = self.parser.parse(dockerfile_content)
        fixes = self.analyzer.analyze_security(dockerfile_content, parsed)

        for fix in fixes:
            assert len(fix.description) > 10  # Should have meaningful description
            assert len(fix.fix) > 5  # Should have meaningful fix suggestion
