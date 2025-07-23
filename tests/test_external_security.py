"""Test cases for external security integration."""

import logging
from unittest.mock import patch

from docker_optimizer.external_security import ExternalSecurityScanner, TrivyScanner
from docker_optimizer.models import CVEDetails, SecurityScore, VulnerabilityReport


class TestExternalSecurityScanner:
    """Test cases for ExternalSecurityScanner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = ExternalSecurityScanner()

    def test_scanner_initialization(self):
        """Test that external security scanner initializes correctly."""
        assert isinstance(self.scanner, ExternalSecurityScanner)
        assert hasattr(self.scanner, 'trivy_scanner')

    def test_scan_dockerfile_for_vulnerabilities(self):
        """Test scanning Dockerfile for vulnerabilities."""
        dockerfile_content = """
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y openssl=1.1.0g-2ubuntu4
"""

        with patch.object(self.scanner.trivy_scanner, 'scan_dockerfile') as mock_scan:
            mock_scan.return_value = VulnerabilityReport(
                total_vulnerabilities=5,
                critical_count=1,
                high_count=2,
                medium_count=2,
                low_count=0,
                cve_details=[
                    CVEDetails(
                        cve_id="CVE-2018-0734",
                        severity="HIGH",
                        package="openssl",
                        installed_version="1.1.0g-2ubuntu4",
                        fixed_version="1.1.0g-2ubuntu4.1",
                        description="OpenSSL vulnerability in DSA signature algorithm"
                    )
                ]
            )

            result = self.scanner.scan_dockerfile_for_vulnerabilities(dockerfile_content)

            assert isinstance(result, VulnerabilityReport)
            assert result.total_vulnerabilities == 5
            assert result.critical_count == 1
            assert len(result.cve_details) == 1
            assert result.cve_details[0].cve_id == "CVE-2018-0734"

    def test_calculate_security_score(self):
        """Test security score calculation."""
        vulnerability_report = VulnerabilityReport(
            total_vulnerabilities=8,
            critical_count=1,
            high_count=3,
            medium_count=3,
            low_count=1,
            cve_details=[]
        )

        score = self.scanner.calculate_security_score(vulnerability_report)

        assert isinstance(score, SecurityScore)
        assert 0 <= score.score <= 100
        assert score.score < 70  # Should be low due to critical vulnerabilities
        assert score.grade in ['A', 'B', 'C', 'D', 'F']
        assert "critical" in score.analysis.lower()

    def test_suggest_security_improvements(self):
        """Test security improvement suggestions."""
        vulnerability_report = VulnerabilityReport(
            total_vulnerabilities=3,
            critical_count=0,
            high_count=1,
            medium_count=2,
            low_count=0,
            cve_details=[
                CVEDetails(
                    cve_id="CVE-2023-1234",
                    severity="HIGH",
                    package="nginx",
                    installed_version="1.14.2",
                    fixed_version="1.18.0",
                    description="Nginx vulnerability"
                )
            ]
        )

        suggestions = self.scanner.suggest_security_improvements(vulnerability_report)

        assert len(suggestions) > 0
        assert any("nginx" in suggestion.lower() for suggestion in suggestions)
        assert any("1.18.0" in suggestion for suggestion in suggestions)

    def test_scan_base_image_vulnerabilities(self):
        """Test scanning base image for vulnerabilities."""
        base_image = "ubuntu:18.04"

        with patch.object(self.scanner.trivy_scanner, 'scan_image') as mock_scan:
            mock_scan.return_value = VulnerabilityReport(
                total_vulnerabilities=50,
                critical_count=5,
                high_count=15,
                medium_count=20,
                low_count=10,
                cve_details=[]
            )

            result = self.scanner.scan_base_image_vulnerabilities(base_image)

            assert result.total_vulnerabilities == 50
            assert result.critical_count == 5
            mock_scan.assert_called_once_with(base_image)

    def test_compare_base_image_security(self):
        """Test comparing security of different base images."""
        images = ["ubuntu:18.04", "ubuntu:20.04", "ubuntu:22.04-slim"]

        with patch.object(self.scanner, 'scan_base_image_vulnerabilities') as mock_scan:
            # Mock different vulnerability counts for each image
            mock_scan.side_effect = [
                VulnerabilityReport(total_vulnerabilities=5, critical_count=1, high_count=2, medium_count=2, low_count=0, cve_details=[]),
                VulnerabilityReport(total_vulnerabilities=3, critical_count=0, high_count=1, medium_count=2, low_count=0, cve_details=[]),
                VulnerabilityReport(total_vulnerabilities=1, critical_count=0, high_count=0, medium_count=1, low_count=0, cve_details=[])
            ]

            comparison = self.scanner.compare_base_image_security(images)

            assert len(comparison) == 3
            # Should be sorted by security score (highest first)
            # ubuntu:22.04-slim should have highest score due to fewest vulnerabilities
            best_image = comparison[0]
            worst_image = comparison[2]

            assert best_image["security_score"] > worst_image["security_score"]
            assert best_image["total_vulnerabilities"] <= worst_image["total_vulnerabilities"]


class TestTrivyScanner:
    """Test cases for TrivyScanner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.trivy = TrivyScanner()

    def test_trivy_initialization(self):
        """Test Trivy scanner initialization."""
        assert isinstance(self.trivy, TrivyScanner)

    @patch('subprocess.run')
    def test_scan_dockerfile_success(self, mock_subprocess):
        """Test successful Dockerfile scanning with Trivy."""
        # Mock Trivy availability
        self.trivy.trivy_available = True

        # Mock successful trivy output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = '''
{
  "Results": [
    {
      "Vulnerabilities": [
        {
          "VulnerabilityID": "CVE-2023-1234",
          "Severity": "HIGH",
          "PkgName": "openssl",
          "InstalledVersion": "1.1.0g-2ubuntu4",
          "FixedVersion": "1.1.0g-2ubuntu4.1",
          "Description": "OpenSSL vulnerability"
        }
      ]
    }
  ]
}
'''

        dockerfile_content = "FROM ubuntu:18.04\nRUN apt-get install openssl"
        result = self.trivy.scan_dockerfile(dockerfile_content)

        assert isinstance(result, VulnerabilityReport)
        assert result.total_vulnerabilities == 1
        assert result.high_count == 1
        assert len(result.cve_details) == 1
        assert result.cve_details[0].cve_id == "CVE-2023-1234"

    @patch('subprocess.run')
    def test_scan_image_success(self, mock_subprocess):
        """Test successful image scanning with Trivy."""
        # Mock Trivy availability
        self.trivy.trivy_available = True

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = '''
{
  "Results": [
    {
      "Vulnerabilities": [
        {
          "VulnerabilityID": "CVE-2023-5678",
          "Severity": "CRITICAL",
          "PkgName": "nginx",
          "InstalledVersion": "1.14.2",
          "FixedVersion": "1.18.0",
          "Description": "Nginx critical vulnerability"
        }
      ]
    }
  ]
}
'''

        result = self.trivy.scan_image("nginx:1.14")

        assert result.total_vulnerabilities == 1
        assert result.critical_count == 1
        assert result.cve_details[0].severity == "CRITICAL"

    @patch('subprocess.run')
    def test_scan_trivy_not_available(self, mock_subprocess):
        """Test behavior when Trivy is not available."""
        mock_subprocess.side_effect = FileNotFoundError("trivy command not found")

        result = self.trivy.scan_dockerfile("FROM ubuntu:20.04")

        # Should return empty report when trivy is not available
        assert result.total_vulnerabilities == 0
        assert len(result.cve_details) == 0

    @patch('subprocess.run')
    def test_scan_trivy_error(self, mock_subprocess):
        """Test handling Trivy scan errors."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Trivy scan failed"

        result = self.trivy.scan_dockerfile("FROM ubuntu:20.04")

        # Should return empty report on error
        assert result.total_vulnerabilities == 0

    def test_parse_trivy_output_malformed_json(self):
        """Test parsing malformed Trivy JSON output."""
        malformed_json = "{ invalid json"

        result = self.trivy._parse_trivy_output(malformed_json)

        assert result.total_vulnerabilities == 0
        assert len(result.cve_details) == 0

    def test_count_vulnerabilities_by_severity(self):
        """Test vulnerability counting by severity."""
        vulnerabilities = [
            {"Severity": "CRITICAL"},
            {"Severity": "HIGH"},
            {"Severity": "HIGH"},
            {"Severity": "MEDIUM"},
            {"Severity": "LOW"}
        ]

        counts = self.trivy._count_vulnerabilities_by_severity(vulnerabilities)

        assert counts["critical"] == 1
        assert counts["high"] == 2
        assert counts["medium"] == 1
        assert counts["low"] == 1

    def test_create_temporary_dockerfile(self):
        """Test temporary Dockerfile creation."""
        dockerfile_content = "FROM ubuntu:20.04\nRUN apt-get update"

        with self.trivy._create_temporary_dockerfile(dockerfile_content) as temp_path:
            assert temp_path.exists()
            assert temp_path.read_text() == dockerfile_content

        # File should be cleaned up after context
        assert not temp_path.exists()

    @patch('subprocess.run')
    def test_error_handling_with_logging(self, mock_subprocess, caplog):
        """Test that errors are properly logged instead of silently ignored."""
        # Mock Trivy availability
        self.trivy.trivy_available = True

        # Mock subprocess to raise an exception
        mock_subprocess.side_effect = Exception("Unexpected error during scan")

        with caplog.at_level(logging.WARNING):
            result = self.trivy.scan_dockerfile("FROM ubuntu:20.04")

        # Should return empty report
        assert result.total_vulnerabilities == 0

        # Should log the error instead of silently ignoring it
        assert len(caplog.records) > 0
        assert "Unexpected error during scan" in caplog.text or "Error during Trivy scan" in caplog.text

    @patch('subprocess.run')
    def test_timeout_error_handling(self, mock_subprocess, caplog):
        """Test that timeout errors are properly logged."""
        import subprocess

        # Mock Trivy availability
        self.trivy.trivy_available = True

        # Mock subprocess to raise TimeoutExpired
        mock_subprocess.side_effect = subprocess.TimeoutExpired("trivy", 30)

        with caplog.at_level(logging.WARNING):
            result = self.trivy.scan_dockerfile("FROM ubuntu:20.04")

        # Should return empty report
        assert result.total_vulnerabilities == 0

        # Should log the timeout
        assert len(caplog.records) > 0
        assert "timeout" in caplog.text.lower() or "scan operation timed out" in caplog.text.lower()

    @patch('subprocess.run')
    def test_file_not_found_error_handling(self, mock_subprocess, caplog):
        """Test that FileNotFoundError during Trivy availability check is properly handled and logged."""
        # Mock subprocess to raise FileNotFoundError
        mock_subprocess.side_effect = FileNotFoundError("trivy command not found")

        with caplog.at_level(logging.INFO):
            # Create a new scanner instance to trigger the availability check
            trivy_scanner = TrivyScanner()
            result = trivy_scanner.scan_dockerfile("FROM ubuntu:20.04")

        # Should return empty report
        assert result.total_vulnerabilities == 0

        # Scanner should be marked as unavailable
        assert not trivy_scanner.trivy_available

        # Should log that Trivy is not available
        assert len(caplog.records) > 0
        assert "trivy not available" in caplog.text.lower() or "trivy command not found" in caplog.text.lower()
