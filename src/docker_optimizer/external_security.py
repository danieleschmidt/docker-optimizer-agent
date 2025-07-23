"""External security vulnerability scanning integration."""

import json
import logging
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Union

from .models import CVEDetails, SecurityScore, VulnerabilityReport

logger = logging.getLogger(__name__)


class TrivyScanner:
    """Trivy vulnerability scanner integration."""

    def __init__(self) -> None:
        """Initialize Trivy scanner."""
        self.trivy_available = self._check_trivy_availability()

    def scan_dockerfile(self, dockerfile_content: str) -> VulnerabilityReport:
        """Scan a Dockerfile for vulnerabilities.

        Args:
            dockerfile_content: Content of the Dockerfile to scan

        Returns:
            VulnerabilityReport: Vulnerability scan results
        """
        if not self.trivy_available:
            return VulnerabilityReport(total_vulnerabilities=0)

        try:
            with self._create_temporary_dockerfile(dockerfile_content) as dockerfile_path:
                cmd = [
                    "trivy",
                    "config",
                    "--format", "json",
                    "--quiet",
                    str(dockerfile_path)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    return self._parse_trivy_output(result.stdout)
                else:
                    # Try fallback: scan the base image directly
                    base_image = self._extract_base_image(dockerfile_content)
                    if base_image:
                        return self.scan_image(base_image)

        except subprocess.TimeoutExpired as e:
            logger.warning("Trivy scan operation timed out after %s seconds", e.timeout)
        except FileNotFoundError:
            logger.info("Trivy command not found - skipping vulnerability scan")
        except Exception as e:
            logger.warning("Error during Trivy scan: %s", str(e))

        return VulnerabilityReport(total_vulnerabilities=0)

    def scan_image(self, image_name: str) -> VulnerabilityReport:
        """Scan a Docker image for vulnerabilities.

        Args:
            image_name: Name of the Docker image to scan

        Returns:
            VulnerabilityReport: Vulnerability scan results
        """
        if not self.trivy_available:
            return VulnerabilityReport(total_vulnerabilities=0)

        try:
            cmd = [
                "trivy",
                "image",
                "--format", "json",
                "--quiet",
                image_name
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                return self._parse_trivy_output(result.stdout)

        except subprocess.TimeoutExpired as e:
            logger.warning("Trivy image scan timed out after %s seconds for image: %s", e.timeout, image_name)
        except FileNotFoundError:
            logger.info("Trivy command not found - skipping image vulnerability scan")
        except Exception as e:
            logger.warning("Error during Trivy image scan for %s: %s", image_name, str(e))

        return VulnerabilityReport(total_vulnerabilities=0)

    def _check_trivy_availability(self) -> bool:
        """Check if Trivy is available in the system."""
        try:
            result = subprocess.run(
                ["trivy", "--version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning("Trivy availability check timed out")
            return False
        except FileNotFoundError:
            logger.info("Trivy not available - trivy command not found")
            return False

    def _parse_trivy_output(self, output: str) -> VulnerabilityReport:
        """Parse Trivy JSON output into VulnerabilityReport.

        Args:
            output: JSON output from Trivy

        Returns:
            VulnerabilityReport: Parsed vulnerability report
        """
        try:
            data = json.loads(output)
            all_vulnerabilities = []

            # Extract vulnerabilities from all results
            for result in data.get("Results", []):
                vulnerabilities = result.get("Vulnerabilities", [])
                all_vulnerabilities.extend(vulnerabilities)

            # Count vulnerabilities by severity
            severity_counts = self._count_vulnerabilities_by_severity(all_vulnerabilities)

            # Create CVE details
            cve_details = []
            for vuln in all_vulnerabilities[:10]:  # Limit to first 10 for performance
                cve_details.append(CVEDetails(
                    cve_id=vuln.get("VulnerabilityID", "Unknown"),
                    severity=vuln.get("Severity", "UNKNOWN"),
                    package=vuln.get("PkgName", "Unknown"),
                    installed_version=vuln.get("InstalledVersion", "Unknown"),
                    fixed_version=vuln.get("FixedVersion"),
                    description=vuln.get("Description", "No description available")[:200]  # Truncate long descriptions
                ))

            return VulnerabilityReport(
                total_vulnerabilities=len(all_vulnerabilities),
                critical_count=severity_counts["critical"],
                high_count=severity_counts["high"],
                medium_count=severity_counts["medium"],
                low_count=severity_counts["low"],
                cve_details=cve_details
            )

        except (json.JSONDecodeError, KeyError, Exception):
            return VulnerabilityReport(total_vulnerabilities=0)

    def _count_vulnerabilities_by_severity(self, vulnerabilities: List[Dict[str, str]]) -> Dict[str, int]:
        """Count vulnerabilities by severity level."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for vuln in vulnerabilities:
            severity = vuln.get("Severity", "").lower()
            if severity in counts:
                counts[severity] += 1

        return counts

    def _extract_base_image(self, dockerfile_content: str) -> str:
        """Extract base image from Dockerfile content."""
        lines = dockerfile_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('FROM '):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1].split(' AS ')[0]  # Remove alias if present
        return ""

    @contextmanager
    def _create_temporary_dockerfile(self, dockerfile_content: str) -> Iterator[Path]:
        """Create a temporary Dockerfile for scanning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(dockerfile_content)
            temp_path = Path(f.name)

        try:
            yield temp_path
        finally:
            if temp_path.exists():
                temp_path.unlink()


class ExternalSecurityScanner:
    """Main external security scanner that integrates multiple tools."""

    def __init__(self) -> None:
        """Initialize the external security scanner."""
        self.trivy_scanner = TrivyScanner()

    def scan_dockerfile_for_vulnerabilities(self, dockerfile_content: str) -> VulnerabilityReport:
        """Scan Dockerfile for vulnerabilities using external tools.

        Args:
            dockerfile_content: Content of the Dockerfile to scan

        Returns:
            VulnerabilityReport: Comprehensive vulnerability report
        """
        # Use Trivy as primary scanner
        report = self.trivy_scanner.scan_dockerfile(dockerfile_content)

        # Could integrate additional scanners here in the future
        # e.g., Snyk, Clair, etc.

        return report

    def scan_base_image_vulnerabilities(self, base_image: str) -> VulnerabilityReport:
        """Scan a base image for vulnerabilities.

        Args:
            base_image: Base image name to scan

        Returns:
            VulnerabilityReport: Vulnerability report for the image
        """
        return self.trivy_scanner.scan_image(base_image)

    def calculate_security_score(self, vulnerability_report: VulnerabilityReport) -> SecurityScore:
        """Calculate security score based on vulnerability report.

        Args:
            vulnerability_report: Vulnerability report to analyze

        Returns:
            SecurityScore: Security score and analysis
        """
        # Base score starts at 100
        score = 100

        # Deduct points based on severity
        score -= vulnerability_report.critical_count * 25
        score -= vulnerability_report.high_count * 10
        score -= vulnerability_report.medium_count * 5
        score -= vulnerability_report.low_count * 1

        # Ensure score doesn't go below 0
        score = max(0, score)

        # Determine grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        # Generate analysis
        analysis_parts = []
        if vulnerability_report.critical_count > 0:
            analysis_parts.append(f"{vulnerability_report.critical_count} critical vulnerabilities found")
        if vulnerability_report.high_count > 0:
            analysis_parts.append(f"{vulnerability_report.high_count} high severity vulnerabilities")
        if vulnerability_report.total_vulnerabilities == 0:
            analysis_parts.append("No vulnerabilities detected")
        elif vulnerability_report.total_vulnerabilities < 5:
            analysis_parts.append("Low vulnerability count")
        elif vulnerability_report.total_vulnerabilities < 20:
            analysis_parts.append("Moderate vulnerability count")
        else:
            analysis_parts.append("High vulnerability count")

        analysis = "; ".join(analysis_parts) if analysis_parts else "Security assessment completed"

        # Generate recommendations
        recommendations = []
        if vulnerability_report.critical_count > 0:
            recommendations.append("Address critical vulnerabilities immediately")
        if vulnerability_report.high_count > 0:
            recommendations.append("Update packages with high severity vulnerabilities")
        if vulnerability_report.total_vulnerabilities > 10:
            recommendations.append("Consider using a more recent base image")
        if score < 70:
            recommendations.append("Review and update all package versions")

        return SecurityScore(
            score=score,
            grade=grade,
            analysis=analysis,
            recommendations=recommendations
        )

    def suggest_security_improvements(self, vulnerability_report: VulnerabilityReport) -> List[str]:
        """Suggest specific security improvements based on vulnerabilities.

        Args:
            vulnerability_report: Vulnerability report to analyze

        Returns:
            List of security improvement suggestions
        """
        suggestions = []

        # Group vulnerabilities by package
        package_vulns: Dict[str, List[CVEDetails]] = {}
        for cve in vulnerability_report.cve_details:
            if cve.package not in package_vulns:
                package_vulns[cve.package] = []
            package_vulns[cve.package].append(cve)

        # Generate package-specific suggestions
        for package, vulns in package_vulns.items():
            critical_vulns = [v for v in vulns if v.severity == "CRITICAL"]
            high_vulns = [v for v in vulns if v.severity == "HIGH"]

            if critical_vulns:
                fixed_versions = [v.fixed_version for v in critical_vulns if v.fixed_version]
                if fixed_versions:
                    suggestions.append(f"CRITICAL: Update {package} to version {fixed_versions[0]} or later")
                else:
                    suggestions.append(f"CRITICAL: {package} has critical vulnerabilities - consider alternative package")

            elif high_vulns:
                fixed_versions = [v.fixed_version for v in high_vulns if v.fixed_version]
                if fixed_versions:
                    suggestions.append(f"HIGH: Update {package} to version {fixed_versions[0]} or later")

        # General suggestions
        if vulnerability_report.total_vulnerabilities > 20:
            suggestions.append("Consider using a more recent or minimal base image (e.g., Alpine, distroless)")

        if vulnerability_report.critical_count > 0:
            suggestions.append("Run security scans regularly in your CI/CD pipeline")

        return suggestions

    def compare_base_image_security(self, images: List[str]) -> List[Dict[str, Union[str, int]]]:
        """Compare security of multiple base images.

        Args:
            images: List of base image names to compare

        Returns:
            List of image security comparisons, sorted by security score
        """
        comparisons: List[Dict[str, Union[str, int]]] = []

        for image in images:
            report = self.scan_base_image_vulnerabilities(image)
            score = self.calculate_security_score(report)

            comparisons.append({
                "image": image,
                "total_vulnerabilities": report.total_vulnerabilities,
                "critical_count": report.critical_count,
                "high_count": report.high_count,
                "security_score": score.score,
                "grade": score.grade
            })

        # Sort by security score (highest first)
        comparisons.sort(key=lambda x: x["security_score"], reverse=True)

        return comparisons
