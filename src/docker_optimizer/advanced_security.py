"""Advanced Security Rule Engine for custom security policies and compliance checking."""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from .models import (
    SecurityFix,
    SecurityRule,
    SecurityRuleEngineResult,
    SecurityRuleSet,
    SecurityScore,
)


class PolicyManager:
    """Manages loading and validation of security policies."""

    def __init__(self) -> None:
        """Initialize the policy manager."""
        self.supported_formats = {".json", ".yaml", ".yml"}

    def load_policy(self, policy_path: Path) -> SecurityRuleSet:
        """Load a security policy from a file.

        Args:
            policy_path: Path to the policy file

        Returns:
            SecurityRuleSet instance

        Raises:
            ValueError: If the policy file is invalid or malformed
        """
        if policy_path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported policy format. Supported: {self.supported_formats}"
            )

        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                if policy_path.suffix.lower() == ".json":
                    policy_data = json.load(f)
                else:
                    policy_data = yaml.safe_load(f)

            if not self._validate_policy_schema(policy_data):
                raise ValueError("Policy file does not match required schema")

            return SecurityRuleSet(**policy_data)

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Failed to parse policy file: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Policy validation failed: {e}") from e

    def _validate_policy_schema(self, policy_data: Dict[str, Any]) -> bool:
        """Validate that policy data has required schema.

        Args:
            policy_data: Parsed policy data

        Returns:
            True if valid, False otherwise
        """
        required_fields = {"name", "version", "description", "rules"}
        return all(field in policy_data for field in required_fields)

    def get_default_policies(self) -> List[SecurityRuleSet]:
        """Get default built-in security policies.

        Returns:
            List of default security policy sets
        """
        # Built-in comprehensive security rules
        default_rules = [
            SecurityRule(
                id="no_latest_tag",
                name="No Latest Tag",
                description="Containers should use specific version tags",
                severity="HIGH",
                category="security",
                rule_type="pattern",
                pattern=r"FROM\s+[^:\s]+(?::latest)?(?:\s|$)",
                function_name=None,
                message="Using latest tag makes builds unpredictable",
                fix_example="FROM ubuntu:22.04",
                compliance_frameworks=["SOC2", "PCI-DSS"],
            ),
            SecurityRule(
                id="no_root_user",
                name="No Root User",
                description="Container should not run as root user",
                severity="HIGH",
                category="security",
                rule_type="function",
                pattern=None,
                function_name="check_root_user",
                message="Container runs as root user",
                fix_example="USER appuser",
                compliance_frameworks=["SOC2", "PCI-DSS", "HIPAA"],
            ),
            SecurityRule(
                id="no_sudo",
                name="No Sudo Usage",
                description="Containers should not use sudo",
                severity="MEDIUM",
                category="security",
                rule_type="pattern",
                pattern=r"RUN.*sudo\s+",
                function_name=None,
                message="Avoid using sudo in containers",
                fix_example="RUN apt-get update && apt-get install -y package",
                compliance_frameworks=["SOC2"],
            ),
            SecurityRule(
                id="expose_port_range",
                name="Specific Port Exposure",
                description="Exposed ports should be specific, not ranges",
                severity="MEDIUM",
                category="security",
                rule_type="pattern",
                pattern=r"EXPOSE\s+\d+-\d+",
                function_name=None,
                message="Avoid exposing port ranges",
                fix_example="EXPOSE 80",
                compliance_frameworks=["PCI-DSS"],
            ),
            SecurityRule(
                id="secrets_in_env",
                name="No Secrets in ENV",
                description="Sensitive data should not be in ENV variables",
                severity="CRITICAL",
                category="security",
                rule_type="pattern",
                pattern=r"ENV\s+.*(?:PASSWORD|SECRET|KEY|TOKEN).*=.*\S",
                function_name=None,
                message="Secrets should not be exposed in environment variables",
                fix_example="Use secret management solutions instead",
                compliance_frameworks=["SOC2", "PCI-DSS", "HIPAA"],
            ),
        ]

        default_policy = SecurityRuleSet(
            name="Default Security Policy",
            version="1.0.0",
            description="Comprehensive default security rules",
            rules=default_rules,
            compliance_framework=None,
            author="Docker Optimizer Agent",
            created_at=None,
        )

        return [default_policy]


class ComplianceChecker:
    """Checks Docker configurations against compliance frameworks."""

    def __init__(self) -> None:
        """Initialize the compliance checker."""
        self.supported_frameworks = {"SOC2", "PCI-DSS", "HIPAA"}

    def check_soc2_compliance(
        self, dockerfile_content: str, parsed_instructions: List[Dict[str, Any]]
    ) -> List[SecurityFix]:
        """Check SOC2 compliance requirements.

        Args:
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            List of SOC2 compliance violations
        """
        violations = []

        # SOC2 CC6.1 - Logical and physical access controls
        if self._uses_latest_tag(dockerfile_content):
            violations.append(
                SecurityFix(
                    vulnerability="SOC2 CC6.1 - Latest tag usage",
                    severity="HIGH",
                    description="Using latest tag violates version control requirements",
                    fix="Use specific version tags for reproducible builds",
                )
            )

        if self._runs_as_root(parsed_instructions):
            violations.append(
                SecurityFix(
                    vulnerability="SOC2 CC6.1 - Root user execution",
                    severity="HIGH",
                    description="Running as root violates principle of least privilege",
                    fix="Create and use non-privileged user account",
                )
            )

        # SOC2 CC6.7 - System boundaries and data flow
        if self._exposes_sensitive_ports(parsed_instructions):
            violations.append(
                SecurityFix(
                    vulnerability="SOC2 CC6.7 - Sensitive port exposure",
                    severity="MEDIUM",
                    description="Exposing administrative ports increases attack surface",
                    fix="Only expose necessary application ports",
                )
            )

        return violations

    def check_pci_dss_compliance(
        self, dockerfile_content: str, parsed_instructions: List[Dict[str, Any]]
    ) -> List[SecurityFix]:
        """Check PCI-DSS compliance requirements.

        Args:
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            List of PCI-DSS compliance violations
        """
        violations = []

        # PCI-DSS Requirement 2 - Default passwords and security parameters
        if self._runs_as_root(parsed_instructions):
            violations.append(
                SecurityFix(
                    vulnerability="PCI-DSS Req 2 - Default root user",
                    severity="HIGH",
                    description="Using default root user violates secure configuration",
                    fix="Configure non-root user with minimal privileges",
                )
            )

        # PCI-DSS Requirement 7 - Restrict access by business need
        if self._has_unnecessary_packages(dockerfile_content):
            violations.append(
                SecurityFix(
                    vulnerability="PCI-DSS Req 7 - Unnecessary packages",
                    severity="MEDIUM",
                    description="Unnecessary packages increase attack surface",
                    fix="Remove unnecessary packages and services",
                )
            )

        # PCI-DSS Requirement 4 - Encrypt transmission of cardholder data
        if not self._uses_tls_ports(parsed_instructions):
            violations.append(
                SecurityFix(
                    vulnerability="PCI-DSS Req 4 - Unencrypted ports",
                    severity="MEDIUM",
                    description="Should use TLS/encrypted ports for data transmission",
                    fix="Use HTTPS (443) instead of HTTP (80) for web services",
                )
            )

        return violations

    def check_hipaa_compliance(
        self, dockerfile_content: str, parsed_instructions: List[Dict[str, Any]]
    ) -> List[SecurityFix]:
        """Check HIPAA compliance requirements.

        Args:
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            List of HIPAA compliance violations
        """
        violations = []

        # HIPAA Security Rule - Access Control
        if self._runs_as_root(parsed_instructions):
            violations.append(
                SecurityFix(
                    vulnerability="HIPAA Access Control - Root user",
                    severity="CRITICAL",
                    description="Root user access violates minimum necessary standard",
                    fix="Implement role-based access with non-privileged user",
                )
            )

        # HIPAA Security Rule - Information System Activity Review
        if not self._has_logging_configured(parsed_instructions):
            violations.append(
                SecurityFix(
                    vulnerability="HIPAA Activity Review - No logging",
                    severity="HIGH",
                    description="Audit logging required for HIPAA compliance",
                    fix="Configure centralized logging for audit trails",
                )
            )

        return violations

    def _uses_latest_tag(self, dockerfile_content: str) -> bool:
        """Check if Dockerfile uses latest tag."""
        return re.search(r"FROM\s+[^:\s]+(?::latest)?(?:\s|$)", dockerfile_content) is not None

    def _runs_as_root(self, instructions: List[Dict[str, Any]]) -> bool:
        """Check if container runs as root."""
        user_instructions = [
            inst for inst in instructions if inst["instruction"] == "USER"
        ]
        if not user_instructions:
            return True
        return any(
            inst["value"].strip() in ["root", "0"] for inst in user_instructions
        )

    def _exposes_sensitive_ports(self, instructions: List[Dict[str, Any]]) -> bool:
        """Check if sensitive administrative ports are exposed."""
        sensitive_ports = {"22", "23", "3389", "5432", "3306", "27017"}
        expose_instructions = [
            inst for inst in instructions if inst["instruction"] == "EXPOSE"
        ]
        for inst in expose_instructions:
            ports = inst["value"].split()
            if any(port in sensitive_ports for port in ports):
                return True
        return False

    def _has_unnecessary_packages(self, dockerfile_content: str) -> bool:
        """Check for unnecessary packages that increase attack surface."""
        unnecessary_patterns = [
            r"vim", r"nano", r"wget", r"curl.*-y", r"openssh-server"
        ]
        return any(re.search(pattern, dockerfile_content, re.IGNORECASE)
                  for pattern in unnecessary_patterns)

    def _uses_tls_ports(self, instructions: List[Dict[str, Any]]) -> bool:
        """Check if TLS ports are used instead of plain text."""
        expose_instructions = [
            inst for inst in instructions if inst["instruction"] == "EXPOSE"
        ]
        tls_ports = {"443", "993", "995", "636"}
        plain_ports = {"80", "143", "110", "389"}

        has_tls = False
        has_plain = False

        for inst in expose_instructions:
            ports = inst["value"].split()
            if any(port in tls_ports for port in ports):
                has_tls = True
            if any(port in plain_ports for port in ports):
                has_plain = True

        return has_tls or not has_plain

    def _has_logging_configured(self, instructions: List[Dict[str, Any]]) -> bool:
        """Check if logging is configured."""
        # Look for logging configuration in RUN commands
        run_instructions = [
            inst for inst in instructions if inst["instruction"] == "RUN"
        ]
        logging_patterns = [r"rsyslog", r"syslog-ng", r"journald", r"log"]
        return any(
            re.search(pattern, inst["value"], re.IGNORECASE)
            for inst in run_instructions
            for pattern in logging_patterns
        )


class AdvancedSecurityEngine:
    """Advanced Security Rule Engine for custom policies and compliance checking."""

    def __init__(self) -> None:
        """Initialize the Advanced Security Engine."""
        self.policy_manager = PolicyManager()
        self.compliance_checker = ComplianceChecker()
        self.loaded_policies: List[SecurityRuleSet] = []

    def load_default_policies(self) -> None:
        """Load default built-in security policies."""
        default_policies = self.policy_manager.get_default_policies()
        self.loaded_policies.extend(default_policies)

    def load_custom_policy(self, policy_path: Path) -> None:
        """Load a custom security policy from file.

        Args:
            policy_path: Path to the policy file

        Raises:
            ValueError: If the policy file is invalid
        """
        policy = self.policy_manager.load_policy(policy_path)
        self.loaded_policies.append(policy)

    def analyze_dockerfile(
        self, dockerfile_content: str, parsed_instructions: List[Dict[str, Any]]
    ) -> List[SecurityFix]:
        """Analyze Dockerfile against all loaded security policies.

        Args:
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            List of security violations found
        """
        violations = []

        for policy in self.loaded_policies:
            for rule in policy.get_enabled_rules():
                violation = self._evaluate_rule(
                    rule, dockerfile_content, parsed_instructions
                )
                if violation:
                    violations.append(violation)

        return violations

    def check_compliance(
        self,
        dockerfile_content: str,
        parsed_instructions: List[Dict[str, Any]],
        framework: str,
    ) -> List[SecurityFix]:
        """Check compliance against a specific framework.

        Args:
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions
            framework: Compliance framework to check against

        Returns:
            List of compliance violations

        Raises:
            ValueError: If framework is not supported
        """
        framework_upper = framework.upper()

        if framework_upper not in self.compliance_checker.supported_frameworks:
            raise ValueError(
                f"Unsupported compliance framework: {framework}. "
                f"Supported: {self.compliance_checker.supported_frameworks}"
            )

        if framework_upper == "SOC2":
            return self.compliance_checker.check_soc2_compliance(
                dockerfile_content, parsed_instructions
            )
        elif framework_upper == "PCI-DSS":
            return self.compliance_checker.check_pci_dss_compliance(
                dockerfile_content, parsed_instructions
            )
        elif framework_upper == "HIPAA":
            return self.compliance_checker.check_hipaa_compliance(
                dockerfile_content, parsed_instructions
            )

        return []

    def _evaluate_rule(
        self,
        rule: SecurityRule,
        dockerfile_content: str,
        parsed_instructions: List[Dict[str, Any]],
    ) -> Optional[SecurityFix]:
        """Evaluate a single security rule.

        Args:
            rule: Security rule to evaluate
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            SecurityFix if rule is violated, None otherwise
        """
        if rule.rule_type == "pattern":
            return self._evaluate_pattern_rule(rule, dockerfile_content)
        elif rule.rule_type == "function":
            return self._evaluate_function_rule(rule, dockerfile_content, parsed_instructions)
        elif rule.rule_type == "compliance":
            # Compliance rules are handled separately
            return None

        return None

    def _evaluate_pattern_rule(
        self, rule: SecurityRule, dockerfile_content: str
    ) -> Optional[SecurityFix]:
        """Evaluate a pattern-based security rule.

        Args:
            rule: Pattern rule to evaluate
            dockerfile_content: Raw Dockerfile content

        Returns:
            SecurityFix if pattern matches, None otherwise
        """
        if not rule.pattern:
            return None

        match = re.search(rule.pattern, dockerfile_content, re.IGNORECASE | re.MULTILINE)
        if match:
            return SecurityFix(
                vulnerability=rule.name,
                severity=rule.severity,
                description=rule.description,
                fix=rule.fix_example or rule.message,
            )

        return None

    def _evaluate_function_rule(
        self,
        rule: SecurityRule,
        dockerfile_content: str,
        parsed_instructions: List[Dict[str, Any]],
    ) -> Optional[SecurityFix]:
        """Evaluate a function-based security rule.

        Args:
            rule: Function rule to evaluate
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            SecurityFix if function returns True, None otherwise
        """
        if not rule.function_name:
            return None

        if self._call_rule_function(rule.function_name, dockerfile_content, parsed_instructions):
            return SecurityFix(
                vulnerability=rule.name,
                severity=rule.severity,
                description=rule.description,
                fix=rule.fix_example or rule.message,
            )

        return None

    def _call_rule_function(
        self,
        function_name: str,
        dockerfile_content: str,
        parsed_instructions: List[Dict[str, Any]],
    ) -> bool:
        """Call a named rule function.

        Args:
            function_name: Name of the function to call
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            True if rule is violated, False otherwise
        """
        # Built-in rule functions
        if function_name == "check_root_user":
            return self._check_root_user(parsed_instructions)
        elif function_name == "check_exposed_secrets":
            return self._check_exposed_secrets(dockerfile_content)
        elif function_name == "check_package_versions":
            return self._check_package_versions(dockerfile_content)

        return False

    def _check_root_user(self, parsed_instructions: List[Dict[str, Any]]) -> bool:
        """Check if container runs as root user."""
        user_instructions = [
            inst for inst in parsed_instructions if inst["instruction"] == "USER"
        ]
        if not user_instructions:
            return True
        return any(
            inst["value"].strip() in ["root", "0"] for inst in user_instructions
        )

    def _check_exposed_secrets(self, dockerfile_content: str) -> bool:
        """Check for exposed secrets in environment variables."""
        secret_patterns = [
            r"ENV\s+.*(?:PASSWORD|SECRET|KEY|TOKEN).*=.*\S",
            r"ARG\s+.*(?:PASSWORD|SECRET|KEY|TOKEN).*=.*\S",
        ]
        return any(
            re.search(pattern, dockerfile_content, re.IGNORECASE | re.MULTILINE)
            for pattern in secret_patterns
        )

    def _check_package_versions(self, dockerfile_content: str) -> bool:
        """Check for unversioned package installations."""
        # Check for package installs without versions
        patterns = [
            r"apt-get install.*[^=]\s+\w+(?:\s|$)",  # apt without version
            r"yum install.*[^-]\s+\w+(?:\s|$)",      # yum without version
            r"apk add.*[^=]\s+\w+(?:\s|$)",          # apk without version
        ]
        return any(
            re.search(pattern, dockerfile_content, re.IGNORECASE | re.MULTILINE)
            for pattern in patterns
        )

    def get_security_score(self, violations: List[SecurityFix]) -> SecurityScore:
        """Calculate security score based on violations.

        Args:
            violations: List of security violations

        Returns:
            SecurityScore with calculated grade and recommendations
        """
        if not violations:
            return SecurityScore(
                score=100,
                grade="A",
                analysis="No security violations found",
                recommendations=["Maintain current security practices"],
            )

        # Calculate score based on violation severity
        score = 100
        for violation in violations:
            if violation.severity == "CRITICAL":
                score -= 25
            elif violation.severity == "HIGH":
                score -= 15
            elif violation.severity == "MEDIUM":
                score -= 10
            elif violation.severity == "LOW":
                score -= 5

        score = max(0, score)

        # Assign letter grade
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

        # Generate recommendations
        recommendations = []
        critical_count = sum(1 for v in violations if v.severity == "CRITICAL")
        high_count = sum(1 for v in violations if v.severity == "HIGH")

        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical security violations immediately")
        if high_count > 0:
            recommendations.append(f"Fix {high_count} high-severity security issues")
        if len(violations) > 5:
            recommendations.append("Consider implementing a comprehensive security policy")

        analysis = f"Found {len(violations)} security violations. Score: {score}/100"

        return SecurityScore(
            score=score,
            grade=grade,
            analysis=analysis,
            recommendations=recommendations or ["Continue following security best practices"],
        )

    def get_violation_report(self, violations: List[SecurityFix]) -> str:
        """Generate a human-readable violation report.

        Args:
            violations: List of security violations

        Returns:
            Formatted violation report
        """
        if not violations:
            return "✅ No security violations found!"

        report = ["🔒 Security Violations Found:\n"]

        for i, violation in enumerate(violations, 1):
            severity_emoji = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🔵"
            }.get(violation.severity, "⚪")

            report.append(f"{i}. {severity_emoji} {violation.vulnerability}")
            report.append(f"   Severity: {violation.severity}")
            report.append(f"   Description: {violation.description}")
            report.append(f"   Fix: {violation.fix}")
            report.append("")

        return "\n".join(report)

    def analyze_with_timing(
        self, dockerfile_content: str, parsed_instructions: List[Dict[str, Any]]
    ) -> SecurityRuleEngineResult:
        """Perform comprehensive security analysis with performance timing.

        Args:
            dockerfile_content: Raw Dockerfile content
            parsed_instructions: Parsed Dockerfile instructions

        Returns:
            Complete analysis result with timing information
        """
        start_time = time.time()

        violations = self.analyze_dockerfile(dockerfile_content, parsed_instructions)
        security_score = self.get_security_score(violations)

        rules_evaluated = sum(
            policy.enabled_rule_count for policy in self.loaded_policies
        )

        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return SecurityRuleEngineResult(
            violations=violations,
            policies_applied=[policy.name for policy in self.loaded_policies],
            rules_evaluated=rules_evaluated,
            execution_time_ms=execution_time,
            security_score=security_score,
        )
