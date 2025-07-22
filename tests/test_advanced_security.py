"""Test cases for Advanced Security Rule Engine."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from docker_optimizer.advanced_security import (
    AdvancedSecurityEngine,
    ComplianceChecker,
    PolicyManager,
    SecurityRule,
    SecurityRuleSet,
)
from docker_optimizer.models import SecurityFix
from docker_optimizer.parser import DockerfileParser


class TestSecurityRule:
    """Test cases for SecurityRule model."""

    def test_security_rule_creation(self):
        """Test creating a security rule."""
        rule = SecurityRule(
            id="test_rule",
            name="Test Rule",
            description="A test security rule",
            severity="HIGH",
            category="security",
            rule_type="pattern",
            pattern="FROM.*:latest",
            message="Do not use latest tag",
            fix_example="FROM ubuntu:22.04",
            compliance_frameworks=["SOC2", "PCI-DSS"],
        )

        assert rule.id == "test_rule"
        assert rule.severity == "HIGH"
        assert rule.is_critical is False
        assert rule.is_high_severity is True
        assert "SOC2" in rule.compliance_frameworks

    def test_security_rule_severity_validation(self):
        """Test security rule severity validation."""
        # Valid severity
        rule = SecurityRule(
            id="test", name="Test", description="Test", severity="CRITICAL",
            category="security", rule_type="pattern", pattern="test", message="Test message"
        )
        assert rule.severity == "CRITICAL"

        # Invalid severity should raise validation error
        with pytest.raises(ValueError):
            SecurityRule(
                id="test", name="Test", description="Test", severity="INVALID",
                category="security", rule_type="pattern", pattern="test", message="Test message"
            )

    def test_security_rule_type_validation(self):
        """Test security rule type validation."""
        # Valid rule types
        for rule_type in ["pattern", "function", "compliance"]:
            rule = SecurityRule(
                id="test", name="Test", description="Test", severity="HIGH",
                category="security", rule_type=rule_type, pattern="test", message="Test message"
            )
            assert rule.rule_type == rule_type


class TestSecurityRuleSet:
    """Test cases for SecurityRuleSet model."""

    def test_security_ruleset_creation(self):
        """Test creating a security ruleset."""
        rule = SecurityRule(
            id="test_rule", name="Test", description="Test", severity="HIGH",
            category="security", rule_type="pattern", pattern="test", message="Test message"
        )

        ruleset = SecurityRuleSet(
            name="Test Ruleset",
            version="1.0",
            description="A test ruleset",
            rules=[rule],
            compliance_framework="SOC2"
        )

        assert ruleset.name == "Test Ruleset"
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].id == "test_rule"

    def test_get_rules_by_severity(self):
        """Test filtering rules by severity."""
        critical_rule = SecurityRule(
            id="critical", name="Critical", description="Critical", severity="CRITICAL",
            category="security", rule_type="pattern", pattern="test", message="Critical message"
        )
        high_rule = SecurityRule(
            id="high", name="High", description="High", severity="HIGH",
            category="security", rule_type="pattern", pattern="test", message="High message"
        )

        ruleset = SecurityRuleSet(
            name="Test", version="1.0", description="Test",
            rules=[critical_rule, high_rule]
        )

        critical_rules = ruleset.get_rules_by_severity("CRITICAL")
        assert len(critical_rules) == 1
        assert critical_rules[0].id == "critical"

    def test_get_rules_by_category(self):
        """Test filtering rules by category."""
        security_rule = SecurityRule(
            id="security", name="Security", description="Security", severity="HIGH",
            category="security", rule_type="pattern", pattern="test", message="Security message"
        )
        performance_rule = SecurityRule(
            id="performance", name="Performance", description="Performance", severity="MEDIUM",
            category="performance", rule_type="pattern", pattern="test", message="Performance message"
        )

        ruleset = SecurityRuleSet(
            name="Test", version="1.0", description="Test",
            rules=[security_rule, performance_rule]
        )

        security_rules = ruleset.get_rules_by_category("security")
        assert len(security_rules) == 1
        assert security_rules[0].id == "security"


class TestPolicyManager:
    """Test cases for PolicyManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.policy_manager = PolicyManager()

    def test_load_policy_from_json(self):
        """Test loading policy from JSON file."""
        policy_data = {
            "name": "Test Policy",
            "version": "1.0",
            "description": "Test policy",
            "rules": [
                {
                    "id": "no_latest_tag",
                    "name": "No Latest Tag",
                    "description": "Do not use latest tag",
                    "severity": "HIGH",
                    "category": "security",
                    "rule_type": "pattern",
                    "pattern": "FROM.*:latest",
                    "message": "Do not use latest tag",
                    "fix_example": "FROM ubuntu:22.04"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(policy_data, f)
            policy_path = Path(f.name)

        try:
            ruleset = self.policy_manager.load_policy(policy_path)
            assert ruleset.name == "Test Policy"
            assert len(ruleset.rules) == 1
            assert ruleset.rules[0].id == "no_latest_tag"
        finally:
            policy_path.unlink()

    def test_load_policy_from_yaml(self):
        """Test loading policy from YAML file."""
        policy_data = {
            "name": "Test YAML Policy",
            "version": "1.0",
            "description": "Test YAML policy",
            "rules": [
                {
                    "id": "root_user_check",
                    "name": "Root User Check",
                    "description": "Container should not run as root",
                    "severity": "CRITICAL",
                    "category": "security",
                    "rule_type": "function",
                    "function_name": "check_root_user",
                    "message": "Container runs as root",
                    "fix_example": "USER nonroot"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(policy_data, f)
            policy_path = Path(f.name)

        try:
            ruleset = self.policy_manager.load_policy(policy_path)
            assert ruleset.name == "Test YAML Policy"
            assert len(ruleset.rules) == 1
            assert ruleset.rules[0].severity == "CRITICAL"
        finally:
            policy_path.unlink()

    def test_load_invalid_policy_file(self):
        """Test loading invalid policy file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            policy_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Failed to parse policy file"):
                self.policy_manager.load_policy(policy_path)
        finally:
            policy_path.unlink()

    def test_validate_policy_schema(self):
        """Test policy schema validation."""
        # Valid policy
        valid_policy = {
            "name": "Test",
            "version": "1.0",
            "description": "Test",
            "rules": []
        }
        assert self.policy_manager._validate_policy_schema(valid_policy) is True

        # Invalid policy - missing required fields
        invalid_policy = {"name": "Test"}
        assert self.policy_manager._validate_policy_schema(invalid_policy) is False


class TestComplianceChecker:
    """Test cases for ComplianceChecker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compliance_checker = ComplianceChecker()
        self.parser = DockerfileParser()

    def test_check_soc2_compliance(self):
        """Test SOC2 compliance checking."""
        # Non-compliant Dockerfile
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update && apt-get install -y curl
CMD ["curl", "--version"]
"""
        parsed = self.parser.parse(dockerfile_content)
        violations = self.compliance_checker.check_soc2_compliance(
            dockerfile_content, parsed
        )

        assert len(violations) > 0
        violation_ids = [v.vulnerability for v in violations]
        assert any("latest tag" in v.lower() for v in violation_ids)
        assert any("root user" in v.lower() for v in violation_ids)

    def test_check_pci_dss_compliance(self):
        """Test PCI-DSS compliance checking."""
        dockerfile_content = """
FROM ubuntu:22.04
USER root
EXPOSE 80
CMD ["nginx"]
"""
        parsed = self.parser.parse(dockerfile_content)
        violations = self.compliance_checker.check_pci_dss_compliance(
            dockerfile_content, parsed
        )

        assert len(violations) > 0
        # Should find root user violation
        assert any("root" in v.vulnerability.lower() for v in violations)

    def test_check_hipaa_compliance(self):
        """Test HIPAA compliance checking."""
        dockerfile_content = """
FROM alpine:3.18
RUN apk add --no-cache nginx
USER nginx
EXPOSE 443
CMD ["nginx"]
"""
        parsed = self.parser.parse(dockerfile_content)
        violations = self.compliance_checker.check_hipaa_compliance(
            dockerfile_content, parsed
        )

        # This compliant Dockerfile should have fewer violations
        assert isinstance(violations, list)


class TestAdvancedSecurityEngine:
    """Test cases for AdvancedSecurityEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AdvancedSecurityEngine()
        self.parser = DockerfileParser()

    def test_engine_initialization(self):
        """Test engine initialization."""
        assert isinstance(self.engine.policy_manager, PolicyManager)
        assert isinstance(self.engine.compliance_checker, ComplianceChecker)
        assert len(self.engine.loaded_policies) == 0

    def test_load_default_policies(self):
        """Test loading default security policies."""
        with patch.object(self.engine.policy_manager, 'get_default_policies') as mock_defaults:
            mock_policy = SecurityRuleSet(
                name="Default", version="1.0", description="Default",
                rules=[]
            )
            mock_defaults.return_value = [mock_policy]

            self.engine.load_default_policies()
            assert len(self.engine.loaded_policies) == 1
            assert self.engine.loaded_policies[0].name == "Default"

    def test_load_custom_policy(self):
        """Test loading custom policy from file."""
        policy_data = {
            "name": "Custom Policy",
            "version": "1.0",
            "description": "Custom security policy",
            "rules": [
                {
                    "id": "custom_rule",
                    "name": "Custom Rule",
                    "description": "A custom rule",
                    "severity": "MEDIUM",
                    "category": "security",
                    "rule_type": "pattern",
                    "pattern": "FROM.*debian",
                    "message": "Avoid Debian base images"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(policy_data, f)
            policy_path = Path(f.name)

        try:
            self.engine.load_custom_policy(policy_path)
            assert len(self.engine.loaded_policies) == 1
            assert self.engine.loaded_policies[0].name == "Custom Policy"
        finally:
            policy_path.unlink()

    def test_analyze_dockerfile_with_custom_rules(self):
        """Test analyzing Dockerfile with custom rules."""
        # Create a custom rule that triggers on Ubuntu
        ubuntu_rule = SecurityRule(
            id="no_ubuntu",
            name="No Ubuntu",
            description="Avoid Ubuntu base images",
            severity="MEDIUM",
            category="security",
            rule_type="pattern",
            pattern=r"FROM\s+ubuntu",
            message="Ubuntu base images are not recommended",
            fix_example="FROM alpine:3.18"
        )

        ruleset = SecurityRuleSet(
            name="Test Rules", version="1.0", description="Test",
            rules=[ubuntu_rule]
        )
        self.engine.loaded_policies.append(ruleset)

        dockerfile_content = """
FROM ubuntu:22.04
RUN apt-get update
"""
        parsed = self.parser.parse(dockerfile_content)

        violations = self.engine.analyze_dockerfile(dockerfile_content, parsed)

        assert len(violations) > 0
        assert any("ubuntu" in v.vulnerability.lower() for v in violations)

    def test_check_compliance_framework(self):
        """Test compliance framework checking."""
        dockerfile_content = """
FROM ubuntu:latest
USER root
CMD ["echo", "hello"]
"""
        parsed = self.parser.parse(dockerfile_content)

        # Test SOC2 compliance
        soc2_violations = self.engine.check_compliance(
            dockerfile_content, parsed, "SOC2"
        )
        assert len(soc2_violations) > 0

        # Test invalid framework
        with pytest.raises(ValueError, match="Unsupported compliance framework"):
            self.engine.check_compliance(dockerfile_content, parsed, "INVALID")

    def test_evaluate_pattern_rule(self):
        """Test evaluating pattern-based rules."""
        pattern_rule = SecurityRule(
            id="test_pattern",
            name="Test Pattern",
            description="Test pattern rule",
            severity="HIGH",
            category="security",
            rule_type="pattern",
            pattern=r"RUN.*sudo",
            message="Avoid using sudo in containers"
        )

        dockerfile_content = "RUN sudo apt-get install package"
        match = self.engine._evaluate_pattern_rule(pattern_rule, dockerfile_content)

        assert match is not None
        assert match.vulnerability == "Test Pattern"
        assert match.severity == "HIGH"

    def test_evaluate_function_rule(self):
        """Test evaluating function-based rules."""
        function_rule = SecurityRule(
            id="test_function",
            name="Test Function",
            description="Test function rule",
            severity="CRITICAL",
            category="security",
            rule_type="function",
            function_name="check_exposed_secrets",
            message="Secrets should not be exposed"
        )

        dockerfile_content = "ENV API_KEY=secret123"
        parsed = self.parser.parse(dockerfile_content)

        # Mock the function
        with patch.object(self.engine, '_call_rule_function') as mock_func:
            mock_func.return_value = True

            match = self.engine._evaluate_function_rule(
                function_rule, dockerfile_content, parsed
            )

            assert match is not None
            assert match.vulnerability == "Test Function"
            assert match.severity == "CRITICAL"

    def test_get_security_score_with_custom_rules(self):
        """Test calculating security score with custom rules."""
        # Add some custom rules
        high_rule = SecurityRule(
            id="high_rule", name="High Rule", description="High severity rule",
            severity="HIGH", category="security", rule_type="pattern",
            pattern="FROM.*:latest", message="Latest tag usage"
        )

        ruleset = SecurityRuleSet(
            name="Test", version="1.0", description="Test",
            rules=[high_rule]
        )
        self.engine.loaded_policies.append(ruleset)

        dockerfile_content = "FROM ubuntu:latest"
        parsed = self.parser.parse(dockerfile_content)

        violations = self.engine.analyze_dockerfile(dockerfile_content, parsed)
        score = self.engine.get_security_score(violations)

        assert score.score <= 85  # Should be penalized for HIGH severity
        assert score.grade in ["A", "B", "C", "D", "F"]
        assert len(score.recommendations) > 0

    def test_get_violation_report(self):
        """Test generating violation report."""
        violation = SecurityFix(
            vulnerability="Test Violation",
            severity="HIGH",
            description="Test violation description",
            fix="Test fix"
        )

        report = self.engine.get_violation_report([violation])

        assert "Test Violation" in report
        assert "HIGH" in report
        assert "Test violation description" in report
        assert "Test fix" in report

    def test_empty_dockerfile_analysis(self):
        """Test analyzing empty dockerfile."""
        violations = self.engine.analyze_dockerfile("", [])
        assert len(violations) == 0

    def test_malformed_custom_policy(self):
        """Test handling malformed custom policy."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": "json"')  # Malformed JSON
            policy_path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                self.engine.load_custom_policy(policy_path)
        finally:
            policy_path.unlink()
